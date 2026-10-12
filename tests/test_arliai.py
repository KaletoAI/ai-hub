"""Guard ArliAI checkpoint pins, Core pixel/batch limits, real image delivery and
final uncertain submits: losing a synchronous POST answer must never re-run it.
Run with: venv/bin/python -m unittest tests.test_arliai -v
"""
import asyncio
import base64
import io
import json
import unittest

import httpx
from PIL import Image

import adapters
import arliai


def png_bytes():
    out = io.BytesIO()
    Image.new("RGB", (2, 3), "red").save(out, format="PNG")
    return out.getvalue()


class TestRequest(unittest.TestCase):
    def cand(self):
        cand = arliai.default_candidate("images")
        cand["model"] = "Exact Checkpoint [ab12]"
        cand["arliai"]["options"].update(width=1000, height=1000, steps=25)
        return cand

    def test_checkpoint_pinned_and_client_controls_win(self):
        body = arliai.build_request(self.cand(), {
            "prompt": "a bird", "negative_prompt": "blur", "steps": 30,
            "cfg": 7.5, "sampler": "DPM++ 2M", "seed": 12,
            "sd_model_checkpoint": "foreign", "model": "foreign",
            "override_settings": {"sd_model_checkpoint": "foreign"}}, {})
        self.assertEqual(body["sd_model_checkpoint"], "Exact Checkpoint [ab12]")
        self.assertEqual((body["steps"], body["cfg_scale"], body["sampler_name"]),
                         (30, 7.5, "DPM++ 2M"))
        self.assertNotIn("override_settings", body)
        self.assertEqual((body["batch_size"], body["n_iter"]), (1, 1))

    def test_pixel_boundary_not_clamped(self):
        body = arliai.build_request(self.cand(), {"width": 2000, "height": 1500}, {})
        self.assertEqual((body["width"], body["height"]), (2000, 1500))
        with self.assertRaises(RuntimeError):
            arliai.build_request(self.cand(), {"width": 2001, "height": 1500}, {})

    def test_batch_and_invalid_dimensions_refused(self):
        for values in ({"batch_size": 2}, {"n_iter": 2}, {"width": 0},
                       {"height": 1.5}, {"width": float("inf")}):
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                arliai.build_request(self.cand(), values, {})

    def test_blank_checkpoint_not_invented(self):
        with self.assertRaises(RuntimeError):
            arliai.build_request(arliai.default_candidate("images"), {}, {})

    def test_playground_cfg_overrides_alias_default(self):
        values = {p["name"]: p.get("default") for p in arliai.public_fields(self.cand())[0]}
        values["cfg"] = 7.5
        values["sampler"] = "DPM++ 2M"
        body = arliai.build_request(self.cand(), values, {})
        self.assertEqual(body["cfg_scale"], 7.5)
        self.assertEqual(body["sampler_name"], "DPM++ 2M")


class TestAdapter(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.counts = [0, 0]
        self.seen = []
        self.reply = {"images": [base64.b64encode(png_bytes()).decode()],
                      "info": json.dumps({"seed": 12})}
        self.boom = None
        self.status = 200

        def handle(request):
            self.seen.append(request)
            if self.boom:
                raise self.boom("secret-key", request=request)
            if request.method == "GET":
                return httpx.Response(200, json=[{"title": "Model [hash]", "model_name": "Model",
                                                  "filename": "/models/model.safetensors"}])
            return httpx.Response(self.status, json=self.reply)

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        self.ctx = adapters.AdapterContext(
            auth_headers=lambda b: {"Authorization": "Bearer secret-key"},
            inflight_inc=lambda bid: self.counts.__setitem__(0, self.counts[0] + 1),
            inflight_dec=lambda bid: self.counts.__setitem__(1, self.counts[1] + 1),
            cost_usd=lambda *a: 0, source_of=lambda r: "test",
            record_call=lambda *a, **k: None, log_enabled=lambda: False,
            http_client=lambda: self.client)
        self.ad = adapters.make_adapter({"name": "images", "type": "arliai",
                                         "url": "https://api.example", "api_key": "secret-key"}, self.ctx)
        self.req = adapters.NormalizedRequest(alias="image", real_model="Model [hash]",
                                              inputs={"prompt": "bird"}, cloud={"endpoint": "txt2img"})

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_real_png_and_metadata(self):
        result = await self.ad.generate(self.req)
        self.assertEqual((result.blobs[0].mime, result.blobs[0].kind), ("image/png", "image"))
        self.assertEqual(result.blobs[0].data, png_bytes())
        self.assertEqual(result.meta["info"], {"seed": 12})
        self.assertNotIn("consumed_credits", result.meta)
        self.assertEqual(self.counts, [1, 1])

    async def test_discovery_exact_title_and_no_chat_context(self):
        caps = await self.ad.discover(self.client)
        self.assertEqual(caps.models, {"Model [hash]"})
        self.assertEqual(caps.context, {})
        self.assertEqual(self.ad.checkpoint_metadata["Model [hash]"]["model_name"], "Model")

    async def test_preclaimed_slot_not_released(self):
        self.req.slot_held = True
        await self.ad.generate(self.req)
        self.assertEqual(self.counts, [0, 0])

    async def test_malformed_result_is_final(self):
        for reply in ([], {}, {"images": []}, {"images": ["!!"]},
                      {"images": [base64.b64encode(b"not an image").decode()]},
                      {"images": ["abc", "abc"]}, {"images": [123]},
                      {"images": [base64.b64encode(png_bytes()).decode()], "info": "bad"}):
            self.reply = reply
            with self.subTest(reply=reply), self.assertRaises(RuntimeError):
                await self.ad.generate(self.req)
        self.assertEqual(self.counts[0], self.counts[1])

    async def test_uncertain_submit_final_not_transport_failover(self):
        for boom in (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError,
                     httpx.WriteError, httpx.RemoteProtocolError):
            self.boom = boom
            with self.subTest(boom=boom), self.assertRaises(RuntimeError) as caught:
                await self.ad.generate(self.req)
            self.assertNotIn("secret-key", str(caught.exception))
        self.assertEqual(len(self.seen), 5)
        self.assertEqual(self.counts, [5, 5])

    async def test_safe_pre_submit_errors_remain_transport_errors(self):
        for boom in (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
            self.boom = boom
            with self.subTest(boom=boom), self.assertRaises(boom):
                await self.ad.generate(self.req)

    async def test_http_error_bounded_and_redacted(self):
        for status in (401, 429, 500):
            self.status = status
            self.reply = {"error": "secret-key" + "x" * 1000}
            with self.subTest(status=status), self.assertRaises(RuntimeError) as caught:
                await self.ad.generate(self.req)
            self.assertNotIn("secret-key", str(caught.exception))
            self.assertLess(len(str(caught.exception)), 400)

    async def test_backend_pixel_budget_prevents_submit(self):
        self.ad.backend["max_image_pixels"] = 100
        with self.assertRaises(RuntimeError):
            await self.ad.generate(self.req)
        self.assertEqual(self.seen, [])
