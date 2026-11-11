"""wiro and LU Labs are composite accounts: their catalog media must route without
aliases (`backend/owner/project` included), stay out of chat and out of `@backends`,
and a real /v1/generations call must reach the vendor with the client's values —
the mechanisms main.py shares with NanoGPT/Venice through DYNAMIC_CATALOG_TYPES.
"""
import asyncio
import io
import json
import os
from unittest.mock import patch

import httpx
from PIL import Image

from tests.test_ui_form_validation import _Fixture
import adapters
import jobs
import main
import mediaprice
import store

SAMPLE = "https://cdn.wiro.ai/uploads/sampleinputs/sample.png"


def _png():
    stream = io.BytesIO()
    Image.new("RGB", (24, 16), "blue").save(stream, format="PNG")
    return stream.getvalue()


WIRO_IMAGE = {"name": "nano", "dynamicprice": '[{"inputs": {}, "price": 0.039, "priceMethod": "cpo"}]',
              "parameters": [{"items": [
                  {"id": "inputImage", "type": "combinefileinput", "required": False,
                   "defaultvalue": [SAMPLE], "maxinputlenght": 3},
                  {"id": "prompt", "type": "textarea", "required": True, "defaultvalue": "sample"},
                  {"id": "aspectRatio", "type": "select", "defaultvalue": "1:1",
                   "options": [{"text": "1:1", "value": "1:1"}, {"text": "16:9", "value": "16:9"}]}]}]}


class _Routing(_Fixture):
    BACKEND: dict = {}
    CATALOGS: dict = {}

    def setUp(self):
        super().setUp()
        self.backend = dict(self.BACKEND)
        self.ad = adapters.make_adapter(self.backend, main.adapter_ctx)
        self.ad._publish_catalogs(json.loads(json.dumps(self.CATALOGS)))
        bid = f"{self.backend['type']}:{self.backend['name']}"
        for name, value in {
            "backends": [self.backend], "backend_adapters": {bid: self.ad},
            "backend_models": {bid: set().union(*self.CATALOGS.values())},
            "backend_healthy": {bid: True}, "backend_inflight": {},
            "image_models": {}, "virtual_models": {}, "model_prefix": True, "gen_speed": {},
            "gen_exec_faults": {}, "gen_progress": {}, "_backend_names": set(),
            "_llm_backends": [], "_gen_backends": [], "_route_index": {},
        }.items():
            p = patch.object(main, name, value)
            p.start()
            self.addCleanup(p.stop)
        main._backend_names.add(self.backend["name"])
        main._llm_backends.append(self.backend)
        main._gen_backends.append(self.backend)
        main.rebuild_route_index()
        saved = {k: getattr(jobs, k) for k in ("_DB_PATH", "_BLOB_DIR", "_DEFAULT_TTL", "_active")}
        self.addCleanup(lambda: [setattr(jobs, k, v) for k, v in saved.items()])
        jobs.init(os.path.join(self.tmp.name, "jobs.db"), os.path.join(self.tmp.name, "blobs"))

    def vendor(self, handler):
        original = httpx.AsyncClient
        return patch("adapters.httpx.AsyncClient", lambda *a, **kw: original(
            *a, transport=httpx.MockTransport(handler),
            **{k: v for k, v in kw.items() if k != "transport"}))


class WiroRouting(_Routing):
    BACKEND = {"name": "wiro", "type": "wiro", "url": "https://llm.test", "paid": True,
               "media_url": "https://api.test/v1", "api_key": "K:S", "max_concurrent": 2,
               "poll_interval": 0}
    CATALOGS = {"text": {"openai/gpt-chat": {}}, "image": {"google/nano-banana": WIRO_IMAGE},
                "video": {}, "audio": {}, "asr": {},
                "speech": {"elevenlabs/tts": {"name": "tts", "parameters": [{"items": [
                    {"id": "prompt", "type": "textarea", "required": True}]}]}}}

    def test_owner_project_ids_list_and_route_without_alias(self):
        """`wiro/google/nano-banana` has TWO slashes: the backend prefix splits at the
        first, the media id keeps owner/project. Chat lists only the LLM side."""
        image = {r["id"] for r in self.c.get("/v1/models?type=image").json()["data"]}
        chat = {r["id"] for r in self.c.get("/v1/models?type=chat").json()["data"]}
        self.assertIn("wiro/google/nano-banana", image)
        self.assertIn("wiro/openai/gpt-chat", chat)
        self.assertNotIn("wiro/google/nano-banana", chat)
        self.assertEqual(main.get_gen_routes("wiro/google/nano-banana")[0][1]["model"],
                         "google/nano-banana")
        self.assertEqual(store.list_aliases(), {})

    def test_generation_reaches_wiro_with_client_values_only(self):
        """A /v1/generations call: the prompt and a client control go out, the sample
        image default never does, and the wiro output becomes the job's image."""
        sent = []

        def provider(request):
            if request.url.host == "cdn.test":
                return httpx.Response(200, content=_png())
            if request.url.path.startswith("/v1/Run/"):
                sent.append(json.loads(request.content))
                return httpx.Response(200, json={"result": True, "taskid": "7",
                                                 "socketaccesstoken": "t7"})
            if request.url.path.endswith("/Task/Detail"):
                return httpx.Response(200, json={"result": True, "tasklist": [{
                    "status": "task_postprocess_end", "pexit": "0", "totalcost": "0.039",
                    "outputs": [{"name": "0.png", "url": "https://cdn.test/0.png"}]}]})
            return httpx.Response(404)
        with self.vendor(provider):
            response = self.c.post("/v1/generations", json={
                "model": "wiro/google/nano-banana", "prompt": "blue lamp",
                "params": {"aspect_ratio": "16:9"}, "output": {"mode": "sync"}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(sent, [{"prompt": "blue lamp", "aspectRatio": "16:9"}])
        self.assertEqual(response.json()["status"], "done", response.text)

    def test_schema_answers_for_slash_ids_and_stored_aliases(self):
        """`GET /v1/generations/{alias}/schema` is how a client learns the field names.
        A catalog id carries slashes (`wiro/google/nano-banana`) — a plain `{alias}`
        segment never matched it (404 for EVERY catalog model) — and catalog ids live
        in no alias table. A stored alias holds no catalog row either, so its wiro
        fields were empty: the live row is attached as `_gen_routes` does."""
        r = self.c.get("/v1/generations/wiro/google/nano-banana/schema")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual([i["name"] for i in body["images"]],
                         ["input_image", "input_image_2", "input_image_3"])
        self.assertIn("aspect_ratio", {p["name"] for p in body["params"]})
        self.assertEqual(body["loras_url"], "/v1/generations/wiro/google/nano-banana/loras")
        loras = self.c.get("/v1/generations/wiro/google/nano-banana/loras")
        self.assertEqual((loras.status_code, loras.json()["loras"]), (200, []))
        self.assertEqual(self.c.get("/v1/generations/wiro/no/such/schema").status_code, 404)
        import wiro
        store.upsert("nano", [wiro.candidate_for("wiro", "google/nano-banana", "image")])
        stored = self.c.get("/v1/generations/nano/schema")
        self.assertEqual(stored.status_code, 200, stored.text)
        self.assertEqual(stored.json()["images"][0]["name"], "input_image")

    def test_paid_speech_models_never_become_llm_routable(self):
        """main's text side reads the catalog key `tts`; a media speech catalog under that
        key made paid task-API models LLM speech models. They are `?type=tts` media only."""
        self.assertNotIn("elevenlabs/tts", main._llm_models(self.backend))
        tts = {r["id"] for r in self.c.get("/v1/models?type=tts").json()["data"]}
        self.assertIn("wiro/elevenlabs/tts", tts)
        self.assertEqual(main.resolve_routes("wiro/elevenlabs/tts"), ([], []))

    def test_bad_values_are_400_before_any_paid_call(self):
        def provider(request):
            self.fail("invalid input must not buy a wiro task")
        with self.vendor(provider):
            for params in ({"aspectRatio": "7:3"}, {"steps": 9}):
                response = self.c.post("/v1/generations", json={
                    "model": "wiro/google/nano-banana", "prompt": "x", "params": params,
                    "output": {"mode": "sync"}})
                self.assertEqual(response.status_code, 400, response.text)

    def test_all_backends_grant_never_opens_paid_media(self):
        llm_only = {"name": "llm", "role": "user", "models": [main.GRANT_ALL_BACKENDS]}
        self.assertTrue(main._model_allowed(llm_only, "wiro/openai/gpt-chat", media=False))
        self.assertFalse(main._model_allowed(llm_only, "wiro/google/nano-banana", media=True))
        account = {"name": "acct", "role": "user", "models": ["wiro"]}
        self.assertTrue(main._model_allowed(account, "wiro/google/nano-banana", media=True))

    def test_media_price_is_the_per_output_usd(self):
        self.assertEqual(main.media_price_info("wiro/google/nano-banana")["usd"], 0.039)


class LuLabsRouting(_Routing):
    BACKEND = {"name": "lu", "type": "lulabs", "url": "https://lu.test/api/inference",
               "paid": True, "api_key": "lu_key", "max_concurrent": 2, "poll_interval": 0}
    CATALOGS = {"text": {"moonshotai/Kimi-K3": {}},
                "image": {"flux-dev": {"id": "flux-dev", "kind": "image", "credits": 1200,
                                       "_ops": ["generate", "edit"]}},
                "image-removebg": {}, "image-upscale": {}, "video": {},
                "video-lipsync": {}, "video-motion": {}, "speech": {}, "music": {}}

    def test_openai_edit_with_mask_lands_in_source_and_mask_slots(self):
        """LU edits take [input_image, input_image_mask]: the OpenAI edits shim maps one image
        and its mask there (other catalog accounts refuse a mask) — both uploaded with
        their roles and sent as source_path / mask_path."""
        roles, jobs_sent = [], []

        def provider(request):
            path = request.url.path
            if request.url.host == "cdn.test":
                return httpx.Response(200, content=_png())
            if path.endswith("/files"):
                roles.append(b'name="role"\r\n\r\nmask' in request.content and "mask" or "source")
                return httpx.Response(200, json={"path": f"me/{len(roles)}.png"})
            if path.endswith("/jobs/quote"):
                return httpx.Response(200, json={"credits": 1200})
            if path.endswith("/jobs"):
                jobs_sent.append(json.loads(request.content))
                return httpx.Response(202, json={"id": "j9", "status": "queued"})
            if path.endswith("/content"):
                return httpx.Response(302, headers={"Location": "https://cdn.test/r"})
            if "/jobs/" in path:
                return httpx.Response(200, json={"id": "j9", "status": "succeeded"})
            return httpx.Response(404)
        with self.vendor(provider):
            response = self.c.post("/v1/images/edits", data={
                "model": "lu/flux-dev", "prompt": "make it red"}, files=[
                ("image", ("s.png", _png(), "image/png")), ("mask", ("m.png", _png(), "image/png"))])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(sorted(roles), ["mask", "source"])
        params = jobs_sent[0]["params"]
        self.assertEqual(params["op"], "edit")
        self.assertTrue(params["source_path"] and params["mask_path"])
        self.assertEqual(params["max_credits"], 1200)

    def test_credit_price_and_chat_listing(self):
        self.assertEqual(main.media_price_info("lu/flux-dev")["credits"], 1200)
        chat = {r["id"] for r in self.c.get("/v1/models?type=chat").json()["data"]}
        self.assertEqual(chat, {"lu/moonshotai/Kimi-K3"})

    def test_alias_before_first_catalog_is_retryable(self):
        import lulabs
        store.upsert("pics", [lulabs.candidate_for("lu", "flux-dev", "image")])
        self.ad._publish_catalogs({})
        response = self.c.post("/v1/generations", json={
            "model": "pics", "prompt": "x", "output": {"mode": "sync"}})
        self.assertEqual(response.status_code, 503, response.text)


if __name__ == "__main__":
    import unittest
    unittest.main()
