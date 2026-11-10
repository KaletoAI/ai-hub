"""LU Labs composite account: chat must never be streamed by accident, a paid job is
booked once (a re-POST only with the same client_request_id), capped at its quote, its
result fetched without leaking the key to a CDN, and the defensive catalog reader
must survive the unpublished catalog shape. All calls use MockTransport.
"""
import asyncio
import copy
import io
import json
import unittest
from unittest.mock import AsyncMock

import httpx
from fastapi import HTTPException
from PIL import Image
from starlette.requests import Request

import adapters
import lulabs
import model_kinds


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (1, 2, 3)).save(buf, "PNG")
    return buf.getvalue()


CREATE = {"data": [
    {"id": "flux-dev", "kind": "image", "ops": ["generate", "edit"], "credits": 1200},
    {"id": "wan-2.2-720p", "kind": "video", "ops": ["generate", "animate"],
     "durations": [5, 8], "credits": {"per_duration": {"5": 40000, "8": 64000}}},
    {"id": "voice-1", "kind": "audio", "ops": ["tts"], "credits": 50},
    {"id": "broken"}, "not-a-row", {"id": "lora", "kind": "image", "ops": ["lora-train"]},
], "tools": {"removebg": {"credits": 100}, "upscale": {"credits": {"2k": 200, "4k": 400}}}}


class Pure(unittest.TestCase):
    def test_catalog_reader_skips_malformed_rows_and_unsupported_ops(self):
        """The create catalog's JSON is unpublished: a row without kind/id, a non-dict
        row or an unsupported op (lora-train) is skipped — one odd row must not take
        the whole account out of routing. A payload with no list at all is an error
        (an empty snapshot would silently hide every model)."""
        cats = lulabs.parse_create_catalog(CREATE)
        self.assertEqual(set(cats["image"]), {"flux-dev"})
        self.assertEqual(cats["image"]["flux-dev"]["_ops"], ["generate", "edit"])
        self.assertEqual(cats["video"]["wan-2.2-720p"]["_ops"], ["generate", "animate"])
        self.assertEqual(set(cats["speech"]), {"voice-1"})
        self.assertEqual(set(cats["image-removebg"]), {"removebg"})
        self.assertEqual(set(cats["image-upscale"]), {"upscale"})
        self.assertNotIn("lora", set().union(*map(set, cats.values())))
        with self.assertRaises(ValueError):
            lulabs.parse_create_catalog({"error": "nope"})

    def test_generation_types_follow_the_op(self):
        """`video-lipsync` is a VIDEO, `music` audio, `tts` speech — the generic label
        table knows none of these endpoint names and would call them images."""
        for endpoint, kind in (("video-lipsync", "video"), ("video", "video"), ("speech", "tts"),
                               ("music", "audio"), ("image-upscale", "image")):
            self.assertEqual(model_kinds.generation_types({"lulabs": {"endpoint": endpoint}}), {kind})

    def test_validation_refuses_before_any_paid_call(self):
        """n > 1, an unpublished duration, unknown keys and missing required inputs are
        the client's mistakes — refused before upload/quote/submit."""
        cats = lulabs.parse_create_catalog(CREATE)
        video = lulabs.candidate_for("lu", "wan-2.2-720p", "video", cats["video"]["wan-2.2-720p"])
        self.assertEqual(lulabs.validate_request(video, {"prompt": "x", "negative_prompt": ""}, (), ()),
                         {"duration": 5, "op": "generate"})
        for bad in ({"prompt": "x", "duration": 7}, {"prompt": "x", "n": 2},
                    {"prompt": "x", "steps": 9}, {}):
            with self.assertRaises(lulabs.INPUT_ERROR):
                lulabs.validate_request(video, bad, (), ())
        # a source image turns the same model id into an animate job
        self.assertEqual(lulabs.validate_request(video, {"prompt": "x"}, ("source_image",), ())["op"],
                         "animate")
        image = lulabs.candidate_for("lu", "flux-dev", "image", cats["image"]["flux-dev"])
        self.assertEqual(lulabs.validate_request(image, {"prompt": "x"}, ("source_image", "mask_image"),
                                                 ())["op"], "edit")
        with self.assertRaises(lulabs.INPUT_ERROR):      # a mask without the image it edits
            lulabs.validate_request(image, {"prompt": "x"}, ("mask_image",), ())

    def test_alias_options_cannot_override_paths_caps_or_op(self):
        """options_json must never smuggle max_credits, client_request_id, op or a path."""
        for key in ("max_credits", "client_request_id", "op", "source_path", "model"):
            with self.assertRaises(lulabs.INPUT_ERROR):
                lulabs.options_of({"lulabs": {"options": {"options_json": json.dumps({key: 1})}}})

    def test_sniff_names_the_file_by_its_bytes(self):
        self.assertEqual(lulabs.sniff(_png())[2], "image")
        self.assertEqual(lulabs.sniff(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8)[0], "mp4")
        self.assertEqual(lulabs.sniff(b"RIFF\x00\x00\x00\x00WAVEfmt ")[0], "wav")
        self.assertIsNone(lulabs.sniff(b"<html>"))


class LuLabsAdapterTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests, self.meta = [], []
        self.png = _png()
        self.job_answers = []          # queued answers for POST /jobs (Response or Exception)
        self.polls = []
        self.block_poll = None
        self.quote = httpx.Response(200, json={"object": "job.quote", "credits": 1200})
        self.content = None
        saved = adapters._CLOUD_DOWNLOAD_BACKOFF_S
        adapters._CLOUD_DOWNLOAD_BACKOFF_S = 0
        self.addCleanup(setattr, adapters, "_CLOUD_DOWNLOAD_BACKOFF_S", saved)
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self.handle))
        self.ctx = adapters.AdapterContext(
            auth_headers=lambda b: {"Authorization": "Bearer " + b["api_key"]},
            inflight_inc=lambda bid: None, inflight_dec=lambda bid: None,
            cost_usd=lambda *a: 0, source_of=lambda r: "test",
            record_call=AsyncMock(), log_enabled=lambda: False,
            note_job_meta=lambda job, meta: self.meta.append(copy.deepcopy(meta)),
            note_progress=lambda job, p: None, http_client=lambda: self.client)
        self.backend = {"type": "lulabs", "name": "lu", "url": "https://lu.test/api/inference",
                        "api_key": "lu_secret", "poll_interval": 0, "max_wait": 5}
        self.ad = adapters.make_adapter(self.backend, self.ctx)
        real = httpx.AsyncClient
        transport = httpx.MockTransport(self.handle)
        adapters.httpx.AsyncClient = lambda *a, **kw: real(*a, transport=transport,
                                                           **{k: v for k, v in kw.items()
                                                              if k != "transport"})
        self.addCleanup(setattr, adapters.httpx, "AsyncClient", real)

    async def asyncSetUp(self):
        await self.ad.discover(self.client)
        self.requests.clear()

    async def asyncTearDown(self):
        await self.client.aclose()

    async def handle(self, request):
        self.requests.append(request)
        path = request.url.path
        if request.url.host == "cdn.test":
            return httpx.Response(200, content=self.png)
        if path.endswith("/v1/models"):
            if request.url.params.get("type") == "create":
                return httpx.Response(200, json=CREATE)
            return httpx.Response(200, json={"data": [{"id": "moonshotai/Kimi-K3", "context_length": 8192}]})
        if path.endswith("/v1/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant",
                                                                      "content": "ok"}}]})
        if path.endswith("/v1/files"):
            return httpx.Response(200, json={"path": "me/abc.png"})
        if path.endswith("/v1/jobs/quote"):
            return self.quote
        if path.endswith("/v1/jobs"):
            answer = self.job_answers.pop(0) if self.job_answers else None
            if isinstance(answer, Exception):
                raise answer
            return answer or httpx.Response(202, json={"object": "job", "id": "j1",
                                                       "status": "queued", "credits": 1200})
        if path.endswith("/content"):
            if self.content is not None:
                return self.content
            return httpx.Response(302, headers={"Location": "https://cdn.test/file?sig=1"})
        if path.endswith("/cancel"):
            return httpx.Response(200, json={"status": "canceled"})
        if "/v1/jobs/" in path:
            if self.block_poll:
                self.block_poll.set()
                await asyncio.sleep(3600)
            job = self.polls.pop(0) if self.polls else {"id": "j1", "status": "succeeded",
                                                        "credits": 1200}
            return httpx.Response(200, json=job)
        return httpx.Response(404)

    def req(self, model="flux-dev", endpoint="image", **kw):
        return adapters.NormalizedRequest(real_model=model, cloud={"endpoint": endpoint, "options": {}},
                                          inputs=kw.pop("inputs", {"prompt": "A tree"}),
                                          job_id="job-1", **kw)

    def calls(self, suffix):
        return [r for r in self.requests if r.url.path.endswith(suffix)]

    async def test_chat_without_stream_flag_is_sent_stream_false(self):
        """LU streams by DEFAULT: a client body without `stream` must go out with
        stream:false, or the gateway would parse an SSE body as JSON."""
        await self.ad.dispatch(adapters.NormalizedRequest(
            path="/v1/chat/completions", real_model="moonshotai/Kimi-K3",
            raw=Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1)}),
            body={"model": "moonshotai/Kimi-K3", "messages": [{"role": "user", "content": "hi"}]}))
        sent = json.loads(self.calls("/v1/chat/completions")[0].content)
        self.assertIs(sent["stream"], False)

    async def test_media_id_on_chat_is_a_400(self):
        with self.assertRaises(HTTPException) as caught:
            await self.ad.dispatch(adapters.NormalizedRequest(
                path="/v1/chat/completions", real_model="flux-dev", body={"messages": []}))
        self.assertEqual(caught.exception.status_code, 400)

    async def test_job_is_quoted_capped_and_result_fetched_without_key(self):
        """Quote first (free), book with max_credits = quote and a client_request_id;
        /content is fetched WITH the key and not followed, the redirect target WITHOUT it;
        the file is named by its bytes; LU credits land as consumed_credits."""
        out = await self.ad.generate(self.req())
        quote = json.loads(self.calls("/v1/jobs/quote")[0].content)
        job = json.loads(self.calls("/v1/jobs")[0].content)
        self.assertEqual(job["params"]["max_credits"], 1200)
        self.assertEqual(job["params"]["op"], "generate")
        self.assertEqual(job["model"], "flux-dev")
        self.assertEqual(job["params"]["client_request_id"], quote["params"]["client_request_id"])
        content = self.calls("/content")[0]
        self.assertEqual(content.headers["authorization"], "Bearer lu_secret")
        cdn = [r for r in self.requests if r.url.host == "cdn.test"][0]
        self.assertNotIn("authorization", cdn.headers)
        self.assertEqual([(b.kind, b.name) for b in out.blobs], [("image", "result.png")])
        self.assertEqual(out.meta["consumed_credits"], 1200)
        self.assertEqual(out.meta["cloud_task_id"], "j1")

    async def test_uncertain_submit_reposts_only_with_the_same_request_id(self):
        """A lost answer to POST /jobs is re-sent with the SAME body and client_request_id
        (LU returns the job already booked) — never a fresh id, never another backend."""
        self.job_answers = [httpx.ReadTimeout("lost"), httpx.Response(502, text="bad gateway")]
        await self.ad.generate(self.req())
        posts = self.calls("/v1/jobs")
        self.assertEqual(len(posts), 3)
        self.assertEqual(len({p.content for p in posts}), 1)

    async def test_bare_proxy_errors_on_the_first_answer_are_resent_with_the_same_id(self):
        """A 503/429 WITHOUT LU's own error code may come from a proxy after LU booked:
        treating it as "nothing booked" (failover) would book the job again elsewhere.
        It is re-sent with the same client_request_id instead."""
        self.job_answers = [httpx.Response(503, text="<html>bad gateway</html>")]
        out = await self.ad.generate(self.req())
        posts = self.calls("/v1/jobs")
        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0].content, posts[1].content)
        self.assertEqual(out.meta["cloud_task_id"], "j1")

    async def test_unreadable_quote_never_books_uncapped(self):
        """Every job is capped at its quote: a quote whose credits cannot be read must not
        silently book a job without `max_credits`."""
        self.quote = httpx.Response(200, json={"object": "job.quote", "price": "?"})
        with self.assertRaises(RuntimeError):
            await self.ad.generate(self.req())
        self.assertEqual(self.calls("/v1/jobs"), [])
        self.quote = httpx.Response(200, json={"quote": {"credits": "1500"}})
        await self.ad.generate(self.req())
        self.assertEqual(json.loads(self.calls("/v1/jobs")[0].content)["params"]["max_credits"], 1500)

    async def test_openai_image_shim_keys_are_dropped_not_refused(self):
        await self.ad.generate(self.req(params={"size": "1024x1024", "quality": "hd",
                                                "style": "vivid", "output_format": "png"}))
        params = json.loads(self.calls("/v1/jobs")[0].content)["params"]
        self.assertEqual(params["size"], "1024x1024")
        self.assertNotIn("quality", params)

    async def test_uncertain_submit_that_never_resolves_is_final(self):
        """After SUBMIT_ATTEMPTS uncertain answers: a final RuntimeError (not a
        failover-class error) and the trace says create_unconfirmed."""
        self.job_answers = [httpx.ReadTimeout("x")] * lulabs.SUBMIT_ATTEMPTS
        req = self.req()
        with self.assertRaises(RuntimeError) as caught:
            await self.ad.generate(req)
        self.assertNotIsInstance(caught.exception, ConnectionError)
        self.assertTrue(req.cloud_trace.get("create_unconfirmed"))
        self.assertEqual(len(self.calls("/v1/jobs")), lulabs.SUBMIT_ATTEMPTS)

    async def test_first_refusals_map_to_the_gateway_classes(self):
        """A first answer that booked nothing: 402/credits_exhausted → CloudNoCredits,
        429 rate_limited → CloudBusy, 503 → CloudTaskRetryable (failover allowed)."""
        for response, cls in (
                (httpx.Response(402, json={"error": "no plan"}), adapters.CloudNoCredits),
                (httpx.Response(429, json={"error": "x", "code": "credits_exhausted"}),
                 adapters.CloudNoCredits),
                (httpx.Response(429, json={"error": "x", "code": "rate_limited"}), adapters.CloudBusy),
                (httpx.Response(503, json={"error": "x", "code": "model_unavailable"}),
                 adapters.CloudTaskRetryable)):
            self.job_answers = [response]
            with self.assertRaises(cls):
                await self.ad.generate(self.req())

    async def test_failed_job_is_final_and_key_redacted(self):
        self.polls = [{"id": "j1", "status": "failed", "error": "render failed lu_secret"}]
        with self.assertRaises(RuntimeError) as caught:
            await self.ad.generate(self.req())
        self.assertIn("render failed", str(caught.exception))
        self.assertNotIn("lu_secret", str(caught.exception))
        self.assertEqual(len(self.calls("/v1/jobs")), 1)

    async def test_lost_result_after_paid_job_is_not_failover_class(self):
        """The job is booked and done; an undownloadable result must not raise a
        failover-class error (main would book the job again on the next candidate)."""
        self.content = httpx.Response(404)
        with self.assertRaises(adapters.CloudArtifactLost):
            await self.ad.generate(self.req())

    async def test_uploads_become_paths_and_cancel_hits_the_job(self):
        """An edit uploads its image (role=source) and sends the returned path; a
        cancelled gateway job asks LU to cancel ITS job and records stop_requested."""
        self.block_poll = asyncio.Event()
        req = self.req(endpoint="image", upload_images={"source_image": self.png})
        task = asyncio.create_task(self.ad.generate(req))
        await asyncio.wait_for(self.block_poll.wait(), 5)
        job = json.loads(self.calls("/v1/jobs")[0].content)
        self.assertEqual(job["params"]["source_path"], "me/abc.png")
        self.assertEqual(job["params"]["op"], "edit")
        self.assertIn(b'name="role"', self.calls("/v1/files")[0].content)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(self.calls("/v1/jobs/j1/cancel")), 1)
        self.assertTrue(self.meta[-1].get("stop_requested"))


if __name__ == "__main__":
    unittest.main()
