"""wiro.ai composite account: the catalog must stay complete and purpose-separated,
a paid task must never be re-submitted after an uncertain answer, sample defaults and
credentials must never leave the gateway, and a cancelled job must Kill its task.
All calls use MockTransport, never a live or paid wiro endpoint.
"""
import asyncio
import copy
import hashlib
import hmac
import io
import json
import unittest
from unittest.mock import AsyncMock

import httpx
from fastapi import HTTPException
from PIL import Image

import adapters
import wiro

SAMPLE = "https://cdn.wiro.ai/uploads/sampleinputs/sample.png"


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (12, 45, 78)).save(buf, "PNG")
    return buf.getvalue()


def _item(fid, typ, **kw):
    base = {"id": fid, "type": typ, "required": False, "advanced": False, "label": fid,
            "defaultvalue": "", "options": [], "minvalue": "", "maxvalue": "", "note": ""}
    base.update(kw)
    return base


def _row(owner, project, categories, items, price=None):
    return {"cleanslugowner": owner, "cleanslugproject": project, "slugowner": owner,
            "slugproject": project.replace("-", " "), "title": project, "categories": categories,
            "parameters": [{"title": "", "items": items}],
            "dynamicprice": json.dumps(price or [{"inputs": {}, "price": 0.039, "priceMethod": "cpo"}])}


IMAGE = _row("google", "nano-banana", ["tool", "text-to-image", "image-editing"], [
    _item("inputImage", "combinefileinput", defaultvalue=[SAMPLE, SAMPLE], maxinputlenght=3),
    _item("prompt", "textarea", required=True, defaultvalue="A koala in a cafe"),
    _item("negativePrompt", "text", defaultvalue="blurry sample"),
    _item("aspectRatio", "select", defaultvalue="1:1",
          options=[{"text": "1:1", "value": "1:1"}, {"text": "16:9", "value": "16:9"}]),
    _item("steps", "number", defaultvalue="20", minvalue="1", maxvalue="50"),
])
LLM = _row("openai", "gpt-chat", ["llm", "chat"], [_item("prompt", "textarea", required=True)])
REALTIME = _row("openai", "gpt-live-1", ["realtime-conversation", "speech-to-speech"], [])
FILE_ONE = _row("acme", "animate", ["image-to-video"], [
    _item("inputImage", "fileinput", required=True, defaultvalue=SAMPLE),
    _item("prompt", "textarea")])
FILLER = [_row("filler", f"img-{i}", ["text-to-image"], [_item("prompt", "textarea")])
          for i in range(100)]


class WiroAdapterTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests, self.meta, self.progress, self.slots = [], [], [], []
        self.pages = [FILLER, [IMAGE, LLM, REALTIME, FILE_ONE]]
        self.page_cap = None
        self.run_answer = httpx.Response(200, json={"result": True, "errors": [], "taskid": "2221",
                                                    "socketaccesstoken": "tok-2221"})
        self.details = []
        self.run_error = None
        self.block_poll = None
        saved = adapters._CLOUD_DOWNLOAD_BACKOFF_S
        adapters._CLOUD_DOWNLOAD_BACKOFF_S = 0
        self.addCleanup(setattr, adapters, "_CLOUD_DOWNLOAD_BACKOFF_S", saved)
        self.png = _png()
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self.handle))
        self.ctx = adapters.AdapterContext(
            auth_headers=lambda b: {"Authorization": "Bearer " + b["api_key"]},
            inflight_inc=lambda bid: self.slots.append(("inc", bid)),
            inflight_dec=lambda bid: self.slots.append(("dec", bid)),
            cost_usd=lambda *a: 0, source_of=lambda r: "test",
            record_call=AsyncMock(), log_enabled=lambda: False,
            note_job_meta=lambda job, meta: self.meta.append(copy.deepcopy(meta)),
            note_progress=lambda job, p: self.progress.append(p),
            http_client=lambda: self.client)
        self.backend = {"type": "wiro", "name": "wiro", "url": "https://llm.test",
                        "media_url": "https://api.test/v1", "api_key": "KEY123:SECRET456",
                        "poll_interval": 0, "max_wait": 5}
        self.ad = adapters.make_adapter(self.backend, self.ctx)
        # CloudTaskAdapter.generate opens its own AsyncClient — route it to the mock too.
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
            return httpx.Response(200, content=self.png, headers={"Content-Type": "image/png"})
        if request.url.host == "llm.test":
            if path == "/v1/models":
                return httpx.Response(200, json={"data": [
                    {"id": "openai/gpt-chat", "context_length": 4096,
                     "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]})
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant",
                                                                      "content": "ok"}}]})
        body = json.loads(request.content or b"{}") if request.headers.get(
            "content-type", "").startswith("application/json") else {}
        if path.endswith("/Tool/List"):
            start = int(body["start"])
            rows = [row for page in self.pages for row in page] if isinstance(self.pages[0], list) \
                else None
            if rows is None:
                return httpx.Response(200, json={"result": True, "tool": self.pages[0]})
            size = min(int(body["limit"]), self.page_cap or 10 ** 6)
            return httpx.Response(200, json={"result": True, "errors": [],
                                             "tool": rows[start:start + size], "total": len(rows)})
        if path.startswith("/v1/Run/"):
            if self.run_error:
                raise self.run_error
            return self.run_answer
        if path.endswith("/Task/Detail"):
            if self.block_poll:
                self.block_poll.set()
                await asyncio.sleep(3600)
            answer = self.details.pop(0) if self.details else _done()
            return httpx.Response(200, json={"result": True, "tasklist": [answer]})
        if path.endswith("/Task/Kill"):
            return httpx.Response(200, json={"result": True, "errors": []})
        return httpx.Response(404)

    def req(self, model="google/nano-banana", endpoint="image", **kw):
        return adapters.NormalizedRequest(real_model=model, cloud={"endpoint": endpoint, "options": {}},
                                          inputs=kw.pop("inputs", {"prompt": "A tree"}),
                                          job_id="job-1", **kw)

    def calls(self, suffix):
        return [r for r in self.requests if r.url.path.endswith(suffix)]

    # ── discovery ─────────────────────────────────────────────────────────────
    async def test_discovery_pages_tool_list_and_maps_categories(self):
        """Tool/List is paginated (a page of `limit` rows means: ask again) and public —
        no account headers on it. Category mapping: llm/chat rows run through the LLM
        gateway, realtime models are rejected by Run — neither may become a paid media
        candidate. The LLM endpoint's /v1/models pages too, so discovery asks limit=1000."""
        await self.ad.discover(self.client)
        lists = self.calls("/Tool/List")
        self.assertEqual([json.loads(r.content)["start"] for r in lists], ["0", "100"])
        for r in lists:
            self.assertNotIn("x-api-key", r.headers)
            self.assertNotIn("authorization", r.headers)
        catalogs = self.ad.model_catalogs
        self.assertIn("google/nano-banana", catalogs["image"])
        self.assertEqual(len(catalogs["image"]), 101)
        self.assertIn("acme/animate", catalogs["video"])
        for endpoint in wiro.ENDPOINTS:
            self.assertNotIn("openai/gpt-chat", catalogs.get(endpoint, {}))
            self.assertNotIn("openai/gpt-live-1", catalogs.get(endpoint, {}))
        self.assertEqual(set(catalogs["text"]), {"openai/gpt-chat"})
        models = [r for r in self.requests if r.url.host == "llm.test"]
        self.assertEqual(models[0].url.params["limit"], "1000")
        # the LLM endpoint takes KEY:SECRET verbatim as Bearer; HMAC headers are not accepted there
        self.assertEqual(models[0].headers["authorization"], "Bearer KEY123:SECRET456")
        self.assertNotIn("x-signature", models[0].headers)

    async def test_short_pages_are_not_the_end_while_total_says_more(self):
        """A server that caps `limit` below 100 answers short pages: `total` decides when
        the catalog is complete, or it was silently cut to its first page."""
        self.page_cap = 40
        await self.ad.discover(self.client)
        self.assertEqual(len(self.ad.model_catalogs["image"]), 101)
        self.assertIn("acme/animate", self.ad.model_catalogs["video"])

    async def test_single_file_input_upload_is_sent_with_empty_url_field(self):
        """A `fileinput` slot sends the file AND an empty `{id}Url` form field — the empty
        field used to crash the adapter (`_mime_and_kind(None)`) on every upload."""
        await self.ad.generate(self.req(model="acme/animate", endpoint="video",
                                        upload_images={"input_image": self.png}))
        raw = self.calls("/Run/acme/animate")[0].content
        self.assertIn(b'name="inputImage"; filename="inputImage.png"', raw)
        self.assertIn(b'name="inputImageUrl"\r\n\r\n\r\n', raw)
        self.assertNotIn(SAMPLE.encode(), raw)

    async def test_params_win_over_blank_inputs(self):
        """main validates {**inputs, **params} and always passes inputs.prompt="" when the
        client put the prompt in params: built the other way round, the blank won and a
        paid run went out WITHOUT the prompt."""
        await self.ad.generate(self.req(inputs={"prompt": "", "negative_prompt": ""},
                                        params={"prompt": "from params"}))
        self.assertEqual(json.loads(self.calls("/Run/google/nano-banana")[0].content)["prompt"],
                         "from params")

    async def test_speech_and_embeddings_are_not_chat(self):
        """wiro's LLM endpoint has no speech/embeddings; its speech models are paid media.
        /v1/audio/speech is a 400 here — it used to forward a task-API id to llm.wiro.ai."""
        for path in ("/v1/audio/speech", "/v1/embeddings"):
            with self.assertRaises(HTTPException) as caught:
                await self.ad.dispatch(adapters.NormalizedRequest(path=path, real_model="openai/gpt-chat",
                                                                  body={}))
            self.assertEqual(caught.exception.status_code, 400)

    async def test_failed_refresh_clears_callable_catalog(self):
        """A failed refresh must not leave stale metadata validating a paid submit."""
        self.pages = [[{"not": "a row"}]]
        self.pages[0] = "broken"
        with self.assertRaises(Exception):
            await self.ad.discover(self.client)
        self.assertEqual(self.ad.model_catalogs, {})
        self.assertIn("google/nano-banana", self.ad.display_media_catalogs["image"])

    async def test_media_id_on_chat_is_a_400(self):
        """A media id sent to chat is the client's mistake: 400, not a 502 + fault row."""
        with self.assertRaises(HTTPException) as caught:
            await self.ad.dispatch(adapters.NormalizedRequest(
                path="/v1/chat/completions", real_model="google/nano-banana",
                body={"model": "google/nano-banana", "messages": []}))
        self.assertEqual(caught.exception.status_code, 400)

    # ── a paid run ────────────────────────────────────────────────────────────
    async def test_run_signs_skips_sample_defaults_and_polls_to_postprocess_end(self):
        """The Run is signed (x-api-key KEY, integer nonce, hex HMAC key=KEY msg=SECRET+NONCE);
        sample defaults (file URLs, text prompts) are never sent, numeric/select defaults
        are; task_end is NOT terminal (outputs are not final before post-processing);
        the artifact is fetched without any credential; totalcost lands as cost_usd."""
        self.details = [{"status": "task_start", "pexit": ""},
                        {"status": "task_end", "pexit": "0", "outputs": []},
                        _done()]
        out = await self.ad.generate(self.req())
        run = self.calls("/Run/google/nano-banana")[0]
        sent = json.loads(run.content)
        self.assertEqual(sent, {"prompt": "A tree", "aspectRatio": "1:1", "steps": 20})
        self.assertNotIn(SAMPLE, run.content.decode())
        nonce = run.headers["x-nonce"]
        self.assertTrue(nonce.isdigit())
        self.assertEqual(run.headers["x-api-key"], "KEY123")
        self.assertEqual(run.headers["x-signature"],
                         hmac.new(b"KEY123", ("SECRET456" + nonce).encode(), hashlib.sha256).hexdigest())
        self.assertEqual(len(self.calls("/Task/Detail")), 3)
        # polled by the numeric id: the socket token never reaches an error text or a job row
        self.assertEqual(json.loads(self.calls("/Task/Detail")[0].content), {"taskid": "2221"})
        asset = [r for r in self.requests if r.url.host == "cdn.test"][0]
        for header in ("x-api-key", "authorization", "x-signature"):
            self.assertNotIn(header, asset.headers)
        self.assertEqual([b.kind for b in out.blobs], ["image"])
        self.assertEqual(out.meta["cost_usd"], 0.039)
        self.assertIsNone(out.meta["consumed_credits"])
        self.assertEqual(out.meta["cloud_task_id"], "2221")
        self.assertEqual(self.meta[0]["cloud_task_id"], "2221")

    async def test_uploaded_images_go_multipart_as_repeated_field(self):
        """Uploaded images replace the sample defaults: multipart with one `inputImage`
        part per slot (input_image, input_image_2 …) in slot order."""
        await self.ad.generate(self.req(upload_images={"input_image": self.png,
                                                       "input_image_2": self.png}))
        run = self.calls("/Run/google/nano-banana")[0]
        self.assertTrue(run.headers["content-type"].startswith("multipart/form-data"))
        raw = run.content
        self.assertEqual(raw.count(b'name="inputImage"'), 2)
        self.assertNotIn(SAMPLE.encode(), raw)

    async def test_required_prompt_missing_is_input_error_before_any_post(self):
        """A required text param has a SAMPLE default — it must be refused (400), never filled."""
        with self.assertRaises(wiro.INPUT_ERROR):
            await self.ad.generate(self.req(inputs={}))
        self.assertEqual(self.calls("/Run/google/nano-banana"), [])

    async def test_failed_exit_code_is_final(self):
        """task_postprocess_end with pexit != "0" is a failed run, not a success without files."""
        self.details = [{"status": "task_postprocess_end", "pexit": "1",
                         "debugerror": "CUDA out of memory", "totalcost": "0"}]
        with self.assertRaises(RuntimeError) as caught:
            await self.ad.generate(self.req())
        self.assertIn("CUDA out of memory", str(caught.exception))
        self.assertEqual(len(self.calls("/Run/google/nano-banana")), 1)

    async def test_refusals_that_created_nothing_keep_failover_semantics(self):
        """Code 96 (concurrency limit, HTTP 200) created no task → CloudTaskRetryable;
        code 97 (balance) → CloudNoCredits. Neither may look like a billed failure."""
        self.run_answer = httpx.Response(200, json={"result": False, "errors": [
            {"code": 96, "message": "concurrency limit"}]})
        with self.assertRaises(adapters.CloudTaskRetryable):
            await self.ad.generate(self.req())
        self.run_answer = httpx.Response(200, json={"result": False, "errors": [
            {"code": 97, "message": "Insufficient balance"}]})
        with self.assertRaises(adapters.CloudNoCredits):
            await self.ad.generate(self.req())

    async def test_uncertain_submit_is_never_reposted(self):
        """The Run went out and the answer was lost: wiro may bill the task — final,
        one POST only, and the trace says create_unconfirmed."""
        self.run_error = httpx.ReadTimeout("lost")
        req = self.req()
        with self.assertRaises(RuntimeError):
            await self.ad.generate(req)
        self.assertEqual(len(self.calls("/Run/google/nano-banana")), 1)
        self.assertTrue(req.cloud_trace.get("create_unconfirmed"))

    async def test_secret_is_redacted_from_errors(self):
        """Neither half of KEY:SECRET may reach a job row or log through an error text."""
        self.run_answer = httpx.Response(500, text="bad key KEY123 with SECRET456")
        with self.assertRaises(RuntimeError) as caught:
            await self.ad.generate(self.req())
        self.assertNotIn("KEY123", str(caught.exception))
        self.assertNotIn("SECRET456", str(caught.exception))

    async def test_cancelled_job_kills_its_task(self):
        """A cancelled gateway job sends /Task/Kill for ITS task (cancelled tasks are not
        billed) and records that a stop was requested — never that wiro stopped."""
        self.block_poll = asyncio.Event()
        req = self.req()
        task = asyncio.create_task(self.ad.generate(req))
        await asyncio.wait_for(self.block_poll.wait(), 5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        kills = self.calls("/Task/Kill")
        self.assertEqual(len(kills), 1)
        self.assertEqual(json.loads(kills[0].content), {"taskid": "2221"})
        self.assertTrue(self.meta[-1].get("stop_requested"))


def _done():
    return {"status": "task_postprocess_end", "pexit": "0", "totalcost": "0.039",
            "outputs": [{"name": "0.png", "contenttype": "image/png",
                         "url": "https://cdn.test/out/0.png"}]}


if __name__ == "__main__":
    unittest.main()
