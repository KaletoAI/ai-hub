"""Venice catalogs must stay atomic and purpose-separated; paid media must never
be re-submitted after uncertainty, lose queue recovery data, or leak keys to assets.
All calls use MockTransport, never a live or paid Venice endpoint.
"""
import asyncio
import base64
import copy
import io
import json
import unittest
import wave
from unittest.mock import AsyncMock

import httpx
from fastapi import HTTPException
from PIL import Image
from starlette.requests import Request

import adapters
import venice


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (12, 45, 78)).save(buf, "PNG")
    return buf.getvalue()


def _wav():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 8000)
    return buf.getvalue()


def _mp4():
    """Reuse the repository's playable, ffmpeg-generated H.264 frame fixture."""
    from tests.test_nanogpt_media_adapter import _mp4 as fixture
    return fixture()


def _m4a():
    """Real 0.125-second mono AAC/M4A from ffmpeg anullsrc, bitexact fragmented MP4."""
    return base64.b64decode(
        "AAAAIGZ0eXBNNEEgAAACAE00QSBpc282aXNvbWlzbzIAAAKEbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAAAAAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAat0cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAEBAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAFHbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAAfQAAAAABVxAAAAAAALWhkbHIAAAAAAAAAAHNvdW4AAAAAAAAAAAAAAABTb3VuZEhhbmRsZXIAAAAA8m1pbmYAAAAQc21oZAAAAAAAAAAAAAAAJGRpbmYAAAAcZHJlZgAAAAAAAAABAAAADHVybCAAAAABAAAAtnN0YmwAAABqc3RzZAAAAAAAAAABAAAAWm1wNGEAAAAAAAAAAQAAAAAAAAAAAAEAEAAAAAAfQAAAAAAANmVzZHMAAAAAA4CAgCUAAQAEgICAF0AVAAAAAAA+gAAAPoAFgICABRWIVuUABoCAgAECAAAAEHN0dHMAAAAAAAAAAAAAABBzdHNjAAAAAAAAAAAAAAAUc3RzegAAAAAAAAAAAAAAAAAAABBzdGNvAAAAAAAAAAAAAAAobXZleAAAACB0cmV4AAAAAAAAAAEAAAABAAAAAAAAAAAAAAAAAAAAPXVkdGEAAAA1bWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAIaWxzdAAAAHRtb29mAAAAEG1maGQAAAAAAAAAAQAAAFx0cmFmAAAAJHRmaGQAAAA5AAAAAQAAAAAAAAKkAAAEAAAAAAQCAAAAAAAAFHRmZHQBAAAAAAAAAAAAAAAAAAAcdHJ1bgAAAQEAAAACAAAAfAAABAAAAAPoAAAAEG1kYXQBGCAHARggBwAAAENtZnJhAAAAK3RmcmEBAAAAAAAAAQAAAAAAAAABAAAAAAAAAAAAAAAAAAACpAEBAQAAABBtZnJvAAAAAAAAAEM=")


def _row(mid, kind, **spec):
    return {"id": mid, "type": kind, "model_spec": spec}


CATALOGS = {
    "text": [_row("chat", "text", pricing={"input": {"usd": 2},
        "output": {"usd": 8}, "cache_input": {"usd": 0.2}},
        availableContextTokens=12345), _row("unknown-chat", "text")],
    "tts": [_row("speech", "tts", pricing={"input": {"usd": 4}},
        voices=["af_sky"], supported_formats=["wav"], default_format="wav"),
        _row("clone", "tts", supported_formats=["wav"], default_format="wav",
             voice_cloning={"accepted_formats": ["wav"], "mode": "zero_shot"})],
    "embedding": [_row("embed", "embedding", pricing={"input": {"usd": 0.1}})],
    "image": [_row("image", "image", constraints={"maxVariations": 4})],
    "inpaint": [_row("edit", "inpaint", constraints={"maxInputImages": 3})],
    "upscale": [_row("upscale", "upscale")],
    "video": [_row("video", "video", constraints={"durations": ["5s"]})],
    "music": [_row("music", "music"), _row("changer", "music", voice_changer=True)],
    "asr": [_row("asr", "asr")],
}


class VeniceAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.slots = []
        self.meta = []
        self.progress = []
        self.catalogs = copy.deepcopy(CATALOGS)
        self.catalog_fail = None
        self.block_kind = None
        self.account_data = {"data": {"apiTier": {"id": "paid", "isCharged": True},
            "accessPermitted": True, "balances": {"USD": 5, "DIEM": 0, "BUNDLED_CREDITS": 0.8652965}}}
        self.account_status, self.account_raw, self.account_error = 200, None, None
        self.block_account = False
        self.entered, self.resume = asyncio.Event(), asyncio.Event()
        self.png, self.wav = _png(), _wav()
        self.mp4 = _mp4()
        self.asset_data, self.asset_mime = self.wav, "audio/wav"
        self.inline_video = False
        self.audio_data, self.audio_mime = self.wav, "audio/wav"
        self.submit_error = None
        self.queue = {"model": "video", "queue_id": "queue-1"}
        self.retrieve = None
        self.asset_fail = False
        self.asset_redirect = None
        self.binary_image = False
        self.transcript_text = False
        self.cancel_poll = False
        self.asset_failures = 0
        self.voice_error = None
        for name, value in (("_CLOUD_DOWNLOAD_BACKOFF_S", 0), ("_VENICE_HICCUP_S", 0.001)):
            saved = getattr(adapters, name)
            setattr(adapters, name, value)
            self.addCleanup(setattr, adapters, name, saved)
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
        self.backend = {"type": "venice", "name": "account", "url": "https://provider.test/api",
                        "api_key": "secret-key", "poll_interval": 0, "max_wait": 1}
        self.ad = adapters.make_adapter(self.backend, self.ctx)

    async def asyncSetUp(self):
        await self.ad.discover(self.client)
        self.requests.clear()

    async def asyncTearDown(self):
        await self.client.aclose()

    async def handle(self, request):
        self.requests.append(request)
        path = request.url.path
        if request.url.host == "assets.test":
            if self.asset_redirect and path.endswith("/redirect"):
                return httpx.Response(302, headers={"Location": self.asset_redirect})
            if self.asset_failures:
                self.asset_failures -= 1
                return httpx.Response(503)
            return httpx.Response(503 if self.asset_fail else 200, content=self.asset_data,
                                  headers={"Content-Type": self.asset_mime})
        if path.endswith("/api_keys/rate_limits"):
            if self.block_account:
                self.entered.set()
                await self.resume.wait()
            if self.account_error:
                raise self.account_error
            if request.headers.get("Authorization") != "Bearer secret-key":
                return httpx.Response(401, text="secret-key should not be rendered")
            if self.account_raw is not None:
                return httpx.Response(self.account_status, text=self.account_raw)
            return httpx.Response(self.account_status, json=self.account_data)
        if path.endswith("/models"):
            kind = request.url.params["type"]
            if kind == self.block_kind:
                self.entered.set()
                await self.resume.wait()
            if kind == self.catalog_fail:
                return httpx.Response(503)
            return httpx.Response(200, json={"data": self.catalogs[kind]})
        if path.endswith("/complete"):
            return httpx.Response(200, json={"success": True})
        if path.endswith("/retrieve"):
            if self.cancel_poll:
                raise asyncio.CancelledError()
            retrieve = self.retrieve.pop(0) if isinstance(self.retrieve, list) else self.retrieve
            if isinstance(retrieve, Exception):
                raise retrieve
            if isinstance(retrieve, httpx.Response):
                return retrieve
            if retrieve is not None:
                return httpx.Response(200, json=retrieve)
            if self.inline_video:
                return httpx.Response(200, content=self.mp4, headers={"Content-Type": "video/mp4"})
            return httpx.Response(200, content=self.audio_data, headers={"Content-Type": self.audio_mime})
        if path.endswith("/voices"):
            if self.voice_error:
                raise self.voice_error
            return httpx.Response(200, json={"id": "vv_voice_test", "model": "clone"})
        if self.submit_error:
            raise self.submit_error
        if path.endswith("/queue"):
            return httpx.Response(200, json=self.queue)
        if path.endswith("/transcriptions"):
            if self.transcript_text:
                return httpx.Response(200, text="Venice transcript")
            return httpx.Response(200, json={"text": "Venice transcript", "duration": 1})
        if path.endswith("/speech"):
            return httpx.Response(200, content=self.wav, headers={"Content-Type": "audio/wav"})
        if path.endswith("/generate"):
            if self.binary_image:
                return httpx.Response(200, content=self.png, headers={"Content-Type": "image/png"})
            return httpx.Response(200, json={"images": [base64.b64encode(self.png).decode()]})
        if "/image/" in path:
            return httpx.Response(200, content=self.png, headers={"Content-Type": "image/png"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    def req(self, endpoint="images", model="image", **kw):
        self.queue["model"] = model
        return adapters.NormalizedRequest(real_model=model, cloud={"endpoint": endpoint, "options": {}},
            inputs=kw.pop("inputs", {"prompt": "A tree"} if endpoint in
                          ("images", "edit", "video", "music", "speech") else {}),
            job_id="job-1", **kw)

    async def test_discovery_keeps_native_types_and_only_token_pricing(self):
        caps = await self.ad.discover(self.client)
        self.assertIsInstance(self.ad, adapters.VeniceAdapter)
        self.assertIn("venice", adapters.DYNAMIC_CATALOG_TYPES)
        self.assertEqual(caps.pricing, {"chat": {"input": 2.0, "output": 8.0, "cache_read": 0.2},
                                       "embed": {"input": 0.1, "output": 0.0}})
        self.assertEqual(caps.context["chat"], 12345)
        self.assertEqual(set(self.ad.model_catalogs["tts"]), {"speech", "clone"})
        self.assertEqual(set(self.ad.model_catalogs["speech"]), {"speech", "clone"})
        self.assertEqual(set(self.ad.model_catalogs["music"]), {"music"})
        self.assertEqual(set(self.ad.model_catalogs["voice-changer"]), {"changer"})
        self.assertEqual(set(self.ad.model_catalogs["transcription"]), {"asr"})
        self.assertNotIn("speech", caps.pricing)
        self.assertFalse(hasattr(self.ad, "credits"))

    async def test_account_status_is_authenticated_and_independent_of_model_prices(self):
        """An inference key's account flags, never zero/absent rates, label billing."""
        self.catalogs["text"][0]["model_spec"]["pricing"] = {
            "input": {"usd": 0}, "output": {"usd": 0}}
        await self.ad.discover(self.client)
        self.assertEqual(self.ad.account_info["state"], "credits")
        self.assertEqual(self.ad.account_info["balances"],
                         {"USD": 5, "DIEM": 0, "BUNDLED_CREDITS": 0.8652965})
        self.assertGreater(self.ad.account_info["checked_at"], 0)
        self.account_data["data"]["apiTier"]["isCharged"] = False
        await self.ad.discover(self.client)
        self.assertEqual(self.ad.account_info["state"], "included")
        self.account_data = {"data": {"accessPermitted": True, "apiTier": {"id": "pro"}}}
        await self.ad.discover(self.client)
        self.assertEqual(self.ad.account_info["state"], "unknown")
        self.assertTrue(self.ad.model_catalogs["text"])
        self.assertTrue(all(r.method == "GET" for r in self.requests))
        self.assertFalse(any("/billing/" in r.url.path for r in self.requests))

    async def test_provider_account_fields_cannot_echo_gateway_credentials(self):
        """Even an echoed key in a recognized label must not reach console/health."""
        self.account_data["data"]["apiTier"]["id"] = "secret-key"
        self.account_data["data"]["api_key"] = "secret-key"
        self.account_data["data"]["error"] = "private account details"
        await self.ad.discover(self.client)
        self.assertEqual(self.ad.account_info["state"], "credits")
        self.assertNotIn("api_tier", self.ad.account_info)
        self.assertNotIn("secret-key", str(self.ad.account_info))
        self.assertNotIn("private account", str(self.ad.account_info))

    async def test_account_failures_clear_assurance_but_keep_catalog_and_chat_usable(self):
        """Optional HTTP/JSON/transport failures must not disable a working account."""
        for status, raw, error in (
                (401, '{"api_key":"secret-key","error":"private account"}', None),
                (503, "private account secret-key", None),
                (200, "not JSON secret-key", None),
                (200, '{"data":[]}', None),
                (200, None, httpx.ReadTimeout("private account secret-key"))):
            with self.subTest(status=status, raw=raw, error=type(error).__name__):
                self.account_status, self.account_raw, self.account_error = 200, None, None
                self.account_data["data"]["apiTier"]["isCharged"] = False
                await self.ad.discover(self.client)
                self.assertEqual(self.ad.account_info["state"], "included")
                self.assertNotIn("error", self.ad.account_info)
                self.account_status, self.account_raw, self.account_error = status, raw, error
                caps = await self.ad.discover(self.client)
                self.assertEqual(self.ad.account_info["state"], "unknown")
                self.assertEqual(self.ad.account_info["balances"], {})
                self.assertIsNone(self.ad.account_info["access_permitted"])
                self.assertGreater(self.ad.account_info["checked_at"], 0)
                if status >= 400:
                    self.assertEqual(self.ad.account_info["error"], "HTTP " + str(status))
                self.assertNotIn("secret-key", str(self.ad.account_info))
                self.assertNotIn("private account", str(self.ad.account_info))
                self.assertIn("chat", caps.models)
                self.assertIn("image", self.ad.model_catalogs["images"])
                raw_request = Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1)})
                await self.ad.dispatch(adapters.NormalizedRequest(
                    path="/v1/chat/completions", real_model="chat", raw=raw_request,
                    body={"model": "chat", "messages": []}))

    async def test_pending_account_check_retains_last_snapshot_until_failure_settles(self):
        """Do not briefly claim unknown/free or retain inclusion after a failed check."""
        self.account_data["data"]["apiTier"]["isCharged"] = False
        await self.ad.discover(self.client)
        before, catalogs = copy.deepcopy(self.ad.account_info), copy.deepcopy(self.ad.model_catalogs)
        self.block_account, self.account_status = True, 401
        pending = asyncio.create_task(self.ad.discover(self.client))
        try:
            await asyncio.wait_for(self.entered.wait(), 2)
            self.assertEqual(self.ad.account_info, before)
            self.assertEqual(self.ad.model_catalogs, catalogs)
        finally:
            self.resume.set()
        await pending
        self.assertEqual(self.ad.account_info["state"], "unknown")
        self.assertEqual(self.ad.account_info["error"], "HTTP 401")
        self.assertEqual(self.ad.model_catalogs, catalogs)

    async def test_account_snapshot_settles_even_when_catalog_refresh_fails(self):
        """Catalog health and account billing assurance have independent outcomes."""
        self.catalog_fail, self.account_status = "asr", 401
        with self.assertRaises(httpx.HTTPStatusError):
            await self.ad.discover(self.client)
        self.assertEqual(self.ad.account_info["state"], "unknown")
        self.assertEqual(self.ad.account_info["error"], "HTTP 401")
        self.assertEqual(self.ad.model_catalogs, {})
        self.assertIn("image", self.ad.display_media_catalogs["images"])

    async def test_account_adoption_requires_same_url_and_key_and_copies_balances(self):
        """A rebinding must not show another account's funds or share mutable state."""
        for changed in ({"api_key": "other-key"}, {"url": "https://elsewhere.test/api"}):
            with self.subTest(changed=changed):
                other = adapters.make_adapter({**self.backend, **changed}, self.ctx)
                other.adopt_state(self.ad)
                self.assertEqual(other.account_info["state"], "unknown")
                self.assertEqual(other.account_info["balances"], {})
                self.assertNotIn("checked_at", other.account_info)
        other = adapters.make_adapter({**self.backend, "models_deny": "image",
                                       "name": "renamed"}, self.ctx)
        other.adopt_discovery(self.ad)
        self.assertEqual(other.account_info, self.ad.account_info)
        self.assertEqual(other.model_catalogs["images"], {})
        other.account_info["balances"]["USD"] = 0
        self.assertEqual(self.ad.account_info["balances"]["USD"], 5)

    async def test_pending_refresh_retains_snapshot_failure_clears_callable_only(self):
        before = copy.deepcopy(self.ad.model_catalogs)
        self.block_kind, self.catalog_fail = "video", "asr"
        pending = asyncio.create_task(self.ad.discover(self.client))
        await asyncio.wait_for(self.entered.wait(), 2)
        self.assertEqual(self.ad.model_catalogs, before)
        self.resume.set()
        with self.assertRaises(httpx.HTTPStatusError):
            await pending
        self.assertEqual(self.ad.model_catalogs, {})
        self.assertEqual(self.ad._media.model_catalogs, {})
        self.assertEqual(self.ad.display_media_catalogs["images"], before["images"])
        with self.assertRaises(venice.INPUT_ERROR):
            await self.ad.generate(self.req())

    async def test_adoption_checks_account_and_reapplies_text_and_name_filters(self):
        other = adapters.make_adapter({**self.backend, "models_deny": "image,speech",
                                      "serverless_only": True, "chat_only": True}, self.ctx)
        other.adopt_state(self.ad)
        self.assertEqual(set(other.model_catalogs["text"]), {"chat"})
        self.assertEqual(set(other.model_catalogs["speech"]), {"clone"})
        self.assertEqual(other.model_catalogs["images"], {})
        foreign = adapters.make_adapter({**self.backend, "api_key": "other"}, self.ctx)
        foreign.adopt_state(self.ad)
        self.assertEqual(foreign.model_catalogs, {})
        self.assertEqual(foreign.display_media_catalogs, {})

    async def test_dispatch_routes_keep_openai_paths_but_refuse_wrong_purpose(self):
        raw = Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1)})
        for path, model in (("/v1/chat/completions", "chat"),
                            ("/v1/embeddings", "embed"), ("/v1/audio/speech", "speech")):
            await self.ad.dispatch(adapters.NormalizedRequest(path=path, real_model=model,
                raw=raw, body={"model": model, "input": "hello", "messages": []}))
            self.assertEqual(self.requests[-1].url.path, "/api" + path)
        for path, model in (("/v1/chat/completions", "image"),
                            ("/v1/embeddings", "speech"), ("/v1/audio/speech", "chat")):
            with self.assertRaises(HTTPException) as refused:
                await self.ad.dispatch(adapters.NormalizedRequest(path=path, real_model=model))
            self.assertEqual(refused.exception.status_code, 400)
        self.assertTrue(all(bid == "venice:account" for _, bid in self.slots))

    async def test_sync_images_and_binary_operations_deliver_valid_images(self):
        for endpoint, model in (("images", "image"), ("edit", "edit"),
                                ("upscale", "upscale"), ("background-remove", "background-remove")):
            req = self.req(endpoint, model, upload_images={} if endpoint == "images" else {"input_image": self.png})
            result = await self.ad.generate(req)
            self.assertEqual((result.blobs[0].data, result.blobs[0].mime), (self.png, "image/png"))
            if endpoint in ("upscale", "background-remove"):
                self.assertNotIn("model", json.loads(self.requests[-1].content))
        self.assertEqual(len([x for x in self.slots if x[0] == "inc"]), 4)
        self.assertEqual(len([x for x in self.slots if x[0] == "dec"]), 4)

    async def test_audio_queue_id_persisted_before_retrieve_and_cleanup(self):
        self.queue = {"model": "music", "queue_id": "queue-1"}
        result = await self.ad.generate(self.req("music", "music"))
        self.assertEqual(result.blobs[0].data, self.wav)
        self.assertEqual(self.meta[0]["cloud_task_id"], "queue-1")
        self.assertEqual(self.meta[0]["model"], "music")
        poll = next(r for r in self.requests if r.url.path.endswith("/retrieve"))
        self.assertEqual(json.loads(poll.content), {"model": "music", "queue_id": "queue-1"})
        self.assertTrue(self.requests[-1].url.path.endswith("/audio/complete"))

    async def test_private_url_is_persisted_and_download_never_gets_bearer(self):
        # Audio body deliberately mismatches video: validated output must fail, not clean up.
        self.queue["download_url"] = "https://assets.test/private.mp4?token=private"
        self.retrieve = {"status": "COMPLETED"}
        req = self.req("video", "video", params={"duration": "5s"})
        with self.assertRaises(RuntimeError):
            await self.ad.generate(req)
        self.assertEqual(req.cloud_trace["download_url"], self.queue["download_url"])
        self.assertEqual(self.meta[0]["download_url"], self.queue["download_url"])
        asset = next(r for r in self.requests if r.url.host == "assets.test")
        self.assertNotIn("authorization", asset.headers)
        self.assertNotIn("x-api-key", asset.headers)
        self.assertFalse(any(r.url.path.endswith("/complete") for r in self.requests))

    async def test_inline_and_private_video_both_deliver_real_bytes(self):
        self.inline_video = True
        result = await self.ad.generate(self.req("video", "video", params={"duration": "5s"}))
        self.assertEqual((result.blobs[0].data, result.blobs[0].mime), (self.mp4, "video/mp4"))
        self.requests.clear()
        self.meta.clear()
        self.inline_video = False
        self.asset_data, self.asset_mime = self.mp4, "video/mp4"
        self.queue["download_url"] = "https://assets.test/private.mp4?token=private"
        self.retrieve = {"status": "COMPLETED"}
        result = await self.ad.generate(self.req("video", "video", params={"duration": "5s"}))
        self.assertEqual(result.blobs[0].data, self.mp4)
        self.assertEqual(self.meta[0]["download_url"], self.queue["download_url"])
        self.assertEqual(self.requests[-2].url.host, "assets.test")
        self.assertTrue(self.requests[-1].url.path.endswith("/video/complete"))

    async def test_voice_changer_uses_own_multipart_queue(self):
        req = self.req("voice-changer", "changer", inputs={},
                       upload_files={"input_audio": ("source.wav", self.wav)})
        result = await self.ad.generate(req)
        queued = next(r for r in self.requests if r.url.path.endswith("/queue"))
        self.assertTrue(queued.url.path.endswith("/audio/voice-changer/queue"))
        self.assertIn("multipart/form-data", queued.headers["content-type"])
        self.assertIn(self.wav, queued.content)
        self.assertEqual(result.blobs[0].kind, "audio")

    async def test_preconnect_is_the_only_retryable_submit_failure(self):
        self.submit_error = httpx.ConnectError("not connected")
        req = self.req()
        with self.assertRaises(httpx.ConnectError):
            await self.ad.generate(req)
        self.assertFalse(req.cloud_trace.get("create_unconfirmed"))
        self.assertEqual(self.slots, [("inc", "venice:account"), ("dec", "venice:account")])


    async def test_uncertain_submit_and_poll_failure_never_become_retryable(self):
        for error in (httpx.ReadTimeout("lost"), httpx.WriteError("partial write")):
            self.submit_error = error
            req = self.req()
            with self.assertRaises(RuntimeError):
                await self.ad.generate(req)
            self.assertTrue(req.cloud_trace["create_unconfirmed"])
        self.submit_error = None
        self.retrieve = httpx.ConnectError("poll lost")
        req = self.req("music", "music")
        with self.assertRaises(RuntimeError):
            await self.ad.generate(req)
        self.assertEqual(req.cloud_trace["cloud_task_id"], "queue-1")
        self.assertEqual(len([r for r in self.requests if r.url.path.endswith("/queue")]), 1)
        self.assertFalse(any(r.url.path.endswith("/complete") for r in self.requests))
        self.assertEqual(len([x for x in self.slots if x[0] == "inc"]),
                         len([x for x in self.slots if x[0] == "dec"]))

    async def test_cancelled_poll_releases_slot_without_deleting_vendor_media(self):
        self.cancel_poll = True
        req = self.req("music", "music")
        with self.assertRaises(asyncio.CancelledError):
            await self.ad.generate(req)
        self.assertEqual(req.cloud_trace["cloud_task_id"], "queue-1")
        self.assertFalse(any(r.url.path.endswith("/complete") for r in self.requests))
        self.assertEqual(self.slots, [("inc", "venice:account"), ("dec", "venice:account")])
        self.assertIsNone(self.progress[-1])

    async def test_transcription_sends_multipart_and_delivers_json_artifact(self):
        req = self.req("transcription", "asr", inputs={},
                       upload_files={"input_audio": ("meeting.wav", self.wav)})
        result = await self.ad.generate(req)
        sent = self.requests[-1]
        self.assertIn("multipart/form-data", sent.headers["content-type"])
        self.assertIn(self.wav, sent.content)
        self.assertIn(b'name="model"', sent.content)
        self.assertIn(b'filename="gw_job-1_input_audio.wav"', sent.content)
        self.assertEqual(result.blobs[0].mime, "application/json")
        self.assertEqual(json.loads(result.blobs[0].data)["text"], "Venice transcript")

    async def test_cloned_speech_uploads_reference_then_uses_returned_handle(self):
        req = self.req("speech", "clone", inputs={"prompt": "Hello there"},
                       upload_files={"input_audio": ("reference.wav", self.wav)})
        result = await self.ad.generate(req)
        self.assertEqual(result.blobs[0].data, self.wav)
        voice, speech = self.requests[-2:]
        self.assertTrue(voice.url.path.endswith("/voices"))
        self.assertIn(self.wav, voice.content)
        self.assertEqual(json.loads(speech.content)["voice"], "vv_voice_test")
        self.assertEqual(json.loads(speech.content)["input"], "Hello there")
        self.assertNotIn("input_audio", json.loads(speech.content))
        self.assertEqual(req.cloud_trace["voice_id"], "vv_voice_test")


    async def test_malformed_catalog_clears_previous_complete_state(self):
        self.catalogs["asr"] = [{"id": "bad", "model_spec": []}]
        with self.assertRaises(ValueError):
            await self.ad.discover(self.client)
        self.assertEqual(self.ad.model_catalogs, {})
        self.assertIn("images", self.ad.display_media_catalogs)

    async def test_missing_invalid_and_character_prices_never_become_free_tokens(self):
        for price in (None, {}, {"input": {"usd": True}, "output": {"usd": 3}},
                      {"input": {"usd": -1}, "output": {"usd": 3}},
                      {"input": {"usd": "NaN"}, "output": {"usd": 3}},
                      {"input": {"diem": 2}, "output": {"diem": 3}}):
            with self.subTest(price=price):
                self.catalogs["text"][0]["model_spec"]["pricing"] = price
                caps = await self.ad.discover(self.client)
                self.assertNotIn("chat", caps.pricing)
                self.assertNotIn("speech", caps.pricing)
        self.catalogs["text"][0]["model_spec"]["pricing"] = {
            "input": {"usd": 0}, "output": {"usd": 0}}
        caps = await self.ad.discover(self.client)
        self.assertEqual(caps.pricing["chat"], {"input": 0.0, "output": 0.0})

    async def test_asset_redirects_strip_injected_headers_and_refuse_lan_targets(self):
        self.client.headers.update({"Authorization": "Bearer secret-key",
                                    "x-api-key": "secret-key", "Cookie": "session=private"})
        self.asset_redirect = "https://assets.test/private.wav"
        response = await self.ad._media._asset(self.client, "https://assets.test/redirect")
        self.assertEqual(response.content, self.wav)
        for request in self.requests:
            self.assertNotIn("authorization", request.headers)
            self.assertNotIn("x-api-key", request.headers)
            self.assertNotIn("cookie", request.headers)
        self.requests.clear()
        self.asset_redirect = "http://127.0.0.1/private"
        with self.assertRaises(RuntimeError):
            await self.ad._media._asset(self.client, "https://assets.test/redirect")
        self.assertEqual(len(self.requests), 1)

    async def test_processing_poll_does_not_submit_again(self):
        self.retrieve = [{"status": "PROCESSING"}, None]
        result = await self.ad.generate(self.req("music", "music"))
        self.assertEqual(result.blobs[0].data, self.wav)
        self.assertEqual(len([r for r in self.requests if r.url.path.endswith("/queue")]), 1)
        self.assertEqual(len([r for r in self.requests if r.url.path.endswith("/retrieve")]), 2)

    async def test_binary_generate_and_multi_edit_use_real_native_paths(self):
        self.binary_image = True
        result = await self.ad.generate(self.req(params={"return_binary": True}))
        self.assertEqual(result.blobs[0].data, self.png)
        result = await self.ad.generate(self.req("edit", "edit", upload_images={
            "input_image": self.png, "input_image_2": self.png}))
        sent = self.requests[-1]
        self.assertTrue(sent.url.path.endswith("/image/multi-edit"))
        self.assertEqual(json.loads(sent.content)["modelId"], "edit")
        self.assertEqual(len(json.loads(sent.content)["images"]), 2)
        self.assertEqual(result.blobs[0].data, self.png)

    async def test_transcription_text_is_downloadable_not_json_wrapped(self):
        self.transcript_text = True
        result = await self.ad.generate(self.req("transcription", "asr",
            params={"response_format": "text"}, upload_files={"input_audio": ("meeting.wav", self.wav)}))
        self.assertEqual((result.blobs[0].data, result.blobs[0].mime, result.blobs[0].name),
                         (b"Venice transcript", "text/plain", "transcript.txt"))

    async def test_standard_media_speech_uses_native_input_and_voice(self):
        result = await self.ad.generate(self.req("speech", "speech",
            inputs={"input": "Native speech text"}, params={"voice": "af_sky"}))
        self.assertEqual(result.blobs[0].data, self.wav)
        body = json.loads(self.requests[-1].content)
        self.assertEqual((body["input"], body["voice"]), ("Native speech text", "af_sky"))
        self.assertFalse(any(r.url.path.endswith("/voices") for r in self.requests))

    async def test_short_image_result_is_final_and_preserves_native_count(self):
        req = self.req(params={"n": 2})
        with self.assertRaises(RuntimeError):
            await self.ad.generate(req)
        body = json.loads(self.requests[-1].content)
        self.assertEqual(body["variants"], 2)
        self.assertNotIn("n", body)
        self.assertEqual(len([r for r in self.requests if r.method == "POST"]), 1)

    async def test_failed_private_download_keeps_paid_queue_without_cleanup(self):
        self.asset_fail = True
        self.queue["download_url"] = "https://assets.test/private.mp4"
        self.retrieve = {"status": "COMPLETED"}
        req = self.req("video", "video", params={"duration": "5s"})
        with self.assertRaises(RuntimeError):
            await self.ad.generate(req)
        self.assertEqual(req.cloud_trace["queue_id"], "queue-1")
        self.assertEqual(len([r for r in self.requests if r.url.path.endswith("/queue")]), 1)
        self.assertFalse(any(r.url.path.endswith("/complete") for r in self.requests))

    async def test_malformed_queue_success_never_submits_twice(self):
        req = self.req("music", "music")
        self.queue.pop("queue_id")
        with self.assertRaises(RuntimeError):
            await self.ad.generate(req)
        self.assertTrue(req.cloud_trace["create_unconfirmed"])
        self.assertEqual(len([r for r in self.requests if r.method == "POST"]), 1)


    async def test_completed_json_url_is_persisted_before_download(self):
        # The Topaz guide documents retrieve.url, unlike private queue.download_url.
        self.asset_data, self.asset_mime = self.mp4, "video/mp4"
        self.retrieve = {"status": "COMPLETED", "url": "https://assets.test/upscaled.mp4"}
        req = self.req("video", "video", params={"duration": "5s"})
        result = await self.ad.generate(req)
        self.assertEqual(result.blobs[0].data, self.mp4)
        self.assertEqual(req.cloud_trace["download_url"], self.retrieve["url"])
        self.assertEqual(self.meta[-1]["download_url"], self.retrieve["url"])


    async def test_advertised_m4a_music_survives_paid_retrieval_and_cleanup(self):
        self.catalogs["music"][0]["model_spec"].update(
            supported_formats=["m4a"], default_format="m4a")
        await self.ad.discover(self.client)
        self.requests.clear()
        self.audio_data, self.audio_mime = _m4a(), "audio/mp4"
        result = await self.ad.generate(self.req("music", "music"))
        self.assertEqual((result.blobs[0].data, result.blobs[0].mime, result.blobs[0].name),
                         (self.audio_data, "audio/mp4", "audio.m4a"))
        self.assertEqual(len([r for r in self.requests if r.url.path.endswith("/queue")]), 1)
        self.assertTrue(self.requests[-1].url.path.endswith("/audio/complete"))

    async def test_iso_audio_aliases_and_octet_stream_keep_audio_extension(self):
        for mime in ("audio/mp4", "audio/x-m4a", "audio/m4a", "application/octet-stream"):
            with self.subTest(mime=mime):
                blob = await asyncio.to_thread(adapters.VeniceMediaAdapter._binary_blob,
                                               _m4a(), mime, "music")
                self.assertEqual((blob.kind, blob.name), ("audio", "audio.m4a"))
                self.assertEqual(blob.mime, "audio/mp4" if mime == "application/octet-stream" else mime)
        blob = await asyncio.to_thread(adapters.VeniceMediaAdapter._binary_blob,
                                       self.wav, "application/octet-stream", "music")
        self.assertEqual((blob.kind, blob.mime, blob.name), ("audio", "audio/wav", "audio.wav"))

    async def test_iso_track_types_and_unsupported_mime_cannot_cross_audio_video(self):
        for data, mime, endpoint in (
                (_m4a(), "video/mp4", "video"),
                (_m4a(), "application/octet-stream", "video"),
                (self.mp4, "audio/mp4", "music"),
                (self.mp4, "application/octet-stream", "music"),
                (b"not an audio container", "application/octet-stream", "music"),
                (_m4a(), "application/pdf", "music"),
                (_m4a()[:100], "audio/mp4", "music")):
            with self.subTest(mime=mime, endpoint=endpoint):
                with self.assertRaises(RuntimeError):
                    await asyncio.to_thread(adapters.VeniceMediaAdapter._binary_blob,
                                            data, mime, endpoint)
        blob = await asyncio.to_thread(adapters.VeniceMediaAdapter._binary_blob,
                                       self.mp4, "application/octet-stream", "video")
        self.assertEqual((blob.kind, blob.mime, blob.name), ("video", "video/mp4", "video.mp4"))

    async def test_decoded_image_dimensions_gate_before_paid_submission(self):
        metadata = self.ad.model_catalogs["video"]["video"]
        metadata["model_spec"]["constraints"].update(
            model_type="image-to-video", reference_image_min_short_side_pixels=300)
        req = self.req("video", "video", params={"duration": "5s"},
                       upload_images={"input_image": self.png})
        with self.assertRaises(venice.INPUT_ERROR):
            await self.ad.generate(req)
        self.assertFalse(any(r.method == "POST" for r in self.requests))
        image = io.BytesIO()
        Image.new("RGB", (512, 512), (12, 45, 78)).save(image, "PNG")
        self.inline_video = True
        req.upload_images["input_image"] = image.getvalue()
        result = await self.ad.generate(req)
        self.assertEqual(result.blobs[0].data, self.mp4)

    async def test_retrieve_hiccups_and_late_bytes_never_discard_a_paid_queue(self):
        """One 429/503/dropped poll, a malformed body or a JSON COMPLETED before the bytes
        used to fail a billed video/music job; retrieve is free, so it is polled again."""
        self.queue = {"model": "music", "queue_id": "queue-1"}
        self.retrieve = [httpx.Response(503), httpx.Response(429), httpx.ReadError("dropped"),
                         httpx.Response(200, text="<html>"), {"status": "COMPLETED"}, None]
        result = await self.ad.generate(self.req("music", "music"))
        self.assertEqual(result.blobs[0].data, self.wav)
        self.assertEqual(len([r for r in self.requests if r.url.path.endswith("/queue")]), 1)
        self.assertEqual(len([r for r in self.requests if r.url.path.endswith("/retrieve")]), 6)
        self.assertTrue(self.requests[-1].url.path.endswith("/audio/complete"))

    async def test_three_client_errors_in_a_row_are_a_final_verdict(self):
        self.queue = {"model": "music", "queue_id": "queue-1"}
        self.retrieve = [httpx.Response(404), httpx.Response(404), httpx.Response(404), None]
        with self.assertRaises(RuntimeError):
            await self.ad.generate(self.req("music", "music"))
        self.assertEqual(len([r for r in self.requests if r.url.path.endswith("/retrieve")]), 3)
        self.assertFalse(any(r.url.path.endswith("/complete") for r in self.requests))

    async def test_cut_off_artifact_and_cdn_5xx_are_fetched_again(self):
        """A CDN 5xx is retried like CloudTaskAdapter._download; bytes that fail validation
        are fetched again from the queue, and only then is the vendor copy deleted."""
        self.asset_data, self.asset_mime = self.mp4, "video/mp4"
        self.asset_failures = 2
        self.queue["download_url"] = "https://assets.test/private.mp4"
        self.retrieve = {"status": "COMPLETED"}
        result = await self.ad.generate(self.req("video", "video", params={"duration": "5s"}))
        self.assertEqual(result.blobs[0].data, self.mp4)
        self.assertEqual(len([r for r in self.requests if r.url.host == "assets.test"]), 3)
        self.requests.clear()
        self.inline_video = True
        self.retrieve = [httpx.Response(200, content=self.mp4[:len(self.mp4) // 2],
                                        headers={"Content-Type": "video/mp4"}), None]
        result = await self.ad.generate(self.req("video", "video", params={"duration": "5s"}))
        self.assertEqual(result.blobs[0].data, self.mp4)
        self.assertTrue(self.requests[-1].url.path.endswith("/video/complete"))

    async def test_content_decides_type_but_foreign_families_still_refuse(self):
        blob = adapters.VeniceMediaAdapter._binary_blob
        for mime in ("binary/octet-stream", "", "video/quicktime"):
            with self.subTest(mime=mime):
                self.assertEqual(blob(self.mp4, mime, "video").kind, "video")
        self.assertEqual(blob(self.mp4, "video/quicktime", "video").name, "video.mov")
        for mime in ("binary/octet-stream", "", "audio/x-wav", "audio/mpeg"):
            with self.subTest(mime=mime):
                self.assertEqual((blob(self.wav, mime, "music").mime,
                                  blob(self.wav, mime, "music").name), ("audio/wav", "audio.wav"))
        for data, mime, endpoint in ((self.wav, "text/html", "music"),
                                     (self.mp4, "text/html", "video")):
            with self.assertRaises(RuntimeError):
                blob(data, mime, endpoint)

    async def test_id3_tag_is_skipped_before_the_container_is_named(self):
        """Venice's ACE-Step FLAC arrives behind an ID3v2 tag (a C2PA GEOB frame). Reading
        the tag as "MP3" stored a playable FLAC as audio.mp3 / audio/mpeg — a player then
        decodes FLAC frames with the MP3 decoder and the paid track is crackling noise."""
        blob = adapters.VeniceMediaAdapter._binary_blob

        def id3(body, footer=False):
            size = 40
            tag = (b"ID3\x04\x00" + bytes([0x10 if footer else 0])
                   + bytes([0, 0, 0, size]) + b"GEOB" + b"\0" * (size - 4))
            return tag + (b"3DI\x04\x00\x10" + bytes([0, 0, 0, size]) if footer else b"") + body

        flac = b"fLaC" + b"\0" * 64
        for data in (id3(flac), id3(flac, footer=True), id3(id3(flac))):
            with self.subTest(data=data[:12]):
                got = blob(data, "audio/mpeg", "music")
                self.assertEqual((got.mime, got.name, got.data), ("audio/flac", "audio.flac", data))
        got = blob(id3(self.wav), "application/octet-stream", "music")
        self.assertEqual((got.mime, got.name), ("audio/wav", "audio.wav"))
        for body in (b"\xff\xfb\x90\x00" + b"\0" * 64, b"\0" * 64):  # MPEG frame / tag padding
            with self.subTest(body=body[:4]):
                self.assertEqual(blob(id3(body), "audio/mpeg", "music").name, "audio.mp3")

    async def test_streaming_and_float_wav_pass_but_a_cut_wav_does_not(self):
        blob = adapters.VeniceMediaAdapter._binary_blob
        fmt_float = (b"fmt " + (16).to_bytes(4, "little") + (3).to_bytes(2, "little")
                     + (1).to_bytes(2, "little") + (8000).to_bytes(4, "little")
                     + (32000).to_bytes(4, "little") + (4).to_bytes(2, "little")
                     + (32).to_bytes(2, "little"))
        samples = b"\0" * 400
        streaming = (b"RIFF" + b"\xff" * 4 + b"WAVE" + fmt_float + b"data" + b"\xff" * 4 + samples)
        self.assertEqual(blob(streaming, "audio/wav", "speech").name, "audio.wav")
        cut = self.wav[:len(self.wav) // 2]
        with self.assertRaises(RuntimeError):
            blob(cut, "audio/wav", "speech")

    async def test_lost_voice_upload_answer_is_final_not_failover(self):
        """A voice upload that was sent but whose answer was lost may exist (and be
        billed): a failover-class error here would upload it again on the next attempt."""
        self.voice_error = httpx.ReadTimeout("lost")
        req = self.req("speech", "clone", inputs={"prompt": "Hello"},
                       upload_files={"input_audio": ("reference.wav", self.wav)})
        with self.assertRaises(RuntimeError) as failed:
            await self.ad.generate(req)
        self.assertNotIsInstance(failed.exception, httpx.HTTPError)
        self.voice_error = httpx.ConnectError("refused")
        with self.assertRaises(httpx.ConnectError):
            await self.ad.generate(req)


if __name__ == "__main__":
    unittest.main()
