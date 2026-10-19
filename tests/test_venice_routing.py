"""Venice's shared account must route catalog media without aliases, keep chat
working, and refuse unsupported edits/counts before buying a generation.
"""
import asyncio
import base64
import io
import json
import os
from unittest.mock import patch

import httpx
from PIL import Image

from tests.test_ui_form_validation import _Fixture
from tests.test_nanogpt_media_adapter import _mp4, _wav
import adapters
import jobs
import main
import store


def _png():
    stream = io.BytesIO()
    Image.new("RGB", (24, 16), "blue").save(stream, format="PNG")
    return stream.getvalue()


class VeniceRouting(_Fixture):
    def setUp(self):
        super().setUp()
        self.backend = {"name": "venice", "type": "venice", "url": "https://provider.test/api",
                        "paid": True, "api_key": "vendor-secret", "max_concurrent": 1}
        self.ad = adapters.make_adapter(self.backend, main.adapter_ctx)
        self.catalogs = {
            "text": {"chat-model": {"id": "chat-model", "type": "text"}},
            "tts": {"tts-model": {"model_spec": {"voices": ["voice-a"]}}},
            "embedding": {"embed-model": {}},
            "images": {"image-model": {"model_spec": {"constraints": {
                "aspectRatios": ["1:1", "16:9"], "widthHeightDivisor": 16}}}},
            "edit": {"edit-model": {"model_spec": {"constraints": {"maxInputImages": 3}}}},
            "upscale": {"upscaler": {}},
            "background-remove": {"background-remove": {}},
            "video": {"video-model": {"model_spec": {"constraints": {
                "model_type": "text-to-video", "durations": ["5s", "10s"],
                "resolutions": ["720p"], "aspect_ratios": ["16:9"]}}}},
            "music": {"music-model": {"model_spec": {"supports_duration": True,
                "duration_constraints": {"min": 1, "max": 30}}}},
            "voice-changer": {"voice-model": {"model_spec": {"voice_changer": True}}},
            "speech": {"tts-model": {"model_spec": {"voices": ["voice-a"]}}},
            "transcription": {"asr-model": {}},
        }
        self.ad._publish_catalogs(self.catalogs)
        for name, value in {
            "backends": [self.backend], "backend_adapters": {"venice:venice": self.ad},
            "backend_models": {"venice:venice": set().union(*self.catalogs.values())},
            "backend_healthy": {"venice:venice": True}, "backend_inflight": {},
            "image_models": {}, "virtual_models": {"existing-chat": "chat-model"},
            "model_prefix": True, "gen_speed": {}, "gen_exec_faults": {}, "gen_progress": {},
            "_backend_names": set(), "_llm_backends": [], "_gen_backends": [], "_route_index": {},
        }.items():
            p = patch.object(main, name, value)
            p.start()
            self.addCleanup(p.stop)
        main.rebuild_route_index()
        saved = {k: getattr(jobs, k) for k in ("_DB_PATH", "_BLOB_DIR", "_DEFAULT_TTL", "_active")}
        self.addCleanup(lambda: [setattr(jobs, k, v) for k, v in saved.items()])
        jobs.init(os.path.join(self.tmp.name, "jobs.db"), os.path.join(self.tmp.name, "blobs"))

    def vendor(self, handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addCleanup(lambda: asyncio.run(client.aclose()))
        return patch.object(main, "http_client", client)

    def test_typed_catalogs_keep_media_out_of_chat_and_preserve_alias(self):
        expected = {"chat": {"venice/chat-model", "existing-chat"},
                    "image": {"venice/image-model", "venice/edit-model", "venice/upscaler",
                              "venice/background-remove"},
                    "video": {"venice/video-model"},
                    "audio": {"venice/music-model", "venice/voice-model"},
                    "asr": {"venice/asr-model"}, "tts": {"venice/tts-model"},
                    "embedding": {"venice/embed-model"}}
        for kind, ids in expected.items():
            with self.subTest(kind=kind):
                response = self.c.get("/v1/models?type=" + kind)
                self.assertEqual(response.status_code, 200)
                self.assertEqual({r["id"] for r in response.json()["data"]}, ids)
        self.assertEqual(main.resolve_routes("venice/image-model"), ([], []))
        self.assertEqual(main.resolve_routes("existing-chat")[0][0][1], "chat-model")
        self.assertEqual(store.list_aliases(), {})

    def test_image_shim_count_follows_native_variants(self):
        """Venice builds the request with `variants` over `n`: every image it bills
        must reach the caller, whatever `n` the OpenAI shim passed along."""
        submitted = []
        def provider(request):
            body = json.loads(request.content)
            submitted.append(body)
            return httpx.Response(200, json={"images": [
                base64.b64encode(_png()).decode()] * body["variants"]})
        with self.vendor(provider):
            response = self.c.post("/v1/images/generations", json={
                "model": "venice/image-model", "prompt": "blue lamp", "n": 1, "variants": 2,
                "response_format": "b64_json"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(submitted[0]["variants"], 2)
        self.assertEqual(len(response.json()["data"]), 2)

    def test_workflow_audio_tasks_share_venice_type_families(self):
        """`?type=asr|audio` must find a ComfyUI transcription / voice-conversion alias
        like Venice's — every task containing "audio" used to be `tts`."""
        for task, kind in (("audio2text", "asr"), ("audio2audio", "audio"),
                           ("text2audio", "tts"), ("text2img", "image")):
            with self.subTest(task=task):
                self.assertEqual(main._generation_type({"backend": "gpu", "task": task}), kind)

    def test_openai_image_shim_delivers_bytes_without_invented_workflow_size(self):
        submitted = []
        def provider(request):
            submitted.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "img-1", "images": [
                base64.b64encode(_png()).decode()]})
        with self.vendor(provider):
            response = self.c.post("/v1/images/generations", json={
                "model": "venice/image-model", "prompt": "blue lamp", "response_format": "b64_json"})
        self.assertEqual(response.status_code, 200, response.text)
        with Image.open(io.BytesIO(base64.b64decode(response.json()["data"][0]["b64_json"]))) as im:
            self.assertEqual(im.size, (24, 16))
        self.assertNotIn("width", submitted[0])
        self.assertNotIn("height", submitted[0])
        self.assertEqual(main.backend_inflight.get("venice:venice", 0), 0)
        self.assertEqual(store.list_aliases(), {})

    def test_invalid_image_count_and_mask_edit_are_refused_before_submit(self):
        def provider(request):
            self.fail("invalid input must not buy a Venice generation")
        original = httpx.AsyncClient
        with patch("adapters.httpx.AsyncClient", lambda **kw:
                   original(transport=httpx.MockTransport(provider), **kw)):
            response = self.c.post("/v1/images/generations", json={
                "model": "venice/image-model", "prompt": "lamp", "n": 1.5})
            self.assertEqual(response.status_code, 400, response.text)
            response = self.c.post("/v1/images/edits", data={
                "model": "venice/edit-model", "prompt": "change the color"}, files={
                "image": ("source.png", _png(), "image/png"),
                "mask": ("mask.png", _png(), "image/png")})
            self.assertEqual(response.status_code, 400, response.text)

    def test_media_grant_cannot_call_colliding_chat_alias(self):
        main.virtual_models = {"image-model": "chat-model"}
        main.rebuild_route_index()
        user = {"name": "images", "role": "user", "models": [main.GRANT_ALL_IMAGE]}
        self.assertTrue(main._model_allowed(user, "venice/image-model", media=True))
        with patch.object(main, "users", [user]), patch.object(main, "_users_by_key", {"image-key": user}):
            response = self.c.post("/v1/chat/completions",
                headers={"Authorization": "Bearer image-key"}, json={
                    "model": "image-model", "messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(response.status_code, 403, response.text)

    def test_all_backends_grant_covers_account_text_but_never_paid_media(self):
        """`@backends` means all LLM backends: it used to open a composite account's
        billed video/music too (prefixed id, bare id, /v1/models). Only the account
        named one by one, `@image`, an alias or the exact id may."""
        llm_only = {"name": "llm", "role": "user", "models": [main.GRANT_ALL_BACKENDS]}
        self.assertTrue(main._model_allowed(llm_only, "venice/chat-model", media=False))
        self.assertTrue(main._model_allowed(llm_only, "chat-model", media=False))
        for model in ("venice/video-model", "video-model", "venice/chat-model"):
            with self.subTest(model=model):
                self.assertFalse(main._model_allowed(llm_only, model, media=True))
        self.assertFalse(main._model_allowed(llm_only, "venice/music-model"))
        account = {"name": "acct", "role": "user", "models": ["venice"]}
        self.assertTrue(main._model_allowed(account, "venice/video-model", media=True))
        with patch.object(main, "users", [llm_only]), \
                patch.object(main, "_users_by_key", {"llm-key": llm_only}):
            listed = {m["id"] for m in self.c.get(
                "/v1/models", headers={"Authorization": "Bearer llm-key"}).json()["data"]}
        self.assertIn("venice/chat-model", listed)
        self.assertNotIn("venice/video-model", listed)

    def test_alias_before_first_catalog_is_retryable_not_a_client_error(self):
        """A key/URL change starts a fresh adapter with no snapshot: an alias call used
        to get a 400 "model is unavailable" — blaming the client for gateway state."""
        import venice
        store.upsert("clips", [venice.candidate_for("venice", "video-model", "video")])
        self.ad._publish_catalogs({})
        response = self.c.post("/v1/generations", json={
            "model": "clips", "prompt": "moving clouds", "output": {"mode": "sync"}})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertIn("retry-after", response.headers)

    def test_failed_refresh_keeps_console_inventory_but_no_callable_media(self):
        self.ad._publish_catalogs({})
        main.rebuild_route_index()
        self.assertEqual(main.get_gen_routes("venice/video-model"), [])
        self.assertEqual(main.generation_models_info(), [])
        rows = main.generation_models_info(console=True)
        row = next(r for r in rows if r["id"] == "venice/video-model")
        self.assertTrue(row["candidate"]["unavailable"])
        self.assertEqual(main.resolve_routes("venice/image-model"), ([], []))

    def test_video_seconds_reach_provider_as_native_duration(self):
        submitted = []
        mp4 = _mp4()
        def provider(request):
            if request.url.path.endswith("/queue"):
                submitted.append(json.loads(request.content))
                return httpx.Response(200, json={"model": "video-model", "queue_id": "vid-1"})
            if request.url.path.endswith("/retrieve"):
                return httpx.Response(200, content=mp4, headers={"Content-Type": "video/mp4"})
            return httpx.Response(200, json={"success": True})
        with self.vendor(provider):
            response = self.c.post("/v1/generations", json={
                "model": "venice/video-model", "prompt": "moving clouds", "seconds": 5,
                "params": {"resolution": "720p", "aspect_ratio": "16:9"},
                "output": {"mode": "sync"}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "done", response.text)
        self.assertEqual(submitted[0]["duration"], "5s")
        self.assertNotIn("frames", submitted[0])
        self.assertNotIn("seconds", submitted[0])

    def test_playground_wav_data_uri_delivers_a_transcript(self):
        """The console's audio/wav MIME must not become a .glb upload at the API boundary."""
        import admin
        def provider(request):
            return httpx.Response(200, json={"text": "A transcribed audio sample."})
        data_uri = ("data:" + admin._pg_file_mime("reference.wav") + ";base64,"
                    + base64.b64encode(_wav()).decode())
        with self.vendor(provider):
            response = self.c.post("/v1/generations", json={
                "model": "venice/asr-model", "files": {"input_audio": data_uri},
                "output": {"mode": "sync"}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "done", response.text)
        artifact = self.c.get(response.json()["results"][0]["url"])
        self.assertEqual(artifact.json()["text"], "A transcribed audio sample.")
