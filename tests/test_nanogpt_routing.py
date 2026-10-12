"""NanoGPT catalog discovery must make media callable without stored aliases;
text routes must not swallow image IDs, and the old provider split must migrate
without losing account credentials or changing a candidate to another account.
"""
import base64
import asyncio
import io
import os
import json
from unittest.mock import patch

import httpx
from PIL import Image
from tests.test_ui_form_validation import _Fixture
import adapters
import jobs
import main
import store
from tests.test_nanogpt_media_adapter import _mp4, _wav, _png


class DirectNanoModels(_Fixture):
    def setUp(self):
        super().setUp()
        self.backend = {"name": "nano", "type": "nanogpt", "url": "https://provider.test/api",
                        "paid": True, "max_concurrent": 1}
        self.ad = adapters.make_adapter(self.backend, main.adapter_ctx)
        self.ad._publish_catalogs({
            "text": {"chat-model": {}}, "tts": {"tts-model": {}},
            "embedding": {"embed-model": {}},
            "images": {"qwen-image": {"capabilities": {"image_to_image": True},
                         "supported_parameters": {"resolutions": ["1k", "2k"],
                                                  "max_input_images": 3, "max_output_images": 4}}},
            "video": {"video-model": {}}, "3d": {"mesh-model": {}}})
        for key, value in {
            "backends": [self.backend], "backend_adapters": {"nanogpt:nano": self.ad},
            "backend_models": {"nanogpt:nano": set().union(*self.ad.model_catalogs.values())},
            "backend_healthy": {"nanogpt:nano": True}, "backend_inflight": {},
            "image_models": {}, "virtual_models": {}, "model_prefix": True,
            "gen_speed": {}, "gen_exec_faults": {}, "gen_progress": {},
            "_backend_names": set(), "_llm_backends": [], "_gen_backends": [], "_route_index": {},
        }.items():
            p = patch.object(main, key, value)
            p.start()
            self.addCleanup(p.stop)
        main.rebuild_route_index()
        saved_jobs = {k: getattr(jobs, k) for k in ("_DB_PATH", "_BLOB_DIR", "_DEFAULT_TTL", "_active")}
        self.addCleanup(lambda: [setattr(jobs, k, v) for k, v in saved_jobs.items()])
        jobs.init(os.path.join(self.tmp.name, "jobs.db"), os.path.join(self.tmp.name, "blobs"))

    def test_discovered_image_is_callable_without_alias_and_not_chat(self):
        offered = self.c.get("/v1/models?type=image").json()["data"]
        self.assertIn("nano/qwen-image", [row["id"] for row in offered])
        self.assertNotIn("nano/video-model", [row["id"] for row in offered])
        self.assertNotIn("qwen-image", main._route_index)
        buf = io.BytesIO()
        Image.new("RGB", (16, 12), "red").save(buf, format="PNG")
        def provider(request):
            return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]})
        original = httpx.AsyncClient
        with patch("adapters.httpx.AsyncClient", lambda **kw: original(transport=httpx.MockTransport(provider), **kw)):
            response = self.c.post("/v1/images/generations", json={
                "model": "nano/qwen-image", "prompt": "a red lamp", "response_format": "b64_json"})
        self.assertEqual(response.status_code, 200, response.text)
        with Image.open(io.BytesIO(base64.b64decode(response.json()["data"][0]["b64_json"]))) as image:
            self.assertEqual(image.size, (16, 12))
        self.assertEqual(store.list_aliases(), {})
        self.assertEqual(main.backend_inflight.get("nanogpt:nano", 0), 0)

    def test_media_grant_does_not_grant_chat_and_alias_overrides_direct_id(self):
        media_user = {"models": [main.GRANT_ALL_IMAGE]}
        self.assertTrue(main._model_allowed(media_user, "nano/qwen-image"))
        self.assertFalse(main._model_allowed(media_user, "nano/chat-model"))
        store.upsert("nano/qwen-image", [{"backend": "nano", "model": "qwen-image", "task": "text2img",
                       "nanogpt": {"endpoint": "images", "options": {"n": 2}}}])
        routes = main.get_gen_routes("nano/qwen-image")
        self.assertEqual(routes[0][1]["nanogpt"]["options"]["n"], 2)

    def test_typed_catalogs_and_prefixed_chat_refuse_media(self):
        for kind, ids in {"chat": {"nano/chat-model"}, "image": {"nano/qwen-image"},
                          "video": {"nano/video-model"}, "3d": {"nano/mesh-model"},
                          "tts": {"nano/tts-model"}, "embedding": {"nano/embed-model"}}.items():
            with self.subTest(kind=kind):
                self.assertEqual({row["id"] for row in
                                  self.c.get("/v1/models?type=" + kind).json()["data"]}, ids)
        self.assertEqual(self.c.get("/v1/models/nano/qwen-image").status_code, 200)
        self.assertEqual(main.resolve_routes("nano/qwen-image"), ([], []))

    def test_pending_catalog_refresh_cannot_turn_cached_media_into_chat(self):
        """Cleared discovery metadata must not expose cached image IDs as paid chat."""
        self.ad._publish_catalogs({})
        main.rebuild_route_index()
        self.assertEqual(main.resolve_routes("nano/qwen-image"), ([], []))
        self.assertEqual(self.c.get("/v1/models?type=chat").json()["data"], [])


    def test_media_picker_survives_pending_and_failed_discovery_without_paid_routes(self):
        """A refresh must not erase 3D selection or expose stale metadata to paid submit."""
        import admin
        from tests.test_nanogpt_console import _Controls
        with patch.object(admin, "_generation_candidate", main.generation_candidate):
            model = "nano/mesh-model"
            first = self.c.get("/ui/playground?sub=media&model=" + model)
            self.assertEqual(_Controls(first.text).selected["model"], model)
            saved_catalogs = self.ad.model_catalogs
            self.ad._publish_catalogs({})
            for healthy in (True, False):
                main.backend_healthy["nanogpt:nano"] = healthy
                page = self.c.get("/ui/playground?sub=media&model=" + model)
                controls = _Controls(page.text)
                self.assertEqual(controls.selected.get("model"), model)
                self.assertIn("nano/qwen-image", controls.options)
                self.assertIn("temporarily unavailable", page.text)
                self.assertIn('type="submit" disabled', page.text)
                self.assertEqual(main.get_gen_routes(model), [])
                self.assertEqual(main.generation_models_info(), [])
            self.ad._publish_catalogs(saved_catalogs)
            main.backend_healthy["nanogpt:nano"] = True
            page = self.c.get("/ui/playground?sub=media&model=" + model)
            self.assertEqual(_Controls(page.text).selected["model"], model)
            self.assertNotIn('type="submit" disabled', page.text)
            self.assertEqual(main.get_gen_routes(model)[0][1]["model"], "mesh-model")

    def test_media_only_user_cannot_call_same_named_chat_alias(self):
        """Catalog media IDs cannot open a same-named chat alias through @image."""
        main.virtual_models = {"qwen-image": "chat-model"}
        main.rebuild_route_index()
        user = {"name": "pictures", "role": "user", "models": [main.GRANT_ALL_IMAGE]}
        original = httpx.AsyncClient
        transport = httpx.MockTransport(lambda r: httpx.Response(200,
            json={"choices": [{"message": {"content": "unauthorized chat"}, "finish_reason": "stop"}]}))
        client = original(transport=transport)
        self.addCleanup(lambda: asyncio.run(client.aclose()))
        with patch.object(main, "users", [user]), \
                patch.object(main, "_users_by_key", {"image-key": user}), \
                patch.object(main, "_usage", {}), \
                patch.object(main, "http_client", client):
            response = self.c.post("/v1/chat/completions",
                headers={"Authorization": "Bearer image-key"},
                json={"model": "qwen-image", "messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual(response.status_code, 403, response.text)

    def test_chat_only_user_cannot_buy_same_named_direct_media(self):
        """@chat must not expand a chat alias into authorization for a colliding image ID."""
        main.virtual_models = {"qwen-image": "chat-model"}
        main.rebuild_route_index()
        user = {"name": "chat-only", "role": "user", "models": [main.GRANT_ALL_CHAT]}
        original = httpx.AsyncClient
        transport = httpx.MockTransport(lambda r: httpx.Response(200,
            json={"data": [{"b64_json": base64.b64encode(_png()).decode()}]}))
        with patch.object(main, "users", [user]), \
                patch.object(main, "_users_by_key", {"chat-key": user}), \
                patch.object(main, "_usage", {}), \
                patch("adapters.httpx.AsyncClient", lambda **kw: original(transport=transport, **kw)):
            response = self.c.post("/v1/images/generations",
                headers={"Authorization": "Bearer chat-key"},
                json={"model": "qwen-image", "prompt": "must not buy an image"})
        self.assertEqual(response.status_code, 403, response.text)

    def test_direct_video_keeps_uploaded_video_and_audio_without_alias(self):
        """Alias-free media must not bill a request with its uploaded files silently dropped."""
        self.ad.model_catalogs["video"]["video-model"] = {
            "capabilities": {"video_to_video": True, "audio_input": True}}
        uploads = {"input_video": ("video/mp4", _mp4()), "input_audio": ("audio/wav", _wav())}
        submitted = []

        def provider(request):
            if request.method == "POST":
                submitted.append(json.loads(request.content))
                return httpx.Response(200, json={"runId": "video-1"})
            if request.url.path.endswith("/status"):
                return httpx.Response(200, json={"status": "completed",
                    "videoUrl": "https://cdn.test/result.mp4"})
            return httpx.Response(200, content=_mp4())

        original = httpx.AsyncClient
        with patch("adapters.httpx.AsyncClient", lambda **kw:
                   original(transport=httpx.MockTransport(provider), **kw)):
            response = self.c.post("/v1/generations", json={
                "model": "nano/video-model", "prompt": "keep the source audio",
                "files": {key: "data:" + mime + ";base64," + base64.b64encode(data).decode()
                          for key, (mime, data) in uploads.items()},
                "output": {"mode": "sync"}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "done")
        for slot, control in (("input_video", "videoDataUrl"), ("input_audio", "audioDataUrl")):
            self.assertEqual(base64.b64decode(submitted[0][control].split(",", 1)[1]), uploads[slot][1])

    def test_cached_models_regain_bare_chat_routes_after_discovery(self):
        """A restart with unchanged cached IDs must restore typed routes and chat aliases."""
        catalogs = self.ad.model_catalogs
        catalogs["tts"]["tts-model"] = {"capabilities": {"text_to_speech": True}}
        main.virtual_models = {"qa": "chat-model"}
        self.ad._publish_catalogs({})
        main.rebuild_route_index()
        paths = {"v1/models": "text", "v1/audio-models": "tts",
                 "v1/embedding-models": "embedding", "v1/images/models": "images",
                 "v1/video-models": "video", "v1/3d-models": "3d"}

        def provider(request):
            if request.method == "GET":
                kind = paths[request.url.path.removeprefix("/api/")]
                return httpx.Response(200, json={"data": [
                    {**meta, "id": model} for model, meta in catalogs[kind].items()]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "restored"},
                                                         "finish_reason": "stop"}]})

        async def refresh():
            async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
                await main.refresh_backend(self.backend, client)
        asyncio.run(refresh())
        original = httpx.AsyncClient
        client = original(transport=httpx.MockTransport(provider))
        self.addCleanup(lambda: asyncio.run(client.aclose()))
        with patch.object(main, "http_client", client):
            response = self.c.post("/v1/chat/completions",
                json={"model": "chat-model", "messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "restored")

    def test_hot_reload_first_generation_backend_activates_startup_job_store(self):
        """Adding NanoGPT through YAML must create readable jobs without moving startup DB paths."""
        boot = {"db_path": os.path.join(self.tmp.name, "hot-jobs.db"),
                "blob_dir": os.path.join(self.tmp.name, "hot-blobs")}
        path = os.path.join(self.tmp.name, "hot-config.yaml")
        with open(path, "w") as stream:
            json.dump({"api_key": "", "backends": [self.backend],
                       "jobs": {"db_path": os.path.join(self.tmp.name, "ignored.db")}}, stream)
        for name in ("config", "config_backends", "config_virtual_models", "jobs_cfg",
                     "stats_cfg", "health_check_interval", "log_per_call", "max_concurrent_default"):
            p = patch.object(main, name, getattr(main, name))
            p.start()
            self.addCleanup(p.stop)
        main.backends, main.config_backends, main._gen_backends = [], [], []
        jobs._active, jobs._DB_PATH, jobs._BLOB_DIR = False, boot["db_path"], boot["blob_dir"]

        async def reload_and_create():
            main.reload_config()
            return jobs.create("text2img", "nano/qwen-image", "nano", owner="ip:testclient")

        with patch.object(main, "CONFIG_PATH", path), \
                patch.object(main, "_jobs_startup_cfg", boot, create=True):
            job_id = asyncio.run(reload_and_create())
        response = self.c.get("/v1/jobs/" + job_id)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "queued")
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "ignored.db")))


class LegacyNanoMigration(_Fixture):
    def test_old_media_account_and_alias_migrate_without_overwriting_text_account(self):
        store.upsert_backend({"name": "nano", "type": "nanogpt", "url": "https://text.test/api",
                              "api_key": "text-secret", "max_concurrent": 2})
        store.upsert_backend({"name": "nano", "type": "nanogpt_media", "url": "https://media.test/api",
                              "api_key": "media-secret", "max_concurrent": 1})
        store.upsert("my-image", [{"backend": "nano", "model": "qwen-image", "task": "text2img",
                     "nanogpt_media": {"endpoint": "images", "options": {"resolution": "2k"}}}])
        store.migrate_nanogpt()
        self.assertEqual(store.get_backend("nano", "nanogpt")["api_key"], "text-secret")
        media = store.get_backend("nano-media", "nanogpt")
        self.assertEqual((media["api_key"], media["max_concurrent"]), ("media-secret", 1))
        self.assertIsNone(store.get_backend("nano", "nanogpt_media"))
        candidate = store.get("my-image")[0]
        self.assertEqual(candidate["backend"], "nano-media")
        self.assertEqual(candidate["nanogpt"]["options"], {"resolution": "2k"})
        self.assertNotIn("nanogpt_media", candidate)
        before = store.list_backends(), store.list_aliases()
        store.migrate_nanogpt()
        self.assertEqual((store.list_backends(), store.list_aliases()), before)

    def test_saved_config_target_refuses_a_new_conflicting_account(self):
        """A later current-type row must not collapse into a previously migrated account."""
        media = {"name": "nano", "type": "nanogpt_media", "url": "https://media.test/api",
                 "api_key": "media-secret", "max_concurrent": 1}
        store.migrate_nanogpt([media])
        other = {"name": "nano", "type": "nanogpt", "url": "https://other.test/api",
                 "api_key": "other-secret", "max_concurrent": 3}
        with self.assertRaises(ValueError):
            store.migrate_nanogpt([media, other])
        self.assertEqual(store.migrate_nanogpt([media]), {"nano": "nano"})

    def test_orphaned_legacy_alias_does_not_bind_an_unrelated_current_account(self):
        """A missing old media profile must stay unroutable, not buy on a same-named account."""
        account = {"name": "nano", "type": "nanogpt", "url": "https://other.test/api",
                   "api_key": "other-secret"}
        store.upsert_backend(account)
        cand = {"backend": "nano", "model": "qwen-image",
                "nanogpt_media": {"endpoint": "images", "options": {}}}
        store.upsert("orphaned", [cand])
        store.migrate_nanogpt()
        with patch.object(main, "_gen_backends", [account]), \
                patch.object(main, "backend_healthy", {"nanogpt:nano": True}):
            self.assertEqual(main.get_gen_routes("orphaned"), [])
        cfg = {"backends": [account], "image_models": {"old-config-alias": [dict(cand)]}}
        with patch.object(store, "_active", False):
            main._normalize_nanogpt_config(cfg)
        self.assertNotEqual(cfg["image_models"]["old-config-alias"][0]["backend"], "nano")

    def test_legacy_config_names_survive_reload_and_avoid_another_account(self):
        """Repeated old-config loads must neither clone a media account nor overwrite one."""
        store.upsert_backend({"name": "nano-media", "type": "nanogpt", "url": "https://other.test/api",
                              "api_key": "other-secret"})
        raw = {"backends": [
            {"name": "nano", "type": "nanogpt", "url": "https://text.test/api"},
            {"name": "nano", "type": "nanogpt_media", "url": "https://media.test/api", "api_key": "media-secret"}],
            "image_models": {"image": [{"backend": "nano", "model": "qwen-image",
                                      "nanogpt_media": {"endpoint": "images", "options": {"n": 2}}}]}}
        for _ in range(2):
            cfg = json.loads(json.dumps(raw))
            main._normalize_nanogpt_config(cfg)
            self.assertEqual([(b["name"], b["type"]) for b in cfg["backends"]],
                             [("nano", "nanogpt"), ("nano-media-2", "nanogpt")])
            cand = cfg["image_models"]["image"][0]
            self.assertEqual((cand["backend"], cand["nanogpt"]["options"]),
                             ("nano-media-2", {"n": 2}))
            self.assertNotIn("nanogpt_media", cand)
        self.assertEqual(store.get_backend("nano-media", "nanogpt")["api_key"], "other-secret")

    def test_config_only_migration_keeps_offline_backend_model_grants(self):
        """Moving a config-only media backend must not turn cached grants into 403s."""
        store.save_backend_models("nanogpt_media:nano", {"qwen-image"})
        cfg = {"backends": [{"name": "nano", "type": "nanogpt_media", "url": "https://nano.test/api"}]}
        main._normalize_nanogpt_config(cfg)
        with patch.object(main, "backends", cfg["backends"]), \
                patch.object(main, "backend_models", store.load_backend_models()), \
                patch.object(main, "_backend_names", {"nano"}), \
                patch.object(main, "virtual_models", {}):
            self.assertTrue(main._model_allowed({"models": ["nano"]}, "qwen-image"))
