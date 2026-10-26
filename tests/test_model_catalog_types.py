"""Mixed local/cloud catalogs retain grants, aliases and every capability membership."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from tests.test_ui_form_validation import _Fixture
import adapters
import main
import store


class ModelCatalogTypes(_Fixture):
    def setUp(self):
        super().setUp()
        local = {"name": "local", "type": "openai", "url": "http://localhost:9999", "local": True}
        cloud = {"name": "cloud", "type": "openai", "url": "https://provider.test", "paid": True}
        nano = {"name": "nano", "type": "nanogpt", "url": "https://nano.test", "paid": True}
        catalogs = {"text": {"cloud-chat": {}}, "tts": {"cloud-voice": {}},
                    "embedding": {"cloud-embed": {}}, "images": {"cloud-image": {}},
                    "video": {"cloud-video": {}}, "3d": {"cloud-mesh": {}}}
        state = {
            "backends": [local, cloud, nano], "config_backends": [],
            "backend_models": {"openai:local": {"legacy", "vision", "kokoro", "whisper-large-v3",
                                                  "nomic-embed-text", "bge-reranker-v2"},
                               "openai:cloud": {"opaque-voice", "opaque-image", "multi", "opaque-embed"},
                               "nanogpt:nano": set().union(*catalogs.values())},
            "backend_model_metadata": {"openai:local": {"vision": {"architecture": {
                "input_modalities": ["text", "image"], "output_modalities": ["text"]}}},
                "openai:cloud": {"opaque-voice": {"type": "tts"}, "opaque-image": {"type": "image"},
                                 "multi": {"architecture": {"output_modalities": ["text", "image"]}},
                                 "opaque-embed": {"type": "embedding"}}},
            "backend_adapters": {"nanogpt:nano": SimpleNamespace(model_catalogs=catalogs)},
            "backend_pricing": {}, "backend_loras": {}, "backend_model_counts": {}, "backend_error": {},
            "backend_healthy": {"openai:local": True, "openai:cloud": True, "nanogpt:nano": True},
            "backend_context": {"openai:local": {"legacy": 8192},
                                "openai:cloud": {"multi": 4096}},
            "backend_running": {"openai:local": []},
            "virtual_models": {"embedding": {"cloud": "opaque-embed"},
                               "tool": {"local": "legacy"},
                               "speaker": {"cloud": "opaque-voice"},
                               "mixed": {"local": "legacy", "cloud": "multi"},
                               "loaded": {"local": "current"}},
            "image_models": {}, "model_prefix": True,
            "_backend_names": {"local", "cloud", "nano"},
            "_llm_backends": [], "_gen_backends": [], "_route_index": {},
        }
        for name, value in state.items():
            p = patch.object(main, name, value)
            p.start()
            self.addCleanup(p.stop)
        main.rebuild_route_index()

    def ids(self, kind=None):
        response = self.c.get("/v1/models" + ("?type=" + kind if kind else ""))
        self.assertEqual(response.status_code, 200, response.text)
        return {row["id"] for row in response.json()["data"]}

    def test_local_extras_and_provider_metadata_filter_without_losing_untyped_chat(self):
        self.assertEqual(self.ids("chat"), {"local/legacy", "legacy", "local/vision", "vision",
            "cloud/multi", "nano/cloud-chat", "tool", "mixed", "loaded", "local/current"})
        self.assertEqual(self.ids("tts"), {"local/kokoro", "kokoro", "cloud/opaque-voice",
                                           "nano/cloud-voice", "speaker"})
        self.assertEqual(self.ids("asr"), {"local/whisper-large-v3", "whisper-large-v3"})
        self.assertEqual(self.ids("embedding"), {"local/nomic-embed-text", "nomic-embed-text",
                                                  "cloud/opaque-embed", "nano/cloud-embed", "embedding"})
        self.assertEqual(self.ids("rerank"), {"local/bge-reranker-v2", "bge-reranker-v2"})
        self.assertEqual(self.ids("image"), {"cloud/opaque-image", "cloud/multi", "nano/cloud-image", "mixed"})

    def test_media_union_excludes_chat_embedding_and_rerank(self):
        expected = {"local/kokoro", "kokoro", "local/whisper-large-v3", "whisper-large-v3",
                    "cloud/opaque-voice", "cloud/opaque-image", "cloud/multi", "speaker", "mixed",
                    "nano/cloud-voice", "nano/cloud-image", "nano/cloud-video", "nano/cloud-mesh"}
        self.assertEqual(self.ids("media"), expected)
        self.assertTrue(expected <= self.ids())
        self.assertIn("embedding", self.ids())
        self.assertIn("legacy", self.ids())

    def test_alias_maps_use_targets_and_context_still_uses_smallest_candidate(self):
        info = main.gateway_info()
        self.assertEqual(info["alias_types"], {"embedding": ["embedding"], "tool": ["chat"],
            "speaker": ["tts"], "mixed": ["chat", "image"], "loaded": ["chat"]})
        rows = {row["name"]: row for row in main.llm_backends_info()}
        self.assertEqual(rows["cloud"]["model_types"]["multi"], ["chat", "image"])
        self.assertEqual(rows["nano"]["model_types"]["cloud-voice"], ["tts"])
        self.assertEqual(rows["local"]["model_types"]["current"], ["chat"])
        self.assertIsNone(rows["cloud"]["billing"])
        mixed = next(row for row in self.c.get("/v1/models?type=chat").json()["data"]
                     if row["id"] == "mixed")
        self.assertEqual(mixed["context_length"], 4096)

    def test_alias_grants_do_not_reveal_raw_models_or_other_aliases(self):
        user = {"name": "restricted", "models": ["speaker"], "role": "user"}
        with patch.object(main, "authenticate", return_value=user):
            self.assertEqual(self.ids("tts"), {"speaker"})
            self.assertEqual(self.ids("media"), {"speaker"})
            self.assertEqual(self.ids("chat"), set())

    def test_global_alias_does_not_gain_chat_from_backends_missing_its_target(self):
        with patch.object(main, "virtual_models", {"voice": "opaque-voice", "embed": "opaque-embed"}):
            self.assertEqual(main.gateway_info()["alias_types"], {"voice": ["tts"], "embed": ["embedding"]})
            self.assertNotIn("voice", self.ids("chat"))
            self.assertNotIn("embed", self.ids("chat"))
            self.assertIn("voice", self.ids("tts"))

    def test_shared_catalog_model_and_alias_keep_multiple_capabilities(self):
        catalogs = main.backend_adapters["nanogpt:nano"].model_catalogs
        catalogs["text"]["dual"] = {}
        catalogs["tts"]["dual"] = {}
        main.backend_models["nanogpt:nano"].add("dual")
        with patch.object(main, "virtual_models", {"hybrid": "dual"}):
            for kind in ("chat", "tts"):
                self.assertTrue({"nano/dual", "hybrid"} <= self.ids(kind))
            self.assertEqual(main.gateway_info()["alias_types"]["hybrid"], ["chat", "tts"])
            nano = next(row for row in main.llm_backends_info() if row["name"] == "nano")
            self.assertEqual(nano["model_types"]["dual"], ["chat", "tts"])

    def test_all_backends_grant_does_not_open_composite_paid_media(self):
        user = {"name": "restricted", "models": ["@backends"], "role": "user"}
        with patch.object(main, "authenticate", return_value=user):
            self.assertIn("nano/cloud-voice", self.ids("media"))
            self.assertTrue({"nano/cloud-image", "nano/cloud-video", "nano/cloud-mesh"}.isdisjoint(
                self.ids("media")))
        user["models"] = ["nano"]
        with patch.object(main, "authenticate", return_value=user):
            self.assertTrue({"nano/cloud-image", "nano/cloud-video", "nano/cloud-mesh"} <= self.ids("media"))

    def test_media_alias_union_and_precedence_over_same_named_raw_model(self):
        store.upsert("nano/cloud-image", [{"backend": "gpu", "task": "text2mesh"},
                                         {"backend": "gpu", "task": "text2audio"}])
        self.assertIn("nano/cloud-image", self.ids("3d"))
        self.assertIn("nano/cloud-image", self.ids("tts"))
        self.assertNotIn("nano/cloud-image", self.ids("image"))

    def test_discovery_preserves_non_dynamic_provider_metadata_and_configured_extras(self):
        backend = {"name": "new", "type": "openai", "url": "https://new.test",
                   "models_extra": ["kokoro", "nomic-embed-text", "whisper-small"]}
        ad = adapters.make_adapter(backend, main.adapter_ctx)
        def provider(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "opaque", "type": "tts"}]})
            return httpx.Response(404)
        async def discover():
            async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:
                await main.refresh_backend(backend, client)
        with patch.object(main, "backends", [backend]), patch.object(main, "backend_adapters", {"openai:new": ad}):
            asyncio.run(discover())
            info = main.llm_backends_info()[0]
            self.assertEqual(set(info["models"]), {"opaque", "kokoro", "nomic-embed-text", "whisper-small"})
            self.assertEqual(info["model_types"], {"opaque": ["tts"], "kokoro": ["tts"],
                "nomic-embed-text": ["embedding"], "whisper-small": ["asr"]})

    def test_raw_metadata_survives_settings_edits_but_not_account_replacement(self):
        cloud = next(b for b in main.backends if b["name"] == "cloud")
        old = adapters.make_adapter(cloud, main.adapter_ctx)
        with patch.object(main, "backends", [{**cloud, "max_concurrent": 2}]), patch.object(
                main, "backend_adapters", {"openai:cloud": old}):
            main.build_backend_adapters()
            self.assertEqual(main.llm_backends_info()[0]["model_types"]["opaque-voice"], ["tts"])
            main.backends = [{**cloud, "api_key": "different-account"}]
            main.build_backend_adapters()
            self.assertEqual(main.llm_backends_info()[0]["model_types"]["opaque-voice"], ["chat"])

    def test_old_account_poll_cannot_restore_metadata_after_account_replacement(self):
        cloud = next(b for b in main.backends if b["name"] == "cloud")
        old = adapters.make_adapter(cloud, main.adapter_ctx)
        async def discover(_client):
            main.backends = [{**cloud, "url": "https://other.test"}]
            main.build_backend_adapters()
            return adapters.Capabilities(models={"opaque-voice"}, pricing={},
                                         metadata={"opaque-voice": {"type": "tts"}})
        with patch.object(main, "backends", [cloud]), patch.object(
                main, "backend_adapters", {"openai:cloud": old}), patch.object(old, "discover", discover):
            asyncio.run(main.refresh_backend(cloud, None))
            self.assertEqual(main.llm_backends_info()[0]["model_types"]["opaque-voice"], ["chat"])

    def test_chat_and_generation_aliases_keep_separate_callable_namespaces(self):
        store.upsert("shared", [{"backend": "gpu", "task": "text2img"}])
        with patch.object(main, "virtual_models", {"shared": {"local": "legacy"}}):
            self.assertIn("shared", self.ids("chat"))
            self.assertIn("shared", self.ids("image"))
            self.assertIn("shared", self.ids("media"))
            with patch.object(main, "authenticate", return_value={"models": ["@image"]}):
                self.assertIn("shared", self.ids())
                self.assertIn("shared", self.ids("image"))
                self.assertNotIn("shared", self.ids("chat"))

    def test_chat_alias_does_not_hide_direct_generation_id(self):
        with patch.object(main, "virtual_models", {"nano/cloud-image": {"local": "legacy"}}):
            self.assertIn("nano/cloud-image", self.ids("image"))
            with patch.object(main, "authenticate", return_value={"models": ["@image"]}):
                self.assertIn("nano/cloud-image", self.ids())

    def test_prefixed_model_types_follow_actual_generic_alias_resolution(self):
        main.virtual_models["kokoro"] = {"local": "legacy"}
        main.rebuild_route_index()
        self.assertEqual(main.resolve_routes("local/kokoro")[0][0][1], "legacy")
        self.assertIn("local/kokoro", self.ids("chat"))
        self.assertNotIn("local/kokoro", self.ids("tts"))
        local = next(row for row in main.llm_backends_info() if row["name"] == "local")
        self.assertEqual(local["model_types"]["kokoro"], ["tts"])
        self.assertEqual(local["callable_model_types"]["kokoro"], ["chat"])

    def test_alias_ignores_same_named_backend_that_does_not_serve_target(self):
        shadow = {"name": "nano", "type": "openai", "url": "http://localhost:1234"}
        main.backends.insert(0, shadow)
        main.backend_models["openai:nano"] = {"legacy"}
        main.backend_healthy["openai:nano"] = True
        next(b for b in main.backends if b["type"] == "nanogpt")["local"] = True
        with patch.object(main, "virtual_models", {"voice": {"nano": "cloud-voice"}}):
            main.rebuild_route_index()
            self.assertEqual(main.gateway_info()["alias_types"]["voice"], ["tts"])
            self.assertNotIn("voice", self.ids("chat"))
            self.assertIn("voice", self.ids("tts"))
            self.assertEqual(main.resolve_routes("nano/cloud-voice"), ([], []))
            nano = next(row for row in main.llm_backends_info() if row["type"] == "nanogpt")
            self.assertEqual(nano["model_types"]["cloud-voice"], ["tts"])
            self.assertEqual(nano["callable_model_types"]["cloud-voice"], [])
            self.assertNotIn("nano/cloud-voice", self.ids("tts"))
            self.assertIn("cloud-voice", self.ids("tts"))
            self.assertIn("cloud-voice", self.ids("media"))

    def test_account_replacement_during_persistence_cannot_erase_new_metadata(self):
        cloud = next(b for b in main.backends if b["name"] == "cloud")
        old = adapters.make_adapter(cloud, main.adapter_ctx)
        async def discover(_client):
            return adapters.Capabilities(models={"opaque-voice"}, pricing={},
                                         metadata={"opaque-voice": {"type": "tts"}})
        async def persist(_label, _save, *_args):
            main.backends = [{**cloud, "api_key": "new-account"}]
            main.build_backend_adapters()
            main.backend_model_metadata["openai:cloud"] = {"opaque-voice": {"type": "image"}}
        with patch.object(main, "backends", [cloud]), patch.object(
                main, "backend_adapters", {"openai:cloud": old}), patch.object(
                old, "discover", discover), patch.object(main, "_persist_discovery", persist):
            asyncio.run(main.refresh_backend(cloud, None))
            self.assertEqual(main.llm_backends_info()[0]["model_types"]["opaque-voice"], ["image"])
