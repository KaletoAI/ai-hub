"""NanoGPT catalogs must discover speech/embeddings without corrupting token costs.

Silent failures guarded: OpenRouter's per-token reader overcounts NanoGPT text
prices by a million; chat_only can delete TTS/embedding ids; character-priced TTS
can look free; a partially failed refresh can keep stale routes looking healthy.
All HTTP is deterministic MockTransport data, not paid vendor requests.

Run: venv/bin/python -m unittest tests.test_nanogpt_text -v
"""
import asyncio
import copy
import json
import threading
import unittest

import httpx

import adapters


_TEXT_PATH = "/api/v1/models"
_AUDIO_PATH = "/api/v1/audio-models"
_EMBED_PATH = "/api/v1/embedding-models"
_CATALOGS = {
    _TEXT_PATH: {"object": "list", "data": [
        {"id": "chat-model", "object": "model", "context_length": 1050000,
         "architecture": {"input_modalities": ["text", "image"],
                          "output_modalities": ["text"]},
         "pricing": {"prompt": 2, "completion": 10, "currency": "USD",
                     "unit": "per_million_tokens", "cacheReadInputPer1kTokens": 0.0001,
                     "cacheWriteInputPer1kTokens": 0.0025}},
        {"id": "unknown-chat", "object": "model", "context_length": None},
    ]},
    _AUDIO_PATH: {"object": "list", "data": [
        {"id": "speech-model", "object": "model",
         "architecture": {"input_modalities": ["text"], "output_modalities": ["audio"]},
         "capabilities": {"text_to_speech": True},
         "pricing": {"per_thousand_chars": 0.136, "currency": "USD"},
         "supported_parameters": {"max_chars": 10000, "voices": ["Aria"]}},
        # The live type=tts catalog also includes music; do not offer it as speech.
        {"id": "music-model", "object": "model",
         "architecture": {"input_modalities": ["text"], "output_modalities": ["music"]},
         "capabilities": {"text_to_music": True},
         "pricing": {"per_second": 0.01, "currency": "USD"}},
    ]},
    _EMBED_PATH: {"object": "list", "data": [
        {"id": "embedding-model", "object": "model", "max_tokens": 8191,
         "dimensions": 1536, "supports_dimensions": True,
         "pricing": {"per_million_tokens": 0.02, "currency": "USD"}},
    ]},
}


def _adapter(**settings):
    ctx = adapters.AdapterContext(
        auth_headers=lambda b: {"authorization": "Bearer catalog-key"},
        inflight_inc=lambda bid: None, inflight_dec=lambda bid: None,
        cost_usd=lambda *args: 0.0, source_of=lambda request: "",
        record_call=lambda **kw: None, log_enabled=lambda: False)
    return adapters.NanoGPTTextAdapter(
        {"name": "nano", "type": "nanogpt", "url": "https://nano.test/api",
         **settings}, ctx)


def _client(catalogs, failure=None, response_type=httpx.Response):
    def handler(request):
        path = request.url.path
        if path not in catalogs or request.method != "GET":
            return httpx.Response(404)
        if request.headers.get("authorization") != "Bearer catalog-key":
            return httpx.Response(401)
        if path == _TEXT_PATH and request.url.params.get("detailed") != "true":
            return httpx.Response(200, json={"object": "list", "data": []})
        if path == _AUDIO_PATH and (request.url.params.get("type") != "tts"
                                   or request.url.params.get("detailed") != "true"):
            return httpx.Response(200, json={"object": "list", "data": []})
        if failure and path == failure[0]:
            kind = failure[1]
            if kind == "transport":
                raise httpx.ConnectError("catalog unreachable", request=request)
            if kind == "json":
                return response_type(200, content=b"not JSON")
            if kind == "status":
                return response_type(503, json={"error": "unavailable"})
        return response_type(200, content=json.dumps(catalogs[path]).encode())
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class NanoGPTDiscovery(unittest.IsolatedAsyncioTestCase):
    async def _discover(self, catalogs=None, **settings):
        async with _client(_CATALOGS if catalogs is None else catalogs) as client:
            return await _adapter(**settings).discover(client)

    async def test_three_catalogs_merge_with_correct_token_and_cache_units(self):
        caps = await self._discover()
        self.assertEqual(caps.models,
                         {"chat-model", "unknown-chat", "speech-model", "embedding-model"})
        self.assertEqual(caps.pricing, {
            "chat-model": {"input": 2.0, "output": 10.0,
                           "cache_read": 0.1, "cache_write": 2.5},
            "embedding-model": {"input": 0.02, "output": 0.0},
        })
        self.assertEqual(caps.context, {"chat-model": 1050000, "embedding-model": 8191})
        self.assertIsNone(caps.running)  # no invented llama-swap current endpoint

    async def test_chat_filter_does_not_delete_audio_or_embedding_models(self):
        catalogs = copy.deepcopy(_CATALOGS)
        catalogs[_TEXT_PATH]["data"].append({
            "id": "image-only", "architecture": {"output_modalities": ["image"]}})
        caps = await self._discover(catalogs, chat_only=True)
        self.assertEqual(caps.models,
                         {"chat-model", "unknown-chat", "speech-model", "embedding-model"})

    async def test_token_serverless_filter_only_applies_to_text_catalog(self):
        caps = await self._discover(chat_only=True, serverless_only=True)
        self.assertEqual(caps.models, {"chat-model", "speech-model", "embedding-model"})

    async def test_refresh_allow_deny_and_extras_can_select_nonchat_models(self):
        adapter = _adapter(chat_only=True, models_allow="speech-*,embedding-*",
                           models_deny="embedding-*", models_extra="private-embedding")
        async with _client(_CATALOGS) as client:
            caps = await adapter.discover(client)
        # These are the real success-boundary helpers refresh_backend applies to all types.
        models = adapters.add_model_extras(
            adapters.filter_models(caps.models, adapter.backend), adapter.backend)
        self.assertEqual(models, {"speech-model", "private-embedding"})

    async def test_unknown_or_invalid_prices_never_become_free_token_prices(self):
        bad_prices = [None, {}, {"prompt": 2},
                      {"prompt": 2, "completion": None, "unit": "per_million_tokens"},
                      {"prompt": 2, "completion": 3, "unit": "per_token"},
                      {"prompt": 2, "completion": 3, "unit": "per_million_tokens",
                       "currency": "EUR"},
                      {"prompt": True, "completion": 3, "unit": "per_million_tokens"},
                      {"prompt": -1, "completion": 3, "unit": "per_million_tokens"},
                      {"prompt": "NaN", "completion": 3, "unit": "per_million_tokens"},
                      {"prompt": 2, "completion": "Infinity", "unit": "per_million_tokens"}]
        for price in bad_prices:
            with self.subTest(price=price):
                catalogs = copy.deepcopy(_CATALOGS)
                catalogs[_TEXT_PATH]["data"][0]["pricing"] = price
                catalogs[_EMBED_PATH]["data"][0]["pricing"] = {"currency": "USD"}
                caps = await self._discover(catalogs)
                self.assertIn("chat-model", caps.models)
                self.assertIn("embedding-model", caps.models)
                self.assertEqual(caps.pricing, {})

    async def test_reported_zero_is_known_but_character_and_flat_fees_are_not_tokens(self):
        catalogs = copy.deepcopy(_CATALOGS)
        catalogs[_TEXT_PATH]["data"][0]["pricing"] = {
            "prompt": 0, "completion": 0, "unit": "per_million_tokens"}
        catalogs[_EMBED_PATH]["data"][0]["pricing"] = {"per_million_tokens": 0}
        catalogs[_AUDIO_PATH]["data"][0]["pricing"] = {
            "per_generation": 0.15, "per_thousand_chars": 0, "currency": "USD"}
        caps = await self._discover(catalogs)
        self.assertEqual(caps.pricing, {
            "chat-model": {"input": 0.0, "output": 0.0},
            "embedding-model": {"input": 0.0, "output": 0.0}})
        self.assertIn("speech-model", caps.models)
        self.assertNotIn("speech-model", caps.context)

    async def test_only_documented_valid_cache_units_are_retained(self):
        catalogs = copy.deepcopy(_CATALOGS)
        price = catalogs[_TEXT_PATH]["data"][0]["pricing"]
        price.pop("cacheReadInputPer1kTokens")
        price["input_cache_read"] = 0.001  # OpenRouter spelling/units are not NanoGPT's
        price["cacheWriteInputPer1kTokens"] = "NaN"
        caps = await self._discover(catalogs)
        self.assertEqual(caps.pricing["chat-model"], {"input": 2.0, "output": 10.0})
        price["cacheReadInputPer1kTokens"] = 0
        caps = await self._discover(catalogs)
        self.assertEqual(caps.pricing["chat-model"],
                         {"input": 2.0, "output": 10.0, "cache_read": 0.0})

    async def test_failed_catalog_aborts_refresh_instead_of_reusing_stale_models(self):
        adapter = _adapter()
        async with _client(_CATALOGS) as client:
            await adapter.discover(client)
        for path in (_TEXT_PATH, _AUDIO_PATH, _EMBED_PATH):
            for failure, error in (("status", httpx.HTTPStatusError),
                                   ("transport", httpx.ConnectError), ("json", ValueError)):
                with self.subTest(path=path, failure=failure):
                    async with _client(_CATALOGS, (path, failure)) as client:
                        with self.assertRaises(error):
                            await adapter.discover(client)
        changed = copy.deepcopy(_CATALOGS)
        changed[_AUDIO_PATH]["data"] = []
        changed[_EMBED_PATH]["data"][0]["id"] = "new-embedding"
        async with _client(changed) as client:
            caps = await adapter.discover(client)
        self.assertEqual(caps.models, {"chat-model", "unknown-chat", "new-embedding"})
        self.assertNotIn("embedding-model", caps.pricing)
        self.assertNotIn("embedding-model", caps.context)

    async def test_malformed_catalog_is_failure_not_healthy_empty_inventory(self):
        for path in (_TEXT_PATH, _AUDIO_PATH, _EMBED_PATH):
            for payload in ({"error": "not a catalog"}, {"data": None}, {"data": {}},
                            {"data": [None]}, {"data": [{"id": 42}]},
                            {"data": [{"id": ""}]}):
                with self.subTest(path=path, payload=payload):
                    catalogs = copy.deepcopy(_CATALOGS)
                    catalogs[path] = payload
                    async with _client(catalogs) as client:
                        with self.assertRaises(ValueError):
                            await _adapter().discover(client)

    async def test_catalog_json_decoding_runs_off_event_loop(self):
        loop_thread = threading.get_ident()
        seen_threads = []

        class TrackedResponse(httpx.Response):
            def json(self, **kwargs):
                seen_threads.append(threading.get_ident())
                return super().json(**kwargs)

        async with _client(_CATALOGS, response_type=TrackedResponse) as client:
            caps = await _adapter().discover(client)
        self.assertEqual(caps.models,
                         {"chat-model", "unknown-chat", "speech-model", "embedding-model"})
        self.assertEqual(len(seen_threads), 3)
        self.assertNotIn(loop_thread, seen_threads)


class NanoUnifiedDiscovery(unittest.IsolatedAsyncioTestCase):
    """A single paid backend exposes six atomic, purpose-separated inventories."""

    def adapter(self, **settings):
        component = _adapter(**settings)
        return adapters.make_adapter(component.backend, component.ctx)

    def catalogs(self):
        catalogs = copy.deepcopy(_CATALOGS)
        catalogs.update({
            "/api/v1/images/models": {"data": [
                {"id": "image-model", "capabilities": {"text_to_image": True},
                 "supported_parameters": {"max_input_images": 0, "max_output_images": 1},
                 "pricing": {"per_image": 0.04}}]},
            "/api/v1/video-models": {"data": [
                {"id": "video-model", "pricing": {"per_second": 0.1}}]},
            "/api/v1/3d-models": {"data": [
                {"id": "mesh-model", "capabilities": {"model_3d_generation": True}},
                {"id": "retexture-only", "capabilities": {"model_3d_generation": False}}]},
        })
        return catalogs

    def client(self, catalogs=None, failure=None, response_type=httpx.Response):
        catalogs = self.catalogs() if catalogs is None else catalogs

        def handler(request):
            path = request.url.path
            if path not in catalogs or request.method != "GET":
                return httpx.Response(404)
            if failure and path == failure:
                return response_type(503, json={"error": "catalog unavailable"})
            return response_type(200, content=json.dumps(catalogs[path]).encode())

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_public_backend_discovers_six_modes_without_media_token_prices(self):
        adapter = self.adapter()
        async with self.client() as client:
            caps = await adapter.discover(client)
        self.assertIsInstance(adapter, adapters.NanoGPTAdapter)
        self.assertEqual(caps.models, {"chat-model", "unknown-chat", "speech-model",
                         "embedding-model", "image-model", "video-model", "mesh-model"})
        self.assertEqual({kind: set(rows) for kind, rows in adapter.model_catalogs.items()}, {
            "text": {"chat-model", "unknown-chat"}, "tts": {"speech-model"},
            "embedding": {"embedding-model"}, "images": {"image-model"},
            "video": {"video-model"}, "3d": {"mesh-model"}})
        self.assertEqual(caps.pricing, {
            "chat-model": {"input": 2.0, "output": 10.0,
                           "cache_read": 0.1, "cache_write": 2.5},
            "embedding-model": {"input": 0.02, "output": 0.0}})
        self.assertEqual(caps.context, {"chat-model": 1050000, "embedding-model": 8191})
        self.assertEqual(adapter.model_catalogs["tts"]["speech-model"]["pricing"],
                         {"per_thousand_chars": 0.136, "currency": "USD"})
        self.assertTrue(adapter.serves_generation and adapter.cloud and adapter.bills)
        for registry in (adapters.ADAPTERS, adapters.GEN_TYPES,
                         adapters.CLOUD_TYPES, adapters.BILLING_TYPES):
            self.assertIn("nanogpt", registry)
            self.assertNotIn("nanogpt_media", registry)

    async def test_filters_narrow_every_kind_without_applying_chat_filter_to_media(self):
        adapter = self.adapter(chat_only=True, serverless_only=True,
            models_allow="chat-*,speech-*,embedding-*,image-*,video-*,mesh-*",
            models_deny="video-*")
        async with self.client() as client:
            caps = await adapter.discover(client)
        self.assertEqual(caps.models, {"chat-model", "speech-model", "embedding-model",
                                      "image-model", "mesh-model"})
        self.assertEqual(adapter.model_catalogs["video"], {})
        self.assertEqual(adapter.model_catalogs["text"].keys(), {"chat-model"})

    async def test_media_output_rows_cannot_leak_into_text_inventory(self):
        catalogs = self.catalogs()
        catalogs[_TEXT_PATH]["data"].extend([
            {"id": "image-model", "architecture": {"output_modalities": ["image"]}},
            {"id": "speech-model", "architecture": {"output_modalities": ["audio"]}},
        ])
        adapter = self.adapter()
        async with self.client(catalogs) as client:
            await adapter.discover(client)
        self.assertEqual(set(adapter.model_catalogs["text"]), {"chat-model", "unknown-chat"})
        self.assertEqual(set(adapter.model_catalogs["images"]), {"image-model"})
        self.assertEqual(set(adapter.model_catalogs["tts"]), {"speech-model"})

    async def test_any_failed_catalog_clears_entire_previous_snapshot(self):
        adapter = self.adapter()
        for path in self.catalogs():
            async with self.client() as client:
                await adapter.discover(client)
            async with self.client(failure=path) as client:
                with self.assertRaises(httpx.HTTPStatusError):
                    await adapter.discover(client)
            self.assertEqual(adapter.model_catalogs, {})

    async def test_snapshot_is_not_published_while_one_catalog_is_pending(self):
        adapter = self.adapter()
        waiting, release = asyncio.Event(), asyncio.Event()
        catalogs = self.catalogs()

        async def handler(request):
            if request.url.path == "/api/v1/video-models":
                waiting.set()
                await release.wait()
            return httpx.Response(200, json=catalogs[request.url.path])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            task = asyncio.create_task(adapter.discover(client))
            try:
                await waiting.wait()
                self.assertEqual(adapter.model_catalogs, {})
            finally:
                release.set()
                await task
        self.assertEqual(set(adapter.model_catalogs),
                         {"text", "tts", "embedding", "images", "video", "3d"})

    async def test_replacement_preserves_account_snapshot_but_refilters_settings(self):
        old = self.adapter(api_key="key")
        async with self.client() as client:
            await old.discover(client)
        for hook in ("adopt_state", "adopt_discovery"):
            replacement = self.adapter(api_key="key", models_deny="image-*", chat_only=True)
            getattr(replacement, hook)(old)
            self.assertEqual(replacement.model_catalogs["images"], {})
            self.assertEqual(set(replacement.model_catalogs["text"]),
                             {"chat-model", "unknown-chat"})
            replacement.model_catalogs["tts"]["speech-model"]["supported_parameters"]["voices"].append("changed")
            self.assertEqual(old.model_catalogs["tts"]["speech-model"]["supported_parameters"]["voices"], ["Aria"])
            changed_account = self.adapter(api_key="different")
            getattr(changed_account, hook)(old)
            self.assertEqual(changed_account.model_catalogs, {})
            changed_host = self.adapter(api_key="key", url="https://other.test/api")
            getattr(changed_host, hook)(old)
            self.assertEqual(changed_host.model_catalogs, {})

    async def test_all_six_catalogs_decode_in_workers(self):
        threads = []
        loop_thread = threading.get_ident()

        class TrackedResponse(httpx.Response):
            def json(self, **kwargs):
                threads.append(threading.get_ident())
                return super().json(**kwargs)

        async with self.client(response_type=TrackedResponse) as client:
            await self.adapter().discover(client)
        self.assertEqual(len(threads), 6)
        self.assertNotIn(loop_thread, threads)


if __name__ == "__main__":
    unittest.main()
