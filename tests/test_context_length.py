"""context_length on /v1/models — the number a client sizes its prompts by.

Why this fails SILENTLY: a client that finds no context_length does not error, it
assumes a default (Oh My Pi: 128000) and happily builds a 33k prompt for a 32k
model — the backend's 400 is the first anyone hears of it (measured 2026-09-08,
glm-5.3-flash on dx10-01, four 400s). So each source, the override precedence and
the alias minimum are pinned here: a wrong number is a plausible number.
"""
import asyncio
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    import admin
    import adapters
    import store
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import httpx
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)     # keep the run's output pristine
logging.getLogger("store").setLevel(logging.WARNING)


# ── the pure parser: one reader for every listing shape ──────────────────────

class ExtractContext(unittest.TestCase):
    def test_openrouter_and_together_context_length(self):
        out = adapters.extract_context({"data": [
            {"id": "z-ai/glm-5.3-flash", "context_length": 202752},
            {"id": "no-ctx"},
        ]})
        self.assertEqual(out, {"z-ai/glm-5.3-flash": 202752})

    def test_vllm_max_model_len(self):
        out = adapters.extract_context({"data": [{"id": "qwen", "max_model_len": 131072}]})
        self.assertEqual(out, {"qwen": 131072})

    def test_llama_server_meta_n_ctx(self):
        out = adapters.extract_context({"data": [
            {"id": "glm", "meta": {"n_ctx": 32768, "n_ctx_train": 1048576}}]})
        self.assertEqual(out, {"glm": 32768})       # the SERVING size, never n_ctx_train

    def test_bare_list_payload(self):
        out = adapters.extract_context([{"id": "m", "context_length": 8192}])
        self.assertEqual(out, {"m": 8192})

    def test_non_positive_and_non_numeric_ignored(self):
        out = adapters.extract_context({"data": [
            {"id": "a", "context_length": 0}, {"id": "b", "context_length": "big"},
            {"id": "c", "context_length": -1}, {"id": "d", "context_length": 4096.0}]})
        self.assertEqual(out, {"d": 4096})


# ── the admin override: glob=tokens lines, first match wins, else learned ──────

class ModelContextRules(unittest.TestCase):
    def test_parse_lines(self):
        rules = adapters.parse_model_context("glm-*=32768\n# comment\n\n qwen3.8-flash-next-ple4 = 131072 \n")
        self.assertEqual(rules, [("glm-*", 32768), ("qwen3.8-flash-next-ple4", 131072)])

    def test_bad_lines_are_dropped_not_fatal(self):
        rules = adapters.parse_model_context("glm-*\nx=abc\ny=0\nok=1")
        self.assertEqual(rules, [("ok", 1)])

    def test_override_beats_learned(self):
        b = {"name": "dx", "model_context": "glm-*=32768"}
        self.assertEqual(adapters.model_context_for(b, "glm-5.3-flash", {"glm-5.3-flash": 131072}), 32768)

    def test_learned_when_no_rule_matches(self):
        b = {"name": "dx", "model_context": "glm-*=32768"}
        self.assertEqual(adapters.model_context_for(b, "qwen", {"qwen": 262144}), 262144)

    def test_unknown_is_none(self):
        self.assertIsNone(adapters.model_context_for({"name": "dx"}, "qwen", {}))

    def test_first_matching_rule_wins(self):
        b = {"name": "dx", "model_context": "glm-5.3-*=32768\nglm-*=65536"}
        self.assertEqual(adapters.model_context_for(b, "glm-5.3-flash", {}), 32768)
        self.assertEqual(adapters.model_context_for(b, "glm-4", {}), 65536)


# ── discovery: llama-swap's list hides n_ctx; ask the LOADED upstreams only ──

class _Swap(BaseHTTPRequestHandler):
    running: list = []          # llama-swap /running entries
    upstream_ctx: dict = {}     # model → n_ctx answered by /upstream/<model>/v1/models
    running_status = 200
    hits: list = []

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        _Swap.hits.append(self.path)
        if self.path == "/v1/models":
            return self._json(200, {"data": [
                {"id": "glm", "owned_by": "llama-swap", "status": {"value": "loaded"}},
                {"id": "qwen", "owned_by": "llama-swap", "status": {"value": "unloaded"}},
                {"id": "gemma", "owned_by": "llama-swap", "status": {"value": "unloaded"}},
            ]})
        if self.path == "/running":
            if _Swap.running_status != 200:
                return self._json(_Swap.running_status, {"error": "nope"})
            return self._json(200, {"running": _Swap.running})
        if self.path.startswith("/upstream/"):
            model = self.path[len("/upstream/"):].split("/", 1)[0]
            if model in _Swap.upstream_ctx:
                return self._json(200, {"data": [{"id": model, "owned_by": "llamacpp",
                                                  "meta": {"n_ctx": _Swap.upstream_ctx[model],
                                                           "n_ctx_train": 1048576}}]})
            return self._json(500, {"error": "would have LOADED the model"})
        self._json(404, {"error": "nope"})


def _ctx():
    return adapters.AdapterContext(
        auth_headers=lambda b: {}, inflight_inc=lambda bid: None, inflight_dec=lambda bid: None,
        cost_usd=lambda *a: 0.0, source_of=lambda r: "test", record_call=lambda *a, **k: None,
        log_enabled=lambda: False)


class LlamaSwapDiscovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _Swap)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        _Swap.running, _Swap.upstream_ctx, _Swap.hits, _Swap.running_status = [], {}, [], 200
        self.ad = adapters.OpenAIAdapter({"name": "dx", "type": "openai", "url": self.url}, _ctx())

    def _discover(self):
        async def go():
            async with httpx.AsyncClient() as c:
                return await self.ad.discover(c)
        return asyncio.run(go())

    def test_loaded_model_context_comes_from_its_upstream(self):
        _Swap.running = [{"model": "glm", "state": "ready"}]
        _Swap.upstream_ctx = {"glm": 32768}
        caps = self._discover()
        self.assertEqual(caps.models, {"glm", "qwen", "gemma"})
        self.assertEqual(caps.context, {"glm": 32768})

    def test_unloaded_models_are_never_asked(self):
        # /upstream/<model>/… on an unloaded model makes llama-swap LOAD it — a discovery
        # poll must never do that. Only entries /running reports as ready are asked.
        _Swap.running = [{"model": "glm", "state": "ready"}, {"model": "qwen", "state": "starting"}]
        _Swap.upstream_ctx = {"glm": 32768, "qwen": 4096}
        caps = self._discover()
        self.assertEqual(caps.context, {"glm": 32768})
        self.assertNotIn("/upstream/qwen/v1/models", _Swap.hits)
        self.assertNotIn("/upstream/gemma/v1/models", _Swap.hits)

    def test_no_running_endpoint_means_listing_only(self):
        _Swap.running_status = 404                    # vLLM, OpenRouter, LocalAI: no /running
        caps = self._discover()
        self.assertEqual(caps.context, {})
        self.assertEqual(caps.models, {"glm", "qwen", "gemma"})

    def test_upstream_failure_does_not_fail_discovery(self):
        _Swap.running = [{"model": "glm", "state": "ready"}]
        _Swap.upstream_ctx = {}                       # /upstream answers 500
        caps = self._discover()
        self.assertEqual(caps.context, {})
        self.assertEqual(caps.models, {"glm", "qwen", "gemma"})


# ── main: learned values are remembered, the catalog carries the number ───────

class _Req:
    def __init__(self, q=None):
        self.query_params = q or {}
        self.state = types.SimpleNamespace()
        self.headers = {}
        self.client = None


class Catalog(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(main, k) for k in
                       ("backends", "virtual_models", "backend_models", "backend_healthy",
                        "backend_context", "model_prefix", "users", "api_key")}
        a = {"name": "a", "type": "openai", "url": "http://a", "enabled": True, "local": True}
        b = {"name": "b", "type": "openai", "url": "http://b", "enabled": True, "local": True,
             "model_context": "glm*=32768"}
        main.backends = [a, b]
        main.virtual_models = {"chat": {"a": "glm", "b": "glm"}, "solo": "qwen", "none": "nothing"}
        main.backend_models = {main.backend_id(a): {"glm", "qwen", "nothing"},
                               main.backend_id(b): {"glm", "qwen"}}
        main.backend_healthy = {main.backend_id(a): True, main.backend_id(b): True}
        main.backend_context = {main.backend_id(a): {"glm": 131072, "qwen": 262144},
                                main.backend_id(b): {"qwen": 65536}}
        main.model_prefix = True
        main.users, main.api_key = [], ""
        main.rebuild_route_index()

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(main, k, v)
        main.rebuild_route_index()

    def _list(self):
        data = asyncio.run(main.list_models(_Req(), None))["data"]
        return {m["id"]: m for m in data}

    def test_prefixed_entry_carries_its_backend_value(self):
        cat = self._list()
        self.assertEqual(cat["a/glm"]["context_length"], 131072)   # learned
        self.assertEqual(cat["b/glm"]["context_length"], 32768)    # admin rule beats learned

    def test_bare_id_and_alias_take_the_minimum_over_backends(self):
        cat = self._list()
        self.assertEqual(cat["glm"]["context_length"], 32768)      # bare id: a AND b serve it
        self.assertEqual(cat["chat"]["context_length"], 32768)     # alias over both
        self.assertEqual(cat["solo"]["context_length"], 65536)     # qwen: 262144 vs 65536

    def test_unknown_stays_absent_not_zero(self):
        cat = self._list()
        self.assertNotIn("context_length", cat["a/nothing"])
        self.assertNotIn("context_length", cat["none"])

    def test_get_model_matches_the_listing(self):
        one = asyncio.run(main.get_model("b/glm", None))
        self.assertEqual(one["context_length"], 32768)
        alias = asyncio.run(main.get_model("chat", None))
        self.assertEqual(alias["context_length"], 32768)
        bare = asyncio.run(main.get_model("glm", None))
        self.assertEqual(bare["context_length"], 32768)

    def test_merge_keeps_what_this_poll_did_not_see(self):
        # A llama-swap model reports its n_ctx only while LOADED; a later poll with it
        # unloaded must not forget the number (that is the whole point of persisting).
        merged = main.merge_learned_context({"glm": 32768, "old": 4096}, {"qwen": 65536, "glm": 16384})
        self.assertEqual(merged, {"glm": 16384, "old": 4096, "qwen": 65536})


class StorePersistence(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            saved = (store._DB_PATH, store._active)
            try:
                store.init(os.path.join(d, "store.db"))
                store.save_backend_context("openai:dx", {"glm": 32768, "qwen": 65536})
                store.save_backend_context("openai:dx", {"glm": 32768, "qwen": 4096})  # overwrite
                self.assertEqual(store.load_backend_context(),
                                 {"openai:dx": {"glm": 32768, "qwen": 4096}})
            finally:
                store._DB_PATH, store._active = saved


# ── console: the rule is editable, the LLM models tab shows the number ─────────

class Console(unittest.TestCase):
    def test_backend_form_renders_the_rules(self):
        html = admin._backend_form({"name": "dx", "type": "openai", "url": "http://x",
                                    "model_context": "glm-*=32768"}, [])
        self.assertIn('name="model_context"', html)
        self.assertIn("glm-*=32768", html)

    def test_host_chip_shows_context(self):
        chip = admin._host_chip({"backend": "dx", "healthy": True, "busy": False, "tps": 0, "ctx": 32768})
        self.assertIn("32k", chip)
        plain = admin._host_chip({"backend": "dx", "healthy": True, "busy": False, "tps": 0})
        self.assertNotIn("ctx", plain)

    def test_snapshot_hosts_carry_context(self):
        saved = {k: getattr(main, k) for k in
                 ("backends", "virtual_models", "backend_models", "backend_healthy", "backend_context")}
        a = {"name": "a", "type": "openai", "url": "http://a", "enabled": True}
        try:
            main.backends, main.virtual_models = [a], {}
            main.backend_models = {main.backend_id(a): {"glm"}}
            main.backend_healthy = {main.backend_id(a): True}
            main.backend_context = {main.backend_id(a): {"glm": 32768}}
            snap = main.routing_snapshot()
            self.assertEqual(snap["models"][0]["hosts"][0]["ctx"], 32768)
        finally:
            for k, v in saved.items():
                setattr(main, k, v)


if __name__ == "__main__":
    unittest.main()
