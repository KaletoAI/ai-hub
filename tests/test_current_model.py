"""`<backend>/current` — a llama-swap call that takes whatever model is loaded.

Why this fails SILENTLY: every wrong pick is a plausible answer. A chat call resolved
to the embedding model llama-swap keeps loaded beside the chat model (bge-m3, `ttl 0`,
measured 2026-09-13 on both boxes) comes back as a 400 that names a model the caller
never asked for; a pick against a stale list, or a fallback that picks "something",
SWAPS a model in — the one thing `current` exists to avoid — and the reply still looks
fine. A backend with nothing suitable loaded taken as a candidate turns a parkable call
into a 503, and one that never counts as `current`-capable just reads as "model absent".
So the pick rule, the routing on top of it, the live refresh, the allow-list, the
catalog and what the console shows are all pinned here.
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
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import httpx
import logging
from fastapi import HTTPException

logging.getLogger("httpx").setLevel(logging.WARNING)

EMBED_CMD = "llama-server --port 9011 -m bge-m3.gguf --embedding --pooling cls --alias bge-m3"
CHAT_CMD = "llama-server --port 9012 -m gemma.gguf --ctx-size 32768 --alias gemma"


def _ent(model, state="ready", kind="chat"):
    return {"model": model, "state": state, "kind": kind}


# ── the pure half: /running → entries, entries → the pick ─────────────────────

class ParseRunning(unittest.TestCase):
    def test_kind_from_llama_server_flags(self):
        out = adapters.parse_running({"running": [
            {"model": "bge-m3", "state": "ready", "cmd": EMBED_CMD},
            {"model": "gemma", "state": "starting", "cmd": CHAT_CMD},
            {"model": "rr", "state": "ready", "cmd": "llama-server --reranking=true"},
            {"model": "nocmd", "state": "ready"},
        ]})
        self.assertEqual(out, [_ent("bge-m3", kind="embedding"), _ent("gemma", "starting"),
                               _ent("rr", kind="rerank"), _ent("nocmd")])

    def test_not_llama_swap_is_none_not_empty(self):
        # A server whose /running is something else must not look like an idle llama-swap.
        self.assertIsNone(adapters.parse_running({"status": "ok"}))
        self.assertIsNone(adapters.parse_running(["x"]))
        self.assertEqual(adapters.parse_running({"running": []}), [])

    def test_entries_without_model_are_dropped(self):
        self.assertEqual(adapters.parse_running({"running": [{"state": "ready"}, "junk"]}), [])


class PickCurrent(unittest.TestCase):
    served = {"bge-m3", "gemma", "qwen"}

    def test_chat_never_gets_the_embedding_model(self):
        running = [_ent("bge-m3", kind="embedding"), _ent("gemma")]
        self.assertEqual(adapters.pick_current(running, "/v1/chat/completions", self.served, None), "gemma")
        self.assertIsNone(adapters.pick_current(running[:1], "/v1/chat/completions", self.served, None))

    def test_embeddings_get_only_the_embedding_model(self):
        running = [_ent("gemma"), _ent("bge-m3", kind="embedding")]
        self.assertEqual(adapters.pick_current(running, "/v1/embeddings", self.served, None), "bge-m3")
        self.assertIsNone(adapters.pick_current(running[:1], "/v1/embeddings", self.served, None))

    def test_ready_beats_starting_and_starting_still_counts(self):
        running = [_ent("qwen", "starting"), _ent("gemma")]
        self.assertEqual(adapters.pick_current(running, "/v1/chat/completions", self.served, None), "gemma")
        self.assertEqual(adapters.pick_current(running[:1], "/v1/chat/completions", self.served, None), "qwen")

    def test_stopping_is_not_loaded(self):
        self.assertIsNone(adapters.pick_current([_ent("gemma", "stopping")], "/v1/chat/completions",
                                                self.served, None))

    def test_last_dispatched_wins_among_equals(self):
        running = [_ent("gemma"), _ent("qwen")]
        self.assertEqual(adapters.pick_current(running, "/v1/chat/completions", self.served, "qwen"), "qwen")
        self.assertEqual(adapters.pick_current(running, "/v1/chat/completions", self.served, "gone"), "gemma")

    def test_model_filter_holds(self):
        # models_deny removed gemma from the served set → `current` must not reach it either
        self.assertIsNone(adapters.pick_current([_ent("gemma")], "/v1/chat/completions", {"qwen"}, None))


# ── discovery carries the list; None where there is no /running ───────────────

class _Swap(BaseHTTPRequestHandler):
    running_status = 200

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
        if self.path == "/v1/models":
            return self._json(200, {"data": [{"id": "gemma"}, {"id": "bge-m3"}]})
        if self.path == "/running" and _Swap.running_status == 200:
            return self._json(200, {"running": [{"model": "bge-m3", "state": "ready", "cmd": EMBED_CMD}]})
        self._json(404, {"error": "nope"})


def _ctx():
    return adapters.AdapterContext(
        auth_headers=lambda b: {}, inflight_inc=lambda bid: None, inflight_dec=lambda bid: None,
        cost_usd=lambda *a: 0.0, source_of=lambda r: "test", record_call=lambda *a, **k: None,
        log_enabled=lambda: False)


class Discovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _Swap)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.ad = adapters.OpenAIAdapter(
            {"name": "sw", "type": "openai", "url": f"http://127.0.0.1:{cls.srv.server_port}"}, _ctx())

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def _discover(self):
        async def go():
            async with httpx.AsyncClient() as c:
                return await self.ad.discover(c)
        return asyncio.run(go())

    def test_llama_swap_reports_running(self):
        _Swap.running_status = 200
        self.assertEqual(self._discover().running, [_ent("bge-m3", kind="embedding")])

    def test_no_running_endpoint_is_none(self):
        _Swap.running_status = 404
        try:
            self.assertIsNone(self._discover().running)
        finally:
            _Swap.running_status = 200


# ── main: routing, refresh, allow-list, catalog, summaries ────────────────────

class _Req:
    def __init__(self):
        self.query_params = {}
        self.state = types.SimpleNamespace()
        self.headers = {}
        self.client = None


class _FakeAdapter:
    """Records what dispatch was handed; answers /running from a settable list."""
    def __init__(self, running=None):
        self.running = running
        self.fetches = 0
        self.dispatched = []

    async def fetch_running(self, client, timeout=5.0):
        self.fetches += 1
        return self.running

    async def dispatch(self, req):
        self.dispatched.append((req.real_model, req.body.get("model")))
        return "ok"


_STATE = ("backends", "virtual_models", "backend_models", "backend_healthy", "backend_running",
          "backend_adapters", "backend_inflight", "backend_last_key", "users", "api_key",
          "model_prefix", "_parked")


class _MainCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(main, k) for k in _STATE}
        self.a = {"name": "a", "type": "openai", "url": "http://a", "enabled": True, "max_concurrent": 1}
        self.b = {"name": "b", "type": "openai", "url": "http://b", "enabled": True}
        self.c = {"name": "c", "type": "openai", "url": "http://c", "enabled": True}   # vLLM: no /running
        self.ida, self.idb, self.idc = (main.backend_id(x) for x in (self.a, self.b, self.c))
        main.backends = [self.a, self.b, self.c]
        main.virtual_models = {"egal": {"a": "current", "b": "current"}, "plain": "gemma"}
        main.backend_models = {self.ida: {"gemma", "bge-m3"}, self.idb: {"gemma", "bge-m3"},
                               self.idc: {"gemma"}}
        main.backend_healthy = {self.ida: True, self.idb: True, self.idc: True}
        main.backend_running = {self.ida: [_ent("bge-m3", kind="embedding"), _ent("gemma")],
                                self.idb: [_ent("bge-m3", kind="embedding")]}
        self.fa, self.fb = _FakeAdapter(main.backend_running[self.ida]), _FakeAdapter(main.backend_running[self.idb])
        main.backend_adapters = {self.ida: self.fa, self.idb: self.fb, self.idc: _FakeAdapter()}
        main.backend_inflight, main.backend_last_key = {}, {}
        main.users, main.api_key, main.model_prefix = [], "", True
        main._parked = []
        main.rebuild_route_index()

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(main, k, v)
        main.rebuild_route_index()


class Routing(_MainCase):
    def test_prefixed_current_resolves_to_the_loaded_chat_model(self):
        self.assertEqual(main.resolve_routes("a/current"), ([(self.a, "gemma")], []))
        self.assertEqual(main.resolve_routes("a/current", "/v1/embeddings"), ([(self.a, "bge-m3")], []))

    def test_nothing_suitable_loaded_is_no_candidate_and_says_why(self):
        self.assertEqual(main.resolve_routes("b/current"), ([], []))
        err = main._nothing_loaded_error("b/current", "/v1/chat/completions")
        self.assertEqual(err.status_code, 503)
        self.assertIn("no chat model loaded on b", err.detail)

    def test_backend_without_running_endpoint_has_no_current(self):
        self.assertEqual(main.resolve_routes("c/current"), ([], []))
        self.assertIsNone(main._nothing_loaded_error("c/current", "/v1/chat/completions"))

    def test_alias_skips_the_backend_with_only_an_embedding_model(self):
        self.assertEqual(main.resolve_routes("egal"), ([(self.a, "gemma")], []))

    def test_busy_loaded_backend_parks_instead_of_spilling(self):
        # b is free but has no chat model; a has one and is at its cap → busy, so the call
        # PARKS for a instead of landing on b (which would have had to load something).
        main.backend_inflight = {self.ida: 1}
        self.assertEqual(main.resolve_routes("egal"), ([], [(self.a, "gemma")]))

    def test_a_listed_model_named_current_is_not_shadowed(self):
        main.backend_models[self.ida] = {"current", "gemma"}
        main.rebuild_route_index()
        self.assertEqual(main.resolve_routes("a/current"), ([(self.a, "current")], []))

    def test_index_follows_running_support(self):
        # discovery of a backend that has /running must put its `current` alias entry in
        # the index — a flip without a model-set change still rebuilds
        main.backend_running.pop(self.ida)
        main.rebuild_route_index()
        self.assertEqual(main.resolve_routes("egal"), ([], []))
        caps = adapters.Capabilities(models={"gemma", "bge-m3"}, pricing={},
                                     running=[_ent("gemma")])

        class _Disc(_FakeAdapter):
            async def discover(self, client):
                return caps
        main.backend_adapters[self.ida] = _Disc()
        asyncio.run(main.refresh_backend(self.a, None))
        self.assertEqual(main.resolve_routes("egal"), ([(self.a, "gemma")], []))
        caps.running = None                             # /running went away again
        asyncio.run(main.refresh_backend(self.a, None))
        self.assertNotIn(self.ida, main.backend_running)
        self.assertEqual(main.resolve_routes("egal"), ([], []))


class Dispatch(_MainCase):
    def test_body_model_is_rewritten_to_the_live_pick(self):
        # discovery still thinks gemma; the live query says qwen was swapped in meanwhile
        main.backend_models[self.ida] |= {"qwen"}
        self.fa.running = [_ent("bge-m3", kind="embedding"), _ent("qwen")]
        body = {"model": "a/current", "messages": []}
        out = asyncio.run(main._dispatch_or_park("a/current", "/v1/chat/completions", body, _Req()))
        self.assertEqual(out, "ok")
        self.assertEqual(self.fa.dispatched, [("qwen", "qwen")])
        self.assertEqual(body["model"], "a/current")                 # the shared body is untouched
        self.assertEqual(main.backend_last_key[self.ida], "qwen")

    def test_nothing_loaded_is_a_503_not_a_load(self):
        with self.assertRaises(HTTPException) as cm:
            asyncio.run(main._dispatch_or_park("b/current", "/v1/chat/completions",
                                               {"model": "b/current"}, _Req()))
        self.assertEqual(cm.exception.status_code, 503)
        self.assertIn("never loads one", cm.exception.detail)
        self.assertEqual(self.fb.dispatched, [])

    def test_plain_alias_asks_nobody(self):
        asyncio.run(main._dispatch_or_park("plain", "/v1/chat/completions", {"model": "plain"}, _Req()))
        self.assertEqual(self.fa.fetches + self.fb.fetches, 0)

    def test_failed_live_query_keeps_the_last_list(self):
        self.fa.running = None
        asyncio.run(main._refresh_loaded("a/current"))
        self.assertEqual(self.fa.fetches, 1)
        self.assertEqual(main.resolve_routes("a/current"), ([(self.a, "gemma")], []))


class AllowList(_MainCase):
    def _ok(self, allow, model):
        return main._model_allowed({"name": "u", "models": allow}, model)

    def test_whole_backend_grant_covers_current(self):
        self.assertTrue(self._ok(["a"], "a/current"))
        self.assertTrue(self._ok(["a/current"], "a/current"))

    def test_a_model_grant_does_not(self):
        # current may land on ANY loaded model — a grant for one of them is not enough
        self.assertFalse(self._ok(["gemma"], "a/current"))
        self.assertFalse(self._ok(["current"], "a/current"))

    def test_alias_with_current_entries(self):
        self.assertTrue(self._ok(["egal"], "egal"))
        self.assertTrue(self._ok(["a"], "egal"))           # alias routes to a granted backend
        self.assertFalse(self._ok(["c"], "egal"))


class Catalog(_MainCase):
    def test_listed_only_where_running_exists(self):
        data = asyncio.run(main.list_models(_Req(), None))["data"]
        ids = {m["id"]: m for m in data}
        self.assertIn("a/current", ids)
        self.assertIn("b/current", ids)                    # up, just nothing chat-loaded right now
        self.assertNotIn("c/current", ids)
        self.assertNotIn("context_length", ids["a/current"])

    def test_get_model(self):
        self.assertEqual(asyncio.run(main.get_model("a/current", None))["id"], "a/current")
        with self.assertRaises(HTTPException):
            asyncio.run(main.get_model("c/current", None))

    def test_routing_snapshot_counts_current_as_present(self):
        routes = {r["backend"]: r for a in main.routing_snapshot()["aliases"]
                  if a["alias"] == "egal" for r in a["routes"]}
        self.assertTrue(routes["a"]["present"])


class Console(_MainCase):
    def test_loaded_info_only_for_up_llama_swap(self):
        self.assertEqual(main._loaded_info(self.a)["loaded"][1]["model"], "gemma")
        self.assertEqual(main._loaded_info(self.c), {})
        main.backend_healthy[self.ida] = False
        self.assertEqual(main._loaded_info(self.a), {})     # a down backend's list is no fact

    def test_summaries_carry_loaded(self):
        snap = {b["name"]: b for b in main.dashboard_snapshot()["backends"]}
        self.assertEqual([e["model"] for e in snap["a"]["loaded"]], ["bge-m3", "gemma"])
        self.assertNotIn("loaded", snap["c"])
        health = asyncio.run(main.health())["backends"]
        self.assertIn("loaded", health[self.ida])

    def test_loaded_text(self):
        html = admin._loaded_text([_ent("bge-m3", kind="embedding"), _ent("qwen", "starting")])
        self.assertIn("<b>bge-m3</b>", html)
        self.assertIn("(embedding)", html)
        self.assertIn("(starting)", html)
        self.assertIn("nothing loaded", admin._loaded_text([]))
        self.assertEqual(admin._loaded_text(None), "")

    def test_backends_list_gets_plain_text(self):
        # the Backends list's sub line is escaped by _item — markup would render as tags
        plain = admin._loaded_text([_ent("bge-m3", kind="embedding"), _ent("gemma")], html=False)
        self.assertEqual(plain, "bge-m3 (embedding), gemma")
        self.assertEqual(admin._loaded_text([], html=False), "nothing loaded")
        info = {"backends": [{"name": "a", "type": "openai", "enabled": True, "healthy": True,
                              "url": "http://a", "models": 2, "loaded": [_ent("gemma")]}]}
        saved = admin._gateway_info
        admin._gateway_info = lambda: info
        try:
            page = asyncio.run(admin.backends_page(_Req()))
        finally:
            admin._gateway_info = saved
        body = page.body.decode() if hasattr(page, "body") else str(page)
        self.assertIn("· loaded gemma", body)
        self.assertNotIn("&lt;b&gt;", body)

    def test_dashboard_panel_shows_the_loaded_model(self):
        html = admin._dash_backends([{"name": "a", "type": "openai", "enabled": True, "healthy": True,
                                      "loaded": [_ent("gemma")]},
                                     {"name": "c", "type": "openai", "enabled": True, "healthy": True}], [])
        self.assertIn(">loaded</th>", html)
        self.assertIn("<b>gemma</b>", html)

    def test_alias_editor_offers_current(self):
        info = {b["name"]: b for b in main.llm_backends_info()}
        self.assertTrue(info["a"]["current"])
        self.assertFalse(info["c"]["current"])


if __name__ == "__main__":
    unittest.main()
