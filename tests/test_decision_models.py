"""Decision models (System One) on `POST /v1/systemone` — discovery, routing, dispatch.

A decision model writes no text: it gets a `state` and typed `questions` and answers each
with a probability per option. Anima-Versa calls OpenRouter's `typesafe/jev-1.14` through
the hub with its hub key (docs/decision-models-anforderung.md), and every way this breaks
is SILENT on the caller's side: Anima treats any non-200, timeout or odd JSON as "no
model" and quietly falls back to its old path, so a broken hub route looks exactly like a
model that is never sure. What is pinned:

- discovery: OpenRouter leaves decision models OUT of its normal `/v1/models` (they are
  listed under `?output_modalities=decisions` only), so without the second listing the
  model does not exist for routing, the allow-list or the console. A listing that names
  output modalities must name `decisions`, or `*` would turn every chat model into a
  decision model; one failed listing must not drop the models for ten minutes; and a
  decision model the whitelist removed is not served;
- routing (`serves_model`): a decision model answers /v1/systemone only and only a decision
  model answers it — the other way round each call is a (paid) upstream 400 — and the
  refusal names the right endpoint instead of "no healthy backend";
- dispatch: the body reaches the model as sent (the state can be long user content — no
  alias/backend sampling, no reasoning rewrite), to OpenRouter's measured Decisions API
  (or llama-swap's `/upstream/<model>`), with a short timeout, and the call is booked with
  OpenRouter's `usage.cost` under /v1/systemone;
- the endpoint end to end: auth + allow-list, a 400 for a body without a model, and the
  `?type=decision` / `?type=chat` split of `/v1/models`.

Run: venv/bin/python -m unittest tests.test_decision_models -v
"""
import asyncio
import json
import os
import sys
import tempfile
import types
import unittest

import httpx

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import adapters
    import main
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import logging

from fastapi import HTTPException
from fastapi.testclient import TestClient

logging.getLogger("store").setLevel(logging.WARNING)
logging.getLogger("main").setLevel(logging.CRITICAL)
logging.getLogger("adapters").setLevel(logging.CRITICAL)

JEV = "typesafe/jev-1.14"
DP = adapters.DECISION_PATH
OR = {"name": "openrouter", "type": "openai", "url": "https://openrouter.ai/api",
      "enabled": True, "paid": True, "decision_models": "typesafe/*"}
LS = {"name": "ls", "type": "openai", "url": "http://192.168.8.31:8080", "enabled": True,
      "decision_models": "laya"}

# OpenRouter /v1/models entries in its listing shape
JEV_ENTRY = {"id": JEV, "architecture": {"output_modalities": ["decisions"]},
             "pricing": {"prompt": "0.0000001", "completion": "0"}}
GPT_ENTRY = {"id": "openai/gpt-5", "architecture": {"output_modalities": ["text"]},
             "pricing": {"prompt": "0.000001", "completion": "0.000008"}}

QUESTION = {"model": JEV, "state": {"text": "smiles and waves at a friend"},
            "questions": {"entry": {"type": "choice",
                                    "instructions": "Which facial expression fits the text best?",
                                    "criteria": {"neutral": "neutral", "positive": "positive",
                                                 "sad": "sad", "none": "none of these"}}}}
ANSWER = {"answers": {"entry": {"type": "choice", "choice": "positive",
                                "probabilities": {"neutral": 0.03, "positive": 0.95,
                                                  "sad": 0.0, "none": 0.02}}},
          "usage": {"prompt_tokens": 120, "completion_tokens": 0, "cost": 0.00004}}


class PureRules(unittest.TestCase):
    def test_glob_and_modality(self):
        self.assertTrue(adapters.is_decision_entry(JEV_ENTRY, ["*"]))
        # THE trap: `*` on OpenRouter must not make a chat model a decision model.
        self.assertFalse(adapters.is_decision_entry(GPT_ENTRY, ["*"]))
        self.assertTrue(adapters.is_decision_entry({"id": "laya"}, ["laya"]))   # no modalities: globs decide
        self.assertFalse(adapters.is_decision_entry({"id": "qwen3"}, ["laya"]))
        self.assertFalse(adapters.is_decision_entry(JEV_ENTRY, ["perplexity/*"]))

    def test_blank_setting_names_none(self):
        self.assertEqual(adapters.extract_decisions({"data": [JEV_ENTRY]}, {}), set())
        self.assertEqual(adapters.extract_decisions({"data": [JEV_ENTRY]},
                                                    {"decision_models": " , "}), set())
        self.assertEqual(adapters.extract_decisions([JEV_ENTRY, GPT_ENTRY],
                                                    {"decision_models": "*"}), {JEV})

    def test_urls(self):
        self.assertEqual(adapters.decision_url(OR, JEV, False),
                         "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(adapters.decision_url(LS, "laya", True),
                         "http://192.168.8.31:8080/upstream/laya/v1/systemone")
        self.assertEqual(adapters.decision_url(LS, "laya", False),
                         "http://192.168.8.31:8080/v1/systemone")


def _ctx(client, rows=None, llamaswap=False):
    async def record_call(**kw):
        if rows is not None:
            rows.append(kw)
    ctx = adapters.AdapterContext(
        auth_headers=lambda b: {"authorization": "Bearer sk-or-backend"},
        inflight_inc=lambda b: None, inflight_dec=lambda b: None,
        cost_usd=lambda *a: 0.777, source_of=lambda r: "anima", record_call=record_call,
        log_enabled=lambda: False, is_llamaswap=lambda bid: llamaswap,
        apply_reasoning=lambda b, m, r, p: ({**p, "reasoning": {"effort": "none"}}, "off"))
    ctx.http_client = lambda: client
    return ctx


class Discovery(unittest.TestCase):
    def _discover(self, backend, handler, adapter=None):
        seen = []

        def h(req):
            seen.append(req)
            return handler(req)
        client = httpx.AsyncClient(transport=httpx.MockTransport(h))
        a = adapter or adapters.OpenAIAdapter(backend, _ctx(client))
        a.ctx.http_client = lambda: client

        async def go():
            try:
                return await a.discover(client)
            finally:
                await client.aclose()
        return asyncio.run(go()), seen, a

    @staticmethod
    def _openrouter(fail_decisions=False):
        def handler(req):
            if req.url.path.endswith("/running"):
                return httpx.Response(404)
            if req.url.params.get("output_modalities") == "decisions":
                if fail_decisions:
                    return httpx.Response(502)
                return httpx.Response(200, json={"data": [JEV_ENTRY]})
            return httpx.Response(200, json={"data": [GPT_ENTRY]})     # decision models left out
        return handler

    def test_cloud_decision_catalog_is_listed_separately(self):
        caps, _, _ = self._discover(OR, self._openrouter())
        self.assertEqual(caps.models, {"openai/gpt-5", JEV})
        self.assertEqual(caps.decisions, {JEV})
        self.assertAlmostEqual(caps.pricing[JEV]["input"], 0.1)    # priced like any model

    def test_a_failed_decision_listing_keeps_the_last_one(self):
        _, _, a = self._discover(OR, self._openrouter())
        caps, _, _ = self._discover(OR, self._openrouter(fail_decisions=True), adapter=a)
        self.assertEqual(caps.decisions, {JEV})
        self.assertIn(JEV, caps.models)

    def test_no_setting_no_second_listing(self):
        b = {k: v for k, v in OR.items() if k != "decision_models"}
        caps, seen, _ = self._discover(b, self._openrouter())
        self.assertEqual(caps.decisions, set())
        self.assertFalse(any(r.url.params.get("output_modalities") for r in seen))

    def test_lan_server_lists_them_with_the_rest(self):
        def handler(req):
            if req.url.path.endswith("/running"):
                return httpx.Response(200, json={"running": []})
            return httpx.Response(200, json={"data": [{"id": "laya"}, {"id": "qwen3"}]})
        caps, seen, _ = self._discover(LS, handler)
        self.assertEqual(caps.decisions, {"laya"})
        self.assertFalse(any(r.url.params.get("output_modalities") for r in seen))

    def test_whitelisted_out_is_not_served(self):
        class _Stub:
            async def discover(self, client):
                return adapters.Capabilities(models={JEV, "openai/gpt-5"}, pricing={},
                                             decisions={JEV})
        keys = ("backends", "backend_adapters", "backend_models", "backend_pricing",
                "backend_loras", "backend_context", "backend_healthy", "backend_error",
                "backend_model_counts", "backend_decisions")
        saved = {k: getattr(main, k) for k in keys}
        try:
            for k in keys[2:]:
                setattr(main, k, {})
            b = dict(OR, models_deny="typesafe/*")
            bid = main.backend_id(b)
            main.backends, main.backend_adapters = [b], {bid: _Stub()}
            main.rebuild_route_index()
            asyncio.run(main.refresh_backend(b, None))
            self.assertEqual(main.backend_decisions[bid], set())
        finally:
            for k, v in saved.items():
                setattr(main, k, v)
            main.rebuild_route_index()


class _Routing(unittest.TestCase):
    KEYS = ("backends", "backend_models", "backend_healthy", "virtual_models",
            "backend_decisions", "backend_adapters", "api_key", "users", "_users_by_key",
            "model_prefix")

    def setUp(self):
        self._saved = {k: getattr(main, k) for k in self.KEYS}
        main.backends = [dict(OR), dict(LS)]
        main.virtual_models = {"jev": {"openrouter": JEV, "ls": "laya"}}
        main.backend_models = {"openai:openrouter": {JEV, "openai/gpt-5"},
                               "openai:ls": {"laya", "qwen3"}}
        main.backend_decisions = {"openai:openrouter": {JEV}, "openai:ls": {"laya"}}
        main.backend_healthy = {"openai:openrouter": True, "openai:ls": True}
        main.rebuild_route_index()

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(main, k, v)
        main.rebuild_route_index()

    @staticmethod
    def _names(routes):
        return sorted((b["name"], real) for b, real in routes)


class Routing(_Routing):
    def test_decision_model_only_on_its_endpoint(self):
        for name in (JEV, f"openrouter/{JEV}"):
            self.assertEqual(main.resolve_routes(name, "/v1/chat/completions"), ([], []), name)
            self.assertEqual(main.resolve_routes(name, "/v1/messages"), ([], []), name)
            self.assertEqual(self._names(main.resolve_routes(name, DP)[0]),
                             [("openrouter", JEV)], name)

    def test_chat_model_never_on_the_decision_endpoint(self):
        for name in ("openai/gpt-5", "openrouter/openai/gpt-5", "qwen3"):
            self.assertEqual(main.resolve_routes(name, DP), ([], []), name)
            ready, busy = main.resolve_routes(name, "/v1/chat/completions")
            self.assertTrue(ready or busy, name)

    def test_alias_fails_over_across_hosts(self):
        self.assertEqual(self._names(main.resolve_routes("jev", DP)[0]),
                         [("ls", "laya"), ("openrouter", JEV)])

    def _refusal(self, model, path):
        req = types.SimpleNamespace(state=types.SimpleNamespace())
        with self.assertRaises(HTTPException) as cm:
            asyncio.run(main._dispatch_or_park(model, path, {"model": model}, req))
        return cm.exception

    def test_refusals_name_the_right_endpoint(self):
        exc = self._refusal(f"openrouter/{JEV}", "/v1/chat/completions")
        self.assertEqual(exc.status_code, 404)
        self.assertIn(DP, exc.detail)
        exc = self._refusal("openai/gpt-5", DP)
        self.assertEqual(exc.status_code, 404)
        self.assertIn("not a decision model", exc.detail)

    def test_down_is_still_503(self):
        main.backend_healthy["openai:openrouter"] = False
        self.assertEqual(self._refusal(JEV, DP).status_code, 503)


class Dispatch(unittest.TestCase):
    def _run(self, backend, llamaswap=False, real=JEV):
        rows, sent = [], []

        def handler(req):
            sent.append(req)
            return httpx.Response(200, json=ANSWER)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        a = adapters.OpenAIAdapter(backend, _ctx(client, rows, llamaswap))
        body = dict(QUESTION, model=real, _reasoning="off")
        req = adapters.NormalizedRequest(
            path=DP, alias=real, real_model=real, body=body, reasoning="off",
            raw=types.SimpleNamespace(headers={"authorization": "Bearer hub-key",
                                               "content-type": "application/json"},
                                      client=None, state=types.SimpleNamespace()))

        async def go():
            resp = await a.dispatch(req)
            for _ in range(5):
                await asyncio.sleep(0)
            await client.aclose()
            return resp
        resp = asyncio.run(go())
        return resp, sent, rows

    def test_forwarded_as_sent_and_booked_with_the_reported_cost(self):
        b = dict(OR, sampling_defaults={"temperature": 0.7, "min_p": 0.05})
        resp, sent, rows = self._run(b)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(json.loads(resp.body), ANSWER)                 # answer passes through
        self.assertEqual(str(sent[0].url), "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(json.loads(sent[0].content), QUESTION)         # no sampling, no reasoning
        self.assertEqual(sent[0].headers["authorization"], "Bearer sk-or-backend")  # never the hub key
        self.assertEqual(sent[0].extensions["timeout"]["read"], adapters._DECISION_TIMEOUT.read)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["endpoint"], DP)
        self.assertEqual(rows[0]["cost_usd"], 0.00004)

    def test_llamaswap_goes_through_upstream(self):
        _, sent, _ = self._run(LS, llamaswap=True, real="laya")
        self.assertEqual(str(sent[0].url), "http://192.168.8.31:8080/upstream/laya/v1/systemone")


class Endpoint(_Routing):
    def setUp(self):
        super().setUp()
        self.sent = []

        def handler(req):
            self.sent.append(req)
            return httpx.Response(200, json=ANSWER)
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ctx = _ctx(self.client)
        ctx.inflight_inc, ctx.inflight_dec = main._inflight_inc, main._inflight_dec
        main.backend_adapters = {"openai:openrouter": adapters.OpenAIAdapter(main.backends[0], ctx)}
        main.api_key = "master"
        main.users = [{"name": "anima", "role": "user", "api_key": "anima-key", "models": [JEV]},
                      {"name": "chat", "role": "user", "api_key": "chat-key",
                       "models": ["openai/gpt-5"]}]
        main._users_by_key = {u["api_key"]: u for u in main.users}
        self.c = TestClient(main.app)

    def tearDown(self):
        asyncio.run(self.client.aclose())
        super().tearDown()

    def _post(self, body, key="anima-key"):
        return self.c.post(DP, json=body, headers={"authorization": f"Bearer {key}"})

    def test_anima_key_gets_the_answer(self):
        r = self._post(dict(QUESTION, stream=True))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["answers"]["entry"]["choice"], "positive")
        self.assertEqual(r.headers["x-gateway-backend"], "openrouter")
        self.assertNotIn("stream", json.loads(self.sent[0].content))

    def test_allow_list_and_bad_bodies(self):
        self.assertEqual(self._post(QUESTION, key="chat-key").status_code, 403)
        self.assertEqual(self._post(QUESTION, key="nope").status_code, 401)
        self.assertEqual(self._post({"state": {}, "questions": {}}).status_code, 400)
        self.assertEqual(self.c.post(DP, content=b"{", headers={
            "authorization": "Bearer anima-key", "content-type": "application/json"}).status_code, 400)
        self.assertEqual(self.sent, [])

    def test_model_listing_splits_the_kinds(self):
        def ids(typ):
            r = self.c.get(f"/v1/models?type={typ}", headers={"authorization": "Bearer master"})
            return {m["id"] for m in r.json()["data"]}
        main.model_prefix = True
        self.assertEqual(ids("decision"), {f"openrouter/{JEV}", "ls/laya", "jev"})
        chat = ids("chat")
        self.assertIn("openrouter/openai/gpt-5", chat)
        self.assertFalse(chat & {f"openrouter/{JEV}", "ls/laya", "jev"})


if __name__ == "__main__":
    unittest.main()
