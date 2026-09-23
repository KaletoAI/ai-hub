"""Call cost with a prompt cache — the number LLM Calls, Statistic and the monthly
cost quota all read.

Why this fails SILENTLY: a wrong cost is a plausible cost. The cache share of the
input was recorded (`cache_read`/`cache_write`) but priced like fresh input, so an
agent session on OpenRouter whose context is ~95 % cache reads was booked at five
times what OpenRouter charged — measured 2026-09-23 on prod, 30 calls to
xiaomi/mimo-v2.6-flash: 0.1625 $ booked, 0.0277 $ billed (cache read 0.0028 $/M
against 0.14 $/M fresh). Nothing errors; a user's cost quota just runs out early.
So the three links are pinned: discovery reads the cache prices, `_cost_usd`
prices each share at its own rate (falling back to the input price where the
backend names none), and the adapter hands the cache split to it.

Where the backend SAYS what the call cost (OpenRouter's `usage.cost`, on every
answer and in the stream's usage chunk), that figure wins over any price list: it
already contains discounts, provider routing and price changes the listing cached
at discovery cannot know. Pinned on the plain path, the stream and the Messages
bridge, BYOK (where `cost` is only OpenRouter's fee), and the fallback when a
backend reports nothing or garbage.
"""
import asyncio
import json

import httpx
import os
import sys
import tempfile
import types
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    import adapters
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import logging

logging.getLogger("store").setLevel(logging.WARNING)

# OpenRouter's /v1/models entry, verbatim shape (2026-09-23)
MIMO = {"id": "xiaomi/mimo-v2.6-flash",
        "pricing": {"prompt": "0.00000014", "completion": "0.00000028",
                    "input_cache_read": "0.0000000028"}}
SONNET = {"id": "anthropic/claude-sonnet-5",
          "pricing": {"prompt": "0.000002", "completion": "0.00001", "web_search": "0.01",
                      "input_cache_read": "0.0000002", "input_cache_write": "0.0000025",
                      "input_cache_write_1h": "0.000004"}}


class Pricing(unittest.TestCase):
    def test_openrouter_cache_prices_are_read_per_million(self):
        p = adapters.normalize_pricing(SONNET)
        self.assertAlmostEqual(p["input"], 2.0)
        self.assertAlmostEqual(p["output"], 10.0)
        self.assertAlmostEqual(p["cache_read"], 0.2)
        self.assertAlmostEqual(p["cache_write"], 2.5)

    def test_absent_cache_price_is_absent_not_zero(self):
        # A 0 would price every cached token as free; absent means "input price".
        p = adapters.normalize_pricing({"id": "x", "pricing": {"prompt": "0.000001",
                                                               "completion": "0.000002"}})
        self.assertNotIn("cache_read", p)
        self.assertNotIn("cache_write", p)
        p = adapters.normalize_pricing({"id": "t", "pricing": {"input": 0.9, "output": 0.9}})
        self.assertEqual(set(p), {"input", "output"})


class CostUsd(unittest.TestCase):
    def setUp(self):
        self._saved = dict(main.backend_pricing)
        main.backend_pricing.clear()
        main.backend_pricing["openai:or"] = adapters.extract_pricing({"data": [MIMO, SONNET]})
        main.backend_pricing["openai:tg"] = {"m": {"input": 1.0, "output": 2.0}}

    def tearDown(self):
        main.backend_pricing.clear()
        main.backend_pricing.update(self._saved)

    def test_measured_mimo_call(self):
        # prod row 53283: 86196 in (85312 of them cache reads), 192 out
        cost = main._cost_usd("openai:or", "xiaomi/mimo-v2.6-flash", 86196, 192, 85312, 0)
        want = (884 * 0.14 + 85312 * 0.0028 + 192 * 0.28) / 1e6
        self.assertAlmostEqual(cost, want, places=12)
        self.assertLess(cost, 0.0005)             # booked 0.0121 before the fix

    def test_cache_write_priced_at_write_rate(self):
        cost = main._cost_usd("openai:or", "anthropic/claude-sonnet-5", 10000, 100, 6000, 3000)
        want = (1000 * 2.0 + 6000 * 0.2 + 3000 * 2.5 + 100 * 10.0) / 1e6
        self.assertAlmostEqual(cost, want, places=12)

    def test_no_cache_price_falls_back_to_input(self):
        self.assertAlmostEqual(main._cost_usd("openai:tg", "m", 1000, 1000, 800, 0),
                               main._cost_usd("openai:tg", "m", 1000, 1000))

    def test_cache_split_larger_than_input_never_goes_negative(self):
        # a backend reporting cache > prompt must not produce a credit
        cost = main._cost_usd("openai:or", "xiaomi/mimo-v2.6-flash", 100, 0, 500, 0)
        self.assertGreaterEqual(cost, 0.0)
        self.assertAlmostEqual(cost, 100 * 0.0028 / 1e6, places=14)

    def test_unknown_model_costs_nothing(self):
        self.assertEqual(main._cost_usd("openai:or", "nope", 1000, 1000, 500, 0), 0.0)
        self.assertEqual(main._cost_usd("openai:or", None, 1000, 1000), 0.0)


class AdapterHandsCacheSplit(unittest.TestCase):
    """`_record` is the one place a call's cost is computed — for every path
    (plain/streamed, chat/Messages), so the cache split must reach it there."""

    def test_record_passes_cache_to_cost_usd(self):
        seen, rows = [], []

        async def record_call(**kw):
            rows.append(kw)

        def cost_usd(*a):
            seen.append(a)
            return 0.5

        ctx = adapters.AdapterContext(
            auth_headers=lambda b: {}, inflight_inc=lambda b: None,
            inflight_dec=lambda b: None, cost_usd=cost_usd, source_of=lambda r: "t",
            record_call=record_call, log_enabled=lambda: False)
        a = adapters.OpenAIAdapter({"name": "or", "url": "http://x"}, ctx)
        req = adapters.NormalizedRequest(
            path="/v1/chat/completions", alias="m", real_model="m",
            body={"model": "m", "messages": []},
            raw=types.SimpleNamespace(headers={}, client=None, state=types.SimpleNamespace()))

        async def go():
            call = a._prepare(req)
            a._record(req, call, 200, 1000, 10, cache=(900, 50))
            a._finish(call)
            await asyncio.sleep(0)

        asyncio.run(go())
        self.assertEqual(seen, [("openai:or", "m", 1000, 10, 900, 50)])
        self.assertEqual(rows[0]["cost_usd"], 0.5)


class CacheWriteCounter(unittest.TestCase):
    """OpenRouter reports cache writes under prompt_tokens_details.cache_write_tokens;
    missed, they are priced like fresh input — cheaper than billed for Anthropic."""

    def test_plain_response(self):
        a = adapters.OpenAIAdapter({"name": "or", "url": "http://x"},
                                   adapters.AdapterContext(
                                       auth_headers=lambda b: {}, inflight_inc=lambda b: None,
                                       inflight_dec=lambda b: None, cost_usd=lambda *a: 0.0,
                                       source_of=lambda r: "t", record_call=lambda **k: None,
                                       log_enabled=lambda: False))
        usage = {"prompt_tokens": 100, "completion_tokens": 5,
                 "prompt_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 30}}
        self.assertEqual(a._cache_of({"usage": usage}), (60, 30))


# ── the backend's own figure ──────────────────────────────────────────────────

# OpenRouter usage, verbatim shape (measured 2026-09-23 through the gateway)
OR_USAGE = {"prompt_tokens": 13907, "completion_tokens": 5, "total_tokens": 13912,
            "cost": 4.29464e-05, "is_byok": False,
            "prompt_tokens_details": {"cached_tokens": 13888, "cache_write_tokens": 0},
            "cost_details": {"upstream_inference_cost": 4.29464e-05}}


class ReportedCostParse(unittest.TestCase):
    def test_openrouter_cost(self):
        self.assertEqual(adapters.reported_cost(OR_USAGE), 4.29464e-05)

    def test_zero_is_a_figure(self):
        # a free model's 0 is what it cost — not "unknown"
        self.assertEqual(adapters.reported_cost({"cost": 0}), 0.0)

    def test_byok_adds_the_upstream_bill(self):
        # BYOK: `cost` is OpenRouter's fee, the provider bills the key separately
        u = {"cost": 0.0001, "is_byok": True,
             "cost_details": {"upstream_inference_cost": 0.002}}
        self.assertAlmostEqual(adapters.reported_cost(u), 0.0021)

    def test_absent_or_garbage_is_none(self):
        for u in (None, {}, {"prompt_tokens": 5}, {"cost": None}, {"cost": "0.1"},
                  {"cost": True}, {"cost": -1.0}, {"cost": float("nan")}, "x"):
            self.assertIsNone(adapters.reported_cost(u), u)


def _adapter(handler, rows, computed=0.777):
    async def record_call(**kw):
        rows.append(kw)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ctx = adapters.AdapterContext(
        auth_headers=lambda b: {}, inflight_inc=lambda b: None, inflight_dec=lambda b: None,
        cost_usd=lambda *a: computed, source_of=lambda r: "t", record_call=record_call,
        log_enabled=lambda: False)
    ctx.http_client = lambda: client
    return adapters.OpenAIAdapter({"name": "or", "url": "https://openrouter.ai/api"}, ctx), client


def _req(path, body, stream):
    return adapters.NormalizedRequest(
        path=path, alias="m", real_model="m", body=body, stream=stream,
        raw=types.SimpleNamespace(headers={}, client=None, state=types.SimpleNamespace()))


def _chat_json(usage):
    return {"id": "c1", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}], "usage": usage}


def _sse(usage):
    lines = [{"id": "c1", "object": "chat.completion.chunk", "model": "m",
              "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}}]},
             {"id": "c1", "object": "chat.completion.chunk", "model": "m",
              "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
             {"id": "c1", "object": "chat.completion.chunk", "model": "m", "choices": [],
              "usage": usage}]
    return "".join(f"data: {json.dumps(x)}\n\n" for x in lines).encode() + b"data: [DONE]\n\n"


class ReportedCostBooked(unittest.TestCase):
    def _run(self, path, body, stream, handler, computed=0.777):
        rows = []
        a, client = _adapter(handler, rows, computed)

        async def go():
            resp = await a.dispatch(_req(path, body, stream))
            if hasattr(resp, "body_iterator"):
                async for _ in resp.body_iterator:
                    pass
            for _ in range(5):
                await asyncio.sleep(0)
            await client.aclose()
        asyncio.run(go())
        self.assertEqual(len(rows), 1)
        return rows[0]

    def _plain(self, usage):
        return lambda r: httpx.Response(200, json=_chat_json(usage))

    def _stream(self, usage):
        return lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"},
                                         content=_sse(usage))

    CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

    def test_plain_books_the_reported_cost(self):
        row = self._run("/v1/chat/completions", self.CHAT, False, self._plain(OR_USAGE))
        self.assertEqual(row["cost_usd"], 4.29464e-05)

    def test_stream_books_the_reported_cost(self):
        row = self._run("/v1/chat/completions", {**self.CHAT, "stream": True}, True,
                        self._stream(OR_USAGE))
        self.assertEqual(row["cost_usd"], 4.29464e-05)
        self.assertEqual(row["cache_read"], 13888)

    def test_messages_bridge_books_the_reported_cost(self):
        body = {"model": "m", "max_tokens": 50, "stream": True,
                "messages": [{"role": "user", "content": "hi"}]}
        row = self._run("/v1/messages", body, True, self._stream(OR_USAGE))
        self.assertEqual(row["cost_usd"], 4.29464e-05)

    def test_no_reported_cost_falls_back_to_the_price_list(self):
        usage = {k: v for k, v in OR_USAGE.items() if k not in ("cost", "cost_details")}
        row = self._run("/v1/chat/completions", self.CHAT, False, self._plain(usage))
        self.assertEqual(row["cost_usd"], 0.777)
        row = self._run("/v1/chat/completions", {**self.CHAT, "stream": True}, True,
                        self._stream(usage))
        self.assertEqual(row["cost_usd"], 0.777)

    def test_reported_cost_never_reaches_a_strict_client(self):
        # the client's usage chunk keeps the strict OpenAI shape
        rows, out = [], []
        a, client = _adapter(self._stream(OR_USAGE), rows)

        async def go():
            resp = await a.dispatch(_req("/v1/chat/completions",
                                         {**self.CHAT, "stream": True,
                                          "stream_options": {"include_usage": True}}, True))
            async for c in resp.body_iterator:
                out.append(c if isinstance(c, bytes) else c.encode())
            await client.aclose()
        asyncio.run(go())
        self.assertNotIn(b'"cost"', b"".join(out))


if __name__ == "__main__":
    unittest.main()
