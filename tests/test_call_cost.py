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
"""
import asyncio
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


if __name__ == "__main__":
    unittest.main()
