"""Prices before a request is sent: `pricing` on /v1/models, the console's price hints,
and the `model_pricing` rule that feeds both — and the LLM Calls cost.

Why this fails SILENTLY: a price is a plausible number either way. A paid model with
no listed price published as 0 reads as "free" to a client that budgets by it (Oh My
Pi reads OpenRouter's `pricing`); an alias published at its CHEAPEST backend's price
under-quotes every call that lands on the dear one; a per-token string written as
"1.5e-07" parses as 1.5 in a naive reader. Nothing errors — the bill is the first
anyone hears of it. So the precedence (rule > discovery > unpaid = 0 > unknown), the
alias maximum, the omit-when-unknown rule and the schema round-trip are pinned here.
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
    import admin
    import adapters
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("store").setLevel(logging.WARNING)


class PricingRules(unittest.TestCase):
    def test_parse_lines_keeps_order_and_absent_cache(self):
        rules = adapters.parse_model_pricing("gpt-*=1.25/10\n# c\n\nclaude-*=3/15/0.3/3.75")
        self.assertEqual(rules, [("gpt-*", {"input": 1.25, "output": 10.0}),
                                 ("claude-*", {"input": 3.0, "output": 15.0,
                                               "cache_read": 0.3, "cache_write": 3.75})])

    def test_bad_lines_are_dropped_never_zero(self):
        # A typo must not become a 0 ("free") — it simply is no rule.
        self.assertEqual(adapters.parse_model_pricing(
            "a=1\nb=x/2\nc=-1/2\nd=1/2/3/4/5\n=1/2\ne=nan/1\nok=1/2"),
            [("ok", {"input": 1.0, "output": 2.0})])

    def test_rule_beats_discovery(self):
        b = {"name": "or", "paid": True, "model_pricing": "m=9/9"}
        self.assertEqual(adapters.model_price_for(b, "m", {"m": {"input": 1.0, "output": 2.0}}),
                         {"input": 9.0, "output": 9.0})

    def test_discovery_when_no_rule_matches(self):
        b = {"name": "or", "paid": True, "model_pricing": "other=9/9"}
        self.assertEqual(adapters.model_price_for(b, "m", {"m": {"input": 1.0, "output": 2.0}}),
                         {"input": 1.0, "output": 2.0})

    def test_unpaid_without_price_is_free_paid_is_unknown(self):
        self.assertEqual(adapters.model_price_for({"name": "l"}, "m", {}),
                         {"input": 0.0, "output": 0.0})
        self.assertIsNone(adapters.model_price_for({"name": "oa", "paid": True}, "m", {}))


class OpenRouterSchema(unittest.TestCase):
    def test_alias_publishes_the_dearest_rate(self):
        s = adapters.summarize_prices([{"input": 0.15, "output": 0.6},
                                       {"input": 3.0, "output": 15.0, "cache_read": 0.3}])
        self.assertEqual(s["input"], [0.15, 3.0])
        # the backend without a cache rate bills cached input at its input rate
        self.assertEqual(s["cache_read"], [0.15, 0.3])
        pr = adapters.openrouter_pricing(s)
        self.assertEqual(pr, {"prompt": "0.000003", "completion": "0.000015",
                              "input_cache_read": "0.0000003"})

    def test_any_unknown_candidate_omits_the_price(self):
        s = adapters.summarize_prices([{"input": 1.0, "output": 2.0}, None])
        self.assertEqual(s["unknown"], 1)
        self.assertIsNone(adapters.openrouter_pricing(s))

    def test_no_exponent_and_round_trip(self):
        p = {"input": 0.15, "output": 0.6, "cache_read": 0.015, "cache_write": 0.1875}
        pr = adapters.openrouter_pricing(adapters.summarize_prices([p]))
        self.assertTrue(all("e" not in v for v in pr.values()), pr)
        back = adapters.normalize_pricing({"pricing": pr})
        for k, v in p.items():
            self.assertAlmostEqual(back[k], v, places=9)

    def test_free_is_published_as_zero(self):
        pr = adapters.openrouter_pricing(adapters.summarize_prices([{"input": 0.0, "output": 0.0}]))
        self.assertEqual(pr, {"prompt": "0", "completion": "0"})


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
                        "backend_pricing", "model_prefix", "users", "api_key")}
        local = {"name": "local", "type": "openai", "url": "http://l", "enabled": True, "local": True}
        cloud = {"name": "cloud", "type": "openai", "url": "http://c", "enabled": True,
                 "paid": True, "model_pricing": "gpt-*=1.25/10"}
        main.backends = [local, cloud]
        main.virtual_models = {"mix": {"local": "qwen", "cloud": "listed"},
                               "blind": {"local": "qwen", "cloud": "unlisted"}}
        main.backend_models = {main.backend_id(local): {"qwen"},
                               main.backend_id(cloud): {"listed", "unlisted", "gpt-5"}}
        main.backend_healthy = {main.backend_id(local): True, main.backend_id(cloud): True}
        main.backend_pricing = {main.backend_id(cloud): {"listed": {"input": 0.5, "output": 2.0}}}
        main.model_prefix = True
        main.users, main.api_key = [], ""
        main.rebuild_route_index()

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(main, k, v)
        main.rebuild_route_index()

    def _list(self):
        data = asyncio.run(main.list_models(_Req(), None, None, None))["data"]
        return {m["id"]: m for m in data}

    def test_each_source_reaches_the_catalog(self):
        cat = self._list()
        self.assertEqual(cat["local/qwen"]["pricing"], {"prompt": "0", "completion": "0"})
        self.assertEqual(cat["cloud/listed"]["pricing"]["prompt"], "0.0000005")
        self.assertEqual(cat["cloud/gpt-5"]["pricing"]["completion"], "0.00001")   # the rule
        self.assertNotIn("pricing", cat["cloud/unlisted"])                          # unknown

    def test_alias_is_its_dearest_backend_or_absent(self):
        cat = self._list()
        self.assertEqual(cat["mix"]["pricing"], {"prompt": "0.0000005", "completion": "0.000002"})
        self.assertNotIn("pricing", cat["blind"])

    def test_get_model_matches_the_listing(self):
        self.assertEqual(asyncio.run(main.get_model("mix", None, None, None))["pricing"],
                         self._list()["mix"]["pricing"])
        self.assertEqual(asyncio.run(main.get_model("cloud/gpt-5", None, None, None))["pricing"],
                         {"prompt": "0.00000125", "completion": "0.00001"})

    def test_call_cost_uses_the_rule(self):
        # What the console quotes is what LLM Calls books.
        cost = main._cost_usd(main.backend_id(main.backends[1]), "gpt-5", 1_000_000, 100_000)
        self.assertAlmostEqual(cost, 1.25 + 1.0)

    def test_console_prices_per_backend(self):
        info = {b["name"]: b for b in main.llm_backends_info()}
        self.assertEqual(info["cloud"]["prices"]["gpt-5"], {"input": 1.25, "output": 10.0})
        self.assertIsNone(info["cloud"]["prices"]["unlisted"])


class Console(unittest.TestCase):
    def test_labels(self):
        one = admin._price_summary({"input": 0.15, "output": 0.6})
        self.assertEqual(admin._price_short(one), "$0.15/$0.60")
        self.assertIn("input $0.15 · output $0.60", admin._price_long(one))
        self.assertEqual(admin._price_short(admin._price_summary({"input": 0.0, "output": 0.0})), "free")
        self.assertEqual(admin._price_short(admin._price_summary(None)), "price ?")
        rng = adapters.summarize_prices([{"input": 0.0, "output": 0.0}, {"input": 3.0, "output": 15.0}, None])
        # an unpriced candidate leaves the upper bound OPEN — never "free", never "$3.00"
        self.assertEqual(admin._price_short(rng), "$0–?/$0–?")
        self.assertIn("1 backend without a price", admin._price_long(rng))
        free_or_unknown = adapters.summarize_prices([{"input": 0.0, "output": 0.0}, None])
        self.assertEqual(admin._price_short(free_or_unknown), "$0–?/$0–?")

    def test_backend_form_renders_the_rules(self):
        html = admin._backend_form({"name": "oa", "type": "openai", "url": "http://x",
                                    "model_pricing": "gpt-*=1.25/10"}, [])
        self.assertIn('name="model_pricing"', html)
        self.assertIn("gpt-*=1.25/10", html)


if __name__ == "__main__":
    unittest.main()
