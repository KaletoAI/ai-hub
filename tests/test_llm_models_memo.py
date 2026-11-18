"""`main._llm_models` for a catalog account (NanoGPT, Venice, wiro, …) is memoized.

Why this file exists (it fails SILENTLY, in both directions): the function rebuilt three
sets, a union and an intersection per call, and `/v1/models`, the route index and the
console call it once or twice PER MODEL — quadratic, ~40 ms of event loop per
`/v1/models` and account backend (review 2026-11-18). The memo is validated by the
IDENTITY of its three inputs, which are only ever replaced, never edited in place
(discovery assigns `backend_models[bid]` and `adapter.model_catalogs`; a config change
rebuilds the backend dicts). Pinned: a repeat call computes nothing; each replaced
input — and nothing less — recomputes; a stale answer is impossible after a discovery.

Run: venv/bin/python -m unittest tests.test_llm_models_memo -v
"""
import os
import sys
import unittest
from unittest import mock

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import adapters  # noqa: E402
import main  # noqa: E402


class _Ad:
    def __init__(self, text):
        self.model_catalogs = {"text": {m: {} for m in text}, "image": {"img-1": {}}}


class LlmModelsMemo(unittest.TestCase):

    def setUp(self):
        kind = sorted(adapters.DYNAMIC_CATALOG_TYPES)[0]
        self.b = {"name": "acct", "type": kind}
        self.bid = main.backend_id(self.b)
        self.ad = _Ad(["a", "b"])
        self._p = [mock.patch.dict(main.backend_models, {self.bid: {"a", "b", "img-1"}}),
                   mock.patch.dict(main.backend_adapters, {self.bid: self.ad}),
                   mock.patch.dict(main._llm_models_memo, {}, clear=True)]
        for p in self._p:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._p])
        self.n = 0
        real = adapters.add_model_extras

        def counting(models, backend):
            self.n += 1
            return real(models, backend)
        ex = mock.patch.object(adapters, "add_model_extras", counting)
        ex.start()
        self.addCleanup(ex.stop)

    def test_repeat_call_computes_nothing(self):
        first = main._llm_models(self.b)
        self.assertEqual(first, {"a", "b"})               # the image model is not an LLM
        for _ in range(50):
            self.assertIs(main._llm_models(self.b), first)
        self.assertEqual(self.n, 1)

    def test_a_discovery_replaces_the_answer(self):
        main._llm_models(self.b)
        main.backend_models[self.bid] = {"a", "b", "c", "img-1"}
        self.assertEqual(main._llm_models(self.b), {"a", "b"})   # catalog does not list c yet
        self.ad.model_catalogs = {"text": {"a": {}, "c": {}}}
        self.assertEqual(main._llm_models(self.b), {"a", "c"})
        self.assertEqual(self.n, 3)

    def test_a_config_change_replaces_the_answer(self):
        main._llm_models(self.b)
        b2 = {**self.b, "models_extra": ["b"]}
        main.backend_models[self.bid] = {"a", "b", "x"}
        self.ad.model_catalogs = {"text": {"a": {}}}
        self.assertEqual(main._llm_models(b2), {"a", "b"})

    def test_in_place_growth_is_noticed_too(self):
        """Nothing in production edits these in place — but a fixture does, and a memo
        that answers stale after an `.add()` would be a trap for the next change."""
        main._llm_models(self.b)
        self.ad.model_catalogs["text"]["c"] = {}
        main.backend_models[self.bid].add("c")
        self.assertEqual(main._llm_models(self.b), {"a", "b", "c"})

    def test_unknown_backend_is_empty_and_uncached(self):
        other = {"name": "nope", "type": self.b["type"]}
        self.assertEqual(main._llm_models(other), set())
        self.assertNotIn(main.backend_id(other), main._llm_models_memo)

    def test_plain_backend_is_the_discovered_set(self):
        plain = {"name": "llama", "type": "openai"}
        with mock.patch.dict(main.backend_models, {main.backend_id(plain): {"m"}}):
            self.assertIs(main._llm_models(plain), main.backend_models[main.backend_id(plain)])


if __name__ == "__main__":
    unittest.main()
