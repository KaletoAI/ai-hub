"""The Media Playground's 2-second live tick must not recompute the whole form.

Why this file exists (it fails SILENTLY — the page still renders; every OTHER client's
stream stutters): while `?job=<id>` is queued/running the page re-fetches itself every
2 s, and each fetch rebuilt — synchronously on the event loop — the model list over all
aliases and catalog models with a price per row, plus the artifact picker over the last
200 jobs (review 2026-11-18: an estimated 50–200 ms of loop time per tick, for as long
as one tab watched a video job). Pinned:
  - a LIVE render reuses those two results for a few seconds;
  - a normal render (no running job) NEVER reads the memo — a just-saved alias must be
    there on the next page load — but refreshes it;
  - what a render does to the rows it got never reaches the next render.

Run: venv/bin/python -m unittest tests.test_playground_live_memo -v
"""
import os
import sys
import unittest
from unittest import mock

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import admin  # noqa: E402


class LiveMemo(unittest.TestCase):

    def setUp(self):
        admin._pg_memo.clear()
        self.calls = []

    def _fn(self):
        self.calls.append(1)
        return [{"id": "a", "n": len(self.calls)}]

    def test_live_reuses(self):
        a = admin._pg_cached("k", self._fn, live=True)
        b = admin._pg_cached("k", self._fn, live=True)
        self.assertEqual((len(self.calls), a, b), (1, [{"id": "a", "n": 1}], [{"id": "a", "n": 1}]))

    def test_not_live_always_computes_and_refreshes(self):
        admin._pg_cached("k", self._fn, live=True)
        fresh = admin._pg_cached("k", self._fn, live=False)
        self.assertEqual(fresh[0]["n"], 2)
        self.assertEqual(admin._pg_cached("k", self._fn, live=True)[0]["n"], 2)
        self.assertEqual(len(self.calls), 2)

    def test_expires(self):
        with mock.patch.object(admin.time, "monotonic", return_value=100.0):
            admin._pg_cached("k", self._fn, live=True)
        with mock.patch.object(admin.time, "monotonic",
                               return_value=100.0 + admin._PG_LIVE_MEMO_S + 0.1):
            admin._pg_cached("k", self._fn, live=True)
        self.assertEqual(len(self.calls), 2)

    def test_keys_are_separate(self):
        admin._pg_cached(("models", "image"), self._fn, live=True)
        admin._pg_cached(("models", "video"), self._fn, live=True)
        self.assertEqual(len(self.calls), 2)

    def test_rows_are_copies(self):
        rows = admin._pg_model_rows("image", True, lambda kind: self._fn())
        rows[0]["price"] = "poisoned"
        rows.append({"id": "extra"})
        again = admin._pg_model_rows("image", True, lambda kind: self._fn())
        self.assertEqual(again, [{"id": "a", "n": 1}])


if __name__ == "__main__":
    unittest.main()
