"""The call log's storage and query paths (stats.py) — what the console reads every tick.

Why this fails SILENTLY: every query here returns the right rows whatever plan SQLite
picks, so a lost index or a full-table scan is invisible in a test on ten rows and only
shows on prod as a Dashboard that ticks slower each week (measured 2026-09 review: the
4-second tick spent 274 ms in `GROUP BY backend` walking `idx_calls_backend` over 300k
rows). The same holds for the body store: a blob written uncompressed and uncapped
still opens in the call view — it just costs gigabytes a day behind a Claude Code
session. So this pins the query PLANS (EXPLAIN QUERY PLAN, not timings), the blob
format, the prune, and the per-user month sum the cost quota reads per request.

Run: venv/bin/python -m unittest tests.test_stats_store -v
"""
import asyncio
import os
import sqlite3
import sys
import tempfile
import time
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import stats  # noqa: E402


def _row(ts, backend="b1", source="kai", endpoint="/v1/chat/completions", cost=0.0):
    return (ts, 10, backend, source, "tool", "m", endpoint, 200, 1, 1, cost, "p", None, 0, 0)


class _StatsDB(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self._saved = (stats._DB_PATH, stats._BLOB_DIR)
        stats.init(os.path.join(self._dir.name, "stats.db"), os.path.join(self._dir.name, "calls"))

    def tearDown(self):
        stats._DB_PATH, stats._BLOB_DIR = self._saved
        self._dir.cleanup()

    def _bulk(self, rows):
        with sqlite3.connect(stats._DB_PATH) as c:
            c.executemany("INSERT INTO calls (ts, duration_ms, backend, source, alias, model, "
                          "endpoint, status, input_tokens, output_tokens, cost_usd, req_preview, "
                          "reasoning, cache_read, cache_write) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          rows)

    def _plan(self, sql, *params):
        with sqlite3.connect(stats._DB_PATH) as c:
            return [r[3] for r in c.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()]


class DashboardQueryPlans(_StatsDB):
    """The two queries the Dashboard runs every 4 s must read the time window through
    an index, never the whole table."""

    def setUp(self):
        super().setUp()
        now = int(time.time())
        self._bulk([_row(now - 86400 * 30 + i * 60, backend=f"b{i % 7}") for i in range(20000)])

    def test_calls_per_backend_in_the_last_hour_reads_the_ts_window(self):
        plan = self._plan(stats._SQL_COUNT_BY_BACKEND_SINCE, int(time.time()) - 3600)
        self.assertTrue(any("idx_calls_ts_backend" in p and "ts>?" in p for p in plan), plan)
        self.assertFalse(any("idx_calls_backend" in p for p in plan), plan)

    def test_recent_calls_window_reads_the_ts_window(self):
        plan = self._plan(stats._SQL_RECENT_SINCE, int(time.time()) - 300, 100)
        self.assertTrue(any("ts>?" in p for p in plan), plan)
        self.assertFalse(any(p.startswith("SCAN calls") for p in plan), plan)

    def test_both_still_answer_the_same(self):
        now = int(time.time())
        self._bulk([_row(now - 5, backend="x"), _row(now - 4, backend="x"), _row(now - 3, backend="y")])
        self.assertEqual(stats.count_by_backend_since(now - 60), {"x": 2, "y": 1})
        got = stats.recent_since(now - 60)
        self.assertEqual([r[3] for r in got], ["y", "x", "x"])        # newest first


if __name__ == "__main__":
    unittest.main()
