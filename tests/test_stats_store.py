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
import types
import unittest
from contextlib import contextmanager

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    import stats
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


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


def _record(**kw):
    base = dict(duration_ms=5, backend="b1", source="kai", alias="tool", model="m",
                endpoint="/v1/chat/completions", status=200)
    base.update(kw)
    asyncio.run(stats.record_call(**base))
    return stats._q("SELECT MAX(id) FROM calls")[0][0]


class BodyStore(_StatsDB):
    def test_a_new_body_is_gzipped_and_reads_back(self):
        cid = _record(request_text='{"messages":[{"role":"user","content":"hi"}]}',
                      response_text='{"choices":[]}')
        path = os.path.join(stats._BLOB_DIR, f"{cid}.json.gz")
        with open(path, "rb") as f:
            self.assertEqual(f.read(2), b"\x1f\x8b")                  # gzip magic
        self.assertFalse(os.path.exists(os.path.join(stats._BLOB_DIR, f"{cid}.json")))
        body = stats.get_body(cid)
        self.assertEqual(body["request"]["messages"][0]["content"], "hi")
        self.assertEqual(body["response"], {"choices": []})

    def test_a_legacy_uncompressed_blob_still_reads(self):
        cid = _record()
        with open(os.path.join(stats._BLOB_DIR, f"{cid}.json"), "w") as f:
            f.write('{"request": {"a": 1}, "response": null}')
        self.assertEqual(stats.get_body(cid), {"request": {"a": 1}, "response": None})

    def test_a_huge_request_keeps_its_head_and_tail_only(self):
        big = '{"messages":"' + "A" * 50 + "x" * (2 * 1024 * 1024) + "Z" * 50 + '"}'
        cid = _record(request_text=big, response_text='{"ok":true}')
        req = stats.get_body(cid)["request"]
        self.assertIn("_truncated", req)
        self.assertTrue(req["head"].startswith('{"messages":"AAAA'))
        self.assertTrue(req["tail"].endswith('ZZZZ"}'))
        self.assertLessEqual(len(req["head"]) + len(req["tail"]), stats._BODY_MAX_CHARS)
        self.assertEqual(stats.get_body(cid)["response"], {"ok": True})

    def test_a_refused_call_keeps_its_reason_but_not_the_request(self):
        cid = _record(backend=stats.REFUSED_BACKEND, status=503, store_request=False,
                      request_text='{"model":"tool","messages":"' + "q" * 100000 + '"}',
                      response_text='{"error":{"message":"No healthy backend"}}')
        body = stats.get_body(cid)
        self.assertNotIn("qqqq", str(body["request"]))
        self.assertEqual(body["response"]["error"]["message"], "No healthy backend")
        prev = stats._q("SELECT req_preview FROM calls WHERE id=?", cid)[0][0]
        self.assertIn("tool", prev)                                    # the list still says what it was

    def test_main_records_a_refusal_without_its_request_body(self):
        request = types.SimpleNamespace(
            state=types.SimpleNamespace(gw_body={"model": "tool", "messages": "Q" * 5000},
                                        gw_alias="tool", gw_endpoint="/v1/chat/completions"),
            headers={}, client=None, url=types.SimpleNamespace(path="/v1/chat/completions"))

        async def go():
            main._record_rejected(request, main.HTTPException(503, "No healthy backend"))
            for _ in range(100):
                await asyncio.sleep(0.01)
                if stats._q("SELECT COUNT(*) FROM calls")[0][0]:
                    break
        asyncio.run(go())
        cid = stats._q("SELECT MAX(id) FROM calls")[0][0]
        body = stats.get_body(cid)
        self.assertNotIn("QQQQ", str(body["request"]))
        self.assertIn("No healthy backend", str(body["response"]))

    def test_old_bodies_are_pruned_while_the_row_stays(self):
        now = int(time.time())
        old = _record(request_text='{"a":1}')
        new = _record(request_text='{"b":2}')
        with sqlite3.connect(stats._DB_PATH) as c:
            c.execute("UPDATE calls SET ts=? WHERE id=?", (now - 20 * 86400, old))
        with open(os.path.join(stats._BLOB_DIR, f"{old}.json"), "w") as f:
            f.write("{}")                                              # a legacy twin goes too
        stats.prune_once(retention_days=0, body_retention_days=14)
        self.assertIsNone(stats.get_body(old))
        self.assertEqual(os.listdir(stats._BLOB_DIR), [f"{new}.json.gz"])
        self.assertEqual(stats._q("SELECT has_body FROM calls WHERE id=?", old)[0][0], 0)
        self.assertEqual(stats._q("SELECT COUNT(*) FROM calls")[0][0], 2)
        self.assertEqual(stats.get_body(new)["request"], {"b": 2})

    def test_row_retention_still_deletes_rows_and_their_bodies(self):
        now = int(time.time())
        old = _record(request_text='{"a":1}')
        with sqlite3.connect(stats._DB_PATH) as c:
            c.execute("UPDATE calls SET ts=? WHERE id=?", (now - 40 * 86400, old))
        stats.prune_once(retention_days=30, body_retention_days=0)
        self.assertEqual(stats._q("SELECT COUNT(*) FROM calls")[0][0], 0)
        self.assertEqual(os.listdir(stats._BLOB_DIR), [])


class WritePath(_StatsDB):
    def test_one_insert_carries_has_body_and_the_connection_is_not_fsync_per_commit(self):
        seen = []
        orig = stats._conn

        @contextmanager
        def traced():
            with orig() as c:
                c.set_trace_callback(seen.append)
                seen.append(f"synchronous={c.execute('PRAGMA synchronous').fetchone()[0]}")
                yield c
        stats._conn = traced
        try:
            cid = _record(request_text='{"a":1}', response_text='{"b":2}')
        finally:
            stats._conn = orig
        writes = [s for s in seen if s.split()[0].upper() in ("INSERT", "UPDATE")]
        self.assertEqual(len(writes), 1, seen)
        self.assertIn("synchronous=1", seen)                           # NORMAL — WAL keeps it safe
        self.assertEqual(stats._q("SELECT has_body FROM calls WHERE id=?", cid)[0][0], 1)


if __name__ == "__main__":
    unittest.main()
