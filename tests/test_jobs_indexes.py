"""The job store's frequent queries must be answered from an index.

Why this file exists (it fails SILENTLY — every query still answers, only slower with
every job): the table had one index (`created`). The short-id lookup behind every
`/v1/jobs/<8 hex>` poll (`id LIKE 'abc%'` cannot use the primary key under SQLite's
default case-insensitive LIKE), the dashboard's status counts every 4 s, the Media
Jobs owner list and the per-alias history (ETA median, node-time profile) were all
full table scans of rows that carry whole workflows in `meta_json` — ~30 ms each at
20 000 rows (measured 2026-11-18), some of them on the event loop.
Pinned with EXPLAIN QUERY PLAN: none of them may scan the table.

Run: venv/bin/python -m unittest tests.test_jobs_indexes -v
"""
import os
import sqlite3
import sys
import tempfile
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import jobs  # noqa: E402


class JobQueriesUseIndexes(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active)
        self.db = os.path.join(self._tmp.name, "jobs.db")
        jobs.init(self.db, os.path.join(self._tmp.name, "blobs"))
        c = sqlite3.connect(self.db)
        for i in range(300):
            jid = f"{i:08x}" + "ab" * 12
            c.execute("INSERT INTO jobs(id,created,updated,status,task,alias,backend,owner,ttl_s,meta_json)"
                      " VALUES(?,?,?,?,?,?,?,?,?,?)",
                      (jid, 1000 + i, 1010 + i, "done" if i % 3 else "error",
                       "chat" if i % 5 == 0 else "image", f"alias{i % 7}", f"b{i % 2}",
                       None if i % 4 == 0 else f"user{i % 3}", 86400,
                       '{"node_times": {"v": 1, "nodes": []}}'))
        c.commit()
        c.close()
        self.plans = []
        real = jobs._conn
        test = self

        import contextlib

        @contextlib.contextmanager
        def spy():
            with real() as conn:
                class C:
                    def execute(self_, sql, args=()):
                        if sql.lstrip().upper().startswith("SELECT"):
                            plan = conn.execute("EXPLAIN QUERY PLAN " + sql, args).fetchall()
                            test.plans.append(" | ".join(r[3] for r in plan))
                        return conn.execute(sql, args)
                yield C()
        self._real, jobs._conn = real, spy

    def tearDown(self):
        jobs._conn = self._real
        jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active = self._saved
        self._tmp.cleanup()

    def _no_scan(self):
        self.assertTrue(self.plans)
        for p in self.plans:        # a SEARCH, or a scan of a narrow covering index —
            self.assertRegex(p, r"SEARCH jobs|USING COVERING INDEX", p)   # never the rows

    def test_prefix_lookup(self):
        ids = jobs.prefix_ids("0000002a", None)
        self.assertEqual(ids, ["0000002a" + "ab" * 12])
        self.assertEqual(len(jobs.prefix_ids("000000", None)), 0)        # < 8 chars: not a prefix
        self.assertEqual(jobs.prefix_ids("0000002A", None), ids)         # case as before
        self.assertEqual(jobs.prefix_ids("0000002a", "user0"), ids)      # 42 % 4 != 0 → user0's
        self.assertEqual(jobs.prefix_ids("0000002a", "user1"), [])
        self._no_scan()

    def test_prefix_is_a_range_not_one_row(self):
        c = sqlite3.connect(self.db)
        for tail in ("0", "f"):
            c.execute("INSERT INTO jobs(id,created,updated,status,ttl_s) VALUES(?,?,?,?,?)",
                      ("deadbeef" + tail * 24, 5, 5, "done", 1))
        c.execute("INSERT INTO jobs(id,created,updated,status,ttl_s) VALUES(?,?,?,?,?)",
                  ("deadbef0" + "0" * 24, 5, 5, "done", 1))
        c.commit()
        c.close()
        self.assertEqual(sorted(jobs.prefix_ids("deadbeef", None)),
                         ["deadbeef" + "0" * 24, "deadbeef" + "f" * 24])

    def test_counts_and_owners(self):
        self.assertEqual(sum(jobs.counts().values()), 300)
        self.assertEqual(sum(jobs.counts(media_only=True).values()), 240)
        self.assertEqual(jobs.owners(), ["user0", "user1", "user2"])
        self._no_scan()

    def test_alias_history(self):
        self.assertIsNotNone(jobs.median_duration("alias3"))
        self.assertIsNotNone(jobs.median_duration("alias3", "b1"))
        self.assertTrue(jobs.node_time_records("alias3"))
        self._no_scan()


if __name__ == "__main__":
    unittest.main()
