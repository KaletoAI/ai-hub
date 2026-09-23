"""The job store's state machine: queued → running → done | failed, and nothing after.

Why this fails SILENTLY: every write here succeeds. A worker that finishes after the
user cancelled used to overwrite `failed: cancelled by user` with `done` (or a later
`running` resurrected the row), so the console showed a job the user had stopped as
delivered; and `prune_once` deleted a job still RUNNING under a short client `ttl_s`,
after which `complete()` wrote its artifacts into a directory no row points to — a
disk leak and a job that simply vanished from the list. None of it raises.

Run: venv/bin/python -m unittest tests.test_jobs_lifecycle -v
"""
import os
import sys
import tempfile
import time
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jobs  # noqa: E402


def _blob(data=b"x"):
    return types.SimpleNamespace(data=data, mime="image/png", kind="image", name=None)


class JobLifecycle(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active)
        jobs.init(os.path.join(self._tmp.name, "jobs.db"), os.path.join(self._tmp.name, "blobs"))

    def tearDown(self):
        jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active = self._saved
        self._tmp.cleanup()

    def _job(self, ttl_s=None):
        return jobs.create("text2img", "a", "gpu", ttl_s=ttl_s)

    def test_a_cancelled_job_is_not_completed_by_its_worker(self):
        jid = self._job()
        jobs.set_status(jid, "running")
        jobs.fail(jid, "cancelled by user")
        jobs.complete(jid, [_blob()], {"backend": "gpu"})
        j = jobs.get(jid)
        self.assertEqual((j["status"], j["error"]), ("failed", "cancelled by user"))
        self.assertFalse(os.path.isdir(os.path.join(jobs._BLOB_DIR, jid)))   # no orphan artifacts

    def test_a_cancelled_job_is_not_set_running_again(self):
        jid = self._job()
        jobs.fail(jid, "cancelled by user")
        self.assertFalse(jobs.set_status(jid, "running"))
        self.assertEqual(jobs.get(jid)["status"], "failed")

    def test_a_live_job_can_be_set_running(self):
        jid = self._job()
        self.assertTrue(jobs.set_status(jid, "running"))
        self.assertEqual(jobs.get(jid)["status"], "running")

    def test_the_first_terminal_write_wins(self):
        jid = self._job()
        jobs.complete(jid, [_blob()], {})
        jobs.fail(jid, "late failure")
        self.assertEqual(jobs.get(jid)["status"], "done")
        jid2 = self._job()
        jobs.fail(jid2, "cancelled")
        jobs.complete_json(jid2, {"x": 1})
        self.assertEqual(jobs.get(jid2)["status"], "failed")

    def test_a_cancelled_job_is_not_re_pointed_to_another_backend(self):
        # the row names the backend it was cancelled on; a worker's late hand-off
        # (chain stage 2, a failover claim) must not rewrite it afterwards
        jid = self._job()
        jobs.set_backend(jid, "gpu-b")
        self.assertEqual(jobs.get(jid)["backend"], "gpu-b")
        jobs.fail(jid, "cancelled by user")
        jobs.set_backend(jid, "gpu-c")
        self.assertEqual(jobs.get(jid)["backend"], "gpu-b")

    def test_merge_meta_reaches_a_terminal_row(self):
        jid = self._job()
        jobs.fail(jid, "cancelled by user")
        jobs.merge_meta(jid, {"cloud_task_id": "t1"})
        self.assertEqual(jobs.get(jid)["meta"]["cloud_task_id"], "t1")

    def test_prune_keeps_a_running_job_past_its_ttl(self):
        jid = self._job(ttl_s=1)
        jobs.set_status(jid, "running")
        with jobs._conn() as c:                         # created long ago
            c.execute("UPDATE jobs SET created=? WHERE id=?", (int(time.time()) - 100, jid))
        self.assertEqual(jobs.prune_once(), 0)
        self.assertIsNotNone(jobs.get(jid))

    def test_prune_removes_a_finished_job_past_its_ttl(self):
        jid = self._job(ttl_s=1)
        jobs.complete(jid, [_blob()], {})
        with jobs._conn() as c:
            c.execute("UPDATE jobs SET created=? WHERE id=?", (int(time.time()) - 100, jid))
        self.assertEqual(jobs.prune_once(), 1)
        self.assertIsNone(jobs.get(jid))
        self.assertFalse(os.path.isdir(os.path.join(jobs._BLOB_DIR, jid)))

    def test_complete_of_a_pruned_row_writes_nothing(self):
        jid = self._job()
        jobs._delete(jid)
        self.assertEqual(jobs.complete(jid, [_blob()], {}), [])
        self.assertFalse(os.path.isdir(os.path.join(jobs._BLOB_DIR, jid)))


if __name__ == "__main__":
    unittest.main()
