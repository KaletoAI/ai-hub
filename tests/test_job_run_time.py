"""A job's duration is its RUN time — from the claim, not from creation.

Why this fails SILENTLY: the `dur` column (Media Jobs, dashboard) was updated − created,
so a job that sat parked behind a busy backend for ten minutes and rendered for one
showed "11 min" — a plausible number, just the wrong one. The same span fed the ETA
median, the Media generation panel's avg and the scheduler's boot seed, so a backend
with a long queue looked SLOW. `jobs.set_status(…, "running")` now stamps `started`
once (a failover / chain re-claim keeps it) and every duration is `jobs.RUN_S`.

Also pinned here: the per-backend Pinned-values tabs carry a "default" button that
copies the primary's slot value — its data-src must name the primary's field exactly,
or the click does nothing (the handler finds no element and returns).

Run: venv/bin/python -m unittest tests.test_job_run_time -v
"""
import asyncio
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import admin  # noqa: E402
import jobs  # noqa: E402


class StartedStamp(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active)
        jobs.init(os.path.join(self._tmp.name, "jobs.db"), os.path.join(self._tmp.name, "blobs"))

    def tearDown(self):
        jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active = self._saved
        self._tmp.cleanup()

    def _set(self, jid, **cols):
        with jobs._conn() as c:
            for k, v in cols.items():
                c.execute(f"UPDATE jobs SET {k}=? WHERE id=?", (v, jid))

    def test_first_claim_stamps_started_and_a_reclaim_keeps_it(self):
        jid = jobs.create("text2img", "a", "gpu")
        self.assertIsNone(jobs.get(jid)["started"])
        jobs.set_status(jid, "running")
        first = jobs.get(jid)["started"]
        self.assertIsNotNone(first)
        self._set(jid, started=first - 50)
        jobs.set_status(jid, "running")                  # failover / chain re-claim
        self.assertEqual(jobs.get(jid)["started"], first - 50)

    def test_median_excludes_the_queue_wait(self):
        jid = jobs.create("text2img", "a", "gpu")
        jobs.set_status(jid, "running")
        jobs.fail(jid, "x")                              # terminal, then fix the clock
        self._set(jid, status="done", created=1000, started=1600, updated=1660)
        self.assertEqual(jobs.median_duration("a"), 60.0)
        self.assertEqual(jobs.gen_speed_rows(), [("a", "gpu", 60000.0)])

    def test_rows_before_the_column_fall_back_to_created(self):
        jid = jobs.create("text2img", "a", "gpu")
        jobs.fail(jid, "x")
        self._set(jid, status="done", created=1000, updated=1030, started=None)
        self.assertEqual(jobs.median_duration("a"), 30.0)

    def test_recent_carries_started(self):
        jid = jobs.create("text2img", "a", "gpu")
        jobs.set_status(jid, "running")
        self.assertIsNotNone(jobs.recent(5)[0]["started"])


class DurCell(unittest.TestCase):

    def test_done_job_shows_run_time_not_wait(self):
        j = {"id": "j1", "status": "done", "created": 1000, "started": 1600, "updated": 1660}
        self.assertIn('data-sv="60000"', admin._job_dur_cell(j, 2000))

    def test_running_job_ticks_from_the_claim(self):
        j = {"id": "j1", "status": "running", "alias": "", "backend": "",
             "created": 1000, "started": 1600, "updated": 1600}
        self.assertIn("data-since='1600'", admin._job_dur_cell(j, 1700))

    def test_old_row_without_started_uses_created(self):
        j = {"id": "j1", "status": "done", "created": 1000, "updated": 1030}
        self.assertIn('data-sv="30000"', admin._job_dur_cell(j, 2000))


class PinDefaultButton(unittest.TestCase):

    def test_extra_tab_default_points_at_the_primary_field(self):
        fixed = [{"node": "4", "field": "ckpt_name", "value": "a.safetensors"}]
        wf = {"4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}}}
        cands = [{"backend": "gpu-a", "fixed": fixed},
                 {"backend": "gpu-b", "fixed": [dict(fixed[0], value="b.safetensors")]}]

        async def _oi(*_a, **_k):
            return {}
        saved = admin._object_info
        admin._object_info = _oi
        try:
            html = asyncio.run(admin._pinned_block("al", cands, fixed, wf, {}))
        finally:
            admin._object_info = saved
        self.assertIn('data-src="fixed__4__ckpt_name"', html)
        self.assertIn('data-dst="ovr__gpu-b__4__ckpt_name"', html)
        self.assertIn('name="fixed__4__ckpt_name"', html)      # the element data-src finds
        self.assertEqual(html.count("pinDef(this)"), 1)        # extra tabs only, not the primary
        self.assertIn("function pinDef", admin._PIN_CSS_JS)


if __name__ == "__main__":
    unittest.main()
