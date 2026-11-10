"""Handing a finished media job over to a user (/ui job page → POST /ui/job/{id}/handover,
jobs.hand_over).

Why this fails SILENTLY: every write here "succeeds". Stamping `updated` on the handover
(the natural thing for an UPDATE) turns the job's run time — RUN_S = updated − started —
into hours, and the median/ETA and speed stats of the alias drift with no error. Moving
a LIVE job hands it away from the client that is still polling it (its next GET is a
403 "not your job"). And a job handed over shortly before its TTL is pruned before the
recipient fetches it — the handover "worked", the job is just gone.

Run: venv/bin/python -m unittest tests.test_job_handover -v
"""
import asyncio
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

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
    import jobs
    import store
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
USERS = [{"name": "alice", "role": "user"}, {"name": "bob", "role": "user"}]


def _blob():
    return types.SimpleNamespace(data=PNG, mime="image/png", kind="image", name="out.png")


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = (jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active)
        jobs.init(os.path.join(self.tmp.name, "jobs.db"), os.path.join(self.tmp.name, "blobs"),
                  default_ttl_s=3600)

    def tearDown(self):
        jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active = self._saved
        self.tmp.cleanup()

    def _done(self, task="txt2img", owner="alice", ttl_s=None):
        jid = jobs.create(task, "al", "gpu", owner=owner, ttl_s=ttl_s)
        jobs.set_status(jid, "running")
        jobs.complete(jid, [_blob()])
        return jid

    def _row(self, jid):
        with jobs._conn() as c:
            return dict(c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone())


class HandOver(_Base):

    def test_a_done_media_job_moves_to_the_new_owner(self):
        jid = self._done()
        self.assertTrue(jobs.hand_over(jid, "bob"))
        self.assertEqual(jobs.get(jid)["owner"], "bob")

    def test_run_time_and_order_columns_are_untouched(self):
        jid = self._done()
        with jobs._conn() as c:      # a run that ended long ago
            c.execute("UPDATE jobs SET created=created-500, started=started-500, "
                      "updated=updated-490 WHERE id=?", (jid,))
        before = self._row(jid)
        jobs.hand_over(jid, "bob")
        after = self._row(jid)
        for col in ("created", "started", "updated", "status", "results_json"):
            self.assertEqual(before[col], after[col], col)

    def test_the_expiry_is_extended_to_a_full_ttl_from_now(self):
        jid = self._done(ttl_s=60)
        with jobs._conn() as c:
            c.execute("UPDATE jobs SET created=created-50 WHERE id=?", (jid,))
        jobs.hand_over(jid, "bob")
        r = self._row(jid)
        self.assertGreaterEqual(r["created"] + r["ttl_s"], int(time.time()) + 3600 - 2)
        self.assertEqual(jobs.prune_once(), 0)

    def test_a_longer_ttl_is_never_shortened(self):
        jid = self._done(ttl_s=7 * 86400)
        jobs.hand_over(jid, "bob")
        self.assertEqual(self._row(jid)["ttl_s"], 7 * 86400)

    def test_live_failed_and_non_media_jobs_do_not_move(self):
        queued = jobs.create("txt2img", "al", "gpu", owner="alice")
        running = jobs.create("txt2img", "al", "gpu", owner="alice")
        jobs.set_status(running, "running")
        failed = jobs.create("txt2img", "al", "gpu", owner="alice")
        jobs.fail(failed, "boom")
        response = self._done(task="response")
        for jid in (queued, running, failed, response):
            self.assertFalse(jobs.hand_over(jid, "bob"), jid)
            self.assertEqual(jobs.get(jid)["owner"], "alice", jid)

    def test_a_blank_owner_or_unknown_job_is_refused(self):
        jid = self._done()
        self.assertFalse(jobs.hand_over(jid, "  "))
        self.assertFalse(jobs.hand_over("0" * 32, "bob"))
        self.assertEqual(jobs.get(jid)["owner"], "alice")


class _Req:
    def __init__(self, body=b"", query=None):
        self._body = body
        self.query_params = query or {}

    async def body(self):
        return self._body


class ConsoleAction(_Base):

    def setUp(self):
        super().setUp()
        self._p = [mock.patch.object(store, "is_active", return_value=True),
                   mock.patch.object(store, "list_users", return_value=USERS)]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in self._p:
            p.stop()
        super().tearDown()

    def _post(self, jid, user):
        r = asyncio.run(admin.job_handover(jid, _Req(f"user={user}".encode())))
        self.assertEqual(r.status_code, 303)
        loc = urlsplit(r.headers["location"])
        self.assertEqual(loc.path, f"/ui/job/{jid}")
        return parse_qs(loc.query)["msg"][0]

    def test_handover_to_a_known_user(self):
        jid = self._done()
        self.assertIn("handed over to bob", self._post(jid, "bob"))
        self.assertEqual(jobs.get(jid)["owner"], "bob")

    def test_an_unknown_user_changes_nothing(self):
        jid = self._done()
        self.assertIn("unknown user", self._post(jid, "mallory"))
        self.assertEqual(jobs.get(jid)["owner"], "alice")

    def test_a_running_job_is_refused_with_a_message(self):
        jid = jobs.create("txt2img", "al", "gpu", owner="alice")
        jobs.set_status(jid, "running")
        self.assertIn("only a finished", self._post(jid, "bob"))
        self.assertEqual(jobs.get(jid)["owner"], "alice")

    def test_the_picker_shows_only_on_a_done_media_job(self):
        done = self._done()
        html = asyncio.run(admin._job_handover_html(done, jobs.get(done)))
        self.assertIn(f"/ui/job/{done}/handover", html)
        self.assertIn('value="alice" selected', html)          # current owner preselected
        self.assertIn('form="gw-handover"', html)              # fields travel with the button
        running = jobs.create("txt2img", "al", "gpu", owner="alice")
        jobs.set_status(running, "running")
        self.assertEqual(asyncio.run(admin._job_handover_html(running, jobs.get(running))), "")
        resp = self._done(task="response")
        self.assertEqual(asyncio.run(admin._job_handover_html(resp, jobs.get(resp))), "")

    def test_the_recipient_can_fetch_it_and_the_old_owner_cannot(self):
        jid = self._done()
        self._post(jid, "bob")
        req = types.SimpleNamespace(state=types.SimpleNamespace(), client=None)
        for name, ok in (("bob", True), ("alice", False)):
            with mock.patch.object(main, "authenticate",
                                   return_value={"name": name, "role": "user"}):
                if ok:
                    self.assertEqual(asyncio.run(main._require_job_owner("Bearer k", req, jid)), jid)
                else:
                    with self.assertRaises(main.HTTPException):
                        asyncio.run(main._require_job_owner("Bearer k", req, jid))


if __name__ == "__main__":
    unittest.main()
