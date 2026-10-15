"""Short job ids in the job API: /v1/jobs/{id}, /result/{n}, /input/{n}, /cancel accept a
unique prefix (>= 8 hex chars), resolved in ONE place (jobs.resolve_id via
main._require_job_owner).

Why this file exists (it fails SILENTLY): the /ui job page shows `job_id[:12]`, a client
copies exactly that, and the API answered a plain 404 "not found" for a job that exists
— reads like a pruned job, not like a lookup that never tried. The prefix lookup in turn
must not become an oracle: a prefix that matches only someone else's job must be the same
404 as no match (never a 403 "not your job", never counted into a 409), and an ambiguous
prefix must be a 409, never silently the first match.

Run: venv/bin/python -m unittest tests.test_job_id_prefix -v
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
    import jobs
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

A1 = "aaaaaaaa" + "1" * 24        # alice
A2 = "aaaaaaaa" + "2" * 24        # alice — makes "aaaaaaaa" ambiguous for her
C = "cccccccc" + "3" * 24         # bob only
D_ALICE = "dddddddd" + "4" * 24   # alice …
D_BOB = "dddddddd" + "5" * 24     # … and bob share this prefix
E = "eeeeeeee" + "6" * 24         # alice, queued (cancellable)
SHARED = "ffffffff" + "7" * 24    # owner 'default' — open to everyone


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._jobs_state = (jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active)
        jobs.init(os.path.join(self.tmp.name, "jobs.db"), os.path.join(self.tmp.name, "jobs"))
        for jid, owner in ((A1, "alice"), (A2, "alice"), (C, "bob"), (D_ALICE, "alice"),
                           (D_BOB, "bob"), (E, "alice"), (SHARED, "default")):
            jobs.create("txt2img", "al", "b", owner=owner, job_id=jid)
        jobs.set_inputs(A1, {"prompt": "x"}, [("input_image", PNG)])
        jobs.complete(A1, [types.SimpleNamespace(data=PNG, mime="image/png", kind="image",
                                                 name="out.png")])

    def tearDown(self):
        jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active = self._jobs_state
        self.tmp.cleanup()


class ResolveId(_Base):
    """jobs.resolve_id(job_id, owner) → (full id | None, matches). owner None = admin."""

    def test_a_full_id_is_returned_unchanged(self):
        self.assertEqual(jobs.resolve_id(C, "alice"), (C, 1))   # owner check is the caller's

    def test_a_unique_prefix_resolves(self):
        self.assertEqual(jobs.resolve_id("aaaaaaaa1", "alice"), (A1, 1))

    def test_an_ambiguous_prefix_counts_the_matches(self):
        self.assertEqual(jobs.resolve_id("aaaaaaaa", "alice"), (None, 2))

    def test_only_visible_jobs_count(self):
        self.assertEqual(jobs.resolve_id("dddddddd", "alice"), (D_ALICE, 1))
        self.assertEqual(jobs.resolve_id("cccccccc", "alice"), (None, 0))
        self.assertEqual(jobs.resolve_id("dddddddd", None), (None, 2))       # admin sees both
        self.assertEqual(jobs.resolve_id("ffffffff", "alice"), (SHARED, 1))  # 'default' is open

    def test_short_or_non_hex_is_not_searched(self):
        self.assertEqual(jobs.resolve_id("aaaaaaa", None), (None, 0))
        self.assertEqual(jobs.resolve_id("aaaaaaa%", None), (None, 0))
        self.assertEqual(jobs.resolve_id("aaaaaaaa_", None), (None, 0))


class JobApiPrefix(_Base):
    def setUp(self):
        super().setUp()
        names = ("api_key", "users", "_users_by_key", "cancel_generation")
        self._saved = {n: getattr(main, n) for n in names}
        alice = {"name": "alice", "role": "user", "api_key": "k-alice"}
        bob = {"name": "bob", "role": "user", "api_key": "k-bob"}
        main.api_key, main.users = "k-admin", [alice, bob]
        main._users_by_key = {"k-alice": alice, "k-bob": bob}
        self.cancelled = []

        async def cancel(job_id):
            self.cancelled.append(job_id)
            return True
        main.cancel_generation = cancel
        from fastapi.testclient import TestClient
        self.client = TestClient(main.app)

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(main, n, v)
        super().tearDown()

    def _get(self, path, key="k-alice"):
        return self.client.get(path, headers={"Authorization": f"Bearer {key}"})

    def test_full_id_unchanged(self):
        r = self._get(f"/v1/jobs/{A1}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["job_id"], A1)
        self.assertEqual(self._get(f"/v1/jobs/{C}").status_code, 403)    # foreign full id: as before

    def test_unique_prefix_answers_with_the_full_id(self):
        r = self._get("/v1/jobs/aaaaaaaa1111")
        self.assertEqual(r.status_code, 200)
        js = r.json()
        self.assertEqual(js["job_id"], A1)
        self.assertTrue(js["results"][0]["url"].endswith(f"/v1/jobs/{A1}/result/0"))

    def test_ambiguous_prefix_is_409(self):
        r = self._get("/v1/jobs/aaaaaaaa")
        self.assertEqual(r.status_code, 409)
        self.assertIn("ambiguous job id prefix, 2 matches", r.json()["detail"])

    def test_too_short_prefix_is_404(self):
        self.assertEqual(self._get("/v1/jobs/aaaaaaa").status_code, 404)
        self.assertEqual(self._get("/v1/jobs/aaaaaaaz").status_code, 404)

    def test_a_prefix_of_only_a_foreign_job_is_404_not_403(self):
        r = self._get("/v1/jobs/cccccccc")
        self.assertEqual(r.status_code, 404)
        self.assertNotIn(C, r.text)

    def test_a_foreign_match_never_makes_a_prefix_ambiguous(self):
        r = self._get("/v1/jobs/dddddddd")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["job_id"], D_ALICE)

    def test_admin_resolves_any_owner(self):
        r = self._get("/v1/jobs/cccccccc", key="k-admin")
        self.assertEqual((r.status_code, r.json()["job_id"]), (200, C))
        self.assertEqual(self._get("/v1/jobs/dddddddd", key="k-admin").status_code, 409)

    def test_result_input_and_cancel_take_the_prefix(self):
        r = self._get("/v1/jobs/aaaaaaaa1/result/0")
        self.assertEqual((r.status_code, r.content), (200, PNG))
        r = self._get("/v1/jobs/aaaaaaaa1/input/0")
        self.assertEqual((r.status_code, r.content), (200, PNG))
        r = self.client.post("/v1/jobs/eeeeeeee/cancel", headers={"Authorization": "Bearer k-alice"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["job_id"], E)
        self.assertEqual(self.cancelled, [E])                 # the worker gets the FULL id

    def test_sub_routes_hide_foreign_and_refuse_ambiguous(self):
        self.assertEqual(self._get("/v1/jobs/cccccccc/result/0").status_code, 404)
        self.assertEqual(self._get("/v1/jobs/aaaaaaaa/input/0").status_code, 409)
        r = self.client.post("/v1/jobs/cccccccc/cancel", headers={"Authorization": "Bearer k-alice"})
        self.assertEqual(r.status_code, 404)
        self.assertEqual(self.cancelled, [])


if __name__ == "__main__":
    unittest.main()
