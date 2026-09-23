"""Limits on what a generation client can make the gateway keep.

Why this fails SILENTLY: a job's `ttl_s` came straight from the client, and anything
positive was stored — `ttl_s: 10**12` kept a job's inputs and results on disk forever,
and the pruner never touched it (review 2026-09-23, S12). And while a parked CHAT call
is capped by `max_parked`, async generation jobs were not capped at all: each one is a
task, a job row and its stored inputs, and a loop of `mode: async` requests queues
without end — the jobs simply sit `queued`, which looks like a busy fleet.

So the client's TTL is clamped to `jobs.max_ttl_s` (config, default 7 days), and a new
async job is refused with 503 + Retry-After once `max_queued_gen` async jobs (Server tab,
default 200) are queued or running.
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
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

from fastapi import HTTPException  # noqa: E402


class TtlClamp(unittest.TestCase):
    def setUp(self):
        self._cfg = dict(main.jobs_cfg)

    def tearDown(self):
        main.jobs_cfg.clear()
        main.jobs_cfg.update(self._cfg)

    def test_clamped_to_the_default_cap(self):
        main.jobs_cfg.pop("max_ttl_s", None)
        self.assertEqual(main._clamp_ttl(10 ** 12), 7 * 86400)
        self.assertEqual(main._clamp_ttl(3600), 3600)

    def test_configurable_cap(self):
        main.jobs_cfg["max_ttl_s"] = 7200
        self.assertEqual(main._clamp_ttl(10 ** 6), 7200)

    def test_nonsense_falls_back_to_the_default(self):
        for v in (None, 0, -5, "abc", True, 1.5):
            self.assertIsNone(main._clamp_ttl(v), v)


class QueueCap(unittest.TestCase):
    def setUp(self):
        self._saved = (main.max_queued_gen, dict(main._gen_tasks), main._gen_pick)
        self.picked = False

        async def pick(alias, force, body):
            self.picked = True
            raise HTTPException(599, "reached the router")
        main._gen_pick = pick

    def tearDown(self):
        main.max_queued_gen, tasks, main._gen_pick = self._saved
        main._gen_tasks.clear()
        main._gen_tasks.update(tasks)

    def _run(self, body):
        req = types.SimpleNamespace(state=types.SimpleNamespace(), client=None, headers={})
        return asyncio.run(main.run_generation(body, req))

    def test_async_job_beyond_the_cap_is_refused(self):
        main.max_queued_gen = 2
        main._gen_tasks.update({"j1": object(), "j2": object()})
        with self.assertRaises(HTTPException) as cm:
            self._run({"model": "a", "mode": "async"})
        self.assertEqual(cm.exception.status_code, 503)
        self.assertIn("Retry-After", cm.exception.headers or {})
        self.assertFalse(self.picked)
        with self.assertRaises(HTTPException) as cm:
            self._run({"model": "a", "output": {"mode": "async"}})
        self.assertEqual(cm.exception.status_code, 503)

    def test_below_the_cap_and_sync_pass(self):
        main.max_queued_gen = 2
        main._gen_tasks.update({"j1": object()})
        with self.assertRaises(HTTPException) as cm:
            self._run({"model": "a", "mode": "async"})
        self.assertEqual(cm.exception.status_code, 599)
        main._gen_tasks.update({"j2": object(), "j3": object()})
        with self.assertRaises(HTTPException) as cm:
            self._run({"model": "a", "mode": "sync"})
        self.assertEqual(cm.exception.status_code, 599)


if __name__ == "__main__":
    unittest.main()
