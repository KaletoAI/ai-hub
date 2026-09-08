"""A job row and a log line must never end in a bare colon.

Measured 2026-09-02 on prod: chain job dbab9494 died with the error text "chain failed: "
and the journal line "✗ chain job … failed:" — nothing after either colon. The cause was
an httpx WriteTimeout on a ~93 MB task-create POST, and EVERY httpx timeout that comes
off the transport is constructed with an empty message, so `str(e)` is "". `_err_text`
falls back to the class name, which is a poor message but an infinitely better one than
none. These tests pin that fallback and the two renderings that reach a job row.

Run: venv/bin/python -m unittest tests.test_err_text -v
"""
import os
import sys
import tempfile
import unittest

import httpx

# `import main` reads ./config.yaml at import time — give it a minimal one in a temp cwd
# (same dance as test_gen_backend_for.py; the dir goes away in the same finally, or
# -W error::ResourceWarning trips on the finalizer).
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (tests/ is one level down)
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import adapters
    import main
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


class TestErrText(unittest.TestCase):
    def test_empty_httpx_timeout_falls_back_to_the_class_name(self):
        self.assertEqual(main._err_text(httpx.WriteTimeout("")), "WriteTimeout")
        self.assertEqual(main._err_text(httpx.ReadTimeout("")), "ReadTimeout")
        self.assertEqual(main._err_text(httpx.ConnectError("")), "ConnectError")

    def test_a_real_message_is_kept_verbatim(self):
        self.assertEqual(main._err_text(RuntimeError("x")), "x")
        self.assertEqual(main._err_text(httpx.WriteTimeout("upload stalled")), "upload stalled")

    def test_bare_exception_still_names_itself(self):
        self.assertEqual(main._err_text(TimeoutError()), "TimeoutError")

    def test_exhausted_message_never_ends_in_a_colon(self):
        for e in (httpx.WriteTimeout(""), TimeoutError(), ConnectionError()):
            msg = main._gen_exhausted_msg(e)
            self.assertFalse(msg.rstrip().endswith(":"), msg)
            self.assertIn(type(e).__name__, msg)
        # …and the no-candidate case (last is None) says so instead of rendering "None"
        self.assertNotIn("None", main._gen_exhausted_msg(None))

    def test_retryable_cloud_fault_is_not_reported_as_unreachable(self):
        # The task REACHED the vendor and the vendor broke it — saying "unreachable" here
        # sends the operator diagnosing a network that answered perfectly well.
        e = adapters.CloudTaskRetryable("Meshy task t1 failed: boom", vendor="Meshy")
        msg = main._gen_exhausted_msg(e)
        self.assertNotIn("unreachable", msg)
        self.assertIn("Meshy", msg)
        self.assertIn("boom", msg)
        self.assertIn("no credits", msg)                 # "consumed no credits"
        self.assertEqual(main._fault_label(e), "Meshy failed the task on its side")


class TestGenFailMeta(unittest.TestCase):
    """What a FAILED generation job carries. A failed cloud run returns no GenOutput, so
    these keys are the only record of the vendor task — and an empty meta is silent: the
    job row simply shows nothing and the task id survives only inside the error text."""

    def test_cloud_trace_reaches_the_failed_row(self):
        trace = {"cloud": "meshy", "cloud_task_id": "t1", "endpoint": "image-to-3d",
                 "request": {"ai_model": "latest"}}
        meta = main._gen_fail_meta(1, trace)
        self.assertEqual(meta["cloud_task_id"], "t1")
        self.assertEqual(meta["endpoint"], "image-to-3d")
        self.assertNotIn("attempts", meta)               # a single attempt is not worth noting

    def test_attempts_ride_along_without_a_cloud_run(self):
        self.assertEqual(main._gen_fail_meta(3, {}), {"attempts": 3})

    def test_nothing_to_say_stays_none(self):
        self.assertIsNone(main._gen_fail_meta(1, {}))

    def test_the_trace_is_copied_not_aliased(self):
        # jobs.fail merges the dict into the row; mutating the caller's trace afterwards
        # must not rewrite what was recorded.
        trace = {"cloud": "tripo", "cloud_task_id": "t1"}
        meta = main._gen_fail_meta(2, trace)
        trace["cloud_task_id"] = "CHANGED"
        self.assertEqual(meta["cloud_task_id"], "t1")

    def test_trace_of_a_request_without_one_is_empty(self):
        self.assertEqual(main._cloud_trace_of(None), {})
        self.assertEqual(main._cloud_trace_of(adapters.NormalizedRequest()), {})


if __name__ == "__main__":
    unittest.main()
