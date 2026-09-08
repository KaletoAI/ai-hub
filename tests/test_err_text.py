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


if __name__ == "__main__":
    unittest.main()
