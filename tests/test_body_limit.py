"""Request body size caps.

Why this fails SILENTLY: every endpoint reads its body whole (`request.json()`,
`request.body()`), and nothing bounded it — a single client could make the gateway hold
an arbitrarily large body in memory, and the first anyone would notice is the OOM kill
of the whole service (review 2026-09-23, S10). So `max_body_mb` (config.yaml, default
200 — far above the ~86 MB a 64 MB mesh takes as base64 JSON) caps every request: a
declared Content-Length over it is refused before a byte is read, a chunked body is
counted while it streams; and the console's url-encoded forms (`admin._form`) have their
own, much smaller cap, because they carry only what an admin typed or pasted.
"""
import asyncio
import os
import sys
import tempfile
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
    import admin
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class BodyLimit(unittest.TestCase):
    def setUp(self):
        self._cfg = dict(main.config)
        self._saved = (main.api_key, main.users, main._users_by_key)
        main.api_key, main.users, main._users_by_key = "", [], {}
        self.c = TestClient(main.app)

    def tearDown(self):
        main.config.clear()
        main.config.update(self._cfg)
        main.api_key, main.users, main._users_by_key = self._saved

    def test_default_is_generous(self):
        main.config.pop("max_body_mb", None)
        self.assertEqual(main.max_body_bytes(), 200 * 1024 * 1024)
        main.config["max_body_mb"] = 0
        self.assertEqual(main.max_body_bytes(), 0)

    def test_declared_length_over_the_cap_is_refused(self):
        main.config["max_body_mb"] = 0.001               # ~1 KB
        r = self.c.post("/v1/chat/completions", content=b"{" + b" " * 4000 + b"}",
                        headers={"content-type": "application/json"})
        self.assertEqual(r.status_code, 413)
        self.assertIn("max_body_mb", r.text)

    def test_chunked_body_is_counted(self):
        main.config["max_body_mb"] = 0.001

        def gen():
            for _ in range(10):
                yield b" " * 500
        r = self.c.post("/v1/chat/completions", content=gen(),
                        headers={"content-type": "application/json"})
        self.assertEqual(r.status_code, 413)

    def test_small_body_passes_the_cap(self):
        main.config["max_body_mb"] = 0.001
        r = self.c.post("/v1/chat/completions", json={"model": "nope", "messages": []})
        self.assertNotEqual(r.status_code, 413)


class FormCap(unittest.TestCase):
    class _Req:
        def __init__(self, n):
            self.n = n

        async def stream(self):
            for _ in range(self.n):
                yield b"a=" + b"x" * 1022

    def test_form_is_capped_while_streaming(self):
        saved = admin._FORM_MAX_BYTES
        admin._FORM_MAX_BYTES = 4096
        try:
            with self.assertRaises(HTTPException) as cm:
                asyncio.run(admin._form(self._Req(10)))
            self.assertEqual(cm.exception.status_code, 413)
            self.assertIn("a", asyncio.run(admin._form(self._Req(2))))
        finally:
            admin._FORM_MAX_BYTES = saved


if __name__ == "__main__":
    unittest.main()
