"""Three credential leaks that would be SILENT (nothing fails, the key is just gone).

  - `admin._self_api` (the playgrounds' call into the gateway's own API) carries the
    admin's or the master key. It once built its target from `request.base_url` — the
    caller's Host header — so a request with `Host: attacker.tld` and a valid session
    cookie made the gateway POST the key to the attacker. It must stay in-process.
  - httpx logs the full URL of every outgoing request; a query string (signed asset
    URLs, tokens in reference-image URLs) must not reach the journal.
  - /docs, /redoc and /openapi.json are unauthenticated FastAPI defaults and load
    third-party JS into the console's origin; they stay off.

Run: venv/bin/python -m unittest tests.test_self_api -v
"""
import asyncio
import logging
import os
import sys
import unittest
from unittest import mock

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import httpx  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

import admin  # noqa: E402
import main  # noqa: E402


class SelfApiStaysInProcess(unittest.TestCase):

    def test_host_header_cannot_redirect_the_key(self):
        seen = {}

        async def echo(request):
            seen["auth"] = request.headers.get("authorization")
            seen["source"] = request.headers.get("x-source")
            return JSONResponse({"ok": True})
        app = Starlette(routes=[Route("/v1/chat/completions", echo, methods=["POST"])])
        scope = {"type": "http", "method": "POST", "path": "/ui/chatplay/send",
                 "headers": [(b"host", b"attacker.tld")], "query_string": b"",
                 "scheme": "http", "server": ("10.0.0.1", 4000), "app": app}
        opened = []
        real = httpx.AsyncHTTPTransport.handle_async_request

        async def no_network(self, request):
            opened.append(str(request.url))
            return await real(self, request)
        with mock.patch.object(admin, "_playground_key", lambda who: "sk-secret"), \
                mock.patch.object(admin, "_session_user", lambda r: "kai"), \
                mock.patch.object(httpx.AsyncHTTPTransport, "handle_async_request", no_network):
            r = asyncio.run(admin._self_api(Request(scope), "POST", "/v1/chat/completions",
                                            json={"x": 1}))
        self.assertEqual(r.json(), {"ok": True})
        self.assertEqual(seen, {"auth": "Bearer sk-secret", "source": "playground"})
        self.assertEqual(opened, [])                  # no socket, to anyone
        self.assertNotIn("attacker", str(r.request.url))


class HttpxLogHasNoQuery(unittest.TestCase):

    def _line(self, url):
        rec = logging.LogRecord("httpx", logging.INFO, __file__, 1,
                                'HTTP Request: %s %s "%s %d %s"',
                                ("GET", httpx.URL(url), "HTTP/1.1", 200, "OK"), None)
        for f in logging.getLogger("httpx").filters:
            f.filter(rec)
        return rec.getMessage()

    def test_filter_is_installed(self):
        self.assertTrue(any(isinstance(f, main._RedactQuery)
                            for f in logging.getLogger("httpx").filters))

    def test_query_is_cut(self):
        line = self._line("https://cdn.example/a.glb?X-Amz-Signature=abc&token=t0p")
        self.assertIn("https://cdn.example/a.glb?…", line)
        self.assertNotIn("abc", line)
        self.assertNotIn("t0p", line)
        self.assertIn("200 OK", line)

    def test_plain_url_untouched(self):
        self.assertIn("GET http://10.0.0.5:8188/history/p1 ", self._line("http://10.0.0.5:8188/history/p1"))

    def test_foreign_record_shape_survives(self):
        rec = logging.LogRecord("httpx", logging.INFO, __file__, 1, "plain %s", ("x",), None)
        self.assertTrue(main._RedactQuery().filter(rec))
        self.assertEqual(rec.getMessage(), "plain x")


class NoInteractiveDocs(unittest.TestCase):

    def test_docs_routes_are_off(self):
        paths = {getattr(r, "path", None) for r in main.app.routes}
        for p in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"):
            self.assertNotIn(p, paths)


if __name__ == "__main__":
    unittest.main()
