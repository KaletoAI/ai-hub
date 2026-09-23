"""The chat dispatch path — failover classes, timeouts, forwarded headers, stats on abort.

Why this fails SILENTLY: every case here ends in a plausible answer. A pooled keep-alive
connection the backend had already closed (`RemoteProtocolError`) or a reset
(`ReadError`) used to skip the failover AND the fault log and surface as a raw
"Internal Server Error" — which Claude Code renders as a blank error, and which reads
like a gateway bug rather than a backend that dropped the call. The opposite mistake is
just as quiet: a paid backend that did not answer within the 300 s read timeout is
likely still generating (and billing) the answer, so failing over bought it twice. A
connect timeout of 300 s held the failover for five minutes on a host that swallows
SYNs. Headers were forwarded by a four-entry denylist, so a browser's cookies (the /ui
session among them), `x-forwarded-for` and `accept-encoding: br` — which the backend
then honours with a body the gateway cannot decode — all reached the backend. And a
streamed call the client aborted (Esc in Claude Code) or the backend dropped never
reached the call log at all, so its tokens were missing from the month-cost quota.

Run: venv/bin/python -m unittest tests.test_chat_dispatch -v
"""
import asyncio
import json
import os
import sys
import tempfile
import types
import unittest

import httpx

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import adapters
    import faults
    import main
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import logging

from fastapi import HTTPException
from fastapi.responses import Response

logging.getLogger("store").setLevel(logging.WARNING)
logging.getLogger("main").setLevel(logging.CRITICAL)


class _Adapter:
    def __init__(self, fail=None, resp=None):
        self.fail, self.resp, self.calls = fail, resp, 0

    async def dispatch(self, req):
        self.calls += 1
        if self.fail:
            raise self.fail
        return self.resp


A = {"name": "llamaswap-strix", "type": "openai", "url": "http://192.168.8.31:8080"}
B = {"name": "dx10-01", "type": "openai", "url": "http://192.168.8.35:8080"}
PAID = {"name": "openrouter", "type": "openai", "url": "https://openrouter.ai/api", "paid": True}


class _MainState(unittest.TestCase):
    KEYS = ("backends", "backend_adapters", "backend_hosts", "hosts_meta")

    def setUp(self):
        faults._DB_PATH = None
        faults._MEM.clear()
        self._saved = {k: getattr(main, k) for k in self.KEYS}
        main.backend_hosts, main.hosts_meta = {}, {}

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(main, k, v)
        faults._MEM.clear()

    def _dispatch(self, pairs, path="/v1/chat/completions"):
        main.backends = [b for b, _ in pairs]
        main.backend_adapters = {main.backend_id(b): a for b, a in pairs}
        request = types.SimpleNamespace(state=types.SimpleNamespace(), headers={}, client=None)
        return asyncio.run(main._dispatch_over([(b, "m") for b, _ in pairs], path,
                                               "tool", {"model": "tool"}, request))


class Failover(_MainState):
    """K8: which transport errors fail over, and which must not."""

    def _ok(self):
        return _Adapter(resp=Response(b"{}", status_code=200))

    def test_a_dropped_connection_fails_over_and_is_logged(self):
        for exc in (httpx.RemoteProtocolError("Server disconnected without sending a response."),
                    httpx.ReadError("Connection reset by peer"),
                    httpx.WriteError("Broken pipe")):
            with self.subTest(exc=type(exc).__name__):
                faults._MEM.clear()
                second = self._ok()
                resp = self._dispatch([(A, _Adapter(fail=exc)), (B, second)])
                self.assertEqual(resp.status_code, 200)
                self.assertEqual(second.calls, 1)
                evs = faults.events_since(0)
                self.assertEqual([(e["backend"], e["source"], e["kind"]) for e in evs],
                                 [("llamaswap-strix", "call", "connection_lost")])

    def test_a_read_timeout_on_a_paid_backend_is_not_bought_twice(self):
        second = self._ok()
        with self.assertRaises(HTTPException) as cm:
            self._dispatch([(PAID, _Adapter(fail=httpx.ReadTimeout(""))), (B, second)])
        self.assertEqual(cm.exception.status_code, 504)
        self.assertEqual(second.calls, 0)
        self.assertIn("openrouter", cm.exception.detail)
        self.assertEqual([e["kind"] for e in faults.events_since(0)], ["timeout"])

    def test_a_read_timeout_on_a_local_backend_still_fails_over(self):
        # Wasted local compute is the lesser evil than a stuck call — a hung
        # llama-swap load is exactly when another box helps.
        second = self._ok()
        self.assertEqual(self._dispatch([(A, _Adapter(fail=httpx.ReadTimeout(""))),
                                         (B, second)]).status_code, 200)
        self.assertEqual(second.calls, 1)

    def test_a_connect_timeout_on_a_paid_backend_fails_over(self):
        # Never connected = nothing was sent, nothing can be billed.
        second = self._ok()
        self.assertEqual(self._dispatch([(PAID, _Adapter(fail=httpx.ConnectTimeout(""))),
                                         (B, second)]).status_code, 200)

    def test_an_unexpected_exception_is_a_clean_502_not_a_raw_500(self):
        second = self._ok()
        with self.assertRaises(HTTPException) as cm:
            self._dispatch([(A, _Adapter(fail=KeyError("usage"))), (B, second)])
        self.assertEqual(cm.exception.status_code, 502)
        self.assertEqual(second.calls, 0)             # a bug is not the backend's fault to retry
        self.assertIn("llamaswap-strix", cm.exception.detail)
        self.assertEqual([e["kind"] for e in faults.events_since(0)], ["error"])

    def test_an_http_exception_from_the_adapter_passes_through(self):
        with self.assertRaises(HTTPException) as cm:
            self._dispatch([(A, _Adapter(fail=HTTPException(413, "too big")))])
        self.assertEqual(cm.exception.status_code, 413)

    def test_all_failed_names_the_error_even_when_it_has_no_text(self):
        with self.assertRaises(HTTPException) as cm:
            self._dispatch([(A, _Adapter(fail=httpx.ReadError("")))])
        self.assertEqual(cm.exception.status_code, 503)
        self.assertIn("ReadError", cm.exception.detail)


class UnexpectedErrorShape(unittest.TestCase):
    """K8: an exception nobody caught answers 502 in the endpoint's own error shape."""

    def setUp(self):
        from fastapi.testclient import TestClient
        self._saved = (main.api_key, main.users, main._users_by_key, main._dispatch_or_park)
        main.api_key, main.users, main._users_by_key = "", [], {}

        async def boom(*a, **k):
            raise RuntimeError("bridge exploded")
        main._dispatch_or_park = boom
        self.c = TestClient(main.app, raise_server_exceptions=False)

    def tearDown(self):
        (main.api_key, main.users, main._users_by_key, main._dispatch_or_park) = self._saved

    def test_chat_path(self):
        r = self.c.post("/v1/chat/completions", json={"model": "m", "messages": []})
        self.assertEqual(r.status_code, 502)
        self.assertIn("bridge exploded", r.json()["detail"])

    def test_messages_path_speaks_anthropic(self):
        r = self.c.post("/v1/messages", json={"model": "m", "max_tokens": 5, "messages": []})
        self.assertEqual(r.status_code, 502)
        body = r.json()
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "api_error")
        self.assertIn("bridge exploded", body["error"]["message"])


class Timeouts(unittest.TestCase):
    """P5: 300 s is a READ budget for long completions, never a connect budget."""

    def test_connect_and_pool_are_short_read_stays_long(self):
        t = adapters._CHAT_TIMEOUT
        self.assertIsInstance(t, httpx.Timeout)
        self.assertEqual(t.read, 300.0)
        self.assertLessEqual(t.connect, 10.0)
        self.assertIsNotNone(t.pool)
        self.assertLessEqual(t.pool, 60.0)


if __name__ == "__main__":
    unittest.main()
