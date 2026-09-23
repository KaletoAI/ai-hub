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

    def test_a_read_timeout_on_an_anthropic_backend_is_not_bought_twice(self):
        # R6: a subscription/API-key Claude backend keeps generating (and consuming quota
        # or billing) after the gateway gave up — the same case as a paid one. It is NOT
        # marked `paid`, which would demote it behind every unpaid chat candidate.
        claude = {"name": "claude", "type": "anthropic", "url": "https://api.anthropic.com"}
        second = self._ok()
        with self.assertRaises(HTTPException) as cm:
            self._dispatch([(claude, _Adapter(fail=httpx.ReadTimeout(""))), (B, second)],
                           path="/v1/messages")
        self.assertEqual(cm.exception.status_code, 504)
        self.assertEqual(second.calls, 0)
        self.assertIn("claude", cm.exception.detail)

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


def _ctx(recorded=None):
    async def record_call(**kw):
        if recorded is not None:
            recorded.append(kw)
    return adapters.AdapterContext(
        auth_headers=lambda b: {"authorization": "Bearer BACKEND-KEY"} if b.get("api_key") else {},
        inflight_inc=lambda bid: None, inflight_dec=lambda bid: None,
        cost_usd=lambda *a: 0.0, source_of=lambda r: "test", record_call=record_call,
        log_enabled=lambda: False)


def _raw(headers):
    return types.SimpleNamespace(headers=headers, client=None, state=types.SimpleNamespace())


BROWSER_HEADERS = {
    "host": "gw:4000", "content-length": "99", "authorization": "Bearer GATEWAY-KEY",
    "x-api-key": "GATEWAY-KEY", "connection": "keep-alive, x-hop-custom", "keep-alive": "timeout=5",
    "x-hop-custom": "1", "te": "trailers", "trailer": "x", "transfer-encoding": "chunked",
    "upgrade": "h2c", "proxy-authorization": "Basic Zm9v", "expect": "100-continue",
    "accept-encoding": "gzip, br, zstd", "cookie": "gw_session=SECRET",
    "x-forwarded-for": "10.0.0.7", "x-forwarded-host": "hub.lan", "forwarded": "for=10.0.0.7",
    "x-real-ip": "10.0.0.7", "x-source": "kai-laptop", "origin": "http://hub.lan",
    "referer": "http://hub.lan/ui/playground", "sec-fetch-site": "same-origin",
    "sec-ch-ua": '"Chromium"', "x-park-mode": "wait",
    # what backends DO read — must survive
    "content-type": "application/json", "accept": "text/event-stream",
    "user-agent": "claude-cli/2.1 (external, cli)", "anthropic-beta": "fine-grained-tool-streaming",
    "anthropic-version": "2023-06-01", "x-app": "cli", "x-stainless-lang": "js",
    "http-referer": "https://n8n.example", "x-title": "N8N agent",
}
KEPT = {"content-type", "accept", "user-agent", "anthropic-beta", "anthropic-version", "x-app",
        "x-stainless-lang", "http-referer", "x-title"}


class ForwardedHeaders(unittest.TestCase):
    """K9/S20: what reaches a backend of the client's headers."""

    def test_only_end_to_end_headers_a_backend_can_use_survive(self):
        self.assertEqual(set(adapters._forward_headers(BROWSER_HEADERS)), KEPT)

    def test_openai_adapter_forwards_the_filtered_set_plus_its_own_credential(self):
        a = adapters.OpenAIAdapter({"name": "or", "url": "http://x", "api_key": "k"}, _ctx())
        req = adapters.NormalizedRequest(path="/v1/chat/completions", alias="m", real_model="m",
                                         body={"model": "m", "messages": []}, raw=_raw(BROWSER_HEADERS))
        call = a._prepare(req)
        a._finish(call)
        self.assertEqual(set(call.headers), KEPT | {"authorization"})
        self.assertEqual(call.headers["authorization"], "Bearer BACKEND-KEY")

    def test_anthropic_passthrough_keeps_what_claude_code_sends(self):
        a = adapters.AnthropicAdapter({"name": "claude", "type": "anthropic",
                                       "url": "https://api.anthropic.com", "api_key": "tok"}, _ctx())
        req = adapters.NormalizedRequest(path="/v1/messages", alias="m", real_model="m",
                                         body={"model": "m", "messages": []}, raw=_raw(BROWSER_HEADERS))
        call = a._prepare(req)
        a._finish(call)
        for k in ("anthropic-beta", "anthropic-version", "user-agent", "x-app", "x-stainless-lang"):
            self.assertIn(k, call.headers)
        self.assertNotIn("cookie", call.headers)
        self.assertNotIn("x-api-key", call.headers)        # the gateway key never leaves


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n".encode()


CHAT_CHUNKS = [_sse({"id": "c", "model": "m", "choices": [{"index": 0, "delta": {"content": w}}]})
               for w in ("one ", "two ", "three")]
ANTH_CHUNKS = [
    b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":40,"output_tokens":1}}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":12}}\n\n',
]


class StreamStats(unittest.TestCase):
    """K11: a stream that does not end normally still books its call."""

    def _run(self, adapter_cls, backend, chunks, explode=None, abort_after=None, path=None):
        rows, counts = [], {"inc": 0, "dec": 0}

        async def body():
            for c in chunks:
                yield c
            if explode is not None:
                raise explode

        def handler(request):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

        async def go():
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            ctx = _ctx(rows)
            ctx.http_client = lambda: client
            ctx.inflight_inc = lambda bid: counts.__setitem__("inc", counts["inc"] + 1)
            ctx.inflight_dec = lambda bid: counts.__setitem__("dec", counts["dec"] + 1)
            a = adapter_cls(backend, ctx)
            req = adapters.NormalizedRequest(
                path=path or "/v1/chat/completions", alias="m", real_model="m",
                body={"model": "m", "stream": True, "messages": [{"role": "user", "content": "x" * 40}]},
                raw=_raw({}), stream=True)
            resp = await a.dispatch(req)
            it = resp.body_iterator
            got, err = 0, None
            self.out = []
            try:
                async for piece in it:
                    self.out.append(piece if isinstance(piece, bytes) else piece.encode())
                    got += 1
                    if abort_after is not None and got >= abort_after:
                        await it.aclose()          # what Starlette does when the client leaves
                        break
            except Exception as e:
                err = e
            for _ in range(3):
                await asyncio.sleep(0)             # let the fire-and-forget record task run
            await client.aclose()
            return err
        err = asyncio.run(go())
        return rows, counts, err

    def test_a_clean_stream_books_one_200(self):
        rows, counts, err = self._run(adapters.OpenAIAdapter, A, CHAT_CHUNKS)
        self.assertIsNone(err)
        self.assertEqual([r["status"] for r in rows], [200])
        self.assertEqual(counts["inc"], counts["dec"])

    def test_a_client_abort_books_499_with_the_tokens_so_far(self):
        rows, counts, _ = self._run(adapters.OpenAIAdapter, A, CHAT_CHUNKS, abort_after=2)
        self.assertEqual([r["status"] for r in rows], [499])
        self.assertGreaterEqual(rows[0]["output_tokens"], 1)
        self.assertGreater(rows[0]["input_tokens"], 0)       # the prompt was processed
        self.assertEqual(counts["inc"], counts["dec"])

    def test_an_upstream_drop_mid_stream_books_502(self):
        rows, counts, err = self._run(adapters.OpenAIAdapter, A, CHAT_CHUNKS,
                                      explode=httpx.ReadError("connection reset"))
        # The client is TOLD, in the stream, instead of getting a cut connection that
        # reads like a finished answer (no [DONE], no reason, an ASGI traceback).
        self.assertIsNone(err)
        last = self.out[-1].decode()
        self.assertTrue(last.startswith("data: "), last)
        self.assertIn("ReadError", json.loads(last[6:])["error"]["message"])
        self.assertNotIn(b"[DONE]", b"".join(self.out))
        self.assertEqual([r["status"] for r in rows], [502])
        self.assertEqual(rows[0]["output_tokens"], 3)
        self.assertEqual(counts["inc"], counts["dec"])

    def test_an_upstream_drop_on_the_anthropic_passthrough_sends_an_error_event(self):
        claude = {"name": "claude", "type": "anthropic", "url": "https://api.anthropic.com"}
        rows, counts, err = self._run(adapters.AnthropicAdapter, claude, ANTH_CHUNKS[:2],
                                      explode=httpx.RemoteProtocolError("peer closed"),
                                      path="/v1/messages")
        self.assertIsNone(err)
        tail = self.out[-1].decode()
        self.assertIn("event: error", tail)
        body = json.loads(tail.split("data: ", 1)[1])
        self.assertEqual(body["type"], "error")
        self.assertIn("RemoteProtocolError", body["error"]["message"])
        self.assertEqual([r["status"] for r in rows], [502])
        self.assertEqual(counts["inc"], counts["dec"])

    def test_the_anthropic_passthrough_books_an_abort_too(self):
        claude = {"name": "claude", "type": "anthropic", "url": "https://api.anthropic.com"}
        rows, counts, _ = self._run(adapters.AnthropicAdapter, claude, ANTH_CHUNKS,
                                    abort_after=3, path="/v1/messages")
        self.assertEqual([r["status"] for r in rows], [499])
        self.assertEqual((rows[0]["input_tokens"], rows[0]["output_tokens"]), (40, 12))
        self.assertEqual(counts["inc"], counts["dec"])


class SerializedOnce(unittest.TestCase):
    """P4: a multi-MB body (a Claude Code context, base64 images) was serialized twice on
    the event loop — once for the stats text, once more by httpx's `json=` — ~25-30 ms
    per MB each. It is now serialized ONCE, sent as bytes and reused as the stats text."""

    def _run(self, stream):
        sent, rows = [], []

        def handler(request):
            sent.append((request.content, request.headers.get("content-type")))
            if stream:
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      content=b"data: [DONE]\n\n")
            return httpx.Response(200, headers={"content-type": "application/json"},
                                  content=b'{"choices":[],"usage":{"prompt_tokens":1}}')

        import httpx._content as hc
        saved = hc.json_dumps

        def refuse(*a, **k):
            raise AssertionError("httpx serialized the body a second time")

        async def go():
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            ctx = _ctx(rows)
            ctx.http_client = lambda: client
            a = adapters.OpenAIAdapter(A, ctx)
            body = {"model": "m", "stream": stream,
                    "messages": [{"role": "user", "content": "Grüße " + "x" * 5000}]}
            req = adapters.NormalizedRequest(path="/v1/chat/completions", alias="m", real_model="m",
                                             body=body, raw=_raw({"content-type": "text/plain"}),
                                             stream=stream)
            resp = await a.dispatch(req)
            if stream:
                async for _ in resp.body_iterator:
                    pass
            for _ in range(3):
                await asyncio.sleep(0)
            await client.aclose()
        hc.json_dumps = refuse
        try:
            asyncio.run(go())
        finally:
            hc.json_dumps = saved
        return sent, rows

    def test_plain_call(self):
        sent, rows = self._run(stream=False)
        content, ctype = sent[0]
        self.assertEqual(ctype, "application/json")
        self.assertEqual(json.loads(content)["messages"][0]["content"][:5], "Grüße")
        self.assertEqual(rows[0]["request_text"].encode("utf-8"), content)

    def test_streamed_call(self):
        sent, rows = self._run(stream=True)
        content, ctype = sent[0]
        self.assertEqual(ctype, "application/json")
        self.assertTrue(json.loads(content)["stream_options"]["include_usage"])
        self.assertEqual(rows[0]["request_text"].encode("utf-8"), content)


class Preview(unittest.TestCase):
    """P4: the call-list preview is head + tail — it must not walk a multi-MB body."""

    def test_same_result_as_collapsing_the_whole_text(self):
        import stats
        for text in ('{"a": 1}', "  lead   ws\n" + "word " * 50000 + "\n tail  end ",
                     "x" * 300, "a b " * 30):
            full = " ".join(text.split())
            want = full if len(full) <= 100 else f"{full[:50]} … {full[-50:]}"
            self.assertEqual(stats._preview(text), want)

    def test_does_not_split_the_whole_body(self):
        import stats

        class Huge(str):
            def split(self, *a, **k):
                raise AssertionError("split over the whole body")
        self.assertTrue(stats._preview(Huge("y " * 200000)).startswith("y y"))


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
