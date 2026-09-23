"""A streamed dispatch's end — slot, connection and call-log row — on every way it can end.

Why this fails SILENTLY: the adapter opens the upstream stream and claims the in-flight
slot BEFORE it answers, and releases both in its body generator's `finally`. A Python
async generator that was never STARTED runs no `finally` — `aclose()` on it is a no-op.
Both bridges (Messages and Responses) yield their own opening events (`message_start`,
`response.created`…) before they touch the adapter's iterator, so a client that leaves
in that window (Esc in Claude Code after a long park, a proxy timeout) closed nothing:
the backend stayed one slot busier for good, the pooled connection was lost and no 499
row was written. Nothing errors; the backend just parks calls it has room for, until a
restart. The same applies to a bridge generator that is dropped without ever running.

The other direction is just as quiet: a backend that reports its own failure IN the
stream (OpenRouter: `data: {"error": …}` under HTTP 200) makes the bridge stop and close
the source — which the adapter then read as the client leaving and booked 499 ("client
closed request"), and on the plain chat path the stream ended "normally" and booked 200.
Either way the call log blamed nobody for a backend failure.

And /v1/responses answered without `x-gateway-backend`/`x-reasoning-control` (streamed
and plain), the two headers every other endpoint carries — "which backend served this?"
had no answer on the one endpoint N8N uses.

Run: venv/bin/python -m unittest tests.test_stream_lifecycle -v
"""
import asyncio
import gc
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
    import anthropic_bridge
    import main
    import responses_bridge
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import logging  # noqa: E402

logging.getLogger("main").setLevel(logging.CRITICAL)
logging.getLogger("adapters").setLevel(logging.CRITICAL)
logging.getLogger("anthropic_bridge").setLevel(logging.CRITICAL)
logging.getLogger("responses_bridge").setLevel(logging.CRITICAL)


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n".encode()


CHAT = [_sse({"id": "c", "model": "m", "choices": [{"index": 0, "delta": {"content": w}}]})
        for w in ("one ", "two ", "three")]
DONE = [_sse({"id": "c", "model": "m", "choices": [{"index": 0, "delta": {},
                                                     "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n"]
# OpenRouter's documented mid-stream failure: HTTP 200, then an error chunk that still
# carries a `choices` entry with finish_reason "error".
OR_ERROR = _sse({"id": "c", "model": "m", "error": {"code": "server_error",
                                                    "message": "Provider disconnected"},
                 "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]})
BARE_ERROR = _sse({"error": {"message": "upstream overloaded"}})

ANTH_BACKEND = {"name": "claude", "type": "anthropic", "url": "https://api.anthropic.com"}
CHAT_BACKEND = {"name": "openrouter", "type": "openai", "url": "https://openrouter.ai/api"}


class _Harness:
    """An adapter over a MockTransport, counting the slot and collecting call-log rows."""

    def __init__(self, chunks, adapter_cls=adapters.OpenAIAdapter, backend=CHAT_BACKEND):
        self.rows, self.inc, self.dec = [], 0, 0
        self.opened = self.closed = 0
        outer = self

        async def body():
            outer.opened += 1
            try:
                for c in chunks:
                    yield c
                    await asyncio.sleep(0)
            finally:
                outer.closed += 1

        def handler(request):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=body())

        async def record_call(**kw):
            outer.rows.append(kw)

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        ctx = adapters.AdapterContext(
            auth_headers=lambda b: {}, cost_usd=lambda *a: 0.0, source_of=lambda r: "test",
            record_call=record_call, log_enabled=lambda: False,
            inflight_inc=lambda bid: setattr(outer, "inc", outer.inc + 1),
            inflight_dec=lambda bid: setattr(outer, "dec", outer.dec + 1))
        ctx.http_client = lambda: self.client
        self.adapter = adapter_cls(backend, ctx)

    def request(self, path, body):
        return adapters.NormalizedRequest(
            path=path, alias="m", real_model="m", body=body,
            raw=types.SimpleNamespace(headers={}, client=None, state=types.SimpleNamespace()),
            stream=True)

    async def settle(self):
        for _ in range(5):
            await asyncio.sleep(0)             # fire-and-forget record / close tasks
        await self.client.aclose()


def _chat_body():
    return {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}


def _messages_body():
    return {"model": "m", "stream": True, "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}]}


class ClientLeavesBeforeTheFirstChunk(unittest.TestCase):
    """R1: the slot and the row must not depend on the adapter's generator having started."""

    def _assert_released(self, h):
        self.assertEqual(h.inc, 1)
        self.assertEqual(h.dec, 1, "in-flight slot leaked")
        self.assertEqual([r["status"] for r in h.rows], [499])

    def test_messages_bridge_closed_after_its_first_own_event(self):
        h = _Harness(CHAT + DONE)

        async def go():
            resp = await h.adapter.dispatch(h.request("/v1/messages", _messages_body()))
            it = resp.body_iterator
            first = await it.__anext__()
            self.assertIn("message_start", first)
            await it.aclose()                  # the client left here
            await h.settle()
        asyncio.run(go())
        self._assert_released(h)

    def test_responses_bridge_closed_after_its_first_own_event(self):
        h = _Harness(CHAT + DONE)

        async def go():
            resp = await h.adapter.dispatch(h.request("/v1/chat/completions", _chat_body()))
            it = responses_bridge.responses_stream(resp, {"model": "m", "stream": True}, "m")
            first = await it.__anext__()
            self.assertIn("response.created", first)
            await it.aclose()
            await h.settle()
        asyncio.run(go())
        self._assert_released(h)

    def test_adapter_iterator_closed_without_ever_being_read(self):
        for cls, backend, path, body in (
                (adapters.OpenAIAdapter, CHAT_BACKEND, "/v1/chat/completions", _chat_body()),
                (adapters.AnthropicAdapter, ANTH_BACKEND, "/v1/messages", _messages_body())):
            with self.subTest(adapter=cls.__name__):
                h = _Harness(CHAT + DONE, cls, backend)

                async def go():
                    resp = await h.adapter.dispatch(h.request(path, body))
                    await resp.body_iterator.aclose()
                    await resp.body_iterator.aclose()      # a second close is harmless
                    await h.settle()
                asyncio.run(go())
                self._assert_released(h)

    def test_a_bridge_dropped_without_ever_running_still_releases(self):
        # Nobody calls aclose() at all — the response object is just garbage.
        h = _Harness(CHAT + DONE)

        async def go():
            resp = await h.adapter.dispatch(h.request("/v1/chat/completions", _chat_body()))
            gen = anthropic_bridge.messages_stream(resp, "m")
            del gen, resp
            gc.collect()
            await h.settle()
        asyncio.run(go())
        self._assert_released(h)
        self.assertEqual(h.opened, h.closed, "upstream response left open")

    def test_a_normal_read_to_the_end_still_books_one_200(self):
        h = _Harness(CHAT + DONE)

        async def go():
            resp = await h.adapter.dispatch(h.request("/v1/messages", _messages_body()))
            async for _ in resp.body_iterator:
                pass
            await h.settle()
        asyncio.run(go())
        self.assertEqual((h.inc, h.dec), (1, 1))
        self.assertEqual([r["status"] for r in h.rows], [200])


class InBandBackendError(unittest.TestCase):
    """R11: a failure the backend reports IN the stream is the backend's, never the client's."""

    def _run(self, chunks, consume):
        h = _Harness(chunks)

        async def go():
            out = await consume(h)
            await h.settle()
            return out
        out = asyncio.run(go())
        self.assertEqual((h.inc, h.dec), (1, 1))
        return h, out

    def test_through_the_messages_bridge(self):
        async def consume(h):
            resp = await h.adapter.dispatch(h.request("/v1/messages", _messages_body()))
            return [p async for p in resp.body_iterator]
        for err in (OR_ERROR, BARE_ERROR):
            with self.subTest(err=err[:40]):
                h, out = self._run(CHAT[:1] + [err] + CHAT[1:] + DONE, consume)
                self.assertIn("event: error", "".join(out))
                self.assertEqual([r["status"] for r in h.rows], [502])

    def test_through_the_responses_bridge(self):
        async def consume(h):
            resp = await h.adapter.dispatch(h.request("/v1/chat/completions", _chat_body()))
            return [p async for p in responses_bridge.responses_stream(
                resp, {"model": "m", "stream": True}, "m")]
        for err in (OR_ERROR, BARE_ERROR):
            with self.subTest(err=err[:40]):
                h, out = self._run(CHAT[:1] + [err] + DONE, consume)
                text = "".join(out)
                self.assertIn("response.failed", text)
                self.assertNotIn("response.completed", text)
                self.assertEqual([r["status"] for r in h.rows], [502])

    def test_on_the_plain_chat_stream(self):
        # Passed through verbatim (the client's SDK raises on it) — but not booked as 200.
        async def consume(h):
            resp = await h.adapter.dispatch(h.request("/v1/chat/completions", _chat_body()))
            return [p async for p in resp.body_iterator]
        h, out = self._run(CHAT[:1] + [BARE_ERROR] + DONE, consume)
        self.assertIn(b"upstream overloaded", b"".join(out))
        self.assertEqual([r["status"] for r in h.rows], [502])

    def test_on_the_anthropic_passthrough(self):
        err = (b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error",'
               b'"message":"Overloaded"}}\n\n')
        start = (b'event: message_start\ndata: {"type":"message_start","message":'
                 b'{"usage":{"input_tokens":4,"output_tokens":1}}}\n\n')
        h = _Harness([start, err], adapters.AnthropicAdapter, ANTH_BACKEND)

        async def go():
            resp = await h.adapter.dispatch(h.request("/v1/messages", _messages_body()))
            out = [p async for p in resp.body_iterator]
            await h.settle()
            return out
        out = asyncio.run(go())
        self.assertIn(b"overloaded_error", b"".join(out))   # still verbatim
        self.assertEqual([r["status"] for r in h.rows], [502])

    def test_a_clean_stream_is_still_200(self):
        async def consume(h):
            resp = await h.adapter.dispatch(h.request("/v1/chat/completions", _chat_body()))
            return [p async for p in resp.body_iterator]
        h, _ = self._run(CHAT + DONE, consume)
        self.assertEqual([r["status"] for r in h.rows], [200])


class ResponsesKeepsGatewayHeaders(unittest.TestCase):
    """R18: /v1/responses said nothing about which backend answered or what reasoning
    control was applied — the two headers every other endpoint carries."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from fastapi.responses import JSONResponse, StreamingResponse
        self._saved = (main.api_key, main.users, main._users_by_key, main._dispatch_or_park)
        main.api_key, main.users, main._users_by_key = "", [], {}
        hdrs = {"x-gateway-backend": "dx10-01", "x-reasoning-control": "enable_thinking=false"}

        async def fake(alias, path, body, request, stats_endpoint=None):
            if body.get("stream"):
                async def gen():
                    for c in CHAT + DONE:
                        yield c
                return StreamingResponse(gen(), media_type="text/event-stream", headers=hdrs)
            r = JSONResponse({"id": "c", "model": "m", "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"},
                 "finish_reason": "stop"}]}, headers=hdrs)
            return r
        main._dispatch_or_park = fake
        self.c = TestClient(main.app)

    def tearDown(self):
        (main.api_key, main.users, main._users_by_key, main._dispatch_or_park) = self._saved

    def test_streamed(self):
        r = self.c.post("/v1/responses", json={"model": "m", "input": "hi", "stream": True})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("x-gateway-backend"), "dx10-01")
        self.assertEqual(r.headers.get("x-reasoning-control"), "enable_thinking=false")

    def test_not_streamed(self):
        r = self.c.post("/v1/responses", json={"model": "m", "input": "hi"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("x-gateway-backend"), "dx10-01")
        self.assertEqual(r.headers.get("x-reasoning-control"), "enable_thinking=false")


if __name__ == "__main__":
    unittest.main()
