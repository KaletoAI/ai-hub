"""Client OAuth must stay request-local while the gateway key never leaves this hop.

Silent failure guards: a saved backend key must not mask missing client OAuth or
charge another account; concurrent clients must not share credentials; a rejected
client token must not remove the shared backend's routes. Discovery has no client
token and must advertise only configured models without any provider request.
The real adapters run against HTTPX's transport seam, including raw payload/SSE
and the foreign-model Messages bridge, so header filtering cannot pass merely by
constructing a plausible response.

Run: venv/bin/python -m unittest tests.test_anthropic_passthrough -v
"""
import asyncio
import json
import os
import sys
import tempfile
import types
import unittest

import httpx
from starlette.datastructures import Headers

# main reads ./config.yaml on import. Never read the workstation's live config.
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
with tempfile.TemporaryDirectory() as _tmp:
    with open(os.path.join(_tmp, "config.yaml"), "w") as _f:
        _f.write('api_key: ""\nbackends: []\n')
    os.chdir(_tmp)
    sys.path.insert(0, _here)
    try:
        import adapters
        import faults
        import main
    finally:
        os.chdir(_prev)


BACKEND = {"name": "claude", "type": "anthropic", "url": "https://anthropic.test",
           "auth_mode": "passthrough", "api_key": "SAVED-KEY-MUST-NOT-BE-USED",
           "models": ["claude-sonnet-test"]}
MESSAGE = (b'{ "id": "msg_test", "type": "message", "role": "assistant", '
           b'"model": "claude-sonnet-test", "content": [{"type":"text","text":"Hello"}], '
           b'"stop_reason": "end_turn", "usage": {"input_tokens": 9, "output_tokens": 2} }')


def _request(authorization="Bearer sk-ant-oat01-client", trusted=True,
             path="/v1/messages", body=None, headers=None):
    incoming = {"x-ai-hub-key": "Bearer GATEWAY-KEY", "x-api-key": "OTHER-GATEWAY-KEY"}
    if authorization is not None:
        incoming["authorization"] = authorization
    incoming.update(headers or {})
    state = types.SimpleNamespace()
    if trusted is not None:
        state.gw_oauth_passthrough = trusted
    raw = types.SimpleNamespace(headers=Headers(incoming), state=state, client=None)
    payload = body if body is not None else {
        "model": "claude-sonnet-test", "max_tokens": 8,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    return adapters.NormalizedRequest(path=path, alias="claude", real_model=payload["model"],
                                      body=payload, raw=raw, stream=bool(payload.get("stream")))


class PassthroughOAuth(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seen, self.records, self.slots, self.active = [], [], [], []
        self.handler = lambda request: httpx.Response(
            200, content=MESSAGE, headers={"content-type": "application/json"})

        async def transport(request):
            self.seen.append(request)
            result = self.handler(request)
            if hasattr(result, "__await__"):
                result = await result
            return result

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(transport))

        async def record_call(**kw):
            self.records.append(kw)

        self.ctx = adapters.AdapterContext(
            auth_headers=lambda b: {"authorization": "Bearer FOREIGN-BACKEND-KEY"},
            inflight_inc=lambda bid: self.slots.append(("inc", bid)),
            inflight_dec=lambda bid: self.slots.append(("dec", bid)),
            cost_usd=lambda *args: 0.0, source_of=lambda raw: "test",
            record_call=record_call, log_enabled=lambda: False,
            active_register=lambda meta: self.active.append(meta),
            http_client=lambda: self.client)
        self.adapter = adapters.AnthropicAdapter(dict(BACKEND), self.ctx)

    async def asyncTearDown(self):
        await asyncio.sleep(0)  # let fire-and-forget stats records finish
        await self.client.aclose()

    async def test_concurrent_clients_use_only_their_own_raw_oauth(self):
        both_open = asyncio.Event()

        async def provider(request):
            if len(self.seen) == 2:
                both_open.set()
            await both_open.wait()
            return httpx.Response(200, content=MESSAGE,
                                  headers={"content-type": "application/json"})

        self.handler = provider
        alice = _request("Bearer sk-ant-oat01-alice")
        bob = _request("Bearer sk-ant-oat01-bob")
        alice.body["messages"][0]["content"] = "Alice"
        bob.body["messages"][0]["content"] = "Bob"
        responses = await asyncio.wait_for(
            asyncio.gather(self.adapter.dispatch(alice), self.adapter.dispatch(bob)), 2)
        self.assertEqual([response.body for response in responses], [MESSAGE, MESSAGE])
        got = {json.loads(request.content)["messages"][0]["content"]:
               request.headers["authorization"] for request in self.seen}
        self.assertEqual(got, {"Alice": "Bearer sk-ant-oat01-alice",
                               "Bob": "Bearer sk-ant-oat01-bob"})
        for request in self.seen:
            self.assertNotIn("x-ai-hub-key", request.headers)
            self.assertNotIn("x-api-key", request.headers)
        self.assertEqual(self.adapter.backend["api_key"], "SAVED-KEY-MUST-NOT-BE-USED")
        self.assertEqual([kind for kind, _ in self.slots].count("inc"), 2)
        self.assertEqual([kind for kind, _ in self.slots].count("dec"), 2)

    async def test_invalid_oauth_is_an_anthropic_401_before_claim_or_network(self):
        cases = [(None, True), ("", True), ("Bearer gateway-key", True),
                 ("Bearer sk-ant-api03-client", True), ("Basic sk-ant-oat01-client", True),
                 ("bearer sk-ant-oat01-client", True), ("Bearer sk-ant-oat", True),
                 ("Bearer sk-ant-oat01-client extra", True),
                 ("Bearer sk-ant-oat01-client", False),
                 ("Bearer sk-ant-oat01-client", None)]
        for path in ("/v1/messages", "/v1/messages/count_tokens"):
            for authorization, trusted in cases:
                with self.subTest(path=path, authorization=authorization, trusted=trusted):
                    response = await self.adapter.dispatch(_request(authorization, trusted, path))
                    self.assertEqual(response.status_code, 401)
                    error = json.loads(response.body)
                    self.assertEqual(error["type"], "error")
                    self.assertEqual(error["error"]["type"], "authentication_error")
        self.assertEqual(self.seen, [])
        self.assertEqual(self.slots, [])
        self.assertEqual(self.active, [])
        self.assertEqual(self.records, [])

    async def test_configured_discovery_never_contacts_provider_or_uses_saved_key(self):
        for configured in (["claude-sonnet-test", "claude-opus-test"],
                           " claude-sonnet-test, claude-opus-test , "):
            with self.subTest(configured=configured):
                self.adapter.backend["models"] = configured
                caps = await self.adapter.discover(self.client)
                self.assertEqual(caps.models, {"claude-sonnet-test", "claude-opus-test"})
                self.assertEqual(caps.pricing, {})
        self.assertEqual(self.seen, [])

    async def test_discovery_without_configured_models_clearly_refuses_without_network(self):
        for configured in (None, [], "", " , ", ["", " "]):
            with self.subTest(configured=configured):
                self.adapter.backend["models"] = configured
                with self.assertRaisesRegex(ValueError, "passthrough.*models|models.*passthrough"):
                    await self.adapter.discover(self.client)
        self.assertEqual(self.seen, [])

    async def test_beta_merge_and_raw_payload_preserve_claude_code_fields(self):
        body = {"model": "claude-sonnet-test", "max_tokens": 1024,
                "messages": [{"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "private", "signature": "signed"}]},
                    {"role": "user", "content": [{"type": "text", "text": "Go",
                     "cache_control": {"type": "ephemeral"}}]}],
                "thinking": {"type": "enabled", "budget_tokens": 512}, "_reasoning": "off"}
        self.adapter.backend["sampling_defaults"] = {"temperature": 0.99}
        for betas, expected in (("tools-test", ["tools-test", "oauth-2025-04-20"]),
                                ("tools-test, oauth-2025-04-20 ",
                                 ["tools-test", "oauth-2025-04-20"])):
            with self.subTest(betas=betas):
                request = _request(body=body, headers={"anthropic-beta": betas,
                                   "anthropic-version": "2024-10-22", "x-app": "cli"})
                response = await self.adapter.dispatch(request)
                self.assertEqual(response.body, MESSAGE)
                outgoing = self.seen[-1]
                self.assertEqual([beta.strip() for beta in
                                  outgoing.headers["anthropic-beta"].split(",")], expected)
                self.assertEqual(outgoing.headers["anthropic-version"], "2024-10-22")
                self.assertEqual(outgoing.headers["x-app"], "cli")
                forwarded = json.loads(outgoing.content)
                self.assertEqual(forwarded["thinking"], {"type": "enabled", "budget_tokens": 512})
                self.assertEqual(forwarded["messages"], body["messages"])
                self.assertNotIn("_reasoning", forwarded)
                self.assertNotIn("temperature", forwarded)

    async def test_count_tokens_uses_oauth_and_default_version(self):
        self.handler = lambda request: httpx.Response(200, json={"input_tokens": 17})
        response = await self.adapter.dispatch(_request(path="/v1/messages/count_tokens"))
        self.assertEqual(json.loads(response.body), {"input_tokens": 17})
        self.assertEqual(self.seen[0].url.path, "/v1/messages/count_tokens")
        self.assertEqual(self.seen[0].headers["authorization"], "Bearer sk-ant-oat01-client")
        self.assertEqual(self.seen[0].headers["anthropic-version"], "2023-06-01")
        self.assertEqual(self.seen[0].headers["anthropic-beta"], "oauth-2025-04-20")
        self.assertEqual([kind for kind, _ in self.slots], ["inc", "dec"])

    async def test_foreign_messages_backend_never_receives_either_client_credential(self):
        self.handler = lambda request: httpx.Response(200, json={
            "id": "chat-test", "object": "chat.completion", "model": "foreign-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 2}})
        foreign = adapters.OpenAIAdapter(
            {"name": "foreign", "type": "openai", "url": "https://foreign.test",
             "api_key": "foreign-key"}, self.ctx)
        response = await foreign.dispatch(_request())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)["content"][0]["text"], "Hello")
        self.assertEqual(self.seen[0].url.path, "/v1/chat/completions")
        self.assertEqual(self.seen[0].headers["authorization"], "Bearer FOREIGN-BACKEND-KEY")
        self.assertNotIn("x-api-key", self.seen[0].headers)
        self.assertNotIn("x-ai-hub-key", self.seen[0].headers)

    async def test_existing_auth_modes_keep_backend_credentials(self):
        for mode, key_header, expected in (("subscription", "authorization", "Bearer saved-token"),
                                            ("api_key", "x-api-key", "saved-token")):
            with self.subTest(mode=mode):
                backend = dict(BACKEND, auth_mode=mode, api_key="saved-token")
                adapter = adapters.AnthropicAdapter(backend, self.ctx)
                response = await adapter.dispatch(_request(trusted=False))
                self.assertEqual(response.status_code, 200)
                self.assertEqual(self.seen[-1].headers[key_header], expected)
                self.assertNotIn("x-ai-hub-key", self.seen[-1].headers)
                if mode == "subscription":
                    self.assertNotIn("x-api-key", self.seen[-1].headers)
                    self.assertEqual(self.seen[-1].headers["anthropic-beta"], "oauth-2025-04-20")
                else:
                    self.assertNotIn("authorization", self.seen[-1].headers)

    async def test_sse_thinking_signatures_and_tool_fragments_are_byte_exact(self):
        chunks = [
            b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":9}}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"signature_delta","signature":"sig+/="}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\\"x\\":"}}\n\n',
            b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":2}}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]

        async def stream():
            for chunk in chunks:
                yield chunk

        self.handler = lambda request: httpx.Response(
            200, content=stream(), headers={"content-type": "text/event-stream"})
        request = _request()
        request.body["stream"] = True
        response = await self.adapter.dispatch(request)
        self.assertEqual([kind for kind, _ in self.slots], ["inc"])
        output = b"".join([chunk async for chunk in response.body_iterator])
        self.assertEqual(output, b"".join(chunks))
        self.assertEqual([kind for kind, _ in self.slots], ["inc", "dec"])
        self.assertEqual(self.seen[0].headers["authorization"], "Bearer sk-ant-oat01-client")

    async def test_upstream_rejected_client_oauth_does_not_poison_shared_health(self):
        error = b'{"type":"error","error":{"type":"authentication_error","message":"expired OAuth"}}'
        self.handler = lambda request: httpx.Response(
            401, content=error, headers={"content-type": "application/json", "retry-after": "5"})
        keys = ("backend_adapters", "backend_healthy", "backend_error", "backend_last_key")
        saved = {key: getattr(main, key) for key in keys}
        saved_faults, saved_path = list(faults._MEM), faults._DB_PATH
        bid = "anthropic:claude"
        try:
            main.backend_adapters = {bid: self.adapter}
            main.backend_healthy = {bid: True}
            main.backend_error = {}
            main.backend_last_key = {}
            faults._MEM.clear()
            faults._DB_PATH = None
            for streaming in (False, True):
                with self.subTest(streaming=streaming):
                    request = _request()
                    request.body["stream"] = streaming
                    response = await main._dispatch_over(
                        [(self.adapter.backend, "claude-sonnet-test")], "/v1/messages", "claude",
                        request.body, request.raw)
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(response.body, error)
                    self.assertEqual(response.headers["retry-after"], "5")
                    self.assertEqual(main.backend_healthy, {bid: True})
                    self.assertEqual(main.backend_error, {})
                    self.assertEqual(faults.events_since(0), [])
            self.assertEqual(len(self.seen), 2)
            self.assertEqual([kind for kind, _ in self.slots], ["inc", "dec", "inc", "dec"])
            caps = await self.adapter.discover(self.client)
            self.assertEqual(caps.models, {"claude-sonnet-test"})
            self.assertEqual(len(self.seen), 2)
        finally:
            for key, value in saved.items():
                setattr(main, key, value)
            faults._MEM.clear()
            faults._MEM.extend(saved_faults)
            faults._DB_PATH = saved_path


if __name__ == "__main__":
    unittest.main()
