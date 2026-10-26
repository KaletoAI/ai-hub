"""Separate AI-Hub authentication must not replace or leak a caller's seat token.

A subscription login sends OAuth in Authorization, while the gateway still needs
its own key. Choosing Authorization first silently rejects valid gateway users;
falling back from an invalid separate key silently bypasses its permissions.
Exercise the HTTP gates, real routing and both protocol adapters. The upstream
transport is isolated so no test can spend money or send credentials externally.
"""
import asyncio
import json
import os
import sys
import tempfile
import unittest
from html.parser import HTMLParser

import httpx
from fastapi.testclient import TestClient

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
with tempfile.TemporaryDirectory() as _tmp:
    with open(os.path.join(_tmp, "config.yaml"), "w") as _f:
        _f.write('api_key: ""\nbackends: []\n')
    os.chdir(_tmp)
    sys.path.insert(0, _here)
    try:
        import main
    finally:
        os.chdir(_prev)


class GatewayCredentials(unittest.TestCase):
    def setUp(self):
        names = ("api_key", "users", "_users_by_key", "backends", "backend_models",
                 "backend_healthy", "virtual_models", "_backend_names", "backend_adapters",
                 "http_client", "backend_inflight", "backend_tps")
        self.saved = {n: getattr(main, n) for n in names}
        main.api_key = "master"
        main.users = [{"name": "seat-user", "role": "user", "api_key": "gateway-user",
                       "models": ["claude-sonnet-test", "kimi-claude-compatible"]}]
        main._users_by_key = {u["api_key"]: u for u in main.users}
        main.backends = [
            {"name": "seat", "type": "anthropic", "url": "https://anthropic.test",
             "auth_mode": "passthrough", "models": ["claude-sonnet-test"]},
            {"name": "foreign", "type": "openai", "url": "https://foreign.test",
             "api_key": "provider-only"}]
        main._backend_names = {"seat", "foreign"}
        main.backend_models = {"anthropic:seat": {"claude-sonnet-test"},
                               "openai:foreign": {"kimi", "private-model"}}
        main.backend_healthy = {bid: True for bid in main.backend_models}
        main.virtual_models = {"kimi-claude-compatible": "kimi", "hidden": "private-model"}
        main.backend_inflight = {}
        main.backend_tps = {}
        main.backend_adapters = {}
        self.upstream = []
        main.http_client = httpx.AsyncClient(transport=httpx.MockTransport(self.provider))
        main.build_backend_adapters()
        main.rebuild_route_index()
        self.client = TestClient(main.app)
        self.headers = {"authorization": "Bearer sk-ant-oat01-client-a",
                        "x-ai-hub-key": "Bearer gateway-user"}

    def tearDown(self):
        self.client.close()
        asyncio.run(main.http_client.aclose())
        for name, value in self.saved.items():
            setattr(main, name, value)
        main.rebuild_route_index()

    def provider(self, request):
        self.upstream.append(request)
        if request.url.path.endswith("count_tokens"):
            return httpx.Response(200, json={"input_tokens": 37})
        if request.url.host == "anthropic.test":
            return httpx.Response(200, json={
                "id": "msg_seat", "type": "message", "role": "assistant",
                "model": "claude-sonnet-test", "content": [{"type": "text", "text": "seat works"}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 8, "output_tokens": 2}})
        return httpx.Response(200, json={
            "id": "chat_foreign", "object": "chat.completion", "model": "kimi",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "foreign works"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2}})

    def post(self, model="claude-sonnet-test", headers=None, path="/v1/messages"):
        return self.client.post(path, headers=self.headers if headers is None else headers,
                                json={"model": model, "max_tokens": 16,
                                      "messages": [{"role": "user", "content": "hello"}]})

    def test_discovery_uses_gateway_permissions_not_oauth(self):
        response = self.client.get("/v1/models?limit=1000", headers=self.headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual({r["id"] for r in response.json()["data"]},
                         {"seat/claude-sonnet-test", "kimi-claude-compatible"})
        response = self.client.get("/v1/models/hidden", headers=self.headers)
        self.assertEqual(response.status_code, 404)

    def test_invalid_separate_credential_never_falls_back_to_master(self):
        headers = {"authorization": "Bearer master", "x-ai-hub-key": "Bearer invalid"}
        for path in ("/v1/models", "/v1/models/hidden"):
            self.assertEqual(self.client.get(path, headers=headers).status_code, 401)
        self.assertEqual(self.post(headers=headers).status_code, 401)
        self.assertEqual(self.upstream, [])

    def test_blank_separate_credential_is_not_ignored(self):
        headers = {"authorization": "Bearer master", "x-ai-hub-key": ""}
        self.assertEqual(self.post(headers=headers).status_code, 401)
        self.assertEqual(self.upstream, [])

    def test_oauth_alone_is_not_a_gateway_credential(self):
        self.assertEqual(self.post(headers={"authorization": self.headers["authorization"]}).status_code, 401)
        self.assertEqual(self.upstream, [])

    def test_gateway_key_without_client_oauth_cannot_use_passthrough(self):
        response = self.post(headers={"x-ai-hub-key": "Bearer gateway-user"})
        self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(self.upstream, [])

    def test_bootstrap_open_cannot_authorize_client_oauth_forwarding(self):
        main.api_key, main.users, main._users_by_key = "", [], {}
        response = self.post()
        self.assertEqual(response.status_code, 401, response.text)
        self.assertEqual(self.upstream, [])

    def test_seat_and_foreign_models_in_one_client_keep_credentials_separate(self):
        seat = self.post()
        self.assertEqual(seat.status_code, 200, seat.text)
        self.assertEqual(seat.json()["content"], [{"type": "text", "text": "seat works"}])
        foreign = self.post("kimi-claude-compatible")
        self.assertEqual(foreign.status_code, 200, foreign.text)
        self.assertEqual(foreign.json()["content"], [{"type": "text", "text": "foreign works"}])
        self.assertEqual(self.upstream[0].headers["authorization"], "Bearer sk-ant-oat01-client-a")
        self.assertEqual(self.upstream[1].headers["authorization"], "Bearer provider-only")
        self.assertEqual(json.loads(self.upstream[1].content)["model"], "kimi")
        for request in self.upstream:
            self.assertNotIn("x-ai-hub-key", request.headers)
            self.assertNotIn("x-api-key", request.headers)
        self.assertEqual(main.backend_inflight, {"anthropic:seat": 0, "openai:foreign": 0})

    def test_count_tokens_uses_the_same_separate_authentication(self):
        response = self.post(path="/v1/messages/count_tokens")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"input_tokens": 37})
        self.assertEqual(self.upstream[0].headers["authorization"], self.headers["authorization"])

    def test_separate_gateway_key_cannot_expand_model_grants(self):
        response = self.post("hidden")
        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(self.upstream, [])

    def test_legacy_gateway_auth_remains_usable_for_foreign_models(self):
        for headers in ({"authorization": "Bearer gateway-user"}, {"x-api-key": "gateway-user"}):
            response = self.post("kimi-claude-compatible", headers=headers)
            self.assertEqual(response.status_code, 200, response.text)

    def test_config_backend_editor_keeps_client_oauth_and_model_ids(self):
        """Saving a YAML backend must not silently change its auth identity or models."""
        class Fields(HTMLParser):
            def __init__(self):
                super().__init__()
                self.models = None
                self.auth = None
                self.select = None

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "input" and attrs.get("name") == "models":
                    self.models = attrs.get("value")
                if tag == "select":
                    self.select = attrs.get("name")
                if tag == "option" and self.select == "auth_mode" and "selected" in attrs:
                    self.auth = attrs.get("value")

            def handle_endtag(self, tag):
                if tag == "select":
                    self.select = None

        response = asyncio.run(main.admin._backends_view({"edit": "anthropic:seat"}))
        fields = Fields()
        fields.feed(response.body.decode())
        self.assertEqual(fields.auth, "passthrough")
        self.assertEqual(fields.models, "claude-sonnet-test")


if __name__ == "__main__":
    unittest.main()
