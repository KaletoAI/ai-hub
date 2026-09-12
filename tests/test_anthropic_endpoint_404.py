"""An Anthropic backend answers /v1/messages ONLY (`main.serves_path`, a licence
boundary: a personal Claude subscription covers Claude Code, not re-serving Claude as
a general-purpose API). A caller who asks for one of its models on
/v1/chat/completions therefore gets nothing — and what the gateway SAYS about that is
the whole point of these tests.

`_dispatch_or_park` has a branch that answers such a call with 404 plus the reason.
It read the candidates out of `_route_index`, which holds aliases and BARE model ids
only — never a '<backend>/<model>' pin (`rebuild_route_index`). So the explaining 404
fired for `claude-sonnet-5` and the pinned `claude/claude-sonnet-5` fell through to the
generic `503 No healthy backend`, which is wrong twice over: it sends the caller
diagnosing a backend that is perfectly healthy, and — as the branch's own comment
notes — a client does not retry a 404 but does retry a 503. Measured 2026-09-12 on
prod: one agent produced 3195 such 503s in three hours, all for the same pinned name.

The failure is silent in both directions, which is why it is pinned here: nothing
raises, a healthy backend is simply reported as absent, and the licence rule that
CAUSED the refusal is never mentioned. The last test is the guard on the fix itself —
the message may change, the boundary may not: after the fix the Anthropic backend must
still be no candidate at all on /v1/chat/completions.

Run: python -m unittest tests.test_anthropic_endpoint_404 -v
"""
import asyncio
import os
import sys
import tempfile
import types
import unittest

# `import main` reads ./config.yaml at import time — give it a minimal one in a temp cwd.
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    from fastapi import HTTPException
finally:
    os.chdir(_prev)
    _tmp.cleanup()


ANTHROPIC = {"name": "claude", "type": "anthropic", "url": "https://api.anthropic.com",
             "enabled": True}
OPENAI = {"name": "openrouter", "type": "openai", "url": "https://openrouter.ai/api",
          "enabled": True}


class AnthropicEndpointReporting(unittest.TestCase):
    """What the gateway tells a caller who asks for an Anthropic model on a chat path."""

    def setUp(self):
        # Rebuild the real routing state instead of hand-stubbing the index: the bug
        # was IN that index's shape, so a hand-built one could hide it.
        self._saved = (main.backends, dict(main.backend_models),
                       dict(main.backend_healthy), dict(main.virtual_models))
        main.backends = [ANTHROPIC, OPENAI]
        main.virtual_models = {}
        main.backend_models = {
            "anthropic:claude": {"claude-sonnet-5", "claude-opus-5"},
            "openai:openrouter": {"anthropic/claude-sonnet-5", "llama-3"},
        }
        main.backend_healthy = {"anthropic:claude": True, "openai:openrouter": True}
        main.rebuild_route_index()

    def tearDown(self):
        (main.backends, main.backend_models,
         main.backend_healthy, main.virtual_models) = self._saved
        main.rebuild_route_index()

    def _call(self, model, path="/v1/chat/completions"):
        """Run _dispatch_or_park far enough to reach its refusal; return the exception."""
        req = types.SimpleNamespace(state=types.SimpleNamespace())
        with self.assertRaises(HTTPException) as cm:
            asyncio.run(main._dispatch_or_park(model, path, {"model": model}, req))
        return cm.exception

    def test_pinned_anthropic_model_on_chat_path_is_404_with_the_reason(self):
        """THE BUG: 'claude/claude-sonnet-5' on /v1/chat/completions answered 503."""
        exc = self._call("claude/claude-sonnet-5")
        self.assertEqual(exc.status_code, 404, f"got {exc.status_code}: {exc.detail}")
        self.assertIn("/v1/messages", exc.detail)

    def test_bare_anthropic_model_on_chat_path_stays_404(self):
        """The case that already worked must keep working."""
        exc = self._call("claude-sonnet-5")
        self.assertEqual(exc.status_code, 404, f"got {exc.status_code}: {exc.detail}")
        self.assertIn("/v1/messages", exc.detail)

    def test_pinned_anthropic_model_on_messages_path_is_503_not_404(self):
        """ON /v1/messages the same empty candidate set means the backend is DOWN.

        A 404 there would tell the caller to use the endpoint they are already on, and
        Claude Code does not retry a 404 — so a transient outage would look permanent.
        """
        main.backend_healthy["anthropic:claude"] = False
        exc = self._call("claude/claude-sonnet-5", "/v1/messages")
        self.assertEqual(exc.status_code, 503, f"got {exc.status_code}: {exc.detail}")

    def test_pinned_model_on_a_chat_backend_that_is_down_stays_503(self):
        """The 404 must not spill onto non-Anthropic backends: down is still down."""
        main.backend_healthy["openai:openrouter"] = False
        exc = self._call("openrouter/llama-3")
        self.assertEqual(exc.status_code, 503, f"got {exc.status_code}: {exc.detail}")

    def test_model_the_anthropic_backend_does_not_serve_stays_503(self):
        """A name that backend never listed is not an endpoint mistake."""
        exc = self._call("claude/claude-nonexistent-9")
        self.assertEqual(exc.status_code, 503, f"got {exc.status_code}: {exc.detail}")

    def test_licence_boundary_stays_closed_on_the_chat_path(self):
        """The GUARD on the fix: only the MESSAGE changes, never the reachability.

        `serves_path` is a licence boundary, so after the fix the Anthropic backend
        must still be no candidate on /v1/chat/completions — neither ready nor busy,
        under either spelling — while it stays reachable on /v1/messages.
        """
        for name in ("claude-sonnet-5", "claude/claude-sonnet-5"):
            self.assertEqual(main.resolve_routes(name, "/v1/chat/completions"), ([], []),
                             f"{name} became routable on the chat path")
            ready, busy = main.resolve_routes(name, "/v1/messages")
            self.assertTrue(ready or busy, f"{name} unreachable on /v1/messages")


if __name__ == "__main__":
    unittest.main()
