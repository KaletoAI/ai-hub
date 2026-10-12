"""Shared account caps must cover chat AND image work without releasing streams early.

Why this fails SILENTLY: separate backend ids can use the same subscription account.
Each backend's own max_concurrent=1 still lets one chat and one image request run at
once, and filtering disabled/draining members loses a slot that their active request
still holds. A stream returning its response headers is not a completed request.

Run: venv/bin/python -m unittest tests.test_concurrency_group -v
"""
import asyncio
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import httpx

# Importing main reads ./config.yaml; never read a developer's config or start lifespan.
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def _backend(name, kind="openai", cap=1, group="arliai-core"):
    return {"name": name, "type": kind, "url": "https://account.invalid",
            "max_concurrent": cap, "concurrency_group": group}


class SharedAccountCap(unittest.TestCase):
    def setUp(self):
        # Rebind, rather than clear, globals so every original object is restored.
        for attr, value in (
                ("backends", []), ("config_backends", []), ("backend_inflight", {}),
                ("max_concurrent_default", None), ("backend_hosts", {}),
                ("host_backends", {}), ("_gen_clash_warned", set()),
                ("_draining", set()), ("_drain_hold", set()), ("_drain_host", {}),
                ("_parked", [])):
            p = patch.object(main, attr, value)
            p.start()
            self.addCleanup(p.stop)
        # Keep backend merging real, but never create controllers or access a DB.
        for obj, attr, kwargs in (
                (main.store, "is_active", {"return_value": False}),
                (main, "sync_host_controllers", {}),
                (main, "sync_volume_controllers", {}),
                (main, "apply_hosts", {}),
                (main, "rebuild_route_index", {})):
            p = patch.object(obj, attr, **kwargs)
            p.start()
            self.addCleanup(p.stop)

    def _install(self, *backends):
        main.config_backends = list(backends)
        main.rebuild_backends()

    def test_chat_and_image_share_one_slot_and_release_wakes_waiters(self):
        # Same display name, distinct types: counters must stay keyed by backend id.
        chat = _backend("arliai")
        image = _backend("arliai", "arliai")
        self._install(chat, image)
        for held in (chat, image):
            with self.subTest(held=held["type"]):
                waiter = asyncio.Event()
                main._parked = [{"event": waiter}]
                self.assertFalse(main.backend_busy(chat))
                self.assertFalse(main.backend_busy(image))
                main._inflight_inc(main.backend_id(held))
                self.assertTrue(main.backend_busy(chat))
                self.assertTrue(main.backend_busy(image))
                self.assertFalse(waiter.is_set())
                main._inflight_dec(main.backend_id(held))
                self.assertFalse(main.backend_busy(chat))
                self.assertFalse(main.backend_busy(image))
                self.assertTrue(waiter.is_set())

    def test_group_uses_total_inflight_and_lowest_finite_cap(self):
        chat = _backend("chat", cap=2)
        image = _backend("image", "arliai", cap=4)
        unlimited = _backend("unlimited", cap=None)
        self._install(chat, image, unlimited)
        main._inflight_inc("arliai:image")
        for b in (chat, image, unlimited):
            self.assertFalse(main.backend_busy(b))
        main._inflight_inc("openai:unlimited")
        for b in (chat, image, unlimited):
            self.assertTrue(main.backend_busy(b))
        main._inflight_dec("arliai:image")
        for b in (chat, image, unlimited):
            self.assertFalse(main.backend_busy(b))

    def test_other_groups_and_ungrouped_backends_keep_independent_caps(self):
        chat = _backend("chat")
        image = _backend("image", "arliai")
        other = _backend("other", group="another-account")
        plain = _backend("plain", group="")
        plain_other = _backend("plain-other", group=None)
        self._install(chat, image, other, plain, plain_other)
        main._inflight_inc("openai:chat")
        self.assertTrue(main.backend_busy(image))
        for b in (other, plain, plain_other):
            self.assertFalse(main.backend_busy(b))
        main._inflight_inc("openai:other")
        main._inflight_inc("openai:plain")
        self.assertTrue(main.backend_busy(other))
        self.assertTrue(main.backend_busy(plain))
        self.assertFalse(main.backend_busy(plain_other))
        main._inflight_dec("openai:chat")
        self.assertFalse(main.backend_busy(image))
        self.assertTrue(main.backend_busy(other))
        self.assertTrue(main.backend_busy(plain))

    def test_group_without_finite_caps_is_unlimited(self):
        members = [_backend("none", cap=None), _backend("zero", "arliai", cap=0),
                   _backend("negative", cap=-1), _backend("text", cap="1")]
        self._install(*members)
        main.backend_inflight = {"openai:none": 7, "arliai:zero": 9,
                                 "openai:negative": 3, "openai:text": 2}
        for b in members:
            self.assertFalse(main.backend_busy(b))

    def test_inherited_global_cap_is_finite_and_remains_hot_reloadable(self):
        chat = _backend("chat", cap=4)
        image = _backend("image", "arliai")
        del image["max_concurrent"]
        main.max_concurrent_default = 2
        self._install(chat, image)
        main.backend_inflight = {"openai:chat": 1, "arliai:image": 1}
        self.assertTrue(main.backend_busy(chat))
        self.assertTrue(main.backend_busy(image))
        main.max_concurrent_default = 3
        self.assertFalse(main.backend_busy(chat))
        self.assertFalse(main.backend_busy(image))
        main._inflight_inc("openai:chat")
        self.assertTrue(main.backend_busy(image))

    def test_disabled_saved_member_still_holds_the_account_slot(self):
        chat = _backend("chat")
        image = _backend("image", "arliai")
        self._install(chat, image)
        main._inflight_inc("arliai:image")
        with patch.object(main.store, "is_active", return_value=True), \
                patch.object(main.store, "list_backends",
                             return_value=[dict(image, enabled=False)]):
            main.rebuild_backends()
        self.assertTrue(main.backend_busy(chat))
        main._inflight_dec("arliai:image")
        self.assertFalse(main.backend_busy(chat))

    def test_disabled_member_still_sets_the_minimum_group_cap(self):
        chat = _backend("chat", cap=4)
        image = dict(_backend("image", "arliai"), enabled=False)
        self._install(chat, image)
        main._inflight_inc("openai:chat")
        self.assertTrue(main.backend_busy(chat))

    def test_draining_member_holds_group_capacity_until_it_finishes(self):
        chat = _backend("chat")
        image = _backend("image", "arliai")
        self._install(chat, image)
        main._inflight_inc("arliai:image")
        main._draining.add("arliai:image")
        main._drain_hold.add("arliai:image")  # managed-host hold: no persisted disable
        main.rebuild_backends()
        self.assertTrue(main.backend_busy(chat))
        main._inflight_dec("arliai:image")
        self.assertFalse(main.backend_busy(chat))
        self.assertFalse(main.backend_busy(image))

    def test_saved_caps_apply_to_current_group_members(self):
        chat = _backend("chat", cap=4)
        image = _backend("image", "arliai", cap=4)
        self._install(chat, image)
        main.backend_inflight = {"openai:chat": 1, "arliai:image": 1}
        self.assertFalse(main.backend_busy(chat))
        self._install(chat, dict(image, max_concurrent=2))
        self.assertTrue(main.backend_busy(chat))
        self._install(chat, dict(image, concurrency_group="another-account"))
        self.assertFalse(main.backend_busy(chat))

    def _stream(self, close_early):
        chat = _backend("chat")
        image = _backend("image", "arliai")
        self._install(chat, image)

        async def go():
            async def body():
                yield (b'data: {"id":"c","model":"m","choices":[{"index":0,'
                       b'"delta":{"content":"hello"}}]}\n\n')
                yield b'data: [DONE]\n\n'

            def handler(request):
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      content=body())

            async def record_call(**kwargs):
                pass  # no SQLite or stats body writes

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                ctx = adapters.AdapterContext(
                    auth_headers=lambda b: {}, inflight_inc=main._inflight_inc,
                    inflight_dec=main._inflight_dec, cost_usd=lambda *a: 0.0,
                    source_of=lambda r: "test", record_call=record_call,
                    log_enabled=lambda: False, http_client=lambda: client)
                adapter = adapters.OpenAIAdapter(chat, ctx)
                req = adapters.NormalizedRequest(
                    path="/v1/chat/completions", alias="m", real_model="m", stream=True,
                    body={"model": "m", "stream": True,
                          "messages": [{"role": "user", "content": "hi"}]},
                    raw=types.SimpleNamespace(headers={}, client=None,
                                              state=types.SimpleNamespace()))
                resp = await adapter.dispatch(req)
                try:
                    self.assertTrue(main.backend_busy(image))  # headers alone are not completion
                    await resp.body_iterator.__anext__()
                    self.assertTrue(main.backend_busy(image))  # neither is one body chunk
                    if close_early:
                        await resp.body_iterator.aclose()
                    else:
                        async for _ in resp.body_iterator:
                            pass
                    self.assertFalse(main.backend_busy(chat))
                    self.assertFalse(main.backend_busy(image))
                    self.assertEqual(main.backend_inflight.get("openai:chat"), 0)
                    await asyncio.sleep(0)  # let the in-memory stats coroutine finish
                finally:
                    await resp.body_iterator.aclose()

        asyncio.run(go())

    def test_stream_holds_the_shared_slot_until_the_body_ends(self):
        self._stream(close_early=False)

    def test_aborted_stream_releases_the_shared_slot(self):
        self._stream(close_early=True)
