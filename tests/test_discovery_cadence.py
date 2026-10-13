"""Discovery cadence — each backend on its own clock, cloud catalogs rarely.

Why this fails SILENTLY: discovery has no caller that waits for it. Measured
2026-10-13 on prod: NanoGPT's direct API host answered its six catalog GETs in 1–18 s
against a 5 s discovery timeout, so the backend was DOWN 17 % of the time
— 259 outages in 36 h, every fault row reading just "ReadTimeout". Nothing errored:
routing simply skipped the backend. The fixes each break quietly again:

- one shared cycle (poll all, then sleep) lets a 60 s vendor catalog stretch every LAN
  poll and ComfyUI's stuck watchdog to a minute;
- a cloud catalog polled every 30 s (and every 3 s by the fast probe while DOWN) asks
  the vendor for ~1.6 MB of catalogs around the clock;
- a changed backend that waits out a 600 s interval keeps serving its OLD url/key;
- a DOWN cloud backend that waits out its catalog interval stays out of routing for
  ten minutes after the vendor is back;
- a fault detail without the request leaves "which catalog?" to be measured by hand.

Run: venv/bin/python -m unittest tests.test_discovery_cadence -v
"""
import asyncio
import os
import sys
import tempfile
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
    import main
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import logging

logging.getLogger("main").setLevel(logging.ERROR)

_LAN_LLM = {"name": "strix", "type": "openai", "url": "http://192.168.8.31:8080"}
_CLOUD_LLM = {"name": "openrouter", "type": "openai", "url": "https://openrouter.ai/api"}
_NANO = {"name": "nano", "type": "nanogpt", "url": "https://api.nano-gpt.com/api", "api_key": "k"}
_MESHY = {"name": "meshy", "type": "meshy", "url": "https://api.meshy.ai"}
_COMFY = {"name": "k12", "type": "comfyui", "url": "http://192.168.8.37:8188"}
_CLAUDE = {"name": "claude", "type": "anthropic", "url": "https://api.anthropic.com"}


class _State:
    """Swap main's backend tables for one test and put them back."""
    _NAMES = ("backends", "backend_healthy", "health_check_interval")

    def setUp(self):
        self._saved = {n: getattr(main, n) for n in self._NAMES}
        self._saved_adapters = dict(main.backend_adapters)
        self._saved_polled = dict(main._polled)
        main.backend_healthy = {}
        main.health_check_interval = 30
        main._polled.clear()

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(main, n, v)
        main.backend_adapters.clear()
        main.backend_adapters.update(self._saved_adapters)
        main._polled.clear()
        main._polled.update(self._saved_polled)
        main._poll_tasks.clear()

    def use(self, *bs):
        main.backends = [dict(b) for b in bs]
        main.build_backend_adapters()
        return main.backends


class LanUrlTest(unittest.TestCase):
    def test_classification(self):
        for url in ("http://192.168.8.31:8080", "http://127.0.0.1:18188", "http://10.0.0.5",
                    "http://100.101.7.3:8080",          # Tailscale CGNAT
                    "http://[fd00::5]:8188", "http://comfy:8188", "http://gpu.lan:8080",
                    "http://box.local", "http://localhost:4000"):
            self.assertTrue(adapters.is_lan_url(url), url)
        for url in ("https://api.nano-gpt.com/api", "https://openrouter.ai/api",
                    "https://api.anthropic.com", "http://8.8.8.8", "", None):
            self.assertFalse(adapters.is_lan_url(url), url)


class TypeDefaultTest(_State, unittest.TestCase):
    def test_defaults_by_what_a_poll_learns(self):
        self.use(_LAN_LLM, _CLOUD_LLM, _NANO, _MESHY, _COMFY, _CLAUDE)
        ad = {b.name: b for b in main.backend_adapters.values()}
        self.assertIsNone(ad["strix"].discovery_interval_default())
        self.assertIsNone(ad["k12"].discovery_interval_default())     # the watchdog needs 30 s
        for name in ("openrouter", "nano", "claude"):
            self.assertEqual(ad[name].discovery_interval_default(), 600.0, name)
        self.assertEqual(ad["meshy"].discovery_interval_default(), 300.0)

    def test_remote_backends_skip_the_fast_probe(self):
        self.use(_LAN_LLM, _CLOUD_LLM, _NANO, _MESHY, _COMFY)
        # all DOWN, and calls are waiting
        self.assertEqual({b["name"] for b in main._fast_probe_targets()}, {"strix", "k12"})

    def test_own_interval_beats_the_type_default(self):
        lan, nano = self.use(dict(_LAN_LLM, discovery_interval_s=90),
                             dict(_NANO, discovery_interval_s=120))
        self.assertEqual(main.discovery_interval(lan), 90.0)
        self.assertEqual(main.discovery_interval(nano), 120.0)
        for junk in (0, -5, True, "60", None):
            (b,) = self.use(dict(_NANO, discovery_interval_s=junk))
            self.assertEqual(main.discovery_interval(b), 600.0, junk)

    def test_lan_follows_health_check_interval(self):
        (lan,) = self.use(_LAN_LLM)
        main.health_check_interval = 45
        self.assertEqual(main.discovery_interval(lan), 45.0)


class DueTest(_State, unittest.TestCase):
    def test_due_rules(self):
        nano, lan = self.use(_NANO, _LAN_LLM)
        bid, lbid = main.backend_id(nano), main.backend_id(lan)
        self.assertTrue(main._poll_is_due(nano, 1000.0))              # never polled
        main._polled[bid] = (1000.0, main.backend_adapters[bid])
        main._polled[lbid] = (1000.0, main.backend_adapters[lbid])
        main.backend_healthy.update({bid: True, lbid: True})
        self.assertFalse(main._poll_is_due(nano, 1000.0 + 599))
        self.assertTrue(main._poll_is_due(nano, 1000.0 + 600))
        self.assertTrue(main._poll_is_due(lan, 1000.0 + 30))
        # DOWN: retried on the health check interval, not the catalog interval
        main.backend_healthy[bid] = False
        self.assertFalse(main._poll_is_due(nano, 1000.0 + 29))
        self.assertTrue(main._poll_is_due(nano, 1000.0 + 30))

    def test_changed_settings_poll_at_once(self):
        (nano,) = self.use(_NANO)
        bid = main.backend_id(nano)
        main._polled[bid] = (1000.0, main.backend_adapters[bid])
        main.backend_healthy[bid] = True
        self.assertFalse(main._poll_is_due(nano, 1001.0))
        (nano,) = self.use(dict(_NANO, api_key="new"))                # new adapter instance
        self.assertTrue(main._poll_is_due(nano, 1001.0))
        # an unrelated edit keeps the instance → stays on its own clock
        main._polled[bid] = (1000.0, main.backend_adapters[bid])
        main.backends = main.backends + [dict(_LAN_LLM)]
        main.build_backend_adapters()
        self.assertFalse(main._poll_is_due(main.backends[0], 1001.0))


class SchedulerTest(_State, unittest.TestCase):
    def test_a_slow_backend_delays_nobody(self):
        """The slow poll runs as its own task; the fast backend is polled again on its
        own clock while the slow one is still in flight — and the slow one is never
        started twice."""
        nano, lan = self.use(_NANO, dict(_LAN_LLM, discovery_interval_s=1))
        calls = []
        release = asyncio.Event()

        async def fake_refresh(b, client):
            calls.append(b["name"])
            main.backend_healthy[main.backend_id(b)] = True
            if b["name"] == "nano":
                await release.wait()

        async def run():
            orig = main.refresh_backend
            main.refresh_backend = fake_refresh
            try:
                main._start_due_polls()
                await asyncio.sleep(0.05)
                self.assertEqual(sorted(calls), ["nano", "strix"])
                # pretend the LAN poll ended long ago → due again; nano still in flight
                lbid = main.backend_id(lan)
                main._polled[lbid] = (main._polled[lbid][0] - 5, main._polled[lbid][1])
                main._start_due_polls()
                await asyncio.sleep(0.05)
                self.assertEqual(calls.count("strix"), 2)
                self.assertEqual(calls.count("nano"), 1)
                release.set()
                await asyncio.sleep(0.05)
                self.assertEqual(main._poll_tasks, {})
            finally:
                main.refresh_backend = orig
                release.set()
        asyncio.run(run())


class FaultDetailTest(unittest.TestCase):
    def test_transport_error_names_the_request_without_query(self):
        req = httpx.Request("GET", "https://api.nano-gpt.com/api/v1/audio-models?type=tts&key=s3cret")
        info = main._classify_error(httpx.ReadTimeout("", request=req))
        self.assertEqual(info["kind"], "timeout")
        self.assertEqual(info["detail"], "ReadTimeout (GET /api/v1/audio-models)")
        self.assertNotIn("s3cret", info["detail"])

    def test_error_without_request_keeps_its_text(self):
        self.assertEqual(main._classify_error(httpx.ReadTimeout(""))["detail"], "ReadTimeout")


class NanoTimeoutTest(unittest.TestCase):
    def test_all_six_catalogs_get_the_long_timeout(self):
        seen = []

        class Client:
            async def get(self, url, headers=None, timeout=None):
                seen.append(timeout)
                raise httpx.ReadTimeout("", request=httpx.Request("GET", url))

        ad = adapters.make_adapter(dict(_NANO), main.adapter_ctx)

        async def run():
            with self.assertRaises(httpx.ReadTimeout):
                await ad.discover(Client())
        asyncio.run(run())
        self.assertEqual(len(seen), 6)
        self.assertTrue(all(t >= 60.0 for t in seen), seen)


if __name__ == "__main__":
    unittest.main()
