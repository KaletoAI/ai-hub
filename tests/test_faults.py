"""The backend fault log — failures that must stay visible after the backend recovers.

Why this fails SILENTLY: every symptom it records used to vanish on its own. The
console's backend status is the CURRENT state (`backend_error` is popped by the next
good poll), a chat call that failed over books as a 200, and a media job that crashed
on one ComfyUI and then finished after a retry is a clean `done` row. Measured
2026-09-13 on prod: comfyui-strix (Evo-X2) crashed five times between 22:03 and 22:22
and nothing in the console said so. A log that records nothing, counts an ongoing
outage twice per poll, or bundles different messages into one line looks exactly
like a healthy fleet — so this pins the recording points in main, the derivation in
faults.py and what the console renders from it.

Run: venv/bin/python -m unittest tests.test_faults -v
"""
import asyncio
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
    import admin
    import faults
    import main
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import logging

logging.getLogger("store").setLevel(logging.WARNING)
logging.getLogger("main").setLevel(logging.ERROR)


def _reset_log():
    faults._DB_PATH = None
    faults._MEM.clear()


def _ev(ts, kind="unreachable", source="health", bid="comfyui:strix", detail="x", dur_s=None,
        status=None):
    return {"ts": ts, "bid": bid, "backend": bid.split(":", 1)[1], "type": bid.split(":", 1)[0],
            "host": "192.168.8.228", "source": source, "kind": kind, "status": status,
            "detail": detail, "dur_s": dur_s}


# ── derivation ────────────────────────────────────────────────────────────────

class BundleKey(unittest.TestCase):
    def test_ids_and_numbers_do_not_split_a_bundle(self):
        a = "img2img: connection issue: ComfyUI unreachable for >15s (job ebbe8cf407ae4ff3)"
        b = "img2img: connection issue: ComfyUI unreachable for >30s (job 19fe94fa542c4ed6)"
        self.assertEqual(faults.bundle_key(a), faults.bundle_key(b))

    def test_different_words_stay_apart(self):
        self.assertNotEqual(faults.bundle_key("node 305 UNETLoader: Unknown quantization"),
                            faults.bundle_key("CUDA out of memory"))


class Bundles(unittest.TestCase):
    def test_groups_by_message_counts_and_keeps_the_latest_text(self):
        evs = [_ev(100, source="job", kind="unreachable", detail="unreachable for >15s (job aaaaaaaa1)"),
               _ev(200, source="job", kind="unreachable", detail="unreachable for >16s (job bbbbbbbb2)"),
               _ev(150, source="job", kind="execution", detail="Unknown quantization format")]
        out = faults.bundles(evs)
        self.assertEqual(len(out), 2)
        top = out[0]                                   # newest group first
        self.assertEqual((top["count"], top["first"], top["last"]), (2, 100, 200))
        self.assertIn("bbbbbbbb2", top["message"])

    def test_recovered_is_not_a_fault(self):
        out = faults.bundles([_ev(100), _ev(160, kind=faults.RECOVERED, dur_s=60)])
        self.assertEqual([g["kind"] for g in out], ["unreachable"])

    def test_same_message_on_two_backends_is_two_bundles(self):
        out = faults.bundles([_ev(1, bid="openai:a"), _ev(2, bid="openai:b")])
        self.assertEqual(len(out), 2)


class PerBackend(unittest.TestCase):
    def test_counts_faults_outages_and_closed_downtime(self):
        evs = [_ev(1000), _ev(1100, kind=faults.RECOVERED, dur_s=100),
               _ev(1200, source="job", kind="execution", detail="boom")]
        s = faults.per_backend(evs, since=0, now=2000)["comfyui:strix"]
        self.assertEqual((s["faults"], s["outages"], s["downtime_s"]), (2, 1, 100))
        self.assertEqual((s["last_kind"], s["last_detail"], s["last_source"]), ("execution", "boom", "job"))

    def test_downtime_is_clipped_to_the_window(self):
        # Went down 500 s before the window opened, came back 100 s into it.
        evs = [_ev(1100, kind=faults.RECOVERED, dur_s=600)]
        s = faults.per_backend(evs, since=1000, now=2000)["comfyui:strix"]
        self.assertEqual(s["downtime_s"], 100)

    def test_an_open_outage_counts_up_to_now_even_without_an_event(self):
        # Down since before the window, never recovered: no event inside it at all.
        out = faults.per_backend([], since=1000, now=2000, down_since={"openai:evo": 400})
        self.assertEqual(out["openai:evo"]["downtime_s"], 1000)


class Storage(unittest.TestCase):
    def setUp(self):
        _reset_log()
        self._dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        _reset_log()
        self._dir.cleanup()

    def test_memory_only_without_init(self):
        faults.record(bid="openai:a", backend="a", source="call", kind="timeout", detail="ReadTimeout", ts=500)
        self.assertFalse(faults.is_persistent())
        self.assertEqual([e["kind"] for e in faults.events_since(0)], ["timeout"])
        self.assertEqual(faults.events_since(500), [])            # strictly newer than `since`

    def test_db_survives_a_restart(self):
        path = os.path.join(self._dir.name, "faults.db")
        faults.init(path)
        faults.record(bid="comfyui:strix", backend="strix", type="comfyui", host="h",
                      source="health", kind="unreachable", detail="All connection attempts failed",
                      ts=1000)
        faults._MEM.clear()                                       # "restart": memory is gone
        faults.init(path)
        evs = faults.events_since(0)
        self.assertEqual(len(evs), 1)
        self.assertEqual((evs[0]["backend"], evs[0]["kind"], evs[0]["host"]), ("strix", "unreachable", "h"))

    def test_unopenable_db_falls_back_to_memory(self):
        faults.init(os.path.join(self._dir.name, "no", "such", "dir", "faults.db"))
        self.assertFalse(faults.is_persistent())
        faults.record(bid="openai:a", backend="a", source="call", kind="timeout")
        self.assertEqual(len(faults.events_since(0)), 1)


# ── recording points in main ──────────────────────────────────────────────────

class _Adapter:
    def __init__(self, fail=None, resp=None):
        self.fail, self.resp = fail, resp

    async def discover(self, client):
        if self.fail:
            raise self.fail
        return adapters.Capabilities(models={"m"}, pricing={})

    async def dispatch(self, req):
        if self.fail:
            raise self.fail
        return self.resp


class _MainState(unittest.TestCase):
    KEYS = ("backends", "backend_adapters", "backend_models", "backend_pricing", "backend_loras",
            "backend_context", "backend_healthy", "backend_error", "backend_model_counts",
            "backend_hosts", "hosts_meta")

    def setUp(self):
        _reset_log()
        self._saved = {k: getattr(main, k) for k in self.KEYS}
        for k in self.KEYS:
            setattr(main, k, [] if k == "backends" else {})

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(main, k, v)
        main.rebuild_route_index()
        _reset_log()

    def _poll(self, backend, adapter):
        main.backends = [backend]
        main.backend_adapters = {main.backend_id(backend): adapter}
        main.rebuild_route_index()
        asyncio.run(main.refresh_backend(backend, None))


EVO = {"name": "comfyui-strix", "type": "comfyui", "url": "http://192.168.8.228:8188", "enabled": True}


class HealthTransitions(_MainState):
    def test_down_is_recorded_once_per_outage_not_per_poll(self):
        bad = _Adapter(fail=httpx.ConnectError("All connection attempts failed"))
        for _ in range(3):
            self._poll(EVO, bad)
        evs = faults.events_since(0)
        self.assertEqual([(e["source"], e["kind"]) for e in evs], [("health", "unreachable")])
        self.assertEqual(evs[0]["detail"], "All connection attempts failed")

    def test_an_empty_timeout_message_still_says_what_happened(self):
        self._poll(EVO, _Adapter(fail=httpx.ReadTimeout("")))
        self.assertEqual(faults.events_since(0)[0]["detail"], "ReadTimeout")

    def test_recovery_closes_the_outage_with_its_length(self):
        self._poll(EVO, _Adapter(fail=httpx.ConnectError("refused")))
        bid = main.backend_id(EVO)
        main.backend_error[bid]["since"] -= 90                    # it has been down 90 s
        self._poll(EVO, _Adapter())
        evs = faults.events_since(0)
        self.assertEqual(evs[-1]["kind"], faults.RECOVERED)
        self.assertGreaterEqual(evs[-1]["dur_s"], 90)
        self.assertNotIn(bid, main.backend_error)

    def test_a_healthy_boot_records_nothing(self):
        self._poll(EVO, _Adapter())
        self.assertEqual(faults.events_since(0), [])


class DispatchFailover(_MainState):
    def _dispatch(self, cands):
        main.backends = [b for b, _ in cands]
        request = types.SimpleNamespace(state=types.SimpleNamespace(), headers={}, client=None)
        return asyncio.run(main._dispatch_over([(b, "m") for b, _ in cands], "/v1/chat/completions",
                                               "tool", {"model": "tool"}, request))

    def test_a_failover_that_ends_in_200_is_still_recorded(self):
        from fastapi.responses import Response
        a = {"name": "llamaswap-strix", "type": "openai", "url": "http://192.168.8.31:8080"}
        b = {"name": "dx10-01", "type": "openai", "url": "http://192.168.8.35:8080"}
        main.backend_adapters = {main.backend_id(a): _Adapter(fail=httpx.ConnectError("refused")),
                                 main.backend_id(b): _Adapter(resp=Response(b"{}", status_code=200))}
        resp = self._dispatch([(a, None), (b, None)])
        self.assertEqual(resp.status_code, 200)
        evs = faults.events_since(0)
        self.assertEqual([(e["backend"], e["source"], e["kind"]) for e in evs],
                         [("llamaswap-strix", "call", "unreachable")])

    def test_a_5xx_passed_to_the_client_keeps_the_backend_text(self):
        from fastapi.responses import Response
        a = {"name": "llamaswap-strix", "type": "openai", "url": "http://192.168.8.31:8080"}
        main.backend_adapters = {main.backend_id(a): _Adapter(
            resp=Response(b'{"error":"model crashed"}', status_code=500))}
        self.assertEqual(self._dispatch([(a, None)]).status_code, 500)
        ev = faults.events_since(0)[0]
        self.assertEqual((ev["kind"], ev["status"]), ("upstream", 500))
        self.assertIn("model crashed", ev["detail"])


class FaultsInfo(_MainState):
    def test_host_label_open_outage_and_totals(self):
        main.backends = [EVO]
        bid = main.backend_id(EVO)
        main.backend_hosts = {bid: "192.168.8.228"}
        main.hosts_meta = {"192.168.8.228": {"label": "Evo-X2"}}
        main.backend_healthy = {bid: False}
        main.backend_error = {bid: {"kind": "unreachable", "status": None, "detail": "x",
                                    "since": int(main.time.time()) - 120}}
        faults.record(bid=bid, backend="comfyui-strix", type="comfyui", host="192.168.8.228",
                      source="job", kind="execution", detail="boom")
        info = main.faults_info()
        s = info["backends"][0]
        self.assertEqual((s["host_label"], s["faults"], info["total"]), ("Evo-X2", 1, 1))
        self.assertGreaterEqual(s["downtime_s"], 120)             # still down → counts to now
        self.assertEqual(info["bundles"][0]["host_label"], "Evo-X2")


# ── console ───────────────────────────────────────────────────────────────────

class Render(unittest.TestCase):
    INFO = {"total": 3, "persistent": True, "backends": [
        {"bid": "comfyui:comfyui-strix", "backend": "comfyui-strix", "type": "comfyui",
         "host": "192.168.8.228", "host_label": "Evo-X2", "enabled": True, "healthy": True,
         "error": None, "faults": 3, "outages": 2, "downtime_s": 95, "last_ts": 1,
         "last_kind": "unreachable", "last_detail": "All connection attempts failed",
         "last_source": "health"},
        {"bid": "openai:dx10-01", "backend": "dx10-01", "type": "openai", "host": "h",
         "host_label": "", "enabled": True, "healthy": True, "error": None, "faults": 0,
         "outages": 0, "downtime_s": 0, "last_ts": None, "last_kind": None, "last_detail": "",
         "last_source": None}],
        "bundles": [{"bid": "comfyui:comfyui-strix", "backend": "comfyui-strix", "type": "comfyui",
                     "host": "192.168.8.228", "host_label": "Evo-X2", "source": "job",
                     "kind": "unreachable", "status": None, "count": 2, "first": 1, "last": 2,
                     "message": "ComfyUI unreachable for >15s during execution"}]}

    def test_dashboard_panel_names_the_failed_backend_even_when_it_is_up_again(self):
        html = admin._dash_faults(self.INFO)
        self.assertIn("comfyui-strix", html)
        self.assertIn("Evo-X2", html)
        self.assertIn("up now", html)
        self.assertNotIn("dx10-01", html)                         # no faults → no row
        self.assertIn("data-sk='dash-faults'", html)

    def test_dashboard_empty_state(self):
        html = admin._dash_faults({"total": 0, "backends": [], "bundles": []})
        self.assertIn("no backend failures", html)

    def test_dashboard_card_counts_faults(self):
        html = admin._dash_cards({}, [], self.INFO)
        self.assertIn("backend faults · 24h", html)
        self.assertIn(">3<", html)

    def test_statistic_panel_lists_every_bundled_message(self):
        html = admin._faults_panel(self.INFO)
        self.assertIn("id='faults'", html)
        self.assertIn("ComfyUI unreachable for &gt;15s during execution", html)
        self.assertIn("<b>2</b>", html)

    def test_statistic_panel_warns_when_memory_only(self):
        html = admin._faults_panel(dict(self.INFO, persistent=False))
        self.assertIn("in memory only", html)

    def test_dashboard_backends_table_carries_the_fault_column_and_down_reason(self):
        be = [{"name": "comfyui-strix", "type": "comfyui", "enabled": True, "healthy": False,
               "error": {"kind": "unreachable", "detail": "refused", "since": 1}}]
        fmap = {s["bid"]: s for s in self.INFO["backends"]}
        html = admin._dash_backends(be, [], fmap)
        self.assertIn("faults · 24h", html)
        self.assertIn("unreachable", html)                        # not just "off"
        self.assertIn("/ui/statistic#faults", html)


if __name__ == "__main__":
    unittest.main()
