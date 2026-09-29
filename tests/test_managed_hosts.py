"""Managed hosts in the store and in main: entries, attached backends, lookups via `host`.

Why this file exists (every case fails SILENTLY):
- A provider token written through `set_settings` as part of a dict is stored in
  PLAINTEXT (only string values of secret keys are encrypted) — and every reader of
  `get_settings()` would see it. Nothing errors; the token just sits in the DB (R-W10).
- The controller set follows the store: a host whose entry is gone while its instance
  runs must keep its controller (else the instance bills with nobody to stop it), and a
  host with an unknown provider must be SHOWN but never driven — and must not take the
  whole backend rebuild down with it (Ruling M4).
- A config-defined backend naming a managed host is NOT attached (R-K3): its store copy
  would be written on every start/stop and override the config wholesale.
- `local_port` is the gateway's end of the tunnel forward and the backend's URL: a port
  that changes on a Save or a rename moves the backend's URL under running jobs, and two
  services on one port make sshrun refuse the WHOLE tunnel.
- The routing gate and the health view resolve backend → host → controller → service
  (R-W7): a lookup by backend name finds nothing and routes a job onto a box whose
  models are still downloading.
- `bootstrap_template` "auto" (Ruling M5): a fixed `comfy-ui` default gave every
  vLLM-only host the template's ComfyUI and its bundled models.

Run: /home/dev/projekte/ai-hub/venv/bin/python -m unittest tests.test_managed_hosts -v
"""
import asyncio
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    import store
    import hostctl
    import thunder
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

from fastapi.testclient import TestClient  # noqa: E402
from tests.fakes import FakeThunder  # noqa: E402
from tests import test_hostctl as th  # noqa: E402

TOKEN = "thunder-SECRET-token-123"


def _host(provider="thunder", **opts):
    return {"provider": provider, "options": dict({"gpu_type": "a6000"}, **opts),
            "api_key": TOKEN}


class _StoreCase(unittest.TestCase):
    """A fresh store per test and main's host globals restored afterwards."""

    NAMES = ("backends", "config_backends", "host_controllers", "_host_tasks",
             "_hosts_booted", "jobs_cfg", "managed_hosts", "_host_errors",
             "_host_not_attachable", "_host_attached", "_not_attach_warned",
             "_host_warned", "backend_hosts", "host_backends", "hosts_meta")

    def setUp(self):
        m = main
        self._saved = {n: getattr(m, n) for n in self.NAMES}
        for n in ("_not_attach_warned", "_host_warned"):
            setattr(m, n, set(getattr(m, n)))
        self._saved_store = (store._DB_PATH, store._active, store._MASTER_KEY)
        self.tmp = tempfile.mkdtemp(prefix="managed-hosts-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        store._MASTER_KEY = os.urandom(32)
        store.init(os.path.join(self.tmp, "store.db"))
        m.host_controllers = {}
        m._host_tasks = {}
        m._hosts_booted = False
        m.managed_hosts = {}
        m._host_errors = {}
        m._host_not_attachable = {}
        m._host_attached = {}
        m.config_backends = []
        m.backends = []
        m.jobs_cfg = dict(m.jobs_cfg, store_path=os.path.join(self.tmp, "store.db"))

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(main, n, v)
        store._DB_PATH, store._active, store._MASTER_KEY = self._saved_store

    @staticmethod
    def comfy(name="comfy", host="vm1", **kw):
        return dict({"name": name, "type": "comfyui", "url": "http://placeholder",
                     "host": host}, **kw)

    @staticmethod
    def llm(name="vllm", host="vm1", **kw):
        return dict({"name": name, "type": "openai", "url": "http://placeholder",
                     "host": host, "svc_start": "serve --port 8000"}, **kw)

    def live(self, name, btype="comfyui"):
        return next(b for b in main.backends
                    if b["name"] == name and b.get("type", "openai") == btype)


class StoreEncryption(_StoreCase):
    def test_token_encrypted_at_rest_and_decrypted_on_read(self):
        store.set_managed_host("vm1", _host())
        with sqlite3.connect(store._DB_PATH) as c:
            raw = c.execute("SELECT value_json FROM settings WHERE key='managed_hosts'"
                            ).fetchone()[0]
        self.assertNotIn(TOKEN, raw)
        self.assertTrue(json.loads(raw)["vm1"]["api_key"].startswith("enc:"))
        # every reader of ALL settings (Server tab, startup overlay) sees ciphertext only
        self.assertNotIn(TOKEN, json.dumps(store.get_settings()))
        self.assertNotIn(TOKEN, json.dumps(store.get_setting("managed_hosts")))
        got = store.get_managed_hosts()
        self.assertEqual(got["vm1"]["api_key"], TOKEN)
        self.assertEqual(got["vm1"]["options"], {"gpu_type": "a6000"})
        # a copy: a caller changing it writes nothing back
        got["vm1"]["api_key"] = "x"
        self.assertEqual(store.get_managed_hosts()["vm1"]["api_key"], TOKEN)

    def test_upsert_blank_token_and_delete(self):
        store.set_managed_host("vm1", _host())
        store.set_managed_host("vm2", dict(_host(), api_key=""))
        self.assertEqual(store.get_managed_hosts()["vm2"]["api_key"], "")
        store.set_managed_host("vm1", None)
        self.assertEqual(list(store.get_managed_hosts()), ["vm2"])
        store.set_managed_host("gone", None)                 # deleting nothing is fine
        self.assertEqual(list(store.get_managed_hosts()), ["vm2"])

    def test_unreadable_setting_reads_empty(self):
        store.set_settings({"managed_hosts": ["garbage"]})
        with self.assertLogs("store", "WARNING"):
            self.assertEqual(store.get_managed_hosts(), {})

    def test_view_and_health_never_carry_the_token(self):
        store.set_managed_host("vm1", _host())
        main.sync_host_controllers()
        v = main.host_view("vm1")
        self.assertTrue(v["api_key_set"])
        self.assertNotIn(TOKEN, json.dumps(v, default=str))
        h = asyncio.run(main.health(verbose=True))
        self.assertNotIn(TOKEN, json.dumps(h, default=str))
        self.assertNotIn(TOKEN, json.dumps(main.gateway_info(), default=str))


class ControllerSync(_StoreCase):
    def test_create_keep_and_attach(self):
        store.set_managed_host("vm1", _host())
        store.upsert_backend(self.comfy())
        store.upsert_backend(self.llm())
        store.upsert_backend({"name": "far", "type": "openai", "url": "http://10.0.0.9:8000"})
        main.rebuild_backends()
        c = main.host_controllers["vm1"]
        self.assertIsInstance(c, hostctl.Controller)
        self.assertEqual((c.name, c.kind, c.host["api_key"]), ("vm1", "thunder", TOKEN))
        self.assertEqual(sorted(c.service_bids()), ["comfyui:comfy", "openai:vllm"])
        comfy, vllm = self.live("comfy"), self.live("vllm", "openai")
        # the services ARE the live backend dicts (the controller reads the current ones)
        self.assertTrue(any(x is comfy for x in c.services))
        # remote port from the profile, local port assigned, URL derived — all persisted
        self.assertEqual((comfy["remote_port"], vllm["remote_port"]), (8188, 8000))
        for b in (comfy, vllm):
            self.assertTrue(18100 <= b["local_port"] <= 18999)
            self.assertEqual(b["url"], f"http://127.0.0.1:{b['local_port']}")
            row = store.get_backend(b["name"], b["type"])
            self.assertEqual((row["local_port"], row["remote_port"], row["url"]),
                             (b["local_port"], b["remote_port"], b["url"]))
        self.assertNotEqual(comfy["local_port"], vllm["local_port"])
        # a rebuild keeps the INSTANCE and hands it the new dicts
        main.rebuild_backends()
        self.assertIs(main.host_controllers["vm1"], c)
        self.assertTrue(any(x is self.live("comfy") for x in c.services))
        # a new token reaches the controller
        store.set_managed_host("vm1", dict(_host(), api_key="new-tok"))
        main.sync_host_controllers()
        self.assertEqual(c.host["api_key"], "new-tok")

    def test_host_without_backends_and_backend_on_no_managed_host(self):
        store.set_managed_host("vm1", _host())
        store.upsert_backend(self.comfy(host="k12"))           # a plain box, not managed
        main.rebuild_backends()
        self.assertEqual(main.host_controllers["vm1"].services, [])
        self.assertNotIn("local_port", self.live("comfy"))
        self.assertEqual(self.live("comfy")["url"], "http://placeholder")

    def test_remove_only_when_off_and_idle(self):
        store.set_managed_host("vm1", _host())
        main.sync_host_controllers()
        c = main.host_controllers["vm1"]
        store.set_managed_host("vm1", None)
        c.state.phase = "ready"
        with self.assertLogs("main", "WARNING") as cm:
            main.sync_host_controllers()
            main.sync_host_controllers()
        self.assertIs(main.host_controllers["vm1"], c)
        self.assertEqual(len([x for x in cm.output if "entry removed" in x]), 1)
        self.assertIn("vm1", main.host_names())               # still shown
        c.state.phase = "off"
        c._op = "starting"                                     # a start is off until created
        main.sync_host_controllers()                           # (warned once already)
        self.assertIs(main.host_controllers["vm1"], c)
        c._op = None
        c.state.pending_snapshot = "s7"                        # the watcher still rotates
        main._host_warned.discard("vm1")
        with self.assertLogs("main", "WARNING") as cm:
            main.sync_host_controllers()
        self.assertIn("s7", "\n".join(cm.output))
        self.assertIs(main.host_controllers["vm1"], c)
        c.state.pending_snapshot = ""
        main.sync_host_controllers()
        self.assertEqual(main.host_controllers, {})
        self.assertNotIn("vm1", main.host_names())

    def test_no_thunder_block_is_read(self):
        # the shim is gone: a comfyui backend with the old block drives nothing
        store.upsert_backend(self.comfy(host="", thunder={"gpu_type": "a6000"}))
        main.rebuild_backends()
        self.assertEqual(main.host_controllers, {})
        self.assertFalse(hasattr(main, "_shim_ctl"))

    def test_booted_gateway_starts_a_new_controller(self):
        runs = []
        self.addCleanup(setattr, main, "_host_run", main._host_run)
        self.addCleanup(setattr, main, "_modelsrc_prepare", main._modelsrc_prepare)
        main._host_run = lambda n, c: runs.append((n, c))
        main._modelsrc_prepare = lambda: runs.append("modelsrc")
        store.set_managed_host("vm1", _host())
        main.sync_host_controllers()
        self.assertEqual(runs, [])                  # before boot: the lifespan starts it
        main._hosts_booted = True
        store.set_managed_host("vm2", _host())
        main.sync_host_controllers()
        self.assertEqual(runs, [("vm2", main.host_controllers["vm2"]), "modelsrc"])


class ConfigBackendNotAttachable(_StoreCase):
    def test_config_backend_naming_a_managed_host_is_not_attached(self):
        store.set_managed_host("vm1", _host())
        main.config_backends = [self.comfy("cb")]
        store.upsert_backend(self.llm())
        with self.assertLogs("main", "WARNING") as cm:
            main.rebuild_backends()
            main.rebuild_backends()
        self.assertEqual(len([x for x in cm.output if "not attached" in x]), 1)
        c = main.host_controllers["vm1"]
        self.assertEqual(c.service_bids(), ["openai:vllm"])
        na = main.host_view("vm1")["not_attachable"]
        self.assertEqual([x["bid"] for x in na], ["comfyui:cb"])
        self.assertIn("config-defined backend", na[0]["reason"])
        # nothing written for it: its URL and ports stay what config says
        self.assertEqual(self.live("cb")["url"], "http://placeholder")
        self.assertIsNone(store.get_backend("cb", "comfyui"))
        # and it is never gated as if it ran there
        self.assertIsNone(main.modelsync_gate(self.live("cb"), "img"))


class LocalPort(_StoreCase):
    def test_stable_over_saves_and_renames(self):
        store.set_managed_host("vm1", _host())
        store.upsert_backend(self.comfy())
        store.upsert_backend(self.llm())
        main.rebuild_backends()
        port = self.live("vllm", "openai")["local_port"]
        self.assertEqual(main.assign_local_port("vllm", "openai"), port)
        # a Save goes from the stored row (admin backend_save) — the port rides along
        row = store.get_backend("vllm", "openai")
        row["svc_start"] = "serve --port 8000 --other"
        store.upsert_backend(row)
        main.rebuild_backends()
        self.assertEqual(self.live("vllm", "openai")["local_port"], port)
        # a rename: the new row is the old one under a new name
        store.delete_backend("vllm", "openai")
        store.upsert_backend(dict(row, name="vllm2"))
        main.rebuild_backends()
        self.assertEqual(self.live("vllm2", "openai")["local_port"], port)
        self.assertEqual(main.assign_local_port("vllm2", "openai"), port)

    def test_unique_over_all_hosts_and_collisions_reassigned(self):
        store.set_managed_host("vm1", _host())
        store.set_managed_host("vm2", _host())
        store.upsert_backend(self.comfy("a", host="vm1"))
        store.upsert_backend(self.llm("b", host="vm1"))
        store.upsert_backend(self.comfy("c", host="vm2", local_port=18100))
        store.upsert_backend(self.llm("d", host="vm2", local_port=18100))   # collides
        store.upsert_backend(self.llm("e", host="vm2", local_port=80))      # out of range
        main.rebuild_backends()
        ports = [self.live(n, t)["local_port"] for n, t in
                 (("a", "comfyui"), ("b", "openai"), ("c", "comfyui"), ("d", "openai"),
                  ("e", "openai"))]
        self.assertEqual(len(set(ports)), 5, ports)
        self.assertTrue(all(18100 <= p <= 18999 for p in ports), ports)
        self.assertIn(18100, ports)                         # one of the two keeps it
        # stable on the next rebuild
        main.rebuild_backends()
        self.assertEqual(ports, [self.live(n, t)["local_port"] for n, t in
                                 (("a", "comfyui"), ("b", "openai"), ("c", "comfyui"),
                                  ("d", "openai"), ("e", "openai"))])

    def test_a_retained_controllers_forward_stays_taken(self):
        # a host deleted while running keeps its controller and its forwards: a port
        # it still forwards must not be handed to another backend
        store.set_managed_host("vm1", _host())
        store.set_managed_host("vm2", _host())
        store.upsert_backend(self.comfy("a", host="vm1"))
        main.rebuild_backends()
        pa = self.live("a")["local_port"]
        c1 = main.host_controllers["vm1"]
        c1.state.phase = "ready"
        store.delete_backend("a", "comfyui")
        store.set_managed_host("vm1", None)
        with self.assertLogs("main", "WARNING"):
            main.rebuild_backends()
        self.assertIs(main.host_controllers["vm1"], c1)
        self.assertEqual(c1.services[0]["local_port"], pa)
        store.upsert_backend(self.llm("b", host="vm2"))
        main.rebuild_backends()
        self.assertNotEqual(self.live("b", "openai")["local_port"], pa)

    def test_exhausted_range_raises(self):
        saved = main.LOCAL_PORT_MAX
        self.addCleanup(setattr, main, "LOCAL_PORT_MAX", saved)
        main.LOCAL_PORT_MAX = main.LOCAL_PORT_MIN
        store.upsert_backend(self.llm("x", local_port=main.LOCAL_PORT_MIN))
        with self.assertRaises(RuntimeError):
            main.assign_local_port("y", "openai")


class _FakeCtl:
    """A controller's routing and action face: services, readiness, recorded ops."""

    def __init__(self, name, services, ready=(), phase="ready"):
        self.name = name
        self.kind = "thunder"
        self.host = {"name": name, "provider": "thunder", "options": {}, "api_key": ""}
        self.services = list(services)
        self.ready = set(ready)
        self.state = hostctl.State(phase=phase)
        self.op = None
        self.calls = []

    def service_bids(self):
        return [f"{x.get('type', 'openai')}:{x['name']}" for x in self.services]

    def has_service(self, bid):
        return bid in self.service_bids()

    def is_alias_ready(self, bid, alias):
        return self.has_service(bid) and alias in self.ready

    def alias_status(self, bid, alias):
        return f"models for {alias} are syncing on {bid}"

    def set_services(self, svcs):
        self.services = list(svcs)

    def view(self):
        return {"name": self.name, "provider": "thunder", "phase": self.state.phase,
                "uptime_s": 7, "cost_per_h": 0.5, "error": "",
                "services": {b: {"status": "up"} for b in self.service_bids()},
                "plan": {"aliases": {"solo": {"ready": False}}}}

    async def _rec(self, *a):
        self.calls.append(a)
        await asyncio.sleep(0)

    async def start(self):
        if self.state.phase != "off":               # refused before the first await
            raise RuntimeError(f"already {self.state.phase}")
        await self._rec("start")

    def stop(self):
        return self._rec("stop")

    def restart_service(self, bid):
        return self._rec("restart_service", bid)

    def resetup(self, bid):
        return self._rec("resetup", bid)

    def sync_now(self):
        return self._rec("sync_now")

    def forget_unreconciled(self):
        self.calls.append(("forget",))

    async def delete_unknown(self, paths):
        self.calls.append(("delete_unknown", list(paths)))
        return len(paths)


class LookupsViaHost(_StoreCase):
    def setUp(self):
        super().setUp()
        self.b = self.comfy("tc")
        main.backends = [self.b, self.comfy("tc2", host=""), self.llm("tc", host="vm1")]
        self.ctl = main.host_controllers["vm1"] = _FakeCtl("vm1", [self.b])

    def test_gate_resolves_backend_host_controller(self):
        self.assertEqual(main.modelsync_gate(self.b, "img"),
                         "models for img are syncing on comfyui:tc")
        self.ctl.ready.add("img")
        self.assertIsNone(main.modelsync_gate(self.b, "img"))
        # no host, another host, the same name as a non-ComfyUI service: never gated
        self.assertIsNone(main.modelsync_gate(self.comfy("tc", host=""), "x"))
        self.assertIsNone(main.modelsync_gate(self.comfy("tc", host="vm9"), "x"))
        self.assertIsNone(main.modelsync_gate(self.llm("tc"), "x"))
        # the host names a controller that does not carry this backend (not attached)
        self.assertIsNone(main.modelsync_gate(self.comfy("other"), "x"))

    def test_gated_only_aliases_and_chain_successor(self):
        store.upsert("solo", [{"backend": "tc", "workflow_json": {}}])
        store.upsert("mixed", [{"backend": "tc", "workflow_json": {}},
                               {"backend": "tc2", "workflow_json": {}}])
        self.assertEqual(main._gated_only_aliases(["solo", "mixed"]), {"solo"})
        rows = main.host_view("vm1")["plan"]["aliases"]
        self.assertTrue(rows["solo"]["gated_only"])
        s2, why = main._chain_successor_on(self.b, "solo")
        self.assertIsNone(s2)
        self.assertIn("syncing", why)

    def test_host_coordination_groups_attached_backends(self):
        store.set_managed_host("vm1", _host())
        store.upsert_backend(self.comfy())
        store.upsert_backend(self.llm())
        main.host_controllers = {}
        main.rebuild_backends()
        self.assertEqual(main.backend_hosts["comfyui:comfy"], "vm1")
        self.assertEqual(main.backend_hosts["openai:vllm"], "vm1")
        self.assertEqual(sorted(main.host_backends["vm1"]), ["comfyui:comfy", "openai:vllm"])
        # the attached backends' URLs point at 127.0.0.1 — the grouping must not follow it
        self.assertNotIn("127.0.0.1", main.host_backends)


class HostActions(_StoreCase):
    def setUp(self):
        super().setUp()
        self.ctl = main.host_controllers["vm1"] = _FakeCtl(
            "vm1", [self.comfy("tc"), self.llm("v")], phase="off")
        main.managed_hosts = {"vm1": _host()}

    def run_(self, *a, **k):
        async def go():
            msg = await main.host_action(*a, **k)
            for ts in main._host_tasks.values():
                await asyncio.gather(*ts, return_exceptions=True)
            return msg
        return asyncio.run(go())

    def test_each_action_reaches_the_controller(self):
        self.assertIn("start", self.run_("vm1", "start"))
        self.ctl.state.phase = "ready"
        self.assertIn("refused: already ready", self.run_("vm1", "start"))
        self.run_("vm1", "stop")
        self.run_("vm1", "restart_service", bid="openai:v")
        self.run_("vm1", "resetup", bid="comfyui:tc")
        self.run_("vm1", "sync")
        self.run_("vm1", "sync_now")
        self.assertIn("forgotten", self.run_("vm1", "forget_unreconciled"))
        self.assertIn("deleted 2 unknown files",
                      self.run_("vm1", "delete_unknown", paths=["models/a", "models/b"]))
        self.assertEqual(self.ctl.calls, [
            ("start",), ("stop",), ("restart_service", "openai:v"),
            ("resetup", "comfyui:tc"), ("sync_now",), ("sync_now",), ("forget",),
            ("delete_unknown", ["models/a", "models/b"])])

    def test_refusals_as_text(self):
        self.assertIn("unknown managed host", self.run_("nope", "start"))
        self.assertIn("unknown action", self.run_("vm1", "explode"))
        self.assertIn("no backend", self.run_("vm1", "restart_service"))
        self.assertIn("nothing deleted", self.run_("vm1", "delete_unknown", paths=[]))
        self.assertEqual(self.ctl.calls, [])

    def test_thunder_card_wrappers_until_task7(self):
        # the Thunder card still posts (name, action) with name = the HOST; its
        # "restart" is the ComfyUI service's restart
        self.ctl.state.phase = "ready"
        asyncio.run(main.thunder_action("vm1", "restart"))
        self.assertEqual(self.ctl.calls, [("restart_service", "comfyui:tc")])
        self.assertEqual(main.host_names(), ["vm1"])
        import admin
        self.assertIs(admin._thunder_names, main.host_names)
        self.assertIs(admin._thunder_view, main.host_view)
        self.assertIs(admin._thunder_longrun, main.host_longrun)
        self.assertIs(admin._thunder_action, main.thunder_action)


class UnknownProvider(_StoreCase):
    def test_shown_not_driven_and_rebuild_survives(self):
        store.set_managed_host("vm1", _host())
        store.set_managed_host("pod", _host(provider="runpod"))
        store.upsert_backend(self.llm("v", host="pod"))
        with self.assertLogs("main", "WARNING") as cm:
            main.rebuild_backends()                          # never raises
            main.rebuild_backends()
        self.assertEqual(len([x for x in cm.output if "unknown provider" in x]), 1)
        self.assertEqual(sorted(main.host_controllers), ["vm1"])
        self.assertEqual(main.host_names(), ["pod", "vm1"])
        v = main.host_view("pod")
        self.assertIn("unknown provider 'runpod'", v["error"])
        self.assertEqual(v["phase"], "off")
        self.assertEqual(list(v["services"]), ["openai:v"])
        self.assertIn("not driven", asyncio.run(main.host_action("pod", "start")))
        h = asyncio.run(main.health(verbose=False))["hosts_managed"]
        self.assertIn("unknown provider", h["pod"]["error"])
        # the Backends tab still renders with it
        import admin
        admin._thunder_panel(admin._thunder_views(), [])


    def test_running_host_whose_entry_turns_unreadable_keeps_its_controller(self):
        # a hand-edited provider on a host with a live instance: never driven by the
        # new entry, but the controller (its instance bills) stays and keeps its list
        store.set_managed_host("vm1", _host())
        store.upsert_backend(self.comfy())
        main.rebuild_backends()
        c = main.host_controllers["vm1"]
        c.state.phase = "ready"
        store.set_managed_host("vm1", _host(provider="runpod"))
        store.upsert_backend(self.llm())
        with self.assertLogs("main", "WARNING"):
            main.rebuild_backends()
        self.assertIs(main.host_controllers["vm1"], c)
        self.assertEqual(c.host["provider"], "thunder")
        self.assertEqual(sorted(c.service_bids()), ["comfyui:comfy", "openai:vllm"])
        self.assertIn("unknown provider", main.host_view("vm1")["host_error"])
        # fixed again: the error goes, and a later relapse is warned again
        store.set_managed_host("vm1", _host())
        main.rebuild_backends()
        self.assertEqual(main.host_view("vm1")["host_error"], "")
        store.set_managed_host("vm1", _host(provider="runpod"))
        with self.assertLogs("main", "WARNING"):
            main.rebuild_backends()


class Health(_StoreCase):
    def test_full_view_carries_hosts_managed(self):
        store.set_managed_host("vm1", _host())
        store.upsert_backend(self.comfy())
        store.upsert_backend(self.llm())
        main.rebuild_backends()
        h = asyncio.run(main.health(verbose=False))
        hm = h["hosts_managed"]["vm1"]
        self.assertEqual((hm["provider"], hm["phase"], hm["uptime_s"]), ("thunder", "off", 0))
        self.assertIn("cost_per_h", hm)
        self.assertEqual(hm["services"], {"comfyui:comfy": "down", "openai:vllm": "down"})
        # the per-backend Thunder block is gone
        for b in h["backends"].values():
            self.assertNotIn("thunder", b)

    def test_stranger_short_form_unchanged(self):
        store.set_managed_host("vm1", _host())
        main.sync_host_controllers()
        saved = (main.api_key, main.users, main._users_by_key)
        self.addCleanup(lambda: (setattr(main, "api_key", saved[0]),
                                 setattr(main, "users", saved[1]),
                                 setattr(main, "_users_by_key", saved[2])))
        main.api_key = "master"
        d = TestClient(main.app).get("/health").json()
        self.assertEqual(set(d), {"status", "backends_total", "backends_healthy"})


class Refusals(_StoreCase):
    def test_name_rule(self):
        for bad in ("", "VM1", "a_b", "-a", "a-", "vm 1", "x" * 41, "ä"):
            self.assertIsNotNone(main.managed_host_refusal(bad, _host()), bad)
        self.assertIsNone(main.managed_host_refusal("gpu-vm-1", _host()))

    def test_collisions(self):
        store.set_managed_host("vm1", _host())
        store.set_host("k12-box", {"label": "K12"})
        main.backends = [{"name": "k", "type": "comfyui", "url": "http://k12:8188"},
                         {"name": "e", "type": "openai", "url": "http://10.0.0.5:8000",
                          "host": "evo"}]
        main.sync_host_controllers()
        self.assertIn("already exists", main.managed_host_refusal("vm1", _host()))
        self.assertIn("Hosts", main.managed_host_refusal("k12-box", _host()))
        # R-W6: a URL hostname without a dot is a host name too
        self.assertIn("k12", main.managed_host_refusal("k12", _host()))
        self.assertIn("evo", main.managed_host_refusal("evo", _host()))
        # an update of the existing host is no collision with itself
        self.assertIsNone(main.managed_host_refusal("vm1", _host(), new=False))
        self.assertIn("unknown", main.managed_host_refusal("vm9", _host(), new=False))

    def test_provider_and_options(self):
        self.assertIn("unknown provider", main.managed_host_refusal("a", _host("runpod")))
        self.assertIn("vcpus", main.managed_host_refusal("a", _host(vcpus=0)))
        self.assertIsNone(main.managed_host_refusal("a", _host(bootstrap_template="")))
        # the provider of an existing host never changes (snapshots, state are its)
        store.set_managed_host("vm1", _host())
        main.sync_host_controllers()
        self.assertIn("provider", main.managed_host_refusal(
            "vm1", dict(_host(), provider="other"), new=False))

    def test_save_helper(self):
        self.assertEqual(main.save_managed_host("vm1", _host(), new=True), "")
        self.assertIn("vm1", main.host_controllers)
        self.assertIn("already exists", main.save_managed_host("vm1", _host(), new=True))
        self.assertEqual(main.save_managed_host("vm1", _host(vcpus=16), new=False), "")
        self.assertEqual(main.host_controllers["vm1"].cfg["vcpus"], 16)


class DeleteHost(_StoreCase):
    def test_only_off_without_pending_snapshot(self):
        store.set_managed_host("vm1", _host())
        store.set_host("vm1", {"label": "cloud GPU"})
        store.upsert_backend(self.comfy())
        main.rebuild_backends()
        self.assertIn("comfyui:comfy", main.delete_managed_host("vm1"))
        store.delete_backend("comfy", "comfyui")
        main.rebuild_backends()
        c = main.host_controllers["vm1"]
        main._host_save_state("vm1", {"phase": "off", "snapshot_id": "s1"})
        main._host_save_state("vm2", {"phase": "off"})
        c.state.phase = "ready"
        self.assertIn("stop it first", main.delete_managed_host("vm1"))
        c.state.phase = "off"
        c._op = "starting"
        self.assertIn("busy", main.delete_managed_host("vm1"))
        c._op = None
        c.state.pending_snapshot = "s9"
        self.assertIn("s9", main.delete_managed_host("vm1"))
        self.assertIn("vm1", store.get_managed_hosts())
        c.state.pending_snapshot = ""
        self.assertEqual(main.delete_managed_host("vm1"), "")
        self.assertEqual(store.get_managed_hosts(), {})
        self.assertEqual(main.host_controllers, {})
        # its state goes too: a new host of that name must not adopt the old snapshot
        self.assertEqual(store.get_setting("host_state"), {"vm2": {"phase": "off"}})
        self.assertNotIn("vm1", store.get_hosts())
        self.assertIsNone(main.managed_host_refusal("vm1", _host()))   # name free again
        self.assertIn("unknown managed host", main.delete_managed_host("vm1"))


class AutoTemplate(unittest.IsolatedAsyncioTestCase):
    """Ruling M5: "auto" is the stored blank and the default."""

    def test_options_of_auto(self):
        f = {x["key"]: x for x in thunder.OPTION_FIELDS}["bootstrap_template"]
        self.assertEqual(f["default"], "")
        self.assertIn("", f["choices"])
        self.assertEqual(f["choice_labels"][""], "auto")
        self.assertEqual(thunder.options_of({})[0]["bootstrap_template"], "")
        for v in ("auto", "", None):
            o, errs, _ = thunder.options_of({"opt__bootstrap_template": v})
            self.assertEqual((o["bootstrap_template"], errs), ("", []), v)
        self.assertEqual(thunder.options_of({"opt__bootstrap_template": "base"})[0]
                         ["bootstrap_template"], "base")

    async def test_auto_picks_by_attached_services(self):
        auto = thunder.options_of({"opt__bootstrap_template": "auto"})[0]["bootstrap_template"]
        fake = FakeThunder()
        c, _, _, _ = th.make(fake)
        c.cfg["bootstrap_template"] = auto
        await c.start()
        self.assertEqual(th._creates(fake)[0]["template"], "comfy-ui")
        fake = FakeThunder()
        c, _, _, _ = th.make(fake, services=[th._svc("vllm", "openai", 18200, 8000)])
        c.cfg["bootstrap_template"] = auto
        await c.start()
        self.assertEqual(th._creates(fake)[0]["template"], "base")


if __name__ == "__main__":
    unittest.main()
