"""Thunder lifecycle controller against a stubbed Thunder API and a fake ssh.
run: venv/bin/python -m unittest tests.test_thunder_controller -v"""
import asyncio
import json
import os
import shutil
import tempfile
import types
import unittest

import httpx

import sshrun
import thunder
import thunderctl


class FakeThunder:
    """Scripted Thunder REST API: instances, snapshots, port forwards."""
    def __init__(self):
        self.instances = {}          # index -> item
        self.snaps = []
        self.calls = []
        self.next_index = 0
        self.status_script = ["PROVISIONING", "RUNNING"]
        self.http_ports_on_create = []
        self.delete_by = "index"     # which id form /delete accepts
        self.ignore_port_remove = False   # a /ports PATCH that answers 200 and changes nothing

    def handler(self, req: httpx.Request) -> httpx.Response:
        p, m = req.url.path, req.method
        self.calls.append((m, p, json.loads(req.content) if req.content else None))
        if p == "/instances/list":
            for it in self.instances.values():
                if self.status_script:
                    it["status"] = self.status_script.pop(0)
            return httpx.Response(200, json=self.instances)
        if p == "/instances/create":
            idx = str(self.next_index); self.next_index += 1
            self.instances[idx] = {"uuid": f"u{idx}", "status": "PROVISIONING", "ip": "10.0.0.5",
                                   "port": 30022, "httpPorts": list(self.http_ports_on_create)}
            return httpx.Response(201, json={"identifier": int(idx), "uuid": f"u{idx}", "key": ""})
        if p.endswith("/delete"):
            ident = p.split("/")[2]
            key = ident if self.delete_by == "index" else next((k for k, v in self.instances.items() if v["uuid"] == ident), None)
            if key not in self.instances:
                return httpx.Response(404, json={"error": "not_found"})
            del self.instances[key]
            return httpx.Response(200, json={"message": "ok"})
        if p.endswith("/ports"):
            ident = p.split("/")[2]
            body = json.loads(req.content)
            if ident not in self.instances:           # uuid form tried first (Ruling 11)
                return httpx.Response(404, json={"error": "not_found"})
            it = self.instances[ident]
            if not self.ignore_port_remove:
                it["httpPorts"] = [x for x in it["httpPorts"] if x not in body.get("remove_ports", [])]
            return httpx.Response(200, json={})
        if p == "/snapshots/create":
            body = json.loads(req.content)
            sid = f"s{len(self.snaps)}"
            self.snaps.append({"id": sid, "name": body["name"], "status": "CREATING", "minimumDiskSizeGb": 120, "createdAt": len(self.snaps) + 1})
            return httpx.Response(202, json={"id": sid, "message": "ok"})
        if p == "/snapshots/list":
            return httpx.Response(200, json=self.snaps)
        if p.startswith("/snapshots/") and m == "DELETE":
            self.snaps = [s for s in self.snaps if s["id"] != p.split("/")[2]]
            return httpx.Response(200, json={})
        if p == "/v2/pricing":
            return httpx.Response(200, json={"pricing": {"a6000_x1": 0.35, "additional_vcpus": 0.04, "disk_gb": 0.0003, "snapshot_gb": 0.00006849}})
        if p == "/v2/specs":
            return httpx.Response(200, json={"specs": {"a6000_x1": {"vcpuOptions": [6, 8], "storageGB": {"min": 100, "max": 500}}}})
        return httpx.Response(404)


COMMIT = "1d61dcc35c35541388c0001bacc7703db14e8bea"
_TMPDIRS = []


def tearDownModule():
    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)


def make(fake, backend=None, ssh_script=None, datadir=None, default_nodes="",
         probe=None):
    """A controller on the stub API. `c.h` carries the harness: `clock` (a list; the
    fake sleep ADVANCES it, so every timeout is reachable without waiting), `phases`
    (every persisted phase in order), `faults`, `tunnels`. `ssh_script` maps a
    substring of the remote command to a result, or to a LIST of results consumed in
    order (the last one sticks); the bootstrap succeeds by default. Every ssh call is also appended to `fake.calls` as
    ("SSH", remote_cmd, stdin), so the order against API calls is checkable. The
    datadir is a fresh temp dir per controller unless given."""
    saved = {}
    enabled = {}
    ssh_calls = []
    clock = [1_790_000_000.0]
    phases, faults, tunnels = [], [], []
    if datadir is None:
        datadir = tempfile.mkdtemp(prefix="thunderctl-test-")
        _TMPDIRS.append(datadir)
    # a bootstrap that succeeds unless the test scripts otherwise
    script = {"bash -s": (0, b"GW:PHASE smoke\nGW:SMOKE ok\nGW:DONE\n", b"")}
    script.update(ssh_script or {})

    async def ssh(argv, stdin=None, timeout=60):
        ssh_calls.append((argv, stdin))
        fake.calls.append(("SSH", argv[-1], stdin))
        for pat, res in script.items():
            if pat in argv[-1]:
                if isinstance(res, list):
                    return res.pop(0) if len(res) > 1 else res[0]
                return res
        return (0, b"", b"")

    async def sleep(s):
        clock[0] += s
        await asyncio.sleep(0)

    def save_state(n, d):
        saved[n] = json.loads(json.dumps(d))
        if not phases or phases[-1] != d.get("phase"):
            phases.append(d.get("phase"))

    async def keygen(path):
        return "ssh-ed25519 AAAAtest ai-hub"

    deps = thunderctl.Deps(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler), base_url="https://t"),
        load_state=lambda n: saved.get(n), save_state=save_state,
        set_enabled=lambda bid, on: enabled.__setitem__(bid, on) or True,
        begin_drain=lambda bid: True, inflight=lambda bid: 0, is_draining=lambda bid: False,
        note_fault=lambda *a, **k: faults.append(a), datadir=datadir, log=lambda m: None,
        now=lambda: clock[0], sleep=sleep,
        ssh=ssh, spawn=None, probe_comfy=probe or (lambda url: asyncio.sleep(0, True)),
        bootstrap_script=lambda: b"#!/bin/bash\necho GW:DONE\n",
        keygen=keygen, default_nodes=lambda: default_nodes)
    b = backend or {"name": "thunder", "type": "comfyui", "api_key": "tok",
                    "thunder": {"gpu_type": "a6000", "num_gpus": 1, "vcpus": 8, "local_port": 18188,
                                "bootstrap_template": "comfy-ui", "reserve_gb": 20,
                                "nodes": ["https://github.com/a/pack@0123abc"],
                                "comfy_commit": COMMIT}}
    c = thunderctl.Controller(b, deps)

    def tunnel():
        t = _NoTunnel(c, fake)
        tunnels.append(t)
        return t
    c._tunnel_factory = tunnel                    # never spawn ssh in tests
    c.h = types.SimpleNamespace(clock=clock, phases=phases, faults=faults, tunnels=tunnels)
    return c, saved, enabled, ssh_calls


class _NoTunnel:
    """Records what the world looked like when the tunnel was started."""
    def __init__(self, c=None, fake=None):
        self.c, self.fake = c, fake
        self.running = False
        self.kh_existed = None

    def start(self):
        self.running = True
        if self.c is not None and self.c.state.uuid:
            self.kh_existed = os.path.exists(self.c._known_hosts_path(self.c.state.uuid))
        if self.fake is not None:
            self.fake.calls.append(("TUNNEL", "start", None))

    async def stop(self):
        self.running = False


def _api(fake, **kw):
    return thunderctl.ThunderApi(
        httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)), "tok", **kw)


class Persistence(unittest.IsolatedAsyncioTestCase):
    async def test_state_roundtrip_excludes_log_and_transfers(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        c.state.phase, c.state.uuid, c.state.index = "ready", "u7", "7"
        c.state.manifests = {"s1": {"models/a.safetensors": {"size": 1}}}
        c.state.log.append("x")
        c.state.transfers["models/a.safetensors"] = {"bytes": 5}
        c._persist()
        d = saved["thunder"]
        self.assertNotIn("log", d)
        self.assertNotIn("transfers", d)
        self.assertEqual(d["manifests"], {"s1": {"models/a.safetensors": {"size": 1}}})
        # a new controller (gateway restart) sees the same instance
        c2 = thunderctl.Controller(c.backend, c.deps)
        self.assertEqual((c2.state.phase, c2.state.uuid, c2.state.index), ("ready", "u7", "7"))
        self.assertEqual(c2.state.log, [])
        self.assertEqual(c2.state.transfers, {})

    async def test_load_tolerates_unknown_keys_and_bad_types(self):
        # a state written by another version must not keep the controller from coming up
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = {"phase": "ready", "uuid": "u1", "port": "30022", "someday": 1,
                            "manifests": None, "log": ["stale"]}
        c2 = thunderctl.Controller(c.backend, c.deps)
        self.assertEqual(c2.state.phase, "ready")
        self.assertEqual(c2.state.port, 30022)
        self.assertEqual(c2.state.manifests, {})
        self.assertEqual(c2.state.log, [])

    async def test_unknown_persisted_phase_becomes_failed_not_off(self):
        # "off" would forget an instance that may still be billing
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = {"phase": "warping", "uuid": "u1", "index": "1"}
        c2 = thunderctl.Controller(c.backend, c.deps)
        self.assertEqual(c2.state.phase, "failed")
        self.assertEqual(c2.state.failed_phase, "warping")
        self.assertEqual(c2.state.uuid, "u1")

    async def test_set_phase_persists_and_failed_records_phase(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        c._set_phase("creating")
        self.assertEqual(saved["thunder"]["phase"], "creating")
        c._set_phase("failed", "boom")
        self.assertEqual(saved["thunder"]["phase"], "failed")
        self.assertEqual(saved["thunder"]["failed_phase"], "creating")
        self.assertEqual(saved["thunder"]["error"], "boom")
        c._set_phase("off")
        self.assertEqual((c.state.error, c.state.failed_phase), ("", ""))
        self.assertTrue(any("creating" in line and "failed" in line for line in c.state.log))
        with self.assertRaises(ValueError):
            c._set_phase("warping")

    async def test_save_failure_is_logged_not_raised(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)

        def boom(n, d):
            raise OSError("disk full")
        c.deps.save_state = boom
        c._set_phase("creating")                 # must not take the lifecycle down
        self.assertTrue(any("disk full" in line for line in c.state.log))

    async def test_log_ring_is_bounded(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        for i in range(250):
            c._log(f"line {i}")
        self.assertEqual(len(c.state.log), 200)
        self.assertTrue(c.state.log[-1].endswith("line 249"))


class LoadFailure(unittest.IsolatedAsyncioTestCase):
    """An unreadable stored record may name a running, billing instance: never "off"."""

    def _check_blocked(self, c, saved):
        self.assertEqual(c.state.phase, "failed")
        self.assertEqual(c.state.failed_phase, thunderctl.LOAD_FAILED)
        self.assertIn("state load failed", c.state.error)
        before = dict(saved)
        c._set_phase("failed", "still unread")
        c._persist()
        self.assertEqual(saved, before)          # the intact record is not overwritten
        self.assertTrue(c.view()["persist_blocked"])

    async def test_load_raising_blocks_persist_and_start(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = {"phase": "ready", "uuid": "u9"}

        def boom(n):
            raise OSError("store locked")
        c.deps.load_state = boom
        c2 = thunderctl.Controller(c.backend, c.deps)
        self._check_blocked(c2, saved)
        self.assertEqual(saved["thunder"]["uuid"], "u9")
        with self.assertRaises(RuntimeError):
            await c2.start()
        self.assertEqual(fake.calls, [])         # no create was even attempted

    async def test_load_non_dict_blocks_persist_and_start(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = ["garbage"]
        c2 = thunderctl.Controller(c.backend, c.deps)
        self._check_blocked(c2, saved)
        self.assertEqual(saved["thunder"], ["garbage"])
        with self.assertRaises(RuntimeError):
            await c2.start()

    async def test_unblock_persist_resumes_saving(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = "garbage"
        c2 = thunderctl.Controller(c.backend, c.deps)
        c2._unblock_persist()                    # what resume() does once reconciled
        c2._set_phase("off")
        self.assertEqual(saved["thunder"]["phase"], "off")
        self.assertFalse(c2.persist_blocked)

    async def test_no_record_is_plain_off(self):
        c, saved, _, _ = make(FakeThunder())
        self.assertEqual(c.state.phase, "off")
        self.assertFalse(c.persist_blocked)


class TokenHygiene(unittest.IsolatedAsyncioTestCase):
    async def test_token_never_in_errors_view_state_or_log(self):
        token = "thunder-SECRET-token-123"
        mode = ["error"]

        def handler(req):
            if mode[0] == "transport":
                raise httpx.ConnectError(f"cannot reach {req.url}")
            # an API that echoes the request back, Authorization included
            return httpx.Response(500, text=f"boom: {dict(req.headers)}")
        fake = FakeThunder()
        backend = {"name": "thunder", "type": "comfyui", "api_key": token,
                   "thunder": {"gpu_type": "a6000", "num_gpus": 1, "vcpus": 8}}
        c, saved, _, _ = make(fake, backend=backend)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        errors = []
        for m in ("error", "transport"):
            mode[0] = m
            try:
                await c.api.list_instances()
            except thunder.ThunderError as e:
                errors.append(str(e))
            await c.refresh_prices()
            await c.refresh_snapshots()
        c._set_phase("failed", errors[0])
        self.assertEqual(len(errors), 2)
        self.assertIn("500", c.state.log[0] + " ".join(c.state.log))
        for where, text in (("errors", " ".join(errors)),
                            ("view", json.dumps(c.view())),
                            ("saved", json.dumps(saved)),
                            ("log", "\n".join(c.state.log))):
            self.assertNotIn(token, text, where)


class Api(unittest.IsolatedAsyncioTestCase):
    async def test_api_delete_falls_back_to_index(self):
        # uuid first (Ruling 11: a stale index can name a stranger's instance)
        fake = FakeThunder()
        fake.delete_by = "index"
        fake.instances["0"] = {"uuid": "u0", "status": "RUNNING"}
        await _api(fake).delete({"index": "0", "uuid": "u0"})
        self.assertEqual(fake.instances, {})
        paths = [p for _, p, _ in fake.calls]
        self.assertEqual(paths, ["/instances/u0/delete", "/instances/0/delete"])

    async def test_api_delete_by_uuid_never_touches_index(self):
        fake = FakeThunder()
        fake.delete_by = "uuid"
        fake.instances["0"] = {"uuid": "u0", "status": "RUNNING"}
        await _api(fake).delete({"index": "0", "uuid": "u0"})
        self.assertEqual([p for _, p, _ in fake.calls], ["/instances/u0/delete"])

    async def test_int_index_zero_is_kept(self):
        # `item.get("index") or ""` dropped index 0
        fake = FakeThunder()
        fake.instances["0"] = {"uuid": "u0", "status": "RUNNING"}
        await _api(fake).delete({"index": 0, "uuid": ""})
        self.assertEqual([p for _, p, _ in fake.calls], ["/instances/0/delete"])

    async def test_api_delete_both_404_is_ok(self):
        fake = FakeThunder()
        await _api(fake).delete({"index": "3", "uuid": "u3"})   # already gone
        self.assertEqual(len(fake.calls), 2)

    async def test_api_non_2xx_raises_thundererror_with_status(self):
        def handler(req):
            return httpx.Response(401, text="unauthorized " + "x" * 1000)
        api = thunderctl.ThunderApi(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tok")
        with self.assertRaises(thunder.ThunderError) as cm:
            await api.list_instances()
        self.assertEqual(cm.exception.status, 401)
        self.assertIn("unauthorized", str(cm.exception))
        self.assertLessEqual(len(str(cm.exception)), 300)

    async def test_transport_error_is_thundererror_without_status(self):
        def handler(req):
            raise httpx.ConnectError("")          # empty str(e), like a real one can be
        api = thunderctl.ThunderApi(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tok")
        with self.assertRaises(thunder.ThunderError) as cm:
            await api.snapshots()
        self.assertIsNone(cm.exception.status)
        self.assertIn("ConnectError", str(cm.exception))

    async def test_bearer_token_sent(self):
        seen = []

        def handler(req):
            seen.append(req.headers.get("authorization"))
            return httpx.Response(200, json={})
        api = thunderctl.ThunderApi(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tok")
        await api.list_instances()
        self.assertEqual(seen, ["Bearer tok"])

    async def test_create_returns_index_as_string(self):
        fake = FakeThunder()
        r = await _api(fake).create({"template": "comfy-ui"})
        self.assertEqual(r, {"index": "0", "uuid": "u0"})

    async def test_list_instances_and_snapshots_are_parsed(self):
        fake = FakeThunder()
        fake.instances["4"] = {"uuid": "u4", "status": "running", "httpPorts": [8188]}
        fake.snaps.append({"id": "s0", "name": "aihub-thunder-20260927t000000z",
                           "status": "READY", "minimumDiskSizeGb": 120, "createdAt": 5})
        api = _api(fake)
        fake.status_script = []
        items = await api.list_instances()
        self.assertEqual(items[0]["index"], "4")
        self.assertEqual(items[0]["http_ports"], [8188])
        snaps = await api.snapshots()
        self.assertEqual(snaps[0]["min_disk_gb"], 120)

    async def test_remove_ports_uuid_then_index(self):
        fake = FakeThunder()
        fake.instances["0"] = {"uuid": "u0", "httpPorts": [8188, 22]}
        await _api(fake).remove_ports({"index": "0", "uuid": "u0"}, [8188])
        self.assertEqual(fake.calls, [("PATCH", "/instances/u0/ports", {"remove_ports": [8188]}),
                                      ("PATCH", "/instances/0/ports", {"remove_ports": [8188]})])
        self.assertEqual(fake.instances["0"]["httpPorts"], [22])

    async def test_modify_falls_back_to_index_and_raises_when_both_404(self):
        calls = []

        def handler(req):
            calls.append(req.url.path)
            return httpx.Response(200, json={}) if req.url.path == "/instances/0/modify" \
                else httpx.Response(404)
        api = thunderctl.ThunderApi(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tok")
        await api.modify({"index": "0", "uuid": "u0"}, {"disk_size_gb": 200})
        self.assertEqual(calls, ["/instances/u0/modify", "/instances/0/modify"])
        with self.assertRaises(thunder.ThunderError) as cm:
            await api.modify({"index": "1", "uuid": "u1"}, {"disk_size_gb": 200})
        self.assertEqual(cm.exception.status, 404)

    async def test_snapshot_create_and_delete(self):
        fake = FakeThunder()
        api = _api(fake)
        sid = await api.create_snapshot({"index": "0", "uuid": "u0"}, "aihub-thunder-x")
        self.assertEqual(sid, "s0")
        self.assertEqual(fake.calls[-1][2]["name"], "aihub-thunder-x")
        # openapi CreateSnapshotRequest.instanceId is a STRING — an int is a 400
        self.assertEqual(fake.calls[-1][2]["instanceId"], "0")
        await api.create_snapshot({"index": 0, "uuid": "u0"}, "aihub-thunder-y")
        self.assertEqual(fake.calls[-1][2]["instanceId"], "0")
        await api.delete_snapshot(sid)
        self.assertEqual([x["id"] for x in fake.snaps], ["s1"])

    async def test_pricing_and_specs_cached_one_hour(self):
        fake = FakeThunder()
        clock = [1000.0]
        api = _api(fake, clock=lambda: clock[0])
        p1 = await api.pricing()
        await api.pricing()
        await api.specs()
        await api.specs()
        self.assertEqual(p1["pricing"]["a6000_x1"], 0.35)
        self.assertEqual([p for _, p, _ in fake.calls], ["/v2/pricing", "/v2/specs"])
        clock[0] += 3601
        await api.pricing()
        self.assertEqual(len(fake.calls), 3)

    async def test_ids_are_path_quoted(self):
        # an id with a "/" must not reach a different endpoint
        seen = []

        def handler(req):
            seen.append(req.url.raw_path)
            return httpx.Response(200, json={})
        api = thunderctl.ThunderApi(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tok")
        await api.delete_snapshot("a/../b")
        self.assertEqual(seen, [b"/snapshots/a%2F..%2Fb"])


class View(unittest.IsolatedAsyncioTestCase):
    async def test_view_without_prices_has_no_cost(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        v = c.view()
        for k in ("phase", "error", "failed_phase", "uuid", "ip", "port", "started_at",
                  "uptime_s", "disk_gb", "cost_per_h", "session_cost", "snapshot", "log",
                  "transfers"):
            self.assertIn(k, v)
        self.assertEqual(v["phase"], "off")
        self.assertIsNone(v["cost_per_h"])
        self.assertIsNone(v["session_cost"])
        self.assertEqual(v["uptime_s"], 0)

    async def test_view_cost_from_cached_pricing(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        await c.refresh_prices()
        c.state.phase, c.state.started_at, c.state.disk_gb = "ready", 1_790_000_000.0 - 7200, 200
        v = c.view()
        # 0.35 + (8-6) vCPUs * 0.04 + (200-100) GB * 0.0003
        self.assertAlmostEqual(v["cost_per_h"], 0.35 + 0.08 + 0.03)
        self.assertEqual(v["uptime_s"], 7200)
        self.assertAlmostEqual(v["session_cost"], 2 * (0.35 + 0.08 + 0.03))

    async def test_refresh_prices_failure_is_logged(self):
        def handler(req):
            return httpx.Response(503, text="down")
        c, _, _, _ = make(FakeThunder())
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await c.refresh_prices()                 # no raise: a price is display only
        self.assertTrue(any("503" in line for line in c.state.log))
        self.assertIsNone(c.view()["cost_per_h"])

    async def test_view_snapshot_monthly(self):
        fake = FakeThunder()
        fake.snaps.append({"id": "s0", "name": "aihub-thunder-20260927t000000z",
                           "status": "READY", "minimumDiskSizeGb": 120, "createdAt": 5})
        c, _, _, _ = make(fake)
        await c.refresh_prices()
        await c.refresh_snapshots()
        c.state.snapshot_id = "s0"
        snap = c.view()["snapshot"]
        self.assertEqual(snap["id"], "s0")
        self.assertEqual(snap["gb"], 120)
        self.assertAlmostEqual(snap["monthly"], 120 * 0.00006849 * 730)


class Tunnel(unittest.IsolatedAsyncioTestCase):
    async def test_default_tunnel_factory_builds_supervisor_with_safe_argv(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        del c._tunnel_factory                    # back to the class default
        c.state.uuid, c.state.ip, c.state.port = "u0", "10.0.0.5", 30022
        sup = c._tunnel_factory()
        self.assertIsInstance(sup, sshrun.Supervisor)
        argv = sup._argv_fn()
        self.assertEqual(argv[-2:], ["--", "ubuntu@10.0.0.5"])
        self.assertIn("127.0.0.1:18188:127.0.0.1:8188", argv)
        self.assertIn("ExitOnForwardFailure=yes", argv)
        d = c.deps.datadir
        self.assertIn(f"UserKnownHostsFile={d}/thunder-known_hosts/u0", argv)
        self.assertEqual(argv[argv.index("-i") + 1], f"{d}/thunder.key")

    async def test_tunnel_argv_refuses_without_instance(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        with self.assertRaises(RuntimeError):
            c._tunnel_argv()

    async def test_bid_and_url(self):
        c, _, _, _ = make(FakeThunder())
        self.assertEqual(c.bid, "comfyui:thunder")
        self.assertEqual(c.url, "http://127.0.0.1:18188")



def _ready_snap(fake, name="aihub-thunder-20260926t120000z", sid="s9", min_gb=120, created=9):
    fake.snaps.append({"id": sid, "name": name, "status": "READY",
                       "minimumDiskSizeGb": min_gb, "createdAt": created})


def _creates(fake):
    return [b for m, p, b in fake.calls if p == "/instances/create"]


def _ssh_cmds(fake):
    return [p for m, p, _ in fake.calls if m == "SSH"]


class Start(unittest.IsolatedAsyncioTestCase):
    """Spec "Start" steps 0–6 (P1: starting → ready, no model sync yet)."""

    async def test_start_first_time_bootstraps_from_template(self):
        fake = FakeThunder()
        c, saved, enabled, _ = make(fake)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertIs(enabled["comfyui:thunder"], True)
        body = _creates(fake)[0]
        self.assertEqual(body["template"], "comfy-ui")
        self.assertEqual(body["public_key"], "ssh-ed25519 AAAAtest ai-hub")
        self.assertEqual(body["disk_size_gb"], 100)
        cmds = _ssh_cmds(fake)
        boot = [x for x in cmds if "bash -s --" in x]
        self.assertEqual(boot, ["bash -o pipefail -c "
                                f"'bash -s -- {COMMIT} 2>&1 | tee ~/gw-bootstrap.log'"])
        self.assertIn(thunderctl._START_CMD, cmds)
        self.assertTrue(thunderctl._START_CMD.endswith(
            "setsid nohup ~/start-comfy.sh >/dev/null 2>&1 < /dev/null &"))
        # "off" first: template + request time are persisted before the create
        self.assertEqual(c.h.phases, ["off", "creating", "connecting", "bootstrapping",
                                      "starting", "ready"])
        self.assertEqual((saved["thunder"]["uuid"], saved["thunder"]["index"]), ("u0", "0"))
        self.assertEqual((c.state.ip, c.state.port), ("10.0.0.5", 30022))
        self.assertEqual(c.state.started_at, 1_790_000_000.0)
        self.assertTrue(c.h.tunnels and c.h.tunnels[-1].running)

    async def test_bootstrap_uploads_node_list_then_runs_script_with_it_on_stdin(self):
        fake = FakeThunder()
        c, _, _, calls = make(fake)
        c.deps.bootstrap_script = lambda: b"#!/bin/bash\necho GW:SMOKE ok\necho GW:DONE\n"
        await c.start()
        remote = [(argv[-1], stdin) for argv, stdin in calls]
        up = [i for i, (cmd, _) in enumerate(remote) if cmd == "cat > ~/.gw-nodes.txt"]
        run = [i for i, (cmd, _) in enumerate(remote) if "bash -s --" in cmd]
        self.assertEqual(len(up), 1)
        self.assertLess(up[0], run[0])
        self.assertEqual(remote[up[0]][1], b"https://github.com/a/pack@0123abc\n")
        self.assertEqual(remote[run[0]][1], b"#!/bin/bash\necho GW:SMOKE ok\necho GW:DONE\n")
        # every argv ends in `-- ubuntu@ip <cmd>` with the instance's port and our key
        argv = calls[0][0]
        self.assertEqual(argv[-3:-1], ["--", "ubuntu@10.0.0.5"])
        self.assertEqual(argv[argv.index("-p") + 1], "30022")
        self.assertEqual(argv[argv.index("-i") + 1], os.path.join(c.deps.datadir, "thunder.key"))

    async def test_empty_node_list_uploads_the_default_list(self):
        # Ruling 12: an empty file would install no packs and fail the smoke test with a
        # misleading reason
        fake = FakeThunder()
        c, _, _, calls = make(fake, default_nodes="# default\nregistry:x@1.0\n")
        c.backend["thunder"]["nodes"] = []
        await c.start()
        up = [stdin for argv, stdin in calls if argv[-1] == "cat > ~/.gw-nodes.txt"]
        self.assertEqual(up, [b"# default\nregistry:x@1.0\n"])

    async def test_no_node_list_at_all_refuses_before_create(self):
        fake = FakeThunder()
        c, _, enabled, _ = make(fake)
        c.backend["thunder"]["nodes"] = []
        await c.start()
        self.assertEqual(c.state.phase, "off")
        self.assertIn("node", c.state.error)
        self.assertEqual(_creates(fake), [])         # nothing is paid for
        self.assertIs(enabled["comfyui:thunder"], False)

    async def test_start_from_newest_ready_snapshot_skips_bootstrap(self):
        fake = FakeThunder()
        _ready_snap(fake, "aihub-thunder-20260925t120000z", "s1", min_gb=110, created=5)
        _ready_snap(fake, "aihub-thunder-20260926t120000z", "s2", min_gb=120, created=9)
        c, _, _, _ = make(fake)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        body = _creates(fake)[0]
        self.assertEqual(body["template"], "aihub-thunder-20260926t120000z")
        self.assertEqual(body["disk_size_gb"], 120)      # never below the snapshot's minimum
        self.assertFalse([x for x in _ssh_cmds(fake) if "bash -s" in x or ".gw-nodes" in x])
        self.assertEqual(c.state.snapshot_id, "s2")
        self.assertEqual(c.state.disk_gb, 120)
        self.assertNotIn("bootstrapping", c.h.phases)

    async def test_restoring_status_is_its_own_phase(self):
        fake = FakeThunder()
        _ready_snap(fake)
        fake.status_script = ["PROVISIONING", "RESTORING", "RESTORING", "RUNNING"]
        c, _, _, _ = make(fake)
        await c.start()
        self.assertEqual(c.h.phases, ["off", "creating", "restoring", "connecting",
                                      "starting", "ready"])

    async def test_uuid_persisted_before_wait(self):
        fake = FakeThunder()
        fake.status_script = ["PROVISIONING"] * 50
        c, saved, _, _ = make(fake)
        at_first_wait = []
        orig_sleep = c.deps.sleep

        async def sleep(sec):
            if not at_first_wait:
                at_first_wait.append((dict(saved.get("thunder") or {}), list(fake.calls)))
            await orig_sleep(sec)
        c.deps.sleep = sleep
        await c.start()
        first_state, calls_then = at_first_wait[0]
        self.assertEqual((first_state["uuid"], first_state["index"]), ("u0", "0"))
        self.assertEqual(first_state["phase"], "creating")
        self.assertIn("/instances/list", [p for _, p, _ in calls_then])
        self.assertEqual(c.state.phase, "failed")
        self.assertEqual(c.state.failed_phase, "creating")
        self.assertEqual(saved["thunder"]["uuid"], "u0")     # the instance is not forgotten
        self.assertEqual(saved["thunder"]["index"], "0")
        self.assertIn("0", fake.instances)                    # and not deleted either
        # 15 min + 8 min per started 100 GB (disk 100 GB)
        self.assertGreaterEqual(c.h.clock[0] - 1_790_000_000.0, 23 * 60)
        self.assertLess(c.h.clock[0] - 1_790_000_000.0, 23 * 60 + 30)
        self.assertEqual(_ssh_cmds(fake), [])
        self.assertEqual(len(c.h.faults), 1)
        self.assertEqual(c.h.faults[0][1:3], ("lifecycle", "error"))

    async def test_open_http_port_is_removed_before_connecting(self):
        fake = FakeThunder()
        fake.http_ports_on_create = [8188]
        c, _, _, _ = make(fake)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        kinds = [(m, p) for m, p, _ in fake.calls]
        patch = kinds.index(("PATCH", "/instances/0/ports"))
        first_ssh = next(i for i, (m, _) in enumerate(kinds) if m in ("SSH", "TUNNEL"))
        self.assertLess(patch, first_ssh)
        self.assertEqual(fake.calls[patch][2], {"remove_ports": [8188]})
        # uuid form first (Ruling 11)
        self.assertEqual(kinds[patch - 1], ("PATCH", "/instances/u0/ports"))

    async def test_port_that_stays_open_fails_never_starts(self):
        fake = FakeThunder()
        fake.http_ports_on_create = [8188]
        fake.ignore_port_remove = True
        c, _, _, _ = make(fake)
        await c.start()
        self.assertEqual(c.state.phase, "failed")
        self.assertIn("8188", c.state.error)
        self.assertFalse([x for x in _ssh_cmds(fake) if "start-comfy.sh" in x])
        self.assertEqual(c.h.tunnels, [])                    # never connected at all
        self.assertIn("0", fake.instances)

    async def test_port_opened_during_bootstrap_is_closed_before_starting(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)

        async def ssh(argv, stdin=None, timeout=60):
            fake.calls.append(("SSH", argv[-1], stdin))
            if "bash -s" in argv[-1]:
                fake.instances["0"]["httpPorts"] = [8188]     # the template opened it
                return (0, b"GW:SMOKE ok\nGW:DONE\n", b"")
            return (0, b"", b"")
        c.deps.ssh = ssh
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        kinds = [(m, p) for m, p, _ in fake.calls]
        boot = next(i for i, (m, p) in enumerate(kinds) if m == "SSH" and "bash -s" in p)
        patch = kinds.index(("PATCH", "/instances/0/ports"))
        start = next(i for i, (m, p) in enumerate(kinds) if m == "SSH" and "start-comfy" in p)
        self.assertLess(boot, patch)
        self.assertLess(patch, start)

    async def test_bootstrap_smoke_fail_keeps_instance(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake, ssh_script={"bash -s": (3, b"GW:SMOKE fail cumesh\n", b"")})
        await c.start()
        self.assertEqual(c.state.phase, "failed")
        self.assertEqual(c.state.failed_phase, "bootstrapping")
        self.assertIn("cumesh", c.state.error)
        self.assertIn("0", fake.instances)
        self.assertEqual(saved["thunder"]["uuid"], "u0")
        self.assertFalse([x for x in _ssh_cmds(fake) if "start-comfy.sh" in x])
        self.assertEqual(len(c.h.faults), 1)

    async def test_node_fail_fails_even_when_done(self):
        # Ruling 9: a missing node pack makes workflows fail later with a plausible error
        out = (b"GW:PHASE nodes\nGW:NODE_FAIL ComfyUI-Foo pip-failed\nGW:SMOKE ok\n"
               b"GW:DONE\n")
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"bash -s": (0, out, b"")})
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "bootstrapping"))
        self.assertIn("ComfyUI-Foo", c.state.error)
        self.assertFalse([x for x in _ssh_cmds(fake) if "start-comfy.sh" in x])

    async def test_bootstrap_nonzero_rc_fails(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"bash -s": (1, b"GW:PHASE venv\n",
                                                        b"x\nbootstrap: cannot build the venv\n")})
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "bootstrapping"))
        self.assertIn("rc 1", c.state.error)
        self.assertIn("venv", c.state.error)

    async def test_bootstrap_without_done_fails(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"bash -s": (0, b"GW:PHASE nodes\n", b"")})
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "bootstrapping"))

    async def test_bootstrap_reports_are_collected_and_persisted(self):
        out = (b"GW:PHASE inventory\nsome noise\n"
               b"GW:TEMPLATE_NODE ComfyUI-Manager\n"
               b"GW:UNKNOWN_MODEL models/checkpoints/sd15.safetensors\t2132625894\n"
               b"GW:UNKNOWN_MODEL models/../../.ssh/id\t5\n"
               b"GW:SMOKE ok\nGW:DONE\n")
        fake = FakeThunder()
        c, saved, _, _ = make(fake, ssh_script={"bash -s": (0, out, b"")})
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(saved["thunder"]["bootstrap_unknown"],
                         {"models/checkpoints/sd15.safetensors": 2132625894})
        self.assertEqual(saved["thunder"]["bootstrap_template_nodes"], ["ComfyUI-Manager"])
        v = c.view()
        self.assertEqual(v["bootstrap_unknown"], {"models/checkpoints/sd15.safetensors": 2132625894})
        self.assertEqual(v["bootstrap_template_nodes"], ["ComfyUI-Manager"])
        joined = "\n".join(c.state.log)
        self.assertIn("GW:TEMPLATE_NODE ComfyUI-Manager", joined)
        self.assertIn("some noise", joined)
        # a restart keeps them (the panel shows them until they are deleted)
        c2 = thunderctl.Controller(c.backend, c.deps)
        self.assertEqual(c2.state.bootstrap_template_nodes, ["ComfyUI-Manager"])

    async def test_known_hosts_reset_for_new_uuid(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        kh = c._known_hosts_path("u0")
        os.makedirs(os.path.dirname(kh), exist_ok=True)
        with open(kh, "w") as f:
            f.write("[10.0.0.5]:30022 ssh-ed25519 AAAAold\n")
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertIs(c.h.tunnels[0].kh_existed, False)
        self.assertFalse(os.path.exists(kh))
        self.assertEqual(os.stat(os.path.dirname(kh)).st_mode & 0o777, 0o700)

    async def test_ssh_probe_retries_until_reachable(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"true": [(255, b"", b"refused"), (255, b"", b"x"),
                                                      (0, b"", b"")]})
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_ssh_cmds(fake).count("true"), 3)

    async def test_ssh_never_reachable_fails_connecting(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"true": (255, b"", b"Connection refused")})
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "connecting"))
        self.assertIn("Connection refused", c.state.error)
        self.assertLessEqual(c.h.clock[0] - 1_790_000_000.0, 5 * 60 + 60)

    async def test_comfy_probe_waits_then_ready(self):
        seen = []

        async def probe(url):
            seen.append(url)
            return len(seen) >= 3
        fake = FakeThunder()
        _ready_snap(fake)
        c, _, _, _ = make(fake, probe=probe)
        await c.start()
        self.assertEqual(c.state.phase, "ready")
        self.assertEqual(seen, ["http://127.0.0.1:18188"] * 3)

    async def test_comfy_never_answers_fails_starting(self):
        async def probe(url):
            raise httpx.ConnectError("refused")      # a raising probe counts as "not yet"
        fake = FakeThunder()
        _ready_snap(fake)
        c, _, _, _ = make(fake, probe=probe)
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "starting"))
        self.assertIn("0", fake.instances)

    async def test_api_error_before_create_is_off_with_message(self):
        fake = FakeThunder()
        c, _, enabled, _ = make(fake)

        def handler(req):
            if req.url.path == "/snapshots/list":
                return httpx.Response(503, text="maintenance")
            return fake.handler(req)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await c.start()
        self.assertEqual(c.state.phase, "off")
        self.assertIn("maintenance", c.state.error)
        self.assertEqual(_creates(fake), [])
        self.assertIs(enabled["comfyui:thunder"], False)

    async def test_bad_comfy_commit_is_refused_up_front(self):
        fake = FakeThunder()
        c, _, enabled, _ = make(fake)
        for bad in ("abc", "", "1d61dcc3", COMMIT + "0", "g" * 40, None):
            c.backend["thunder"]["comfy_commit"] = bad
            with self.assertRaises(RuntimeError) as cm:
                await c.start()
            self.assertIn("comfy_commit", str(cm.exception))
        self.assertEqual(fake.calls, [])
        self.assertEqual(enabled, {})

    async def test_start_refused_while_an_instance_is_known(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        for phase, uuid, index in (("ready", "u1", "1"), ("creating", "u1", "1"),
                                   ("failed", "u1", "1"), ("failed", "", "3")):
            c.state.phase, c.state.uuid, c.state.index = phase, uuid, index
            with self.assertRaises(RuntimeError):
                await c.start()
        self.assertEqual(fake.calls, [])

    async def test_failed_without_instance_may_start_again(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        c.state.phase, c.state.failed_phase, c.state.error = "failed", "creating", "x"
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)

    async def test_concurrent_start_is_refused(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        first = asyncio.ensure_future(c.start())
        await asyncio.sleep(0)
        with self.assertRaises(RuntimeError):
            await c.start()
        await first
        self.assertEqual(len(_creates(fake)), 1)

    async def test_create_error_is_off_not_failed(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)

        def handler(req):
            if req.url.path == "/instances/create":
                return httpx.Response(400, text="bad gpu")
            return fake.handler(req)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await c.start()
        self.assertEqual(c.state.phase, "off")
        self.assertIn("bad gpu", c.state.error)
        self.assertEqual(c.state.uuid, "")

    async def test_instance_that_vanishes_while_waiting_fails(self):
        fake = FakeThunder()
        fake.status_script = ["PROVISIONING", "DELETED"]
        c, _, _, _ = make(fake)
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "creating"))
        self.assertIn("DELETED", c.state.error)

    async def test_list_error_while_waiting_is_retried(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        n = [0]

        def handler(req):
            if req.url.path == "/instances/list":
                n[0] += 1
                if n[0] == 1:
                    return httpx.Response(502, text="gateway")
            return fake.handler(req)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)


    async def test_enable_failure_ends_in_off_before_create(self):
        # a backend that stays disabled is never polled or routed to: an instance
        # created for it would bill for nothing
        for behaviour in ("raise", "false"):
            fake = FakeThunder()
            c, _, _, _ = make(fake)
            calls = []

            def set_enabled(bid, on, behaviour=behaviour):
                calls.append(on)
                if behaviour == "raise":
                    raise OSError("store locked")
                return False
            c.deps.set_enabled = set_enabled
            await c.start()
            self.assertEqual(c.state.phase, "off", behaviour)
            self.assertIn("cannot enable", c.state.error)
            self.assertEqual(_creates(fake), [])
            self.assertEqual(fake.calls, [])
            self.assertEqual(calls, [True])       # no pointless disable afterwards

    async def test_bootstrap_timeout_reads_the_log_on_the_instance(self):
        # sshrun.run answers a timeout with (124, b"", b"timeout") — everything it read
        # is gone; the tee'd log file on the instance says how far it got
        tail = (b"GW:PHASE nodes\nGW:PHASE extensions\nbuilding cumesh wheel ...\n")
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"bash -s": (124, b"", b"timeout"),
                                            "tail -n 200 ~/gw-bootstrap.log": (0, tail, b"")})
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "bootstrapping"))
        self.assertIn("timed out", c.state.error)
        self.assertIn("phase extensions", c.state.error)
        self.assertTrue(any("building cumesh wheel" in ln for ln in c.state.log))
        self.assertIn("tail -n 200 ~/gw-bootstrap.log", _ssh_cmds(fake))

    async def test_bootstrap_dropped_connection_reads_the_log(self):
        tail = b"GW:PHASE venv\nbootstrap: cannot build the venv\n"
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"bash -s": (255, b"", b"Connection reset"),
                                            "tail -n 200": (0, tail, b"")})
        await c.start()
        self.assertIn("rc 255 in phase venv", c.state.error)
        self.assertIn("cannot build the venv", c.state.error)

    async def test_bootstrap_output_present_needs_no_log_fetch(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"bash -s": (1, b"GW:PHASE venv\nbootstrap: x\n", b"")})
        await c.start()
        self.assertIn("rc 1 in phase venv: bootstrap: x", c.state.error)
        self.assertFalse([x for x in _ssh_cmds(fake) if x.startswith("tail ")])

    async def test_missing_start_script_fails_fast_with_a_reason(self):
        fake = FakeThunder()
        _ready_snap(fake)
        c, _, _, _ = make(fake, ssh_script={"test -x ~/start-comfy.sh":
                                            (3, b"", b"start-comfy.sh missing\n")})
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "starting"))
        self.assertIn("start-comfy.sh missing", c.state.error)
        self.assertLess(c.h.clock[0] - 1_790_000_000.0, 60)     # no 10-min probe wait

    def _uuidless_create(self, c, fake, template):
        def handler(req):
            r = fake.handler(req)
            if req.url.path == "/instances/create":
                idx = str(fake.next_index - 1)
                fake.instances[idx]["template"] = template
                return httpx.Response(201, json={"identifier": int(idx)})
            return r
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_create_without_uuid_adopts_our_instance_by_index_and_template(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        self._uuidless_create(c, fake, "comfy-ui")
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(saved["thunder"]["uuid"], "u0")

    async def test_create_without_uuid_never_adopts_a_stranger(self):
        # Ruling 10: Thunder reuses indices — the item at our index with another
        # template is not ours, so it is never connected to (let alone deleted)
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        self._uuidless_create(c, fake, "base")
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "creating"))
        self.assertEqual((saved["thunder"]["uuid"], saved["thunder"]["index"]), ("", "0"))
        self.assertEqual(_ssh_cmds(fake), [])
        with self.assertRaises(RuntimeError):          # an index still names an instance
            await c.start()

    def _restored(self, fake, **state):
        c, saved, _, _ = make(fake)
        saved["thunder"] = dict({"phase": "creating", "uuid": "", "index": "0",
                                 "created_template": "comfy-ui",
                                 "create_requested_at": 1_790_000_000.0}, **state)
        return thunderctl.Controller(c.backend, c.deps), saved

    async def test_template_and_request_time_are_persisted_before_create(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        seen = []

        def handler(req):
            if req.url.path == "/instances/create":
                seen.append(dict(saved.get("thunder") or {}))
            return fake.handler(req)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await c.start()
        self.assertEqual(seen[0]["created_template"], "comfy-ui")
        self.assertEqual(seen[0]["create_requested_at"], 1_790_000_000.0)

    def test_restored_state_adopts_uuidless_instance(self):
        # a gateway restart during `creating`: the persisted template still identifies it
        c, saved = self._restored(FakeThunder())
        items = [{"index": "0", "uuid": "u0", "template": "comfy-ui",
                  "created_at": "1790000030"}]
        self.assertEqual(c._find_ours(items)["uuid"], "u0")
        self.assertEqual(saved["thunder"]["uuid"], "u0")

    def test_created_at_outside_the_window_is_not_ours(self):
        c, _ = self._restored(FakeThunder())
        for created in ("1790003600", 1_790_003_600_000, "2026-09-27T12:00:00Z",
                        "1789990000"):
            items = [{"index": "0", "uuid": "u9", "template": "comfy-ui",
                      "created_at": created}]
            self.assertIsNone(c._find_ours(items), created)
        # milliseconds and ISO inside the window, and an unreadable value, pass
        for created in (1_790_000_060_000, "2026-09-21T14:14:00+00:00", "soon", None):
            c.state.uuid = ""
            items = [{"index": "0", "uuid": "u9", "template": "comfy-ui",
                      "created_at": created}]
            self.assertIsNotNone(c._find_ours(items), created)

    def test_none_template_in_list_is_not_ours(self):
        c, _ = self._restored(FakeThunder())
        self.assertIsNone(c._find_ours([{"index": "0", "uuid": "u9", "template": None}]))

    def test_no_persisted_template_adopts_nothing(self):
        c, _ = self._restored(FakeThunder(), created_template="")
        self.assertIsNone(c._find_ours([{"index": "0", "uuid": "u9", "template": ""}]))

class RestartComfy(unittest.IsolatedAsyncioTestCase):
    async def _ready(self, **kw):
        fake = FakeThunder()
        _ready_snap(fake)
        c, saved, enabled, calls = make(fake, **kw)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        return fake, c

    async def test_restart_kills_comfy_and_runs_the_loop_again(self):
        fake, c = await self._ready()
        n = len(fake.calls)
        await c.restart_comfy()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        cmds = [p for m, p, _ in fake.calls[n:] if m == "SSH"]
        self.assertEqual(len(cmds), 1)
        self.assertIn("pkill -f", cmds[0])
        self.assertTrue(cmds[0].endswith("; " + thunderctl._START_CMD))
        self.assertEqual(c.h.phases[-2:], ["starting", "ready"])

    async def test_pkill_pattern_matches_comfy_but_not_the_remote_shell(self):
        # The command line to match is the one start-comfy.sh really runs — read from
        # the heredoc the bootstrap writes, not copied by hand. The remote
        # `bash -c "<cmd>"` carries the pattern in ITS command line: a plain pattern
        # would kill the shell running the restart before it starts the loop.
        # pkill -f reads the pattern as an ERE; for brackets and literals re agrees.
        import re
        import shlex
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "ops", "thunder-bootstrap.sh")
        with open(path, encoding="utf-8") as f:
            script = f.read()
        heredoc = script.split("<<'START_COMFY_EOF'\n", 1)[1].split("\nSTART_COMFY_EOF", 1)[0]
        line = next(ln for ln in heredoc.splitlines() if " main.py " in ln)
        argv = shlex.split(line.split("\\")[0].split("9>&-")[0])
        self.assertEqual(argv[0], "$COMFY_PY")
        comfy = " ".join(["/home/ubuntu/ComfyUI/venv/bin/python"] + argv[1:])
        self.assertIn("--listen 127.0.0.1", comfy)
        pat = shlex.split(thunderctl._RESTART_CMD.split(";")[0])[2]
        self.assertTrue(re.search(pat, comfy), comfy)
        self.assertFalse(re.search(pat, "bash -c " + thunderctl._RESTART_CMD))
        self.assertFalse(re.search(pat, "bash -c " + shlex.quote(thunderctl._RESTART_CMD)))
        self.assertFalse(re.search(pat, "/bin/bash /home/ubuntu/start-comfy.sh"))

    async def test_restart_closes_ports_first(self):
        fake, c = await self._ready()
        fake.instances["0"]["httpPorts"] = [8188]
        n = len(fake.calls)
        await c.restart_comfy()
        kinds = [(m, p) for m, p, _ in fake.calls[n:]]
        self.assertLess(kinds.index(("PATCH", "/instances/0/ports")),
                        next(i for i, (m, _) in enumerate(kinds) if m == "SSH"))

    async def test_restart_from_failed_bootstrap(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"bash -s": (3, b"GW:SMOKE fail x\n", b"")})
        await c.start()
        self.assertEqual(c.state.phase, "failed")
        await c.restart_comfy()
        self.assertEqual(c.state.phase, "ready", c.state.error)

    async def test_restart_refused_without_instance(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        with self.assertRaises(RuntimeError):
            await c.restart_comfy()
        c.state.phase, c.state.uuid = "failed", ""
        with self.assertRaises(RuntimeError):
            await c.restart_comfy()
        self.assertEqual(fake.calls, [])

    async def test_restart_refused_when_unreconciled(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = "garbage"
        c2 = thunderctl.Controller(c.backend, c.deps)
        with self.assertRaises(RuntimeError):
            await c2.restart_comfy()

    async def test_restart_that_never_answers_fails_starting(self):
        answers = [True]

        async def probe(url):
            return answers[0]
        fake, c = await self._ready(probe=probe)
        answers[0] = False
        await c.restart_comfy()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "starting"))

    async def test_restart_waits_before_first_probe(self):
        # the old process may still answer for a moment after pkill
        seen, box = [], []

        async def probe(url):
            seen.append(box[0].h.clock[0] if box else None)
            return True
        fake, c = await self._ready(probe=probe)
        box.append(c)
        t0 = c.h.clock[0]
        seen.clear()
        await c.restart_comfy()
        self.assertGreater(seen[0], t0)


class BootstrapParse(unittest.TestCase):
    def test_prefix_not_position(self):
        r = thunderctl.parse_bootstrap(
            "noise GW:SMOKE ok\nGW:SMOKEY ok\nGW:PHASE smoke\n  GW:NODE_FAIL x y\n"
            "GW:SMOKE fail a,b\nGW:DONE\n")
        self.assertEqual(r["smoke"], "fail a,b")
        self.assertEqual(r["node_fails"], [])      # indented = not a GW line
        self.assertTrue(r["done"])
        self.assertEqual(r["phase"], "smoke")

    def test_unknown_model_bad_size_is_skipped(self):
        r = thunderctl.parse_bootstrap("GW:UNKNOWN_MODEL models/a.bin\tlots\n"
                                       "GW:UNKNOWN_MODEL models/b c.bin\t7\n")
        self.assertEqual(r["unknown"], {"models/b c.bin": 7})


if __name__ == "__main__":
    unittest.main()
