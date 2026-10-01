"""Managed-host lifecycle controller against a stubbed Thunder API and a fake ssh.

The fixture is ONE host (named like its backend, `thunder`) with ONE ComfyUI service —
the shape every test before the host/service split was written for, so their money
invariants read unchanged; the host-level tests (several services, drain of all, host
faults, forwards on the running master) build their own.
run: venv/bin/python -m unittest tests.test_hostctl -v"""
import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
import types
import unittest
from unittest import mock

import httpx

import modelsync as ms
import services
import sshrun
import thunder
import hostctl
from tests.fakes import FakeThunder  # the scripted Thunder REST API


COMMIT = "1d61dcc35c35541388c0001bacc7703db14e8bea"
BID = "comfyui:thunder"         # the fixture's one ComfyUI service
HOST_BOOT = "gw-host-bootstrap.log"       # in the host bootstrap's command only
HOST_SCRIPT = b"#!/bin/bash\necho host; echo GW:DONE\n"
_TMPDIRS = []


def tearDownModule():
    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)


def make(fake, backend=None, ssh_script=None, datadir=None, default_nodes="",
         probe=None, state=None, host_name=None, services=None, probe_http=None):
    """A controller on the stub API. `c.h` carries the harness: `clock` (a list; the
    fake sleep ADVANCES it, so every timeout is reachable without waiting), `phases`
    (every persisted phase in order), `faults`, `tunnels`. `ssh_script` maps a
    substring of the remote command to a result, or to a LIST of results consumed in
    order (the last one sticks); the bootstrap succeeds by default. Every ssh call is also appended to `fake.calls` as
    ("SSH", remote_cmd, stdin), so the order against API calls is checkable. The
    datadir is a fresh temp dir per controller unless given. `state` is a persisted
    record the controller loads at construction (a gateway restart). The host is named
    like the backend unless `host_name` says otherwise, its options ARE the backend's
    `thunder` block (the fixture's shape) and `services` replaces the one ComfyUI service."""
    b = backend or {"name": "thunder", "type": "comfyui", "api_key": "tok",
                    "thunder": {"gpu_type": "a6000", "num_gpus": 1, "vcpus": 8, "local_port": 18188,
                                "bootstrap_template": "comfy-ui", "reserve_gb": 20,
                                "nodes": ["https://github.com/a/pack@0123abc"],
                                "comfy_commit": COMMIT}}
    hname = host_name or b["name"]
    saved = {} if state is None else {hname: state}
    enabled = {}
    ssh_calls = []
    clock = [1_790_000_000.0]
    phases, faults, tunnels = [], [], []
    if datadir is None:
        datadir = tempfile.mkdtemp(prefix="hostctl-test-")
        _TMPDIRS.append(datadir)
    # both bootstraps succeed unless the test scripts otherwise; the host bootstrap's
    # key comes FIRST (its command also says `bash -s`), so a test that scripts
    # "bash -s" scripts the ComfyUI bootstrap only, as before the split; the test's own
    # keys come before the default "bash -s" (a service's setup also says `bash -s`)
    script = {HOST_BOOT: (0, b"GW:PHASE tools\nGW:DONE\n", b"")}
    script.update(ssh_script or {})
    script.setdefault("bash -s", (0, b"GW:PHASE smoke\nGW:SMOKE ok\nGW:DONE\n", b""))

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

    deps = hostctl.Deps(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler), base_url="https://t"),
        load_state=lambda n: saved.get(n), save_state=save_state,
        set_enabled=lambda bid, on: enabled.__setitem__(bid, on) or True,
        begin_drain=lambda bid: True, inflight=lambda bid: 0, is_draining=lambda bid: False,
        note_fault=lambda *a, **k: faults.append(a), datadir=datadir, log=lambda m: None,
        now=lambda: clock[0], sleep=sleep,
        ssh=ssh, spawn=None, probe_comfy=probe or (lambda url: asyncio.sleep(0, True)),
        probe_http=probe_http or (lambda url: asyncio.sleep(0, 200)),
        bootstrap_script=lambda: b"#!/bin/bash\necho GW:DONE\n",
        host_bootstrap_script=lambda: HOST_SCRIPT,
        keygen=keygen, default_nodes=lambda: default_nodes)
    host = {"name": hname, "provider": "thunder", "options": b["thunder"],
            "api_key": b.get("api_key", "")}
    if services is None:
        services = [dict(b, local_port=int(b["thunder"].get("local_port") or 18188),
                         remote_port=8188)]
    c = hostctl.Controller(host, services, deps)

    def tunnel():
        t = _NoTunnel(c, fake)
        tunnels.append(t)
        return t
    c._tunnel_factory = tunnel                    # never spawn ssh in tests
    c.h = types.SimpleNamespace(clock=clock, phases=phases, faults=faults, tunnels=tunnels,
                                saved=saved)
    return c, saved, enabled, ssh_calls


def again(c):
    """The same host after a gateway restart: a new controller on the same deps."""
    return hostctl.Controller(c.host, c.services, c.deps)


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
        c2 = again(c)
        self.assertEqual((c2.state.phase, c2.state.uuid, c2.state.index), ("ready", "u7", "7"))
        self.assertEqual(c2.state.log, [])
        self.assertEqual(c2.state.transfers, {})

    async def test_load_tolerates_unknown_keys_and_bad_types(self):
        # a state written by another version must not keep the controller from coming up
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = {"phase": "ready", "uuid": "u1", "port": "30022", "someday": 1,
                            "manifests": None, "log": ["stale"]}
        c2 = again(c)
        self.assertEqual(c2.state.phase, "ready")
        self.assertEqual(c2.state.port, 30022)
        self.assertEqual(c2.state.manifests, {})
        self.assertEqual(c2.state.log, [])

    async def test_unknown_persisted_phase_becomes_failed_not_off(self):
        # "off" would forget an instance that may still be billing
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = {"phase": "warping", "uuid": "u1", "index": "1"}
        c2 = again(c)
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
        self.assertEqual(c.state.failed_phase, hostctl.LOAD_FAILED)
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
        c2 = again(c)
        self._check_blocked(c2, saved)
        self.assertEqual(saved["thunder"]["uuid"], "u9")
        with self.assertRaises(RuntimeError):
            await c2.start()
        self.assertEqual(fake.calls, [])         # no create was even attempted

    async def test_load_non_dict_blocks_persist_and_start(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = ["garbage"]
        c2 = again(c)
        self._check_blocked(c2, saved)
        self.assertEqual(saved["thunder"], ["garbage"])
        with self.assertRaises(RuntimeError):
            await c2.start()

    async def test_unblock_persist_resumes_saving(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = "garbage"
        c2 = again(c)
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


def _backend_named(name):
    return {"name": name, "type": "comfyui", "api_key": "tok",
            "thunder": {"gpu_type": "a6000", "num_gpus": 1, "vcpus": 8, "local_port": 18188,
                        "bootstrap_template": "comfy-ui", "reserve_gb": 20,
                        "nodes": [], "comfy_commit": COMMIT}}


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
        # R-W1: a ControlMaster whose socket sits under the data dir, 0700
        self.assertIn("-M", argv)
        ctl = argv[argv.index("-S") + 1]
        self.assertTrue(ctl.startswith(f"{d}/thunder-ctl/"), ctl)
        self.assertIn("ControlPersist=no", argv)
        self.assertEqual(os.stat(os.path.dirname(ctl)).st_mode & 0o777, 0o700)

    async def test_ctl_path_is_short_and_per_backend(self):
        a, _, _, _ = make(FakeThunder(), backend=_backend_named("GPU 1"))
        b, _, _, _ = make(FakeThunder(), backend=_backend_named("gpu-1"),
                          datadir=a.deps.datadir)
        self.assertNotEqual(a._ctl_path(), b._ctl_path())      # slugs alike, not the path
        long, _, _, _ = make(FakeThunder(), backend=_backend_named("x" * 200))
        self.assertLessEqual(len(long._ctl_path().encode()), sshrun.CTL_PATH_MAX)

    async def test_tunnel_argv_refuses_without_instance(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        with self.assertRaises(RuntimeError):
            c._tunnel_argv()

    async def test_bid_and_url(self):
        c, _, _, _ = make(FakeThunder())
        self.assertEqual(c.service_bids(), ["comfyui:thunder"])
        self.assertTrue(c.has_service("comfyui:thunder"))
        self.assertEqual(c._svc_url(c.services[0]), "http://127.0.0.1:18188")



def _ready_snap(fake, name="aihub-thunder-20260926t120000z", sid="s9", min_gb=120, created=9):
    fake.snaps.append({"id": sid, "name": name, "status": "READY",
                       "minimumDiskSizeGb": min_gb, "createdAt": created})


def _creates(fake):
    return [b for m, p, b in fake.calls if p == "/instances/create"]


def _sv(c, bid=BID):
    """A service's row of the view: {status, error, …}."""
    return c.view()["services"][bid]


def _ssh_cmds(fake):
    return [p for m, p, _ in fake.calls if m == "SSH"]


class Start(unittest.IsolatedAsyncioTestCase):
    """Spec "Start" steps 0–6 (starting → syncing → ready)."""

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
        # R-W3: the host bootstrap first, then the ComfyUI part
        self.assertEqual(boot, ["bash -o pipefail -c "
                                "'bash -s -- 2>&1 | tee ~/gw-host-bootstrap.log'",
                                "bash -o pipefail -c "
                                f"'bash -s -- {COMMIT} 2>&1 | tee ~/gw-bootstrap.log'"])
        self.assertIn(hostctl._start_cmd(8188), cmds)
        self.assertTrue(hostctl._start_cmd(8188).endswith(
            "setsid nohup ~/start-comfy.sh 8188 >/dev/null 2>&1 < /dev/null &"))
        # "off" first: template + request time are persisted before the create
        self.assertEqual(c.h.phases, ["off", "creating", "connecting", "bootstrapping",
                                      "starting", "syncing", "ready"])
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
        host = [i for i, (cmd, _) in enumerate(remote) if HOST_BOOT in cmd]
        run = [i for i, (cmd, _) in enumerate(remote) if "tee ~/gw-bootstrap.log" in cmd]
        self.assertEqual((len(up), len(host), len(run)), (1, 1, 1))
        # the list goes up before the HOST bootstrap: its template report reads it
        self.assertLess(up[0], host[0])
        self.assertLess(host[0], run[0])
        self.assertEqual(remote[up[0]][1], b"https://github.com/a/pack@0123abc\n")
        self.assertEqual(remote[host[0]][1], HOST_SCRIPT)
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
        c.cfg["nodes"] = []
        await c.start()
        up = [stdin for argv, stdin in calls if argv[-1] == "cat > ~/.gw-nodes.txt"]
        self.assertEqual(up, [b"# default\nregistry:x@1.0\n"])

    async def test_no_node_list_at_all_refuses_before_create(self):
        fake = FakeThunder()
        c, _, enabled, _ = make(fake)
        c.cfg["nodes"] = []
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
                                      "starting", "syncing", "ready"])

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
        # Ruling M4 (g): a failing ComfyUI bootstrap is that SERVICE's `setup failed` —
        # the host goes `ready` (instance kept, other services would run), ComfyUI is
        # never started, and the snapshot of this host stays marked incomplete
        fake = FakeThunder()
        c, saved, _, _ = make(fake, ssh_script={"bash -s": (3, b"GW:SMOKE fail cumesh\n", b"")})
        await c.start()
        self.assertEqual((c.state.phase, c.state.error), ("ready", ""))
        self.assertEqual(_sv(c)["status"], "setup failed")
        self.assertIn("cumesh", _sv(c)["error"])
        self.assertIn("0", fake.instances)
        self.assertEqual(saved["thunder"]["uuid"], "u0")
        self.assertTrue(saved["thunder"]["bootstrap_incomplete"])
        self.assertFalse([x for x in _ssh_cmds(fake) if "start-comfy.sh" in x])
        self.assertEqual(len(c.h.faults), 1)
        self.assertEqual((c.h.faults[0][0]["name"], c.h.faults[0][0]["type"]),
                         ("thunder", "comfyui"))

    async def test_node_fail_fails_even_when_done(self):
        # Ruling 9: a missing node pack makes workflows fail later with a plausible error
        out = (b"GW:PHASE nodes\nGW:NODE_FAIL ComfyUI-Foo pip-failed\nGW:SMOKE ok\n"
               b"GW:DONE\n")
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"bash -s": (0, out, b"")})
        await c.start()
        self.assertEqual(c.state.phase, "ready")
        self.assertEqual(_sv(c)["status"], "setup failed")
        self.assertIn("ComfyUI-Foo", _sv(c)["error"])
        self.assertFalse([x for x in _ssh_cmds(fake) if "start-comfy.sh" in x])

    async def test_bootstrap_nonzero_rc_fails(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"bash -s": (1, b"GW:PHASE venv\n",
                                                        b"x\nbootstrap: cannot build the venv\n")})
        await c.start()
        self.assertEqual((c.state.phase, _sv(c)["status"]), ("ready", "setup failed"))
        self.assertIn("rc 1", _sv(c)["error"])
        self.assertIn("venv", _sv(c)["error"])

    async def test_bootstrap_without_done_fails(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"bash -s": (0, b"GW:PHASE nodes\n", b"")})
        await c.start()
        self.assertEqual((c.state.phase, _sv(c)["status"]), ("ready", "setup failed"))

    async def test_bootstrap_reports_are_collected_and_persisted(self):
        # the template's models and packs come from the HOST bootstrap (R-W3); the
        # ComfyUI bootstrap after it (no such lines) must not wipe them
        out = (b"GW:PHASE inventory\nsome noise\n"
               b"GW:TEMPLATE_NODE ComfyUI-Manager\n"
               b"GW:UNKNOWN_MODEL models/checkpoints/sd15.safetensors\t2132625894\n"
               b"GW:UNKNOWN_MODEL models/../../.ssh/id\t5\n"
               b"GW:DONE\n")
        fake = FakeThunder()
        c, saved, _, _ = make(fake, ssh_script={HOST_BOOT: (0, out, b"")})
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
        c2 = again(c)
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

    async def test_comfy_never_answers_is_the_services_down(self):
        async def probe(url):
            raise httpx.ConnectError("refused")      # a raising probe counts as "not yet"
        fake = FakeThunder()
        _ready_snap(fake)
        c, _, _, _ = make(fake, probe=probe)
        await c.start()
        self.assertEqual(c.state.phase, "ready")
        self.assertEqual(_sv(c)["status"], "down")
        self.assertIn("did not answer", _sv(c)["error"])
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
            c.cfg["comfy_commit"] = bad
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
        self.assertEqual((c.state.phase, _sv(c)["status"]), ("ready", "setup failed"))
        self.assertIn("timed out", _sv(c)["error"])
        self.assertIn("phase extensions", _sv(c)["error"])
        self.assertTrue(any("building cumesh wheel" in ln for ln in c.state.log))
        self.assertIn("tail -n 200 ~/gw-bootstrap.log", _ssh_cmds(fake))

    async def test_bootstrap_dropped_connection_reads_the_log(self):
        tail = b"GW:PHASE venv\nbootstrap: cannot build the venv\n"
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"bash -s": (255, b"", b"Connection reset"),
                                            "tail -n 200": (0, tail, b"")})
        await c.start()
        self.assertIn("rc 255 in phase venv", _sv(c)["error"])
        self.assertIn("cannot build the venv", _sv(c)["error"])

    async def test_bootstrap_output_present_needs_no_log_fetch(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, ssh_script={"bash -s": (1, b"GW:PHASE venv\nbootstrap: x\n", b"")})
        await c.start()
        self.assertIn("rc 1 in phase venv: bootstrap: x", _sv(c)["error"])
        self.assertFalse([x for x in _ssh_cmds(fake) if x.startswith("tail ")])

    async def test_missing_start_script_fails_fast_with_a_reason(self):
        fake = FakeThunder()
        _ready_snap(fake)
        c, _, _, _ = make(fake, ssh_script={"test -x ~/start-comfy.sh":
                                            (3, b"", b"start-comfy.sh missing\n")})
        await c.start()
        self.assertEqual((c.state.phase, _sv(c)["status"]), ("ready", "down"))
        self.assertIn("start-comfy.sh missing", _sv(c)["error"])
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
        return again(c), saved

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
        self.assertEqual(cmds[0], hostctl._restart_cmd(8188))
        self.assertTrue(cmds[0].endswith("; " + hostctl._start_cmd(8188)))
        self.assertEqual(c.h.phases[-2:], ["starting", "ready"])

    async def test_pkill_pattern_matches_comfy_but_not_the_remote_shell(self):
        # The command line to match is the one start-comfy.sh really runs — read from
        # the heredoc the bootstrap writes, not copied by hand (its port is the loop's
        # argument now, any port matches). The remote `bash -c "<cmd>"` carries the
        # pattern in ITS command line: a plain pattern would kill the shell running the
        # restart before it starts the loop.
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
        self.assertEqual(argv[argv.index("--port") + 1], "$PORT")
        argv[argv.index("--port") + 1] = "8190"
        comfy = " ".join(["/home/ubuntu/ComfyUI/venv/bin/python"] + argv[1:])
        self.assertIn("--listen 127.0.0.1", comfy)
        pat = re.search(r"pkill -f '([^']*)'", hostctl._restart_cmd(8188)).group(1)
        self.assertTrue(re.search(pat, comfy), comfy)
        self.assertFalse(re.search(pat, "bash -c " + hostctl._restart_cmd(8188)))
        self.assertFalse(re.search(pat, "bash -c " + shlex.quote(hostctl._restart_cmd(8188))))
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
        self.assertEqual((c.state.phase, _sv(c)["status"]), ("ready", "setup failed"))
        self.assertTrue(c.state.bootstrap_incomplete)
        # fixed by hand on the box: a restart that answers confirms the install
        await c.restart_comfy()
        self.assertEqual((c.state.phase, _sv(c)["status"]), ("ready", "up"), c.state.error)
        self.assertFalse(c.state.bootstrap_incomplete)

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
        c2 = again(c)
        with self.assertRaises(RuntimeError):
            await c2.restart_comfy()

    async def test_restart_that_never_answers_is_the_services_down(self):
        # the host stays ready (other services run on); ComfyUI shows why
        answers = [True]

        async def probe(url):
            return answers[0]
        fake, c = await self._ready(probe=probe)
        answers[0] = False
        await c.restart_comfy()
        self.assertEqual(c.state.phase, "ready")
        self.assertEqual(_sv(c)["status"], "down")
        self.assertIn("did not answer", _sv(c)["error"])
        # from `failed`, a restart that fixes nothing leaves the host failed
        c.state.phase, c.state.failed_phase = "failed", "bootstrapping"
        await c.restart_comfy()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "starting"))
        self.assertIn("comfyui:thunder", c.state.error)

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


def _paths(fake, start=0):
    return [(m, p) for m, p, _ in fake.calls[start:]]


def _inst(fake, idx="0", uuid="u0", **kw):
    """A listed, RUNNING instance (no status script: the list reports what is set)."""
    fake.status_script = []
    fake.instances[idx] = dict({"uuid": uuid, "status": "RUNNING", "ip": "10.0.0.5",
                                "port": 30022, "httpPorts": [], "template": "comfy-ui"}, **kw)
    return fake.instances[idx]


def _persisted(**kw):
    return dict({"phase": "ready", "uuid": "u0", "index": "0", "ip": "10.0.0.5",
                 "port": 30022, "started_at": 1_789_990_000.0, "disk_gb": 120}, **kw)


class Stop(unittest.IsolatedAsyncioTestCase):
    """Spec "Stop" 1–4: draining → pruning → snapshotting → deleting → off."""

    async def _ready(self, **kw):
        fake = FakeThunder()
        c, saved, enabled, calls = make(fake, **kw)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        return fake, c, saved, enabled

    async def test_stop_order_drain_snapshot_delete(self):
        fake, c, saved, enabled = await self._ready()
        n, nphases = len(fake.calls), len(c.h.phases)
        await c.refresh_prices()
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(c.h.phases[nphases:], ["draining", "pruning", "snapshotting",
                                                "deleting", "off"])
        kinds = _paths(fake, n)
        snap = kinds.index(("POST", "/snapshots/create"))
        delete = kinds.index(("POST", "/instances/0/delete"))
        du = next(i for i, (m, p) in enumerate(kinds) if m == "SSH" and p == hostctl._DU_CMD)
        self.assertLess(du, snap)
        self.assertLess(snap, delete)
        body = next(b for m, p, b in fake.calls[n:] if p == "/snapshots/create")
        self.assertEqual(body["instanceId"], "0")
        self.assertRegex(body["name"], r"^aihub-thunder-\d{8}t\d{6}z$")
        self.assertEqual(fake.instances, {})
        # nothing bills any more: ids gone, session cost stops
        for k in ("uuid", "index", "ip"):
            self.assertEqual(saved["thunder"][k], "", k)
        self.assertEqual((saved["thunder"]["port"], saved["thunder"]["started_at"]), (0, 0.0))
        self.assertIsNone(c.view()["session_cost"])
        self.assertEqual(c.state.pending_snapshot, "s0")
        self.assertEqual(c.state.manifests, {"s0": {}})
        self.assertIs(enabled["comfyui:thunder"], False)
        self.assertFalse(c.h.tunnels[-1].running)
        self.assertEqual(c.h.faults, [])

    async def test_stop_waits_for_inflight(self):
        fake, c, _, _ = await self._ready()
        seq, drained = [2, 1, 0], []
        c.deps.begin_drain = lambda bid: drained.append(bid) or True

        def inflight(bid):
            v = seq.pop(0) if seq else 0
            fake.calls.append(("INFLIGHT", str(v), None))
            return v
        c.deps.inflight = inflight
        n = len(fake.calls)
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(drained, ["comfyui:thunder"])
        kinds = _paths(fake, n)
        polls = [i for i, (m, _) in enumerate(kinds) if m == "INFLIGHT"]
        self.assertEqual(len(polls), 3)                      # 2, 1, 0 → three rounds
        self.assertLess(polls[-1], kinds.index(("POST", "/snapshots/create")))
        log = "\n".join(c.state.log)
        self.assertIn("waiting for 2 job(s)", log)
        self.assertIn("waiting for 1 job(s)", log)

    async def test_drain_waits_until_the_drain_itself_is_over(self):
        fake, c, _, _ = await self._ready()
        states = [True, True, False]
        c.deps.is_draining = lambda bid: states.pop(0) if states else False
        seen = []

        async def sleep(sec):
            seen.append((sec, c.view()["waiting_jobs"]))
            c.h.clock[0] += sec
        c.deps.sleep = sleep
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        # per service now ({bid: jobs}): 0 jobs, but the drain itself is not over
        self.assertEqual(seen[:2], [(2, {BID: 0}), (2, {BID: 0})])
        self.assertIsNone(c.view()["waiting_jobs"])

    async def test_stop_measures_base_bytes(self):
        fake, c, saved, _ = await self._ready(ssh_script={"du -sb": (0, b"12345678\n", b"")})
        await c.stop()
        self.assertEqual(saved["thunder"]["base_bytes"], 12345678)
        self.assertIn(hostctl._DU_CMD, _ssh_cmds(fake))
        self.assertEqual(hostctl._DU_CMD,
                         "du -sb --exclude=ComfyUI/models --exclude=hf-cache ~ | cut -f1")

    async def test_failed_du_keeps_old_base_and_still_stops(self):
        fake, c, saved, _ = await self._ready(ssh_script={"du -sb": (255, b"", b"refused\n")})
        c.state.base_bytes = 777
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(saved["thunder"]["base_bytes"], 777)
        self.assertTrue(any("base size not measured" in ln for ln in c.state.log))

    async def test_delete_that_never_disappears_fails_not_off(self):
        fake, c, saved, _ = await self._ready()

        def handler(req):
            if req.url.path.endswith("/delete"):
                fake.calls.append((req.method, req.url.path, None))
                return httpx.Response(200, json={"message": "ok"})   # … and deletes nothing
            return fake.handler(req)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        c._api = c._client = None
        n = len(c.h.phases)
        t0 = c.h.clock[0]
        await c.stop()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "deleting"))
        self.assertNotIn("off", c.h.phases[n:])
        self.assertEqual(saved["thunder"]["uuid"], "u0")       # still known: it bills
        self.assertGreater(saved["thunder"]["started_at"], 0)
        self.assertGreaterEqual(c.h.clock[0] - t0, 5 * 60)
        self.assertLess(c.h.clock[0] - t0, 5 * 60 + 60)
        self.assertEqual(c.h.faults[-1][1:3], ("lifecycle", "error"))
        # a second stop resumes at deleting: no second snapshot
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
        c._api = c._client = None
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(len(fake.snaps), 1)

    async def test_failed_snapshot_request_resumes_with_the_same_name(self):
        fake, c, _, _ = await self._ready()
        fail = [True]

        def handler(req):
            if req.url.path == "/snapshots/create" and fail[0]:
                fail[0] = False
                return httpx.Response(500, text="snapshot service down")
            return fake.handler(req)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        c._api = c._client = None
        await c.stop()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "snapshotting"))
        self.assertIn("0", fake.instances)                    # kept: its changes are unsaved
        name = c.state.pending_snapshot_name
        c.h.clock[0] += 3600                                   # a later retry …
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual([s["name"] for s in fake.snaps], [name])   # … keeps the name

    async def test_stop_refusals(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        with self.assertRaises(RuntimeError):                  # nothing running
            await c.stop()
        saved["thunder"] = "garbage"
        c2 = again(c)
        with self.assertRaises(RuntimeError):                  # unreconciled (Note for Task 6)
            await c2.stop()
        self.assertEqual(fake.calls, [])

    async def test_restart_comfy_refused_after_a_failed_stop_step(self):
        fake, c, _, _ = await self._ready()
        c.state.phase, c.state.failed_phase = "failed", "deleting"
        with self.assertRaises(RuntimeError):
            await c.restart_comfy()

    async def test_stop_while_resume_of_an_off_backend_is_refused(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, state={"phase": "off"})
        gate = asyncio.Event()

        async def handler(req):
            await gate.wait()
            return fake.handler(req)
        c.state.pending_snapshot = "s1"
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        resume = asyncio.ensure_future(c.resume())
        while c._op != "resuming":
            await asyncio.sleep(0)
        with self.assertRaises(RuntimeError):
            await c.stop()
        gate.set()
        await resume

    async def test_concurrent_stop_is_refused(self):
        fake, c, _, _ = await self._ready()
        c.deps.inflight = lambda bid: 1                        # a job that runs on
        first = asyncio.ensure_future(c.stop())
        for _ in range(5):
            await asyncio.sleep(0)
        with self.assertRaises(RuntimeError):
            await c.stop()
        with self.assertRaises(RuntimeError):                  # and no start meanwhile
            await c.start()
        c.deps.inflight = lambda bid: 0
        await first
        self.assertEqual(c.state.phase, "off", c.state.error)

    async def test_stop_from_failed_connecting_deletes_without_snapshot(self):
        fake = FakeThunder()
        c, _, enabled, _ = make(fake, ssh_script={"true": (255, b"", b"Connection refused")})
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "connecting"))
        n = len(fake.calls)
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertNotIn(("POST", "/snapshots/create"), _paths(fake, n))
        self.assertEqual(fake.instances, {})
        self.assertIs(enabled["comfyui:thunder"], False)

    async def test_failed_bootstrap_snapshot_is_marked_and_bootstraps_again(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake, ssh_script={"bash -s": (124, b"", b"timeout")})
        await c.start()
        self.assertEqual((c.state.phase, _sv(c)["status"]), ("ready", "setup failed"))
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(saved["thunder"]["incomplete_snapshots"], ["s0"])
        self.assertTrue(any("half-finished" in ln for ln in c.state.log))
        fake.snaps[0]["status"] = "READY"
        await c.watch_snapshots()
        self.assertEqual(c.state.snapshot_id, "s0")
        # a start from it restores the snapshot AND runs the bootstrap again
        c.deps.ssh = make(fake)[0].deps.ssh                   # a bootstrap that succeeds
        fake.status_script = ["PROVISIONING", "RUNNING"]
        n = len(fake.calls)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_creates(fake)[-1]["template"], fake.snaps[0]["name"])
        self.assertTrue([p for m, p, _ in fake.calls[n:] if m == "SSH" and "bash -s" in p])
        self.assertFalse(c.state.bootstrap_incomplete)
        # … and the snapshot of THAT session is a complete one
        await c.stop()
        self.assertEqual(c.state.incomplete_snapshots, ["s0"])
        self.assertEqual(c.state.pending_snapshot, "s1")

    async def test_instance_vanished_before_snapshot_is_off_with_fault(self):
        fake, c, _, enabled = await self._ready()
        fake.instances.clear()
        n = len(fake.calls)
        await c.stop()
        self.assertEqual(c.state.phase, "off")
        self.assertIn("vanished", c.state.error)
        self.assertNotIn(("POST", "/snapshots/create"), _paths(fake, n))
        self.assertEqual(c.h.faults[-1][1:3], ("lifecycle", "instance_vanished"))
        self.assertEqual(c.state.started_at, 0.0)
        self.assertIs(enabled["comfyui:thunder"], False)

    async def test_known_hosts_file_goes_with_the_instance(self):
        fake, c, _, _ = await self._ready()
        kh = c._known_hosts_path("u0")
        with open(kh, "w") as f:
            f.write("[10.0.0.5]:30022 ssh-ed25519 AAAA\n")
        await c.stop()
        self.assertFalse(os.path.exists(kh))


class ReviewFixes(unittest.IsolatedAsyncioTestCase):
    """Fix round 1 of Task 6."""

    async def _ready(self, **kw):
        fake = FakeThunder()
        c, saved, enabled, _ = make(fake, **kw)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        return fake, c, saved, enabled

    async def test_failed_snapshot_with_the_pending_name_is_not_taken_over(self):
        # (1) taking a FAILED row over would delete the instance with no good snapshot
        fake = FakeThunder()
        _inst(fake)
        name = "aihub-thunder-20260927t100000z"
        fake.snaps.append({"id": "s5", "name": name, "status": "FAILED",
                           "minimumDiskSizeGb": 120, "createdAt": 50})
        c, _, _, _ = make(fake, state=_persisted(phase="snapshotting",
                                                 pending_snapshot_name=name))
        await c.resume()
        self.assertEqual(c.state.phase, "off", c.state.error)
        creates = [(i, b) for i, (m, p, b) in enumerate(fake.calls) if p == "/snapshots/create"]
        self.assertEqual(len(creates), 1)
        self.assertNotEqual(creates[0][1]["name"], name)
        delete = next(i for i, (m, p, _) in enumerate(fake.calls) if p.endswith("/delete"))
        self.assertLess(creates[0][0], delete)
        self.assertEqual(c.state.pending_snapshot, "s1")
        self.assertNotIn("s5", c.state.manifests)
        self.assertEqual(c.h.faults[-1][2:], ("snapshot_failed", name))

    async def test_failed_row_is_never_adopted_by_name_on_resume(self):
        fake = FakeThunder()
        fake.status_script = []
        name = "aihub-thunder-20260927t100000z"
        fake.snaps.append({"id": "s5", "name": name, "status": "FAILED",
                           "minimumDiskSizeGb": 120, "createdAt": 50})
        c, _, _, _ = make(fake, state=_persisted(phase="snapshotting",
                                                 pending_snapshot_name=name))
        await c.resume()
        self.assertEqual(c.state.phase, "off")
        self.assertEqual(c.state.pending_snapshot, "")
        self.assertIn("vanished", c.state.error)              # the session is lost: said so

    def _flaky_list(self, c, fake, empties):
        """/instances/list answers `{}` for the next `empties[0]` calls."""
        def handler(req):
            if req.url.path == "/instances/list" and empties[0] > 0:
                empties[0] -= 1
                fake.calls.append(("GET", "/instances/list", "EMPTY"))
                return httpx.Response(200, json={})
            return fake.handler(req)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        c._api = c._client = None

    async def test_one_empty_list_before_the_snapshot_does_not_forget_the_instance(self):
        # (2) parse_instances reads any odd 2xx body as [] — one such answer is no proof
        fake, c, saved, _ = await self._ready()
        self._flaky_list(c, fake, [1])
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(c.h.faults, [])
        self.assertEqual(len(fake.snaps), 1)                  # the snapshot was taken
        self.assertEqual(fake.instances, {})                  # and the instance deleted

    async def test_two_empty_lists_are_gone(self):
        fake, c, _, _ = await self._ready()
        self._flaky_list(c, fake, [2])
        await c.stop()
        self.assertEqual(c.state.phase, "off")
        self.assertIn("vanished", c.state.error)
        self.assertEqual(fake.snaps, [])

    async def test_resume_single_empty_list_is_not_vanished(self):
        fake = FakeThunder()
        _inst(fake)
        c, _, _, _ = make(fake, state=_persisted())
        self._flaky_list(c, fake, [1])
        await c.resume()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(c.state.uuid, "u0")
        self.assertEqual(c.h.faults, [])

    async def test_resume_two_empty_lists_is_vanished(self):
        fake = FakeThunder()
        _inst(fake)
        c, _, _, _ = make(fake, state=_persisted())
        self._flaky_list(c, fake, [2])
        await c.resume()
        self.assertEqual(c.state.phase, "off")
        self.assertIn("vanished", c.state.error)

    async def test_delete_precheck_sends_the_uuid_form_delete_anyway(self):
        fake, c, _, _ = await self._ready()
        c.state.phase, c.state.failed_phase = "failed", "deleting"
        fake.delete_by = "uuid"
        self._flaky_list(c, fake, [10])                       # the lists never show it
        n = len(fake.calls)
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertIn(("POST", "/instances/u0/delete"), _paths(fake, n))
        self.assertNotIn(("POST", "/instances/0/delete"), _paths(fake, n))
        self.assertEqual(fake.instances, {})                  # it did exist: now deleted

    async def test_one_empty_list_after_the_delete_is_not_yet_gone(self):
        fake, c, _, _ = await self._ready()
        c.state.phase, c.state.failed_phase = "failed", "deleting"

        def handler(req):
            if req.url.path.endswith("/delete"):
                fake.calls.append((req.method, req.url.path, None))
                return httpx.Response(200, json={})        # accepted, not done yet
            if req.url.path == "/instances/list":
                n = sum(1 for _, p, _ in fake.calls if p == "/instances/list")
                fake.calls.append(("GET", "/instances/list", None))
                if n % 3 == 1:
                    return httpx.Response(200, json={})    # every third answer is odd
                return httpx.Response(200, json=fake.instances)
            return fake.handler(req)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        c._api = c._client = None
        await c.stop()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "deleting"))

    async def test_abort_during_a_transport_failed_create_names_the_orphan_risk(self):
        # (3) same hint as the non-aborted path
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(req):
            if req.url.path == "/instances/create":
                entered.set()
                await release.wait()
                raise httpx.ReadError("connection dropped")
            return fake.handler(req)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        start = asyncio.ensure_future(c.start())
        await entered.wait()
        stop = asyncio.ensure_future(c.stop())
        for _ in range(20):
            await asyncio.sleep(0)
        release.set()
        await stop
        await start
        self.assertEqual(c.state.phase, "off")
        self.assertIn("aborted", c.state.error)
        self.assertIn("check the orphan list", c.state.error)

    async def test_unreconciled_strangers_block_start_until_gone_or_forgotten(self):
        # (4) one of them may be this backend's own: a start would make a second one
        fake = FakeThunder()
        _inst(fake, idx="4", uuid="u4")
        c, saved, _, _ = make(fake, state="garbage")
        await c.resume()
        self.assertEqual(saved["thunder"]["unreconciled_uuids"], ["u4"])
        self.assertEqual(c.view()["unreconciled_uuids"], ["u4"])
        with self.assertRaises(RuntimeError) as cm:
            await c.start()
        self.assertIn("u4", str(cm.exception))
        self.assertEqual(_creates(fake), [])
        self.assertIsNone(c._op)
        # a restart keeps the block
        c2 = again(c)
        c2._tunnel_factory = c._tunnel_factory
        with self.assertRaises(RuntimeError):
            await c2.start()
        # gone → allowed again
        del fake.instances["4"]
        fake.status_script = ["PROVISIONING", "RUNNING"]
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(saved["thunder"]["unreconciled_uuids"], [])

    async def test_forget_unreconciled_allows_start(self):
        fake = FakeThunder()
        _inst(fake, idx="4", uuid="u4")
        c, saved, _, _ = make(fake, state="garbage")
        await c.resume()
        c.forget_unreconciled()
        self.assertEqual(saved["thunder"]["unreconciled_uuids"], [])
        fake.status_script = ["PROVISIONING", "RUNNING"]
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertIn("4", fake.instances)                   # still never touched

    async def test_missing_pending_keeps_its_incomplete_mark(self):
        # (5) only a confirmed FAILED (or deleted) snapshot loses the mark
        fake = FakeThunder()
        c, saved, _, _ = make(fake, state={"phase": "off", "pending_snapshot": "s7",
                                           "incomplete_snapshots": ["s7"],
                                           "manifests": {"s7": {}}})
        for _ in range(hostctl._PENDING_MISSES):
            await c.watch_snapshots()
        self.assertEqual(c.state.pending_snapshot, "")
        self.assertEqual(saved["thunder"]["incomplete_snapshots"], ["s7"])
        # should it reappear READY, a start from it still bootstraps
        fake.snaps.append({"id": "s7", "name": "aihub-thunder-20260926t120000z",
                           "status": "READY", "minimumDiskSizeGb": 120, "createdAt": 9})
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertTrue([x for x in _ssh_cmds(fake) if "bash -s" in x])

    async def test_begin_drain_error_fails_the_stop(self):
        # (6) routing would go on and the wait could never end, with nothing saying why
        fake, c, _, _ = await self._ready()

        def boom(bid):
            raise OSError("backend list locked")
        c.deps.begin_drain = boom
        n = len(fake.calls)
        await c.stop()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "draining"))
        self.assertIn("backend list locked", c.state.error)
        self.assertNotIn(("POST", "/snapshots/create"), _paths(fake, n))
        self.assertIn("0", fake.instances)


class AbortStart(unittest.IsolatedAsyncioTestCase):
    """Ruling 13: stop() while a start() runs aborts the start and stops from the phase
    it reached."""

    async def _settle(self):
        for _ in range(20):
            await asyncio.sleep(0)

    async def test_abort_before_create_is_off_and_disabled(self):
        fake = FakeThunder()
        c, _, enabled, _ = make(fake)
        gate, entered = asyncio.Event(), asyncio.Event()

        async def keygen(path):
            entered.set()
            await gate.wait()
            return "ssh-ed25519 AAAA"
        c.deps.keygen = keygen
        start = asyncio.ensure_future(c.start())
        await entered.wait()
        self.assertIs(enabled["comfyui:thunder"], True)
        await c.stop()
        await start                                          # returns, does not raise
        self.assertEqual(c.state.phase, "off")
        self.assertIn("aborted", c.state.error)
        self.assertIs(enabled["comfyui:thunder"], False)
        self.assertEqual(_creates(fake), [])
        self.assertIsNone(c._op)
        gate.set()
        await c.start()                                      # nothing blocks a new start
        self.assertEqual(c.state.phase, "ready", c.state.error)

    async def test_abort_during_create_post_still_deletes_the_instance(self):
        # the POST was out when the stop came: the instance exists and bills — its id
        # must be learned before giving up, or nobody ever deletes it
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(req):
            r = fake.handler(req)
            if req.url.path == "/instances/create":
                entered.set()
                await release.wait()
            return r
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        start = asyncio.ensure_future(c.start())
        await entered.wait()
        stop = asyncio.ensure_future(c.stop())
        await self._settle()
        release.set()
        await stop
        await start
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(fake.instances, {})
        self.assertIn(("POST", "/instances/0/delete"), _paths(fake))
        self.assertNotIn(("POST", "/snapshots/create"), _paths(fake))

    async def test_abort_while_waiting_for_running_deletes_without_snapshot(self):
        fake = FakeThunder()
        fake.status_script = ["PROVISIONING"] * 1000
        c, _, enabled, _ = make(fake)
        start = asyncio.ensure_future(c.start())
        while c.state.phase != "creating":
            await asyncio.sleep(0)
        await c.stop()
        await start
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(fake.instances, {})
        self.assertNotIn(("POST", "/snapshots/create"), _paths(fake))
        self.assertIs(enabled["comfyui:thunder"], False)
        self.assertEqual(c.h.faults, [])

    async def test_abort_during_bootstrap_snapshots_it_marked(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        orig, running = c.deps.ssh, []

        async def ssh(argv, stdin=None, timeout=60):
            if "bash -s" in argv[-1]:
                running.append(True)
                try:
                    await asyncio.Event().wait()             # a bootstrap that takes hours
                finally:
                    running.pop()                            # sshrun.run kills its ssh here
            return await orig(argv, stdin=stdin, timeout=timeout)
        c.deps.ssh = ssh
        start = asyncio.ensure_future(c.start())
        while not running:
            await asyncio.sleep(0)
        self.assertEqual(c.state.phase, "bootstrapping")
        await c.stop()
        await start
        self.assertEqual(running, [])                        # the bootstrap's ssh ended
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(c.state.incomplete_snapshots, ["s0"])
        self.assertEqual(fake.instances, {})

    async def test_abort_restart_comfy(self):
        fake = FakeThunder()
        _ready_snap(fake)
        answers = [True]

        async def probe(url):
            return answers[0]
        c, _, _, _ = make(fake, probe=probe)
        await c.start()
        answers[0] = False                                   # the restart hangs probing
        restart = asyncio.ensure_future(c.restart_comfy())
        while c.state.phase != "starting":
            await asyncio.sleep(0)
        await c.stop()
        await restart
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(fake.instances, {})

    async def test_outer_cancel_is_not_swallowed(self):
        # a gateway shutdown cancels the start task itself: that must propagate
        fake = FakeThunder()
        fake.status_script = ["PROVISIONING"] * 1000
        c, _, _, _ = make(fake)
        start = asyncio.ensure_future(c.start())
        while c.state.phase != "creating":
            await asyncio.sleep(0)
        start.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await start
        self.assertEqual(c.state.phase, "creating")          # resume() picks it up
        self.assertIsNone(c._op)


class Resume(unittest.IsolatedAsyncioTestCase):
    """Spec "Nach Gateway-Neustart": the persisted state against /instances/list."""

    async def test_resume_mid_snapshotting_does_not_double_snapshot(self):
        fake = FakeThunder()
        _inst(fake)
        name = "aihub-thunder-20260927t100000z"
        fake.snaps.append({"id": "s5", "name": name, "status": "CREATING",
                           "minimumDiskSizeGb": 120, "createdAt": 50})
        # a restart between the snapshot POST and persisting its id
        c, saved, enabled, _ = make(fake, state=_persisted(
            phase="snapshotting", pending_snapshot_name=name))
        await c.resume()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertNotIn(("POST", "/snapshots/create"), _paths(fake))
        self.assertEqual(len(fake.snaps), 1)
        self.assertEqual(c.state.pending_snapshot, "s5")
        self.assertEqual(fake.instances, {})
        self.assertIs(enabled["comfyui:thunder"], False)

    async def test_resume_mid_snapshotting_with_id_skips_to_delete_idempotently(self):
        fake = FakeThunder()
        _inst(fake)
        name = "aihub-thunder-20260927t100000z"
        fake.snaps.append({"id": "s5", "name": name, "status": "CREATING",
                           "minimumDiskSizeGb": 120, "createdAt": 50})
        c, _, _, _ = make(fake, state=_persisted(
            phase="snapshotting", pending_snapshot="s5", pending_snapshot_name=name))
        await c.resume()
        self.assertEqual(c.state.phase, "off")
        self.assertEqual(len(fake.snaps), 1)

    async def test_resume_finds_instance_by_index_when_uuid_unknown(self):
        # Review focus 1 under Ruling 10: create answered without a uuid, the gateway
        # restarted mid-stop; the index is used only because the item there carries the
        # template we asked for, created around our request — its uuid is adopted
        fake = FakeThunder()
        _inst(fake, idx="3", uuid="u3", template="comfy-ui", createdAt=1_790_000_010)
        c, saved, _, _ = make(fake, state=_persisted(
            phase="pruning", uuid="", index="3", created_template="comfy-ui",
            create_requested_at=1_790_000_000.0))
        await c.resume()
        self.assertEqual(c.state.phase, "off", c.state.error)
        kinds = _paths(fake)
        self.assertIn(("POST", "/snapshots/create"), kinds)
        self.assertIn(("POST", "/instances/u3/delete"), kinds)   # by the ADOPTED uuid
        self.assertEqual(fake.instances, {})

    async def test_resume_never_takes_a_stranger_at_our_index(self):
        # Thunder reuses indices: our uuid is gone, somebody else's instance sits at
        # our old index — it is never snapshotted or deleted (Ruling 10)
        fake = FakeThunder()
        _inst(fake, idx="0", uuid="u9")
        c, _, _, _ = make(fake, state=_persisted(phase="draining"))
        await c.resume()
        self.assertEqual(c.state.phase, "off")
        self.assertIn("vanished", c.state.error)
        self.assertIn("0", fake.instances)
        kinds = _paths(fake)
        self.assertFalse([p for m, p in kinds if p.endswith("/delete")
                          or p == "/snapshots/create"])
        self.assertEqual(_ssh_cmds(fake), [])                # not even an ssh to it
        self.assertEqual(c.h.phases, ["off"])                # no stop step was entered
        self.assertEqual(c.h.faults[-1][1:3], ("lifecycle", "instance_vanished"))

    async def test_resume_ready_closes_ports_again(self):
        # Review focus 2: the port guard also runs after a restart
        fake = FakeThunder()
        _inst(fake, httpPorts=[8188])
        c, _, _, _ = make(fake, state=_persisted())
        await c.resume()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        kinds = _paths(fake)
        patch = kinds.index(("PATCH", "/instances/0/ports"))
        self.assertLess(patch, kinds.index(("TUNNEL", "start")))
        self.assertEqual(fake.instances["0"]["httpPorts"], [])
        self.assertTrue(c.h.tunnels[-1].running)

    async def test_resume_ready_with_port_that_stays_open_fails(self):
        fake = FakeThunder()
        _inst(fake, httpPorts=[8188])
        fake.ignore_port_remove = True
        c, _, _, _ = make(fake, state=_persisted())
        await c.resume()
        self.assertEqual(c.state.phase, "failed")
        self.assertEqual(c.h.tunnels, [])

    async def test_resume_ready_restarts_comfy_that_does_not_answer(self):
        fake = FakeThunder()
        _inst(fake)
        restarted = []

        async def probe(url):
            return bool(restarted)
        c, _, _, _ = make(fake, probe=probe, state=_persisted())
        orig = c.deps.ssh

        async def ssh(argv, stdin=None, timeout=60):
            if "pkill" in argv[-1]:
                restarted.append(True)
            return await orig(argv, stdin=stdin, timeout=timeout)
        c.deps.ssh = ssh
        await c.resume()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(restarted, [True])

    async def test_resume_ready_that_answers_needs_no_restart(self):
        fake = FakeThunder()
        _inst(fake)
        c, _, _, _ = make(fake, state=_persisted())
        await c.resume()
        self.assertEqual(c.state.phase, "ready")
        self.assertEqual(_ssh_cmds(fake), [])

    async def test_resume_vanished_instance_is_off_with_fault(self):
        fake = FakeThunder()
        fake.status_script = []
        c, saved, enabled, _ = make(fake, state=_persisted())
        await c.resume()
        self.assertEqual(c.state.phase, "off")
        self.assertIn("vanished", c.state.error)
        self.assertEqual(saved["thunder"]["uuid"], "")
        self.assertEqual(saved["thunder"]["started_at"], 0.0)
        self.assertEqual(c.h.faults[-1][1:3], ("lifecycle", "instance_vanished"))
        self.assertIs(enabled["comfyui:thunder"], False)

    async def test_resume_after_the_delete_is_plain_off(self):
        fake = FakeThunder()
        fake.status_script = []
        fake.snaps.append({"id": "s5", "name": "aihub-thunder-20260927t100000z",
                           "status": "CREATING", "minimumDiskSizeGb": 120, "createdAt": 50})
        c, _, _, _ = make(fake, state=_persisted(phase="deleting", pending_snapshot="s5"))
        await c.resume()
        self.assertEqual((c.state.phase, c.state.error), ("off", ""))
        self.assertEqual(c.h.faults, [])
        self.assertEqual(c.state.pending_snapshot, "s5")      # still watched

    async def test_resume_off_runs_the_watcher(self):
        fake = FakeThunder()
        _ready_snap(fake, "aihub-thunder-20260925t120000z", "s1", created=5)
        _ready_snap(fake, "aihub-thunder-20260926t120000z", "s2", created=9)
        c, _, _, _ = make(fake, state={"phase": "off", "snapshot_id": "s1",
                                       "pending_snapshot": "s2",
                                       "manifests": {"s1": {}, "s2": {}}})
        await c.resume()
        self.assertEqual(c.state.snapshot_id, "s2")
        self.assertEqual([s["id"] for s in fake.snaps], ["s2"])
        self.assertNotIn("/instances/list", [p for _, p in _paths(fake)])

    async def test_resume_continues_an_interrupted_restore(self):
        fake = FakeThunder()
        _inst(fake, template="aihub-thunder-20260926t120000z")
        c, _, _, _ = make(fake, state=_persisted(phase="restoring", ip="", port=0))
        await c.resume()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertIn(hostctl._start_cmd(8188), _ssh_cmds(fake))
        self.assertFalse([x for x in _ssh_cmds(fake) if "bash -s" in x])

    async def test_resume_interrupted_first_start_runs_the_bootstrap(self):
        fake = FakeThunder()
        _inst(fake)
        c, _, _, _ = make(fake, state=_persisted(phase="creating", ip="", port=0,
                                                 bootstrap_incomplete=True))
        await c.resume()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertTrue([x for x in _ssh_cmds(fake) if "bash -s" in x])

    async def test_resume_interrupted_bootstrap_fails_and_keeps_instance(self):
        fake = FakeThunder()
        _inst(fake)
        c, _, _, _ = make(fake, state=_persisted(phase="bootstrapping",
                                                 bootstrap_incomplete=True))
        await c.resume()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "bootstrapping"))
        self.assertIn("0", fake.instances)
        self.assertTrue(c.h.tunnels[-1].running)             # Restart ComfyUI can work
        self.assertFalse([x for x in _ssh_cmds(fake) if "bash -s" in x])
        await c.stop()                                       # its snapshot is marked
        self.assertEqual(c.state.incomplete_snapshots, ["s0"])

    async def test_resume_failed_reattaches_but_stays_failed(self):
        fake = FakeThunder()
        _inst(fake, httpPorts=[8188])
        c, _, _, _ = make(fake, state=_persisted(phase="failed", failed_phase="starting",
                                                 error="x"))
        await c.resume()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "starting"))
        self.assertEqual(fake.instances["0"]["httpPorts"], [])
        self.assertTrue(c.h.tunnels[-1].running)

    async def test_resume_failed_stop_continues_the_stop(self):
        fake = FakeThunder()
        _inst(fake)
        c, _, _, _ = make(fake, state=_persisted(phase="failed", failed_phase="deleting",
                                                 pending_snapshot="s5"))
        await c.resume()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(fake.instances, {})

    async def test_resume_list_error_is_retried_by_run_forever(self):
        fake = FakeThunder()
        _inst(fake)
        down = [True]

        def handler(req):
            if req.url.path == "/instances/list" and down[0]:
                return httpx.Response(503, text="maintenance")
            return fake.handler(req)
        c, _, _, _ = make(fake, state=_persisted())
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await c.resume()
        self.assertEqual(c.state.phase, "ready")               # untouched …
        self.assertEqual(c.h.tunnels, [])                      # … and not yet reattached
        down[0] = False
        await _one_round(c)
        self.assertTrue(c.h.tunnels and c.h.tunnels[-1].running)

    async def test_unreadable_state_reconciles_to_off_without_strangers(self):
        fake = FakeThunder()
        fake.status_script = []
        c, saved, _, _ = make(fake, state="garbage")
        self.assertTrue(c.persist_blocked)
        await c.resume()
        self.assertFalse(c.persist_blocked)
        self.assertEqual(saved["thunder"]["phase"], "off")

    async def test_unreadable_state_with_strangers_is_failed_without_instance(self):
        fake = FakeThunder()
        _inst(fake, idx="4", uuid="u4")
        c, saved, _, _ = make(fake, state="garbage")
        c.deps.known_uuids = lambda: {"u-other"}
        await c.resume()
        self.assertFalse(c.persist_blocked)
        self.assertEqual((c.state.phase, c.state.uuid, c.state.index), ("failed", "", ""))
        self.assertIn("u4", c.state.error)
        self.assertNotEqual(c.state.failed_phase, hostctl.LOAD_FAILED)
        self.assertEqual([o["uuid"] for o in c.view()["orphans"]], ["u4"])
        self.assertIn("4", fake.instances)                    # never deleted
        await c.stop()                                         # nothing of ours to stop
        self.assertEqual(c.state.phase, "off")
        self.assertIn("4", fake.instances)

    async def test_unreadable_state_reread_on_resume(self):
        fake = FakeThunder()
        _inst(fake)
        c0, saved, _, _ = make(fake)
        record = _persisted()
        tries = [0]

        def load(name):
            tries[0] += 1
            if tries[0] == 1:
                raise OSError("store locked")
            return record
        c0.deps.load_state = load
        c = again(c0)
        c._tunnel_factory = c0._tunnel_factory
        c.h = c0.h
        self.assertTrue(c.persist_blocked)
        await c.resume()
        self.assertFalse(c.persist_blocked)
        self.assertEqual((c.state.phase, c.state.uuid), ("ready", "u0"))

    async def test_resume_unreconciled_list_error_stays_blocked(self):
        fake = FakeThunder()

        def handler(req):
            return httpx.Response(503, text="down")
        c, saved, _, _ = make(fake, state="garbage")
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await c.resume()
        self.assertTrue(c.persist_blocked)
        self.assertEqual(saved["thunder"], "garbage")
        with self.assertRaises(RuntimeError):
            await c.start()


async def _one_round(c):
    """One 60 s run_forever round: its 5 s ticks pass until the 60 s work ran, the next
    tick ends the loop."""
    orig, n = c.deps.sleep, [0]
    ticks = hostctl._WATCH_S // hostctl._SYNC_POLL_S

    async def sleep(sec):
        if sec == hostctl._SYNC_POLL_S:
            n[0] += 1
            if n[0] > ticks:
                raise asyncio.CancelledError()
        await orig(sec)
    c.deps.sleep = sleep
    try:
        await c.run_forever()
    except asyncio.CancelledError:
        pass
    finally:
        c.deps.sleep = orig


class Watcher(unittest.IsolatedAsyncioTestCase):
    """Spec "Stop" 4: the pending snapshot → READY (rotation) or FAILED (fault)."""

    async def test_rotation_after_ready(self):
        fake = FakeThunder()
        _ready_snap(fake, "aihub-thunder-20260924t120000z", "s1", created=3)
        _ready_snap(fake, "aihub-thunder-20260925t120000z", "s2", created=5)
        fake.snaps.append({"id": "s3", "name": "aihub-thunder-20260926t120000z",
                           "status": "CREATING", "minimumDiskSizeGb": 120, "createdAt": 9})
        fake.snaps.append({"id": "x1", "name": "aihub-other-20260920t120000z",
                           "status": "READY", "minimumDiskSizeGb": 120, "createdAt": 1})
        c, saved, _, _ = make(fake, state={"phase": "off", "snapshot_id": "s2",
                                           "pending_snapshot": "s3",
                                           "manifests": {"s1": {}, "s2": {}, "s3": {"m": 1}}})
        await c.watch_snapshots()                             # still CREATING
        self.assertEqual(c.state.pending_snapshot, "s3")
        self.assertEqual(len(fake.snaps), 4)
        fake.snaps[2]["status"] = "READY"
        await c.watch_snapshots()
        self.assertEqual((c.state.snapshot_id, c.state.pending_snapshot), ("s3", ""))
        self.assertEqual(sorted(s["id"] for s in fake.snaps), ["s3", "x1"])
        self.assertEqual(saved["thunder"]["manifests"], {"s3": {"m": 1}})

    async def test_failed_snapshot_keeps_old_and_falls_back_manifest(self):
        fake = FakeThunder()
        _ready_snap(fake, "aihub-thunder-20260925t120000z", "s2", created=5)
        fake.snaps.append({"id": "s3", "name": "aihub-thunder-20260926t120000z",
                           "status": "FAILED", "minimumDiskSizeGb": 120, "createdAt": 9})
        c, saved, _, _ = make(fake, state={"phase": "off", "snapshot_id": "s2",
                                           "pending_snapshot": "s3",
                                           "incomplete_snapshots": ["s3"],
                                           "manifests": {"s2": {"a": 1}, "s3": {"b": 2}}})
        await c.watch_snapshots()
        self.assertEqual((c.state.snapshot_id, c.state.pending_snapshot), ("s2", ""))
        self.assertEqual(saved["thunder"]["manifests"], {"s2": {"a": 1}})
        self.assertEqual(saved["thunder"]["incomplete_snapshots"], [])
        self.assertEqual([s["id"] for s in fake.snaps], ["s2", "s3"])   # s2 stays
        self.assertEqual(c.h.faults[-1][1:], ("lifecycle", "snapshot_failed",
                                              "aihub-thunder-20260926t120000z"))
        self.assertEqual(thunder.newest_ready(await c.api.snapshots(), "thunder")["id"], "s2")

    async def test_new_snapshot_without_created_at_is_never_rotated_away(self):
        fake = FakeThunder()
        _ready_snap(fake, "aihub-thunder-20260925t120000z", "s2", created=5)
        fake.snaps.append({"id": "s3", "name": "aihub-thunder-20260926t120000z",
                           "status": "READY", "minimumDiskSizeGb": 120})
        c, _, _, _ = make(fake, state={"phase": "off", "pending_snapshot": "s3"})
        await c.watch_snapshots()
        self.assertEqual(c.state.snapshot_id, "s3")
        self.assertIn("s3", [s["id"] for s in fake.snaps])

    async def test_pending_missing_from_the_list_fails_after_some_rounds(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, state={"phase": "off", "pending_snapshot": "s7",
                                       "pending_snapshot_name": "aihub-thunder-x",
                                       "manifests": {"s7": {}}})
        for _ in range(hostctl._PENDING_MISSES - 1):
            await c.watch_snapshots()
            self.assertEqual(c.state.pending_snapshot, "s7")
        await c.watch_snapshots()
        self.assertEqual(c.state.pending_snapshot, "")
        self.assertEqual(c.state.manifests, {})
        self.assertEqual(c.h.faults[-1][2:], ("snapshot_failed", "aihub-thunder-x"))

    async def test_list_error_changes_nothing(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, state={"phase": "off", "pending_snapshot": "s7"})
        c.deps.client_factory = lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(502)))
        await c.watch_snapshots()
        self.assertEqual(c.state.pending_snapshot, "s7")
        self.assertEqual(c.h.faults, [])

    async def test_rotation_delete_error_is_logged(self):
        fake = FakeThunder()
        _ready_snap(fake, "aihub-thunder-20260925t120000z", "s2", created=5)
        _ready_snap(fake, "aihub-thunder-20260926t120000z", "s3", created=9)
        c, _, _, _ = make(fake, state={"phase": "off", "pending_snapshot": "s3",
                                       "manifests": {"s2": {}, "s3": {}}})

        def handler(req):
            if req.method == "DELETE":
                return httpx.Response(500, text="nope")
            return fake.handler(req)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await c.watch_snapshots()
        self.assertEqual(c.state.snapshot_id, "s3")
        self.assertIn("s2", c.state.manifests)                # not deleted → kept
        self.assertTrue(any("deleting snapshot s2 failed" in ln for ln in c.state.log))

    async def test_stop_recording_a_newer_pending_meanwhile_wins(self):
        # the watcher awaits the list while a stop records the NEXT pending snapshot:
        # the READY answer about the old one must not promote the new id
        fake = FakeThunder()
        _ready_snap(fake, "aihub-thunder-20260926t120000z", "s3", created=9)
        c, _, _, _ = make(fake, state={"phase": "off", "snapshot_id": "s1",
                                       "pending_snapshot": "s3"})
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(req):
            if req.url.path == "/snapshots/list":
                entered.set()
                await release.wait()
            return fake.handler(req)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        watch = asyncio.ensure_future(c.watch_snapshots())
        await entered.wait()
        c.state.pending_snapshot = "s9"
        release.set()
        await watch
        self.assertEqual((c.state.snapshot_id, c.state.pending_snapshot), ("s1", "s9"))

    async def test_run_forever_watches_while_pending(self):
        fake = FakeThunder()
        _ready_snap(fake, "aihub-thunder-20260926t120000z", "s3", created=9)
        c, _, _, _ = make(fake, state={"phase": "off", "pending_snapshot": "s3"})
        await _one_round(c)
        self.assertEqual(c.state.snapshot_id, "s3")
        n = len(fake.calls)
        await _one_round(c)                                   # nothing pending: no calls
        self.assertEqual(len(fake.calls), n)


class Orphans(unittest.IsolatedAsyncioTestCase):
    async def test_orphans_are_unowned_instances_only_and_never_deleted(self):
        fake = FakeThunder()
        _inst(fake, idx="0", uuid="u0")
        _inst(fake, idx="1", uuid="u1")
        _inst(fake, idx="2", uuid="u2")
        _inst(fake, idx="3", uuid="u3", status="DELETED")
        c, _, _, _ = make(fake, state=_persisted())
        c.deps.known_uuids = lambda: {"u1"}                    # another backend's
        out = await c.orphans()
        self.assertEqual([o["uuid"] for o in out], ["u2"])
        self.assertEqual([o["uuid"] for o in c.view()["orphans"]], ["u2"])
        self.assertEqual([m for m, _ in _paths(fake)], ["GET"])


class CostGuard(unittest.IsolatedAsyncioTestCase):
    """Spec "Fehlerbehandlung": phase != off for more than 24 h → a warning banner. An
    instance forgotten over a weekend bills ~60 h unseen; the flag is what the panel and
    the Dashboard key the banner on, judged on the controller's own clock."""

    def test_long_running_after_24h(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, state=_persisted())
        now = c.h.clock[0]
        c.state.started_at = now - 25 * 3600
        self.assertTrue(c.view()["long_running"])
        c.state.started_at = now - 23 * 3600
        self.assertFalse(c.view()["long_running"])
        c.state.started_at = 0.0                           # no session: never a banner
        self.assertFalse(c.view()["long_running"])
        c.state.started_at, c.state.phase = now - 30 * 3600, "off"
        self.assertFalse(c.view()["long_running"])
        c.state.phase = "failed"                           # a failed instance still bills
        self.assertTrue(c.view()["long_running"])
        for ph in ("draining", "pruning", "snapshotting", "deleting", "starting"):
            c.state.phase = ph                             # a stop in flight still bills
            self.assertTrue(c.view()["long_running"], ph)


class AccountRefresh(unittest.IsolatedAsyncioTestCase):
    """run_forever keeps the account view current (price list, snapshot list, foreign
    instances) every 10 min — the console only ever renders caches, so without this the
    orphan warnings and the $/h figure would show only what a start/stop happened to
    fetch (the instance orphans: never)."""

    async def test_refreshed_every_10_min_not_per_round(self):
        fake = FakeThunder()
        _inst(fake, idx="4", uuid="stranger")
        fake.snaps.append({"id": "x1", "name": "aihub-old-20260920t120000z", "status": "READY",
                           "minimumDiskSizeGb": 80, "createdAt": 1})
        c, _, _, _ = make(fake, state={"phase": "off"})
        await _one_round(c)
        got = {p for m, p, _ in fake.calls}
        self.assertTrue({"/instances/list", "/snapshots/list", "/v2/pricing"} <= got, got)
        self.assertEqual([o["uuid"] for o in c.view()["orphans"]], ["stranger"])
        self.assertEqual([s["id"] for s in c.snapshots()], ["x1"])
        self.assertAlmostEqual(c.pricing_table()["snapshot_gb"], 0.00006849)
        n = len(fake.calls)
        await _one_round(c)                                 # 60 s later: cached
        self.assertEqual(len(fake.calls), n)
        c.h.clock[0] += 600
        await _one_round(c)
        self.assertGreater(len(fake.calls), n)

    async def test_price_list_fetched_once_per_hour(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, state={"phase": "off"})
        pricing = lambda: sum(1 for m, p, _ in fake.calls if p == "/v2/pricing")  # noqa: E731
        c.api._clock = lambda: c.h.clock[0]                # the cache's TTL on the test clock
        await _one_round(c)
        self.assertEqual(pricing(), 1)
        for _ in range(4):                                 # 4 more refreshes within the hour
            c.h.clock[0] += 600
            await _one_round(c)
        self.assertEqual(pricing(), 1)
        self.assertEqual(sum(1 for m, p, _ in fake.calls if p == "/snapshots/list"), 5)
        c.h.clock[0] += 3600
        await _one_round(c)
        self.assertEqual(pricing(), 2)

    async def test_no_foreign_instance_check_during_an_op_or_unreconciled(self):
        fake = FakeThunder()
        _inst(fake, idx="4", uuid="ours-not-yet-persisted")
        c, _, _, _ = make(fake, state={"phase": "off"})
        c._op = "start"
        await c.refresh_account()
        self.assertEqual(c.view()["orphans"], [])
        self.assertNotIn("/instances/list", {p for m, p, _ in fake.calls})
        c._op, c._persist_blocked = None, True
        await c.refresh_account()
        self.assertNotIn("/instances/list", {p for m, p, _ in fake.calls})
        c._persist_blocked = False
        await c.refresh_account()
        self.assertEqual(len(c.view()["orphans"]), 1)

    async def test_a_failing_refresh_never_costs_the_sync_tick(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, state={"phase": "off"})
        ticks = []

        async def boom():
            raise KeyError("bug")

        async def tick():
            ticks.append(1)
        c.refresh_account, c._sync_tick = boom, tick
        await _one_round(c)
        self.assertEqual(len(ticks), hostctl._WATCH_S // hostctl._SYNC_POLL_S)
        self.assertTrue([ln for ln in c.state.log if "account refresh failed" in ln])

    async def test_foreign_instances_priced_from_their_own_config(self):
        fake = FakeThunder()
        _inst(fake, idx="4", uuid="x", gpuType="a6000", numGpus="1", cpuCores="8",
              storage=300)
        _inst(fake, idx="5", uuid="y", gpuType="h200", numGpus="1")    # no price for it
        c, _, _, _ = make(fake, state={"phase": "off"})
        await c.refresh_account()
        o = {x["uuid"]: x for x in c.view()["orphans"]}
        want = thunder.hourly_cost({"a6000_x1": 0.35, "additional_vcpus": 0.04,
                                    "disk_gb": 0.0003}, "a6000", 1, 8, 300,
                                   {"vcpuOptions": [6, 8]})
        self.assertAlmostEqual(o["x"]["cost_per_h"], want)
        self.assertIsNone(o["y"]["cost_per_h"])

    async def test_no_token_no_calls(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, state={"phase": "off"})
        c.host = dict(c.host, api_key="")
        await _one_round(c)
        self.assertEqual(fake.calls, [])
        self.assertEqual(c.snapshots(), [])
        self.assertIsNone(c.pricing_table())


class BootstrapParse(unittest.TestCase):
    def test_prefix_not_position(self):
        r = hostctl.parse_bootstrap(
            "noise GW:SMOKE ok\nGW:SMOKEY ok\nGW:PHASE smoke\n  GW:NODE_FAIL x y\n"
            "GW:SMOKE fail a,b\nGW:DONE\n")
        self.assertEqual(r["smoke"], "fail a,b")
        self.assertEqual(r["node_fails"], [])      # indented = not a GW line
        self.assertTrue(r["done"])
        self.assertEqual(r["phase"], "smoke")

    def test_unknown_model_bad_size_is_skipped(self):
        r = hostctl.parse_bootstrap("GW:UNKNOWN_MODEL models/a.bin\tlots\n"
                                       "GW:UNKNOWN_MODEL models/b c.bin\t7\n")
        self.assertEqual(r["unknown"], {"models/b c.bin": 7})

    def test_host_verdict_needs_done_not_smoke(self):
        # the host bootstrap has no smoke test: rc 0 + GW:DONE is success, and a
        # timeout names ITS budget, not the ComfyUI bootstrap's hours
        v = hostctl.bootstrap_verdict
        done = hostctl.parse_bootstrap("GW:PHASE tools\nGW:DONE\n")
        self.assertEqual(v(0, done, "", smoke_test=False), "")
        self.assertIn("without GW:DONE", v(0, hostctl.parse_bootstrap("GW:PHASE tools\n"),
                                            "", smoke_test=False))
        self.assertIn("without GW:SMOKE ok", v(0, done, ""))    # the ComfyUI rule stays
        self.assertIn("rc 3", v(3, done, "x", smoke_test=False))
        self.assertIn("after 30 min", v(124, done, "", smoke_test=False,
                                        timeout_s=hostctl._HOST_BOOTSTRAP_S))
        self.assertIn("after 3 h", v(124, done, ""))




_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script_phases(name: str) -> list[str]:
    """The `phase <name>` calls of an ops script's `main()` body, in textual order. The
    pin ASSUMES that order is the execution order: every `phase` call sits in `main()`,
    in sequence. A `phase` call moved into a helper is outside the slice — the pin then
    fails loudly instead of trusting a textual order that may no longer hold."""
    with open(os.path.join(_HERE, "ops", name), encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"^main\(\) \{\n(.*?)^\}", text, re.M | re.S)
    assert m, f"no main() in {name}"
    calls = re.findall(r"^\s*phase ([A-Za-z0-9-]+)\s*$", m.group(1), re.M)
    everywhere = re.findall(r"^\s*phase ([A-Za-z0-9-]+)\s*$", text, re.M)
    assert calls == everywhere, f"{name}: a phase call outside main()"
    return calls


class BootstrapProgressPure(unittest.TestCase):
    """Task 2 (2026-10-01, thunder-1): the bootstrap ran for 20 min while the card showed
    a red "did not finish" and nothing moved. What the poller makes of the remote log."""

    def test_phase_orders_are_the_scripts(self):
        # the progress bar counts on these: a script that gains, loses or reorders a
        # phase must fail here, not silently draw a wrong bar
        self.assertEqual(list(hostctl.HOST_BOOTSTRAP_PHASES),
                         _script_phases("host-bootstrap.sh"))
        self.assertEqual(list(hostctl.COMFY_BOOTSTRAP_PHASES),
                         _script_phases("thunder-bootstrap.sh"))
        self.assertEqual(hostctl.HOST_BOOTSTRAP_PHASES,
                         ("locate", "autostart", "inventory", "tools"))
        self.assertEqual(hostctl.COMFY_BOOTSTRAP_PHASES,
                         ("locate", "stop", "checkout", "venv", "nodes", "extensions",
                          "node-install-scripts", "start-script", "smoke"))

    def test_tail_last_phase_wins_and_gw_lines_are_skipped(self):
        t = hostctl.bootstrap_tail
        self.assertEqual(t(""), (None, ""))                 # no marker = unknown, not "starting"
        self.assertEqual(t("cloning\n"), (None, "cloning"))
        text = ("GW:PHASE checkout\nchecked out abc\nGW:PHASE venv\n"
                "Collecting torch==2.11.0\n\n   \nGW:NODE_FAIL x y\nGW:SMOKE ok\n")
        self.assertEqual(t(text), ("venv", "Collecting torch==2.11.0"))
        # an indented GW: is no GW line (parse_bootstrap's rule) — a normal line
        self.assertEqual(t("GW:PHASE nodes\n  GW:PHASE fake\n"), ("nodes", "GW:PHASE fake"))

    def test_tail_line_is_clean_and_clipped(self):
        line = "\x1b[1;32mok\x1b[0m pip\x07 says\tthis" + "x" * 300
        ph, got = hostctl.bootstrap_tail("GW:PHASE venv\n" + line + "\n")
        self.assertEqual(ph, "venv")
        self.assertTrue(got.startswith("ok pip says this"), got)
        self.assertLessEqual(len(got), 160)
        self.assertFalse(any(ord(ch) < 32 or 127 <= ord(ch) < 160 for ch in got))
        # pip's \r progress redraws: the last state counts
        self.assertEqual(hostctl.bootstrap_tail("a 10%\rb 90%\r\n")[1], "b 90%")
        # a phase name from the log is a word, never markup or a whole sentence
        ph, _ = hostctl.bootstrap_tail("GW:PHASE <b>x</b> y\n")
        self.assertIsNone(ph)

    def test_poll_command_is_fixed_text(self):
        for log in (hostctl._BOOTSTRAP_LOG, hostctl._HOST_BOOTSTRAP_LOG):
            cmd = hostctl.bootstrap_poll_cmd(log)
            self.assertEqual(cmd, hostctl._BOOTSTRAP_POLL_CMD.replace("{log}", log))
            self.assertEqual(cmd.count(log), 3)
            self.assertNotIn("{", cmd.replace("${n:-0}", "").replace("${p#GW:PHASE }", ""))
        # the template itself: a grep count and a tail of the ONE path, nothing else
        self.assertEqual(hostctl._BOOTSTRAP_POLL_CMD,
                         "n=$(grep -c '^node .* @ ' {log} 2>/dev/null); "
                         "p=$(grep '^GW:PHASE ' {log} 2>/dev/null | tail -n 1); "
                         "echo \"GW-POLL ${n:-0} ${p#GW:PHASE }\"; tail -n 40 {log}")
        for bad in ("~/x; rm -rf ~", "/etc/passwd", "~/../x", "~/a b", "~/$(id)", ""):
            with self.assertRaises(ValueError):
                hostctl.bootstrap_poll_cmd(bad)

    def test_poll_output(self):
        p = hostctl.parse_poll_output
        self.assertEqual(p("GW-POLL 5 nodes\nGW:PHASE nodes\nnode x @ 1\n"),
                         (5, "nodes", "GW:PHASE nodes\nnode x @ 1\n"))
        self.assertEqual(p("GW-POLL 0 \nx\n"), (0, None, "x\n"))     # no marker yet
        self.assertEqual(p("GW-POLL 0\nx\n"), (0, None, "x\n"))
        self.assertEqual(p("GW-POLL 3 venv\r\n"), (3, "venv", ""))
        self.assertEqual(p("GW-POLL 0 <b>x\n")[1], None)            # not a phase word
        self.assertEqual(p("GW:PHASE x\n"), (None, None, "GW:PHASE x\n"))
        self.assertEqual(p("GW-POLL lots venv\ny\n")[:2], (None, None))

    def test_progress_values(self):
        p = hostctl.bootstrap_progress
        v = p("comfyui", "venv")
        self.assertEqual((v["step"], v["steps"]), (4, 9))
        self.assertAlmostEqual(v["fraction"], 3 / 9)
        self.assertIsNone(v["nodes_done"])
        v = p("comfyui", "nodes", nodes_seen=5, nodes_total=14)
        self.assertEqual((v["step"], v["nodes_done"], v["nodes_total"]), (5, 5, 14))
        self.assertAlmostEqual(v["fraction"], (4 + 4 / 14) / 9)   # 4 packs finished
        v = p("comfyui", "nodes", nodes_seen=40, nodes_total=14)   # never past the phase
        self.assertEqual(v["nodes_done"], 14)
        self.assertLess(v["fraction"], 5 / 9 + 1e-9)
        v = p("comfyui", "nodes", nodes_seen=None, nodes_total=14)
        self.assertIsNone(v["nodes_done"])
        self.assertAlmostEqual(v["fraction"], 4 / 9)
        v = p("host", "tools")
        self.assertEqual((v["step"], v["steps"]), (4, 4))
        self.assertAlmostEqual(v["fraction"], 3 / 4)
        self.assertIsNone(p("comfyui", "starting"))     # unknown phase: no bar
        self.assertIsNone(p("comfyui", "tools"))        # another script's phase
        self.assertIsNone(p("service", "venv"))         # a setup script has no order

    def test_node_entry_rule_is_shared(self):
        # one rule for what _node_list accepts and what the bar counts (M-6)
        self.assertTrue(hostctl._is_node_entry("  https://x/y@1 "))
        self.assertFalse(hostctl._is_node_entry("  # c"))
        self.assertFalse(hostctl._is_node_entry("   "))

    def test_node_list_count(self):
        self.assertEqual(hostctl.count_node_lines(
            "# list\nhttps://github.com/a/b@1\n\n  registry:x@2  \n#c\nbad\n"), 3)
        self.assertEqual(hostctl.count_node_lines(""), 0)


class BootstrapPoller(unittest.IsolatedAsyncioTestCase):
    """The side task beside a running bootstrap: display only, never in the way."""

    def _ctl(self, poll_results, boot=(0, b"GW:PHASE smoke\nGW:SMOKE ok\nGW:DONE\n", b"")):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        c.state.uuid, c.state.ip, c.state.port = "uuid-1", "203.0.113.5", 22   # ssh-able
        self.release = asyncio.Event()
        self.polls = []

        async def ssh(argv, stdin=None, timeout=60):
            cmd = argv[-1]
            if "GW-POLL" in cmd:
                self.polls.append((cmd, timeout))
                r = poll_results.pop(0) if len(poll_results) > 1 else poll_results[0]
                if isinstance(r, BaseException):
                    raise r
                if r == "hang":
                    await asyncio.Event().wait()
                return r
            if "bash -s" in cmd:
                await self.release.wait()
                return boot
            return (0, b"", b"")
        c.deps.ssh = ssh

        async def quick(_s):
            await asyncio.sleep(0)
        c._poll_sleep = quick
        c._nodes_text = "https://github.com/a/p1@1\nhttps://github.com/a/p2@2\n#x\n"
        return c

    async def _until(self, pred, n=200):
        for _ in range(n):
            if pred():
                return True
            await asyncio.sleep(0)
        return False

    def _run(self, c, which="comfyui", log=hostctl._BOOTSTRAP_LOG):
        return asyncio.ensure_future(c._run_script(b"#!/bin/bash\n", "", log, 100,
                                                   which=which))

    async def test_progress_while_running_then_cleared(self):
        tail = (b"GW-POLL 2 nodes\nGW:PHASE venv\nGW:PHASE nodes\nnode p1 @ 1\n"
                b"node p2 @ 2\nCloning into 'p2'...\n")
        c = self._ctl([(0, tail, b"")])
        t = self._run(c)
        self.assertTrue(await self._until(
            lambda: (c.view()["bootstrap_running"] or {}).get("phase") == "nodes"))
        v = c.view()["bootstrap_running"]
        self.assertEqual(v["which"], "comfyui")
        self.assertEqual(v["line"], "Cloning into 'p2'...")
        self.assertEqual((v["step"], v["steps"]), (5, 9))
        self.assertEqual((v["nodes_done"], v["nodes_total"]), (2, 2))
        self.assertGreaterEqual(v["elapsed_s"], 0)
        await self._until(lambda: len(self.polls) >= 5)
        self.assertGreaterEqual(len(self.polls), 5)
        # each phase once in the log ring, however often it was polled
        log = c.state.log
        self.assertEqual(sum("bootstrap phase: nodes" in ln for ln in log), 1, log)
        self.assertEqual(sum("bootstrap phase: venv" in ln for ln in log), 0, log)
        # the fixed poll command, a short timeout
        cmd, timeout = self.polls[0]
        self.assertEqual(cmd, hostctl.bootstrap_poll_cmd(hostctl._BOOTSTRAP_LOG))
        self.assertLessEqual(timeout, 30)
        self.release.set()
        rc, text, _ = await t
        self.assertEqual(rc, 0)
        self.assertIsNone(c.view()["bootstrap_running"])
        self.assertIsNone(c._bs_task)
        n = len(self.polls)
        for _ in range(20):
            await asyncio.sleep(0)
        self.assertEqual(len(self.polls), n)            # the poller is gone

    async def test_phase_change_logged_once_each(self):
        c = self._ctl([(0, b"GW-POLL 0 venv\nGW:PHASE venv\nx\n", b""),
                       (0, b"GW-POLL 0 venv\nGW:PHASE venv\ny\n", b""),
                       (0, b"GW-POLL 0 nodes\nGW:PHASE venv\nGW:PHASE nodes\n", b"")])
        t = self._run(c)
        await self._until(lambda: len(self.polls) >= 6)
        log = [ln for ln in c.state.log if "bootstrap phase:" in ln]
        self.assertEqual([ln.split("bootstrap phase: ")[1] for ln in log], ["venv", "nodes"])
        self.release.set()
        await t

    async def test_a_verbose_tail_keeps_the_phase(self):
        # review I-1: `nodes` and `smoke` print more than 40 lines after their marker.
        # The phase comes from the WHOLE log (the command's own grep) and never moves
        # backwards: no "phase starting", no bar lost, no bogus ring line.
        verbose = b"".join(b"smoke: import mod%d ok\n" % i for i in range(60))
        c = self._ctl([(0, b"GW-POLL 3 nodes\nGW:PHASE nodes\nnode a @ 1\n", b""),
                       (0, b"GW-POLL 3 \n" + verbose, b""),     # marker unknown to this poll
                       (0, b"GW-POLL 3\n" + verbose, b""),      # an older header shape
                       (0, b"GW-POLL lots\n" + verbose, b"")])
        t = self._run(c)
        await self._until(lambda: len(self.polls) >= 8)
        v = c.view()["bootstrap_running"]
        self.assertEqual(v["phase"], "nodes")
        self.assertEqual((v["step"], v["steps"]), (5, 9))
        self.assertIsNotNone(v["fraction"])
        self.assertEqual(v["line"], "smoke: import mod59 ok")
        log = [ln for ln in c.state.log if "phase:" in ln]
        self.assertEqual(len(log), 1, log)
        self.assertFalse(any("starting" in ln for ln in log))
        self.release.set()
        await t

    async def test_header_phase_wins_over_an_older_window(self):
        # the whole-log phase moves on even when the tail still shows an earlier marker
        c = self._ctl([(0, b"GW-POLL 0 extensions\nGW:PHASE nodes\nx\n", b"")])
        t = self._run(c)
        self.assertTrue(await self._until(
            lambda: (c.view()["bootstrap_running"] or {}).get("phase") == "extensions"))
        self.release.set()
        await t

    async def test_a_straggling_poller_never_writes_into_the_next_run(self):
        # review M-2: a poll answer that arrives for an earlier run is dropped
        c = self._ctl([(0, b"GW-POLL 0 venv\n", b"")])
        c._bs_run = {"id": object(), "which": "comfyui", "service": "", "since": 0,
                     "phase": "starting", "line": "", "nodes_seen": None,
                     "nodes_total": None, "logged": set()}
        c._apply_poll(b"GW-POLL 0 tools\nhost line\n", object())
        self.assertEqual(c._bs_run["phase"], "starting")
        self.assertEqual(c._bs_run["line"], "")

    async def test_nodes_total_fixed_at_the_start_of_the_run(self):
        # review M-3: counted once per run, falling back to the list the bootstrap uploads
        c = self._ctl([(0, b"GW-POLL 1 nodes\nnode a @ 1\n", b"")])
        c._nodes_text = ""
        t = self._run(c)
        self.assertTrue(await self._until(
            lambda: (c.view()["bootstrap_running"] or {}).get("phase") == "nodes"))
        self.assertEqual(c.view()["bootstrap_running"]["nodes_total"],
                         hostctl.count_node_lines(c._node_list()))
        self.release.set()
        await t

    async def test_host_bootstrap_label(self):
        c = self._ctl([(0, b"GW-POLL 0 autostart\nGW:PHASE autostart\n", b"")])
        t = self._run(c, which="host", log=hostctl._HOST_BOOTSTRAP_LOG)
        self.assertTrue(await self._until(
            lambda: (c.view()["bootstrap_running"] or {}).get("phase") == "autostart"))
        v = c.view()["bootstrap_running"]
        self.assertEqual((v["which"], v["step"], v["steps"]), ("host", 2, 4))
        self.assertIsNone(v["nodes_total"])
        self.assertTrue(any("host bootstrap phase: autostart" in ln for ln in c.state.log))
        self.assertEqual(self.polls[0][0],
                         hostctl.bootstrap_poll_cmd(hostctl._HOST_BOOTSTRAP_LOG))
        self.release.set()
        await t

    async def test_failing_polls_leave_the_bootstrap_alone(self):
        for res in ([RuntimeError("ssh broke")], [(255, b"", b"refused")],
                    [(124, b"", b"timeout")], [(0, b"\xff\xfe garbage", b"")], ["hang"]):
            with self.subTest(res=res):
                c = self._ctl(list(res))
                t = self._run(c)
                await self._until(lambda: len(self.polls) >= 1)
                for _ in range(20):
                    await asyncio.sleep(0)
                v = c.view()["bootstrap_running"]
                self.assertIsNotNone(v)
                self.assertIn(v["phase"], ("starting",))
                self.assertFalse(t.done())
                self.release.set()
                rc, text, err = await t
                self.assertEqual(rc, 0)
                self.assertIn("GW:DONE", text)
                self.assertIsNone(c.view()["bootstrap_running"])
                self.assertFalse(any("ssh broke" in ln or "refused" in ln
                                     for ln in c.state.log))     # silently

    async def test_failed_bootstrap_clears_too(self):
        c = self._ctl([(0, b"GW-POLL 0 venv\nGW:PHASE venv\n", b"")],
                      boot=(1, b"GW:PHASE venv\nboom\n", b"err"))
        t = self._run(c)
        await self._until(lambda: len(self.polls) >= 2)
        self.release.set()
        rc, _, _ = await t
        self.assertEqual(rc, 1)
        self.assertIsNone(c.view()["bootstrap_running"])
        self.assertIsNone(c._bs_task)

    async def test_raising_exec_clears_too(self):
        c = self._ctl([(0, b"GW-POLL 0\n", b"")])
        orig = c.deps.ssh

        async def ssh(argv, stdin=None, timeout=60):
            if "bash -s" in argv[-1]:
                await self.release.wait()
                raise OSError("ssh binary gone")
            return await orig(argv, stdin, timeout)
        c.deps.ssh = ssh
        t = self._run(c)
        await self._until(lambda: len(self.polls) >= 1)
        self.release.set()
        with self.assertRaises(OSError):
            await t
        self.assertIsNone(c.view()["bootstrap_running"])

    async def test_cancellation_cancels_the_poller(self):
        c = self._ctl(["hang"])
        t = self._run(c)
        await self._until(lambda: len(self.polls) >= 1)
        poller = c._bs_task
        self.assertIsNotNone(poller)
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        await self._until(lambda: poller.done())
        self.assertTrue(poller.done())
        self.assertIsNone(c.view()["bootstrap_running"])
        self.assertIsNone(c._bs_task)

    async def test_stop_aborting_a_start_clears_it(self):
        # the real path: start() reaches the ComfyUI bootstrap, stop() aborts it
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        orig = c.deps.ssh
        entered, release = asyncio.Event(), asyncio.Event()

        async def ssh(argv, stdin=None, timeout=60):
            cmd = argv[-1]
            if "GW-POLL" in cmd:
                return (0, b"GW-POLL 0 venv\nGW:PHASE venv\n", b"")
            if "bash -s" in cmd and "gw-bootstrap.log" in cmd:
                entered.set()
                await release.wait()
            return await orig(argv, stdin, timeout)
        c.deps.ssh = ssh

        async def quick(_s):
            await asyncio.sleep(0)
        c._poll_sleep = quick
        start = asyncio.ensure_future(c.start())
        await asyncio.wait_for(entered.wait(), 5)
        await self._until(lambda: (c.view()["bootstrap_running"] or {}).get("phase") == "venv")
        self.assertEqual(c.view()["bootstrap_running"]["phase"], "venv")
        poller = c._bs_task
        await c.stop()
        await start
        self.assertTrue(poller.done())
        self.assertIsNone(c.view()["bootstrap_running"])

    async def test_no_poller_without_which_or_with_an_odd_log(self):
        c = self._ctl([(0, b"GW-POLL 0\n", b"")])
        t = asyncio.ensure_future(c._run_script(b"x", "", hostctl._BOOTSTRAP_LOG, 100))
        for _ in range(30):
            await asyncio.sleep(0)
        self.assertEqual(self.polls, [])
        self.assertIsNone(c.view()["bootstrap_running"])
        self.release.set()
        await t
        self.release.clear()
        t = asyncio.ensure_future(c._run_script(b"x", "", "~/a b.log", 100, which="service",
                                                label="openai:v"))
        for _ in range(30):
            await asyncio.sleep(0)
        self.assertEqual(self.polls, [])
        # display, but nothing polled for a path that is not plain
        self.assertEqual(c.view()["bootstrap_running"]["which"], "service")
        self.release.set()
        await t

    async def test_command_setup_is_labelled_with_its_service(self):
        c = self._ctl([(0, b"GW-POLL 0\npip install vllm\n", b"")])
        t = asyncio.ensure_future(c._run_script(b"x", "", "~/gw-svc-v.setup.log", 100,
                                                which="service", label="openai:v"))
        self.assertTrue(await self._until(
            lambda: (c.view()["bootstrap_running"] or {}).get("line") == "pip install vllm"))
        v = c.view()["bootstrap_running"]
        self.assertEqual((v["which"], v["service"], v["step"]), ("service", "openai:v", None))
        self.release.set()
        await t


# ── main wiring (Task 7) ──────────────────────────────────────────────────────

# ── model sync (spec "Controller": URL transfer, triggers, disk growth; "Stop" 1–2) ──

_GiB = 1024 ** 3
_TOKEN = "hf_SECRET_TOKEN_123"


class FakeVM:
    """The instance's side of the model sync, in memory. `files` holds PLAN paths
    (`models/…`, `hf-cache/…`) incl. the `.part`/`.part.lock`/`.part.log` artefacts,
    `manifest` the text of ~/.gw-modelsync.json, `avail` what `df` reports. It answers
    the controller's remote commands by their `: gw-<verb> [path] ;` tag and passes
    every other command to the harness ssh. A curl moves `chunk` bytes of its URL's
    size (`sizes`) per poll, ends with `fail[url]` on its stderr, or makes no progress
    while its URL is in `hold`; `sha[url]` is what sha256sum reports for its bytes."""

    def __init__(self, fake):
        self.fake = fake
        self.files = {}
        self.manifest = None
        self.manifest_writes = []
        self.avail = 400 * _GiB
        self.sizes, self.fail, self.sha = {}, {}, {}
        self.hold = set()
        self.cap = {}             # url -> bytes its curl stops at (a slow download)
        self.chunk = 10 ** 12
        self.procs = {}           # pid -> url
        self.lockpid = {}         # plan path -> pid in its .part.lock
        self.part_url = {}        # plan path -> url its .part came from
        self.logs = {}            # plan path -> curl stderr
        self.next_pid = 4000
        self.started = []         # (plan path, curl config on stdin)
        self.killed = []
        self.argvs = []
        self.seen = []            # (verb, controller phase) per sync command
        self.heads = []           # (plan path, curl config) per HEAD
        self.head_len = {}        # url -> Content-Length a HEAD reports instead of sizes[url]
        self.no_length = set()    # urls whose HEAD names no Content-Length
        self.content = {}         # plan path -> bytes of its LAN-streamed .part
        self.links = {}           # plan path -> symlink target text
        self.poll_fail = set()    # plan paths whose poll the instance never answers
        fake.on_modify = lambda old, new: setattr(self, "avail", self.avail + (new - old) * _GiB)

    def wrap(self, c):
        orig = c.deps.ssh

        async def ssh(argv, stdin=None, timeout=60):
            self.argvs.append(list(argv))
            cmd = argv[-1]
            if not cmd.startswith(": gw-"):
                return await orig(argv, stdin=stdin, timeout=timeout)
            self.fake.calls.append(("SSH", cmd, stdin))
            self.seen.append((cmd.split()[1], c.state.phase))
            await asyncio.sleep(0)        # a real ssh suspends: cancels land here
            return self.run(cmd, stdin)
        c.deps.ssh = ssh

    @staticmethod
    def plan_of(remote):
        return remote[len("ComfyUI/"):] if remote.startswith("ComfyUI/models/") else remote

    @staticmethod
    def remote(path):
        return "ComfyUI/" + path if path.startswith("models/") else path

    @staticmethod
    def _rm_args(toks):
        i = toks.index("rm")
        j = toks.index("--", i) + 1
        out = []
        while j < len(toks) and toks[j] not in ("&&", ";"):
            out.append(FakeVM.plan_of(toks[j]))
            j += 1
        return out

    def _start(self, path, cfg):
        url = re.search(r'^url = "(.*)"$', cfg, re.M).group(1)
        pid = self.next_pid
        self.next_pid += 1
        self.procs[pid] = url
        self.lockpid[path] = pid
        self.part_url[path] = url
        self.logs[path] = ""
        self.files.setdefault(path + ".part", 0)
        self.files[path + ".part.lock"] = 5
        self.files[path + ".part.log"] = 0
        self.started.append((path, cfg))

    def _advance(self, path):
        pid = self.lockpid.get(path)
        url = self.procs.get(pid)
        if url is None:
            return
        if url in self.fail:
            self.logs[path] = self.fail[url]
            del self.procs[pid]
        elif url not in self.hold:
            have = self.files.get(path + ".part", 0)
            n = min(self.sizes[url], have + self.chunk, self.cap.get(url, self.sizes[url]))
            self.files[path + ".part"] = n
            if n >= self.sizes[url]:
                del self.procs[pid]

    def run(self, cmd, stdin):
        toks = shlex.split(cmd)
        verb = toks[1]
        path = toks[2] if len(toks) > 2 and toks[2] != ";" else None
        text = (stdin or b"").decode()
        out = ""
        if verb == "gw-index":
            out = "".join(f"{self.remote(p)}\t{n}\n" for p, n in sorted(self.files.items()))
            out += "".join(f"L\t{self.remote(p)}\t{t}\n" for p, t in sorted(self.links.items()))
            out += "GW:END\n"
        elif verb == "gw-part":
            out = f"{self.files.get(path + '.part', 0)}\n"
        elif verb == "gw-link":
            ls = text.split("\n")
            for i in range(0, len(ls) - 1, 2):
                self.links[self.plan_of(ls[i])] = ls[i + 1]
            out = "GW:LINKED\n"
        elif verb == "gw-manifest":
            out = self.manifest if self.manifest is not None else "{}\n"
        elif verb == "gw-manifest-write":
            self.manifest = text
            self.manifest_writes.append(json.loads(text))
        elif verb == "gw-df":
            out = f"{self.avail}\n"
        elif verb == "gw-fetch":
            if self.lockpid.get(path) in self.procs:
                out = "GW:ADOPT\n"
            else:
                self._start(path, text)
                out = "GW:STARTED\n"
        elif verb == "gw-poll":
            if path in self.poll_fail:
                return (255, b"", b"ssh: connection lost")
            self._advance(path)
            state = "GW:RUN" if self.lockpid.get(path) in self.procs else "GW:END"
            out = f"{state}\n{self.files.get(path + '.part', -1)}\n{self.logs.get(path, '')}"
        elif verb == "gw-head":
            url = re.search(r'^url = "(.*)"$', text, re.M).group(1)
            self.heads.append((path, text))
            n = self.head_len.get(url, self.sizes.get(url))
            if n is None:
                return (0, b"HTTP/2 404\r\ncontent-length: 9\r\n\r\n", b"")
            length = "" if url in self.no_length else f"Content-Length: {n}\r\n"
            out = ("HTTP/2 302\r\nlocation: https://cdn.example/x\r\ncontent-length: 0\r\n"
                   f"\r\nHTTP/2 200\r\n{length}\r\n")
        elif verb == "gw-sha":
            if path in self.content:
                out = f"{hashlib.sha256(self.content[path]).hexdigest()}  x\n"
            else:
                out = f"{self.sha.get(self.part_url.get(path), '0' * 64)}  x\n"
        elif verb == "gw-done":
            n = self.files.pop(path + ".part")
            self.files[path] = n
            self.files.pop(path + ".part.lock", None)
            self.files.pop(path + ".part.log", None)
            self.lockpid.pop(path, None)
            out = f"{n}\n"
        elif verb == "gw-discard":
            for suf in (".part", ".part.lock", ".part.log"):
                self.files.pop(path + suf, None)
            self.lockpid.pop(path, None)
            self.content.pop(path, None)
        elif verb == "gw-abandon":
            pid = self.lockpid.get(path)
            if pid in self.procs:
                del self.procs[pid]
                self.killed.append(path)
            for suf in (".part", ".part.lock", ".part.log"):
                self.files.pop(path + suf, None)
            self.lockpid.pop(path, None)
            self.content.pop(path, None)
            out = "GW:ABANDONED\n"
        elif verb == "gw-kill":
            for p, pid in list(self.lockpid.items()):
                if pid in self.procs:
                    del self.procs[pid]
                    self.killed.append(p)
            out = "GW:KILLED\n"
        elif verb == "gw-prune":
            if "rm" in toks:
                for p in self._rm_args(toks):
                    self.files.pop(p, None)
                    self.links.pop(p, None)
                out += "GW:RM-OK\n"
            for p in [p for p in self.files if p.endswith((".part", ".part.lock", ".part.log"))]:
                del self.files[p]
            self.lockpid.clear()
            out += "GW:PRUNED\n"
        elif verb == "gw-delete":
            for p in self._rm_args(toks):
                self.files.pop(p, None)
            out = "GW:DELETED\n"
        else:
            return (127, b"", f"unknown verb {verb}".encode())
        return (0, out.encode(), b"")


def _wf(*files):
    """A workflow with one UNETLoader per weight file (→ models/diffusion_models/…)."""
    return {str(i): {"class_type": "UNETLoader", "inputs": {"unet_name": f, "weight_dtype": "default"}}
            for i, f in enumerate(files, 1)}


def _dm(name):
    return f"models/diffusion_models/{name}"


def _sync_make(aliases=None, catalog=None, src=None, token=_TOKEN, **kw):
    """A controller with the model-sync deps wired to `box` (aliases → candidate,
    catalog) and a FakeVM behind its ssh."""
    fake = FakeThunder()
    vm = FakeVM(fake)
    c, saved, enabled, calls = make(fake, **kw)
    box = {"aliases": dict(aliases or {}), "catalog": list(catalog or [])}

    def needs(bid):                     # called with the ComfyUI service's backend id
        return [ms.alias_need(a, ms.refs_for(cand, cand["workflow_json"], []), box["catalog"])
                for a, cand in sorted(box["aliases"].items())
                if f"comfyui:{cand.get('backend')}" == bid]
    c.deps.alias_needs = needs
    c.deps.alias_signature = lambda name: json.dumps(box, sort_keys=True)
    c.deps.url_catalog = lambda src: ms.url_catalog(box["catalog"], src)
    c.deps.source_index = lambda: dict(src or {})
    c.deps.hf_token = lambda: token
    vm.wrap(c)
    c.h.calls = calls
    return fake, vm, c, box, saved


def _cand(*files):
    return {"backend": "thunder", "workflow_json": _wf(*files)}


def _url(name, host="https://example.com", sha=None):
    e = {"file": _dm(name), "url": f"{host}/{name}"}
    if sha:
        e["sha256"] = sha
    return e


async def _until(pred, secs=5.0):
    """Spin the loop until `pred()` holds (the transfer tasks and the thread hops of a
    sync need real turns of the event loop)."""
    loop = asyncio.get_running_loop()
    end = loop.time() + secs
    while loop.time() < end:
        if pred():
            return True
        await asyncio.sleep(0)
    return pred()


def _idle(c):
    return not c._fetches and (c._kicker is None or c._kicker.done())


def _gw(fake, verb, start=0):
    return [i for i, (m, p, _) in enumerate(fake.calls[start:]) if m == "SSH"
            and p.startswith(f": {verb} ")]


class ModelSync(unittest.IsolatedAsyncioTestCase):
    """The sync loop: plan, per-alias readiness, URL transfers on the VM, disk growth,
    prune at stop (spec "Controller", "Stop" 1–2)."""

    async def test_ready_only_after_all_files_present(self):
        a, b = _dm("a.safetensors"), _dm("b.safetensors")
        fake, vm, c, box, _ = _sync_make(
            aliases={"img": _cand("a.safetensors", "b.safetensors")},
            catalog=[_url("a.safetensors"), _url("b.safetensors")])
        vm.sizes = {"https://example.com/a.safetensors": 100,
                    "https://example.com/b.safetensors": 300}
        vm.chunk = 50
        vm.hold.add("https://example.com/b.safetensors")
        self.assertFalse(c.is_alias_ready(BID, "img"))               # no plan yet
        self.assertIsNone(c.plan)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(c.h.phases[-3:], ["starting", "syncing", "ready"])
        self.assertIsNotNone(c.plan)
        self.assertFalse(c.is_alias_ready(BID, "img"))
        # a finished, b still downloading: a is present, the alias is not ready
        self.assertTrue(await _until(lambda: a in vm.files and c.plan is not None and any(
            f["path"] == a and f["present"] for f in c.plan["per_alias"]["img"]["files"])))
        self.assertFalse(c.is_alias_ready(BID, "img"))
        self.assertNotIn("img", c.ready_aliases)
        self.assertIn("syncing on thunder", c.alias_status(BID, "img"))
        self.assertIn(b, [t["file"] for t in c.view()["transfers"]])
        vm.hold.clear()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        self.assertEqual(c.alias_status(BID, "img"), "models for img are ready on thunder")
        man = json.loads(vm.manifest)
        self.assertEqual(man[a]["size"], 100)
        self.assertEqual(man[b]["size"], 300)
        self.assertEqual((man[a]["source"], man[a]["aliases"]), ("url", ["img"]))
        self.assertEqual(vm.files[a], 100)
        self.assertNotIn(a + ".part", vm.files)
        self.assertEqual(c.view()["transfers"], [])
        # readiness is also a phase question
        c.state.phase = "draining"
        self.assertFalse(c.is_alias_ready(BID, "img"))
        self.assertEqual(c.alias_status(BID, "img"), "thunder instance is draining")
        c.state.phase = "syncing"
        self.assertTrue(c.is_alias_ready(BID, "img"))
        self.assertFalse(c.is_alias_ready(BID, "other"))
        self.assertIn("not planned", c.alias_status(BID, "other"))

    async def test_at_most_three_transfers_at_once(self):
        names = [f"f{i}.safetensors" for i in range(5)]
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand(*names)},
                                         catalog=[_url(n) for n in names])
        vm.sizes = {f"https://example.com/{n}": 10 for n in names}
        vm.hold = set(vm.sizes)
        await c.start()
        await _until(lambda: len(vm.started) >= 3)
        for _ in range(50):
            await asyncio.sleep(0)
        self.assertEqual(len(vm.started), 3)
        self.assertEqual(len(c._fetches), 3)
        vm.hold.clear()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        self.assertEqual(len(vm.started), 5)

    async def test_alias_removed_during_session_keeps_files_until_stop(self):
        k, g = _dm("keep.safetensors"), _dm("gone.safetensors")
        fake, vm, c, box, saved = _sync_make(
            aliases={"keep": _cand("keep.safetensors"), "gone": _cand("gone.safetensors")},
            catalog=[_url("keep.safetensors"), _url("gone.safetensors")])
        vm.sizes = {"https://example.com/keep.safetensors": 7,
                    "https://example.com/gone.safetensors": 9}
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.ready_aliases == {"keep", "gone"}))
        n = len(fake.calls)
        del box["aliases"]["gone"]                  # signature changes
        await c._sync_tick()
        self.assertEqual(c.plan["prune"], [g])
        self.assertIn(g, vm.files)                  # never deleted during a session
        self.assertEqual(_gw(fake, "gw-prune", n), [])
        self.assertEqual(c.ready_aliases, {"keep"})
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertNotIn(g, vm.files)
        self.assertIn(k, vm.files)
        man = json.loads(vm.manifest)
        self.assertEqual(sorted(man), [k])
        self.assertEqual(man[k]["aliases"], ["keep"])
        self.assertEqual(c.state.manifests["s0"], man)
        self.assertEqual(saved["thunder"]["manifests"]["s0"], man)
        kinds = _paths(fake, n)
        prune = _gw(fake, "gw-prune", n)[0]
        self.assertLess(prune, kinds.index(("POST", "/snapshots/create")))

    async def test_held_files_of_a_blocked_alias_survive_the_stop(self):
        h = _dm("held.safetensors")
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("held.safetensors")},
                                         catalog=[_url("held.safetensors")])
        vm.sizes = {"https://example.com/held.safetensors": 5}
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        # the alias now loads a file no source has → blocked; its synced file is held
        box["aliases"]["img"] = _cand("nowhere.safetensors")
        await c._sync_tick()
        self.assertFalse(c.is_alias_ready(BID, "img"))
        self.assertEqual(c.plan["held"], [[h, 5, "img"]])
        await c.stop()
        self.assertIn(h, vm.files)
        self.assertIn(h, json.loads(vm.manifest))

    async def test_running_download_adopted_not_restarted(self):
        p, url = _dm("big.safetensors"), "https://example.com/big.safetensors"
        fake = FakeThunder()
        _inst(fake)
        vm = FakeVM(fake)
        c, _, _, _ = make(fake, state=_persisted())
        box = {"img": _cand("big.safetensors")}
        cat = [_url("big.safetensors")]
        c.deps.alias_needs = lambda n: [ms.alias_need(
            "img", ms.refs_for(box["img"], box["img"]["workflow_json"], []), cat)]
        c.deps.alias_signature = lambda n: "sig1"
        c.deps.url_catalog = lambda src: ms.url_catalog(cat, src)
        vm.wrap(c)
        # a curl the previous gateway process started is still running on the box
        vm.sizes[url] = 30
        vm.chunk = 10
        vm.procs[3999] = url
        vm.lockpid[p] = 3999
        vm.part_url[p] = url
        vm.files.update({p + ".part": 10, p + ".part.lock": 5, p + ".part.log": 0})
        await c.resume()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertIsNone(c.plan)                     # not before the first sync
        await c._sync_tick()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        self.assertEqual(vm.started, [])              # no second curl on the same .part
        self.assertEqual(vm.files[p], 30)
        self.assertTrue(any("adopted" in ln for ln in c.state.log))

    async def test_hf_token_only_for_hf_hosts_and_never_in_argv(self):
        hf = "https://huggingface.co/org/repo/resolve/main"
        cat = [_url("h.safetensors", hf), _url("e.safetensors"),
               _url("s.safetensors", "https://huggingface.co.evil.example"),
               _url("w.safetensors", "https://hf.co/org/repo")]
        names = ["h.safetensors", "e.safetensors", "s.safetensors", "w.safetensors"]
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand(*names)}, catalog=cat)
        vm.sizes = {e["url"]: 4 for e in cat}
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        cfg = {p: t for p, t in vm.started}
        self.assertIn(f'header = "Authorization: Bearer {_TOKEN}"', cfg[_dm("h.safetensors")])
        self.assertIn(f'header = "Authorization: Bearer {_TOKEN}"', cfg[_dm("w.safetensors")])
        for other in ("e.safetensors", "s.safetensors"):
            self.assertNotIn(_TOKEN, cfg[_dm(other)])
            self.assertNotIn("Authorization", cfg[_dm(other)])
        self.assertIn(f'url = "{hf}/h.safetensors"', cfg[_dm("h.safetensors")])
        # never in any argv (ps shows argv), the log, the view or a fault
        self.assertTrue(vm.argvs)
        for argv in vm.argvs + [a for a, _ in c.h.calls]:
            self.assertFalse([x for x in argv if _TOKEN in x], argv)
        self.assertNotIn(_TOKEN, "\n".join(c.state.log))
        self.assertNotIn(_TOKEN, json.dumps(c.view(), default=str))
        self.assertNotIn(_TOKEN, json.dumps(c.h.faults, default=str))

    async def test_token_left_out_of_a_failed_transfers_messages(self):
        url = "https://huggingface.co/o/r/resolve/main/x.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors", "https://huggingface.co/o/r/resolve/main")])
        vm.fail[url] = f"curl: (22) The requested URL returned error: 401 (sent {_TOKEN})\n"
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and "img" in (c.plan or {}).get(
            "per_alias", {}) and c.plan["per_alias"]["img"]["blocked"]))
        blob = "\n".join(c.state.log) + json.dumps(c.view(), default=str) + json.dumps(
            c.h.faults, default=str) + c.alias_status(BID, "img")
        self.assertNotIn(_TOKEN, blob)
        self.assertIn("401", c.alias_status(BID, "img"))

    async def test_failed_transfer_blocks_after_three_attempts_with_fault(self):
        url = "https://example.com/x.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        # a 5xx is the server's trouble, not the URL's: retried (model sources R-5)
        vm.fail[url] = "curl: (22) The requested URL returned error: 503\n"
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.plan["per_alias"]["img"]["blocked"]))
        self.assertEqual(len(vm.started), 3)
        self.assertFalse(c.is_alias_ready(BID, "img"))
        st = c.alias_status(BID, "img")
        self.assertIn("blocked on thunder", st)
        self.assertIn("transfer failed", st)
        self.assertIn("503", st)
        sync_faults = [f for f in c.h.faults if f[1:3] == ("sync", "transfer")]
        self.assertEqual(len(sync_faults), 1)
        # a transfer is the ComfyUI SERVICE's event (R-K1), not the host's
        self.assertEqual((sync_faults[0][0]["name"], sync_faults[0][0]["type"]),
                         ("thunder", "comfyui"))
        self.assertIn(_dm("x.safetensors"), sync_faults[0][3])
        # no retry loop: the next sync keeps it blocked without a fourth curl
        await c.sync_once()
        await _until(lambda: _idle(c))
        self.assertEqual(len(vm.started), 3)
        # the operator fixes the catalog (signature changes) → tried again
        vm.fail.clear()
        vm.sizes[url] = 3
        box["catalog"] = [_url("x.safetensors")] + [{"match": {"alias": "other"}, "paths": []}]
        await c._sync_tick()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))

    async def test_sha256_mismatch_discards_the_part(self):
        url = "https://example.com/x.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors", sha="ab" * 32)])
        vm.sizes[url] = 8
        vm.sha[url] = "cd" * 32
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.plan["per_alias"]["img"]["blocked"]))
        self.assertIn("sha256", c.alias_status(BID, "img"))
        self.assertNotIn(_dm("x.safetensors"), vm.files)
        self.assertNotIn(_dm("x.safetensors") + ".part", vm.files)
        # deterministic: the same URL serves the same bytes — ONE download, not three
        # (model sources I-1); the file is on no share, so it gives up and blocks
        self.assertEqual(len(_gw(fake, "gw-discard")), 1)
        self.assertEqual(len(vm.started), 1)
        self.assertEqual(c._url_fallback, {})
        # the right bytes pass and their sha is recorded
        vm.sha[url] = "AB" * 32
        box["catalog"] = [_url("x.safetensors", sha="AB" * 32)]
        await c._sync_tick()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        self.assertEqual(json.loads(vm.manifest)[_dm("x.safetensors")]["sha256"], "ab" * 32)

    async def test_disk_growth_calls_modify(self):
        big, url = _dm("big.safetensors"), "https://example.com/big.safetensors"
        fake, vm, c, box, saved = _sync_make(
            aliases={"img": _cand("big.safetensors")}, catalog=[_url("big.safetensors")],
            src={big: 50 * _GiB})
        vm.sizes[url] = 50 * _GiB
        vm.avail = 30 * _GiB                      # < 50 GiB fetch + 20 GiB reserve
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_creates(fake)[0]["disk_size_gb"], 100)
        mods = [(p, b) for m, p, b in fake.calls if p.endswith("/modify")]
        # uuid form first (404 in the stub), then the index — both from a fresh list
        self.assertEqual(mods, [("/instances/u0/modify", {"disk_size_gb": 140}),
                                ("/instances/0/modify", {"disk_size_gb": 140})])
        lists = [i for i, (m, p, _) in enumerate(fake.calls) if p == "/instances/list"]
        first_mod = next(i for i, (m, p, _) in enumerate(fake.calls) if p.endswith("/modify"))
        self.assertTrue([i for i in lists if i < first_mod])
        self.assertEqual((c.state.disk_gb, saved["thunder"]["disk_gb"]), (140, 140))
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))

    async def test_disk_beyond_spec_max_blocks_the_alias(self):
        big, url = _dm("big.safetensors"), "https://example.com/big.safetensors"
        fake, vm, c, box, _ = _sync_make(
            aliases={"img": _cand("big.safetensors"), "small": _cand("s.safetensors")},
            catalog=[_url("big.safetensors"), _url("s.safetensors")], src={big: 300 * _GiB})
        vm.sizes = {url: 300 * _GiB, "https://example.com/s.safetensors": 1}
        vm.avail = 25 * _GiB                      # something else fills the disk
        await c.start()
        self.assertEqual(c.state.disk_gb, 320)
        self.assertEqual([p for m, p, b in fake.calls if p.endswith("/modify")], [])
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "small")))
        self.assertFalse(c.is_alias_ready(BID, "img"))
        self.assertIn("disk", c.alias_status(BID, "img"))
        self.assertNotIn(big, [p for p, _ in vm.started])

    async def test_stop_kills_transfers_before_snapshot(self):
        p, url = _dm("x.safetensors"), "https://example.com/x.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.sizes[url] = 100
        vm.chunk = 10
        vm.hold.add(url)
        await c.start()
        self.assertTrue(await _until(lambda: c.view()["transfers"]))
        n = len(fake.calls)
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        kinds = _paths(fake, n)
        kill = _gw(fake, "gw-kill", n)
        self.assertTrue(kill)
        self.assertLess(kill[0], kinds.index(("POST", "/snapshots/create")))
        self.assertLess(kill[0], _gw(fake, "gw-prune", n)[0])
        # ended while DRAINING (jobs may still run for a long time)
        self.assertIn(("gw-kill", "draining"), vm.seen)
        self.assertEqual(vm.killed, [p])
        self.assertEqual(vm.procs, {})
        self.assertFalse([f for f in vm.files if ".part" in f])
        self.assertEqual(c.view()["transfers"], [])
        self.assertEqual(c._fetches, {})
        self.assertNotIn(p, json.loads(vm.manifest))

    async def test_stop_resumed_after_restart_kills_downloads_before_prune(self):
        # the gateway restarted mid-stop: no transfer task exists, the curl still runs
        p, url = _dm("x.safetensors"), "https://example.com/x.safetensors"
        fake = FakeThunder()
        _inst(fake)
        vm = FakeVM(fake)
        c, _, _, _ = make(fake, state=_persisted(phase="pruning"))
        vm.wrap(c)
        vm.sizes[url] = 30
        vm.hold.add(url)
        vm.procs[3999] = url
        vm.lockpid[p] = 3999
        vm.files.update({p + ".part": 10, p + ".part.lock": 5, p + ".part.log": 0})
        await c.resume()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(vm.killed, [p])
        verbs = [v for v, _ in vm.seen]
        self.assertLess(verbs.index("gw-kill"), verbs.index("gw-prune"))
        self.assertFalse([f for f in vm.files if ".part" in f])

    async def test_stop_during_a_sync_neither_grows_the_disk_nor_restores_the_old_manifest(self):
        import threading
        k, g = _dm("keep.safetensors"), _dm("gone.safetensors")
        fake, vm, c, box, _ = _sync_make(
            aliases={"keep": _cand("keep.safetensors"), "gone": _cand("gone.safetensors")},
            catalog=[_url("keep.safetensors"), _url("gone.safetensors")])
        vm.sizes = {"https://example.com/keep.safetensors": 7,
                    "https://example.com/gone.safetensors": 9}
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.ready_aliases == {"keep", "gone"}))
        # the aliases change: "gone" goes, "big" needs more disk than there is free
        del box["aliases"]["gone"]
        box["aliases"]["big"] = _cand("big.safetensors")
        box["catalog"].append(_url("big.safetensors"))
        vm.sizes["https://example.com/big.safetensors"] = 50 * _GiB
        vm.avail = 10 * _GiB
        entered, release = threading.Event(), threading.Event()
        orig, armed = c.deps.alias_needs, [True]

        def needs(name):                  # the tick's plan hangs in its store read …
            if armed[0]:
                armed[0] = False
                entered.set()
                release.wait(10)
            return orig(name)
        c.deps.alias_needs = needs
        real = vm.run
        at_prune = []

        def run(cmd, stdin):              # … until the stop has written the pruned manifest
            if cmd.startswith(": gw-prune "):
                at_prune.append(len(c._syncs))      # the sync is ended BEFORE the prune
            r = real(cmd, stdin)
            if cmd.startswith(": gw-manifest-write ") and c.state.phase == "pruning":
                release.set()
            return r
        vm.run = run
        tick = asyncio.ensure_future(c._sync_tick())        # what run_forever does
        try:
            self.assertTrue(await _until(entered.is_set))
            await c.stop()
            self.assertEqual(c.state.phase, "off", c.state.error)
            release.set()
            await _until(tick.done)
            for _ in range(20):
                await asyncio.sleep(0)
            self.assertIsNone(await tick)                     # the loop is not cancelled
        finally:
            release.set()
        self.assertEqual([p for m, p, b in fake.calls if p.endswith("/modify")], [])
        self.assertEqual(sorted(c.state.manifests["s0"]), [k])
        self.assertEqual(sorted(c._current_manifest()), [k])
        self.assertEqual(c._syncs, set())
        self.assertEqual(at_prune, [0])

    async def test_cancel_between_mv_and_manifest_keeps_the_entry(self):
        p, url = _dm("x.safetensors"), "https://example.com/x.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.sizes[url] = 5
        await c.start()
        # another manifest writer holds the lock: the finished file waits for it after
        # its mv — and a stop cancels it right there
        await c._manifest_lock.acquire()
        self.assertTrue(await _until(lambda: vm.files.get(p) == 5 and p in c._fetches))
        for _ in range(10):
            await asyncio.sleep(0)
        c._fetches[p].cancel()
        self.assertTrue(await _until(lambda: _idle(c)))
        c._manifest_lock.release()
        self.assertEqual(vm.files[p], 5)
        self.assertNotIn(p, json.loads(vm.manifest or "{}"))  # its write never happened
        self.assertEqual(c._unsaved[p]["size"], 5)
        await c.sync_once()
        self.assertTrue(c.is_alias_ready(BID, "img"))              # present, not re-fetched
        self.assertEqual(len(vm.started), 1)
        await c.stop()
        self.assertEqual(json.loads(vm.manifest)[p]["size"], 5)
        self.assertEqual(c.state.manifests["s0"][p]["size"], 5)

    async def test_retries_back_off(self):
        url = "https://example.com/x.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.fail[url] = "curl: (7) Failed to connect\n"
        waits = []
        orig = c.deps.sleep

        async def sleep(sec):
            if asyncio.current_task() in c._fetches.values() and sec != hostctl._SYNC_POLL_S:
                waits.append(sec)
            await orig(sec)
        c.deps.sleep = sleep
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.plan["per_alias"]["img"]["blocked"]))
        self.assertEqual(waits, [15, 45])
        self.assertEqual(len(vm.started), 3)

    async def test_head_learns_url_sizes_with_the_token_rules(self):
        hf = "https://huggingface.co/o/r/resolve/main"
        cat = [_url("h.safetensors", hf), _url("e.safetensors")]
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("h.safetensors", "e.safetensors")},
                                         catalog=cat)
        vm.sizes = {f"{hf}/h.safetensors": 40, "https://example.com/e.safetensors": 60}
        vm.chunk = 10
        vm.cap = {u: 20 for u in vm.sizes}
        await c.start()
        heads = {p: t for p, t in vm.heads}
        self.assertIn(f'header = "Authorization: Bearer {_TOKEN}"', heads[_dm("h.safetensors")])
        self.assertNotIn(_TOKEN, heads[_dm("e.safetensors")])
        self.assertIn("max-time", heads[_dm("e.safetensors")])
        for argv in vm.argvs:
            self.assertFalse([x for x in argv if _TOKEN in x], argv)
        self.assertEqual(c.plan["per_alias"]["img"]["need_bytes"], 100)
        self.assertEqual(sorted(e["size"] for e in c.plan["fetch"]), [40, 60])
        self.assertTrue(await _until(lambda: len(c.view()["transfers"]) == 2 and all(
            t["bytes"] == 20 and t["eta"] is not None for t in c.view()["transfers"])))
        self.assertEqual(sorted(t["total"] for t in c.view()["transfers"]), [40, 60])
        vm.cap.clear()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        self.assertEqual(len(vm.heads), 2)                    # once per URL and session
        self.assertNotIn(_TOKEN, "\n".join(c.state.log))

    async def test_head_without_length_stays_unknown_and_a_wrong_size_fails(self):
        a, b = "https://example.com/a.safetensors", "https://example.com/b.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"one": _cand("a.safetensors"),
                                                  "two": _cand("b.safetensors")},
                                         catalog=[_url("a.safetensors"), _url("b.safetensors")])
        vm.sizes = {a: 5, b: 5}
        vm.no_length.add(a)
        vm.head_len[b] = 99                           # the server lied / the file changed
        await c.start()
        self.assertEqual({e["path"]: e["size"] for e in c.plan["fetch"]},
                         {_dm("a.safetensors"): None, _dm("b.safetensors"): 99})
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "one")
                                     and c.plan["per_alias"]["two"]["blocked"]))
        self.assertIn("size 5 ≠ 99", c.alias_status(BID, "two"))
        self.assertTrue(any("no Content-Length" in ln for ln in c.state.log))

    async def test_unreadable_manifest_is_warned_about(self):
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.sizes["https://example.com/x.safetensors"] = 3
        vm.manifest = "{not json"
        await c.start()
        self.assertTrue([ln for ln in c.state.log if "WARNING: manifest" in ln
                         and "unreadable" in ln])
        n = len(c.state.log)
        vm.manifest = "{}"
        await c.sync_once()
        self.assertFalse([ln for ln in c.state.log[n:] if "WARNING: manifest" in ln])

    async def test_token_a_curl_config_cannot_carry_is_withheld(self):
        hf = "https://huggingface.co/o/r/resolve/main"
        for bad in ('hf_a"b', "hf_a\\b", "hf_a\nheader = x", "hf_a\x01b"):
            fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("h.safetensors")},
                                             catalog=[_url("h.safetensors", hf)], token=bad)
            vm.sizes[f"{hf}/h.safetensors"] = 4
            await c.start()
            self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
            for _, cfg in vm.started + vm.heads:
                self.assertNotIn("header", cfg, repr(bad))
                self.assertNotIn(bad, cfg)
            self.assertTrue([ln for ln in c.state.log if "hf_token withheld" in ln], repr(bad))

    def test_parse_head(self):
        ok = "HTTP/1.1 302 Found\r\nContent-Length: 0\r\n\r\nHTTP/2 200\r\ncontent-length: 1234\r\n"
        self.assertEqual(hostctl.parse_head(ok), 1234)
        self.assertIsNone(hostctl.parse_head("HTTP/2 200\r\netag: x\r\n"))
        self.assertIsNone(hostctl.parse_head("HTTP/2 302\r\ncontent-length: 5\r\n\r\n"
                                                "HTTP/2 403\r\ncontent-length: 7\r\n"))
        self.assertIsNone(hostctl.parse_head(""))
        self.assertIsNone(hostctl.parse_head("HTTP/2 200\r\ncontent-length: -1\r\n"))
        # a 2xx naming length 0 is unknown, not an empty model file: taken as the size,
        # every download would fail its size check and block the file's aliases
        self.assertIsNone(hostctl.parse_head("HTTP/2 200\r\ncontent-length: 0\r\n"))
        self.assertIsNone(hostctl.parse_head("HTTP/1.1 302 Found\r\nContent-Length: 9\r\n"
                                                "\r\nHTTP/2 200\r\ncontent-length: 0\r\n"))

    async def test_unknown_files_listed_never_pruned(self):
        stranger, old = "models/checkpoints/stranger.safetensors", "models/loras/old.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.sizes["https://example.com/x.safetensors"] = 3
        vm.files = {stranger: 123, old: 44, "hf-cache/token": 40}
        vm.manifest = json.dumps({old: {"size": 44, "source": "lan", "aliases": "nobody"}})
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        self.assertEqual(c.plan["unknown"], [[stranger, 123]])
        self.assertEqual(c.plan["prune"], [old])
        self.assertEqual(c.view()["plan"]["unknown"], [[stranger, 123]])
        n = len(fake.calls)
        await c.stop()
        self.assertIn(stranger, vm.files)
        self.assertIn("hf-cache/token", vm.files)
        self.assertNotIn(old, vm.files)
        prune_cmd = fake.calls[n + _gw(fake, "gw-prune", n)[0]][1]
        self.assertNotIn("stranger", prune_cmd)
        self.assertNotIn("token", prune_cmd)
        # the manifest written at stop carries `aliases` as a list everywhere
        for e in json.loads(vm.manifest).values():
            self.assertIsInstance(e["aliases"], list)

    async def test_delete_unknown_only_deletes_unknown_files(self):
        stranger = "models/checkpoints/stranger.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.sizes["https://example.com/x.safetensors"] = 3
        vm.files = {stranger: 123}
        with self.assertRaises(RuntimeError):
            await c.delete_unknown([stranger])          # no instance
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        with self.assertRaises(ValueError):             # a needed file is no "unknown"
            await c.delete_unknown([stranger, _dm("x.safetensors")])
        self.assertIn(stranger, vm.files)
        self.assertEqual(await c.delete_unknown([stranger]), 1)
        self.assertNotIn(stranger, vm.files)
        self.assertEqual(c.plan["unknown"], [])
        self.assertIn(_dm("x.safetensors"), vm.files)

    async def test_delete_unknown_drops_the_files_from_the_template_report(self):
        # thunder-1, 2026-10-01 16:32: "delete unknown" removed the template's checkpoint,
        # and the card went on listing it under "Models the template brought along"
        tmpl, other = "models/checkpoints/v1-5-pruned.safetensors", "models/vae/kept.pt"
        fake, vm, c, box, saved = _sync_make(aliases={"img": _cand("x.safetensors")},
                                             catalog=[_url("x.safetensors")])
        vm.sizes["https://example.com/x.safetensors"] = 3
        vm.files = {tmpl: 123, other: 7}
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        c.state.bootstrap_unknown = {tmpl: 123, other: 7}
        self.assertEqual(await c.delete_unknown([tmpl]), 1)
        self.assertEqual(c.state.bootstrap_unknown, {other: 7})
        self.assertEqual(c.view()["bootstrap_unknown"], {other: 7})
        self.assertEqual(saved["thunder"]["bootstrap_unknown"], {other: 7})   # persisted

    async def test_sync_now_retries_failed_transfers(self):
        """Task 13: "Sync now" — without it a transfer that gave up is retried only when
        the aliases or the catalog change, however the operator fixed the cause."""
        url = "https://example.com/x.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        with self.assertRaises(RuntimeError):
            await c.sync_now()                          # no instance: refused at once
        vm.sizes[url] = 4
        vm.fail[url] = "curl: (7) Failed to connect\n"
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.plan["per_alias"]["img"]["blocked"]))
        n = len(vm.started)
        await c.sync_once()                             # a plain re-plan keeps it given up
        self.assertEqual(len(vm.started), n)
        del vm.fail[url]
        await c.sync_now()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        self.assertGreater(len(vm.started), n)
        self.assertTrue(any("tried again" in ln for ln in c.state.log))

    async def test_plan_view_sizes_the_prune_preview(self):
        old = "models/checkpoints/old.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.sizes["https://example.com/x.safetensors"] = 3
        vm.files = {old: 7}
        vm.manifest = json.dumps({old: {"size": 7, "aliases": ["gone"], "source": "url"}})
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        pv = c.view()["plan"]
        self.assertEqual(pv["prune"], [old])
        self.assertEqual(pv["prune_sizes"], [[old, 7]])

    async def test_lan_files_wait_for_the_lan_source(self):
        lanf = _dm("lan.safetensors")
        # "img": in no source at all; "known": in a (P3-style) source index but no URL
        fake, vm, c, box, _ = _sync_make(
            aliases={"img": _cand("nowhere.safetensors"), "known": _cand("lan.safetensors")},
            src={lanf: 9})
        await c.start()
        self.assertEqual(c.plan["fetch"][0]["source"], "lan")
        for a in ("img", "known"):
            st = c.alias_status(BID, a)
            self.assertIn("waiting for LAN source (not configured)", st)
            self.assertFalse(c.is_alias_ready(BID, a))
            self.assertTrue([b for b in c.view()["plan"]["aliases"][a]["blocked"]
                             if b.startswith("waiting for LAN source (not configured)")])
        self.assertNotIn("not in source", c.alias_status(BID, "img"))
        self.assertEqual(vm.started, [])

    async def test_signature_change_triggers_a_sync(self):
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.sizes["https://example.com/x.safetensors"] = 3
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c)))
        n = len(_gw(fake, "gw-index"))
        await c._sync_tick()                            # unchanged, nothing running
        self.assertEqual(len(_gw(fake, "gw-index")), n)
        box["aliases"]["two"] = _cand("x.safetensors")
        await c._sync_tick()
        self.assertEqual(len(_gw(fake, "gw-index")), n + 1)
        self.assertTrue(c.is_alias_ready(BID, "two"))
        # not outside syncing|ready
        box["aliases"]["three"] = _cand("x.safetensors")
        c.state.phase = "starting"
        await c._sync_tick()
        self.assertEqual(len(_gw(fake, "gw-index")), n + 1)

    async def test_transfer_ending_during_a_comfy_restart_still_starts_the_next(self):
        names = [f"f{i}.safetensors" for i in range(4)]
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand(*names)},
                                         catalog=[_url(n) for n in names])
        vm.sizes = {f"https://example.com/{n}": 10 for n in names}
        vm.hold = set(vm.sizes)
        await c.start()
        self.assertTrue(await _until(lambda: len(c._fetches) == 3))
        c.state.phase = "starting"                    # ComfyUI restarting
        vm.hold.clear()
        self.assertTrue(await _until(lambda: not c._fetches))
        self.assertEqual(len(vm.started), 3)          # no sync outside syncing|ready …
        c.state.phase = "ready"
        await c._sync_tick()                          # … the pending re-plan runs now
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        self.assertEqual(len(vm.started), 4)

    async def test_run_forever_polls_the_signature_every_5s(self):
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.sizes["https://example.com/x.safetensors"] = 3
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c)))
        box["aliases"]["two"] = _cand("x.safetensors")
        n = len(_gw(fake, "gw-index"))
        orig, ticks = c.deps.sleep, [0]

        async def sleep(sec):
            if sec == hostctl._SYNC_POLL_S and asyncio.current_task() is loop_task:
                ticks[0] += 1
                if ticks[0] > 1:
                    raise asyncio.CancelledError()
            await orig(sec)
        c.deps.sleep = sleep
        loop_task = asyncio.ensure_future(c.run_forever())
        with self.assertRaises(asyncio.CancelledError):
            await loop_task
        c.deps.sleep = orig
        self.assertEqual(len(_gw(fake, "gw-index")), n + 1)
        self.assertTrue(c.is_alias_ready(BID, "two"))

    async def test_required_bytes_hint_uses_the_snapshots_manifest(self):
        f = _dm("big.safetensors")
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("big.safetensors")},
                                         catalog=[_url("big.safetensors")])
        _ready_snap(fake, sid="s9", min_gb=100)
        c.state.manifests = {"s9": {f: {"size": 150 * _GiB, "source": "url", "aliases": ["img"]}}}
        self.assertEqual(await c._required_bytes_hint("s9"), 150 * _GiB)
        self.assertEqual(await c._required_bytes_hint(""), 0)       # URL size unknown
        vm.files[f] = 150 * _GiB                        # the restored disk holds it
        vm.manifest = json.dumps(c.state.manifests["s9"])
        await c.start()
        self.assertEqual(_creates(fake)[0]["disk_size_gb"], 170)
        # a broken dep never stops a start: the hint is 0 then
        c.deps.alias_needs = lambda n: 1 / 0
        self.assertEqual(await c._required_bytes_hint("s9"), 0)

    async def test_view_plan_summary_and_transfers(self):
        fake, vm, c, box, _ = _sync_make(
            aliases={"img": _cand("x.safetensors"), "bad": _cand("nowhere.safetensors")},
            catalog=[_url("x.safetensors", "https://huggingface.co/o/r/resolve/main")])
        url = "https://huggingface.co/o/r/resolve/main/x.safetensors"
        vm.sizes[url] = 100
        vm.chunk = 25
        vm.cap[url] = 50
        self.assertIsNone(c.view()["plan"])
        await c.start()
        self.assertTrue(await _until(lambda: c.view()["transfers"]
                                     and c.view()["transfers"][0]["bytes"] == 50
                                     and c.view()["transfers"][0]["rate"] is not None))
        v = c.view()
        row = v["transfers"][0]
        for k in ("file", "source", "bytes", "total", "rate", "eta"):
            self.assertIn(k, row)
        self.assertEqual((row["file"], row["source"]), (_dm("x.safetensors"), "url"))
        self.assertNotIn(_TOKEN, json.dumps(v, default=str))
        self.assertNotIn(url, json.dumps(v["transfers"]))
        a = v["plan"]["aliases"]["img"]
        for k in ("need_bytes", "have_bytes", "missing", "blocked", "hints", "held", "ready"):
            self.assertIn(k, a)
        self.assertEqual((a["missing"], a["ready"], a["held"]), (1, False, 0))
        self.assertTrue(v["plan"]["aliases"]["bad"]["blocked"])
        vm.cap.clear()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        self.assertTrue(c.view()["plan"]["aliases"]["img"]["ready"])
        self.assertEqual(c.view()["ready_aliases"], ["img"])

    async def test_sync_failure_keeps_start_and_is_retried(self):
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.sizes["https://example.com/x.safetensors"] = 3
        real = vm.run
        broken = [True]

        def run(cmd, stdin):
            if broken[0] and cmd.startswith(": gw-index "):
                return (255, b"", b"Connection reset")
            return real(cmd, stdin)
        vm.run = run
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertIsNone(c.plan)
        self.assertFalse(c.is_alias_ready(BID, "img"))
        self.assertTrue(c.view()["sync_error"])
        broken[0] = False
        c.h.clock[0] += hostctl._SYNC_REFRESH_S
        await c._sync_tick()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        self.assertEqual(c.view()["sync_error"], "")

    async def test_manifest_write_is_atomic_and_manifest_aliases_are_lists(self):
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.sizes["https://example.com/x.safetensors"] = 3
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        cmd = fake.calls[_gw(fake, "gw-manifest-write")[0]][1]
        self.assertIn("cat > .gw-modelsync.json.tmp && mv -f .gw-modelsync.json.tmp "
                      ".gw-modelsync.json", cmd)
        for m in vm.manifest_writes:
            for e in m.values():
                self.assertIsInstance(e["aliases"], list)

    def test_remote_paths_stay_under_the_two_roots(self):
        self.assertEqual(hostctl.remote_path("models/vae/a.safetensors"),
                         "ComfyUI/models/vae/a.safetensors")
        self.assertEqual(hostctl.remote_path("hf-cache/hub/x"), "hf-cache/hub/x")
        for bad in ("models/../x", "/etc/passwd", "other/x", "models/", "hf-cache/.ssh/k",
                    "models/a\nb"):
            with self.assertRaises(ValueError, msg=bad):
                hostctl.remote_path(bad)
        idx = hostctl.parse_index("ComfyUI/models/vae/a.st\t5\nhf-cache/hub/b\t7\n"
                                     "hf-cache/.locks/c\t1\nComfyUI/custom_nodes/x\t3\n"
                                     "junk\nGW:END\n")
        self.assertEqual(idx, {"models/vae/a.st": 5, "hf-cache/hub/b": 7})
        with self.assertRaises(RuntimeError):            # cut off → never "all missing"
            hostctl.parse_index("ComfyUI/models/vae/a.st\t5\n")

    def test_curl_config_quotes_nothing_it_cannot_carry(self):
        cfg = hostctl.curl_config("https://example.com/a", "")
        self.assertIn('url = "https://example.com/a"', cfg)
        self.assertNotIn("header", cfg)
        self.assertNotIn("location-trusted", cfg)
        cfg = hostctl.curl_config("https://huggingface.co/a", "tok")
        self.assertIn('header = "Authorization: Bearer tok"', cfg)


_STUB_CURL = """#!/bin/sh
# stub curl: options from stdin (--config -), writes to -o; a TERM ends it
cfg=$(cat)
case " $* " in *" -sIL "*)
  printf '%s' "$cfg" > "$HOME/head-cfg"
  printf 'HTTP/1.1 302 Found\r\nContent-Length: 0\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\n'
  exit 0;;
esac
[ -n "$STUB_BECOME" ] && exec sleep "$STUB_BECOME"   # the pid lives on, as no curl
out=
while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done
printf '%s' "$cfg" > "$HOME/cfg-seen"
i=0
while [ $i -lt "${STUB_SLEEP:-0}" ]; do sleep 1; i=$((i+1)); done
if [ -n "$STUB_FAIL" ]; then echo "curl: (22) $STUB_FAIL" >&2; exit 22; fi
printf 'payload' >> "$out"
"""


@unittest.skipUnless(shutil.which("bash") and shutil.which("setsid") and os.path.isdir("/proc"),
                     "needs bash, setsid and /proc")
class RemoteShell(unittest.TestCase):
    """The remote sync commands run for real in `bash -c` (what sshd runs) against a
    temp HOME and a stub curl — quoting, the lock/adopt logic, the kill and the prune
    only ever fail on a live instance otherwise."""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="thunder-vm-")
        _TMPDIRS.append(self.home)
        os.makedirs(os.path.join(self.home, "bin"))
        os.makedirs(os.path.join(self.home, "ComfyUI", "models"))
        stub = os.path.join(self.home, "bin", "curl")
        with open(stub, "w") as f:
            f.write(_STUB_CURL)
        os.chmod(stub, 0o755)
        self.env = dict(os.environ, HOME=self.home,
                        PATH=os.path.join(self.home, "bin") + os.pathsep + os.environ["PATH"])

    def sh(self, cmd, stdin=b"", **env):
        import subprocess
        r = subprocess.run(["bash", "-c", cmd], input=stdin, capture_output=True,
                           env=dict(self.env, **env), timeout=30)
        return r.returncode, r.stdout.decode(), r.stderr.decode()

    def poll(self, rel, secs=10.0):
        import time as _t
        end = _t.monotonic() + secs
        while True:
            rc, out, err = self.sh(hostctl._poll_cmd(rel))
            self.assertEqual(rc, 0, err)
            if out.startswith("GW:END") or _t.monotonic() > end:
                return out.splitlines()
            _t.sleep(0.1)

    def test_fetch_poll_done_roundtrip(self):
        rel = "models/diffusion_models/it's a file.safetensors"
        cfg = hostctl.curl_config("https://huggingface.co/x", _TOKEN)
        rc, out, err = self.sh(hostctl._fetch_cmd(rel), cfg.encode(), STUB_SLEEP="2")
        self.assertEqual((rc, out.strip()), (0, "GW:STARTED"), err)
        # while it runs: a second start adopts, and no process shows the token
        rc, out, _ = self.sh(hostctl._fetch_cmd(rel), cfg.encode())
        self.assertEqual(out.strip(), "GW:ADOPT")
        for pid in [p for p in os.listdir("/proc") if p.isdigit()]:
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    self.assertNotIn(_TOKEN.encode(), f.read(), pid)
            except OSError:
                pass
        state, size, *log = self.poll(rel)
        self.assertEqual((state, size, log), ("GW:END", "7", []))
        with open(os.path.join(self.home, "cfg-seen")) as f:
            self.assertEqual(f.read(), cfg.rstrip("\n"))
        rc, out, err = self.sh(hostctl._done_cmd(rel))
        self.assertEqual((rc, out.strip()), (0, "7"), err)
        final = os.path.join(self.home, "ComfyUI", hostctl.remote_path(rel)[len("ComfyUI/"):])
        with open(final) as f:
            self.assertEqual(f.read(), "payload")
        self.assertEqual(sorted(os.listdir(os.path.dirname(final))),
                         ["it's a file.safetensors"])
        rc, out, _ = self.sh(hostctl._INDEX_CMD)
        self.assertEqual(hostctl.parse_index(out), {rel: 7})
        rc, out, _ = self.sh(hostctl._sha_cmd(rel.replace(".safetensors", ".x")))
        self.assertNotEqual(rc, 0)                    # no .part: sha fails, not "empty"

    def test_curl_error_lands_in_the_poll(self):
        rel = "hf-cache/hub/x.bin"
        cfg = hostctl.curl_config("https://example.com/x", "")
        self.sh(hostctl._fetch_cmd(rel), cfg.encode(), STUB_FAIL="404 not found")
        state, size, *log = self.poll(rel)
        self.assertEqual(state, "GW:END")
        self.assertIn("curl: (22) 404 not found", "\n".join(log))
        rc, _, err = self.sh(hostctl._discard_cmd(rel))
        self.assertEqual(rc, 0, err)
        self.assertEqual(os.listdir(os.path.join(self.home, "hf-cache", "hub")), [])

    def test_stale_lock_of_another_process_is_not_adopted_or_killed(self):
        rel = "models/vae/v.safetensors"
        d = os.path.join(self.home, "ComfyUI", "models", "vae")
        os.makedirs(d)
        with open(os.path.join(d, "v.safetensors.part.lock"), "w") as f:
            f.write(f"{os.getpid()}\n")               # alive, but no curl (this test)
        rc, out, _ = self.sh(hostctl._KILL_CMD)
        self.assertEqual((rc, out.strip()), (0, "GW:KILLED"))   # we are still here
        cfg = hostctl.curl_config("https://example.com/v", "")
        rc, out, err = self.sh(hostctl._fetch_cmd(rel), cfg.encode())
        self.assertEqual(out.strip(), "GW:STARTED", err)
        self.assertEqual(self.poll(rel)[:2], ["GW:END", "7"])

    def test_kill_then_prune(self):
        rel = "models/loras/l.safetensors"
        keep = os.path.join(self.home, "ComfyUI", "models", "loras", "keep.safetensors")
        old = os.path.join(self.home, "ComfyUI", "models", "loras", "old one.safetensors")
        cfg = hostctl.curl_config("https://example.com/l", "")
        self.sh(hostctl._fetch_cmd(rel), cfg.encode(), STUB_SLEEP="20")
        for p in (keep, old):
            with open(p, "w") as f:
                f.write("x")
        self.assertEqual(self.sh(hostctl._poll_cmd(rel))[1].splitlines()[0], "GW:RUN")
        rc, out, err = self.sh(hostctl._KILL_CMD)
        self.assertEqual((rc, out.strip()), (0, "GW:KILLED"), err)
        self.assertEqual(self.sh(hostctl._poll_cmd(rel))[1].splitlines()[0], "GW:END")
        rc, out, err = self.sh(hostctl._prune_cmd(["models/loras/old one.safetensors"]))
        self.assertEqual((rc, out.split()), (0, ["GW:RM-OK", "GW:PRUNED"]), err)
        self.assertEqual(os.listdir(os.path.dirname(keep)), ["keep.safetensors"])
        rc, out, err = self.sh(hostctl._prune_cmd([]))
        self.assertEqual(out.split(), ["GW:PRUNED"])

    def test_abandon_ends_the_files_curl_and_discards_its_part(self):
        # model sources fallback: before the LAN takes over a file, the URL's curl (which
        # may run on after an attempt gave up) is ended and the shared .part removed
        rel = "models/loras/f b.safetensors"
        d = os.path.join(self.home, "ComfyUI", "models", "loras")
        cfg = hostctl.curl_config("https://example.com/f", "")
        rc, out, err = self.sh(hostctl._fetch_cmd(rel), cfg.encode(), STUB_SLEEP="20")
        self.assertEqual(out.strip(), "GW:STARTED", err)
        with open(os.path.join(d, "f b.safetensors.part.lock")) as f:
            pid = f.read().strip()
        rc, out, err = self.sh(hostctl._abandon_cmd(rel))
        self.assertEqual((rc, out.strip()), (0, "GW:ABANDONED"), err)
        self.assertFalse(os.path.exists(f"/proc/{pid}/comm") and
                         open(f"/proc/{pid}/comm").read().strip() == "curl")
        self.assertEqual(os.listdir(d), [])
        # a lock naming a live process that is no curl: nothing is killed (this test)
        with open(os.path.join(d, "f b.safetensors.part.lock"), "w") as f:
            f.write(f"{os.getpid()}\n")
        rc, out, err = self.sh(hostctl._abandon_cmd(rel))
        self.assertEqual((rc, out.strip()), (0, "GW:ABANDONED"), err)
        self.assertEqual(os.listdir(d), [])
        # nothing there at all: still confirmed
        rc, out, err = self.sh(hostctl._abandon_cmd("models/none/x.safetensors"))
        self.assertEqual((rc, out.strip()), (0, "GW:ABANDONED"), err)

    def test_started_only_once_the_lock_names_the_curl(self):
        # GW:STARTED used to be echoed before the child had written its pid and exec'd
        # curl: a kill in that window found no curl and the download ran on. Now the
        # lock names a running curl whenever GW:STARTED arrives — every time.
        rel = "models/loras/r.safetensors"
        lock = os.path.join(self.home, "ComfyUI", "models", "loras", "r.safetensors.part.lock")
        cfg = hostctl.curl_config("https://example.com/r", "")
        for _ in range(5):
            rc, out, err = self.sh(hostctl._fetch_cmd(rel), cfg.encode(), STUB_SLEEP="20")
            self.assertEqual((rc, out.strip()), (0, "GW:STARTED"), err)
            with open(lock) as f:
                pid = f.read().strip()
            with open(f"/proc/{pid}/comm") as f:
                self.assertEqual(f.read().strip(), "curl")
            self.assertEqual(self.sh(hostctl._KILL_CMD)[1].strip(), "GW:KILLED")
            self.assertEqual(self.sh(hostctl._poll_cmd(rel))[1].splitlines()[0], "GW:END")

    def test_a_stale_lock_of_a_dead_pid_does_not_count_as_started(self):
        # the lock a dead curl left behind must not pass for the new child's: removed
        # before the spawn, so what appears is the child's own pid
        rel = "models/vae/d.safetensors"
        d = os.path.join(self.home, "ComfyUI", "models", "vae")
        os.makedirs(d)
        with open(os.path.join(d, "d.safetensors.part.lock"), "w") as f:
            f.write("999999999\n")                        # no such process
        cfg = hostctl.curl_config("https://example.com/d", "")
        rc, out, err = self.sh(hostctl._fetch_cmd(rel), cfg.encode(), STUB_SLEEP="20")
        self.assertEqual(out.strip(), "GW:STARTED", err)
        with open(os.path.join(d, "d.safetensors.part.lock")) as f:
            self.assertNotEqual(f.read().strip(), "999999999")
        self.sh(hostctl._KILL_CMD)

    def test_curl_that_never_comes_up_is_a_start_failure(self):
        # the lock's pid lives on but never as a curl: no GW:STARTED — the controller
        # counts a failed attempt instead of polling a download a kill could not end
        rel = "models/vae/nc.safetensors"
        cfg = hostctl.curl_config("https://example.com/nc", "")
        rc, out, err = self.sh(hostctl._fetch_cmd(rel), cfg.encode(), STUB_BECOME="5")
        self.assertEqual(rc, 0, err)
        self.assertEqual(out.splitlines()[0], "GW:START-FAIL")
        self.assertNotIn("GW:STARTED", out)

    def test_head_reports_the_final_content_length(self):
        rel = "models/vae/v.safetensors"
        cfg = hostctl.curl_config("https://huggingface.co/v", _TOKEN, head=True)
        rc, out, err = self.sh(hostctl._head_cmd(rel), cfg.encode())
        self.assertEqual(rc, 0, err)
        self.assertEqual(hostctl.parse_head(out), 7)
        with open(os.path.join(self.home, "head-cfg")) as f:
            self.assertEqual(f.read(), cfg.rstrip("\n"))
        self.assertNotIn(_TOKEN, hostctl._head_cmd(rel))

    def test_manifest_read_write_and_df(self):
        rc, out, _ = self.sh(hostctl._MANIFEST_READ)
        self.assertEqual(hostctl.parse_manifest(out), {})
        rc, out, err = self.sh(hostctl._MANIFEST_WRITE, b'{"models/a": {"size": 1}}')
        self.assertEqual(rc, 0, err)
        rc, out, _ = self.sh(hostctl._MANIFEST_READ)
        self.assertEqual(hostctl.parse_manifest(out), {"models/a": {"size": 1, "aliases": []}})
        self.assertFalse(os.path.exists(os.path.join(self.home, ".gw-modelsync.json.tmp")))
        rc, out, err = self.sh(hostctl._DF_CMD)
        self.assertEqual(rc, 0, err)
        self.assertTrue(out.strip().isdigit(), out)


_MAIN = None


def _main():
    """Import main lazily (only MainWiring needs it) from a temp cwd with an empty
    config — the same trick test_faults/test_health_access use."""
    global _MAIN
    if _MAIN is None:
        import sys
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prev = os.getcwd()
        tmp = tempfile.mkdtemp(prefix="thunder-main-")
        _TMPDIRS.append(tmp)
        with open(os.path.join(tmp, "config.yaml"), "w") as f:
            f.write('api_key: ""\nbackends: []\n')
        os.chdir(tmp)
        sys.path.insert(0, here)
        try:
            import main as m
        finally:
            os.chdir(prev)
        _MAIN = m
    return _MAIN


class _FakeCtl:
    """Stands in for a Controller in the wiring tests: records calls, raises what the
    test scripts (a refusal is a RuntimeError before the first await). `name` is the
    managed HOST, carrying the ComfyUI backend of the same name as its service."""
    def __init__(self, name, phase="off", uuid="", refuse=None):
        self.name = name
        self.kind = "thunder"
        self.host = {"name": self.name, "provider": "thunder",
                     "options": {"gpu_type": "a6000"}, "api_key": "tok"}
        self.services = [{"name": name, "type": "comfyui", "host": name,
                          "local_port": 18188, "remote_port": 8188}]
        self.state = hostctl.State(phase=phase, uuid=uuid)
        self.refuse = refuse or {}
        self.calls = []
        self.gate = asyncio.Event()
        self.op = None               # DATA, like Controller.op — a method would be truthy

    async def _call(self, name):
        self.calls.append(name)
        if name in self.refuse:
            raise RuntimeError(self.refuse[name])
        await self.gate.wait()

    def start(self):
        return self._call("start")

    def stop(self):
        return self._call("stop")

    def restart_service(self, bid):
        return self._call("restart")

    def resume(self):
        return self._call("resume")

    def run_forever(self):
        return self._call("run_forever")

    def forget_unreconciled(self):
        self.calls.append("forget")

    async def aclose(self):
        self.calls.append("aclose")

    def has_service(self, bid):
        return bid in [f"{x['type']}:{x['name']}" for x in self.services]

    def set_services(self, services):
        self.services = list(services)

    def view(self):
        return {"phase": self.state.phase, "uptime_s": 42, "cost_per_h": 0.57,
                "log": ["x"], "provider": "thunder",
                "services": {f"{x['type']}:{x['name']}": {"status": "up"}
                             for x in self.services}}


class MainWiring(unittest.IsolatedAsyncioTestCase):
    """main's side of the managed hosts: the store's `managed_hosts` entries become
    controllers, the store backends naming a host become its services."""

    def setUp(self):
        m = self.m = _main()
        import store
        self.store = store
        self._saved = {n: getattr(m, n) for n in (
            "backends", "config_backends", "host_controllers", "_host_tasks",
            "_hosts_booted", "backend_inflight", "_draining", "jobs_cfg", "managed_hosts",
            "_host_errors", "_host_attached", "_host_not_attachable")}
        self._saved_store = (store._DB_PATH, store._active)
        self.tmp = tempfile.mkdtemp(prefix="thunder-wiring-")
        _TMPDIRS.append(self.tmp)
        store.init(os.path.join(self.tmp, "store.db"))
        m.host_controllers = {}
        m._host_tasks = {}
        m._hosts_booted = False
        m.managed_hosts = {}
        m._host_errors = {}
        m._host_attached = {}
        m._host_not_attachable = {}
        m.config_backends = []
        m.jobs_cfg = dict(m.jobs_cfg, store_path=os.path.join(self.tmp, "store.db"))

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(self.m, n, v)
        self.store._DB_PATH, self.store._active = self._saved_store

    def _host(self, name="tc", token="tok", **opts):
        """A managed host in the store; the token is its PROVIDER's (one per provider,
        encrypted there)."""
        self.store.set_provider_token("thunder", token)
        self.store.set_managed_host(name, {"provider": "thunder",
                                           "options": {"gpu_type": "a6000", **opts}})

    @staticmethod
    def _tb(name="tc", host=None, **kw):
        """A ComfyUI backend attached to the managed host `host` (default: its name)."""
        return dict({"name": name, "type": "comfyui", "url": "http://127.0.0.1:18188",
                     "host": name if host is None else host}, **kw)

    def test_controllers_follow_host_entries(self):
        m = self.m
        plain = {"name": "k12", "type": "comfyui", "url": "http://10.0.0.1:8188"}
        self._host()
        m.backends = [plain, self._tb()]
        m.sync_host_controllers()
        self.assertEqual(list(m.host_controllers), ["tc"])
        c = m.host_controllers["tc"]
        self.assertIsInstance(c, hostctl.Controller)
        self.assertEqual((c.name, c.host["provider"], c.host["api_key"]), ("tc", "thunder", "tok"))
        svc = c.services[0]
        self.assertIs(svc, m.backends[1])                   # the live dict itself
        self.assertEqual((svc["name"], svc["remote_port"]), ("tc", 8188))
        port = svc["local_port"]
        self.assertTrue(18100 <= port <= 18999)
        self.assertEqual(svc["url"], f"http://127.0.0.1:{port}")
        # a rebuild hands NEW dicts: the instance stays, its host/services are current
        self._host(gpu_type="h100")
        fresh = self._tb()
        m.backends = [dict(plain), fresh]
        m.sync_host_controllers()
        self.assertIs(m.host_controllers["tc"], c)
        self.assertEqual(c.cfg["gpu_type"], "h100")
        self.assertIs(c.services[0], fresh)
        self.assertEqual(fresh["local_port"], port)
        # entry removed while an instance runs → kept, warned
        c.state.phase = "ready"
        self.store.set_managed_host("tc", None)
        with self.assertLogs("main", "WARNING") as cm:
            m.sync_host_controllers()
        self.assertIs(m.host_controllers.get("tc"), c)
        self.assertIn("managed host entry removed while instance runs", "\n".join(cm.output))
        # … and once it is off, the next sync removes it
        c.state.phase = "off"
        m.sync_host_controllers()
        self.assertEqual(m.host_controllers, {})
        # a backend naming no managed host (or an old thunder block) drives nothing
        m.backends = [{"name": "tc", "type": "comfyui", "url": "http://x", "host": "tc",
                       "thunder": {"gpu_type": "a6000"}}]
        m.sync_host_controllers()
        self.assertEqual(m.host_controllers, {})

    def test_rebuild_backends_syncs_controllers(self):
        m = self.m
        self._host()
        self.store.upsert_backend(self._tb())
        m.rebuild_backends()
        c = m.host_controllers["tc"]
        m.rebuild_backends()
        self.assertIs(m.host_controllers["tc"], c)
        live = next(b for b in m.backends if b["name"] == "tc")
        self.assertIs(c.services[0], live)
        self.assertEqual(c.cfg, {"gpu_type": "a6000"})

    def test_set_backend_enabled_keeps_config_backend_whole(self):
        # a config-defined backend toggled in the console: each toggle writes a store
        # copy that overrides config WHOLESALE — every configured key must survive
        m = self.m
        extra = {"host": "gpu-box", "comfy_output_dir": "/home/ubuntu/ComfyUI/output",
                 "max_wait": 900, "auto_restart": True,
                 "models_allow": "flux*", "bypass": ["12"],
                 "sampling_defaults": {"stop": ["a"]}}
        m.config_backends = [dict(self._tb(host=""), api_key="tok", **extra)]
        m.rebuild_backends()
        for on in (True, False, True, False):          # two start/stop cycles
            self.assertTrue(m.set_backend_enabled("comfyui:tc", on))
            stored = self.store.get_backend("tc", "comfyui") or {}
            live = next(b for b in m.backends if b["name"] == "tc")
            for k, v in extra.items():
                self.assertEqual(stored.get(k), v, k)
                self.assertEqual(live.get(k), v, k)
            self.assertEqual(stored.get("api_key"), "tok")   # decrypted on read
            self.assertIs(stored.get("enabled"), on)
        self.assertEqual(m.host_controllers, {})           # a plain host drives nothing

    def test_set_backend_enabled_keeps_an_attached_backends_forward(self):
        # the lifecycle enables/disables its services on every start/stop: the stored
        # forward ends and the derived URL must survive every toggle
        m = self.m
        self._host()
        self.store.upsert_backend(self._tb())
        m.rebuild_backends()
        c = m.host_controllers["tc"]
        live = next(b for b in m.backends if b["name"] == "tc")
        ends = (live["local_port"], live["remote_port"], live["url"])
        for on in (False, True):
            self.assertTrue(m.set_backend_enabled("comfyui:tc", on))
            stored = self.store.get_backend("tc", "comfyui")
            self.assertEqual((stored["local_port"], stored["remote_port"], stored["url"]), ends)
            self.assertIs(stored.get("enabled"), on)
            self.assertIs(m.host_controllers["tc"], c)
            self.assertEqual(m.backend_host(c.services[0]), "tc")

    def test_config_backend_entry_is_the_whole_dict_minus_enabled(self):
        m = self.m
        b = {"name": "n", "type": "comfyui", "enabled": False, "_tmp": 1,
             "sampling_defaults": {"stop": ["a"]}, "paid": False, "stuck_after_s": 120}
        e = m._config_backend_entry(b)
        self.assertEqual(e, {"name": "n", "type": "comfyui", "sampling_defaults": {"stop": ["a"]},
                             "paid": False, "stuck_after_s": 120})
        e["sampling_defaults"]["stop"].append("b")          # a copy, not the live dict
        self.assertEqual(b["sampling_defaults"]["stop"], ["a"])

    def test_removed_entry_warns_once_per_controller(self):
        m = self.m
        self._host()
        m.backends = [self._tb()]
        m.sync_host_controllers()
        c = m.host_controllers["tc"]
        c.state.phase = "ready"
        self.store.set_managed_host("tc", None)
        with self.assertLogs("main", "WARNING") as cm:
            m.sync_host_controllers()
            m.sync_host_controllers()
            m.sync_host_controllers()
        self.assertEqual(len([x for x in cm.output if "entry removed" in x]), 1)
        # the entry returns and goes again → warned again
        self._host()
        m.sync_host_controllers()
        self.store.set_managed_host("tc", None)
        with self.assertLogs("main", "WARNING") as cm:
            m.sync_host_controllers()
        self.assertEqual(len([x for x in cm.output if "entry removed" in x]), 1)

    def test_off_controller_with_op_in_flight_is_not_retired(self):
        # a start is `off` until the create — retiring it then would orphan the instance
        m = self.m
        fc = _FakeCtl("tc")
        fc.op = "starting"
        m.host_controllers = {"tc": fc}
        m.backends = []
        with self.assertLogs("main", "WARNING"):
            m.sync_host_controllers()
        self.assertIs(m.host_controllers.get("tc"), fc)
        fc.op = None
        m.sync_host_controllers()
        self.assertEqual(m.host_controllers, {})

    def test_off_controller_with_a_pending_snapshot_is_not_retired(self):
        # the stop is done, but the watcher still has to rotate the old snapshot out —
        # retired now, both snapshots would bill per GB-month with nobody watching
        m = self.m
        fc = _FakeCtl("tc")
        fc.state.pending_snapshot = "s7"
        m.host_controllers = {"tc": fc}
        m.backends = []
        with self.assertLogs("main", "WARNING") as cm:
            m.sync_host_controllers()
        self.assertIs(m.host_controllers.get("tc"), fc)
        self.assertIn("s7", "\n".join(cm.output))
        fc.state.pending_snapshot = ""
        m.sync_host_controllers()
        self.assertEqual(m.host_controllers, {})

    def test_config_backend_entry_is_json_safe(self):
        # an unquoted YAML date is a datetime.date: the store's json.dumps raised on
        # every enable/disable
        import datetime
        m = self.m
        b = {"name": "n", "type": "comfyui", "note": datetime.date(2026, 9, 28),
             "extra": {"since": datetime.datetime(2026, 9, 28, 12, 0)}}
        e = m._config_backend_entry(b)
        self.assertEqual(e["note"], "2026-09-28")
        self.assertEqual(e["extra"]["since"], "2026-09-28 12:00:00")
        json.dumps(e)

    async def test_real_controller_refusal_comes_back_as_text(self):
        # pins the contract host_action relies on: Controller.stop() refuses BEFORE
        # its first await, so the refusal is the answer — not a background log line
        m = self.m
        self._host()
        m.backends = [self._tb()]
        m.sync_host_controllers()
        c = m.host_controllers["tc"]
        self.assertIsInstance(c, hostctl.Controller)
        self.assertEqual(c.state.phase, "off")
        with self.assertNoLogs("main", "WARNING"):          # answered, not logged twice
            self.assertEqual(await m.host_action("tc", "stop"), "stop refused: not running")
            self.assertEqual(await m.host_action("tc", "restart_service", bid="comfyui:tc"),
                             "restart of comfyui:tc refused: no running instance to restart "
                             "ComfyUI on (off)")
            self.assertEqual(await m.host_action("tc", "resetup", bid="comfyui:tc"),
                             "setup re-run of comfyui:tc refused: no running instance (off)")
            await asyncio.sleep(0)                          # let the done-callbacks run
        self.assertIsNone(c.op)

    def test_deps_wiring(self):
        m = self.m
        self._host()
        self._host("tb")
        m.backends = [self._tb(), self._tb("tb")]
        m.sync_host_controllers()
        deps = m.host_controllers["tc"].deps
        self.assertEqual(deps.datadir, self.tmp)
        self.assertIs(deps.set_enabled, m.set_backend_enabled)
        self.assertIs(deps.begin_drain, m.begin_drain)
        self.assertIs(deps.cancel_drain, m.cancel_drain)
        self.assertIs(deps.hold_routing, m._hold_routing)
        self.assertIs(deps.note_fault, m._note_fault)
        m.backend_inflight = {"comfyui:tc": 2}
        m._draining = {"comfyui:tc"}
        self.assertEqual(deps.inflight("comfyui:tc"), 2)
        self.assertEqual(deps.inflight("comfyui:tb"), 0)
        self.assertTrue(deps.is_draining("comfyui:tc"))
        self.assertFalse(deps.is_draining("comfyui:tb"))
        # persistence: one settings key, one entry per HOST name, others untouched
        deps.save_state("tc", {"phase": "ready", "uuid": "u1"})
        deps.save_state("tb", {"phase": "off"})
        self.assertEqual(self.store.get_setting("host_state"),
                         {"tc": {"phase": "ready", "uuid": "u1"},
                          "tb": {"phase": "off"}})
        self.assertEqual(deps.load_state("tc"), {"phase": "ready", "uuid": "u1"})
        self.assertIsNone(deps.load_state("nope"))
        # every controller's uuid is known (orphans = instances nobody owns)
        m.host_controllers["tc"].state.uuid = "u1"
        m.host_controllers["tb"].state.uuid = "u2"
        self.assertEqual(deps.known_uuids(), {"u1", "u2"})
        # the repo files
        self.assertTrue(deps.bootstrap_script().startswith(b"#!"))
        # R-W3: the host bootstrap is its own repo file
        self.assertTrue(deps.host_bootstrap_script().startswith(b"#!"))
        self.assertIn(b"disable_rc_autostart", deps.host_bootstrap_script())
        self.assertNotIn(b"disable_rc_autostart", deps.bootstrap_script())
        self.assertIsInstance(deps.default_nodes(), str)
        self.assertTrue(deps.default_nodes().strip())
        # forwards on the running master go through its control socket (R-W1)
        self.assertIs(deps.control, sshrun.control)
        cl = deps.client_factory()
        self.assertIsNot(cl, m.http_client)
        self.assertIsNot(cl, deps.client_factory())

    async def _moved_during_stop(self, job_ends_at_move):
        """H1 (a real Controller on the stub API) stops while a long job runs on its
        backend; during the drain the backend moves to H2 and H2 enables it — with
        main's REAL begin_drain / _finalize_drain / cancel_drain. Returns (controller,
        backend id)."""
        m = self.m
        self._saved["_drain_host"] = dict(m._drain_host)
        m._drain_host = {}
        m._draining = set()
        m.backend_inflight = {}

        async def no_refresh(*a, **k):          # apply_backend_change's discovery
            return None
        with mock.patch.object(m, "refresh_backend", no_refresh):
            self._host("tc")
            self._host("tb")
            self.store.upsert_backend(self._tb("gpu", host="tc"))
            m.rebuild_backends()
            live = next(b for b in m.backends if b["name"] == "gpu")
            bid = m.backend_id(live)
            c, _, _, _ = make(FakeThunder(), host_name="tc", services=[live])
            c.deps.set_enabled = m.set_backend_enabled
            c.deps.begin_drain = m.begin_drain
            c.deps.cancel_drain = m.cancel_drain
            c.deps.inflight = lambda x: m.backend_inflight.get(x, 0)
            c.deps.is_draining = lambda x: x in m._draining
            await c.start()
            self.assertEqual(c.state.phase, "ready", c.state.error)
            self.assertIs(self.store.get_backend("gpu", "comfyui")["enabled"], True)
            m._inflight_inc(bid)                                # the long job
            sleep, polls, moved = c.deps.sleep, [], []

            async def sleeping(sec):
                if c.state.phase == "draining":
                    polls.append(1)
                    self.assertLess(len(polls), 50, "the stop waits on a moved backend")
                    if not moved:
                        moved.append(1)
                        e = dict(self.store.get_backend("gpu", "comfyui"), host="tb")
                        self.store.upsert_backend(e)
                        m.rebuild_backends()                    # main hands the lists over
                        c.set_services([])
                        self.assertTrue(m.set_backend_enabled(bid, True))   # H2 start
                        if job_ends_at_move:
                            m._inflight_dec(bid)
                await sleep(sec)
            c.deps.sleep = sleeping
            await c.stop()
            self.assertEqual(c.state.phase, "off", c.state.error)
            if not job_ends_at_move:
                m._inflight_dec(bid)                            # the job ends after all
        return c, bid

    async def test_backend_moved_during_a_stop_stays_enabled_when_its_job_ends(self):
        # R-K2 through main's drain-finalize: the job through H1 ends right after the
        # move (before H1's next poll) — _finalize_drain must not disable it for H2
        m = self.m
        c, bid = await self._moved_during_stop(job_ends_at_move=True)
        self.assertIs(self.store.get_backend("gpu", "comfyui")["enabled"], True)
        self.assertNotIn(bid, m._draining)
        self.assertEqual(m._drain_host, {})

    async def test_backend_moved_during_a_stop_leaves_the_drain(self):
        # … and with the job still running: H1 stops waiting on it and cancels its
        # drain, so routing reaches it on H2; its job ending later disables nothing
        m = self.m
        c, bid = await self._moved_during_stop(job_ends_at_move=False)
        self.assertIs(self.store.get_backend("gpu", "comfyui")["enabled"], True)
        self.assertNotIn(bid, m._draining)
        live = next(b for b in m.backends if b["name"] == "gpu")
        self.assertFalse(m.is_draining(live))

    def test_hold_routing_never_disables(self):
        # an automatic restart's hold: routing skips the backend, its last request
        # ending does NOT disable it, and the release gives routing back; a real
        # take-offline meanwhile wins (the release leaves that drain alone)
        m = self.m
        self._saved["_drain_host"] = dict(m._drain_host)
        self._saved["_drain_hold"] = set(m._drain_hold)
        m._drain_host, m._drain_hold = {}, set()
        m._draining, m.backend_inflight = set(), {}
        self._host("tc")
        self.store.upsert_backend(self._tb("gpu", host="tc"))
        m.rebuild_backends()
        bid = "comfyui:gpu"
        live = lambda: next(b for b in m.backends if b["name"] == "gpu")
        m._inflight_inc(bid)
        self.assertTrue(m._hold_routing(bid, True))
        self.assertTrue(m.is_draining(live()))
        self.assertFalse(m._hold_routing(bid, True))        # already held
        m._inflight_dec(bid)
        self.assertIsNot(self.store.get_backend("gpu", "comfyui").get("enabled"), False)
        self.assertTrue(m._hold_routing(bid, False))
        self.assertFalse(m.is_draining(live()))
        # take-offline during a hold
        m._inflight_inc(bid)
        self.assertTrue(m._hold_routing(bid, True))
        self.assertTrue(m.begin_drain(bid))
        self.assertFalse(m._hold_routing(bid, False))
        self.assertTrue(m.is_draining(live()))
        m._inflight_dec(bid)
        self.assertIs(self.store.get_backend("gpu", "comfyui")["enabled"], False)

    def test_finalize_still_disables_a_backend_on_its_host(self):
        # the console's take-offline (and a host's own stop) keep disabling
        m = self.m
        self._saved["_drain_host"] = dict(m._drain_host)
        m._drain_host, m._draining, m.backend_inflight = {}, set(), {}
        self._host("tc")
        self.store.upsert_backend(self._tb("gpu", host="tc"))
        m.rebuild_backends()
        bid = "comfyui:gpu"
        m._inflight_inc(bid)
        self.assertTrue(m.begin_drain(bid))
        m._inflight_dec(bid)
        self.assertIs(self.store.get_backend("gpu", "comfyui")["enabled"], False)
        self.assertNotIn(bid, m._draining)

    def test_model_sync_deps(self):
        m = self.m
        saved_img = m.image_models
        wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux.safetensors"}},
              "2": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}}}
        path = os.path.join(self.tmp, "wf.json")
        with open(path, "w") as f:
            json.dump({"1": {"class_type": "CLIPLoader",
                             "inputs": {"clip_name": "t5.safetensors"}}}, f)
        self.store.upsert("img", [
            {"backend": "tc", "workflow_json": wf,
             "mapping": {"model": {"node": "1", "field": "unet_name", "label": "input_model"}},
             "fixed": [{"node": "2", "field": "vae_name", "value": "pinned.safetensors"}]},
            {"backend": "k12", "workflow_json": {"9": {"class_type": "VAELoader",
                                                       "inputs": {"vae_name": "k.st"}}}}])
        self.store.upsert("mesh", [{"backend": "tc", "meshy": {"endpoint": "image-to-3d"}}])
        m.image_models = {"cfgalias": [{"backend": "tc", "workflow": path}],
                          "img": [{"backend": "tc", "workflow_json": {}}]}   # store wins
        try:
            self._host()
            m.backends = [self._tb()]
            m.sync_host_controllers()
            deps = m.host_controllers["tc"].deps
            self.assertIs(deps.alias_needs, m.service_alias_needs)
            # called with the ComfyUI service's backend id; any other type needs nothing
            self.assertEqual(deps.alias_needs("openai:tc"), [])
            needs = {n.alias: n for n in deps.alias_needs("comfyui:tc")}
            self.assertEqual(sorted(needs), ["cfgalias", "img"])     # no cloud, no k12
            self.assertEqual(sorted((r.cls, r.value, r.selectable) for r in needs["img"].refs),
                             [("UNETLoader", "flux.safetensors", True),
                              ("VAELoader", "pinned.safetensors", False)])
            self.assertEqual([r.value for r in needs["cfgalias"].refs], ["t5.safetensors"])
            self.assertEqual(deps.alias_needs("comfyui:k12")[0].refs[0].value, "k.st")
            # the catalog: defaults while unset, the setting once set, [] when garbage
            self.assertEqual(deps.url_catalog({}), {})
            cat = [{"file": "models/vae/ae.safetensors", "url": "https://hf.co/x/ae.safetensors"},
                   {"match": {"alias": "img"}, "paths": ["models/loras/"]}]
            sig0 = deps.alias_signature("comfyui:tc")
            self.assertEqual(sig0, deps.alias_signature("comfyui:tc"))          # stable
            self.store.set_settings({"modelsync_catalog": cat})
            self.assertEqual(deps.url_catalog({}), {"models/vae/ae.safetensors":
                                                  {"url": "https://hf.co/x/ae.safetensors"}})
            self.assertEqual(needs["img"].catalog, [])
            self.assertEqual({n.alias: n for n in deps.alias_needs("comfyui:tc")}["img"].catalog,
                             ["models/loras/"])
            sig1 = deps.alias_signature("comfyui:tc")
            self.assertNotEqual(sig1, sig0)                              # catalog counts
            # another backend's candidate changes nothing, this backend's does
            self.store.upsert("other", [{"backend": "k12", "workflow_json": wf}])
            self.assertEqual(deps.alias_signature("comfyui:tc"), sig1)
            self.store.upsert("other", [{"backend": "tc", "workflow_json": wf}])
            sig2 = deps.alias_signature("comfyui:tc")
            self.assertNotEqual(sig2, sig1)
            self.store.delete("other")
            self.assertEqual(deps.alias_signature("comfyui:tc"), sig1)
            # a path workflow's content counts (same path, new file)
            with open(path, "w") as f:
                json.dump({"1": {"class_type": "CLIPLoader",
                                 "inputs": {"clip_name": "t5-v2.safetensors"}}}, f)
            os.utime(path, (1, 1))
            self.assertNotEqual(deps.alias_signature("comfyui:tc"), sig1)
            self.store.set_settings({"modelsync_catalog": {"not": "a list"}})
            self.assertEqual(deps.url_catalog({}), {})
            self.assertEqual(deps.source_index(), {})
            self.assertEqual(deps.hf_token(), "")
            self.store.set_settings({"hf_token": "hf_x"})
            self.assertEqual(deps.hf_token(), "hf_x")
        finally:
            m.image_models = saved_img

    def test_share_sha_cache_is_the_store_setting_and_feeds_the_plan(self):
        """Model sources: `modelsrc_sha` is read through ONE accessor (`_share_sha_files`,
        the host check inside LanSource), handed to `url_catalog` so a confirmed share
        hash that differs makes the entry outdated, and written back by the LanSource."""
        m = self.m
        path, h = "models/vae/a.st", "ab" * 32
        self.store.set_settings({"modelsrc_host": _SRCHOST})
        self.store.set_settings({"modelsrc_sha": {"host": _SRCHOST, "files": {path: [3, h]}}})
        m._modelsrc_obj = None
        self.assertEqual(m._share_sha_files(), {path: [3, h]})
        # review-2 I-2: a setting that grows with the share is no "all settings" member
        self.assertNotIn("modelsrc_sha", self.store.get_settings())
        self.assertEqual(self.store.get_setting("modelsrc_sha")["files"], {path: [3, h]})
        self._host()
        m.backends = [self._tb()]
        m.sync_host_controllers()
        deps = m.host_controllers["tc"].deps
        entry = {"file": path, "url": "https://example.com/a.st", "sha256": "cd" * 32, "size": 3}
        self.store.set_settings({"modelsync_catalog": [entry]})
        self.assertEqual(deps.url_catalog({path: 3}), {})          # the share disagrees
        self.store.set_settings({"modelsync_catalog": [dict(entry, sha256=h)]})
        self.assertEqual(deps.url_catalog({path: 3}),
                         {path: {"url": "https://example.com/a.st", "sha256": h}})
        # written back through the store; another host's record is no record
        m.modelsrc().forget_sha(path, 3)
        self.assertEqual(self.store.get_setting("modelsrc_sha"),
                         {"host": _SRCHOST, "files": {}})
        self.store.set_settings({"modelsrc_sha": {"host": "x@y", "files": {path: [3, h]}}})
        m._modelsrc_obj = None
        self.assertEqual(m._share_sha_files(), {})
        m._modelsrc_obj = None

    def test_unreadable_setting_is_never_overwritten(self):
        m = self.m
        self.store.set_settings({"host_state": ["garbage"]})
        self._host()
        m.backends = [self._tb()]
        m.sync_host_controllers()
        c = m.host_controllers["tc"]
        self.assertTrue(c.persist_blocked)          # load failed → no start, no save
        with self.assertRaises(ValueError):
            c.deps.save_state("tc", {"phase": "off"})
        self.assertEqual(self.store.get_setting("host_state"), ["garbage"])

    async def test_probe_comfy(self):
        m = self.m
        seen = []

        def handler(req):
            seen.append((str(req.url), req.extensions.get("timeout")))
            return httpx.Response(200 if "good" in str(req.url) else 502)
        saved = m.http_client
        m.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            deps = m._host_deps()
            self.assertTrue(await deps.probe_comfy("http://good:1"))
            self.assertFalse(await deps.probe_comfy("http://bad:1"))
        finally:
            await m.http_client.aclose()
            m.http_client = saved
        self.assertEqual(seen[0][0], "http://good:1/object_info")
        self.assertEqual(seen[0][1]["read"], 10)

    async def test_probe_comfy_transport_error_is_false(self):
        m = self.m

        def handler(req):
            raise httpx.ConnectError("refused")
        saved = m.http_client
        m.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            self.assertFalse(await m._host_deps().probe_comfy("http://x:1"))
        finally:
            await m.http_client.aclose()
            m.http_client = saved

    async def test_actions_run_in_background_and_refusals_are_text(self):
        m = self.m
        c = _FakeCtl("tc", refuse={"stop": "not running"})
        m.host_controllers = {"tc": c}
        msg = await m.host_action("tc", "start")
        self.assertIn("start", msg)
        self.assertEqual(c.calls, ["start"])
        # held (not GC-able) and still running: the call did not wait for the op
        held = [t for t in m._bg_refs if not t.done()]
        self.assertTrue(held)
        msg = await m.host_action("tc", "stop")
        self.assertIn("not running", msg)
        await m.host_action("tc", "restart_service", bid="comfyui:tc")
        self.assertEqual(c.calls[-1], "restart")
        c.state.unreconciled_uuids = ["u9"]
        await m.host_action("tc", "forget_unreconciled")
        self.assertEqual(c.calls[-1], "forget")
        self.assertIn("unknown", await m.host_action("nope", "start"))
        self.assertIn("unknown", await m.host_action("tc", "explode"))
        c.gate.set()
        await asyncio.sleep(0)

    async def test_host_view_and_names(self):
        # the host card's binds read managed hosts (card name = host name)
        m = self.m
        c = _FakeCtl("tc", phase="ready")
        m.host_controllers = {"tc": c}
        self.assertEqual(m.host_view("tc")["phase"], "ready")
        self.assertIsNone(m.host_view("nope"))
        import admin
        self.assertEqual(admin._host_names(), ["tc"])
        self.assertIs(admin._host_view, m.host_view)
        self.assertIs(admin._host_action, m.host_action)

    async def test_boot_resumes_and_runs_each_controller_and_shutdown_closes(self):
        m = self.m
        a, b = _FakeCtl("a", phase="ready"), _FakeCtl("b")
        self._host("a")
        self._host("b")
        m.host_controllers = {"a": a, "b": b}
        m._hosts_boot()
        await asyncio.sleep(0)
        self.assertEqual(sorted(a.calls), ["resume", "run_forever"])
        self.assertEqual(sorted(b.calls), ["resume", "run_forever"])
        # a controller that appears later (host added in the console) is started too
        self._host("late")
        m.backends = [self._tb("late")]
        m.sync_host_controllers()
        self.assertIs(m.host_controllers["a"], a)          # kept, handed the new lists
        late = m.host_controllers["late"]
        self.assertEqual(len(m._host_tasks["late"]), 2)
        for t in m._host_tasks["late"]:
            t.cancel()
        await m._hosts_shutdown()
        self.assertIn("aclose", a.calls)
        self.assertIn("aclose", b.calls)
        # the background loops are gone, the instance was never touched
        for ts in m._host_tasks.values():
            self.assertTrue(all(t.done() for t in ts))
        self.assertNotIn("stop", a.calls)
        self.assertEqual(late.state.phase, "off")

    async def test_health_carries_hosts_managed_in_full_view(self):
        m = self.m
        tb = self._tb()
        m.backends = [tb, {"name": "k12", "type": "comfyui", "url": "http://10.0.0.1:8188"}]
        m.host_controllers = {"tc": _FakeCtl("tc", phase="ready")}
        h = await m.health(verbose=False)
        self.assertEqual(h["hosts_managed"],
                         {"tc": {"provider": "thunder", "phase": "ready", "uptime_s": 42,
                                 "cost_per_h": 0.57, "services": {"comfyui:tc": "up"}}})
        # the per-backend Thunder block is gone (spec "main.py")
        self.assertNotIn("thunder", h["backends"]["comfyui:tc"])
        self.assertNotIn("thunder", h["backends"]["comfyui:k12"])

    def test_host_fault_pseudo_backend_is_recorded(self):
        # R-K1: the controller books machine events on {"name": <host>, "type":
        # "managed-host"} — main._note_fault and faults.record must take it (it has no
        # url, and no backend of that id exists), grouped apart from its services
        m = self.m
        with mock.patch.object(m.faults, "record") as rec:
            m._note_fault({"name": "thunder-tc", "type": hostctl.HOST_FAULT_TYPE},
                          "lifecycle", "instance_vanished", "gone")
        kw = rec.call_args.kwargs
        self.assertEqual((kw["bid"], kw["backend"], kw["type"], kw["host"]),
                         ("managed-host:thunder-tc", "thunder-tc", "managed-host",
                          "thunder-tc"))
        with mock.patch.object(m.faults, "_DB_PATH", None):
            ev = m.faults.record(**kw)
        self.assertEqual((ev["bid"], ev["type"], ev["kind"]),
                         ("managed-host:thunder-tc", "managed-host", "instance_vanished"))

    def test_deploy_and_gitignore_exclude_keys(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # thunder-ctl: the ControlMaster sockets — a deploy's rsync --delete must not
        # pull the live socket out from under a running master
        names = ["thunder.key", "thunder.key.pub", "thunder-known_hosts", "thunder-ctl",
                 "modelsrc.key", "modelsrc.key.pub", "modelsrc-known_hosts"]
        with open(os.path.join(root, ".gitignore")) as f:
            gi = {ln.strip().rstrip("/") for ln in f}
        with open(os.path.join(root, "deploy.sh")) as f:
            sh = f.read()
        rs = sh.split("RSYNC_EXCLUDES=(", 1)[1].split("\n)", 1)[0]
        tr = sh.split("TAR_EXCLUDES=(", 1)[1].split("\n)", 1)[0]
        for n in names:
            self.assertIn(n, gi, ".gitignore")
            self.assertRegex(rs, r"--exclude='%s/?'" % n.replace(".", r"\."), "rsync")
            self.assertRegex(tr, r"--exclude='\./%s/?'" % n.replace(".", r"\."), "tar")



# ── the LAN model source (Task 15: pinned host key, index, resumable stream) ──────────

_SRCHOST = "modelsrc@192.168.8.24"
_ED_B64 = "AAAAC3NzaC1lZDI1NTE5AAAAIHm4E0tb6VPU5qn5zKm6c1tJ4HQ1Pdu6Wf7k6o7V3tJ2"


class FakeShare:
    """The model share behind `ops/modelsrc-serve.sh`, answering LanSource's ssh by
    the forced command's verbs (`list`, `sha256 <rel>`) and `ssh-keyscan`. `files` are
    SHARE paths → bytes, `links` share path → target text."""

    def __init__(self):
        self.files, self.links = {}, {}
        self.calls = []
        self.list_rc, self.list_err = 0, b""
        self.sha_answers = []          # reported before the real digests (a mismatch)
        self.keyscan = (0, f"# 192.168.8.24:22 SSH-2.0-OpenSSH_9.6\n"
                           f"192.168.8.24 ssh-ed25519 {_ED_B64}\n".encode(), b"")

    def lists(self):
        return [a for a in self.calls if a[0] == "ssh" and a[-1] == "list"]

    async def ssh(self, argv, stdin=None, timeout=60):
        self.calls.append(list(argv))
        await asyncio.sleep(0)
        if argv[0] == "ssh-keyscan":
            return self.keyscan
        w = shlex.split(argv[-1])
        if w[0] == "list":
            if self.list_rc:
                return (self.list_rc, b"F\tpartial\t1\n", self.list_err)
            out = "".join(f"F\t{r}\t{len(b)}\n" for r, b in sorted(self.files.items()))
            out += "".join(f"L\t{r}\t{t}\n" for r, t in sorted(self.links.items()))
            return (0, out.encode(), b"")
        if w[0] == "sha256" and w[1] in self.files:
            if self.sha_answers:
                return (0, (self.sha_answers.pop(0) + "\n").encode(), b"")
            return (0, (hashlib.sha256(self.files[w[1]]).hexdigest() + "\n").encode(), b"")
        return (2, b"", b"modelsrc-serve: refused: nope")


async def _fake_keygen(path):
    return "ssh-ed25519 AAAAlan ai-hub"


def _lan(share, datadir, clock, pinned=True, host=_SRCHOST):
    lan = hostctl.LanSource(datadir, host=lambda: host, ssh=share.ssh,
                               keygen=_fake_keygen, now=lambda: clock[0])
    if pinned:
        with open(lan.known_hosts_path, "w") as f:
            f.write(f"192.168.8.24 ssh-ed25519 {_ED_B64}\n")
    return lan


class LanPipe:
    """`deps.pipe` for the tests: moves the share file's bytes from the offset the
    source argv names into the FakeVM `.part` the destination command names. `gate`
    (an Event) holds every stream until set; `active`/`peak` count concurrent streams."""

    def __init__(self, vm, share):
        self.vm, self.share = vm, share
        self.log = []
        self.gate = None
        self.active = self.peak = 0
        self.cancelled = 0

    async def __call__(self, src_argv, dst_argv, on_bytes, timeout_idle=120):
        self.log.append((list(src_argv), list(dst_argv)))
        w = shlex.split(src_argv[-1])
        rel, off = w[1], int(w[2])
        path = shlex.split(dst_argv[-1])[2]
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.gate is not None:
                await self.gate.wait()
            await asyncio.sleep(0)
            data = self.share.files[rel][off:]
            self.vm.content[path] = self.vm.content.get(path, b"") + data
            self.vm.files[path + ".part"] = len(self.vm.content[path])
            on_bytes(len(data))
            return (0, 0, "")
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1


def _lan_make(aliases, share, pinned=True, catalog=None, **kw):
    fake, vm, c, box, saved = _sync_make(aliases=aliases, catalog=catalog, **kw)
    lan = _lan(share, c.deps.datadir, c.h.clock, pinned=pinned)
    c.deps.lan = lan
    c.deps.source_index = lan.cached
    pipe = c.deps.pipe = LanPipe(vm, share)
    c.h.box = box
    return fake, vm, c, lan, pipe


class LanTransfer(unittest.IsolatedAsyncioTestCase):
    """Spec "Übertragung LAN": one stream at a time, resumed from the `.part`, verified by
    sha256 on both sides — and nothing at all without a pinned host key."""

    def share(self, **files):
        sh = FakeShare()
        for name, data in files.items():
            sh.files[f"diffusion_models/{name}.safetensors"] = data
        return sh

    async def test_lan_resume_from_part_offset(self):
        sh = self.share(a=b"0123456789")
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh)
        path = _dm("a.safetensors")
        vm.content[path] = b"012"
        vm.files[path + ".part"] = 3
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        src, dst = pipe.log[0]
        self.assertTrue(src[-1].endswith(" 3"), src)
        self.assertEqual(shlex.split(src[-1]), ["cat", "diffusion_models/a.safetensors", "3"])
        self.assertIn("StrictHostKeyChecking=yes", src)
        self.assertIn(f"UserKnownHostsFile={lan.known_hosts_path}", src)
        self.assertEqual(src[src.index("--") + 1], _SRCHOST)
        self.assertIn("cat >> ", dst[-1])
        self.assertIn("flock -n", dst[-1])
        self.assertEqual(vm.files[path], 10)
        man = json.loads(vm.manifest)[path]
        self.assertEqual((man["source"], man["size"]),  ("lan", 10))
        self.assertEqual(man["sha256"], hashlib.sha256(b"0123456789").hexdigest())
        self.assertEqual(len(pipe.log), 1)

    async def test_catalog_entry_judged_against_the_listing_the_plan_uses(self):
        """model sources: a sized catalog entry the share has outgrown (the file was
        replaced) is dropped against the SAME listing the plan is built from — the
        share's copy syncs over the LAN; an entry that still matches downloads by URL."""
        path = _dm("a.safetensors")
        for size, via in ((9, "lan"), (10, "url")):
            with self.subTest(size=size):
                sh = self.share(a=b"0123456789")
                entry = dict(_url("a.safetensors"), size=size)
                fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh,
                                                   catalog=[entry])
                vm.sizes[entry["url"]] = 10
                listings, seen = [], []
                real_src, real_urls = c.deps.source_index, c.deps.url_catalog
                c.deps.source_index = lambda: (listings.append(real_src()), listings[-1])[1]
                c.deps.url_catalog = lambda src: (seen.append(src), real_urls(src))[1]
                await c.start()
                self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                                c.state.log[-8:])
                self.assertEqual(json.loads(vm.manifest)[path]["source"], via)
                self.assertEqual(len(pipe.log), 1 if via == "lan" else 0)
                self.assertEqual(len(vm.started), 0 if via == "lan" else 1)
                listed = [s for s in seen if s]
                self.assertTrue(listed and all(any(s is x for x in listings) for s in listed))

    async def test_sha_mismatch_restarts_and_counts(self):
        sh = self.share(a=b"abcdef")
        sh.sha_answers = ["f" * 64]
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh)
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        # attempt 1 streamed everything, the digests differed: .part discarded, attempt 2
        # starts from byte 0 (not resumed onto bad bytes), then verifies
        self.assertEqual([shlex.split(s[-1])[2] for s, _ in pipe.log], ["0", "0"])
        self.assertTrue(any("LAN transfer models/diffusion_models/a.safetensors attempt 1/3 "
                            "failed: sha256 mismatch" in ln for ln in c.state.log), c.state.log)
        self.assertEqual(c.h.faults, [])
        # the source digest was asked again after the mismatch (not the cached bad one)
        self.assertEqual(len([a for a in sh.calls if a[-1].startswith("sha256 ")]), 2)

    async def test_three_mismatches_block_the_alias_and_log_a_fault(self):
        sh = self.share(a=b"abcdef")
        sh.sha_answers = ["f" * 64] * 3
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh)
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and "transfer failed"
                                     in c.alias_status(BID, "img")), c.state.log[-8:])
        self.assertEqual(len(pipe.log), 3)
        self.assertEqual(c.h.faults[-1][1:3], ("sync", "transfer"))
        self.assertNotIn(_dm("a.safetensors") + ".part", vm.files)

    async def test_short_stream_is_resumed_not_discarded(self):
        sh = self.share(a=b"0123456789")
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh)
        real = LanPipe.__call__
        calls = []

        async def short_once(src, dst, on_bytes, timeout_idle=120):
            calls.append(1)
            if len(calls) == 1:           # the connection dropped after 4 bytes
                path = shlex.split(dst[-1])[2]
                vm.content[path] = b"0123"
                vm.files[path + ".part"] = 4
                return (255, 0, "source: Connection reset")
            return await real(pipe, src, dst, on_bytes, timeout_idle)
        c.deps.pipe = short_once
        pipe.log = []
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        self.assertEqual(shlex.split(pipe.log[0][0][-1])[2], "4")
        self.assertTrue(any("stream failed (source rc 255, instance rc 0)" in ln
                            for ln in c.state.log))

    async def test_no_pin_no_lan_transfer(self):
        sh = self.share(a=b"abc")
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh, pinned=False)
        await c.start()
        await _until(lambda: _idle(c))
        self.assertEqual(pipe.log, [])
        self.assertEqual(sh.calls, [])            # not even a `list`
        self.assertIn("waiting for LAN source (not configured)", c.alias_status(BID, "img"))
        self.assertFalse(c.is_alias_ready(BID, "img"))

    async def test_hostile_host_setting_never_reaches_ssh(self):
        sh = self.share(a=b"abc")
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh)
        for bad in ("-oProxyCommand=touch /tmp/x", "a b@host", "user@ho'st", ""):
            lan._host_fn = lambda bad=bad: bad
            self.assertFalse(lan.configured(), bad)
            with self.assertRaises(ValueError):
                lan.cat_argv(_dm("a.safetensors"), 0)
        await c.start()
        await _until(lambda: _idle(c))
        self.assertEqual((sh.calls, pipe.log), ([], []))
        self.assertIn("waiting for LAN source (not configured", c.alias_status(BID, "img"))

    async def test_one_lan_stream_at_a_time(self):
        sh = self.share(a=b"a" * 5, b=b"b" * 7, c=b"c" * 9)
        fake, vm, c, lan, pipe = _lan_make(
            {"img": _cand("a.safetensors", "b.safetensors", "c.safetensors")}, sh)
        pipe.gate = asyncio.Event()
        await c.start()
        self.assertTrue(await _until(lambda: pipe.active == 1))
        for _ in range(50):
            await asyncio.sleep(0)
        self.assertEqual(pipe.active, 1)
        self.assertEqual([t["source"] for t in c.view()["transfers"]], ["lan"])
        pipe.gate.set()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        self.assertEqual((pipe.peak, len(pipe.log)), (1, 3))

    async def test_lan_and_url_run_side_by_side_and_url_not_withheld(self):
        # Ruling 18: with the source usable, an alias's URL files are no longer held back
        # for its LAN files (and the URL curls do not wait for the one LAN stream)
        sh = self.share(a=b"lan-bytes")
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors", "u.safetensors")},
                                           sh, catalog=[_url("u.safetensors")])
        vm.sizes["https://example.com/u.safetensors"] = 4
        pipe.gate = asyncio.Event()
        await c.start()
        self.assertTrue(await _until(lambda: vm.started and pipe.active == 1))
        pipe.gate.set()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        # and without the pin, the URL file of that alias is withheld as before
        sh2 = self.share(a=b"lan-bytes")
        fake, vm2, c2, lan2, pipe2 = _lan_make({"img": _cand("a.safetensors", "u.safetensors")},
                                               sh2, pinned=False,
                                               catalog=[_url("u.safetensors")])
        vm2.sizes["https://example.com/u.safetensors"] = 4
        await c2.start()
        await _until(lambda: _idle(c2))
        self.assertEqual((vm2.started, pipe2.log), ([], []))

    async def test_unreachable_source_waits_and_keeps_the_last_listing(self):
        sh = self.share(a=b"abc")
        sh.list_rc, sh.list_err = 255, b"ssh: connect to host 192.168.8.24 port 22: No route"
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh)
        await c.start()
        await _until(lambda: _idle(c))
        st = c.alias_status(BID, "img")
        self.assertIn("waiting for LAN source (unreachable: ssh: connect to host", st)
        self.assertEqual(pipe.log, [])
        # a good listing, then an incomplete one (rc 1): the good one stays, transfers wait
        sh.list_rc = 0
        await lan.refresh(force=True)
        good = lan.cached()
        self.assertEqual(good, {_dm("a.safetensors"): 3})
        sh.list_rc, sh.list_err = 1, b"modelsrc-serve: list incomplete (find exit 1)"
        gen = lan.generation
        await lan.refresh(force=True)
        self.assertEqual(lan.cached(), good)             # never "the source is empty"
        self.assertFalse(lan.usable())
        self.assertIn("list incomplete", lan.problem())
        self.assertGreater(lan.generation, gen)

    async def test_index_cached_and_invalidated_by_start_and_sync_now(self):
        sh = self.share(a=b"abc")
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh)
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        n = len(sh.lists())
        self.assertEqual(n, 1)
        c.h.clock[0] += 300
        await c._sync_tick()
        self.assertEqual(len(sh.lists()), n)             # within 10 min: cached
        c.h.clock[0] += 301
        await c._sync_tick()
        self.assertEqual(len(sh.lists()), n + 1)         # stale: listed again
        await c.sync_now()
        self.assertEqual(len(sh.lists()), n + 2)         # the button lists at once

    async def test_pin_makes_waiting_aliases_sync(self):
        sh = self.share(a=b"abc")
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh, pinned=False)
        await c.start()
        await _until(lambda: _idle(c))
        self.assertIn("waiting for LAN source (not configured)", c.alias_status(BID, "img"))
        fp = await lan.scan()
        lan.pin(fp)
        await c._sync_tick()                              # the pin alone triggers a plan
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])

    async def test_stop_ends_the_lan_stream(self):
        sh = self.share(a=b"abc")
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh)
        pipe.gate = asyncio.Event()                        # never set: a stream in flight
        await c.start()
        self.assertTrue(await _until(lambda: pipe.active == 1))
        await c._stop_transfers()
        self.assertEqual((pipe.active, pipe.cancelled), (0, 1))
        self.assertEqual(c.view()["transfers"], [])
        self.assertIsNone(c._lan_path)

    async def test_hf_links_created_recorded_and_not_recreated(self):
        sh = FakeShare()
        repo = "hf-cache/hub/models--o--n/"
        sh.files = {repo + "blobs/abc": b"weights", repo + "refs/main": b"r1"}
        sh.links = {repo + "snapshots/r1/model.safetensors": "../../blobs/abc",
                    # crosses into the other root: never recreated
                    "vae/x.safetensors": "../hf-cache/hub/models--o--n/blobs/abc"}
        cat = [{"match": {"alias": "hf"}, "paths": [repo]}]
        fake, vm, c, lan, pipe = _lan_make({"hf": {"backend": "thunder", "workflow_json": {}}},
                                           sh, catalog=cat)
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "hf")),
                        c.state.log[-8:])
        self.assertIn(repo + "blobs/abc", lan.cached())
        self.assertNotIn("models/vae/x.safetensors", lan.cached())
        snap = repo + "snapshots/r1/model.safetensors"
        self.assertEqual(vm.links, {snap: "../../blobs/abc"})
        linked = [st for m, cmd, st in fake.calls if m == "SSH" and cmd.startswith(": gw-link ")]
        self.assertEqual(linked, [f"{snap}\n../../blobs/abc\n".encode()])
        man = json.loads(vm.manifest)
        self.assertEqual({k: man[snap][k] for k in ("source", "target", "size", "aliases")},
                         {"source": "link", "target": "../../blobs/abc", "size": 0,
                          "aliases": ["hf"]})
        await c.sync_once()
        linked = [cmd for m, cmd, _ in fake.calls if m == "SSH" and cmd.startswith(": gw-link ")]
        self.assertEqual(len(linked), 1)                  # present links are not re-made
        self.assertTrue(c.is_alias_ready(BID, "hf"))
        # the alias goes: the stop prunes the link with the files it belongs to
        c.h.box["aliases"].clear()
        await c._before_snapshot()
        self.assertEqual(vm.links, {})
        self.assertNotIn(repo + "blobs/abc", vm.files)
        self.assertEqual(json.loads(vm.manifest), {})

    async def test_derived_hf_blob_downloads_by_url_and_its_link_is_made(self):
        """model sources Stage 1 end to end: a share HF-cache blob with a 64-hex name is
        fetched from huggingface.co by curl (checked against that sha), its snapshot link
        recreated, and only the tiny `refs/main` crosses the LAN (review M-4)."""
        sh = FakeShare()
        rev, oid = "0123456789abcdef0123456789abcdef01234567", "ab" * 32
        repo = "hf-cache/hub/models--org--repo/"
        blob, snap = repo + "blobs/" + oid, repo + f"snapshots/{rev}/model.safetensors"
        sh.files = {blob: b"weights", repo + "refs/main": rev.encode()}
        sh.links = {snap: "../../blobs/" + oid}
        cat = [{"match": {"alias": "hf"}, "paths": [repo]}]
        fake, vm, c, lan, pipe = _lan_make({"hf": {"backend": "thunder", "workflow_json": {}}},
                                           sh, catalog=cat)
        url = f"https://huggingface.co/org/repo/resolve/{rev}/model.safetensors"
        vm.sizes[url], vm.sha[url] = 7, oid
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "hf")),
                        c.state.log[-8:])
        self.assertEqual([p for p, _ in vm.started], [blob])
        self.assertIn(url, vm.started[0][1])
        man = json.loads(vm.manifest)
        self.assertEqual((man[blob]["source"], man[blob]["size"]), ("url", 7))
        self.assertEqual(man[repo + "refs/main"]["source"], "lan")
        self.assertEqual(len(pipe.log), 1)
        self.assertEqual(vm.links, {snap: "../../blobs/" + oid})
        self.assertEqual(man[snap]["source"], "link")

    async def test_plan_view_names_each_files_source(self):
        """model sources, the card's badge: every file row of the plan view says how THIS
        plan fetches it — `url` with `origin` hf-auto (derived) or catalog (an entry), a
        `link`, else `lan`. Without it the card can only guess from the catalog, which
        the plan may have overruled (an outdated entry, a fallback)."""
        sh = FakeShare()
        rev, oid = "0123456789abcdef0123456789abcdef01234567", "ab" * 32
        repo = "hf-cache/hub/models--org--repo/"
        blob, snap = repo + "blobs/" + oid, repo + f"snapshots/{rev}/model.safetensors"
        sh.files = {blob: b"weights", repo + "refs/main": rev.encode(),
                    "models/x/m.bin": b"0123456789"}
        sh.links = {snap: "../../blobs/" + oid}
        cat = [{"match": {"alias": "hf"}, "paths": [repo, "models/x/m.bin"]},
               {"file": "models/x/m.bin", "url": "https://mirror.example/m.bin", "size": 10}]
        fake, vm, c, lan, pipe = _lan_make({"hf": {"backend": "thunder", "workflow_json": {}}},
                                           sh, catalog=cat)
        vm.sizes[f"https://huggingface.co/org/repo/resolve/{rev}/model.safetensors"] = 7
        vm.sha[f"https://huggingface.co/org/repo/resolve/{rev}/model.safetensors"] = oid
        vm.sizes["https://mirror.example/m.bin"] = 10
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "hf")),
                        c.state.log[-8:])
        files = {f["path"]: f for f in c.view()["plan"]["aliases"]["hf"]["files"]}
        self.assertEqual((files[blob]["source"], files[blob]["origin"]), ("url", "hf-auto"))
        self.assertEqual((files["models/x/m.bin"]["source"], files["models/x/m.bin"]["origin"]),
                         ("url", "catalog"))
        self.assertEqual(files[repo + "refs/main"]["source"], "lan")
        self.assertNotIn("origin", files[repo + "refs/main"])
        self.assertEqual(files[snap]["source"], "link")
        # no URL in the view (a catalog URL may carry a query token)
        self.assertNotIn("mirror.example", json.dumps(c.view()["plan"]))

    async def test_links_wait_while_the_lan_source_waits(self):
        sh = FakeShare()
        repo = "hf-cache/hub/models--o--n/"
        sh.files = {repo + "blobs/abc": b"weights"}
        sh.links = {repo + "snapshots/r1/m.bin": "../../blobs/abc"}
        fake, vm, c, lan, pipe = _lan_make(
            {"hf": {"backend": "thunder", "workflow_json": {}}}, sh, pinned=False,
            catalog=[{"match": {"alias": "hf"}, "paths": [repo]}])
        # unpinned: no index at all, so the catalog dir is "waiting", and nothing is linked
        await c.start()
        await _until(lambda: _idle(c))
        self.assertEqual(vm.links, {})
        self.assertIn("waiting for LAN source (not configured)", c.alias_status(BID, "hf"))


class UrlRules(unittest.IsolatedAsyncioTestCase):
    """Model sources, "Download on the instance": a URL that served the wrong bytes or
    answered 4xx is FINAL for that URL at once — retrying downloads the same bytes, on a
    billed instance (review I-1, R-4, R-5); transport, 5xx and 429 keep the attempts."""

    def test_curl_http_status(self):
        st = hostctl.curl_http_status
        self.assertEqual(st("curl: (22) The requested URL returned error: 404"), 404)
        self.assertEqual(st("x\ncurl: (22) The requested URL returned error: 403 Forbidden\n"), 403)
        self.assertEqual(st("curl: (22) 404 not found"), 404)
        self.assertEqual(st("curl: (22) The requested URL returned error: 503"), 503)
        for text in ("curl: (7) Failed to connect", "curl: (28) Operation timed out after "
                     "300000 milliseconds", "", None, "curl: (56) error 404 in recv",
                     "curl: (22) no status here"):
            self.assertIsNone(st(text), text)
        final = hostctl.http_status_final
        self.assertTrue(all(final(n) for n in (400, 401, 403, 404, 410, 451)))
        self.assertFalse(any(final(n) for n in (408, 429, 500, 502, 503, None)))

    async def test_http_4xx_is_final_after_one_download(self):
        url = "https://example.com/x.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")],
                                         src={_dm("other.safetensors"): 5})
        vm.fail[url] = "curl: (22) The requested URL returned error: 404\n"
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.plan["per_alias"]["img"]["blocked"]))
        self.assertEqual(len(vm.started), 1)
        self.assertIn("404", c.alias_status(BID, "img"))
        self.assertIn(_dm("x.safetensors"), c._failed)
        # a URL-only file (the share does not list it) gives up and blocks as before
        self.assertEqual(c._url_fallback, {})
        kinds = [f[1:3] for f in c.h.faults]
        self.assertIn(("sync", "transfer"), kinds)
        self.assertNotIn(("sync", "url_fallback"), kinds)

    async def test_429_is_retried(self):
        url = "https://example.com/x.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.fail[url] = "curl: (22) The requested URL returned error: 429\n"
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.plan["per_alias"]["img"]["blocked"]))
        self.assertEqual(len(vm.started), 3)

    async def test_size_mismatch_is_final_after_one_download(self):
        url = "https://example.com/x.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.head_len[url] = 10                  # the HEAD names 10, the body has 8
        vm.sizes[url] = 8
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.plan["per_alias"]["img"]["blocked"]))
        self.assertEqual(len(vm.started), 1)
        self.assertEqual(len(_gw(fake, "gw-discard")), 1)
        self.assertIn("size 8", c._failed[_dm("x.safetensors")])


class UrlFallback(unittest.IsolatedAsyncioTestCase):
    """Model sources, "Fallback" (review I-2): a URL that ended final for a file the
    share also lists syncs the share's copy instead — after its curl is ended and the
    shared `.part` discarded, keyed on the URL, cleared by Sync now, Ruling 18 intact."""

    PATH = _dm("a.safetensors")
    URL = "https://example.com/a.safetensors"

    def share(self, data=b"0123456789"):
        sh = FakeShare()
        sh.files["diffusion_models/a.safetensors"] = data
        return sh

    async def test_hash_mismatch_falls_back_to_the_share_copy(self):
        sh = self.share()
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh,
                                           catalog=[_url("a.safetensors", sha="ab" * 32)])
        vm.sizes[self.URL], vm.sha[self.URL] = 10, "cd" * 32
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        self.assertEqual(len(vm.started), 1)                # ONE download, then the LAN
        man = json.loads(vm.manifest)
        self.assertEqual(man[self.PATH]["source"], "lan")
        self.assertEqual(len(pipe.log), 1)
        self.assertTrue(pipe.log[0][0][-1].endswith(" 0"), pipe.log[0][0])
        # the abandon (kill + discard) ran before the LAN attempt's resume offset read
        ab, part = _gw(fake, "gw-abandon"), _gw(fake, "gw-part")
        self.assertEqual(len(ab), 1)
        self.assertLess(ab[0], part[0])
        self.assertEqual(c._url_fallback, {self.PATH: self.URL})
        self.assertEqual(c.h.saved["thunder"]["url_fallback"], {self.PATH: self.URL})
        fb = [f for f in c.h.faults if f[1:3] == ("sync", "url_fallback")]
        self.assertEqual(len(fb), 1)
        self.assertIn("hash differs", fb[0][3])
        self.assertIn(self.PATH, fb[0][3])
        self.assertNotIn(("sync", "transfer"), [f[1:3] for f in c.h.faults])
        self.assertEqual(sum("syncing the share's copy" in ln for ln in c.state.log), 1)
        v = c.view()
        self.assertEqual(list(v["url_fallback"]), [self.PATH])
        self.assertIn("hash differs", v["url_fallback"][self.PATH])
        self.assertNotIn(self.URL, json.dumps(v["url_fallback"]))   # a URL may carry a token
        # the next plans keep it on the LAN: no second curl
        await c.sync_once()
        await _until(lambda: _idle(c))
        self.assertEqual(len(vm.started), 1)

    async def test_size_differs_names_both_sizes_and_syncs_the_share(self):
        sh = self.share()
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh,
                                           catalog=[_url("a.safetensors")])
        vm.sizes[self.URL] = 8                  # the URL's file is not the share's copy
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        self.assertEqual(len(vm.started), 1)
        why = c.view()["url_fallback"][self.PATH]
        # the overview's cheap accessor answers the same, without the whole view
        self.assertEqual(c.url_fallback_view(), c.view()["url_fallback"])
        self.assertIn("size differs", why)
        self.assertIn("8 bytes", why)
        self.assertIn("10 bytes", why)
        self.assertEqual(json.loads(vm.manifest)[self.PATH]["source"], "lan")

    async def test_derived_hf_blob_that_differs_falls_back(self):
        """Review M-3: a share copy that is not HF's canonical file — the derived URL's
        bytes have another size; the fallback says so and syncs the share's blob."""
        sh = FakeShare()
        rev, oid = "0123456789abcdef0123456789abcdef01234567", "ab" * 32
        repo = "hf-cache/hub/models--org--repo/"
        blob, snap = repo + "blobs/" + oid, repo + f"snapshots/{rev}/model.safetensors"
        sh.files = {blob: b"weights"}
        sh.links = {snap: "../../blobs/" + oid}
        fake, vm, c, lan, pipe = _lan_make(
            {"hf": {"backend": "thunder", "workflow_json": {}}}, sh,
            catalog=[{"match": {"alias": "hf"}, "paths": [repo]}])
        url = f"https://huggingface.co/org/repo/resolve/{rev}/model.safetensors"
        vm.sizes[url], vm.sha[url] = 9, oid
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "hf")),
                        c.state.log[-8:])
        self.assertEqual([p for p, _ in vm.started], [blob])
        self.assertIn("size differs", c.view()["url_fallback"][blob])
        man = json.loads(vm.manifest)
        self.assertEqual(man[blob]["source"], "lan")
        self.assertEqual(vm.links, {snap: "../../blobs/" + oid})

    async def test_fallback_kills_a_live_curl_and_discards_the_part_first(self):
        sh = self.share()
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh,
                                           catalog=[_url("a.safetensors")])
        vm.sizes[self.URL] = 10
        vm.files[self.PATH + ".part"] = 3        # bytes from the URL, never verified
        vm.poll_fail.add(self.PATH)              # the instance stops answering polls
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img"), 30),
                        c.state.log[-8:])
        # three attempts, each adopting the SAME curl (it ran on), then the fallback
        self.assertEqual(len(vm.started), 1)
        self.assertTrue(any("attempt 3/3 failed: the instance did not answer" in ln
                            for ln in c.state.log), c.state.log)
        self.assertEqual(vm.killed, [self.PATH])
        # the LAN stream started from byte 0, not on top of the URL's bytes
        self.assertEqual(shlex.split(pipe.log[0][0][-1]), ["cat", "diffusion_models/a.safetensors", "0"])
        self.assertEqual(vm.files[self.PATH], 10)
        self.assertEqual(json.loads(vm.manifest)[self.PATH]["source"], "lan")
        self.assertIn("did not answer", c.view()["url_fallback"][self.PATH])

    async def test_failed_abandon_gives_up_instead_of_switching(self):
        sh = self.share()
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh,
                                           catalog=[_url("a.safetensors", sha="ab" * 32)])
        vm.sizes[self.URL], vm.sha[self.URL] = 10, "cd" * 32
        run = vm.run

        def broken(cmd, stdin):
            if cmd.startswith(": gw-abandon "):
                return (255, b"", b"ssh: connection lost")
            return run(cmd, stdin)
        vm.run = broken
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.plan["per_alias"]["img"]["blocked"]))
        self.assertEqual(pipe.log, [])                 # never two writers on one .part
        self.assertEqual(c._url_fallback, {})
        self.assertIn("share's copy", c._failed[self.PATH])
        self.assertIn(("sync", "transfer"), [f[1:3] for f in c.h.faults])

    async def _fallen_back_lan_streaming(self):
        """A fallback whose LAN transfer is in flight (the stream held at `pipe.gate`)."""
        sh = self.share()
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh,
                                           catalog=[_url("a.safetensors", sha="ab" * 32)])
        vm.sizes[self.URL], vm.sha[self.URL] = 10, "cd" * 32
        pipe.gate = asyncio.Event()
        await c.start()
        self.assertTrue(await _until(lambda: pipe.active == 1), c.state.log[-8:])
        self.assertEqual(c._url_fallback, {self.PATH: self.URL})
        return sh, fake, vm, c, lan, pipe

    async def test_fallback_survives_a_gateway_restart_mid_lan_transfer(self):
        """Review-2 I-1: a restart that forgot the fallback planned the URL again — its
        curl resumed onto the LAN's `.part`, failed the same way, and the abandon threw
        the LAN progress away (a stream from zero through the uplink)."""
        sh, fake, vm, c, lan, pipe = await self._fallen_back_lan_streaming()
        for t in list(c._fetches.values()):           # the old process dies
            t.cancel()
        await _until(lambda: not c._fetches)
        vm.content[self.PATH] = b"0123"                # what the stream had delivered
        vm.files[self.PATH + ".part"] = 4
        n = len(vm.started)
        c2 = again(c)                                   # same store record, same VM
        self.assertEqual(c2._url_fallback, {self.PATH: self.URL})
        self.assertIn("hash differs", c2.view()["url_fallback"][self.PATH])
        pipe.gate.set()
        await c2.sync_once()
        self.assertTrue(await _until(lambda: _idle(c2) and c2.is_alias_ready(BID, "img")),
                        c2.state.log[-8:])
        self.assertEqual(len(vm.started), n)            # no curl onto the LAN's .part
        self.assertEqual(shlex.split(pipe.log[-1][0][-1]),
                         ["cat", "diffusion_models/a.safetensors", "4"])   # resumed
        self.assertEqual(json.loads(vm.manifest)[self.PATH]["source"], "lan")

    async def _restart_instance(self, c):
        """Stop (snapshot + delete) and start again: a NEW instance (`_created`)."""
        await _until(lambda: _idle(c))
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        await c.start()
        await _until(lambda: _idle(c))

    async def test_transport_fallback_is_forgotten_by_a_new_instance(self):
        """Task-2 re-review ruling: three TRANSPORT failures are about that session's
        network — a persisted record must not keep the next instance on the LAN."""
        sh = self.share()
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh,
                                           catalog=[_url("a.safetensors")])
        vm.sizes[self.URL] = 10
        vm.poll_fail.add(self.PATH)              # attempts run out on unanswered polls
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img"), 30),
                        c.state.log[-8:])
        self.assertEqual(c._url_fallback, {self.PATH: self.URL})
        self.assertEqual(c.state.url_fallback_cause, {self.PATH: "transport"})
        self.assertEqual(c.h.saved["thunder"]["url_fallback_cause"], {self.PATH: "transport"})
        await self._restart_instance(c)
        self.assertEqual(c._url_fallback, {})
        self.assertEqual(c.state.url_fallback_cause, {})
        self.assertEqual(c.h.saved["thunder"]["url_fallback"], {})
        self.assertTrue(any("tried again on this instance" in ln for ln in c.state.log),
                        c.state.log[-8:])

    async def test_mismatch_fallback_outlives_a_new_instance(self):
        sh = self.share()
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh,
                                           catalog=[_url("a.safetensors", sha="ab" * 32)])
        vm.sizes[self.URL], vm.sha[self.URL] = 10, "cd" * 32
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        self.assertEqual(c.state.url_fallback_cause, {self.PATH: "verdict"})
        await self._restart_instance(c)
        self.assertEqual(c._url_fallback, {self.PATH: self.URL})
        self.assertEqual(c.state.url_fallback_cause, {self.PATH: "verdict"})

    async def test_http_4xx_fallback_outlives_a_new_instance(self):
        sh = self.share()
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh,
                                           catalog=[_url("a.safetensors")])
        vm.fail[self.URL] = "curl: (22) The requested URL returned error: 404\n"
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        self.assertEqual(c.state.url_fallback_cause, {self.PATH: "verdict"})
        await self._restart_instance(c)
        self.assertEqual(c._url_fallback, {self.PATH: self.URL})

    def test_a_record_without_a_cause_counts_as_a_verdict(self):
        st = hostctl.state_from({"url_fallback": {"models/a": "https://x/a"},
                                 "url_fallback_why": {"models/a": "old"}})
        self.assertEqual(st.url_fallback_cause, {})
        fake = FakeThunder()
        c, saved, enabled, calls = make(fake)
        c.state = st
        c._created({"index": "0", "uuid": "u-1"}, 100, None, (False, False))
        self.assertEqual(c._url_fallback, {"models/a": "https://x/a"})

    async def test_forget_fallback_lets_the_next_plan_use_the_url(self):
        """Final review I-1: a VERDICT record for URL X, then a successful Check & save
        of the same X (main calls `forget_fallback`): the next plan downloads from X —
        `_without_fallbacks` alone keeps a record whose entry names the same URL."""
        sh = self.share()
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh,
                                           catalog=[_url("a.safetensors", sha="ab" * 32)])
        vm.sizes[self.URL], vm.sha[self.URL] = 10, "cd" * 32
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        self.assertEqual(c.state.url_fallback_cause, {self.PATH: "verdict"})
        # the operator fixed the cause; the file is gone from the disk again
        vm.sha[self.URL] = "ab" * 32
        vm.files.pop(self.PATH, None)
        vm.content.pop(self.PATH, None)
        n = len(vm.started)
        await c.sync_once()
        await _until(lambda: _idle(c))
        self.assertEqual(len(vm.started), n)            # still on the LAN while kept
        vm.files.pop(self.PATH, None)
        vm.content.pop(self.PATH, None)
        self.assertEqual(c.forget_fallback([self.PATH, "models/never/had.st"]), 1)
        self.assertEqual(c._url_fallback, {})
        self.assertEqual(c.h.saved["thunder"]["url_fallback"], {})
        self.assertEqual(c.forget_fallback([self.PATH]), 0)
        await c.sync_once()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        self.assertEqual(len(vm.started), n + 1)        # downloaded from the URL
        self.assertEqual(json.loads(vm.manifest)[self.PATH]["source"], "url")

    def test_forget_fallback_while_off_persists(self):
        fake = FakeThunder()
        c, saved, enabled, calls = make(fake)
        self.assertEqual(c.state.phase, "off")
        c.state.url_fallback = {"models/a": "https://x.example/a"}
        c.state.url_fallback_why = {"models/a": "hash differs"}
        c.state.url_fallback_cause = {"models/a": "verdict"}
        self.assertEqual(c.forget_fallback(["models/a"]), 1)
        st = hostctl.state_from(saved["thunder"])
        self.assertEqual((st.url_fallback, st.url_fallback_why, st.url_fallback_cause),
                         ({}, {}, {}))

    def test_template_report_kept_when_the_index_lists_no_models(self):
        """Final review M-4: an index without any `models/` path (a broken `~/ComfyUI`
        link) is no evidence that the template's files are gone."""
        fake = FakeThunder()
        c, saved, enabled, calls = make(fake)
        c.state.bootstrap_unknown = {"models/checkpoints/t.st": 5}
        c._prune_template_report({"hf-cache/hub/x": 1})
        self.assertEqual(c.state.bootstrap_unknown, {"models/checkpoints/t.st": 5})
        c._prune_template_report({})
        self.assertEqual(c.state.bootstrap_unknown, {"models/checkpoints/t.st": 5})
        c._prune_template_report({"models/vae/v.st": 1})
        self.assertEqual(c.state.bootstrap_unknown, {})

    async def test_sync_now_keeps_a_fallback_whose_lan_transfer_runs(self):
        sh, fake, vm, c, lan, pipe = await self._fallen_back_lan_streaming()
        n = len(vm.started)
        vm.sha[self.URL] = "ab" * 32
        await c.sync_now()
        self.assertEqual(c._url_fallback, {self.PATH: self.URL})
        self.assertTrue(any("stay given up while that copy is transferred" in ln
                            for ln in c.state.log), c.state.log[-5:])
        pipe.gate.set()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        self.assertEqual(len(vm.started), n)            # no curl started meanwhile
        self.assertEqual(json.loads(vm.manifest)[self.PATH]["source"], "lan")

    async def _fallen_back_with_lan_down(self, extra_catalog=()):
        """A fallback recorded while the LAN source has become unreachable (its last
        listing kept): the file is `lan` now and waits (Ruling 18)."""
        sh = self.share()
        fake, vm, c, lan, pipe = _lan_make(
            {"img": _cand("a.safetensors")}, sh,
            catalog=[_url("a.safetensors", sha="ab" * 32)] + list(extra_catalog))
        vm.sizes[self.URL], vm.sha[self.URL] = 10, "cd" * 32
        vm.hold.add(self.URL)
        await c.start()
        self.assertTrue(await _until(lambda: self.PATH in c._fetches))
        sh.list_rc = 255
        lan.invalidate()
        await lan.refresh()
        self.assertFalse(lan.usable())
        self.assertIn(self.PATH, lan.cached())
        vm.hold.clear()
        self.assertTrue(await _until(lambda: _idle(c) and self.PATH in c._url_fallback))
        await _until(lambda: _idle(c))
        return sh, fake, vm, c, lan, pipe

    async def test_ruling_18_and_a_changed_url_retries_by_itself(self):
        sh, fake, vm, c, lan, pipe = await self._fallen_back_with_lan_down()
        st = c.alias_status(BID, "img")
        self.assertIn("waiting for LAN source", st)
        self.assertEqual(pipe.log, [])
        self.assertEqual(c._fetchable(c.plan), [])
        # Ruling 18: a URL file the alias gains now is withheld while it waits
        g = _dm("g.safetensors")
        vm.sizes["https://example.com/g.safetensors"] = 4
        c.h.box["aliases"]["img"] = _cand("a.safetensors", "g.safetensors")
        c.h.box["catalog"].append(_url("g.safetensors"))
        await c._sync_tick()
        await _until(lambda: _idle(c))
        self.assertNotIn(g, [p for p, _ in vm.started])
        self.assertIn("waiting for LAN source", c.alias_status(BID, "img"))
        # a NEW url for the file (the operator fixed the entry) is tried by itself
        new = "https://mirror.example/a.safetensors"
        vm.sizes[new], vm.sha[new] = 10, "ab" * 32
        c.h.box["catalog"][0] = {"file": self.PATH, "url": new, "sha256": "ab" * 32}
        await c._sync_tick()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        self.assertIn(new, [re.search(r'url = "(.*)"', cfg).group(1) for _, cfg in vm.started])
        self.assertEqual(c._url_fallback, {})               # the stale record went with it
        self.assertEqual(c.view()["url_fallback"], {})
        self.assertEqual(json.loads(vm.manifest)[self.PATH]["source"], "url")
        self.assertEqual(pipe.log, [])

    async def test_sync_now_clears_the_fallback(self):
        sh, fake, vm, c, lan, pipe = await self._fallen_back_with_lan_down()
        n = len(vm.started)
        vm.sha[self.URL] = "ab" * 32                    # fixed upstream
        await c.sync_now()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")),
                        c.state.log[-8:])
        self.assertEqual(len(vm.started), n + 1)
        self.assertEqual(c._url_fallback, {})
        self.assertEqual(json.loads(vm.manifest)[self.PATH]["source"], "url")

    async def test_template_report_pruned_by_the_destination_index(self):
        fake, vm, c, box, saved = _sync_make(aliases={"img": _cand("x.safetensors")},
                                             catalog=[_url("x.safetensors")])
        vm.sizes["https://example.com/x.safetensors"] = 3
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        here, gone = "models/checkpoints/here.safetensors", "models/checkpoints/gone.safetensors"
        vm.files[here] = 5
        c.state.bootstrap_unknown = {here: 5, gone: 7}
        await c.sync_once()
        self.assertEqual(c.state.bootstrap_unknown, {here: 5})
        self.assertEqual(c.view()["bootstrap_unknown"], {here: 5})
        self.assertEqual(saved["thunder"]["bootstrap_unknown"], {here: 5})   # persisted


async def _sha_flushed(lan):
    """The share-sha write a synchronous path scheduled off the loop has landed."""
    if lan._sha_pending is not None:
        await lan._sha_pending


class ShareShaCache(unittest.IsolatedAsyncioTestCase):
    """Model sources, "Persistent share-sha cache" (review I-4, R-2): every share hash
    is kept in the store setting `modelsrc_sha`, bound to the share host, pruned by the
    listing, and computed one at a time."""

    PATH = "models/vae/a.st"

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="lan-sha-")
        self.addCleanup(shutil.rmtree, self.d, True)
        self.clock = [1000.0]
        self.box = {"v": None}
        self.saves = 0

    def lan(self, sh, host=None):
        hosts = host if isinstance(host, list) else [host or _SRCHOST]

        def save(v):
            self.saves += 1
            self.box["v"] = json.loads(json.dumps(v))
        lan = hostctl.LanSource(self.d, host=lambda: hosts[0], ssh=sh.ssh,
                                keygen=_fake_keygen, now=lambda: self.clock[0],
                                load_sha=lambda: self.box["v"], save_sha=save)
        with open(lan.known_hosts_path, "w") as f:
            f.write(f"192.168.8.24 ssh-ed25519 {_ED_B64}\n")
        return lan

    def share(self):
        sh = FakeShare()
        sh.files["vae/a.st"] = b"xyz"
        sh.files["vae/b.st"] = b"bbbb"
        return sh

    def sha_calls(self, sh):
        return [a for a in sh.calls if a[0] == "ssh" and shlex.split(a[-1])[0] == "sha256"]

    async def test_persisted_across_a_new_lansource(self):
        sh = self.share()
        h = await self.lan(sh).sha256(self.PATH, 3)
        self.assertEqual(self.box["v"], {"host": _SRCHOST, "files": {self.PATH: [3, h]}})
        lan2 = self.lan(sh)                           # a gateway restart
        self.assertEqual(lan2.sha_files(), {self.PATH: [3, h]})
        self.assertEqual(lan2.known_sha(self.PATH, 3), h)
        self.assertIsNone(lan2.known_sha(self.PATH, 4))
        self.assertEqual(await lan2.sha256(self.PATH, 3), h)
        self.assertEqual(len(self.sha_calls(sh)), 1)  # not hashed again

    async def test_record_of_another_host_is_ignored(self):
        sh = self.share()
        self.box["v"] = {"host": "src@10.0.0.9", "files": {self.PATH: [3, "ab" * 32]}}
        lan = self.lan(sh)
        self.assertEqual(lan.sha_files(), {})
        self.assertIsNone(lan.known_sha(self.PATH, 3))
        h = await lan.sha256(self.PATH, 3)
        self.assertEqual(h, hashlib.sha256(b"xyz").hexdigest())
        self.assertEqual(len(self.sha_calls(sh)), 1)
        self.assertEqual(self.box["v"], {"host": _SRCHOST, "files": {self.PATH: [3, h]}})

    async def test_junk_rows_are_ignored(self):
        sh = self.share()
        good = "cd" * 32
        self.box["v"] = {"host": _SRCHOST, "files": {
            self.PATH: [3, good], "models/vae/b.st": [True, good], "models/x": [1, "zz"],
            "models/y": "nope", 7: [1, good], "models/../z": [1, good]}}
        self.assertEqual(self.lan(sh).sha_files(), {self.PATH: [3, good]})
        for junk in ("junk", None, [], {"host": _SRCHOST, "files": "x"}):
            self.box["v"] = junk
            self.assertEqual(self.lan(sh).sha_files(), {}, junk)

    async def test_dropped_whole_on_a_host_change(self):
        sh = self.share()
        hosts = [_SRCHOST]
        lan = self.lan(sh, hosts)
        await lan.sha256(self.PATH, 3)
        gen = lan.sha_generation
        hosts[0] = "src@10.0.0.9"
        self.assertEqual(lan.sha_files(), {})
        await _sha_flushed(lan)
        self.assertEqual(self.box["v"], {"host": "src@10.0.0.9", "files": {}})
        self.assertGreater(lan.sha_generation, gen)
        hosts[0] = _SRCHOST                           # back: the old hashes are gone
        self.assertEqual(lan.sha_files(), {})

    async def test_forget_drops_both_copies(self):
        sh = self.share()
        lan = self.lan(sh)
        await lan.sha256(self.PATH, 3)
        lan.forget_sha(self.PATH, 3)
        self.assertEqual(lan.sha_files(), {})
        await _sha_flushed(lan)
        self.assertEqual(self.box["v"]["files"], {})
        self.assertEqual(self.lan(sh).sha_files(), {})
        await lan.sha256(self.PATH, 3)
        self.assertEqual(len(self.sha_calls(sh)), 2)

    async def test_pruned_by_a_fresh_listing(self):
        sh = self.share()
        lan = self.lan(sh)
        ha = await lan.sha256(self.PATH, 3)
        await lan.sha256("models/vae/b.st", 4)
        self.box["v"]["files"]["models/vae/gone.st"] = [1, "ab" * 32]   # an older session
        lan = self.lan(sh)
        del sh.files["vae/b.st"]                      # removed from the share
        sh.files["vae/a.st"] = b"xyz"
        await lan.refresh(force=True)
        self.assertEqual(lan.sha_files(), {self.PATH: [3, ha]})
        self.assertEqual(self.box["v"]["files"], {self.PATH: [3, ha]})
        sh.files["vae/a.st"] = b"wxyz"                 # replaced at another size
        await lan.refresh(force=True)
        self.assertEqual(lan.sha_files(), {})
        self.assertEqual(self.box["v"]["files"], {})
        # a failed listing prunes nothing
        await lan.sha256(self.PATH, 4)
        sh.list_rc = 255
        await lan.refresh(force=True)
        self.assertEqual(list(lan.sha_files()), [self.PATH])

    async def test_one_hash_at_a_time_with_a_visible_queue(self):
        sh = self.share()
        gate, active, peak = asyncio.Event(), [0], [0]
        real = sh.ssh

        async def ssh(argv, stdin=None, timeout=60):
            if argv[0] != "ssh-keyscan" and shlex.split(argv[-1])[0] == "sha256":
                active[0] += 1
                peak[0] = max(peak[0], active[0])
                try:
                    await gate.wait()
                    return await real(argv, stdin, timeout)
                finally:
                    active[0] -= 1
            return await real(argv, stdin, timeout)
        sh.ssh = ssh
        lan = self.lan(sh)
        lan._ssh = ssh
        self.assertEqual(lan.hash_queue(), [])
        lan.sha_files()                 # the record is read (in a thread on first need)
        ts = [asyncio.ensure_future(lan.sha256(self.PATH, 3)),
              asyncio.ensure_future(lan.sha256("models/vae/b.st", 4)),
              asyncio.ensure_future(lan.sha256(self.PATH, 3))]
        for _ in range(20):
            await asyncio.sleep(0)
        self.assertEqual(lan.hash_queue(), [self.PATH, "models/vae/b.st"])
        self.assertEqual(active[0], 1)
        gate.set()
        res = await asyncio.gather(*ts)
        self.assertEqual(res[0], res[2])
        self.assertEqual(peak[0], 1)
        self.assertEqual(len(self.sha_calls(sh)), 2)   # the duplicate waited for the first
        self.assertEqual(lan.hash_queue(), [])

    async def test_transfer_hashes_overtake_queued_background_ones(self):
        """Model sources M-5: a LAN transfer holds the one stream slot until its hash
        answers — it must not wait behind a queue of Check & save hashes. A background
        hash that already runs is not interrupted; a cancelled waiter frees its place."""
        sh = self.share()
        sh.files["vae/c.st"] = b"ccccc"
        sh.files["vae/d.st"] = b"dddddd"
        gates, order = {}, []
        real = sh.ssh

        async def ssh(argv, stdin=None, timeout=60):
            words = shlex.split(argv[-1]) if argv[0] != "ssh-keyscan" else [""]
            if words[0] == "sha256":
                order.append(words[1])
                g = gates.setdefault(words[1], asyncio.Event())
                await g.wait()
            return await real(argv, stdin, timeout)
        sh.ssh = ssh
        lan = self.lan(sh)
        lan._ssh = ssh
        lan.sha_files()
        a = asyncio.ensure_future(lan.sha256(self.PATH, 3, background=True))
        await asyncio.sleep(0.01)
        b = asyncio.ensure_future(lan.sha256("models/vae/b.st", 4, background=True))
        d = asyncio.ensure_future(lan.sha256("models/vae/d.st", 6, background=True))
        await asyncio.sleep(0.01)
        c = asyncio.ensure_future(lan.sha256("models/vae/c.st", 5))     # a transfer's
        await asyncio.sleep(0.01)
        self.assertEqual(lan.hash_queue(), [self.PATH, "models/vae/c.st", "models/vae/b.st",
                                            "models/vae/d.st"])
        d.cancel()                                   # a waiter gone: no hole in the queue
        await asyncio.sleep(0.01)
        self.assertEqual(lan.hash_queue(), [self.PATH, "models/vae/c.st", "models/vae/b.st"])
        for k in ("vae/a.st", "vae/c.st", "vae/b.st", "vae/d.st"):
            gates.setdefault(k, asyncio.Event()).set()
        await asyncio.wait_for(asyncio.gather(a, b, c), 5)
        self.assertEqual(order, ["vae/a.st", "vae/c.st", "vae/b.st"])
        self.assertEqual(lan.hash_queue(), [])
        # the slot is free again: the next request runs at once
        self.assertEqual(await asyncio.wait_for(lan.sha256("models/vae/d.st", 6), 5),
                         hashlib.sha256(b"dddddd").hexdigest())

    async def test_three_priorities_transfer_check_confirmation(self):
        """Review-3 M-3: a directory's background confirmations (2) queue behind the next
        Check & save (1), which queues behind a transfer (0)."""
        sh = self.share()
        sh.files["vae/c.st"] = b"ccccc"
        gate, order = asyncio.Event(), []
        real = sh.ssh

        async def ssh(argv, stdin=None, timeout=60):
            words = shlex.split(argv[-1]) if argv[0] != "ssh-keyscan" else [""]
            if words[0] == "sha256":
                order.append(words[1])
                await gate.wait()
            return await real(argv, stdin, timeout)
        sh.ssh = ssh
        lan = self.lan(sh)
        lan._ssh = ssh
        lan.sha_files()
        run = asyncio.ensure_future(lan.sha256("models/vae/c.st", 5, background=2))
        await asyncio.sleep(0.01)
        conf = asyncio.ensure_future(lan.sha256("models/vae/b.st", 4, background=2))
        chk = asyncio.ensure_future(lan.sha256(self.PATH, 3, background=1))
        await asyncio.sleep(0.01)
        self.assertEqual(lan.hash_queue(), ["models/vae/c.st", self.PATH, "models/vae/b.st"])
        gate.set()
        await asyncio.wait_for(asyncio.gather(run, conf, chk), 5)
        self.assertEqual(order, ["vae/c.st", "vae/a.st", "vae/b.st"])

    async def test_a_hash_answered_after_a_host_change_is_not_kept(self):
        sh = self.share()
        hosts = [_SRCHOST]
        lan = self.lan(sh, hosts)
        real = sh.ssh

        async def ssh(argv, stdin=None, timeout=60):
            res = await real(argv, stdin, timeout)
            hosts[0] = "src@10.0.0.9"                 # changed while it hashed
            return res
        lan._ssh = ssh
        with self.assertRaises(RuntimeError):
            await lan.sha256(self.PATH, 3)
        self.assertEqual(lan.sha_files(), {})
        await _sha_flushed(lan)
        self.assertEqual(self.box["v"], {"host": "src@10.0.0.9", "files": {}})

    async def test_synchronous_writes_leave_the_loop_and_never_go_backwards(self):
        """Review-2 M-2: `forget_sha`/`_follow_host` write the whole record — off the
        event loop when one runs; and a record built earlier never overwrites a later
        one, whichever worker thread finishes first."""
        import threading as _th
        sh = self.share()
        lan = self.lan(sh)
        await lan.sha256(self.PATH, 3)
        writers = []
        real = lan._save_sha

        def save(v):
            writers.append(_th.current_thread() is _th.main_thread())
            real(v)
        lan._save_sha = save
        lan.forget_sha(self.PATH, 3)
        await _sha_flushed(lan)
        self.assertEqual(writers, [False])
        self.assertEqual(self.box["v"]["files"], {})
        # ordering: the older record arrives last and is dropped
        old = lan._sha_record(_SRCHOST)
        new = (old[0] + 1, {"host": _SRCHOST, "files": {self.PATH: [3, "ab" * 32]}})
        lan._write_sha(*new)
        lan._write_sha(old[0], {"host": _SRCHOST, "files": {"models/vae/old.st": [1, "cd" * 32]}})
        self.assertEqual(self.box["v"], new[1])

    async def test_sha256_failures_are_runtime_errors(self):
        """Review-2 M-3: Check & save catches RuntimeError — an unset host or a path with
        no share mapping must not escape as a ValueError."""
        sh = self.share()
        with self.assertRaises(RuntimeError):
            await self.lan(sh, [""]).sha256(self.PATH, 3)
        with self.assertRaises(RuntimeError):
            await self.lan(sh).sha256("models/hf-cache/x", 3)

    async def test_memory_only_without_store_callables(self):
        sh = self.share()
        lan = _lan(sh, self.d, self.clock)
        h = await lan.sha256(self.PATH, 3)
        self.assertEqual(lan.sha_files(), {self.PATH: [3, h]})

    async def test_lan_transfer_hash_lands_in_the_store(self):
        sh = FakeShare()
        sh.files["diffusion_models/a.safetensors"] = b"0123456789"
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh)
        lan._load_sha = lambda: self.box["v"]
        lan._save_sha = lambda v: self.box.__setitem__("v", json.loads(json.dumps(v)))
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready(BID, "img")))
        self.assertEqual(self.box["v"]["files"],
                         {_dm("a.safetensors"): [10, hashlib.sha256(b"0123456789").hexdigest()]})


class LanSourceUnit(unittest.IsolatedAsyncioTestCase):
    """The pure half and the pin: mapping share ↔ plan paths, the list parser, argv
    shape, fingerprint, scan → pin."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="lan-test-")
        self.addCleanup(shutil.rmtree, self.d, True)
        self.clock = [1000.0]

    def test_share_mapping(self):
        self.assertEqual(hostctl.share_rel("models/vae/a.st"), "vae/a.st")
        self.assertEqual(hostctl.share_rel("hf-cache/hub/m/blobs/x"), "hf-cache/hub/m/blobs/x")
        for bad in ("models/hf-cache/hub/x", "models/hf-cache", "other/x", "models/",
                    "models/../x", "hf-cache/", "models/a/"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                hostctl.share_rel(bad)
        self.assertEqual(hostctl.plan_of_share("vae/a.st"), "models/vae/a.st")
        self.assertEqual(hostctl.plan_of_share("hf-cache/hub/x"), "hf-cache/hub/x")
        for bad in ("hf-cache", "../x", ".hidden/x", "-x/y", "a//b"):
            self.assertIsNone(hostctl.plan_of_share(bad), bad)

    def test_parse_source_list(self):
        text = ("F\tvae/a.st\t12\nF\thf-cache/hub/m/blobs/b\t7\n"
                "F\tbad size\tx\nF\t../x\t1\ngarbage\n"
                "L\thf-cache/hub/m/snapshots/r/w.st\t../../blobs/b\n"
                "L\tvae/cross.st\t../hf-cache/hub/m/blobs/b\n"      # other root: dropped
                "L\thf-cache/hub/m/snapshots/r/up.st\t../../../../../x\n"   # escapes
                "L\tvae/abs.st\t/etc/passwd\n"
                "L\tvae/same.st\ta.st\n")
        self.assertEqual(hostctl.parse_source_list(text), {
            "models/vae/a.st": 12, "hf-cache/hub/m/blobs/b": 7,
            "hf-cache/hub/m/snapshots/r/w.st": {"link": "../../blobs/b"},
            "models/vae/same.st": {"link": "a.st"}})

    def test_parse_index_reads_links(self):
        idx = hostctl.parse_index("ComfyUI/models/vae/a.st\t5\n"
                                     "L\thf-cache/hub/m/snapshots/r/x\t../../blobs/b\n"
                                     "L\t.hidden/l\tx\nGW:END\n")
        self.assertEqual(idx, {"models/vae/a.st": 5,
                               "hf-cache/hub/m/snapshots/r/x": {"link": "../../blobs/b"}})
        self.assertIn("-type l -printf 'L", hostctl._INDEX_CMD)

    def test_argv_pinned_and_quoted(self):
        sh = FakeShare()
        lan = _lan(sh, self.d, self.clock)
        a = lan.cat_argv("models/loras/my 'odd' lora.safetensors", 17)
        self.assertEqual(a[:5], ["ssh", "-F", "/dev/null", "-i",
                                 os.path.join(self.d, "modelsrc.key")])
        self.assertIn("StrictHostKeyChecking=yes", a)
        self.assertIn(f"UserKnownHostsFile={os.path.join(self.d, 'modelsrc-known_hosts')}", a)
        self.assertEqual(a[-3:-1], ["--", _SRCHOST])
        self.assertEqual(shlex.split(a[-1]), ["cat", "loras/my 'odd' lora.safetensors", "17"])
        with self.assertRaises(ValueError):
            lan.cat_argv("models/hf-cache/hub/x", 0)

    def test_fingerprint_is_openssh_s(self):
        import subprocess
        k = os.path.join(self.d, "k")
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", k], check=True)
        with open(k + ".pub") as f:
            b64 = f.read().split()[1]
        want = subprocess.run(["ssh-keygen", "-lf", k + ".pub"], check=True,
                              capture_output=True, text=True).stdout.split()[1]
        self.assertEqual(hostctl.host_key_fingerprint(b64), want)

    async def test_scan_then_pin_writes_known_hosts(self):
        sh = FakeShare()
        lan = _lan(sh, self.d, self.clock, pinned=False)
        self.assertFalse(lan.configured())
        with self.assertRaises(ValueError):
            lan.pin("SHA256:x")                           # nothing scanned
        fp = await lan.scan()
        self.assertEqual(sh.calls[-1], ["ssh-keyscan", "-t", "ed25519", "--", "192.168.8.24"])
        self.assertEqual(fp, hostctl.host_key_fingerprint(_ED_B64))
        self.assertFalse(os.path.exists(lan.known_hosts_path))    # memory only
        with self.assertRaises(ValueError):
            lan.pin("SHA256:someone-else")               # not the confirmed one
        gen = lan.generation
        lan.pin(fp)
        with open(lan.known_hosts_path) as f:
            self.assertEqual(f.read(), f"192.168.8.24 ssh-ed25519 {_ED_B64}\n")
        self.assertEqual(os.stat(lan.known_hosts_path).st_mode & 0o777, 0o600)
        self.assertTrue(lan.configured())
        self.assertEqual(lan.pinned_fingerprint(), fp)
        self.assertGreater(lan.generation, gen)

    async def test_scan_refuses_what_is_no_ed25519_key_of_that_host(self):
        sh = FakeShare()
        lan = _lan(sh, self.d, self.clock, pinned=False)
        for ans in ((0, b"# comment only\n", b""),
                    (0, f"10.0.0.9 ssh-ed25519 {_ED_B64}\n".encode(), b""),
                    (0, b"192.168.8.24 ssh-ed25519 not*base64\n", b""),
                    (1, b"", b"getaddrinfo: no such host")):
            sh.keyscan = ans
            with self.subTest(ans=ans), self.assertRaises(RuntimeError):
                await lan.scan()
        self.assertEqual(lan.scanned_fingerprint(), "")

    async def test_host_change_drops_the_index_and_names_the_stale_pin(self):
        """Task 15 review: after `modelsrc_host` changes, the old listing belongs to
        ANOTHER share and the pin to another host — neither may be used, and the reason
        must say what to do instead of a bare "unreachable"."""
        sh = FakeShare()
        sh.files["vae/a.st"] = b"xyz"
        host = ["modelsrc@192.168.8.24"]
        lan = hostctl.LanSource(self.d, host=lambda: host[0], ssh=sh.ssh,
                                   keygen=_fake_keygen, now=lambda: self.clock[0])
        with open(lan.known_hosts_path, "w") as f:
            f.write(f"192.168.8.24 ssh-ed25519 {_ED_B64}\n")
        await lan.refresh()
        await lan.sha256("models/vae/a.st", 3)
        self.assertEqual(lan.cached(), {"models/vae/a.st": 3})
        await lan.scan()
        gen = lan.generation
        host[0] = "src@10.0.0.9"
        self.assertEqual(lan.problem(),
                         "pinned for 192.168.8.24, not 10.0.0.9 — fetch its key")
        self.assertFalse(lan.configured())
        self.assertEqual(lan.cached(), {})
        self.assertGreater(lan.generation, gen)
        self.assertEqual(lan.scanned_fingerprint(), "")
        v = lan.view()
        self.assertFalse(v["pinned"])
        self.assertEqual(v["files"], 0)
        # back to the pinned host: pinned again, but the old listing is gone for good
        host[0] = "modelsrc@192.168.8.24"
        self.assertTrue(lan.configured())
        self.assertEqual(lan.problem(), "not listed yet")
        n = len(sh.calls)
        await lan.sha256("models/vae/a.st", 3)             # sha cache dropped with it
        self.assertEqual(len(sh.calls), n + 1)

    def test_one_host_read_per_view_and_the_pin_file_cached(self):
        """`problem()` runs every 5 s per controller and `view()` per page tick: the host
        setting is a store read in main and the pin a file open."""
        from unittest import mock
        reads = [0]

        def host():
            reads[0] += 1
            return "modelsrc@192.168.8.24"
        lan = hostctl.LanSource(self.d, host=host, ssh=FakeShare().ssh,
                                   keygen=_fake_keygen, now=lambda: self.clock[0])
        with open(lan.known_hosts_path, "w") as f:
            f.write(f"192.168.8.24 ssh-ed25519 {_ED_B64}\n")
        lan.view()
        reads[0] = 0
        opened = []
        real_open = open

        def spy(path, *a, **k):
            opened.append(str(path))
            return real_open(path, *a, **k)
        with mock.patch("builtins.open", spy):
            v = lan.view()
            lan.problem()
            lan.usable()
        self.assertEqual(reads[0], 3)                      # one per public entry point
        self.assertTrue(v["pinned"])
        self.assertNotIn(lan.known_hosts_path, opened)     # cached by inode/mtime/size
        # a changed file is read again
        with open(lan.known_hosts_path, "w") as f:
            f.write(f"10.1.1.1 ssh-ed25519 {_ED_B64}\n")
        self.assertEqual(lan.problem(),
                         "pinned for 10.1.1.1, not 192.168.8.24 — fetch its key")

    async def test_a_failed_host_read_is_no_host_change(self):
        sh = FakeShare()
        sh.files["vae/a.st"] = b"xyz"
        state = {"fail": False}

        def host():
            if state["fail"]:
                raise RuntimeError("database is locked")
            return _SRCHOST
        lan = hostctl.LanSource(self.d, host=host, ssh=sh.ssh, keygen=_fake_keygen,
                                   now=lambda: self.clock[0])
        with open(lan.known_hosts_path, "w") as f:
            f.write(f"192.168.8.24 ssh-ed25519 {_ED_B64}\n")
        await lan.refresh()
        gen = lan.generation
        state["fail"] = True
        self.assertEqual(lan.cached(), {"models/vae/a.st": 3})
        self.assertEqual(lan.generation, gen)
        state["fail"] = False
        self.assertEqual(lan.cached(), {"models/vae/a.st": 3})
        self.assertEqual(lan.generation, gen)

    async def test_sha_cached_per_path_and_size(self):
        sh = FakeShare()
        sh.files["vae/a.st"] = b"xyz"
        lan = _lan(sh, self.d, self.clock)
        h = await lan.sha256("models/vae/a.st", 3)
        self.assertEqual(h, hashlib.sha256(b"xyz").hexdigest())
        await lan.sha256("models/vae/a.st", 3)
        self.assertEqual(len(sh.calls), 1)
        await lan.sha256("models/vae/a.st", 4)            # another size: asked again
        self.assertEqual(len(sh.calls), 2)
        lan.forget_sha("models/vae/a.st", 3)
        await lan.sha256("models/vae/a.st", 3)
        self.assertEqual(len(sh.calls), 3)
        with self.assertRaises(RuntimeError):
            await lan.sha256("models/vae/missing.st", 3)  # refused (rc 2)

    async def test_refresh_is_cached_and_failures_retry_sooner(self):
        sh = FakeShare()
        sh.files["vae/a.st"] = b"x"
        lan = _lan(sh, self.d, self.clock)
        self.assertEqual(lan.problem(), "not listed yet")
        await lan.refresh()
        await lan.refresh()
        self.assertEqual(len(sh.lists()), 1)
        self.assertTrue(lan.usable())
        self.clock[0] += 599
        await lan.refresh()
        self.assertEqual(len(sh.lists()), 1)
        self.clock[0] += 2
        sh.list_rc, sh.list_err = 255, b"Host key verification failed."
        await lan.refresh()
        self.assertEqual(lan.problem(), "unreachable: Host key verification failed.")
        self.clock[0] += 61                               # a failure retries after 1 min
        sh.list_rc = 0
        await lan.refresh()
        self.assertEqual(len(sh.lists()), 3)
        self.assertTrue(lan.usable())

    async def test_no_default_host_means_not_configured_and_no_ssh(self):
        """There is no default share host: a baked-in LAN address sent the operator to
        create a user on a hypervisor. Empty = "not configured", and nothing — no list,
        no keyscan, no argv — ever reaches ssh; even a leftover pin changes nothing."""
        self.assertFalse(hasattr(hostctl, "MODELSRC_HOST_DEFAULT"))
        sh = FakeShare()
        sh.files["vae/a.st"] = b"x"
        lan = _lan(sh, self.d, self.clock, pinned=True, host="")
        # the LAN source lives in Server → Models (2026-10-01) — "below" was the
        # Backends tab, where it no longer is
        want = "LAN model source not configured — enter the share host under Server → Models"
        self.assertEqual(hostctl.SRC_UNSET, want)
        self.assertEqual(lan.problem(), want)
        self.assertFalse(lan.configured())
        self.assertFalse(lan.usable())
        self.assertEqual(lan.view()["problem"], want)
        # a pin left from an earlier host is pinned for NOBODY: the card must not read
        # "pinned · List now" beside "not configured" (prod's state after the upgrade)
        self.assertFalse(lan.pinned())
        self.assertFalse(lan.view()["pinned"])
        self.assertEqual(lan.view()["pinned_fp"], "")
        await lan.refresh(force=True)
        with self.assertRaises(ValueError) as cm:
            await lan.scan()
        self.assertIn("not configured", str(cm.exception))
        with self.assertRaises(ValueError):
            lan.cat_argv("models/vae/a.st", 0)
        self.assertEqual(sh.calls, [])
        # a whitespace-only setting is no host either
        lan2 = _lan(sh, self.d, self.clock, pinned=False, host="   ")
        self.assertEqual(lan2.problem(), want)
        self.assertFalse(lan2.pinned())
        # nor for a host that is no plain [user@]host
        bad = _lan(sh, self.d, self.clock, pinned=False, host="a;b")
        self.assertFalse(bad.pinned())
        self.assertFalse(bad.view()["pinned"])
        with self.assertRaises(ValueError) as cm:
            bad.cat_argv("models/vae/a.st", 0)
        self.assertTrue(str(cm.exception).startswith("LAN source not configured: "))
        self.assertEqual(sh.calls, [])
        # a host that is set but not pinned keeps the short "not configured"
        lan3 = _lan(sh, self.d, self.clock, pinned=False)
        os.remove(lan3.known_hosts_path)               # the first one's pin, same datadir
        self.assertEqual(lan3.problem(), "not configured")

    async def test_forced_refresh_lists_now_and_counts(self):
        """"List now" = refresh(force=True): lists inside the TTL, and the view carries
        what the last listing held."""
        sh = FakeShare()
        sh.files["vae/a.st"] = b"x"
        sh.files["hf-cache/hub/m/blobs/b1"] = b"yy"
        sh.links["hf-cache/hub/m/snapshots/r/w.st"] = "../../blobs/b1"
        lan = _lan(sh, self.d, self.clock)
        v = lan.view()
        self.assertEqual((v["files"], v["links"], v["listed_at"]), (0, 0, 0.0))
        await lan.refresh()
        await lan.refresh(force=True)
        self.assertEqual(len(sh.lists()), 2)
        v = lan.view()
        self.assertEqual((v["files"], v["links"], v["listed_at"]), (2, 1, 1000.0))


if __name__ == "__main__":
    unittest.main()


class FinalReviewFixes(unittest.IsolatedAsyncioTestCase):
    """The final whole-branch review's findings: each test fails without its fix."""

    async def _settle(self, n=20):
        for _ in range(n):
            await asyncio.sleep(0)

    @staticmethod
    def _gated_lists(fake, answers):
        """A handler whose /instances/list answers come from `answers` (callables on the
        fake, consumed in order, the last one sticks) — to script list sequences."""
        n = [0]

        def handler(req):
            if req.url.path == "/instances/list":
                fn = answers[min(n[0], len(answers) - 1)]
                n[0] += 1
                fake.calls.append(("GET", "/instances/list", None))
                return fn()
            return fake.handler(req)
        return handler, n

    # I1 ─ a stop during start's unreconciled check
    async def test_stop_during_the_unreconciled_check_means_no_create(self):
        fake = FakeThunder()
        _inst(fake, idx="4", uuid="u4")
        c, saved, _, _ = make(fake, state="garbage")
        await c.resume()
        self.assertEqual(c.state.unreconciled_uuids, ["u4"])
        del fake.instances["4"]                        # the check would let the start on
        fake.status_script = ["PROVISIONING", "RUNNING"]
        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(req):
            if req.url.path == "/instances/list" and not release.is_set():
                entered.set()
                await release.wait()
            return fake.handler(req)
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        c._api = c._client = None
        start = asyncio.ensure_future(c.start())
        await entered.wait()
        stopped = None
        try:
            await c.stop()
            stopped = True
        except RuntimeError:
            stopped = False
        release.set()
        await self._settle(50)
        try:
            await start
        except RuntimeError:
            pass
        await self._settle(50)
        # never both: a stop that answered "done" and an instance created after it
        self.assertFalse(stopped and _creates(fake), (stopped, _creates(fake)))
        self.assertEqual(_creates(fake), [])
        self.assertEqual(c.state.phase, "off")
        self.assertIsNone(c._op)

    async def test_stop_refuses_an_abortable_op_without_a_task(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        c._op, c._op_task = "starting", None
        with self.assertRaises(RuntimeError) as cm:
            await c.stop()
        self.assertIn("retry in a moment", str(cm.exception))
        self.assertEqual(c._op, "starting")            # the op is left alone

    # I2 ─ absence must hold over consecutive fresh lists
    async def test_unreconciled_check_needs_two_lists_without_the_instance(self):
        fake = FakeThunder()
        _inst(fake, idx="4", uuid="u4")
        c, saved, _, _ = make(fake, state="garbage")
        await c.resume()
        self.assertEqual(c.state.unreconciled_uuids, ["u4"])
        handler, n = self._gated_lists(fake, [lambda: httpx.Response(200, json={}),
                                              lambda: httpx.Response(200, json=fake.instances)])
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        c._api = c._client = None
        with self.assertRaises(RuntimeError) as cm:
            await c.start()
        self.assertIn("u4", str(cm.exception))
        self.assertEqual(n[0], 2)
        self.assertEqual(_creates(fake), [])
        self.assertEqual(c.state.unreconciled_uuids, ["u4"])
        self.assertEqual(saved["thunder"]["unreconciled_uuids"], ["u4"])

    async def test_unreadable_state_not_reconciled_as_off_on_one_empty_list(self):
        fake = FakeThunder()
        _inst(fake, idx="4", uuid="u4")
        c, saved, _, _ = make(fake, state="garbage")
        handler, n = self._gated_lists(fake, [lambda: httpx.Response(200, json={}),
                                              lambda: httpx.Response(200, json=fake.instances)])
        c.deps.client_factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await c.resume()
        self.assertEqual(n[0], 2)
        self.assertEqual(c.state.phase, "failed")
        self.assertEqual(c.state.unreconciled_uuids, ["u4"])
        self.assertIn("u4", c.state.error)

    # I3 ─ a failed save is visible and stops a create
    async def test_failed_save_before_the_post_refuses_the_create(self):
        fake = FakeThunder()
        c, saved, enabled, _ = make(fake)
        good = c.deps.save_state

        def boom(n, d):
            raise OSError("disk full")
        c.deps.save_state = boom
        await c.start()
        self.assertEqual(_creates(fake), [])
        self.assertEqual(c.state.phase, "off")
        self.assertIn("could not be saved", c.state.error)
        self.assertIs(enabled["comfyui:thunder"], False)
        self.assertIn("disk full", c.view()["persist_error"])
        c.deps.save_state = good
        c._persist()
        self.assertEqual(c.view()["persist_error"], "")

    # I5 ─ a transient HEAD failure can be retried
    async def test_sync_now_forgets_unknown_and_failed_head_sizes(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, state=_persisted())
        c._head_sizes = {"https://x/unknown": None, "https://x/good": 5, "https://x/failed": 7}
        c._failed = {"models/vae/f.safetensors": "size mismatch"}
        c._plan_inputs = ([], {}, {}, {}, {"models/vae/f.safetensors": {"url": "https://x/failed"}})

        async def nothing():
            return None
        c.sync_once = nothing
        await c.sync_now()
        self.assertEqual(c._head_sizes, {"https://x/good": 5})
        self.assertEqual(c._failed, {})

    async def test_fetch_start_failure_is_a_failed_attempt(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, state=_persisted(),
                          ssh_script={"gw-fetch": (0, b"GW:START-FAIL\ncurl: (6) no host\n", b"")})
        why, final = await c._fetch_attempt(
            {"path": "models/vae/v.safetensors", "url": "https://example.com/v", "size": None}, 1)
        self.assertEqual((why, final), ("curl did not start on the instance: curl: (6) no host",
                                        False))
        self.assertNotIn("models/vae/v.safetensors", c.state.transfers)

    # M1 ─ our instance vanishing at Thunder is noticed, the phase is not touched
    async def test_own_instance_absent_twice_is_a_fault_not_a_phase_change(self):
        fake = FakeThunder()
        fake.status_script = []
        c, _, _, _ = make(fake, state=_persisted())
        vanished = lambda: [f for f in c.h.faults if f[2] == "instance_vanished"]  # noqa: E731
        await c.refresh_account()
        self.assertEqual(vanished(), [])                   # one list is no proof
        await c.refresh_account()
        self.assertEqual(len(vanished()), 1)
        self.assertEqual(vanished()[0][1], "lifecycle")
        self.assertIn("u0", vanished()[0][3])
        await c.refresh_account()
        self.assertEqual(len(vanished()), 1)               # once per disappearance
        self.assertEqual((c.state.phase, c.state.uuid), ("ready", "u0"))
        self.assertTrue([ln for ln in c.state.log
                         if "no longer listed at Thunder Compute (" in ln])
        _inst(fake)                                        # listed again: reset
        await c.refresh_account()
        self.assertEqual(c._own_absent, 0)

    async def test_own_instance_listed_is_no_fault(self):
        fake = FakeThunder()
        _inst(fake)
        c, _, _, _ = make(fake, state=_persisted())
        for _ in range(3):
            await c.refresh_account()
        self.assertEqual([f for f in c.h.faults if f[2] == "instance_vanished"], [])

    # M7 ─ a Start without a token
    async def test_start_without_token_is_refused_before_any_call(self):
        fake = FakeThunder()
        c, _, enabled, _ = make(fake)
        c.host = dict(c.host, api_key="")
        with self.assertRaises(RuntimeError) as cm:
            await c.start()
        self.assertIn("no Thunder Compute API token set", str(cm.exception))
        self.assertIn("enter it under Server → API Keys", str(cm.exception))
        self.assertEqual(fake.calls, [])
        self.assertEqual(enabled, {})
        self.assertEqual(c.state.phase, "off")
        self.assertIsNone(c._op)


class IncludedVcpus(unittest.IsolatedAsyncioTestCase):
    """`vcpus` blank = the GPU configuration's included count (the smallest
    `vcpuOptions` entry), resolved at START before the create — never guessed: a count
    the specs cannot confirm ends the start in `off` with no create call."""

    def _make(self, fake, vcpus):
        c, saved, enabled, calls = make(fake)
        c.host["options"]["vcpus"] = vcpus
        return c, enabled

    async def test_blank_creates_with_the_included_count(self):
        fake = FakeThunder()
        fake.vcpu_options = [8, 6, 12]
        c, _ = self._make(fake, "")
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_creates(fake)[0]["cpu_cores"], 6)
        self.assertEqual(c.state.vcpus, 6)                  # what the instance has
        self.assertEqual(c.host["options"]["vcpus"], "")    # the stored option untouched
        self.assertEqual(c.view()["vcpus"], 6)

    async def test_unreadable_specs_end_in_off_without_a_create(self):
        for status, opts in ((500, [6, 8]), (200, [])):      # unreadable / no option
            fake = FakeThunder()
            fake.specs_status, fake.vcpu_options = status, opts
            c, enabled = self._make(fake, "")
            await c.start()
            self.assertEqual(c.state.phase, "off", status)
            self.assertEqual(_creates(fake), [], status)
            self.assertIn("cannot read Thunder's vCPU options for a6000 ×1 — set vcpus "
                          "explicitly or try again", c.state.error)
            self.assertNotIn("an instance may exist", c.state.error)
            self.assertFalse(any(enabled.values()), status)  # its enable undone

    async def test_typed_count_not_offered_is_refused_before_the_create(self):
        fake = FakeThunder()
        c, _ = self._make(fake, 12)
        await c.start()
        self.assertEqual(c.state.phase, "off")
        self.assertEqual(_creates(fake), [])
        self.assertIn("a6000 ×1 offers vCPUs 6, 8", c.state.error)

    async def test_typed_offered_count_is_used_as_is(self):
        fake = FakeThunder()
        c, _ = self._make(fake, 8)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_creates(fake)[0]["cpu_cores"], 8)

    async def test_typed_count_with_unreadable_specs_fails_before_the_create(self):
        # as before the included default: a specs outage ends the start before the
        # create (the disk limits come from the same answer) — nothing bills
        fake = FakeThunder()
        fake.specs_status = 503
        c, _ = self._make(fake, 8)
        await c.start()
        self.assertEqual(c.state.phase, "off")
        self.assertEqual(_creates(fake), [])
        self.assertIn("start failed", c.state.error)
        # nothing was POSTed: no orphan-list hint for an instance that cannot exist
        self.assertNotIn("an instance may exist", c.state.error)

    async def test_specs_blip_falls_back_to_the_cached_list(self):
        # the provider's own list from the last fetch (what the card and the form just
        # showed) is no guess: a failing fresh fetch uses it, for a blank and a typed count
        for vcpus, want in (("", 6), (8, 8)):
            fake = FakeThunder()
            c, _ = self._make(fake, vcpus)
            await c.refresh_prices()                        # the cache the card reads
            fake.specs_status = 500
            c.api._cache = {k: (-1e12, v) for k, (_, v) in c.api._cache.items()}  # expired
            await c.start()
            self.assertEqual(c.state.phase, "ready", c.state.error)
            self.assertEqual(_creates(fake)[0]["cpu_cores"], want)
            self.assertTrue([ln for ln in c.state.log if "using the cached list" in ln])

    async def test_create_transport_error_still_names_the_orphan_list(self):
        # the note stays where it is true: a create POST without an answer
        fake = FakeThunder()
        c, _ = self._make(fake, "")
        orig = fake.handler

        def handler(req):
            if req.url.path == "/instances/create":
                raise httpx.ConnectError("boom")
            return orig(req)
        fake.handler = handler
        await c.start()
        self.assertEqual(c.state.phase, "off")
        self.assertIn("an instance may exist anyway", c.state.error)

    async def test_view_and_cost_use_the_resolved_count(self):
        fake = FakeThunder()
        c, _ = self._make(fake, "")
        # nothing cached yet: the count is unknown — the card says "included" — and
        # the $/h of an included count bills no extra vCPU
        self.assertIsNone(c.view()["vcpus"])
        await c.refresh_prices()
        self.assertEqual(c.view()["vcpus"], 6)          # what a start would use
        self.assertAlmostEqual(c.view()["cost_per_h"], 0.35)
        c.host["options"]["vcpus"] = 8
        self.assertEqual(c.view()["vcpus"], 8)
        self.assertAlmostEqual(c.view()["cost_per_h"], 0.35 + 2 * 0.04)
        # a running instance: the count it was created with, whatever the options say now
        c.host["options"]["vcpus"] = ""
        await c.start()
        c.host["options"]["vcpus"] = 8
        self.assertEqual(c.view()["vcpus"], 6)
        self.assertAlmostEqual(c.view()["cost_per_h"], 0.35)
        # persisted with the instance and cleared once it is gone
        self.assertEqual(again(c).state.vcpus, 6)
        await c.stop()
        self.assertEqual((c.state.phase, c.state.vcpus), ("off", 0))


def _svc(name, typ, lport, rport, **kw):
    """An attached service as main hands it over: the backend dict plus its forward (a
    command service with a start command — the store fields arrive in Task 6)."""
    b = {"name": name, "type": typ, "url": f"http://127.0.0.1:{lport}",
         "local_port": lport, "remote_port": rport}
    if typ == "openai":
        b["svc_start"] = f"serve-stub --port {rport}"
    b.update(kw)
    return b


class _MasterTunnel(_NoTunnel):
    """Like the Supervisor: builds the master's argv from `_tunnel_argv` at every
    (re)spawn — which is when the controller learns what the master carries."""
    def __init__(self, c, fake):
        super().__init__(c, fake)
        self.argvs = []

    def start(self):
        super().start()
        self.argvs.append(self.c._tunnel_argv())

    def respawn(self):
        """The master died; the Supervisor spawns it again from the SAME argv_fn."""
        self.argvs.append(self.c._tunnel_argv())


def _fwds(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a == "-L"]


def _master(c, fake):
    def factory():
        t = _MasterTunnel(c, fake)
        c.h.tunnels.append(t)
        return t
    c._tunnel_factory = factory


def _controls(c, answer=(0, "")):
    calls = []

    async def control(ctl, host, op, lport, rport, timeout=15):
        calls.append((ctl, host, op, lport, rport))
        return answer() if callable(answer) else answer
    c.deps.control = control
    return calls


class ManagedHost(unittest.IsolatedAsyncioTestCase):
    """One host, several services (spec 2026-09-29 "Host-Controller", R-K1, R-W1,
    R-W4): what is the MACHINE's (enable/drain/disable of every service, snapshots and
    host faults by host name, one tunnel) and what is one SERVICE's (its forward, its
    status, its faults)."""

    def _three(self):
        return [_svc("thunder", "comfyui", 18188, 8188), _svc("vllm", "openai", 18200, 8000),
                _svc("embed", "openai", 18201, 8001)]

    async def test_stop_drains_every_service(self):
        # Review Focus 5: three services, one with a running job — all three drained,
        # the snapshot only after the last job, then all three off; a start turns all
        # three on again
        fake = FakeThunder()
        c, saved, enabled, _ = make(fake, services=self._three())
        bids = ["comfyui:thunder", "openai:vllm", "openai:embed"]
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(enabled, {b: True for b in bids})
        sv = c.view()["services"]
        self.assertEqual(sv["comfyui:thunder"]["status"], "up")
        self.assertEqual((sv["openai:vllm"]["local_port"], sv["openai:vllm"]["remote_port"]),
                         (18200, 8000))
        # the command services run through their wrappers (services.CommandProfile)
        self.assertEqual((sv["openai:vllm"]["status"], sv["openai:embed"]["status"]),
                         ("up", "up"))
        self.assertIn(services.COMMAND.start_cmd(self._three()[1]), _ssh_cmds(fake))
        drained, busy, seen = [], {"openai:vllm": [1, 1, 0]}, []
        c.deps.begin_drain = lambda bid: drained.append(bid) or True

        def inflight(bid):
            seq = busy.get(bid)
            v = seq.pop(0) if seq else 0
            fake.calls.append(("INFLIGHT", f"{bid}={v}", None))
            return v
        c.deps.inflight = inflight
        sleep = c.deps.sleep

        async def watching_sleep(sec):
            if c.state.phase == "draining":
                seen.append(c.view()["waiting_jobs"])
            await sleep(sec)
        c.deps.sleep = watching_sleep
        n = len(fake.calls)
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(drained, bids)
        kinds = _paths(fake, n)
        snap = kinds.index(("POST", "/snapshots/create"))
        busy_at = [i for i, k in enumerate(kinds) if k == ("INFLIGHT", "openai:vllm=1")]
        idle_at = kinds.index(("INFLIGHT", "openai:vllm=0"))
        self.assertEqual(len(busy_at), 2)
        self.assertLess(max(busy_at), snap)
        self.assertLess(idle_at, snap)
        self.assertEqual(seen, [{"openai:vllm": 1}, {"openai:vllm": 1}])
        self.assertIsNone(c.view()["waiting_jobs"])
        self.assertIn("waiting for 1 job(s) on openai:vllm", "\n".join(c.state.log))
        self.assertEqual(enabled, {b: False for b in bids})
        # … and the next start switches every one of them on again
        fake.status_script = ["PROVISIONING", "RUNNING"]      # the new instance's
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(enabled, {b: True for b in bids})

    async def test_drain_of_one_service_that_cannot_start_stops_the_stop(self):
        # a second service's begin_drain failing must not let routing go on sending it
        # jobs during the snapshot: the stop fails on it (instance kept), as before
        fake = FakeThunder()
        c, _, _, _ = make(fake, services=self._three())
        await c.start()

        def begin(bid):
            if bid == "openai:embed":
                raise OSError("store locked")
            return True
        c.deps.begin_drain = begin
        await c.stop()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "draining"))
        self.assertIn("openai:embed", c.state.error)
        self.assertEqual(c.state.uuid, "u0")                 # it bills: never forgotten

    async def test_enable_failure_of_a_later_service_disables_the_earlier_ones(self):
        fake = FakeThunder()
        c, _, enabled, _ = make(fake, services=self._three())
        calls = []

        def set_enabled(bid, on):
            calls.append((bid, on))
            if bid == "openai:embed" and on:
                return False
            enabled[bid] = on
            return True
        c.deps.set_enabled = set_enabled
        await c.start()
        self.assertEqual(c.state.phase, "off")
        self.assertIn("cannot enable backend openai:embed", c.state.error)
        self.assertEqual(_creates(fake), [])
        # the two it did enable go back off; the one that failed is not touched again
        self.assertEqual(enabled, {"comfyui:thunder": False, "openai:vllm": False})
        self.assertNotIn(("openai:embed", False), calls)

    async def test_no_service_refused(self):
        fake = FakeThunder()
        c, saved, enabled, _ = make(fake, services=[])
        with self.assertRaises(RuntimeError) as cm:
            await c.start()
        self.assertIn("no backend is attached to managed host thunder", str(cm.exception))
        self.assertEqual(fake.calls, [])                    # not one provider call
        self.assertEqual(enabled, {})
        self.assertEqual(c.state.phase, "off")
        self.assertIsNone(c._op)
        self.assertEqual(saved, {})

    async def test_snapshots_named_after_host(self):
        # R-W4: prefix, ownership, rotation and the restore template follow the HOST
        # name; a snapshot carrying the backend's name is foreign (display only)
        fake = FakeThunder()
        fake.snaps += [
            {"id": "sh", "name": "aihub-gpu-box-20260925t120000z", "status": "READY",
             "minimumDiskSizeGb": 100, "createdAt": 1},
            {"id": "sb", "name": "aihub-thunder-20260926t120000z", "status": "READY",
             "minimumDiskSizeGb": 100, "createdAt": 2}]
        c, saved, _, _ = make(fake, host_name="gpu-box")
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_creates(fake)[0]["template"], "aihub-gpu-box-20260925t120000z")
        self.assertEqual(c.state.snapshot_id, "sh")
        self.assertEqual(list(saved), ["gpu-box"])           # the state is the host's
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        body = next(b for m, p, b in fake.calls if p == "/snapshots/create")
        self.assertRegex(body["name"], r"^aihub-gpu-box-\d{8}t\d{6}z$")
        new = c.state.pending_snapshot
        for x in fake.snaps:
            if x["id"] == new:
                x["status"] = "READY"
        await c.watch_snapshots()
        ids = {x["id"] for x in fake.snaps}
        self.assertEqual(c.state.snapshot_id, new)
        self.assertNotIn("sh", ids)                          # the host's older one rotated
        self.assertIn("sb", ids)                             # the backend-named one never
        self.assertEqual([x["name"] for x in thunder.foreign_snapshots(
            await c.api.snapshots(), ["gpu-box"])], ["aihub-thunder-20260926t120000z"])
        # key and known_hosts per provider kind, the control socket per host
        d = c.deps.datadir
        self.assertEqual(c._key_path(), os.path.join(d, "thunder.key"))
        self.assertEqual(c._known_hosts_path("u0"), os.path.join(d, "thunder-known_hosts", "u0"))
        self.assertTrue(os.path.basename(c._ctl_path()).startswith("gpu-box-"))

    async def test_host_faults_use_pseudo_backend(self):
        # R-K1: machine events on {"name": <host>, "type": "managed-host"}, a service's
        # own failure (its bootstrap, its start, its transfers) on its backend
        host = {"name": "box", "type": "managed-host"}
        # instance gone (two account refreshes without it)
        fake = FakeThunder()
        c, _, _, _ = make(fake, host_name="box", state=_persisted())
        await c.refresh_account()
        await c.refresh_account()
        self.assertEqual([f[0] for f in c.h.faults if f[2] == "instance_vanished"], [host])
        # a pending snapshot FAILED
        fake = FakeThunder()
        fake.snaps.append({"id": "s5", "name": "aihub-box-20260926t120000z",
                           "status": "FAILED", "minimumDiskSizeGb": 100, "createdAt": 1})
        c, _, _, _ = make(fake, host_name="box", state=_persisted(phase="off", uuid="",
                                                                   index="", pending_snapshot="s5"))
        await c.watch_snapshots()
        self.assertEqual(c.h.faults[-1][0], host)
        self.assertEqual(c.h.faults[-1][2], "snapshot_failed")
        # the ComfyUI bootstrap failing → the ComfyUI service's fault and status; the
        # host is not failed for it (Ruling M4 g)
        fake = FakeThunder()
        c, _, _, _ = make(fake, host_name="box",
                          ssh_script={"bash -s": (1, b"GW:PHASE nodes\n", b"boom")})
        await c.start()
        self.assertEqual(c.state.phase, "ready")
        f = c.h.faults[-1]
        self.assertEqual((f[0]["name"], f[0]["type"], f[1], f[2]),
                         ("thunder", "comfyui", "lifecycle", "error"))
        self.assertEqual(len(c.h.faults), 1)
        self.assertEqual(c.view()["services"]["comfyui:thunder"]["status"], "setup failed")
        # ComfyUI never answering → the service's fault, status down with the reason
        fake = FakeThunder()
        c, _, _, _ = make(fake, host_name="box", probe=lambda url: asyncio.sleep(0, False))
        await c.start()
        self.assertEqual(c.state.phase, "ready")
        self.assertEqual((c.h.faults[-1][0]["name"], c.h.faults[-1][0]["type"]),
                         ("thunder", "comfyui"))
        sv = c.view()["services"]["comfyui:thunder"]
        self.assertEqual(sv["status"], "down")
        self.assertIn("did not answer", sv["error"])
        # a failure of the machine itself (the instance gone while starting) → the host
        fake = FakeThunder()
        fake.status_script = ["PROVISIONING", "TERMINATED"]
        c, _, _, _ = make(fake, host_name="box")
        await c.start()
        self.assertEqual(c.state.phase, "failed")
        self.assertEqual(c.h.faults[-1][0], host)

    async def test_forward_added_without_tunnel_restart(self):
        # Review Focus 2 part 1 (R-W1): a service attached while the host runs gets its
        # forward through the master's control socket — the tunnel carrying the other
        # service's stream is neither restarted nor respawned
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        _master(c, fake)
        calls = _controls(c)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        t = c.h.tunnels[0]
        argv = t.argvs[0]
        self.assertEqual(_fwds(argv), ["127.0.0.1:18188:127.0.0.1:8188"])
        ctl = argv[argv.index("-S") + 1]
        c.set_services(c.services + [_svc("vllm", "openai", 18200, 8000)])
        self.assertTrue(await _until(lambda: calls))
        self.assertEqual(calls, [(ctl, "ubuntu@10.0.0.5", "forward", 18200, 8000)])
        self.assertEqual(len(c.h.tunnels), 1)               # no new Supervisor
        self.assertTrue(t.running)
        self.assertEqual(len(t.argvs), 1)                   # and no respawn either
        self.assertEqual(c._fwd_active, {(18188, 8188), (18200, 8000)})
        # the same set again changes nothing
        c.set_services(list(c.services))
        await _until(lambda: c._fwd_task.done())
        self.assertEqual(len(calls), 1)
        # detached → cancel on the same socket, still no restart
        c.set_services(c.services[:1])
        self.assertTrue(await _until(lambda: len(calls) == 2))
        self.assertEqual(calls[-1], (ctl, "ubuntu@10.0.0.5", "cancel", 18200, 8000))
        self.assertEqual((len(c.h.tunnels), len(t.argvs)), (1, 1))
        self.assertEqual(c._fwd_active, {(18188, 8188)})

    async def test_failed_forward_marks_only_that_service_down(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        _master(c, fake)
        calls = _controls(c, (255, "Port forwarding failed."))
        await c.start()
        c.set_services(c.services + [_svc("vllm", "openai", 18200, 8000)])
        self.assertTrue(await _until(lambda: calls and c._fwd_task.done()))
        sv = c.view()["services"]
        self.assertEqual(sv["openai:vllm"]["status"], "down")
        self.assertIn("Port forwarding failed", sv["openai:vllm"]["error"])
        self.assertEqual(sv["comfyui:thunder"]["status"], "up")
        self.assertEqual(c.state.phase, "ready")
        # a master that dies meanwhile is respawned WITHOUT the failed forward — with
        # ExitOnForwardFailure it would take every service's tunnel down with it
        c.h.tunnels[0].respawn()
        self.assertEqual(_fwds(c.h.tunnels[0].argvs[-1]), ["127.0.0.1:18188:127.0.0.1:8188"])
        # the port is free again: the retry (run_forever's tick, once its backoff has
        # passed) adds it on the master
        _controls(c)                                        # answers rc 0 from now on
        c.h.clock[0] += hostctl._FWD_RETRY_MIN_S
        await c.reconcile_forwards()
        self.assertEqual(c._fwd_active, {(18188, 8188), (18200, 8000)})
        c.h.tunnels[0].respawn()
        self.assertEqual(len(_fwds(c.h.tunnels[0].argvs[-1])), 2)

    async def test_master_restart_carries_all_forwards(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        _master(c, fake)
        _controls(c)
        await c.start()
        c.set_services(c.services + [_svc("vllm", "openai", 18200, 8000),
                                     _svc("embed", "openai", 18201, 8001)])
        await _until(lambda: c._fwd_task is not None and c._fwd_task.done())
        # the master dies: the Supervisor's respawn builds its argv from the CURRENT set
        c.h.tunnels[0].respawn()
        argv = c.h.tunnels[0].argvs[-1]
        self.assertEqual(sorted(_fwds(argv)), ["127.0.0.1:18188:127.0.0.1:8188",
                                              "127.0.0.1:18200:127.0.0.1:8000",
                                              "127.0.0.1:18201:127.0.0.1:8001"])
        self.assertEqual(argv[-2:], ["--", "ubuntu@10.0.0.5"])
        self.assertEqual(c._fwd_active, {(18188, 8188), (18200, 8000), (18201, 8001)})
        # a service whose local port another one already forwards is left out (sshrun
        # refuses one port for two targets — the whole master would never start)
        c.set_services(c.services + [_svc("dup", "openai", 18200, 9000)])
        c.h.tunnels[0].respawn()
        self.assertNotIn("127.0.0.1:18200:127.0.0.1:9000", _fwds(c.h.tunnels[0].argvs[-1]))

    async def test_comfy_bootstrap_only_with_comfy_service(self):
        # R-W3: a host with no ComfyUI service runs the host bootstrap only — no
        # ComfyUI bootstrap, no commit or node list, starts no ComfyUI, syncs no models
        fake = FakeThunder()
        c, _, enabled, calls = make(fake, services=[_svc("vllm", "openai", 18200, 8000)])
        c.cfg["comfy_commit"] = "not-a-sha"              # ComfyUI-only requirements …
        c.cfg["nodes"] = []                               # … do not refuse this host
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        cmds = _ssh_cmds(fake)
        self.assertEqual([x for x in cmds if "bash -s" in x or "gw-nodes" in x],
                         ["bash -o pipefail -c "
                          "'bash -s -- 2>&1 | tee ~/gw-host-bootstrap.log'"])
        self.assertNotIn(hostctl._start_cmd(8188), cmds)
        self.assertEqual(enabled, {"openai:vllm": True})
        self.assertIsNone(c.plan)
        self.assertFalse(c.is_alias_ready("openai:vllm", "img"))
        # with a ComfyUI service the bootstrap runs, fed the script deps hand over
        fake = FakeThunder()
        c, _, _, calls = make(fake)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        boot = [(argv[-1], stdin) for argv, stdin in calls if "bash -s --" in argv[-1]]
        self.assertEqual(len(boot), 2)
        self.assertEqual([b[1] for b in boot], [HOST_SCRIPT, b"#!/bin/bash\necho GW:DONE\n"])
        self.assertIn(hostctl._start_cmd(8188), _ssh_cmds(fake))

    async def test_restart_pattern_uses_the_service_port(self):
        # the pkill pattern names the SERVICE's port: a fixed 8188 would kill nothing
        # (or another ComfyUI) for a service on another port
        # (Ruling M4 d: the loop takes the port; the restart starts it on the service's)
        self.assertTrue(hostctl._restart_cmd(8190).endswith(hostctl._start_cmd(8190)))
        self.assertIn("~/start-comfy.sh 8190 ", hostctl._start_cmd(8190))
        with self.assertRaises(ValueError):
            hostctl._restart_cmd(0)
        fake = FakeThunder()
        _ready_snap(fake)
        c, _, _, _ = make(fake, services=[_svc("thunder", "comfyui", 18188, 8190)])
        await c.start()
        n = len(fake.calls)
        await c.restart_comfy()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        cmds = [p for m, p, _ in fake.calls[n:] if m == "SSH"]
        self.assertEqual(cmds[0], hostctl._restart_cmd(8190))

    async def test_resume_probes_each_service_on_its_own_forward(self):
        # after a gateway restart every service is probed through ITS local port, and
        # the one that does not answer is restarted with ITS remote port in the pattern
        fake = FakeThunder()
        _inst(fake)
        up, asked = {"http://127.0.0.1:18190": False}, []

        async def probe(url):
            asked.append(url)
            await asyncio.sleep(0)
            return up.get(url, True)
        c, _, _, _ = make(fake, state=_persisted(), probe=probe,
                          services=[_svc("thunder", "comfyui", 18190, 8190),
                                    _svc("vllm", "openai", 18200, 8000)])
        n = len(fake.calls)
        task = asyncio.ensure_future(c.resume())
        self.assertTrue(await _until(
            lambda: any("pkill" in p for m, p, _ in fake.calls[n:] if m == "SSH")))
        up["http://127.0.0.1:18190"] = True
        await task
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(set(asked), {"http://127.0.0.1:18190"})
        kill = next(p for m, p, _ in fake.calls[n:] if m == "SSH" and "pkill" in p)
        self.assertEqual(kill, hostctl._restart_cmd(8190))
        self.assertEqual(c.view()["services"]["comfyui:thunder"]["status"], "up")

    async def test_tunnel_error_is_in_the_view(self):
        # Ruling M3: a tunnel whose spawn keeps failing says why on the card
        c, _, _, _ = make(FakeThunder())
        self.assertEqual(c.view()["tunnel_error"], "")
        c._tunnel = types.SimpleNamespace(running=False, last_spawn_error=
                                          "control socket in use, left in place: '/x'")
        self.assertIn("control socket in use", c.view()["tunnel_error"])

    async def test_ctl_path_falls_back_to_tmp_when_the_datadir_is_long(self):
        # Ruling M2: a data dir so deep that `<datadir>/<kind>-ctl/<name>` exceeds a Unix
        # socket's limit would respawn the tunnel forever on "path too long"
        base = tempfile.mkdtemp(prefix="hostctl-m2-")
        _TMPDIRS.append(base)
        deep = os.path.join(base, "d" * 70)
        os.makedirs(deep)
        c, _, _, _ = make(FakeThunder(), datadir=deep)
        p = c._ctl_path()
        self.assertTrue(p.startswith(f"/tmp/ai-hub-{os.getuid()}-ctl/thunder-thunder-"), p)
        self.assertLessEqual(len(p.encode()), sshrun.CTL_PATH_MAX)
        short, _, _, _ = make(FakeThunder())
        self.assertTrue(short._ctl_path().startswith(
            os.path.join(short.deps.datadir, "thunder-ctl") + "/"))
        # the fallback directory must be this uid's own real directory
        fb = os.path.join(base, "fb-{uid}")
        with mock.patch.object(hostctl, "_CTL_FALLBACK", fb):
            d = fb.format(uid=os.getuid())
            os.symlink(base, d)
            with self.assertRaisesRegex(ValueError, "not this gateway's own"):
                c._prepare_ctl()
            os.remove(d)
            path = c._prepare_ctl()
            self.assertEqual(os.path.dirname(path), d)
            self.assertEqual(os.stat(d).st_mode & 0o777, 0o700)


def _boots(fake, n=0):
    """(host, comfy) bootstrap runs among the ssh calls since fake.calls[n]."""
    cmds = [p for m, p, _ in fake.calls[n:] if m == "SSH"]
    return ([x for x in cmds if HOST_BOOT in x],
            [x for x in cmds if "tee ~/gw-bootstrap.log" in x])


class SplitBootstrap(unittest.IsolatedAsyncioTestCase):
    """R-W3 + Ruling M4: the host bootstrap runs on every host's first start, the
    ComfyUI bootstrap only with a ComfyUI service; each has its own done-flag, and a
    snapshot inherits both."""

    def _vllm(self, fake, **kw):
        c, saved, enabled, calls = make(fake, services=[_svc("vllm", "openai", 18200, 8000)],
                                        **kw)
        c.cfg.pop("bootstrap_template")          # the host form's default: unset
        return c, saved

    async def _stop_ready(self, c, fake):
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        fake.snaps[-1]["status"] = "READY"
        await c.watch_snapshots()
        fake.status_script = ["PROVISIONING", "RUNNING"]

    async def test_vllm_only_host_runs_the_host_bootstrap_on_base(self):
        fake = FakeThunder()
        c, saved = self._vllm(fake)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_creates(fake)[0]["template"], thunder.DEFAULT_TEMPLATE_NO_COMFY)
        host, comfy = _boots(fake)
        self.assertEqual((len(host), comfy), (1, []))
        self.assertIn("bootstrapping", c.h.phases)
        self.assertTrue(saved["thunder"]["host_bootstrapped"])
        self.assertTrue(c.view()["host_bootstrapped"])
        self.assertFalse(c.state.bootstrap_incomplete)
        self.assertTrue(c.state.comfy_absent)
        # the stop neither warns about an unfinished bootstrap nor marks the snapshot
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        joined = "\n".join(c.state.log)
        self.assertNotIn("did not finish", joined)
        self.assertNotIn("marked", joined)
        self.assertEqual((c.state.incomplete_snapshots, c.state.host_incomplete_snapshots),
                         ([], []))
        self.assertEqual(c.state.no_comfy_snapshots, ["s0"])

    async def test_configured_template_wins_and_comfy_default_is_comfy_ui(self):
        fake = FakeThunder()
        c, _ = self._vllm(fake)
        c.cfg["bootstrap_template"] = "comfy-ui"
        await c.start()
        self.assertEqual(_creates(fake)[0]["template"], "comfy-ui")
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        c.cfg.pop("bootstrap_template")
        await c.start()
        self.assertEqual(_creates(fake)[0]["template"], "comfy-ui")

    async def test_comfy_host_runs_host_then_comfy_bootstrap(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        cmds = _ssh_cmds(fake)
        host, comfy = _boots(fake)
        self.assertEqual((len(host), len(comfy)), (1, 1))
        self.assertLess(cmds.index("cat > ~/.gw-nodes.txt"), cmds.index(host[0]))
        self.assertLess(cmds.index(host[0]), cmds.index(comfy[0]))
        self.assertLess(cmds.index(comfy[0]), cmds.index(hostctl._start_cmd(8188)))
        self.assertEqual((saved["thunder"]["host_bootstrapped"],
                          saved["thunder"]["bootstrap_incomplete"],
                          saved["thunder"]["comfy_absent"]), (True, False, False))

    async def test_host_bootstrap_failure_fails_the_host(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake, ssh_script={
            HOST_BOOT: (1, b"GW:PHASE tools\n", b"host-bootstrap: failed in phase tools")})
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "bootstrapping"))
        self.assertIn("host bootstrap", c.state.error)
        self.assertIn("phase tools", c.state.error)
        self.assertIn("0", fake.instances)                  # kept for diagnosis
        self.assertEqual(_boots(fake)[1], [])               # the ComfyUI part never ran
        self.assertFalse(saved["thunder"]["host_bootstrapped"])
        # booked on the HOST's pseudo backend, not the ComfyUI service
        self.assertEqual(c.h.faults[-1][0], {"name": "thunder", "type": "managed-host"})
        self.assertNotEqual(c.view()["services"][BID]["status"], "setup failed")
        # its snapshot carries both marks; a start from it runs both bootstraps again
        await self._stop_ready(c, fake)
        self.assertTrue(any("host bootstrap did not finish" in ln for ln in c.state.log))
        self.assertEqual((c.state.host_incomplete_snapshots, c.state.incomplete_snapshots),
                         (["s0"], ["s0"]))
        c.deps.ssh = make(fake)[0].deps.ssh                 # both succeed now
        n = len(fake.calls)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_creates(fake)[-1]["template"], fake.snaps[0]["name"])
        host, comfy = _boots(fake, n)
        self.assertEqual((len(host), len(comfy)), (1, 1))

    async def test_host_flag_carried_into_snapshots(self):
        # vLLM-only: a failed host bootstrap marks the snapshot, a finished one does not
        fake = FakeThunder()
        c, saved = self._vllm(fake, ssh_script={HOST_BOOT: (124, b"", b"timeout")})
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "bootstrapping"))
        self.assertIn("30 min", c.state.error)
        await self._stop_ready(c, fake)
        self.assertEqual(saved["thunder"]["host_incomplete_snapshots"], ["s0"])
        self.assertEqual(c.state.incomplete_snapshots, [])  # no ComfyUI: never "incomplete"
        c.deps.ssh = make(fake)[0].deps.ssh
        n = len(fake.calls)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(len(_boots(fake, n)[0]), 1)       # ran again on the snapshot
        await self._stop_ready(c, fake)
        self.assertEqual(c.state.snapshot_id, "s1")
        self.assertNotIn("s1", c.state.host_incomplete_snapshots)   # (s0 rotated away)
        n = len(fake.calls)
        await c.start()                                     # from s1: nothing to run
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_boots(fake, n), ([], []))
        self.assertNotIn("bootstrapping", c.h.phases[-4:])

    async def test_comfy_attached_later_bootstraps_from_a_no_comfy_snapshot(self):
        fake = FakeThunder()
        c, _ = self._vllm(fake)
        await c.start()
        await self._stop_ready(c, fake)
        c.services = [_svc("thunder", "comfyui", 18188, 8188)]
        n = len(fake.calls)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_creates(fake)[-1]["template"], fake.snaps[0]["name"])
        host, comfy = _boots(fake, n)
        self.assertEqual((len(host), len(comfy)), (0, 1))  # host part is in the snapshot
        self.assertFalse(c.state.comfy_absent or c.state.bootstrap_incomplete)

    async def test_comfy_attached_to_a_running_host_is_bootstrapped(self):
        # a ComfyUI service attached to a running host without ComfyUI gets exactly the
        # ComfyUI bootstrap, once — through the attach op (Ruling M4 f)
        fake = FakeThunder()
        c, saved = self._vllm(fake)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        c.set_services(c.services + [_svc("thunder", "comfyui", 18188, 8188)])
        n = len(fake.calls)
        t = c._reconcile_services()
        self.assertEqual(c.op, "updating services")
        await t
        cmds = [p for m, p, _ in fake.calls[n:] if m == "SSH"]
        host, comfy = _boots(fake, n)
        self.assertEqual((host, len(comfy)), ([], 1))
        self.assertLess(cmds.index("cat > ~/.gw-nodes.txt"), cmds.index(comfy[0]))
        self.assertLess(cmds.index(comfy[0]), cmds.index(hostctl._start_cmd(8188)))
        self.assertEqual((saved["thunder"]["bootstrap_incomplete"],
                          saved["thunder"]["comfy_absent"]), (False, False))
        self.assertEqual(_sv(c)["status"], "up")
        self.assertIsNone(c.op)
        n = len(fake.calls)
        self.assertIsNone(c._reconcile_services())          # done: nothing runs
        self.assertEqual(_boots(fake, n), ([], []))

    async def test_ensure_comfy_bootstrap_only_inside_an_op(self):
        # Ruling M4 (f): never outside the op guard, never while the host stops
        fake = FakeThunder()
        c, _ = self._vllm(fake)
        await c.start()
        c.set_services(c.services + [_svc("thunder", "comfyui", 18188, 8188)])
        n = len(fake.calls)
        with self.assertRaisesRegex(RuntimeError, "inside a host operation"):
            await c._ensure_comfy_bootstrap()
        c._op = "stopping"
        with self.assertRaisesRegex(RuntimeError, "never while the host stops"):
            await c._ensure_comfy_bootstrap()
        with self.assertRaisesRegex(RuntimeError, "already stopping"):
            await c.resetup(BID)
        self.assertIsNone(c._reconcile_services())          # no attach into a stop
        c._op = None
        self.assertEqual(_boots(fake, n), ([], []))

    async def test_ensure_comfy_bootstrap_failure_is_the_services(self):
        fake = FakeThunder()
        c, saved = self._vllm(fake, ssh_script={"bash -s": (3, b"GW:SMOKE fail x\n", b"")})
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        comfy = _svc("thunder", "comfyui", 18188, 8188)
        c.set_services(c.services + [comfy])
        await c._reconcile_services()
        self.assertEqual(c.state.phase, "ready")            # the host is not failed
        self.assertEqual(_sv(c)["status"], "setup failed")
        self.assertEqual(_sv(c, "openai:vllm")["status"], "up")
        self.assertEqual(c.h.faults[-1][0]["name"], "thunder")
        self.assertTrue(saved["thunder"]["bootstrap_incomplete"])
        self.assertFalse(saved["thunder"]["comfy_absent"])
        self.assertFalse([x for x in _ssh_cmds(fake) if "start-comfy.sh" in x])
        # not retried every tick — the card's buttons are the way
        self.assertIsNone(c._reconcile_services())
        # a snapshot of this instance now holds a half-install: marked incomplete
        await c.stop()
        self.assertEqual(c.state.incomplete_snapshots, ["s0"])

    async def test_resume_of_an_interrupted_host_bootstrap_names_its_log(self):
        fake = FakeThunder()
        _inst(fake)
        c, _, _, _ = make(fake, state=_persisted(phase="bootstrapping",
                                                 host_bootstrapped=False))
        await c.resume()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "bootstrapping"))
        self.assertIn("gw-host-bootstrap.log", c.state.error)

    async def test_resume_interrupted_first_start_runs_both(self):
        fake = FakeThunder()
        _inst(fake)
        c, _, _, _ = make(fake, state=_persisted(phase="creating", ip="", port=0,
                                                 host_bootstrapped=False,
                                                 bootstrap_incomplete=True))
        await c.resume()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        host, comfy = _boots(fake)
        self.assertEqual((len(host), len(comfy)), (1, 1))

    async def test_host_bootstrap_finished_by_hand_is_not_rerun(self):
        # review fix 1: an operator who finished a failed host bootstrap by hand and
        # restarted ComfyUI has a working host — a later start must not re-run the host
        # bootstrap, whose inventory would take the synced models for the template's
        tpl = b"GW:UNKNOWN_MODEL models/checkpoints/tpl.safetensors\t5000000\n"
        fake = FakeThunder()
        c, saved, _, _ = make(fake, ssh_script={
            HOST_BOOT: (1, b"GW:PHASE tools\n" + tpl, b"host-bootstrap: failed in phase tools")})
        await c.start()
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "bootstrapping"))
        unknown = dict(c.state.bootstrap_unknown)
        self.assertEqual(unknown, {"models/checkpoints/tpl.safetensors": 5000000})
        await c.restart_comfy()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertTrue(saved["thunder"]["host_bootstrapped"])
        self.assertTrue(any("host bootstrap counts as done" in ln for ln in c.state.log))
        await self._stop_ready(c, fake)
        self.assertEqual((c.state.host_incomplete_snapshots, c.state.incomplete_snapshots),
                         ([], []))
        n = len(fake.calls)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_boots(fake, n), ([], []))
        self.assertEqual(c.state.bootstrap_unknown, unknown)

    async def test_host_rerun_uploads_nodes_and_skips_our_models(self):
        # review fixes 1+4: a host bootstrap re-run on a snapshot (the ComfyUI part is
        # complete) gets the current node list first — its template-node report leaves
        # our packs out — and never reports a file the snapshot's manifest knows
        fake = FakeThunder()
        _ready_snap(fake)
        out = (b"GW:UNKNOWN_MODEL models/checkpoints/ours.safetensors\t9000000\n"
               b"GW:UNKNOWN_MODEL models/checkpoints/tpl.safetensors\t5000000\n"
               b"GW:DONE\n")
        c, saved, _, _ = make(fake, ssh_script={HOST_BOOT: (0, out, b"")}, state={
            "phase": "off", "host_bootstrapped": False, "host_incomplete_snapshots": ["s9"],
            "manifests": {"s9": {"models/checkpoints/ours.safetensors": {"size": 9000000}}},
            "bootstrap_unknown": {"models/checkpoints/tpl.safetensors": 5000000}})
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        cmds = _ssh_cmds(fake)
        host, comfy = _boots(fake)
        self.assertEqual((len(host), comfy), (1, []))
        self.assertLess(cmds.index("cat > ~/.gw-nodes.txt"), cmds.index(host[0]))
        self.assertEqual(saved["thunder"]["bootstrap_unknown"],
                         {"models/checkpoints/tpl.safetensors": 5000000})

    async def test_host_rerun_node_upload_failure_is_not_fatal(self):
        # only the host bootstrap's report needs the list: it runs unfiltered instead
        fake = FakeThunder()
        _ready_snap(fake)
        c, _, _, _ = make(fake, ssh_script={"cat > ~/.gw-nodes.txt": (1, b"", b"disk full")},
                          state={"phase": "off", "host_bootstrapped": False,
                                 "host_incomplete_snapshots": ["s9"]})
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(len(_boots(fake)[0]), 1)
        self.assertTrue(any("template-node report lists every pack" in ln
                            for ln in c.state.log))


SECRET_START = "serve-model --api-key SECRET-START-9f3"
SECRET_SETUP = "pip install thing  # SECRET-SETUP-7c1\n"
COMFY = _svc("thunder", "comfyui", 18188, 8188)


def _cmdsvc(name="vllm", lport=18200, rport=8000, **kw):
    """A command service (services.CommandProfile) as a dict fixture — the store fields
    `svc_setup`/`svc_start`/`svc_health` arrive in Task 6."""
    return _svc(name, "openai", lport, rport, **dict(
        {"svc_start": SECRET_START, "svc_setup": SECRET_SETUP, "svc_health": "/health"}, **kw))


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _setups(fake, n=0, slug="vllm"):
    """(remote command, stdin) of every setup run of service <slug> since fake.calls[n]."""
    return [(p, stdin) for m, p, stdin in fake.calls[n:]
            if m == "SSH" and f"gw-svc-{slug}.setup.log" in p]


def _cmds(fake, n=0):
    return [p for m, p, _ in fake.calls[n:] if m == "SSH"]


class CommandServices(unittest.IsolatedAsyncioTestCase):
    """Command services (vLLM & co., services.CommandProfile) on a managed host, and the
    per-service lifecycle every service shares now: setup per hash, a failing service
    that is its own trouble, attach/detach while the host runs, the buttons, resume, and
    the Ruling M4 items (a list changed mid-stop, forward backoff, a service that cannot
    run, the no-ComfyUI stop)."""

    async def _stop_ready(self, c, fake):
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        fake.snaps[-1]["status"] = "READY"
        await c.watch_snapshots()
        fake.status_script = ["PROVISIONING", "RUNNING"]

    async def test_setup_runs_once_per_hash_and_per_disk(self):
        fake = FakeThunder()
        svc = _cmdsvc()
        c, saved, _, _ = make(fake, services=[svc])
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        su = _setups(fake)
        self.assertEqual(len(su), 1)
        self.assertEqual(su[0][1], SECRET_SETUP.encode())       # the script on stdin
        self.assertEqual(su[0][0], "bash -o pipefail -c "
                         + shlex.quote(services.COMMAND.setup_cmd(svc)))
        self.assertIn("bootstrapping", c.h.phases)
        cmds = _cmds(fake)
        self.assertLess(cmds.index(su[0][0]), cmds.index(services.COMMAND.start_cmd(svc)))
        h = _sha(SECRET_SETUP)
        self.assertEqual(saved["thunder"]["services"]["openai:vllm"]["setup_hash"], h)
        self.assertEqual(_sv(c, "openai:vllm")["status"], "up")
        # the snapshot records it: a start from that snapshot runs no setup
        await self._stop_ready(c, fake)
        self.assertEqual(saved["thunder"]["setup_snapshots"], {"s0": {"openai:vllm": h}})
        n = len(fake.calls)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_setups(fake, n), [])
        self.assertNotIn("bootstrapping", c.h.phases[-4:])
        # a changed script while the host runs: set up, then restarted, by the tick
        new = dict(svc, svc_setup="pip install other\n")
        c.set_services([new])
        n = len(fake.calls)
        await c._reconcile_services()
        su = _setups(fake, n)
        self.assertEqual([x[1] for x in su], [b"pip install other\n"])
        cmds = _cmds(fake, n)
        self.assertLess(cmds.index(su[0][0]), cmds.index(services.COMMAND.restart_cmd(new)))
        self.assertEqual(saved["thunder"]["services"]["openai:vllm"]["setup_hash"],
                         _sha("pip install other\n"))
        self.assertIsNone(c._reconcile_services())          # nothing changed since
        # a start from a TEMPLATE (no snapshot of ours): the stored hash is another
        # disk's — the setup runs again
        state = json.loads(json.dumps(saved["thunder"]))
        state.update(phase="off", uuid="", index="", ip="", port=0)
        fake2 = FakeThunder()
        c2, _, _, _ = make(fake2, services=[new], state=state)
        await c2.start()
        self.assertEqual(c2.state.phase, "ready", c2.state.error)
        self.assertEqual(len(_setups(fake2)), 1)

    async def test_failed_setup_is_that_service_alone(self):
        # spec: rc ≠ 0 → that service `setup failed`, the host stays, the others run
        fake = FakeThunder()
        vllm = _cmdsvc()
        embed = _cmdsvc("embed", 18201, 8001, svc_setup="pip install embed\n")
        c, saved, _, _ = make(fake, services=[COMFY, vllm, embed], ssh_script={
            "gw-svc-vllm.setup.log": (1, b"Collecting vllm\nERROR: No space left on device\n",
                                      b"")})
        await c.start()
        self.assertEqual((c.state.phase, c.state.error), ("ready", ""))
        v = _sv(c, "openai:vllm")
        self.assertEqual(v["status"], "setup failed")
        self.assertIn("rc 1", v["error"])
        self.assertIn("No space left on device", v["error"])
        self.assertIn("~/gw-svc-vllm.setup.log", v["error"])
        self.assertEqual((_sv(c)["status"], _sv(c, "openai:embed")["status"]), ("up", "up"))
        self.assertNotIn(services.COMMAND.start_cmd(vllm), _cmds(fake))
        self.assertEqual(saved["thunder"]["services"]["openai:vllm"]["setup_hash"], "")
        self.assertEqual([(f[0]["name"], f[0]["type"]) for f in c.h.faults],
                         [("vllm", "openai")])
        self.assertTrue([ln for ln in c.state.log
                         if ln.endswith(" openai:vllm setup: ERROR: No space left on device")])
        # not retried by every tick; the card's button re-runs it
        self.assertIsNone(c._reconcile_services())
        c.deps.ssh = make(fake)[0].deps.ssh                 # the setup succeeds now
        await c.resetup("openai:vllm")
        self.assertEqual(_sv(c, "openai:vllm")["status"], "up")
        self.assertEqual(saved["thunder"]["services"]["openai:vllm"]["setup_hash"],
                         _sha(SECRET_SETUP))
        self.assertEqual(c.state.phase, "ready")

    async def test_attach_while_running_leaves_the_other_service_alone(self):
        # Review Focus 2: a service attached while ComfyUI serves a stream gets its
        # forward on the running master, is enabled, set up, started and probed — no
        # tunnel restart or respawn, nothing sent to ComfyUI
        fake = FakeThunder()
        c, _, enabled, _ = make(fake)
        _master(c, fake)
        ctl = _controls(c)
        asked = []

        async def probe_http(url):
            asked.append(url)
            return 401                          # its own API key: listening = up (R-W9)
        c.deps.probe_http = probe_http
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        t = c.h.tunnels[0]
        n = len(fake.calls)
        vllm = _cmdsvc()
        c.set_services(c.services + [vllm])
        op = c._reconcile_services()
        self.assertEqual(c.op, "updating services")
        await op
        self.assertIsNone(c.op)
        self.assertEqual([x[2:] for x in ctl], [("forward", 18200, 8000)])
        self.assertEqual((len(c.h.tunnels), len(t.argvs)), (1, 1))
        self.assertTrue(t.running)
        self.assertIs(enabled["openai:vllm"], True)
        calls = [(p, stdin) for m, p, stdin in fake.calls[n:] if m == "SSH"]
        names = [p for p, _ in calls]
        self.assertFalse([p for p in names if "comfy" in p])  # ComfyUI never touched
        up = names.index(services.COMMAND.upload_cmd(vllm))
        self.assertEqual(calls[up][1], services.COMMAND.wrapper_script(vllm))
        self.assertLess(up, names.index(services.COMMAND.start_cmd(vllm)))
        # the port guard ran before anything was started
        kinds = _paths(fake, n)
        self.assertLess(kinds.index(("GET", "/instances/list")),
                        kinds.index(("SSH", services.COMMAND.start_cmd(vllm))))
        self.assertEqual(asked[-1], "http://127.0.0.1:18200/health")
        self.assertEqual((_sv(c, "openai:vllm")["status"], _sv(c)["status"]), ("up", "up"))
        self.assertEqual(c.state.phase, "ready")
        # detached: stopped on the VM (by its lock), its forward cancelled — and its
        # backend NEVER disabled (R-K2)
        n = len(fake.calls)
        c.set_services([x for x in c.services if x is not vllm])
        await c._reconcile_services()
        self.assertEqual(_cmds(fake, n), [services.COMMAND.stop_cmd(vllm)])
        self.assertTrue(await _until(lambda: ctl[-1][2] == "cancel"))
        self.assertEqual(ctl[-1][2:], ("cancel", 18200, 8000))
        self.assertIs(enabled["openai:vllm"], True)
        self.assertNotIn("openai:vllm", c.view()["services"])
        self.assertEqual((len(c.h.tunnels), len(t.argvs)), (1, 1))

    async def test_moving_a_service_to_another_host_never_disables_it(self):
        # Review Focus 1: H1 → H2 while H1 runs — H1 ends the process on its VM and
        # later, at `off`, disables only what is attached to it then; H2 enables it
        calls = []
        vllm = _cmdsvc()
        h1, _, _, _ = make(FakeThunder(), services=[COMFY, vllm])
        f2 = FakeThunder()
        h2, _, _, _ = make(f2, host_name="h2", services=[])
        for h in (h1, h2):
            h.deps.set_enabled = lambda bid, on, h=h: calls.append((h.name, bid, on)) or True
        await h1.start()
        self.assertEqual(h1.state.phase, "ready", h1.state.error)
        h1.set_services([COMFY])
        h2.set_services([vllm])
        await h1._reconcile_services()
        await h1.stop()
        self.assertEqual(h1.state.phase, "off", h1.state.error)
        await h2.start()
        self.assertEqual(h2.state.phase, "ready", h2.state.error)
        self.assertEqual([x for x in calls if x[1] == "openai:vllm"],
                         [("thunder", "openai:vllm", True), ("h2", "openai:vllm", True)])
        self.assertIn(("thunder", "comfyui:thunder", False), calls)

    async def test_services_changed_during_a_stop_wait_until_off(self):
        # Ruling M4 (a): mid-stop the list waits — the drain's services stand, nothing
        # is attached into the stopping host, and `off` disables what is attached THEN
        # (the one that moved away is not touched)
        fake = FakeThunder()
        vllm, embed = _cmdsvc(), _cmdsvc("embed", 18201, 8001)
        c, _, _, _ = make(fake, services=[COMFY, vllm])
        calls, drained, seen = [], [], []
        c.deps.set_enabled = lambda bid, on: calls.append((bid, on)) or True
        await c.start()
        c.deps.begin_drain = lambda bid: drained.append(bid) or True
        busy = [1, 1]
        c.deps.inflight = lambda bid: busy.pop(0) if bid == "openai:vllm" and busy else 0
        sleep = c.deps.sleep

        async def sleeping(sec):
            if c.state.phase == "draining" and not seen:
                c.set_services([COMFY, embed])              # embed in, vllm moved away
                seen.append((list(c.services), c._reconcile_services(),
                             c.view()["services"].keys()))
            await sleep(sec)
        c.deps.sleep = sleeping
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        during, op, shown = seen[0]
        self.assertEqual(during, [COMFY, vllm])             # the drain's list stood
        self.assertIsNone(op)                               # no attach into a stop
        self.assertNotIn("openai:embed", shown)
        self.assertEqual(drained, ["comfyui:thunder", "openai:vllm"])
        self.assertEqual(c.services, [COMFY, embed])        # applied at off
        self.assertNotIn(("openai:embed", True), calls)     # never enabled by the stop
        self.assertEqual(sorted(x for x in calls if x[1] is False),
                         [("comfyui:thunder", False), ("openai:embed", False)])
        # a stop that FAILED at a step is still a stop: a list keeps waiting
        c.state.phase, c.state.failed_phase = "failed", "pruning"
        c.set_services([COMFY])
        self.assertEqual(c.services, [COMFY, embed])
        self.assertEqual(c._pending_services, [COMFY])

    async def test_failed_forward_backs_off_and_recovers(self):
        # Ruling M4 (b): a forward that fails is retried with a doubling backoff, the
        # service shows why meanwhile and is not started behind a missing forward; once
        # the forward is added the service comes up by itself
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        _master(c, fake)
        answers = [(255, "Port forwarding failed.")] * 3
        ctl = _controls(c, lambda: answers.pop(0) if answers else (0, ""))
        await c.start()
        vllm = _cmdsvc()
        c.set_services(c.services + [vllm])
        self.assertTrue(await _until(lambda: ctl and c._fwd_task.done()))
        self.assertEqual(len(ctl), 1)
        await c._reconcile_services()                       # the tick's attach
        v = _sv(c, "openai:vllm")
        self.assertEqual(v["status"], "down")
        self.assertIn("Port forwarding failed", v["error"])
        self.assertNotIn(services.COMMAND.start_cmd(vllm), _cmds(fake))
        self.assertEqual(len(ctl), 1)                       # backing off (5 s)
        c.h.clock[0] += 5
        await c.reconcile_forwards()
        self.assertEqual(len(ctl), 2)                       # failed again: 10 s now
        c.h.clock[0] += 5
        await c.reconcile_forwards()
        self.assertEqual(len(ctl), 2)
        c.h.clock[0] += 5
        await c.reconcile_forwards()
        self.assertEqual(len(ctl), 3)                       # 20 s now
        c.h.clock[0] += 19
        await c.reconcile_forwards()
        self.assertEqual(len(ctl), 3)
        c.h.clock[0] += 1
        await c.reconcile_forwards()
        self.assertEqual(len(ctl), 4)                       # added
        self.assertEqual((_sv(c, "openai:vllm")["status"], _sv(c, "openai:vllm")["error"]),
                         ("starting", "forward restored"))
        await c._reconcile_services()
        self.assertEqual(_sv(c, "openai:vllm")["status"], "up")
        self.assertIn(services.COMMAND.start_cmd(vllm), _cmds(fake))
        self.assertEqual(c._fwd_active, {(18188, 8188), (18200, 8000)})

    async def test_services_that_cannot_run_are_down_with_the_cause(self):
        # Ruling M4 (c) + R-W8: shown down with the reason, never forwarded, enabled
        # or started — the first of two that clash keeps running
        fake = FakeThunder()
        svcs = [COMFY, _cmdsvc("vllm", 18188, 8000),                   # local port taken
                _cmdsvc("Qwen_1", 18201, 8001), _cmdsvc("qwen-1", 18202, 8002),  # slug
                _svc("gpu2", "comfyui", 18203, 8189),                   # second ComfyUI
                _cmdsvc("dup-remote", 18204, 8001),                     # remote port taken
                {"name": "m", "type": "meshy", "local_port": 18205, "remote_port": 8005},
                _cmdsvc("nostart", 18206, 8006, svc_start="  ")]
        c, _, enabled, _ = make(fake, services=svcs)
        _master(c, fake)
        _controls_log = _controls(c)
        v = c.view()["services"]
        want = {"openai:vllm": "local port 18188 is already forwarded for comfyui:thunder",
                "openai:qwen-1": "name slug 'qwen-1' collides with openai:Qwen_1",
                "comfyui:gpu2": "a second ComfyUI on one host is not supported "
                                "(comfyui:thunder runs there)",
                "openai:dup-remote": "remote port 8001 is already used by openai:Qwen_1",
                "meshy:m": "type 'meshy' cannot run on a managed host",
                "openai:nostart": "start command (svc_start) is required"}
        for bid, why in want.items():
            self.assertEqual(v[bid]["status"], "down", bid)
            self.assertIn(why, v[bid]["error"], bid)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(sorted(b for b, on in enabled.items() if on),
                         ["comfyui:thunder", "openai:Qwen_1"])
        self.assertEqual(sorted(_fwds(c.h.tunnels[0].argvs[0])),
                         ["127.0.0.1:18188:127.0.0.1:8188", "127.0.0.1:18201:127.0.0.1:8001"])
        self.assertEqual([x for x in _cmds(fake) if "gw-svc-" in x and "qwen-1" not in x], [])
        for bid, why in want.items():
            self.assertIn(why, _sv(c, bid)["error"], bid)
        # the clash resolved (the first one detached): the other one is attached now —
        # enabled, started, up
        c.set_services([x for x in svcs if x["name"] != "thunder"])
        await c._reconcile_services()
        self.assertEqual((_sv(c, "openai:vllm")["status"], _sv(c, "openai:vllm")["error"]),
                         ("up", ""))
        self.assertIs(enabled["openai:vllm"], True)
        self.assertIn("127.0.0.1:18188:127.0.0.1:8000", [
            ":".join(["127.0.0.1", str(x[3]), "127.0.0.1", str(x[4])]) for x in _controls_log])
        # nothing that can run: the start is refused before anything bills
        c2, _, _, _ = make(FakeThunder(), services=[svcs[6]])
        with self.assertRaisesRegex(RuntimeError, "can run: meshy:m: type 'meshy'"):
            await c2.start()

    async def test_no_comfy_host_stop_has_no_model_prune(self):
        # Ruling M4 (e): "no ComfyUI service" is a branch, not an exception text — no
        # plan, no index, no manifest; unfinished downloads still go
        fake = FakeThunder()
        c, _, _, _ = make(fake, services=[_cmdsvc()])
        c.cfg.pop("bootstrap_template")
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        with self.assertRaisesRegex(RuntimeError, "no ComfyUI service attached"):
            await c.delete_unknown(["models/checkpoints/x.safetensors"])
        n = len(fake.calls)
        with mock.patch.object(c, "_compute_plan", side_effect=AssertionError("planned")):
            await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        cmds = _cmds(fake, n)
        self.assertFalse([x for x in cmds if x.startswith((": gw-index", ": gw-manifest"))])
        self.assertEqual([x for x in cmds if x.startswith(": gw-prune")],
                         [hostctl._prune_cmd([])])
        log = "\n".join(c.state.log)
        self.assertIn("no ComfyUI service — no model prune", log)
        self.assertNotIn("model prune skipped", log)
        self.assertIn(("POST", "/snapshots/create"), _paths(fake, n))

    async def test_admin_text_travels_on_stdin_only(self):
        # security (binding): svc_setup/svc_start never in an ssh argv, a log line, an
        # error, a fault, the view or the persisted state — only in the stdin of the
        # wrapper upload and of the setup
        fake = FakeThunder()
        vllm = _cmdsvc()
        embed = _cmdsvc("embed", 18201, 8001, svc_setup="pip install SECRET-SETUP-2\n",
                        svc_start="serve SECRET-START-2")
        c, saved, _, calls = make(fake, services=[COMFY, vllm], ssh_script={
            "gw-svc-embed.setup.log": [(1, b"", b"boom"), (0, b"ok\n", b"")]})
        _master(c, fake)
        _controls(c)
        await c.start()
        c.set_services([COMFY, vllm, embed])
        await c._reconcile_services()                       # attach; its setup fails
        self.assertEqual(_sv(c, "openai:embed")["status"], "setup failed")
        await c.restart_service("openai:vllm")
        await c.resetup("openai:embed")
        c.set_services([COMFY, dict(vllm, svc_start=SECRET_START + " --v2")])
        await c._reconcile_services()                       # changed + detached
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        argvs = [" ".join(a) for a, _ in calls] + [" ".join(a) for a in c.h.tunnels[0].argvs]
        self.assertTrue(argvs)
        for text in argvs + c.state.log + [json.dumps(c.view()), repr(c.h.faults),
                                           json.dumps(saved)]:
            self.assertNotIn("SECRET", text)
        stdins = b"\0".join(stdin or b"" for _, stdin in calls)
        for secret in (SECRET_START, SECRET_SETUP, "SECRET-SETUP-2", "SECRET-START-2",
                       SECRET_START + " --v2"):
            self.assertIn(secret.encode(), stdins)

    async def test_service_buttons(self):
        fake = FakeThunder()
        vllm = _cmdsvc()
        c, _, _, _ = make(fake, services=[COMFY, vllm])
        await c.start()
        phases = list(c.h.phases)
        n = len(fake.calls)
        await c.restart_service("openai:vllm")
        cmds = _cmds(fake, n)
        self.assertEqual(cmds[-2:], [services.COMMAND.upload_cmd(vllm),
                                     services.COMMAND.restart_cmd(vllm)])
        self.assertEqual(_setups(fake, n), [])
        self.assertFalse([p for p in cmds if "comfy" in p])  # ComfyUI untouched
        self.assertEqual(c.h.phases, phases)                 # the host's phase too
        n = len(fake.calls)
        await c.resetup("openai:vllm")                       # the same hash: runs anyway
        self.assertEqual(len(_setups(fake, n)), 1)
        self.assertEqual(_cmds(fake, n)[-1], services.COMMAND.restart_cmd(vllm))
        # ComfyUI: its restart is restart_comfy; its re-setup the ComfyUI bootstrap
        n = len(fake.calls)
        await c.restart_service(BID)
        self.assertEqual(_cmds(fake, n)[-1], hostctl._restart_cmd(8188))
        n = len(fake.calls)
        await c.resetup(BID)
        self.assertEqual((len(_boots(fake, n)[0]), len(_boots(fake, n)[1])), (0, 1))
        self.assertEqual(_cmds(fake, n)[-1], hostctl._restart_cmd(8188))
        self.assertEqual((c.state.phase, _sv(c)["status"]), ("ready", "up"))
        # refusals
        with self.assertRaisesRegex(RuntimeError, "not attached"):
            await c.restart_service("openai:nope")
        c._op = "starting"
        with self.assertRaisesRegex(RuntimeError, "already starting"):
            await c.resetup("openai:vllm")
        c._op = None
        await c.stop()
        with self.assertRaisesRegex(RuntimeError, "no running instance"):
            await c.restart_service("openai:vllm")

    async def test_stop_aborts_an_attach(self):
        # never an attach going on into a stopping host: the stop aborts it (a setup
        # may take an hour) and takes over
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        await c.start()
        ssh, entered, gate = c.deps.ssh, [], asyncio.Event()

        async def slow(argv, stdin=None, timeout=60):
            if "gw-svc-vllm.setup.log" in argv[-1]:
                entered.append(1)
                await gate.wait()
            return await ssh(argv, stdin=stdin, timeout=timeout)
        c.deps.ssh = slow
        vllm = _cmdsvc()
        c.set_services(c.services + [vllm])
        op = c._reconcile_services()
        self.assertTrue(await _until(lambda: entered))
        await c.stop()
        self.assertTrue(op.cancelled())
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertNotIn(services.COMMAND.start_cmd(vllm), _cmds(fake))
        self.assertIsNone(c.op)

    async def test_resume_restarts_only_the_service_that_does_not_answer(self):
        fake = FakeThunder()
        _inst(fake)
        url = "http://127.0.0.1:18200/health"
        answer = {url: 0}

        async def probe_http(u):
            await asyncio.sleep(0)
            return answer.get(u, 200)
        vllm = _cmdsvc()
        st = {"openai:vllm": {"status": "up", "error": "", "setup_hash": _sha(SECRET_SETUP)}}
        c, _, _, _ = make(fake, state=_persisted(services=st), services=[COMFY, vllm],
                          probe_http=probe_http)
        n = len(fake.calls)
        task = asyncio.ensure_future(c.resume())
        self.assertTrue(await _until(
            lambda: services.COMMAND.restart_cmd(vllm) in _cmds(fake, n)))
        answer[url] = 403
        await task
        self.assertEqual(c.state.phase, "ready", c.state.error)
        cmds = _cmds(fake, n)
        self.assertFalse([p for p in cmds if "comfy" in p])  # ComfyUI answered
        self.assertEqual(_setups(fake, n), [])               # its hash matches
        self.assertEqual((_sv(c)["status"], _sv(c, "openai:vllm")["status"]), ("up", "up"))
        self.assertIsNone(c._reconcile_services())           # both recorded as running
        # a setup script changed while the gateway was down runs on resume
        fake = FakeThunder()
        _inst(fake)
        st["openai:vllm"]["setup_hash"] = _sha("old script\n")
        c, _, _, _ = make(fake, state=_persisted(services=st), services=[COMFY, vllm])
        await c.resume()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        su = _setups(fake)
        self.assertEqual(len(su), 1)
        self.assertLess(_cmds(fake).index(su[0][0]),
                        _cmds(fake).index(services.COMMAND.restart_cmd(vllm)))

    async def test_resume_of_interrupted_setups_is_per_service(self):
        # Ruling M4 (g): a gateway restart during the per-service setups — a setup that
        # may still run is that service's `setup failed` (naming its log), the host
        # comes up with the rest
        fake = FakeThunder()
        _inst(fake)
        vllm = _cmdsvc()
        c, _, _, _ = make(fake, state=_persisted(phase="bootstrapping", host_bootstrapped=True,
                                                 bootstrap_incomplete=True),
                          services=[COMFY, vllm, _cmdsvc("embed", 18201, 8001, svc_setup="")])
        await c.resume()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(_sv(c)["status"], "setup failed")
        self.assertIn("gw-bootstrap.log", _sv(c)["error"])
        self.assertEqual(_sv(c, "openai:vllm")["status"], "setup failed")
        self.assertIn("gw-svc-vllm.setup.log", _sv(c, "openai:vllm")["error"])
        self.assertEqual(_sv(c, "openai:embed")["status"], "up")
        self.assertTrue(c.state.bootstrap_incomplete)        # its snapshot stays marked
        self.assertEqual(_boots(fake), ([], []))            # nothing run twice
        self.assertEqual(_setups(fake), [])


    async def test_health_path_change_only_reprobes(self):
        # the health path is probe-only: a changed one must not restart a serving service
        fake = FakeThunder()
        vllm = _cmdsvc()
        asked = []

        async def probe(url):
            asked.append(url)
            return 200
        c, _, _, _ = make(fake, services=[COMFY, vllm], probe_http=probe)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        new = dict(vllm, svc_health="/v1/models")
        c.set_services([COMFY, new])
        n = len(fake.calls)
        await c._reconcile_services()
        self.assertEqual(_cmds(fake, n), [])                 # nothing stopped or started
        self.assertEqual(asked[-1], "http://127.0.0.1:18200/v1/models")
        self.assertEqual(_sv(c, "openai:vllm")["status"], "up")
        self.assertIsNone(c._reconcile_services())          # recorded: no second probe

    async def test_config_change_restart_waits_for_requests_in_flight(self):
        # an AUTOMATIC restart holds routing back, waits for the requests in flight
        # (shown as `restart pending`), restarts, and gives routing back
        fake = FakeThunder()
        vllm = _cmdsvc()
        c, _, _, _ = make(fake, services=[COMFY, vllm])
        await c.start()
        holds, seen = [], []
        c.deps.hold_routing = lambda bid, on: holds.append((bid, on)) or True
        busy = [2, 2, 1]
        c.deps.inflight = lambda bid: (busy.pop(0) if busy else 0) \
            if bid == "openai:vllm" else 0
        sleep = c.deps.sleep

        async def sleeping(sec):
            seen.append((_sv(c, "openai:vllm")["status"], _sv(c, "openai:vllm")["error"],
                         services.COMMAND.restart_cmd(new) in _cmds(fake, n), list(holds)))
            await sleep(sec)
        c.deps.sleep = sleeping
        new = dict(vllm, svc_start=SECRET_START + " --v2")
        c.set_services([COMFY, new])
        n = len(fake.calls)
        await c._reconcile_services()
        pending = [x for x in seen if x[0] == "restart pending"]
        self.assertEqual(len(pending), 3)                   # 2, 2, 1 in flight
        self.assertEqual(seen[:3], pending)
        for status, err, restarted, h in pending:
            self.assertEqual(status, "restart pending")
            self.assertFalse(restarted)                     # not while requests run
            self.assertEqual(h, [("openai:vllm", True)])
        self.assertEqual(seen[0][1][:22], "2 request(s) in flight")
        self.assertEqual(_cmds(fake, n)[-1], services.COMMAND.restart_cmd(new))
        self.assertEqual(holds, [("openai:vllm", True), ("openai:vllm", False)])
        self.assertEqual(_sv(c, "openai:vllm")["status"], "up")

    async def test_changed_setup_runs_before_routing_is_held(self):
        # the setup runs while the old process still serves; only the restart waits
        fake = FakeThunder()
        vllm = _cmdsvc()
        c, _, _, _ = make(fake, services=[COMFY, vllm])
        await c.start()
        order = []
        c.deps.hold_routing = lambda bid, on: order.append(("hold", on)) or True
        ssh = c.deps.ssh

        async def spy(argv, stdin=None, timeout=60):
            if "gw-svc-vllm.setup.log" in argv[-1]:
                order.append(("setup", None))
            return await ssh(argv, stdin=stdin, timeout=timeout)
        c.deps.ssh = spy
        new = dict(vllm, svc_setup="pip install other\n")
        c.set_services([COMFY, new])
        n = len(fake.calls)
        await c._reconcile_services()
        self.assertEqual(order, [("setup", None), ("hold", True), ("hold", False)])
        self.assertEqual(_cmds(fake, n)[-1], services.COMMAND.restart_cmd(new))
        self.assertIsNone(c._reconcile_services())          # recorded
        # a FAILING setup: no hold, no restart — the old process serves on
        order.clear()
        c.deps.ssh = ssh
        bad = dict(vllm, svc_setup="exit 3\n")
        fake2_n = len(fake.calls)
        orig = c.deps.ssh

        async def failing(argv, stdin=None, timeout=60):
            if "gw-svc-vllm.setup.log" in argv[-1]:
                fake.calls.append(("SSH", argv[-1], stdin))
                return (3, b"", b"boom")
            return await orig(argv, stdin=stdin, timeout=timeout)
        c.deps.ssh = failing
        c.set_services([COMFY, bad])
        await c._reconcile_services()
        self.assertEqual(order, [])
        self.assertNotIn(services.COMMAND.restart_cmd(bad), _cmds(fake, fake2_n))
        self.assertEqual(_sv(c, "openai:vllm")["status"], "setup failed")
        self.assertIsNone(c._reconcile_services())

    async def test_config_change_restart_wait_is_bounded(self):
        fake = FakeThunder()
        vllm = _cmdsvc()
        c, _, _, _ = make(fake, services=[COMFY, vllm])
        await c.start()
        c.deps.hold_routing = lambda bid, on: True
        c.deps.inflight = lambda bid: 1 if bid == "openai:vllm" else 0
        t0 = c.h.clock[0]
        new = dict(vllm, svc_start=SECRET_START + " --v2")
        c.set_services([COMFY, new])
        n = len(fake.calls)
        await c._reconcile_services()
        self.assertEqual(_cmds(fake, n)[-1], services.COMMAND.restart_cmd(new))
        self.assertGreaterEqual(c.h.clock[0] - t0, hostctl._RESTART_WAIT_MAX_S)
        self.assertLess(c.h.clock[0] - t0, hostctl._RESTART_WAIT_MAX_S + 60)

    async def test_stop_during_a_pending_restart_gives_routing_back(self):
        fake = FakeThunder()
        vllm = _cmdsvc()
        c, _, _, _ = make(fake, services=[COMFY, vllm])
        await c.start()
        holds, drained = [], []
        c.deps.hold_routing = lambda bid, on: holds.append((bid, on)) or True
        c.deps.begin_drain = lambda bid: drained.append((bid, list(holds))) or True
        c.deps.inflight = lambda bid: 1 if bid == "openai:vllm" and not drained else 0
        c.set_services([COMFY, dict(vllm, svc_start=SECRET_START + " --v2")])
        c._reconcile_services()
        self.assertTrue(await _until(lambda: _sv(c, "openai:vllm")["status"]
                                     == "restart pending"))
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        # released BEFORE the stop's own drain began
        self.assertEqual(drained[0][1], [("openai:vllm", True), ("openai:vllm", False)])

    async def test_restart_button_is_immediate(self):
        fake = FakeThunder()
        vllm = _cmdsvc()
        c, _, _, _ = make(fake, services=[COMFY, vllm])
        await c.start()
        holds = []
        c.deps.hold_routing = lambda bid, on: holds.append((bid, on)) or True
        c.deps.inflight = lambda bid: 3
        n = len(fake.calls)
        await c.restart_service("openai:vllm")
        self.assertEqual(_cmds(fake, n)[-1], services.COMMAND.restart_cmd(vllm))
        self.assertEqual(holds, [])

    async def test_listener_check_warns_on_a_non_loopback_bind(self):
        # Ruling M6: after a command service is up, `ss -ltnH` on the VM; a listener on
        # its port bound to anything but loopback → a visible warning, nothing killed
        fake = FakeThunder()
        vllm = _cmdsvc()
        ss = [(0, b"LISTEN 0 4096 0.0.0.0:8000 0.0.0.0:*\n"
                  b"LISTEN 0 4096 127.0.0.1:8188 0.0.0.0:*\n", b"")]
        c, saved, _, _ = make(fake, services=[COMFY, vllm],
                              ssh_script={services.LISTEN_CMD: ss})
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(c.view()["services"]["openai:vllm"]["warning"], "")
        n = len(fake.calls)
        await c._check_exposures()
        self.assertEqual(_cmds(fake, n), [services.LISTEN_CMD])      # ComfyUI not checked
        w = c.view()["services"]["openai:vllm"]["warning"]
        self.assertEqual(w, "listening on all interfaces (0.0.0.0:8000) — reachable from "
                            "outside the VM; bind to 127.0.0.1")
        self.assertEqual(c.view()["services"][BID]["warning"], "")
        self.assertEqual(saved["thunder"]["services"]["openai:vllm"]["warning"], w)
        self.assertEqual(_sv(c, "openai:vllm")["status"], "up")      # still up, not killed
        self.assertFalse([p for p in _cmds(fake, n) if "kill" in p])
        self.assertTrue([ln for ln in c.state.log if "reachable from outside" in ln])
        n = len(fake.calls)
        await c._check_exposures()                                   # once per up
        self.assertEqual(_cmds(fake, n), [])
        # restarted with a loopback bind: the warning goes with the restart, and the
        # check after it finds nothing
        ss[0] = (0, b"LISTEN 0 4096 127.0.0.1:8000 0.0.0.0:*\n", b"")
        await c.restart_service("openai:vllm")
        await c._check_exposures()
        self.assertEqual(c.view()["services"]["openai:vllm"]["warning"], "")
        self.assertNotIn("warning", saved["thunder"]["services"]["openai:vllm"])

    async def test_listener_check_runs_in_the_tick_and_survives_a_failure(self):
        fake = FakeThunder()
        vllm = _cmdsvc()
        c, _, _, _ = make(fake, services=[COMFY, vllm],
                          ssh_script={services.LISTEN_CMD: (127, b"", b"ss: not found\n")})
        await c.start()
        await _one_round(c)
        self.assertEqual(len([p for p in _cmds(fake) if p == services.LISTEN_CMD]), 1)
        self.assertTrue([ln for ln in c.state.log if "listener check failed (rc 127)" in ln])
        self.assertEqual(c.view()["services"]["openai:vllm"]["warning"], "")
        self.assertEqual(_sv(c, "openai:vllm")["status"], "up")


class ProviderNeutralTexts(unittest.TestCase):
    """A host's option errors name the HOST OPTION (the form field), never a
    `thunder.<key>` block that no longer exists — and a provider is named by its NAME."""

    def _c(self, **opts):
        c, _, _, _ = make(FakeThunder())
        c.host = dict(c.host, options=dict(c.cfg, **opts))
        return c

    def test_option_errors_name_the_host_option(self):
        with self.assertRaisesRegex(hostctl._PreCreate,
                                    r"^host option vcpus is not a number: 'lots'$"):
            self._c(vcpus="lots")._cfg_int("vcpus", 1)
        c = self._c(nodes=[])
        c.deps.default_nodes = lambda: ""
        with self.assertRaisesRegex(hostctl._PreCreate, r"\(host option nodes is empty"):
            c._node_list()
        with self.assertRaisesRegex(RuntimeError, r"^host option comfy_commit must be"):
            self._c(comfy_commit="main")._commit()
        for c in (self._c(vcpus="lots"), self._c(nodes=[]), self._c(comfy_commit="x")):
            for fn in (lambda: c._cfg_int("vcpus", 1), c._commit):
                try:
                    fn()
                except Exception as e:
                    self.assertNotIn("thunder.", str(e))

    def test_no_token_names_the_form_field(self):
        # the token is the PROVIDER's (one per provider, entered once in Server → API
        # Keys) — the text must send the operator there
        self.assertEqual(hostctl._NO_TOKEN.format(name="X"),
                         "no X API token set — enter it under Server → API Keys")


class StartBlockers(unittest.IsolatedAsyncioTestCase):
    """`start_blockers()` is what the card shows BEFORE Start is pressed (the button is
    disabled with the first one as its title), `start()` raises its first item. Two
    copies of the refusal rules would drift: a card offering a Start the controller
    refuses (two presses refused with "no backend attached" that the card never
    announced — operator test 2026-09-30), or one hiding a Start that would run."""

    def _c(self, **kw):
        fake = FakeThunder()
        c, _, enabled, _ = make(fake, **kw)
        return c, fake, enabled

    async def _refusal(self, c) -> str:
        with self.assertRaises(RuntimeError) as cm:
            await c.start()
        return str(cm.exception)

    async def test_each_refusal_is_the_first_blocker(self):
        cases = {}
        c, _, _ = self._c()
        c.host = dict(c.host, api_key="")
        cases["token"] = c
        c, _, _ = self._c(services=[])
        cases["no service"] = c
        c, _, _ = self._c(services=[_svc("m", "meshy", 18100, 9000)])
        cases["none can run"] = c
        c, _, _ = self._c()
        c.host = dict(c.host, options=dict(c.cfg, comfy_commit="main"))
        cases["commit"] = c
        c, _, _ = self._c(state=_persisted())
        cases["phase"] = c
        c, _, _ = self._c()
        c._op = "syncing models"
        cases["op"] = c
        c, _, _ = self._c()
        c._persist_blocked = True
        cases["persist"] = c
        want = {"token": "no Thunder Compute API token set",
                "no service": "no backend is attached to managed host thunder",
                "none can run": "no attached service of thunder can run: meshy:m",
                "commit": "host option comfy_commit must be a full 40-hex",
                "phase": "already ready (instance u0)",
                "op": "already syncing models",
                "persist": "state not loaded"}
        for k, c in cases.items():
            b = c.start_blockers()
            self.assertTrue(b, k)
            self.assertIn(want[k], b[0], k)
            self.assertEqual(await self._refusal(c), b[0], k)

    async def test_blockers_are_pure_and_ordered_like_start(self):
        # no token AND no service: both listed, start() raises the FIRST (its own order)
        c, fake, enabled = self._c(services=[])
        c.host = dict(c.host, api_key="")
        b = c.start_blockers()
        self.assertEqual(len(b), 2)
        self.assertIn("no backend is attached", b[0])
        self.assertIn("API token", b[1])
        self.assertEqual(await self._refusal(c), b[0])
        self.assertEqual((fake.calls, enabled, c.state.phase, c._op), ([], {}, "off", None))

    async def test_no_blocker_when_it_can_start(self):
        c, fake, _ = self._c()
        self.assertEqual(c.start_blockers(), [])
        self.assertEqual(fake.calls, [])                 # no provider call to judge it
        # a ComfyUI-less host never needs the commit
        c, _, _ = self._c(services=[_svc("vllm", "openai", 18101, 8000)])
        c.host = dict(c.host, options=dict(c.cfg, comfy_commit=""))
        self.assertEqual(c.start_blockers(), [])
        # unreconciled uuids are checked against a FRESH list inside the start op (they
        # may be gone by now) — not a blocker the card could judge
        c.state.unreconciled_uuids = ["u-x"]
        self.assertEqual(c.start_blockers(), [])

    def test_checklist(self):
        class Lan:
            ok = False

            def usable(self):
                return self.ok

            def problem(self):
                return "" if self.ok else "not set up"
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        lan = c.deps.lan = Lan()
        items = c.checklist()
        self.assertEqual([(i["ok"], i["required"]) for i in items],
                         [(True, True), (True, True), (False, False)])
        self.assertIn("Thunder Compute API token", items[0]["text"])
        self.assertEqual(items[0]["key"], "token")       # the card links it to API Keys
        self.assertIn("comfyui:thunder", items[1]["text"])
        self.assertIn("only needed for model files that no URL/catalog entry provides",
                      items[2]["text"])
        self.assertEqual(items[2]["key"], "lan")         # … and this one to Server → Models
        lan.ok = True
        self.assertTrue(c.checklist()[2]["ok"])
        # no token, no backend: both required items fail; no ComfyUI → no LAN item
        c.host = dict(c.host, api_key="")
        c.set_services([])
        items = c.checklist()
        self.assertEqual([(i["ok"], i["required"]) for i in items], [(False, True), (False, True)])
        self.assertEqual(items[0]["text"],
                         "Thunder Compute API token not set — enter it under Server → API Keys")
        self.assertIn("no backend attached — add one below", items[1]["text"])
        # attached but none can run: the causes are named
        c.set_services([_svc("m", "meshy", 18100, 9000)])
        it = c.checklist()[1]
        self.assertFalse(it["ok"])
        self.assertIn("meshy:m", it["text"])
        # a LAN source whose check throws is "not usable", never a broken card
        c.set_services([_svc("tc", "comfyui", 18100, 8188)])

        class Boom:
            def usable(self):
                raise RuntimeError("store")
        c.deps.lan = Boom()
        self.assertFalse(c.checklist()[2]["ok"])

class LegacyBootstrapState(unittest.TestCase):
    """A record written before the split has no host flag: the one-piece bootstrap did
    the host part too, so its verdict carries over — and what it marked incomplete may
    lack the host part."""

    def test_complete_record_counts_as_host_bootstrapped(self):
        s = hostctl.state_from({"phase": "ready", "uuid": "u1", "incomplete_snapshots": ["s1"]})
        self.assertTrue(s.host_bootstrapped)
        self.assertEqual(s.host_incomplete_snapshots, ["s1"])
        self.assertEqual((s.comfy_absent, s.no_comfy_snapshots), (False, []))

    def test_incomplete_record_is_not(self):
        s = hostctl.state_from({"phase": "bootstrapping", "uuid": "u1",
                                "bootstrap_incomplete": True})
        self.assertFalse(s.host_bootstrapped)

    def test_new_record_is_read_as_written(self):
        s = hostctl.state_from({"phase": "ready", "host_bootstrapped": False,
                                "host_incomplete_snapshots": [],
                                "incomplete_snapshots": ["s1"]})
        self.assertFalse(s.host_bootstrapped)
        self.assertEqual(s.host_incomplete_snapshots, [])


class DepsContract(unittest.TestCase):
    """The main↔hostctl seam: every field `hostctl.Deps` declares is provided by
    `main._host_deps()` with a callable that accepts the arguments hostctl calls
    it with. A renamed or re-shaped dependency fails only at run time otherwise — in
    the middle of a start, after the instance was paid for."""

    # how hostctl calls each callable field (positional arity); None = not a callable
    ARITY = {"client_factory": 0, "load_state": 1, "save_state": 2, "set_enabled": 2,
             "begin_drain": 1, "inflight": 1, "is_draining": 1, "note_fault": 4,
             "datadir": None, "probe_comfy": 1, "bootstrap_script": 0, "log": 1, "now": 0,
             "sleep": 1, "ssh": 1, "spawn": None, "known_uuids": 0, "keygen": 1,
             "default_nodes": 0, "alias_needs": 1, "alias_signature": 1, "source_index": 0,
             "url_catalog": 1, "hf_token": 0, "lan": None, "pipe": 3, "control": 5,
             "host_bootstrap_script": 0, "probe_http": 1, "cancel_drain": 1,
             "hold_routing": 2}
    KWARGS = {"ssh": ("stdin", "timeout"), "pipe": ("timeout_idle",), "control": ("timeout",)}

    def test_every_field_provided_with_the_called_arity(self):
        import dataclasses
        import inspect
        m = _main()
        d = m._host_deps()
        names = [f.name for f in dataclasses.fields(hostctl.Deps)]
        self.assertEqual(sorted(names), sorted(self.ARITY), "Deps grew or lost a field — "
                         "add it to ARITY with the arity hostctl calls it with")
        for name in names:
            v = getattr(d, name)
            arity = self.ARITY[name]
            if name == "datadir":
                self.assertIsInstance(v, str)
                self.assertTrue(v)
                continue
            if name == "lan":
                self.assertIsInstance(v, hostctl.LanSource)
                continue
            if name == "spawn":
                self.assertTrue(v is None or callable(v))
                continue
            self.assertTrue(callable(v), name)
            try:
                sig = inspect.signature(v)
            except ValueError:                  # a builtin (time.time): nothing to read
                continue
            try:
                sig.bind(*([None] * arity),
                                          **{k: None for k in self.KWARGS.get(name, ())})
            except TypeError as e:
                self.fail(f"Deps.{name}: {e}")


class StopServiceEvents(unittest.IsolatedAsyncioTestCase):
    """Addendum bug 1 (thunder-1, 2026-10-01 16:51): the stop's OWN disable came back
    through main's rebuild as a "services changed while stopping" event, was applied
    again at `off` ("now applied"), the service was disabled a second time, and one more
    "while stopping" line followed after the phase was already `off`. Main is modelled
    as it behaves: `begin_drain` with nothing in flight finalizes at once (disable), and
    every `set_enabled` rebuilds and hands the controller FRESH dicts carrying the new
    `enabled` flag."""

    def _wire(self, c, extra=None):
        logs, calls = [], []
        c.deps.log = logs.append
        live = {}
        roster = {"svcs": [dict(x) for x in c.services]}

        def rebuild():
            c.set_services([dict(x, enabled=live.get(hostctl.service_bid(x), True))
                            for x in roster["svcs"]])

        def set_enabled(bid, on):
            calls.append((bid, on))     # main logs "disabled via console" on EVERY call
            live[bid] = on
            rebuild()
            return True

        def begin_drain(bid):
            set_enabled(bid, False)     # idle: main's finalize disables at once
            return True
        c.deps.set_enabled = set_enabled
        c.deps.begin_drain = begin_drain
        return logs, calls, roster, rebuild

    async def test_the_stops_own_disable_is_no_service_change(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake, services=[dict(COMFY)])
        logs, calls, _, _ = self._wire(c)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        n = len(logs)
        calls.clear()
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        said = logs[n:]
        self.assertFalse([m for m in said if "services changed" in m], said)
        # disabled ONCE (by the drain's finalize) — `off` does not disable it again
        self.assertEqual(calls, [("comfyui:thunder", False)])
        self.assertEqual(_sv(c, "comfyui:thunder")["status"], "down")
        self.assertIsNone(c._pending_services)
        self.assertIs(c.services[0].get("enabled"), False)     # the current dicts

    async def test_off_disables_a_service_the_drain_left_enabled(self):
        # nothing disabled it on the way (a drain that did not finalize): `off` does
        fake = FakeThunder()
        c, _, _, _ = make(fake, services=[dict(COMFY)])
        logs, calls, _, _ = self._wire(c)
        await c.start()
        c.deps.begin_drain = lambda bid: True
        calls.clear()
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(calls, [("comfyui:thunder", False)])

    async def test_a_real_change_mid_stop_still_waits_and_is_said_once(self):
        # Ruling M4 (a) unchanged: an attach during the drain waits for `off`, is logged
        # once when handed over and once when applied — and nothing after `off` says
        # "while stopping"
        fake = FakeThunder()
        vllm = _cmdsvc()
        c, _, _, _ = make(fake, services=[dict(COMFY)])
        logs, calls, roster, rebuild = self._wire(c)
        await c.start()
        n = len(logs)
        sleep, seen = c.deps.sleep, []

        async def sleeping(sec):
            if c.state.phase == "draining" and not seen:
                roster["svcs"] = [dict(COMFY), dict(vllm)]
                rebuild()
                seen.append([hostctl.service_bid(x) for x in c.services])
            await sleep(sec)
        c.deps.sleep = sleeping
        c.deps.inflight = lambda bid, busy=[1]: busy.pop() if busy else 0
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(seen, [["comfyui:thunder"]])           # the drain's list stood
        said = logs[n:]
        self.assertEqual(len([m for m in said if "while stopping" in m]), 1, said)
        self.assertEqual(len([m for m in said if "now applied" in m]), 1, said)
        off_at = max(i for i, m in enumerate(said) if "services changed" in m)
        self.assertIn("now applied", said[off_at])              # the last word on it
        self.assertEqual([hostctl.service_bid(x) for x in c.services],
                         ["comfyui:thunder", "openai:vllm"])
        self.assertNotIn(("openai:vllm", True), calls)           # never enabled by a stop
        self.assertEqual(calls.count(("comfyui:thunder", False)), 1)

    async def test_a_list_handed_over_after_off_applies_at_once(self):
        # the rebuild that `off`'s own disable triggers arrives while the op is still
        # "stopping" but the phase is `off`: nothing is waiting for anything any more
        fake = FakeThunder()
        vllm = _cmdsvc()
        c, _, _, _ = make(fake, services=[dict(COMFY)])
        logs, calls, roster, rebuild = self._wire(c)
        await c.start()
        c.deps.begin_drain = lambda bid: True                  # leave it to `off`
        n = len(logs)

        def set_enabled(bid, on):
            calls.append((bid, on))
            roster["svcs"] = [dict(COMFY), dict(vllm)]        # attached meanwhile
            rebuild()
            return True
        c.deps.set_enabled = set_enabled
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertFalse([m for m in logs[n:] if "services changed" in m], logs[n:])
        self.assertEqual([hostctl.service_bid(x) for x in c.services],
                         ["comfyui:thunder", "openai:vllm"])
        self.assertIsNone(c._pending_services)

    async def test_off_disables_unless_definitely_disabled(self):
        # fix round 1 (coordinator): the skip is for `enabled is False` ONLY — an absent,
        # None or odd value is disabled again; a backend left enabled after `off` is
        # polled and routed to a dead tunnel, a redundant disable costs one log line
        for val in ("absent", None, 0, "false", True, False):
            with self.subTest(enabled=val):
                fake = FakeThunder()
                c, _, _, _ = make(fake, services=[dict(COMFY)])
                await c.start()
                self.assertEqual(c.state.phase, "ready", c.state.error)
                svc = dict(COMFY) if val == "absent" else dict(COMFY, enabled=val)
                c.services = [svc]                      # main's last rebuild said so
                calls = []
                c.deps.set_enabled = lambda bid, on: calls.append((bid, on)) or True
                c.deps.begin_drain = lambda bid: True   # no finalize: off decides
                await c.stop()
                self.assertEqual(c.state.phase, "off", c.state.error)
                want = [] if val is False else [("comfyui:thunder", False)]
                self.assertEqual(calls, want)
                self.assertEqual(_sv(c, "comfyui:thunder")["status"], "down")

    async def test_resume_mid_stop_skips_what_the_drain_already_disabled(self):
        # a gateway restart during the stop: the new controller is BUILT with main's live
        # dicts (the drain finalize already disabled the service) and finishes the stop —
        # `off` must not disable it a second time, and still shows it down
        fake = FakeThunder()
        fake.status_script = []
        fake.snaps.append({"id": "s5", "name": "aihub-thunder-20260927t100000z",
                           "status": "CREATING", "minimumDiskSizeGb": 120, "createdAt": 50})
        c, _, _, _ = make(fake, services=[dict(COMFY, enabled=False)],
                          state=_persisted(phase="deleting", pending_snapshot="s5"))
        logs, calls = [], []
        c.deps.log = logs.append
        c.deps.set_enabled = lambda bid, on: calls.append((bid, on)) or True
        await c.resume()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(calls, [])
        self.assertEqual(_sv(c, "comfyui:thunder")["status"], "down")
        self.assertEqual(len([m for m in logs if "comfyui:thunder: down" in m]), 1, logs)
        self.assertFalse([m for m in logs if "services changed" in m], logs)

    async def test_a_reordered_list_is_no_change(self):
        # a `priority` edit re-sorts main's list without changing any service
        fake = FakeThunder()
        vllm = _cmdsvc()
        c, _, _, _ = make(fake, services=[dict(COMFY), dict(vllm)])
        logs, calls, roster, rebuild = self._wire(c)
        await c.start()
        n = len(logs)
        sleep, done = c.deps.sleep, []

        async def sleeping(sec):
            if c.state.phase == "draining" and not done:
                roster["svcs"] = [dict(vllm), dict(COMFY)]
                rebuild()
                done.append(1)
            await sleep(sec)
        c.deps.sleep = sleeping
        c.deps.inflight = lambda bid, busy=[1]: busy.pop() if busy else 0
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        self.assertEqual(done, [1])
        self.assertFalse([m for m in logs[n:] if "services changed" in m], logs[n:])
        self.assertIsNone(c._pending_services)

    async def test_a_change_reverted_before_off_closes_its_log_line(self):
        # attach X, then detach X, during one stop: "while stopping" is answered at `off`
        fake = FakeThunder()
        vllm = _cmdsvc()
        c, _, _, _ = make(fake, services=[dict(COMFY)])
        logs, calls, roster, rebuild = self._wire(c)
        await c.start()
        n = len(logs)
        sleep, step = c.deps.sleep, []

        async def sleeping(sec):
            if c.state.phase == "draining" and len(step) < 2:
                roster["svcs"] = ([dict(COMFY), dict(vllm)] if not step else [dict(COMFY)])
                rebuild()
                step.append(1)
            await sleep(sec)
        c.deps.sleep = sleeping
        c.deps.inflight = lambda bid, busy=[1, 1]: busy.pop() if busy else 0
        await c.stop()
        self.assertEqual(c.state.phase, "off", c.state.error)
        said = logs[n:]
        self.assertEqual(len([m for m in said if "while stopping" in m]), 1, said)
        self.assertEqual(len([m for m in said if "unchanged after all" in m]), 1, said)
        self.assertFalse([m for m in said if "now applied" in m], said)
        self.assertEqual([hostctl.service_bid(x) for x in c.services], ["comfyui:thunder"])
