"""Thunder lifecycle controller against a stubbed Thunder API and a fake ssh.
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

import httpx

import modelsync as ms
import sshrun
import thunder
import hostctl
from tests.fakes import FakeThunder  # the scripted Thunder REST API


COMMIT = "1d61dcc35c35541388c0001bacc7703db14e8bea"
_TMPDIRS = []


def tearDownModule():
    for d in _TMPDIRS:
        shutil.rmtree(d, ignore_errors=True)


def make(fake, backend=None, ssh_script=None, datadir=None, default_nodes="",
         probe=None, state=None):
    """A controller on the stub API. `c.h` carries the harness: `clock` (a list; the
    fake sleep ADVANCES it, so every timeout is reachable without waiting), `phases`
    (every persisted phase in order), `faults`, `tunnels`. `ssh_script` maps a
    substring of the remote command to a result, or to a LIST of results consumed in
    order (the last one sticks); the bootstrap succeeds by default. Every ssh call is also appended to `fake.calls` as
    ("SSH", remote_cmd, stdin), so the order against API calls is checkable. The
    datadir is a fresh temp dir per controller unless given. `state` is a persisted
    record the controller loads at construction (a gateway restart)."""
    saved = {} if state is None else {"thunder": state}
    enabled = {}
    ssh_calls = []
    clock = [1_790_000_000.0]
    phases, faults, tunnels = [], [], []
    if datadir is None:
        datadir = tempfile.mkdtemp(prefix="hostctl-test-")
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

    deps = hostctl.Deps(
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
    c = hostctl.Controller(b, deps)

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
        c2 = hostctl.Controller(c.backend, c.deps)
        self.assertEqual((c2.state.phase, c2.state.uuid, c2.state.index), ("ready", "u7", "7"))
        self.assertEqual(c2.state.log, [])
        self.assertEqual(c2.state.transfers, {})

    async def test_load_tolerates_unknown_keys_and_bad_types(self):
        # a state written by another version must not keep the controller from coming up
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = {"phase": "ready", "uuid": "u1", "port": "30022", "someday": 1,
                            "manifests": None, "log": ["stale"]}
        c2 = hostctl.Controller(c.backend, c.deps)
        self.assertEqual(c2.state.phase, "ready")
        self.assertEqual(c2.state.port, 30022)
        self.assertEqual(c2.state.manifests, {})
        self.assertEqual(c2.state.log, [])

    async def test_unknown_persisted_phase_becomes_failed_not_off(self):
        # "off" would forget an instance that may still be billing
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = {"phase": "warping", "uuid": "u1", "index": "1"}
        c2 = hostctl.Controller(c.backend, c.deps)
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
        c2 = hostctl.Controller(c.backend, c.deps)
        self._check_blocked(c2, saved)
        self.assertEqual(saved["thunder"]["uuid"], "u9")
        with self.assertRaises(RuntimeError):
            await c2.start()
        self.assertEqual(fake.calls, [])         # no create was even attempted

    async def test_load_non_dict_blocks_persist_and_start(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = ["garbage"]
        c2 = hostctl.Controller(c.backend, c.deps)
        self._check_blocked(c2, saved)
        self.assertEqual(saved["thunder"], ["garbage"])
        with self.assertRaises(RuntimeError):
            await c2.start()

    async def test_unblock_persist_resumes_saving(self):
        fake = FakeThunder()
        c, saved, _, _ = make(fake)
        saved["thunder"] = "garbage"
        c2 = hostctl.Controller(c.backend, c.deps)
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
        self.assertEqual(boot, ["bash -o pipefail -c "
                                f"'bash -s -- {COMMIT} 2>&1 | tee ~/gw-bootstrap.log'"])
        self.assertIn(hostctl._START_CMD, cmds)
        self.assertTrue(hostctl._START_CMD.endswith(
            "setsid nohup ~/start-comfy.sh >/dev/null 2>&1 < /dev/null &"))
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
        c2 = hostctl.Controller(c.backend, c.deps)
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
        return hostctl.Controller(c.backend, c.deps), saved

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
        self.assertTrue(cmds[0].endswith("; " + hostctl._START_CMD))
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
        pat = shlex.split(hostctl._RESTART_CMD.split(";")[0])[2]
        self.assertTrue(re.search(pat, comfy), comfy)
        self.assertFalse(re.search(pat, "bash -c " + hostctl._RESTART_CMD))
        self.assertFalse(re.search(pat, "bash -c " + shlex.quote(hostctl._RESTART_CMD)))
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
        c2 = hostctl.Controller(c.backend, c.deps)
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
        self.assertEqual(seen[:2], [(2, 0), (2, 0)])
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
        c2 = hostctl.Controller(c.backend, c.deps)
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
        self.assertEqual((c.state.phase, c.state.failed_phase), ("failed", "bootstrapping"))
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
        c2 = hostctl.Controller(c.backend, c.deps)
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
        self.assertIn(hostctl._START_CMD, _ssh_cmds(fake))
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
        c = hostctl.Controller(c0.backend, c0.deps)
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
        c.backend = dict(c.backend, api_key="")
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

    def needs(name):
        return [ms.alias_need(a, ms.refs_for(cand, cand["workflow_json"], []), box["catalog"])
                for a, cand in sorted(box["aliases"].items()) if cand.get("backend") == name]
    c.deps.alias_needs = needs
    c.deps.alias_signature = lambda name: json.dumps(box, sort_keys=True)
    c.deps.url_catalog = lambda: ms.url_catalog(box["catalog"])
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
        self.assertFalse(c.is_alias_ready("img"))               # no plan yet
        self.assertIsNone(c.plan)
        await c.start()
        self.assertEqual(c.state.phase, "ready", c.state.error)
        self.assertEqual(c.h.phases[-3:], ["starting", "syncing", "ready"])
        self.assertIsNotNone(c.plan)
        self.assertFalse(c.is_alias_ready("img"))
        # a finished, b still downloading: a is present, the alias is not ready
        self.assertTrue(await _until(lambda: a in vm.files and c.plan is not None and any(
            f["path"] == a and f["present"] for f in c.plan["per_alias"]["img"]["files"])))
        self.assertFalse(c.is_alias_ready("img"))
        self.assertNotIn("img", c.ready_aliases)
        self.assertIn("syncing on thunder", c.alias_status("img"))
        self.assertIn(b, [t["file"] for t in c.view()["transfers"]])
        vm.hold.clear()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
        self.assertEqual(c.alias_status("img"), "models for img are ready on thunder")
        man = json.loads(vm.manifest)
        self.assertEqual(man[a]["size"], 100)
        self.assertEqual(man[b]["size"], 300)
        self.assertEqual((man[a]["source"], man[a]["aliases"]), ("url", ["img"]))
        self.assertEqual(vm.files[a], 100)
        self.assertNotIn(a + ".part", vm.files)
        self.assertEqual(c.view()["transfers"], [])
        # readiness is also a phase question
        c.state.phase = "draining"
        self.assertFalse(c.is_alias_ready("img"))
        self.assertEqual(c.alias_status("img"), "thunder instance is draining")
        c.state.phase = "syncing"
        self.assertTrue(c.is_alias_ready("img"))
        self.assertFalse(c.is_alias_ready("other"))
        self.assertIn("not planned", c.alias_status("other"))

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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
        # the alias now loads a file no source has → blocked; its synced file is held
        box["aliases"]["img"] = _cand("nowhere.safetensors")
        await c._sync_tick()
        self.assertFalse(c.is_alias_ready("img"))
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
        c.deps.url_catalog = lambda: ms.url_catalog(cat)
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
            c.h.faults, default=str) + c.alias_status("img")
        self.assertNotIn(_TOKEN, blob)
        self.assertIn("401", c.alias_status("img"))

    async def test_failed_transfer_blocks_after_three_attempts_with_fault(self):
        url = "https://example.com/x.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.fail[url] = "curl: (22) The requested URL returned error: 404\n"
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.plan["per_alias"]["img"]["blocked"]))
        self.assertEqual(len(vm.started), 3)
        self.assertFalse(c.is_alias_ready("img"))
        st = c.alias_status("img")
        self.assertIn("blocked on thunder", st)
        self.assertIn("transfer failed", st)
        self.assertIn("404", st)
        sync_faults = [f for f in c.h.faults if f[1:3] == ("sync", "transfer")]
        self.assertEqual(len(sync_faults), 1)
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))

    async def test_sha256_mismatch_discards_the_part(self):
        url = "https://example.com/x.safetensors"
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors", sha="ab" * 32)])
        vm.sizes[url] = 8
        vm.sha[url] = "cd" * 32
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.plan["per_alias"]["img"]["blocked"]))
        self.assertIn("sha256", c.alias_status("img"))
        self.assertNotIn(_dm("x.safetensors"), vm.files)
        self.assertNotIn(_dm("x.safetensors") + ".part", vm.files)
        self.assertEqual(len(_gw(fake, "gw-discard")), 3)
        # the right bytes pass and their sha is recorded
        vm.sha[url] = "AB" * 32
        box["catalog"] = [_url("x.safetensors", sha="AB" * 32)]
        await c._sync_tick()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))

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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("small")))
        self.assertFalse(c.is_alias_ready("img"))
        self.assertIn("disk", c.alias_status("img"))
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
        self.assertTrue(c.is_alias_ready("img"))              # present, not re-fetched
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("one")
                                     and c.plan["per_alias"]["two"]["blocked"]))
        self.assertIn("size 5 ≠ 99", c.alias_status("two"))
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
            self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
        with self.assertRaises(ValueError):             # a needed file is no "unknown"
            await c.delete_unknown([stranger, _dm("x.safetensors")])
        self.assertIn(stranger, vm.files)
        self.assertEqual(await c.delete_unknown([stranger]), 1)
        self.assertNotIn(stranger, vm.files)
        self.assertEqual(c.plan["unknown"], [])
        self.assertIn(_dm("x.safetensors"), vm.files)

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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
            st = c.alias_status(a)
            self.assertIn("waiting for LAN source (not configured)", st)
            self.assertFalse(c.is_alias_ready(a))
            self.assertTrue([b for b in c.view()["plan"]["aliases"][a]["blocked"]
                             if b.startswith("waiting for LAN source (not configured)")])
        self.assertNotIn("not in source", c.alias_status("img"))
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
        self.assertTrue(c.is_alias_ready("two"))
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
        self.assertTrue(c.is_alias_ready("two"))

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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
        self.assertFalse(c.is_alias_ready("img"))
        self.assertTrue(c.view()["sync_error"])
        broken[0] = False
        c.h.clock[0] += hostctl._SYNC_REFRESH_S
        await c._sync_tick()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
        self.assertEqual(c.view()["sync_error"], "")

    async def test_manifest_write_is_atomic_and_manifest_aliases_are_lists(self):
        fake, vm, c, box, _ = _sync_make(aliases={"img": _cand("x.safetensors")},
                                         catalog=[_url("x.safetensors")])
        vm.sizes["https://example.com/x.safetensors"] = 3
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
    test scripts (a refusal is a RuntimeError before the first await)."""
    def __init__(self, name, phase="off", uuid="", refuse=None):
        self.name = name
        self.backend = {"name": name, "type": "comfyui", "thunder": {"gpu_type": "a6000"}}
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

    def restart_comfy(self):
        return self._call("restart")

    def resume(self):
        return self._call("resume")

    def run_forever(self):
        return self._call("run_forever")

    def forget_unreconciled(self):
        self.calls.append("forget")

    async def aclose(self):
        self.calls.append("aclose")

    def view(self):
        return {"phase": self.state.phase, "uptime_s": 42, "cost_per_h": 0.57,
                "log": ["x"]}


class MainWiring(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        m = self.m = _main()
        import store
        self.store = store
        self._saved = {n: getattr(m, n) for n in (
            "backends", "thunder_controllers", "_thunder_tasks", "_thunder_booted",
            "backend_inflight", "_draining", "jobs_cfg")}
        self._saved_store = (store._DB_PATH, store._active)
        self.tmp = tempfile.mkdtemp(prefix="thunder-wiring-")
        _TMPDIRS.append(self.tmp)
        store.init(os.path.join(self.tmp, "store.db"))
        m.thunder_controllers = {}
        m._thunder_tasks = {}
        m._thunder_booted = False
        m.jobs_cfg = dict(m.jobs_cfg, store_path=os.path.join(self.tmp, "store.db"))

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(self.m, n, v)
        self.store._DB_PATH, self.store._active = self._saved_store

    @staticmethod
    def _tb(name="tc", **thunder_kw):
        return {"name": name, "type": "comfyui", "url": "http://127.0.0.1:18188",
                "api_key": "tok", "thunder": {"gpu_type": "a6000", **thunder_kw}}

    def test_controllers_follow_backend_list(self):
        m = self.m
        plain = {"name": "k12", "type": "comfyui", "url": "http://10.0.0.1:8188"}
        m.backends = [plain, self._tb()]
        m.sync_thunder_controllers()
        self.assertEqual(list(m.thunder_controllers), ["tc"])
        c = m.thunder_controllers["tc"]
        self.assertIsInstance(c, hostctl.Controller)
        # a rebuild hands NEW dicts: the instance stays, its backend is the current one
        fresh = self._tb(gpu_type="h100")
        m.backends = [dict(plain), fresh]
        m.sync_thunder_controllers()
        self.assertIs(m.thunder_controllers["tc"], c)
        self.assertIs(c.backend, fresh)
        self.assertEqual(c.cfg["gpu_type"], "h100")
        # block removed while an instance runs → kept, warned
        c.state.phase = "ready"
        m.backends = [plain, {"name": "tc", "type": "comfyui", "url": "http://x"}]
        with self.assertLogs("main", "WARNING") as cm:
            m.sync_thunder_controllers()
        self.assertIs(m.thunder_controllers.get("tc"), c)
        self.assertIn("backend config removed while instance runs", "\n".join(cm.output))
        # … and once it is off, the next sync removes it
        c.state.phase = "off"
        m.sync_thunder_controllers()
        self.assertEqual(m.thunder_controllers, {})
        # a thunder block on a non-ComfyUI backend is no Thunder backend
        m.backends = [{"name": "llm", "type": "openai", "url": "http://x", "thunder": {"a": 1}}]
        m.sync_thunder_controllers()
        self.assertEqual(m.thunder_controllers, {})

    def test_rebuild_backends_syncs_controllers(self):
        m = self.m
        self.store.upsert_backend(self._tb())
        saved_cfg = m.config_backends
        m.config_backends = []
        try:
            m.rebuild_backends()
            c = m.thunder_controllers["tc"]
            m.rebuild_backends()
            self.assertIs(m.thunder_controllers["tc"], c)
            self.assertIs(c.backend, next(b for b in m.backends if b["name"] == "tc"))
        finally:
            m.config_backends = saved_cfg

    def test_set_backend_enabled_keeps_config_backend_whole(self):
        # a config-defined Thunder backend: the lifecycle enables it on every start and
        # disables it on every stop, and each toggle writes a store copy that overrides
        # config WHOLESALE — every configured key must survive, the thunder block included
        m = self.m
        saved_cfg = m.config_backends
        extra = {"host": "thunder-box", "comfy_output_dir": "/home/ubuntu/ComfyUI/output",
                 "max_wait": 900, "auto_restart": True,
                 "models_allow": "flux*", "bypass": ["12"]}
        m.config_backends = [dict(self._tb(), **extra)]
        try:
            m.rebuild_backends()
            c = m.thunder_controllers["tc"]
            for on in (True, False, True, False):          # two start/stop cycles
                self.assertTrue(m.set_backend_enabled("comfyui:tc", on))
                stored = self.store.get_backend("tc", "comfyui") or {}
                live = next(b for b in m.backends if b["name"] == "tc")
                for k, v in extra.items():
                    self.assertEqual(stored.get(k), v, k)
                    self.assertEqual(live.get(k), v, k)
                self.assertEqual(stored.get("thunder"), {"gpu_type": "a6000"})
                self.assertEqual(stored.get("api_key"), "tok")   # decrypted on read
                self.assertIs(stored.get("enabled"), on)
                self.assertIs(m.thunder_controllers["tc"], c)
                self.assertIs(c.backend, live)
            self.assertEqual(c.cfg, {"gpu_type": "a6000"})
            self.assertEqual(m.backend_host(c.backend), "thunder-box")
        finally:
            m.config_backends = saved_cfg

    def test_config_backend_entry_is_the_whole_dict_minus_enabled(self):
        m = self.m
        b = {"name": "n", "type": "comfyui", "enabled": False, "_tmp": 1,
             "thunder": {"nodes": ["a"]}, "paid": False, "stuck_after_s": 120}
        e = m._config_backend_entry(b)
        self.assertEqual(e, {"name": "n", "type": "comfyui", "thunder": {"nodes": ["a"]},
                             "paid": False, "stuck_after_s": 120})
        e["thunder"]["nodes"].append("b")                   # a copy, not the live block
        self.assertEqual(b["thunder"]["nodes"], ["a"])

    def test_removed_block_warns_once_per_controller(self):
        m = self.m
        m.backends = [self._tb()]
        m.sync_thunder_controllers()
        c = m.thunder_controllers["tc"]
        c.state.phase = "ready"
        gone = [{"name": "tc", "type": "comfyui", "url": "http://x"}]
        m.backends = gone
        with self.assertLogs("main", "WARNING") as cm:
            m.sync_thunder_controllers()
            m.sync_thunder_controllers()
            m.sync_thunder_controllers()
        self.assertEqual(len([x for x in cm.output if "config removed" in x]), 1)
        # the block returns and goes again → warned again
        m.backends = [self._tb()]
        m.sync_thunder_controllers()
        m.backends = gone
        with self.assertLogs("main", "WARNING") as cm:
            m.sync_thunder_controllers()
        self.assertEqual(len([x for x in cm.output if "config removed" in x]), 1)

    def test_off_controller_with_op_in_flight_is_not_retired(self):
        # a start is `off` until the create — retiring it then would orphan the instance
        m = self.m
        fc = _FakeCtl("tc")
        fc.op = "starting"
        m.thunder_controllers = {"tc": fc}
        m.backends = []
        with self.assertLogs("main", "WARNING"):
            m.sync_thunder_controllers()
        self.assertIs(m.thunder_controllers.get("tc"), fc)
        fc.op = None
        m.sync_thunder_controllers()
        self.assertEqual(m.thunder_controllers, {})

    def test_off_controller_with_a_pending_snapshot_is_not_retired(self):
        # the stop is done, but the watcher still has to rotate the old snapshot out —
        # retired now, both snapshots would bill per GB-month with nobody watching
        m = self.m
        fc = _FakeCtl("tc")
        fc.state.pending_snapshot = "s7"
        m.thunder_controllers = {"tc": fc}
        m.backends = []
        with self.assertLogs("main", "WARNING") as cm:
            m.sync_thunder_controllers()
        self.assertIs(m.thunder_controllers.get("tc"), fc)
        self.assertIn("s7", "\n".join(cm.output))
        fc.state.pending_snapshot = ""
        m.sync_thunder_controllers()
        self.assertEqual(m.thunder_controllers, {})

    def test_config_backend_entry_is_json_safe(self):
        # an unquoted YAML date is a datetime.date: the store's json.dumps raised on
        # every enable/disable — a Thunder start and stop included
        import datetime
        m = self.m
        b = {"name": "n", "type": "comfyui", "note": datetime.date(2026, 9, 28),
             "thunder": {"since": datetime.datetime(2026, 9, 28, 12, 0)}}
        e = m._config_backend_entry(b)
        self.assertEqual(e["note"], "2026-09-28")
        self.assertEqual(e["thunder"]["since"], "2026-09-28 12:00:00")
        json.dumps(e)

    async def test_real_controller_refusal_comes_back_as_text(self):
        # pins the contract thunder_action relies on: Controller.stop() refuses BEFORE
        # its first await, so the refusal is the answer — not a background log line
        m = self.m
        m.backends = [self._tb()]
        m.sync_thunder_controllers()
        c = m.thunder_controllers["tc"]
        self.assertIsInstance(c, hostctl.Controller)
        self.assertEqual(c.state.phase, "off")
        with self.assertNoLogs("main", "WARNING"):          # answered, not logged twice
            self.assertEqual(await m.thunder_action("tc", "stop"),
                             "stop refused: not running")
            self.assertEqual(await m.thunder_action("tc", "restart"),
                             "ComfyUI restart refused: no running instance to restart "
                             "ComfyUI on (off)")
            await asyncio.sleep(0)                          # let the done-callbacks run
        self.assertIsNone(c.op)

    def test_deps_wiring(self):
        m = self.m
        m.backends = [self._tb(), self._tb("tb")]
        m.sync_thunder_controllers()
        deps = m.thunder_controllers["tc"].deps
        self.assertEqual(deps.datadir, self.tmp)
        self.assertIs(deps.set_enabled, m.set_backend_enabled)
        self.assertIs(deps.begin_drain, m.begin_drain)
        self.assertIs(deps.note_fault, m._note_fault)
        m.backend_inflight = {"comfyui:tc": 2}
        m._draining = {"comfyui:tc"}
        self.assertEqual(deps.inflight("comfyui:tc"), 2)
        self.assertEqual(deps.inflight("comfyui:tb"), 0)
        self.assertTrue(deps.is_draining("comfyui:tc"))
        self.assertFalse(deps.is_draining("comfyui:tb"))
        # persistence: one settings key, one entry per backend name, others untouched
        deps.save_state("tc", {"phase": "ready", "uuid": "u1"})
        deps.save_state("tb", {"phase": "off"})
        self.assertEqual(self.store.get_setting("thunder_state"),
                         {"tc": {"phase": "ready", "uuid": "u1"}, "tb": {"phase": "off"}})
        self.assertEqual(deps.load_state("tc"), {"phase": "ready", "uuid": "u1"})
        self.assertIsNone(deps.load_state("nope"))
        # every controller's uuid is known (orphans = instances nobody owns)
        m.thunder_controllers["tc"].state.uuid = "u1"
        m.thunder_controllers["tb"].state.uuid = "u2"
        self.assertEqual(deps.known_uuids(), {"u1", "u2"})
        # the repo files
        self.assertTrue(deps.bootstrap_script().startswith(b"#!"))
        self.assertIsInstance(deps.default_nodes(), str)
        self.assertTrue(deps.default_nodes().strip())
        cl = deps.client_factory()
        self.assertIsNot(cl, m.http_client)
        self.assertIsNot(cl, deps.client_factory())

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
            m.backends = [self._tb()]
            m.sync_thunder_controllers()
            deps = m.thunder_controllers["tc"].deps
            self.assertIs(deps.alias_needs, m.thunder_alias_needs)
            needs = {n.alias: n for n in deps.alias_needs("tc")}
            self.assertEqual(sorted(needs), ["cfgalias", "img"])     # no cloud, no k12
            self.assertEqual(sorted((r.cls, r.value, r.selectable) for r in needs["img"].refs),
                             [("UNETLoader", "flux.safetensors", True),
                              ("VAELoader", "pinned.safetensors", False)])
            self.assertEqual([r.value for r in needs["cfgalias"].refs], ["t5.safetensors"])
            self.assertEqual(deps.alias_needs("k12")[0].refs[0].value, "k.st")
            # the catalog: defaults while unset, the setting once set, [] when garbage
            self.assertEqual(deps.url_catalog(), {})
            cat = [{"file": "models/vae/ae.safetensors", "url": "https://hf.co/x/ae.safetensors"},
                   {"match": {"alias": "img"}, "paths": ["models/loras/"]}]
            sig0 = deps.alias_signature("tc")
            self.assertEqual(sig0, deps.alias_signature("tc"))          # stable
            self.store.set_settings({"modelsync_catalog": cat})
            self.assertEqual(deps.url_catalog(), {"models/vae/ae.safetensors":
                                                  {"url": "https://hf.co/x/ae.safetensors"}})
            self.assertEqual(needs["img"].catalog, [])
            self.assertEqual({n.alias: n for n in deps.alias_needs("tc")}["img"].catalog,
                             ["models/loras/"])
            sig1 = deps.alias_signature("tc")
            self.assertNotEqual(sig1, sig0)                              # catalog counts
            # another backend's candidate changes nothing, this backend's does
            self.store.upsert("other", [{"backend": "k12", "workflow_json": wf}])
            self.assertEqual(deps.alias_signature("tc"), sig1)
            self.store.upsert("other", [{"backend": "tc", "workflow_json": wf}])
            sig2 = deps.alias_signature("tc")
            self.assertNotEqual(sig2, sig1)
            self.store.delete("other")
            self.assertEqual(deps.alias_signature("tc"), sig1)
            # a path workflow's content counts (same path, new file)
            with open(path, "w") as f:
                json.dump({"1": {"class_type": "CLIPLoader",
                                 "inputs": {"clip_name": "t5-v2.safetensors"}}}, f)
            os.utime(path, (1, 1))
            self.assertNotEqual(deps.alias_signature("tc"), sig1)
            self.store.set_settings({"modelsync_catalog": {"not": "a list"}})
            self.assertEqual(deps.url_catalog(), {})
            self.assertEqual(deps.source_index(), {})
            self.assertEqual(deps.hf_token(), "")
            self.store.set_settings({"hf_token": "hf_x"})
            self.assertEqual(deps.hf_token(), "hf_x")
        finally:
            m.image_models = saved_img

    def test_unreadable_setting_is_never_overwritten(self):
        m = self.m
        self.store.set_settings({"thunder_state": ["garbage"]})
        m.backends = [self._tb()]
        m.sync_thunder_controllers()
        c = m.thunder_controllers["tc"]
        self.assertTrue(c.persist_blocked)          # load failed → no start, no save
        with self.assertRaises(ValueError):
            c.deps.save_state("tc", {"phase": "off"})
        self.assertEqual(self.store.get_setting("thunder_state"), ["garbage"])

    async def test_probe_comfy(self):
        m = self.m
        seen = []

        def handler(req):
            seen.append((str(req.url), req.extensions.get("timeout")))
            return httpx.Response(200 if "good" in str(req.url) else 502)
        saved = m.http_client
        m.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            deps = m._thunder_deps()
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
            self.assertFalse(await m._thunder_deps().probe_comfy("http://x:1"))
        finally:
            await m.http_client.aclose()
            m.http_client = saved

    async def test_actions_run_in_background_and_refusals_are_text(self):
        m = self.m
        c = _FakeCtl("tc", refuse={"stop": "not running"})
        m.thunder_controllers = {"tc": c}
        msg = await m.thunder_action("tc", "start")
        self.assertIn("start", msg)
        self.assertEqual(c.calls, ["start"])
        # held (not GC-able) and still running: the call did not wait for the op
        held = [t for t in m._bg_refs if not t.done()]
        self.assertTrue(held)
        msg = await m.thunder_action("tc", "stop")
        self.assertIn("not running", msg)
        await m.thunder_action("tc", "restart")
        self.assertEqual(c.calls[-1], "restart")
        c.state.unreconciled_uuids = ["u9"]
        await m.thunder_action("tc", "forget_unreconciled")
        self.assertEqual(c.calls[-1], "forget")
        self.assertIn("unknown", await m.thunder_action("nope", "start"))
        self.assertIn("unknown", await m.thunder_action("tc", "explode"))
        c.gate.set()
        await asyncio.sleep(0)

    async def test_thunder_view_and_names(self):
        m = self.m
        c = _FakeCtl("tc", phase="ready")
        m.thunder_controllers = {"tc": c}
        self.assertEqual(m.thunder_view("tc")["phase"], "ready")
        self.assertIsNone(m.thunder_view("nope"))
        import admin
        self.assertEqual(admin._thunder_names(), ["tc"])
        self.assertIs(admin._thunder_view, m.thunder_view)
        self.assertIs(admin._thunder_action, m.thunder_action)

    async def test_boot_resumes_and_runs_each_controller_and_shutdown_closes(self):
        m = self.m
        a, b = _FakeCtl("a", phase="ready"), _FakeCtl("b")
        m.thunder_controllers = {"a": a, "b": b}
        m._thunder_boot()
        await asyncio.sleep(0)
        self.assertEqual(sorted(a.calls), ["resume", "run_forever"])
        self.assertEqual(sorted(b.calls), ["resume", "run_forever"])
        # a controller that appears later (backend added in the console) is started too
        m.backends = [self._tb("late")]
        m.sync_thunder_controllers()
        late = m.thunder_controllers["late"]
        self.assertEqual(len(m._thunder_tasks["late"]), 2)
        for t in m._thunder_tasks["late"]:
            t.cancel()
        await m._thunder_shutdown()
        self.assertIn("aclose", a.calls)
        self.assertIn("aclose", b.calls)
        # the background loops are gone, the instance was never touched
        for ts in m._thunder_tasks.values():
            self.assertTrue(all(t.done() for t in ts))
        self.assertNotIn("stop", a.calls)
        self.assertEqual(late.state.phase, "off")

    async def test_health_carries_thunder_block_in_full_view(self):
        m = self.m
        tb = self._tb()
        m.backends = [tb, {"name": "k12", "type": "comfyui", "url": "http://10.0.0.1:8188"}]
        m.thunder_controllers = {"tc": _FakeCtl("tc", phase="ready")}
        h = await m.health(verbose=False)
        self.assertEqual(h["backends"]["comfyui:tc"]["thunder"],
                         {"phase": "ready", "uptime_s": 42, "cost_per_h": 0.57})
        self.assertNotIn("thunder", h["backends"]["comfyui:k12"])

    def test_deploy_and_gitignore_exclude_keys(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        names = ["thunder.key", "thunder.key.pub", "thunder-known_hosts",
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")),
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

    async def test_sha_mismatch_restarts_and_counts(self):
        sh = self.share(a=b"abcdef")
        sh.sha_answers = ["f" * 64]
        fake, vm, c, lan, pipe = _lan_make({"img": _cand("a.safetensors")}, sh)
        await c.start()
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")),
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
                                     in c.alias_status("img")), c.state.log[-8:])
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")),
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
        self.assertIn("waiting for LAN source (not configured)", c.alias_status("img"))
        self.assertFalse(c.is_alias_ready("img"))

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
        self.assertIn("waiting for LAN source (not configured", c.alias_status("img"))

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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")),
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")),
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
        st = c.alias_status("img")
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")))
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
        self.assertIn("waiting for LAN source (not configured)", c.alias_status("img"))
        fp = await lan.scan()
        lan.pin(fp)
        await c._sync_tick()                              # the pin alone triggers a plan
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("img")),
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
        self.assertTrue(await _until(lambda: _idle(c) and c.is_alias_ready("hf")),
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
        self.assertTrue(c.is_alias_ready("hf"))
        # the alias goes: the stop prunes the link with the files it belongs to
        c.h.box["aliases"].clear()
        await c._before_snapshot()
        self.assertEqual(vm.links, {})
        self.assertNotIn(repo + "blobs/abc", vm.files)
        self.assertEqual(json.loads(vm.manifest), {})

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
        self.assertIn("waiting for LAN source (not configured)", c.alias_status("hf"))


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
        self.assertEqual(a[:3], ["ssh", "-i", os.path.join(self.d, "modelsrc.key")])
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
        self.assertTrue([ln for ln in c.state.log if "no longer listed" in ln])
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
        c.backend = dict(c.backend, api_key="")
        with self.assertRaises(RuntimeError) as cm:
            await c.start()
        self.assertIn("no Thunder API token set", str(cm.exception))
        self.assertIn("API key field", str(cm.exception))
        self.assertEqual(fake.calls, [])
        self.assertEqual(enabled, {})
        self.assertEqual(c.state.phase, "off")
        self.assertIsNone(c._op)


class DepsContract(unittest.TestCase):
    """The main↔hostctl seam: every field `hostctl.Deps` declares is provided by
    `main._thunder_deps()` with a callable that accepts the arguments hostctl calls
    it with. A renamed or re-shaped dependency fails only at run time otherwise — in
    the middle of a start, after the instance was paid for."""

    # how hostctl calls each callable field (positional arity); None = not a callable
    ARITY = {"client_factory": 0, "load_state": 1, "save_state": 2, "set_enabled": 2,
             "begin_drain": 1, "inflight": 1, "is_draining": 1, "note_fault": 4,
             "datadir": None, "probe_comfy": 1, "bootstrap_script": 0, "log": 1, "now": 0,
             "sleep": 1, "ssh": 1, "spawn": None, "known_uuids": 0, "keygen": 1,
             "default_nodes": 0, "alias_needs": 1, "alias_signature": 1, "source_index": 0,
             "url_catalog": 0, "hf_token": 0, "lan": None, "pipe": 3}
    KWARGS = {"ssh": ("stdin", "timeout"), "pipe": ("timeout_idle",)}

    def test_every_field_provided_with_the_called_arity(self):
        import dataclasses
        import inspect
        m = _main()
        d = m._thunder_deps()
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
