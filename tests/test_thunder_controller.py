"""Thunder lifecycle controller against a stubbed Thunder API and a fake ssh.
run: venv/bin/python -m unittest tests.test_thunder_controller -v"""
import asyncio
import json
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


def make(fake, backend=None, ssh_script=None):
    saved = {}
    enabled = {}
    ssh_calls = []

    async def ssh(argv, stdin=None, timeout=60):
        ssh_calls.append((argv, stdin))
        for pat, res in (ssh_script or {}).items():
            if pat in argv[-1]:
                return res
        return (0, b"", b"")

    deps = thunderctl.Deps(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler), base_url="https://t"),
        load_state=lambda n: saved.get(n), save_state=lambda n, d: saved.__setitem__(n, json.loads(json.dumps(d))),
        set_enabled=lambda bid, on: enabled.__setitem__(bid, on) or True,
        begin_drain=lambda bid: True, inflight=lambda bid: 0, is_draining=lambda bid: False,
        note_fault=lambda *a, **k: None, datadir="/tmp", log=lambda m: None,
        now=lambda: 1_790_000_000.0, sleep=lambda s: asyncio.sleep(0),
        ssh=ssh, spawn=None, probe_comfy=lambda url: asyncio.sleep(0, True),
        bootstrap_script=lambda: b"#!/bin/bash\necho GW:DONE\n")
    b = backend or {"name": "thunder", "type": "comfyui", "api_key": "tok",
                    "thunder": {"gpu_type": "a6000", "num_gpus": 1, "vcpus": 8, "local_port": 18188,
                                "bootstrap_template": "comfy-ui", "reserve_gb": 20, "nodes": [], "comfy_commit": "abc"}}
    c = thunderctl.Controller(b, deps)
    c._tunnel_factory = lambda: _NoTunnel()       # never spawn ssh in tests
    return c, saved, enabled, ssh_calls


class _NoTunnel:
    running = True
    def start(self): pass
    async def stop(self): self.running = False


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
        self.assertIn("UserKnownHostsFile=/tmp/thunder-known_hosts/u0", argv)
        self.assertEqual(argv[argv.index("-i") + 1], "/tmp/thunder.key")

    async def test_tunnel_argv_refuses_without_instance(self):
        fake = FakeThunder()
        c, _, _, _ = make(fake)
        with self.assertRaises(RuntimeError):
            c._tunnel_argv()

    async def test_bid_and_url(self):
        c, _, _, _ = make(FakeThunder())
        self.assertEqual(c.bid, "comfyui:thunder")
        self.assertEqual(c.url, "http://127.0.0.1:18188")


if __name__ == "__main__":
    unittest.main()
