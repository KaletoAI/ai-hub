"""Shared test doubles — NOT a test module (no TestCase here, and the name does not
match `test*.py`, so discovery never collects it). The scripted Thunder REST API lives
here so the API tests and the controller tests share ONE fake whatever those files are
called."""
import json

import httpx


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
        self.storage = {"min": 100, "max": 500}   # /v2/specs storageGB of a6000_x1
        self.on_modify = None        # callback(old_gb, new_gb) — the VM's disk grows

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
                                   "port": 30022, "httpPorts": list(self.http_ports_on_create),
                                   "disk_size_gb": json.loads(req.content).get("disk_size_gb", 0)}
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
        if p.endswith("/modify"):
            ident = p.split("/")[2]
            body = json.loads(req.content)
            if ident not in self.instances:           # uuid form tried first (Ruling 11)
                return httpx.Response(404, json={"error": "not_found"})
            it = self.instances[ident]
            old = it.get("disk_size_gb", 0)
            it["disk_size_gb"] = body.get("disk_size_gb")
            if self.on_modify is not None:
                self.on_modify(old, body.get("disk_size_gb"))
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
            return httpx.Response(200, json={"specs": {"a6000_x1": {"vcpuOptions": [6, 8], "storageGB": dict(self.storage)}}})
        return httpx.Response(404)
