"""Thunder Compute (https://www.thundercompute.com) as an on-demand ComfyUI host — the
PURE half.

Everything here is a function of dicts and numbers: the body `POST /instances/create`
takes, what `GET /instances/list`, `GET /snapshots/list`, `GET /v2/specs` and
`GET /v2/pricing` hand back, how big a disk a start needs, what an hour of it costs,
and which of this backend's snapshots may be deleted. No `main`/`adapters` imports, no
I/O, no config cached at module level — the controller does the HTTP and the SSH and
asks this module WHAT to send and what came back, so the decisions that fail silently
(a rotation that deletes the last usable snapshot, a disk the snapshot does not fit
on, a cost shown per hour that is off by a factor) are testable without an account.

The API facts come from the design spec 2026-09-27 ("Fakten über Thunder", read from
the docs and the public openapi.json that day). Thunder has NO stop: an instance is a
container, and "stopping" is snapshot → delete → later create from that snapshot
(`template` = the snapshot's name). IP, SSH port and host key change with every
instance. Instance status values are undocumented, so everything unknown reads as
"not finished yet" and the controller bounds the wait. Covered by test_thunder.py.
"""
from __future__ import annotations

import math
import re
import time
from typing import Optional

API = "https://api.thundercompute.com:8443"

_GB = 1024 ** 3
_GONE = {"DELETED", "TERMINATED", "DELETING"}
_HOURS_PER_MONTH = 730          # Thunder bills snapshots per GB·hour; 730 h ≈ one month
_DISK_PER_GPU_GB = 100          # included per GPU and the smallest disk Thunder creates
# What follows `aihub-<slug>-` in a name this module wrote. Fully lowercase so the whole
# name stays within [a-z0-9-] (the one character rule Thunder could enforce).
_STAMP_FMT = "%Y%m%dt%H%M%Sz"
_STAMP_RE = re.compile(r"\d{8}t\d{6}z")


class ThunderError(RuntimeError):
    """A Thunder API call failed; `status` is the HTTP status, None for transport."""

    def __init__(self, msg, status: Optional[int] = None):
        super().__init__(msg)
        self.status = status


class DiskTooSmall(ValueError):
    """What the aliases need does not fit on the largest disk this GPU config allows."""


def _int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        # createdAt/storage may arrive as "1.7e9" or 120.0 — a float string is still a number
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None


def _slug(s) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "backend"


# ---- create / instances -------------------------------------------------------------

def create_body(cfg: dict, template: str, disk_gb: int, public_key: str) -> dict:
    """The `POST /instances/create` body. The API wants integers; the backend form
    stores whatever was typed, so an int() here is what keeps "8" from being a 422."""
    return {"cpu_cores": int(cfg["vcpus"]), "disk_size_gb": int(disk_gb),
            "gpu_type": str(cfg["gpu_type"]), "num_gpus": int(cfg.get("num_gpus") or 1),
            "template": str(template), "public_key": str(public_key)}


def _index_of(it: dict, key) -> str:
    v = it.get("id")
    return str(key if v is None or v == "" else v)


def parse_instances(obj) -> list[dict]:
    """`/instances/list` → normalized rows. The docs show a map `{index: item}`, other
    places a list whose items carry `id`; both are accepted, and every field may be
    missing (`cpuCores`/`numGpus` are even strings) — a KeyError here would take the
    whole lifecycle down over one field Thunder renamed."""
    if isinstance(obj, dict):
        rows = list(obj.items())
    else:
        rows = [(i, x) for i, x in enumerate(obj or [])]
    out = []
    for key, it in rows:
        if not isinstance(it, dict):
            continue
        out.append({
            "index": _index_of(it, key),
            "uuid": str(it.get("uuid") or ""),
            "status": str(it.get("status") or "").upper(),
            "ip": str(it.get("ip") or ""),
            "port": _int(it.get("port")),
            "http_ports": [p for p in (_int(x) for x in (it.get("httpPorts") or [])) if p],
            "storage": _int(it.get("storage")),
            "template": str(it.get("template") or ""),
            "created_at": str(it.get("createdAt") or ""),
        })
    return out


def find_instance(items: list[dict], uuid: str, index: Optional[str]) -> Optional[dict]:
    """Our instance among the listed ones: by UUID first (stable), else by index. The
    index is only a fallback — Thunder reuses small integers, so an index alone could
    name somebody else's newer instance once ours is gone."""
    if uuid:
        for it in items:
            if it.get("uuid") == uuid:
                return it
    if index is not None and str(index) != "":
        for it in items:
            if it.get("index") == str(index):
                return it
    return None


def is_running(status) -> bool:
    return str(status or "").upper() == "RUNNING"


def is_gone_status(status) -> bool:
    return str(status or "").upper() in _GONE


def ports_open(item: dict) -> list[int]:
    """HTTP ports Thunder forwards publicly (`https://<uuid>-<port>.thundercompute.net`,
    no auth). The controller checks ComfyUI's port is NOT among them."""
    return list(item.get("http_ports") or [])


# ---- snapshots ----------------------------------------------------------------------

def parse_snapshots(obj) -> list[dict]:
    """`/snapshots/list` → `{id, name, status (UPPER), min_disk_gb, created_at}`.
    `created_at` is the epoch integer rotation sorts by (0 when absent)."""
    rows = obj.values() if isinstance(obj, dict) else (obj or [])
    out = []
    for it in rows:
        if not isinstance(it, dict):
            continue
        out.append({"id": str(it.get("id") or ""), "name": str(it.get("name") or ""),
                    "status": str(it.get("status") or "").upper(),
                    "min_disk_gb": _int(it.get("minimumDiskSizeGb")) or 0,
                    "created_at": _int(it.get("createdAt")) or 0})
    return out


def snapshot_prefix(backend_name: str) -> str:
    return f"aihub-{_slug(backend_name)}-"


def snapshot_name(backend_name: str, now: float) -> str:
    return snapshot_prefix(backend_name) + time.strftime(_STAMP_FMT, time.gmtime(now))


def _owned(snap: dict, backend_name: str) -> bool:
    """Is this snapshot one THIS backend wrote? The prefix alone is not enough: backend
    `thunder`'s prefix `aihub-thunder-` is also the start of every snapshot of a backend
    `thunder-a6000`, and a hand-made `aihub-thunder-manual` carries it too — rotation
    would delete both. So what follows the prefix must be exactly the timestamp
    `snapshot_name` writes."""
    name, prefix = snap.get("name") or "", snapshot_prefix(backend_name)
    return name.startswith(prefix) and _STAMP_RE.fullmatch(name[len(prefix):]) is not None


def newest_ready(snaps: list[dict], backend_name: str) -> Optional[dict]:
    """The snapshot a start restores from: this backend's newest READY one by
    `created_at` (not by name — a clock-skewed name must not win)."""
    ready = [s for s in snaps if _owned(s, backend_name) and s.get("status") == "READY"]
    return max(ready, key=lambda s: s.get("created_at") or 0) if ready else None


def rotation(snaps: list[dict], backend_name: str) -> list[str]:
    """Ids of this backend's snapshots that may be deleted: every READY one OLDER than
    the newest READY, and every FAILED one. The newest READY is never returned (it is
    the only way back to the instance), CREATING never (it may become that newest one),
    and a foreign or hand-made snapshot never. A row without an id is never returned
    either: `DELETE /snapshots/` with an empty id is not a request anyone should send."""
    mine = [s for s in snaps if _owned(s, backend_name)]
    keep = newest_ready(mine, backend_name)
    out = [s["id"] for s in mine if s.get("status") == "FAILED"]
    if keep is not None:
        out += [s["id"] for s in mine if s.get("status") == "READY" and s is not keep
                and (s.get("created_at") or 0) < (keep.get("created_at") or 0)]
    return [i for i in out if i]


# ---- disk / specs / cost ------------------------------------------------------------

def choose_disk_gb(required_bytes: int, base_bytes: int, reserve_gb: int,
                   snapshot_min_gb: int, spec_min: int, spec_max: int,
                   num_gpus: int = 1) -> int:
    """Disk for a new instance, in GB, rounded up to 10: the models the aliases need +
    what the base install takes + a reserve, never below the snapshot's
    `minimumDiskSizeGb` (Thunder refuses the restore), the spec minimum or 100 GB per
    GPU. Disks only grow on Thunder, so this is also the size every later snapshot
    carries. Above the spec's maximum → DiskTooSmall naming both numbers; the
    rounding alone never causes that (a need that fits is clamped to the maximum)."""
    need = math.ceil((int(required_bytes or 0) + int(base_bytes or 0)) / _GB) + int(reserve_gb or 0)
    gb = max(need, int(snapshot_min_gb or 0), int(spec_min or 0),
             _DISK_PER_GPU_GB * max(1, int(num_gpus or 1)))
    if spec_max and gb > spec_max:
        raise DiskTooSmall(f"needs {gb} GB, max {spec_max} GB")
    gb = math.ceil(gb / 10) * 10
    if spec_max and gb > spec_max:
        gb = int(spec_max)
    return gb


def _config_key(gpu_type: str, num_gpus: int) -> str:
    return f"{str(gpu_type).lower()}_x{int(num_gpus or 1)}"


def spec_for(specs_obj, gpu_type: str, num_gpus: int) -> Optional[dict]:
    """This GPU configuration's entry from `/v2/specs` (`{"specs": {"a6000_x1": …}}`;
    the bare map is accepted too), or None when Thunder does not offer it."""
    if not isinstance(specs_obj, dict):
        return None
    table = specs_obj.get("specs") if isinstance(specs_obj.get("specs"), dict) else specs_obj
    spec = table.get(_config_key(gpu_type, num_gpus))
    return spec if isinstance(spec, dict) else None


def hourly_cost(pricing: dict, gpu_type: str, num_gpus: int, vcpus: int, disk_gb: int,
                spec: Optional[dict]) -> Optional[float]:
    """$/h of a running instance from `/v2/pricing`: the GPU configuration's rate +
    every vCPU above the SMALLEST option of the spec (that is how Thunder bills them —
    pinned in the test, to verify against a real invoice) + disk per GB·h for what lies
    BEYOND the 100 GB per GPU Thunder includes. None when the configuration has no
    price: a made-up number is worse than none."""
    rate = (pricing or {}).get(_config_key(gpu_type, num_gpus))
    if rate is None:
        return None
    extra = 0
    opts = [o for o in (_int(x) for x in ((spec or {}).get("vcpuOptions") or [])) if o is not None]
    if opts:
        extra = max(0, int(vcpus or 0) - min(opts))
    billable_gb = max(0, int(disk_gb or 0) - _DISK_PER_GPU_GB * max(1, int(num_gpus or 1)))
    return (float(rate) + extra * float(pricing.get("additional_vcpus") or 0)
            + billable_gb * float(pricing.get("disk_gb") or 0))


def snapshot_monthly(pricing: dict, gb: float) -> Optional[float]:
    """$/month a snapshot of `gb` GB costs while the instance is off (None: no price)."""
    rate = (pricing or {}).get("snapshot_gb")
    if rate is None:
        return None
    return float(gb) * float(rate) * _HOURS_PER_MONTH
