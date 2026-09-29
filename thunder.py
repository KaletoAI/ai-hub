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

# ---- provider interface ---------------------------------------------------------------
# The names every provider module exports (duck-typed like meshy.py/tripo.py; the API
# class lives in hostapi.py, the registry there too). A second provider (RunPod) is a
# module with these same names — the host controller and the console read nothing else.
KIND = "thunder"                      # the key in hostapi.PROVIDERS and a host entry
NAME = "Thunder Compute"              # what the console shows
SSH_USER = "ubuntu"                   # the login on every instance of every template
# Thunder has no stop: "stop" is snapshot → delete, "start" a create from the snapshot.
# ("native" = the provider stops the machine and keeps its disk — RunPod.)
STOP_MODE = "snapshot"
# The template a host starts from when no ComfyUI service is attached: `comfy-ui`
# carries a ComfyUI install nobody would use (and whose port the guard then watches).
DEFAULT_TEMPLATE_NO_COMFY = "base"
# The ComfyUI revision the bootstrap pins (the k12-gpu build). A FULL sha only: the
# bootstrap's fetch-by-sha fallback needs it and exits 2 on anything else — after the
# instance was already created and billed. (admin._THUNDER_COMMIT_DEFAULT, _THUNDER_GPUS
# and _THUNDER_TEMPLATES are pinned equal to these until the console reads them.)
COMFY_COMMIT_DEFAULT = "1d61dcc35c35541388c0001bacc7703db14e8bea"
GPU_TYPES = ("a6000", "l40", "a100xl", "h100")
TEMPLATES = ("comfy-ui", "base")

# The host form as data (read by `options_of`, rendered by the console). `min` bounds an
# int; every field has a default, which is also what an absent or blank field becomes.
OPTION_FIELDS: list = [
    {"key": "gpu_type", "label": "gpu", "type": "select", "choices": list(GPU_TYPES),
     "default": "a6000",
     "hint": "The GPU of the instance; its price per hour comes from Thunder's price list."},
    {"key": "num_gpus", "label": "gpus", "type": "int", "min": 1, "default": 1,
     "hint": "GPUs per instance; each one includes 100 GB of disk."},
    {"key": "vcpus", "label": "vcpus", "type": "int", "min": 1, "default": 8,
     "hint": "vCPUs; every one above the GPU configuration's smallest option is billed "
             "extra."},
    {"key": "bootstrap_template", "label": "template", "type": "select",
     "choices": list(TEMPLATES), "default": "comfy-ui",
     "hint": "Thunder's image for the FIRST start (later starts restore the snapshot). "
             "<code>base</code> when no ComfyUI runs on this host."},
    {"key": "reserve_gb", "label": "disk reserve GB", "type": "int", "min": 0, "default": 20,
     "hint": "Free space kept on top of the models and the install. Disks only grow."},
    {"key": "comfy_commit", "label": "ComfyUI commit", "type": "text",
     "default": COMFY_COMMIT_DEFAULT,
     "hint": "A full 40-hex sha (blank = the default pin)."},
    {"key": "nodes", "label": "custom nodes", "type": "textarea", "default": [],
     "hint": "One node pack per line: <code>&lt;git-url&gt;@&lt;commit&gt;</code> or "
             "<code>registry:&lt;id&gt;@&lt;version&gt;</code>; <code>#</code> comments and "
             "blank lines are ignored. Empty = the default list at bootstrap time."},
]
_COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")
_WHOLE_RE = re.compile(r"[0-9]+")


def _form_str(v) -> str:
    """A form value as the string a browser would have sent. A form built by code may
    carry an int or a float; those are VALIDATED as their string (a 1.5 is an error,
    not a silent default). Only None and a bool (an unchecked/checked box) read as
    blank; `str()` of anything else never raises for the types a form carries."""
    if isinstance(v, bool) or v is None:
        return ""
    try:
        return str(v).strip()
    except Exception:                   # a pathological __str__: still never a raise
        return "?"


def options_of(form) -> tuple[dict, list, dict]:
    """The `opt__<key>` values of the host form → `(options, errors, typed)`. Never
    raises. `options` holds a VALID value on every key, always — the parsed value, or
    the field's default where the field is blank, absent or in error — so whatever a
    caller stores or hands to `create_body` is never a typo (a stored "rtx9090" would
    become the gpu_type, a "0" vcpus a 422 after the start began). `errors` names each
    refused field; `typed` is what the form carried, as strings, for re-rendering the
    form as typed (only keys the form sent). Rules: blank/absent is the ONE "unset"; an
    int must be whole digits (`1.5`, `-1`, `1e3`, `+2` are errors — the console's
    `_int_field` rule) and at least the field's `min`; a select one of its choices;
    `comfy_commit` a full 40-hex sha (the bootstrap exits 2 on anything else, after the
    instance was paid for). `nodes` becomes a list of lines (trailing blank lines
    dropped, like a textarea's final newline); a list is taken as the lines."""
    form = form if isinstance(form, dict) else {}
    out, errors, typed = {}, [], {}
    for fld in OPTION_FIELDS:
        k, t, default = fld["key"], fld["type"], fld["default"]
        dflt = list(default) if isinstance(default, list) else default
        name = f"opt__{k}"
        raw = form.get(name)
        if t == "textarea":
            if isinstance(raw, (list, tuple)):
                lines = [_form_str(x) if not isinstance(x, str) else x.rstrip() for x in raw]
            elif isinstance(raw, str):
                lines = [ln.rstrip() for ln in raw.splitlines()]
            else:
                lines = [ln.rstrip() for ln in _form_str(raw).splitlines()]
            while lines and not lines[-1].strip():
                lines.pop()
            if name in form:
                typed[k] = "\n".join(lines)
            out[k] = lines if lines else dflt
            continue
        s = _form_str(raw)
        if name in form:
            typed[k] = s
        out[k] = dflt
        if not s:
            continue
        if t == "int":
            lo = fld.get("min", 0)
            if _WHOLE_RE.fullmatch(s) and int(s) >= lo:
                out[k] = int(s)
            else:
                errors.append(f"{fld['label']}: '{s}' is not a whole number ≥ {lo}")
        elif t == "select":
            if s in fld["choices"]:
                out[k] = s
            else:
                errors.append(f"{fld['label']}: '{s}' is not one of "
                              f"{', '.join(fld['choices'])}")
        elif k == "comfy_commit" and not _COMMIT_RE.fullmatch(s):
            errors.append(f"{fld['label']}: '{s}' is not a full 40-hex commit sha "
                          "(blank = the default pin)")
        else:
            out[k] = s
    return out, errors, typed

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
            # what an instance costs (OpenAPI InstanceListItem; counts come as strings) —
            # a foreign instance is priced from these, never from our own backend's form
            "gpu_type": str(it.get("gpuType") or ""),
            "num_gpus": _int(it.get("numGpus")),
            "cpu_cores": _int(it.get("cpuCores")),
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


SNAPSHOT_FAMILY = "aihub-"            # every snapshot this gateway writes starts so


def snapshot_prefix(backend_name: str) -> str:
    return f"{SNAPSHOT_FAMILY}{_slug(backend_name)}-"


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


def foreign_snapshots(snaps, backend_names, pricing: Optional[dict] = None) -> list[dict]:
    """Snapshots named like this gateway's (`aihub-…`) that NO current Thunder backend
    owns (`_owned` against every name) — `{id, name, status, gb, monthly}`, by name.
    Rotation only ever looks at the current name's snapshots, so a renamed or deleted
    backend leaves its old ones billing $/month unseen. Display only: a hand-made
    `aihub-…` snapshot lands here too, which is why nothing ever deletes from this list.
    `gb` is Thunder's minimum restore disk (it reports no size), `monthly` None without a
    price list or a size — never a made-up figure."""
    names = [str(n) for n in (backend_names or [])]
    out = []
    for s in snaps or []:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or "")
        if not name.startswith(SNAPSHOT_FAMILY) or any(_owned(s, n) for n in names):
            continue
        gb = _int(s.get("min_disk_gb")) or None
        out.append({"id": str(s.get("id") or ""), "name": name,
                    "status": str(s.get("status") or ""), "gb": gb,
                    "monthly": snapshot_monthly(pricing, gb) if (gb and pricing) else None})
    return sorted(out, key=lambda x: (x["name"], x["id"]))


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
