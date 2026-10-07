"""RunPod volume ownership and lifecycle, with dependencies injected by main.

A lost create answer must be recovered by listing, and destructive calls must
verify ownership afresh: a stale saved id is never permission to spend or delete.
"""
import asyncio
import copy
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import httpx

import modelsync
import s3vol

REST = "https://rest.runpod.io/v1"
NAME_RE = re.compile(r"^[a-z0-9-]{1,40}$")
DCS = ("EU-RO-1", "EU-CZ-1", "EUR-IS-1", "EUR-NO-1", "US-CA-2",
       "US-GA-2", "US-IL-1", "US-KS-2", "US-MD-1", "US-MO-1", "US-MO-2",
       "US-NC-1", "US-NC-2", "US-NE-1", "US-WA-1")
GONE_AFTER = 2
FETCH_ATTEMPTS = 3
MANIFEST_KEY = ".gw-modelsync.json"


def rp_name(name: str) -> str:
    return "aihub-" + name


class RestError(RuntimeError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"RunPod REST {status}: {message}")


class RestAuthError(RestError):
    pass


async def _request(client, key, method, path, body=None):
    response = await client.request(method, REST + path,
                                    headers={"Authorization": "Bearer " + key},
                                    json=body, timeout=30)
    if not 200 <= response.status_code < 300:
        cls = RestAuthError if response.status_code in (401, 403) else RestError
        raise cls(response.status_code, response.text[:200])
    return response


async def list_volumes(client, key) -> list[dict]:
    data = (await _request(client, key, "GET", "/networkvolumes")).json()
    if isinstance(data, dict):
        data = data.get("networkVolumes")
    # Treating a malformed list as empty could order another billed volume.
    if not isinstance(data, list) or any(not isinstance(v, dict) for v in data):
        raise RestError(200, "invalid network volume list")
    return data


async def create_volume(client, key, name, size_gb, dc) -> dict:
    return (await _request(client, key, "POST", "/networkvolumes",
                           {"name": rp_name(name), "size": size_gb,
                            "dataCenterId": dc})).json()


async def grow_volume(client, key, vid, size_gb) -> dict:
    return (await _request(client, key, "PATCH", "/networkvolumes/" + vid,
                           {"size": size_gb})).json()


async def delete_volume(client, key, vid) -> None:
    await _request(client, key, "DELETE", "/networkvolumes/" + vid)


@dataclass
class VolumeDeps:
    client_factory: Callable[[], Any]
    load_state: Callable[[str], Optional[dict]]
    save_state: Callable[[str, dict], None]
    creds: Callable[[], tuple]
    backends: Callable[[str], list]
    alias_needs: Callable[[str], list]
    alias_signature: Callable[[str], str]
    source_index: Callable[[], dict]
    url_catalog: Callable[[dict], dict]
    lan: Any
    run_fetch: Callable[..., Awaitable[dict]]
    fetch_status: Callable[..., Awaitable[Optional[dict]]]
    fetch_cancel: Callable[..., Awaitable[bool]]
    note_fault: Callable[[str, str], None]
    log: Callable[[str], None]
    now: Callable[[], float] = time.time
    s3_factory: Optional[Callable] = None


class VolumeController:
    def __init__(self, name: str, cfg: dict, deps: VolumeDeps):
        self.name = name
        self.cfg = cfg
        self.deps = deps
        loaded = deps.load_state(name)
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError("invalid RunPod volume state")
        self.state = self._empty_state()
        if loaded is not None:
            self.state.update(copy.deepcopy(loaded))
        self._problem = ""
        self._api_rejected = None
        self._lock = asyncio.Lock()
        self._plan_lock = asyncio.Lock()
        self.plan = None
        self.ready_aliases = set()
        self.dest, self.manifest, self.urls = {}, {}, {}
        self.transfers = []
        self.sync_error = ""
        self._backends = []
        self._s3 = None
        self._round_plan = None
        self.last_error: Optional[BaseException] = None   # the last round's S3/REST error (backoff, auth pause)

    def _empty_state(self) -> dict:
        return {"id": None, "dc": self.cfg["datacenter"], "size_gb": None,
                "missing": 0, "fetch_job": None, "mpu": [],
                "url_fallback": {}, "url_fallback_why": {},
                "sync_cost_usd": 0.0, "sync_seconds": 0.0, "blocked": {}}

    def _save(self):
        self.deps.save_state(self.name, self.state)

    def problems(self) -> list[str]:
        """Keep credential and ownership checks readable without doing request-time I/O."""
        key, access, secret = self.deps.creds()
        lines = []
        if not key:
            lines.append("runpod API key missing")
        elif self._api_rejected == key:
            lines.append("API key rejected")
        if not access or not secret:
            lines.append("runpod_s3 key missing")
        if self._problem:
            lines.append(self._problem)
        if self.state["missing"] >= GONE_AFTER:
            lines.append("deleted outside AI-Hub? Sync now to create a new billed volume")
        return lines

    def _owned(self, vol) -> bool:
        if vol.get("name") != rp_name(self.name):
            self._problem = "RunPod volume name does not match; refusing ownership"
            return False
        if vol.get("dataCenterId") != self.cfg["datacenter"]:
            self._problem = "RunPod volume DC does not match; refusing ownership"
            return False
        self._problem = ""
        return True

    def _remember(self, vol, size_gb=None):
        """Save what RunPod answered — tolerant of a partial record: the create/PATCH
        answers are not documented field by field [U], and a KeyError AFTER a billed
        call would lose the new id or size. Without an id nothing is saved; the next
        round's list adopts the volume by name."""
        vid = vol.get("id") if isinstance(vol, dict) else None
        if not vid:
            raise RestError(200, "RunPod answered without a volume id")
        size = vol.get("size")
        self.state.update(id=str(vid), dc=str(vol.get("dataCenterId") or self.cfg["datacenter"]),
                          size_gb=int(size) if isinstance(size, int) and size > 0
                          else (size_gb or self.state.get("size_gb")),
                          missing=0)
        self._save()

    async def _list(self, client, key):
        try:
            vols = await list_volumes(client, key)
        except RestAuthError:
            self._api_rejected = key
            raise
        self._api_rejected = None
        return vols

    async def _ensure(self, client, key, create):
        vols = await self._list(client, key)
        vid = self.state["id"]
        if vid:
            vol = next((v for v in vols if v.get("id") == vid), None)
        else:
            vol = next((v for v in vols if v.get("name") == rp_name(self.name)), None)
        if vol is not None:
            if not self._owned(vol):
                return None
            self._remember(vol)
            return vol
        if vid:
            self.state["missing"] += 1
            self._save()
            if self.state["missing"] < GONE_AFTER or not create:
                return None
            # Explicit re-creation still checks for a same-named replacement first.
            replacement = next((v for v in vols if v.get("name") == rp_name(self.name)), None)
            if replacement is not None:
                if not self._owned(replacement):
                    return None
                self._remember(replacement)
                return replacement
        if not create:
            return None
        try:
            vol = await create_volume(client, key, self.name, self.cfg["size_gb"],
                                      self.cfg["datacenter"])
        except httpx.TransportError:
            # The server may have created it; the next round lists before retrying.
            return None
        except RestAuthError:
            self._api_rejected = key
            raise
        self._remember(vol, self.cfg["size_gb"])
        return vol

    async def ensure_volume(self, create: bool) -> Optional[dict]:
        key = self.deps.creds()[0]
        if not key:
            return None
        async with self._lock:
            async with self.deps.client_factory() as client:
                return await self._ensure(client, key, create)

    async def grow_to(self, size_gb: int) -> bool:
        key = self.deps.creds()[0]
        if not key:
            return False
        async with self._lock:
            async with self.deps.client_factory() as client:
                vol = await self._ensure(client, key, False)
                if vol is None:
                    return False
                old = vol["size"]
                if size_gb <= old:
                    return True
                if size_gb > self.cfg["max_size_gb"]:
                    return False
                try:
                    grown = await grow_volume(client, key, vol["id"], size_gb)
                except RestAuthError:
                    self._api_rejected = key
                    raise
                answer = grown if isinstance(grown, dict) else {}
                # the listed size is the OLD one: an answer without `size` means size_gb
                self._remember({**vol, **answer, "size": answer.get("size") or size_gb}, size_gb)
                new = self.state["size_gb"]
                self.deps.note_fault("info", f"RunPod volume {self.name} grew {old} → "
                                     f"{new} GB; monthly cost ${old * .07:.2f} → "
                                     f"${new * .07:.2f}")
                return True

    async def delete_volume(self) -> str:
        async with self._lock:
            if self.deps.backends(self.name):
                return "Volume is referenced by a RunPod backend"
            key = self.deps.creds()[0]
            if not key:
                return "runpod API key missing"
            async with self.deps.client_factory() as client:
                vol = await self._ensure(client, key, False)
                if vol is None:
                    return self._problem or "Volume id not found in fresh list"
                # A backend may have attached while the list request was in flight.
                if self.deps.backends(self.name):
                    return "Volume is referenced by a RunPod backend"
                try:
                    await delete_volume(client, key, vol["id"])
                except RestAuthError:
                    self._api_rejected = key
                    raise
                self.state = self._empty_state()
                self._save()
                return ""


    def _used_bytes(self) -> int:
        return sum(v for v in self.dest.values() if isinstance(v, int))

    async def plan_round(self) -> None:
        """Refresh the routing snapshot before any transfer can spend disk or money."""
        async with self._plan_lock:
            # The routing snapshot (plan, ready_aliases) is swapped only at the END of a
            # good round: cleared up front, every round would 503 the RunPod aliases for
            # its whole duration, and a transient S3 error would block models that are
            # still on the volume. Only a volume that is gone or not ours clears it.
            self.last_error = None
            key, access, secret = self.deps.creds()
            if not key or not access or not secret:
                return
            if not self.state["id"]:
                self._problem = "volume id missing"
                self.plan, self.ready_aliases = None, set()
                return
            if self._problem == "volume id missing":
                self._problem = ""
            if self._problem or self.state["missing"] >= GONE_AFTER:
                self.plan, self.ready_aliases = None, set()
                return

            def inputs():
                backends = self.deps.backends(self.name)
                needs = modelsync._merge_needs([
                    n for b in backends
                    for n in self.deps.alias_needs(f"runpod:{b['name']}")])
                src = self.deps.source_index()
                return backends, needs, src, self.deps.url_catalog(src)

            self._backends, needs, src, urls = await asyncio.to_thread(inputs)
            async with self.deps.client_factory() as client:
                self._s3 = (self.deps.s3_factory(client) if self.deps.s3_factory else
                            s3vol.S3Volume(client, s3vol.endpoint_for(self.cfg["datacenter"]),
                                           self.state["id"], access, secret,
                                           self.cfg["datacenter"].lower()))
                try:
                    dest = await self._s3.list_objects("models/")
                    dest.update(await self._s3.list_objects("hf-cache/"))
                    self.dest = {p: v for p, v in dest.items() if not p.endswith(".gw-part")}
                    raw = await self._s3.get(MANIFEST_KEY)
                    try:
                        text = (raw or b"{}").decode("utf-8")
                        if not isinstance(json.loads(text), dict):
                            raise ValueError("manifest is not an object")
                    except (ValueError, UnicodeError):
                        self.deps.log(f"RunPod volume {self.name}: unreadable manifest; files are unknown")
                        text = "{}"
                    self.manifest = modelsync.parse_manifest(text)
                    for p, entry in self.manifest.items():
                        target = entry.get("link")
                        if entry.get("source") == "link":
                            target = entry.get("target")
                        if isinstance(target, str) and not p.endswith(".gw-part"):
                            self.dest[p] = {"link": target}
                    self.urls, stale = modelsync.without_fallbacks(urls, self.state["url_fallback"])
                    if stale:
                        for p in stale:
                            self.state["url_fallback"].pop(p, None)
                            self.state["url_fallback_why"].pop(p, None)
                        self._save()
                    plan = self._round_plan = modelsync.plan(needs, src, self.dest,
                                                             self.manifest, self.urls)
                    if self._backends:
                        await self._space_for(sum(e["size"] or 0 for e in plan["fetch"]))
                        plan = modelsync.plan(needs, src, self.dest, self.manifest, self.urls)
                    else:
                        # An idle volume keeps all owned leftovers, even if it is full.
                        plan["prune"] = []
                    ready = {a for a, row in plan["per_alias"].items()
                             if not self._problem and modelsync.ready(plan, a) and not any(
                                 f["path"] in self.state["blocked"] for f in row["files"])}
                    self.plan, self.ready_aliases = plan, ready
                    self.sync_error = ""
                except (s3vol.S3Error, httpx.TransportError, RestError) as exc:
                    # Provider error bodies may contain credentials or signed URLs. The
                    # last good snapshot stays: the files it saw do not vanish with a 503.
                    self.sync_error = f"volume sync failed ({type(exc).__name__})"
                    self.last_error = exc
                finally:
                    self._round_plan = None
                    self._s3 = None

    async def _write_manifest(self, man: dict) -> None:
        """Whole-object replacement keeps deleted paths out of persisted ownership."""
        await self._s3.put(MANIFEST_KEY, json.dumps(man, sort_keys=True).encode("utf-8"))

    async def _space_for(self, need_bytes: int) -> int:
        """Only owned leftovers may make room; unknown objects still consume capacity."""
        size = self.state["size_gb"] or self.cfg["size_gb"]
        if self._used_bytes() + need_bytes > size * 10**9 * .95:
            job = self.state["fetch_job"] or {}
            inflight = {item["path"] for item in job.get("items", [])}
            inflight.update(item["key"] for item in self.state["mpu"])
            for p in self._round_plan["prune"]:
                if p in inflight:
                    continue
                if await self.ensure_volume(False) is None:
                    return max(0, self._used_bytes() + need_bytes - int(size * 10**9 * .95))
                await self._s3.delete(p)
                self.dest.pop(p, None)
                self.manifest.pop(p, None)
                await self._write_manifest(self.manifest)
            have = self._used_bytes()
            if have + need_bytes > size * 10**9 * .95:
                target = math.ceil((have + need_bytes) * 1.10 / 1e9)
                await self.grow_to(min(target, self.cfg["max_size_gb"]))
        size = self.state["size_gb"] or self.cfg["size_gb"]
        available = max(0, int(size * 10**9 * .95) - self._used_bytes())
        blocked = self.state["blocked"]
        # Re-evaluate capacity blocks, but preserve transfer-failure reasons.
        for p in list(blocked):
            if blocked[p].startswith("needs ") and ", limit " in blocked[p]:
                del blocked[p]
        allocated = set()
        for alias, row in sorted(self._round_plan["per_alias"].items(),
                                 key=lambda item: (item[1]["need_bytes"] - item[1]["have_bytes"], item[0])):
            missing = [f for f in row["files"] if not f["present"] and f["path"] not in allocated]
            amount = sum(f["size"] or 0 for f in missing)
            if amount > available:
                reason = f"needs {amount / 1e9:.1f} GB, limit {self.cfg['max_size_gb']} GB"
                for f in missing:
                    blocked.setdefault(f["path"], reason)
            elif not row["blocked"]:
                available -= amount
                allocated.update(f["path"] for f in missing)
        self._save()
        return max(0, self._used_bytes() + need_bytes - int(size * 10**9 * .95))

    def is_alias_ready(self, alias: str) -> bool:
        return alias in self.ready_aliases

    def alias_status(self, alias: str) -> str:
        prefix = f"RunPod volume {self.name}: "
        if self.plan is None:
            return prefix + "no plan yet"
        row = self.plan["per_alias"].get(alias, {})
        reasons = sorted({self.state["blocked"][f["path"]] for f in row.get("files", [])
                          if f["path"] in self.state["blocked"]})
        return prefix + ("; ".join(reasons) if reasons else
                         modelsync.status_text(self.plan, alias, self.name))

    def view(self) -> dict:
        """Expose the cached round without URLs or request-time network traffic."""
        size = self.state["size_gb"] or 0
        job = self.state["fetch_job"]
        if job:
            job = {k: copy.deepcopy(job[k]) for k in ("id", "bid", "ts") if k in job}
            job["items"] = [{k: item[k] for k in ("path", "size") if k in item}
                            for item in self.state["fetch_job"].get("items", [])]
        return {"name": self.name, "dc": self.cfg["datacenter"], "id": self.state["id"],
                "size_gb": size, "max_size_gb": self.cfg["max_size_gb"],
                "used_bytes": self._used_bytes(),
                "phase": ("off" if not self.state["id"] else
                          "syncing" if self.plan is None or self.plan["fetch"] or self.transfers
                          else "ready"),
                "plan": modelsync.plan_view(self.plan, self.ready_aliases, self.dest,
                                            self.manifest, self.urls) if self.plan is not None else {},
                "transfers": copy.deepcopy(self.transfers),
                "fetch_job": job,
                "url_fallback": dict(self.state["url_fallback_why"]),
                "blocked": dict(self.state["blocked"]), "sync_error": self.sync_error,
                "problems": self.problems(), "cost_month_usd": size * .07,
                "sync_cost_usd": self.state["sync_cost_usd"],
                "backends": [b["name"] for b in self._backends]}
