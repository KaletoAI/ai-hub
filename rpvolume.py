"""RunPod volume ownership and lifecycle, with dependencies injected by main.

A lost create answer must be recovered by listing, and destructive calls must
verify ownership afresh: a stale saved id is never permission to spend or delete.
"""
import asyncio
import copy
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import httpx

REST = "https://rest.runpod.io/v1"
NAME_RE = re.compile(r"^[a-z0-9-]{1,40}$")
DCS = ("EU-RO-1", "EU-CZ-1", "EUR-IS-1", "EUR-NO-1", "US-CA-2",
       "US-GA-2", "US-IL-1", "US-KS-2", "US-MD-1", "US-MO-1", "US-MO-2",
       "US-NC-1", "US-NC-2", "US-NE-1", "US-WA-1")
GONE_AFTER = 2
FETCH_ATTEMPTS = 3


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
