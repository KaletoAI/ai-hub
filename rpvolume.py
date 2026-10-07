"""RunPod volume ownership and lifecycle, with dependencies injected by main.

A lost create answer must be recovered by listing, and destructive calls must
verify ownership afresh: a stale saved id is never permission to spend or delete.
"""
import asyncio
import copy
import hashlib
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
PART_SIZE = 128 * 1024 * 1024
JOB_BYTES = 20 * 10**9
JOB_FILES = 50
MANIFEST_KEY = ".gw-modelsync.json"
# A /run whose answer was lost may still have started a job we have no id for: its
# paths stay reserved this long (≥ a fetch job's executionTimeout + queue) before
# another writer may touch them — two writers on one `.gw-part` corrupt it.
GHOST_HOLD_S = 1800
_HF_HOSTS = ("huggingface.co", "hf.co")


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
    hf_token: Callable[[], str] = lambda: ""   # only for the size HEAD of an HF URL


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
        self._inflight = set()
        self.attempts, self._stream_failures = {}, {}
        self._transfer_lock = asyncio.Lock()
        self._resumed = False
        self._closed = False
        self._runner = None
        self._sync_requested = False
        self._recreate = False
        self._backoff = 0
        self._retry_at = 0
        self._paused_creds = None
        self._s3_rejected = None
        self._transfer_problem = ""
        self._ghost: dict = {}                 # path → monotonic-ish deadline (deps.now) of a lost /run
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
        if self._s3_rejected == (access, secret):
            lines.append("S3 key rejected")
        if self._transfer_problem:
            lines.append(self._transfer_problem)
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
            inflight.update(self._inflight)
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

    def _client_s3(self, client):
        _, access, secret = self.deps.creds()
        return (self.deps.s3_factory(client) if self.deps.s3_factory else
                s3vol.S3Volume(client, s3vol.endpoint_for(self.cfg['datacenter']),
                               self.state['id'], access, secret,
                               self.cfg['datacenter'].lower()))

    async def _record_present(self, s3, entry, sha256, source):
        """Only independent S3 verification establishes manifest ownership."""
        path = entry['path']
        if await s3.head(path) != entry['size']:
            return False
        man = dict(self.manifest)
        man[path] = dict(size=entry['size'], sha256=sha256, source=source,
                         aliases=entry.get('aliases', []), ts=self.deps.now())
        await s3.put(MANIFEST_KEY, json.dumps(man, sort_keys=True).encode())
        self.manifest = man
        self.dest[path] = entry['size']
        return True

    def _attempt_failed(self, entry, error):
        path = entry['path']
        self.attempts[path] = self.attempts.get(path, 0) + 1
        if self.attempts[path] < FETCH_ATTEMPTS and not error.startswith('final:'):
            return
        url = entry.get('url') or self.urls.get(path, {}).get('url')
        self.state['url_fallback'][path] = url
        # Worker error bodies can echo a signed URL or credentials.
        reason = error
        for value in (url, *self.deps.creds()):
            if value:
                reason = reason.replace(value, '[redacted]')
        self.state['url_fallback_why'][path] = reason
        if not isinstance(self.deps.source_index().get(path), int):
            self.state['blocked'][path] = reason
            self.deps.note_fault('error', f'RunPod volume {self.name}: {path}: {reason}')
        self._save()
        self._sync_requested = True

    async def _finish_fetch(self, s3, job, status, entries=None):
        results = {r.get('path'): r for r in (status.get('output') or {}).get('results', [])}
        by_path = {e['path']: e for e in entries or []}
        for item in job['items']:
            entry = {**item, **by_path.get(item['path'], {})}
            result = results.get(item['path'], {})
            if (result.get('ok') and await self._record_present(
                    s3, entry, result.get('sha256'), 'url')):
                self.attempts.pop(item['path'], None)
            else:
                self._attempt_failed(entry, result.get('error') or 'fetch result not verified')
        backend = next((b for b in self.deps.backends(self.name)
                        if 'runpod:' + b['name'] == job['bid']), {})
        ms = status.get('executionTime') or 0
        self.state['sync_cost_usd'] += ms / 3.6e6 * float(backend.get('cost_per_hour') or 0)
        self.state['fetch_job'] = None
        self._save()

    async def _fetch_batch(self, bid: str, entries: list) -> None:
        """Persist the billed job before polling and keep its paths reserved until settled."""
        if self.state['fetch_job'] or any(e['path'] in self._inflight for e in entries):
            return
        items = [{k: e[k] for k in ('path', 'url', 'size', 'sha256') if k in e}
                 for e in entries]
        paths = {e['path'] for e in entries}
        self._inflight.update(paths)
        def on_id(jid):
            self.state['fetch_job'] = dict(id=jid, bid=bid, items=[
                {k: i[k] for k in ('path', 'size', 'url')} for i in items], ts=self.deps.now())
            self._save()
        try:
            try:
                status = await self.deps.run_fetch(bid, {'op': 'fetch', 'items': items}, on_id)
            except BaseException:
                if not self.state['fetch_job']:
                    # no id saved: the job may exist anyway (lost /run answer) — hold
                    until = self.deps.now() + GHOST_HOLD_S
                    self._ghost.update({p: until for p in paths})
                raise
            if status.get('status') not in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
                self._resumed = False
                return
            async with self.deps.client_factory() as client:
                await self._finish_fetch(self._client_s3(client), self.state['fetch_job'], status, entries)
        finally:
            if self.state['fetch_job']:
                self._resumed = False
            else:
                self._inflight.difference_update(paths)

    def _stream_failed(self, path):
        self._stream_failures[path] = self._stream_failures.get(path, 0) + 1
        if self._stream_failures[path] >= 2:
            self.state['blocked'][path] = 'share file changing?'
            self._save()

    async def _stream_parts(self, s3, entry, proc, uid, progress):
        digest, etags, body = hashlib.sha256(), [], b''
        while True:
            try:
                chunk = await proc.stdout.readexactly(PART_SIZE)
            except asyncio.IncompleteReadError as exc:
                chunk = exc.partial
            if not chunk:
                break
            digest.update(chunk)
            progress['bytes'] += len(chunk)
            if uid:
                etags.append(await s3.mpu_part(entry['path'], uid, len(etags) + 1, chunk))
            else:
                body += chunk
            if len(chunk) < PART_SIZE:
                break
        rc = await proc.wait()
        expected = await self.deps.lan.sha256(entry['path'], entry['size'])
        if rc or progress['bytes'] != entry['size'] or digest.hexdigest() != expected:
            return None
        if uid:
            try:
                await s3.mpu_complete(entry['path'], uid, etags)
            except httpx.TimeoutException:
                if await s3.head(entry['path']) != entry['size']:
                    return None
        else:
            await s3.put(entry['path'], body)
        return digest.hexdigest()

    async def _lan_stream(self, entry: dict) -> None:
        path = entry['path']
        if path in self._inflight:
            return
        self._inflight.add(path)
        progress = dict(file=path, bytes=0, total=entry['size'], via='lan')
        self.transfers.append(progress)
        proc, uid, verified = None, None, False
        try:
            async with self.deps.client_factory() as client:
                s3 = self._client_s3(client)
                try:
                    if entry['size'] >= PART_SIZE:
                        uid = await s3.mpu_create(path)
                        self.state['mpu'].append(dict(key=path, upload_id=uid))
                        self._save()
                    proc = await asyncio.create_subprocess_exec(
                        *self.deps.lan.cat_argv(path, 0), stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL)
                    sha = await self._stream_parts(s3, entry, proc, uid, progress)
                    verified = bool(sha) and await self._record_present(s3, entry, sha, 'lan')
                    if not verified:
                        self._stream_failed(path)
                    else:
                        self._stream_failures.pop(path, None)
                finally:
                    if uid:
                        if not verified:
                            await s3.mpu_abort(path, uid)
                        self.state['mpu'] = [m for m in self.state['mpu'] if m['upload_id'] != uid]
                        self._save()
        finally:
            if proc and proc.returncode is None:
                proc.kill()
                await proc.wait()
            self.transfers.remove(progress)
            self._inflight.discard(path)

    async def _make_links(self, plan) -> None:
        entries = [e for e in plan['fetch'] if e['source'] == 'link' and
                   e['path'] not in self._inflight and e['path'] not in self.state['blocked']
                   and modelsync.link_target(e['path'], e['target']) in self.dest]
        if not entries:
            return
        if not self._backends:
            self._transfer_problem = 'needs a RunPod endpoint on this volume'
            return
        # Link jobs are writers too; use the same restart record as fetch jobs.
        paths = {e['path'] for e in entries}
        self._inflight.update(paths)
        bid = 'runpod:' + sorted(self._backends, key=lambda b: b['name'])[0]['name']
        def on_id(jid):
            self.state['fetch_job'] = dict(id=jid, bid=bid, ts=self.deps.now(),
                items=[dict(path=e['path'], size=0, url='', target=e['target']) for e in entries])
            self._save()
        try:
            status = await self.deps.run_fetch(bid, {'op': 'link', 'links': [
                dict(path=e['path'], target=e['target']) for e in entries]}, on_id)
            async with self.deps.client_factory() as client:
                await self._finish_links(self._client_s3(client), self.state['fetch_job'], status, entries)
        finally:
            if self.state['fetch_job']:
                self._resumed = False
            else:
                self._inflight.difference_update(paths)

    async def _finish_links(self, s3, job, status, entries=None):
        if status.get('status') not in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
            return
        results = {r.get('path'): r for r in (status.get('output') or {}).get('results', [])}
        man = dict(self.manifest)
        for e in entries or job['items']:
            if results.get(e['path'], {}).get('ok'):
                man[e['path']] = dict(source='link', target=e['target'], link=e['target'],
                                      aliases=e.get('aliases', []), ts=self.deps.now())
        await s3.put(MANIFEST_KEY, json.dumps(man, sort_keys=True).encode())
        self.manifest = man
        backend = next((b for b in self.deps.backends(self.name)
                        if 'runpod:' + b['name'] == job['bid']), {})
        self.state['sync_cost_usd'] += (status.get('executionTime') or 0) / 3.6e6 * float(backend.get('cost_per_hour') or 0)
        self.state['fetch_job'] = None
        self._save()

    async def _fetch_entry(self, entry):
        """Catalog-only paths need a known size before bounded jobs can accept them."""
        if entry['size'] is not None:
            return entry
        entry = dict(entry)
        headers = {}
        host = (httpx.URL(entry['url']).host or '').lower()
        if any(host == h or host.endswith('.' + h) for h in _HF_HOSTS):
            tok = str(self.deps.hf_token() or '')
            if tok:
                # httpx drops Authorization on a redirect to another origin (the CDN)
                headers['Authorization'] = 'Bearer ' + tok
        async with self.deps.client_factory() as client:
            response = await client.head(entry['url'], headers=headers, timeout=30,
                                         follow_redirects=True)
            if response.status_code == 429 or response.status_code >= 500:
                raise httpx.TransportError('URL size lookup unavailable')
            if 400 <= response.status_code < 500:
                self._attempt_failed(entry, f'final: HTTP {response.status_code}')
                return None
            try:
                size = int(response.headers['content-length'])
            except (KeyError, ValueError):
                self._transfer_problem = 'fetch size unknown'
                return None
            if size < 0:
                self._transfer_problem = 'fetch size unknown'
                return None
            entry['size'] = size
            async with self._plan_lock:
                self._s3, self._round_plan = self._client_s3(client), self.plan
                try:
                    need = size + sum(e['size'] or 0 for e in self.plan['fetch'])
                    if await self._space_for(need):
                        self.state['blocked'][entry['path']] = (
                            f"needs {need / 1e9:.1f} GB, limit {self.cfg['max_size_gb']} GB")
                        self._save()
                        return None
                finally:
                    self._s3 = self._round_plan = None
        return entry

    async def _transfer_round(self) -> None:
        """Concurrent triggers cannot create a second writer or a second billed job."""
        if (not self._resumed or self._transfer_lock.locked() or self.plan is None
                or not self.plan['fetch']):
            return
        async with self._transfer_lock:
            self._transfer_problem = ''
            now = self.deps.now()
            self._ghost = {p: t for p, t in self._ghost.items() if t > now}
            batch, amount = [], 0
            backend = next(iter(sorted(self._backends, key=lambda b: b['name'])), None)
            for entry in self.plan['fetch']:
                path = entry['path']
                if (path in self._inflight or path in self._ghost or path in self.state['blocked']
                        or entry['source'] == 'link'):
                    continue
                if path in self.urls:
                    if not backend:
                        self._transfer_problem = 'needs a RunPod endpoint on this volume'
                        continue
                    entry = await self._fetch_entry(entry)
                    if entry is None:
                        continue
                    size = entry['size']
                    if size is None:
                        self._transfer_problem = 'fetch size unknown'
                        continue
                    # JOB_BYTES bounds a batch, not a file: a bigger file goes alone (its
                    # .gw-part resumes across jobs if one job's time runs out)
                    if batch and (len(batch) == JOB_FILES or amount + size > JOB_BYTES):
                        await self._fetch_batch('runpod:' + backend['name'], batch)
                        if not self._resumed: return
                        batch, amount = [], 0
                    batch.append(entry); amount += size
                elif self.deps.lan and self.deps.lan.usable():
                    await self._lan_stream(entry)
                else:
                    problem = self.deps.lan.problem() if self.deps.lan else 'not configured'
                    self._transfer_problem = f'waiting for LAN source ({problem})'
            if batch:
                await self._fetch_batch('runpod:' + backend['name'], batch)
            if self._resumed:
                await self._make_links(self.plan)
                await self.plan_round()

    async def _resume_job(self, s3, job):
        """An unconfirmed cancel keeps the restart gate closed, even after the deadline."""
        paths = {i['path'] for i in job['items']}
        self._inflight.update(paths)
        deadline = self.deps.now() + 1800
        while True:
            status = await self.deps.fetch_status(job['bid'], job['id'])
            if status is None:
                self.state['fetch_job'] = None
                self._save()
                break
            if status.get('status') in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
                if any('target' in i for i in job['items']):
                    await self._finish_links(s3, job, status)
                else:
                    await self._finish_fetch(s3, job, status)
                break
            if self.deps.now() >= deadline:
                if not await self.deps.fetch_cancel(job['bid'], job['id']):
                    return False
                self.state['fetch_job'] = None
                self._save()
                break
            await asyncio.sleep(5)
        self._inflight.difference_update(paths)
        return True

    async def resume(self) -> None:
        if self._resumed or not self.state['id'] or not all(self.deps.creds()):
            return
        async with self._transfer_lock:
            async with self.deps.client_factory() as client:
                s3 = self._client_s3(client)
                raw = await s3.get(MANIFEST_KEY)
                self.manifest = modelsync.parse_manifest((raw or b'{}').decode('utf-8', errors='replace'))
                job = self.state['fetch_job']
                if job and not await self._resume_job(s3, job):
                    return
                uploads = {(m['key'], m['upload_id']) for m in self.state['mpu']}
                uploads.update(await s3.mpu_list('models/'))
                uploads.update(await s3.mpu_list('hf-cache/'))
                for path, uid in sorted(uploads):
                    await s3.mpu_abort(path, uid)
                    self.state['mpu'] = [m for m in self.state['mpu'] if (m['key'], m['upload_id']) != (path, uid)]
                    self._save()
                self._resumed = True

    async def delete_unknown(self, paths: list) -> int:
        count = 0
        async with self._transfer_lock:
            if await self.ensure_volume(False) is None:
                return 0
            async with self.deps.client_factory() as client:
                s3 = self._client_s3(client)
                dest = await s3.list_objects('models/')
                dest.update(await s3.list_objects('hf-cache/'))
                raw = await s3.get(MANIFEST_KEY)
                man = modelsync.parse_manifest((raw or b'{}').decode('utf-8', errors='replace'))
                def inputs():                   # blocking store reads: off the loop
                    needs = modelsync._merge_needs([n for b in self.deps.backends(self.name)
                        for n in self.deps.alias_needs('runpod:' + b['name'])])
                    src = self.deps.source_index()
                    return needs, src, self.deps.url_catalog(src)
                needs, src, cat = await asyncio.to_thread(inputs)
                urls, _ = modelsync.without_fallbacks(cat, self.state['url_fallback'])
                unknown = {p for p, size in modelsync.plan(needs, src, dest, man, urls)['unknown']}
                held = set(self._inflight)
                held.update(i['path'] for i in (self.state['fetch_job'] or {}).get('items', []))
                held.update(m['key'] for m in self.state['mpu'])
                for path in dict.fromkeys(paths):
                    if path in unknown and path not in man and path not in held:
                        await s3.delete(path)
                        self.dest.pop(path, None)
                        count += 1
        self._sync_requested = True
        return count

    def sync_now(self, recreate: bool = False) -> None:
        self._sync_requested = True
        self._recreate = self._recreate or recreate

    def _signature(self):
        """Blocking store reads — called through asyncio.to_thread only."""
        backends = self.deps.backends(self.name)
        return ([(b, self.deps.alias_signature('runpod:' + b['name'])) for b in backends],
                getattr(self.deps.lan, 'generation', None),
                copy.deepcopy(self.deps.url_catalog(self.deps.source_index())), dict(self.cfg))

    def _round_failed(self, error):
        self.last_error = error
        self._sync_requested = True
        self.sync_error = f'volume sync failed ({type(error).__name__})'
        if isinstance(error, (s3vol.S3AuthError, RestAuthError)):
            self._paused_creds = self.deps.creds()
            if isinstance(error, s3vol.S3AuthError):
                self._s3_rejected = self._paused_creds[1:]
            else:
                self._api_rejected = self._paused_creds[0]
        else:
            self._backoff = min(self._backoff * 2 if self._backoff else 30, 600)
            self._retry_at = self.deps.now() + self._backoff

    async def run_forever(self) -> None:
        self._runner = asyncio.current_task()
        signature, planned_at = None, -600
        while not self._closed:
            creds = self.deps.creds()
            paused = self._paused_creds == creds
            if self._paused_creds and not paused:
                self._paused_creds = self._s3_rejected = None
                self._retry_at = 0
                self._sync_requested = True
            if not paused and all(creds) and self.deps.now() >= self._retry_at:
                try:
                    current = await asyncio.to_thread(self._signature)
                    due = self._sync_requested or current != signature or self.deps.now() - planned_at >= 600
                    if due or not self._resumed:
                        create = not self.state['id'] or self._recreate
                        await self.ensure_volume(create)
                        self._recreate = False
                        await self.resume()
                    if due:
                        await self.plan_round()
                        if self.last_error: raise self.last_error
                        signature, planned_at = copy.deepcopy(current), self.deps.now()
                        self._sync_requested = False
                    await self._transfer_round()
                    if self.last_error: raise self.last_error
                    self._backoff = 0
                    self.sync_error = ''
                except (s3vol.S3Error, RestError, httpx.TransportError) as exc:
                    self._round_failed(exc)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # anything else (a FAILED fetch job's RuntimeError, a bug) backs off and
                    # retries: a loop that died here would leave the volume unsynced for good
                    self.deps.log(f"RunPod volume {self.name}: sync round failed: "
                                  f"{type(exc).__name__}: {str(exc)[:300]}")
                    self._round_failed(exc)
            await asyncio.sleep(5)

    async def aclose(self) -> None:
        """Stopping the loop preserves unsettled job/upload records for restart recovery."""
        self._closed = True
        if self._runner and self._runner is not asyncio.current_task():
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
