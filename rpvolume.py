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
        self.leftovers = {}
        self._plan_inputs = None
        self._head_sizes, self._head_retry_at = {}, {}
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
        self._fetch_problem = ""
        self._ghost = {p: t for p, t in self.state['ghost'].items() if t > deps.now()}
        if self._ghost != self.state['ghost']:
            self.state['ghost'] = dict(self._ghost)
            self._save()
        self.last_error: Optional[BaseException] = None   # the last round's S3/REST error (backoff, auth pause)

    def _empty_state(self) -> dict:
        return {"id": None, "dc": self.cfg["datacenter"], "size_gb": None,
                "missing": 0, "fetch_job": None, "mpu": [],
                "url_fallback": {}, "url_fallback_why": {},
                "sync_cost_usd": 0.0, "sync_seconds": 0.0, "blocked": {}, "ghost": {},
                "job_failures": 0, "job_failure_reason": ""}

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
        if self.state['job_failures'] >= 3:
            lines.append('fetch jobs paused after 3 failed jobs (last: ' +
                         self.state['job_failure_reason'] + ') — Sync now retries')
        elif self._fetch_problem:
            lines.append(self._fetch_problem)
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
        return (sum(v for v in self.dest.values() if isinstance(v, int))
                + sum(self.leftovers.values()))

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
                    self.leftovers = {p: v for p, v in dest.items() if p.endswith(".gw-part")}
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
                    self._plan_inputs = (needs, src, self.urls)
                    plan = self._round_plan = modelsync.plan(needs, src, self.dest,
                                                             self.manifest, self.urls)
                    self._drop_present_blocks(plan)
                    # A raised ceiling releases the old capacity decision for this round.
                    before = dict(self.state['blocked'])
                    for p, reason in list(before.items()):
                        if (self._capacity_reason(reason) and
                                not reason.endswith(f"limit {self.cfg['max_size_gb']} GB")):
                            self.state['blocked'].pop(p)
                    if before != self.state['blocked']:
                        self._save()
                    if self._backends:
                        await self._space_for(self._remaining_need(plan))
                    self._refresh_plan()
                    self.sync_error = ""
                except (s3vol.S3Error, httpx.TransportError, RestError) as exc:
                    # Provider error bodies may contain credentials or signed URLs. The
                    # last good snapshot stays: the files it saw do not vanish with a 503.
                    self.sync_error = f"volume sync failed ({type(exc).__name__})"
                    self.last_error = exc
                finally:
                    self._round_plan = None
                    self._s3 = None

    @staticmethod
    def _capacity_reason(reason):
        return reason.startswith('needs ') and ', limit ' in reason

    def _held_paths(self):
        held = set(self._inflight)
        held.update(i['path'] for i in (self.state['fetch_job'] or {}).get('items', []))
        held.update(m['key'] for m in self.state['mpu'])
        held.update(p for p, t in self._ghost.items() if t > self.deps.now())
        return held

    def _drop_present_blocks(self, plan):
        before = dict(self.state['blocked'])
        for row in plan['per_alias'].values():
            for f in row['files']:
                if f['present']:
                    self.state['blocked'].pop(f['path'], None)
        if before != self.state['blocked']:
            self._save()

    def _refresh_plan(self):
        """Publish each completed unit immediately without paying for another S3 listing."""
        if self._plan_inputs is None:
            return
        needs, src, urls = self._plan_inputs
        urls, _ = modelsync.without_fallbacks(urls, self.state['url_fallback'])
        self.urls = urls
        plan = modelsync.plan(needs, src, self.dest, self.manifest, urls)
        self._drop_present_blocks(plan)
        if not self._backends:
            plan['prune'] = []
        ready = {a for a, row in plan['per_alias'].items()
                 if not self._problem and modelsync.ready(plan, a) and not any(
                     f['path'] in self.state['blocked'] for f in row['files'])}
        self.plan, self.ready_aliases = plan, ready

    def _entry_need(self, entry):
        """A partial already occupies used bytes, so only its remaining bytes need room."""
        return max(0, (entry['size'] or 0) - self.leftovers.get(entry['path'] + '.gw-part', 0))

    def _remaining_need(self, plan):
        """Growth counts only missing transferable bytes in the current destination."""
        entries = {e['path']: e for e in plan['fetch']}
        return sum(self._entry_need(e) for e in entries.values()
                   if e['source'] != 'link' and e['path'] not in self.state['blocked']
                   and (e['size'] is None or self.dest.get(e['path']) != e['size']))

    def _can_progress(self):
        if self._inflight or self.transfers or self.state['fetch_job']:
            return True
        if (self.plan is None or not all(self.deps.creds()) or self._paused_creds == self.deps.creds()
                or self.state['missing'] >= GONE_AFTER or self._problem):
            return False
        held = self._held_paths()
        for e in self.plan['fetch']:
            if e['path'] in held or e['path'] in self.state['blocked']:
                continue
            if e['source'] == 'link':
                if self._backends and modelsync.link_target(e['path'], e['target']) in self.dest:
                    return True
            elif e['path'] in self.urls:
                if self._backends and self.state['job_failures'] < 3:
                    return True
            elif self.deps.lan and self.deps.lan.usable():
                return True
        return False

    async def _write_manifest(self, man: dict) -> None:
        """Whole-object replacement keeps deleted paths out of persisted ownership."""
        await self._s3.put(MANIFEST_KEY, json.dumps(man, sort_keys=True).encode("utf-8"))

    async def _space_for(self, need_bytes: int) -> int:
        """Only owned leftovers may make room; unknown objects still consume capacity."""
        before = dict(self.state["blocked"])
        size = self.state["size_gb"] or self.cfg["size_gb"]
        if self._used_bytes() + need_bytes > size * 10**9 * .95:
            inflight = self._held_paths()
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
            if self._capacity_reason(blocked[p]):
                del blocked[p]
        allocated = set()
        for alias, row in sorted(self._round_plan["per_alias"].items(),
                                 key=lambda item: (item[1]["need_bytes"] - item[1]["have_bytes"], item[0])):
            missing = [f for f in row["files"] if not f["present"] and f["path"] not in allocated
                       and f["path"] not in blocked and "link" not in f]
            amount = sum(self._entry_need(f) for f in missing)
            if amount > available:
                reason = f"needs {amount / 1e9:.1f} GB, limit {self.cfg['max_size_gb']} GB"
                for f in missing:
                    blocked.setdefault(f["path"], reason)
            elif not row["blocked"]:
                available -= amount
                allocated.update(f["path"] for f in missing)
        if before != blocked:
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
                "size_gb": size, "start_size_gb": self.cfg["size_gb"],
                "max_size_gb": self.cfg["max_size_gb"],
                "used_bytes": self._used_bytes(), "missing": self.state["missing"],
                "leftovers": [[p, n] for p, n in sorted(self.leftovers.items())],
                "phase": ("off" if not self.state["id"] else
                          "syncing" if self._can_progress() else
                          "waiting" if self.plan is None or self.plan["fetch"] or self.problems()
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
        size = await s3.head(path)
        if size is None or (entry['size'] is not None and size != entry['size']):
            return False
        man = dict(self.manifest)
        man[path] = dict(size=size, sha256=sha256, source=source,
                         aliases=entry.get('aliases', []), ts=self.deps.now())
        await s3.put(MANIFEST_KEY, json.dumps(man, sort_keys=True).encode())
        self.manifest = man
        self.dest[path] = size
        self.leftovers.pop(path + '.gw-part', None)
        return True

    def _attempt_failed(self, entry, error):
        path = entry['path']
        self.attempts[path] = self.attempts.get(path, 0) + 1
        if self.attempts[path] < FETCH_ATTEMPTS and not error.startswith('final:'):
            return
        url = entry.get('url') or self.urls.get(path, {}).get('url')
        self.state['url_fallback'][path] = url
        # Worker error bodies can echo a signed URL or credentials.
        reason = self._scrub_error(error)
        self.state['url_fallback_why'][path] = reason
        if not isinstance(self.deps.source_index().get(path), int):
            self.state['blocked'][path] = reason
            self.deps.note_fault('error', f'RunPod volume {self.name}: {path}: {reason}')
        self._save()
        self._sync_requested = True

    def _scrub_error(self, detail):
        """Provider errors may echo credentials and signed URLs into the console."""
        detail = str(detail)
        for value in (*self.deps.creds(), self.deps.hf_token()):
            if value:
                detail = detail.replace(value, '[redacted]')
        return re.sub(r'https?://[^\s]+', '[redacted]', detail)[:200]

    def _job_failed(self, status):
        output = status.get('output') or {}
        detail = self._scrub_error(status.get('error') or output.get('error') or
                                  status.get('status') or 'result not verified')
        self._fetch_problem = ('fetch jobs paused after 3 failed jobs (last: ' + detail +
                               ') — Sync now retries' if self.state['job_failures'] >= 3 else
                               'fetch job failed: ' + detail)
        raise RuntimeError(self._fetch_problem)

    def _job_outcome(self, status, results):
        """Only consecutive job-wide failures should pause paid endpoint work."""
        output = status.get('output') or {}
        if results:
            self.state['job_failures'] = 0
            self.state['job_failure_reason'] = ''
        else:
            self.state['job_failures'] += 1
            self.state['job_failure_reason'] = self._scrub_error(status.get('error') or
                output.get('error') or status.get('status') or 'result not verified')

    async def _finish_fetch(self, s3, job, status, entries=None):
        output = status.get('output') or {}
        results = {r.get('path'): r for r in output.get('results', [])}
        by_path = {e['path']: e for e in entries or []}
        failed = not results
        for item in job['items']:
            entry = {**item, **by_path.get(item['path'], {})}
            result = results.get(item['path'])
            if result is None:
                if not results:
                    await self._record_present(s3, entry, '', 'url')
                failed = True
            elif result.get('ok') is False:
                self._attempt_failed(entry, result.get('error') or 'fetch item failed')
            elif result.get('ok') and await self._record_present(s3, entry, result.get('sha256'), 'url'):
                self.attempts.pop(item['path'], None)
            else:
                failed = True
        self._job_outcome(status, results)
        backend = next((b for b in self.deps.backends(self.name)
                        if 'runpod:' + b['name'] == job['bid']), {})
        ms = status.get('executionTime') or 0
        self.state['sync_cost_usd'] += ms / 3.6e6 * float(backend.get('cost_per_hour') or 0)
        self.state['fetch_job'] = None
        self._save()
        self._refresh_plan()
        if failed:
            self._job_failed(status)
        self._fetch_problem = ''

    def _hold_uncertain(self, exc, paths, bid, budget_s):
        """Only an ambiguous /run reserves paths without an id, including across restart."""
        if not (isinstance(exc, (httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError,
                                 asyncio.CancelledError)) or
                isinstance(exc, RuntimeError) and
                str(exc) == 'RunPod answered /run without a readable job id'):
            return
        backend = next((b for b in self._backends or self.deps.backends(self.name)
                        if 'runpod:' + b['name'] == bid), {})
        until = self.deps.now() + budget_s + float(backend.get('queue_max_s', 300)) + 60
        self._ghost.update({p: until for p in paths})
        self.state['ghost'] = dict(self._ghost)
        self._save()

    async def _fetch_batch(self, bid: str, entries: list) -> None:
        """Persist the billed job before polling and keep its paths reserved until settled."""
        if self.state['fetch_job'] or any(e['path'] in self._inflight for e in entries):
            return
        budget_s = (4 * 3600 if any(e['size'] is None for e in entries) else
                    max(600, min(4 * 3600, sum(e['size'] for e in entries) / 20e6 + 300)))
        items = [{k: e[k] for k in ('path', 'url', 'size', 'sha256') if k in e}
                 for e in entries]
        paths = {e['path'] for e in entries}
        self._inflight.update(paths)
        def on_id(jid):
            self.state['fetch_job'] = dict(id=jid, bid=bid, budget_s=budget_s, items=[
                {k: e[k] for k in ('path', 'size', 'url', 'aliases') if k in e}
                for e in entries], ts=self.deps.now())
            self._save()
        try:
            try:
                status = await self.deps.run_fetch(bid, {'op': 'fetch', 'items': items}, on_id,
                                                   max_wait=budget_s)
            except BaseException as exc:
                if not self.state['fetch_job']:
                    self._hold_uncertain(exc, paths, bid, budget_s)
                elif not isinstance(exc, asyncio.CancelledError):
                    try:
                        self._job_failed({'error': str(exc)})
                    except RuntimeError:
                        pass
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
            if len(chunk) >= 1024 * 1024:
                await asyncio.to_thread(digest.update, chunk)
            else:
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
        if digest.hexdigest() != expected and hasattr(self.deps.lan, 'forget_sha'):
            self.deps.lan.forget_sha(entry['path'], entry['size'])
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
                        self._refresh_plan()
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
                   e['path'] not in self._held_paths() and e['path'] not in self.state['blocked']
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
            self.state['fetch_job'] = dict(id=jid, bid=bid, ts=self.deps.now(), budget_s=600,
                items=[dict(path=e['path'], size=0, url='', target=e['target'],
                            aliases=e.get('aliases', [])) for e in entries])
            self._save()
        try:
            try:
                status = await self.deps.run_fetch(bid, {'op': 'link', 'links': [
                    dict(path=e['path'], target=e['target']) for e in entries]}, on_id, max_wait=600)
            except BaseException as exc:
                if not self.state['fetch_job']:
                    self._hold_uncertain(exc, paths, bid, 600)
                else:
                    try:
                        self._job_failed({'error': str(exc)})
                    except RuntimeError:
                        pass
                raise
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
                self.dest[e['path']] = {'link': e['target']}
                man[e['path']] = dict(source='link', target=e['target'], link=e['target'],
                                      aliases=e.get('aliases', []), ts=self.deps.now())
        await s3.put(MANIFEST_KEY, json.dumps(man, sort_keys=True).encode())
        self.manifest = man
        self._job_outcome(status, results)
        backend = next((b for b in self.deps.backends(self.name)
                        if 'runpod:' + b['name'] == job['bid']), {})
        self.state['sync_cost_usd'] += (status.get('executionTime') or 0) / 3.6e6 * float(backend.get('cost_per_hour') or 0)
        self.state['fetch_job'] = None
        self._save()
        self._refresh_plan()
        if not results:
            self._job_failed(status)
        self._fetch_problem = ''

    async def _fetch_entry(self, entry):
        """A failed HEAD does not prove GET will fail, especially with endpoint HF_TOKEN."""
        if entry['size'] is not None:
            return entry
        entry = dict(entry)
        url, now = entry['url'], self.deps.now()
        if url not in self._head_sizes or (self._head_sizes[url] is None and
                                          now >= self._head_retry_at[url]):
            headers = {}
            host = (httpx.URL(url).host or '').lower()
            if any(host == h or host.endswith('.' + h) for h in _HF_HOSTS):
                tok = str(self.deps.hf_token() or '')
                if tok:
                    headers['Authorization'] = 'Bearer ' + tok
            size = None
            try:
                async with self.deps.client_factory() as client:
                    response = await client.head(url, headers=headers, timeout=30, follow_redirects=True)
                    if 200 <= response.status_code < 300:
                        n = int(response.headers.get('content-length', '-1'))
                        if n >= 0:
                            size = n
            except (httpx.TransportError, ValueError):
                pass
            self._head_sizes[url] = size
            self._head_retry_at[url] = now + 600
        entry['size'] = self._head_sizes[url]
        if entry['size'] is not None:
            async with self._plan_lock:
                async with self.deps.client_factory() as client:
                    self._s3 = self._client_s3(client)
                    self._round_plan = copy.deepcopy(self.plan)
                    self._round_plan['fetch'] = [entry if e['path'] == entry['path'] else e
                                                 for e in self._round_plan['fetch']]
                    for row in self._round_plan['per_alias'].values():
                        for f in row['files']:
                            if f['path'] == entry['path']:
                                f['size'] = entry['size']
                                f['present'] = self.dest.get(f['path']) == f['size']
                        row['need_bytes'] = sum(f['size'] or 0 for f in row['files'])
                        row['have_bytes'] = sum(f['size'] or 0 for f in row['files'] if f['present'])
                    try:
                        await self._space_for(self._remaining_need(self._round_plan))
                        # Later HEADs in this batch must include sizes already learned.
                        self.plan = self._round_plan
                        if entry['path'] in self.state['blocked']:
                            return None
                    finally:
                        self._s3 = self._round_plan = None
        return entry

    async def _transfer_round(self) -> None:
        """One unit per tick exposes readiness and lets signature changes interrupt long queues."""
        if (not self._resumed or self._transfer_lock.locked() or self.plan is None
                or not self.plan['fetch']):
            return
        async with self._transfer_lock:
            self._transfer_problem = ''
            ghost = {p: t for p, t in self._ghost.items() if t > self.deps.now()}
            if ghost != self._ghost:
                self._ghost = ghost
                self.state['ghost'] = dict(ghost)
                self._save()
            batch, amount = [], 0
            backend = next(iter(sorted(self._backends, key=lambda b: b['name'])), None)
            for entry in self.plan['fetch']:
                path = entry['path']
                if (path in self._held_paths() or path in self.state['blocked']
                        or entry['source'] == 'link'):
                    continue
                if path in self.urls:
                    if self.state['job_failures'] >= 3:
                        continue
                    if not backend:
                        self._transfer_problem = 'needs a RunPod endpoint on this volume'
                        continue
                    entry = await self._fetch_entry(entry)
                    if entry is None:
                        continue
                    size = entry['size']
                    if batch and (size is None or batch[0]['size'] is None or
                                  len(batch) == JOB_FILES or amount + size > JOB_BYTES):
                        break
                    batch.append(entry)
                    amount += size or 0
                    if size is None or amount >= JOB_BYTES or len(batch) == JOB_FILES:
                        break
                elif self.deps.lan and self.deps.lan.usable():
                    if batch:
                        break
                    await self._lan_stream(entry)
                    return
                else:
                    problem = self.deps.lan.problem() if self.deps.lan else 'not configured'
                    self._transfer_problem = f'waiting for LAN source ({problem})'
            batch = [e for e in batch if e['path'] not in self.state['blocked']]
            if batch:
                await self._fetch_batch('runpod:' + backend['name'], batch)
                return
            if self._resumed:
                await self._make_links(self.plan)

    async def _resume_job(self, s3, job):
        """An unconfirmed cancel keeps the restart gate closed, even after the deadline."""
        paths = {i['path'] for i in job['items']}
        self._inflight.update(paths)
        deadline = self.deps.now() + job.get('budget_s', 4 * 3600) + 300 + 60
        while True:
            status = await self.deps.fetch_status(job['bid'], job['id'])
            if status is None:
                self._fetch_problem = (f"fetch job {job['id']} result lost (expired at RunPod)"
                                          ' — files are re-checked')
                self.deps.log(f'RunPod volume {self.name}: {self._fetch_problem}')
                self.state['fetch_job'] = None
                self._save()
                break
            if status.get('status') in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'):
                try:
                    if any('target' in i for i in job['items']):
                        await self._finish_links(s3, job, status)
                    else:
                        await self._finish_fetch(s3, job, status)
                finally:
                    if self.state['fetch_job'] is None:
                        self._inflight.difference_update(paths)
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
                held = self._held_paths()
                unknown.update(p for p in dest if p.endswith('.gw-part'))
                for path in dict.fromkeys(paths):
                    final = path[:-8] if path.endswith('.gw-part') else path
                    if path in unknown and path not in man and final not in held:
                        await s3.delete(path)
                        self.dest.pop(path, None)
                        self.leftovers.pop(path, None)
                        count += 1
        self._sync_requested = True
        return count

    def sync_now(self, recreate: bool = False) -> None:
        """An operator repair releases stale failures without disturbing active writers."""
        self._sync_requested = True
        self._recreate = self._recreate or recreate
        held = self._held_paths()
        forgotten = (set(self._stream_failures) |
                     {p for p, reason in self.state['blocked'].items()
                      if reason == 'share file changing?'}) - held
        for path in forgotten:
            if hasattr(self.deps.lan, 'forget_sha'):
                size = next((e['size'] for e in (self.plan or {}).get('fetch', [])
                             if e['path'] == path), 0) or 0
                # LanSource requires size but invalidates by path even before a plan.
                self.deps.lan.forget_sha(path, size)
        for mapping in (self.attempts, self._stream_failures, self.state['url_fallback'],
                        self.state['url_fallback_why']):
            for path in set(mapping) - held:
                mapping.pop(path)
        for path, reason in list(self.state['blocked'].items()):
            if path not in held and not self._capacity_reason(reason):
                self.state['blocked'].pop(path)
        self.state['job_failures'] = 0
        self.state['job_failure_reason'] = ''
        self._fetch_problem = ''
        self._head_sizes.clear()
        self._head_retry_at.clear()
        self._save()

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
