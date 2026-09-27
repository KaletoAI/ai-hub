"""Thunder Compute lifecycle controller: one object per `comfyui` backend that carries
a `thunder` block. It talks to the Thunder REST API, supervises the SSH tunnel the
ComfyUI adapter reaches the instance through, and remembers WHICH instance it owns.

That last part is what everything else rests on. An instance bills by the hour
whether or not the gateway remembers it, and Thunder has no "stop" — so a gateway
restart that forgot the instance's uuid would leave it running (and billing) with
nobody to snapshot or delete it. The state is therefore persisted on EVERY phase
change (store setting `thunder_state`, one entry per backend name, written by
`main` through the injected `save_state`) and read back by the constructor;
`resume()` then reconciles it with `/instances/list`. The log ring and the live
transfer table are NOT persisted: both describe the running process, and a stale
"downloading 40 %" after a restart would be a lie.

The steps are imperative and idempotent, not a pure `next_step` state machine
(ledger Ruling 3): every step re-checks the world before acting (a snapshot with the
name already exists → not created twice; an instance already gone → delete is
done), so a resume can re-enter any phase.

Never imports `main`: everything the controller needs from the gateway arrives in
`Deps`, so it stays hot-reload-safe and testable against a stub API and a fake ssh.
`ThunderApi` owns the HTTP: Bearer token, `httpx.Timeout(30, connect=10)`, any 2xx
is success (`create` answers 201, `/snapshots/create` 202), anything else a
`thunder.ThunderError` carrying the status. Which id form `/instances/{id}/…` wants
is documented contradictorily (index vs uuid), so every such call tries the UUID
first and the index only on a 404 (a reused index could name a stranger's instance, a
uuid cannot) — to verify live. Covered by test_thunder_controller.py.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import quote

import httpx

import sshrun
import thunder

PHASES = ("off", "creating", "restoring", "connecting", "bootstrapping", "starting",
          "syncing", "ready", "draining", "pruning", "snapshotting", "deleting", "failed")

_TIMEOUT = httpx.Timeout(30, connect=10)
_PRICE_TTL_S = 3600             # /v2/pricing and /v2/specs change rarely; one fetch per hour
_ERR_MAX = 300                  # chars of an API error body kept in the message
_LOG_MAX = 200                  # lines in the per-backend log ring
_COMFY_PORT = 8188              # ComfyUI on the instance, loopback only
_SSH_USER = "ubuntu"


# ── Thunder REST client ──────────────────────────────────────────────────────

def _q(s) -> str:
    """One path segment. An id is Thunder's, but quoting keeps a `/` or `..` in it from
    addressing a DIFFERENT endpoint (`/snapshots/a/../b`)."""
    return quote(str(s), safe="")


class ThunderApi:
    """Thin async client for the endpoints the controller uses. Parsing lives in the
    pure `thunder` module; this class only does HTTP, errors and the price cache."""

    def __init__(self, client: httpx.AsyncClient, token: str, base: str = thunder.API,
                 clock: Callable[[], float] = time.monotonic):
        self._client = client
        self._token = token or ""
        self._base = base.rstrip("/")
        self._clock = clock
        self._cache: dict[str, tuple[float, Any]] = {}

    async def _req(self, method: str, path: str, body=None) -> httpx.Response:
        """One request → the response, whatever its status. A transport failure is a
        `ThunderError` without status, named by its type: `str()` of an httpx error can
        be EMPTY, and "Thunder API: " alone sends nobody anywhere."""
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            return await self._client.request(method, self._base + path, json=body,
                                              headers=headers, timeout=_TIMEOUT)
        except httpx.HTTPError as e:
            raise thunder.ThunderError(
                f"{method} {path}: {type(e).__name__}: {e}"[:_ERR_MAX], None) from e

    def _redact(self, text: str) -> str:
        """An error body goes into the panel log and the fault log; an API that echoes
        the request (Authorization included) must not put the token there."""
        return text.replace(self._token, "***") if self._token else text

    def _check(self, r: httpx.Response) -> httpx.Response:
        if not 200 <= r.status_code < 300:
            raise thunder.ThunderError(
                self._redact(r.text or f"HTTP {r.status_code}")[:_ERR_MAX], r.status_code)
        return r

    def _json(self, r: httpx.Response):
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError as e:
            raise thunder.ThunderError(f"invalid JSON: {self._redact(r.text[:200])}",
                                       r.status_code) from e

    async def _call(self, method: str, path: str, body=None):
        return self._json(self._check(await self._req(method, path, body)))

    async def _by_id(self, method: str, item: dict, suffix: str, body=None,
                     gone_ok: bool = False) -> Optional[httpx.Response]:
        """`/instances/{id}/<suffix>` with the UUID first, the index only on a 404
        (ledger Ruling 11; the openapi names the parameter "Instance ID (index)" for
        modify/ports but plain "Instance ID" for delete, so which form each accepts is
        verified live). UUID first because Thunder REUSES small indices: a stale index
        can name somebody else's instance, a uuid never can. `gone_ok`: 404 on every
        form means the instance no longer exists — the goal of a delete, so not an
        error there. The index is still only safe while `item` came from a fresh list
        matched by uuid (Ruling 10) — never pass an index taken from the state alone."""
        ids = ["" if item.get(k) is None else str(item.get(k)) for k in ("uuid", "index")]
        ids = [i for n, i in enumerate(ids) if i and i not in ids[:n]]
        if not ids:
            raise thunder.ThunderError("instance has neither index nor uuid", None)
        r = None
        for ident in ids:
            r = await self._req(method, f"/instances/{_q(ident)}/{suffix}", body)
            if r.status_code != 404:
                return self._check(r)
        if gone_ok:
            return None
        return self._check(r)

    # instances
    async def list_instances(self) -> list[dict]:
        return thunder.parse_instances(await self._call("GET", "/instances/list"))

    async def create(self, body: dict) -> dict:
        """→ `{"index": str, "uuid": str}`. `identifier` is an int in the API; the
        controller keeps every id as a string."""
        d = await self._call("POST", "/instances/create", body)
        d = d if isinstance(d, dict) else {}
        ident, uuid = d.get("identifier"), d.get("uuid")
        out = {"index": "" if ident is None else str(ident), "uuid": str(uuid or "")}
        if not out["index"] and not out["uuid"]:
            # the instance may exist and bill anyway — the orphan list is where it shows
            raise thunder.ThunderError(f"create answered without identifier/uuid: {d}"
                                       [:_ERR_MAX], None)
        return out

    async def delete(self, item: dict) -> None:
        await self._by_id("POST", item, "delete", gone_ok=True)

    async def modify(self, item: dict, body: dict) -> None:
        await self._by_id("POST", item, "modify", body)

    async def remove_ports(self, item: dict, ports: list[int]) -> None:
        await self._by_id("PATCH", item, "ports", {"remove_ports": [int(p) for p in ports]})

    # snapshots
    async def snapshots(self) -> list[dict]:
        return thunder.parse_snapshots(await self._call("GET", "/snapshots/list"))

    async def create_snapshot(self, item: dict, name: str) -> str:
        """→ the new snapshot's id. `instanceId` is a STRING in the openapi
        (`CreateSnapshotRequest`) — an int would be a 400 on every stop — and carries
        the index, as the instance endpoints' "(index)" parameters do; the uuid only
        when no index is known. To verify live."""
        idx, uuid = item.get("index"), item.get("uuid")
        inst = str(idx) if idx is not None and str(idx) != "" else str(uuid or "")
        if not inst:
            raise thunder.ThunderError("instance has neither index nor uuid", None)
        d = await self._call("POST", "/snapshots/create", {"instanceId": inst, "name": name})
        sid = str((d or {}).get("id") or "") if isinstance(d, dict) else ""
        if not sid:
            raise thunder.ThunderError(f"snapshot create answered without id: {d}"[:_ERR_MAX],
                                       None)
        return sid

    async def delete_snapshot(self, sid: str) -> None:
        """404 = already gone → fine (rotation is re-run after a restart)."""
        if not sid:
            raise thunder.ThunderError("empty snapshot id", None)
        r = await self._req("DELETE", f"/snapshots/{_q(sid)}")
        if r.status_code != 404:
            self._check(r)

    # public price lists, cached
    async def _cached(self, key: str, path: str):
        hit = self._cache.get(key)
        now = self._clock()
        if hit is not None and now - hit[0] < _PRICE_TTL_S:
            return hit[1]
        val = await self._call("GET", path)
        self._cache[key] = (now, val)
        return val

    async def pricing(self) -> dict:
        return await self._cached("pricing", "/v2/pricing")

    async def specs(self) -> dict:
        return await self._cached("specs", "/v2/specs")

    def cached(self, key: str):
        """The last fetched `pricing`/`specs` body (any age) or None — for the sync
        `view()`, which must not wait on the network."""
        hit = self._cache.get(key)
        return hit[1] if hit is not None else None


# ── state / dependencies ─────────────────────────────────────────────────────

@dataclass
class State:
    phase: str = "off"
    failed_phase: str = ""
    error: str = ""
    index: str = ""
    uuid: str = ""
    ip: str = ""
    port: int = 0
    started_at: float = 0.0
    disk_gb: int = 0
    base_bytes: int = 0
    snapshot_id: str = ""           # the snapshot the running instance was started from
    pending_snapshot: str = ""      # id of the snapshot taken at the last stop, until READY/FAILED
    manifests: dict = field(default_factory=dict)   # snapshot id → model manifest
    log: list = field(default_factory=list)         # not persisted
    transfers: dict = field(default_factory=dict)   # not persisted


_VOLATILE = ("log", "transfers")
LOAD_FAILED = "load"            # failed_phase marker: the stored record could not be read


def _coerce(default, v):
    """A persisted value into the field's type, else the default: a state written by
    another version must never keep the controller (and its instance) from coming up."""
    try:
        if isinstance(default, bool):
            return bool(v)
        if isinstance(default, int):
            return int(float(v))
        if isinstance(default, float):
            return float(v)
        if isinstance(default, str):
            return "" if v is None else str(v)
        if isinstance(default, dict):
            return dict(v) if isinstance(v, dict) else {}
    except (TypeError, ValueError):
        pass
    return default


def state_from(d: Optional[dict]) -> State:
    s = State()
    if not isinstance(d, dict):
        return s
    for f in dataclasses.fields(State):
        if f.name in _VOLATILE or f.name not in d:
            continue
        setattr(s, f.name, _coerce(getattr(s, f.name), d[f.name]))
    if s.phase not in PHASES:
        # never "off": that would forget an instance that may still be billing
        s.failed_phase, s.phase = s.phase, "failed"
        s.error = s.error or f"unknown persisted phase {s.failed_phase!r}"
    return s


def _default_log(msg: str) -> None:
    pass


@dataclass
class Deps:
    """What the controller needs from the gateway, injected by `main`. Every field a
    later phase adds carries a default, so a `Deps(...)` built for an earlier one keeps
    working."""
    client_factory: Callable[[], httpx.AsyncClient]
    load_state: Callable[[str], Optional[dict]]
    save_state: Callable[[str, dict], None]
    set_enabled: Callable[[str, bool], bool]
    begin_drain: Callable[[str], bool]
    inflight: Callable[[str], int]
    is_draining: Callable[[str], bool]
    note_fault: Callable[..., None]
    datadir: str
    probe_comfy: Callable[[str], Awaitable[bool]]
    bootstrap_script: Callable[[], bytes]
    log: Callable[[str], None] = _default_log
    now: Callable[[], float] = time.time
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    ssh: Callable[..., Awaitable[tuple]] = sshrun.run
    spawn: Optional[Callable[..., Awaitable[Any]]] = None     # None = create_subprocess_exec
    known_uuids: Callable[[], set] = field(default=lambda: set())


# ── controller ───────────────────────────────────────────────────────────────

class Controller:
    """Lifecycle of one Thunder-backed ComfyUI backend. The start and stop paths
    build on `_set_phase` (persist on every change) and `_log` (the panel's ring)."""

    def __init__(self, backend: dict, deps: Deps):
        self.backend = backend
        self.deps = deps
        self._api: Optional[ThunderApi] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._tunnel = None
        self._snaps: Optional[list[dict]] = None      # last /snapshots/list, for view()
        self._persist_blocked = False
        self.state = State()
        try:
            loaded = deps.load_state(self.name)
        except Exception as e:
            self._load_failed(f"state load failed: {e!r}")
        else:
            if loaded is None:
                pass                    # no record: this backend never had an instance
            elif not isinstance(loaded, dict):
                self._load_failed(f"state load failed: stored entry is a "
                                  f"{type(loaded).__name__}, not a dict")
            else:
                self.state = state_from(loaded)

    def _load_failed(self, msg: str) -> None:
        """The stored record exists but could not be read. It may name a RUNNING,
        billing instance, so this is not "off": `off` would let start() create a second
        instance, and its first `_persist` would overwrite the intact record with an
        empty uuid — the first instance then bills with nobody knowing it. Instead:
        `failed` with the `LOAD_FAILED` marker, saving suppressed and start() refused
        until `resume()` has reconciled with `/instances/list` and calls
        `_unblock_persist()`."""
        self.state = State(phase="failed", failed_phase=LOAD_FAILED, error=msg)
        self._persist_blocked = True
        self._log(msg)

    @property
    def persist_blocked(self) -> bool:
        return self._persist_blocked

    def _unblock_persist(self) -> None:
        """Allow saving again. Only for `resume()`, once it has established from
        `/instances/list` which instance (if any) this backend owns — from then on the
        in-memory state is the truth and may overwrite the unreadable record."""
        if self._persist_blocked:
            self._persist_blocked = False
            self._log("state reconciled — saving resumed")

    # identity / config
    @property
    def name(self) -> str:
        return str(self.backend["name"])

    @property
    def bid(self) -> str:
        # = main.backend_id; recomputed, not imported (thunderctl never imports main)
        return f'{self.backend.get("type", "comfyui")}:{self.name}'

    @property
    def cfg(self) -> dict:
        return self.backend.get("thunder") or {}

    @property
    def lport(self) -> int:
        return int(self.cfg.get("local_port") or 18188)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.lport}"

    @property
    def api(self) -> ThunderApi:
        """One client for the controller's lifetime; a new token (backend saved in the
        console) gets a new ThunderApi on the same client."""
        token = str(self.backend.get("api_key") or "")
        if self._api is None or self._api._token != token:
            if self._client is None:
                self._client = self.deps.client_factory()
            self._api = ThunderApi(self._client, token)
        return self._api

    async def aclose(self) -> None:
        """Gateway shutdown: end the tunnel process and the HTTP client. The INSTANCE is
        not touched — it keeps running and `resume()` picks it up again."""
        await self._stop_tunnel()
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client, self._api = None, None

    # persistence / log
    def _persist(self) -> None:
        if self._persist_blocked:
            # never overwrite a record we could not read (see _load_failed)
            self._log("state not saved: stored record unread, waiting for reconcile")
            return
        d = asdict(self.state)
        for k in _VOLATILE:
            d.pop(k, None)
        try:
            self.deps.save_state(self.name, d)
        except Exception as e:
            # logged, not raised: a store hiccup must not abort a create half-way and
            # leave the instance running with nobody following it up
            self._log(f"state save failed: {e!r}")

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.deps.now()))
        line = f"{stamp} {msg}"
        self.state.log.append(line)
        if len(self.state.log) > _LOG_MAX:
            del self.state.log[:-_LOG_MAX]
        try:
            self.deps.log(f"[thunder {self.name}] {msg}")
        except Exception:
            pass

    def _set_phase(self, p: str, error: str = "") -> None:
        """Enter phase `p` and persist. `failed` remembers the phase it interrupted
        (what a retry or the panel needs to know); any other phase clears the error."""
        if p not in PHASES:
            raise ValueError(f"unknown phase {p!r}")
        old = self.state.phase
        if p == "failed":
            if old != "failed":
                self.state.failed_phase = old
            self.state.error = error or self.state.error
        else:
            self.state.failed_phase = ""
            self.state.error = error
        self.state.phase = p
        self._log(f"phase {old} → {p}" + (f": {error}" if error else ""))
        self._persist()

    # tunnel
    def _key_path(self) -> str:
        return os.path.join(self.deps.datadir, "thunder.key")

    def _known_hosts_path(self, uuid: str) -> str:
        """Per instance: IP, port AND host key change with every instance, so one shared
        file would either refuse the next instance or have to trust any key."""
        return os.path.join(self.deps.datadir, "thunder-known_hosts", sshrun.safe_rel(uuid))

    def _tunnel_argv(self) -> list[str]:
        """Called by the Supervisor per spawn, so it follows the CURRENT instance."""
        s = self.state
        if not (s.uuid and s.ip and s.port):
            raise RuntimeError("no instance to tunnel to")
        return sshrun.tunnel_argv(self._key_path(), self._known_hosts_path(s.uuid),
                                  f"{_SSH_USER}@{s.ip}", s.port, self.lport, _COMFY_PORT)

    def _tunnel_factory(self):
        """The tunnel Supervisor seam (tests replace it per instance)."""
        return sshrun.Supervisor(self._tunnel_argv, lambda m: self._log(m),
                                 spawn=self.deps.spawn or asyncio.create_subprocess_exec)

    async def _stop_tunnel(self) -> None:
        t, self._tunnel = self._tunnel, None
        if t is not None:
            await t.stop()

    # prices / snapshots for the panel
    async def refresh_prices(self) -> None:
        """Fetch (or reuse, 1 h) pricing + specs. Display only: a failure is logged."""
        try:
            await self.api.pricing()
            await self.api.specs()
        except thunder.ThunderError as e:
            self._log(f"price list unavailable ({e.status or 'transport'}): {e}")

    async def refresh_snapshots(self) -> Optional[list[dict]]:
        try:
            self._snaps = await self.api.snapshots()
        except thunder.ThunderError as e:
            self._log(f"snapshot list unavailable ({e.status or 'transport'}): {e}")
        return self._snaps

    def cost_per_h(self) -> Optional[float]:
        api = self._api
        pricing = api.cached("pricing") if api is not None else None
        specs = api.cached("specs") if api is not None else None
        if not isinstance(pricing, dict):
            return None
        table = pricing.get("pricing") if isinstance(pricing.get("pricing"), dict) else pricing
        cfg = self.cfg
        gpu, n = str(cfg.get("gpu_type") or ""), int(cfg.get("num_gpus") or 1)
        return thunder.hourly_cost(table, gpu, n, int(cfg.get("vcpus") or 0),
                                   self.state.disk_gb, thunder.spec_for(specs, gpu, n))

    def _snapshot_view(self) -> dict:
        s = self.state
        row = next((x for x in (self._snaps or []) if x.get("id") == s.snapshot_id), None)
        gb = (row or {}).get("min_disk_gb") or None
        monthly = None
        api = self._api
        pricing = api.cached("pricing") if api is not None else None
        if gb and isinstance(pricing, dict):
            table = pricing.get("pricing") if isinstance(pricing.get("pricing"), dict) else pricing
            monthly = thunder.snapshot_monthly(table, gb)
        return {"id": s.snapshot_id, "pending": s.pending_snapshot,
                "name": (row or {}).get("name", ""), "status": (row or {}).get("status", ""),
                # Thunder reports no snapshot size; its minimum restore disk is the
                # closest figure it gives, so $/month is an upper bound
                "gb": gb, "monthly": monthly}

    def view(self) -> dict:
        s = self.state
        running = s.phase != "off" and s.started_at > 0
        uptime = max(0, int(self.deps.now() - s.started_at)) if running else 0
        cph = self.cost_per_h()
        return {"name": self.name, "phase": s.phase, "error": s.error,
                "failed_phase": s.failed_phase, "index": s.index, "uuid": s.uuid,
                "ip": s.ip, "port": s.port, "started_at": s.started_at,
                "uptime_s": uptime, "disk_gb": s.disk_gb, "cost_per_h": cph,
                "session_cost": (cph * uptime / 3600) if (cph is not None and running) else None,
                "snapshot": self._snapshot_view(), "log": list(s.log[-_LOG_MAX:]),
                "transfers": dict(s.transfers), "persist_blocked": self._persist_blocked}

    # lifecycle — the start and stop paths come next; until then they refuse loudly
    def _refuse_if_unreconciled(self) -> None:
        if self._persist_blocked:
            raise RuntimeError(f"state not loaded ({self.state.error or 'unreadable'}) — "
                               "an instance may still be running; resume first")

    async def start(self) -> None:
        self._refuse_if_unreconciled()
        raise NotImplementedError("Thunder start path not implemented yet")

    async def stop(self) -> None:
        raise NotImplementedError("Thunder stop path not implemented yet")

    async def restart_comfy(self) -> None:
        raise NotImplementedError("Thunder ComfyUI restart not implemented yet")

    async def resume(self) -> None:
        raise NotImplementedError("Thunder resume not implemented yet")

    async def run_forever(self) -> None:
        raise NotImplementedError("Thunder background loop not implemented yet")
