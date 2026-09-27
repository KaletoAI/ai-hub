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

The start path (`start()`, spec "Start" 0–6) has two rules that hold whatever fails:
the instance's uuid is persisted BEFORE the first wait (a restart during the up to
half an hour to RUNNING must not forget it), and `bootstrapping`/`starting` are never
entered while Thunder forwards a public HTTP port (`_ensure_ports_closed`: its port
forwarding has no auth, and ComfyUI behind it is code execution for anyone). A failure
before `create` ends in `off`; after it in `failed(<phase>)` with the instance KEPT for
diagnosis. The bootstrap's verdict is read from its `GW:` lines by tag
(`parse_bootstrap`/`bootstrap_verdict`, ledger Ruling 9).

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
import math
import os
import re
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

# start path timing
_CREATE_POLL_S = 10             # /instances/list while creating/restoring
_CREATE_BASE_S = 15 * 60        # create timeout = base + per started 100 GB of disk:
_RESTORE_PER_100GB_S = 8 * 60   #   Thunder's docs: a restore takes up to 8 min / 100 GB
_SSH_READY_S = 5 * 60           # a RUNNING instance whose sshd does not answer by then
_SSH_PROBE_S = 5
_BOOTSTRAP_S = 3 * 3600         # venv + torch + node packs + CUDA extension builds
_COMFY_READY_S = 10 * 60        # first start loads custom nodes (3D packs import slowly)
_COMFY_PROBE_S = 3
_STDERR_LOG_LINES = 20          # stderr tail logged for a failed remote step

_COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")
# The loop script is flock-guarded, so starting it while it runs is a no-op.
_START_CMD = "setsid nohup ~/start-comfy.sh >/dev/null 2>&1 < /dev/null &"
# Kill ComfyUI (the loop restarts it 2 s later) and start the loop in case it is not
# alive (a failed bootstrap, a killed wrapper). The bracket keeps the pattern from
# matching the remote `bash -c "<this command>"` itself — pkill spares only its own
# process, and a pattern that hit the shell would kill the restart half-way. It
# matches the loop's `<venv python> main.py --listen 127.0.0.1 --port 8188 …`, whatever
# venv the bootstrap picked (`ComfyUI/main.py` never appears in that command line).
_RESTART_CMD = ("pkill -f '[m]ain[.]py --listen 127[.]0[.]0[.]1 --port "
                f"{_COMFY_PORT}'; " + _START_CMD)


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
    # what the last bootstrap reported: models the TEMPLATE brought (rel → bytes; paid
    # for in every snapshot until deleted) and custom_nodes packs it brought
    bootstrap_unknown: dict = field(default_factory=dict)
    bootstrap_template_nodes: list = field(default_factory=list)
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
        if isinstance(default, list):
            return list(v) if isinstance(v, list) else []
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
    keygen: Callable[[str], Awaitable[str]] = sshrun.keygen   # path → public key
    default_nodes: Callable[[], str] = field(default=lambda: "")  # ops/thunder-nodes.default.txt


# ── bootstrap output ─────────────────────────────────────────────────────────

def parse_bootstrap(out: str) -> dict:
    """The `GW:` lines of `ops/thunder-bootstrap.sh`'s stdout (its header documents
    them). Matched by TAG — the first whitespace-separated token of a line that starts
    with `GW:` — never by position: the script prints other lines in between, and a
    later version may add tags (ledger Ruling 9). An UNKNOWN_MODEL path that could
    leave the model tree (`safe_rel`) is dropped here — the panel later offers to
    DELETE these paths."""
    r = {"phase": "", "smoke": None, "done": False, "node_fails": [], "unknown": {},
         "template_nodes": [], "bad": []}
    for line in (out or "").splitlines():
        if not line.startswith("GW:"):
            continue
        tag, _, rest = line.partition(" ")
        rest = rest.strip("\r")
        if tag == "GW:PHASE":
            r["phase"] = rest.strip()
        elif tag == "GW:SMOKE":
            r["smoke"] = rest.strip()
        elif tag == "GW:DONE":
            r["done"] = True
        elif tag == "GW:NODE_FAIL":
            r["node_fails"].append(rest.strip())
        elif tag == "GW:TEMPLATE_NODE":
            if rest.strip():
                r["template_nodes"].append(rest.strip())
        elif tag == "GW:UNKNOWN_MODEL":
            rel, _, size = rest.rpartition("\t")   # no tab → rel "" → refused
            try:
                n = int(size)
                r["unknown"][sshrun.safe_rel(rel)] = n
            except ValueError:
                r["bad"].append(line)
    return r


def bootstrap_verdict(rc: int, rep: dict, err: str) -> str:
    """"" when the bootstrap succeeded, else why not. Success needs ALL of: rc 0,
    `GW:SMOKE ok`, `GW:DONE` and no `GW:NODE_FAIL` at all (Ruling 9: a pack that did not
    install fails workflows later with a plausible-looking error)."""
    why = []
    if rep["node_fails"]:
        why.append("node packs failed: " + "; ".join(rep["node_fails"]))
    smoke = rep["smoke"]
    if smoke is not None and smoke != "ok":
        why.append("smoke test " + smoke)
    where = f" in phase {rep['phase']}" if rep["phase"] else ""
    if rc == 124:
        why.append(f"timed out after {_BOOTSTRAP_S // 3600} h{where}")
    elif rc == 3 and smoke is None:
        why.append("smoke test failed")
    elif rc not in (0, 3):
        last = next((ln.strip() for ln in reversed((err or "").splitlines()) if ln.strip()), "")
        why.append(f"bootstrap exited rc {rc}{where}" + (f": {last}" if last else ""))
    if not why and not (smoke == "ok" and rep["done"]):
        why.append(f"bootstrap ended without GW:SMOKE ok / GW:DONE{where}")
    return "; ".join(why)


def _tail(b: bytes, n: int = _STDERR_LOG_LINES) -> list[str]:
    lines = [ln for ln in (b or b"").decode("utf-8", "replace").splitlines() if ln.strip()]
    return lines[-n:]


def _errtext(e: BaseException) -> str:
    """`str()` of a TimeoutError or an httpx error can be empty."""
    return str(e) or type(e).__name__


class _PreCreate(Exception):
    """A start failed before any instance existed → `off`, not `failed`."""


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
        self._op: Optional[str] = None                 # start/restart in flight
        self._created_template = ""                    # what the last create asked for
        self._nodes_text = ""                          # node list of the pending bootstrap
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

    def _reset_known_hosts(self, uuid: str) -> str:
        """A NEW instance: its host key is new too, so a file left for this uuid (a
        retry, a reused uuid) must go before the first ssh — accept-new then records
        the key this instance presents. The directory is 0700: the files say which
        hosts this gateway talks to."""
        path = self._known_hosts_path(uuid)
        d = os.path.dirname(path)
        os.makedirs(d, mode=0o700, exist_ok=True)
        os.chmod(d, 0o700)
        try:
            os.remove(path)
            self._log(f"known_hosts for {uuid} reset")
        except FileNotFoundError:
            pass
        return path

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

    async def _start_tunnel(self) -> None:
        """A fresh Supervisor for the current instance. An old one is stopped (and its
        ssh reaped) first: its process holds the local port, and the new tunnel would
        exit at once on ExitOnForwardFailure."""
        await self._stop_tunnel()
        self._tunnel = self._tunnel_factory()
        self._tunnel.start()

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
                "transfers": dict(s.transfers), "persist_blocked": self._persist_blocked,
                "bootstrap_unknown": dict(s.bootstrap_unknown),
                "bootstrap_template_nodes": list(s.bootstrap_template_nodes)}

    # lifecycle
    def _refuse_if_unreconciled(self) -> None:
        if self._persist_blocked:
            raise RuntimeError(f"state not loaded ({self.state.error or 'unreadable'}) — "
                               "an instance may still be running; resume first")

    def _refuse_if_busy(self) -> None:
        if self._op is not None:
            raise RuntimeError(f"already {self._op}")

    def _fail(self, msg: str) -> None:
        """An instance exists (and bills): `failed(<phase>)`, never `off` — the stop
        path needs the uuid to snapshot and delete it. The fault log keeps it after the
        panel has moved on."""
        self._set_phase("failed", msg)
        try:
            self.deps.note_fault(self.backend, "lifecycle", "error", msg)
        except Exception as e:
            self._log(f"fault log unavailable: {e!r}")

    def _cfg_int(self, key: str, default: int) -> int:
        v = self.cfg.get(key)
        try:
            return int(v) if v is not None and v != "" else default
        except (TypeError, ValueError):
            raise _PreCreate(f"thunder.{key} is not a number: {v!r}")

    def _required_bytes_hint(self) -> int:
        """Bytes the aliases' models need on the new disk. P1: 0 (no model sync); P2
        returns the plan's total so the disk is sized before the create."""
        return 0

    def _node_list(self) -> str:
        """The node list the bootstrap installs: the backend's own, else the default
        list (Ruling 12) — never an empty file, which installs nothing and then fails
        the smoke test with a reason that points nowhere near the cause."""
        nodes = self.cfg.get("nodes")
        if isinstance(nodes, str):
            nodes = nodes.splitlines()
        text = "\n".join(str(x) for x in (nodes or []) if str(x).strip())
        if not text.strip():
            text = str(self.deps.default_nodes() or "")
        if not any(ln.strip() and not ln.strip().startswith("#") for ln in text.splitlines()):
            raise _PreCreate("no custom-node list to bootstrap with (thunder.nodes is "
                             "empty and there is no default list)")
        return text if text.endswith("\n") else text + "\n"

    def _commit(self) -> str:
        c = self.cfg.get("comfy_commit")
        c = "" if c is None else str(c).strip()
        if not _COMMIT_RE.fullmatch(c):
            # the bootstrap exits 2 on anything else — after an instance was paid for
            raise RuntimeError(f"thunder.comfy_commit must be a full 40-hex commit sha, "
                               f"got {c!r}")
        return c

    def _find_ours(self, items: list[dict]) -> Optional[dict]:
        """Our instance in a FRESH list (Ruling 10): by uuid only. Only when create
        answered without a uuid, the item at our index is accepted — and only if
        nobody else owns its uuid and it carries the template we asked for (Thunder
        reuses indices); its uuid is then adopted and persisted."""
        s = self.state
        if s.uuid:
            return next((it for it in items if it.get("uuid") == s.uuid), None)
        if not s.index:
            return None
        it = next((x for x in items if x.get("index") == s.index), None)
        if (it is None or not it.get("uuid") or it["uuid"] in self._known_uuids()
                or it.get("template", "").lower() != self._created_template.lower()):
            return None
        s.uuid = it["uuid"]
        self._log(f"instance at index {s.index} adopted: uuid {s.uuid}")
        self._persist()
        return it

    def _known_uuids(self) -> set:
        try:
            return set(self.deps.known_uuids() or ())
        except Exception:
            return set()

    async def _fresh_item(self) -> dict:
        it = self._find_ours(await self.api.list_instances())
        if it is None:
            raise thunder.ThunderError(
                f"instance {self.state.uuid or self.state.index} is not in /instances/list",
                None)
        if thunder.is_gone_status(it.get("status")):
            raise thunder.ThunderError(f"instance {it.get('uuid')} is {it.get('status')}", None)
        return it

    async def _ensure_ports_closed(self, item: dict) -> dict:
        """Port guard (spec "Start", hard): Thunder's HTTP port forwarding is PUBLIC
        without auth, and ComfyUI behind it means code execution (Manager) and file
        reads (`/view`) for anyone. `item` must come from a fresh list (Ruling 10).
        Open ports are removed and the list read AGAIN — a 200 on the PATCH is not
        proof — and ports still open raise, so `bootstrapping`/`starting` are never
        entered with a public port. → the fresh item."""
        ports = thunder.ports_open(item)
        if not ports:
            return item
        self._log(f"public http ports {ports} open on the instance — removing")
        await self.api.remove_ports(item, ports)
        fresh = await self._fresh_item()
        still = thunder.ports_open(fresh)
        if still:
            raise thunder.ThunderError(
                f"public http ports {still} still open after removing them — "
                "ComfyUI is not started behind a public port", None)
        self._log("public http ports closed")
        return fresh

    def _ssh_argv(self, cmd: str) -> list[str]:
        s = self.state
        return sshrun.exec_argv(self._key_path(), self._known_hosts_path(s.uuid),
                                f"{_SSH_USER}@{s.ip}", s.port, cmd)

    async def _exec(self, cmd: str, stdin: Optional[bytes] = None,
                    timeout: float = 60) -> tuple[int, bytes, bytes]:
        return await self.deps.ssh(self._ssh_argv(cmd), stdin=stdin, timeout=timeout)

    async def _wait_running(self, disk_gb: int) -> dict:
        """Poll until RUNNING with ip + port. Status values are undocumented, so
        anything else is "not yet" — except RESTORING (its own phase) and a gone
        status. A failing list is retried: one 502 must not end a 20-minute wait."""
        limit = _CREATE_BASE_S + _RESTORE_PER_100GB_S * math.ceil(max(1, disk_gb) / 100)
        deadline = self.deps.now() + limit
        status, last_err = "not listed", ""
        while True:
            try:
                it = self._find_ours(await self.api.list_instances())
            except thunder.ThunderError as e:
                it, last_err = None, str(e)
                self._log(f"instance list failed while waiting: {e}")
            if it is not None:
                status = it.get("status") or "?"
                if thunder.is_gone_status(status):
                    raise thunder.ThunderError(f"instance {it.get('uuid')} became {status} "
                                               "while starting", None)
                if status == "RESTORING" and self.state.phase != "restoring":
                    self._set_phase("restoring")
                if thunder.is_running(status) and it.get("ip") and it.get("port"):
                    return it
            if self.deps.now() >= deadline:
                raise TimeoutError(f"instance not RUNNING after {limit // 60} min "
                                   f"(status {status})" +
                                   (f"; last list error: {last_err}" if last_err else ""))
            await self.deps.sleep(_CREATE_POLL_S)

    async def _wait_ssh(self) -> None:
        deadline = self.deps.now() + _SSH_READY_S
        while True:
            rc, _, err = await self._exec("true", timeout=30)
            if rc == 0:
                self._log("ssh reachable")
                return
            last = (_tail(err, 1) or [f"rc {rc}"])[0]
            if self.deps.now() >= deadline:
                raise TimeoutError(f"ssh not reachable after {_SSH_READY_S // 60} min: {last}")
            await self.deps.sleep(_SSH_PROBE_S)

    async def _bootstrap(self) -> None:
        s = self.state
        rc, _, err = await self._exec("cat > ~/.gw-nodes.txt",
                                      stdin=self._nodes_text.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"node list upload failed (rc {rc}): "
                               + " | ".join(_tail(err, 3)))
        self._log(f"bootstrap: ComfyUI {self._commit()[:12]}, node list uploaded")
        rc, out, err = await self._exec(f"bash -s -- {sshrun.q(self._commit())}",
                                        stdin=self.deps.bootstrap_script(),
                                        timeout=_BOOTSTRAP_S)
        text = (out or b"").decode("utf-8", "replace")
        for line in text.splitlines():
            if line.strip():
                self._log(line)
        rep = parse_bootstrap(text)
        for line in rep["bad"]:
            self._log(f"bootstrap: unreadable report line ignored: {line!r}")
        s.bootstrap_unknown = dict(rep["unknown"])
        s.bootstrap_template_nodes = list(rep["template_nodes"])
        self._persist()
        why = bootstrap_verdict(rc, rep, (err or b"").decode("utf-8", "replace"))
        if why:
            for line in _tail(err):
                self._log(f"stderr: {line}")
            raise RuntimeError(why)
        if s.bootstrap_unknown:
            gb = sum(s.bootstrap_unknown.values()) / 1024 ** 3
            self._log(f"bootstrap: {len(s.bootstrap_unknown)} template model file(s), "
                      f"{gb:.1f} GB — delete them before the first stop or every "
                      "snapshot carries them")
        self._log("bootstrap done")

    async def _wait_comfy(self, settle: bool = False) -> None:
        """Probe ComfyUI through the tunnel every 3 s until it answers. `settle`: wait
        one interval first — right after a pkill the old process may still answer."""
        deadline = self.deps.now() + _COMFY_READY_S
        if settle:
            await self.deps.sleep(_COMFY_PROBE_S)
        while True:
            try:
                ok = bool(await self.deps.probe_comfy(self.url))
            except Exception:
                ok = False              # tunnel not up yet, ComfyUI still importing
            if ok:
                self._log("ComfyUI answers")
                return
            if self.deps.now() >= deadline:
                raise TimeoutError(f"ComfyUI did not answer on {self.url} within "
                                   f"{_COMFY_READY_S // 60} min (see ~/comfy.log)")
            await self.deps.sleep(_COMFY_PROBE_S)

    async def _start_comfy(self, cmd: str, settle: bool = False) -> None:
        rc, _, err = await self._exec(cmd, timeout=60)
        if rc != 0:
            raise RuntimeError(f"starting ComfyUI failed (rc {rc}): "
                               + " | ".join(_tail(err, 3)))
        await self._wait_comfy(settle)

    async def start(self) -> None:
        """Spec "Start" 0–6 (P1: no model sync — `starting` → `ready`).

        Refusals (unreconciled state, an instance already known, a bad commit, a
        start in flight) RAISE before anything happens. Everything else ends in the
        state: a failure before `create` → `off` with the reason (nothing bills), a
        failure after it → `failed(<phase>)` with the instance KEPT (diagnosis; the
        stop path removes it). The uuid is persisted before the first wait, so a
        gateway restart during the up to ~30 min to RUNNING still knows the instance."""
        self._refuse_if_unreconciled()
        self._refuse_if_busy()
        s = self.state
        if s.phase != "off" and not (s.phase == "failed" and not s.uuid and not s.index):
            # `failed` with an index but no uuid still names an instance (Ruling 10)
            raise RuntimeError(f"already {s.phase}"
                               + (f" (instance {s.uuid or s.index})" if s.uuid or s.index else ""))
        self._commit()
        self._op = "starting"
        try:
            await self._start()
        finally:
            self._op = None

    async def _start(self) -> None:
        s = self.state
        try:
            self.deps.set_enabled(self.bid, True)
        except Exception as e:
            self._log(f"enabling the backend failed: {e!r}")
        try:
            created = await self._create()
        except Exception as e:
            msg = _errtext(e)
            if isinstance(e, thunder.ThunderError) and e.status is None:
                msg += " (an instance may exist anyway — check the orphan list)"
            self._set_phase("off", f"start failed: {msg}")
            try:
                self.deps.set_enabled(self.bid, False)    # off = disabled (spec "Stop" 1)
            except Exception as e2:
                self._log(f"disabling the backend failed: {e2!r}")
            return
        needs_bootstrap = created
        try:
            item = await self._wait_running(s.disk_gb)
            s.ip, s.port = str(item["ip"]), int(item["port"])
            self._persist()
            item = await self._ensure_ports_closed(item)
            self._set_phase("connecting")
            self._reset_known_hosts(s.uuid)
            await self._start_tunnel()
            await self._wait_ssh()
            if needs_bootstrap:
                self._set_phase("bootstrapping")
                await self._bootstrap()
            await self._ensure_ports_closed(await self._fresh_item())
            self._set_phase("starting")
            await self._start_comfy(_START_CMD)
            self._set_phase("ready")
        except Exception as e:
            self._fail(_errtext(e))

    async def _create(self) -> bool:
        """Steps 1–3 up to the create: template, disk, key, `POST /instances/create`;
        index/uuid persisted with phase `creating`. → whether the instance needs the
        bootstrap (no READY snapshot of ours to restore from)."""
        s, cfg = self.state, self.cfg
        for k in ("gpu_type", "vcpus"):
            if not cfg.get(k):
                raise _PreCreate(f"thunder.{k} is not set")
        num_gpus = self._cfg_int("num_gpus", 1)
        snaps = await self.api.snapshots()
        self._snaps = snaps
        snap = thunder.newest_ready(snaps, self.name)
        if snap is not None:
            template, needs_bootstrap = snap["name"], False
            self._nodes_text = ""
        else:
            template = str(cfg.get("bootstrap_template") or "comfy-ui")
            needs_bootstrap = True
            self._nodes_text = self._node_list()
        spec = thunder.spec_for(await self.api.specs(), str(cfg["gpu_type"]), num_gpus)
        storage = (spec or {}).get("storageGB") if isinstance((spec or {}).get("storageGB"), dict) else {}
        if spec is None:
            self._log(f"no /v2/specs entry for {cfg['gpu_type']} x{num_gpus} — disk "
                      "limits unknown, Thunder decides")
        disk_gb = thunder.choose_disk_gb(
            required_bytes=self._required_bytes_hint(), base_bytes=s.base_bytes,
            reserve_gb=self._cfg_int("reserve_gb", 20),
            snapshot_min_gb=(snap or {}).get("min_disk_gb") or 0,
            spec_min=int(storage.get("min") or 0), spec_max=int(storage.get("max") or 0),
            num_gpus=num_gpus)
        pub = await self.deps.keygen(self._key_path())
        self._log(f"creating instance: template {template}, disk {disk_gb} GB"
                  + ("" if snap else " (first start: bootstrap follows)"))
        self._created_template = template
        created = await self.api.create(thunder.create_body(cfg, template, disk_gb, pub))
        # from here on an instance exists and bills: persist it BEFORE any wait
        s.index, s.uuid = created["index"], created["uuid"]
        s.ip, s.port = "", 0
        s.disk_gb = disk_gb
        s.started_at = self.deps.now()
        s.snapshot_id = snap["id"] if snap else ""
        if needs_bootstrap:
            s.bootstrap_unknown, s.bootstrap_template_nodes = {}, []
        self._set_phase("creating")
        return needs_bootstrap

    async def restart_comfy(self) -> None:
        """Restart ComfyUI on the running instance (panel button; spec "Start" 5 says
        `starting` is repeatable from `ready`). Also the way out of a failed bootstrap
        or start once the cause is fixed on the box. Port guard first, like any entry
        into `starting`."""
        self._refuse_if_unreconciled()
        self._refuse_if_busy()
        s = self.state
        if s.phase not in ("ready", "failed") or not (s.uuid and s.ip and s.port):
            raise RuntimeError(f"no running instance to restart ComfyUI on ({s.phase})")
        self._op = "restarting ComfyUI"
        try:
            try:
                await self._ensure_ports_closed(await self._fresh_item())
                if self._tunnel is None:
                    await self._start_tunnel()
                self._set_phase("starting")
                await self._start_comfy(_RESTART_CMD, settle=True)
                self._set_phase("ready")
            except Exception as e:
                self._fail(_errtext(e))
        finally:
            self._op = None

    async def stop(self) -> None:
        raise NotImplementedError("Thunder stop path not implemented yet")

    async def resume(self) -> None:
        raise NotImplementedError("Thunder resume not implemented yet")

    async def run_forever(self) -> None:
        raise NotImplementedError("Thunder background loop not implemented yet")
