"""Managed-host lifecycle controller: one object per MANAGED HOST — a machine a
provider (Thunder Compute today, `hostapi.PROVIDERS`) creates, lists and deletes — with
the backends ATTACHED to it as its services. It talks to the provider's REST API,
supervises the ONE ssh tunnel every service is reached through (a ControlMaster with a
forward per service), and remembers WHICH machine it owns.

That last part is what everything else rests on. An instance bills by the hour
whether or not the gateway remembers it, and Thunder has no "stop" — so a gateway
restart that forgot the instance's uuid would leave it running (and billing) with
nobody to snapshot or delete it. The state is therefore persisted on EVERY phase
change (store setting `host_state`, one entry per HOST name, written by `main` through
the injected `save_state`) and read back by the constructor; `resume()` then reconciles
it with `/instances/list`. The log ring and the live transfer table are NOT persisted:
both describe the running process, and a stale "downloading 40 %" after a restart would
be a lie.

Host vs. service (spec 2026-09-29, R-K1). Everything about the MACHINE is the host's:
the phase, the instance ids, the snapshots (named, owned and rotated by the host name,
R-W4), the ssh key and known_hosts (per provider kind), the ControlMaster socket, the
bootstrap state, and the faults of creating, snapshotting and deleting it (booked on the
pseudo backend `{"name": <host>, "type": "managed-host"}`). Everything about ONE
service is that backend's: `enabled`, its drain, its forward (`local_port` →
`remote_port` on the VM), its start/probe status in `State.services`, and the faults of
starting it or syncing its models. Start enables EVERY attached backend before the
create; the stop drains them all and `off` disables those attached at that moment. A
forward that changes while the host runs goes through the master's control socket
(`sshrun.control`), never a tunnel restart that would cut another service's stream
(R-W1); a master that died is respawned with every current forward. Only a ComfyUI
service is started, probed and model-synced in this version (one per host).

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
diagnosis. The setup is two scripts (R-W3): the HOST bootstrap (`ops/host-bootstrap.sh`:
the template's autostart off, the models and node packs it brought reported, flock/uv)
runs on the first start of EVERY host, the ComfyUI bootstrap (`ops/thunder-bootstrap.sh`)
after it and only when a ComfyUI service is attached — or later, when one is attached
to a running host (`_ensure_comfy_bootstrap`). Each keeps its own done-flag
(`host_bootstrapped`, `bootstrap_incomplete`) that a snapshot inherits, so a start from
a snapshot taken before a script finished runs that script again. Both verdicts are
read from the `GW:` lines by tag (`parse_bootstrap`/`bootstrap_verdict`, ledger
Ruling 9).

The stop path (`stop()`, spec "Stop") is draining → pruning → snapshotting → deleting →
`off`, each step re-entrant: the snapshot NAME is persisted before its POST and looked
up before creating (a restart in between never takes a second one), and the delete is
only done once a fresh `/instances/list` no longer shows the instance — one that stays
is `failed(deleting)`, never `off`, because it bills. Only a confirmed-gone instance
clears the ids and `started_at` (the session cost stops there). A stop() aborts a start
in flight and stops from the phase it reached (Ruling 13); a snapshot taken before the
bootstrap finished is marked (`incomplete_snapshots`) so a start from it bootstraps
again (the host bootstrap: `host_incomplete_snapshots`; a snapshot of a host that never
had a ComfyUI service: `no_comfy_snapshots`, so attaching one later bootstraps it
without calling the snapshot broken). `watch_snapshots()` settles the pending snapshot
(READY → rotation, FAILED → fault, the last READY one stays the template) and `resume()` reconciles the persisted
state with the instance list after a gateway restart. Every mutating call uses an item
found in a FRESH list by `_find_ours` (Ruling 10); `orphans()` only ever displays.

The model sync (`sync_once`, spec "Controller"): the destination index is rebuilt by
`find` over both roots on every plan (the manifest `~/.gw-modelsync.json` says only
where a file CAME from, never that it exists), `modelsync.plan` decides, and
`plan`/`ready_aliases` are written there and nowhere else — routing (Task 12) asks
`is_alias_ready`, which is False before the first plan and outside `syncing|ready`. What
only the controller knows joins an alias's `blocked` list: a LAN source that is not
usable ("waiting for LAN source (not configured)" without a pinned host key, "(unreachable:
…)" when its last `list` failed), a transfer that gave up after three attempts with
backoff (also a fault), a disk that cannot grow enough. A HEAD learns a URL file's size
first (disk sizing, the size check). URL files download ON the instance (≤ 3 curls,
`setsid`, a lockfile holding the curl's pid, options incl. the Hugging Face token — only
for a Hugging Face host — on stdin, never argv), and a live lockfile is ADOPTED after a
gateway restart, never answered with a second curl on the same `.part`. LAN files
(`LanSource`, the share behind `ops/modelsrc-serve.sh`, pinned host key) stream THROUGH the
gateway — exactly one stream per backend (`deps.pipe`: the share's `cat <rel> <offset>`
into `cat >> <rel>.part` on the instance), resumed from the `.part`'s size, verified by
sha256 on both sides; the HF cache's snapshot symlinks are recreated (`_make_links`) and
recorded like files. Files leave the disk only at stop (`_before_snapshot`): the plan's
`prune` list, never `held` or `unknown` ones — those go only by the operator's
`delete_unknown`. Triggers: the start path, a 5-s signature poll in `run_forever` (plus
a changed LAN source: a pin, a new listing, a source gone or back), a re-plan when a
transfer ends and every 60 s while transfers run. A stop cancels and awaits every sync task (`_cancel_sync_tasks`) and
a sync re-checks the phase after planning: a plan about an instance on its way out never
grows its disk or replaces the manifest the snapshot records.

Never imports `main`: everything the controller needs from the gateway arrives in
`Deps`, so it stays hot-reload-safe and testable against a stub API and a fake ssh.
The provider's pure module and API class come from `hostapi.provider(kind)` (the
provider seam: Bearer token, timeout, any 2xx is success, uuid first and the index only
on a 404, token redaction, the one-hour price cache — test_hostapi.py). Covered by
test_hostctl.py.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import dataclasses
import hashlib
import json
import math
import os
import posixpath
import re
import stat
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

import httpx

import hostapi
import modelsync
import sshrun

PHASES = ("off", "creating", "restoring", "connecting", "bootstrapping", "starting",
          "syncing", "ready", "draining", "pruning", "snapshotting", "deleting", "failed")

_LOG_MAX = 200                  # lines in the per-host log ring
COMFY_PORT = 8188               # ComfyUI's default port on the instance, loopback only
# the fault log groups by backend+type: host events go to this pseudo backend's type
HOST_FAULT_TYPE = "managed-host"

# start path timing
_CREATE_POLL_S = 10             # /instances/list while creating/restoring
_CREATE_BASE_S = 15 * 60        # create timeout = base + per started 100 GB of disk:
_RESTORE_PER_100GB_S = 8 * 60   #   Thunder's docs: a restore takes up to 8 min / 100 GB
_SSH_READY_S = 5 * 60           # a RUNNING instance whose sshd does not answer by then
_SSH_PROBE_S = 5
_BOOTSTRAP_S = 3 * 3600         # venv + torch + node packs + CUDA extension builds
_HOST_BOOTSTRAP_S = 30 * 60     # kill + inventory + at most an apt install and uv
_COMFY_READY_S = 10 * 60        # first start loads custom nodes (3D packs import slowly)
_COMFY_PROBE_S = 3
_STDERR_LOG_LINES = 20          # stderr tail logged for a failed remote step

_CREATED_SKEW_S = 5 * 60        # createdAt may precede our request by clock skew …
_CREATED_LATE_S = 15 * 60       # … and follow it by Thunder's own queueing
_COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}")
# The loop script is flock-guarded, so starting it while it runs is a no-op. A missing
# script (a bootstrap that died before writing it) fails in seconds with a named reason
# instead of a 10-minute probe timeout that says nothing.
_START_CMD = ("test -x ~/start-comfy.sh || { echo 'start-comfy.sh missing' >&2; exit 3; }; "
              "setsid nohup ~/start-comfy.sh >/dev/null 2>&1 < /dev/null &")
# The bootstrap's output also goes to a file ON THE INSTANCE: `sshrun.run` answers a
# timeout with (124, b"", b"timeout") and loses everything it read, and a dropped ssh
# loses it too — the log file is where the last GW:PHASE is fetched from then. Wrapped
# in `bash -o pipefail -c` so the SCRIPT's exit status survives the `| tee` whatever
# the login shell is; `bash -s` reads the script from ssh's stdin, tee only the pipe.
_BOOTSTRAP_LOG = "~/gw-bootstrap.log"
_HOST_BOOTSTRAP_LOG = "~/gw-host-bootstrap.log"   # the host bootstrap's own (R-W3)
_BOOTSTRAP_TAIL = 200


def _restart_cmd(port: int) -> str:
    """Kill ComfyUI (the loop restarts it 2 s later) and start the loop in case it is
    not alive (a failed bootstrap, a killed wrapper). The bracket keeps the pattern from
    matching the remote `bash -c "<this command>"` itself — pkill spares only its own
    process, and a pattern that hit the shell would kill the restart half-way. It
    matches the loop's `<venv python> main.py --listen 127.0.0.1 --port <port> …`,
    whatever venv the bootstrap picked (`ComfyUI/main.py` never appears in that command
    line). `port` is the SERVICE's remote port: a pattern with another service's port
    kills nothing, or the wrong ComfyUI. An int by construction — no quoting needed."""
    return ("pkill -f '[m]ain[.]py --listen 127[.]0[.]0[.]1 --port "
            f"{sshrun._port(int(port))}'; " + _START_CMD)

# stop path / background timing
_DRAIN_POLL_S = 2               # inflight check while draining (no timeout: a job may finish)
_DELETE_POLL_S = 10             # /instances/list after the delete …
_DELETE_WAIT_S = 5 * 60         # … until the instance is gone, else failed(deleting)
_DU_TIMEOUT_S = 10 * 60         # `du` over the home directory (a venv has 100k files)
# Everything under ~ except the two model roots: what the NEXT disk needs besides models.
_DU_CMD = "du -sb --exclude=ComfyUI/models --exclude=hf-cache ~ | cut -f1"
_WATCH_S = 60                   # snapshot watcher / resume retry interval
_ACCOUNT_S = 600                # account view re-read: snapshot list and foreign instances
                                # every 10 min; the price list inside it only hourly
                                # (`hostapi.ProviderApi._cached`, `_PRICE_TTL_S`)
_LONG_RUN_S = 24 * 3600         # phase != off for longer → the cost banner (spec)
_PENDING_MISSES = 3             # rounds a pending snapshot may be absent from the list
# An instance counts as gone only when this many CONSECUTIVE fresh lists lack it:
# `thunder.parse_instances` reads any odd 2xx body as `[]`, and one bad answer must not
# make the controller forget (and stop deleting) an instance that bills.
_ABSENT_CONFIRM = 2
_ABSENT_RECHECK_S = 5
_RESUME_PROBE_S = 30            # a freshly started tunnel needs a moment before ComfyUI answers
# a Start without a token is refused before any call (each would be the provider's 401)
_NO_TOKEN = "no {name} API token set — put it into the API key field"
# a Start without a service creates a machine that serves nothing and bills anyway
_NO_SERVICE = "no backend is attached to managed host {host} — nothing to start"
_STOP_STEPS = ("draining", "pruning", "snapshotting", "deleting")
# Phases before any bootstrap ran: the instance holds nothing a snapshot should keep (a
# template, or an unchanged copy of the snapshot it was restored from).
_PRE_BOOT = ("creating", "restoring", "connecting")
# ops a stop() may abort (Ruling 13): an operator must be able to end a hanging, billing
# start; "stopping" itself is never aborted
_ABORTABLE = ("starting", "restarting ComfyUI", "resuming")
# the fallback directory for control sockets when `<datadir>/<kind>-ctl/<name>` exceeds
# `sshrun.CTL_PATH_MAX` (Ruling M2); the ai-hub unit has PrivateTmp, so /tmp is its own
_CTL_FALLBACK = "/tmp/ai-hub-{uid}-ctl"
_SVC_STATUSES = ("starting", "up", "setup failed", "down")

# model sync (spec "Controller": URL transfer, triggers, disk growth; "Stop" 1–2)
_SYNC_PHASES = ("syncing", "ready")   # the phases a plan is made (and trusted) in
_LIVE_PHASES = ("starting", "syncing", "ready")   # our instance must be listed then
_SYNC_POLL_S = 5                # signature check / transfer poll interval
_SYNC_REFRESH_S = 60            # a re-plan while transfers run, or after a failed sync
_MAX_FETCH = 3                  # URL transfers at once
_FETCH_ATTEMPTS = 3             # then the file's aliases are blocked (+ fault)
_RETRY_DELAYS_S = (15, 45)      # backoff before attempt 2 and 3 (spec "mit Backoff")
_HEAD_TIMEOUT_S = 30            # a HEAD that learns a URL file's size
_POLL_FAILS_MAX = 60            # consecutive failed polls (5 min) end an attempt
_STALL_S = 600                  # curl aborts itself below 1 B/s for this long
_SHA_TIMEOUT_S = 30 * 60        # sha256sum of a 20 GB file on a network disk
_PRUNE_TIMEOUT_S = 10 * 60
_GROW_RECHECK = 6               # df re-reads after a disk modify …
_GROW_RECHECK_S = 10            # … this far apart (the filesystem grows behind it)
_HF_HOSTS = frozenset({"huggingface.co", "hf.co"})
_MANIFEST = ".gw-modelsync.json"
# plan path prefix → path under the instance's home (spec "Zwei Wurzeln")
_ROOT_MAP = (("models/", "ComfyUI/models/"), ("hf-cache/", "hf-cache/"))
_LAN_WAIT = "waiting for LAN source (not configured)"
_NOT_IN_SOURCE = "not in source: "
# the LAN model share (spec "Quell-Runner (LAN)", "Übertragung LAN")
MODELSRC_HOST_DEFAULT = "modelsrc@192.168.8.24"
# = main._VOICE_HOST_RE (pinned by a test): `[user@]host` of plain characters — no
# leading `-`, no spaces or quotes; it reaches an ssh argv (after `--`, but still)
_SRC_HOST_RE = re.compile(r"^(?:[A-Za-z0-9_][A-Za-z0-9._-]*@)?[A-Za-z0-9_][A-Za-z0-9._-]*$")
_SRC_TTL_S = 600                # the source index is re-listed after 10 min …
_SRC_RETRY_S = 60               # … a failed list after one
_SRC_LIST_TIMEOUT_S = 300       # one `find` pass over ~1 TB of models
_SRC_SHA_TIMEOUT_S = 30 * 60    # sha256 of a 20 GB file on the share
_KEYSCAN_TIMEOUT_S = 30
_LAN_IDLE_S = 120               # a LAN stream moving no byte this long is ended
_FLOCK_BUSY = 75                # rc of `flock -n -E 75`: another stream still appends


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
    # its name, persisted BEFORE the create: a restart between the POST and persisting
    # the id finds the snapshot by name instead of taking a second one
    pending_snapshot_name: str = ""
    manifests: dict = field(default_factory=dict)   # snapshot id → model manifest
    # what the last HOST bootstrap reported: models the TEMPLATE brought (rel → bytes;
    # paid for in every snapshot until deleted) and custom_nodes packs it brought
    bootstrap_unknown: dict = field(default_factory=dict)
    bootstrap_template_nodes: list = field(default_factory=list)
    # The host bootstrap (R-W3, every host): True once it finished on the running
    # instance (or the instance came from a snapshot where it had). A snapshot taken
    # while False lands in `host_incomplete_snapshots`, and a start from one runs it
    # again. (A record written before the split has no such key: `state_from` derives
    # it — the old one-piece bootstrap did the host part too.)
    host_bootstrapped: bool = False
    host_incomplete_snapshots: list = field(default_factory=list)
    # The COMFYUI bootstrap: True from the create of an instance that needs it — with a
    # ComfyUI service attached — until it succeeded (or ComfyUI was confirmed running):
    # a snapshot taken meanwhile holds a half-finished install — a timed-out bootstrap
    # may even still run on the box
    bootstrap_incomplete: bool = False
    # snapshot ids taken while `bootstrap_incomplete`: a start from one runs the
    # bootstrap again (kept apart from `manifests`, whose entries are model files)
    incomplete_snapshots: list = field(default_factory=list)
    # The instance carries no ComfyUI install of ours because no ComfyUI service was
    # attached (a vLLM-only host). Not "incomplete" — nothing failed, nothing is logged —
    # but its snapshots (`no_comfy_snapshots`) still get the ComfyUI bootstrap once a
    # ComfyUI service is attached, and `_ensure_comfy_bootstrap` runs it on this instance.
    comfy_absent: bool = False
    no_comfy_snapshots: list = field(default_factory=list)
    # uuids of unowned instances seen when the stored record was unreadable: one may be
    # ours, so no start while any of them is still listed (or the operator forgets them)
    unreconciled_uuids: list = field(default_factory=list)
    # what the last create asked for and when — the only way to recognise OUR instance
    # when create answered without a uuid (Ruling 10), also after a gateway restart
    created_template: str = ""
    create_requested_at: float = 0.0
    # per attached service (backend id → {"status", "error", "setup_hash"}): what the last
    # start/probe of it said. Host state stays above — a service's failure is its own
    services: dict = field(default_factory=dict)
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
    if "host_bootstrapped" not in d:
        # a record from before the host/ComfyUI split: the one-piece bootstrap did the
        # host part as well, so what it called complete is host-bootstrapped too, and a
        # snapshot it marked incomplete may lack the host part (re-running it is safe)
        s.host_bootstrapped = not s.bootstrap_incomplete
        if "host_incomplete_snapshots" not in d:
            s.host_incomplete_snapshots = list(s.incomplete_snapshots)
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
    # model sync (P2): what the aliases on a ComfyUI service need (called with the
    # service's BACKEND ID), a hash of exactly that input (candidates + catalog), the
    # source's files, the public download sources and the Hugging Face token. The two
    # alias readers are blocking store reads — the controller calls them in a worker
    # thread.
    alias_needs: Callable[[str], list] = field(default=lambda bid: [])
    alias_signature: Callable[[str], str] = field(default=lambda bid: "")
    source_index: Callable[[], dict] = field(default=lambda: {})
    url_catalog: Callable[[], dict] = field(default=lambda: {})
    hf_token: Callable[[], str] = field(default=lambda: "")
    # the LAN model share (P3): `LanSource` (main: one per gateway; its `cached` is
    # `source_index`) and the stream runner. None = no LAN source.
    lan: Optional[Any] = None
    pipe: Callable[..., Awaitable[tuple]] = sshrun.pipe
    # a forward added/cancelled on the RUNNING master (R-W1): `sshrun.control`
    control: Callable[..., Awaitable[tuple]] = sshrun.control
    # ops/host-bootstrap.sh (R-W3); b"" fails the host bootstrap (no GW:DONE)
    host_bootstrap_script: Callable[[], bytes] = field(default=lambda: b"")


# ── bootstrap output ─────────────────────────────────────────────────────────

def parse_bootstrap(out: str) -> dict:
    """The `GW:` lines of `ops/host-bootstrap.sh`'s and `ops/thunder-bootstrap.sh`'s
    stdout (their headers document them — one protocol for both). Matched by TAG — the
    first whitespace-separated token of a line that starts with `GW:` — never by
    position: the script prints other lines in between, and a
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


def bootstrap_verdict(rc: int, rep: dict, err: str, smoke_test: bool = True,
                      timeout_s: int = _BOOTSTRAP_S) -> str:
    """"" when the bootstrap succeeded, else why not. Success needs ALL of: rc 0,
    `GW:SMOKE ok`, `GW:DONE` and no `GW:NODE_FAIL` at all (Ruling 9: a pack that did not
    install fails workflows later with a plausible-looking error). `smoke_test=False`
    (the host bootstrap, which has none): rc 0 and `GW:DONE`."""
    why = []
    if rep["node_fails"]:
        why.append("node packs failed: " + "; ".join(rep["node_fails"]))
    smoke = rep["smoke"]
    if smoke is not None and smoke != "ok":
        why.append("smoke test " + smoke)
    where = f" in phase {rep['phase']}" if rep["phase"] else ""
    if rc == 124:
        took = (f"{timeout_s // 3600} h" if timeout_s % 3600 == 0
                else f"{timeout_s // 60} min")
        why.append(f"timed out after {took}{where}")
    elif rc == 3 and smoke is None and smoke_test:
        why.append("smoke test failed")
    elif rc not in ((0, 3) if smoke_test else (0,)):
        last = next((ln.strip() for ln in reversed((err or "").splitlines())
                     if ln.strip() and not ln.startswith("GW:")), "")
        why.append(f"bootstrap exited rc {rc}{where}" + (f": {last}" if last else ""))
    if not smoke_test:
        if not why and not rep["done"]:
            why.append(f"bootstrap ended without GW:DONE{where}")
    elif not why and not (smoke == "ok" and rep["done"]):
        why.append(f"bootstrap ended without GW:SMOKE ok / GW:DONE{where}")
    return "; ".join(why)


def _tail(b: bytes, n: int = _STDERR_LOG_LINES) -> list[str]:
    lines = [ln for ln in (b or b"").decode("utf-8", "replace").splitlines() if ln.strip()]
    return lines[-n:]


def _redact(text: str, token: str) -> str:
    """curl's stderr and remote errors go to the log, the panel and the fault log: the
    Hugging Face token a command was given never does."""
    return text.replace(token, "***") if token else text


def _errtext(e: BaseException) -> str:
    """`str()` of a TimeoutError or an httpx error can be empty."""
    return str(e) or type(e).__name__


# ── model sync: paths and remote commands ────────────────────────────────────
# Every remote sync command starts with a `: gw-<verb> [path] ;` no-op: `ps` and the
# journal then say which file a process is about (and the test VM parses it). Each
# runs from the home directory with ROOT-RELATIVE paths, so what `find` prints maps to
# a plan path without knowing the home. Paths go through `remote_path` (`safe_rel`)
# and `sshrun.q`, and `--` precedes them wherever a command takes options.

def remote_path(rel: str) -> str:
    """A plan path (`models/…` | `hf-cache/…`) → its path under the instance's home.
    Anything else — a dot segment, an absolute path, a third root, a directory — is a
    ValueError: an `rm` or a download must never reach outside the two model roots."""
    rel = sshrun.safe_rel(rel)
    if rel.endswith("/"):
        raise ValueError(f"directory, not a file: {rel!r}")
    for key, remote in _ROOT_MAP:
        if rel.startswith(key) and len(rel) > len(key):
            return remote + rel[len(key):]
    raise ValueError(f"path outside the model roots: {rel!r}")


def plan_path(remote: str) -> Optional[str]:
    """The inverse of `remote_path` (None for anything outside the roots or unsafe)."""
    for key, rp in _ROOT_MAP:
        if remote.startswith(rp) and len(remote) > len(rp):
            try:
                return sshrun.safe_rel(key + remote[len(rp):])
            except ValueError:
                return None
    return None


# files as `<path>\t<size>`, symlinks as `L\t<path>\t<target text>`: the HF cache's
# snapshots/ are links, and a present one must not be re-created (or look missing)
_INDEX_CMD = (": gw-index ; cd ~ && find ComfyUI/models hf-cache -type f "
              "-printf '%p\\t%s\\n' -o -type l -printf 'L\\t%p\\t%l\\n' 2>/dev/null ; "
              "echo GW:END")
_MANIFEST_READ = f": gw-manifest ; cat ~/{_MANIFEST} 2>/dev/null || echo {{}}"
# atomic: a dropped connection leaves the old manifest, never half a new one
_MANIFEST_WRITE = (f": gw-manifest-write ; cd ~ && cat > {_MANIFEST}.tmp && "
                   f"mv -f {_MANIFEST}.tmp {_MANIFEST}")
_DF_CMD = ": gw-df ; df -B1 --output=avail ~ | tail -1"
# A lock is live only while its pid is a CURL: a pid from a lock that outlived its
# process (a restored disk, a reboot) may by now be any other process — adopting it
# would wait forever, killing it could end ComfyUI.
_ALIVE_FN = ('_alive() { [ -f "$1" ] && '
             '[ "$(cat /proc/"$(head -n 1 -- "$1")"/comm 2>/dev/null)" = curl ]; }; ')
# After a fetch spawned its curl: has it got that far? The lock names a CURL (the child
# wrote its pid and exec'd), or a pid that is already gone (curl ran and ended — the
# poll reads how). A lock still naming the pre-exec `sh`, an empty lock or none: not
# yet — a kill in that window would miss the curl (`_KILL_CMD` kills curls only).
_STARTED_FN = ('_started() { p=$(head -n 1 -- "$1" 2>/dev/null); [ -n "$p" ] || return 1; '
               '[ -e /proc/"$p" ] || return 0; '
               '[ "$(cat /proc/"$p"/comm 2>/dev/null)" = curl ]; }; ')
_START_WAIT = 30                # × 0.1 s: how long a fetch waits for its curl to be up
# Ends every download of ours: the curl pids named by the lockfiles, TERM, up to 10 s
# to exit, then KILL (a snapshot must not freeze a growing .part).
_KILL_CMD = (": gw-kill ; cd ~ && pids=$(find ComfyUI/models hf-cache -type f "
             "-name '*.part.lock' 2>/dev/null | while IFS= read -r f; do "
             "p=$(head -n 1 -- \"$f\"); [ \"$(cat /proc/\"$p\"/comm 2>/dev/null)\" = curl ] "
             "&& echo \"$p\"; done); for p in $pids; do kill \"$p\" 2>/dev/null; done; "
             "i=0; while [ $i -lt 10 ]; do a=; for p in $pids; do [ -e /proc/\"$p\" ] && a=1; "
             "done; [ -z \"$a\" ] && break; sleep 1; i=$((i+1)); done; "
             "for p in $pids; do kill -9 \"$p\" 2>/dev/null; done; echo GW:KILLED")
_PART_FIND = ("find ComfyUI/models hf-cache -type f \\( -name '*.part' -o -name '*.part.lock' "
              "-o -name '*.part.log' \\) -delete 2>/dev/null")


def _parts(rel: str) -> tuple[str, str, str, str, str]:
    """(dir, base, part, lock, log) of a plan path on the instance."""
    d, base = posixpath.split(remote_path(rel))
    return d, base, base + ".part", base + ".part.lock", base + ".part.log"


def _fetch_cmd(rel: str) -> str:
    """Adopt a live download of `rel`, else start one — in ONE command, so no second
    curl can start between the check and the start. curl reads its options (URL, and
    for a Hugging Face host the token header) from stdin, never argv (`ps` shows argv
    to everyone): the foreground reads stdin to the end first (`cfg=$(cat)`) so nothing
    is lost when ssh closes, and `printf` is a builtin, so no process ever carries the
    config on its command line. `< /dev/stdin` is explicit because a background job's
    stdin is /dev/null otherwise. `setsid`: the curl outlives the ssh session; its pid
    (`exec`) is what the lockfile holds.

    `GW:STARTED` only once the lock names the curl (`_started`): echoed right after the
    spawn, a stop's `_KILL_CMD` landing before the child wrote its pid and exec'd found
    no curl to kill, and the download ran on into the snapshot. A stale lock is removed
    first (`_alive` just said nobody holds it), so the lock that appears is this
    child's. Not up within ~3 s → `GW:START-FAIL` plus the log tail: a failed attempt."""
    d, base, part, lock, log = _parts(rel)
    q = sshrun.q
    inner = f"echo $$ > {q(lock)}; exec curl -L --fail -sS -C - -o {q(part)} --config -"
    return (f": gw-fetch {q(rel)} ; {_ALIVE_FN}{_STARTED_FN}cd ~ && mkdir -p -- {q(d)} && "
            f"cd -- {q(d)} && "
            f"if _alive {q(lock)}; then echo GW:ADOPT; else cfg=$(cat) && rm -f -- {q(lock)} && "
            f"printf '%s\\n' \"$cfg\" | (setsid sh -c {q(inner)} < /dev/stdin > {q(log)} 2>&1 &) "
            f"&& i=0 && while [ $i -lt {_START_WAIT} ] && ! _started {q(lock)}; do sleep 0.1; "
            f"i=$((i+1)); done; if _started {q(lock)}; then echo GW:STARTED; else "
            f"echo GW:START-FAIL; tail -c 2000 -- {q(log)} 2>/dev/null; fi; fi")


def _poll_cmd(rel: str) -> str:
    """→ `GW:RUN|GW:END`, the .part's size (-1 = none), then curl's stderr (-sS: empty
    unless it failed)."""
    d, base, part, lock, log = _parts(rel)
    q = sshrun.q
    return (f": gw-poll {q(rel)} ; {_ALIVE_FN}cd ~ && cd -- {q(d)} && {{ if _alive {q(lock)}; "
            f"then echo GW:RUN; else echo GW:END; fi; stat -c %s -- {q(part)} 2>/dev/null "
            f"|| echo -1; tail -c 2000 -- {q(log)} 2>/dev/null; }}")


def _sha_cmd(rel: str) -> str:
    part = remote_path(rel) + ".part"
    return f": gw-sha {sshrun.q(rel)} ; cd ~ && sha256sum -- {sshrun.q(part)}"


def _done_cmd(rel: str) -> str:
    d, base, part, lock, log = _parts(rel)
    q = sshrun.q
    return (f": gw-done {q(rel)} ; cd ~ && cd -- {q(d)} && mv -f -- {q(part)} {q(base)} && "
            f"rm -f -- {q(lock)} {q(log)} && stat -c %s -- {q(base)}")


def _discard_cmd(rel: str) -> str:
    d, base, part, lock, log = _parts(rel)
    q = sshrun.q
    return f": gw-discard {q(rel)} ; cd ~ && cd -- {q(d)} && rm -f -- {q(part)} {q(lock)} {q(log)}"


def _prune_cmd(rels: list) -> str:
    """Delete the plan's prune list (manifest files only) and every transfer artefact.
    `GW:RM-OK` says the list is gone — only then does the manifest drop its entries."""
    rm = ""
    if rels:
        files = " ".join(sshrun.q(remote_path(r)) for r in rels)
        rm = f"rm -f -- {files} && echo GW:RM-OK ; "
    return f": gw-prune ; cd ~ && {rm}{_PART_FIND} ; echo GW:PRUNED"


def _delete_cmd(rels: list) -> str:
    return (": gw-delete ; cd ~ && rm -f -- "
            + " ".join(sshrun.q(remote_path(r)) for r in rels) + " && echo GW:DELETED")


def _head_cmd(rel: str) -> str:
    """HEAD the URL (redirects followed) to learn a file's size before downloading it —
    the options on stdin exactly as for the download (`_fetch_cmd`): the token, where
    there is one, never reaches an argv."""
    return (f": gw-head {sshrun.q(rel)} ; cfg=$(cat) && printf '%s\\n' \"$cfg\" | "
            "curl -sIL --config -")


def _part_size_cmd(rel: str) -> str:
    """The size of `rel`'s `.part` (0 when there is none): where a LAN stream resumes."""
    part = remote_path(rel) + ".part"
    return (f": gw-part {sshrun.q(rel)} ; cd ~ && stat -c %s -- {sshrun.q(part)} "
            "2>/dev/null || echo 0")


def _lan_recv_cmd(rel: str) -> str:
    """The instance end of a LAN stream: append stdin to `rel`'s `.part`. `flock -n` on
    the `.part.lock`: a second appender (a stream an earlier gateway process left
    running) would interleave bytes — this one exits 75 instead, having written
    nothing. The lock holds no pid, so the URL path's `_alive` never adopts it."""
    d, base, part, lock, log = _parts(rel)
    q = sshrun.q
    return (f": gw-lan-recv {q(rel)} ; cd ~ && mkdir -p -- {q(d)} && cd -- {q(d)} && "
            f"exec flock -n -E {_FLOCK_BUSY} -- {q(lock)} cat >> {q(part)}")


# Symlinks (the HF cache's snapshots/) from stdin, two lines each: the path under the
# home, then the target text — on stdin, not argv: a repo has hundreds of them and one
# argument is capped at 128 KiB. Neither can hold a newline (safe_rel / modelsync).
_LINK_CMD = (": gw-link ; cd ~ && while IFS= read -r p && IFS= read -r t; do "
             "mkdir -p -- \"${p%/*}\" && ln -sfn -- \"$t\" \"$p\" || exit 1; done; "
             "echo GW:LINKED")


def parse_head(text: str) -> Optional[int]:
    """The FINAL response's Content-Length of `curl -sIL` output (one header block per
    hop), None when that response is no 2xx or names no length. A 2xx naming length 0
    is unknown too: no model file is empty, and servers answer a HEAD they do not
    really serve that way — taken as the size, every download of the file would then
    fail its size check and its aliases would end up blocked."""
    blocks, cur = [], None
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln.upper().startswith("HTTP/"):
            cur = {"status": ln, "len": None}
            blocks.append(cur)
        elif cur is not None and ln.lower().startswith("content-length:"):
            v = ln.split(":", 1)[1].strip()
            cur["len"] = int(v) if v.isdigit() else None
    if not blocks:
        return None
    last = blocks[-1]
    parts = last["status"].split()
    if len(parts) < 2 or not parts[1].startswith("2"):
        return None
    return last["len"] or None


def parse_index(text: str) -> dict:
    """`find -printf '%p\\t%s\\n'` (root-relative) → `{plan path: bytes}`, and a symlink
    line `L\\t<path>\\t<target>` → `{plan path: {"link": target}}` (modelsync's form).
    Paths outside the roots or with a dot segment are skipped (the HF cache's `.locks`, a
    `.git`). Without the end marker the listing was cut off — a RuntimeError, never a
    shorter index: a file missing from it is downloaded again from zero."""
    lines = (text or "").splitlines()
    if "GW:END" not in lines:
        raise RuntimeError("model index incomplete (no end marker)")
    out: dict = {}
    for ln in lines[:lines.index("GW:END")]:
        f = ln.split("\t")
        if len(f) == 3 and f[0] == "L":
            key = plan_path(f[1])
            if key is not None and f[2]:
                out[key] = {"link": f[2]}
            continue
        path, sep, size = ln.rpartition("\t")
        if not sep or not size.isdigit():
            continue
        key = plan_path(path)
        if key is not None:
            out[key] = int(size)
    return out


def parse_manifest(text: str) -> dict:
    """The manifest file → `{path: entry}`, `aliases` always a list. Unreadable → `{}`:
    its files then count as "unknown" (listed, never deleted) — the safe direction."""
    try:
        d = json.loads(text or "{}")
    except ValueError:
        return {}
    return normalize_manifest(d)


def normalize_manifest(d) -> dict:
    """`{path: entry}` with dict entries only and `aliases` always a list (a string
    there would otherwise be iterated letter by letter)."""
    if not isinstance(d, dict):
        return {}
    out: dict = {}
    for k, v in d.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        e = dict(v)
        al = e.get("aliases")
        e["aliases"] = [str(x) for x in al] if isinstance(al, (list, tuple)) else []
        out[k] = e
    return out


def curl_config(url: str, token: str, head: bool = False) -> str:
    """curl's options on stdin (`--config -`): the URL and — only when the caller
    decided the host is Hugging Face — the token header. Redirects stay on https;
    `--location-trusted` is never set, so curl drops the header when a redirect leaves
    the host (HF sends its files from a CDN). curl aborts a download that stalls."""
    lines = [f'url = "{url}"', 'proto = "=https"', 'proto-redir = "=https"']
    lines += ([f"max-time = {_HEAD_TIMEOUT_S}"] if head
              else ["speed-limit = 1", f"speed-time = {_STALL_S}"])
    if token:
        lines.append(f'header = "Authorization: Bearer {token}"')
    return "\n".join(lines) + "\n"


# ── the LAN model share ──────────────────────────────────────────────────────
# Mapping (ledger note for Tasks 14/15): the share root IS the models tree, with the
# Hugging Face cache as `hf-cache/` inside it. Plan `models/<x>` ↔ share `<x>` for every
# `<x>` not under `hf-cache/`; plan `hf-cache/<y>` ↔ share `hf-cache/<y>`.

def share_rel(path: str) -> str:
    """A plan path → its path on the model share (ValueError outside the mapping —
    `models/hf-cache/…` would name the share's HF cache, which is `hf-cache/…`)."""
    path = sshrun.safe_rel(path)
    if path.endswith("/"):
        raise ValueError(f"directory, not a file: {path!r}")
    if path.startswith("hf-cache/") and len(path) > len("hf-cache/"):
        return path
    if path.startswith("models/"):
        x = path[len("models/"):]
        if x and x != "hf-cache" and not x.startswith("hf-cache/"):
            return x
    raise ValueError(f"no model-share path for {path!r}")


def plan_of_share(rel: str) -> Optional[str]:
    """The inverse of `share_rel` (None for anything unsafe or unmapped)."""
    try:
        rel = sshrun.safe_rel(rel)
    except ValueError:
        return None
    if rel.endswith("/") or rel == "hf-cache":
        return None
    return rel if rel.startswith("hf-cache/") else "models/" + rel


def parse_source_list(text: str) -> dict:
    """`modelsrc-serve list` (`F\t<rel>\t<size>` / `L\t<rel>\t<target>`, share paths) →
    the plan-path index modelsync reads: `{path: size}` and `{path: {"link": target}}`.
    A link whose target leaves its root is dropped HERE, in share space: a share link
    `vae/x → ../hf-cache/hub/…` looks like a `models/` path once mapped, but its target
    sits in the other root, which is not beside it on the instance. Unparseable lines
    are skipped (the script never writes one; the caller judges completeness by rc)."""
    out: dict = {}
    for ln in (text or "").split("\n"):
        f = ln.split("\t")
        if len(f) != 3:
            continue
        kind, rel, val = f
        key = plan_of_share(rel)
        if key is None:
            continue
        if kind == "F" and re.fullmatch(r"[0-9]{1,18}", val):
            out[key] = int(val)
        elif kind == "L":
            tgt = modelsync.resolve_link(rel, val)
            if (tgt is None or plan_of_share(tgt) is None
                    or tgt.startswith("hf-cache/") != rel.startswith("hf-cache/")):
                continue
            out[key] = {"link": val}
    return out


def host_key_fingerprint(b64: str) -> str:
    """OpenSSH's `SHA256:…` fingerprint of a public key blob (what `ssh-keygen -lf` and
    the first-connect prompt print), so the operator can compare it on the share host."""
    blob = base64.b64decode(b64, validate=True)
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")


def parse_keyscan(text: str, hostname: str) -> Optional[str]:
    """The `<host> ssh-ed25519 <key>` line of `ssh-keyscan` output for `hostname` (a
    known_hosts line as-is), None when there is none or its key is no valid base64."""
    for ln in (text or "").splitlines():
        parts = ln.split()
        if len(parts) < 3 or parts[0] != hostname or parts[1] != "ssh-ed25519":
            continue
        try:
            host_key_fingerprint(parts[2])
        except ValueError:
            continue
        return f"{parts[0]} ssh-ed25519 {parts[2]}"
    return None


def hf_token_ok(tok: str) -> bool:
    """Can a curl config line (`header = "Authorization: Bearer <tok>"`) carry this HF
    token verbatim? The ONE rule: the console refuses a token failing it at Save
    (main.save_hf_token) and the transfer withholds one (`_hf_token_for`) — two rules
    would let a token be saved that every download then silently goes without."""
    tok = str(tok or "")
    return 0 < len(tok) <= 512 and all(33 <= ord(c) <= 126 and c not in '"\\' for c in tok)


class LanSource:
    """The LAN model share behind `ops/modelsrc-serve.sh` (spec "Quell-Runner (LAN)"),
    ONE per gateway — every Thunder controller reads the same share. It holds what
    guards the transfers from it:

    - **the host key is pinned.** `StrictHostKeyChecking=yes` against
      `<datadir>/modelsrc-known_hosts`, which exists only after the operator compared
      the fingerprint `scan()` fetched and confirmed it (`pin`). Without it there is no
      LAN source at all (`configured()` False): an accept-new first connect would trust
      whatever answers on the LAN and hand it nothing less than the model traffic.
    - **a failed `list` is no empty source.** Exit status 1 is "list incomplete", 255
      an ssh failure — judged by rc, never by stderr (the share host's setlocale noise).
      The last good listing is KEPT and the failure becomes `problem()`: LAN transfers
      wait, files already synced stay verified.
    - **the index is cached** (10 min, a failed list is retried after 1), `invalidate()`d
      by every instance start and by "Sync now"; `generation` changes whenever the index
      or the problem does, which is how a controller notices a re-plan is due.
    - **sha256 is cached per (path, size)** — a 20 GB file is hashed on the share once.

    `ssh`/`keygen`/`now` are injected (tests); nothing here imports `main`."""

    def __init__(self, datadir: str, host: Callable[[], str], ssh=sshrun.run,
                 keygen=sshrun.keygen, now: Callable[[], float] = time.time,
                 log: Callable[[str], None] = _default_log):
        self.datadir = datadir
        self._host_fn = host
        self._ssh = ssh
        self._keygen = keygen
        self._now = now
        self._logf = log
        self._index: Optional[dict] = None     # last GOOD listing (plan paths)
        self._listed_at = 0.0
        self._tried_at = 0.0
        self._error = ""
        self._gen = 0
        self._lock = asyncio.Lock()
        self._key_lock = asyncio.Lock()
        self._key_ready = False
        self._sha: dict = {}
        self._scanned: Optional[tuple] = None   # (host, known_hosts line, fingerprint)
        self._for_host: Optional[str] = None    # the modelsrc_host everything above is for
        self._pin_cache: Optional[tuple] = None  # ((ino, mtime_ns, size), known_hosts fields)

    def _log(self, msg: str) -> None:
        try:
            self._logf(f"[modelsrc] {msg}")
        except Exception:
            pass

    @property
    def key_path(self) -> str:
        return os.path.join(self.datadir, "modelsrc.key")

    @property
    def known_hosts_path(self) -> str:
        return os.path.join(self.datadir, "modelsrc-known_hosts")

    def _look(self) -> str:
        """Read the `modelsrc_host` setting ONCE for this entry point (a store read in
        main) and follow a change; every helper below takes the value it returned. A
        FAILED read is no change: the last known host stands — treating a transient
        store error as "" would drop the listing and the sha256 cache for nothing."""
        try:
            raw = str(self._host_fn() or "").strip()
        except Exception:
            return self._for_host or ""
        self._follow_host(raw)
        return raw

    def raw_host(self) -> str:
        return self._look()

    @staticmethod
    def _plain(raw: str) -> str:
        return raw if _SRC_HOST_RE.match(raw) else ""

    def host(self) -> str:
        """The `[user@]host`, or "" when the setting is not a plain one."""
        return self._plain(self._look())

    def _follow_host(self, raw: str) -> None:
        """A changed `modelsrc_host` (the console's field) makes the listing, the sha256
        cache and a fetched-but-unconfirmed key describe ANOTHER share: dropped here, on
        the first look after the change, however the setting was changed. Without it the
        new host would be planned from the old share's index until the TTL ran out."""
        if self._for_host is None:
            self._for_host = raw
            return
        if raw == self._for_host:
            return
        self._log(f"modelsrc_host changed ({self._for_host or '-'} → {raw or '-'}) — "
                  "listing dropped")
        self._for_host = raw
        self._index, self._listed_at, self._tried_at, self._error = None, 0.0, 0.0, ""
        self._sha.clear()
        self._scanned = None
        self._gen += 1

    def _pin_parts(self) -> list:
        """The pinned key line's fields ([] when nothing is pinned) — re-read only when
        the file's mtime/size changed: `problem()` runs every few seconds per controller,
        and a file open per look added up."""
        try:
            st = os.stat(self.known_hosts_path)
        except OSError:
            self._pin_cache = None
            return []
        key = (st.st_ino, st.st_mtime_ns, st.st_size)   # pin() replaces the file: new inode
        if self._pin_cache is None or self._pin_cache[0] != key:
            parts: list = []
            if st.st_size > 0:
                try:
                    with open(self.known_hosts_path, encoding="utf-8") as f:
                        parts = f.readline().split()
                except OSError:
                    parts = []
            self._pin_cache = (key, parts)
        return self._pin_cache[1]

    def _stale_pin(self, raw: str) -> str:
        """The host a pin is for when it is NOT the configured one ("" otherwise): with
        StrictHostKeyChecking=yes every connect would fail as "unreachable", which
        points the operator at the network instead of the missing pin."""
        host = self._plain(raw)
        parts = self._pin_parts()
        pn = parts[0] if parts else ""
        name = host.rsplit("@", 1)[-1] if host else ""
        return pn if (pn and name and pn != name) else ""

    def _pinned(self, raw: str) -> bool:
        return bool(self._pin_parts()) and not self._stale_pin(raw)

    def pinned(self) -> bool:
        """A key is pinned FOR THE CONFIGURED HOST (a pin for the previous host is none)."""
        return self._pinned(self._look())

    def _configured(self, raw: str) -> bool:
        return bool(self._plain(raw)) and self._pinned(raw)

    def configured(self) -> bool:
        return self._configured(self._look())

    def _problem(self, raw: str) -> str:
        host = self._plain(raw)
        if raw and not host:
            return f"not configured: modelsrc_host {raw!r} is not a plain [user@]host"
        old = self._stale_pin(raw)
        if old:
            return f"pinned for {old}, not {host.rsplit('@', 1)[-1]} — fetch its key"
        if not host or not self._pinned(raw):
            return "not configured"
        if self._error:
            return self._error
        if self._index is None:
            return "not listed yet"
        return ""

    def problem(self) -> str:
        """Why LAN transfers wait ("" = they may run) — the text inside "waiting for LAN
        source (…)"."""
        return self._problem(self._look())

    def usable(self) -> bool:
        return self.problem() == ""

    @property
    def generation(self) -> int:
        self._look()
        return self._gen

    def cached(self) -> dict:
        """The last good listing (a copy) — {} while not configured."""
        raw = self._look()
        return dict(self._index or {}) if self._configured(raw) else {}

    def invalidate(self) -> None:
        self._tried_at = 0.0

    def _stale(self, raw: str) -> bool:
        if not self._configured(raw):
            return False
        if not self._tried_at:
            return True
        age = self._now() - self._tried_at
        return age >= (_SRC_RETRY_S if self._error else _SRC_TTL_S)

    def stale(self) -> bool:
        return self._stale(self._look())

    def _argv(self, *words) -> list[str]:
        raw = self._look()
        host = self._plain(raw)
        if not host:
            raise ValueError(f"LAN source {self._problem(raw)}")
        return sshrun.ssh_base(self.key_path, self.known_hosts_path, strict="yes") + [
            "--", host, " ".join(sshrun.q(w) for w in words)]

    def cat_argv(self, path: str, offset: int) -> list[str]:
        return self._argv("cat", share_rel(path), str(int(offset)))

    async def ensure_key(self) -> str:
        """The public key of `modelsrc.key`, generated at first need (serialised: two
        controllers booting at once must not race two ssh-keygens onto one file)."""
        async with self._key_lock:
            pub = await self._keygen(self.key_path)
            self._key_ready = True
            return pub

    async def refresh(self, force: bool = False) -> None:
        """List the share when the cache is stale (or `force`). Never raises: a failure
        is `problem()`, and the last good listing stays."""
        raw = self._look()
        if not self._configured(raw) or (not force and not self._stale(raw)):
            return
        async with self._lock:
            listed_for = self._look()
            if not force and not self._stale(listed_for):
                return                          # another controller just listed
            before = (self._error, self._index)
            self._tried_at = self._now()
            try:
                if not self._key_ready:
                    await self.ensure_key()
                rc, out, err = await self._ssh(self._argv("list"), timeout=_SRC_LIST_TIMEOUT_S)
            except Exception as e:
                rc, out, err = -1, b"", _errtext(e).encode()
            if self._look() != listed_for:
                return                          # the host changed meanwhile: another share's answer
            if rc == 0:
                self._index = parse_source_list((out or b"").decode("utf-8", "replace"))
                self._listed_at = self._tried_at
                if self._error:
                    self._log("reachable again")
                self._error = ""
            else:
                last = " | ".join(_tail(err, 2)) or f"rc {rc}"
                what = ("unreachable" if rc in (255, 124, -1)
                        else "list incomplete" if rc == 1 else f"list failed (rc {rc})")
                err_text = f"{what}: {last}"[:300]
                if err_text != self._error:
                    self._log(f"list failed (rc {rc}) — keeping the last good listing: {last}")
                self._error = err_text
            if (self._error, self._index) != before:
                self._gen += 1

    async def sha256(self, path: str, size: int) -> str:
        self._look()
        key = (path, size)
        if key in self._sha:
            return self._sha[key]
        rc, out, err = await self._ssh(self._argv("sha256", share_rel(path)),
                                       timeout=_SRC_SHA_TIMEOUT_S)
        if rc != 0:
            raise RuntimeError(f"source sha256 failed (rc {rc}): "
                               + (" | ".join(_tail(err, 2)) or "no message"))
        hexd = (out or b"").decode("utf-8", "replace").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", hexd):
            raise RuntimeError(f"source sha256 answered {hexd[:40]!r}")
        self._sha[key] = hexd
        return hexd

    def forget_sha(self, path: str, size: int) -> None:
        """A mismatch: the share's file may have changed at the same size."""
        self._sha.pop((path, size), None)

    # host-key pin (the console's "Fetch host key" / "Confirm fingerprint")
    async def scan(self) -> str:
        """`ssh-keyscan -t ed25519` the host → its fingerprint. The key line is kept in
        memory only; nothing trusts it until `pin` is given the same fingerprint."""
        raw = self._look()
        host = self._plain(raw)
        if not host:
            raise ValueError(f"LAN source {self._problem(raw)}")
        name = host.rsplit("@", 1)[-1]
        rc, out, err = await self._ssh(["ssh-keyscan", "-t", "ed25519", "--", name],
                                       timeout=_KEYSCAN_TIMEOUT_S)
        line = parse_keyscan((out or b"").decode("utf-8", "replace"), name) if rc == 0 else None
        if line is None:
            raise RuntimeError(f"no ed25519 host key from {name} (rc {rc})"
                               + (": " + " | ".join(_tail(err, 2)) if err else ""))
        fp = host_key_fingerprint(line.split()[2])
        self._scanned = (host, line, fp)
        self._log(f"host key of {name} fetched: {fp} (not trusted until confirmed)")
        return fp

    def _scanned_fp(self, raw: str) -> str:
        sc = self._scanned
        return sc[2] if sc is not None and sc[0] == self._plain(raw) else ""

    def scanned_fingerprint(self) -> str:
        return self._scanned_fp(self._look())

    def pin(self, fingerprint: str) -> str:
        """Trust the scanned key: write its line to `modelsrc-known_hosts` (0600,
        atomically). `fingerprint` is the one the operator confirmed — a key scanned
        again in between (a different answer) is refused, never pinned unseen."""
        host = self.host()
        sc = self._scanned
        if not host or sc is None or sc[0] != host:
            raise ValueError(f"no host key fetched for {host or '?'} — fetch it first")
        if fingerprint != sc[2]:
            raise ValueError("the fetched host key is not the one confirmed — fetch it again "
                             "and compare")
        path = self.known_hosts_path
        tmp = path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, (sc[1] + "\n").encode())
        finally:
            os.close(fd)
        os.replace(tmp, path)
        self._pin_cache = None
        self._scanned = None
        self._error = ""
        self.invalidate()
        self._gen += 1
        self._log(f"host key of {host} pinned: {sc[2]}")
        return sc[2]

    def pinned_fingerprint(self) -> str:
        parts = self._pin_parts()
        try:
            return host_key_fingerprint(parts[2]) if len(parts) >= 3 else ""
        except ValueError:
            return ""

    def public_key(self) -> str:
        try:
            with open(self.key_path + ".pub", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return ""

    def view(self) -> dict:
        """The console's LAN block: ONE read of the host setting, the pin from its
        mtime-keyed cache, and the small public-key file."""
        raw = self._look()
        pinned = self._pinned(raw)
        idx = self._index or {}
        links = sum(1 for v in idx.values() if modelsync.link_of(v) is not None)
        return {"host": raw, "host_ok": bool(self._plain(raw)), "pinned": pinned,
                "pinned_fp": self.pinned_fingerprint() if pinned else "",
                "scanned_fp": self._scanned_fp(raw), "public_key": self.public_key(),
                "problem": self._problem(raw), "files": len(idx) - links, "links": links,
                "listed_at": self._listed_at, "error": self._error}


def _gb(n) -> str:
    return f"{(n or 0) / 1024 ** 3:.1f}"


class _PreCreate(Exception):
    """A start failed before any instance existed → `off`, not `failed`."""


class _Vanished(Exception):
    """The instance disappeared under a stop before its snapshot → `off` + fault."""


class _ServiceError(RuntimeError):
    """A failure that belongs to ONE service (its bootstrap, its start): the host fails
    as before, but the fault is booked on that service's backend (R-K1)."""

    def __init__(self, svc: dict, msg: str):
        super().__init__(msg)
        self.svc = svc


def service_bid(svc: dict) -> str:
    """= main.backend_id (recomputed, not imported: hostctl never imports main)."""
    return f'{svc.get("type", "openai")}:{svc["name"]}'


# ── controller ───────────────────────────────────────────────────────────────

class Controller:
    """Lifecycle of one managed host and its attached services. The start and stop
    paths build on `_set_phase` (persist on every change) and `_log` (the panel's ring).

    `host` = `{"name", "provider", "options", "api_key"}` (the provider's pure module and
    API class come from `hostapi.provider`; an unknown provider is a ValueError — such a
    host is shown, never driven). `services` = the attached backend dicts, each with
    `local_port` (the gateway's end of its forward) and `remote_port` (the service's
    loopback port on the VM); `set_services` replaces the list after every rebuild."""

    def __init__(self, host: dict, services: list, deps: Deps):
        prov = hostapi.provider((host or {}).get("provider"))
        if prov is None:
            raise ValueError(f"managed host {(host or {}).get('name')!r}: unknown provider "
                             f"{(host or {}).get('provider')!r}")
        self.host = host
        self._prov, self._api_cls = prov
        self._Error = self._api_cls.Error              # the provider's API error class
        self.deps = deps
        self.services: list = []
        self._api = None
        self._client: Optional[httpx.AsyncClient] = None
        self._tunnel = None
        # what the running master carries: set per spawn from its argv (the forwards it
        # was started with), then kept in step by every control forward/cancel
        self._fwd_active: set = set()
        # forwards `control forward` could not add (a local port taken): left out of a
        # respawned master's argv — with ExitOnForwardFailure one of them would take
        # the WHOLE tunnel down — and retried through the control socket instead
        self._fwd_failed: set = set()
        self._ctl_used = ""                            # the socket path the master got
        self._fwd_task: Optional[asyncio.Task] = None  # a pending forward reconcile
        self._fwd_dirty = False
        self._plan_bid: Optional[str] = None           # the ComfyUI service the plan is for
        self._snaps: Optional[list[dict]] = None      # last /snapshots/list, for view()
        self._persist_blocked = False
        self._persist_error = ""                       # the last failed save ("" = saved)
        self._op: Optional[str] = None                 # start/restart/stop/resume in flight
        self._op_task: Optional[asyncio.Task] = None   # the abortable op's task (Ruling 13)
        self._aborted: Optional[asyncio.Task] = None   # the op task stop() cancelled
        self._drain_waiting: Optional[dict] = None     # {bid: jobs} a draining stop waits for
        self._orphans: list[dict] = []                 # last orphans() answer, for view()
        self._resume_pending = False                   # resume() could not reach the API
        self._pending_misses = 0                       # watcher rounds without the pending row
        self._abort_note = ""                          # why an aborted create is uncertain
        self._nodes_text = ""                          # node list of the pending bootstrap
        # model sync: `plan`/`ready_aliases` are written by sync_once() ONLY
        self.plan: Optional[dict] = None
        self.ready_aliases: set = set()
        self._manifest: dict = {}                      # the manifest as last read/written
        self._unsaved: dict = {}                       # manifest entries a write lost
        self._sig: Optional[str] = None                # alias_signature the plan is for
        self._sync_lock = asyncio.Lock()
        self._manifest_lock = asyncio.Lock()
        self._fetches: dict[str, asyncio.Task] = {}    # plan path → its transfer task
        self._lan_path: Optional[str] = None           # the ONE LAN stream's plan path
        self._lan_gen: Optional[int] = None            # LanSource.generation the plan saw
        self._failed: dict[str, str] = {}              # plan path → why it gave up
        self._kicker: Optional[asyncio.Task] = None    # re-plan after a finished transfer
        self._syncs: set = set()                       # running sync_once bodies (stop cancels)
        self._plan_inputs: Optional[tuple] = None      # last _compute_plan inputs (re-plan)
        self._head_sizes: dict = {}                    # url → Content-Length (None = unknown)
        self._dirty = False
        self._sync_error = ""
        self._last_try = 0.0
        self._account_at: Optional[float] = None       # last account refresh (run_forever)
        self._own_absent = 0                           # account refreshes without our uuid
        self.state = State()
        try:
            loaded = deps.load_state(self.name)
        except Exception as e:
            self._load_failed(f"state load failed: {e!r}")
        else:
            if loaded is None:
                pass                    # no record: this host never had an instance
            elif not isinstance(loaded, dict):
                self._load_failed(f"state load failed: stored entry is a "
                                  f"{type(loaded).__name__}, not a dict")
            else:
                self.state = state_from(loaded)
        self.set_services(services)

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

    @property
    def op(self) -> Optional[str]:
        """The operation in flight (start/stop/restart/resume), None when idle."""
        return self._op

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
        """The HOST name — the identity of the state record, the snapshots and the
        control socket (R-W5: never renamed)."""
        return str(self.host["name"])

    @property
    def kind(self) -> str:
        """The provider kind (`thunder`): names the key file and the known_hosts dir."""
        return str(self._prov.KIND)

    @property
    def cfg(self) -> dict:
        """The provider options of the host (GPU, vCPUs, template, reserve, nodes …)."""
        o = self.host.get("options")
        return o if isinstance(o, dict) else {}

    def _token(self) -> str:
        return str(self.host.get("api_key") or "")

    def _host_backend(self) -> dict:
        """The pseudo backend host events are booked on (R-K1): the fault log groups by
        backend + type, so a host's faults form their own group next to its services'."""
        return {"name": self.name, "type": HOST_FAULT_TYPE}

    @property
    def api(self):
        """One client for the controller's lifetime; a new token (saved in the console)
        gets a new API object on the same client."""
        token = self._token()
        if self._api is None or self._api._token != token:
            if self._client is None:
                self._client = self.deps.client_factory()
            self._api = self._api_cls(self._client, token)
        return self._api

    # attached services
    def set_services(self, services) -> None:
        """The attached backends as main sees them now (every rebuild hands NEW dicts —
        kept as given, so the controller always reads the current ones). While the
        tunnel runs, a changed forward set is applied through the master's control
        socket in the background, never by restarting the tunnel (R-W1)."""
        self.services = [x for x in (services or []) if isinstance(x, dict) and x.get("name")]
        self._kick_forwards()

    def service_bids(self) -> list:
        return [service_bid(x) for x in self.services]

    def has_service(self, bid: str) -> bool:
        return bid in self.service_bids()

    def _comfy(self) -> Optional[dict]:
        """The ComfyUI service — one per host (spec; the console refuses a second), so
        the first one is THE one: the bootstrap, the model sync and the routing gate
        are about it."""
        return next((x for x in self.services if x.get("type") == "comfyui"), None)

    def _comfy_bid(self) -> Optional[str]:
        c = self._comfy()
        return service_bid(c) if c is not None else None

    @staticmethod
    def _ports(svc: dict) -> Optional[tuple]:
        """(local, remote) of a service, None when either is no port."""
        try:
            lp, rp = svc.get("local_port"), svc.get("remote_port")
            sshrun._fwd(lp, rp)             # the one port rule (ints 1–65535, no bools)
        except (TypeError, ValueError):
            return None
        return lp, rp

    def _svc_url(self, svc: dict) -> str:
        p = self._ports(svc)
        return f"http://127.0.0.1:{p[0]}" if p else ""

    def _forwards(self) -> list:
        """(local, remote) per attached service with valid ports, in list order. A
        local port an EARLIER service already forwards elsewhere is left out: sshrun
        refuses one port for two targets, and the whole tunnel would never start."""
        out: list = []
        seen: dict = {}
        for x in self.services:
            p = self._ports(x)
            if p is None or p[0] in seen:
                continue
            seen[p[0]] = p[1]
            out.append(p)
        return out

    def _svc_set(self, svc: dict, status: str, error: str = "") -> None:
        """Record a service's status (persisted — `setup_hash` rides along); logged only
        when it changes, so a retried probe does not fill the ring."""
        bid = service_bid(svc)
        cur = self.state.services.get(bid)
        cur = dict(cur) if isinstance(cur, dict) else {}
        new = dict(cur, status=status, error=str(error or ""))
        new.setdefault("setup_hash", "")
        if new != cur:
            self.state.services[bid] = new
            self._log(f"service {bid}: {status}" + (f" — {error}" if error else ""))
            self._persist()

    def _services_view(self) -> dict:
        out: dict = {}
        for x in self.services:
            bid = service_bid(x)
            st = self.state.services.get(bid)
            st = st if isinstance(st, dict) else {}
            p = self._ports(x)
            out[bid] = {"name": str(x.get("name")), "type": str(x.get("type") or "openai"),
                        "local_port": p[0] if p else x.get("local_port"),
                        "remote_port": p[1] if p else x.get("remote_port"),
                        "status": str(st.get("status") or "down"),
                        "error": str(st.get("error") or "")}
        return out

    async def aclose(self) -> None:
        """Gateway shutdown: end the tunnel process and the HTTP client. The INSTANCE is
        not touched — it keeps running and `resume()` picks it up again."""
        await self._stop_tunnel()
        t, self._fwd_task = self._fwd_task, None
        if t is not None and not t.done():
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
        # local tasks only: the curls on the instance run on and are adopted by the
        # next gateway process through their lockfiles
        await self._cancel_sync_tasks()
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client, self._api = None, None

    # persistence / log
    def _persist(self) -> bool:
        """Save the state. → whether it was saved. A failure is remembered in
        `_persist_error` (independent of the state, so it survives a state swap and is
        cleared only by the next good save) and shown on the card: a record that stopped
        being written is invisible otherwise — until a gateway restart forgets a billing
        instance."""
        if self._persist_blocked:
            # never overwrite a record we could not read (see _load_failed)
            self._log("state not saved: stored record unread, waiting for reconcile")
            return False
        d = asdict(self.state)
        for k in _VOLATILE:
            d.pop(k, None)
        try:
            self.deps.save_state(self.name, d)
        except Exception as e:
            # logged, not raised: a store hiccup must not abort a create half-way and
            # leave the instance running with nobody following it up
            stamp = time.strftime("%H:%M:%S", time.localtime(self.deps.now()))
            self._persist_error = f"{stamp} {e!r}"
            self._log(f"state save failed: {e!r}")
            return False
        if self._persist_error:
            self._log("state saved again")
        self._persist_error = ""
        return True

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.deps.now()))
        line = f"{stamp} {msg}"
        self.state.log.append(line)
        if len(self.state.log) > _LOG_MAX:
            del self.state.log[:-_LOG_MAX]
        try:
            self.deps.log(f"[host {self.name}] {msg}")
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
        """One key per PROVIDER (R-W4): `<kind>.key` — for Thunder the file it always was."""
        return os.path.join(self.deps.datadir, f"{self.kind}.key")

    def _known_hosts_path(self, uuid: str) -> str:
        """Per instance: IP, port AND host key change with every instance, so one shared
        file would either refuse the next instance or have to trust any key."""
        return os.path.join(self.deps.datadir, f"{self.kind}-known_hosts",
                            sshrun.safe_rel(uuid))

    def _login(self) -> str:
        return f"{self._prov.SSH_USER}@{self.state.ip}"

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

    def _ctl_path(self) -> str:
        """The tunnel's ControlMaster socket, `<datadir>/<kind>-ctl/<slug>-<hash>`. The
        hash keeps two hosts whose names slug alike apart — a shared path would have one
        master clear the other's LIVE socket as stale — and the slug is cut so the path
        stays short. A data dir so deep that even then the path exceeds what a Unix
        socket can bind (`sshrun.CTL_PATH_MAX`) moves the socket to
        `/tmp/ai-hub-<uid>-ctl/` (Ruling M2) — refused there, the tunnel would respawn
        forever on "path too long"."""
        leaf = (re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-") or "host")[:24]
        leaf += "-" + hashlib.sha256(self.name.encode("utf-8")).hexdigest()[:8]
        path = os.path.join(self.deps.datadir, f"{self.kind}-ctl", leaf)
        if len(path.encode("utf-8")) <= sshrun.CTL_PATH_MAX:
            return path
        return os.path.join(_CTL_FALLBACK.format(uid=os.getuid()), f"{self.kind}-{leaf}")

    def _prepare_ctl(self) -> str:
        """`sshrun.prepare_ctl_path` on `_ctl_path()`. The /tmp fallback directory must be
        OURS — a real directory owned by this uid, no symlink: in a shared /tmp another
        user could have made it first and would then own the socket that grants this
        gateway's session on the VM."""
        path = self._ctl_path()
        d = os.path.dirname(path)
        if not path.startswith(os.path.join(os.path.abspath(self.deps.datadir), "")):
            try:
                st = os.lstat(d)
            except FileNotFoundError:
                pass
            else:
                if (stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode)
                        or st.st_uid != os.getuid()):
                    raise ValueError(f"control socket directory {d!r} is not this "
                                     "gateway's own (symlink or another owner) — not used")
        return sshrun.prepare_ctl_path(path)

    def _tunnel_argv(self) -> list[str]:
        """Called by the Supervisor per spawn, so it follows the CURRENT instance and
        carries EVERY current forward (a respawned master has them all) — and prepares
        the control socket each time (0700 dir; a socket a SIGKILLed master left behind
        is removed, or the new master would run without one)."""
        s = self.state
        if not (s.uuid and s.ip and s.port):
            raise RuntimeError("no instance to tunnel to")
        fwds = [f for f in self._forwards() if f not in self._fwd_failed]
        ctl = self._prepare_ctl()
        argv = sshrun.tunnel_argv(self._key_path(), self._known_hosts_path(s.uuid),
                                  self._login(), s.port, fwds, ctl)
        if ctl != self._ctl_used and os.path.dirname(os.path.dirname(ctl)) != \
                os.path.abspath(self.deps.datadir):
            self._log(f"control socket in {os.path.dirname(ctl)} — the data dir path is "
                      "too long for a Unix socket")
        self._ctl_used = ctl
        self._fwd_active = set(fwds)
        return argv

    def _tunnel_factory(self):
        """The tunnel Supervisor seam (tests replace it per instance)."""
        return sshrun.Supervisor(self._tunnel_argv, lambda m: self._log(m),
                                 spawn=self.deps.spawn or asyncio.create_subprocess_exec)

    async def _start_tunnel(self) -> None:
        """A fresh Supervisor for the current instance. An old one is stopped (and its
        ssh reaped) first: its process holds the local ports, and the new tunnel would
        exit at once on ExitOnForwardFailure."""
        await self._stop_tunnel()
        self._tunnel = self._tunnel_factory()
        self._tunnel.start()

    async def _stop_tunnel(self) -> None:
        t, self._tunnel = self._tunnel, None
        self._fwd_active = set()
        if t is not None:
            await t.stop()

    def _tunnel_error(self) -> str:
        """Why the tunnel's last spawn failed ("" = it did not): shown on the card as
        "tunnel will not come up" — otherwise a tunnel that can never start looks like
        one that restarts now and then (Ruling M3)."""
        t = self._tunnel
        return str(getattr(t, "last_spawn_error", "") or "") if t is not None else ""

    def _kick_forwards(self) -> None:
        """Apply a changed forward set to the running master soon: `set_services` is
        synchronous, the control call is not. One task at a time; a change while it
        runs makes it look once more."""
        self._fwd_dirty = True
        if self._tunnel is None:
            return                      # no master: its next spawn carries the set
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._fwd_task is None or self._fwd_task.done():
            self._fwd_task = asyncio.ensure_future(self._forward_loop())

    async def _forward_loop(self) -> None:
        while self._fwd_dirty:
            self._fwd_dirty = False
            try:
                await self.reconcile_forwards()
            except Exception as e:      # the next change or tick tries again
                self._log(f"forward update failed: {_errtext(e)}")

    async def reconcile_forwards(self) -> None:
        """Bring the RUNNING master's forwards to the attached services' set through its
        control socket — `forward` for a new one, `cancel` for one no longer wanted.
        Never a tunnel restart (R-W1: it would cut every other service's stream). A
        master that is not up has nothing to change: its next spawn carries the current
        set (`_tunnel_argv`). A forward that fails (its local port taken) makes only
        THAT service `down`, with ssh's reason."""
        t, s = self._tunnel, self.state
        if (t is None or not getattr(t, "running", False) or not self._ctl_used
                or not (s.ip and s.port)):
            return
        want = self._forwards()
        self._fwd_failed &= set(want)
        by_fwd = {self._ports(x): x for x in self.services}
        for lp, rp in sorted(self._fwd_active - set(want)):
            rc, why = await self.deps.control(self._ctl_used, self._login(), "cancel", lp, rp)
            if rc == 0:
                self._fwd_active.discard((lp, rp))
                self._log(f"forward {lp} → {rp} removed")
            else:
                self._log(f"removing forward {lp} → {rp} failed (rc {rc}): {why}")
        for lp, rp in want:
            if (lp, rp) in self._fwd_active:
                continue
            rc, why = await self.deps.control(self._ctl_used, self._login(), "forward", lp, rp)
            svc = by_fwd.get((lp, rp))
            if rc == 0:
                self._fwd_active.add((lp, rp))
                self._fwd_failed.discard((lp, rp))
                self._log(f"forward {lp} → {rp} added")
            elif (lp, rp) not in self._fwd_active:
                # (a master respawned meanwhile carries it already — then it is no failure)
                self._fwd_failed.add((lp, rp))
                if svc is not None:
                    self._svc_set(svc, "down",
                                  f"forward {lp} → {rp} failed: {why or f'rc {rc}'}")

    # prices / snapshots for the panel
    async def refresh_prices(self) -> None:
        """Fetch (or reuse, 1 h) pricing + specs. Display only: a failure is logged."""
        try:
            await self.api.pricing()
            await self.api.specs()
        except self._Error as e:
            self._log(f"price list unavailable ({e.status or 'transport'}): {e}")

    async def refresh_snapshots(self) -> Optional[list[dict]]:
        try:
            self._snaps = await self.api.snapshots()
        except self._Error as e:
            self._log(f"snapshot list unavailable ({e.status or 'transport'}): {e}")
        return self._snaps

    async def refresh_account(self) -> None:
        """The account view the console renders from caches only: the price list (1 h
        cache), the snapshot list and the foreign instances. Every call is display only
        and logs its own failure. No token → nothing (a 401 every 10 min says nothing)."""
        if not self._token():
            return
        await self.refresh_prices()
        await self.refresh_snapshots()
        # not while an op runs (a start between create and the persisted uuid would list
        # our own instance as foreign) nor on an unreconciled state (resume() owns the
        # orphan list then — it names them as possibly ours)
        foreign = self._op is None and not self._persist_blocked and not self._resume_pending
        live = self.state.phase in _LIVE_PHASES and bool(self.state.uuid)
        items = None
        if live or foreign:
            try:
                items = await self.api.list_instances()
            except self._Error as e:
                self._log(f"instance list unavailable ({e.status or 'transport'}): {e}")
        if live and items is not None:
            self._check_own_listed(items)
        elif not live:
            self._own_absent = 0
        if foreign and items is not None:           # a failed list keeps the last answer
            await self.orphans(items)

    def _check_own_listed(self, items: list[dict]) -> None:
        """Our instance must be in the account's list while we think it runs. Absent
        from `_ABSENT_CONFIRM` consecutive refreshes → logged and a fault, once per
        disappearance — the phase stays: ending it (snapshot? delete?) is the stop
        path's call, and one odd list must never make a billing instance look gone."""
        uuid = self.state.uuid
        if any(it.get("uuid") == uuid and not self._prov.is_gone_status(it.get("status"))
               for it in items):
            self._own_absent = 0
            return
        self._own_absent += 1
        if self._own_absent == _ABSENT_CONFIRM:
            msg = (f"instance {uuid} is no longer listed at Thunder (phase "
                   f"{self.state.phase}) — deleted outside the gateway? Stop clears it")
            self._log(msg)
            self._fault(None, "lifecycle", "instance_vanished", msg)

    def _fault(self, svc: Optional[dict], source: str, kind: str, detail: str) -> None:
        """Book a fault (R-K1): on the service's backend when `svc` is given (its start,
        its bootstrap, its transfers), else on the host's pseudo backend (create,
        snapshot, delete, instance gone). Never raises: the fault log is a record, not
        a reason to stop."""
        try:
            self.deps.note_fault(svc if svc is not None else self._host_backend(),
                                 source, kind, detail)
        except Exception as e:
            self._log(f"fault log unavailable: {e!r}")

    def snapshots(self) -> list[dict]:
        """The last `/snapshots/list` answer (a copy; [] before the first)."""
        return [dict(x) for x in (self._snaps or [])]

    def pricing_table(self) -> Optional[dict]:
        """The cached `/v2/pricing` table (any age), None before the first fetch."""
        api = self._api
        pricing = api.cached("pricing") if api is not None else None
        if not isinstance(pricing, dict):
            return None
        return pricing.get("pricing") if isinstance(pricing.get("pricing"), dict) else pricing

    def _orphan_view(self, o: dict) -> dict:
        """A foreign instance with its $/h from ITS OWN configuration (the list's
        gpuType/numGpus/cpuCores/storage) — None when the price list lacks it."""
        out = dict(o)
        gpu, n = str(o.get("gpu_type") or ""), int(o.get("num_gpus") or 1)
        table = self.pricing_table()
        api = self._api
        specs = api.cached("specs") if api is not None else None
        out["cost_per_h"] = (self._prov.hourly_cost(table, gpu, n, int(o.get("cpu_cores") or 0),
                                                 int(o.get("storage") or 0),
                                                 self._prov.spec_for(specs, gpu, n))
                             if (gpu and table is not None) else None)
        return out

    def cost_per_h(self) -> Optional[float]:
        api = self._api
        specs = api.cached("specs") if api is not None else None
        table = self.pricing_table()
        if table is None:
            return None
        cfg = self.cfg
        gpu, n = str(cfg.get("gpu_type") or ""), int(cfg.get("num_gpus") or 1)
        return self._prov.hourly_cost(table, gpu, n, int(cfg.get("vcpus") or 0),
                                   self.state.disk_gb, self._prov.spec_for(specs, gpu, n))

    def _snapshot_view(self) -> dict:
        s = self.state
        row = next((x for x in (self._snaps or []) if x.get("id") == s.snapshot_id), None)
        gb = (row or {}).get("min_disk_gb") or None
        monthly = None
        table = self.pricing_table()
        if gb and table is not None:
            monthly = self._prov.snapshot_monthly(table, gb)
        return {"id": s.snapshot_id, "pending": s.pending_snapshot,
                "pending_name": s.pending_snapshot_name,
                "name": (row or {}).get("name", ""), "status": (row or {}).get("status", ""),
                # Thunder reports no snapshot size; its minimum restore disk is the
                # closest figure it gives, so $/month is an upper bound
                "gb": gb, "monthly": monthly}

    def view(self) -> dict:
        s = self.state
        running = s.phase != "off" and s.started_at > 0
        uptime = max(0, int(self.deps.now() - s.started_at)) if running else 0
        cph = self.cost_per_h()
        # spec "Kosten-Wächter": an instance up for more than a day is almost always one
        # somebody forgot — the panel and the Dashboard key their banner on this
        return {"name": self.name, "provider": self.kind, "phase": s.phase, "error": s.error,
                "failed_phase": s.failed_phase, "index": s.index, "uuid": s.uuid,
                "ip": s.ip, "port": s.port, "started_at": s.started_at,
                "uptime_s": uptime, "long_running": running and uptime > _LONG_RUN_S,
                "disk_gb": s.disk_gb, "cost_per_h": cph,
                "session_cost": (cph * uptime / 3600) if (cph is not None and running) else None,
                "snapshot": self._snapshot_view(), "log": list(s.log[-_LOG_MAX:]),
                "transfers": [dict(v) for _, v in sorted(s.transfers.items())],
                "plan": self._plan_view(), "ready_aliases": sorted(self.ready_aliases),
                "sync_error": self._sync_error, "persist_blocked": self._persist_blocked,
                "persist_error": self._persist_error,
                "bootstrap_unknown": dict(s.bootstrap_unknown),
                "bootstrap_template_nodes": list(s.bootstrap_template_nodes),
                "bootstrap_incomplete": s.bootstrap_incomplete,
                "host_bootstrapped": s.host_bootstrapped,
                "op": self._op,
                # {bid: jobs} while a stop drains, else None
                "waiting_jobs": (dict(self._drain_waiting)
                                 if self._drain_waiting is not None else None),
                "services": self._services_view(),
                "tunnel_error": self._tunnel_error(),
                "unreconciled_uuids": list(s.unreconciled_uuids),
                "orphans": [self._orphan_view(x) for x in self._orphans]}

    # lifecycle
    def _refuse_if_unreconciled(self) -> None:
        if self._persist_blocked:
            raise RuntimeError(f"state not loaded ({self.state.error or 'unreadable'}) — "
                               "an instance may still be running; resume first")

    def _refuse_if_busy(self) -> None:
        if self._op is not None:
            raise RuntimeError(f"already {self._op}")

    def _fail(self, msg: str, svc: Optional[dict] = None) -> None:
        """An instance exists (and bills): `failed(<phase>)`, never `off` — the stop
        path needs the uuid to snapshot and delete it. The fault log keeps it after the
        panel has moved on — on the service's backend when the failure was that
        service's (`_ServiceError`), else on the host."""
        self._set_phase("failed", msg)
        self._fault(svc, "lifecycle", "error", msg)

    def _cfg_int(self, key: str, default: int) -> int:
        v = self.cfg.get(key)
        try:
            return int(v) if v is not None and v != "" else default
        except (TypeError, ValueError):
            raise _PreCreate(f"thunder.{key} is not a number: {v!r}")

    async def _required_bytes_hint(self, snapshot_id: str = "") -> int:
        """Bytes the aliases' models need on the new disk (spec "Start" 2): a plan
        against the source index, with the manifest of the snapshot the instance is
        started from as the destination (its URL files carry their measured size). A
        size nobody knows yet (a URL never downloaded) counts 0, and so does a host
        without a ComfyUI service (nothing is synced there). Never raises: a broken
        input sizes the disk by the other terms, it must not stop a start."""
        bid = self._comfy_bid()
        if bid is None:
            return 0
        try:
            man = normalize_manifest(self.state.manifests.get(snapshot_id) or {})
            dest = {k: v["size"] for k, v in man.items() if isinstance(v.get("size"), int)}
            def inputs():                       # blocking store reads: off the loop
                return (self.deps.alias_needs(bid) or [], self.deps.source_index() or {},
                        self.deps.url_catalog() or {})
            needs, src, urls = await asyncio.to_thread(inputs)
            p = modelsync.plan(needs, self._with_head_sizes(src, urls), dest, man, urls)
            return int(p["need_total"])
        except Exception as e:
            self._log(f"model size estimate unavailable: {_errtext(e)}")
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
        reuses indices) and, where the list carries a readable `createdAt`, was created
        around our request; its uuid is then adopted and persisted."""
        s = self.state
        if s.uuid:
            return next((it for it in items if it.get("uuid") == s.uuid), None)
        if not s.index:
            return None
        it = next((x for x in items if x.get("index") == s.index), None)
        if (it is None or not it.get("uuid") or it["uuid"] in self._known_uuids()
                or not s.created_template
                or str(it.get("template") or "").lower() != s.created_template.lower()
                or not self._created_near(it.get("created_at"))):
            return None
        s.uuid = it["uuid"]
        self._log(f"instance at index {s.index} adopted: uuid {s.uuid}")
        self._persist()
        return it

    def _created_near(self, created) -> bool:
        """Was an instance with this `createdAt` created around our create request?
        Epoch seconds or milliseconds, or an ISO timestamp; anything unreadable (or no
        request time) is no evidence either way → True, the template check stands
        alone. A LATER instance at a reused index is what this excludes: ours must have
        disappeared for a stranger to get the index, so its createdAt lies after it."""
        t0 = self.state.create_requested_at
        if not t0 or created in (None, ""):
            return True
        t = None
        try:
            t = float(created)
            if t > 1e12:
                t /= 1000.0     # milliseconds
        except (TypeError, ValueError):
            try:
                from datetime import datetime
                t = datetime.fromisoformat(str(created).replace("Z", "+00:00")).timestamp()
            except (TypeError, ValueError):
                return True
        return t0 - _CREATED_SKEW_S <= t <= t0 + _CREATED_LATE_S

    def _known_uuids(self) -> set:
        try:
            return set(self.deps.known_uuids() or ())
        except Exception:
            return set()

    async def _fresh_item(self) -> dict:
        it = self._find_ours(await self.api.list_instances())
        if it is None:
            raise self._Error(
                f"instance {self.state.uuid or self.state.index} is not in /instances/list",
                None)
        if self._prov.is_gone_status(it.get("status")):
            raise self._Error(f"instance {it.get('uuid')} is {it.get('status')}", None)
        return it

    async def _ensure_ports_closed(self, item: dict) -> dict:
        """Port guard (spec "Start", hard): Thunder's HTTP port forwarding is PUBLIC
        without auth, and ComfyUI behind it means code execution (Manager) and file
        reads (`/view`) for anyone. `item` must come from a fresh list (Ruling 10).
        Open ports are removed and the list read AGAIN — a 200 on the PATCH is not
        proof — and ports still open raise, so `bootstrapping`/`starting` are never
        entered with a public port. → the fresh item."""
        ports = self._prov.ports_open(item)
        if not ports:
            return item
        self._log(f"public http ports {ports} open on the instance — removing")
        await self.api.remove_ports(item, ports)
        fresh = await self._fresh_item()
        still = self._prov.ports_open(fresh)
        if still:
            raise self._Error(
                f"public http ports {still} still open after removing them — "
                "ComfyUI is not started behind a public port", None)
        self._log("public http ports closed")
        return fresh

    def _ssh_argv(self, cmd: str) -> list[str]:
        s = self.state
        return sshrun.exec_argv(self._key_path(), self._known_hosts_path(s.uuid),
                                self._login(), s.port, cmd)

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
            except self._Error as e:
                it, last_err = None, str(e)
                self._log(f"instance list failed while waiting: {e}")
            if it is not None:
                status = it.get("status") or "?"
                if self._prov.is_gone_status(status):
                    raise self._Error(f"instance {it.get('uuid')} became {status} "
                                               "while starting", None)
                if status == "RESTORING" and self.state.phase != "restoring":
                    self._set_phase("restoring")
                if self._prov.is_running(status) and it.get("ip") and it.get("port"):
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

    async def _run_script(self, script: bytes, args: str, logfile: str,
                          timeout: int) -> tuple[int, str, bytes]:
        """Stream a bootstrap script to `bash -s` on the instance, tee'd to `logfile`
        there. → (rc, stdout text, stderr); the stdout comes from the log file when the
        ssh lost it (a timeout keeps nothing, a dropped connection little). Every
        non-blank line goes to the panel log."""
        inner = f"bash -s --{(' ' + args) if args else ''} 2>&1 | tee {logfile}"
        rc, out, err = await self._exec(f"bash -o pipefail -c {sshrun.q(inner)}",
                                        stdin=script, timeout=timeout)
        text = (out or b"").decode("utf-8", "replace")
        if rc != 0 and not text.strip():
            # timeout (sshrun.run keeps nothing) or a dropped connection: the log on
            # the instance still says how far it got
            text = await self._bootstrap_log_tail(logfile)
        for line in text.splitlines():
            if line.strip():
                self._log(line)
        return rc, text, err or b""

    async def _host_bootstrap(self) -> None:
        """`ops/host-bootstrap.sh` (R-W3): every host, first. Its failure is the HOST's
        (`failed(bootstrapping)`, the instance kept) — no service can be trusted on a box
        whose template may still run its own ComfyUI on every interface."""
        s = self.state
        self._log("host bootstrap: template autostart, template inventory, tools")
        rc, text, err = await self._run_script(self.deps.host_bootstrap_script(), "",
                                               _HOST_BOOTSTRAP_LOG, _HOST_BOOTSTRAP_S)
        rep = parse_bootstrap(text)
        for line in rep["bad"]:
            self._log(f"host bootstrap: unreadable report line ignored: {line!r}")
        s.bootstrap_unknown = dict(rep["unknown"])
        s.bootstrap_template_nodes = list(rep["template_nodes"])
        self._persist()
        why = bootstrap_verdict(rc, rep, err.decode("utf-8", "replace") + "\n" + text,
                                smoke_test=False, timeout_s=_HOST_BOOTSTRAP_S)
        if why:
            for line in _tail(err):
                self._log(f"stderr: {line}")
            raise RuntimeError(f"host bootstrap: {why}")
        if s.bootstrap_unknown:
            gb = sum(s.bootstrap_unknown.values()) / 1024 ** 3
            self._log(f"host bootstrap: {len(s.bootstrap_unknown)} template model file(s), "
                      f"{gb:.1f} GB — delete them before the first stop or every "
                      "snapshot carries them")
        s.host_bootstrapped = True
        self._persist()
        self._log("host bootstrap done")

    async def _upload_nodes(self) -> None:
        """The ComfyUI bootstrap's node list → `~/.gw-nodes.txt`. Before the host
        bootstrap when both run: its template-node report leaves our packs out."""
        rc, _, err = await self._exec("cat > ~/.gw-nodes.txt",
                                      stdin=self._nodes_text.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"node list upload failed (rc {rc}): "
                               + " | ".join(_tail(err, 3)))

    async def _bootstrap(self) -> None:
        """`ops/thunder-bootstrap.sh` — the ComfyUI part; the node list is uploaded."""
        s = self.state
        self._log(f"bootstrap: ComfyUI {self._commit()[:12]}, node list uploaded")
        rc, text, err = await self._run_script(self.deps.bootstrap_script(),
                                               sshrun.q(self._commit()), _BOOTSTRAP_LOG,
                                               _BOOTSTRAP_S)
        rep = parse_bootstrap(text)
        for line in rep["bad"]:
            self._log(f"bootstrap: unreadable report line ignored: {line!r}")
        # the script's own last line names the cause; ssh's stderr only when it has none
        why = bootstrap_verdict(rc, rep, err.decode("utf-8", "replace") + "\n" + text)
        if why:
            for line in _tail(err):
                self._log(f"stderr: {line}")
            raise RuntimeError(why)
        s.bootstrap_incomplete = False
        self._persist()
        self._log("bootstrap done")

    def _comfy_pending(self) -> bool:
        """A ComfyUI service is attached and this instance lacks a finished ComfyUI
        bootstrap (an unfinished one, or none because no ComfyUI was attached before)."""
        s = self.state
        return self._comfy() is not None and (s.bootstrap_incomplete or s.comfy_absent)

    async def _ensure_comfy_bootstrap(self, upload: bool = True) -> bool:
        """Run the ComfyUI bootstrap when a ComfyUI service is attached and the instance
        lacks it — at a start right after the host bootstrap, or on a RUNNING host a
        ComfyUI service was just attached to (Task 5 wires the attach). → True when it
        ran (and succeeded), False when there was nothing to do. A failure raises
        `_ServiceError` (fault + `setup failed` on the service); the caller decides what
        it means for the host — today the start fails the whole host (Task 5 makes it
        the service's alone). `upload=False`: the node list is already on the box."""
        comfy = self._comfy()
        if not self._comfy_pending():
            return False
        s = self.state
        # from here on a snapshot of this instance holds a (half-)install: "incomplete"
        s.bootstrap_incomplete, s.comfy_absent = True, False
        self._persist()
        try:
            if not self._nodes_text:
                self._nodes_text = self._node_list()     # lost with a restart
            if upload:
                await self._upload_nodes()
            await self._bootstrap()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # the bootstrap is the ComfyUI service's setup: its fault (R-K1)
            self._svc_set(comfy, "setup failed", _errtext(e))
            raise _ServiceError(comfy, _errtext(e)) from e
        return True

    async def _bootstrap_log_tail(self, logfile: str = _BOOTSTRAP_LOG) -> str:
        try:
            rc, out, err = await self._exec(f"tail -n {_BOOTSTRAP_TAIL} {logfile}",
                                            timeout=60)
        except Exception as e:
            self._log(f"bootstrap log unavailable: {_errtext(e)}")
            return ""
        if rc != 0:
            self._log(f"bootstrap log unavailable (rc {rc}): " + " | ".join(_tail(err, 3)))
            return ""
        self._log(f"bootstrap output lost — last {_BOOTSTRAP_TAIL} lines of "
                  f"{logfile} follow")
        return (out or b"").decode("utf-8", "replace")

    # services: start / probe / restart (ComfyUI only in this version — the provisional
    # internal profile; command services join with services.py)
    async def _wait_comfy(self, svc: dict, settle: bool = False) -> None:
        """Probe a ComfyUI service through its forward every 3 s until it answers.
        `settle`: wait one interval first — right after a pkill the old process may
        still answer."""
        url = self._svc_url(svc)
        deadline = self.deps.now() + _COMFY_READY_S
        if settle:
            await self.deps.sleep(_COMFY_PROBE_S)
        while True:
            try:
                ok = bool(await self.deps.probe_comfy(url))
            except Exception:
                ok = False              # tunnel not up yet, ComfyUI still importing
            if ok:
                self._log("ComfyUI answers")
                return
            if self.deps.now() >= deadline:
                raise TimeoutError(f"ComfyUI did not answer on {url} within "
                                   f"{_COMFY_READY_S // 60} min (see ~/comfy.log)")
            await self.deps.sleep(_COMFY_PROBE_S)

    async def _start_comfy(self, svc: dict, restart: bool = False) -> None:
        """Start (or, `restart`, kill and restart) one ComfyUI service and wait until it
        answers. Its status follows (`starting` → `up`, else `down` with the reason); a
        failure raises `_ServiceError` — the host fails as before, the fault is the
        service's."""
        p = self._ports(svc)
        if p is None:
            why = "no valid local/remote port"
            self._svc_set(svc, "down", why)
            raise _ServiceError(svc, f"{service_bid(svc)}: {why}")
        self._svc_set(svc, "starting")
        try:
            rc, _, err = await self._exec(_restart_cmd(p[1]) if restart else _START_CMD,
                                          timeout=60)
            if rc != 0:
                raise RuntimeError(f"starting ComfyUI failed (rc {rc}): "
                                   + " | ".join(_tail(err, 3)))
            await self._wait_comfy(svc, settle=restart)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._svc_set(svc, "down", _errtext(e))
            raise _ServiceError(svc, _errtext(e)) from e
        self._svc_set(svc, "up")

    async def _start_services(self, restart: bool = False,
                              only: Optional[list] = None) -> None:
        """Every attached service (or just `only`), in list order. A service type
        without a start profile yet (anything but ComfyUI in this version) is left
        `down` with that reason — it does not fail the host."""
        for svc in list(self.services if only is None else only):
            if svc.get("type") == "comfyui":
                await self._start_comfy(svc, restart=restart)
            else:
                self._svc_set(svc, "down", f"no start profile for type "
                                           f"{svc.get('type') or 'openai'!r} yet")

    # ops: one lifecycle operation at a time
    async def _run_op(self, name: str, coro) -> None:
        """Run `coro` as THE operation of this controller, in its own task, so a stop()
        can cancel exactly it (Ruling 13) without cancelling whoever awaits it. A cancel
        that stop() issued is swallowed here — stop owns the outcome from then on; any
        other cancel (gateway shutdown) propagates."""
        self._op = name
        t = asyncio.ensure_future(coro)
        self._op_task = t
        try:
            await t
        except asyncio.CancelledError:
            if not (t.cancelled() and self._aborted is t):
                raise
        finally:
            if self._op_task is t:
                self._op, self._op_task = None, None

    async def start(self) -> None:
        """Spec "Start" 0–6 (`starting` → `syncing` → first plan → `ready`).

        Refusals (unreconciled state, an instance already known, no attached service, a
        bad commit, a start in flight) RAISE before anything happens. Everything else ends in the
        state: a failure before `create` → `off` with the reason (nothing bills), a
        failure after it → `failed(<phase>)` with the instance KEPT (diagnosis; the
        stop path removes it). The uuid is persisted before the first wait, so a
        gateway restart during the up to ~30 min to RUNNING still knows the instance.
        A stop() meanwhile aborts it and takes over from the phase it reached."""
        self._refuse_if_unreconciled()
        self._refuse_if_busy()
        s = self.state
        if s.phase != "off" and not (s.phase == "failed" and not s.uuid and not s.index):
            # `failed` with an index but no uuid still names an instance (Ruling 10)
            raise RuntimeError(f"already {s.phase}"
                               + (f" (instance {s.uuid or s.index})" if s.uuid or s.index else ""))
        if not self.services:
            # a machine serving nothing would bill for nothing
            raise RuntimeError(_NO_SERVICE.format(host=self.name))
        if self._comfy() is not None:
            self._commit()              # only the ComfyUI bootstrap pins a commit
        if not self._token():
            # every provider call would be a 401; say what to do instead of showing it
            raise RuntimeError(_NO_TOKEN.format(name=self._prov.NAME))
        self._abort_note = ""
        await self._run_op("starting", self._checked_start())

    async def _checked_start(self) -> None:
        """The start op: the unreconciled check INSIDE it, so the op's task exists from
        the first await on — a stop() landing during the check aborts exactly this task
        (and nothing is created) instead of finding an op without a task, answering
        "stop done" and letting the start go on to create an instance. A refusal of the
        check raises out of start() like the other refusals (and is logged: it arrives
        after the console already answered "start requested")."""
        if self.state.unreconciled_uuids:
            try:
                await self._check_unreconciled()
            except RuntimeError as e:
                self._log(f"start refused: {e}")
                raise
        await self._start()

    async def _check_unreconciled(self) -> None:
        """Refuse a start while an instance seen next to an unreadable state record is
        still listed: it may be this backend's own, and a start would create a second
        one (the risk `_load_failed` exists to prevent). Gone ones are forgotten — but
        only once `_ABSENT_CONFIRM` consecutive fresh lists all lack them (the
        `_find_live` rule): one empty answer is not proof, and forgetting on it would
        let the start create a second instance next to a billing one."""
        s = self.state
        for i in range(_ABSENT_CONFIRM):
            if i:
                await self.deps.sleep(_ABSENT_RECHECK_S)
            try:
                items = await self.api.list_instances()
            except self._Error as e:
                raise RuntimeError(f"cannot check the unreconciled instances "
                                   f"({', '.join(s.unreconciled_uuids)}): {e}") from e
            listed = {it.get("uuid") for it in items
                      if not self._prov.is_gone_status(it.get("status"))}
            still = [u for u in s.unreconciled_uuids if u in listed]
            if still:
                raise RuntimeError(f"instance(s) {', '.join(still)} seen while the stored "
                                   "state was unreadable are still running — one may be "
                                   "this backend's: delete them by hand or forget them first")
        self._log("the unreconciled instances are gone — start allowed again")
        s.unreconciled_uuids = []
        self._persist()

    def forget_unreconciled(self) -> None:
        """Operator reset (panel): the listed unowned instances are not this backend's."""
        if self.state.unreconciled_uuids:
            self._log("unreconciled instances forgotten by the operator: "
                      + ", ".join(self.state.unreconciled_uuids))
            self.state.unreconciled_uuids = []
            self._persist()

    def _enable(self, done: list) -> None:
        """Step 0: EVERY attached backend. One that stays disabled is never polled or
        routed to — an instance created for it would bill for nothing, so any failure
        here ends the start before the create. `done` collects the ones enabled, which
        the caller disables again (and only those — the one that failed never was)."""
        for svc in list(self.services):
            bid = service_bid(svc)
            try:
                ok = self.deps.set_enabled(bid, True)
            except Exception as e:
                raise _PreCreate(f"cannot enable backend {bid}: {_errtext(e)}") from e
            if ok is False:
                raise _PreCreate(f"cannot enable backend {bid}: not a known backend")
            done.append(svc)

    def _disable(self, svcs: Optional[list] = None) -> None:
        """`off` = disabled (spec "Stop" 1): no discovery against a dead tunnel port —
        for every backend attached AT THIS MOMENT (R-K2: one that moved to another host
        or a real URL is that one's business now), or just `svcs`."""
        for svc in list(self.services if svcs is None else svcs):
            bid = service_bid(svc)
            try:
                self.deps.set_enabled(bid, False)
            except Exception as e:
                self._log(f"disabling backend {bid} failed: {e!r}")
            self._svc_set(svc, "down")

    async def _start(self) -> None:
        enabled: list = []
        try:
            self._enable(enabled)
            await self._create()
        except Exception as e:
            msg = _errtext(e)
            if isinstance(e, self._Error) and e.status is None:
                msg += " (an instance may exist anyway — check the orphan list)"
            self._set_phase("off", f"start failed: {msg}")
            if enabled:
                self._disable(enabled)
            return
        await self._after_create()

    async def _after_create(self) -> None:
        """Steps 3–6 from a created instance on: RUNNING → port guard → tunnel (every
        service's forward) → ssh → the host bootstrap (if this instance lacks it) → the
        ComfyUI bootstrap (if needed, and only with a ComfyUI service attached) → each
        service started and probed. What runs is read from the state flags `_created`
        set (`host_bootstrapped`, `bootstrap_incomplete`/`comfy_absent`), so this is
        also how resume() continues a start the gateway restart interrupted before the
        bootstrap."""
        s = self.state
        comfy = self._comfy()
        need_host = not s.host_bootstrapped
        need_comfy = self._comfy_pending()
        try:
            if need_comfy and not self._nodes_text:
                self._nodes_text = self._node_list()     # lost with a restart
            item = await self._wait_running(s.disk_gb)
            s.ip, s.port = str(item["ip"]), int(item["port"])
            self._persist()
            item = await self._ensure_ports_closed(item)
            self._set_phase("connecting")
            self._reset_known_hosts(s.uuid)
            await self._start_tunnel()
            await self._wait_ssh()
            if need_host or need_comfy:
                self._set_phase("bootstrapping")
            if need_comfy:
                try:
                    await self._upload_nodes()      # first: the host bootstrap reads it
                except Exception as e:
                    self._svc_set(comfy, "setup failed", _errtext(e))
                    raise _ServiceError(comfy, _errtext(e)) from e
            if need_host:
                await self._host_bootstrap()        # a failure is the host's
            if need_comfy:
                # the ComfyUI service's setup: a failure is booked on it, but still
                # fails the whole host start here — Task 5 makes it that service's alone
                await self._ensure_comfy_bootstrap(upload=False)
            elif comfy is None and s.comfy_absent:
                self._log("no ComfyUI service attached — ComfyUI bootstrap skipped")
            await self._ensure_ports_closed(await self._fresh_item())
            self._set_phase("starting")
            await self._start_services()
            # spec "Start" 6: the first plan before `ready`; aliases go live one by one
            # as their files arrive (`ready_aliases`), `ready` is the INSTANCE's state
            self._set_phase("syncing")
            self._invalidate_source()           # every start lists the share afresh
            await self.sync_once()
            self._ready()
        except Exception as e:
            self._fail(_errtext(e), getattr(e, "svc", None))

    def _ready(self) -> None:
        """ComfyUI answers: whatever the bootstrap left undone, this instance works — a
        snapshot of it is a good template (an operator who fixed a failed bootstrap by
        hand and restarted ComfyUI has confirmed exactly that)."""
        if self.state.bootstrap_incomplete and self._comfy() is not None:
            self._log("ComfyUI answers — the unfinished bootstrap counts as done now")
            self.state.bootstrap_incomplete = False
        self._set_phase("ready")

    async def _create(self) -> None:
        """Steps 1–3 up to the create: template, disk, key, `POST /instances/create`;
        index/uuid persisted with phase `creating`, together with what the instance
        needs: the host bootstrap (no READY snapshot of ours to restore from, or the
        newest one was taken before it finished) and the ComfyUI bootstrap (with a
        ComfyUI service: the same, or a snapshot of a host that had none). Without a
        snapshot the template is the configured one, else `comfy-ui` with a ComfyUI
        service and the provider's `DEFAULT_TEMPLATE_NO_COMFY` (`base`) without (R-W3)."""
        s, cfg = self.state, self.cfg
        for k in ("gpu_type", "vcpus"):
            if not cfg.get(k):
                raise _PreCreate(f"thunder.{k} is not set")
        num_gpus = self._cfg_int("num_gpus", 1)
        snaps = await self.api.snapshots()
        self._snaps = snaps
        snap = self._prov.newest_ready(snaps, self.name)
        comfy = self._comfy() is not None
        if snap is not None:
            template = snap["name"]
            need_host = snap["id"] in s.host_incomplete_snapshots
            comfy_missing = (snap["id"] in s.incomplete_snapshots
                             or snap["id"] in s.no_comfy_snapshots)
            if need_host:
                self._log(f"snapshot {snap['name']} was taken before its host bootstrap "
                          "finished — the host bootstrap runs again on it")
            if comfy and snap["id"] in s.incomplete_snapshots:
                self._log(f"snapshot {snap['name']} was taken before its bootstrap finished "
                          "— the bootstrap runs again on it")
            elif comfy and comfy_missing:
                self._log(f"snapshot {snap['name']} carries no ComfyUI — the ComfyUI "
                          "bootstrap runs on it")
        else:
            template = str(cfg.get("bootstrap_template") or "").strip() or (
                "comfy-ui" if comfy
                else getattr(self._prov, "DEFAULT_TEMPLATE_NO_COMFY", "comfy-ui"))
            need_host = comfy_missing = True
        # the ComfyUI bootstrap (and its node list) only with a ComfyUI service (R-W3)
        self._nodes_text = self._node_list() if (comfy_missing and comfy) else ""
        needs = (need_host, comfy_missing)
        spec = self._prov.spec_for(await self.api.specs(), str(cfg["gpu_type"]), num_gpus)
        storage = (spec or {}).get("storageGB") if isinstance((spec or {}).get("storageGB"), dict) else {}
        if spec is None:
            self._log(f"no /v2/specs entry for {cfg['gpu_type']} x{num_gpus} — disk "
                      "limits unknown, Thunder decides")
        disk_gb = self._prov.choose_disk_gb(
            required_bytes=await self._required_bytes_hint((snap or {}).get("id") or ""),
            base_bytes=s.base_bytes,
            reserve_gb=self._cfg_int("reserve_gb", 20),
            snapshot_min_gb=(snap or {}).get("min_disk_gb") or 0,
            spec_min=int(storage.get("min") or 0), spec_max=int(storage.get("max") or 0),
            num_gpus=num_gpus)
        pub = await self.deps.keygen(self._key_path())
        self._log(f"creating instance: template {template}, disk {disk_gb} GB"
                  + ("" if snap else " (first start: bootstrap follows)"))
        s.created_template, s.create_requested_at = template, self.deps.now()
        if not self._persist():
            # the POST would make an instance whose record the store never got: a
            # gateway restart could not find (nor stop) it while it bills
            raise _PreCreate(f"state could not be saved ({self._persist_error or 'blocked'}) "
                             "— not creating an instance nobody could find after a restart")
        fut = asyncio.ensure_future(
            self.api.create(self._prov.create_body(cfg, template, disk_gb, pub)))
        try:
            created = await asyncio.shield(fut)
        except asyncio.CancelledError as cancel:
            # stop() aborted the start while the POST was out: the instance may exist
            # and bill already — learn its id before giving up, or nobody deletes it
            try:
                created = await fut
            except Exception as e:
                note = f"create answered {_errtext(e)}"
                if isinstance(e, self._Error) and e.status is None:
                    note += " (an instance may exist anyway — check the orphan list)"
                self._abort_note = note
                self._log(f"create aborted; {note}")
                raise cancel
            self._created(created, disk_gb, snap, needs)
            raise cancel
        self._created(created, disk_gb, snap, needs)

    def _created(self, created: dict, disk_gb: int, snap: Optional[dict],
                 needs: tuple) -> None:
        """`needs` = (host bootstrap needed, ComfyUI install missing). The flags are
        what `_after_create` (also after a gateway restart) runs from, and what a
        snapshot of this instance inherits."""
        # from here on an instance exists and bills: persist it BEFORE any wait
        s = self.state
        need_host, comfy_missing = needs
        s.index, s.uuid = created["index"], created["uuid"]
        s.ip, s.port = "", 0
        s.disk_gb = disk_gb
        s.started_at = self.deps.now()
        s.snapshot_id = snap["id"] if snap else ""
        s.host_bootstrapped = not need_host
        # with a ComfyUI service the missing install is an unfinished bootstrap from the
        # create on; without one it is just absent (nothing will run, nothing failed)
        comfy = self._comfy() is not None
        s.bootstrap_incomplete = comfy_missing and comfy
        s.comfy_absent = comfy_missing and not comfy
        if need_host:
            s.bootstrap_unknown, s.bootstrap_template_nodes = {}, []
        self._set_phase("creating")

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
        if s.phase == "failed" and s.failed_phase in _STOP_STEPS:
            raise RuntimeError(f"a stop did not finish ({s.failed_phase}) — stop again")
        comfy = self._comfy()
        if comfy is None:
            raise RuntimeError(f"no ComfyUI service attached to {self.name}")
        await self._run_op("restarting ComfyUI", self._restart(only=[comfy]))

    async def _restart(self, only: Optional[list] = None) -> None:
        try:
            await self._ensure_ports_closed(await self._fresh_item())
            if self._tunnel is None:
                await self._start_tunnel()
            self._set_phase("starting")
            await self._start_services(restart=True, only=only)
            self._ready()
        except Exception as e:
            self._fail(_errtext(e), getattr(e, "svc", None))

    # ── stop (spec "Stop" 1–4) ──────────────────────────────────────────────

    async def _before_snapshot(self) -> None:
        """Hook of the `pruning` step (spec "Stop" 2) — the ONLY place synced files are
        deleted: a file a removed alias no longer needs stays for the rest of the
        session (the alias may come back in a minute). Ends every download (again: a
        stop resumed after a gateway restart has no transfer tasks, but the curls may
        still run), deletes the plan's `prune` list — manifest files only, never a
        `held` or `unknown` one — and every `.part`/`.part.lock`/`.part.log`, then
        writes the manifest the snapshot carries (`_current_manifest`). Failures are
        logged, never raised: pruning saves money, it must not keep a billing instance
        from being stopped."""
        s = self.state
        if not (s.ip and s.port):
            self._log("model prune skipped: no ssh address known")
            return
        await self._kill_remote()
        async with self._manifest_lock:
            try:
                plan, man = await self._compute_plan()
            except Exception as e:
                plan, man = None, None
                self._log(f"model prune skipped ({_errtext(e)}) — only unfinished "
                          "downloads are removed")
            prune = list(plan["prune"]) if plan else []
            try:
                rc, out, err = await self._exec(_prune_cmd(prune), timeout=_PRUNE_TIMEOUT_S)
            except Exception as e:
                rc, out, err = -1, b"", _errtext(e).encode()
            text = (out or b"").decode("utf-8", "replace")
            removed = bool(prune) and rc == 0 and "GW:RM-OK" in text
            if rc != 0 or "GW:PRUNED" not in text:
                self._log(f"model prune failed (rc {rc}): " + " | ".join(_tail(err, 3)))
            elif prune:
                self._log(f"pruned {len(prune)} model file(s) no alias needs: "
                          + ", ".join(prune[:10]) + (" …" if len(prune) > 10 else ""))
            if man is None:
                return
            new = {k: v for k, v in man.items() if not (removed and k in prune)}
            users: dict = {}
            for a, row in plan["per_alias"].items():
                for f in row["files"]:
                    users.setdefault(f["path"], set()).add(a)
            for k in new:
                if k in users:                  # who needs it NOW (held files keep theirs)
                    new[k] = dict(new[k], aliases=sorted(users[k]))
            try:
                await self._write_manifest(new)
                self._unsaved.clear()
                self._manifest = new
            except Exception as e:
                self._log(f"manifest not written: {_errtext(e)}")
                self._manifest = man

    async def _stop_transfers(self) -> None:
        """Hook of the `draining` step (spec "Stop" 1: transfers end BEFORE the
        snapshot, or it freezes growing `.part`s): the local tasks, then the curls on
        the instance (ended there, by pid — `_kill_remote`)."""
        self._dirty = False
        await self._cancel_sync_tasks()
        self.state.transfers.clear()
        await self._kill_remote()

    async def _cancel_sync_tasks(self) -> None:
        """Cancel and AWAIT every local sync task: the transfers, the kicker and any
        running `sync_once` body — also one `run_forever` started before the phase
        changed. Left running, such a sync grows the disk of an instance being
        snapshotted, or sets `_manifest` back to the pre-prune one after
        `_before_snapshot` wrote the pruned manifest (the snapshot then records it)."""
        tasks = [t for t in list(self._fetches.values()) + [self._kicker] + list(self._syncs)
                 if t is not None and not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._fetches.clear()
        self._lan_path = None

    def _current_manifest(self) -> dict:
        """The model manifest the snapshot carries (`manifests[sid]`): the one
        `_before_snapshot` wrote (else the last one read)."""
        return copy.deepcopy(self._manifest)

    async def stop(self) -> None:
        """Spec "Stop": draining → pruning → snapshotting → deleting → `off`.

        Every step is re-entrant, so the same call finishes a stop a gateway restart
        or a failure interrupted: it starts at the step the state names. A start or
        ComfyUI restart in flight is ABORTED first and the stop runs from the phase it
        reached (Ruling 13) — before `create` that is just `off` with the backend
        disabled again; before any bootstrap ran (`creating|restoring|connecting`) the
        instance holds nothing worth a snapshot and is deleted straight away.

        Refused (raises): an unreconciled state (it may name a billing instance we
        cannot identify), nothing running, a stop already in flight. A failure after
        that ends in `failed(<step>)` with the instance kept — never `off` while it
        may still bill; stop() again resumes from that step."""
        self._refuse_if_unreconciled()
        if self._op == "stopping":
            raise RuntimeError("already stopping")
        prev_op = self._op
        if prev_op in _ABORTABLE and self._op_task is None:
            # an abortable op with no task to cancel: cancelling nothing and answering
            # "stop done" would let it go on (a start would still create) — refuse
            raise RuntimeError(f"{prev_op} is checking Thunder — retry in a moment")
        prev = self._op_task if prev_op in _ABORTABLE else None
        if prev is not None and prev.done():
            prev = None
        if self.state.phase == "off" and prev_op != "starting":
            raise RuntimeError("not running")
        self._op, self._op_task = "stopping", None
        try:
            if prev is not None:
                self._log(f"stop: aborting {prev_op}")
                self._aborted = prev
                prev.cancel()
                await asyncio.wait([prev])
            s = self.state
            if not (s.uuid or s.index):
                # no instance was ever created (or it is long forgotten): nothing bills
                self._disable()
                msg = ""
                if prev is not None and prev_op == "starting":
                    msg = "start aborted before an instance was created"
                    if self._abort_note:
                        msg += f"; {self._abort_note}"
                self._abort_note = ""
                self._set_phase("off", msg)
                return
            await self._stop_run()
        finally:
            self._op = None
            self._drain_waiting = None

    async def _stop_run(self) -> None:
        s = self.state
        p = s.failed_phase if s.phase == "failed" else s.phase
        if p in _STOP_STEPS:
            first = p
            self._log(f"stop: resuming at {p}")
        elif p in _PRE_BOOT:
            first = "deleting"
            self._log(f"stop: instance never got past {p} — nothing to snapshot, deleting it")
        else:
            first = "draining"
        if not s.host_bootstrapped and first != "deleting":
            self._log("stop: the host bootstrap did not finish — the snapshot is marked, "
                      "and a start from it runs the host bootstrap again")
        if s.bootstrap_incomplete and first != "deleting":
            self._log("stop: the bootstrap did not finish (after a timeout it may even "
                      "still run on the instance) — the snapshot may hold a half-finished "
                      "install; it is marked, and a start from it bootstraps again")
        steps = _STOP_STEPS[_STOP_STEPS.index(first):]
        try:
            if "draining" in steps:
                await self._drain()
            if "pruning" in steps:
                await self._prune()
            if "snapshotting" in steps:
                await self._snapshot(fresh_name=first != "snapshotting")
            await self._delete()
        except _Vanished as e:
            await self._gone(str(e), fault=True)
        except Exception as e:
            self._fail(_errtext(e))

    async def _drain(self) -> None:
        """Step 1. The existing drain for EVERY attached backend (routing stops now,
        each backend is disabled once idle — wanted for `off`). The snapshot waits until
        none has a job left AND none is still draining — one busy service keeps the
        whole machine. No timeout: a running job may finish; the panel shows how many
        are left per service (`waiting_jobs`)."""
        self._set_phase("draining")
        svcs = list(self.services)
        for svc in svcs:
            bid = service_bid(svc)
            try:
                started = self.deps.begin_drain(bid)
            except Exception as e:
                # routing would go on sending jobs, and the wait below could never end
                # with nothing saying why
                raise RuntimeError(f"drain of {bid} could not start: {_errtext(e)}") from e
            if not started:
                self._log(f"drain: backend {bid} already offline")
        await self._stop_transfers()
        last = None
        while True:
            waiting: dict = {}
            for svc in svcs:
                bid = service_bid(svc)
                n = int(self.deps.inflight(bid) or 0)
                if n > 0 or self.deps.is_draining(bid):
                    waiting[bid] = max(0, n)
            if not waiting:
                break
            if waiting != last:
                self._log("draining: " + "; ".join(
                    f"waiting for {n} job(s) on {bid}" if n > 0
                    else f"waiting for the drain of {bid} to complete"
                    for bid, n in sorted(waiting.items())))
                last = waiting
            self._drain_waiting = waiting
            await self.deps.sleep(_DRAIN_POLL_S)
        self._drain_waiting = None

    async def _prune(self) -> None:
        """Step 2: the model prune (`_before_snapshot`), then `base_bytes` (everything
        but the model roots) — what the next disk needs besides models. A failed
        measurement keeps the last value: it sizes a disk, it must not stop a stop."""
        s = self.state
        self._set_phase("pruning")
        await self._before_snapshot()
        if not (s.ip and s.port):
            self._log("base size not measured: no ssh address known")
            return
        try:
            rc, out, err = await self._exec(_DU_CMD, timeout=_DU_TIMEOUT_S)
        except Exception as e:
            rc, out, err = -1, b"", _errtext(e).encode()
        text = (out or b"").decode("utf-8", "replace").strip()
        if rc == 0 and text.isdigit():
            s.base_bytes = int(text)
            self._log(f"base install: {s.base_bytes / 1024 ** 3:.1f} GB")
            self._persist()
        else:
            self._log(f"base size not measured (rc {rc}): " + " | ".join(_tail(err, 3)))

    async def _our_item(self) -> Optional[dict]:
        """Our instance in a FRESH list (Ruling 10, via `_find_ours`), None when it is
        gone: not listed, or listed with a gone status. A list error raises."""
        it = self._find_ours(await self.api.list_instances())
        if it is None or self._prov.is_gone_status(it.get("status")):
            return None
        return it

    async def _find_live(self) -> Optional[dict]:
        """`_our_item`, but None only after `_ABSENT_CONFIRM` consecutive fresh lists
        without the instance — a single empty answer never forgets one that bills."""
        for i in range(_ABSENT_CONFIRM):
            if i:
                self._log("instance not listed — checking again")
                await self.deps.sleep(_ABSENT_RECHECK_S)
            it = await self._our_item()
            if it is not None:
                return it
        return None

    def _record_pending(self, sid: str) -> None:
        s = self.state
        s.pending_snapshot = sid
        self._pending_misses = 0
        s.manifests[sid] = self._current_manifest()
        if not s.host_bootstrapped and sid not in s.host_incomplete_snapshots:
            s.host_incomplete_snapshots.append(sid)
            self._log(f"snapshot {s.pending_snapshot_name} marked: host bootstrap incomplete")
        if s.bootstrap_incomplete and sid not in s.incomplete_snapshots:
            s.incomplete_snapshots.append(sid)
            self._log(f"snapshot {s.pending_snapshot_name} marked: bootstrap incomplete")
        if s.comfy_absent and sid not in s.no_comfy_snapshots:
            s.no_comfy_snapshots.append(sid)       # not a fault: no ComfyUI was attached
        self._persist()

    async def _snapshot(self, fresh_name: bool) -> None:
        """Step 3. Idempotent: the name is persisted before the POST, and a snapshot
        that already carries it (a restart between POST and persist, a retry) is taken
        over instead of created twice."""
        s = self.state
        if fresh_name or not s.pending_snapshot_name:
            if s.pending_snapshot:
                # the rotation removes it once the new one is READY (or it failed)
                self._log(f"previous snapshot {s.pending_snapshot} is not READY yet — "
                          "the watcher follows the new one")
            s.pending_snapshot_name = self._prov.snapshot_name(self.name, self.deps.now())
        self._set_phase("snapshotting")
        snaps = await self.api.snapshots()
        self._snaps = snaps
        row = next((x for x in snaps if x.get("name") == s.pending_snapshot_name
                    and x.get("id")), None)
        if row is not None and row.get("status") == "FAILED":
            # never "taken over": the instance would be deleted with no good snapshot
            # of this session — take a new one under a new name
            await self._failed_snapshot_row(row)
            row = None
        if row is not None:
            self._log(f"snapshot {row['name']} exists already ({row['status']}) — not taken twice")
            self._record_pending(row["id"])
            return
        item = await self._find_live()
        if item is None:
            raise _Vanished(f"instance {s.uuid or s.index} vanished before its snapshot "
                            "was taken — this session's changes are lost")
        sid = await self.api.create_snapshot(item, s.pending_snapshot_name)
        self._log(f"snapshot {s.pending_snapshot_name} requested ({sid})")
        self._record_pending(sid)

    async def _failed_snapshot_row(self, row: dict) -> None:
        s = self.state
        self._log(f"snapshot {row['name']} FAILED — taking a new one")
        self._fault(None, "lifecycle", "snapshot_failed", row["name"])
        if s.pending_snapshot == row["id"]:
            s.pending_snapshot = ""
        s.manifests.pop(row["id"], None)
        old = s.pending_snapshot_name
        for _ in range(3):                  # the name has 1-s resolution
            s.pending_snapshot_name = self._prov.snapshot_name(self.name, self.deps.now())
            if s.pending_snapshot_name != old:
                break
            await self.deps.sleep(1)
        self._persist()

    async def _delete(self) -> None:
        """Step 3b. Delete, then CONFIRM through fresh lists that the instance is gone;
        one that stays listed for 5 min is `failed(deleting)`, never `off` — it bills."""
        s = self.state
        self._set_phase("deleting")
        await self._stop_tunnel()
        item = await self._find_live()
        if item is None and not s.uuid:
            # Ruling 10: never delete by a stored index alone
            await self._gone(f"instance at index {s.index} could not be identified as "
                             "ours — not deleted; if it runs it shows in the orphan list",
                             fault=True)
            return
        if item is None:
            # not listed twice — still delete by UUID (it cannot name a stranger): a 404
            # confirms it is gone, anything else means the lists were wrong
            self._log(f"instance {s.uuid} not listed — deleting it by uuid anyway")
            item = {"uuid": s.uuid}
        await self.api.delete(item)
        self._log(f"delete requested for {item.get('uuid') or item.get('index')}")
        deadline = self.deps.now() + _DELETE_WAIT_S
        misses = 0
        while True:
            try:
                still = await self._our_item()
            except self._Error as e:
                still = item                     # unknown ≠ gone
                self._log(f"instance list failed while deleting: {e}")
            if still is None:
                misses += 1
                if misses >= _ABSENT_CONFIRM:
                    break
            else:
                misses = 0
                if self.deps.now() >= deadline:
                    raise TimeoutError(f"instance {item.get('uuid')} still listed "
                                       f"{_DELETE_WAIT_S // 60} min after its delete "
                                       f"(status {still.get('status') or '?'})")
            await self.deps.sleep(_DELETE_POLL_S)
        self._log("instance deleted")
        await self._gone("")

    async def _gone(self, why: str, fault: bool = False) -> None:
        """The instance is confirmed gone: nothing bills any more, so the session cost
        stops (started_at 0) and the ids go — a later index is somebody else's."""
        s = self.state
        uuid = s.uuid
        await self._stop_tunnel()
        s.index, s.uuid, s.ip, s.port = "", "", "", 0
        s.started_at = 0.0
        s.bootstrap_incomplete = s.comfy_absent = s.host_bootstrapped = False
        self._set_phase("off", why)
        if uuid:
            try:
                os.remove(self._known_hosts_path(uuid))
            except (OSError, ValueError):
                pass
        self._disable()
        if fault:
            self._fault(None, "lifecycle", "instance_vanished", why)

    # ── snapshot watcher (spec "Stop" 4) ────────────────────────────────────

    def _forget_snapshot(self, sid: str) -> None:
        s = self.state
        s.manifests.pop(sid, None)
        for marks in (s.incomplete_snapshots, s.host_incomplete_snapshots,
                      s.no_comfy_snapshots):
            if sid in marks:
                marks.remove(sid)

    async def watch_snapshots(self) -> None:
        """One round: the pending snapshot READY → it becomes `snapshot_id` and the
        rotation deletes what `thunder.rotation` names (never the new one); FAILED (or
        missing from the list for several rounds) → fault, `snapshot_id` stays on the
        last READY one and the failed one's manifest goes. CREATING → next round."""
        s = self.state
        pid = s.pending_snapshot
        if not pid:
            return
        try:
            snaps = await self.api.snapshots()
        except self._Error as e:
            self._log(f"snapshot list unavailable ({e.status or 'transport'}): {e}")
            return
        if s is not self.state or s.pending_snapshot != pid:
            return                       # a stop recorded a newer one meanwhile
        self._snaps = snaps
        row = next((x for x in snaps if x.get("id") == pid), None)
        if row is None:
            self._pending_misses += 1
            if self._pending_misses < _PENDING_MISSES:
                self._log(f"pending snapshot {pid} not listed (yet)")
                return
            status = "MISSING"
        else:
            status = row.get("status") or ""
        name = (row or {}).get("name") or s.pending_snapshot_name or pid
        if status == "READY":
            self._pending_misses = 0
            s.snapshot_id, s.pending_snapshot, s.pending_snapshot_name = pid, "", ""
            self._log(f"snapshot {name} READY")
            self._persist()
            for sid in self._prov.rotation(snaps, self.name):
                if sid in (pid, s.pending_snapshot):
                    continue                 # never the one just made (a missing createdAt)
                try:
                    await self.api.delete_snapshot(sid)
                except self._Error as e:
                    self._log(f"rotation: deleting snapshot {sid} failed: {e}")
                    continue
                self._forget_snapshot(sid)
                self._log(f"rotation: snapshot {sid} deleted")
            self._persist()
        elif status in ("FAILED", "MISSING"):
            self._pending_misses = 0
            s.pending_snapshot, s.pending_snapshot_name = "", ""
            if status == "FAILED":
                self._forget_snapshot(pid)
            else:
                # not confirmed failed: should it reappear READY, its incomplete-bootstrap
                # mark must still make a start from it bootstrap again
                s.manifests.pop(pid, None)
            self._persist()
            why = "failed" if status == "FAILED" else "vanished from the snapshot list"
            self._log(f"snapshot {name} {why} — the last READY one ({s.snapshot_id or 'none'}) "
                      "stays the start template; this session's changes are lost")
            self._fault(None, "lifecycle", "snapshot_failed", name)

    # ── model sync (spec "Controller": URL transfer, triggers, disk growth) ─────

    def is_alias_ready(self, bid: str, alias: str) -> bool:
        """May `alias` route to service `bid` of this host? Only for the ComfyUI service
        the plan was made for, with a plan (none before the first sync) that has every
        file of the alias present and nothing blocking it — and only while the instance
        is `ready|syncing`: a plan is no promise about an instance that is draining,
        restarting ComfyUI or gone."""
        return (self.state.phase in _SYNC_PHASES and self.plan is not None
                and bid == self._plan_bid and bid == self._comfy_bid()
                and alias in self.ready_aliases)

    def alias_status(self, bid: str, alias: str) -> str:
        """Why (not) — the text a client's 503 carries."""
        if self.state.phase not in _SYNC_PHASES:
            return f"{self.kind} instance is {self.state.phase}"
        comfy = self._comfy()
        if comfy is None or service_bid(comfy) != bid:
            return f"{bid} is not the ComfyUI service of managed host {self.name}"
        return modelsync.status_text(self.plan if bid == self._plan_bid else None, alias,
                                     str(comfy.get("name")))

    def _lan_ok(self) -> bool:
        """LAN transfers may run: a pinned, reachable source with a good listing."""
        lan = self.deps.lan
        return lan is not None and lan.usable()

    def _lan_wait(self) -> str:
        """"waiting for LAN source (<why>)" — not configured, unreachable, …"""
        lan = self.deps.lan
        why = lan.problem() if lan is not None else ""
        return f"waiting for LAN source ({why or 'not configured'})"

    def _invalidate_source(self) -> None:
        if self.deps.lan is not None:
            self.deps.lan.invalidate()

    async def _signature(self) -> str:
        bid = self._comfy_bid()
        # the service id is part of it: a different ComfyUI service is a new plan
        return (f"{bid}|" + str(await asyncio.to_thread(self.deps.alias_signature, bid))
                if bid is not None else "")

    async def _sync_tick(self) -> None:
        """One 5-s trigger round: re-plan when the aliases or the catalog changed (the
        signature covers every store write, deletions and config aliases), else every
        60 s while transfers run (progress, readiness) or after a failed sync."""
        if self.state.phase not in _SYNC_PHASES or self._op is not None:
            return
        try:
            sig = await self._signature()
        except Exception as e:
            self._sync_fail(f"alias signature unavailable: {_errtext(e)}")
            return
        due = self.deps.now() - self._last_try >= _SYNC_REFRESH_S
        lan = self.deps.lan
        if lan is not None and lan.stale():
            await lan.refresh()             # at most every 10 min (1 after a failure)
        # a pin, a new listing or a source that went away/came back changes what waits
        lan_moved = lan is not None and lan.generation != self._lan_gen
        if (sig != self._sig or lan_moved) and (not self._sync_error or due):
            await self.sync_once()
        elif self._dirty and (self._kicker is None or self._kicker.done()):
            self._dirty = False
            await self.sync_once()
        elif (self._fetches or self._sync_error) and due:
            await self.sync_once()

    def _sync_fail(self, msg: str) -> None:
        self._last_try = self.deps.now()
        if msg != self._sync_error:
            self._log(f"model sync failed: {msg}")
        self._sync_error = msg

    async def _exec_ok(self, cmd: str, stdin: Optional[bytes] = None,
                       timeout: float = 60, token: str = "") -> str:
        """`_exec` that raises on a non-zero exit (with stderr's last line) → stdout.
        `token`: the secret this command was given on stdin — redacted from the error."""
        if not (self.state.ip and self.state.port):
            raise RuntimeError("no ssh address known")
        rc, out, err = await self._exec(cmd, stdin=stdin, timeout=timeout)
        if rc != 0:
            last = " | ".join(_tail(err, 2)) or f"rc {rc}"
            raise RuntimeError(f"remote command failed (rc {rc}): {_redact(last, token)}")
        return (out or b"").decode("utf-8", "replace")

    async def _dest_index(self) -> dict:
        return parse_index(await self._exec_ok(_INDEX_CMD, timeout=300))

    async def _read_manifest(self) -> dict:
        text = await self._exec_ok(_MANIFEST_READ)
        man = parse_manifest(text)
        if not man and text.strip() not in ("", "{}"):
            # silently {} would make every synced file "unknown" and every URL file a
            # download from zero
            self._log(f"WARNING: manifest ~/{_MANIFEST} is unreadable — its files count as "
                      "unknown and URL files are downloaded again")
        return man

    async def _write_manifest(self, man: dict) -> None:
        await self._exec_ok(_MANIFEST_WRITE, stdin=json.dumps(
            normalize_manifest(man), sort_keys=True, indent=1).encode("utf-8"))

    async def _df(self) -> int:
        text = (await self._exec_ok(_DF_CMD)).strip()
        if not text.isdigit():
            raise RuntimeError(f"df answered {text[:40]!r}")
        return int(text)

    async def _compute_plan(self) -> tuple[dict, dict]:
        """(plan, manifest) from a fresh destination index (the manifest is never the
        truth about which files exist, spec "Manifest"), the manifest (plus entries a
        failed write lost) and the gateway's inputs."""
        bid = self._comfy_bid()
        if bid is None:
            # never a plan with no needs: its prune list would be every synced file
            raise RuntimeError("no ComfyUI service attached — nothing to sync")
        dest = await self._dest_index()
        man = await self._read_manifest()
        man.update(copy.deepcopy(self._unsaved))

        def inputs():
            return (self.deps.alias_needs(bid) or [], self.deps.source_index() or {},
                    self.deps.url_catalog() or {})
        needs, src, urls = await asyncio.to_thread(inputs)
        self._plan_inputs = (needs, src, dest, man, urls)
        return modelsync.plan(needs, self._with_head_sizes(src, urls), dest, man, urls), man

    def _with_head_sizes(self, src: dict, urls: dict) -> dict:
        """The source index plus the sizes HEAD requests learned for URL files: a file
        with a known size counts in disk sizing and the syncing text, and is present
        only at that size (an upstream file that changed is fetched again)."""
        out = dict(src or {})
        for path, e in (urls or {}).items():
            n = self._head_sizes.get(e.get("url")) if isinstance(e, dict) else None
            if n is not None and path not in out:
                out[path] = n
        return out

    async def _learn_sizes(self, plan: dict) -> dict:
        """HEAD every URL file about to be fetched whose size nobody knows (once per URL
        and session; a HEAD that fails or names no length stays unknown), then re-plan
        with what was learned — without a second index/manifest read."""
        todo = [e for e in self._fetchable(plan)
                if e["source"] == "url" and e["size"] is None and e["url"] not in self._head_sizes
                and e["path"] not in self._fetches]
        if not todo:
            return plan
        for e in todo:
            self._head_sizes[e["url"]] = await self._head_size(e["path"], e["url"])
        needs, src, dest, man, urls = self._plan_inputs
        return modelsync.plan(needs, self._with_head_sizes(src, urls), dest, man, urls)

    async def _head_size(self, path: str, url: str) -> Optional[int]:
        if modelsync.validate_catalog([{"file": path, "url": url}]):
            return None
        token = self._hf_token_for(url)
        try:
            out = await self._exec_ok(_head_cmd(path), timeout=_HEAD_TIMEOUT_S + 30,
                                      stdin=curl_config(url, token, head=True).encode(),
                                      token=token)
        except Exception as ex:
            self._log(f"size of {path} unknown: HEAD failed ({_errtext(ex)})")
            return None
        n = parse_head(out)
        if n is None:
            self._log(f"size of {path} unknown: no Content-Length")
        return n

    async def sync_once(self) -> None:
        """Plan, readiness, disk, transfers (spec "Controller"). Serialised; a no-op
        outside `syncing|ready`. The ONLY writer of `plan` and `ready_aliases`. A
        failure keeps the last plan and is retried by `run_forever` after 60 s — it is
        logged and shown, never raised (a start must not fail over a sync)."""
        t = asyncio.ensure_future(self._sync_body())
        self._syncs.add(t)
        try:
            await t
        except asyncio.CancelledError:
            # a stop cancelled the BODY (`_cancel_sync_tasks`): the caller — maybe the
            # run_forever loop — goes on; a cancel of the caller itself propagates
            me = asyncio.current_task()
            if not t.cancelled() or (me is not None and me.cancelling()):
                raise
        finally:
            self._syncs.discard(t)

    async def sync_now(self) -> None:
        """The panel's "Sync now": a sync at once, and the transfers that gave up are
        tried again — otherwise only a change of the aliases or the catalog retries them,
        and an operator who fixed the cause (a URL, the disk) has no way to say so.
        Refused (RuntimeError, before the first await) without a running instance."""
        if not self._syncing():
            raise RuntimeError(f"no running instance ({self.state.phase})")
        # a HEAD that failed or named no length is asked again, and so is every URL of a
        # file that gave up (its size may be what was wrong): once per session otherwise
        urls = (self._plan_inputs or (None,) * 5)[4] or {}
        drop = {e.get("url") for p, e in urls.items()
                if p in self._failed and isinstance(e, dict)}
        self._head_sizes = {u: n for u, n in self._head_sizes.items()
                            if n is not None and u not in drop}
        if self._failed:
            self._log(f"sync requested — {len(self._failed)} failed transfer(s) are tried again")
            self._failed.clear()
        self._invalidate_source()           # the share is listed again, now
        await self.sync_once()

    def _syncing(self) -> bool:
        return self.state.phase in _SYNC_PHASES

    async def _sync_body(self) -> None:
        async with self._sync_lock:
            bid = self._comfy_bid()
            if not self._syncing() or bid is None:
                return                  # no ComfyUI service: nothing is synced here
            self._last_try = self.deps.now()
            try:
                sig = await self._signature()
                if self._sig is not None and sig != self._sig and self._failed:
                    self._log("aliases or catalog changed — failed transfers are tried again")
                    self._failed.clear()
                lan = self.deps.lan
                if lan is not None:
                    await lan.refresh()
                    self._lan_gen = lan.generation
                plan, man = await self._compute_plan()
                if self._syncing():
                    plan = await self._learn_sizes(plan)
            except Exception as e:
                self._sync_fail(_errtext(e))
                return
            if not self._syncing():
                # the phase changed while planning (a stop began): this plan is about an
                # instance on its way out — it must not touch the manifest or the disk
                return
            self._manifest = man
            plan = await self._make_links(plan)
            try:
                go, disk = await self._fit_disk(plan)
            except Exception as e:
                # df unreadable: go on — a full disk fails the curl, which is reported
                self._log(f"free disk space unknown ({_errtext(e)}) — downloading anyway")
                go, disk = self._fetchable(plan), {}
            before = set(self.ready_aliases) if self._plan_bid == bid else set()
            self.plan = self._annotate(plan, disk)
            self._plan_bid = bid
            self.ready_aliases = {a for a in self.plan["per_alias"]
                                  if modelsync.ready(self.plan, a)}
            self._sig, self._sync_error = sig, ""
            for a in sorted(self.ready_aliases - before):
                self._log(f"models for {a} ready")
            for a in sorted(before - self.ready_aliases):
                self._log(f"models for {a} no longer ready")
            if self._syncing():
                self._start_fetches(go)

    def _stuck_aliases(self, plan: dict) -> set:
        """Aliases a transfer cannot make ready now: a file only the LAN has while the
        LAN source is not usable (not pinned, unreachable), or a transfer that gave up.
        Once the source is usable, a LAN file no longer withholds its alias's URL files
        (Ruling 18)."""
        lan_ok = self._lan_ok()
        stuck: set = set()
        for e in plan["fetch"]:
            if (e["source"] == "lan" and not lan_ok) or e["path"] in self._failed:
                stuck.update(e["aliases"])
        return stuck

    def _fetchable(self, plan: dict) -> list:
        """The URL and LAN fetch entries worth transferring: not given up, and needed by
        at least one alias a transfer can make ready (modelsync's rule for blocked
        aliases, extended to the LAN-waiting and failed ones). Links are no transfer —
        `_make_links` creates them."""
        stuck = self._stuck_aliases(plan)
        return [e for e in plan["fetch"] if e["source"] in ("url", "lan")
                and e["path"] not in self._failed and set(e["aliases"]) - stuck]

    def _reserve_bytes(self) -> int:
        try:
            return int(self.cfg.get("reserve_gb") or 20) * 1024 ** 3
        except (TypeError, ValueError):
            return 20 * 1024 ** 3

    async def _fit_disk(self, plan: dict) -> tuple[list, dict]:
        """(entries to fetch, {alias: disk reason}). Free space below the known fetch
        bytes + reserve → grow the disk (`_grow_disk`); still short → the aliases that
        do not fit (smallest first, the plan's order) are blocked "disk". A URL file of
        unknown size counts 0 — its download fails on a full disk and says so."""
        entries = self._fetchable(plan)
        todo = [e for e in entries if e["path"] not in self._fetches]
        if not todo:
            return entries, {}
        need = sum(e["size"] or 0 for e in todo)
        reserve = self._reserve_bytes()
        avail = await self._df()
        why = ""
        if avail < need + reserve:
            avail, why = await self._grow_disk(avail, need, reserve)
        if avail >= need + reserve:
            return entries, {}
        budget, used, take, disk = avail - reserve, 0, set(), {}
        order: list = []
        for e in todo:
            for a in e["aliases"]:
                if a not in order:
                    order.append(a)
        for a in order:
            mine = [e for e in todo if a in e["aliases"] and e["path"] not in take]
            extra = sum(e["size"] or 0 for e in mine)
            if used + extra <= budget:
                used += extra
                take.update(e["path"] for e in mine)
            else:
                disk[a] = (f"disk: {_gb(extra)} GB to fetch, {_gb(max(0, budget - used))} GB "
                           f"free above the reserve{why}")
        go = [e for e in entries if e["path"] in take or e["path"] in self._fetches]
        return go, disk

    async def _grow_disk(self, avail: int, need: int, reserve: int) -> tuple[int, str]:
        """Grow the disk so the fetch fits (spec "Disk-Wachstum"): `choose_disk_gb` over
        what is used now + the fetch, never below the current size, within the spec's
        maximum. The instance is found by uuid in a FRESH list (Rulings 10/11). Thunder
        grows the filesystem itself — `df` is re-read until the space appears. →
        (free bytes now, "" or why it could not grow — for the disk block text)."""
        s, cfg = self.state, self.cfg
        if not self._syncing():
            return avail, f" (instance is {s.phase})"
        try:
            num_gpus = int(cfg.get("num_gpus") or 1)
            spec = self._prov.spec_for(await self.api.specs(), str(cfg.get("gpu_type") or ""),
                                    num_gpus)
            storage = (spec or {}).get("storageGB")
            storage = storage if isinstance(storage, dict) else {}
            used = max(0, int(s.disk_gb or 0) * 1024 ** 3 - avail)
            new = self._prov.choose_disk_gb(
                required_bytes=used + need, base_bytes=0,
                reserve_gb=math.ceil(reserve / 1024 ** 3), snapshot_min_gb=s.disk_gb,
                spec_min=int(storage.get("min") or 0), spec_max=int(storage.get("max") or 0),
                num_gpus=num_gpus)
        except self._prov.DiskTooSmall as e:
            self._log(f"disk cannot grow enough: {e}")
            return avail, f" (disk cannot grow: {e})"
        except (self._Error, TypeError, ValueError) as e:
            self._log(f"disk growth unavailable: {_errtext(e)}")
            return avail, f" (disk growth unavailable: {_errtext(e)})"
        if new <= int(s.disk_gb or 0):
            return avail, f" (disk is {s.disk_gb} GB)"
        try:
            item = await self._fresh_item()
            if not self._syncing():             # a stop began during the list
                return avail, f" (instance is {s.phase})"
            await self.api.modify(item, {"disk_size_gb": new})
        except self._Error as e:
            self._log(f"disk growth to {new} GB failed: {e}")
            return avail, f" (disk growth to {new} GB failed)"
        self._log(f"disk grown {s.disk_gb} → {new} GB: {_gb(need)} GB to fetch, "
                  f"{_gb(avail)} GB free, reserve {_gb(reserve)} GB")
        s.disk_gb = new
        self._persist()
        for i in range(_GROW_RECHECK):
            if i:
                await self.deps.sleep(_GROW_RECHECK_S)
            try:
                avail = await self._df()
            except Exception as e:
                self._log(f"free disk space unknown after growing: {_errtext(e)}")
                continue
            if avail >= need + reserve:
                return avail, ""
        return avail, f" (disk grown to {new} GB, the space has not appeared yet)"

    def _annotate(self, plan: dict, disk: dict) -> dict:
        """The controller's view of modelsync's plan (a copy): what only the controller
        knows joins `blocked` — the LAN source is not configured (P2: modelsync's "not
        in source" and every LAN fetch become "waiting for LAN source (not
        configured)"), a transfer gave up, the disk is too small."""
        p = copy.deepcopy(plan)
        rows = p["per_alias"]
        lan = self._lan_ok()
        wait = self._lan_wait()
        for row in rows.values():
            row["blocked"] = [f"{wait}: {b[len(_NOT_IN_SOURCE):]}"
                              if not lan and b.startswith(_NOT_IN_SOURCE) else b
                              for b in row["blocked"]]
        for e in p["fetch"]:
            extra = ""
            if e["source"] == "lan" and not lan:
                extra = f"{wait}: {e['path']}"
            elif e["path"] in self._failed:
                extra = f"transfer failed: {e['path']}: {self._failed[e['path']]}"
            for a in e["aliases"] if extra else ():
                rows[a]["blocked"].append(extra)
        for a, why in disk.items():
            rows[a]["blocked"].append(why)
        for row in rows.values():
            row["blocked"] = sorted(set(row["blocked"]))
        return p

    def _plan_view(self) -> Optional[dict]:
        """The panel's plan summary (Task 13): per alias sizes, counts and texts, the
        fetch/prune/held/unknown lists. No URL (a catalog URL may carry a query token)."""
        p = self.plan
        if p is None:
            return None
        held: dict = {}
        for h in p["held"]:
            held[h[2]] = held.get(h[2], 0) + 1
        # what the stop will delete, with sizes for the preview ("N files, X GB"): the
        # destination's size, else the manifest's; None when neither knows
        dest = (self._plan_inputs or (None, None, {}))[2] or {}
        man = self._manifest if isinstance(self._manifest, dict) else {}

        def prune_size(path):
            n = dest.get(path)
            if n is None and isinstance(man.get(path), dict):
                n = man[path].get("size")
            return n if isinstance(n, int) and not isinstance(n, bool) else None
        return {
            "aliases": {a: {"ready": a in self.ready_aliases, "need_bytes": r["need_bytes"],
                            "have_bytes": r["have_bytes"], "missing": len(r["missing"]),
                            "blocked": list(r["blocked"]), "hints": list(r["hints"]),
                            "held": held.get(a, 0), "selectable": list(r["selectable"]),
                            "files": [dict(f) for f in r["files"]]}
                        for a, r in p["per_alias"].items()},
            "fetch": [{k: e[k] for k in ("path", "size", "source", "aliases")}
                      for e in p["fetch"]],
            "prune": list(p["prune"]),
            "prune_sizes": [[x, prune_size(x)] for x in p["prune"]],
            "held": [list(h) for h in p["held"]],
            "unknown": [list(u) for u in p["unknown"]],
            "need_total": p["need_total"], "have_total": p["have_total"]}

    async def _make_links(self, plan: dict) -> dict:
        """Create the plan's missing symlinks (the HF cache's snapshots/, modelsync
        `source: "link"`) in ONE command, record them in the manifest (a link is pruned
        like a file) and re-plan with them present. Only for aliases a transfer can make
        ready — without a LAN source an HF repo's links would just dangle. A failure is
        logged and retried with the next plan; the plan is returned unchanged."""
        stuck = self._stuck_aliases(plan)
        todo = [e for e in plan["fetch"] if e["source"] == "link" and set(e["aliases"]) - stuck]
        lines, entries = [], {}
        now = int(self.deps.now())
        for e in todo:
            path, t = e["path"], e.get("target")
            try:
                rp = remote_path(path)
            except ValueError as ex:
                self._log(f"link {path} skipped: {ex}")
                continue
            if modelsync.link_target(path, t) is None:
                self._log(f"link {path} skipped: target {t!r} leaves its root")
                continue
            lines.append(f"{rp}\n{t}\n")
            entries[path] = {"size": 0, "source": "link", "target": t,
                             "aliases": sorted(e["aliases"]), "ts": now}
        if not entries:
            return plan
        # recorded BEFORE the ln (the `_finish` rule): a cancel between the links and the
        # manifest write must not leave them unrecorded ("unknown" next session)
        self._unsaved.update(entries)
        try:
            out = await self._exec_ok(_LINK_CMD, stdin="".join(lines).encode("utf-8"),
                                      timeout=120)
            if "GW:LINKED" not in out:
                raise RuntimeError("links not confirmed by the instance")
        except Exception as ex:
            for path, ent in entries.items():
                if self._unsaved.get(path) is ent:
                    del self._unsaved[path]
            self._log(f"creating {len(entries)} model link(s) failed: {_errtext(ex)}")
            return plan
        await self._manifest_add_many(entries)
        self._log(f"linked {len(entries)} model file link(s)")
        needs, src, dest, man, urls = self._plan_inputs
        dest = dict(dest, **{p: {"link": e["target"]} for p, e in entries.items()})
        man = dict(man, **copy.deepcopy(entries))
        self._plan_inputs = (needs, src, dest, man, urls)
        return modelsync.plan(needs, self._with_head_sizes(src, urls), dest, man, urls)

    # URL transfers ─────────────────────────────────────────────────────────

    def _start_fetches(self, entries: list) -> None:
        """Up to `_MAX_FETCH` URL downloads (they run ON the instance) and exactly ONE LAN
        stream (it runs through the gateway, and the uplink is the bottleneck)."""
        for e in entries:
            path = e["path"]
            if path in self._fetches:
                continue
            if e["source"] == "lan":
                if self._lan_path is not None and self._lan_path in self._fetches:
                    continue
                self._lan_path = path
            elif sum(1 for p in self._fetches if p != self._lan_path) >= _MAX_FETCH:
                continue
            self._fetches[path] = asyncio.ensure_future(self._fetch(dict(e)))

    def _kick_sync(self) -> None:
        """Re-plan soon (a transfer ended): readiness and the next download start
        without waiting for the 60-s round. One kicker at a time; a kick while it runs
        makes it plan once more. Outside `syncing|ready` (a ComfyUI restart) the kick
        stays pending and `_sync_tick` runs it — dropped, the files still to fetch
        would never start."""
        self._dirty = True
        if self.state.phase not in _SYNC_PHASES:
            return
        if self._kicker is None or self._kicker.done():
            self._kicker = asyncio.ensure_future(self._kick_loop())

    async def _kick_loop(self) -> None:
        while self._dirty:
            self._dirty = False
            try:
                await self.sync_once()
            except Exception as e:           # sync_once logs its own failures
                self._log(f"model sync failed: {_errtext(e)}")

    def _hf_token_for(self, url: str) -> str:
        """The token, only for a Hugging Face HOST — decided from the parsed hostname
        before any config exists (`huggingface.co.evil.example` is not one) — and only
        when a curl config line can carry it verbatim."""
        host = (urlparse(url).hostname or "").lower()
        if host not in _HF_HOSTS:
            return ""
        try:
            tok = str(self.deps.hf_token() or "")
        except Exception:
            return ""
        if tok and not hf_token_ok(tok):
            self._log("hf_token withheld: it is longer than 512 characters or contains a "
                      "quote, a backslash, whitespace, a control or non-ASCII character a "
                      "curl config line cannot carry — requesting without it")
            return ""
        return tok

    async def _fetch(self, e: dict) -> None:
        """One file's transfer task: up to `_FETCH_ATTEMPTS` attempts (curl resumes the
        `.part`), then the file is given up: its aliases are blocked with the reason and
        a fault is logged. Always re-plans at the end (the alias may be ready now)."""
        path = e["path"]
        me = asyncio.current_task()
        lan = e.get("source") == "lan"
        attempt_fn, what = ((self._lan_attempt, "LAN transfer") if lan
                            else (self._fetch_attempt, "download"))
        why = ""
        try:
            for attempt in range(1, _FETCH_ATTEMPTS + 1):
                why, final = await attempt_fn(e, attempt)
                if not why:
                    break
                self._log(f"{what} {path} attempt {attempt}/{_FETCH_ATTEMPTS} failed: {why}")
                if final:
                    break
                if attempt < _FETCH_ATTEMPTS:
                    await self.deps.sleep(_RETRY_DELAYS_S[min(attempt, len(_RETRY_DELAYS_S)) - 1])
            if why:
                self._failed[path] = why
                # a transfer is the ComfyUI service's (its models); the host's pseudo
                # backend only when that service was detached meanwhile
                self._fault(self._comfy(), "sync", "transfer", f"{path}: {why}")
        finally:
            if self._fetches.get(path) is me:
                del self._fetches[path]
            if self._lan_path == path:
                self._lan_path = None
            self.state.transfers.pop(path, None)
        self._kick_sync()

    async def _fetch_attempt(self, e: dict, attempt: int) -> tuple[str, bool]:
        """Start (or adopt) the curl, poll it every 5 s, verify, move it in place and
        record it in the manifest. → ("" | why it failed, whether retrying is pointless)."""
        path, url = e["path"], str(e.get("url") or "")
        errs = modelsync.validate_catalog([{"file": path, "url": url}])
        if errs:
            return "invalid source: " + "; ".join(errs), True
        try:
            cmd = _fetch_cmd(path)
        except ValueError as ex:
            return f"invalid path: {ex}", True
        token = self._hf_token_for(url)          # read ONCE per attempt (a store read)
        cfg = curl_config(url, token)
        try:
            out = await self._exec_ok(cmd, stdin=cfg.encode("utf-8"), token=token)
        except Exception as ex:
            return f"start failed: {_errtext(ex)}", False
        if "GW:ADOPT" in out:
            self._log(f"download of {path} still running on the instance — adopted")
        elif "GW:START-FAIL" in out:
            tail = _redact(out.split("GW:START-FAIL", 1)[1], token).strip()
            return ("curl did not start on the instance"
                    + (f": {tail.splitlines()[-1][:300]}" if tail else "")), False
        elif "GW:STARTED" in out:
            self._log(f"downloading {path}" + (f" (attempt {attempt})" if attempt > 1 else ""))
        else:
            return "start failed: no answer from the instance", False
        row = self.state.transfers[path] = {
            "file": path, "source": "url", "bytes": 0, "total": e.get("size"),
            "rate": None, "eta": None, "attempt": attempt}
        prev = None
        fails = 0
        while True:
            await self.deps.sleep(_SYNC_POLL_S)
            try:
                lines = (await self._exec_ok(_poll_cmd(path))).splitlines()
                if len(lines) < 2 or lines[0] not in ("GW:RUN", "GW:END"):
                    raise RuntimeError("unreadable poll answer")
            except Exception as ex:
                fails += 1
                if fails == 1:
                    self._log(f"download {path}: poll failed ({_errtext(ex)}) — retrying")
                if fails >= _POLL_FAILS_MAX:
                    return "the instance did not answer while downloading", False
                continue
            fails = 0
            size = int(lines[1]) if lines[1].lstrip("-").isdigit() else -1
            now = self.deps.now()
            if size >= 0:
                if prev is not None and now > prev[1]:
                    row["rate"] = max(0.0, (size - prev[0]) / (now - prev[1]))
                    total = row["total"]
                    row["eta"] = ((total - size) / row["rate"]
                                  if total and row["rate"] else None)
                prev = (size, now)
                row["bytes"] = size
            if lines[0] == "GW:RUN":
                continue
            err = _redact("\n".join(lines[2:]).strip(), token)
            if err:
                return err.splitlines()[-1][:300], False
            break
        return await self._finish(e, row["bytes"])

    async def _finish(self, e: dict, size: int) -> tuple[str, bool]:
        """curl ended cleanly (with --fail and a Content-Length it refuses a short body
        itself). Size against what the source says, the catalog's sha256 when it has
        one; bytes that fail either are discarded, never resumed. Then `mv` into place
        and the manifest entry."""
        path, want = e["path"], e.get("size")
        bad = ""
        if size <= 0:
            bad = "download is empty"
        elif want is not None and size != want:
            bad = f"size {size} ≠ {want} expected"
        sha = str(e.get("sha256") or "").lower()
        if not bad and sha:
            try:
                out = await self._exec_ok(_sha_cmd(path), timeout=_SHA_TIMEOUT_S)
            except Exception as ex:
                return f"sha256 check failed: {_errtext(ex)}", False
            got = (out.split() or [""])[0].lower()
            if got != sha:
                bad = f"sha256 mismatch ({got[:12] or '?'}… ≠ {sha[:12]}…)"
        if bad:
            try:
                await self._exec_ok(_discard_cmd(path))
            except Exception as ex:
                self._log(f"discarding {path}.part failed: {_errtext(ex)}")
            return bad, False
        entry = {"size": size, "sha256": sha or None, "source": e.get("source") or "url",
                 "aliases": sorted(e.get("aliases") or []), "ts": int(self.deps.now())}
        # recorded BEFORE the mv: a cancel (stop) between the mv and the manifest write
        # would otherwise leave a finished file without its entry — "unknown" next
        # session and downloaded again in full. `_before_snapshot` writes `_unsaved`.
        self._unsaved[path] = entry
        try:
            out = (await self._exec_ok(_done_cmd(path))).strip()
        except Exception as ex:
            if self._unsaved.get(path) is entry:
                del self._unsaved[path]
            return f"moving the download into place failed: {_errtext(ex)}", False
        final = int(out) if out.isdigit() else size
        await self._manifest_add(path, dict(entry, size=final))
        self._log(f"{'transferred' if entry['source'] == 'lan' else 'downloaded'} {path} "
                  f"({_gb(final)} GB)")
        return "", False

    async def _lan_attempt(self, e: dict, attempt: int) -> tuple[str, bool]:
        """One LAN transfer attempt (spec "Übertragung LAN"): resume at the size of the
        instance's `.part`, stream the share's `cat <rel> <offset>` into `cat >> .part`
        through the gateway (`deps.pipe`: two ssh processes, nothing stored on the
        gateway), then sha256 on BOTH sides — the source's cached per (path, size) —
        before `_finish` moves it in place. A short `.part` is kept (the next attempt
        resumes it), a long one or a sha mismatch is discarded. → ("" | why, final)."""
        path, want = e["path"], e.get("size")
        lan = self.deps.lan
        if lan is None or not lan.usable():
            return f"LAN source unavailable ({lan.problem() if lan else 'not configured'})", False
        try:
            _parts(path)
            lan.cat_argv(path, 0)
        except ValueError as ex:
            return f"invalid path: {ex}", True
        try:
            out = (await self._exec_ok(_part_size_cmd(path))).strip()
        except Exception as ex:
            return f"resume offset unknown: {_errtext(ex)}", False
        offset = int(out) if re.fullmatch(r"[0-9]{1,18}", out) else 0
        if want is not None and offset > want:
            self._log(f"{path}.part is larger than the source file — starting over")
            try:
                await self._exec_ok(_discard_cmd(path))
            except Exception as ex:
                return f"discarding an oversized .part failed: {_errtext(ex)}", False
            offset = 0
        row = self.state.transfers[path] = {
            "file": path, "source": "lan", "bytes": offset, "total": want,
            "rate": None, "eta": None, "attempt": attempt}
        if want is None or offset < want:
            self._log(f"LAN transfer {path}" + (f" from byte {offset}" if offset else "")
                      + (f" (attempt {attempt})" if attempt > 1 else ""))
            t0, moved = self.deps.now(), [0]

            def on_bytes(n: int) -> None:
                moved[0] += n
                row["bytes"] = offset + moved[0]
                dt = self.deps.now() - t0
                if dt > 0:
                    row["rate"] = moved[0] / dt
                    row["eta"] = ((want - row["bytes"]) / row["rate"]
                                  if want and row["rate"] else None)
            rc_s, rc_d, tail = await self.deps.pipe(
                lan.cat_argv(path, offset), self._ssh_argv(_lan_recv_cmd(path)), on_bytes,
                timeout_idle=_LAN_IDLE_S)
            if rc_s != 0 or rc_d != 0:
                why = ("another stream still appends to the .part" if rc_d == _FLOCK_BUSY
                       else f"stream failed (source rc {rc_s}, instance rc {rc_d})")
                return why + (f": {tail[-300:]}" if tail else ""), False
        try:
            out = (await self._exec_ok(_part_size_cmd(path))).strip()
        except Exception as ex:
            return f"size check failed: {_errtext(ex)}", False
        size = int(out) if re.fullmatch(r"[0-9]{1,18}", out) else 0
        row["bytes"] = size
        if want is not None and size < want:
            return f"stream ended at {size} of {want} bytes", False     # resumed next attempt
        if want is not None and size > want:
            try:
                await self._exec_ok(_discard_cmd(path))
            except Exception as ex:
                self._log(f"discarding {path}.part failed: {_errtext(ex)}")
            return f"size {size} ≠ {want} expected", False
        try:
            src_sha = await lan.sha256(path, size)
        except Exception as ex:
            return f"source sha256 unavailable: {_errtext(ex)}", False
        why, final = await self._finish(dict(e, sha256=src_sha), size)
        if why.startswith("sha256 mismatch"):
            lan.forget_sha(path, size)      # the share's file may have changed in place
        return why, final

    async def _manifest_add(self, path: str, entry: dict) -> None:
        await self._manifest_add_many({path: entry})

    async def _manifest_add_many(self, entries: dict) -> None:
        """Record finished files/links (read-modify-write, serialised). A failed write
        keeps the entries in memory (`_unsaved`) — plans and the next write include them;
        without them a file would count as not present and be transferred again."""
        async with self._manifest_lock:
            self._unsaved.update(entries)
            try:
                man = await self._read_manifest()
                man.update(copy.deepcopy(self._unsaved))
                await self._write_manifest(man)
            except Exception as ex:
                self._log(f"manifest not written ({_errtext(ex)}) — kept in memory")
                return
            self._unsaved.clear()
            self._manifest = man

    async def _kill_remote(self) -> None:
        """End every download of ours on the instance (by the lockfiles' curl pids)."""
        s = self.state
        if not (s.ip and s.port):
            return
        try:
            rc, out, err = await self._exec(_KILL_CMD, timeout=120)
        except Exception as e:
            rc, out, err = -1, b"", _errtext(e).encode()
        if rc != 0 or b"GW:KILLED" not in (out or b""):
            self._log(f"ending the downloads on the instance failed (rc {rc}): "
                      + " | ".join(_tail(err, 3)))

    async def delete_unknown(self, paths: list) -> int:
        """Delete files of the `unknown` list (neither synced by us nor needed — e.g. a
        template's models) on the operator's request. Judged against a FRESH plan: a
        file that became needed or was synced meanwhile is refused (ValueError), and so
        is the whole request then — nothing is deleted by a stale button. → count."""
        if self.state.phase not in _SYNC_PHASES:
            raise RuntimeError(f"no running instance ({self.state.phase})")
        paths = [str(p) for p in paths or []]
        if not paths:
            return 0
        async with self._sync_lock:
            plan, _ = await self._compute_plan()
            if not self._syncing():
                raise RuntimeError(f"instance is {self.state.phase} — nothing deleted")
            unknown = {u[0] for u in plan["unknown"]}
            bad = [p for p in paths if p not in unknown]
            if bad:
                raise ValueError("not in the unknown list (needed, synced or gone): "
                                 + ", ".join(bad))
            out = await self._exec_ok(_delete_cmd(paths), timeout=_PRUNE_TIMEOUT_S)
            if "GW:DELETED" not in out:
                raise RuntimeError("delete not confirmed by the instance")
            self._log(f"deleted {len(paths)} unknown file(s): " + ", ".join(paths[:10])
                      + (" …" if len(paths) > 10 else ""))
        await self.sync_once()
        return len(paths)

    # ── orphans ─────────────────────────────────────────────────────────────

    async def orphans(self, items: Optional[list] = None) -> list[dict]:
        """Instances of this account no controller owns (`deps.known_uuids()`, plus our
        own) — shown with their cost, NEVER deleted: a stranger's instance cannot be
        told from a lost one of ours. A list error keeps the last answer. `items`: a
        list the caller has just fetched (None = fetch one)."""
        if items is None:
            try:
                items = await self.api.list_instances()
            except self._Error as e:
                self._log(f"orphan check: instance list unavailable: {e}")
                return list(self._orphans)
        known = self._known_uuids()
        if self.state.uuid:
            known.add(self.state.uuid)
        self._orphans = [it for it in items if it.get("uuid") not in known
                         and not self._prov.is_gone_status(it.get("status"))]
        return list(self._orphans)

    # ── after a gateway restart ─────────────────────────────────────────────

    async def resume(self) -> None:
        """Lifespan, after `rebuild_backends`: reconcile the persisted state with
        `/instances/list` (spec "Nach Gateway-Neustart"). An interrupted stop runs on,
        an interrupted start before the bootstrap continues, a live instance gets its
        tunnel back (ComfyUI restarted when it does not answer), a vanished one ends in
        `off`. An API that cannot be reached leaves everything as it is and
        `run_forever` retries. Runs as an op, so a stop() meanwhile aborts it."""
        if self._op is not None:
            self._log(f"resume skipped: {self._op} in progress")
            return
        await self._run_op("resuming", self._resume())

    async def _resume(self) -> None:
        if self._persist_blocked and not await self._reconcile_unread():
            return
        s = self.state
        if s.phase == "off":
            self._resume_pending = False
            await self.watch_snapshots()
            return
        if not (s.uuid or s.index):
            self._resume_pending = False
            if s.phase != "failed":      # `failed` without instance stays for the panel
                await self._gone(f"no instance recorded in phase {s.phase}")
            return
        try:
            items = await self.api.list_instances()
        except self._Error as e:
            self._log(f"resume: instance list unavailable ({e}) — retrying")
            self._resume_pending = True
            return
        it = self._find_ours(items)
        if it is not None and self._prov.is_gone_status(it.get("status")):
            it = None
        if it is None:
            # one answer without it is not proof (see _ABSENT_CONFIRM)
            await self.deps.sleep(_ABSENT_RECHECK_S)
            try:
                it = await self._our_item()
            except self._Error as e:
                self._log(f"resume: instance list unavailable ({e}) — retrying")
                self._resume_pending = True
                return
        self._resume_pending = False
        p = s.failed_phase if s.phase == "failed" else s.phase
        if it is None and p == "deleting":
            await self._stop_run()           # _delete finishes it (uuid-form delete)
            await self.watch_snapshots()
            return
        if it is None:
            if p == "snapshotting":
                await self._adopt_pending_by_name()
            if p == "snapshotting" and s.pending_snapshot:
                self._log("resume: instance gone — the stop had finished its snapshot")
                await self._gone("")
            else:
                await self._gone(f"instance {s.uuid or s.index} vanished while the gateway "
                                 f"was down (phase {p})", fault=True)
            await self.watch_snapshots()
            return
        if p in _STOP_STEPS:
            self._log(f"resume: continuing the interrupted stop at {p}")
            await self._stop_run()
            return
        if s.phase == "failed":
            await self._reattach(it, keep_failed=True)
            return
        if s.phase in _PRE_BOOT:
            self._log(f"resume: continuing the interrupted start at {s.phase}")
            await self._after_create()
            return
        if s.phase == "bootstrapping":
            await self._reattach(it, keep_failed=True)
            if not s.host_bootstrapped:
                self._fail("host bootstrap interrupted by a gateway restart (it may still "
                           f"run on the instance, see {_HOST_BOOTSTRAP_LOG}) — Stop, and "
                           "the next start runs it again")
            else:
                self._fail("bootstrap interrupted by a gateway restart (it may still run on "
                           f"the instance, see {_BOOTSTRAP_LOG}) — Restart ComfyUI once it is "
                           "done, or Stop")
            return
        # starting / syncing / ready
        try:
            await self._reattach(it)
        except Exception as e:
            self._fail(_errtext(e))
            return
        # every service probed on its own forward; only one that does not answer is
        # restarted — the others keep their jobs
        dead = []
        for svc in [x for x in self.services if x.get("type") == "comfyui"]:
            if await self._probe_briefly(svc):
                self._svc_set(svc, "up")
            else:
                dead.append(svc)
        if not dead:
            self._ready()
        else:
            self._log("resume: ComfyUI does not answer — restarting it")
            await self._restart(only=dead)

    async def _reattach(self, item: dict, keep_failed: bool = False) -> None:
        """Port guard first (a restart must not find the instance behind a public port
        the template reopened), then ip/port from the fresh list and a new tunnel."""
        s = self.state
        try:
            item = await self._ensure_ports_closed(item)
        except Exception as e:
            if not keep_failed:
                raise
            self._log(f"resume: {_errtext(e)} — no tunnel")
            return
        if item.get("ip") and item.get("port"):
            s.ip, s.port = str(item["ip"]), int(item["port"])
            self._persist()
        if s.ip and s.port:
            await self._start_tunnel()

    async def _probe_briefly(self, svc: dict) -> bool:
        url = self._svc_url(svc)
        if not url:
            return False
        deadline = self.deps.now() + _RESUME_PROBE_S
        while True:
            try:
                if await self.deps.probe_comfy(url):
                    self._log("ComfyUI answers")
                    return True
            except Exception:
                pass
            if self.deps.now() >= deadline:
                return False
            await self.deps.sleep(_COMFY_PROBE_S)

    async def _adopt_pending_by_name(self) -> None:
        """A restart hit `snapshotting` between the POST and persisting its id: the
        persisted name finds it."""
        s = self.state
        if s.pending_snapshot or not s.pending_snapshot_name:
            return
        try:
            snaps = await self.api.snapshots()
        except self._Error as e:
            self._log(f"snapshot list unavailable ({e.status or 'transport'}): {e}")
            return
        row = next((x for x in snaps if x.get("name") == s.pending_snapshot_name
                    and x.get("id") and x.get("status") != "FAILED"), None)
        if row is not None:
            self._log(f"snapshot {row['name']} found by name ({row['id']})")
            self._record_pending(row["id"])

    async def _reconcile_unread(self) -> bool:
        """The stored record could not be read at construction (`_load_failed`). Read it
        again (a locked store recovers); still unreadable → `/instances/list` decides:
        no unowned instance → nothing of ours can bill → `off`; unowned ones → `failed`
        WITHOUT an instance, naming them (they show in the orphan list; never adopted,
        Ruling 10). Saving resumes either way. → whether resume() may continue with a
        re-read state."""
        try:
            loaded = self.deps.load_state(self.name)
            readable = loaded is None or isinstance(loaded, dict)
        except Exception:
            readable = False
        if readable:
            log = self.state.log
            self.state = state_from(loaded)
            self.state.log = log
            self._unblock_persist()
            self._log("stored state re-read")
            return True
        # `off` needs "no stranger" to hold over `_ABSENT_CONFIRM` consecutive fresh lists
        # (the `_find_live` rule): one empty answer reconciled as off would let the next
        # start create a second instance next to a billing one of ours
        known = self._known_uuids()
        found: dict = {}
        for i in range(_ABSENT_CONFIRM):
            if i:
                self._log("no unowned instance listed — checking again")
                await self.deps.sleep(_ABSENT_RECHECK_S)
            try:
                items = await self.api.list_instances()
            except self._Error as e:
                self._log(f"resume: instance list unavailable ({e}) — state stays "
                          "unreconciled")
                self._resume_pending = True
                return False
            for it in items:
                if it.get("uuid") not in known and not self._prov.is_gone_status(it.get("status")):
                    found.setdefault(str(it.get("uuid") or it.get("index") or len(found)), it)
            if found:
                break
        self._resume_pending = False
        strangers = list(found.values())
        self._orphans = strangers
        self._unblock_persist()
        s = self.state
        s.failed_phase = ""
        s.unreconciled_uuids = [it["uuid"] for it in strangers if it.get("uuid")]
        if strangers:
            ids = ", ".join(it.get("uuid") or it.get("index") or "?" for it in strangers)
            self._set_phase("failed", f"stored state unreadable; unowned instance(s) listed "
                                      f"({ids}) — one may be this backend's: check and "
                                      "delete by hand")
        else:
            self._set_phase("off", "stored state was unreadable; no unowned instance is "
                                   "running")
        return False

    # ── background ──────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """Background loop (spawned by main next to resume()): every 5 s a forward that
        is still missing on the running master (a failed `forward`, a port that was
        busy) is tried again, and the model-sync trigger (`_sync_tick`) runs; every 60 s the snapshot watcher while a snapshot is
        pending, and resume() again while it could not reach the API; every 10 min the
        account view (`refresh_account` — prices, snapshots, foreign instances)."""
        every = max(1, _WATCH_S // _SYNC_POLL_S)
        tick = 0
        while True:
            await self.deps.sleep(_SYNC_POLL_S)
            tick += 1
            try:
                if tick % every == 0:
                    if self._resume_pending and self._op is None:
                        await self.resume()
                    if self.state.pending_snapshot:
                        await self.watch_snapshots()
                    now = self.deps.now()
                    if self._account_at is None or now - self._account_at >= _ACCOUNT_S:
                        self._account_at = now
                        try:
                            await self.refresh_account()
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:  # display only: never costs the sync tick
                            self._log(f"account refresh failed: {_errtext(e)}")
                if (self._tunnel is not None and self._fwd_active != set(self._forwards())
                        and (self._fwd_task is None or self._fwd_task.done())):
                    await self.reconcile_forwards()
                await self._sync_tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._log(f"background round failed: {_errtext(e)}")
