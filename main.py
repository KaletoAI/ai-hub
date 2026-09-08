import asyncio
import base64
import calendar
import fnmatch
import json
import logging
import mimetypes
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from watchfiles import awatch

import adapters
import admin
import anthropic_bridge
import jobs
import reasoning
import scheduler
import netscan
import socket
import stats
import store
from adapters import (AdapterContext, ComfyExecutorStuck, NormalizedRequest, image_params,
                      is_image_field, lora_counterpart, lora_groups,
                      make_adapter, normalize_delivery, validate_delivery)
from openai_image_bridge import (EDIT_KNOWN, OAI_IMG_KEYS, coerce_scalar, gen_done_or_502,
                                 images_response, images_uploads, multipart_list, parse_size)
from responses_bridge import (chat_to_responses, response_shell, responses_stream,
                              responses_to_chat)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path("config.yaml")

def load_config() -> None:
    """Read config.yaml and (re)bind module-level config values."""
    global config, config_backends, config_virtual_models, virtual_models
    global health_check_interval, api_key
    global stats_cfg, log_per_call, model_prefix, max_concurrent_default
    global image_models, jobs_cfg
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config_backends = sorted(config["backends"], key=lambda b: b.get("priority", 100))
    # Chat aliases: config is the base, UI-managed store entries merge over it (see
    # rebuild_virtual_models). `virtual_models` is the effective dict the router reads.
    config_virtual_models = config.get("virtual_models", {})
    virtual_models = dict(config_virtual_models)
    health_check_interval = config.get("health_check_interval", 30)
    api_key = config.get("api_key")
    stats_cfg = config.get("stats") or {}
    log_per_call = config.get("log_per_call", True)
    model_prefix = config.get("model_prefix", True)
    # Global in-flight cap applied to every backend that doesn't set its own
    # `max_concurrent`. None = unlimited (legacy behaviour).
    max_concurrent_default = config.get("max_concurrent")
    # Generation (image/video/tts) aliases → ordered backend+workflow candidates.
    # Kept separate from `virtual_models` (LLM routing) in Phase 1; unified into
    # the capability router in Phase 3. Hot-reloaded. `jobs` is startup-only.
    image_models = config.get("image_models") or {}
    jobs_cfg = config.get("jobs") or {}


def log_config_summary() -> None:
    logger.info(f"Loaded {len(backends)} backend(s):")
    for b in backends:
        state = "ENABLED " if is_enabled(b) else "DISABLED"
        cap = backend_max_concurrent(b)
        cap_s = f"  max_concurrent={cap}" if cap is not None else ""
        tier = "paid" if b.get("paid") else "free"
        logger.info(f"  [{state}] {b['name']:25} {tier}  url={b['url']}{cap_s}")
    logger.info(f"Loaded {len(virtual_models)} virtual alias(es):")
    for alias, mapping in virtual_models.items():
        if isinstance(mapping, dict):
            for bname, entry in mapping.items():
                if isinstance(entry, dict):
                    logger.info(f"  {alias:15} → [{bname}] {entry.get('model')}")
                else:
                    logger.info(f"  {alias:15} → [{bname}] {entry}")
        else:
            logger.info(f"  {alias:15} → {mapping}  (all backends)")
    logger.info(f"health_check_interval={health_check_interval}s  api_key={'set' if api_key else 'unset'}")


# Initial load — populates module globals
config: dict
config_backends: list[dict]                                    # backends from config.yaml
backends: list[dict]                                           # effective = config + store (UI-added)
config_virtual_models: dict                                    # chat aliases from config.yaml
virtual_models: dict                                           # effective = config + store (UI-managed)
health_check_interval: int
api_key: Optional[str]
stats_cfg: dict
log_per_call: bool
model_prefix: bool
max_concurrent_default: Optional[int]
image_models: dict
jobs_cfg: dict
load_config()
backends = list(config_backends)                              # store merges in once it's active (lifespan)


def backend_host(b: dict) -> str:
    """The physical box a backend runs on: the explicit `host` field, else the
    URL's hostname/IP. Backends sharing an IP group automatically — k12/evo run
    llama-swap AND ComfyUI on one GPU, and host-level policies (see
    docs/host-coordination-plan.md) need to know they belong together."""
    h = (b.get("host") or "").strip()
    if h:
        return h
    try:
        return urlparse(b.get("url") or "").hostname or b["name"]
    except Exception:
        return b["name"]


def rebuild_backends() -> None:
    """Effective backend list = config backends, with UI-added (store) backends
    merged in by name (store overrides config for the same name). Kept in the
    configured `priority` order so listings stay stable — priority no longer routes
    anything (spec 2026-09-01). Also rebuilds the host grouping maps. Call after
    config reload or any store backend change."""
    global backends, backend_hosts, host_backends
    merged = {backend_id(b): b for b in config_backends}
    if store.is_active():
        for b in store.list_backends():
            merged[backend_id(b)] = b      # store overrides config per (name, type)
    backends = sorted(merged.values(), key=lambda b: b.get("priority", 100))   # list order only
    for b in backends:
        # Cost tier for the scheduler (spec 2026-09-01): a paid backend is a candidate
        # only when no unpaid one is free. Normalized here — config and store entries
        # may omit the key entirely. A cloud backend (Meshy, Tripo) bills per task, so it
        # is ALWAYS paid.
        b["paid"] = True if b.get("type") in adapters.CLOUD_TYPES else bool(b.get("paid"))
    backend_hosts = {backend_id(b): backend_host(b) for b in backends}
    host_backends = {}
    for bid, h in backend_hosts.items():
        host_backends.setdefault(h, []).append(bid)
    apply_hosts()
    rebuild_route_index()                  # backend set/enabled flags changed


def apply_hosts() -> None:
    """Refresh the per-host settings cache (labels + policy flags) from the store —
    read on the request path by _media_busy_hosts, so no DB hit per request."""
    global hosts_meta
    hosts_meta = store.get_hosts() if store.is_active() else {}


def _media_busy_hosts() -> set:
    """Hosts whose ComfyUI backend is generating RIGHT NOW. Their LLM siblings
    share the box's GPU — a llama-swap model (re)load then aborts on the VRAM
    ComfyUI holds (measured on k12-gpu, see docs/host-coordination-plan.md).
    resolve_routes sorts chat candidates on these hosts LAST (never drops them:
    with alternatives the collision never happens, without them best effort +
    the 502-failover still apply). Per-host opt-out: avoid_llm_during_media
    false (Hosts editor); absent = on — it only ever bites shared boxes."""
    out = set()
    for bid, n in backend_inflight.items():
        if n > 0 and bid.startswith("comfyui:"):
            h = backend_hosts.get(bid)
            if h and (hosts_meta.get(h) or {}).get("avoid_llm_during_media", True):
                out.add(h)
    return out


def rebuild_virtual_models() -> None:
    """Effective chat aliases = config `virtual_models`, with UI-managed (store)
    entries merged over them by alias (store overrides config for the same name).
    Also refreshes the per-alias park times, reasoning defaults, voice defaults
    and sampling defaults. Call after config reload or any store chat-alias
    change."""
    global virtual_models, alias_park_s, alias_reasoning, alias_voice, alias_sampling
    merged = dict(config_virtual_models)
    if store.is_active():
        merged.update(store.list_chat_aliases())
    virtual_models = merged
    park = dict(config.get("alias_park") or {}) if isinstance(config, dict) else {}
    if store.is_active():
        park.update(store.get_alias_park())
    alias_park_s = park
    alias_reasoning = store.get_alias_reasoning() if store.is_active() else {}
    alias_voice = store.get_alias_voice() if store.is_active() else {}
    alias_sampling = store.get_alias_sampling() if store.is_active() else {}
    apply_voice_library()
    apply_reasoning_rules()                # refresh the store-backed reasoning rule cache
    rebuild_route_index()                  # alias mappings changed

# ── State ─────────────────────────────────────────────────────────────────────

backend_models: dict[str, set[str]] = {}                       # name → {model_id, ...}
backend_healthy: dict[str, bool] = {}                          # name → bool
backend_error: dict[str, dict] = {}                            # bid → why discovery failed (see _classify_error)
backend_pricing: dict[str, dict[str, dict[str, float]]] = {}   # name → {model_id → {input, output}}
backend_loras: dict[str, set[str]] = {}                        # id → {lora filename, ...} (ComfyUI)
# id → {model_id → context window (tokens)} as LEARNED by discovery (persisted, merged —
# a llama-swap model says its n_ctx only while loaded). The admin's per-backend
# `model_context` rules sit above this; adapters.model_context_for() resolves both.
backend_context: dict[str, dict[str, int]] = {}
backend_inflight: dict[str, int] = {}                          # name → current in-flight requests
backend_hosts: dict[str, str] = {}                             # bid → host (explicit `host` or URL IP)
backend_tps: dict[str, float] = {}                             # bid → EWMA output tok/s (speed routing; runtime-only, resets on restart)
host_backends: dict[str, list] = {}                            # host → [bid, …] (rebuild_backends)
hosts_meta: dict[str, dict] = {}                               # host → {label, avoid_llm_during_media, …}
backend_adapters: dict = {}                                    # name → BackendAdapter instance
gen_speed: dict = {}                                           # "alias|bid" → EMA seconds of a successful media job
gen_exec_faults: dict = {}                                     # "alias|bid" → proven execution-fault record (scheduler.exec_fault_*)
gen_progress: dict = {}                                        # job id → live progress from the backend's ws feed


def _note_progress(job_id: str, info: Optional[dict]) -> None:
    """Adapter hook (`AdapterContext.note_progress`): record one job's live progress,
    or drop it when `info` is None (the run ended — the job ROW owns the outcome from
    then on, and a stale 'running 25/35' next to a finished job would be a lie).

    In memory on purpose: this updates several times a second per running job, and none
    of it is worth a SQLite write — it is a view of something already in flight, and a
    gateway restart legitimately forgets it (the jobs it belonged to are failed by
    `reconcile_orphans` anyway)."""
    if not job_id:
        return
    if info is None:
        gen_progress.pop(job_id, None)
        return
    gen_progress[job_id] = {**info, "at": time.time()}
backend_last_key: dict = {}                                    # bid → type key last DISPATCHED (media: alias, LLM: real model)
backend_vram_key: dict = {}                                    # bid → alias whose model set a ComfyUI GPU HOLDS; None = unknown/mixed (see _claim_gen_backend)


def _note_gen_speed(alias: str, bid: str, seconds: float) -> None:
    """Fold one successful media generation into the alias+backend duration EMA.
    This is the media counterpart of _note_speed (tok/s): a mesh/image job has no
    token count, so wall-clock seconds per job is the only comparable signal."""
    if seconds <= 0:
        return
    k = f"{alias}|{bid}"
    gen_speed[k] = scheduler.ema(gen_speed.get(k), seconds)


def _settle_exec_faults(alias: str, ok_bid: str, exec_faults: list) -> None:
    """Called on a SUCCESS: settle the execution faults this job collected on the way.

    A candidate that failed to execute where this one then succeeded is PROVEN to be
    the broken part — same alias, same workflow, same request, different outcome. Only
    such proven faults are charged, which is what separates "this backend is broken"
    from "this request is broken": if every candidate had failed we would never get
    here and nobody is charged (see the tail of `_run_job`).

    The success itself clears `ok_bid`, so the fault count means CONSECUTIVE faults and
    a repaired backend is first-class again the moment it delivers once."""
    scheduler.exec_fault_clear(gen_exec_faults, f"{alias}|{ok_bid}")
    now = time.time()
    for bid, name, err in exec_faults:
        rec = scheduler.exec_fault_note(gen_exec_faults, f"{alias}|{bid}", now,
                                        error=_err_text(err))
        if rec["until"] > now:
            logger.warning(
                f"[{name}] quarantined for '{alias}' for "
                f"{int(scheduler.EXEC_QUARANTINE_S / 60)} min — {rec['fails']} execution "
                f"failures in a row where another backend succeeded: {rec['error']}")
        else:
            logger.info(f"[{name}] execution fault {rec['fails']}/"
                        f"{scheduler.EXEC_FAULT_THRESHOLD} for '{alias}' "
                        f"(another backend ran the same job): {rec['error']}")


def _gen_speed_of(alias: str):
    """speed_of callable for scheduler.order_ready over media candidates: higher is
    better, and an unmeasured backend sorts first (probe-once).

    "Probe-ONCE" is the operative word: a candidate that has produced a proven
    execution fault has HAD its probe and sorts LAST instead. Without that, a backend
    which answers but cannot execute is unbeatable — it never completes a job, so it
    never gets a gen_speed sample, so it keeps the unmeasured head start and wins the
    ordering again on the very next retry (measured 2026-09-03 on comfyui-strix: four
    consecutive retries, two idle healthy backends)."""
    def speed(backend: dict, _cand) -> float:
        k = f"{alias}|{backend_id(backend)}"
        s = gen_speed.get(k)
        if s is None:
            return 0.0 if scheduler.exec_probed(gen_exec_faults, k) else float("inf")
        return 1.0 / max(s, 0.001)
    return speed


# ── Health / Discovery ────────────────────────────────────────────────────────

def is_enabled(backend: dict) -> bool:
    return backend.get("enabled", True)


def backend_id(backend: dict) -> str:
    """Stable unique key for a backend = type:name. The *name* is only a display label
    and a type-scoped routing reference, so an LLM and a ComfyUI backend may share a
    name. All runtime state (models/health/inflight/adapters) is keyed by this id."""
    return f'{backend.get("type", "openai")}:{backend["name"]}'


def _is_gen(b: dict) -> bool:
    """A generation backend (ComfyUI, Meshy, Tripo): routed by POST /v1/generations, never
    listed in the chat catalogs. Type-agnostic replacement for `type == "comfyui"`."""
    return b.get("type") in adapters.GEN_TYPES


def enabled_backends() -> list[dict]:
    return [b for b in backends if is_enabled(b)]


def backend_auth_headers(backend: dict) -> dict:
    key = backend.get("api_key")
    return {"authorization": f"Bearer {key}"} if key else {}


# ── In-flight cap / "busy" routing ─────────────────────────────────────────────
# Per-backend live request counter. A backend at/above its `max_concurrent` cap is
# "busy": routing skips it (spilling to the next backend) and the routing
# dashboard flags it. Lets one slow llama.cpp box (--parallel 1) shed concurrent
# load onto the rest of the fleet instead of queueing/overflowing.

def backend_max_concurrent(backend: dict) -> Optional[int]:
    """In-flight cap for this backend: its own `max_concurrent`, else the global
    default, else None (unlimited)."""
    v = backend.get("max_concurrent", max_concurrent_default)
    return v if isinstance(v, int) and v > 0 else None


def backend_busy(backend: dict) -> bool:
    """True when the backend is at/above its in-flight cap → temporarily skipped."""
    cap = backend_max_concurrent(backend)
    return cap is not None and backend_inflight.get(backend_id(backend), 0) >= cap


# ── Graceful drain (take a backend offline once idle) ─────────────────────────
# A draining backend takes NO new requests (excluded from routing) but lets its
# in-flight requests finish; once in-flight hits 0 it is disabled (persisted) — so a
# backend can be pulled for maintenance without aborting running requests.
_draining: set = set()                  # backend ids currently draining


def is_draining(backend: dict) -> bool:
    return backend_id(backend) in _draining


def _inflight_inc(name: str) -> None:
    backend_inflight[name] = backend_inflight.get(name, 0) + 1


def _inflight_dec(name: str) -> None:
    backend_inflight[name] = max(0, backend_inflight.get(name, 0) - 1)
    if name in _draining and backend_inflight.get(name, 0) <= 0:
        _finalize_drain(name)            # last in-flight request finished → go offline
    _notify_slot_free()


# ── Speed signal (throughput EWMA) ──────────────────────────────────────────────
# Per-backend generation throughput in output tokens/sec, folded from each completed
# call. This is the LLM speed signal of the unified scheduler: resolve_routes orders
# ready candidates fastest-first within their cost tier. Runtime-only — no
# persistence; unmeasured backends sort first so each gets probed once.
_TPS_ALPHA = 0.3                        # EWMA weight of the newest sample (higher = more reactive)
_TPS_MIN_TOKENS = 16                    # ignore tiny completions — fixed overhead dominates their tok/s


def _note_speed(bid: str, out_tok: int, duration_ms: int, status: int) -> None:
    """Fold one completed dispatch into a backend's tok/s EWMA. Only successful
    text generations with enough tokens count; TTS/embeddings (0 tokens) and errors
    are skipped. Called synchronously from the adapter's stats path (loop thread)."""
    if status != 200 or out_tok < _TPS_MIN_TOKENS or duration_ms <= 0:
        return
    tps = out_tok / (duration_ms / 1000.0)
    prev = backend_tps.get(bid)
    backend_tps[bid] = tps if prev is None else _TPS_ALPHA * tps + (1 - _TPS_ALPHA) * prev


# ── Live LLM-call registry (dashboard "running calls") ────────────────────────
# Currently-running chat/completions/embeddings forwards, so the console can show
# what's in flight right now. Registered when dispatch starts, dropped on
# completion (incl. the streamed-finally) — same lifecycle as the in-flight
# counter. Once finished, the call lands in stats; the dashboard's 5-minute
# "recently ended" view reads from there, so this holds only the live set.
_active_calls: dict = {}                 # token → {alias, model, backend, source, endpoint, stream, started}
_active_seq: list = [0]


def _active_register(meta: dict) -> int:
    _active_seq[0] += 1
    token = _active_seq[0]
    _active_calls[token] = {**meta, "started": time.time()}
    return token


def _active_done(token) -> None:
    _active_calls.pop(token, None)


# ── Call parking (a queue: hold instead of 503 when all backends are busy) ─────
# Parking is the DEFAULT for chat: when every backend mapping an alias is at its
# in-flight cap, the call is held in a FIFO queue until a mapping backend frees
# (then dispatched) or its park time elapses (→ 503). Each entry stays in `_parked`
# for its whole wait — keeping its FIFO position and staying visible in the console.
# `_inflight_dec` wakes all waiters in order; the event loop is single-threaded, so
# the oldest resumes first and claims the freed slot (dispatch increments in-flight
# before its first await), the rest re-check and wait again. Park time is per-alias
# (`alias_park_s`), else the global default below; 0 disables parking for an alias.
park_timeout_s: float = 60.0            # global default park time (Server tab); per-alias overrides in alias_park_s
async_park_timeout_s: float = 600.0
# How long a parked generation job rides out a poll that finds ZERO candidates before
# giving up. A job only parks because it HAD candidates (all busy), so an empty poll is
# a transient health flap — a busy ComfyUI routinely drops its /object_info discovery
# poll mid-generation and is briefly marked DOWN. Covers a multi-cycle flap (observed
# ~30 s) without hanging a genuinely-offline alias to the full park deadline.
park_health_grace_s: float = 90.0
max_parked: int = 100
# Freed-backend type affinity (spec 2026-09-01): a woken parked call claims a freed
# backend only if the scheduler designates IT for that backend, so a backend prefers a
# waiter that needs the model it just ran (no reload). The affinity may hold a queued
# call back at most this long — beyond it the call counts as overdue and is served
# strictly oldest-first by the next free backend that can run it.
affinity_max_wait_s: float = 120.0
alias_park_s: dict = {}                 # alias → park seconds (config + store); absent → default, 0 → off
alias_reasoning: dict = {}              # alias → "off"|"on" default (store); absent → auto. Client wins.
alias_voice: dict = {}                  # alias → {voice, ref_text} TTS defaults (store). Client wins.
alias_sampling: dict = {}               # alias → {param: value} sampling defaults (store). Client wins.
voice_library: dict = {}                # name → {ref_text, file, remote, shipped} (store voice_library)
_parked: list = []                     # ordered FIFO of live parked-call entries (rich, for the console)
_park_seq: list = [0]
_probing: set = set()                  # backend ids with a discovery poll in flight (see refresh_backend)

# How often an UNHEALTHY backend is re-polled while something waits for capacity, so a
# backend that came back outside the gateway is noticed in seconds instead of a whole
# health_check_interval. 0 disables the fast probe (Server tab).
fast_probe_interval_s: float = 3.0
# LAN scan for backends (Backends tab → Scan network; docs/superpowers/specs/
# 2026-09-08-lan-scan-design.md). Manual only — one scan at a time, nothing is added
# by itself. `scan_cidrs` empty = the /24 of every IPv4 address of this host.
scan_cidrs: list[str] = []
scan_ports: list[int] = list(netscan.DEFAULT_PORTS)
_scan: dict = {"task": None, "result": None}
# Media jobs park in their own poll loops rather than in `_parked`, so they announce
# themselves here instead. A TIMESTAMP, not a counter: a cancelled or crashed job task
# can never leave a phantom waiter behind, it just stops refreshing.
_gen_wait_at: list = [0.0]
_GEN_WAIT_TTL_S = 5.0                  # how long one ping counts as "still waiting"
# The media queue (spec 2026-09-01, "Designated taker"): every generation job that has
# to wait registers here for its whole run, so a freed ComfyUI backend goes to the job
# it belongs to (overdue first, else one needing the alias it just ran) instead of to
# whoever polls next. Separate pool from `_parked` — an LLM backend never serves media
# and vice versa. Entries in enqueue order:
#   {job_id, alias, enqueued_at (monotonic), eligible, force, claimed?}
# `claimed` marks an entry that holds a backend slot: still registered, no longer
# waiting (see _gen_waiting_pool).
_gen_waiting: list = []


def _gen_wait_ping() -> None:
    """A parked generation job just polled for a free backend (see _run_gen_parked)."""
    _gen_wait_at[0] = time.monotonic()


def _capacity_wanted() -> bool:
    """Is anything waiting for a backend right now? Drives the fast probe."""
    return bool(_parked) or (time.monotonic() - _gen_wait_at[0]) < _GEN_WAIT_TTL_S

# Normalized reasoning rules (store-backed, UI-editable). Cached here and refreshed on
# save so the resolver doesn't hit the DB per request. See reasoning.py.
reasoning_rules: list = []


def apply_reasoning_rules() -> None:
    global reasoning_rules
    reasoning_rules = store.get_reasoning_rules() if store.is_active() else []


# Shared outbound HTTP client — ONE connection pool (keep-alive, no TLS re-handshake)
# for every proxied call, discovery poll, and ComfyUI helper. Constructed at import
# (connections open lazily), closed in lifespan shutdown. Carries only a safety-net
# timeout: every hot call site passes its own `timeout=` per request.
http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=200, max_keepalive_connections=50,
                        keepalive_expiry=30.0),
    timeout=httpx.Timeout(30.0),
)


def _next_park_id() -> int:
    _park_seq[0] += 1
    return _park_seq[0]


def _park_time_for(alias: str) -> float:
    v = alias_park_s.get(alias)
    if v is None:
        return park_timeout_s
    try:
        return max(0.0, float(v))
    except (TypeError, ValueError):
        return park_timeout_s


def _notify_slot_free() -> None:
    # Wake every parked call in FIFO order; each re-checks its own alias's routes.
    # The oldest resumes first and claims the freed slot; the rest wait again.
    for entry in _parked:
        ev = entry.get("event")
        if ev is not None and not ev.is_set():
            ev.set()


_comfy_restarting: set[str] = set()      # backend ids with a restart() in flight


def _spawn_comfy_restart(backend: dict, adapter, why: str) -> None:
    bid = backend_id(backend)
    _comfy_restarting.add(bid)
    logger.warning(f"[{backend['name']}] restarting ComfyUI service ({why})")

    async def _run():
        try:
            await adapter.restart()
        except Exception as e:
            logger.warning(f"[{backend['name']}] ComfyUI restart failed: {e}")
        finally:
            _comfy_restarting.discard(bid)
    asyncio.create_task(_run())


def _maybe_auto_restart(backend: dict, adapter) -> None:
    """Opt-in: one restart attempt per cooldown when the executor is stuck.
    Deliberately NOT gated on inflight — stuck means nothing is executing, a
    pending gateway prompt is lost either way (its poll fails over/parks)."""
    bid = backend_id(backend)
    if not backend.get("auto_restart") or bid in _comfy_restarting:
        return
    cooldown = int(backend.get("restart_cooldown_s") or 600)
    if adapter.last_restart and time.time() - adapter.last_restart < cooldown:
        return
    _spawn_comfy_restart(backend, adapter, "executor stuck — auto-restart")


def restart_comfy_backend(bid: str) -> bool:
    """UI hook (Backends tab): fire-and-forget ComfyUI service restart."""
    b = next((x for x in backends if backend_id(x) == bid), None)
    adapter = backend_adapters.get(bid)
    if b is None or adapter is None or b.get("type") != "comfyui" or bid in _comfy_restarting:
        return False
    _spawn_comfy_restart(b, adapter, "manual via UI")
    return True


def _classify_error(e: Exception) -> dict:
    """Why a discovery poll failed, in a form the console can act on.

    "down" alone sends you hunting a network fault that may not exist: a rejected
    credential and an unplugged host look identical in the Backends tab, yet one is
    fixed in the api-key field and the other on the host. `kind` carries that
    distinction; `detail` keeps the raw message for the tooltip."""
    detail = str(e) or e.__class__.__name__
    status = None
    resp = getattr(e, "response", None)
    if resp is not None:
        status = getattr(resp, "status_code", None)
    if isinstance(e, ComfyExecutorStuck):
        kind = "stuck"
    elif status in (401, 403):
        kind = "auth"                       # credential rejected — not a network problem
    elif status == 404:
        kind = "not_found"                  # wrong base url / path (e.g. url ends in /v1)
    elif status == 429:
        kind = "rate_limit"
    elif status is not None and status >= 500:
        kind = "upstream"
    elif isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
        kind = "unreachable"
    elif isinstance(e, httpx.TimeoutException):
        kind = "timeout"
    else:
        kind = "error"
    return {"kind": kind, "status": status, "detail": detail[:300], "since": int(time.time())}


def merge_learned_context(old: Optional[dict], new: Optional[dict]) -> dict[str, int]:
    """Fold one discovery poll's context values into what earlier polls learned. A
    model absent from THIS poll keeps its value (llama-swap reports n_ctx only for
    loaded models); a model present again overwrites (the operator may have changed
    --ctx-size). Pure, so test_context_length.py can pin it."""
    merged = dict(old or {})
    for k, v in (new or {}).items():
        if isinstance(v, (int, float)) and v > 0:
            merged[str(k)] = int(v)
    return merged


def model_context(backend: dict, model: str) -> Optional[int]:
    """Context window of `model` on `backend` — admin rule first, learned value second,
    None when neither knows (see adapters.model_context_for)."""
    return adapters.model_context_for(backend, model, backend_context.get(backend_id(backend), {}))


def _context_min(pairs) -> Optional[int]:
    """The catalog value for an entry SEVERAL backends may serve (a bare id, an alias):
    the smallest known window, because the client cannot pick the backend — a prompt
    sized for the largest one 400s on the smallest. Unknown backends do not vote."""
    vals = [v for v in (model_context(b, m) for b, m in pairs) if v]
    return min(vals) if vals else None


def _alias_context(alias: str, candidates: list) -> Optional[int]:
    """Smallest known window over the backends an alias resolves on (see _context_min)."""
    pairs = []
    for b in candidates:
        real, _prio = alias_entry(alias, b["name"])
        if real is not None and real in backend_models.get(backend_id(b), set()):
            pairs.append((b, real))
    return _context_min(pairs)


def _with_context(entry: dict, ctx: Optional[int]) -> dict:
    """Attach `context_length` only when known — never 0, never a guess."""
    if ctx:
        entry["context_length"] = ctx
    return entry


async def refresh_backend(backend: dict, client: httpx.AsyncClient) -> None:
    """Poll a backend's capabilities via its adapter and update discovery state.

    Discovery is protocol-specific (delegated to the adapter); the up/down
    logging and the module-global state (`backend_models` / `backend_pricing` /
    `backend_healthy`) stay owned here so routing keeps a single source of truth.
    """
    bid, label = backend_id(backend), backend["name"]
    adapter = backend_adapters.get(bid)
    if adapter is None:
        return
    if bid in _probing:
        return                             # a poll of this backend is already in flight
    _probing.add(bid)
    try:
        caps = await adapter.discover(client)
        changed = caps.models != backend_models.get(bid)
        if changed and store.is_active():
            await asyncio.to_thread(store.save_backend_models, bid, caps.models)  # persist on change
        backend_models[bid] = caps.models
        backend_pricing[bid] = caps.pricing
        backend_loras[bid] = getattr(caps, "loras", set()) or set()
        learned = merge_learned_context(backend_context.get(bid), getattr(caps, "context", None))
        if learned != backend_context.get(bid, {}):
            backend_context[bid] = learned
            if store.is_active():
                await asyncio.to_thread(store.save_backend_context, bid, learned)
        if changed:
            rebuild_route_index()          # model set changed → refresh routing candidates
        was_healthy = backend_healthy.get(bid, False)
        if not was_healthy:
            logger.info(f"[{label}] UP  — {len(caps.models)} models, {len(caps.pricing)} priced")
        backend_healthy[bid] = True
        backend_error.pop(bid, None)
        if not was_healthy or changed:     # a backend came online / gained models →
            _notify_slot_free()            # let parked calls re-evaluate and grab it
    except Exception as e:
        info = _classify_error(e)
        # Keep the ORIGINAL failure time across repeated polls of the same fault, so
        # the console can say how long it has been broken.
        prev = backend_error.get(bid)
        if prev and prev.get("kind") == info["kind"] and prev.get("status") == info["status"]:
            info["since"] = prev["since"]
        backend_error[bid] = info
        if backend_healthy.get(bid, True):
            hint = " (credential rejected — check the api key)" if info["kind"] == "auth" else ""
            logger.warning(f"[{label}] DOWN — {e}{hint}")
        backend_healthy[bid] = False
        backend_pricing[bid] = {}
        backend_loras[bid] = set()
        # backend_models is intentionally NOT cleared — keep the last-known (persisted)
        # set so a bare model id still resolves to this offline backend → 503, not 403.
        if isinstance(e, ComfyExecutorStuck):
            _maybe_auto_restart(backend, adapter)
    finally:
        _probing.discard(bid)


async def health_loop() -> None:
    """Poll every enabled backend, then sleep. Backends are polled CONCURRENTLY: a
    sequential loop adds each unreachable backend's connect timeout to the cycle, so
    the wait for a returning backend grew with the number of broken ones."""
    while True:
        await asyncio.gather(*[refresh_backend(b, http_client) for b in enabled_backends()],
                             return_exceptions=True)     # refresh_backend absorbs its own errors
        await asyncio.sleep(health_check_interval)


async def fast_probe_loop() -> None:
    """Re-poll UNHEALTHY backends quickly while calls or jobs are waiting for capacity.

    Re-routing onto a backend that just came back already works — `refresh_backend`
    wakes the park queue on DOWN→UP and parked generation jobs re-resolve their routes
    every 2 s. What was slow is NOTICING: on the normal `health_check_interval`
    (default 30 s) a backend restarted outside the gateway stays invisible for up to a
    full cycle, and that wait is the whole delay a queued job sees.

    Only unhealthy backends are probed — a healthy-but-busy one needs no poll, its
    freed slot is announced by `_inflight_dec` → `_notify_slot_free`."""
    while True:
        await asyncio.sleep(max(1.0, fast_probe_interval_s or 1.0))
        if not fast_probe_interval_s or not _capacity_wanted():
            continue
        targets = [b for b in enabled_backends()
                   if not backend_healthy.get(backend_id(b), False)]
        if targets:
            await asyncio.gather(*[refresh_backend(b, http_client) for b in targets],
                                 return_exceptions=True)


def reload_config() -> None:
    """Re-read config.yaml and apply. Keeps old config on parse error."""
    old_ids = {backend_id(b) for b in backends}
    try:
        load_config()
    except Exception as e:
        logger.error(f"Config reload FAILED, keeping previous config: {e}")
        return
    rebuild_backends()                 # re-merge config + store backends
    rebuild_virtual_models()           # re-merge config + store chat aliases
    apply_server_settings()            # re-apply UI server overrides over fresh config
    rebuild_users()                    # reload multi-user identities
    # Drop state for backends removed from config
    new_ids = {backend_id(b) for b in backends}
    for stale in old_ids - new_ids:
        backend_healthy.pop(stale, None)
        backend_error.pop(stale, None)
        backend_models.pop(stale, None)
        backend_pricing.pop(stale, None)
        backend_loras.pop(stale, None)
        backend_inflight.pop(stale, None)
        logger.info(f"  removed backend [{stale}] — state cleared")
    build_backend_adapters()       # rebind adapters to the new backend dicts
    logger.info("Config reloaded.")
    log_config_summary()


async def watch_config(path: "str | Path", on_change: Callable[[], None]) -> None:
    """Call `on_change` once per save of `path`, for the life of the process.

    Watches the config's DIRECTORY and filters events down to the one file, because an
    inotify watch on a FILE dies with that file's inode — and nearly every editor saves
    by writing a sibling and renaming it over the original, which mints a new one.
    Measured 2026-09-08 on a fresh Debian 13 / Python 3.13 / watchfiles 1.2.0 install:
    the old `awatch(CONFIG_PATH)` form reported exactly ONE of four edits — the first
    `sed -i`; `stat` showed the inode had changed, and from then on nothing was seen,
    not an in-place `echo >>`, not a second `sed -i`, not a `cp good config.yaml`. The
    failure is SILENT: the gateway keeps serving the config it read first while README
    and CLAUDE.md promise a hot reload on save, so the operator "restoring" a bad edit
    gets no reload and no complaint. A directory watch survives the rename-replace.
    `recursive=False` keeps jobs/ and its artifacts out of the watch entirely; the
    store/stats/jobs DB files that sit beside config.yaml still raise events, and the
    filter drops them before anything reloads.

    A config.yaml may also BE a symlink — the stub-instance harness is a directory of
    them — and then it has two identities: the LEXICAL path the operator's editor writes
    and the RESOLVED path the bytes live at. Measured 2026-09-08, same install: resolving
    first and watching only the target's parent saw 0 of the saves made through the link,
    because an editor saving the instance-dir config.yaml REPLACES the link with a regular
    file — an event in the lexical parent, which was not watched — and from then on the
    watch is aimed at a file nobody edits. So both parents are watched (deduplicated to
    one entry when the path is not a link) and an event matching EITHER identity counts.
    `target` is computed once at start: after the link has been replaced, `resolve()`
    returns the lexical path itself, and the lexical arm of the filter is what keeps
    later in-place edits of the now-regular file visible. The old target's parent stays
    watched and an edit of the orphaned target still triggers one (idempotent) reload —
    a log line, nothing more. The mirror case is NOT covered on purpose: a config that
    BECOMES a symlink after start keeps only the lexical arm.
    """
    lexical = Path(path).absolute()      # where the operator's editor writes
    target = lexical.resolve()           # where the bytes live (== lexical when not a link)

    def _is_config(_change, p: str) -> bool:
        q = Path(p)
        if q.absolute() == lexical:
            return True
        try:
            return q.resolve() == target
        except (OSError, RuntimeError):  # a symlink LOOP raises RuntimeError; an exception
            return False                 # out of a watch_filter kills the awatch silently

    async for _ in awatch(*{lexical.parent, target.parent}, recursive=False,
                          watch_filter=_is_config):
        on_change()


async def watch_config_loop() -> None:
    def _reload() -> None:
        logger.info(f"Detected change in {CONFIG_PATH} — reloading")
        reload_config()

    await watch_config(CONFIG_PATH, _reload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI-Hub")

    # Writable store backs all UI-managed state (backends, chat aliases, generation
    # aliases) — always on so the /ui console works regardless of image_models.
    # Brought up BEFORE discovery so UI-added (store) backends are merged in and
    # discovered too.
    store.init(jobs_cfg.get("store_path", "store.db"))
    store.bootstrap(image_models)
    backend_models.update(store.load_backend_models())   # seed last-known models (offline → 503, not 403)
    backend_context.update(store.load_backend_context())  # learned context windows survive a restart
    apply_server_settings()            # overlay UI-managed server settings onto config
    rebuild_users()                    # load multi-user identities from the store
    rebuild_backends()                 # merge UI-added backends from the store
    rebuild_virtual_models()           # merge UI-managed chat aliases from the store
    build_backend_adapters()           # rebind adapters to the merged backend list
    logger.info("UI: console at /ui")

    # Generation job store: on whenever image_models are configured (or forced via
    # `jobs.enabled`).
    jobs_prune_task: Optional[asyncio.Task] = None
    if image_models or jobs_cfg.get("enabled"):
        jobs.init(jobs_cfg.get("db_path", "jobs.db"),
                  jobs_cfg.get("blob_dir", "jobs"),
                  jobs_cfg.get("default_ttl_s", 86400))
        jobs_prune_task = asyncio.create_task(jobs.prune_loop(jobs_cfg.get("prune_interval_s", 3600)))
        # Seed the media gen-speed EMA from the job store so a restart does not have
        # to re-probe every backend once per alias. Best-effort: a missing/odd jobs DB
        # must never hold up the boot.
        try:
            for seed_alias, seed_bname, seed_avg_ms in jobs.gen_speed_rows():
                seed_b = next((b for b in backends if b["name"] == seed_bname
                               and _is_gen(b)), None)
                if seed_b is not None and seed_avg_ms:
                    gen_speed.setdefault(f"{seed_alias}|{backend_id(seed_b)}",
                                         float(seed_avg_ms) / 1000.0)
        except Exception as e:
            logger.info(f"gen-speed seed skipped: {e}")

    log_config_summary()
    await asyncio.gather(*[refresh_backend(b, http_client) for b in enabled_backends()])
    health_task = asyncio.create_task(health_loop())
    probe_task = asyncio.create_task(fast_probe_loop())
    watch_task = asyncio.create_task(watch_config_loop())

    # Stats: record calls + prune, but NO separate server — the dashboard lives in
    # /ui → Statistic now (so no extra port/bind).
    prune_task: Optional[asyncio.Task] = None
    if stats_cfg.get("enabled"):
        stats.init(stats_cfg.get("db_path", "stats.db"), stats_cfg.get("blob_dir", "calls"))
        prune_task = asyncio.create_task(stats.prune_loop(stats_cfg.get("retention_days", 0)))
        logger.info("stats: recording on; dashboard at /ui → Statistic")
    # snapshot the restart-only server state actually in effect, so the UI can flag
    # when an edited setting needs a restart to apply.
    _server_runtime.update(
        stats_enabled=bool(stats_cfg.get("enabled")),
        stats_db_path=stats_cfg.get("db_path", "stats.db"),
        stats_retention_days=stats_cfg.get("retention_days", 0),
        jobs_enabled=jobs_prune_task is not None,        # actually running
        jobs_db_path=jobs_cfg.get("db_path", "jobs.db"),
        jobs_blob_dir=jobs_cfg.get("blob_dir", "jobs"),
        jobs_default_ttl_s=jobs_cfg.get("default_ttl_s", 86400),
        jobs_prune_interval_s=jobs_cfg.get("prune_interval_s", 3600),
    )

    yield
    health_task.cancel()
    probe_task.cancel()
    watch_task.cancel()
    if jobs_prune_task is not None:
        jobs_prune_task.cancel()
    if prune_task is not None:
        prune_task.cancel()
    await http_client.aclose()         # drain the shared connection pool last


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI-Hub", lifespan=lifespan)
admin.register(app)                     # generation management UI at /ui


@app.exception_handler(HTTPException)
async def _rejected_call(request: Request, exc: HTTPException):
    """Log every API call the gateway REFUSES, then render it.

    Stats rows were written by the adapter only — so a request that never reached a
    backend ("No healthy backend", park timeout, quota, unknown alias) left no trace
    at all, which is exactly the request you go looking for in LLM Calls. Recorded
    here, centrally, because a refusal can be raised from a dozen places.

    `gw_dispatched` guards the double entry: once a backend answered, the adapter
    owns the row (e.g. /v1/responses re-raises an upstream error as HTTPException).
    """
    if request.url.path.startswith("/v1/") and not getattr(request.state, "gw_dispatched", False):
        _record_rejected(request, exc)
    if request.url.path.startswith("/v1/messages"):
        return _messages_error(exc.status_code, exc.detail, getattr(exc, "headers", None))
    return await http_exception_handler(request, exc)


def _record_rejected(request: Request, exc: HTTPException) -> None:
    """Fire-and-forget stats row for a refused call. Never raises into the response
    path: a logging failure must not turn a clean 503 into a 500."""
    if not stats.is_active():
        return
    try:
        body = getattr(request.state, "gw_body", None)
        alias = getattr(request.state, "gw_alias", None)
        if alias is None and isinstance(body, dict):
            alias = body.get("model")
        asyncio.create_task(stats.record_call(
            duration_ms=0, backend=_REJECTED_BACKEND, source=_source_of(request),
            # `model` stays empty: it holds the REAL model a backend served, and no
            # backend ever resolved one here — filling it with the alias would render
            # as "x→x" in the call list and claim a resolution that never happened.
            alias=alias, model=None,
            endpoint=getattr(request.state, "gw_endpoint", None) or request.url.path,
            status=exc.status_code, input_tokens=0, output_tokens=0, cost_usd=0.0,
            request_text=(json.dumps(body, ensure_ascii=False) if isinstance(body, dict) else None),
            response_text=json.dumps({"error": {"message": str(exc.detail)}}, ensure_ascii=False),
        ))
    except Exception as e:                       # never let logging break the answer
        logger.warning(f"stats: could not record a rejected call: {e}")


# Shown in the backend column for calls no backend ever saw. A name rather than a
# blank so it reads as a statement ("nothing served this") and so the Statistic tab
# can show how many requests were turned away at the door. It lives in stats.py
# because the aggregates there must exclude it from the per-backend/per-model tables —
# one definition, or the marker silently starts counting as a backend again.
_REJECTED_BACKEND = stats.REFUSED_BACKEND

# ── Helpers ───────────────────────────────────────────────────────────────────

# ── Multi-user auth ──────────────────────────────────────────────────────────────
# Each request's Bearer token resolves to a user (own key). The global `api_key` acts
# as a master-admin key. **Bootstrap-open**: with no users AND no master key set, the
# gateway is fully open (today's behaviour). A user carries a role, an optional model
# allow-list (empty = all), and an optional per-day request quota.
users: list = []
_users_by_key: dict = {}
_MASTER_ADMIN = {"name": "admin", "role": "admin", "models": [], "_master": True}
_usage: dict = {}                # (user_name, utc_date) → request count (in-memory)


def rebuild_users() -> None:
    global users, _users_by_key
    users = store.list_users() if store.is_active() else []
    _users_by_key = {u["api_key"]: u for u in users if u.get("api_key") and u.get("enabled", True)}


def apply_users() -> None:
    rebuild_users()
    logger.info(f"users changed → {len(users)} user(s)")


def authenticate(authorization: Optional[str]) -> Optional[dict]:
    """Resolve the Bearer token to a user. Returns None only in bootstrap-open mode
    (no users + no master key). Raises 401 on a missing/invalid token otherwise."""
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
    if not users and not api_key:
        return None
    if not token:
        raise HTTPException(401, "Missing Authorization header")
    if api_key and token == api_key:
        return _MASTER_ADMIN
    u = _users_by_key.get(token)
    if u:
        return u
    raise HTTPException(401, "Invalid API key")


def check_auth(authorization: Optional[str]) -> None:
    authenticate(authorization)


def _model_allowed(user: dict, model: Optional[str]) -> bool:
    allow = user.get("models") or []
    if not allow or not model:
        return True                              # empty allow-list = all models
    if model in allow:                           # exact id / chat alias / image alias
        return True
    bname, bare = split_backend_prefix(model)    # backend/model
    if bname and bname in allow:                 # whole-backend grant (prefixed request)
        return True
    if bare in allow:                            # bare model id explicitly granted
        return True
    # Otherwise allow if a GRANTED backend either HOSTS this model (a bare real id) or
    # MAPS it as a virtual alias — so a whole-backend grant also covers the bare ids and
    # the aliases that route to it. Routing then decides availability (503 when the host
    # is offline) instead of a misleading 403. Disabled backends keep their last model set.
    is_alias = model in virtual_models
    for b in backends:
        if _is_gen(b) or b.get("name") not in allow:
            continue
        served = backend_models.get(backend_id(b), set())
        if bare in served:                                       # real model on a granted backend
            return True
        if is_alias and alias_entry(model, b["name"])[0] in served:   # alias → granted backend's model
            return True
    return False


async def gate_request(authorization: Optional[str], request: Request, model: Optional[str]) -> Optional[dict]:
    """Authenticate + enforce model allow-list and per-day quota; attribute the call
    to the user (stats source / job owner read it off request.state). Returns the user
    (None = anonymous bootstrap mode). Async only for the monthly-cost DB scan (run
    off-loop); the daily-quota check+increment stays await-free → atomic."""
    # Every caller passes the requested model here, and a rejection below (403/402/429)
    # happens before dispatch — so this is the earliest point where a refused call can
    # be logged with the model it asked for.
    if model and getattr(request.state, "gw_alias", None) is None:
        request.state.gw_alias = model
    user = authenticate(authorization)
    if user is None:
        return None
    request.state.gw_user = user["name"]
    if not _model_allowed(user, model):
        raise HTTPException(403, f"user '{user['name']}' is not allowed model '{model}'")
    cap = user.get("quota_cost_month")                 # monthly cost/credit quota (E1)
    if cap and stats.is_active():
        t = time.gmtime()
        month_start = calendar.timegm((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, 0))
        spent = await asyncio.to_thread(stats.month_cost, user["name"], month_start)
        if spent >= float(cap):
            raise HTTPException(402, f"monthly cost quota (${float(cap):.4f}) exceeded for "
                                     f"'{user['name']}' — spent ${spent:.4f}")
    limit = user.get("quota_req_day")
    day = time.strftime("%Y-%m-%d", time.gmtime())
    if limit and _usage.get((user["name"], day), 0) >= int(limit):
        raise HTTPException(429, f"daily request quota ({limit}) exceeded for '{user['name']}'")
    _usage[(user["name"], day)] = _usage.get((user["name"], day), 0) + 1
    return user


def _request_owner(request: Request) -> str:
    """Job/response owner attribution: the authenticated user, else — open mode
    (no users, no master key) — the caller's IP as 'ip:<addr>'. IP owners give
    keyless LAN services a best-effort separation (each sees only its own jobs);
    NAT/spoofing means it is NOT a security boundary — per-service users + keys
    are. No user and no client address → the legacy shared owner 'default'."""
    u = getattr(request.state, "gw_user", None)
    if u:
        return u
    ip = getattr(getattr(request, "client", None), "host", None)
    return f"ip:{ip}" if ip else "default"


def _check_owner(job: Optional[dict], user: Optional[dict], *, status: int, detail: str,
                 anon_owner: Optional[str] = None) -> None:
    """Non-admin users — and, in open mode, anonymous callers — may only touch
    their own jobs/responses; admin/master pass. Anonymous identity is the caller
    IP (`anon_owner`, see _request_owner). `status` picks the flavor: jobs answer
    403, background responses hide foreign ids as 404 (no existence leak). Owner
    None/'default' (legacy/shared) stays open to everyone."""
    if user and (user.get("_master") or user.get("role") == "admin"):
        return
    owner = (job or {}).get("owner")
    if owner in (None, "default"):
        return
    caller = user.get("name") if user else anon_owner
    if job and owner != caller:
        raise HTTPException(status, detail)


def resolve_admin(key: Optional[str]) -> Optional[dict]:
    """Resolve a bare key to an admin (master api_key or an admin-role user). Drives
    the /ui login. Returns None if the key isn't an admin credential."""
    if not key:
        return None
    if api_key and key == api_key:
        return _MASTER_ADMIN
    u = _users_by_key.get(key)
    return u if (u and u.get("role") == "admin") else None


def ui_locked() -> bool:
    """True once an admin credential exists (master key or an admin user) → /ui needs
    login. Bootstrap-open (no admin anywhere) returns False so you can't lock yourself
    out before setting one up."""
    return bool(api_key) or any(u.get("role") == "admin" for u in users)


def alias_entry(alias: str, backend_name: str) -> tuple[Optional[str], Optional[int]]:
    """(real_model, priority_override) for this alias on this backend.

    real_model is None when the alias isn't mapped to this backend.
    priority_override is parsed from the stored `{model, priority}` shape (old entries
    still carry it) but is NOT read anywhere — routing is unpaid-then-fastest since
    spec 2026-09-01, and the UI no longer offers the field.

    - alias not in virtual_models → (alias, None)         pass-through
    - alias maps to string         → (string, None)        same model everywhere
    - alias maps to dict, value …
        … string                   → (string, None)        per-backend model
        … object {model, priority} → (model, priority)     model + stored (unused) prio
    """
    mapping = virtual_models.get(alias)
    if mapping is None:
        return alias, None
    if isinstance(mapping, str):
        return mapping, None
    if isinstance(mapping, dict):
        entry = mapping.get(backend_name)
        if isinstance(entry, dict):
            return entry.get("model"), entry.get("priority")
        return entry, None        # string value, or None if backend absent
    return None, None


def resolve_for_backend(alias: str, backend_name: str) -> Optional[str]:
    """Real model name for this alias on this backend, or None if not mapped here."""
    return alias_entry(alias, backend_name)[0]


def split_backend_prefix(model: str) -> tuple[Optional[str], str]:
    """Split a '<backend-name>/<real-model>' id into (backend_name, real_model).

    Returns (None, model) when the first path segment isn't a known backend
    name — so bare aliases ('fast') and vendor-prefixed ids ('moonshotai/Kimi…')
    are left untouched. Backend names (together, openrouter, dx-10-1, …) never
    collide with vendor prefixes, so the first '/' disambiguates cleanly.
    """
    if "/" in model:
        prefix, rest = model.split("/", 1)
        if prefix in _backend_names:
            return prefix, rest
    return None, model


# ── Route index (precomputed candidates; health/busy/drain stay live checks) ────
# The static routing inputs — which enabled LLM backend maps an alias to which real
# model, and which bare model ids pass through — change only on config/store edits or
# a discovery model-set change. They are precomputed here instead of rescanned on
# every request; resolve_routes() then only evaluates the live flags (healthy / busy /
# draining) and applies the scheduler's ordering.
# Rebuilt by rebuild_backends() / rebuild_virtual_models() and by refresh_backend()
# whenever a backend's model set changes. Swapped atomically (built local, then
# assigned) — safe for the off-loop readers (get_gen_routes runs in a thread).
_backend_names: set = set()            # all backend names — split_backend_prefix test
_llm_backends: list = []               # enabled non-ComfyUI backends
_gen_backends: list = []               # enabled generation backends (ComfyUI, Meshy, Tripo)
_route_index: dict = {}                # alias/model-id → [(backend, real_model)] candidates


def rebuild_route_index() -> None:
    """Recompute the per-key candidate lists. The index no longer sorts: dispatch
    order is decided per request by the scheduler (unpaid before paid, then fastest
    first), so candidates keep their insertion order as the stable tiebreak."""
    global _backend_names, _llm_backends, _gen_backends, _route_index
    _backend_names = {b["name"] for b in backends}
    _llm_backends = [b for b in enabled_backends() if not _is_gen(b)]
    _gen_backends = [b for b in enabled_backends() if _is_gen(b)]
    index: dict[str, list] = {}
    for alias in virtual_models:                       # aliases (they shadow same-named real ids)
        for b in _llm_backends:
            real, _prio = alias_entry(alias, b["name"])
            if real is not None and real in backend_models.get(backend_id(b), set()):
                index.setdefault(alias, []).append((b, real))
    for b in _llm_backends:                            # bare model ids → pass-through routing
        for mid in backend_models.get(backend_id(b), set()):
            if mid not in virtual_models:
                index.setdefault(mid, []).append((b, mid))
    _route_index = index


def serves_path(backend: dict, path: str) -> bool:
    """May this backend serve a request on `path`?

    Anthropic backends answer `/v1/messages` ONLY. That is a licence boundary, not
    a technical one: the credential is normally a personal Claude subscription,
    which covers using Claude Code — not re-serving Claude as a general-purpose
    API through the gateway's OpenAI endpoints. Enforced here, in routing, so the
    backend cannot be reached by any other endpoint, alias or playground; the
    README and the Backends tab state the same rule in words.

    It is also what keeps a mixed alias honest: an Anthropic backend would receive
    a chat-completions body it cannot parse, so it must not be a candidate there.

    The restriction runs one way only — a chat backend still serves `/v1/messages`,
    translated by the bridge. That is what lets one alias fail over from Anthropic
    to an open-weight model.
    """
    if backend.get("type") == "anthropic":
        return path.startswith("/v1/messages")
    return True


def resolve_routes(alias: str, path: str = "/v1/chat/completions") -> tuple[list, list]:
    """(ready, busy) (backend, real_model) candidate lists.

    `ready` is routable now and comes back in DISPATCH order: unpaid backends before
    paid ones, fastest (measured tok/s) first inside each tier, unmeasured backends
    first so each gets probed once — then the shared-GPU demotion on top. `busy` maps
    + serves the alias but sits at its in-flight cap (→ parkable). A
    '<backend>/<model>' alias resolves to a single backend in whichever bucket. Drives
    both normal routing (ready) and call parking (busy). Candidates come pre-resolved
    from `_route_index`; only healthy/busy/draining — and the endpoint a backend is
    allowed to serve (`serves_path`) — are evaluated per request.
    """
    bname, bare = split_backend_prefix(alias)
    if bname is not None:
        # chat routing only considers LLM backends, so a name shared with a ComfyUI
        # backend is unambiguous here.
        b = next((b for b in _llm_backends if b["name"] == bname), None)
        if b is None or not backend_healthy.get(backend_id(b)) or is_draining(b):
            return [], []
        if not serves_path(b, path):
            return [], []
        real = resolve_for_backend(bare, bname)
        if real is None or real not in backend_models.get(backend_id(b), set()):
            return [], []
        return ([], [(b, real)]) if backend_busy(b) else ([(b, real)], [])

    ready, busy = [], []
    for b, real in _route_index.get(alias, ()):
        if not backend_healthy.get(backend_id(b)) or is_draining(b):
            continue
        if not serves_path(b, path):
            continue
        (busy if backend_busy(b) else ready).append((b, real))
    if len(ready) > 1:
        # Unified scheduling (spec 2026-09-01): unpaid before paid, then fastest
        # first by measured tok/s; unmeasured backends sort first so each gets
        # probed once. Priority and the per-key speed switch are gone.
        ready = scheduler.order_ready(
            ready,
            lambda b, real: backend_tps.get(backend_id(b), float("inf")),
            lambda b: bool(b.get("paid")))
        # Shared-GPU consideration: candidates whose host is generating media go
        # LAST (stable — the scheduler order is kept within both groups), never
        # dropped. Runs after the sort so the shared-GPU guarantee stays dominant.
        mb = _media_busy_hosts()
        if mb:
            ready.sort(key=lambda br: backend_hosts.get(backend_id(br[0]), "") in mb)
    return ready, busy


def get_routes_for(alias: str) -> list[tuple[dict, str]]:
    """(backend, real_model) pairs to try, in dispatch order (see resolve_routes) —
    ready (non-busy) backends only. Thin wrapper over resolve_routes()."""
    return resolve_routes(alias)[0]


def alias_model_conflicts() -> list[dict]:
    """Aliases whose name also exists as a real model id on some backend.

    Setting an alias named like a real model *shadows* that model: a bare
    request for the name routes only via the alias mapping (the pass-through
    that would otherwise reach the real model is disabled), and even the
    '<backend>/<name>' form fails on backends the alias doesn't map (because
    resolve_for_backend returns None there). So any backend that actually hosts
    a model of that exact id but is absent from the alias mapping becomes
    unreachable by that name.

    Returns one entry per colliding alias with the hosting backends split into
    `covered` (in the mapping → still routable) and `shadowed` (hosting the real
    model but not mapped → unreachable by that name). `shadowed` non-empty is the
    actionable conflict; empty means the alias intentionally shadows a model it
    fully covers (e.g. one id mapped across exactly the backends that serve it).
    """
    out = []
    for name in virtual_models:
        hosting = [b["name"] for b in enabled_backends()
                   if not _is_gen(b)
                   and name in backend_models.get(backend_id(b), set())]
        if not hosting:
            continue
        covered = [bn for bn in hosting if alias_entry(name, bn)[0] is not None]
        shadowed = [bn for bn in hosting if alias_entry(name, bn)[0] is None]
        out.append({
            "name": name,
            "hosting_backends": hosting,
            "covered": covered,
            "shadowed": shadowed,
        })
    return out


def routing_snapshot() -> dict:
    """Diagnostic view of how every alias and discovered model resolves.

    Unlike get_routes_for(), this keeps unhealthy backends and not-yet-discovered
    models in the result (flagged), so the dashboard shows the full configured
    picture rather than only what's routable right now.
    """
    enabled = enabled_backends()

    aliases = []
    for name in virtual_models:
        rows = []
        for b in backends:                     # all (incl. disabled) LLM backends, so a
            if _is_gen(b):                     # mapping doesn't vanish when off
                continue                       # chat aliases route only to LLM backends
            real, _prio = alias_entry(name, b["name"])
            if real is None:
                continue                       # alias not mapped to this backend
            bid, enbl = backend_id(b), is_enabled(b)
            healthy = enbl and backend_healthy.get(bid, False)
            present = real in backend_models.get(bid, set())
            busy = enbl and healthy and backend_busy(b)
            rows.append({
                "backend": b["name"],
                "model": real,
                "enabled": enbl,
                "healthy": healthy,
                "error": backend_error.get(bid) if enbl and not healthy else None,
                "present": present,
                "busy": busy,
                "routable": enbl and healthy and present,
            })
        aliases.append({"alias": name, "routes": rows})
    aliases.sort(key=lambda a: a["alias"].lower())

    model_hosts: dict[str, list] = {}
    for b in enabled:
        healthy = backend_healthy.get(backend_id(b), False)
        for mid in backend_models.get(backend_id(b), set()):
            model_hosts.setdefault(mid, []).append({
                "backend": b["name"],
                "type": b.get("type", "openai"),
                "healthy": healthy,
                "busy": healthy and backend_busy(b),
                "tps": round(backend_tps.get(backend_id(b), 0.0), 1),
                "paid": bool(b.get("paid")),
                "ctx": model_context(b, mid),   # admin rule or learned; None = unknown
            })
    models = []
    for mid, hosts in sorted(model_hosts.items(), key=lambda kv: kv[0].lower()):
        models.append({
            "model": mid,
            "hosts": hosts,
            "shadowed_by_alias": mid in virtual_models,
        })

    return {"aliases": aliases, "models": models, "conflicts": alias_model_conflicts()}


def _source_of(request: Request) -> str:
    u = getattr(request.state, "gw_user", None)        # authenticated user wins
    if u:
        return u
    return request.headers.get("x-source") or (request.client.host if request.client else "unknown")


def _normalize_reasoning(body: dict) -> Optional[str]:
    """Client reasoning control → 'off' | 'on' | None(auto). Pops the gateway control
    field `reasoning` (string, or a Responses-style object {effort}); falls back to the
    OpenAI `reasoning_effort` alias (minimal→off, else→on). `reasoning_effort` itself is
    left in the body (it's a real field a native-effort backend may use)."""
    v = body.pop("reasoning", None)
    if isinstance(v, str):
        v = v.strip().lower()
        if v in ("off", "on"):
            return v
        return None                                  # "auto"/unknown → default
    if isinstance(v, dict):                          # Responses API shape: {"effort": ...}
        eff = v.get("effort")
        if isinstance(eff, str):
            return "off" if eff.strip().lower() == "minimal" else "on"
        return None
    eff = body.get("reasoning_effort")
    if isinstance(eff, str):
        return "off" if eff.strip().lower() == "minimal" else "on"
    return None


def _reasoning_apply(backend: dict, model: Optional[str], requested: Optional[str], payload: dict):
    """Adapter-context hook: resolve the reasoning rule for (backend, model) and apply
    it. Rules live in the store (UI-editable, hot); empty → everything 'unsupported'."""
    if requested not in ("off", "on"):
        return payload, None
    rule = reasoning.resolve(reasoning_rules, backend.get("name", ""), model or "")
    return reasoning.apply(rule, requested, payload)


async def probe_reasoning(backend_name: str, model: str, adapter: str, param: dict,
                          requested: str, prompt: str) -> dict:
    """Live-test a reasoning adapter against ONE (backend, model): fire a baseline call
    and a call with `adapter` applied, and report whether the model's reasoning channel
    was actually suppressed. Goes DIRECT to the backend (bypasses routing + the stored
    rules) so a candidate mechanism can be validated BEFORE a rule exists. Never raises —
    the /ui Reasoning-tab live test drives this. Returns a plain dict for the UI."""
    requested = requested if requested in ("off", "on") else "off"
    b = next((x for x in backends if x.get("name") == backend_name
              and not _is_gen(x)), None)
    if b is None:
        return {"error": f"backend '{backend_name}' not found or not an LLM backend"}
    if b.get("type") == "anthropic":
        # The one path that could otherwise reach an Anthropic backend off
        # /v1/messages: this probe talks to the backend DIRECTLY, and
        # api.anthropic.com does answer /v1/chat/completions. Same licence rule as
        # serves_path — a subscription is not a chat-completions API.
        return {"error": f"backend '{backend_name}' is an Anthropic backend — "
                         "reachable through /v1/messages only, so there is nothing to probe here"}
    url = f"{b['url']}/v1/chat/completions"
    headers = {"content-type": "application/json", **backend_auth_headers(b)}
    base_body = {"model": model, "stream": False, "temperature": 0.1, "max_tokens": 300,
                 "messages": [{"role": "user",
                               "content": (prompt or "").strip() or "Say hello in one short sentence."}]}
    rule = {"adapter": adapter, "param": param or {}}
    cand_body, control = reasoning.apply(rule, requested, dict(base_body))

    async def _one(body: dict) -> dict:
        try:
            r = await http_client.post(url, headers=headers, json=body, timeout=120.0)
            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text[:600]}
            msg = ((data.get("choices") or [{}])[0].get("message") or {}) if isinstance(data, dict) else {}
            return {"status": r.status_code,
                    "content": (msg.get("content") or ""),
                    "reasoning": (msg.get("reasoning") or msg.get("reasoning_content") or ""),
                    "err": None if r.status_code == 200 else json.dumps(data, ensure_ascii=False)[:400]}
        except Exception as e:
            return {"status": 0, "content": "", "reasoning": "", "err": f"{type(e).__name__}: {e}"}

    # Sequential, not concurrent: single-slot backends (LocalAI / llama.cpp --parallel 1)
    # give unreliable results when baseline + candidate hit them at once.
    base = await _one(base_body)
    cand = await _one(cand_body)
    return {
        "backend": backend_name, "model": model, "adapter": adapter,
        "control": control, "requested": requested,
        "baseline": {"status": base["status"], "reasoning_len": len(base["reasoning"]),
                     "content_len": len(base["content"]), "err": base["err"]},
        "candidate": {"status": cand["status"], "reasoning_len": len(cand["reasoning"]),
                      "content_len": len(cand["content"]),
                      "content_preview": cand["content"][:240], "err": cand["err"]},
    }


# ── Voice reference library (TTS voice cloning) ─────────────────────────────────
# WAV blobs live on the GATEWAY (voiceref/ — gitignored, deploy-excluded) and are
# additionally SHIPPED via scp to the TTS backend host: qwen3-tts-style models read
# `voice` strictly as a local file (measured: no base64/data-URI, no URL, no files
# API). API/UI reference entries as voice:"lib:<name>"; route() resolves that to the
# shipped path + fills ref_text.

VOICE_REF_DIR = Path("voiceref")


def _voice_safe(name: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (name or "voice"))
    return keep[:48] or "voice"


def apply_voice_library() -> None:
    global voice_library
    voice_library = store.get_voice_library() if store.is_active() else {}


def voice_ship_config() -> tuple[list[str], str]:
    """(targets, voice_dir) for shipping references.

    `voice_ref_hosts` = comma-separated scp TARGETS 'user@host:/abs/host/dir' — one
    per host serving a cloning model; the host-side dir may differ per host (e.g. a
    docker bind-mount source). `voice_ref_dir` = the dir AS THE MODEL SEES IT (e.g.
    the container path '/models/voices') — this single path goes into `voice`, so
    it must be identical on every host (failover may pick any of them)."""
    s = store.get_settings() if store.is_active() else {}
    targets = [h.strip() for h in str(s.get("voice_ref_hosts") or "").split(",") if h.strip()]
    return targets, str(s.get("voice_ref_dir") or "").rstrip("/")


_whisper_model = None                       # lazy faster-whisper instance (CPU, loaded on first use)


def _local_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel        # heavy import — only on first transcription
        s = store.get_settings() if store.is_active() else {}
        size = str(s.get("whisper_model") or "small")
        _whisper_model = WhisperModel(size, device="cpu", compute_type="int8")
    return _whisper_model


def _whisper_route() -> Optional[tuple[dict, str]]:
    """First healthy LLM backend serving a whisper* model (fallback transcription)."""
    for b in enabled_backends():
        if _is_gen(b):
            continue
        for m in sorted(backend_models.get(backend_id(b), set())):
            if "whisper" in m.lower() and backend_healthy.get(backend_id(b)):
                return b, m
    return None


async def transcribe_audio(data: bytes, filename: str = "ref.wav") -> tuple[Optional[str], str]:
    """(text, source-or-error). Local faster-whisper on the gateway CPU first (small,
    lazy-loaded); a backend's /v1/audio/transcriptions as fallback when the local
    model is unavailable."""
    import io
    try:
        def _run():
            segs, _info = _local_whisper().transcribe(io.BytesIO(data))
            return " ".join(s.text.strip() for s in segs).strip()
        txt = await asyncio.to_thread(_run)
        if txt:
            return txt, "gateway whisper"
    except ImportError:
        pass                                            # not installed → try a backend
    except Exception as ex:
        logger.warning(f"local whisper failed: {type(ex).__name__}: {ex}")
    hit = _whisper_route()
    if hit is None:
        return None, "no local faster-whisper and no whisper model on any backend"
    b, model = hit
    try:
        r = await http_client.post(f"{b['url']}/v1/audio/transcriptions",
                                   headers=backend_auth_headers(b), data={"model": model},
                                   files={"file": (filename, data, "audio/wav")}, timeout=180.0)
        if r.status_code == 200:
            txt = str((r.json() or {}).get("text") or "").strip()
            return (txt or None), f"{b['name']}/{model}"
        return None, f"{b['name']}/{model} HTTP {r.status_code}: {r.text[:150]}"
    except Exception as ex:
        return None, f"{type(ex).__name__}: {ex}"


async def _scp(src: Path, host: str, remote: str) -> tuple[bool, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            str(src), f"{host}:{remote}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        return proc.returncode == 0, (out or b"").decode(errors="replace")[-200:]
    except Exception as ex:
        return False, f"{type(ex).__name__}: {ex}"


async def ship_voice_ref(name: str, notify=None) -> tuple[bool, str]:
    """scp the library WAV to EVERY configured target (host-side dirs may differ —
    docker bind mounts); the entry's `remote` is the MODEL-VISIBLE path
    (voice_ref_dir/<file>), identical across hosts. (all_ok, message) — never
    raises (UI renders the message). `notify(kind, text)` (kind ok|err|run)
    streams per-step progress to the UI poller."""
    nb = notify or (lambda k, t: None)
    e = (store.get_voice_library() if store.is_active() else {}).get(name)
    if not e:
        nb("err", f"unknown voice '{name}'")
        return False, f"unknown voice '{name}'"
    targets, vdir = voice_ship_config()
    bad = [t for t in targets if ":" not in t or not t.split(":", 1)[1].startswith("/")]
    if not targets or bad or not vdir.startswith("/"):
        msg = ("no/invalid ship config — targets are 'user@host:/abs/host/dir' (comma-"
               "separated) + a model-visible voice dir (e.g. /models/voices)")
        nb("err", msg)
        return False, msg
    src = Path(e.get("file") or "")
    if not src.is_file():
        nb("err", f"gateway blob missing: {src}")
        return False, f"gateway blob missing: {src}"
    remote = f"{vdir}/{src.name}"                       # what goes into `voice`
    results = {}
    for t in targets:
        host, hdir = t.split(":", 1)
        hdir = hdir.rstrip("/")
        nb("run", f"upload → {host}")
        try:                                            # scp can't create dirs — mkdir -p first
            proc = await asyncio.create_subprocess_exec(
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, f"mkdir -p {hdir}",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await proc.communicate()
        except Exception:
            pass                                        # scp below reports the real error
        ok, msg = await _scp(src, host, f"{hdir}/{src.name}")
        results[t] = "ok" if ok else (msg or "scp failed")
        nb("ok" if ok else "err", f"{host}: {'ok' if ok else (msg or 'scp failed')[:120]}")
    all_ok = all(v == "ok" for v in results.values())
    e.update({"remote": remote if all_ok else e.get("remote", ""),
              "shipped": all_ok, "hosts": results})
    store.set_voice_entry(name, e)
    apply_voice_library()
    summary = " · ".join(f"{t.split('@')[-1].split(':')[0]}: {v if v == 'ok' else v[:80]}"
                         for t, v in results.items())
    return all_ok, (f"voice={remote} · {summary}" if all_ok else summary)


async def save_voice_ref(name: str, data: bytes, ref_text: str = "", notify=None) -> dict:
    """Create/replace a library entry: write the gateway blob, auto-transcribe when
    ref_text is empty (whisper backend, if any), then try to ship. Returns UI status;
    `notify(kind, text)` streams per-step progress (store → whisper → per-host scp)."""
    nb = notify or (lambda k, t: None)
    VOICE_REF_DIR.mkdir(exist_ok=True)
    path = VOICE_REF_DIR / (_voice_safe(name) + ".wav")
    path.write_bytes(data)
    nb("ok", f"stored on the gateway ({len(data) // 1024} KB)")
    note = ""
    if not (ref_text or "").strip():
        nb("run", "whisper — transcribing the reference")
        txt, src = await transcribe_audio(data, path.name)
        ref_text = txt or ""
        note = f"ref_text transcribed via {src}" if txt else f"ref_text empty — {src}"
        nb("ok" if txt else "err", note + (f": “{txt[:80]}…”" if txt and len(txt) > 80
                                           else (f": “{txt}”" if txt else "")))
    store.set_voice_entry(name, {"ref_text": (ref_text or "").strip(), "file": str(path),
                                 "remote": "", "shipped": False})
    apply_voice_library()                         # cache now — ship's early returns don't refresh
    ok, msg = await ship_voice_ref(name, notify=notify)
    return {"shipped": ok, "ship_msg": msg, "note": note,
            "entry": voice_library.get(name) or {}}


def delete_voice_ref(name: str) -> None:
    e = (store.get_voice_library() if store.is_active() else {}).get(name) or {}
    try:
        p = Path(e.get("file") or "")
        if p.is_file():
            p.unlink()
    except OSError:
        pass
    store.set_voice_entry(name, None)
    apply_voice_library()


def _cost_usd(bid: str, model_id: Optional[str], in_tok: int, out_tok: int) -> float:
    """USD cost for a call from cached pricing (Together-style /v1/models), keyed by
    the backend id (`type:name`, same as `backend_pricing`). 0 if unknown."""
    if not model_id:
        return 0.0
    p = backend_pricing.get(bid, {}).get(model_id, {})
    return ((in_tok or 0) * p.get("input", 0.0) + (out_tok or 0) * p.get("output", 0.0)) / 1_000_000


# ── Adapter wiring ─────────────────────────────────────────────────────────────
# Services handed to every backend adapter so it stays import-cycle-free and
# hot-reload-safe (the log flag is read per-call via a callable, never cached).

adapter_ctx = AdapterContext(
    auth_headers=backend_auth_headers,
    inflight_inc=_inflight_inc,
    inflight_dec=_inflight_dec,
    cost_usd=_cost_usd,
    source_of=_source_of,
    record_call=stats.record_call,
    note_speed=_note_speed,
    note_progress=_note_progress,
    log_enabled=lambda: log_per_call,
    active_register=_active_register,
    active_done=_active_done,
    apply_reasoning=_reasoning_apply,
    http_client=lambda: http_client,   # shared pool; callable so adapters never cache it
    loras_of=lambda bid: backend_loras.get(bid, set()),
)


def build_backend_adapters() -> None:
    """(Re)instantiate one adapter per configured backend. Called at import and
    after every config reload so adapters point at the current backend dicts."""
    backend_adapters.clear()
    for b in backends:
        backend_adapters[backend_id(b)] = make_adapter(b, adapter_ctx)


build_backend_adapters()
rebuild_route_index()                      # initial index (config backends; models fill in via discovery)


async def _dispatch_or_park(alias, path, body, request, stats_endpoint=None, deadline=None):
    """Forward to a ready backend; if all mapping backends are busy, hold the call in
    the park queue until one frees (then dispatch) or 503. Park window = the alias's
    park time, or an explicit `deadline` (background responses use the longer async
    window). Shared by chat routing, the Responses bridge, and background responses."""
    # Stash what a refusal needs to be logged with (see _record_rejected): the alias
    # and body only exist here, but the rejection is rendered by the error handler.
    request.state.gw_alias = alias
    request.state.gw_body = body
    request.state.gw_endpoint = stats_endpoint or path
    ready, busy = resolve_routes(alias, path)
    # Spec rule 4: always into the queue. A free backend that a parked call is designated
    # for is NOT up for grabs — this request parks instead and competes from inside the
    # pool, so a fresh arrival can never overtake the waiters it just queued behind.
    reserved = bool(ready) and _reserved_for_waiter(ready[0][0])
    if ready and not reserved:
        return await _dispatch_over(ready, path, alias, body, request, stats_endpoint=stats_endpoint)
    if not ready and not busy:
        # Say WHY when the only backends that could serve this alias are Anthropic
        # ones: they answer /v1/messages alone (see serves_path), and "no healthy
        # backend" would send the caller hunting a fault that isn't there. Only off
        # the messages path — ON it the very same candidate set means the backend is
        # simply down, and a 404 there would tell the caller to use the endpoint they
        # are already on (and Claude Code does not retry a 404, a 503 it does).
        cands = _route_index.get(alias) or []
        if (not path.startswith("/v1/messages")
                and cands and all(b.get("type") == "anthropic" for b, _ in cands)):
            raise HTTPException(404, f"model '{alias}' is served by an Anthropic backend — "
                                     "reachable through POST /v1/messages only")
        raise HTTPException(503, f"No healthy backend for model '{alias}'")
    if deadline is None:
        ptime = _park_time_for(alias)
        if ptime <= 0:                                 # parking disabled for this alias → 503 now
            raise HTTPException(503, f"all backends for '{alias}' are busy (parking disabled)",
                                headers={"Retry-After": "1"})
        deadline = time.monotonic() + ptime
    if len(_parked) >= max_parked:
        raise HTTPException(503, f"park queue full ({max_parked}) — retry later", headers={"Retry-After": "2"})
    if reserved:
        # We stepped aside for a backend that is free RIGHT NOW: wake the pool so its
        # designated taker claims it instead of leaving it idle until the next slot
        # frees. One broadcast per gated arrival — each waiter just re-checks itself.
        _notify_slot_free()
    return await _park_and_dispatch(alias, path, body, request, deadline,
                                    source=_source_of(request), stats_endpoint=stats_endpoint)


def _retryable_upstream_error(resp) -> bool:
    """A 502 whose body is llama-swap's "unable to start process" — the backend
    is alive but can't load the model right now (VRAM held by another process on
    the shared GPU, measured on k12-gpu). Backend-local by definition, so another
    candidate may well serve the call — treated like connect/timeout in
    _dispatch_over. Plain Response only (streams surface errors that way since
    the adapter opens upstream before answering)."""
    if getattr(resp, "status_code", 0) != 502:
        return False
    return b"unable to start process" in (getattr(resp, "body", b"") or b"")[:300]


async def _dispatch_over(candidates, path, alias, body, request, stats_endpoint=None):
    """Forward to the first candidate, failing over to the next only on
    connect/timeout or a backend-local load failure (_retryable_upstream_error);
    other HTTP errors return as-is. The first candidate's dispatch increments
    in-flight before its first await, so a parked waiter claims that slot
    atomically here. `stats_endpoint` overrides the recorded endpoint label
    (e.g. /v1/responses)."""
    last_error: Exception = Exception("unknown")
    last_resp = None
    for backend, real_model in candidates:
        cand_body = dict(body, model=real_model)  # per-candidate copy — the shared body stays untouched
        adapter = backend_adapters.get(backend_id(backend))
        if adapter is None:                       # config raced a reload — skip
            continue
        try:
            if log_per_call:
                logger.info(f"→ [{backend['name']}] {alias} → {real_model}")
            req = NormalizedRequest(
                path=path, alias=alias, real_model=real_model,
                body=cand_body, raw=request, stream=bool(body.get("stream")),
                stats_endpoint=stats_endpoint, reasoning=body.get("_reasoning"),
            )
            # Type affinity (spec 2026-09-01): remember what this backend last ran, so
            # a freed backend prefers a waiter needing the same model (no reload).
            backend_last_key[backend_id(backend)] = real_model
            resp = await adapter.dispatch(req)
            # A backend answered — the adapter has written (or will write) the stats
            # row, so the error handler must not add a second one for this request.
            request.state.gw_dispatched = True
            if _retryable_upstream_error(resp):
                logger.warning(f"✗ [{backend['name']}] upstream can't start the model (502) — trying next")
                last_resp = resp
                continue
            return resp
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning(f"✗ [{backend['name']}] {e} — trying next")
            last_error = e
    if last_resp is not None:                     # every candidate failed to load → the real 502
        return last_resp
    raise HTTPException(503, f"All backends failed: {last_error}")


def _waiting_pool() -> list:
    """Snapshot of the park queue as the scheduler sees it (`_parked` mutates across
    awaits). Entries that already claimed a backend are out: a claimant stays in
    `_parked` until its response is done, but it is no longer waiting and must not
    shadow the others out of a second free backend."""
    return [e for e in _parked if not e.get("claimed")]


def _designated_waiter(backend, pool: list, now: Optional[float] = None):
    """The parked call this backend belongs to, or None if it belongs to none of them.

    Type affinity (spec 2026-09-01, "Designated taker"): overdue waiters first (oldest),
    else the one needing the model the backend last ran, else the oldest it can serve.
    THE one place the designation is computed — the wake gate and the fresh-arrival gate
    both go through here so they cannot drift apart.
    """
    bid, bname = backend_id(backend), backend["name"]
    return scheduler.designated_taker(
        pool,
        can_serve=lambda e: any(backend_id(rb) == bid for rb, _ in
                                resolve_routes(e["alias"], e["path"])[0]),
        type_key=lambda e: alias_entry(e["alias"], bname)[0] or e["alias"],
        last_key=backend_last_key.get(bid),
        now=time.monotonic() if now is None else now,
        max_wait_s=affinity_max_wait_s)


def _reserved_for_waiter(backend) -> bool:
    """Is this free backend spoken for by someone already in the queue?

    Spec rule 4: a fresh request never overtakes the waiters — it may dispatch straight
    to a backend only when no parked call is designated for it; otherwise it joins the
    queue as the youngest entry and competes from there. Empty queue → never reserved.
    """
    pool = _waiting_pool()
    return bool(pool) and _designated_waiter(backend, pool) is not None


def _designated_index(entry, ready) -> Optional[int]:
    """Index in `ready` of the best candidate this parked call may claim now, or None.

    A woken entry dispatches to the best ready candidate it is designated for, and parks
    again when every ready backend is somebody else's (the rightful waiter claims it on
    the same broadcast wake). A lone waiter always passes.

    Every ready backend has exactly one designated taker among the WAITING pool, and that
    taker has the backend in its own ready list — so on every wake at least one waiter
    proceeds while a backend is free. A designation is never left behind by an entry that
    stopped waiting: claiming a backend and leaving the queue (dispatch, timeout, cancel)
    both drop the entry out of the pool and re-broadcast, so the rest re-evaluate at once
    (see `_park_and_dispatch`).
    """
    if not ready:
        return None
    pool = _waiting_pool()
    if len(pool) <= 1:                       # only us waiting → nothing to yield to
        return 0
    now = time.monotonic()
    for i, (b, _real) in enumerate(ready):
        if _designated_waiter(b, pool, now) is entry:
            return i
    return None


async def _park_and_dispatch(alias, path, body, request, deadline, source="?", stats_endpoint=None):
    """Hold a request in the park queue until a mapping backend frees (then dispatch),
    or until `deadline` (→ 503). The entry stays in `_parked` for the whole wait so it
    keeps its FIFO position and shows in the console; `_notify_slot_free` wakes it."""
    entry = {"id": _next_park_id(), "alias": alias, "path": path, "source": source,
             "enqueued": time.time(), "enqueued_at": time.monotonic(),
             "deadline": deadline, "event": asyncio.Event()}
    _parked.append(entry)
    try:
        while True:
            entry["event"].clear()                     # arm before checking → no lost wakeup
            ready, busy = resolve_routes(alias, path)
            i = _designated_index(entry, ready)
            if i is not None:
                entry["claimed"] = True    # out of the waiting pool (see _designated_index)
                # We hold a designation no more: let the rest re-evaluate the backends we
                # are NOT taking. They only resume once we await below, by which time our
                # own candidate is in-flight (no double claim) — and re-checking is free,
                # the loop arms its event before looking.
                _notify_slot_free()
                # Try the designated candidate first, the rest stay as failover tail.
                cands = [ready[i]] + ready[:i] + ready[i + 1:]
                return await _dispatch_over(cands, path, alias, body, request, stats_endpoint=stats_endpoint)
            if not ready and not busy:
                raise HTTPException(503, f"No healthy backend for model '{alias}'")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HTTPException(503, f"all backends for '{alias}' are busy — parked with no free "
                                    "backend in time", headers={"Retry-After": "2"})
            try:
                await asyncio.wait_for(entry["event"].wait(), remaining)
            except asyncio.TimeoutError:
                raise HTTPException(503, f"all backends for '{alias}' are busy — parked with no free "
                                    "backend in time", headers={"Retry-After": "2"})
    finally:
        try:
            _parked.remove(entry)
        except ValueError:
            pass
        if _parked:
            # Leaving the queue — dispatched, timed out or cancelled — releases whatever
            # this entry was designated for, so the remaining waiters must look again.
            _notify_slot_free()


def _apply_alias_sampling(alias: str, body: dict) -> None:
    """Fill this alias's sampling defaults into a chat body — only keys the CLIENT
    did not send (an explicit client value always wins). The serving backend's own
    `sampling_defaults` apply later, in the adapter, so the effective precedence is
    client > alias > backend. A malformed store entry is ignored rather than failing
    the request."""
    d = alias_sampling.get(alias)
    if not isinstance(d, dict):
        if d is not None:
            logger.warning(f"alias '{alias}': sampling defaults are not a dict — ignored")
        return
    for k, v in d.items():
        if k not in body:
            body[k] = v


async def route(path: str, request: Request, authorization: Optional[str]) -> JSONResponse | StreamingResponse:
    body = await request.json()
    alias = body.get("model", "")
    await gate_request(authorization, request, alias)        # auth + model allow-list + quota
    body.pop("park", None)                              # legacy control field — parking is automatic; never forward
    if path.startswith("/v1/audio/"):
        body.pop("stream", None)                        # audio is a binary passthrough — never SSE
        av = alias_voice.get(alias)
        if av:                                          # per-alias TTS defaults; explicit client fields win
            if av.get("voice") and not body.get("voice"):
                body["voice"] = av["voice"]
            if av.get("ref_text") and not (body.get("params") or {}).get("ref_text"):
                body.setdefault("params", {})["ref_text"] = av["ref_text"]
        v = body.get("voice")                           # voice:"lib:<name>" → shipped path + ref_text
        if isinstance(v, str) and v.startswith("lib:"):
            e = voice_library.get(v[4:])
            if not e:
                raise HTTPException(400, f"unknown voice library entry '{v[4:]}'")
            if not (e.get("shipped") and e.get("remote")):
                raise HTTPException(409, f"voice '{v[4:]}' is not on the backend host yet — "
                                         "configure the scp target / retry ship in the Voice tab")
            body["voice"] = e["remote"]
            if e.get("ref_text") and not (body.get("params") or {}).get("ref_text"):
                body.setdefault("params", {})["ref_text"] = e["ref_text"]
    if not (path.startswith("/v1/audio/") or path.startswith("/v1/embeddings")):
        _apply_alias_sampling(alias, body)              # per-alias sampling; client fields win
    r = _normalize_reasoning(body)                      # off|on|None; strips `reasoning`, stashes for dispatch
    if r is None:
        r = alias_reasoning.get(alias)                  # per-alias default (tool vs tool-thinking)
    if r is not None:
        body["_reasoning"] = r
    # Sync park is the default: a ready backend dispatches now; all busy → queue until one
    # frees (per-alias park time) or 503. Async is not on chat/completions — it lives on the
    # standard /v1/responses background mode.
    return await _dispatch_or_park(alias, path, body, request)


# The Responses API ↔ Chat Completions translation layer (request/response/
# stream/shell builders) lives in responses_bridge.py — pure functions, no
# gateway state. The endpoints below own dispatch/parking + background mode.


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models(request: Request, authorization: Optional[str] = Header(None)):
    user = authenticate(authorization)
    # Per-user allow-list FILTERS the catalog (empty = all). Entries may be a model id,
    # a chat/image alias, or a backend name (grants all of that backend's models).
    allow = (user.get("models") if user else None) or []
    typ = (request.query_params.get("type") or "").lower()    # ""=both, "chat", "image"
    now = int(time.time())
    seen: set[str] = set()
    data = []

    def visible(keys: set) -> bool:                           # any grant-key in the allow-list?
        return (not allow) or any(k in allow for k in keys)

    # CHAT/LLM catalog = LLM backends only (ComfyUI "models" are checkpoints, not
    # chat-callable). Names are unique within the LLM type, so the prefix is unambiguous.
    if typ != "image":
        offered = [b for b in enabled_backends()
                   if not (_is_gen(b) or is_draining(b) or not backend_healthy.get(backend_id(b)))]
        for backend in offered:                   # comfy / draining / down → not offered
            bname = backend["name"]
            expose_bare = backend.get("local", False)
            for mid in sorted(backend_models.get(backend_id(backend), set())):
                # model_prefix on → '<backend>/<model>' (provider visible, distinct across
                # backends). off → legacy bare ids deduplicated across backends.
                disp = f"{bname}/{mid}" if model_prefix else mid
                if disp not in seen and visible({disp, mid, bname}):
                    seen.add(disp)
                    data.append(_with_context(
                        {"id": disp, "object": "model", "created": now, "owned_by": bname},
                        model_context(backend, mid)))
                # `local: true` backends ALSO list the bare id; a bare request routes
                # across every backend that exposes it (like a virtual alias) — so its
                # context_length is the SMALLEST of theirs.
                if expose_bare and mid not in seen and visible({mid, bname}):
                    seen.add(mid)
                    data.append(_with_context(
                        {"id": mid, "object": "model", "created": now, "owned_by": bname},
                        _context_min((b2, mid) for b2 in offered
                                     if mid in backend_models.get(backend_id(b2), set()))))
        # Virtual chat aliases are cross-backend → always listed bare (no prefix).
        for alias in virtual_models:
            if alias not in seen and visible({alias}):
                seen.add(alias)
                data.append(_with_context(
                    {"id": alias, "object": "model", "created": now, "owned_by": "ai-hub (virtual)"},
                    _alias_context(alias, offered)))

    # IMAGE generation aliases (separate namespace) — listed so image clients (anima-verse)
    # can discover them; granted by alias name. `?type=image` returns only these.
    if typ != "chat":
        img_aliases = (list((await asyncio.to_thread(store.list_aliases)).keys())
                       if store.is_active() else list(image_models.keys()))
        for alias in img_aliases:
            if alias not in seen and visible({alias}):
                seen.add(alias)
                data.append({"id": alias, "object": "model", "created": now, "owned_by": "ai-hub (image)"})

    return {"object": "list", "data": data}


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    now = int(time.time())
    llm = [b for b in enabled_backends() if not _is_gen(b)]
    if model_id in virtual_models:
        return _with_context(
            {"id": model_id, "object": "model", "created": now, "owned_by": "ai-hub (virtual)"},
            _alias_context(model_id, llm))
    bname, bare = split_backend_prefix(model_id)
    if bname is not None:
        b = next((b for b in llm if b["name"] == bname), None)
        if b is not None and bare in backend_models.get(backend_id(b), set()):
            return _with_context({"id": model_id, "object": "model", "created": now, "owned_by": bname},
                                 model_context(b, bare))
    else:
        hosts = [b for b in llm if model_id in backend_models.get(backend_id(b), set())]
        if hosts:      # a bare id routes across every host → the smallest window, as in the listing
            return _with_context(
                {"id": model_id, "object": "model", "created": now, "owned_by": hosts[0]["name"]},
                _context_min((b, model_id) for b in hosts))
    raise HTTPException(404, f"Model '{model_id}' not found")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    return await route("/v1/chat/completions", request, authorization)


def _playground_key(name: str) -> Optional[str]:
    """API key the /ui playgrounds use to call the gateway's OWN endpoints as a real
    client (they test the API, so they go through it — dispatch, parking, reasoning,
    stats, quotas). The logged-in admin's own key wins (correct attribution), else the
    master key; None in bootstrap-open mode (anonymous, x-source attributes it)."""
    u = next((u for u in users if u.get("name") == name and u.get("api_key")), None)
    return (u or {}).get("api_key") or api_key


# ── Responses background mode (official async) ────────────────────────────────
# `POST /v1/responses` with `background: true` returns immediately with a queued
# response object; the worker parks/dispatches in the same queue and stores the
# result. Poll `GET /v1/responses/{id}`; cancel `POST /v1/responses/{id}/cancel`.
_bg_tasks: dict = {}                        # response_id → asyncio.Task (for cancellation)
_RESP_PREFIX = "resp_"                      # background response id = resp_<job_id>


def _bg_job_id(response_id: str) -> str:
    """jobs.py id behind a background response id (strips the resp_ prefix)."""
    return response_id[len(_RESP_PREFIX):]


def _dispatch_json(resp) -> dict:
    """Parsed JSON of a non-stream dispatch result. The adapter attaches the body it
    already parsed for usage as `resp.parsed_json` — reuse it instead of re-parsing
    the raw bytes; fall back to the body for foreign Response objects."""
    parsed = getattr(resp, "parsed_json", None)
    if isinstance(parsed, dict):
        return parsed
    try:
        return json.loads(bytes(resp.body)) if getattr(resp, "body", None) else {}
    except Exception:
        return {}


async def _run_bg_response(job_id, rid, alias, chat_body, request, created):
    """Worker for a background response: dispatch (parking in the shared queue up to the
    async window), translate the chat completion → Responses object, store it on the job."""
    try:
        resp = await _dispatch_or_park(alias, "/v1/chat/completions", chat_body, request,
                                       stats_endpoint="/v1/responses",
                                       deadline=time.monotonic() + async_park_timeout_s)
        chat_json = _dispatch_json(resp)
        if getattr(resp, "status_code", 200) >= 400:
            await asyncio.to_thread(jobs.fail, job_id,
                                    f"{resp.status_code}: {(json.dumps(chat_json) or '')[:300]}")
            return
        obj = chat_to_responses(chat_json)
        obj["id"], obj["background"], obj["created_at"] = rid, True, created
        await asyncio.to_thread(jobs.complete_json, job_id, obj,
                                meta={"model": chat_json.get("model")})
    except asyncio.CancelledError:
        jobs.fail(job_id, "cancelled")   # sync on purpose (no await while being cancelled);
        raise                            # GET maps this back to status "cancelled"
    except HTTPException as e:
        await asyncio.to_thread(jobs.fail, job_id, f"{e.status_code}: {e.detail}")
    except Exception as e:                              # never let a background task vanish silently
        logger.warning(f"background response {rid} failed: {e}")
        await asyncio.to_thread(jobs.fail, job_id, str(e))


async def _create_bg_response(alias, chat_body, request) -> JSONResponse:
    if not jobs.is_active():
        raise HTTPException(503, "background responses unavailable (job store off)")
    if len(_parked) >= max_parked:
        raise HTTPException(503, f"too many queued ({max_parked}) — retry later", headers={"Retry-After": "2"})
    owner = _request_owner(request)
    job_id = await asyncio.to_thread(jobs.create, "response", alias, "(background)", owner=owner)
    rid, created = f"{_RESP_PREFIX}{job_id}", int(time.time())
    body = dict(chat_body); body["stream"] = False     # background result is fetched, not streamed
    t = asyncio.create_task(_run_bg_response(job_id, rid, alias, body, request, created))
    _bg_tasks[rid] = t
    t.add_done_callback(lambda _: _bg_tasks.pop(rid, None))
    if log_per_call:
        logger.info(f"→ background response {rid} (alias '{alias}') queued")
    return JSONResponse(response_shell(rid, "queued", alias, created, background=True),
                        status_code=200)


async def _bg_job_for(response_id: str) -> dict:
    if not response_id.startswith(_RESP_PREFIX) or not jobs.is_active():
        raise HTTPException(404, f"response '{response_id}' not found")
    job = await asyncio.to_thread(jobs.get, _bg_job_id(response_id))
    if job is None or job.get("task") != "response":
        raise HTTPException(404, f"response '{response_id}' not found")
    return job


def _bg_owner_check(job: dict, user: Optional[dict], request: Request) -> None:
    # non-admin users (and, open mode, anonymous IPs) only touch their own
    # responses; hide others as 404 (no leak).
    _check_owner(job, user, status=404, detail="response not found",
                 anon_owner=_request_owner(request))


def _bg_view(response_id: str, job: dict) -> dict:
    status = job["status"]
    created = int(job.get("created") or time.time())
    model = (job.get("meta") or {}).get("model") or job.get("alias")
    shell = lambda st, **kw: response_shell(response_id, st, model, created, background=True, **kw)
    if status == "done":
        return (job.get("results") or [None])[0] or shell("completed")
    if status == "failed":
        err = job.get("error") or "failed"
        if err == "cancelled":
            return shell("cancelled")
        return shell("failed", error={"message": err})
    return shell("in_progress" if status == "running" else "queued")


@app.post("/v1/responses")
async def responses(request: Request, authorization: Optional[str] = Header(None)):
    """OpenAI Responses API → Chat Completions bridge.

    LangChain.js (N8N's AI Agent) calls this endpoint by default. Backends that
    only speak Chat Completions still work — request and response are translated
    transparently. `stream:true` translates the backend's chat SSE into Responses
    SSE (A3); `background:true` runs it async (queued → poll GET /v1/responses/{id}).
    Like chat, a busy backend parks instead of 503.
    """
    raw_body = await request.json()
    chat_body = responses_to_chat(raw_body)
    alias = chat_body.get("model", "")
    await gate_request(authorization, request, alias)        # auth + model allow-list + quota

    _apply_alias_sampling(alias, chat_body)            # same per-alias defaults as the chat path
    r = _normalize_reasoning(raw_body)                 # honors reasoning + reasoning_effort + {effort}
    if r is None:
        r = alias_reasoning.get(alias)                 # per-alias default (tool vs tool-thinking)
    if r is not None:
        chat_body["_reasoning"] = r

    if raw_body.get("background") is True:             # official async: immediate queued object
        return await _create_bg_response(alias, chat_body, request)

    wants_stream = bool(raw_body.get("stream"))
    chat_body["stream"] = wants_stream
    if wants_stream:
        # The bridge consumes the chat usage chunk for `response.completed`;
        # since the adapter now hides it from clients that don't ask (strict
        # OpenAI shape), ask explicitly — it never reaches the client raw.
        chat_body["stream_options"] = {"include_usage": True}
    # Shared dispatch/park path (failover + in-flight + stats, labelled /v1/responses).
    resp = await _dispatch_or_park(alias, "/v1/chat/completions", chat_body, request,
                                   stats_endpoint="/v1/responses")
    if wants_stream:                                   # A3: translate chat SSE → Responses SSE
        if isinstance(resp, StreamingResponse):
            return StreamingResponse(responses_stream(resp, raw_body, alias),
                                     media_type="text/event-stream")
        err = _dispatch_json(resp)
        raise HTTPException(resp.status_code, (json.dumps(err) or "")[:500])
    chat_resp_json = _dispatch_json(resp)
    if resp.status_code >= 400:
        raise HTTPException(resp.status_code, (json.dumps(chat_resp_json) or "")[:500])
    return JSONResponse(chat_to_responses(chat_resp_json), status_code=resp.status_code)


@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Poll a background response: queued → in_progress → completed | failed | cancelled."""
    user = authenticate(authorization)
    job = await _bg_job_for(response_id)
    _bg_owner_check(job, user, request)
    return JSONResponse(_bg_view(response_id, job))


@app.post("/v1/responses/{response_id}/cancel")
async def cancel_response(response_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Cancel a background response (no-op if already terminal)."""
    user = authenticate(authorization)
    job = await _bg_job_for(response_id)
    _bg_owner_check(job, user, request)
    if job["status"] in ("queued", "running"):
        t = _bg_tasks.get(response_id)
        if t and not t.done():
            t.cancel()                                 # stop a parked/in-flight worker at its next await
        # Mark cancelled right away so the response reflects it now — but only if it
        # hasn't just reached a terminal state (cancel racing completion).
        cur = await asyncio.to_thread(jobs.get, _bg_job_id(response_id))
        if cur and cur.get("status") in ("queued", "running"):
            await asyncio.to_thread(jobs.fail, _bg_job_id(response_id), "cancelled")
        job = await _bg_job_for(response_id)           # re-read post-cancel
    return JSONResponse(_bg_view(response_id, job))


# ── Anthropic Messages frontdoor (Claude Code) ────────────────────────────────
# Claude Code speaks this protocol, so the gateway speaks it too: point it at the
# gateway with ANTHROPIC_BASE_URL and it can use Anthropic models through a
# subscription backend (verbatim passthrough, see AnthropicAdapter) AND open-weight
# models through any chat backend (translated by anthropic_bridge) — with the same
# routing, parking, failover, quotas and stats as every other endpoint.
#
# Claude Code authenticates with `x-api-key`; ANTHROPIC_AUTH_TOKEN sends
# `Authorization: Bearer` instead. Both carry a GATEWAY key here (never an
# Anthropic credential — that one lives on the backend).

def _client_credential(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[str]:
    """The caller's gateway credential from either header, in Bearer form."""
    if authorization:
        return authorization
    return f"Bearer {x_api_key}" if x_api_key else None


def _messages_error(status: int, message: str, headers: Optional[dict] = None) -> JSONResponse:
    """Gateway-side failure in the shape Claude Code renders (it reads
    error.message; an OpenAI-shaped or FastAPI `detail` body shows as blank).
    `headers` carries the ones that steer a client's retry — a park-timeout 503
    without its `Retry-After` tells the caller nothing about when to come back."""
    etype = {400: "invalid_request_error", 401: "authentication_error",
             403: "permission_error", 404: "not_found_error", 429: "rate_limit_error",
             402: "billing_error"}.get(status, "api_error")
    return JSONResponse({"type": "error", "error": {"type": etype, "message": str(message)}},
                        status_code=status, headers=headers or None)


async def _messages_route(path: str, request: Request, credential: Optional[str]):
    """Shared body of both Messages endpoints: authenticate, fold the thinking
    control into the gateway's normalized toggle (so a translated backend thinks
    when Claude Code asks it to), then hand over to the normal dispatch/park path.

    Deliberately NOT applied here: per-alias sampling defaults. Claude Code sends a
    complete, deliberate request, and a chat-shaped `min_p`/`repetition_penalty`
    would 400 against Anthropic and change behaviour everywhere else."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "request body is not valid JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")
    alias = body.get("model", "")
    await gate_request(credential, request, alias)          # auth + allow-list + quota
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") in ("enabled", "disabled"):
        # An explicit client control always wins over the per-alias default — same
        # rule as the chat path. Honoured by translated backends; the Anthropic
        # passthrough sets `thinking` itself and never sees this.
        body["_reasoning"] = "on" if thinking["type"] == "enabled" else "off"
    elif alias_reasoning.get(alias) is not None:
        body["_reasoning"] = alias_reasoning[alias]
    if path.startswith("/v1/messages/count_tokens"):
        # No chat backend has this endpoint — it is answered from an estimate. Doing
        # that BEFORE dispatch keeps it out of the in-flight/park machinery: sizing a
        # context must not queue behind real calls (or hold a slot on a
        # max_concurrent:1 backend) when no backend is needed at all.
        ready, busy = resolve_routes(alias, path)
        if not any(b.get("type") == "anthropic" for b, _ in ready + busy):
            return JSONResponse({"input_tokens": anthropic_bridge.estimate_input_tokens(body)})
    return await _dispatch_or_park(alias, path, body, request)


@app.post("/v1/messages")
async def messages(request: Request, authorization: Optional[str] = Header(None),
                   x_api_key: Optional[str] = Header(None)):
    """Anthropic Messages API — Claude Code's endpoint. Errors are shaped by the
    app-wide HTTPException handler (which also logs the refusal)."""
    return await _messages_route("/v1/messages", request,
                                 _client_credential(authorization, x_api_key))


@app.post("/v1/messages/count_tokens")
async def messages_count_tokens(request: Request, authorization: Optional[str] = Header(None),
                                x_api_key: Optional[str] = Header(None)):
    """Token count for a Messages body: passed through to an Anthropic backend,
    estimated by the bridge for a chat backend (which has no such endpoint)."""
    return await _messages_route("/v1/messages/count_tokens", request,
                                 _client_credential(authorization, x_api_key))


@app.post("/v1/completions")
async def completions(request: Request, authorization: Optional[str] = Header(None)):
    return await route("/v1/completions", request, authorization)


@app.post("/v1/embeddings")
async def embeddings(request: Request, authorization: Optional[str] = Header(None)):
    # Same routing as chat: body["model"] is the alias/model, picked up
    # by get_routes_for(). Embedding responses carry usage.prompt_tokens only
    # (no completion_tokens) → cost falls out of the input-price path for free.
    # Backends that filter out embedding models (chat_only) simply won't be
    # candidates here, so the request routes to a backend that actually serves it.
    return await route("/v1/embeddings", request, authorization)


@app.post("/v1/audio/speech")
async def audio_speech(request: Request, authorization: Optional[str] = Header(None)):
    # OpenAI-shaped TTS / voice cloning: routed like chat by body["model"] (bare id
    # or alias) through the same dispatch/parking/failover machinery. The backend's
    # binary audio (audio/wav etc.) passes through untouched — route() strips any
    # `stream` flag (audio is never SSE) and the adapter skips body parsing/stats
    # blobs for non-text responses. Extra fields (`voice`, `params.ref_text`, …)
    # are forwarded verbatim; a per-alias voice default may fill them in.
    return await route("/v1/audio/speech", request, authorization)


# ── Generation (image / video / TTS) ───────────────────────────────────────────
# Native job-based API. A generation alias resolves via `image_models` to ordered
# (backend, workflow) candidates; the request is rendered into the backend's
# protocol by its adapter. Results are persisted in the job store and retrievable
# by id (TTL) — sync mode also returns them inline. No VRAM coordinator yet
# (Phase 2): candidates are filtered by health + the existing busy cap only.

def _gen_backend_for(name: str, cand: Optional[dict],
                     pool: Optional[list] = None) -> Optional[dict]:
    """The generation backend `name` of the SAME KIND as `cand` (a cloud candidate ↔ a
    backend of that cloud type, workflow candidate ↔ comfyui backend). Backends are keyed
    (name, type), so a ComfyUI and a Meshy/Tripo backend may share a name; a bare-name
    match could route a cloud alias onto a GPU box (or a workflow alias onto the cloud
    API) and fail opaquely. `pool` defaults to the enabled generation backends."""
    want = adapters.cand_kind(cand or {})
    for b in (pool if pool is not None else _gen_backends):
        if b.get("name") == name and adapters.backend_kind(b) == want:
            return b
    return None


def _gen_routes(alias: str) -> tuple[list, list]:
    """(ready, all) (backend, candidate) pairs for a generation alias, each ordered
    **unpaid before paid, then fastest first** (`gen_speed` EMA per alias+backend; an
    unmeasured backend sorts first, probe-once) and filtered to enabled + healthy
    backends; `ready` additionally drops busy ones. ONE store read serves both lists.
    Priority sorts nothing any more (spec 2026-09-01).

    An alias holds a flat list of *allowed* backends (no primary/fallback). They are
    tried in that order; on a connection error the job runner moves to the
    next. An optional per-alias `retries` caps how many backends are attempted (1 +
    retries); blank = try all eligible. Each list is capped independently — same
    result as the old per-include_busy filtering.

    Reads from the writable store (UI source of truth) when active, falling back
    to the `image_models` config for aliases the store doesn't hold. Blocking
    (store read) — call via asyncio.to_thread from async code."""
    candidates = store.get(alias) if store.is_active() else None
    if candidates is None:
        candidates = image_models.get(alias, [])
    # generation routes only to generation backends (ComfyUI, Meshy, Tripo), so a name
    # shared with an LLM backend resolves to the right one — and, since backends are keyed
    # (name, type), the candidate's own KIND picks between a same-named ComfyUI and
    # cloud backend (see _gen_backend_for).
    gen = [b for b in _gen_backends if not is_draining(b)]
    allc = []
    for cand in candidates:
        b = _gen_backend_for(cand.get("backend"), cand, gen)
        if b is None or not backend_healthy.get(backend_id(b)):
            continue
        allc.append((b, cand))
    allc = scheduler.order_ready(allc, _gen_speed_of(alias), lambda b: bool(b.get("paid")))
    # A candidate under execution-fault quarantine drops out of `ready` but stays at the
    # END of `allc`. Both halves matter: out of `ready` so it is never CHOSEN while a
    # working backend exists, still in `allc` so "all busy" logic keeps seeing a
    # candidate and the job PARKS for a healthy backend instead of 503-ing — and so a
    # failover can still reach it as the last resort if everything else fails.
    usable, held = scheduler.split_quarantined(
        allc, lambda bx: f"{alias}|{backend_id(bx[0])}", gen_exec_faults, time.time())
    allc = usable + held
    ready = [r for r in usable if not backend_busy(r[0])]   # busy → only parkable, not ready
    raw = next((c.get("retries") for c in candidates if c.get("retries") not in (None, "")), None)
    if raw is not None:
        try:
            cap = max(1, int(raw) + 1)                      # first attempt + N retries
            allc, ready = allc[:cap], ready[:cap]
        except (ValueError, TypeError):
            pass
    return ready, allc


def get_gen_routes(alias: str, include_busy: bool = False) -> list[tuple[dict, dict]]:
    """Single-list view of _gen_routes() — kept for callers that need only one side
    (image slots, LoRA listing, admin)."""
    ready, allc = _gen_routes(alias)
    return allc if include_busy else ready


def _force_filter(routes: list, force: str) -> list:
    """Keep only the force-pinned backend's candidates (no-op without a pin)."""
    return [r for r in routes if r[0].get("name") == force] if force else routes


def _entry_can_use(entry: dict, backend: dict) -> bool:
    """May this waiting generation job run on `backend` right now? Mirrors the filters
    the job applies in its own poll: its own `exclude` list, the force pin, LoRA
    eligibility, and the alias actually routing to this backend while it is free.

    `exclude` (backend NAMES) is how a waiter declares a candidate it would refuse for
    a reason the routing tables cannot show — the chain maintains it from its per-pass
    `usable()` verdicts and its `tried` failover set. Without it a chain could be
    designated for a backend it will never claim and, once overdue, hold that idle
    backend against every other waiter until its own park deadline.

    Blocking (store read via _gen_routes) — call via asyncio.to_thread from async code."""
    if backend.get("name") in (entry.get("exclude") or ()):
        return False
    if entry.get("force") and backend.get("name") != entry["force"]:
        return False
    if entry.get("eligible") is not None and backend.get("name") not in entry["eligible"]:
        return False
    ready, _allc = _gen_routes(entry["alias"])
    bid = backend_id(backend)
    return any(backend_id(b) == bid for b, _cand in ready)


def _gen_waiting_pool() -> list:
    """Snapshot of the media queue as the scheduler sees it (`_gen_waiting` mutates
    across awaits). Entries that hold a backend slot are out: a claimant stays
    registered until its job is done, but it is no longer waiting and must not shadow
    the others out of a second free backend."""
    return [e for e in _gen_waiting if not e.get("claimed")]


def _designated_gen_waiter(backend: dict, pool: list, now: Optional[float] = None):
    """The queued generation job this free backend belongs to, or None.

    Media twin of _designated_waiter (spec 2026-09-01, "Designated taker"): overdue
    jobs first (oldest), else one needing the alias the backend last ran — one alias is
    one workflow, i.e. the model set already in VRAM — else the oldest job it can
    serve. THE one place the media designation is computed, so the poll gate and the
    fresh-arrival gate cannot drift apart. Blocking — via asyncio.to_thread."""
    return scheduler.designated_taker(
        pool,
        can_serve=lambda e: _entry_can_use(e, backend),
        type_key=lambda e: e["alias"],
        last_key=backend_last_key.get(backend_id(backend)),
        now=time.monotonic() if now is None else now,
        max_wait_s=affinity_max_wait_s)


def _may_claim_gen(entry: dict, backend: dict) -> bool:
    """Is this free backend THIS waiting job's to claim? A lone waiter always passes.
    Blocking — via asyncio.to_thread."""
    pool = _gen_waiting_pool()
    if len(pool) <= 1:                       # only us waiting → nothing to yield to
        return True
    return _designated_gen_waiter(backend, pool) is entry


def _designated_gen_index(entry: dict, ready: list) -> Optional[int]:
    """Index in `ready` of the best candidate this waiting job may claim now, or None
    (→ keep parking). All ready candidates are scanned, not just the best one, and that
    is what guarantees progress: every free backend has exactly one designated taker
    among the waiting jobs; `_entry_can_use` rejects every backend a waiter would
    refuse (`exclude`, force pin, LoRA eligibility, routability), so the taker really
    can claim it, and it has the backend in its own ready list — hence on every poll at
    least one waiter proceeds while a backend is free. The chain rebuilds its `exclude`
    set once per pass, so a designation it cannot honour is released within one 2 s
    poll. Blocking — via asyncio.to_thread."""
    for i, (b, _cand) in enumerate(ready):
        if _may_claim_gen(entry, b):
            return i
    return None


def _gen_reserved(backend: dict) -> bool:
    """Is this free media backend spoken for by a job already in the queue?

    Spec rule 4: a fresh request never overtakes the waiters — it may dispatch straight
    to a free backend only while no queued job is designated for it; otherwise it joins
    the queue as the youngest entry and competes from there. Empty queue → never
    reserved. Blocking — via asyncio.to_thread."""
    pool = _gen_waiting_pool()
    return bool(pool) and _designated_gen_waiter(backend, pool) is not None


def _gen_inputs_params(body: dict) -> tuple[dict, dict]:
    inputs = {
        "prompt": body.get("prompt", ""),
        "negative_prompt": body.get("negative_prompt", ""),
    }
    params = dict(body.get("params") or {})
    for k in ("width", "height", "steps", "cfg", "seed", "sampler", "scheduler", "seconds"):
        if k in body and k not in params:        # top-level convenience knobs
            params[k] = body[k]
    return inputs, params


def _mapping_param(mapping: dict, name: str) -> Optional[str]:
    """The mapping param whose EXTERNAL name (label, else param) is `name`."""
    for p, m in (mapping or {}).items():
        if p == name or ((m or {}).get("label") or "").strip().lower() == name:
            return p
    return None


def _apply_seconds(params: dict, cand: dict) -> None:
    """Video convenience: `seconds` → the alias's `frames` param, when the alias
    declares `fps` (mapping editor). Explicit frames always win; `frames_snap: S`
    rounds onto the S·k+1 raster (Wan). A mapping that itself exposes a param
    named `seconds` is left alone; `seconds` without alias fps is a clear 400
    instead of a silent ignore."""
    if _mapping_param(cand.get("mapping") or {}, "seconds"):
        return                                   # the workflow maps it directly
    sec = params.pop("seconds", None)
    if sec in (None, ""):
        return
    fps = cand.get("fps")
    mapping = cand.get("mapping") or {}
    fparam = _mapping_param(mapping, "frames")
    if not fps or fparam is None:
        raise HTTPException(400, f"'seconds' is not supported for this alias "
                                 f"({'no fps configured' if not fps else 'no frames param'}) — send frames directly")
    lbl = ((mapping.get(fparam) or {}).get("label") or "").strip()
    if params.get(fparam) not in (None, "") or (lbl and params.get(lbl) not in (None, "")):
        return                                   # explicit frames beats the convenience knob
    try:
        frames = max(1, round(float(sec) * float(fps)))
    except (TypeError, ValueError):
        raise HTTPException(400, f"'seconds' must be a number (got {sec!r})")
    snap = cand.get("frames_snap")
    if snap:
        s = max(1, int(snap))
        frames = max(1, round((frames - 1) / s) * s + 1)
    params[fparam] = frames


async def _job_view(job_id: str, request: Request) -> dict:
    job = await asyncio.to_thread(jobs.get, job_id)
    if job is None:
        raise HTTPException(404, f"job '{job_id}' not found")
    view = {
        "job_id": job_id, "status": job["status"], "task": job["task"],
        "alias": job["alias"], "backend": job["backend"], "error": job["error"],
        "meta": job["meta"],
    }
    if job["task"] == "chat":                           # parked chat → inline completion JSON
        view["completion"] = job["results"][0] if job["results"] else None
        return view
    base = str(request.base_url).rstrip("/")
    view["results"] = [{
        "n": r["n"], "mime": r["mime"], "kind": r["kind"], "name": r.get("name"),
        "sha256": r.get("sha256"),
        "url": f"{base}/v1/jobs/{job_id}/result/{r['n']}",
    } for r in job["results"]]
    meta = job.get("meta") or {}
    # Client-facing delivery metadata: rig type (mixamo → shared anim library
    # applies; generic → procedural idle) and any web-suitability warnings — see the
    # character-model spec. The workflow identity is already `view["alias"]`.
    if meta.get("rig"):
        view["rig"] = meta["rig"]
    if meta.get("rig_spec"):
        # Which bone-naming convention a `rig: "tripo"` delivery carries (mixamo|tripo).
        # The client spec documents it as a TOP-LEVEL job field, so lift it like `rig`:
        # a client that reads only the job object cannot see meta.
        view["rig_spec"] = meta["rig_spec"]
    if meta.get("warnings"):
        view["warnings"] = meta["warnings"]
    view["inputs"] = meta.get("inputs")
    # sha256 mirrors `results[]`: it identifies the exact bytes that went INTO the run,
    # so a client can prove which reference image a delivered artifact was made from.
    view["input_images"] = [{
        "n": r["n"], "slot": r.get("slot"), "mime": r["mime"],
        "sha256": r.get("sha256"), "bytes": r.get("bytes"),
        "url": f"{base}/v1/jobs/{job_id}/input/{r['n']}",
    } for r in meta.get("input_images", [])]
    # Live jobs get an honest progress estimate: elapsed vs the median runtime of
    # the alias's recent done jobs (same backend when it has history). Capped at
    # 0.97 — only completion says 100%. No history → elapsed only.
    if job["status"] in ("queued", "running"):
        elapsed = max(0, int(time.time()) - int(job.get("created") or 0))
        view["elapsed_s"] = elapsed
        if job["status"] == "running":
            # The backend's own step counter wins when we have it: the median can only
            # say what jobs of this shape USUALLY take, while this one knows that this
            # run is at step 25 of 35 — and derives its ETA from the seconds per step
            # actually measured here. Falls back to the median the moment the feed is
            # unavailable (old ComfyUI, no websockets module, a dropped socket).
            live = gen_progress.get(job_id)
            if live and live.get("fraction") is not None:
                view["progress"] = round(min(live["fraction"], 0.99), 2)
                view["progress_basis"] = "live"
                view["progress_step"] = f"{live.get('step')}/{live.get('steps')}"
                if live.get("node"):
                    view["progress_node"] = live["node"]
                if live.get("eta_s") is not None:
                    view["eta_s"] = live["eta_s"]
                view["progress_age_s"] = max(0, int(time.time() - (live.get("at") or 0)))
                return view
            med = (await asyncio.to_thread(jobs.median_duration, job["alias"], job["backend"])
                   or await asyncio.to_thread(jobs.median_duration, job["alias"]))
            if med and med > 0:
                view["progress"] = round(min(elapsed / med, 0.97), 2)
                view["eta_s"] = max(0, int(med - elapsed))
                view["progress_basis"] = "history-median"
    return view


# Connection-type errors that warrant failing over to the next candidate backend.
# A crashed/unreachable ComfyUI raises these; a content error (ComfyUI validation/
# execution → RuntimeError) does not — it would fail identically elsewhere.
_GEN_FAILOVER_ERRORS = (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError,
                        ConnectionError, TimeoutError)


def _err_text(e: BaseException) -> str:
    """The text of an exception, never empty — falls back to the class name.

    Several exceptions that reach a job row carry NO message at all: every httpx
    timeout is constructed as `WriteTimeout("")` when it comes off the transport, so
    `str(e)` is "". Measured 2026-09-02 on prod: a chain's stage-2 create POST timed
    out and the job's error read exactly "chain failed: " — nothing after the colon,
    the journal line equally blank, and the only way to learn WHAT failed was to
    re-derive it. A class name ("WriteTimeout") is a poor message but an infinitely
    better one than none, so every `{e}` that ends up in a job row or a log line goes
    through here."""
    return str(e) or type(e).__name__


# ── Per-backend rolling generation fail-rate (runbook C) ────────────────────────
def _fault_label(e: BaseException) -> str:
    """How to name a failover-worthy generation fault in a log line. Three distinct
    causes hide behind one failover path — say which one it was."""
    if isinstance(e, adapters.CloudNoCredits):
        return "no credits left"
    if isinstance(e, adapters.CloudBusy):
        return f"{e.vendor} queue full"
    if isinstance(e, adapters.CloudTaskRetryable):
        return f"{e.vendor} failed the task on its side"
    if isinstance(e, TimeoutError):            # the adapter's own max_wait cap
        return "did not finish in time"
    if isinstance(e, httpx.TimeoutException):  # a single HTTP round trip timed out
        return "timed out mid-request"
    return "connection issue"


def _gen_exhausted_msg(last: Optional[BaseException]) -> str:
    """The message a generation job dies with once every candidate is used up.

    `_GEN_FAILOVER_ERRORS` lumps timeouts in with genuine connection faults — correct
    for routing (all warrant trying another backend), wrong for the report: a job that
    hit `max_wait` reached its backend perfectly well and was simply not finished in
    time. Saying "unreachable (connection)" there sends you diagnosing the network
    instead of the workflow (measured 2026-08-25). The `max_wait` hint belongs ONLY on
    the adapter's own TimeoutError — an httpx timeout is a transport fault and naming
    max_wait there would mislead exactly the same way."""
    # `_err_text`, not `{last}`: an httpx timeout stringifies to "" and would leave the
    # message ending in a bare colon (see _err_text). None only reaches here when no
    # candidate was ever tried, which has no exception to name.
    txt = _err_text(last) if last is not None else "no candidate was tried"
    if isinstance(last, adapters.CloudNoCredits):
        return f"no candidate backend could run it — {last.vendor} account out of credits: {txt}"
    if isinstance(last, adapters.CloudBusy):
        return f"no candidate backend could run it — {last.vendor} queue limit reached: {txt}"
    if isinstance(last, adapters.CloudTaskRetryable):
        # Named apart from "unreachable": the task REACHED the vendor and the vendor broke
        # it. Sending someone to check the network here is the same wrong turn the max_wait
        # case warns about, and the credits question ("was I billed?") is answered too.
        return (f"every attempt failed inside {last.vendor} — the vendor called it a "
                f"retryable fault and consumed no credits: {txt}")
    if isinstance(last, TimeoutError):
        return (f"no candidate backend finished in time — the gateway's per-backend "
                f"`max_wait` (default 600s; raise it in Backends for slow workflows): {txt}")
    if isinstance(last, httpx.TimeoutException):
        return f"no candidate backend answered in time (transport timeout): {txt}"
    return f"all candidate backends unreachable (connection): {txt}"


def _cloud_trace_of(req) -> dict:
    """A COPY of what a cloud adapter recorded on this request (`NormalizedRequest.
    cloud_trace`), empty for a ComfyUI request or an attempt that never got that far.
    Copied because the request is re-built per attempt and the trace outlives it."""
    return dict(getattr(req, "cloud_trace", None) or {})


def _gen_fail_meta(attempts: int, cloud_trace: dict) -> Optional[dict]:
    """The meta a failed generation job carries: the retry count (kept visible, runbook B)
    plus whatever a cloud candidate had already created. The cloud keys are the SAME ones
    the success meta uses, so `admin._cloud_table` and the job view render a failed run
    with no special case. None when there is nothing to say."""
    meta = dict(cloud_trace)
    if attempts > 1:
        meta["attempts"] = attempts
    return meta or None


# bid → deque[(ts, conn_fail)] of the last generate() attempts. In-memory on
# purpose (a gateway restart resets the sample — fine) and module-global so
# adapter rebinds on config hot-reload don't lose it. Display-only: NEVER used
# to drop a backend from rotation (the operator decides / runbook A1).
backend_gen_window: dict = {}
_GEN_WINDOW_N = 50            # last N attempts …
_GEN_WINDOW_S = 86400         # … no older than 24 h


def _record_gen_attempt(bid: str, conn_fail: bool, exec_fail: bool = False) -> None:
    """Count one generate() attempt: every attempt lands in the window; `conn_fail`
    marks connection-type aborts (_GEN_FAILOVER_ERRORS), `exec_fail` marks a prompt the
    backend RAN and blew up on.

    The two are counted apart because they mean opposite things to an operator: a
    connection rate says the backend keeps falling over, an execution rate says it is
    up and burning every job it is given. Lumping them lost the second entirely — an
    execution failure used to book as a clean attempt, so a backend that reliably
    destroyed every job made its own fail-rate go DOWN with each one (measured
    2026-09-03 on comfyui-strix: 53 %, and all of it from the crash phase)."""
    dq = backend_gen_window.setdefault(bid, deque(maxlen=_GEN_WINDOW_N))
    dq.append((time.time(), bool(conn_fail), bool(exec_fail)))


def _gen_fail_stats(bid: str) -> Optional[dict]:
    """{fail_rate, gen_fails, exec_fail_rate, exec_fails, gen_attempts} over the
    window, or None without data. `fail_rate` keeps its old meaning (connection-type
    faults) so nothing that reads it changes meaning; the execution rate is reported
    beside it — see `_record_gen_attempt` for why they must not be merged."""
    dq = backend_gen_window.get(bid)
    if not dq:
        return None
    cutoff = time.time() - _GEN_WINDOW_S
    total = fails = xfails = 0
    for entry in dq:
        ts, cf = entry[0], entry[1]
        xf = entry[2] if len(entry) > 2 else False   # window may predate the exec column
        if ts >= cutoff:
            total += 1
            fails += 1 if cf else 0
            xfails += 1 if xf else 0
    if not total:
        return None
    return {"fail_rate": round(fails / total, 2), "gen_fails": fails,
            "exec_fail_rate": round(xfails / total, 2), "exec_fails": xfails,
            "gen_attempts": total}


async def _wait_backend_up(backend: dict, timeout_s: float = 30.0) -> None:
    """Give a crashed-and-systemd-restarting ComfyUI time to come back before a
    self-retry: poll /system_stats until it answers (or timeout_s passes). Never
    raises — a still-down backend just makes the retry fail fast into failover."""
    if backend.get("type") != "comfyui":
        return                      # a cloud task API has no VRAM to free / no host siblings
    url = (backend.get("url") or "").rstrip("/")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await asyncio.sleep(2.0)
        try:
            r = await http_client.get(f"{url}/system_stats", timeout=5.0)
            if r.status_code == 200:
                return
        except Exception:
            pass


def _host_flag(host: str, key: str) -> bool:
    """A host's GPU-policy flag: stored value, else the default `scheduler.HOST_FLAGS`
    gives it (the console reads the same table — one answer for an untouched host)."""
    return scheduler.host_flag(hosts_meta.get(host), key, _shared_host(host))


def _shared_host(host: str) -> bool:
    """A host running BOTH an LLM and a ComfyUI backend — they share its GPU."""
    kinds = {bid.split(":", 1)[0] for bid in host_backends.get(host, ())}
    return "comfyui" in kinds and len(kinds) > 1


_FREE_SETTLE_S = 30.0            # how long a free may take to SHOW in /system_stats before we stop trusting it
_FREE_REPOST_S = 2.0             # re-post /free this often while the VRAM has not moved (see below)
_FREE_FLOOR_BYTES = 256 << 20    # torch's reserved pool at or under this = "nothing loaded"


async def _comfy_vram_reserved(url: str) -> Optional[int]:
    """Bytes torch holds on the backend's devices per /system_stats (`torch_vram_total`
    is `torch.cuda.memory_reserved()` — the pool OTHER processes cannot use until
    `empty_cache()` gives it back, which is exactly what an unload must achieve for a
    rigger in its own process or a llama-swap load). None when unreadable."""
    try:
        r = await http_client.get(f"{url}/system_stats", timeout=5.0)
        devs = r.json().get("devices") or []
        return sum(int(d.get("torch_vram_total") or 0) for d in devs)
    except Exception:
        return None


async def _comfy_free(backend: dict, why: str, settle_s: float = 0.0,
                      abort_when: Optional[Callable[[], bool]] = None) -> bool:
    """POST /free (unload models + free memory) to a ComfyUI backend and, with
    `settle_s`, wait until the VRAM is actually gone. The raw call — every policy lives
    in the two callers below. Never raises: a box that cannot be freed must not fail the
    job that asked for it. Returns whether the free is KNOWN to have happened — the
    callers record what the GPU holds from that verdict, never from the attempt.

    Why waiting on the POST is not enough (ComfyUI server.py/main.py, checked
    2026-09-08): `/free` only sets two flags on the prompt queue and answers 200. The
    single prompt worker reads the flags AFTER `q.get()` returns — after executing the
    prompt it just took — and `set_flag` wakes it with a `Condition.notify` that is
    LOST when nobody is waiting: right after a prompt the worker is in `gc.collect()` +
    `soft_empty_cache()` (a second or more after a heavy 3D run), which is precisely
    when the gateway's 1 s /history poll sees the result and posts the free. The flag
    then sits until the NEXT prompt has run — i.e. stage 2 of a chain executes on stage
    1's whole cache first and the unload lands after it (the cc604da29e0e OOM by another
    route). So: post, watch `torch_vram_total`, and re-post every `_FREE_REPOST_S` until
    it drops — a re-post is idempotent (the flags are booleans) and the second one lands
    once the worker waits again. `settle_s = 0` keeps the fire-and-forget shape;
    `abort_when` stops the re-posting (verdict False) once the caller's reason is gone —
    the after-job free passes "a job claimed this backend", because a re-post under a
    prompt that just started would unload the models it is loading."""
    url = (backend.get("url") or "").rstrip("/")
    name = backend.get("name")
    body = {"unload_models": True, "free_memory": True}
    # 5 s: the POST only sets two flags and answers in milliseconds when the box is
    # alive. This call is AWAITED on the claim path with the slot already held, so a
    # box whose HTTP layer is gone must cost one short timeout there, not 30 s per
    # phase before generate() spends its own (review 2026-09-08).
    before = await _comfy_vram_reserved(url) if settle_s > 0 else None
    try:
        await http_client.post(f"{url}/free", json=body, timeout=5.0)
    except Exception as e:
        logger.warning(f"host: /free on [{name}] failed: {e}")
        return False
    if settle_s <= 0 or before is None:
        logger.info(f"host: freed ComfyUI VRAM on [{name}] — {why}")
        return True                 # nothing to watch (or no stats to watch it by)
    if before <= _FREE_FLOOR_BYTES:
        logger.info(f"host: ComfyUI VRAM on [{name}] already empty ({before >> 20} MiB) — {why}")
        return True
    target = max(before // 5, _FREE_FLOOR_BYTES)      # "gone" = ≤ 20 % of what it held
    t0 = last_post = time.monotonic()
    while True:
        await asyncio.sleep(0.5)
        held = await _comfy_vram_reserved(url)
        now = time.monotonic()
        if held is not None and held <= target:
            logger.info(f"host: freed ComfyUI VRAM on [{name}] — {why} "
                        f"({before >> 20} → {held >> 20} MiB in {now - t0:.1f} s)")
            return True
        if now - t0 >= settle_s:
            logger.warning(f"host: /free on [{name}] did not settle in {settle_s:.0f} s "
                           f"(still {(held or 0) >> 20} of {before >> 20} MiB) — {why}")
            return False
        if now - last_post >= _FREE_REPOST_S:
            if abort_when is not None and abort_when():
                logger.info(f"host: /free on [{name}] not repeated — {why} no longer applies")
                return False
            try:
                await http_client.post(f"{url}/free", json=body, timeout=5.0)
                last_post = now
            except Exception as e:
                logger.warning(f"host: /free on [{name}] failed: {e}")
                return False


def _model_set_key(adapter, req) -> Optional[str]:
    """The request's VRAM key from the adapter that will run it, None for one that
    has no notion of it (a cloud task API) — see _claim_gen_backend."""
    fn = getattr(adapter, "model_set_key", None)
    return fn(req) if callable(fn) else None


async def _claim_gen_backend(backend: dict, key: str, vram_key: Optional[str] = None) -> None:
    """Record the type key a generation backend now runs AND, for ComfyUI, free its
    VRAM first when the MODEL SET changed (scheduler.free_vram_before_job).

    `vram_key` is what the request will load (`ComfyUIAdapter.model_set_key` →
    `scheduler.model_set_key`: the loader nodes' weight names after pins/mapping/LoRAs),
    falling back to the alias `key` when nothing recognisable is loaded. The alias
    alone was the key at first — wrong both ways, see `scheduler.model_set_key`.

    This is the one moment where the question can be answered instead of guessed:
    the job is claimed, so what comes next is decided. Freeing after a job is the
    older, weaker half of it — it throws away a cache the next job may want and, on
    a box nobody hands another media job to, happens far too late to matter. What
    made it not merely wasteful: ComfyUI holds its cache across a GATEWAY restart
    too, so a first job after one starts against a full GPU (measured 2026-09-05 on
    k12-gpu — 21.4 of 23.5 GiB held by the ComfyUI process, and the Make-It-Animatable
    node, which runs in its OWN venv/process and can therefore never share that cache,
    died on `CUDA out of memory. Tried to allocate 20.00 MiB`).

    Awaited, unlike the after-job free: the prompt must not be submitted before the
    VRAM is actually gone — `_comfy_free` waits for that, because the POST alone only
    queues the unload (see there). Host flag `comfy_free_before_job`, default ON for
    every ComfyUI host (a same-alias run keeps its models either way, so the flag only
    buys back reload time for a box whose VRAM is never contended).

    Two records, deliberately: `backend_last_key` is the scheduler's AFFINITY key (what
    ran here last — set for every backend type, before any early return, because a cloud
    backend is designated by the same rule) and `backend_vram_key` is what the GPU is
    KNOWN to hold, written from the free's verdict and never from the attempt. They used
    to be one field, written up front: a /free that timed out (or was skipped for a job
    in flight) left the alias on record, so every later same-alias job trusted a cache
    that was still another alias's — the OOM this mechanism exists to prevent, made
    permanent (review 2026-09-08). None means unknown OR mixed, and both free next."""
    bid = backend_id(backend)
    backend_last_key[bid] = key
    if backend.get("type") != "comfyui":
        return                      # a cloud task API has no VRAM to free / no host siblings
    key = vram_key or key
    held = backend_vram_key.get(bid)
    host = backend_hosts.get(bid, "")
    if not scheduler.free_vram_before_job(
            held, key,
            others_inflight=max(0, backend_inflight.get(bid, 0) - 1),   # our own slot is held
            enabled=_host_flag(host, "comfy_free_before_job")):
        if held != key:
            backend_vram_key[bid] = None   # our set joins whatever is there → nobody knows
        return
    ok = await _comfy_free(backend, f"claiming '{key}' (was {held or 'unknown'})",
                           settle_s=_FREE_SETTLE_S)
    backend_vram_key[bid] = key if ok else None


async def _free_comfy_vram(backend: dict, why: str) -> None:
    """POST /free to a ComfyUI backend after its media job ended (fire-and-forget via
    create_task). The SECOND half of the policy above, and the only one that serves a
    non-media purpose: on a shared box a llama-swap load aborts on the VRAM ComfyUI
    still holds hours after its last job (phase 3, host plan), so that host frees
    eagerly. Skipped while ANOTHER generation runs there (the free would drop its
    cache) and when the job the scheduler will hand this backend next
    (`_designated_gen_waiter` — the ONE place that designation is computed) wants the
    alias its VRAM holds: freeing would make it reload the model set it is queued for.
    Asking the scheduler is the point: a same-alias waiter that can never run here (it
    excluded this backend after a failed chain stage, it is force-pinned elsewhere)
    must not suppress the free — it did, when this scanned the whole queue by alias
    (review 2026-09-08), and on a shared box nothing else ever freed the GPU again.
    Policy: host flag `comfy_free_after_job`; absent = ON for shared hosts only (a
    dedicated comfy box keeps its cache — the before-job free above already guarantees
    an empty GPU whenever the alias changes)."""
    if backend.get("type") != "comfyui":
        return                      # a cloud task API has no VRAM to free / no host siblings
    bid = backend_id(backend)
    host = backend_hosts.get(bid, "")
    if not _host_flag(host, "comfy_free_after_job"):
        return
    if backend_inflight.get(bid, 0) > 0:
        return
    held = backend_vram_key.get(bid)         # None = unknown/mixed → free regardless of waiters
    if held:
        pool = _gen_waiting_pool()
        if pool:
            nxt = await asyncio.to_thread(_designated_gen_waiter, backend, pool)
            if nxt is not None and nxt.get("alias") == held:
                return
            if backend_inflight.get(bid, 0) > 0:
                return              # claimed while we were in the store — its prompt is next
    # Settled too: the after-job POST lands in the worker's post-prompt gc window more
    # often than not (see _comfy_free), where a lone post is simply lost.
    if await _comfy_free(backend, f"after {why}", settle_s=_FREE_SETTLE_S,
                         abort_when=lambda: backend_inflight.get(bid, 0) > 0):
        backend_vram_key[bid] = None


async def _unload_host_llms(backend: dict) -> None:
    """Best-effort GET /unload on the LLM siblings of a ComfyUI backend's host
    right before a media job runs there, so the generation doesn't start against
    VRAM a loaded llama-swap model holds. Host flag `llm_unload_before_media`,
    default OFF — llama-swap's TTL usually clears the model anyway (endpoint
    verified on k12-gpu: llama-swap GET /unload → 200; other servers ignore it)."""
    if backend.get("type") != "comfyui":
        return                      # a cloud task API has no VRAM to free / no host siblings
    bid = backend_id(backend)
    host = backend_hosts.get(bid, "")
    if not _host_flag(host, "llm_unload_before_media"):
        return
    for obid in host_backends.get(host, ()):
        if not obid.startswith("openai:"):
            continue
        ob = next((x for x in backends if backend_id(x) == obid), None)
        if ob and ob.get("url"):
            try:
                await http_client.get(f"{ob['url'].rstrip('/')}/unload", timeout=8.0)
                logger.info(f"host: unloaded LLMs on [{ob['name']}] before media job")
            except Exception:
                pass


async def _run_job(job_id: str, alias: str, candidates: list, build_req) -> None:
    """Run a generation job, failing over to the next candidate on connection-type
    errors. A backend with `self_retries: n` gets n extra attempts on ITSELF first
    (runbook B: for sporadic driver faults the same host is the cheapest second
    try — no model re-load elsewhere, same success odds as attempt one). Its ONE
    job slot is held across the repeats, so no parked job slips in between.
    An execution error (the backend ran the prompt and it blew up) moves to the next
    candidate too, but never repeats on the SAME backend — see the `except Exception`
    arm for why that is not the same thing as a connection failover, and why a CLOUD
    candidate is excluded from it. Stops at the first success."""
    await asyncio.to_thread(jobs.set_status, job_id, "running")
    last = None
    attempts = 0
    # (bid, name, error) of candidates that ran the prompt and failed. Kept until the
    # job ends because a fault only counts as the BACKEND's once a later candidate has
    # succeeded — until then it is indistinguishable from a broken request.
    exec_faults: list = []
    # What a CLOUD candidate had already created when it failed (task id, endpoint, the
    # request summary). A failed run returns no GenOutput, so these facts reach the job row
    # only from here — and they are the ones a cloud failure is diagnosed with: without
    # them the task id survives only inside the error TEXT and everything else not at all
    # (measured 2026-09-03, job 9cf448115b4b). Last writer wins: the candidate the job
    # actually died on is the one worth describing.
    cloud_trace: dict = {}
    for backend, cand in candidates:
        bid = backend_id(backend)
        adapter = backend_adapters.get(bid)
        if adapter is None:
            continue
        try:
            tries = 1 + max(0, int(backend.get("self_retries") or 0))
        except (TypeError, ValueError):
            tries = 1                          # malformed config value → no self-retry
        _inflight_inc(bid)                     # hold ONE slot across all self-retries
        # The job row was stamped with the FIRST candidate at creation; re-point it at
        # the backend actually claiming it. A parked job routinely lands somewhere else
        # (a different backend freed first, or one came back while it waited), and a row
        # naming the wrong backend sends you reading the wrong ComfyUI's log. Same
        # reason the chain re-points at claim and hand-off.
        await asyncio.to_thread(jobs.set_backend, job_id, backend["name"])
        try:
            for attempt in range(1, tries + 1):
                attempts += 1
                req = None                             # so a build_req failure cannot leave
                try:                                   # the PREVIOUS attempt's trace behind
                    req = build_req(backend, cand)
                    req.slot_held = True               # we hold it — generate() must not double-count
                    # Type affinity + an empty GPU when the MODEL SET changed (see
                    # _claim_gen_backend); the request is built first because the key
                    # is what it loads. A self-retry re-runs the same key → no free.
                    await _claim_gen_backend(backend, alias, _model_set_key(adapter, req))
                    await _unload_host_llms(backend)   # opt-in host policy, no-op by default
                    t0 = time.monotonic()
                    out = await adapter.generate(req)
                    _record_gen_attempt(bid, conn_fail=False)
                    _note_gen_speed(alias, bid, time.monotonic() - t0)
                    _settle_exec_faults(alias, bid, exec_faults)
                    meta = dict(out.meta or {})
                    if attempts > 1:
                        meta["attempts"] = attempts    # retries must stay visible (runbook B)
                    await asyncio.to_thread(jobs.complete, job_id, out.blobs, meta)
                    if log_per_call:
                        logger.info(f"✓ job {job_id} done on [{backend['name']}] — "
                                    f"{len(out.blobs)} artifact(s)"
                                    + (f" after {attempts} attempts" if attempts > 1 else ""))
                    asyncio.create_task(_free_comfy_vram(backend, "job done"))
                    return
                except _GEN_FAILOVER_ERRORS as e:
                    _record_gen_attempt(bid, conn_fail=True)
                    last = e
                    cloud_trace = _cloud_trace_of(req) or cloud_trace
                    # both fail over, but they are different faults — name them apart so
                    # the log points at the workflow, not the network (see _gen_exhausted_msg)
                    what = _fault_label(e)
                    if attempt < tries:
                        logger.warning(f"✗ job {job_id} [{backend['name']}] {what} "
                                       f"({type(e).__name__}: {e}) — retrying same backend "
                                       f"(self-retry {attempt}/{tries - 1})")
                        await _wait_backend_up(backend)
                        continue
                    logger.warning(f"✗ job {job_id} [{backend['name']}] {what} "
                                   f"({type(e).__name__}: {e}) — failing over")
                    asyncio.create_task(_free_comfy_vram(backend, "job failover"))
                except Exception as e:
                    # Execution error: the backend accepted the prompt and it blew up.
                    # NEVER retried on the same backend (a re-run reproduces it), but it
                    # does move on to the next CANDIDATE — the old assumption that such an
                    # error "would fail identically on any backend" is only true when the
                    # REQUEST is at fault. It is false when the backend is: a broken torch
                    # build, a missing custom node, an incompatible platform all surface
                    # here while /object_info keeps answering and the backend keeps
                    # reporting healthy (measured 2026-09-03 on comfyui-strix — a ROCm
                    # update broke every Flux-family model load, and four user retries in a
                    # row died on it while two backends that could run the alias sat idle).
                    _record_gen_attempt(bid, conn_fail=False, exec_fail=True)
                    cloud_trace = _cloud_trace_of(req) or cloud_trace
                    if adapters.cloud_kind(cand):
                        # A cloud task is BILLED. Whatever failed here may have happened
                        # after the paid task was created, and re-running the job on the
                        # next candidate would buy the same mesh twice — the invariant
                        # tripo.py/adapters.py go out of their way to preserve. Final.
                        logger.warning(f"✗ job {job_id} [{backend['name']}] failed: {_err_text(e)}")
                        await asyncio.to_thread(jobs.fail, job_id, _err_text(e),
                                                _gen_fail_meta(attempts, cloud_trace))
                        return
                    exec_faults.append((bid, backend["name"], e))
                    last = e
                    logger.warning(f"✗ job {job_id} [{backend['name']}] execution failed: "
                                   f"{_err_text(e)} — trying the next backend")
                    asyncio.create_task(_free_comfy_vram(backend, "job failure"))
                    break                      # next candidate; never the same backend
        finally:
            _inflight_dec(bid)
    # Every candidate is used up. If any of them EXECUTED and failed, that error is the
    # report — "all candidate backends unreachable" would send you diagnosing a network
    # that was never involved. And no fault is charged to anyone here: when they all
    # failed, the request is the common factor, not the backends (this is what keeps one
    # bad workflow from quarantining every backend an alias has).
    if exec_faults:
        msg = _err_text(exec_faults[0][2])
        if len(exec_faults) > 1:
            msg += (f" — the same on {len(exec_faults)} backends "
                    f"({', '.join(n for _, n, _ in exec_faults)}), so the request is at fault")
        await asyncio.to_thread(jobs.fail, job_id, msg,
                                _gen_fail_meta(attempts, cloud_trace))
        return
    await asyncio.to_thread(jobs.fail, job_id, _gen_exhausted_msg(last),
                            _gen_fail_meta(attempts, cloud_trace))


def _chain_mesh_param_error(s2: dict, mesh_param: str, succ_alias: str) -> Optional[str]:
    """Why `mesh_param` cannot carry the mesh into the successor candidate `s2` (None = it
    can). `mesh_param` must be a request field of the successor (accepted under param OR
    label, exactly like any incoming value) — every adapter silently drops an unknown
    param, so stage 2 would otherwise run on the workflow's baked-in mesh path (or, on a
    cloud backend, on no mesh at all) and deliver a stale/WRONG mesh as a "done" job. A
    cloud successor (Meshy, Tripo) carries no mapping: its request fields are the fixed
    label table, and the mesh is a FILE field (public_fields()[2]). Pure over `s2` —
    unit-tested."""
    k = adapters.cloud_kind(s2)
    if k:
        vendor = adapters.cloud_module(k).VENDOR
        s2_files = [f["name"] for f in adapters.public_fields(s2)[2]]
        if mesh_param in s2_files:
            return None
        return (f"chain mesh param '{mesh_param}' is not a file field of the {vendor} "
                f"successor '{succ_alias}' — it takes "
                + (", ".join(f"'{n}'" for n in s2_files) if s2_files else
                   "no file input at all (only a rigging alias does)"))
    if any(p == mesh_param or ((m or {}).get("label") or "").strip() == mesh_param
           for p, m in (s2.get("mapping") or {}).items()):
        return None
    return (f"chain mesh param '{mesh_param}' is not a request field "
            f"(param or label) of successor '{succ_alias}' — map the "
            f"successor's mesh-load input or fix 'successor mesh param'")


async def _wait_and_hold(backend: dict, job_id: str, label: str) -> bool:
    """Wait until `backend` is free, then claim a slot (inflight_inc) atomically — the
    last busy-check and the inc run with no await between them (the dispatch invariant).
    Returns True holding the slot, or fails the job and returns False on park-timeout.
    The caller must _inflight_dec(backend_id(backend)) once done with the slot."""
    deadline = time.monotonic() + async_park_timeout_s
    while backend_busy(backend):
        if time.monotonic() > deadline:
            await asyncio.to_thread(jobs.fail, job_id,
                                    f"chain: backend '{backend.get('name')}' stayed busy past park time ({label})")
            return False
        await asyncio.sleep(0.5)
    _inflight_inc(backend_id(backend))
    return True


async def _run_chain(job_id: str, alias: str, succ: dict, body: dict, request,
                     upload_images, upload_files, inputs: dict, params: dict,
                     force: str = "", eligible: Optional[set] = None) -> None:
    """Run a two-stage workflow chain: stage 1 (the client-facing mesh alias) exports
    a mesh under a filename WE pin, and stage 2 (the `successor`, e.g. a rigger) is fed
    that mesh + stage-1's threaded params; ONLY stage 2's result is delivered.

    Stage 1 gets the same routing guarantees as a plain generation: candidates are
    re-resolved while parked (fresh health/busy, `force`/LoRA-`eligible` kept), a
    misconfigured candidate is skipped for the next one, and a connection-type error
    fails over. Stage-2/hand-off errors are FINAL — once the mesh is in hand the chain
    never restarts stage 1 elsewhere.

    Stage 1 also QUEUES like a plain generation: the job registers in `_gen_waiting`
    for the whole chain and claims a free backend only when the scheduler designates it
    for it (spec 2026-09-01). The entry is marked `claimed` while a slot is held — the
    hand-off health park included, where stage 1 is already done — so it never shadows
    another waiter out of a backend it is not going to take. It deliberately STAYS in
    the pool while the chain parks on an unhealthy SUCCESSOR before stage 1: nothing is
    held there, so the wait just ages the entry into the overdue rule.

    Two hand-off modes (`relay`):
      • `path` (default) — both stages on the SAME backend (shared disk); stage 2 gets
        the mesh's absolute output path. Backend needs comfy_output_dir. The one slot is
        held across both stages so nothing from the queue runs between them.
      • `upload` — CROSS-backend: the gateway fetches stage 1's mesh (/view), uploads
        it into stage 2's backend input dir and passes its ABSOLUTE PATH there
        (`comfy_input_dir`, else derived from comfy_output_dir's …/input sibling) — the
        successor consumes it exactly like a path hand-off, no special loader needed.
        Only when no input dir is known does the bare stored name go over (a
        load-from-input node can still resolve that). Stage 2 may run on a DIFFERENT
        backend (e.g. mesh on dx10-02, rig on a UniRig box). The stage-1 slot is
        released (and its ComfyUI VRAM freed) once its mesh is in hand, then the
        stage-2 slot is claimed — different backends, so they need not be atomic.
        A cloud stage (Meshy, Tripo; either side) shares no disk with anything, so it
        FORCES this mode regardless of what the alias stored.

    successor config: {alias, export_node, mesh_param, relay?, keep_from_mesh?, rig?}."""
    succ_alias = (succ.get("alias") or "").strip()
    # `export_node` is read by the stage-1 adapter (chain_export), not here.
    mesh_param = (succ.get("mesh_param") or "mesh_path").strip()
    keep_globs = [g.strip() for g in (succ.get("keep_from_mesh") or []) if g.strip()]
    chain_rig = (succ.get("rig") or "").strip() or None
    relay = (succ.get("relay") or "path").strip().lower()      # "path" (shared disk) | "upload" (relay bytes)

    # A cloud stage (Meshy, Tripo) shares no disk with anything: with one on EITHER side
    # the mesh travels as bytes, whatever the stored relay says (the editor hides the
    # field for such aliases).
    def _kind_cloud(alias_name: str) -> bool:
        c = (store.get(alias_name) if store.is_active() else None) or image_models.get(alias_name) or []
        return bool(c) and adapters.cloud_kind(c[0]) is not None
    s1_cloud = await asyncio.to_thread(_kind_cloud, alias)
    s2_cloud = await asyncio.to_thread(_kind_cloud, succ_alias)
    if s1_cloud or s2_cloud:
        relay = "upload"
    # inputs/params come pre-computed from the endpoint (one _gen_inputs_params +
    # _apply_seconds pass, which also raised any 400); the per-attempt _apply_seconds
    # below then re-derives for failover freshness (a no-op once seconds is resolved).
    prefix = f"gwchain_{job_id}"

    async def stage2_for(backend: dict) -> tuple:
        """(successor candidate on `backend`, skip reason) for the path relay. Store
        first, `image_models` config fallback — the same order as _gen_routes."""
        cands = (await asyncio.to_thread(store.get, succ_alias)) if store.is_active() else None
        if cands is None:
            cands = image_models.get(succ_alias, [])
        s2 = next((c for c in cands if c.get("backend") == backend["name"]), None)
        if s2 is None:
            return None, f"successor '{succ_alias}' is not configured for backend '{backend['name']}'"
        return s2, None

    async def usable(backend: dict) -> tuple:
        """(s2, outdir, skip reason) — whether this stage-1 candidate can run the
        chain. s2/outdir are only resolved for the path relay (stage 2 pinned here)."""
        if backend_adapters.get(backend_id(backend)) is None:
            return None, "", f"chain needs an adapter on backend '{backend.get('name')}'"
        if relay != "path":
            return None, "", None
        outdir = (backend.get("comfy_output_dir") or "").rstrip("/")
        if not outdir:
            return None, "", (f"chain (path relay) needs a comfy_output_dir on backend "
                              f"'{backend.get('name')}'")
        s2, why = await stage2_for(backend)
        return s2, outdir, why

    def fail_meta() -> Optional[dict]:
        """Extra meta for a chain `jobs.fail`. A stage-2 failure is exactly when the
        hand-off matters most, so what stage 2 was handed is kept on the failed row too
        (`jobs.fail` merges; `complete`'s _mark_done rewrites, hence the explicit key
        there). `s2_info` is None until the hand-off. `chain_stage1` rides along whenever
        stage 1 was a PAID cloud task: a chain that dies after it must still name the cloud
        task that was billed, which `complete`'s meta would otherwise be the only record of.

        A stage that FAILED produced no GenOutput at all, so `s1_meta` stays None for it and
        the adapter's own trace (`NormalizedRequest.cloud_trace`) is the only record that a
        vendor task ever existed. It goes exactly where the success path puts the same
        facts: stage 1 under `chain_stage1`, a cloud stage 2 at the TOP level."""
        s1 = s1_meta or _cloud_trace_of(req1)
        m = {**_cloud_trace_of(req2),
             **({"attempts": gen_attempts} if gen_attempts > 2 else {}),
             **({"chain_stage2": s2_info} if s2_info else {}),
             **({"chain_stage1": s1} if s1 else {})}
        return m or None

    deadline = time.monotonic() + async_park_timeout_s
    tried: set = set()                       # stage-1 backends that failed with a connection error
    skip_reason = None
    gen_attempts = 0                         # generate() calls across candidates + self-retries
    req1 = req2 = None                       # the live stage requests — `fail_meta` reads their
                                             # cloud_trace, and it may run before either exists
    # In the media queue for the whole chain (see the docstring): registered before
    # the first pass, dropped again on every exit path.
    entry = {"job_id": job_id, "alias": alias, "enqueued_at": time.monotonic(),
             "eligible": eligible, "force": force}
    _gen_waiting.append(entry)
    try:
        while True:
            entry["claimed"] = False             # a new pass = waiting for a stage-1 slot again
            cur = await asyncio.to_thread(jobs.get, job_id)
            if not cur or cur.get("status") not in ("queued", "running"):
                return                           # cancelled (or externally finished) meanwhile
            ready, allc = await asyncio.to_thread(_gen_routes, alias)   # fresh health/busy per attempt
            ready, allc = _force_filter(ready, force), _force_filter(allc, force)
            if eligible is not None:
                ready = [r for r in ready if r[0].get("name") in eligible]
                allc = [r for r in allc if r[0].get("name") in eligible]
            ready = [r for r in ready if r[0].get("name") not in tried]
            allc = [r for r in allc if r[0].get("name") not in tried]
            if not allc:
                await asyncio.to_thread(jobs.fail, job_id,
                                        skip_reason or f"chain: no healthy backend for '{alias}'"
                                        + (f" on '{force}'" if force else ""))
                return

            # pick the first READY candidate that satisfies the chain's per-backend needs
            # and is OURS to claim — a free backend the media queue designates for another
            # waiter is left to it (type affinity), exactly like for a plain parked job.
            picked = None
            # Backends this chain would refuse — the failover set plus everything
            # usable() rejects this pass. Published on the queue entry so the scheduler
            # never designates one of them for us and leaves it idle for the others.
            # EVERY ready candidate is judged, not just up to the pick: we may still
            # end up parking (unhealthy successor, lost busy-race) with the pick in
            # hand, and an unjudged free backend would stay designated for us
            # throughout that wait.
            rejected = set(tried)
            for backend, cand in ready:
                s2p, outdir, why = await usable(backend)
                if why:
                    rejected.add(backend["name"])
                    if picked is None:
                        skip_reason = why        # first-pick reporting, unchanged
                    continue
                if picked is not None:
                    continue                     # picked already — only completing `rejected`
                if not await asyncio.to_thread(_may_claim_gen, entry, backend):
                    continue                     # another waiter is designated for it
                picked = (backend, cand, s2p, outdir)
            entry["exclude"] = set(rejected)
            if picked is None:
                # nothing ready is usable — park while a usable candidate is merely busy
                waitable = False
                for backend, _cand in allc:
                    _s2, _out, why = await usable(backend)
                    if why is None:
                        waitable = True
                        break
                    skip_reason = why
                    rejected.add(backend["name"])
                entry["exclude"] = set(rejected)
                if not waitable:
                    await asyncio.to_thread(jobs.fail, job_id,
                                            skip_reason or f"chain: no usable backend for '{alias}'")
                    return
                if time.monotonic() > deadline:
                    await asyncio.to_thread(jobs.fail, job_id,
                                            f"chain: all usable backends stayed busy past park "
                                            f"time ({async_park_timeout_s:.0f}s)")
                    return
                _gen_wait_ping()                     # stage 1 is waiting → keep the fast probe awake
                await asyncio.sleep(2.0)
                continue

            backend, stage1_cand, s2, outdir = picked
            bid = backend_id(backend)
            adapter = backend_adapters[bid]
            # Resolve the stage-2 (successor) backend + candidate. Path relay pinned it to
            # stage 1's backend above (shared disk). Upload relay picks the successor
            # alias's best candidate — preferring stage 1's backend if it is itself one
            # (keeps the fast in-process path).
            if relay == "upload":
                cands2 = await asyncio.to_thread(get_gen_routes, succ_alias, True)   # successor's allowed+healthy backends
                if not cands2:
                    # No candidate configured at all is a config error — fail fast. A
                    # configured successor whose backend is merely in a transient health
                    # dip parks instead (nothing is held yet): re-enter the stage-1 loop,
                    # which re-resolves everything fresh, until the park deadline.
                    raw2 = (await asyncio.to_thread(store.get, succ_alias)) if store.is_active() else None
                    if raw2 is None:
                        raw2 = image_models.get(succ_alias, [])
                    gen_names = {b["name"] for b in _gen_backends}
                    if not any(c.get("backend") in gen_names for c in raw2):
                        # unconfigured, or every candidate backend is disabled/gone — an
                        # admin state, not a transient: parking would never recover it.
                        await asyncio.to_thread(jobs.fail, job_id, f"successor '{succ_alias}' has no "
                                                "enabled candidate backend for the upload relay")
                        return
                    if time.monotonic() > deadline:
                        await asyncio.to_thread(jobs.fail, job_id, f"successor '{succ_alias}' had no "
                                                "enabled+healthy candidate backend for the upload relay "
                                                f"within park time ({async_park_timeout_s:.0f}s)")
                        return
                    # A successor wait must not hold stage-1 designations: while we sit
                    # here we claim nothing, so release every stage-1 backend of this pass
                    # (rebuilt from scratch next pass → self-heals once the successor is back).
                    entry["exclude"] = {b["name"] for b, _ in ready} | rejected
                    _gen_wait_ping()             # keep the fast probe awake → quick recovery pickup
                    await asyncio.sleep(2.0)
                    continue
                backend2, s2 = next((bc for bc in cands2 if backend_id(bc[0]) == bid), None) or cands2[0]
            else:
                backend2 = backend
            bid2 = backend_id(backend2)
            adapter2 = backend_adapters.get(bid2)
            if adapter2 is None:
                await asyncio.to_thread(jobs.fail, job_id, f"successor backend "
                                        f"'{backend2.get('name')}' has no adapter")
                return
            why = _chain_mesh_param_error(s2, mesh_param, succ_alias)
            if why:
                await asyncio.to_thread(jobs.fail, job_id, why)
                return
            s1_wf = stage1_cand.get("workflow_json") or {}
            # Stage-1 export is backend-specific (ComfyUI pins an export node; a cloud
            # backend delivers the mesh as a blob) — the adapter decides, and a candidate
            # that cannot export as configured is refused HERE, before GPU-minutes/credits.
            export = adapter.chain_export(stage1_cand, succ, params, prefix)
            if export.error:
                await asyncio.to_thread(jobs.fail, job_id, export.error)
                return
            # A cloud successor (Meshy, Tripo) takes ONE mesh format: the API's mesh url
            # is a glb. Refused here — before the slot claim and the GPU minutes — because
            # the mismatch is in the stage-1 export node's `file_format`, knowable up
            # front; discovering it from the cloud's rejection would cost a full stage-1 run.
            s2_kind = adapters.cloud_kind(s2)
            if s2_kind and not export.mesh_name.lower().endswith(".glb"):
                await asyncio.to_thread(
                    jobs.fail, job_id,
                    f"chain: successor '{succ_alias}' runs on "
                    f"{adapters.cloud_module(s2_kind).VENDOR} and takes a .glb mesh, "
                    f"but stage 1 would export '{export.mesh_name}' — set the export node's "
                    f"file_format to glb")
                return
            mesh_name = export.mesh_name
            cross = relay == "upload" and bid2 != bid

            # Claim the stage-1 slot: the busy-check and inc run with no await between them
            # (the dispatch invariant); the awaits above may have let someone else claim it.
            if backend_busy(backend):
                await asyncio.sleep(0.5)
                continue
            _inflight_inc(bid)
            entry["claimed"] = True              # holding a slot → out of the waiting pool
            # `held` tracks which slot we owe a decrement, so the per-attempt finally never
            # over/under-counts across the hand-off. `active` = backend to free VRAM on.
            held = bid
            active = backend
            s1_done = False                             # mesh in hand → stage-2 errors are final
            s1_meta = None                              # a paid stage-1 cloud task, for the job view
            s2_info = None                              # what stage 2 was actually handed (job view)
            req1 = req2 = None                          # per CANDIDATE: a later stage-1 failure must
                                                        # not report the previous pass's tasks
            await asyncio.to_thread(jobs.set_status, job_id, "running")
            await asyncio.to_thread(jobs.set_backend, job_id, backend["name"])   # cancel targets the LIVE backend
            await asyncio.to_thread(jobs.set_stage, job_id, "1/2")   # multi-stage progress → "running 1/2"
            try:
                _apply_seconds(params, stage1_cand)     # no-op re-derive (endpoint validated it)
                # ── Stage 1: mesh (pin the export filename; ignore its own outputs) ──
                req1 = NormalizedRequest(
                    alias=alias, real_model=stage1_cand.get("model"),
                    task=stage1_cand.get("task", "text2img"), inputs=inputs, params=params, output={},
                    workflow=stage1_cand.get("workflow"), workflow_json=s1_wf,
                    node_mapping=stage1_cand.get("mapping") or {},
                    fixed=list(stage1_cand.get("fixed") or []) + list(export.extra_fixed),
                    cloud=adapters.cloud_block(stage1_cand),
                    bypass=(stage1_cand.get("bypass") or []),
                    upload_images=dict(upload_images or {}), raw=request,
                    upload_files=dict(upload_files or {}),
                    upload_prefix=_upload_prefix(job_id, "s1"), job_id=job_id,
                    loras=body.get("loras"), slot_held=True)
                # type affinity + free VRAM when the model set changes (keyed on what
                # req1 loads, hence after it is built)
                await _claim_gen_backend(backend, alias, _model_set_key(adapter, req1))
                await _unload_host_llms(backend)        # opt-in host policy, no-op by default
                # runbook B: retry a sporadic fault on the SAME backend first — the held
                # slot (`held`) spans the repeats; the last attempt re-raises into the
                # existing stage-1 failover (next candidate via `tried`).
                s1_tries = 1 + max(0, int(backend.get("self_retries") or 0))
                for s1_attempt in range(1, s1_tries + 1):
                    gen_attempts += 1
                    try:
                        t0 = time.monotonic()
                        out1 = await adapter.generate(req1)
                        _record_gen_attempt(bid, conn_fail=False)
                        _note_gen_speed(alias, bid, time.monotonic() - t0)
                        break
                    except _GEN_FAILOVER_ERRORS as e:
                        if s1_attempt >= s1_tries:
                            raise                # outer handler records + fails over
                        _record_gen_attempt(bid, conn_fail=True)
                        logger.warning(f"✗ chain job {job_id} stage 1 [{backend['name']}] "
                                       f"{_fault_label(e)} ({type(e).__name__}: {e}) — retrying "
                                       f"same backend (self-retry {s1_attempt}/{s1_tries - 1})")
                        await _wait_backend_up(backend)
                # keep_from_mesh: files the successor can't make itself (e.g. the basecolor
                # PNG — the mesh/texturing stage bakes it; the UniRig fbx only references its
                # texture) travel from stage 1 into the final delivery.
                mesh_extras = [b for b in (out1.blobs or [])
                               if keep_globs and any(fnmatch.fnmatch((b.name or "").lower(), g.lower())
                                                     for g in keep_globs)]
                # The path relay only needs the mesh to EXIST on the shared disk (stage 2
                # reads it by absolute path); only the upload relay needs the bytes. The
                # adapter takes it (ComfyUI: existence-only = cheap 1-byte Range GET on
                # /view; bytes otherwise), keeping backend conventions out of the router.
                need_bytes = relay == "upload"
                mesh = await adapter.chain_take_mesh(out1, export, need_bytes)
                if mesh is None:
                    raise RuntimeError(f"stage-1 produced no mesh at '{mesh_name}' — "
                                       "check the export node / file_format")
                mesh_bytes = mesh if need_bytes else None
                s1_done = True
                # A cloud stage 1 is a PAID task: its kind and task id (what the vendor's
                # own dashboard is searched by), endpoint, request, sub-tasks and credits
                # are knowable only from THIS run — and stage 2's meta would overwrite
                # every one of those keys in the merge below, so they are kept under their
                # own key. `meshy_task_id` rides along beside the neutral `cloud_task_id`:
                # existing job rows and the job view still read the Meshy-era name.
                s1_meta = ({k: out1.meta.get(k) for k in
                            ("backend", "cloud", "cloud_task_id", "meshy_task_id", "endpoint",
                             "consumed_credits", "request", "tasks")
                            if out1.meta.get(k) is not None}
                           if (out1.meta.get("cloud_task_id") or out1.meta.get("meshy_task_id"))
                           else None)

                # Stage 2's request is built BEFORE the hand-off: the feed is the
                # stage-2 adapter's business and may have to put the mesh ON the request
                # (a cloud backend uploads/embeds it) instead of somewhere it can name by
                # path. `params` is filled in after the feed, once the mesh_ref is known.
                req2 = NormalizedRequest(
                    alias=succ_alias, real_model=s2.get("model"),
                    task=s2.get("task", "text2img"), inputs={}, params={}, output={},
                    workflow=s2.get("workflow"), workflow_json=s2.get("workflow_json"),
                    node_mapping=s2.get("mapping") or {}, fixed=s2.get("fixed") or [], upload_images={},
                    upload_files={},
                    upload_prefix=_upload_prefix(job_id, "s2"), job_id=job_id,
                    raw=request, output_node=(s2.get("output_node") or None),
                    output_ext=(s2.get("output_ext") or None), output_globs=(s2.get("output_globs") or None),
                    output_cases=(s2.get("output_cases") or None),
                    texture_format=(s2.get("texture_format") or None),
                    dummy_check=(s2.get("dummy_check") is not False),
                    cloud=adapters.cloud_block(s2),
                    bypass=(s2.get("bypass") or []), slot_held=True)

                # ── Hand-off: give stage 2 either a shared-disk path or an uploaded input name ──
                if cross:
                    # cross-backend: stage 1 is done — release its slot AND free its VRAM (it
                    # stops being `active`, so the end-of-chain free would never reach it),
                    # then claim stage 2's and upload the mesh into its input dir (released
                    # first, so no A-holds-waits-B cycle).
                    _inflight_dec(held); held = None
                    asyncio.create_task(_free_comfy_vram(backend, "chain stage 1 done"))
                    # Stage 1 ran for minutes — the successor backend picked up front may be
                    # in a transient health dip right now. The mesh bytes are in hand and no
                    # slot is held, so waiting is free: park until it is healthy again (or
                    # the park timeout passes) instead of losing the finished mesh to a
                    # connection error on the upload.
                    h_deadline = time.monotonic() + async_park_timeout_s
                    while not backend_healthy.get(bid2):
                        cur = await asyncio.to_thread(jobs.get, job_id)
                        if not cur or cur.get("status") not in ("queued", "running"):
                            return               # cancelled meanwhile
                        if time.monotonic() > h_deadline:
                            await asyncio.to_thread(jobs.fail, job_id,
                                                    f"chain: successor backend '{backend2['name']}' "
                                                    f"stayed unhealthy past park time for the mesh "
                                                    f"hand-off ({async_park_timeout_s:.0f}s)")
                            return
                        _gen_wait_ping()         # keep the fast probe awake → quick recovery pickup
                        await asyncio.sleep(2.0)
                    if not await _wait_and_hold(backend2, job_id, "stage 2"):
                        return
                    held = bid2
                    active = backend2
                    await asyncio.to_thread(jobs.set_backend, job_id, backend2["name"])
                    mesh_ref = await adapter2.chain_feed_mesh(req2, backend2, mesh_param,
                                                             mesh_name, mesh_bytes, outdir)
                    if log_per_call:
                        logger.info(f"chain job {job_id}: relayed mesh {mesh_name} "
                                    f"[{backend['name']}]→[{backend2['name']}] as '{mesh_ref}'")
                else:
                    # same backend: the relayed bytes go through stage 2's own input path
                    # (`upload`), else the mesh already lies on the shared disk (`path`).
                    mesh_ref = await adapter2.chain_feed_mesh(req2, backend2, mesh_param,
                                                             mesh_name, mesh_bytes, outdir)

                # ── Stage 2: successor, fed the mesh + stage-1 params (name, no_fingers, …) ──
                # Thread stage-1 params to the successor keyed by their mapping LABEL, never by
                # the raw/node-based field name. A client field like `value` (or `value_307`) is
                # tied to a specific node id that changes when the workflow is rebuilt, and a
                # generic `value` from stage 1 would otherwise collide with the successor's own
                # `value` param — clobbering the mesh path (seen: face-num 100000 landed on the
                # mesh-load node). Labels are the stable, unique public names.
                s1_map = stage1_cand.get("mapping") or {}
                label_of = {p: (((m or {}).get("label") or "").strip() or p) for p, m in s1_map.items()}
                s2_params = {label_of.get(k, k): v for k, v in params.items()}
                s2_params[mesh_param] = mesh_ref
                # What stage 2 was HANDED, recorded for the job view — the threading is
                # label-keyed and a successor silently ignores what it doesn't map, so
                # "did my param reach the rigger?" is otherwise only answerable from the
                # backend's own ComfyUI history. Recorded, never re-derived from config:
                # the alias mapping may have changed since the run. `applied` (what the
                # successor actually mapped) is filled in from out2 below.
                s2_info = {"alias": succ_alias, "backend": backend2["name"], "relay": relay,
                           "mesh_param": mesh_param, "mesh_ref": mesh_ref, "params": s2_params}
                # This REPLACES req2.params, so a stage-2 feed hook must put the mesh on
                # `req2.upload_files` (or the request body) — never into `req2.params`.
                req2.params = s2_params                  # built before the hand-off (see above)
                await asyncio.to_thread(jobs.set_stage, job_id, "2/2")   # → "running 2/2"
                # Stage 2 is its own type key, and on the SAME backend it is also the
                # moment stage 1's model set has to go: the two stages of a chain are
                # two different workflows, and the second one (a rigger) may not even
                # run in ComfyUI's process — measured on Meshy→mesh-mia, where stage 2
                # OOM'd on the 21 GiB stage 1 left behind (job cc604da29e0e).
                await _claim_gen_backend(backend2, succ_alias, _model_set_key(adapter2, req2))
                await _unload_host_llms(backend2)
                gen_attempts += 1
                t2 = time.monotonic()
                out2 = await adapter2.generate(req2)
                _record_gen_attempt(bid2, conn_fail=False)
                _note_gen_speed(succ_alias, bid2, time.monotonic() - t2)
                blobs = list(out2.blobs) + mesh_extras       # successor result + kept mesh-stage files
                # out2.meta["applied"] is STAGE 2's applied set (the merge below makes it the
                # row's top-level one); copied in here so the job view can mark each handed
                # param as applied-or-dropped without guessing which stage it came from.
                s2_info["applied"] = list(out2.meta.get("applied") or [])
                meta = {**out2.meta, "backend": backend2["name"], "chain": [alias, succ_alias],
                        "chain_stage2": s2_info,
                        **({"chain_stage1": s1_meta} if s1_meta else {})}
                if gen_attempts > 2:             # a clean chain is exactly 2 generate() calls
                    meta["attempts"] = gen_attempts
                if cross:
                    meta["chain_backends"] = [backend["name"], backend2["name"]]
                if chain_rig:
                    meta["rig"] = chain_rig              # the client-facing rig tag, whatever its kind
                    # `generic`/`mixamo` name deliveries the GATEWAY shapes and checks:
                    # V-flip (+ optional jpeg) textures — normalize-once flagged; the knob
                    # lives on the CLIENT-FACING (stage-1) alias, covering kept stage-1 files
                    # too — then validate the COMBINED delivery at chain level. `meshy`/`tripo`
                    # are rigs the cloud built to its own conventions: tag them, never re-flip
                    # them or fail them against ComfyUI-shaped rules.
                    if chain_rig in ("generic", "mixamo"):
                        normalize_delivery(blobs, chain_rig, stage1_cand.get("texture_format"))
                        warnings = validate_delivery(blobs, chain_rig)   # raises → job fails clearly
                        if warnings:
                            meta["warnings"] = warnings
                await asyncio.to_thread(jobs.complete, job_id, blobs, meta)
                if log_per_call:
                    route = (f"{backend['name']}→{backend2['name']}" if cross else backend["name"])
                    logger.info(f"✓ chain job {job_id} on [{route}] ({alias}→{succ_alias}) "
                                f"— {len(blobs)} artifact(s)")
                asyncio.create_task(_free_comfy_vram(active, "chain done"))
                return
            except _GEN_FAILOVER_ERRORS as e:
                _record_gen_attempt(backend_id(active), conn_fail=True)
                if s1_done:                              # mesh already relayed — a stage-2 loss is final
                    # `_err_text`: this is the branch an httpx WriteTimeout lands in, and
                    # its str() is empty — the row used to read "chain failed: " (2026-09-02).
                    logger.warning(f"✗ chain job {job_id} [{active['name']}] ({alias}→{succ_alias}) "
                                   f"{_fault_label(e)}: {_err_text(e)}")
                    await asyncio.to_thread(jobs.fail, job_id, f"chain failed: {_err_text(e)}",
                                            fail_meta())
                    asyncio.create_task(_free_comfy_vram(active, "chain failure"))
                    return
                logger.warning(f"✗ chain job {job_id} stage 1 [{backend['name']}] {_fault_label(e)} "
                               f"({type(e).__name__}: {e}) — failing over")
                tried.add(backend["name"])
                asyncio.create_task(_free_comfy_vram(backend, "chain stage-1 failure"))
                continue
            except Exception as e:
                logger.warning(f"✗ chain job {job_id} [{active['name']}] ({alias}→{succ_alias}) "
                               f"failed: {_err_text(e)}")
                await asyncio.to_thread(jobs.fail, job_id, f"chain failed: {_err_text(e)}", fail_meta())
                asyncio.create_task(_free_comfy_vram(active, "chain failure"))
                return
            finally:
                if held is not None:
                    _inflight_dec(held)
    finally:
        try:
            _gen_waiting.remove(entry)
        except ValueError:
            pass


_gen_tasks: dict = {}                       # job_id → asyncio.Task (for cancellation)


def _spawn_gen(job_id: str, coro) -> None:
    """Run a generation coroutine as a tracked background task so it can be cancelled."""
    t = asyncio.create_task(coro)
    _gen_tasks[job_id] = t
    t.add_done_callback(lambda _: _gen_tasks.pop(job_id, None))


async def cancel_generation(job_id: str) -> bool:
    """Cancel a queued/running generation job: best-effort interrupt on the backend
    (adapter.cancel — ComfyUI /interrupt frees the GPU; a cloud task API has nothing
    to stop, the vendor finishes and bills the task), cancel the worker task, mark the job
    failed. Returns False if the job is already finished/unknown."""
    job = await asyncio.to_thread(jobs.get, job_id)
    if not job or job.get("status") not in ("queued", "running"):
        return False
    # The job row names the backend by NAME, which is unique only per TYPE — so the
    # alias's candidate for that name decides the kind, or an /interrupt would hit an
    # unrelated same-named ComfyUI. A row whose alias is gone (legacy/deleted) falls
    # back to the bare name match: cancelling the worker task still beats not cancelling.
    job_alias = job.get("alias") or ""
    cands = (await asyncio.to_thread(store.get, job_alias)
             if store.is_active() else None) or image_models.get(job_alias) or []
    cand = next((c for c in cands if c.get("backend") == job.get("backend")), None)
    pool = [x for x in backends if _is_gen(x)]       # incl. disabled: cancel must still reach it
    b = (_gen_backend_for(job.get("backend"), cand, pool) if cand is not None
         else next((x for x in pool if x.get("name") == job.get("backend")), None))
    adapter = backend_adapters.get(backend_id(b)) if b else None
    if adapter is not None:
        await adapter.cancel()
    t = _gen_tasks.get(job_id)
    if t and not t.done():
        t.cancel()
    await asyncio.to_thread(jobs.fail, job_id, "cancelled by user")
    if b:
        asyncio.create_task(_free_comfy_vram(b, "cancel"))     # no-op for non-ComfyUI (step 3)
    return True


async def _run_gen_parked(job_id, alias, force, build_req, eligible: Optional[set] = None):
    """Hold a generation job until a backend frees (polls backend-busy), then run it —
    so a busy backend queues instead of 503'ing (async/playground). `eligible` keeps
    the LoRA constraint through the park: the job waits for a LoRA-capable backend
    instead of spilling to whichever frees first.

    A poll that finds ZERO candidates is NOT an immediate fail: the job only parked
    because it HAD candidates (all busy), so an empty poll is a transient health flap
    (a busy ComfyUI drops its /object_info discovery poll mid-generation and is briefly
    marked DOWN). Ride it out for `park_health_grace_s`; only if EVERY candidate stays
    gone that long is the alias treated as having no healthy backend."""
    deadline = time.monotonic() + async_park_timeout_s
    unhealthy_since = None                       # when the candidate set first went empty (flap timer)
    # In the media queue for the whole wait, so a backend that frees can be handed to
    # THIS job when the scheduler designates it (same alias / overdue) — see
    # _designated_gen_index. Removed again on every exit path (finally).
    entry = {"job_id": job_id, "alias": alias, "enqueued_at": time.monotonic(),
             "eligible": eligible, "force": force}
    _gen_waiting.append(entry)
    try:
        while True:
            _gen_wait_ping()                     # this job is waiting → keep the fast probe awake
            ready, allc = await asyncio.to_thread(_gen_routes, alias)   # fresh health/busy per poll
            ready, allc = _force_filter(ready, force), _force_filter(allc, force)
            if eligible is not None:
                ready = [r for r in ready if r[0].get("name") in eligible]
                allc = [r for r in allc if r[0].get("name") in eligible]
            # A free candidate is only ours if the scheduler designates it for us;
            # otherwise it belongs to another waiter and we keep parking (2 s poll).
            i = (await asyncio.to_thread(_designated_gen_index, entry, ready)) if ready else None
            if i is not None:
                entry["claimed"] = True          # holding a slot → out of the waiting pool
                # designated candidate first, the rest stay as the failover tail
                await _run_job(job_id, alias, [ready[i]] + ready[:i] + ready[i + 1:], build_req)
                return
            now = time.monotonic()
            if allc:                             # candidates exist but are busy → keep parking
                unhealthy_since = None
            else:                                # all candidates transiently unhealthy → grace, not fail
                unhealthy_since = unhealthy_since or now
                if now - unhealthy_since > park_health_grace_s:
                    await asyncio.to_thread(jobs.fail, job_id,
                                            f"no healthy backend for '{alias}' (all candidates down for "
                                            f">{park_health_grace_s:.0f}s)" + (f" on '{force}'" if force else ""))
                    return
            if now > deadline:
                await asyncio.to_thread(jobs.fail, job_id,
                                        f"park timeout: backend busy for >{async_park_timeout_s:.0f}s")
                return
            await asyncio.sleep(2.0)
    finally:
        try:
            _gen_waiting.remove(entry)
        except ValueError:
            pass


def _requested_loras(body: dict) -> set:
    """LoRA filenames a request asks for — used for LoRA-aware backend preference.
    Sources: lora_* params (top-level or in `params`; also mapped names like
    lora_02_172 and label aliases like lora1_high) and the `loras:[{name,…}]`
    array incl. the high/low counterparts it will resolve to."""
    out = set()
    merged = {**body, **(body.get("params") or {})}
    for k, v in merged.items():
        if isinstance(v, str) and v and v != "None" and re.match(r"^lora[_\d]", str(k)):
            out.add(v)
    for e in (body.get("loras") or []):
        n = str((e.get("name") if isinstance(e, dict) else e) or "").strip()
        if n and n != "None":
            out.add(n)
            cp = lora_counterpart(n)
            if cp:
                out.add(cp)                    # eligibility needs both pair halves
    return out


def _lora_eligible_names(all_cands: list, body: dict) -> Optional[set]:
    """LoRA-aware backend eligibility: backends lacking a requested LoRA are dropped —
    but only for LoRAs installed on SOME candidate; a LoRA installed nowhere is ignored
    so the normal ordering still decides (per spec). Decided over ALL candidates (incl. busy), so
    the eligible backend is parked-for rather than spilling to a backend without the
    LoRA. None = no constraint."""
    req_loras = _requested_loras(body)
    if not req_loras or not all_cands:
        return None
    avail = set().union(*(backend_loras.get(backend_id(b), set()) for b, _ in all_cands))
    need = req_loras & avail
    if not need:
        return None
    elig = [r for r in all_cands if need <= backend_loras.get(backend_id(r[0]), set())]
    if not elig:                                          # loras split across backends → no constraint
        return None
    return {r[0].get("name") for r in elig}


async def _gen_pick(alias: str, force: str, body: dict) -> tuple[list, bool, Optional[set]]:
    """Resolve a generation request's candidates ONCE (a single store read):
    force-pin filter, LoRA eligibility, ready/busy split. Returns
    (routes, parked, eligible_names); raises 503 when nothing is eligible."""
    ready, allc = await asyncio.to_thread(_gen_routes, alias)
    ready, allc = _force_filter(ready, force), _force_filter(allc, force)
    if not allc:
        raise HTTPException(503, f"No healthy backend for generation model '{alias}'"
                                 + (f" on backend '{force}'" if force else ""))
    eligible = None if force else _lora_eligible_names(allc, body)   # a pin is never overridden
    if eligible is not None:
        ready = [r for r in ready if r[0].get("name") in eligible]
        allc = [r for r in allc if r[0].get("name") in eligible]
    # Spec rule 4, "immer in die Queue": a free backend that a WAITING job is designated
    # for is not up for grabs — this request parks instead and competes from inside the
    # media queue, so a fresh arrival never overtakes the jobs it just queued behind.
    if ready and not await asyncio.to_thread(_gen_reserved, ready[0][0]):
        return ready, False, eligible                    # free and unclaimed → dispatch now
    return allc, True, eligible                  # busy/reserved → park (async: queue, sync: block)


def _upload_prefix(job_id: str, stage: str = "") -> str:
    """The job-unique namespace every input file of this job is uploaded under
    (`gw_<job id>[_<stage>]_<param>.<ext>`, built by adapters.upload_slot_name).

    Job-unique is NOT an optimisation, it is the correctness contract: ComfyUI opens
    an input file when the prompt EXECUTES, not when it is submitted, so any name two
    jobs can both write is a corruption window — and the gateway's one-slot cap does
    not close it (a poll timeout releases the slot while ComfyUI keeps running the
    prompt). Measured 2026-08: a client job was delivered another subject's mesh."""
    return f"gw_{job_id}{('_' + stage) if stage else ''}"


async def run_generation(body: dict, request: Request,
                         upload_images: Optional[dict] = None,
                         upload_files: Optional[dict] = None) -> dict:
    """Resolve a generation alias, create a job, and run it (sync) or schedule it
    (async). Returns a job view (sync) or `{job_id, status:"queued"}` (async).
    Fails over across candidate backends on connection errors (e.g. a crashed
    ComfyUI). Shared by the HTTP endpoint and the UI playground.

    `upload_files` ({param: (slot name, bytes)}, decoded once by the endpoint) is
    handed to whichever backend ends up dispatching — the adapter uploads it there,
    so parking and failover need no special casing."""
    alias = body.get("model", "")
    output = dict(body.get("output") or {})
    mode = output.get("mode") or body.get("mode") or "sync"
    ttl_s = output.get("ttl_s") or body.get("ttl_s")
    force = (body.get("backend") or "").strip()          # pin to one backend (playground testing)

    routes, parked, eligible = await _gen_pick(alias, force, body)
    inputs, params = _gen_inputs_params(body)
    _apply_seconds(params, routes[0][1])         # seconds → frames (alias fps; 400 if unsupported)

    def build_req(backend: dict, cand: dict) -> NormalizedRequest:
        # `job_id` below is bound by the time this runs (dispatch happens after the job
        # row exists) — every input upload is namespaced by it.
        return NormalizedRequest(
            alias=alias, real_model=cand.get("model"),
            task=cand.get("task", body.get("task", "text2img")),
            inputs=inputs, params=params, output=output,
            workflow=cand.get("workflow"), workflow_json=cand.get("workflow_json"),
            node_mapping=cand.get("mapping") or {}, fixed=cand.get("fixed") or [],
            upload_images=dict(upload_images or {}), raw=request,
            upload_files=dict(upload_files or {}), upload_prefix=_upload_prefix(job_id),
            job_id=job_id,                                        # keys the live progress feed
            loras=body.get("loras"), output_node=(cand.get("output_node") or None),
            output_ext=(cand.get("output_ext") or None),
            output_globs=(cand.get("output_globs") or None),
            output_cases=(cand.get("output_cases") or None),
            texture_format=(cand.get("texture_format") or None),
            dummy_check=(cand.get("dummy_check") is not False),   # default on; alias opt-out
            bypass=(cand.get("bypass") or []),                    # per-backend node bypass
            cloud=adapters.cloud_block(cand),                     # cloud candidate block (None on ComfyUI)
        )

    first, cand0 = routes[0]
    task = cand0.get("task", body.get("task", "text2img"))
    owner = _request_owner(request)
    job_id = await asyncio.to_thread(jobs.create, task, alias, first["name"], owner=owner, ttl_s=ttl_s)
    # From here on the outcome is recorded in the JOB, so the refusal handler must not
    # write a call-log row for it (see _record_rejected). Without this, a failed media
    # job surfaced as HTTPException 502 by gen_done_or_502 was logged a second time as
    # "(refused), 0 ms, no backend" — a request that was in fact served and failed.
    # Refusals BEFORE this point (no eligible backend, quota, missing prompt) still get
    # their row: no job exists there, so it would otherwise leave no trace at all.
    request.state.gw_dispatched = True
    # persist the request inputs so the job stays inspectable in the UI within its TTL.
    # Uploaded FILES are only noted, never stored: a 40 MB mesh per job would flood the
    # disk, and the job's value is knowing which file went in, not keeping it.
    ref_blobs = [(slot, data) for slot, data in (upload_images or {}).items() if data]
    shown = {**params, **{p: f"<upload:{nm} ({len(d) / (1024 * 1024):.1f} MB)>"
                          for p, (nm, d) in (upload_files or {}).items()}}
    await asyncio.to_thread(jobs.set_inputs, job_id,
                            {"prompt": inputs.get("prompt", ""),
                             "negative_prompt": inputs.get("negative_prompt", ""),
                             "params": shown}, ref_blobs)
    if log_per_call:
        cands = ", ".join(b["name"] for b, _ in routes)
        logger.info(f"→ generation '{alias}' ({task}) job {job_id} mode={mode}"
                    f"{' PARKED' if parked else ''} candidates=[{cands}]")

    # Workflow chain: stage 1 has a `successor` → run both stages back-to-back,
    # deliver only stage 2. The chain resolves/parks/fails over its stage-1 backend
    # itself (keeping the force pin + LoRA eligibility), so parked/ready both route here.
    succ = cand0.get("successor")
    if succ and (succ.get("alias") or "").strip():
        runner = _run_chain(job_id, alias, succ, body, request, upload_images, upload_files,
                            inputs, params, force, eligible)
        if mode == "async":
            _spawn_gen(job_id, runner)
            return {"job_id": job_id, "status": "queued"}
        await runner
        return await _job_view(job_id, request)

    if parked:
        if mode == "async":                              # queue and hand back a job id
            _spawn_gen(job_id, _run_gen_parked(job_id, alias, force, build_req, eligible))
            return {"job_id": job_id, "status": "queued"}
        await _run_gen_parked(job_id, alias, force, build_req, eligible)   # sync: block through the park
        return await _job_view(job_id, request)
    if mode == "async":
        _spawn_gen(job_id, _run_job(job_id, alias, routes, build_req))
        return {"job_id": job_id, "status": "queued"}

    await _run_job(job_id, alias, routes, build_req)      # sync: block until done/failed
    return await _job_view(job_id, request)


_UPLOAD_MAX_BYTES = 64 * 1024 * 1024        # per-file cap for `files` (413) — a mesh is big,
                                            # but a request body is not a stream


def _gen_alias_mapping(alias: str) -> tuple[dict, dict]:
    """A generation alias's (workflow, mapping) — from its first candidate, since both
    are backend-independent. Includes busy backends: resolving a request field must not
    depend on which backend happens to be free."""
    routes = get_gen_routes(alias, include_busy=True)
    if not routes:
        return {}, {}
    _, cand = routes[0]
    return (cand.get("workflow_json") or {}), (cand.get("mapping") or {})


def _file_param(wf: dict, mapping: dict, key: str) -> str:
    """Which mapping param a `files` key addresses — param name or public label, the
    same two names every request field accepts. Image slots are rejected on purpose:
    they carry their own upload path (`images`) with placeholders and empty-modes."""
    hit = next((p for p in mapping if p == key), None) \
        or next((p for p, m in mapping.items()
                 if ((m or {}).get("label") or "").strip() == key), None)
    if not hit:
        raise HTTPException(400, f"unknown `files` key '{key}' — not a parameter of this "
                                 f"generation alias (see GET /v1/generations/<alias>/schema)")
    if is_image_field(wf, (mapping.get(hit) or {}).get("node")):
        raise HTTPException(400, f"`files` key '{key}' addresses an image slot — "
                                 f"send images via `images`")
    return hit


async def _decode_one_file(key: str, param: str, val) -> tuple:
    """One `files` entry → (slot name, bytes). The name is a HINT (display + extension);
    a ComfyUI upload is renamed under the job's prefix, a cloud one is embedded as a
    data-URI — neither ever writes the raw client name."""
    got = await _decode_ref_blob(val)
    if not got or not got[0]:
        raise HTTPException(400, f"`files.{key}` could not be read — expected base64, "
                                 f"a data-URI or an http(s) URL")
    data, ext = got
    if len(data) > _UPLOAD_MAX_BYTES:
        raise HTTPException(413, f"`files.{key}` is {len(data) / (1024 * 1024):.1f} MB — "
                                 f"the limit is {_UPLOAD_MAX_BYTES // (1024 * 1024)} MB")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", param)[:40] or "file"
    return f"gwup_{slug}.{ext}", data


async def _decode_upload_files(alias: str, files: dict) -> dict:
    """`files: {param|label: base64|data-URI|URL}` → {param: (slot name, bytes)}.

    Deliberately STRICT where `params` are lenient (unknown names are ignored there):
    a dropped file would not degrade the job, it would run the workflow against its
    baked-in default and hand back a confidently wrong result."""
    wf, mapping = await asyncio.to_thread(_gen_alias_mapping, alias)
    # Same lookup as every other alias read (_gen_routes, gen_alias_schema): store first,
    # config `image_models` for aliases the store doesn't hold — a config-defined cloud
    # alias must hit the branch below too, not fall through and drop the files silently.
    cands = ((await asyncio.to_thread(store.get, alias)) if store.is_active() else None) \
        or image_models.get(alias)
    k = adapters.cloud_kind(cands[0]) if cands else None
    if k:
        # A cloud alias (Meshy, Tripo) has no mapping — its file inputs are the endpoint's
        # fixed table (the rig endpoint: input_mesh_path; the image ones: none), so the
        # keys are checked against public_fields instead, and are their OWN param names.
        vendor = adapters.cloud_module(k).VENDOR
        allowed = {f["name"] for f in adapters.public_fields(cands[0])[2]}
        if not allowed:
            raise HTTPException(400, f"generation alias '{alias}' runs on {vendor} and accepts"
                                     f" no `files` — send images under `images`")
        out = {}
        for key, val in files.items():
            if key not in allowed:
                raise HTTPException(400, f"unknown `files` key '{key}' — this alias takes "
                                         f"{', '.join(sorted(allowed))} (see GET "
                                         f"/v1/generations/<alias>/schema)")
            out[key] = await _decode_one_file(key, key, val)
        return out
    if not mapping:
        return {}                       # no candidate at all → run_generation 503s in a moment
    out = {}
    for key, val in files.items():
        param = _file_param(wf, mapping, key)
        out[param] = await _decode_one_file(key, param, val)
    return out


@app.post("/v1/generations")
async def generations(request: Request, authorization: Optional[str] = Header(None)):
    body = await request.json()
    await gate_request(authorization, request, body.get("model"))    # auth + allow-list + quota
    # Optional per-field reference images: {"images": {<image-param>: <base64|data-URI|URL>}}
    # — the native counterpart of the playground's per-field uploads and the OpenAI
    # shims' positional ref_images.
    uploads = None
    imgs = body.pop("images", None)
    if isinstance(imgs, dict):
        uploads = {}
        for param, val in imgs.items():
            data = await _decode_ref_image(val)
            if data:
                uploads[param] = data
    # Optional client files for NON-image params: {"files": {<param>: <base64|data-URI|URL>}}
    # — e.g. the mesh a shrink/rig alias works on. The gateway uploads it onto whichever
    # backend runs the job, so a client never needs a path on a backend.
    files = body.pop("files", None)
    if files is not None and not isinstance(files, dict):
        raise HTTPException(400, "`files` must be an object of {param: base64|data-URI|URL}")
    upload_files = await _decode_upload_files(body.get("model", ""), files) if files else None
    view = await run_generation(body, request, upload_images=uploads, upload_files=upload_files)
    code = {"queued": 202, "done": 200}.get(view.get("status"), 502)
    return JSONResponse(view, status_code=code)


@app.get("/v1/generations/{alias}/loras")
async def gen_alias_loras(alias: str, request: Request, authorization: Optional[str] = Header(None)):
    """LoRA filenames valid for a generation alias — the union of what's installed on
    the alias's backends. Lets a client present a valid LoRA picker per alias."""
    await gate_request(authorization, request, alias)                # auth + allow-list
    known = (await asyncio.to_thread(store.get, alias)) if store.is_active() else None
    if not (known or image_models.get(alias)):
        raise HTTPException(404, f"generation alias '{alias}' not found")
    loras: set = set()
    for b, _ in await asyncio.to_thread(get_gen_routes, alias, include_busy=True):
        loras |= backend_loras.get(backend_id(b), set())
    return {"object": "list", "alias": alias, "loras": sorted(loras)}


@app.get("/v1/generations/{alias}/schema")
async def gen_alias_schema(alias: str, request: Request, authorization: Optional[str] = Header(None)):
    """Self-description of a generation alias — enough for a client (or an agent)
    to build a valid request without out-of-band docs: params under their EXTERNAL
    names (label, else param; both are accepted on requests) with type + default
    from the workflow, image slots with their empty behaviour, `files` (uploads that
    are not images — a mesh a rig/shrink alias works on), fps/frames raster, and
    where to list valid LoRAs."""
    await gate_request(authorization, request, alias)                # auth + allow-list
    cands = ((await asyncio.to_thread(store.get, alias)) if store.is_active() else None) \
        or image_models.get(alias)
    if not cands:
        raise HTTPException(404, f"generation alias '{alias}' not found")
    cand = cands[0]
    # ONE seam for both candidate kinds (ComfyUI: workflow + mapping labels; a cloud
    # backend: the endpoint's fixed label table) — see adapters.public_fields.
    params, images, files = adapters.public_fields(cand)
    wf = cand.get("workflow_json") or {}
    mapping = cand.get("mapping") or {}
    kinds = sorted(k for _, k in lora_groups(wf, mapping) if k)
    out: dict = {"object": "generation.schema", "alias": alias,
                 "backends": [c.get("backend") for c in cands],
                 "params": params, "images": images, "files": files,
                 "modes": ["sync", "async"],
                 "loras_url": f"/v1/generations/{alias}/loras",
                 "loras": {"list_url": f"/v1/generations/{alias}/loras",
                           "request": "loras: [{name, strength}]",
                           **({"paired_stacks": kinds,
                               "note": "send ONE pair half; the counterpart is resolved server-side"}
                              if len(kinds) > 1 else {})}}
    if cand.get("fps"):
        out["fps"] = cand["fps"]
        if cand.get("frames_snap"):
            out["frames_snap"] = cand["frames_snap"]     # frames land on snap·k+1
        if _mapping_param(mapping, "frames"):
            out["seconds_supported"] = True              # params.seconds → frames via fps
    return out


async def _require_job_owner(authorization: Optional[str], request: Request, job_id: str) -> None:
    """A caller may only touch its own jobs: non-admin users by name, and — open
    mode — anonymous callers by IP (_check_owner does the matching). admin/master
    see all; a 'default'/legacy-owned job stays open to everyone."""
    user = authenticate(authorization)
    if user and (user.get("_master") or user.get("role") == "admin"):
        return
    job = await asyncio.to_thread(jobs.get, job_id)
    _check_owner(job, user, status=403, detail="not your job",
                 anon_owner=_request_owner(request))


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str, request: Request, authorization: Optional[str] = Header(None)):
    await _require_job_owner(authorization, request, job_id)
    return await _job_view(job_id, request)


@app.get("/v1/jobs/{job_id}/result/{n}")
async def get_job_result(job_id: str, n: int, request: Request, authorization: Optional[str] = Header(None)):
    await _require_job_owner(authorization, request, job_id)
    rp = await asyncio.to_thread(jobs.result_path, job_id, n)
    if rp is None:
        raise HTTPException(404, f"result {n} of job '{job_id}' not found")
    path, mime, name = rp
    headers = None
    if name:                                            # suggest the original filename on download,
        headers = {"Content-Disposition": jobs.content_disposition(name)}   # inline so media still previews
    return FileResponse(path, media_type=mime, headers=headers)


@app.get("/v1/jobs/{job_id}/input/{n}")
async def get_job_input(job_id: str, n: int, request: Request, authorization: Optional[str] = Header(None)):
    """Reference image `n` that was submitted with a generation job (kept within TTL)."""
    await _require_job_owner(authorization, request, job_id)
    ip = await asyncio.to_thread(jobs.input_path, job_id, n)
    if ip is None:
        raise HTTPException(404, f"input {n} of job '{job_id}' not found")
    path, mime = ip
    return FileResponse(path, media_type=mime)


@app.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Cancel a queued/running generation job (interrupts the backend, frees the GPU)."""
    await _require_job_owner(authorization, request, job_id)
    if not await cancel_generation(job_id):
        raise HTTPException(409, f"job '{job_id}' is not cancellable (already done/failed/unknown)")
    return {"job_id": job_id, "status": "failed", "cancelled": True}


# ── OpenAI-compatible image endpoints (C4) ───────────────────────────────────
# Thin shims over the native job path so OpenAI image clients (anima-verse's image
# provider, SDKs) reach the gateway's ComfyUI generation aliases.
#  /v1/images/generations : JSON, text->image (+ bonus LocalAI-style ref_images)
#  /v1/images/edits       : multipart, reference image(s) -> alias image-input slots

# Request/response plumbing (multipart parsing, size/scalar coercion, the OpenAI
# images response shape) lives in openai_image_bridge.py; main keeps only what
# needs gateway state: the slot lookup below and the endpoints.

def _gen_image_slots(alias: str) -> list:
    """Ordered image-input param names of a generation alias (its workflow's image
    loaders per the mapping) — reference images map onto these positionally. Includes
    busy backends: slots are a workflow property, not gated on backend availability
    (else a busy backend would silently drop the uploaded reference images)."""
    routes = get_gen_routes(alias, include_busy=True)
    if not routes:
        return []
    _, cand = routes[0]
    if adapters.cloud_kind(cand):
        # [1] = the IMAGE slots only: a `files` entry (a rigging mesh) is not something
        # a positional reference image may ever land on.
        return [i["name"] for i in adapters.public_fields(cand)[1]]   # labels ARE the params
    return image_params(cand.get("workflow_json") or {}, cand.get("mapping") or {})


_EXT_BY_MIME = {                            # what a data-URI MIME means as a file extension —
    "model/gltf-binary": "glb",             # the model types mimetypes doesn't know
    "model/gltf+json": "gltf",
    "model/obj": "obj", "model/stl": "stl", "model/ply": "ply",
    "model/fbx": "fbx", "application/x-fbx": "fbx",
}


async def _decode_ref_blob(ref) -> Optional[tuple[bytes, str]]:
    """A client-supplied blob as base64 / data-URI / http(s) URL → (bytes, extension).

    The extension comes from the data-URI MIME or the URL path, never from sniffing the
    bytes: this carries meshes as well as images, and a wrong guess would hand ComfyUI a
    file its loader refuses. `.glb` is the fallback — the mesh params are the only
    consumers of a payload that names no type."""
    if not isinstance(ref, str) or not ref:
        return None
    ext = ""
    if ref.startswith("data:"):
        head, _, rest = ref.partition(",")
        mime = head[len("data:"):].split(";")[0].strip().lower()
        ext = _EXT_BY_MIME.get(mime) or (mimetypes.guess_extension(mime) or "").lstrip(".")
        ref = rest
    if ref.startswith(("http://", "https://")):
        ext = ext or Path(urlparse(ref).path).suffix.lstrip(".")
        try:
            r = await http_client.get(ref, timeout=20.0)
        except Exception:
            return None
        return (r.content, _clean_ext(ext)) if r.status_code == 200 else None
    try:
        return base64.b64decode(ref), _clean_ext(ext)
    except Exception:
        return None


def _clean_ext(ext: str) -> str:
    """A safe filename extension (the value ends up in an upload filename)."""
    ext = re.sub(r"[^A-Za-z0-9]", "", ext or "")[:8].lower()
    return ext or "glb"


async def _decode_ref_image(ref) -> Optional[bytes]:
    """Reference image as base64 data-URI / raw base64 / http(s) URL -> bytes."""
    got = await _decode_ref_blob(ref)
    return got[0] if got else None


@app.post("/v1/images/generations")
async def images_generations(request: Request, authorization: Optional[str] = Header(None)):
    """OpenAI Images API (text->image). Bonus: LocalAI-style `ref_images`
    (base64/URL list) are accepted and mapped onto the alias's image slots."""
    body = await request.json()
    alias = body.get("model", "")
    await gate_request(authorization, request, alias)
    if not body.get("prompt"):
        raise HTTPException(400, "`prompt` is required")
    w, h = parse_size(body.get("size"))
    refs = body.get("ref_images") or []
    decoded = [await _decode_ref_image(r) for r in refs]
    slots = await asyncio.to_thread(_gen_image_slots, alias)   # ONE lookup — uploads + log
    uploads = images_uploads(decoded, slots) if refs else None
    extra = {k: v for k, v in body.items() if k not in OAI_IMG_KEYS}   # dynamic workflow params
    logger.info(f"images/generations '{alias}': ref_images={len(refs)} "   # where client images land
                f"decoded_ok={sum(1 for d in decoded if d)} slots={slots} "
                f"filled={sorted((uploads or {}).keys())} extra_keys={sorted(extra)}")
    native = {
        "model": alias, "mode": "sync",
        "prompt": body.get("prompt", ""), "negative_prompt": body.get("negative_prompt", ""),
        "params": {"width": w, "height": h, **(body.get("params") or {}), **extra},
        "output": {"n": int(body.get("n") or 1), "mode": "sync"},
    }
    view = gen_done_or_502(await run_generation(native, request, upload_images=uploads))
    return JSONResponse(await asyncio.to_thread(          # b64 branch reads result files
        images_response, view, body.get("response_format") or "url"))


@app.post("/v1/images/edits")
async def images_edits(request: Request, authorization: Optional[str] = Header(None)):
    """OpenAI Images Edit API (multipart): `image` field(s) carry reference images,
    mapped positionally onto the alias's declared image-input slots (cap = slot count;
    OpenAI itself allows 1 for dall-e-2, up to 16 for gpt-image-1)."""
    f = await multipart_list(request)
    one = lambda k, d="": (f.get(k) or [d])[0]
    alias = (one("model") or "").strip()
    await gate_request(authorization, request, alias)
    images = [v for v in (f.get("image") or []) if isinstance(v, (bytes, bytearray))]
    if not images:
        raise HTTPException(400, "at least one `image` file is required")
    masks = [v for v in (f.get("mask") or []) if isinstance(v, (bytes, bytearray))]
    slots = await asyncio.to_thread(_gen_image_slots, alias)   # ONE lookup — uploads + log
    logger.info(f"images/edits '{alias}': image_files={len(images)} mask_files={len(masks)} "  # where they land
                f"slots={slots} "
                f"scalar_keys={sorted(k for k, vs in f.items() if not isinstance((vs or [None])[0], (bytes, bytearray)))}")
    # OpenAI `mask` field → the next positional slot (the mask slot is normally last in
    # the mapping order, after the reference image[s]).
    images = (images + masks)[:16]                      # OpenAI gpt-image-1 max; workflow may use fewer
    w, h = parse_size(one("size") or None)
    extra = {k: coerce_scalar(one(k)) for k, vs in f.items()  # dynamic scalar params (loras, seed, …)
             if k not in EDIT_KNOWN and not isinstance((vs or [None])[0], (bytes, bytearray))}
    native = {
        "model": alias, "mode": "sync", "task": "img2img",
        "prompt": one("prompt"), "negative_prompt": one("negative_prompt"),
        "params": {"width": w, "height": h, **extra},
        "output": {"n": int((one("n") or "1") or 1), "mode": "sync"},
    }
    view = gen_done_or_502(await run_generation(native, request, upload_images=images_uploads(images, slots)))
    return JSONResponse(await asyncio.to_thread(          # b64 branch reads result files
        images_response, view, one("response_format") or "url"))


def dashboard_snapshot() -> dict:
    """Live 'what's happening now' for the dashboard: per-backend status + in-flight,
    LLM/image activity totals, image-job counts + recent, and recent calls."""
    hour_ago = int(time.time()) - 3600
    calls_1h = stats.count_by_backend_since(hour_ago) if stats.is_active() else {}
    jobs_1h = jobs.count_by_backend_since(hour_ago) if jobs.is_active() else {}
    bes = []
    for b in backends:
        bid, en = backend_id(b), is_enabled(b)
        # requests handled in the last hour: LLM calls from stats, image jobs from the
        # job store (an LLM and a ComfyUI backend may share a name → pick by type).
        src_1h = jobs_1h if _is_gen(b) else calls_1h
        bes.append({
            "name": b["name"], "type": b.get("type", "openai"),
            "enabled": en, "healthy": en and backend_healthy.get(bid, False),
            "busy": en and backend_busy(b), "inflight": backend_inflight.get(bid, 0),
            "draining": is_draining(b),
            "max_concurrent": backend_max_concurrent(b),
            "models": len(backend_models.get(bid, set())),
            "reqs_1h": src_1h.get(b["name"], 0),
            "sampling_defaults": b.get("sampling_defaults") or None,
        })
    is_comfy = _is_gen
    return {
        "backends": bes,
        "llm_inflight": sum(backend_inflight.get(backend_id(b), 0) for b in backends if not is_comfy(b)),
        "img_inflight": sum(backend_inflight.get(backend_id(b), 0) for b in backends if is_comfy(b)),
        "parked": len(_parked),
        "parked_calls": [{
            "id": e["id"], "alias": e["alias"], "source": e["source"],
            "waited_s": max(0.0, time.time() - e["enqueued"]),
            "remaining_s": max(0.0, e["deadline"] - time.monotonic()),
        } for e in list(_parked)],
        "jobs_active": jobs.is_active(),
        "jobs_counts": jobs.counts(media_only=True),     # the dashboard cards/panel are MEDIA
        "jobs_recent": jobs.recent(15, media_only=True),  # jobs — chat/response rows stay out
        "stats_active": stats.is_active(),
        "calls_24h": stats.count_since(int(time.time()) - 86400),
        # Recent LLM calls = currently running (live registry) + ended in the last 5 min (stats).
        "llm_running": sorted(_active_calls.values(), key=lambda c: c.get("started", 0)),
        "llm_recent": stats.recent_since(int(time.time()) - 300) if stats.is_active() else [],
    }


def gen_speed_info() -> dict:
    """What the media scheduler is actually routing on, keyed "alias|backend name".

    `speed` is the live EMA in seconds (`gen_speed`, seeded at boot from the job store)
    — the number `order_ready` sorts by; `None` there means UNMEASURED, which sorts
    FIRST (probe-once) unless the candidate has spent its probe on a fault. `quarantine`
    carries the seconds remaining per key. Keyed by backend NAME, not `backend_id`,
    because that is what a job row and the console table show."""
    speed, quar = {}, {}
    by_bid = {backend_id(b): b["name"] for b in backends}
    now = time.time()
    for key, secs in gen_speed.items():
        alias, _, bid = key.rpartition("|")
        speed[f"{alias}|{by_bid.get(bid, bid)}"] = secs
    for key, rec in gen_exec_faults.items():
        alias, _, bid = key.rpartition("|")
        left = rec.get("until", 0.0) - now
        if left > 0:
            quar[f"{alias}|{by_bid.get(bid, bid)}"] = int(left)
    return {"speed": speed, "quarantine": quar}


def _quarantine_info(bid: str) -> dict:
    """`quarantined`: the aliases this backend is currently held out of rotation for,
    with the error that earned it. Unlike the fail-rate this one DOES change routing,
    so it has to be visible — an operator must never have to guess why a backend sits
    idle while jobs run elsewhere."""
    now = time.time()
    held = []
    for key, rec in gen_exec_faults.items():
        alias, _, key_bid = key.rpartition("|")
        if key_bid != bid or now >= rec.get("until", 0.0):
            continue
        held.append({"alias": alias, "until": int(rec["until"]),
                     "for_s": int(rec["until"] - now), "fails": rec.get("fails", 0),
                     "error": (rec.get("error") or "")[:200]})
    return {"quarantined": sorted(held, key=lambda h: h["alias"])} if held else {}


def _comfy_watch_info(b: dict) -> dict:
    """Executor-watchdog + rolling fail-rate fields for comfy backends (merged
    into /health + UI snapshot). Both fail rates stay display-only (runbook C): the
    operator decides — they never reorder routing (A1). The `quarantined` list beside
    them is the one thing here that does, and is reported for exactly that reason."""
    if b.get("type") != "comfyui":
        return {}
    info: dict = {}
    fs = _gen_fail_stats(backend_id(b))
    if fs:
        info.update(fs)
    info.update(_quarantine_info(backend_id(b)))
    ad = backend_adapters.get(backend_id(b))
    if ad is not None:
        info.update({"exec_stuck": bool(getattr(ad, "exec_stuck", False)),
                     "last_restart": int(ad.last_restart) if getattr(ad, "last_restart", 0.0) else None,
                     "last_restart_result": getattr(ad, "last_restart_result", "") or None})
    return info


def _cloud_info(b: dict) -> dict:
    """Credit balance seen at the last discovery of a cloud backend (Meshy, Tripo), plus
    the same rolling fail-rate the comfy backends carry (merged into /health + the UI
    snapshot); {} for every other type. fail_rate is display-only — it never
    reorders routing."""
    if b.get("type") not in adapters.CLOUD_TYPES:
        return {}
    info: dict = {}
    fs = _gen_fail_stats(backend_id(b))      # same rolling fail-rate as comfy backends
    if fs:
        info.update(fs)
    ad = backend_adapters.get(backend_id(b))
    if ad is None:
        return info
    info.update({"credits": getattr(ad, "credits", None),
                 "credits_at": int(getattr(ad, "credits_at", 0) or 0) or None})
    return info


async def _scan_fetch(url: str):
    """netscan's fetch: (status, json|None) with a short timeout — an open port that
    does not answer HTTP in 2 s is not a backend worth waiting for."""
    async with httpx.AsyncClient(timeout=2.0) as c:
        r = await c.get(url)
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, None


async def _scan_resolve(host: str) -> Optional[str]:
    try:
        return (await asyncio.to_thread(socket.gethostbyaddr, host))[0]
    except Exception:
        return None


def start_scan() -> bool:
    """Start one LAN scan (Backends tab button). False when one is still running —
    the console then simply shows that one. Never writes to the store."""
    t = _scan.get("task")
    if t is not None and not t.done():
        return False
    cidrs = list(scan_cidrs) or netscan.local_cidrs()
    hosts, truncated = netscan.expand_targets(cidrs, netscan.HOST_CAP)
    res = netscan.ScanResult(cidrs=cidrs, ports=list(scan_ports), hosts_total=len(hosts),
                             truncated=truncated)
    _scan["result"] = res
    if not hosts:
        res.finished = time.time()           # nothing to scan: report, do not spawn
        _scan["task"] = None
        return True
    _scan["task"] = asyncio.create_task(netscan.scan(
        hosts, list(scan_ports), fetch=_scan_fetch, resolve=_scan_resolve,
        backends=[{"name": b["name"], "url": b.get("url", "")} for b in backends],
        result=res))
    return True


def scan_status() -> dict:
    """Snapshot for admin._scan_panel — plain dicts, no dataclasses across the seam."""
    res = _scan.get("result")
    if res is None:
        return {"running": False, "cidrs": [], "ports": list(scan_ports), "hosts_total": 0,
                "hosts_done": 0, "truncated": False, "error": None, "findings": [], "no_range": False}
    return {
        "running": res.running,
        "cidrs": list(res.cidrs), "ports": list(res.ports),
        "hosts_total": res.hosts_total, "hosts_done": res.hosts_done,
        "truncated": res.truncated, "error": res.error,
        "no_range": res.hosts_total == 0,
        "findings": [{"host": f.host, "port": f.port, "url": f.url, "type": f.type,
                      "flavor": f.flavor, "models": f.models, "needs_key": f.needs_key,
                      "known_as": f.known_as, "hostname": f.hostname} for f in res.findings],
    }


def gateway_info() -> dict:
    """Snapshot the UI's Backends/Input/Server tabs read from."""
    config_ids = {backend_id(b) for b in config_backends}
    return {
        "backends": [{
            "name": b["name"], "type": b.get("type", "openai"),
            "enabled": is_enabled(b), "healthy": backend_healthy.get(backend_id(b), False),
            "error": backend_error.get(backend_id(b)),      # why it is down (kind/status/detail)
            "inflight": backend_inflight.get(backend_id(b), 0), "draining": is_draining(b),
            "models": len(backend_models.get(backend_id(b), set())), "url": b["url"],
            "max_concurrent": b.get("max_concurrent"),
            "chat_only": bool(b.get("chat_only")), "serverless_only": bool(b.get("serverless_only")),
            "local": bool(b.get("local")), "paid": bool(b.get("paid")),
            "sampling_defaults": b.get("sampling_defaults") or None,
            "host": backend_hosts.get(backend_id(b), ""),
            "host_explicit": bool((b.get("host") or "").strip()),
            "source": "config" if backend_id(b) in config_ids else "ui",
            **_comfy_watch_info(b), **_cloud_info(b),
        } for b in backends],
        "virtual_models": list(virtual_models.keys()),
        "endpoints": ["/v1/chat/completions", "/v1/completions", "/v1/embeddings",
                      "/v1/responses", "/v1/models", "/v1/generations", "/v1/jobs/{id}"],
    }


def apply_backend_change() -> None:
    """Re-merge config + store backends, rebind adapters, and kick an immediate
    discovery — called by the UI after a backend is added/edited/deleted/enabled.
    Wakes the park queue so calls waiting on 'all busy' re-evaluate against the new
    backend set now (a taken-offline backend drops out; a brought-online one is
    grabbed once discovery marks it healthy — see refresh_backend)."""
    rebuild_backends()
    build_backend_adapters()
    _notify_slot_free()

    async def _discover():
        await asyncio.gather(*[refresh_backend(b, http_client) for b in enabled_backends()])
    try:
        asyncio.get_running_loop().create_task(_discover())
    except RuntimeError:
        pass


def begin_drain(bid: str) -> bool:
    """Take a backend offline gracefully: stop routing new requests to it now, and
    disable it once its in-flight requests finish. False if unknown/already disabled."""
    b = next((x for x in backends if backend_id(x) == bid), None)
    if b is None or not is_enabled(b):
        return False
    _draining.add(bid)
    n = backend_inflight.get(bid, 0)
    logger.info(f"[{b['name']}] draining — {n} in-flight; goes offline when idle")
    _notify_slot_free()                  # parked calls re-evaluate: this backend is out now
    if n <= 0:
        _finalize_drain(bid)
    return True


def cancel_drain(bid: str) -> bool:
    """Abort a drain → the backend rejoins rotation. False if it wasn't draining."""
    if bid not in _draining:
        return False
    _draining.discard(bid)
    nm = next((x["name"] for x in backends if backend_id(x) == bid), bid)
    logger.info(f"[{nm}] drain cancelled — back in rotation")
    _notify_slot_free()                  # back in rotation → parked calls can grab it now
    return True


def set_backend_enabled(bid: str, on: bool) -> bool:
    """Persist a backend's `enabled` flag (store) and rebuild. Backs the drain-finalize
    and the UI take-offline / bring-online actions. False if the backend is unknown."""
    b = next((x for x in backends if backend_id(x) == bid), None)
    if b is None:
        return False
    entry = dict(store.get_backend(b["name"], b.get("type", "openai")) or
                 {k: b[k] for k in ("name", "type", "url", "priority", "max_concurrent",
                                    "api_key", "chat_only", "serverless_only", "local",
                                    "paid") if k in b})
    entry.update({"name": b["name"], "type": b.get("type", "openai"), "enabled": bool(on)})
    store.upsert_backend(entry)
    logger.info(f"[{b['name']}] {'enabled' if on else 'disabled'} via console")
    apply_backend_change()
    return True


def _finalize_drain(bid: str) -> None:
    """Backend is idle → take it offline (persist enabled=false) and rebuild."""
    _draining.discard(bid)
    set_backend_enabled(bid, False)
    logger.info(f"backends changed → {len(backends)} effective")


def apply_chat_aliases() -> None:
    """Re-merge config + store chat aliases into the live router — called by the UI
    after a chat alias is added/edited/deleted. No discovery/adapter rebind needed;
    routing reads `virtual_models` directly."""
    rebuild_virtual_models()
    logger.info(f"chat aliases changed → {len(virtual_models)} effective")


# Restart-only server state actually in effect (snapshotted at startup), so the UI
# can flag settings whose change needs a restart.
_server_runtime: dict = {}


def _apply_scan_settings(s: dict) -> None:
    """Server-tab text fields → the scan globals. Blank cidrs = derive from this host;
    blank ports = DEFAULT_PORTS. Pure over `s`, so tests can call it directly."""
    global scan_cidrs, scan_ports
    if "scan_cidrs" in s:
        scan_cidrs = [c.strip() for c in str(s.get("scan_cidrs") or "").split(",") if c.strip()]
    if "scan_ports" in s:
        scan_ports = netscan.parse_ports(s.get("scan_ports")) or list(netscan.DEFAULT_PORTS)


def apply_server_settings() -> None:
    """Overlay UI-managed server settings (store) onto the live config globals.

    Runtime knobs (api_key, log_per_call, model_prefix, max_concurrent, health
    interval) take effect immediately. stats.* are overlaid too but only bite on the
    next restart (the stats server is built once at startup). The gateway listening
    port is set by the launch command, so it is informational here."""
    global api_key, log_per_call, model_prefix, max_concurrent_default, health_check_interval
    global park_timeout_s, async_park_timeout_s, park_health_grace_s, max_parked
    global fast_probe_interval_s, affinity_max_wait_s
    s = store.get_settings() if store.is_active() else {}
    if "api_key" in s:
        api_key = s["api_key"] or None
    if "log_per_call" in s:
        log_per_call = bool(s["log_per_call"])
    if "model_prefix" in s:
        model_prefix = bool(s["model_prefix"])
    if "max_concurrent" in s:
        max_concurrent_default = s["max_concurrent"] if s["max_concurrent"] not in ("", None) else None
    if "health_check_interval" in s:
        try:
            health_check_interval = int(s["health_check_interval"])
        except (TypeError, ValueError):
            pass
    if "park_timeout_s" in s:
        try:
            park_timeout_s = float(s["park_timeout_s"])
        except (TypeError, ValueError):
            pass
    if "async_park_timeout_s" in s:
        try:
            async_park_timeout_s = float(s["async_park_timeout_s"])
        except (TypeError, ValueError):
            pass
    if "park_health_grace_s" in s:
        try:
            park_health_grace_s = float(s["park_health_grace_s"])
        except (TypeError, ValueError):
            pass
    if "max_parked" in s:
        try:
            max_parked = int(s["max_parked"])
        except (TypeError, ValueError):
            pass
    if "affinity_max_wait_s" in s:
        try:
            affinity_max_wait_s = float(s["affinity_max_wait_s"])
        except (TypeError, ValueError):
            pass
    if "fast_probe_interval_s" in s:
        try:
            fast_probe_interval_s = max(0.0, float(s["fast_probe_interval_s"]))
        except (TypeError, ValueError):
            pass
    _apply_scan_settings(s)
    # restart-only: overlaid onto stats_cfg / jobs_cfg so the next start picks them up
    # (these init once at startup). Lets config.yaml shed the jobs/stats db knobs.
    for skey, ckey in (("stats_enabled", "enabled"),
                       ("stats_db_path", "db_path"), ("stats_retention_days", "retention_days")):
        if skey in s:
            stats_cfg[ckey] = s[skey]
    for skey, ckey in (("jobs_enabled", "enabled"), ("jobs_db_path", "db_path"),
                       ("jobs_blob_dir", "blob_dir"), ("jobs_default_ttl_s", "default_ttl_s"),
                       ("jobs_prune_interval_s", "prune_interval_s")):
        if skey in s:
            jobs_cfg[ckey] = s[skey]
    if s:
        logger.info(f"server settings: applied {len(s)} UI override(s)")


def server_info() -> dict:
    """Effective server settings + the restart-only values actually running, for the
    UI's Server tab."""
    return {
        "effective": {
            "api_key_set": bool(api_key),
            "log_per_call": bool(log_per_call),
            "model_prefix": bool(model_prefix),
            "max_concurrent": max_concurrent_default,
            "health_check_interval": health_check_interval,
            # park knobs must be echoed back — the Server tab renders the form from
            # `effective`, so a key missing here shows blank and a Save writes "" over
            # the stored value (apply then drops it → the setting looks unsaveable).
            "park_timeout_s": int(park_timeout_s) if park_timeout_s == int(park_timeout_s) else park_timeout_s,
            "max_parked": max_parked,
            "affinity_max_wait_s": (int(affinity_max_wait_s)
                                    if affinity_max_wait_s == int(affinity_max_wait_s)
                                    else affinity_max_wait_s),
            "fast_probe_interval_s": (int(fast_probe_interval_s)
                                      if fast_probe_interval_s == int(fast_probe_interval_s)
                                      else fast_probe_interval_s),
            "scan_cidrs": ", ".join(scan_cidrs),
            "scan_ports": ", ".join(str(p) for p in scan_ports),
            "port": (config or {}).get("port", 4000),
            "stats_enabled": bool(stats_cfg.get("enabled")),
            "stats_db_path": stats_cfg.get("db_path", "stats.db"),
            "stats_retention_days": stats_cfg.get("retention_days", 0),
            "jobs_enabled": bool(jobs_cfg.get("enabled")),
            "jobs_db_path": jobs_cfg.get("db_path", "jobs.db"),
            "jobs_blob_dir": jobs_cfg.get("blob_dir", "jobs"),
            "jobs_default_ttl_s": jobs_cfg.get("default_ttl_s", 86400),
            "jobs_prune_interval_s": jobs_cfg.get("prune_interval_s", 3600),
        },
        "runtime": dict(_server_runtime),     # restart-only state in effect now
    }


def apply_server_settings_hook() -> None:
    """UI save hook: re-apply settings + log."""
    apply_server_settings()


def llm_backends_info() -> list[dict]:
    """LLM (non-ComfyUI) backends + their discovered model ids — feeds the chat-alias
    editor's per-backend model pickers."""
    return [{"name": b["name"], "type": b.get("type", "openai"),
             "enabled": is_enabled(b),
             "models": sorted(backend_models.get(backend_id(b), set()))}
            for b in backends if not _is_gen(b)]


# Wire the UI to the generation core, the ComfyUI backends, the status snapshot,
# and the backend-change hook.
admin.bind(comfy_backends=lambda: [b for b in backends if b.get("type") == "comfyui"],
           gen_backends=lambda: [b for b in backends if _is_gen(b)],
           gateway_info=gateway_info,
           gen_speed_info=gen_speed_info,
           job_progress=lambda job_id: gen_progress.get(job_id),
           apply_backends=apply_backend_change,
           llm_backends=llm_backends_info,
           config_chat_aliases=lambda: dict(config_virtual_models),
           apply_chat_aliases=apply_chat_aliases,
           playground_key=_playground_key,
           routing_snapshot=routing_snapshot,
           server_info=server_info,
           apply_server_settings=apply_server_settings_hook,
           scan_start=start_scan,
           scan_status=scan_status,
           apply_users=apply_users,
           resolve_admin=resolve_admin, ui_locked=ui_locked,
           dashboard_snapshot=dashboard_snapshot, cancel_generation=cancel_generation,
           drain_backend=begin_drain, cancel_drain=cancel_drain,
           set_backend_enabled=set_backend_enabled,
           restart_comfy=restart_comfy_backend,
           llm_backend_names=lambda: sorted({b["name"] for b in backends
                                             if not _is_gen(b)}),
           resolve_for_backend=resolve_for_backend,
           apply_reasoning=apply_reasoning_rules,
           probe_reasoning=probe_reasoning,
           voice_lib_save=save_voice_ref, voice_lib_delete=delete_voice_ref,
           voice_lib_ship=ship_voice_ref, voice_ship_config=voice_ship_config,
           apply_voice_library=apply_voice_library,
           apply_hosts=apply_hosts,
           backend_loras=lambda: {b["name"]: sorted(backend_loras.get(backend_id(b), set()))
                                  for b in backends if b.get("type") == "comfyui"})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "parked": len(_parked),
        "backends": {
            backend_id(b): {
                "name": b["name"], "type": b.get("type", "openai"),
                "enabled": is_enabled(b),
                "healthy": is_enabled(b) and backend_healthy.get(backend_id(b), False),
                "error": backend_error.get(backend_id(b)) if is_enabled(b) else None,
                "busy": is_enabled(b) and backend_busy(b),
                "inflight": backend_inflight.get(backend_id(b), 0),
                "max_concurrent": backend_max_concurrent(b),
                "paid": bool(b.get("paid")),
                "tps": round(backend_tps.get(backend_id(b), 0.0), 1),
                "sampling_defaults": b.get("sampling_defaults") or None,
                "models": sorted(backend_models.get(backend_id(b), set())) if is_enabled(b) else [],
                **_comfy_watch_info(b), **_cloud_info(b),
            }
            for b in backends
        },
        "virtual_models": virtual_models,
        # Physical-box grouping (explicit `host` field or URL IP) — which backends
        # share a machine/GPU. Basis for the host-level policies (see
        # docs/host-coordination-plan.md).
        "hosts": host_backends,
        # Aliases that shadow a real model on a backend they don't map (→ that
        # model is unreachable by its bare name). Empty list = no such conflict.
        "alias_model_conflicts": [c for c in alias_model_conflicts() if c["shadowed"]],
    }
