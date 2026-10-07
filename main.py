import asyncio
import base64
import calendar
import copy
import fnmatch
import hashlib
import hmac
import ipaddress
import json
import logging
import mimetypes
import re
import shlex
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from watchfiles import awatch

import adapters
import admin
import anthropic_bridge
import faults
import jobs
import reasoning
import scheduler
import netscan
import os
import socket
import stats
import store
import modelsync
import loratags
import hostapi
import hostctl
import rpvolume
import services
import sshrun
import thunder
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
    _warn_gen_name_clashes(backends)
    for b in backends:
        # Cost tier for the scheduler (spec 2026-09-01): a paid backend is a candidate
        # only when no unpaid one is free. Normalized here — config and store entries
        # may omit the key entirely. A cloud backend (Meshy, Tripo) bills per task, so it
        # is ALWAYS paid, and a RunPod endpoint bills per second of every run.
        b["paid"] = True if (b.get("type") in adapters.CLOUD_TYPES
                             or b.get("type") in adapters.BILLING_TYPES) else bool(b.get("paid"))
    # Every rebuild hands out NEW backend dicts; a managed-host controller reads its
    # entry and services from what it was handed, so it must be given the current ones.
    # BEFORE the grouping and the route index: an attached backend's forward ends and
    # URL are derived there (and written onto these dicts).
    sync_host_controllers()
    sync_volume_controllers()
    backend_hosts = {backend_id(b): backend_host(b) for b in backends}
    host_backends = {}
    for bid, h in backend_hosts.items():
        host_backends.setdefault(h, []).append(bid)
    apply_hosts()
    rebuild_route_index()                  # backend set/enabled flags changed


_gen_clash_warned: set = set()


def gen_name_clashes(blist: list) -> list:
    """[(name, (type, type, …))] — generation backends of the SAME kind sharing a name
    (a `comfyui` and a `runpod` backend, the first two types of one kind). Names are
    unique only per (name, type), and a workflow candidate names its backend by NAME:
    `_gen_backend_for` and every other name-only lookup would then pick whichever comes
    first, so local work could silently run (and bill) on RunPod. The console refuses
    such a name (admin.backend_save); config.yaml can still hold one."""
    seen: dict = {}
    for b in blist:
        if b.get("type") in adapters.GEN_TYPES:
            seen.setdefault((b.get("name"), adapters.backend_kind(b)), set()).add(b.get("type"))
    return [(n, tuple(sorted(ts))) for (n, _k), ts in seen.items() if len(ts) > 1]


def _warn_gen_name_clashes(blist: list) -> None:
    for name, types in gen_name_clashes(blist):
        if (name, types) in _gen_clash_warned:
            continue
        _gen_clash_warned.add((name, types))
        logger.warning(f"backends: {' and '.join(types)} backends are both named '{name}' — "
                       "an alias candidate names its backend by name, so its jobs may run on "
                       "either one (a paid RunPod run included); rename one of them")


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
backend_pricing: dict[str, dict[str, dict[str, float]]] = {}   # name → {model_id → {input, output[, cache_read, cache_write]}}
backend_loras: dict[str, set[str]] = {}                        # id → {lora filename, ...} (ComfyUI)
# bid → (kept, total) of the last discovery poll, when the backend carries an
# allow/deny model filter. A whitelist typo would otherwise leave the backend healthy
# and serving 0 models with nothing anywhere saying why — so the counts are reported.
backend_model_counts: dict[str, tuple[int, int]] = {}
# id → {model_id → context window (tokens)} as LEARNED by discovery (persisted, merged —
# a llama-swap model says its n_ctx only while loaded). The admin's per-backend
# `model_context` rules sit above this; adapters.model_context_for() resolves both.
backend_context: dict[str, dict[str, int]] = {}
# bid → what llama-swap's `/running` last said is loaded ([{model, state, kind}],
# adapters.parse_running). Only backends that HAVE the endpoint carry a key — that key
# is what makes `<backend>/current` routable. Fed by every discovery poll and, for a
# `current` call, by a live query right before routing (_refresh_loaded). In memory
# only: a restart must ask again, the loaded model is not the gateway's to remember.
backend_running: dict[str, list] = {}
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
# backend id → the `host` it named when its drain began. The finalize disables only a
# backend still on that host: one moved to another managed host (or a plain URL) during
# a host's stop belongs to its new host now (R-K2), which may have enabled it already.
_drain_host: dict = {}
# backend ids in `_draining` only for a managed host's automatic restart (hostctl
# `hold_routing`): routing skips them, but they are never finalized (disabled)
_drain_hold: set = set()


def is_draining(backend: dict) -> bool:
    return backend_id(backend) in _draining


def _inflight_inc(name: str) -> None:
    backend_inflight[name] = backend_inflight.get(name, 0) + 1


def _inflight_dec(name: str) -> None:
    backend_inflight[name] = max(0, backend_inflight.get(name, 0) - 1)
    if (name in _draining and name not in _drain_hold
            and backend_inflight.get(name, 0) <= 0):
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
# Cap on async generation jobs queued or running at once (`_gen_tasks`) — the media
# counterpart of max_parked: each is a task + a job row + stored inputs, and without a
# cap a loop of `mode: async` requests queues without end. Server tab.
max_queued_gen: int = 200
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


_bg_refs: set = set()                    # fire-and-forget tasks, held until they finish


def _bg(coro) -> asyncio.Task:
    """`asyncio.create_task` for fire-and-forget work, with the reference HELD until the
    task is done. The event loop keeps only a weak reference to a task, so an unreferenced
    one can be garbage-collected mid-flight — the after-job VRAM free silently never
    happening, or a restart whose `finally` never clears `_comfy_restarting`, after which
    that backend can never be restarted again (K20)."""
    t = asyncio.create_task(coro)
    _bg_refs.add(t)
    t.add_done_callback(_bg_refs.discard)
    return t


_comfy_restarting: set[str] = set()      # backend ids with a restart() in flight


def _spawn_comfy_restart(backend: dict, adapter, why: str) -> None:
    bid = backend_id(backend)
    logger.warning(f"[{backend['name']}] restarting ComfyUI service ({why})")
    _note_fault(backend, "watchdog", "restart", why)
    _comfy_restarting.add(bid)               # only once nothing before the task can raise

    async def _run():
        try:
            await adapter.restart()
        except Exception as e:
            logger.warning(f"[{backend['name']}] ComfyUI restart failed: {_err_text(e)}")
            _note_fault(backend, "watchdog", "restart_failed", _err_text(e))
        finally:
            _comfy_restarting.discard(bid)
    _bg(_run())


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


def _note_fault(backend: dict, source: str, kind: str, detail: str = "",
                status: Optional[int] = None, dur_s: Optional[int] = None) -> None:
    """Put one backend failure into the fault log (faults.py) — the console's only
    memory of it once the backend is healthy again. Never raises into the caller."""
    try:
        bid = backend_id(backend)
        faults.record(bid=bid, backend=backend["name"], type=backend.get("type", "openai"),
                      host=backend_hosts.get(bid) or backend_host(backend), source=source,
                      kind=kind, detail=detail, status=status, dur_s=dur_s)
    except Exception as e:
        logger.warning(f"faults: could not record a fault for [{backend.get('name')}]: {e}")


def _resp_snippet(resp) -> str:
    """The first bytes of an upstream error body, for the fault log (a streamed
    response carries no body attribute — then just the status)."""
    body = getattr(resp, "body", None)
    if isinstance(body, (bytes, bytearray)) and body:
        return bytes(body[:300]).decode("utf-8", "replace")
    return f"HTTP {getattr(resp, 'status_code', '?')}"


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


async def _persist_discovery(label: str, save, *args) -> None:
    """Store what a poll learned (models, context windows) — bookkeeping that must never
    fail the POLL: inside refresh_backend's try, a locked store.db marked a backend that
    had just answered as DOWN and opened a fault-log outage for it. Logged instead; the
    in-memory state is updated anyway and the next change persists again."""
    if not store.is_active():
        return
    try:
        await asyncio.to_thread(save, *args)
    except Exception as e:
        logger.warning(f"[{label}] discovery result not persisted ({save.__name__}): {_err_text(e)}")


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
        try:
            caps = await adapter.discover(client)
        finally:
            # A backend save may have REPLACED the adapter while the poll was in flight
            # (build_backend_adapters): what discover() wrote (slot types, watchdog,
            # balance) went onto the old instance. Carry it over where it still applies,
            # and act on the CURRENT instance from here on (auto-restart cooldown).
            cur = backend_adapters.get(bid)
            if cur is not None and cur is not adapter:
                cur.adopt_discovery(adapter)
                adapter = cur
        # The admin's allow/deny globs narrow the discovered set HERE — type-neutrally
        # (extract_models is the openai path only) and BEFORE `changed`, the persist and
        # the route-index rebuild, so /v1/models, routing, alias candidates and the
        # ComfyUI checkpoint dropdowns all read the one filtered set.
        total = len(caps.models)
        caps.models = adapters.filter_models(caps.models, backend)
        filtered = len(caps.models) != total
        backend_model_counts[bid] = (len(caps.models), total)
        # `models_extra` goes the other way — ids the backend SERVES but does not LIST
        # (FastFlowLM answers /v1/embeddings and /v1/audio/transcriptions while listing
        # chat models only). Added AFTER the filter so a narrow whitelist cannot remove
        # them again, and after the counts so `kept/total` keeps describing what the
        # filter did to what discovery MEASURED. Here and not in the adapter: like the
        # filter it must hold for every backend type, and only a successful poll may
        # publish them — an unreachable backend offering a model is a 503 waiting to
        # happen.
        caps.models = adapters.add_model_extras(caps.models, backend)
        changed = caps.models != backend_models.get(bid)
        if changed:
            await _persist_discovery(label, store.save_backend_models, bid, caps.models)
        backend_models[bid] = caps.models
        backend_pricing[bid] = caps.pricing
        backend_loras[bid] = getattr(caps, "loras", set()) or set()
        learned = merge_learned_context(backend_context.get(bid), getattr(caps, "context", None))
        if learned != backend_context.get(bid, {}):
            backend_context[bid] = learned
            await _persist_discovery(label, store.save_backend_context, bid, learned)
        running = getattr(caps, "running", None)
        had_running = bid in backend_running
        if running is None:
            backend_running.pop(bid, None)
        else:
            backend_running[bid] = running
        if changed or had_running != (running is not None):
            # model set changed, or `/running` appeared/vanished (which decides whether
            # an alias's `current` entry is a candidate at all) → refresh the candidates
            rebuild_route_index()
        was_healthy = backend_healthy.get(bid, False)
        if not was_healthy:
            logger.info(f"[{label}] UP  — {len(caps.models)} models, {len(caps.pricing)} priced")
        if filtered and (not was_healthy or changed):   # same cadence as the UP log — never every poll
            logger.info(f"[{label}] model filter — {len(caps.models)} of {total} models kept")
        backend_healthy[bid] = True
        prev_err = backend_error.pop(bid, None)
        # Closes the outage the fault log opened → its length, from the moment it became a
        # fault (`fault_since`), whatever kind it showed last. A switched-off backend that
        # never turned into a fault opened none (faults.NOT_FAULT_KINDS) — none to close.
        if prev_err and prev_err.get("fault_since") is not None:
            down_s = max(0, int(time.time()) - int(prev_err["fault_since"]))
            _note_fault(backend, "health", faults.RECOVERED,
                        f"back after {down_s} s ({prev_err.get('kind')})", dur_s=down_s)
        if not was_healthy or changed:     # a backend came online / gained models →
            _notify_slot_free()            # let parked calls re-evaluate and grab it
    except Exception as e:
        info = _classify_error(e)
        # Keep the ORIGINAL failure time across repeated polls of the same fault, so
        # the console can say how long it has been broken.
        prev = backend_error.get(bid)
        if prev and prev.get("kind") == info["kind"] and prev.get("status") == info["status"]:
            info["since"] = prev["since"]
        # The fault log sees ONE outage per DOWN, whatever kinds it runs through: it is
        # opened the first time it is a fault (the DOWN poll, or later — a switched-off box
        # that starts timing out), never again, and `fault_since` rides along every poll
        # until the recovery closes it. Keyed on the kind it had WHEN it opened, a
        # timeout→unreachable outage was never closed (no downtime) and an
        # unreachable→timeout one closed without ever opening.
        fault_since = prev.get("fault_since") if prev and not backend_healthy.get(bid, True) else None
        if backend_healthy.get(bid, True):
            hint = " (credential rejected — check the api key)" if info["kind"] == "auth" else ""
            # `_err_text`: an httpx timeout stringifies to "" — the journal read "DOWN —"
            # with nothing after it (measured 2026-09-12/13 on prod, a dozen times).
            logger.warning(f"[{label}] DOWN — {_err_text(e)}{hint}")
        if fault_since is None and info["kind"] not in faults.NOT_FAULT_KINDS:
            fault_since = int(time.time())
            _note_fault(backend, "health", info["kind"], info["detail"], status=info["status"])
        info["fault_since"] = fault_since
        backend_error[bid] = info
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
        backend_model_counts.pop(stale, None)
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
    for _alias, _cands in store.list_aliases().items():
        if adapters.migrate_upload_pins(_cands):
            store.upsert(_alias, _cands)
            logger.warning(f"store: alias '{_alias}': pinned image 'playground upload' → 8×8 "
                           "placeholder (that option always ran on the placeholder and is gone; "
                           "bind the loader as an image request field instead)")
    try:
        migrate_provider_tokens()      # per-host API tokens → one per provider (idempotent)
    except Exception as e:
        logger.warning(f"managed hosts: token migration failed: {type(e).__name__}")
    lora_meta_boot()                   # LoRA trigger words: stored records → memory
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
        _bg(_cancel_orphaned_runpod(jobs.last_orphans()))
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

    # Backend fault log (faults.py): always on, like the store — it is what the
    # Dashboard and Statistic show once a failed backend is healthy again. Opened
    # BEFORE the first discovery, so a backend already down at boot is recorded.
    faults_cfg = config.get("faults") or {} if isinstance(config, dict) else {}
    faults.init(faults_cfg.get("db_path", "faults.db"))
    faults_prune_task = asyncio.create_task(faults.prune_loop(faults_cfg.get("retention_days", 7)))

    log_config_summary()
    await asyncio.gather(*[refresh_backend(b, http_client) for b in enabled_backends()])
    # Managed-host controllers exist since rebuild_backends(); now each reconciles its
    # persisted state with the provider's instance list (an instance that billed through
    # the restart gets its tunnel back) and starts its background loop.
    _hosts_boot()
    health_task = asyncio.create_task(health_loop())
    probe_task = asyncio.create_task(fast_probe_loop())
    watch_task = asyncio.create_task(watch_config_loop())
    lora_task = asyncio.create_task(lora_meta_loop())

    # Stats: record calls + prune, but NO separate server — the dashboard lives in
    # /ui → Statistic now (so no extra port/bind).
    prune_task: Optional[asyncio.Task] = None
    if stats_cfg.get("enabled"):
        stats.init(stats_cfg.get("db_path", "stats.db"), stats_cfg.get("blob_dir", "calls"),
                   body_max_kb=stats_cfg.get("body_max_kb"))
        prune_task = asyncio.create_task(stats.prune_loop(
            stats_cfg.get("retention_days", 0),
            stats_cfg.get("body_retention_days")))          # None/"" = stats' default
        logger.info("stats: recording on; dashboard at /ui → Statistic")
    # snapshot the restart-only server state actually in effect, so the UI can flag
    # when an edited setting needs a restart to apply.
    _server_runtime.update(
        stats_enabled=bool(stats_cfg.get("enabled")),
        stats_db_path=stats_cfg.get("db_path", "stats.db"),
        stats_retention_days=stats_cfg.get("retention_days", 0),
        stats_body_retention_days=stats_cfg.get("body_retention_days", stats.BODY_RETENTION_DAYS_DEFAULT),
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
    lora_task.cancel()
    faults_prune_task.cancel()
    if jobs_prune_task is not None:
        jobs_prune_task.cancel()
    if prune_task is not None:
        prune_task.cancel()
    await _hosts_shutdown()            # tunnels + provider clients; the instances run on
    await http_client.aclose()         # drain the shared connection pool last


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI-Hub", lifespan=lifespan)

# Request bodies are read whole (request.json()/body()) — nothing bounded them, so one
# client could hand the gateway an arbitrarily large body to hold in memory (review S10).
# `max_body_mb` (config.yaml, hot-reloaded; 0 = off) caps every request: a declared
# Content-Length over it is refused before a byte is read, a chunked body is counted
# while it streams. The default stays far above the largest legitimate body — a 64 MB
# mesh under `files` is ~86 MB as base64 JSON.
MAX_BODY_MB_DEFAULT = 200


def max_body_bytes() -> int:
    try:
        mb = float((config or {}).get("max_body_mb", MAX_BODY_MB_DEFAULT))
    except (TypeError, ValueError, NameError):
        mb = MAX_BODY_MB_DEFAULT
    return int(mb * 1024 * 1024) if mb > 0 else 0


class _BodyLimit:
    """Pure ASGI (not BaseHTTPMiddleware): it has to sit in the receive path."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        limit = max_body_bytes() if scope["type"] == "http" else 0
        if not limit:
            return await self.app(scope, receive, send)
        msg = f"request body exceeds {limit / (1024 * 1024):g} MB (max_body_mb)"
        cl = dict(scope.get("headers") or []).get(b"content-length")
        try:
            declared = int(cl) if cl is not None else None
        except ValueError:
            declared = None
        if declared is not None and declared > limit:
            return await JSONResponse({"detail": msg}, status_code=413)(scope, receive, send)
        seen = 0

        async def counted():
            nonlocal seen
            m = await receive()
            if m.get("type") == "http.request":
                seen += len(m.get("body") or b"")
                if seen > limit:
                    # Raised inside the endpoint's body read → the app's HTTPException
                    # handler answers 413 (and logs the refusal like any other).
                    raise HTTPException(413, msg)
            return m
        await self.app(scope, counted, send)


# Added BEFORE admin.register: middleware added later wraps the earlier one, so this
# sits INSIDE the console's BaseHTTPMiddleware guard — whose receive runs in a task
# group that would turn the 413 into an ExceptionGroup (a 500) on its way out.
app.add_middleware(_BodyLimit)
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


@app.exception_handler(Exception)
async def _unexpected_error(request: Request, exc: Exception):
    """An exception nobody caught. On the API it becomes a 502 in the endpoint's own
    error shape — through the HTTPException handler, so it is logged like a refusal
    and `/v1/messages` answers in Anthropic form (Claude Code renders a plain
    "Internal Server Error" as a blank message). Starlette still re-raises it after
    this response, so the traceback reaches the journal. Elsewhere (/ui) the stock
    500 stays."""
    if request.url.path.startswith("/v1/"):
        return await _rejected_call(request, HTTPException(
            502, f"gateway error: {type(exc).__name__}: {_err_text(exc)}"))
    return PlainTextResponse("Internal Server Error", status_code=500)


# Refusals are logged before (or without) authentication, so the row holds what the
# CALLER chose: the model field, x-source, the path. Cut to a fixed length — an
# anonymous client could otherwise store a 50 MB "model name" per request — and 401
# rows, the refusal any stranger can produce at will, are capped per minute so a key
# scanner cannot push every real call out of the LLM Calls view.
_LOG_FIELD_MAX = 200
_UNAUTH_LOG_PER_MIN = 60
_unauth_log: dict = {}            # {"minute": int, "n": recorded, "dropped": int}


def _clip(v) -> Optional[str]:
    if v is None:
        return None
    return (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str))[:_LOG_FIELD_MAX]


def _unauth_row_allowed() -> bool:
    minute = int(time.time() // 60)
    if _unauth_log.get("minute") != minute:
        if _unauth_log.get("dropped"):
            logger.warning(f"stats: {_unauth_log['dropped']} further 401 refusal(s) not logged "
                           f"(cap {_UNAUTH_LOG_PER_MIN}/min)")
        _unauth_log.update(minute=minute, n=0, dropped=0)
    if _unauth_log["n"] >= _UNAUTH_LOG_PER_MIN:
        _unauth_log["dropped"] += 1
        return False
    _unauth_log["n"] += 1
    return True


# How much of each END of a refused body `_ends_only` keeps exact: at least the window
# `stats._preview` reads from each end before collapsing whitespace (8 × (50 + 50)).
_PREVIEW_ENDS = 1024


def _ends_only(v, w: int = _PREVIEW_ENDS):
    """A stand-in for `v` whose `json.dumps` has the SAME first and last `w` characters
    as `json.dumps(v)`, at a size bounded by `w` per nesting level instead of the body's.

    A refusal stores no request (`store_request=False`), only the preview built from
    the serialised body's two ends — and serialising a multi-MB Claude Code context on
    the event loop for that cost ~25-30 ms per MB, per refused retry. Strings keep
    their first and last `w` characters; a list or object keeps items from the front
    until they serialise to `w` characters, the same from the back, and drops the
    middle. Only the item where a run reaches `w` can itself be shortened, so the two
    ends stay character-exact (the same separators as json.dumps' defaults)."""
    if isinstance(v, str):
        return v if len(v) <= 2 * w else v[:w] + v[-w:]
    if isinstance(v, list):
        items, dump = list(range(len(v))), (lambda i, x: json.dumps(x, ensure_ascii=False))
    elif isinstance(v, dict):
        keys = list(v)
        items = keys
        dump = (lambda k, x: json.dumps(str(k), ensure_ascii=False) + json.dumps(x, ensure_ascii=False))
    else:
        return v
    front, back, acc = [], [], 0
    for i in items:
        x = _ends_only(v[i], w)
        front.append((i, x))
        acc += len(dump(i, x)) + 2
        if acc >= w:
            break
    acc, stop = 0, len(front)
    for i in reversed(items[stop:]):
        x = _ends_only(v[i], w)
        back.append((i, x))
        acc += len(dump(i, x)) + 2
        if acc >= w:
            break
    kept = front + back[::-1]
    if isinstance(v, list):
        return [x for _, x in kept]
    return {k: x for k, x in kept}


def _record_rejected(request: Request, exc: HTTPException) -> None:
    """Fire-and-forget stats row for a refused call. Never raises into the response
    path: a logging failure must not turn a clean 503 into a 500."""
    if not stats.is_active():
        return
    if exc.status_code == 401 and not _unauth_row_allowed():
        return
    try:
        body = getattr(request.state, "gw_body", None)
        alias = getattr(request.state, "gw_alias", None)
        if alias is None and isinstance(body, dict):
            alias = body.get("model")
        asyncio.create_task(stats.record_call(
            duration_ms=0, backend=_REJECTED_BACKEND, source=_clip(_source_of(request)),
            # `model` stays empty: it holds the REAL model a backend served, and no
            # backend ever resolved one here — filling it with the alias would render
            # as "x→x" in the call list and claim a resolution that never happened.
            alias=_clip(alias), model=None,
            endpoint=_clip(getattr(request.state, "gw_endpoint", None) or request.url.path),
            status=exc.status_code, input_tokens=0, output_tokens=0, cost_usd=0.0,
            # Feeds the preview only (store_request=False) — its two ends are all it reads.
            request_text=(json.dumps(_ends_only(body), ensure_ascii=False)
                          if isinstance(body, dict) else None),
            response_text=json.dumps({"error": {"message": str(exc.detail)}}, ensure_ascii=False),
            # The reason is the body worth keeping; the request only feeds the preview —
            # an agent retrying a refused 1 MB request stored it once per retry.
            store_request=False,
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


def _key_eq(given: str, expected: str) -> bool:
    """Constant-time compare for the master key (`==` leaks the matching prefix length)."""
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def authenticate(authorization: Optional[str]) -> Optional[dict]:
    """Resolve the Bearer token to a user. Returns None only in bootstrap-open mode
    (no users + no master key). Raises 401 on a missing/invalid token otherwise."""
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
    if not users and not api_key:
        return None
    if not token:
        raise HTTPException(401, "Missing Authorization header")
    if api_key and _key_eq(token, api_key):
        return _MASTER_ADMIN
    u = _users_by_key.get(token)
    if u:
        return u
    raise HTTPException(401, "Invalid API key")


def check_auth(authorization: Optional[str]) -> None:
    authenticate(authorization)


# Group grants: the user editor's "all chat / all image / all backend" boxes store one of
# these TOKENS, resolved on every request — a snapshot of today's names would silently
# leave out every alias or backend added later.
GRANT_ALL_CHAT, GRANT_ALL_IMAGE, GRANT_ALL_BACKENDS = "@chat", "@image", "@backends"


def _gen_alias_exists(name: str) -> bool:
    if name in image_models:
        return True
    return bool(store.is_active() and store.get(name))


def _expand_grants(allow) -> set:
    """The allow-list with `@chat`/`@backends` resolved against the LIVE config (`@image`
    is checked per model by _model_allowed / listed by /v1/models — resolving it here
    would read the whole generation store on every request)."""
    out = set(allow)
    if GRANT_ALL_CHAT in out:
        out |= set(virtual_models)
    if GRANT_ALL_BACKENDS in out:
        out |= {b["name"] for b in backends if b.get("name") and not _is_gen(b)}
    return out


def _model_allowed(user: dict, model: Optional[str]) -> bool:
    raw = user.get("models") or []
    if not raw or not model:
        return True                              # empty allow-list = all models
    allow = _expand_grants(raw)
    if model in allow:                           # exact id / chat alias / image alias
        return True
    if GRANT_ALL_IMAGE in allow and _gen_alias_exists(model):
        return True
    bname, bare = split_backend_prefix(model)    # backend/model
    if bname and bname in allow:                 # whole-backend grant (prefixed request)
        return True
    if bname and bare == adapters.CURRENT_MODEL and model not in virtual_models:
        # `<backend>/current` may land on ANY model that backend has loaded, so only a
        # grant covering all of them (the backend, or this exact entry — both checked
        # above) allows it; a bare `current` in the list must not open every backend.
        return False
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
        if is_alias:                                             # alias → granted backend's model
            real = alias_entry(model, b["name"])[0]
            if real in served or _is_current(b, real):
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
    # admin-only request powers downstream (a backend path in generation params)
    request.state.gw_admin = bool(user.get("_master") or user.get("role") == "admin")
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
    if api_key and _key_eq(key, api_key):
        return _MASTER_ADMIN
    u = _users_by_key.get(key)
    return u if (u and u.get("role") == "admin") else None


def admin_session_tag(name: Optional[str], master: bool) -> Optional[str]:
    """Fingerprint of the credential a /ui session belongs to — the master api_key, or
    the named admin user's key — or None once that is no longer an admin credential
    (key removed or rotated, user deleted/disabled/demoted). The session cookie stores
    it at login and admin re-checks it on every request, so revoking the credential
    revokes its sessions instead of leaving them valid for the cookie's lifetime."""
    if master:
        key = api_key
    else:
        u = next((u for u in users if u.get("name") == name), None)
        key = (u.get("api_key") if u and u.get("role") == "admin" and u.get("enabled", True)
               else None)
    return hashlib.sha256(f"{'m' if master else 'u'}:{key}".encode()).hexdigest() if key else None


def ui_locked() -> bool:
    """True once ANY credential exists (master key or any user) → /ui needs login —
    the same condition that closes the API (`authenticate`). Locking only on an ADMIN
    credential left a gateway with nothing but `role: user` accounts API-closed and
    console-open to the whole LAN. Bootstrap-open (no users, no key) returns False."""
    return bool(api_key) or bool(users)


def admin_credential_exists(user_list: Optional[list] = None,
                            key: Optional[str] = None) -> bool:
    """Whether someone can sign in to /ui: a master key, or an ENABLED admin user that
    has a key (what `resolve_admin` accepts). Defaults to the live state."""
    ulist = users if user_list is None else user_list
    if key if key is not None else api_key:
        return True
    return any(u.get("role") == "admin" and u.get("enabled", True) and u.get("api_key")
               for u in ulist)


def admin_change_refusal(users_after: list) -> Optional[str]:
    """Why a users-editor change must be refused, or None. Two states are never entered
    from the console: users without any admin credential (the console locks and nobody
    can sign in), and a locked gateway losing its last admin credential (the console —
    and with no users left, the API — silently opens again). A master key covers both."""
    if admin_credential_exists(users_after):
        return None
    if admin_credential_exists():
        return ("last_admin" if users_after else "last_admin_open")
    return "no_admin" if users_after else None


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
            if real is not None and (real in backend_models.get(backend_id(b), set())
                                     or _is_current(b, real)):
                # a `current` entry stays the placeholder here — what it resolves to
                # changes with every swap, so resolve_routes picks it per request
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


def anthropic_only_candidates(alias: str) -> bool:
    """Is every backend that could serve `alias` an Anthropic one?

    Asked ONLY to explain a refusal (see `_dispatch_or_park`), never to route: such a
    backend answers `/v1/messages` alone (`serves_path`), so off that path it is no
    candidate and the caller must be told WHICH endpoint to use instead of "no healthy
    backend". Health and busy state are deliberately not consulted — the answer is
    about who OWNS the name, not who is up.

    A '<backend>/<model>' pin is resolved the same way `resolve_routes` resolves it,
    against `_llm_backends` — reading `_route_index` cannot work here: the index holds
    aliases and BARE model ids only (`rebuild_route_index`), so every pinned name came
    back with no candidates and fell through to the generic 503.
    """
    bname, bare = split_backend_prefix(alias)
    if bname is not None:
        b = next((b for b in _llm_backends if b["name"] == bname), None)
        if b is None:
            return False
        real = resolve_for_backend(bare, bname)
        # A model that backend never listed is not an endpoint mistake — leave it 503.
        if real is None or real not in backend_models.get(backend_id(b), set()):
            return False
        return b.get("type") == "anthropic"
    cands = _route_index.get(alias) or []
    return bool(cands) and all(b.get("type") == "anthropic" for b, _ in cands)


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
        if _is_current(b, real):
            real = _current_model(b, path)
        if real is None or real not in backend_models.get(backend_id(b), set()):
            return [], []
        return ([], [(b, real)]) if backend_busy(b) else ([(b, real)], [])

    ready, busy = [], []
    for b, real in _route_index.get(alias, ()):
        if not backend_healthy.get(backend_id(b)) or is_draining(b):
            continue
        if not serves_path(b, path):
            continue
        if _is_current(b, real):
            real = _current_model(b, path)
            if real is None:              # nothing suitable loaded → not a candidate (never load one)
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


def _is_current(backend: dict, real: Optional[str]) -> bool:
    """Is `real` the `current` placeholder ON this backend? Only where llama-swap's
    `/running` answered (backend_running carries the key) — elsewhere `current` is just
    a model name nobody serves. A backend that really lists a model named `current`
    keeps it: a listed id is never shadowed by the placeholder."""
    bid = backend_id(backend)
    return (real == adapters.CURRENT_MODEL and bid in backend_running
            and real not in backend_models.get(bid, set()))


def _current_model(backend: dict, path: str) -> Optional[str]:
    """What `current` resolves to on this backend for `path`, from backend_running —
    None when nothing suitable is loaded (see adapters.pick_current)."""
    bid = backend_id(backend)
    return adapters.pick_current(backend_running.get(bid), path,
                                 backend_models.get(bid, set()), backend_last_key.get(bid))


def _current_backends(alias: str) -> list[dict]:
    """The backends on which `alias` means `current` — the ones a live `/running`
    query has to ask before routing it. Empty for every other alias, so ordinary calls
    pay nothing."""
    bname, bare = split_backend_prefix(alias)
    if bname is not None:
        b = next((b for b in _llm_backends if b["name"] == bname), None)
        return [b] if b is not None and _is_current(b, resolve_for_backend(bare, bname)) else []
    return [b for b, real in _route_index.get(alias, ()) if _is_current(b, real)]


_CURRENT_LIVE_TIMEOUT = 2.0


async def _refresh_loaded(alias: str) -> None:
    """Re-read `/running` on every healthy backend where `alias` means `current`, so the
    pick is made against what is loaded NOW and not up to a discovery interval ago (a
    client talking to llama-swap directly swaps without the gateway knowing). Runs
    BEFORE resolve_routes — never between it and dispatch, where the in-flight claim
    must stay await-free. A failed query keeps the last known list: if the backend is
    really gone, dispatch fails over as it would for any other call."""
    targets = [b for b in _current_backends(alias)
               if backend_healthy.get(backend_id(b)) and not is_draining(b)]
    if not targets:
        return

    async def one(b):
        ad = backend_adapters.get(backend_id(b))
        fetch = getattr(ad, "fetch_running", None)
        if fetch is None:
            return
        running = await fetch(http_client, timeout=_CURRENT_LIVE_TIMEOUT)
        if running is not None and backend_id(b) in backend_running:
            backend_running[backend_id(b)] = running

    await asyncio.gather(*(one(b) for b in targets), return_exceptions=True)


def _nothing_loaded_error(alias: str, path: str) -> Optional[HTTPException]:
    """The 503 for a `current` call no candidate can take because none has a suitable
    model loaded — said as such, since 'no healthy backend' would send the caller
    looking for a fault on a backend that is up and simply idle."""
    names = [b["name"] for b in _current_backends(alias)
             if backend_healthy.get(backend_id(b)) and not is_draining(b)]
    if not names:
        return None
    return HTTPException(503, f"model '{alias}': no {adapters.current_kind_for(path)} model "
                              f"loaded on {', '.join(names)} — 'current' never loads one")


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

    Unlike resolve_routes(), this keeps unhealthy backends and not-yet-discovered
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
            present = real in backend_models.get(bid, set()) or _is_current(b, real)
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
    # x-source is the caller's to choose and lands in every stats row → bounded.
    return ((request.headers.get("x-source") or "")[:_LOG_FIELD_MAX]
            or (request.client.host if request.client else "unknown"))


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


# A ship target's parts end up on ssh/scp command lines (and in a remote shell for the
# mkdir), so they are held to plain host and path characters: `[user@]host` where
# neither part can start with `-` (an ssh OPTION), and an absolute dir without `..`.
_VOICE_HOST_RE = re.compile(r"^(?:[A-Za-z0-9_][A-Za-z0-9._-]*@)?[A-Za-z0-9_][A-Za-z0-9._-]*$")
_VOICE_DIR_RE = re.compile(r"^/[A-Za-z0-9._/-]*$")


def _voice_dir_ok(d: str) -> bool:
    return bool(_VOICE_DIR_RE.match(d)) and ".." not in d.split("/")


def parse_voice_target(t: str) -> Optional[tuple[str, str]]:
    """'user@host:/abs/dir' → (host, dir without trailing /), None if the target is not
    that shape. The ONLY gate before ship_voice_ref spawns ssh/scp with these values."""
    host, sep, hdir = (t or "").partition(":")
    hdir = hdir.rstrip("/") or ("/" if hdir else "")
    if not sep or not _VOICE_HOST_RE.match(host) or not _voice_dir_ok(hdir):
        return None
    return host, hdir


async def _scp(src: Path, host: str, remote: str) -> tuple[bool, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "--",
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
    parsed = [(t, parse_voice_target(t)) for t in targets]
    bad = [t for t, p in parsed if p is None]
    if not targets or bad or not _voice_dir_ok(vdir):
        msg = ("no/invalid ship config — targets are 'user@host:/abs/host/dir' (comma-"
               "separated) + a model-visible voice dir (e.g. /models/voices); host and "
               "dirs may only use letters, digits, . _ - and /"
               + (f" — refused: {', '.join(bad)}" if bad else ""))
        nb("err", msg)
        return False, msg
    src = Path(e.get("file") or "")
    if not src.is_file():
        nb("err", f"gateway blob missing: {src}")
        return False, f"gateway blob missing: {src}"
    remote = f"{vdir}/{src.name}"                       # what goes into `voice`
    results = {}
    for t, (host, hdir) in parsed:
        nb("run", f"upload → {host}")
        try:                                            # scp can't create dirs — mkdir -p first
            # `--` ends ssh's options before the host; the remote command is shell-parsed,
            # hence shlex.quote on top of the validated dir.
            proc = await asyncio.create_subprocess_exec(
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "--", host,
                f"mkdir -p -- {shlex.quote(hdir)}",
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


def _cost_usd(bid: str, model_id: Optional[str], in_tok: int, out_tok: int,
              cache_read: int = 0, cache_write: int = 0) -> float:
    """USD cost for a call from cached pricing (Together-style /v1/models), keyed by
    the backend id (`type:name`, same as `backend_pricing`). 0 if unknown.

    `cache_read`/`cache_write` are SUBSETS of `in_tok`; each is priced at its own
    rate where the backend lists one, else as input. Clamped so a backend reporting
    more cached than prompt tokens never books a credit."""
    if not model_id:
        return 0.0
    p = backend_pricing.get(bid, {}).get(model_id, {})
    inp = p.get("input", 0.0)
    total = max(in_tok or 0, 0)
    read = min(max(cache_read or 0, 0), total)
    write = min(max(cache_write or 0, 0), total - read)
    fresh = total - read - write
    return (fresh * inp + read * p.get("cache_read", inp) + write * p.get("cache_write", inp)
            + (out_tok or 0) * p.get("output", 0.0)) / 1_000_000


# ── Adapter wiring ─────────────────────────────────────────────────────────────
# Services handed to every backend adapter so it stays import-cycle-free and
# hot-reload-safe (the log flag is read per-call via a callable, never cached).

def _note_job_meta(job_id: str, meta: dict) -> None:
    """Best effort: a failed write (sqlite locked) must not end a job whose RunPod run
    already started — the poll and the cancel handling still have to happen."""
    try:
        if jobs._active:
            jobs.merge_meta(job_id, meta)
    except Exception as e:
        logger.warning(f"note_job_meta {job_id}: {type(e).__name__}: {e}")


def _runpod_probe_save(name: str, rec: dict) -> None:
    """Store a RunPod probe record under its backend name (worker thread)."""
    if not store.is_active():
        return
    allp = dict(store.get_setting("runpod_probe") or {})
    allp[name] = rec
    store.set_settings({"runpod_probe": allp})


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
    note_job_meta=_note_job_meta,
    runpod_probe_load=lambda name: (store.get_setting("runpod_probe") or {}).get(name)
        if store.is_active() else None,
    runpod_probe_save=_runpod_probe_save,
    runpod_account_key=lambda: provider_token("runpod"),
)


def build_backend_adapters() -> None:
    """(Re)bind one adapter per configured backend. Called at import, after every config
    reload and after every backend save, so adapters point at the current backend dicts.

    An adapter carries RUNTIME state — ComfyUI's restart cooldown (`last_restart`), the
    executor watchdog's tracking, the `/object_info` slot-type cache, the prompts its jobs
    are running (the cancel target); a cloud adapter its last credit balance. Rebuilding
    them all on every save threw that away (K19): the auto-restart cooldown reset, so a
    stuck box could be restarted again right after any unrelated edit, and a job running on
    the old instance could no longer be stopped through the new one. So a backend whose
    settings did not change KEEPS its instance, and a changed one gets a new instance that
    `adopt_state`s what still applies (see the adapters)."""
    old = dict(backend_adapters)
    backend_adapters.clear()
    for b in backends:
        bid = backend_id(b)
        prev = old.get(bid)
        if prev is not None and prev.backend == b:
            prev.backend = b                 # same settings, the current dict object
            backend_adapters[bid] = prev
            continue
        ad = make_adapter(b, adapter_ctx)
        if prev is not None:
            ad.adopt_state(prev)
        backend_adapters[bid] = ad


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
    await _refresh_loaded(alias)                       # `current` only; a no-op for every other alias
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
        if not path.startswith("/v1/messages") and anthropic_only_candidates(alias):
            raise HTTPException(404, f"model '{alias}' is served by an Anthropic backend — "
                                     "reachable through POST /v1/messages only")
        raise _nothing_loaded_error(alias, path) or HTTPException(503, f"No healthy backend for model '{alias}'")
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
                _note_fault(backend, "call", "load_failed", f"{real_model}: {_resp_snippet(resp)}", 502)
                last_resp = resp
                continue
            # A 5xx the client gets as-is is still the BACKEND failing — the call log
            # keeps only the status, the fault log keeps what the backend said.
            if getattr(resp, "status_code", 0) >= 500:
                _note_fault(backend, "call", "upstream", f"{real_model}: {_resp_snippet(resp)}",
                            resp.status_code)
            return resp
        except HTTPException:
            raise                                 # already a deliberate answer
        except httpx.ReadTimeout as e:
            # Connected and sent, then no answer within the read budget: the backend is
            # most likely STILL generating. On a paid backend a failover buys the same
            # answer twice, so it ends here; a local one only wastes its own compute, and
            # a hung llama-swap load is exactly when another box helps.
            _note_fault(backend, "call", "timeout", f"{real_model}: {_err_text(e)}")
            if _bills_while_generating(backend):
                logger.warning(f"✗ [{backend['name']}] no answer in {_READ_BUDGET_S:g} s — "
                               "billed backend, not retried elsewhere")
                raise HTTPException(504, f"backend '{backend['name']}' did not answer within "
                                         f"{_READ_BUDGET_S:g} s — not retried on another backend, "
                                         "because a paid or subscription backend may still be "
                                         "generating (and billing) this request")
            logger.warning(f"✗ [{backend['name']}] {_err_text(e)} — trying next")
            last_error = e
        except httpx.PoolTimeout:
            # The GATEWAY's shared connection pool is exhausted — no backend saw the
            # call. It is the same pool for every candidate, so a failover only waits
            # the pool timeout again per backend, and charging each one a "timeout"
            # fault blames a fleet for the gateway's own load.
            logger.warning(f"✗ connection pool exhausted — {alias} refused "
                           f"({adapters._CHAT_TIMEOUT.pool:g} s wait for a free connection)")
            raise HTTPException(503, "gateway busy: no free upstream connection in the "
                                     f"shared pool within {adapters._CHAT_TIMEOUT.pool:g} s",
                                headers={"Retry-After": "5"})
        except httpx.TransportError as e:
            # Every transport failure surfaces BEFORE the client saw a byte (a stream is
            # opened, headers and status read, before the adapter answers): connect
            # errors, a pooled keep-alive connection the backend had closed
            # (RemoteProtocolError), a reset (ReadError/WriteError). All fail over.
            logger.warning(f"✗ [{backend['name']}] {_err_text(e)} — trying next")
            # Failed over — the call may still end as a 200 elsewhere, which is exactly
            # why this is invisible anywhere but here.
            _note_fault(backend, "call", _call_fault_kind(e), f"{real_model}: {_err_text(e)}")
            last_error = e
        except Exception as e:
            # Not a transport failure: a bug, or a backend answer the adapter could not
            # handle. Retrying elsewhere would most likely reproduce it, so it ends here —
            # as a clean 502 the error handler renders in the endpoint's own shape (a raw
            # 500 reads blank in Claude Code), and in the fault log.
            logger.exception(f"✗ [{backend['name']}] dispatch failed")
            _note_fault(backend, "call", "error", f"{real_model}: {type(e).__name__}: {_err_text(e)}")
            raise HTTPException(502, f"backend '{backend['name']}' failed: "
                                     f"{type(e).__name__}: {_err_text(e)}")
    if last_resp is not None:                     # every candidate failed to load → the real 502
        return last_resp
    raise HTTPException(503, f"All backends failed: {_err_text(last_error)}")


_READ_BUDGET_S = adapters._CHAT_TIMEOUT.read


def _bills_while_generating(backend: dict) -> bool:
    """Does a request this backend gave up on still cost something? `paid` backends
    bill per token, and an `anthropic` one keeps generating against the subscription
    quota or the API key's bill. It is deliberately NOT marked `paid` for that: the
    scheduler orders unpaid before paid, and a flat subscription tagged `paid` would
    sort behind every unpaid candidate of a mixed alias — Claude Code sessions moving
    to per-token OpenRouter or a local model. Only this no-failover rule needs it."""
    return bool(backend.get("paid")) or backend.get("type") == "anthropic"


def _call_fault_kind(e: BaseException) -> str:
    """Fault-log kind for a transport error on a forwarded call. A connection that was
    established and then lost (reset, closed without a response) is `connection_lost`
    — the backend dropped the call — while never connecting stays `unreachable`, which
    the fault log ignores (a backend that is off is a state, not an error)."""
    if isinstance(e, (httpx.NetworkError, httpx.ProtocolError)) and not isinstance(e, httpx.ConnectError):
        return "connection_lost"
    return _classify_error(e)["kind"]


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

    def type_key(e):
        # A `current` waiter needs no particular model — whatever runs here is its
        # type, so it counts as the no-reload match it is.
        if any(backend_id(cb) == bid for cb in _current_backends(e["alias"])):
            return backend_last_key.get(bid)
        return alias_entry(e["alias"], bname)[0] or e["alias"]

    return scheduler.designated_taker(
        pool,
        can_serve=lambda e: any(backend_id(rb) == bid for rb, _ in
                                resolve_routes(e["alias"], e["path"])[0]),
        type_key=type_key,
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
            await _refresh_loaded(alias)               # `current`: the loaded model may have changed while parked
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
                raise _nothing_loaded_error(alias, path) or HTTPException(
                    503, f"No healthy backend for model '{alias}'")
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
    allow = _expand_grants((user.get("models") if user else None) or [])
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
            # `<backend>/current` (llama-swap only) — always prefixed, a bare `current`
            # would be ambiguous. No context_length: it changes with every swap.
            cur = f"{bname}/{adapters.CURRENT_MODEL}"
            if _is_current(backend, adapters.CURRENT_MODEL) and cur not in seen and visible({cur, bname}):
                seen.add(cur)
                data.append({"id": cur, "object": "model", "created": now, "owned_by": bname})
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
            if alias not in seen and visible({alias, GRANT_ALL_IMAGE}):
                seen.add(alias)
                data.append({"id": alias, "object": "model", "created": now, "owned_by": "ai-hub (image)"})

    return {"object": "list", "data": data}


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str, authorization: Optional[str] = Header(None)):
    user = authenticate(authorization)
    # The same grant the request path enforces; outside it the answer is the unknown-model
    # 404, so a restricted key cannot probe what exists beyond its allow-list.
    if user is not None and not _model_allowed(user, model_id):
        raise HTTPException(404, f"Model '{model_id}' not found")
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
        if b is not None and _is_current(b, bare):
            return {"id": model_id, "object": "model", "created": now, "owned_by": bname}
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
                                     media_type="text/event-stream",
                                     headers=adapters._gateway_headers(resp.headers))
        err = _dispatch_json(resp)
        # The upstream's retry-after is an instruction to the caller, not diagnostics —
        # it must survive this re-raise like every other response rebuild (see
        # adapters._ratelimit_headers).
        raise HTTPException(resp.status_code, (json.dumps(err) or "")[:500],
                            headers=adapters._ratelimit_headers(resp.headers) or None)
    chat_resp_json = _dispatch_json(resp)
    if resp.status_code >= 400:
        raise HTTPException(resp.status_code, (json.dumps(chat_resp_json) or "")[:500],
                            headers=adapters._ratelimit_headers(resp.headers) or None)
    return JSONResponse(chat_to_responses(chat_resp_json), status_code=resp.status_code,
                        headers=adapters._gateway_headers(resp.headers))


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
    # by resolve_routes(). Embedding responses carry usage.prompt_tokens only
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


def _gen_routes(alias: str, gated: Optional[list] = None) -> tuple[list, list]:
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

    A candidate whose backend has not synced this alias's models (`modelsync_gate`, a
    managed host's ComfyUI) leaves BOTH lists: such an alias then 503s at once naming
    the sync instead of parking for hours behind it. `gated`, when given, collects
    (backend name, reason) of every such candidate — the 503's text.

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
        why = modelsync_gate(b, alias)
        if why:
            if gated is not None:
                gated.append((b.get("name"), why))
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


def get_gen_routes(alias: str) -> list[tuple[dict, dict]]:
    """Every allowed + healthy candidate of _gen_routes(), busy ones included — for
    callers that ask what an alias CAN run on (image slots, LoRA listing, the chain's
    successor check), not what is free right now."""
    return _gen_routes(alias)[1]


def _force_filter(routes: list, force: str) -> list:
    """Keep only the force-pinned backend's candidates (no-op without a pin)."""
    return [r for r in routes if r[0].get("name") == force] if force else routes


def _entry_can_use(entry: dict, backend: dict, memo: Optional[dict] = None) -> bool:
    """May this waiting generation job run on `backend` right now? Mirrors the filters
    the job applies in its own poll: its own `exclude` list, the force pin, LoRA
    eligibility, and the alias actually routing to this backend while it is free.

    `exclude` (backend NAMES) is how a waiter declares a candidate it would refuse for
    a reason the routing tables cannot show — the chain maintains it from its per-pass
    `usable()` verdicts and its `tried` failover set. Without it a chain could be
    designated for a backend it will never claim and, once overdue, hold that idle
    backend against every other waiter until its own park deadline.

    `memo` ({alias: ready backend ids}) is shared by one designation pass: `_gen_routes`
    reads and JSON-parses the alias's candidates — workflow JSON included — from the
    store, and a pass asks this for every waiter × every free backend (P11).

    Blocking (store read via _gen_routes) — call via asyncio.to_thread from async code."""
    if backend.get("name") in (entry.get("exclude") or ()):
        return False
    if entry.get("force") and backend.get("name") != entry["force"]:
        return False
    if entry.get("eligible") is not None and backend.get("name") not in entry["eligible"]:
        return False
    memo = {} if memo is None else memo
    ids = memo.get(entry["alias"])
    if ids is None:
        ready, _allc = _gen_routes(entry["alias"])
        ids = memo[entry["alias"]] = {backend_id(b) for b, _cand in ready}
    return backend_id(backend) in ids


def _gen_waiting_pool() -> list:
    """Snapshot of the media queue as the scheduler sees it (`_gen_waiting` mutates
    across awaits). Entries that hold a backend slot are out: a claimant stays
    registered until its job is done, but it is no longer waiting and must not shadow
    the others out of a second free backend."""
    return [e for e in _gen_waiting if not e.get("claimed")]


def _designated_gen_waiter(backend: dict, pool: list, now: Optional[float] = None,
                           memo: Optional[dict] = None):
    """The queued generation job this free backend belongs to, or None.

    Media twin of _designated_waiter (spec 2026-09-01, "Designated taker"): overdue
    jobs first (oldest), else one needing the alias the backend last ran — one alias is
    one workflow, i.e. the model set already in VRAM — else the oldest job it can
    serve. THE one place the media designation is computed, so the poll gate and the
    fresh-arrival gate cannot drift apart. Blocking — via asyncio.to_thread."""
    memo = {} if memo is None else memo          # one store read per alias, not per waiter
    return scheduler.designated_taker(
        pool,
        can_serve=lambda e: _entry_can_use(e, backend, memo),
        type_key=lambda e: e["alias"],
        last_key=backend_last_key.get(backend_id(backend)),
        now=time.monotonic() if now is None else now,
        max_wait_s=affinity_max_wait_s)


def _may_claim_gen(entry: dict, backend: dict, memo: Optional[dict] = None) -> bool:
    """Is this free backend THIS waiting job's to claim? A lone waiter always passes.
    Blocking — via asyncio.to_thread."""
    pool = _gen_waiting_pool()
    if len(pool) <= 1:                       # only us waiting → nothing to yield to
        return True
    return _designated_gen_waiter(backend, pool, memo=memo) is entry


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
    memo: dict = {}                          # the routing tables cannot change within this pass
    for i, (b, _cand) in enumerate(ready):
        if _may_claim_gen(entry, b, memo):
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
    # The same rule _client_param_refusal applies to `params`: a list or object is not a
    # workflow value. The injector skips it with a WARNING, so the job would run `done`
    # on the workflow's DEFAULT prompt instead of failing.
    for k, v in inputs.items():
        if isinstance(v, (list, tuple, dict)):
            raise HTTPException(400, f"`{k}` must be a single value — a list or object is "
                                     f"not a workflow value")
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


def _gen_fault_kind(e: BaseException) -> str:
    """`_fault_label` as a short fault-log kind (same distinctions, same order)."""
    if isinstance(e, adapters.CloudNoCredits):
        return "no_credits"
    if isinstance(e, adapters.CloudBusy):
        return "rate_limit"
    if isinstance(e, adapters.CloudTaskRetryable):
        return "vendor_failed"
    if isinstance(e, TimeoutError):
        return "max_wait"
    if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "unreachable"         # never connected: the backend is off — not a fault
    if isinstance(e, httpx.TimeoutException):
        return "timeout"
    # Connected, then lost it — ComfyUI gone mid-execution (the adapter's ConnectionError
    # after disconnect_grace), a dropped read: the box died WHILE working. A real error,
    # unlike a backend that was simply switched off.
    return "connection_lost"


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


def _billed_cloud_task(cand: dict, trace: dict, e: BaseException,
                       prior_task: Optional[str] = None) -> Optional[str]:
    """The vendor task a failed CLOUD attempt leaves behind — named for the job row — or
    None when repeating the attempt (self-retry, next candidate) cannot buy anything twice.

    The failover arm exists for faults that happen BEFORE work starts (a refused connect,
    a full vendor queue, no credits). A cloud candidate also raises failover-class errors
    AFTER its paid task was created: `_poll` gives up on a vendor unreachable past
    `disconnect_grace` (ConnectionError) or on `max_wait` (TimeoutError, whose own text
    says the task is "still running"), and a create POST whose ANSWER was lost (read
    timeout, dropped connection) may well have created the task. Every one of those used to
    re-create the task on the next attempt and pay for it again (review 2026-09-18, K1).

    `prior_task` is the task id the request carried before this attempt (the chain reuses
    one request across self-retries): a task from an EARLIER attempt that the vendor failed
    unbilled must not make a later, never-created attempt look billed.
    `CloudTaskRetryable` stays retryable — `_poll` raises it only for a vendor-side fault
    that consumed no credits, which is the one case a re-run is meant for."""
    if trace.get("runpod"):
        # A RunPod run of a WORKFLOW candidate (cloud_kind is None for it). The adapter
        # sets `runpod_settled` only once RunPod confirms the job is no longer running —
        # anything else may still be billing, and a re-run would pay for it twice.
        if trace.get("create_unconfirmed"):
            return "a RunPod job (the /run request was sent but its answer was lost)"
        rp = trace.get("runpod_job_id")
        if rp and not trace.get("runpod_settled"):
            return f"RunPod job {rp}"
        return None
    if not adapters.cloud_kind(cand) or isinstance(e, adapters.CloudTaskRetryable):
        return None
    vendor = adapters.cloud_module(adapters.cloud_kind(cand)).VENDOR
    tid = trace.get("cloud_task_id")
    if tid and tid != prior_task:
        return f"{vendor} task {tid}"
    if trace.get("create_unconfirmed"):
        return (f"a {vendor} {trace.get('endpoint') or ''} task (the create request was sent "
                f"but its answer was lost)").replace("  ", " ")
    return None


def _billed_final_msg(e: BaseException, billed: str) -> str:
    """The job-row error for a cloud failure that must not be repeated (_billed_cloud_task)."""
    return (f"{_fault_label(e)}: {_err_text(e)} — {billed} may still be running (and "
            f"billing) at the vendor; not re-created on another attempt, which would pay "
            f"for it twice")


def _gen_fail_meta(attempts: int, cloud_trace: dict) -> Optional[dict]:
    """The meta a failed generation job carries: the retry count (kept visible, runbook B)
    plus whatever a cloud candidate had already created. The cloud keys are the SAME ones
    the success meta uses, so `admin._cloud_table` and the job view render a failed run
    with no special case. None when there is nothing to say."""
    meta = dict(cloud_trace)
    if attempts > 1:
        meta["attempts"] = attempts
    return meta or None


def _note_cancelled_trace(job_id: str, attempts: int, trace: dict) -> None:
    """Put a cancelled job's cloud trace on its row (F1): the vendor finishes and bills a
    created task whatever the gateway does, and a cancelled row without the task id leaves
    no way to find it. Nothing to write → nothing written; merged, never a status change
    (cancel_generation owns that). Synchronous — called from a CancelledError handler."""
    if not trace:
        return
    try:
        jobs.merge_meta(job_id, _gen_fail_meta(attempts, trace) or {})
    except Exception as e:                      # a cancel must never turn into a crash
        logger.warning(f"job {job_id}: could not record the cancelled cloud task: {e}")


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


async def _run_job(job_id: str, alias: str, candidates: list, build_req,
                   state: Optional[dict] = None) -> bool:
    """Run a generation job, failing over to the next candidate on connection-type
    errors. A backend with `self_retries: n` gets n extra attempts on ITSELF first
    (runbook B: for sporadic driver faults the same host is the cheapest second
    try — no model re-load elsewhere, same success odds as attempt one). Its ONE
    job slot is held across the repeats, so no parked job slips in between.
    An execution error (the backend ran the prompt and it blew up) moves to the next
    candidate too, but never repeats on the SAME backend — see the `except Exception`
    arm for why that is not the same thing as a connection failover, and why a CLOUD
    candidate is excluded from it. Stops at the first success.

    A candidate is CLAIMED only if it is not busy at that very moment — the busy check
    and `_inflight_inc` run with no await between them (the dispatch invariant). The list
    was computed before several awaits (job creation, the designation lookup) and a
    failover target was never checked at all, so either could push a backend past its
    `max_concurrent` (K5). A busy candidate is skipped; if nothing ran to an end and some
    candidate was skipped as busy, this returns False and the caller parks the job again,
    passing `state` back in so attempts, the backends already tried and the execution
    faults carry over. True = the job row holds its outcome (or the job was cancelled)."""
    st = state if state is not None else {}
    last = st.get("last")
    attempts = st.get("attempts", 0)
    tried: set = st.setdefault("tried", set())      # bids that ran — never claimed twice
    busy_skipped = False
    # (bid, name, error) of candidates that ran the prompt and failed. Kept until the
    # job ends because a fault only counts as the BACKEND's once a later candidate has
    # succeeded — until then it is indistinguishable from a broken request.
    exec_faults: list = st.setdefault("exec_faults", [])
    # What a CLOUD candidate had already created when it failed (task id, endpoint, the
    # request summary). A failed run returns no GenOutput, so these facts reach the job row
    # only from here — and they are the ones a cloud failure is diagnosed with: without
    # them the task id survives only inside the error TEXT and everything else not at all
    # (measured 2026-09-03, job 9cf448115b4b). Last writer wins: the candidate the job
    # actually died on is the one worth describing.
    cloud_trace: dict = st.get("cloud_trace") or {}
    for backend, cand in candidates:
        bid = backend_id(backend)
        adapter = backend_adapters.get(bid)
        if adapter is None or bid in tried:
            continue
        try:
            tries = 1 + max(0, int(backend.get("self_retries") or 0))
        except (TypeError, ValueError):
            tries = 1                          # malformed config value → no self-retry
        if backend_busy(backend):              # checked HERE, right before the claim (K5)
            busy_skipped = True
            continue
        _inflight_inc(bid)                     # hold ONE slot across all self-retries
        tried.add(bid)
        # `try` right after the claim: an await between inc and try leaked the slot for
        # good when the job was cancelled in it (K4).
        try:
            if not st.get("running"):
                if not await asyncio.to_thread(jobs.set_status, job_id, "running"):
                    return True                # cancelled while it waited — never start it
                st["running"] = True
            # The job row was stamped with the FIRST candidate at creation; re-point it at
            # the backend actually claiming it. A parked job routinely lands somewhere else
            # (a different backend freed first, or one came back while it waited), and a
            # row naming the wrong backend sends you reading the wrong ComfyUI's log. Same
            # reason the chain re-points at claim and hand-off.
            await asyncio.to_thread(jobs.set_backend, job_id, backend["name"])
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
                    _bg(_free_comfy_vram(backend, "job done"))
                    return True
                except _GEN_FAILOVER_ERRORS as e:
                    _record_gen_attempt(bid, conn_fail=True)
                    last = e
                    cloud_trace = _cloud_trace_of(req) or cloud_trace
                    # both fail over, but they are different faults — name them apart so
                    # the log points at the workflow, not the network (see _gen_exhausted_msg)
                    what = _fault_label(e)
                    # Recorded on EVERY attempt: a self-retry that then succeeds leaves a
                    # clean `done` row, and the crash that forced it is only visible here.
                    _note_fault(backend, "job", _gen_fault_kind(e), f"{alias}: {what}: {_err_text(e)}")
                    billed = _billed_cloud_task(cand, _cloud_trace_of(req), e)
                    if billed:
                        # A paid task exists (or may): neither a self-retry nor the next
                        # candidate may create another one — see _billed_cloud_task.
                        logger.warning(f"✗ job {job_id} [{backend['name']}] {what} after "
                                       f"{billed} was created — final, not re-run")
                        await asyncio.to_thread(jobs.fail, job_id, _billed_final_msg(e, billed),
                                                _gen_fail_meta(attempts, cloud_trace))
                        return True
                    if attempt < tries:
                        logger.warning(f"✗ job {job_id} [{backend['name']}] {what} "
                                       f"({type(e).__name__}: {e}) — retrying same backend "
                                       f"(self-retry {attempt}/{tries - 1})")
                        await _wait_backend_up(backend)
                        continue
                    logger.warning(f"✗ job {job_id} [{backend['name']}] {what} "
                                   f"({type(e).__name__}: {e}) — failing over")
                    _bg(_free_comfy_vram(backend, "job failover"))
                except adapters.ComfyPromptInterrupted as e:
                    # Somebody stopped OUR prompt on the backend (an operator in ComfyUI's
                    # own UI). A decision, not a fault: no failover — that would run what was
                    # just stopped — and no execution fault, which could quarantine a
                    # healthy backend for 15 minutes.
                    logger.warning(f"✗ job {job_id} [{backend['name']}] {_err_text(e)}")
                    await asyncio.to_thread(jobs.fail, job_id, _err_text(e),
                                            _gen_fail_meta(attempts, cloud_trace))
                    return True
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
                    _note_fault(backend, "job", "execution", f"{alias}: {_err_text(e)}")
                    cloud_trace = _cloud_trace_of(req) or cloud_trace
                    billed_x = _billed_cloud_task(cand, _cloud_trace_of(req), e)
                    if adapters.cloud_kind(cand) or billed_x:
                        # A cloud task is BILLED (and a RunPod job that is not settled may
                        # still be running). Whatever failed here may have happened
                        # after the paid task was created, and re-running the job on the
                        # next candidate would buy the same mesh twice — the invariant
                        # tripo.py/adapters.py go out of their way to preserve. Final.
                        logger.warning(f"✗ job {job_id} [{backend['name']}] failed: {_err_text(e)}")
                        await asyncio.to_thread(
                            jobs.fail, job_id,
                            _billed_final_msg(e, billed_x) if billed_x else _err_text(e),
                            _gen_fail_meta(attempts, cloud_trace))
                        return True
                    exec_faults.append((bid, backend["name"], e))
                    last = e
                    logger.warning(f"✗ job {job_id} [{backend['name']}] execution failed: "
                                   f"{_err_text(e)} — trying the next backend")
                    _bg(_free_comfy_vram(backend, "job failure"))
                    break                      # next candidate; never the same backend
                except asyncio.CancelledError:
                    # Cancelled by the user (cancel_generation marks the row itself). The
                    # except arms above never see a cancel — it is a BaseException — so a
                    # cloud task that was already created and billed would leave the row
                    # with no task id at all. Written synchronously on purpose: no await
                    # while being cancelled.
                    _note_cancelled_trace(job_id, attempts, _cloud_trace_of(req) or cloud_trace)
                    raise
        finally:
            _inflight_dec(bid)
    if busy_skipped:
        # A candidate was taken by someone else between our pick and our claim: park
        # again for it (the caller re-resolves) instead of reporting the job as exhausted.
        st.update(last=last, attempts=attempts, cloud_trace=cloud_trace)
        return False
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
        return True
    await asyncio.to_thread(jobs.fail, job_id, _gen_exhausted_msg(last),
                            _gen_fail_meta(attempts, cloud_trace))
    return True


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


def _chain_successor_on(backend: dict, succ_alias: str) -> tuple:
    """(successor candidate on `backend`, skip reason) for a chain's PATH relay, which
    pins stage 2 to stage 1's backend. Store first, `image_models` config fallback — the
    same order as _gen_routes. It reads the store directly, not through _gen_routes, so
    the model-sync gate is applied here itself: a Thunder box ready for stage 1 must not
    be handed a stage 2 whose weights are still downloading.
    Blocking (store read) — call via asyncio.to_thread from async code."""
    cands = store.get(succ_alias) if store.is_active() else None
    if cands is None:
        cands = image_models.get(succ_alias, [])
    s2 = next((c for c in cands if c.get("backend") == backend["name"]), None)
    if s2 is None:
        return None, f"successor '{succ_alias}' is not configured for backend '{backend['name']}'"
    why = modelsync_gate(backend, succ_alias)
    if why:
        return None, f"successor: {why}"
    return s2, None


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
        """(successor candidate on `backend`, skip reason) for the path relay."""
        return await asyncio.to_thread(_chain_successor_on, backend, succ_alias)

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
            adapter = backend_adapters.get(bid)
            if adapter is None:                  # backend removed/re-saved since usable() looked
                await asyncio.sleep(0.5)         # (K10) — the next pass re-judges it
                continue
            # Resolve the stage-2 (successor) backend + candidate. Path relay pinned it to
            # stage 1's backend above (shared disk). Upload relay picks the successor
            # alias's best candidate — preferring stage 1's backend if it is itself one
            # (keeps the fast in-process path).
            if relay == "upload":
                cands2 = await asyncio.to_thread(get_gen_routes, succ_alias)   # successor's allowed+healthy backends
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
            s1_prior_task = None                        # see the stage-1 retry loop
            req1 = req2 = None                          # per CANDIDATE: a later stage-1 failure must
                                                        # not report the previous pass's tasks
            # `try` right after the claim — the row updates below are awaits, and a cancel
            # landing in one of them leaked the slot for good when they sat before it (K4).
            try:
                if not await asyncio.to_thread(jobs.set_status, job_id, "running"):
                    return                              # cancelled meanwhile (finally frees the slot)
                await asyncio.to_thread(jobs.set_backend, job_id, backend["name"])   # cancel targets the LIVE backend
                await asyncio.to_thread(jobs.set_stage, job_id, "1/2")   # multi-stage progress → "running 1/2"
                _apply_seconds(params, stage1_cand)     # no-op re-derive (endpoint validated it)
                # ── Stage 1: mesh (pin the export filename; ignore its own outputs) ──
                req1 = NormalizedRequest(
                    alias=alias, real_model=stage1_cand.get("model"),
                    inputs=inputs, params=params,
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
                try:
                    s1_tries = 1 + max(0, int(backend.get("self_retries") or 0))
                except (TypeError, ValueError):
                    s1_tries = 1                        # malformed config value → no self-retry (K15)
                for s1_attempt in range(1, s1_tries + 1):
                    gen_attempts += 1
                    # req1 is reused across self-retries, so its trace may already name an
                    # earlier attempt's (unbilled, retryable) task — only a NEW one counts.
                    s1_prior_task = req1.cloud_trace.get("cloud_task_id")
                    try:
                        t0 = time.monotonic()
                        out1 = await adapter.generate(req1)
                        _record_gen_attempt(bid, conn_fail=False)
                        _note_gen_speed(alias, bid, time.monotonic() - t0)
                        break
                    except _GEN_FAILOVER_ERRORS as e:
                        if s1_attempt >= s1_tries or _billed_cloud_task(
                                stage1_cand, _cloud_trace_of(req1), e, s1_prior_task):
                            raise                # outer handler records + fails over (or, for a
                                                 # billed cloud task, ends the chain)
                        _record_gen_attempt(bid, conn_fail=True)
                        _note_fault(backend, "job", _gen_fault_kind(e),
                                    f"{alias} (chain stage 1): {_fault_label(e)}: {_err_text(e)}")
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
                    inputs={}, params={},
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
                    _bg(_free_comfy_vram(backend, "chain stage 1 done"))
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
                        # PIL on textures — off the event loop (P9)
                        await asyncio.to_thread(normalize_delivery, blobs, chain_rig,
                                                stage1_cand.get("texture_format"))
                        warnings = await asyncio.to_thread(validate_delivery, blobs, chain_rig)   # raises → job fails clearly
                        if warnings:
                            meta["warnings"] = warnings
                await asyncio.to_thread(jobs.complete, job_id, blobs, meta)
                if log_per_call:
                    route = (f"{backend['name']}→{backend2['name']}" if cross else backend["name"])
                    logger.info(f"✓ chain job {job_id} on [{route}] ({alias}→{succ_alias}) "
                                f"— {len(blobs)} artifact(s)")
                _bg(_free_comfy_vram(active, "chain done"))
                return
            except _GEN_FAILOVER_ERRORS as e:
                _record_gen_attempt(backend_id(active), conn_fail=True)
                _note_fault(active, "job", _gen_fault_kind(e),
                            f"{alias}→{succ_alias}: {_fault_label(e)}: {_err_text(e)}")
                if s1_done:                            # mesh already relayed — a stage-2 loss is final
                    # `_err_text`: this is the branch an httpx WriteTimeout lands in, and
                    # its str() is empty — the row used to read "chain failed: " (2026-09-02).
                    logger.warning(f"✗ chain job {job_id} [{active['name']}] ({alias}→{succ_alias}) "
                                   f"{_fault_label(e)}: {_err_text(e)}")
                    await asyncio.to_thread(jobs.fail, job_id, f"chain failed: {_err_text(e)}",
                                            fail_meta())
                    _bg(_free_comfy_vram(active, "chain failure"))
                    return
                billed = _billed_cloud_task(stage1_cand, _cloud_trace_of(req1), e, s1_prior_task)
                if billed:
                    # a paid stage-1 task exists — failing over would buy it again
                    logger.warning(f"✗ chain job {job_id} stage 1 [{backend['name']}] "
                                   f"{_fault_label(e)} after {billed} was created — final")
                    await asyncio.to_thread(jobs.fail, job_id,
                                            f"chain stage 1: {_billed_final_msg(e, billed)}",
                                            fail_meta())
                    return
                logger.warning(f"✗ chain job {job_id} stage 1 [{backend['name']}] {_fault_label(e)} "
                               f"({type(e).__name__}: {e}) — failing over")
                tried.add(backend["name"])
                _bg(_free_comfy_vram(backend, "chain stage-1 failure"))
                continue
            except Exception as e:
                _note_fault(active, "job", "execution", f"{alias}→{succ_alias}: {_err_text(e)}")
                logger.warning(f"✗ chain job {job_id} [{active['name']}] ({alias}→{succ_alias}) "
                               f"failed: {_err_text(e)}")
                await asyncio.to_thread(jobs.fail, job_id, f"chain failed: {_err_text(e)}", fail_meta())
                _bg(_free_comfy_vram(active, "chain failure"))
                return
            except asyncio.CancelledError:
                # cancelled by the user: keep a created (billed) cloud task findable (F1)
                _note_cancelled_trace(job_id, 0, fail_meta() or {})   # carries its own attempts
                raise
            finally:
                if held is not None:
                    _inflight_dec(held)
    finally:
        try:
            _gen_waiting.remove(entry)
        except ValueError:
            pass


_gen_tasks: dict = {}                       # job_id → asyncio.Task (for cancellation)


def _spawn_gen(job_id: str, coro) -> asyncio.Task:
    """Run a generation coroutine as a tracked background task so it can be cancelled.
    The done callback is the job's last line of defence: a worker that dies of an
    exception nobody anticipated left its row `queued`/`running` until the next process
    restart (reconcile_orphans) — a job that looks alive, forever (K10). It is marked
    failed with the error instead; a row that already has its outcome is untouched
    (jobs keeps terminal states final), and a cancelled task was marked by the cancel."""
    t = asyncio.create_task(coro)
    _gen_tasks[job_id] = t

    def _done(task: asyncio.Task) -> None:
        if _gen_tasks.get(job_id) is task:
            _gen_tasks.pop(job_id, None)
        if task.cancelled() or task.exception() is None:
            return
        e = task.exception()
        logger.error(f"job {job_id}: generation worker crashed: {type(e).__name__}: {_err_text(e)}",
                     exc_info=e)
        try:
            jobs.fail(job_id, f"internal error: {type(e).__name__}: {_err_text(e)}")
        except Exception as e2:                  # the store itself is what failed — nothing left to do
            logger.error(f"job {job_id}: could not mark the crashed job failed: {e2}")
    t.add_done_callback(_done)
    return t


async def _run_gen_sync(job_id: str, coro) -> None:
    """Run a SYNC generation as a tracked task and wait for it. Sync jobs used to run
    inline in the request handler, invisible to `_gen_tasks` — so a cancel marked the row
    failed while the worker went on to fail over or finish (K3). `asyncio.wait` never
    raises the task's outcome into the handler: the JOB row owns it (_job_view renders
    it), and a client that disconnects no longer takes a running job down with it."""
    await asyncio.wait({_spawn_gen(job_id, coro)})


async def cancel_generation(job_id: str) -> bool:
    """Cancel a queued/running generation job: mark it failed FIRST (the job store keeps
    a terminal state final, so nothing the worker does afterwards can mark it done), then
    cancel the worker task — its adapter stops its OWN ComfyUI prompt on the way out
    (targeted: a queued job never interrupts somebody else's run; a cloud task API has
    nothing to stop, the vendor finishes and bills the task, whose id the worker records).
    Without a worker task in this process, `adapter.cancel(job_id)` is the targeted
    fallback. Returns False if the job is already finished/unknown."""
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
    await asyncio.to_thread(jobs.fail, job_id, "cancelled by user")
    t = _gen_tasks.get(job_id)
    if t and not t.done():
        t.cancel()
        # Let the worker unwind — stop its prompt, release its slot — before the free
        # below looks at the backend: with the slot still held it would skip the free.
        await asyncio.wait({t}, timeout=15.0)
    elif adapter is not None:
        await adapter.cancel(job_id)
    if b:
        _bg(_free_comfy_vram(b, "cancel"))     # no-op for non-ComfyUI (step 3)
    return True


async def _run_gen_now(job_id, alias, force, routes, build_req, eligible: Optional[set] = None):
    """A job that found a free backend at request time: run it on `routes`, and if the
    backend was claimed by someone else before this job could (_run_job returns False),
    queue it like any parked job — with the failover bookkeeping carried over."""
    st: dict = {}
    if not await _run_job(job_id, alias, routes, build_req, st):
        await _run_gen_parked(job_id, alias, force, build_req, eligible, st)


async def _run_gen_parked(job_id, alias, force, build_req, eligible: Optional[set] = None,
                          run_state: Optional[dict] = None):
    """Hold a generation job until a backend frees (polls backend-busy), then run it —
    so a busy backend queues instead of 503'ing (async/playground). `eligible` keeps
    the LoRA constraint through the park: the job waits for a LoRA-capable backend
    instead of spilling to whichever frees first.

    A poll that finds ZERO candidates is NOT an immediate fail: the job only parked
    because it HAD candidates (all busy), so an empty poll is a transient health flap
    (a busy ComfyUI drops its /object_info discovery poll mid-generation and is briefly
    marked DOWN). Ride it out for `park_health_grace_s`; only if EVERY candidate stays
    gone that long is the alias treated as having no healthy backend.

    `run_state` is `_run_job`'s bookkeeping across claims: a claim can be lost to a
    faster job between the designation and the claim (_run_job then returns False), and
    the job simply keeps parking — the backends it already ran on stay tried."""
    run_state = run_state if run_state is not None else {}
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
            tried = run_state.get("tried")
            if tried:                            # back from a lost claim: never re-run a backend
                ready = [r for r in ready if backend_id(r[0]) not in tried]
                allc = [r for r in allc if backend_id(r[0]) not in tried]
                if not allc:                     # everything has run → report how it ended
                    await _run_job(job_id, alias, [], build_req, run_state)
                    return
            # A free candidate is only ours if the scheduler designates it for us;
            # otherwise it belongs to another waiter and we keep parking (2 s poll).
            i = (await asyncio.to_thread(_designated_gen_index, entry, ready)) if ready else None
            if i is not None:
                entry["claimed"] = True          # holding a slot → out of the waiting pool
                # designated candidate first, the rest stay as the failover tail
                if await _run_job(job_id, alias, [ready[i]] + ready[:i] + ready[i + 1:],
                                  build_req, run_state):
                    return
                entry["claimed"] = False         # lost the claim race → back in the queue
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
    gated: list = []
    ready, allc = await asyncio.to_thread(_gen_routes, alias, gated)
    ready, allc = _force_filter(ready, force), _force_filter(allc, force)
    if not allc:
        # A backend that is up but has not synced this alias's models is the reason worth
        # naming — "no healthy backend" would send the caller looking at a box that is fine.
        # A force pin filters these too: a pin elsewhere is not about the Thunder box.
        why = [w for name, w in gated if not force or name == force]
        raise HTTPException(503, "; ".join(why) if why else
                            f"No healthy backend for generation model '{alias}'"
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


def _params_trusted(request) -> bool:
    """Whether this generation request may name BACKEND paths in its params: an admin
    key (gate_request marks it `gw_admin`), or bootstrap-open mode, where everything is
    open anyway. The console needs no rule of its own: its playground reaches
    /v1/generations as a self-call carrying the logged-in admin's key (admin._self_api)."""
    if not users and not api_key:
        return True
    return bool(getattr(getattr(request, "state", None), "gw_admin", False))


def _numberish(v: str) -> bool:
    try:
        float(v.strip())
        return True
    except ValueError:
        return v.strip().lower() in ("true", "false")


def _client_param_refusal(params: dict, wf_maps: list, trusted: bool) -> Optional[str]:
    """Why these client params must not reach the workflows in `wf_maps` [(wf, mapping)]
    — the alias's own and, for a chain, its successor's (params are threaded there by
    label) — or None. Only MAPPED names are judged; unknown ones are ignored downstream.

    A list or object is never a value (in ComfyUI's API format a list is a LINK), and a
    mapped file field (adapters.is_file_param) is a path on the backend box: from a
    plain user that reads any file ComfyUI can — another job's output included — so it
    needs `trusted` or the mapping entry's `client_path: true`; `files` is the way in."""
    flat = dict(params or {})
    extra = flat.get("extra")
    if isinstance(extra, dict):
        flat.update(extra)
    for wf, mapping in wf_maps:
        for p, m in (mapping or {}).items():
            m = m or {}
            lbl = (m.get("label") or "").strip()
            for name in {p, lbl} - {""}:
                if name not in flat:
                    continue
                v = flat[name]
                if isinstance(v, (list, tuple, dict)):
                    return (f"`params.{name}` must be a single value — a list or object is "
                            f"not a workflow value")
                # is_file_param is a NAME heuristic, so the VALUE decides: only a string
                # that names a file (looks_like_path) is judged — "5000", "quad" are not
                if (not trusted and not m.get("client_path") and isinstance(v, str)
                        and not _numberish(v) and adapters.looks_like_path(v)
                        and not is_image_field(wf or {}, m.get("node"))
                        and adapters.is_file_param(p, m)):
                    return (f"`params.{name}` looks like a file path on the backend, which "
                            f"only an admin key may name — send the file itself under "
                            f"`files.{name}`, or have an admin tick \"client may send a "
                            f"backend path\" on this field in the Mapping editor")
    return None


JOB_MAX_TTL_DEFAULT = 7 * 86400


def _clamp_ttl(v) -> Optional[int]:
    """A client's job `ttl_s`, capped at `jobs.max_ttl_s` (config, default 7 days) —
    unbounded, `10**12` kept a job's inputs and results on disk for good. Anything that
    is not a positive int stays None (→ the store's default TTL), as before."""
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        return None
    try:
        cap = int(jobs_cfg.get("max_ttl_s", JOB_MAX_TTL_DEFAULT))
    except (TypeError, ValueError):
        cap = JOB_MAX_TTL_DEFAULT
    return min(v, cap) if cap > 0 else v


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
    ttl_s = _clamp_ttl(output.get("ttl_s") or body.get("ttl_s"))
    if mode == "async" and len(_gen_tasks) >= max_queued_gen > 0:
        raise HTTPException(503, f"too many queued generation jobs ({max_queued_gen}) — retry "
                                 f"later", headers={"Retry-After": "10"})
    force = (body.get("backend") or "").strip()          # pin to one backend (playground testing)

    routes, parked, eligible = await _gen_pick(alias, force, body)
    inputs, params = _gen_inputs_params(body)
    _apply_seconds(params, routes[0][1])         # seconds → frames (alias fps; 400 if unsupported)
    c0 = routes[0][1]
    wf_maps = [(adapters.cand_workflow(c0) or {}, c0.get("mapping") or {})]
    succ_alias = ((c0.get("successor") or {}).get("alias") or "").strip()
    if succ_alias:
        sc = (((await asyncio.to_thread(store.get, succ_alias)) if store.is_active() else None)
              or image_models.get(succ_alias) or [])
        if sc:
            wf_maps.append((adapters.cand_workflow(sc[0]) or {}, sc[0].get("mapping") or {}))
    refusal = _client_param_refusal(params, wf_maps, _params_trusted(request))
    if refusal:
        raise HTTPException(400, refusal)

    def build_req(backend: dict, cand: dict) -> NormalizedRequest:
        # `job_id` below is bound by the time this runs (dispatch happens after the job
        # row exists) — every input upload is namespaced by it.
        return NormalizedRequest(
            alias=alias, real_model=cand.get("model"),
            inputs=inputs, params=params,
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
        await _run_gen_sync(job_id, runner)
        return await _job_view(job_id, request)

    if parked:
        if mode == "async":                              # queue and hand back a job id
            _spawn_gen(job_id, _run_gen_parked(job_id, alias, force, build_req, eligible))
            return {"job_id": job_id, "status": "queued"}
        await _run_gen_sync(job_id, _run_gen_parked(job_id, alias, force, build_req, eligible))
        return await _job_view(job_id, request)                  # sync: blocked through the park
    runner = _run_gen_now(job_id, alias, force, routes, build_req, eligible)
    if mode == "async":
        _spawn_gen(job_id, runner)
        return {"job_id": job_id, "status": "queued"}

    await _run_gen_sync(job_id, runner)                  # sync: block until done/failed
    return await _job_view(job_id, request)


_UPLOAD_MAX_BYTES = 64 * 1024 * 1024        # per-file cap for `files` (413) — a mesh is big,
                                            # but a request body is not a stream


def _gen_alias_mapping(alias: str) -> tuple[dict, dict]:
    """A generation alias's (workflow, mapping) — from its first candidate, since both
    are backend-independent. Includes busy backends: resolving a request field must not
    depend on which backend happens to be free."""
    routes = get_gen_routes(alias)
    if not routes:
        return {}, {}
    _, cand = routes[0]
    return (adapters.cand_workflow(cand) or {}), (cand.get("mapping") or {})


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


async def _decode_ref_images(imgs: dict) -> dict:
    """`images: {slot: base64|data-URI|URL}` → {slot: bytes}. An EMPTY value is a slot the
    client leaves empty (the slot's own on_empty rule applies); a value that cannot be read
    — a 404 URL, broken base64 — is a 400 naming the slot. It used to be dropped silently,
    so the job ran on the slot's placeholder (or its baked-in image) and came back `done`
    with a confidently wrong result (K12)."""
    out = {}
    for param, val in imgs.items():
        if val in (None, ""):
            continue
        data = await _decode_ref_image(val)
        if not data:
            raise HTTPException(400, f"`images.{param}` could not be read — expected base64, "
                                     f"a data-URI or an http(s) URL that answers 200")
        out[param] = data
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
        # Only keys that ARE image slots of this alias (param or label) are fetched and
        # kept: anything else was ignored by the adapter anyway, but it was still
        # downloaded and stored as a job input — a free fetch-and-keep for any URL.
        # A workflow this process cannot read (None) is NOT "no slots": the adapter
        # loads it itself and matches the images there, so nothing is filtered.
        slots = await asyncio.to_thread(_gen_image_slot_names, body.get("model", ""))
        if slots is not None:
            for param in imgs:
                if param not in slots:
                    logger.info(f"generations: ignoring images.{str(param)[:60]} — not an "
                                f"image slot of '{str(body.get('model', ''))[:80]}'")
        uploads = await _decode_ref_images({p: v for p, v in imgs.items()
                                            if slots is None or p in slots})
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


def _alias_paired(cands) -> bool:
    """Whether the alias's workflow has BOTH high and low LoRA stacks — exactly when
    `_apply_lora_list` loads a pair's counterpart itself (F7: a client then gets both
    halves' trigger words without knowing about pairs). May read a workflow FILE: call
    it in a worker thread."""
    for c in cands or []:
        wf = adapters.cand_workflow(c) or {}
        if {"high", "low"} <= {k for _, k in lora_groups(wf, (c or {}).get("mapping") or {})}:
            return True
    return False


async def _alias_lora_items(alias: str, request: Request, authorization: Optional[str]) -> tuple:
    """(sorted LoRA names valid for the alias, their trigger-word items). The names are
    the union installed across the alias's backends, as always; the items come from the
    in-memory snapshot only — never a hash, a listing or Civitai (N1)."""
    await gate_request(authorization, request, alias)                # auth + allow-list
    cands = ((await asyncio.to_thread(store.get, alias)) if store.is_active() else None) \
        or image_models.get(alias)
    if not cands:
        raise HTTPException(404, f"generation alias '{alias}' not found")
    loras: set = set()
    for b, _ in await asyncio.to_thread(get_gen_routes, alias):
        loras |= backend_loras.get(backend_id(b), set())
    names = sorted(loras)
    paired = await asyncio.to_thread(_alias_paired, cands)
    return names, loratags.items(names, _lm_snapshot(),
                                 adapters.lora_counterpart if paired else None)


@app.get("/v1/generations/{alias}/loras")
async def gen_alias_loras(alias: str, request: Request, authorization: Optional[str] = Header(None)):
    """LoRA filenames valid for a generation alias — the union of what's installed on
    the alias's backends (`loras`, unchanged), plus `items`: per name its trigger words
    (curated before Civitai), status, Civitai record and pair. Information only — the
    gateway never edits a prompt."""
    names, its = await _alias_lora_items(alias, request, authorization)
    return {"object": "list", "alias": alias, "loras": names, "items": its}


@app.get("/v1/generations/{alias}/loras/{name:path}")
async def gen_alias_lora(alias: str, name: str, request: Request,
                         authorization: Optional[str] = Header(None)):
    """One LoRA's trigger-word item for an alias; 404 when the name is not valid there."""
    _names, its = await _alias_lora_items(alias, request, authorization)
    for it in its:
        if it["name"] == name:
            return it
    raise HTTPException(404, f"LoRA '{name}' is not valid for generation alias '{alias}'")


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
    wf = adapters.cand_workflow(cand) or {}
    mapping = cand.get("mapping") or {}
    kinds = sorted(k for _, k in lora_groups(wf, mapping) if k)
    out: dict = {"object": "generation.schema", "alias": alias,
                 "backends": [c.get("backend") for c in cands],
                 "params": params, "images": images, "files": files,
                 "modes": ["sync", "async"],
                 "loras_url": f"/v1/generations/{alias}/loras",
                 "loras": {"list_url": f"/v1/generations/{alias}/loras",
                           "item_url": f"/v1/generations/{alias}/loras/{{name}}",
                           "trigger_words": ("items[] in list_url carry each LoRA's "
                                             "trigger_words (curated before Civitai), "
                                             "status, base model and Civitai link — "
                                             "information only: the gateway never edits "
                                             "the prompt"),
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

def _gen_image_slots_known(alias: str) -> Optional[list]:
    """Ordered image-input param names of a generation alias (its workflow's image
    loaders per the mapping) — reference images map onto these positionally. Includes
    busy backends: slots are a workflow property, not gated on backend availability
    (else a busy backend would silently drop the uploaded reference images). None when
    the workflow cannot be read here (adapters.cand_workflow) — "unknown", which must
    never be read as "no slots"."""
    routes = get_gen_routes(alias)
    if not routes:
        return []
    _, cand = routes[0]
    if adapters.cloud_kind(cand):
        # [1] = the IMAGE slots only: a `files` entry (a rigging mesh) is not something
        # a positional reference image may ever land on.
        return [i["name"] for i in adapters.public_fields(cand)[1]]   # labels ARE the params
    wf = adapters.cand_workflow(cand)      # workflow_json, else the `workflow:` FILE
    if wf is None:
        return None
    return image_params(wf, cand.get("mapping") or {})


def _gen_image_slots(alias: str) -> list:
    """_gen_image_slots_known for the positional shims: unknown slots map nothing."""
    return _gen_image_slots_known(alias) or []


def _gen_image_slot_names(alias: str) -> Optional[set]:
    """Every name `images` may use for this alias: each image slot's param AND its
    public label (the adapter accepts both; a cloud alias's slot names are its labels).
    None = the workflow is not readable here, so nothing can be judged."""
    known = _gen_image_slots_known(alias)
    if known is None:
        return None
    names = set(known)
    wf, mapping = _gen_alias_mapping(alias)
    for p in names.copy():
        lbl = ((mapping.get(p) or {}).get("label") or "").strip()
        if lbl:
            names.add(lbl)
    return names


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
        data = await _fetch_ref_url(ref)
        return (data, _clean_ext(ext)) if data is not None else None
    try:
        return base64.b64decode(ref), _clean_ext(ext)
    except Exception:
        return None


# ── Client-supplied URLs (reference images, `files`) ──────────────────────────────
# The gateway fetches these ON THE CLIENT'S BEHALF and keeps the bytes readable at
# /v1/jobs/<id>/input/<n> — so an unfiltered fetch lets any key holder read what only
# the gateway can reach: a backend's admin port, a router UI, 169.254.169.254, the
# gateway's own /ui on localhost (review S5). Every address the name resolves to must be
# PUBLIC, the connection goes to exactly the address that was checked (a second lookup
# could answer differently — DNS rebinding), redirects are never followed (the shared
# client's default; a 3xx is a failed fetch), and the body is counted while it streams.
# `ref_url_allow_cidrs` (config.yaml) opens chosen private ranges, e.g. the LAN NAS.
_REF_FETCH_MAX_BYTES = _UPLOAD_MAX_BYTES


def _ref_allow_nets() -> list:
    nets = []
    for c in (config.get("ref_url_allow_cidrs") or []) if isinstance(config, dict) else []:
        try:
            nets.append(ipaddress.ip_network(str(c).strip(), strict=False))
        except ValueError:
            logger.warning(f"ref_url_allow_cidrs: ignoring '{c}' (not a network)")
    return nets


def ref_addr_blocked(addr: str, allow: Optional[list] = None) -> bool:
    """True for an address a client URL must not reach: anything not globally routable
    (loopback, RFC 1918, link-local, CGNAT, ULA, unspecified, documentation, …) and
    multicast — unless an `allow` network contains it. An IPv4-mapped IPv6 address is
    judged as the IPv4 it carries (`::ffff:127.0.0.1` is loopback)."""
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if any(ip in n for n in (allow or [])):
        return False
    return (not ip.is_global) or ip.is_multicast


async def _resolve_ref_host(host: str, port: int) -> list:
    """Every address `host` resolves to (a literal IP resolves to itself)."""
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(i[4][0] for i in infos))


async def _fetch_ref_url(url: str) -> Optional[bytes]:
    """GET a client-supplied http(s) URL under the rules above. None = not readable
    (unresolvable, refused, non-200); 400 = the target is not allowed; 413 = too big."""
    u = urlparse(url)
    host = u.hostname
    if not host:
        return None
    try:
        port = u.port or (443 if u.scheme == "https" else 80)
    except ValueError:
        return None
    try:
        addrs = await _resolve_ref_host(host, port)
    except (OSError, UnicodeError):
        return None
    allow = _ref_allow_nets()
    bad = [a for a in addrs if ref_addr_blocked(a, allow)]
    if not addrs or bad:
        logger.warning(f"ref url refused: {host} → {', '.join(bad or ['nothing'])}")
        raise HTTPException(400, f"reference URL host '{host}' resolves to a private, loopback "
                                 f"or link-local address — the gateway only fetches public "
                                 f"URLs (send the bytes as base64 instead, or allow the range "
                                 f"in ref_url_allow_cidrs)")
    ip = addrs[0].split("%", 1)[0]
    pinned = u._replace(netloc=(f"[{ip}]" if ":" in ip else ip) + f":{port}").geturl()
    ext = {"sni_hostname": host} if u.scheme == "https" else {}
    limit = _REF_FETCH_MAX_BYTES
    try:
        async with http_client.stream("GET", pinned, headers={"Host": u.netloc.rsplit("@", 1)[-1]},
                                      extensions=ext, timeout=20.0) as r:
            if r.status_code != 200:
                return None
            if int(r.headers.get("content-length") or 0) > limit:
                raise HTTPException(413, f"reference URL body exceeds {limit // (1024 * 1024)} MB")
            buf = bytearray()
            async for chunk in r.aiter_bytes():
                buf += chunk
                if len(buf) > limit:
                    raise HTTPException(413, f"reference URL body exceeds {limit // (1024 * 1024)} MB")
            return bytes(buf)
    except HTTPException:
        raise
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
    if not isinstance(refs, list):
        raise HTTPException(400, "`ref_images` must be a list")
    slots = await asyncio.to_thread(_gen_image_slots, alias)   # ONE lookup — uploads + log
    # images_uploads keeps one image per slot; fetching the rest would only download
    # (and, for URLs, reach out for) bytes that are thrown away.
    decoded = [await _decode_ref_image(r) for r in refs[:len(slots)]]
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
            "error": backend_error.get(bid) if en else None,   # why it is down → the status chip
            "busy": en and backend_busy(b), "inflight": backend_inflight.get(bid, 0),
            "draining": is_draining(b),
            "max_concurrent": backend_max_concurrent(b),
            "models": len(backend_models.get(bid, set())),
            "reqs_1h": src_1h.get(b["name"], 0),
            "sampling_defaults": b.get("sampling_defaults") or None,
            **_loaded_info(b),
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


# faults_info runs on every Dashboard tick (per viewer) and every /health: up to 20k
# events read and bundled each time. Memoised briefly — keyed on the fault log's
# generation, so a NEW fault shows at once; only an open outage's running downtime
# may lag by the TTL.
_FAULTS_INFO_TTL_S = 5.0
_faults_info_memo: dict = {}


def faults_info_memo_clear() -> None:
    _faults_info_memo.clear()


def faults_info(window_s: int = faults.WINDOW_S) -> dict:
    """Memoised _faults_info (see _FAULTS_INFO_TTL_S). Callers must not mutate it."""
    key = (int(window_s), faults.generation())
    hit = _faults_info_memo.get(key)
    mono = time.monotonic()
    if hit is not None and mono - hit[0] < _FAULTS_INFO_TTL_S:
        return hit[1]
    val = _faults_info(window_s)
    _faults_info_memo.clear()
    _faults_info_memo[key] = (mono, val)
    return val


def _faults_info(window_s: int = faults.WINDOW_S) -> dict:
    """The last `window_s` of the fault log, derived for the console (Dashboard +
    Statistic): `backends` = one summary per backend (faults, outages, downtime incl.
    an outage STILL open, the last error), `bundles` = the faults grouped by message.
    Host labels come from the Hosts settings — the operator thinks "the Evo-X2 fell
    over", not "192.168.8.228"."""
    now = int(time.time())
    since = now - int(window_s)
    evs = faults.events_since(since)
    by_bid = {backend_id(b): b for b in backends}
    # Open outages = the ones the log OPENED (fault_since, see refresh_backend) — the same
    # clock the recovery closes them with, whatever kind they show now.
    down_since = {bid: int(backend_error[bid]["fault_since"])
                  for bid, b in by_bid.items()
                  if is_enabled(b) and not backend_healthy.get(bid, False)
                  and (backend_error.get(bid) or {}).get("fault_since") is not None}
    per = faults.per_backend(evs, since, now, down_since)

    def label(host: str) -> str:
        return ((hosts_meta.get(host) or {}).get("label") or "").strip()

    rows = []
    for bid, s in per.items():
        b = by_bid.get(bid)
        if b is not None:
            s["backend"], s["type"] = b["name"], b.get("type", "openai")
            s["host"] = backend_hosts.get(bid) or backend_host(b)
            s["enabled"] = is_enabled(b)
            s["healthy"] = s["enabled"] and backend_healthy.get(bid, False)
            s["error"] = backend_error.get(bid) if s["enabled"] else None
        else:                                    # removed since — history stays readable
            s.update(enabled=False, healthy=False, error=None)
        s["host_label"] = label(s["host"])
        rows.append(s)
    rows.sort(key=lambda s: (s["healthy"], -(s["faults"]), -(s["last_ts"] or 0), s["backend"].lower()))
    bundles = faults.bundles(evs)
    for g in bundles:
        b = by_bid.get(g["bid"])
        if b is not None:
            g["host"] = backend_hosts.get(g["bid"]) or backend_host(b)
        g["host_label"] = label(g["host"])
    return {"since": since, "now": now, "window_s": int(window_s),
            "persistent": faults.is_persistent(), "backends": rows, "bundles": bundles,
            "total": sum(s["faults"] for s in rows)}


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
    if b.get("type") not in ("comfyui", "runpod"):
        return {}
    info: dict = {}
    fs = _gen_fail_stats(backend_id(b))
    if fs:
        info.update(fs)
    # A RunPod endpoint runs the same workflows, so an execution fault quarantines it
    # like a ComfyUI box — and a quarantine that changes routing must be visible.
    info.update(_quarantine_info(backend_id(b)))
    if b.get("type") != "comfyui":
        return info                 # no executor watchdog, nothing to restart on RunPod
    ad = backend_adapters.get(backend_id(b))
    if ad is not None:
        info.update({"exec_stuck": bool(getattr(ad, "exec_stuck", False)),
                     "last_restart": int(ad.last_restart) if getattr(ad, "last_restart", 0.0) else None,
                     "last_restart_result": getattr(ad, "last_restart_result", "") or None})
    return info


def _model_filter_info(b: dict) -> dict:
    """What the two model-set knobs did: `{"models_filtered": {"kept": k, "total": t}}`
    for a backend carrying an allow/deny filter, `{"models_added": [ids]}` for one
    carrying manual additions, `{}` for a backend with neither — so an absent key means
    "not configured", not "changed nothing".

    Both are VERDICTS, reported only where a poll has happened, and named apart from
    the config fields they come from (`models_allow`/`models_deny`/`models_extra`,
    which this same summary carries verbatim for the editor to pre-fill). One key
    cannot be both: the form needs the comma string always, the badge needs the
    measurement only when there is one.

    Reported because the filter is otherwise INVISIBLE: a typo in `models_allow`
    leaves the backend healthy, discovered and empty, and every downstream symptom
    (an alias with no candidates, a model missing from /v1/models) points somewhere
    else. `kept`/`total` are the numbers a discovery poll actually MEASURED.

    A backend that never polled successfully therefore reports nothing at all —
    deriving `(0, 0)` from the empty model set puts "filter matches nothing" next to
    an UNREACHABLE backend and blames the filter for a dead host (measured 2026-09-09
    on a fresh instance: a down backend carrying a filter showed exactly that badge).
    One that polled and went down afterwards keeps its last measured numbers, which
    stay true."""
    bid = backend_id(b)
    counts = backend_model_counts.get(bid)
    if counts is None:
        return {}                    # never polled — there is no measurement to report
    out: dict = {}
    if (adapters.parse_model_filter(b.get("models_allow"))
            or adapters.parse_model_filter(b.get("models_deny"))):
        kept, total = counts
        out["models_filtered"] = {"kept": int(kept), "total": int(total)}
    # `models_extra` is reported under the same never-polled rule, for a reason of its
    # own: refresh_backend adds these ids only where discovery SUCCEEDED, so announcing
    # them for a backend that has never answered would list models nothing can serve.
    extra = adapters.parse_model_filter(b.get("models_extra"))
    if extra:
        out["models_added"] = extra
    return out


def _loaded_info(b: dict) -> dict:
    """`loaded`: what llama-swap's `/running` last reported ([{model, state, kind}]) for a
    backend that has the endpoint and is up — the same list `current` resolves against,
    so the console shows exactly what a `current` call would get. `{}` for every other
    backend, and for a down one (its last list is no longer a fact); an empty list means
    'up, nothing loaded'."""
    bid = backend_id(b)
    if bid not in backend_running or not (is_enabled(b) and backend_healthy.get(bid, False)):
        return {}
    return {"loaded": [dict(e) for e in backend_running[bid]]}


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


def _runpod_info(b: dict) -> dict:
    """What the last discovery and probe of a RunPod backend saw (merged into /health +
    the Backends tab); {} for every other type."""
    if b.get("type") != "runpod":
        return {}
    ad = backend_adapters.get(backend_id(b))
    if ad is None:
        return {}
    h = getattr(ad, "health", {}) or {}
    return {"runpod": {"workers_idle": (h.get("workers") or {}).get("idle"),
                       "workers_running": (h.get("workers") or {}).get("running"),
                       "in_queue": (h.get("jobs") or {}).get("inQueue"),
                       "workers_max": (getattr(ad, "endpoint_info", {}) or {}).get("workersMax"),
                       "probe": dict(getattr(ad, "probe_state", {}) or {})}}


def runpod_probe(bid: str) -> bool:
    """Start one probe of a RunPod backend in the background (console action)."""
    ad = backend_adapters.get(bid)
    if not isinstance(ad, adapters.RunpodAdapter) or ad.probe_state.get("state") == "running":
        return False
    _bg(_runpod_probe_run(bid, ad))
    return True


async def _runpod_probe_run(bid: str, ad) -> None:
    """The probe runs to its end on THIS instance — a backend save meanwhile may have
    replaced it (build_backend_adapters), and the replacement would otherwise never learn
    the result (its probe_state stayed "running" for good). Same endpoint → hand it over."""
    await ad.probe()
    cur = backend_adapters.get(bid)
    if cur is not ad and isinstance(cur, adapters.RunpodAdapter):
        cur.adopt_probe(ad)


def runpod_object_info(name: str) -> Optional[dict]:
    """The probed /object_info of the RunPod backend `name` (the mapping editor's widget
    source — the endpoint has no /object_info of its own to ask)."""
    for b in backends:
        if b.get("name") == name and b.get("type") == "runpod":
            ad = backend_adapters.get(backend_id(b))
            return getattr(ad, "object_info_full", None)
    return None


async def _cancel_orphaned_runpod(orphans: list) -> int:
    """Startup: a job the restart orphaned may still run — and bill — at RunPod. Cancel
    each one whose row names a RunPod job (best effort; the job's own ttl ends it
    otherwise). Returns how many cancels were sent."""
    n = 0
    for job_id, bname, meta in orphans:
        rp = (meta or {}).get("runpod_job_id")
        if not rp:
            continue
        try:
            b = next((x for x in backends if x.get("name") == bname and x.get("type") == "runpod"), None)
            ad = backend_adapters.get(backend_id(b)) if b else None
            if ad is None:
                logger.warning(f"startup: orphaned job {job_id} names RunPod job {rp} but "
                               f"'{bname}' is no runpod backend now — check the RunPod console")
                continue
            n += 1
            # The job ran on the endpoint its row names — the backend's url may have been
            # changed since, and a cancel sent to the new endpoint ends nothing.
            ep = (meta or {}).get("runpod_endpoint")
            ep_url = f"https://api.runpod.ai/v2/{ep}" if isinstance(ep, str) else ""
            if ep_url and adapters.runpod_endpoint_id(ep_url) == ep \
                    and ep != adapters.runpod_endpoint_id(b.get("url") or ""):
                ok = await ad.cancel_runpod_id(rp, url=ep_url)
            else:
                ok = await ad.cancel_runpod_id(rp)
            logger.warning(f"startup: RunPod job {rp} of orphaned job {job_id} "
                           f"{'cancelled' if ok else 'cancel UNCONFIRMED — check the RunPod console'}")
            await asyncio.to_thread(jobs.merge_meta, job_id, {"runpod_cancelled_at_restart": ok})
        except Exception as e:
            logger.warning(f"startup: RunPod job {rp} of orphaned job {job_id}: "
                           f"{type(e).__name__}: {e} — check the RunPod console")
    return n


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
            "api_key_set": bool(b.get("api_key")),      # the key itself never leaves main
            "sampling_defaults": b.get("sampling_defaults") or None,
            # The filter globs themselves, not just the resulting counts: the backend
            # editor falls back to THIS summary for a config-defined backend (nothing in
            # the store yet), and a field the summary omits renders blank — the next Save
            # would then write that blank back and drop the filter without a word.
            # Joined through the parser, so config.yaml's YAML-list form arrives as the
            # comma string the editor's text input expects (never "['gpt-*']").
            "models_allow": ", ".join(adapters.parse_model_filter(b.get("models_allow"))),
            "models_deny": ", ".join(adapters.parse_model_filter(b.get("models_deny"))),
            "models_extra": ", ".join(adapters.parse_model_filter(b.get("models_extra"))),
            "host": backend_hosts.get(backend_id(b), ""),
            "host_explicit": bool((b.get("host") or "").strip()),
            "source": "config" if backend_id(b) in config_ids else "ui",
            **_comfy_watch_info(b), **_cloud_info(b), **_runpod_info(b), **_model_filter_info(b), **_loaded_info(b),
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
        asyncio.get_running_loop()           # outside a loop (import-time callers) → skip
        _bg(_discover())
    except RuntimeError:
        pass


def begin_drain(bid: str) -> bool:
    """Take a backend offline gracefully: stop routing new requests to it now, and
    disable it once its in-flight requests finish. False if unknown/already disabled."""
    b = next((x for x in backends if backend_id(x) == bid), None)
    if b is None or not is_enabled(b):
        return False
    _draining.add(bid)
    _drain_hold.discard(bid)             # a real drain takes over a restart's hold
    _drain_host[bid] = _host_name_of(b)
    n = backend_inflight.get(bid, 0)
    logger.info(f"[{b['name']}] draining — {n} in-flight; goes offline when idle")
    _notify_slot_free()                  # parked calls re-evaluate: this backend is out now
    if n <= 0:
        _finalize_drain(bid)
    return True


def cancel_drain(bid: str) -> bool:
    """Abort a drain → the backend rejoins rotation. False if it wasn't draining."""
    _drain_host.pop(bid, None)
    _drain_hold.discard(bid)
    if bid not in _draining:
        return False
    _draining.discard(bid)
    nm = next((x["name"] for x in backends if backend_id(x) == bid), bid)
    logger.info(f"[{nm}] drain cancelled — back in rotation")
    _notify_slot_free()                  # back in rotation → parked calls can grab it now
    return True


def _config_backend_entry(b: dict) -> dict:
    """A config-defined backend as the store copy `set_backend_enabled` writes. The
    store entry replaces the config entry WHOLESALE (rebuild_backends), so it must carry
    EVERY configured key — an allowlist silently dropped `host`, the ComfyUI output/input
    dirs, `max_wait`, `auto_restart`, the model filters, … on the first toggle (a config
    backend is never attached to a managed host, R-K3, so only the console toggles it).
    Excluded: `enabled` (set by the
    caller) and `_`-prefixed runtime keys. The one key rebuild_backends derives onto the
    live dict is `paid`, and it is kept: for a cloud type it is forced True on every
    rebuild anyway, for the rest it is `bool(paid)` — exactly what the config meant.
    `api_key` stays plaintext here; store.upsert_backend encrypts it like any store entry."""
    # JSON-safe: the store writes it as JSON, and a YAML config can hold what JSON
    # cannot (an unquoted `2026-09-28` is a datetime.date) — json.dumps then raised on
    # every enable/disable
    return json.loads(json.dumps({k: v for k, v in b.items()
                                  if k != "enabled" and not str(k).startswith("_")},
                                 default=str))


def set_backend_enabled(bid: str, on: bool) -> bool:
    """Persist a backend's `enabled` flag (store) and rebuild. Backs the drain-finalize
    and the UI take-offline / bring-online actions. False if the backend is unknown."""
    b = next((x for x in backends if backend_id(x) == bid), None)
    if b is None:
        return False
    entry = dict(store.get_backend(b["name"], b.get("type", "openai")) or
                 _config_backend_entry(b))
    entry.update({"name": b["name"], "type": b.get("type", "openai"), "enabled": bool(on)})
    store.upsert_backend(entry)
    logger.info(f"[{b['name']}] {'enabled' if on else 'disabled'} via console")
    apply_backend_change()
    return True


def _hold_routing(bid: str, on: bool) -> bool:
    """A managed host's automatic restart (hostctl `hold_routing`): `on` keeps new
    requests off an enabled backend while its in-flight ones finish — `_draining` for
    routing, without the finalize that would disable it. False when it could not hold
    (unknown, disabled, already draining — a real drain is left alone). `off` gives
    routing back, only if it is still this hold (a take-offline meanwhile wins)."""
    if on:
        b = next((x for x in backends if backend_id(x) == bid), None)
        if b is None or not is_enabled(b) or bid in _draining:
            return False
        _draining.add(bid)
        _drain_hold.add(bid)
        _notify_slot_free()
        return True
    if bid not in _drain_hold:
        return False
    _drain_hold.discard(bid)
    _draining.discard(bid)
    _notify_slot_free()
    return True


def _host_name_of(b: dict) -> str:
    return str(b.get("host") or "").strip()


def _finalize_drain(bid: str) -> None:
    """Backend is idle → take it offline (persist enabled=false) and rebuild — unless
    it moved to another host since its drain began (R-K2): the old host's stop must not
    disable a backend its new host already enabled."""
    _draining.discard(bid)
    began_on = _drain_host.pop(bid, None)
    b = next((x for x in backends if backend_id(x) == bid), None)
    if b is not None and began_on is not None and _host_name_of(b) != began_on:
        logger.info(f"[{b['name']}] drain ended after a move to host "
                    f"{_host_name_of(b) or '(none)'} — left enabled")
        return
    set_backend_enabled(bid, False)
    logger.info(f"backends changed → {len(backends)} effective")


# ── Managed hosts (hostctl.py) ────────────────────────────────────────────────────
# One Controller per MANAGED HOST (store setting `managed_hosts`: {name: {provider,
# options, api_key}}), keyed by host name. Its services are the STORE backends whose
# `host` names it; each carries `local_port` (the gateway's end of the tunnel forward,
# assigned here, stable across saves and renames) and `remote_port` (the service's
# loopback port on the VM, the profile's default when unset), and its URL is derived
# from the local port. A config-defined backend naming a managed host is NOT attached
# (R-K3): the lifecycle would write a store copy of it on every start and stop, and
# that copy overrides config wholesale. A controller owns a billing cloud instance, so
# it outlives its entry: a host deleted (or its entry unreadable) while the instance
# runs keeps its controller — with its last service list — because dropping it would
# leave the instance billing with nobody to snapshot or delete it. Every lookup from a
# backend goes backend → `host` → controller → service (R-W7).

_HERE = Path(__file__).resolve().parent
_HOST_STATE_KEY = "host_state"              # store setting: host name → State dict
_THUNDER_PROBE_S = 10
host_controllers: dict = {}                 # host name → hostctl.Controller
# host name → the controller's background tasks (resume, run_forever, console actions),
# so a retired controller and the shutdown can cancel them; also held by _bg.
_host_tasks: dict = {}
_hosts_booted = False                       # lifespan ran _hosts_boot (new ones start at once)
_host_warned: set = set()                   # hosts warned "entry removed while instance runs"
managed_hosts: dict = {}                    # name → entry (token decrypted), per sync
_host_errors: dict = {}                     # name → why the host is not driven (as it stands)
_host_error_warned: set = set()             # (name, error) already logged
_host_attached: dict = {}                   # name → [bid] of the backends attached to it
_host_not_attachable: dict = {}             # name → [bid] config backends naming it
_not_attach_warned: set = set()             # (name, bid) warned "not attached"
# The gateway's ends of the tunnel forwards (spec "Datenmodell"): one range, unique over
# every host — two services on one local port make sshrun refuse the whole tunnel.
LOCAL_PORT_MIN, LOCAL_PORT_MAX = 18100, 18999
# A host name is the identity of its state record, snapshots (`aihub-<name>-<stamp>`)
# and control socket (R-W5): its own slug, short enough for the socket path.
_HOST_NAME_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?")
NOT_ATTACHABLE_REASON = "config-defined backend — create it in the console"


def _live_backend(bid: str) -> Optional[dict]:
    return next((b for b in backends if backend_id(b) == bid), None)


def _host_ctl(backend: Optional[dict]) -> Optional["hostctl.Controller"]:
    """The controller of the managed host `backend` runs on — backend → `host` →
    controller (R-W7) — only if that controller really carries this backend as a
    service: a config backend naming the host is not attached, and a same-named
    backend of another type is another backend."""
    if not isinstance(backend, dict) or not backend.get("name"):
        return None
    hn = str(backend.get("host") or "").strip()
    c = host_controllers.get(hn) if hn else None
    if c is None or not c.has_service(backend_id(backend)):
        return None
    return c


def _lport_ok(v) -> bool:
    return (isinstance(v, int) and not isinstance(v, bool)
            and LOCAL_PORT_MIN <= v <= LOCAL_PORT_MAX)


def _pick_local_port(bid: str, cur, host: str, rows: list, own: tuple = ()) -> int:
    """`cur` when it is in the range and nobody else holds it, else the lowest free
    port. Held = any OTHER backend's `local_port` (store rows and the live list), and a
    port another host's controller still forwards for a backend that no longer exists
    (a host deleted while running keeps its forwards). The backend's OWN host's old
    service list is not counted: after a rename its old entry is replaced by this one.
    `own` = further ids that ARE this backend (the identity a console rename comes
    from, whose row and live entry still exist until the Save writes)."""
    mine = {bid, *own}
    used = set()
    for b in list(rows) + list(backends):
        if backend_id(b) not in mine and _lport_ok(b.get("local_port")):
            used.add(b["local_port"])
    live = {backend_id(b) for b in backends}
    # a port a controller FORWARDS for this very backend (its service list of the last
    # sync) is its, whoever else's row claims it: resolving the clash by list order
    # could move a RUNNING service's forward (and URL) for a hand-edited row
    if _lport_ok(cur) and any(
            hostctl.service_bid(x) in mine and x.get("local_port") == cur
            for c in host_controllers.values() for x in getattr(c, "services", []) or []):
        return cur
    for hn, c in host_controllers.items():
        if hn == host:
            continue
        for x in getattr(c, "services", []) or []:
            if (hostctl.service_bid(x) not in live and hostctl.service_bid(x) not in mine
                    and _lport_ok(x.get("local_port"))):
                used.add(x["local_port"])
    if _lport_ok(cur) and cur not in used:
        return cur
    for p in range(LOCAL_PORT_MIN, LOCAL_PORT_MAX + 1):
        if p not in used:
            return p
    raise RuntimeError(f"no free local port in {LOCAL_PORT_MIN}–{LOCAL_PORT_MAX}")


def assign_local_port(backend_name: str, btype: str = "openai",
                      prev: Optional[tuple] = None) -> int:
    """The local port of backend `(backend_name, btype)` on a managed host: the one its
    store row carries when that is still valid and unique, else the lowest free one in
    18100–18999. Stable across Saves AND renames because it is read from the row (a
    Save — and a rename — builds the new row from the old one). `prev` = the (name,
    type) the console's Save is renaming FROM: its row is still in the store and its
    backend still live, so without it the backend's own old port would count as
    another's and a rename would move the URL. Returns, never writes: the console's
    Save stores it with the row; `sync_host_controllers` assigns lazily for rows that
    have none yet. RuntimeError when the range is exhausted."""
    bid = f"{btype}:{backend_name}"
    pbid = f"{prev[1]}:{prev[0]}" if prev else bid
    rows = store.list_backends() if store.is_active() else []
    row = (next((b for b in rows if backend_id(b) == bid), None) or _live_backend(bid)
           or next((b for b in rows if backend_id(b) == pbid), None) or _live_backend(pbid)
           or {})
    return _pick_local_port(bid, row.get("local_port"), str(row.get("host") or ""), rows,
                            own=(pbid,))


def _attach_fields(b: dict, rows: list) -> None:
    """An attached store backend's forward ends and URL: `remote_port` from the profile
    when unset, `local_port` assigned, `url` = http://127.0.0.1:<local_port>. Written
    onto the live dict (this rebuild's adapters are built from it next) and persisted
    to its store row when anything changed. A type without a profile gets nothing — the
    controller shows it `down` with the reason."""
    prof = services.profile_for(b)
    if prof is None:
        return
    ch = {}
    if b.get("remote_port") in (None, ""):
        ch["remote_port"] = prof.default_port
    lp = _pick_local_port(backend_id(b), b.get("local_port"), str(b.get("host") or ""), rows)
    if b.get("local_port") != lp:
        ch["local_port"] = lp
    url = f"http://127.0.0.1:{lp}"
    if b.get("url") != url:
        ch["url"] = url
    if not ch:
        return
    b.update(ch)
    row = store.get_backend(b["name"], b.get("type", "openai"))
    if row is not None:
        row.update(ch)
        store.upsert_backend(row)
        for r in rows:                       # later picks in this sync see the new port
            if backend_id(r) == backend_id(b):
                r.update(ch)
    logger.info(f"[host {b.get('host')}] {backend_id(b)}: "
                + ", ".join(f"{k}={v}" for k, v in sorted(ch.items())))


def _host_load_state(name: str) -> Optional[dict]:
    d = store.get_setting(_HOST_STATE_KEY)
    if d is None:
        return None
    if not isinstance(d, dict):
        # raised, not read as {}: the controller then refuses to start and never saves
        # over a record that may name a running instance (hostctl._load_failed)
        raise ValueError(f"store setting {_HOST_STATE_KEY} is a {type(d).__name__}, "
                         "not a dict")
    return d.get(name)


def _host_save_state(name: str, d: dict) -> None:
    """Read-modify-write of the ONE shared setting. Safe only because neither this
    function nor hostctl's `_persist` awaits between the read and the write — every
    controller saves on the event loop thread, one after the other. Moving this to a
    worker thread (asyncio.to_thread) would open a lost-update race: two controllers
    reading the same dict, the later write erasing the other's instance record."""
    cur = store.get_setting(_HOST_STATE_KEY)
    if cur is None:
        cur = {}
    if not isinstance(cur, dict):
        # writing {name: d} would erase every other host's instance record
        raise ValueError(f"store setting {_HOST_STATE_KEY} is unreadable — not saved")
    cur[name] = d
    store.set_settings({_HOST_STATE_KEY: cur})


def _host_known_uuids() -> set:
    """Instances some controller owns — `orphans()` lists everything else."""
    return {c.state.uuid for c in host_controllers.values() if c.state.uuid}


async def _comfy_probe(url: str) -> bool:
    """Does ComfyUI answer through the tunnel? Streamed: /object_info is megabytes and
    the status is all the controller asks — it probes every few seconds while starting."""
    try:
        async with http_client.stream("GET", url.rstrip("/") + "/object_info",
                                      timeout=_THUNDER_PROBE_S) as r:
            return r.status_code == 200
    except httpx.HTTPError:
        return False


async def _host_probe_http(url: str) -> int:
    """A command service's health probe through its tunnel forward → the HTTP status
    (0 = no answer). Streamed, body never read: the status is all the controller asks
    (services.CommandProfile.probe_ok — 200/401/403 = up)."""
    try:
        async with http_client.stream("GET", url, timeout=_THUNDER_PROBE_S) as r:
            return r.status_code
    except httpx.HTTPError:
        return 0


def _thunder_default_nodes() -> str:
    """ops/thunder-nodes.default.txt for the backend FORM only (a new Thunder block's
    nodes textarea): "" when unreadable, so the console still renders. The controller
    deliberately uses the UNGUARDED reader in `_host_deps` — there an unreadable
    default list must fail the start, not bootstrap with nothing."""
    try:
        return (_HERE / "ops" / "thunder-nodes.default.txt").read_text("utf-8")
    except OSError as e:
        logger.warning(f"thunder: default node list unreadable: {e}")
        return ""


# ── model sync inputs (modelsync.py; the controller runs the plan) ────────────
_MODELSYNC_CATALOG_KEY = "modelsync_catalog"
_catalog_warned: list = []                  # the last unreadable catalog value warned about


def _modelsync_catalog() -> list:
    """The model-sync catalog: store setting `modelsync_catalog`. While it is absent,
    the first read copies `DEFAULT_CATALOG` into it — from then on the setting is
    authoritative (an operator who empties it gets an empty catalog, not the seed back,
    and a later release's seed never rewrites a catalog someone edited). A value that is
    no list is read as `[]` (warned once per value) — falling back to the defaults would
    sync what the operator replaced; entries `validate_catalog` refuses are dropped one
    by one inside modelsync."""
    if not store.is_active():
        return copy.deepcopy(modelsync.DEFAULT_CATALOG)
    raw = store.get_setting(_MODELSYNC_CATALOG_KEY)
    if raw is None:
        raw = store.setdefault_setting(_MODELSYNC_CATALOG_KEY,
                                       copy.deepcopy(modelsync.DEFAULT_CATALOG))
        logger.info(f"{_MODELSYNC_CATALOG_KEY}: seeded with the default catalog")
    if not isinstance(raw, list):
        if _catalog_warned != [raw]:
            _catalog_warned[:] = [raw]
            logger.warning(f"{_MODELSYNC_CATALOG_KEY} is a {type(raw).__name__}, not a "
                           "list — read as empty")
        return []
    return raw


def _modelsync_catalog_view() -> list:
    """`_modelsync_catalog()` for VIEWS: the same list, but an absent setting is answered
    with the default IN MEMORY — a page view (the console's editor, the Model sources
    overview, a 400 re-render of them) never writes the store. The seed lands on the
    controller's first plan or the editor's first Save, as before."""
    if not store.is_active():
        return copy.deepcopy(modelsync.DEFAULT_CATALOG)
    raw = store.get_setting(_MODELSYNC_CATALOG_KEY)
    if raw is None:
        return copy.deepcopy(modelsync.DEFAULT_CATALOG)
    return raw if isinstance(raw, list) else []


def save_modelsync_catalog(cat, expect_hash: Optional[str] = None) -> list:
    """The console's catalog Save: `validate_catalog`'s refusals ([] = saved). Nothing is
    written unless the WHOLE catalog is valid — a partial save would drop the refused
    entry silently and leave the alias it was meant for blocked without a word. Taken
    under `_catalog_lock` (Check & save writes the same setting); `expect_hash` = the
    `modelsync_catalog_hash()` the form was rendered with → `[CATALOG_STALE]` when the
    stored catalog changed since."""
    errs = modelsync.validate_catalog(cat)
    if errs:
        return errs
    if not store.is_active():
        return ["the store is not active — the catalog cannot be saved"]
    with _catalog_lock:                 # never between Check & save's read and write
        if expect_hash is not None and modelsync_catalog_hash() != expect_hash:
            return [CATALOG_STALE]
        store.set_settings({_MODELSYNC_CATALOG_KEY: cat})
    return []


def _comfy_alias_cands(backend_name: str) -> list:
    """(alias, candidate) of every generation alias with a ComfyUI candidate on this
    backend — store aliases over same-named config ones, the order `_gen_routes` reads
    them in. A cloud candidate of the same backend NAME is another backend (keyed
    name+type) and never counts."""
    merged = dict(image_models or {})
    if store.is_active():
        merged.update(store.list_aliases())
    out = []
    for alias in sorted(merged):
        for cand in merged[alias] or []:
            if (isinstance(cand, dict) and cand.get("backend") == backend_name
                    and adapters.cand_kind(cand) == "comfyui"):
                out.append((alias, cand))
    return out


def _mapping_fields(cand: dict) -> list:
    """The `(node, field)` pairs the candidate's mapping lets a client set."""
    m = cand.get("mapping") or {}
    return [(str(b.get("node")), str(b.get("field"))) for b in m.values()
            if isinstance(b, dict) and b.get("node") is not None and b.get("field")]


def _comfy_name_of(bid: str) -> Optional[str]:
    """A RunPod backend runs ComfyUI workflows; the candidate kind is comfyui.
    Both backend id kinds therefore participate in the same model planning."""
    kind, sep, name = str(bid or "").partition(":")
    return name if sep and kind in ("comfyui", "runpod") else None


def service_alias_needs(bid: str, catalog=None) -> list:
    """`modelsync.AliasNeed` per alias candidate on the ComfyUI service `bid` (built by
    `modelsync.alias_need`, the only builder that fills `covered`). Blocking store
    read — the controller calls it in a worker thread. `catalog` = the list to match
    against (the overview hands in the one it already read); None = the setting."""
    name = _comfy_name_of(bid)
    if name is None:
        return []
    if catalog is None:
        catalog = _modelsync_catalog()
    return [modelsync.alias_need(alias, modelsync.refs_for(
                cand, adapters.cand_workflow(cand), _mapping_fields(cand)), catalog)
            for alias, cand in _comfy_alias_cands(name)]


def service_alias_signature(bid: str) -> str:
    """A stable hash of exactly what `service_alias_needs` reads: this service's
    candidates (a path workflow's CONTENT too — the file may change under the same
    path) and the catalog. Polled every 5 s: any store write, deletion or config
    change that matters to the sync shows up here, none that does not."""
    name = _comfy_name_of(bid)
    parts = []
    for alias, cand in (_comfy_alias_cands(name) if name is not None else []):
        wf = None if "workflow_json" in cand else adapters.cand_workflow(cand)
        parts.append([alias, cand, wf])
    data = [parts, _modelsync_catalog()]
    try:
        blob = json.dumps(data, sort_keys=True, default=str)
    except TypeError:                       # mixed key types (a YAML workflow's int ids)
        blob = json.dumps(data, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _thunder_hf_token() -> str:
    return str((store.get_setting("hf_token") if store.is_active() else "") or "")


def hf_token_set() -> bool:
    """Is an HF token stored? (The console says so; it never shows the value.)"""
    return bool(_thunder_hf_token())


def save_hf_token(token: str) -> str:
    """The console's HF-token Save: "" removes the token, anything else replaces it
    (encrypted at rest — `hf_token` is in `store._SECRET_SETTINGS`). → the refusal, ""
    = saved. The refusal never repeats the value: it is a secret."""
    token = str(token or "")
    # the transfer's own rule (hostctl.hf_token_ok): a token it would withhold must
    # not be saveable — every gated download would then go without it, silently
    if token and not hostctl.hf_token_ok(token):
        return ("the HF token may hold only printable ASCII without spaces, quotes or "
                "backslashes (at most 512 characters) — not saved")
    if not store.is_active():
        return "the store is not active — not saved"
    store.set_settings({"hf_token": token})
    return ""


def _thunder_datadir() -> str:
    """thunder.key, modelsrc.key and the known_hosts files sit next to store.db (and
    secret.key)."""
    return os.path.dirname(os.path.abspath(jobs_cfg.get("store_path", "store.db")))


# ── the LAN model source (hostctl.LanSource, ops/modelsrc-serve.sh on the share) ──
# ONE per gateway: every host controller reads the same share, so the index cache,
# the sha256 cache and the pinned host key are shared. Rebuilt only when the data dir
# moves (a test, a store_path change).
_modelsrc_obj: Optional["hostctl.LanSource"] = None
_modelsrc_key_task: Optional[asyncio.Task] = None


def _modelsrc_host() -> str:
    """Store setting `modelsrc_host`, "" when unset — there is no default share host
    ("" = not configured, and the LanSource then never reaches ssh). Returned raw: the
    LanSource holds it to plain characters (the `_VOICE_HOST_RE` rule) before any argv."""
    v = store.get_setting("modelsrc_host") if store.is_active() else None
    return str(v or "").strip()


def save_modelsrc_host(value: str) -> str:
    """The console's `modelsrc_host` Save → the refusal ("" = saved). Blank stores ""
    (= no LAN source); anything else must be a plain `[user@]host` (`_VOICE_HOST_RE`, the
    rule LanSource holds it to before any ssh argv). The LanSource notices the change on
    its next look and drops the old share's listing (`LanSource._follow_host`)."""
    v = str(value or "").strip()
    if v and not _VOICE_HOST_RE.match(v):
        return f"modelsrc_host {v!r} is not a plain [user@]host — not saved"
    if not store.is_active():
        return "the store is not active — not saved"
    store.set_settings({"modelsrc_host": v})
    return ""


def thunder_orphan_snapshots() -> list:
    """`aihub-` snapshots no managed host owns (`thunder.foreign_snapshots`, by HOST
    name — R-W4) over every controller's CACHED snapshot list (deduplicated by id — two
    hosts of one account list the same snapshots) and the first cached price list. No
    API call: the controllers refresh both in their background loop."""
    snaps, seen, table = [], set(), None
    for c in list(host_controllers.values()):
        for sn in c.snapshots() or []:
            sid = sn.get("id") or sn.get("name")
            if sid not in seen:
                seen.add(sid)
                snaps.append(sn)
        if table is None:
            table = c.pricing_table()
    return thunder.foreign_snapshots(snaps, list(host_controllers), table)


_MODELSRC_SHA_KEY = "modelsrc_sha"


def _modelsrc_sha_load():
    """The persistent share-sha cache (`{"host", "files": {path: [size, sha256]}}`);
    LanSource holds it to the configured share host (R-2) and validates every row."""
    return store.get_setting(_MODELSRC_SHA_KEY) if store.is_active() else None


def _modelsrc_sha_save(rec: dict) -> None:
    if store.is_active():
        store.set_settings({_MODELSRC_SHA_KEY: rec})


def modelsrc() -> "hostctl.LanSource":
    global _modelsrc_obj
    d = _thunder_datadir()
    if _modelsrc_obj is None or _modelsrc_obj.datadir != d:
        _modelsrc_obj = hostctl.LanSource(d, host=_modelsrc_host, log=logger.info,
                                          load_sha=_modelsrc_sha_load,
                                          save_sha=_modelsrc_sha_save)
    return _modelsrc_obj


def _share_sha_files() -> dict:
    """The ONE reader of the share-sha cache for planning and the overview: `{path:
    [size, sha256]}` of the CONFIGURED share host ({} for a record of another host).
    Never starts a hash."""
    return modelsrc().sha_files()


# ── LoRA trigger words (spec 2026-10-02-lora-trigger-words-design) ─────────────────
# AI-Hub STORES and DELIVERS a LoRA's trigger words — it never edits a prompt. The
# metadata hangs on the share file's sha256 (the share renames files): the worker below
# hashes every share LoRA through the LanSource queue at the LOWEST priority (3) and
# asks Civitai by hash; the API and the console read only the in-memory snapshot
# (`_lm_snapshot`), never a hash, a listing or Civitai (N1).
_CIVITAI_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash/"
_CIVITAI_TIMEOUT = httpx.Timeout(20.0)
_CIVITAI_MAX_BYTES = 1 << 20
_CIVITAI_MIN_GAP_S = 2.0                  # at most one Civitai request per 2 s
_CIVITAI_BACKOFF = (300.0, 21600.0)       # per sha, transient failure: 5 min doubling → 6 h
_CIVITAI_PAUSE = (60.0, 3600.0)           # 429 without Retry-After: 1 min doubling → 1 h
_LM_TICK_S = 60.0
_LM_HASH_PAUSE_MIN_S = 10.0
_LM_HASH_RETRY_S = 600.0
_civitai_client: Optional[httpx.AsyncClient] = None     # None = http_client (tests: a mock)

lora_meta: dict = {}          # sha256 → record (store table lora_meta; see loratags)
_lm_errors: dict = {}         # sha or real share path → {"error", "next_try", "backoff_s"}
_lm_refetch: set = set()      # shas a manual refresh wants asked again


def _lm_initial_state() -> dict:
    return {"configured": False, "problem": "", "share": {}, "shas": {}, "hashing": None,
            "pause_until": 0.0, "pause_s": 0.0, "last_civitai": 0.0, "last_run": None,
            "last_error": "", "store_ok": True}


_lm_persist_lock: Optional[asyncio.Lock] = None   # serialises the table writes (lazy, per loop)
_lm_state: dict = _lm_initial_state()
_lm_wake: Optional[asyncio.Event] = None


async def _civitai_lookup(sha: str) -> tuple:
    """Ask Civitai about ONE sha256 — the only thing that leaves the gateway (N5).
    → ("found", record) | ("not_found", None) | ("busy", Retry-After s | None) |
    ("error", text). Never raises."""
    if not loratags.valid_sha(sha):
        return ("error", "invalid sha256")
    client = _civitai_client or http_client
    try:
        async with client.stream("GET", _CIVITAI_BY_HASH + sha, timeout=_CIVITAI_TIMEOUT,
                                 headers={"User-Agent": "ai-hub",
                                          "Accept": "application/json"},
                                 follow_redirects=False) as r:
            if r.status_code == 404:
                return ("not_found", None)
            if r.status_code == 429:
                return ("busy", loratags.retry_after_s(r.headers.get("retry-after")))
            if r.status_code != 200:
                return ("error", f"HTTP {r.status_code}")
            buf = bytearray()
            async for chunk in r.aiter_bytes():
                buf += chunk
                if len(buf) > _CIVITAI_MAX_BYTES:
                    return ("error", "answer larger than 1 MB")
    except Exception as e:                  # noqa: BLE001 — "never raises": a pass must progress
        return ("error", type(e).__name__)  # (httpx.HTTPError included; Cancelled passes)
    try:
        return ("found", loratags.parse_civitai(json.loads(bytes(buf)), time.time()))
    except Exception:                       # noqa: BLE001 — incl. RecursionError on nested JSON
        return ("error", "unparsable answer")


_LM_STORE_BAD = ("store unreadable at boot — trigger words are not saved; "
                 "restart the gateway")


def _lm_snapshot() -> dict:
    """What the API and the console read — memory only (N1)."""
    st = _lm_state
    return {"configured": st["configured"], "problem": st["problem"], "share": st["share"],
            "shas": st["shas"], "meta": lora_meta, "errors": _lm_errors}


def _lm_wake_event() -> asyncio.Event:
    global _lm_wake
    if _lm_wake is None:
        _lm_wake = asyncio.Event()
    return _lm_wake


def _lm_wake_up() -> None:
    _lm_wake_event().set()


def lora_meta_boot() -> None:
    """Lifespan: the stored records into memory, and the snapshot's starting state — a
    configured share reads "not listed yet" (every LoRA pending) until the first pass,
    never an empty listing that would call every LoRA not_on_share."""
    global _lm_persist_lock
    _lm_persist_lock = None
    try:
        lora_meta.clear()
        lora_meta.update(store.lora_meta_all())
        _lm_state["store_ok"] = True
    except Exception as e:                               # noqa: BLE001 — never block the boot
        # Memory is EMPTY now: writing from it would replace every stored record —
        # curated lists included. Nothing is written until a restart reads the store.
        _lm_state["store_ok"] = False
        logger.warning(f"lora meta: store read failed: {type(e).__name__}: {e}")
    configured = bool(_modelsrc_host())
    _lm_state.update(configured=configured, problem="not listed yet" if configured else "")


async def _lm_persist(sha: str, rec: dict) -> bool:
    """Write the record of `sha` to the store → True when written. Writes are serialised
    and each carries what memory holds when its turn comes (not the caller's copy), so a
    worker write and an operator's Save for one sha cannot land out of order and drop
    `curated`."""
    global _lm_persist_lock
    if not _lm_state["store_ok"]:
        if not _lm_state.get("store_warned"):
            _lm_state["store_warned"] = True
            logger.warning("lora meta: store unreadable at boot — not writing trigger words")
        return False
    if _lm_persist_lock is None:
        _lm_persist_lock = asyncio.Lock()
    async with _lm_persist_lock:
        try:
            await asyncio.to_thread(store.lora_meta_put, sha, lora_meta.get(sha, rec))
            return True
        except Exception as e:                           # noqa: BLE001 — the memory value stands
            logger.warning(f"lora meta: store write failed for {sha[:12]}: {type(e).__name__}")
            _lm_state["last_error"] = f"store write failed: {type(e).__name__}"
            return False


def _lm_backoff(key: str, error: str, now: float, fixed: Optional[float] = None) -> None:
    lo, hi = _CIVITAI_BACKOFF
    prev = (_lm_errors.get(key) or {}).get("backoff_s") or 0.0
    b = fixed if fixed is not None else min(hi, max(lo, prev * 2))
    _lm_errors[key] = {"error": error, "next_try": now + b, "backoff_s": b}
    _lm_state["last_error"] = error


def _lm_wants_civitai(sha: str, now: float) -> bool:
    if (_lm_errors.get(sha) or {}).get("next_try", 0) > now:
        return False
    return sha in _lm_refetch or "civitai" not in (lora_meta.get(sha) or {})


async def _lm_apply(sha: str, kind: str, data, now: float) -> bool:
    """One Civitai answer → memory (+ the store for 200/404; a transient failure stays
    in memory, N2). False = stop asking this pass (429)."""
    if kind in ("found", "not_found"):
        rec = dict(lora_meta.get(sha) or {})               # a copy: `curated` is kept as is
        rec["civitai"] = data if kind == "found" else loratags.not_found_record(now)
        lora_meta[sha] = rec
        _lm_errors.pop(sha, None)
        _lm_refetch.discard(sha)
        _lm_state["pause_s"] = 0.0
        await _lm_persist(sha, rec)
        return True
    if kind == "busy":
        if data is not None:
            pause = float(data)
        else:
            lo, hi = _CIVITAI_PAUSE
            pause = min(hi, max(lo, _lm_state["pause_s"] * 2))
            _lm_state["pause_s"] = pause
        _lm_state["pause_until"] = now + pause
        _lm_state["last_error"] = f"Civitai rate limit (429) — paused for {int(pause)} s"
        return False
    _lm_backoff(sha, f"Civitai: {data}", now)
    return True


async def lora_meta_pass() -> Optional[float]:
    """One worker pass: list the share (TTL'd), ask Civitai for every hash that needs it
    (before the next hash starts — results show while the rest still hashes), then hash
    at most ONE share LoRA at priority 3. → the pause before the next pass (after a
    hash: as long as it took, at least 10 s — the share reads for this at most half the
    time, N3), None = the regular tick."""
    if not _modelsrc_host():
        _lm_state.update(configured=False, problem="", share={}, shas={})
        return None
    lan = modelsrc()
    await lan.refresh()
    problem = lan.problem()
    _lm_state.update(configured=True, problem=problem)
    if problem:
        return None
    share = loratags.share_loras(lan.cached())
    files = await asyncio.to_thread(lan.sha_files)
    offered = set()
    for names in list(backend_loras.values()):
        for n in names:
            p, _ = loratags.share_path(n, share)
            if p is not None:
                offered.add(share[p][0])
    reals: dict = {}                                     # real path → size, in work order
    for p in sorted(share, key=lambda p: (share[p][0] not in offered, p)):
        reals.setdefault(*share[p])
    shas = {r: files[r][1] for r, n in reals.items() if r in files and files[r][0] == n}
    _lm_state.update(share=share, shas=shas)
    for sha in dict.fromkeys(shas.values()):
        now = time.time()
        if now < _lm_state["pause_until"]:
            break
        if not _lm_wants_civitai(sha, now):
            continue
        gap = _CIVITAI_MIN_GAP_S - (time.monotonic() - _lm_state["last_civitai"])
        if gap > 0:
            await asyncio.sleep(gap)
        _lm_state["last_civitai"] = time.monotonic()
        kind, data = await _civitai_lookup(sha)
        if not await _lm_apply(sha, kind, data, time.time()):
            break
    for real, size in reals.items():
        if real in shas or (_lm_errors.get(real) or {}).get("next_try", 0) > time.time():
            continue
        t0 = time.monotonic()
        _lm_state["hashing"] = real
        try:
            sha = await lan.sha256(real, size, background=3)
            _lm_errors.pop(real, None)
            new = dict(_lm_state["shas"])
            new[real] = sha
            _lm_state["shas"] = new
        except Exception as e:                           # noqa: BLE001 — else the next pass
            _lm_backoff(real, f"hash: {e}" if isinstance(e, RuntimeError)   # picks this file again
                        else f"hash: {type(e).__name__}: {e}",
                        time.time(), fixed=_LM_HASH_RETRY_S)
        finally:
            _lm_state["hashing"] = None
        return max(_LM_HASH_PAUSE_MIN_S, time.monotonic() - t0)
    return None


async def lora_meta_loop() -> None:
    """The lifespan task: a pass, then the pause it asked for (or the 60-s tick); a
    console refresh wakes it early. A failing pass is logged, never the loop's end."""
    while True:
        try:
            pause = await lora_meta_pass()
        except asyncio.CancelledError:
            raise
        except Exception as e:                           # noqa: BLE001 — never die
            logger.warning(f"lora meta: pass failed: {type(e).__name__}: {e}")
            _lm_state["last_error"] = f"pass failed: {type(e).__name__}"
            pause = None
        _lm_state["last_run"] = time.time()
        ev = _lm_wake_event()
        try:
            await asyncio.wait_for(ev.wait(), pause or _LM_TICK_S)
        except asyncio.TimeoutError:
            pass
        ev.clear()


async def lora_refresh_all() -> str:
    """Console: ask Civitai again for every current share LoRA's hash — earlier 404s
    included; nothing is re-hashed (54 GB) and no curated list is touched."""
    shas = set(_lm_state["shas"].values())
    _lm_refetch.update(shas)
    for s in shas:
        _lm_errors.pop(s, None)
    _lm_wake_up()
    return f"asking Civitai again for {len(shas)} LoRA hash(es)"


async def lora_refresh(name: str) -> str:
    """Console: one LoRA — forget its cached sha (re-hash: catches a change in place at
    the same size) and ask Civitai again. The curated list stays."""
    st = _lm_state
    if not st["configured"] or st["problem"]:
        return f"not refreshed: the LAN share is not usable ({st['problem'] or 'not set up'})"
    path, why = loratags.share_path(name, st["share"])
    if path is None:
        return f"not refreshed: {name} is not a LoRA on the share ({why})"
    real, size = st["share"][path]
    sha = st["shas"].get(real)
    await asyncio.to_thread(modelsrc().forget_sha, real, size)
    st["shas"] = {r: h for r, h in st["shas"].items() if r != real}
    _lm_errors.pop(real, None)
    if sha:
        _lm_refetch.add(sha)
        _lm_errors.pop(sha, None)
    _lm_wake_up()
    return f"refreshing {name}: re-hash and ask Civitai again"


async def lora_curate(sha: str, words: Optional[list]) -> str:
    """Console: the curated trigger words of the LoRA file `sha` (None = back to
    Civitai's list; [] = deliberately none). → refusal text, "" = saved. Refused for a
    sha no current share LoRA has — the form showed a file that changed since (F1)."""
    if not loratags.valid_sha(sha) or sha not in set(_lm_state["shas"].values()):
        return "the file changed since this page was loaded — reload and edit again"
    if not _lm_state["store_ok"]:
        return _LM_STORE_BAD
    rec = dict(lora_meta.get(sha) or {})
    if words is None:
        rec["curated"], rec["curated_at"] = None, None
    else:
        rec["curated"], rec["curated_at"] = loratags.clean_words(list(words)), time.time()
    lora_meta[sha] = rec
    if not await _lm_persist(sha, rec):
        return "not saved to the store (kept in memory until a restart) — see the last error"
    return ""


def lora_meta_view() -> dict:
    """The LoRAs tab: one row per LoRA a ComfyUI backend offers plus every share LoRA
    no backend offers (by its name under models/loras/), each a `loratags.lookup` item
    with `backends`; the worker's counts and state. Memory only."""
    snap = _lm_snapshot()
    hosts: dict = {}
    for b in backends:
        if b.get("type") == "comfyui":
            for n in backend_loras.get(backend_id(b), set()):
                hosts.setdefault(n, []).append(b["name"])
    names = set(hosts)
    usable = snap["configured"] and not snap["problem"]
    if usable:
        claimed = {loratags.share_path(n, snap["share"])[0] for n in names}
        names |= {p[len(loratags.LORA_ROOT):] for p in snap["share"] if p not in claimed}
    rows = []
    for n in sorted(names):
        it = loratags.lookup(n, snap)
        it["backends"] = sorted(hosts.get(n, []))
        rows.append(it)
    now = time.time()
    reals = {v[0] for v in snap["share"].values()} if usable else set()
    shas = {snap["shas"][r] for r in reals if r in snap["shas"]}

    def civ(s):
        return ((lora_meta.get(s) or {}).get("civitai") or {}).get("status")
    counts = {"files": len(reals), "hashed": sum(1 for r in reals if r in snap["shas"]),
              "found": sum(1 for s in shas if civ(s) == "found"),
              "not_found": sum(1 for s in shas if civ(s) == "not_found")}
    counts["civitai_pending"] = len(shas) - counts["found"] - counts["not_found"]
    unhashed = [r for r in reals if r not in snap["shas"]
                and (_lm_errors.get(r) or {}).get("next_try", 0) <= now]
    paused = now < _lm_state["pause_until"]            # Civitai waits; hashing still counts
    busy = bool(usable and (_lm_state["hashing"] or unhashed
                            or (not paused and any(_lm_wants_civitai(s, now) for s in shas))))
    pause = _lm_state["pause_until"]
    return {"configured": snap["configured"], "problem": snap["problem"], "rows": rows,
            "counts": counts, "hashing": _lm_state["hashing"],
            "pause_until": pause if pause > now else None,
            "store_ok": _lm_state["store_ok"],
            "last_error": _lm_state["last_error"], "last_run": _lm_state["last_run"],
            "busy": busy}


# ── model sources, Stage 3: Check & save (spec "enter a URL once, verified") ─────
# The operator names a public URL for a share file (or a Hugging Face repo for a share
# directory); the gateway HEADs it and compares with the share before the catalog
# entry is written — never on the request path: a task, ONE check at a time.
#
# The HEAD is the gateway reaching out on an operator's word, with the HF token: the
# SSRF rule of `_fetch_ref_url` holds on EVERY hop (review C-2) — each name resolved,
# every address `ref_addr_blocked`, the connection made to exactly the checked address
# (original Host + SNI, no second lookup to rebind), no automatic redirects (each
# `Location` is a new hop through the same rule, ≤ `_HEAD_MAX_HOPS`). The HF token rides
# on the FIRST hop only and only to `hostctl._HF_HOSTS`. HF's 302 carries the facts
# (`X-Linked-Size`, `X-Linked-Etag` = the LFS/Xet content sha256, `X-Repo-Commit`); the
# CDN's final ETag is no sha256 and `X-Xet-Hash` is another hash — both ignored. Refusals
# are FIXED texts: a response body is never read, let alone echoed.
_HEAD_MAX_HOPS = 5
_HEAD_TIMEOUT_S = 20.0
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DIGITS = re.compile(r"[0-9]{1,18}")


@dataclass
class UrlHead:
    """What a HEAD of a source URL said. `error` "" = answered: `size` (`X-Linked-Size`
    of the first response, else the last `Content-Length`; None = unknown), `sha256`
    (the first response's `X-Linked-Etag` when it is a 64-hex content hash),
    `commit` (`X-Repo-Commit`, 40 hex), `status` of the last response, `hops` made."""
    error: str = ""
    size: Optional[int] = None
    sha256: Optional[str] = None
    commit: Optional[str] = None
    status: Optional[int] = None
    hops: int = 0


def _head_size(v) -> Optional[int]:
    s = str(v or "").strip()
    n = int(s) if _DIGITS.fullmatch(s) else None
    return n if n else None                 # 0 = unknown: no model file is empty


def _linked_etag(v) -> Optional[str]:
    """`X-Linked-Etag` → the sha256 it names, else None (a 40-hex sha1 of a non-LFS
    file, a CDN etag, junk). Quotes and a weak `W/` prefix are stripped."""
    s = str(v or "").strip()
    if s.startswith("W/"):
        s = s[2:].strip()
    s = s.strip('"')
    return s if _HEX64.fullmatch(s) else None


async def _head_ref_url(url: str, hf_token_ok: bool = True,
                        token: Optional[str] = None) -> UrlHead:
    """HEAD a source URL under the rules above → `UrlHead` (never raises).
    `hf_token_ok` False = never send the HF token (it is sent at most on the first hop,
    and only to huggingface.co/hf.co). `token` = the HF token read once by the caller (a
    directory check HEADs N files; None = read it here). The HF headers are believed only
    from an HF host (review-3 M-2): another server's `X-Linked-*` would end the HEAD early
    or earn "verified by sha256" on that server's word — its file is "size only"."""
    res = UrlHead()
    first = urlparse(str(url or ""))
    hf = (first.hostname or "").lower() in hostctl._HF_HOSTS
    tok = ""
    if hf_token_ok and hf:
        tok = _thunder_hf_token() if token is None else str(token or "")
        tok = tok if hostctl.hf_token_ok(tok) else ""
    token = tok
    cur = str(url or "")
    for hop in range(_HEAD_MAX_HOPS + 1):
        u = urlparse(cur)
        host = u.hostname
        if u.scheme != "https" or not host:
            res.error = "redirect to a non-https URL" if hop else "the URL must be https://"
            return res
        try:
            port = u.port or 443
        except ValueError:
            res.error = "redirect to an invalid port" if hop else "the URL has an invalid port"
            return res
        try:
            addrs = await _resolve_ref_host(host, port)
        except (OSError, UnicodeError):
            addrs = []
        if not addrs:
            res.error = ("redirect to a host that does not resolve" if hop
                         else "the URL's host does not resolve")
            return res
        if any(ref_addr_blocked(a) for a in addrs):
            logger.warning(f"model source check refused: {host} resolves to a non-public "
                           f"address (hop {hop + 1})")
            res.error = ("redirect to a private address" if hop
                         else "the URL's host resolves to a private address")
            return res
        ip = addrs[0].split("%", 1)[0]
        pinned = u._replace(netloc=(f"[{ip}]" if ":" in ip else ip) + f":{port}").geturl()
        headers = {"Host": u.netloc.rsplit("@", 1)[-1], "Accept-Encoding": "identity"}
        if hop == 0 and token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            r = await http_client.request("HEAD", pinned, headers=headers,
                                          extensions={"sni_hostname": host},
                                          timeout=_HEAD_TIMEOUT_S, follow_redirects=False)
        except Exception as e:
            res.error = f"the URL did not answer ({type(e).__name__})"
            return res
        res.hops, res.status = hop + 1, r.status_code
        h = r.headers
        if hop == 0 and hf:
            # HF's facts sit on ITS response (the 302 for an LFS file), not the CDN's
            res.size = _head_size(h.get("x-linked-size"))
            res.sha256 = _linked_etag(h.get("x-linked-etag"))
            commit = str(h.get("x-repo-commit") or "").strip()
            res.commit = commit if _HEX40.fullmatch(commit) else None
        if r.status_code in (301, 302, 303, 307, 308):
            if res.size is not None:
                return res                  # the size is known: no further connection
            loc = h.get("location")
            if not loc:
                res.error = "redirect without a Location"
                return res
            cur = urljoin(cur, loc)
            continue
        if not 200 <= r.status_code < 300:
            res.error = f"HTTP {r.status_code}"
            if (hop and hf and r.status_code in (401, 403)
                    and (u.hostname or "").lower() in hostctl._HF_HOSTS):
                # the token rides on the first hop only (a gated repo behind a rename)
                res.error += (" after a redirect within Hugging Face — enter the URL the "
                              "redirect names")
            return res
        if res.size is None:
            res.size = _head_size(h.get("content-length"))
        if res.size is None:
            res.error = "the URL names no size"
        return res
    res.error = f"more than {_HEAD_MAX_HOPS} redirects"
    return res


# The checks. `_src_checks` (key = the share path, or the directory ending in `/`) is
# what the overview reads through `source_checks()`; ONE runs at a time (`_src_lock`),
# the rest wait as `queued`. A directory's provisional rows are confirmed by background
# share hashes (`_src_confirming`) after the check itself is done.
_SRC_PENDING = ("queued", "heading", "hashing")
_SRC_CHECKS_MAX = 200
_src_checks: dict = {}
_src_check_tasks: dict = {}
_src_confirming: dict = {}                  # plan path → (dir key, run token) it confirms
# the background confirmation of a directory check, per dir key, and the token of the
# CURRENT run: a re-check or a remove cancels the old task, and a hash that already ran
# writes nothing for a run that is no longer current (review-3 I-1)
_src_confirm_tasks: dict = {}
_src_confirm_run: dict = {}
_src_run_seq = [0]
_src_lock_obj: Optional[tuple] = None       # (loop, asyncio.Lock): one per event loop

# Writers of `modelsync_catalog` (Check & save, remove, the console's JSON editor): one
# lock, so no read-modify-write interleaves with another (a store write takes ms; the
# lock is never held across an await).
_catalog_lock = threading.Lock()
CATALOG_STALE = ("the catalog changed since this form was opened — your text is kept "
                 "below; merge and save again")


def modelsync_catalog_hash(cat=None) -> str:
    """The hash the catalog editor renders with and `save_modelsync_catalog(…,
    expect_hash=)` compares (Task 4's stale-form refusal, R-3). The stored catalog is read
    without seeding (the default's hash equals the seeded default's)."""
    cat = _modelsync_catalog_view() if cat is None else cat
    try:
        blob = json.dumps(cat, sort_keys=True, default=str)
    except TypeError:
        blob = json.dumps(cat, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _src_lock() -> asyncio.Lock:
    global _src_lock_obj
    loop = asyncio.get_running_loop()
    if _src_lock_obj is None or _src_lock_obj[0] is not loop:
        _src_lock_obj = (loop, asyncio.Lock())
    return _src_lock_obj[1]


def source_checks() -> dict:
    """`{key: status}` of the Check & save runs this process knows (the overview's
    "queued / heading / hashing / done / refused" + reason). A status carries `kind`
    (file|dir), `url` or `repo` + `rev` (+ `commit` once resolved), `state`, `reason`
    (refused), `note` (done: "verified by
    sha256" / "size-verified only" / provisional count), `left_out` ({relpath: reason}
    — a directory's files that got no row), `outdated` ({relpath: reason} — a
    provisional row the share's hash disagreed with), `confirming` (rows still waiting
    for their share hash), `progress` ("3 of 12") and `at`."""
    return copy.deepcopy(_src_checks)


def source_checks_pending() -> bool:
    """A check queued or running, or a directory's share hash still pending — the
    overview is live while this holds."""
    return (any(s.get("state") in _SRC_PENDING for s in _src_checks.values())
            or bool(_src_confirming))


def _src_stop_confirm(key: str) -> None:
    """End the background confirmation of directory `key` (a re-check, a remove): its
    queued hashes leave the LanSource queue (the cancel), a running one finishes but
    writes nothing (`_src_confirm_run` no longer names its token)."""
    _src_confirm_run.pop(key, None)
    t = _src_confirm_tasks.pop(key, None)
    if t is not None and not t.done():
        t.cancel()


def _src_set(key: str, **kw) -> None:
    st = _src_checks.get(key)
    if st is not None:
        st.update(kw, at=time.time())


def _src_refuse(key: str, reason: str) -> None:
    _src_set(key, state="refused", reason=reason, progress="")
    logger.info(f"[model sources] {key}: not saved — {reason}")


def _src_trim() -> None:
    while len(_src_checks) > _SRC_CHECKS_MAX:
        old = next((k for k, s in _src_checks.items() if s.get("state") not in _SRC_PENDING),
                   None)
        if old is None:
            return
        del _src_checks[old]


def _queue_source_check(key: str, info: dict, factory) -> str:
    cur = _src_checks.get(key)
    if cur is not None and cur.get("state") in _SRC_PENDING:
        return f"a check of {key} is already {cur['state']}"
    _src_checks.pop(key, None)
    _src_checks[key] = dict(info, state="queued", reason="", note="", left_out={},
                            outdated={}, confirming=0, progress="", at=time.time())
    _src_trim()
    _src_check_tasks[key] = _bg(_run_source_check(key, factory))
    return f"check of {key} queued"


async def _run_source_check(key: str, factory) -> None:
    try:
        async with _src_lock():
            if _src_checks.get(key, {}).get("state") != "queued":
                return                          # removed while it waited
            await factory()
    except asyncio.CancelledError:
        _src_refuse(key, "cancelled")
        raise
    except Exception as e:
        logger.warning(f"[model sources] check of {key} failed: {type(e).__name__}: {e}")
        _src_refuse(key, "the check failed — see the gateway log")
    finally:
        if _src_check_tasks.get(key) is asyncio.current_task():
            del _src_check_tasks[key]


async def _src_listing(key: str):
    """(LanSource, the share listing) for a check, else None after refusing it."""
    lan = modelsrc()
    await lan.refresh()
    idx = lan.cached()
    if not lan.configured() or not idx:
        _src_refuse(key, f"the share is not usable ({lan.problem() or 'not listed yet'}) — "
                         "press List now")
        return None
    return lan, idx


async def _share_hash(key: str, lan, path: str, size: int) -> Optional[str]:
    """The share's sha256 for a check: the persistent cache, else a hash queued behind
    every transfer's and ahead of every directory confirmation (priority 1). None after
    refusing the check."""
    sha = lan.known_sha(path, size)
    if sha is not None:
        return sha
    _src_set(key, state="hashing")
    try:
        return await lan.sha256(path, size, background=1)
    except RuntimeError as e:
        logger.warning(f"[model sources] {path}: share sha256 failed: {e}")
        _src_refuse(key, "the share's sha256 could not be computed — see the gateway log")
        return None


def _catalog_write(fn) -> list:
    """Read-modify-write of `modelsync_catalog` under `_catalog_lock`: `fn(copy)` → the
    new list, or a str refusal. → [] written, else the refusal(s)."""
    if not store.is_active():
        return ["the store is not active — the catalog cannot be saved"]
    with _catalog_lock:
        cat = copy.deepcopy(_modelsync_catalog())
        out = fn(cat)
        if isinstance(out, str):
            return [out]
        if out is not None:
            store.set_settings({_MODELSYNC_CATALOG_KEY: out})
    return []


def _put_entry(entry: dict, same, still=None) -> list:
    """Write `entry` in place of the catalog entries `same(e)` names (its position: the
    first of them), else at the end. Only the NEW entry is validated — an unrelated
    broken entry (dropped one by one by modelsync anyway) never blocks a check."""
    errs = modelsync.validate_catalog([entry])
    if errs:
        return [e.removeprefix("entry 1: ") for e in errs]

    def fn(cat):
        if still is not None and not still():
            return "removed while it was checked"
        out, placed = [], False
        for e in cat:
            if isinstance(e, dict) and same(e):
                if not placed:
                    out.append(entry)
                    placed = True
                continue
            out.append(e)
        return out if placed else out + [entry]
    return _catalog_write(fn)


SRC_URL_REFUSED = ("the URL must start with https:// and contain no whitespace, quote or "
                   "control character")


async def check_source(path: str, url: str) -> str:
    """Check & save for ONE share file (spec Stage 3): queued, then HEAD → share size
    compare → share sha256 (cache, else hashed) → accept iff the size is equal AND an
    LFS sha256 the URL named equals the share's (none named: "size-verified only") →
    `{file, url, size, sha256: <the SHARE's>, verified}` written in place of the path's
    entry. → what happened, for the console banner."""
    path, url = str(path or "").strip(), str(url or "").strip()
    # the answer becomes the banner — a `?msg=` redirect (browser history, the access
    # log) — so it never carries the typed URL, whose query may hold a token: a FIXED
    # text for a refused URL, and the path's own refusal only when the URL is fine
    if modelsync._url_error(url):
        return "not checked: " + SRC_URL_REFUSED
    errs = modelsync.validate_catalog([{"file": path, "url": url}])
    if errs:
        msg = errs[0].removeprefix("entry 1: ")
        return "not checked: " + (SRC_URL_REFUSED if url in msg else msg)
    return _queue_source_check(path, {"kind": "file", "url": url},
                               lambda: _check_file(path, url))


async def _check_file(path: str, url: str) -> None:
    got = await _src_listing(path)
    if got is None:
        return
    lan, idx = got
    size = idx.get(path)
    if not isinstance(size, int) or isinstance(size, bool):
        _src_refuse(path, "the share does not list this file")
        return
    _src_set(path, state="heading")
    h = await _head_ref_url(url, token=await asyncio.to_thread(_thunder_hf_token))
    if h.error:
        _src_refuse(path, h.error)
        return
    if h.size != size:
        _src_refuse(path, f"size differs: share {modelsync.size_text(size)}, URL "
                          f"{modelsync.size_text(h.size)}")
        return
    share = await _share_hash(path, lan, path, size)
    if share is None:
        return
    if h.sha256 is not None and h.sha256 != share:
        _src_refuse(path, "hash differs: the URL's sha256 (X-Linked-Etag) is not the share "
                          "file's")
        return
    verified = "sha256" if h.sha256 is not None else "size"
    entry = {"file": path, "url": url, "size": size, "sha256": share, "verified": verified}
    mine = _src_checks.get(path)
    errs = await asyncio.to_thread(_put_entry, entry, lambda e: e.get("file") == path,
                                   lambda: _src_checks.get(path) is mine)
    if errs:
        _src_refuse(path, "not saved: " + errs[0])
        return
    _src_set(path, state="done", reason="", progress="",
             note="verified by sha256" if verified == "sha256" else "size-verified only")
    logger.info(f"[model sources] {path}: URL source saved ({verified})")
    _forget_fallbacks([path])


async def check_dir_source(dir_: str, repo: str, rev: str = "main") -> str:
    """Check & save for a share DIRECTORY against a Hugging Face repo (spec Stage 3,
    R-1/R-6): `rev` resolved ONCE to the commit (`X-Repo-Commit`) and every further HEAD
    made at `resolve/<commit>/<relpath>`; each share file under `dir_` is accepted on
    its SIZE (`modelsync.dir_check_row`), a file whose HEAD fails or whose size differs
    is left OUT (→ LAN, listed with its reason), and the check is refused only when NO
    file verifies. Rows without the share's sha256 are confirmed by background hashes
    afterwards (a provisional one the share disagrees with turns that file outdated)."""
    d, repo, rev = str(dir_ or "").strip(), str(repo or "").strip(), str(rev or "").strip()
    if d and not d.endswith("/"):
        d += "/"
    err = modelsync.dir_source_error(d, repo)
    if err:
        return f"not checked: {err}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", rev) or ".." in rev:
        return f"not checked: rev {rev!r} is no branch, tag or commit name"
    return _queue_source_check(d, {"kind": "dir", "repo": repo, "rev": rev},
                               lambda: _check_dir(d, repo, rev))


async def _check_dir(d: str, repo: str, rev: str) -> None:
    got = await _src_listing(d)
    if got is None:
        return
    lan, idx = got
    files = {p[len(d):]: n for p, n in modelsync.expand_dir(d, idx).items()
             if isinstance(n, int) and not isinstance(n, bool) and not modelsync._is_part(p)}
    if not files:
        _src_refuse(d, "the share lists no file under this directory")
        return
    commit = rev if _HEX40.fullmatch(rev) else None
    rows, left = {}, {}
    _src_set(d, state="heading")
    token = await asyncio.to_thread(_thunder_hf_token)     # once, off the loop (M-4)
    for i, rel in enumerate(sorted(files)):
        _src_set(d, progress=f"{i + 1} of {len(files)} · {rel}")
        size = files[rel]
        h = await _head_ref_url(modelsync.hf_resolve_url(repo, commit or rev, rel),
                                token=token)
        if commit is None:
            if h.commit:
                commit = h.commit           # ONE commit for the whole entry (R-6 c)
            elif not h.error:
                _src_refuse(d, "the URL named no commit (X-Repo-Commit) — is it a "
                               "Hugging Face model repo?")
                return
        if h.error:
            left[rel] = h.error
            continue
        row, why = modelsync.dir_check_row(size, h.size, h.sha256, lan.known_sha(d + rel, size))
        if row is None:
            left[rel] = why
        else:
            rows[rel] = row
    _src_set(d, left_out=left)
    if not rows or commit is None:
        first = next(iter(left.values()), "")
        _src_refuse(d, "no file of this directory verified" + (f" ({first})" if first else ""))
        return
    entry = {"dir": d, "repo": repo, "rev": commit, "files": rows}
    mine = _src_checks.get(d)
    errs = await asyncio.to_thread(_put_entry, entry, lambda e: e.get("dir") == d,
                                   lambda: _src_checks.get(d) is mine)
    if errs:
        _src_refuse(d, "not saved: " + errs[0])
        return
    todo = [(rel, row[0]) for rel, row in sorted(rows.items()) if row[1] is None or row[2]]
    note = f"{len(rows)} of {len(files)} files verified by size"
    if todo:
        note += f" — {len(todo)} wait for the share's sha256"
    _src_set(d, state="done", reason="", progress="", note=note, confirming=len(todo),
             commit=commit)
    logger.info(f"[model sources] {d}: directory source saved ({len(rows)} of {len(files)} "
                f"files, commit {commit})")
    _src_stop_confirm(d)
    _forget_fallbacks([d + rel for rel in rows])
    if todo:
        _src_run_seq[0] += 1
        run = (d, _src_run_seq[0])
        _src_confirm_run[d] = run
        for rel, _ in todo:
            _src_confirming[d + rel] = run
        _src_confirm_tasks[d] = _bg(_confirm_dir_rows(d, repo, commit, todo, run))


async def _confirm_dir_rows(d: str, repo: str, commit: str, todo: list, run=None) -> None:
    """The background half of a directory check: the share's sha256 of every row that
    lacks it (priority 2 — behind transfers AND the next Check & save), then the row
    confirmed — or, for a provisional sha the share disagrees with, left as it is: the
    persistent share-sha cache now makes `modelsync.source_kinds` call that file
    outdated. `run` is this run's token: a newer run or a remove ends this one, and
    only this run's `_src_confirming` markers are popped."""
    lan = modelsrc()
    current = lambda: _src_confirm_run.get(d) == run
    try:
        for rel, size in todo:
            path = d + rel
            if not current():
                return
            try:
                share = await lan.sha256(path, size, background=2)
            except RuntimeError as e:
                logger.warning(f"[model sources] {path}: share sha256 failed: {e}")
                share = None
            if not current():
                return                      # re-checked or removed while it hashed
            verdict = ("failed" if share is None else
                       await asyncio.to_thread(_confirm_row, d, repo, commit, rel, size,
                                               share, run))
            if _src_confirming.get(path) == run:
                del _src_confirming[path]
            st = _src_checks.get(d)
            if st is not None and current():
                st["confirming"] = max(0, int(st.get("confirming") or 0) - 1)
                if verdict == "outdated":
                    st["outdated"][rel] = ("hash differs: the share's copy differs from "
                                           "the Hugging Face copy")
                elif verdict == "failed":
                    st["outdated"][rel] = "the share's sha256 could not be computed"
                st["at"] = time.time()
    finally:
        for rel, _ in todo:
            if _src_confirming.get(d + rel) == run:
                del _src_confirming[d + rel]
        if _src_confirm_tasks.get(d) is asyncio.current_task():
            del _src_confirm_tasks[d]
            if current():
                _src_confirm_run.pop(d, None)


def _confirm_row(d: str, repo: str, commit: str, rel: str, size: int, share: str,
                 run=None) -> str:
    """Apply one background share hash to the stored dir entry → "confirmed",
    "outdated" (left provisional; the cache says it differs) or "gone" (the entry or the
    row changed since, or the run is no longer current — nothing written)."""
    result = ["gone"]

    def fn(cat):
        if run is not None and _src_confirm_run.get(d) != run:
            return None                     # judged under the lock: a remove won
        for e in cat:
            if not (isinstance(e, dict) and e.get("dir") == d and e.get("repo") == repo
                    and e.get("rev") == commit and isinstance(e.get("files"), dict)):
                continue
            row = e["files"].get(rel)
            if not (isinstance(row, list) and len(row) == 3 and row[0] == size):
                return None
            if row[1] is not None and str(row[1]).lower() != share:
                result[0] = "outdated"
                return None
            e["files"][rel] = [size, share, False]
            result[0] = "confirmed"
            return cat
        return None
    errs = _catalog_write(fn)
    return result[0] if not errs else "gone"


def _forget_fallbacks(paths) -> None:
    """After a successful Check & save write or a remove: every host controller forgets
    its URL-fallback records of `paths` (final review I-1) — in any phase, an `off` host
    included; otherwise a URL the operator just re-verified stayed "given up" and the
    next session streamed the file from the LAN. Best-effort: a controller that fails
    is logged, the others still run."""
    paths = [p for p in paths if isinstance(p, str)]
    if not paths:
        return
    for name, c in list(host_controllers.items()):
        try:
            n = c.forget_fallback(paths)
        except Exception as e:
            logger.warning(f"[model sources] host {name}: fallback records not cleared: "
                           f"{type(e).__name__}: {e}")
            continue
        if n:
            logger.info(f"[model sources] host {name}: {n} URL fallback record(s) cleared")


async def remove_source(key: str) -> str:
    """The overview's "remove": drop the per-file source entry (`key` = its path) or
    the directory source (`key` ending in `/`) — and a check of it still waiting, and a
    directory's background confirmation. ASYNC on purpose (review-3 RR-1): the task
    cancels must run ON the loop — `Task.cancel()` from a worker thread is not
    thread-safe and may be lost — so only the catalog write goes to a thread. Await it
    on the loop; never wrap it in `asyncio.to_thread`."""
    key = str(key or "").strip()
    found = [False]
    # loop side first: the status goes (an in-flight Check & save write — a worker
    # thread a cancel cannot stop — re-checks it under the catalog lock and writes
    # nothing, M-6), the confirmation run ends, the check task is cancelled
    _src_checks.pop(key, None)
    _src_stop_confirm(key)
    t = _src_check_tasks.get(key)
    if t is not None and not t.done():
        t.cancel()

    covered: list = []

    def fn(cat):
        out = []
        for e in cat:
            if isinstance(e, dict) and "url" in e and e.get("file") == key:
                covered.append(key)
            elif isinstance(e, dict) and "repo" in e and e.get("dir") == key:
                fm = e.get("files")
                covered.extend(key + rel for rel in (fm if isinstance(fm, dict) else {}))
            else:
                out.append(e)
        if len(out) == len(cat):
            return None
        found[0] = True
        return out
    errs = await asyncio.to_thread(_catalog_write, fn)
    if errs:
        return f"not removed: {errs[0]}"
    if not found[0]:
        return f"no source entry for {key}"
    _forget_fallbacks(covered)
    logger.info(f"[model sources] {key}: source removed")
    return f"source of {key} removed"


# ── model sources: the overview (Server → Models → "Model sources") ─────────────────
# Which NEEDED files (spec Concepts: `per_alias[*].files` of a plan over every ComfyUI
# backend's aliases against the share listing — never `fetch`, which drops the blocked
# aliases, the ones most worth seeing) come from a public URL and which only from the
# LAN share. No running host needed. Planning reads the store, the workflows and the
# whole listing, so the view is built in a worker thread (`model_sources_overview`) and
# memoised on exactly what it reads; the live parts (checks, hash queue, the
# controllers' fallbacks) are laid over it per call, on the loop.
_msrc_memo: list = [None, None]             # [key, view]
_msrc_kinds_memo: list = [None, None]       # [key, source_kinds]
_msrc_memo_lock = threading.Lock()


def _comfy_backend_names() -> list:
    """Every ComfyUI backend name (config + store: `backends` is the merged list)."""
    return sorted({str(b.get("name")) for b in list(backends)
                   if isinstance(b, dict) and b.get("type") == "comfyui" and b.get("name")})


def _source_kinds_memo(catalog, cat_hash, lan) -> dict:
    """`modelsync.source_kinds` of the share listing, memoised on (catalog hash, listing
    generation, sha-cache generation) — the card's badges read it per Backends tick."""
    key = (cat_hash, lan.generation, lan.sha_generation)
    with _msrc_memo_lock:
        if _msrc_kinds_memo[0] == key:
            return _msrc_kinds_memo[1]
    kinds = modelsync.source_kinds(catalog, lan.cached(), lan.sha_files())
    with _msrc_memo_lock:
        _msrc_kinds_memo[:] = [key, kinds]
    return kinds


def model_source_kinds() -> dict:
    """`{path: {"kind", "reason"?}}` for every share/catalog file with a public source
    (absent = `lan`) — the host card's badge. BLOCKING (store reads, one pass over the
    listing on a change): call it through `asyncio.to_thread`."""
    lan = modelsrc()
    catalog = _modelsync_catalog_view()
    kinds = _source_kinds_memo(catalog, modelsync_catalog_hash(catalog), lan)
    return {p: {k: v for k, v in i.items() if k in ("kind", "reason", "origin")}
            for p, i in kinds.items()}


def _dir_entries_by_path(catalog) -> dict:
    """`{path: dir entry}` over the valid directory entries' `files` (a later entry wins,
    as in `source_kinds`) — the key `remove_source` takes for a dir row. Each entry is
    validated ONCE (`validate_catalog` walks its whole `files` map)."""
    out: dict = {}
    for e in catalog if isinstance(catalog, list) else []:
        if (isinstance(e, dict) and "repo" in e and isinstance(e.get("dir"), str)
                and isinstance(e.get("files"), dict) and not modelsync.validate_catalog([e])):
            for rel in e["files"]:
                out[e["dir"] + rel] = e
    return out


def _overview_alias_key(names) -> str:
    """One hash over exactly what the overview's needs read besides the catalog: every
    alias candidate on one of these ComfyUI backends (a path workflow's CONTENT too).
    ONE alias read and ONE dump for all backends — a memo hit stays cheap (the catalog
    is keyed by its own hash)."""
    wanted = set(names)
    merged = dict(image_models or {})
    if store.is_active():
        merged.update(store.list_aliases())
    parts = []
    for alias in sorted(merged):
        for cand in merged[alias] or []:
            if (isinstance(cand, dict) and cand.get("backend") in wanted
                    and adapters.cand_kind(cand) == "comfyui"):
                wf = None if "workflow_json" in cand else adapters.cand_workflow(cand)
                parts.append([alias, cand, wf])
    try:
        blob = json.dumps(parts, sort_keys=True, default=str)
    except TypeError:                       # mixed key types (a YAML workflow's int ids)
        blob = json.dumps(parts, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def model_sources_view() -> dict:
    """The overview's rows (spec "The overview"): one per NEEDED file — `path`, `size`
    (the listing's, else the entry's), `in_share`, `kind` (`lan` | `url` | `hf-auto` |
    `outdated`), `url`, `origin`, `verified`, `provisional`, `reason` (outdated),
    `outdated_entry`, `entry_key` (what `remove_source` takes for the row's explicit
    entry, "" = none), `dir_entry`/`dir_repo` (a LAN file under a directory source whose
    check predates it) and `aliases` (`[[alias, blocked reason or ""]]`). A link is not a
    file: its alias is credited to the file it points at. Plus `listed` (the share has
    been listed) and `backends` (the ComfyUI backends planned over).

    BLOCKING and memoised on (the ComfyUI backend names, `_overview_alias_key` — one alias
    read and one dump —, the catalog hash, the listing generation, the sha-cache
    generation): call it via `asyncio.to_thread`
    (`model_sources_overview`). Never starts a hash or a listing."""
    lan = modelsrc()
    names = _comfy_backend_names()
    catalog = _modelsync_catalog_view()
    cat_hash = modelsync_catalog_hash(catalog)
    key = (tuple(names), _overview_alias_key(names), cat_hash, lan.generation,
           lan.sha_generation)
    with _msrc_memo_lock:
        if _msrc_memo[0] == key:
            return _msrc_memo[1]
    idx = lan.cached()
    share_sha = lan.sha_files()
    needs = []
    for n in names:
        needs += service_alias_needs(f"comfyui:{n}", catalog)
    urls = modelsync.url_catalog(catalog, idx, share_sha)
    p = modelsync.plan(needs, idx, {}, {}, urls)
    kinds = _source_kinds_memo(catalog, cat_hash, lan)
    dir_rows = _dir_entries_by_path(catalog)
    rows: dict = {}
    for alias, r in sorted(p["per_alias"].items()):
        why = "; ".join(r["blocked"])
        for f in r["files"]:
            path = f["path"]
            if f.get("link") is not None:
                path = modelsync.link_target(path, f["link"])
                if path is None:
                    continue
            row = rows.get(path)
            if row is None:
                info = kinds.get(path) or {}
                size = idx.get(path)
                in_share = isinstance(size, int) and not isinstance(size, bool)
                kind = info.get("kind", "lan")
                row = {"path": path, "size": size if in_share else info.get("size"),
                       "in_share": in_share, "kind": kind, "url": info.get("url", ""),
                       "origin": info.get("origin", ""), "verified": info.get("verified", ""),
                       "provisional": bool(info.get("provisional")),
                       "reason": info.get("reason", ""),
                       "outdated_entry": info.get("outdated_entry", ""),
                       "entry_key": "", "dir_entry": "", "dir_repo": "", "aliases": {}}
                if kind in ("url", "outdated"):
                    if info.get("origin") == "file":
                        row["entry_key"] = path
                    elif info.get("origin") == "dir":
                        d = dir_rows.get(path)
                        row["entry_key"] = d["dir"] if d else ""
                if kind == "lan":
                    d = modelsync.dir_for(path, catalog)
                    if d is not None:
                        row["dir_entry"], row["dir_repo"] = d["dir"], d["repo"]
                rows[path] = row
            row["aliases"].setdefault(alias, why)
    out = []
    for path in sorted(rows):
        row = rows[path]
        row["aliases"] = [[a, w] for a, w in sorted(row["aliases"].items())]
        out.append(row)
    view = {"rows": out, "listed": bool(idx), "backends": names}
    with _msrc_memo_lock:
        _msrc_memo[:] = [key, view]
    return view


async def model_sources_overview() -> dict:
    """The console's "Model sources" section: `model_sources_view()` (built in a worker
    thread, memoised) plus what changes while nothing else does — `fallback` (`{path:
    "<host>: <reason>[; …]"}` of every controller's "URL failed — LAN",
    `Controller.url_fallback_view` — in memory, not the whole `view()`), `checks`
    (`source_checks()`), `hashing` (the share-hash queue, running first), `pending` (a
    check or a hash still running: the section is live) and `problem` (the LAN source's,
    "" = usable)."""
    view = dict(await asyncio.to_thread(model_sources_view))
    # per HOST: a URL that failed on one instance may download fine on another, so
    # the row names every host it failed on (`"<host>: <reason>"`)
    fallback: dict = {}
    for name, c in sorted(host_controllers.items()):
        try:
            fb = c.url_fallback_view() or {}
        except Exception as e:              # a courtesy column, never the section
            logger.warning(f"[model sources] {name}: fallback view unavailable: {e!r}")
            continue
        for path, why in fb.items():
            line = f"{name}: {why or 'URL given up'}"
            fallback[str(path)] = (f"{fallback[str(path)]}; {line}" if str(path) in fallback
                                   else line)
    lan = modelsrc()
    try:
        hashing = [str(x) for x in lan.hash_queue()]
    except Exception:                       # noqa: BLE001 — a LanSource without a queue
        hashing = []
    try:
        problem = str(lan.problem() or "")
    except Exception as e:                  # noqa: BLE001
        problem = f"{type(e).__name__}"
    view.update(fallback=fallback, checks=source_checks(), hashing=hashing,
                pending=source_checks_pending() or bool(hashing), problem=problem)
    return view


def _modelsrc_prepare() -> None:
    """modelsrc.key exists before the console shows it: its public half is what the
    operator installs on the share host FIRST. Generated in the background at boot (and
    for a managed host added later), never by a page view."""
    global _modelsrc_key_task
    if not host_controllers or os.path.exists(modelsrc().key_path + ".pub"):
        return
    if _modelsrc_key_task is not None and not _modelsrc_key_task.done():
        return

    async def gen():
        try:
            await modelsrc().ensure_key()
        except Exception as e:              # the panel then shows no key; logged
            logger.warning(f"[modelsrc] key generation failed: {type(e).__name__}: {e}")
    _modelsrc_key_task = _bg(gen())


def modelsrc_view() -> dict:
    """The console's LAN-source block (LanSource.view: no network)."""
    return modelsrc().view()


async def modelsrc_scan() -> str:
    """The console's "Fetch host key": ssh-keyscan → the fingerprint to compare. Nothing
    is trusted yet."""
    lan = modelsrc()
    try:
        fp = await lan.scan()
    except (RuntimeError, ValueError) as e:
        return f"host key not fetched: {e}"
    return (f"host key of {lan.host()}: {fp} — compare it with the share host's "
            "(ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub) before confirming")


async def modelsrc_pin(fingerprint: str) -> str:
    """The console's "Confirm fingerprint": pins the key fetched before (refused when it
    is not the one `fingerprint` names), then the share is listed at the next sync."""
    try:
        fp = modelsrc().pin(str(fingerprint or "").strip())
    except (ValueError, OSError) as e:
        return f"host key not pinned: {e}"
    return f"host key {fp} pinned — press List now to list the share"


async def modelsrc_list() -> str:
    """The console's "List now": list the share at once (`LanSource.refresh(force=True)`
    — one async ssh call, the cache's own lock). It needs the share only, no running
    instance: without it the card said "not listed yet" until the next model sync of a
    RUNNING host, which read as "the pin did not work"."""
    lan = modelsrc()
    if not lan.configured():
        return f"not listed: {lan.problem()}"
    await lan.refresh(force=True)
    v = lan.view()
    if v.get("error") or not v.get("listed_at"):
        return f"not listed: {v.get('problem') or v.get('error') or 'no answer'}"
    return f"listed {v['files']} files and {v['links']} links from {lan.host()}"


def _host_deps() -> "hostctl.Deps":
    ops = _HERE / "ops"
    lan = modelsrc()
    return hostctl.Deps(
        # own client per controller (closed by Controller.aclose): provider calls must
        # not compete with proxied traffic for the shared pool, nor outlive a closed one
        client_factory=lambda: httpx.AsyncClient(),
        load_state=_host_load_state, save_state=_host_save_state,
        set_enabled=set_backend_enabled, begin_drain=begin_drain, cancel_drain=cancel_drain,
        hold_routing=_hold_routing,
        inflight=lambda bid: backend_inflight.get(bid, 0),
        is_draining=lambda bid: bid in _draining,
        note_fault=_note_fault,
        datadir=_thunder_datadir(),
        probe_comfy=_comfy_probe, probe_http=_host_probe_http,
        bootstrap_script=lambda: (ops / "thunder-bootstrap.sh").read_bytes(),
        host_bootstrap_script=lambda: (ops / "host-bootstrap.sh").read_bytes(),
        log=logger.info,
        known_uuids=_host_known_uuids,
        default_nodes=lambda: (ops / "thunder-nodes.default.txt").read_text("utf-8"),
        alias_needs=service_alias_needs, alias_signature=service_alias_signature,
        # the LAN share's last good listing ({} until pinned and listed), and the share
        # itself for its refresh, the stream and the sha256
        source_index=lan.cached, lan=lan,
        # the catalog's URL sources judged against the SAME listing the plan uses
        # (outdated entries dropped — by size, and by a confirmed share hash from the
        # persistent cache) plus the share's HF cache derived (Stage 1)
        url_catalog=lambda src: modelsync.url_catalog(_modelsync_catalog(), src,
                                                      _share_sha_files()),
        hf_token=_thunder_hf_token, control=sshrun.control)


def _host_task_done(name: str, what: str, answered: Optional[set] = None):
    def done(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        e = t.exception()
        if e is not None and not (answered and t in answered):
            # a refusal that came after the first await (start's unreconciled check),
            # or a bug — either way the console's action already answered
            logger.warning(f"[host {name}] {what}: {type(e).__name__}: {e}")
    return done


def _host_spawn(name: str, what: str, coro, answered: Optional[set] = None) -> asyncio.Task:
    t = _bg(coro)
    t.add_done_callback(_host_task_done(name, what, answered))
    held = [x for x in _host_tasks.get(name, []) if not x.done()]
    held.append(t)
    _host_tasks[name] = held
    return t


def _host_run(name: str, c) -> None:
    _host_spawn(name, "resume", c.resume())
    _host_spawn(name, "background loop", c.run_forever())


def _host_retire(name: str) -> None:
    c = host_controllers.pop(name)
    for t in _host_tasks.pop(name, []):
        t.cancel()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return                              # no loop → nothing was ever opened
    _bg(c.aclose())


def _load_managed_hosts() -> dict:
    """The store's managed hosts; the last good read when the store cannot answer (a
    read error must not look like "every host was deleted")."""
    if not store.is_active():
        return {}
    try:
        return store.get_managed_hosts()
    except Exception as e:
        logger.warning(f"managed_hosts unreadable — keeping the last read: "
                       f"{type(e).__name__}: {e}")
        return dict(managed_hosts)


def _host_idle_off(c) -> bool:
    """Off with nothing left to do — the only state a controller may be retired in. An
    `off` controller whose snapshot is still CREATING is not idle: its watcher still has
    to rotate the old snapshot out (and mark or drop the new one) — retired, both would
    sit at the provider and bill per GB-month, unseen. A start is `off` until the create."""
    return c.state.phase == "off" and c.op is None and not c.state.pending_snapshot


def _host_error(name: str, msg: str) -> None:
    _host_errors[name] = msg
    if (name, msg) not in _host_error_warned:        # once per error, not per rebuild
        _host_error_warned.add((name, msg))
        logger.warning(f"[host {name}] {msg}")


def _sync_one_host(name: str, h: dict, svcs: list) -> None:
    prov = hostapi.provider(h.get("provider"))
    c = host_controllers.get(name)
    if prov is None:
        # shown, never driven (Ruling M4) — a running controller of this name keeps its
        # last host entry and gets the service list, so it can still be stopped
        _host_error(name, f"unknown provider {h.get('provider')!r} — this host is not "
                          "driven (known: " + ", ".join(sorted(hostapi.PROVIDERS)) + ")")
        if c is not None:
            c.set_services(svcs)
        return
    if c is not None and c.kind != prov[0].KIND:
        # the provider owns the state record and the snapshots: never swapped under a
        # controller (managed_host_refusal refuses it; this is a hand-edited store)
        _host_error(name, f"provider changed to {h.get('provider')!r} — not applied; "
                          "delete the host and create a new one")
        c.set_services(svcs)
        return
    _host_errors.pop(name, None)
    _host_error_warned.difference_update({x for x in _host_error_warned if x[0] == name})
    _host_warned.discard(name)                  # entry is back: warn again if it goes
    if c is None:
        c = host_controllers[name] = hostctl.Controller(h, svcs, _host_deps())
        if _hosts_booted:
            _host_run(name, c)
            _modelsrc_prepare()
    else:
        c.host = h
        c.set_services(svcs)


def sync_host_controllers() -> None:
    """Match the controllers to the store's managed hosts and the current backend list
    (rebuild_backends calls it BEFORE it groups hosts and builds the route index, since
    an attached backend's URL may be derived here). A new host gets a controller (started
    at once after boot), an existing one keeps its INSTANCE and is handed the current
    host entry and service list, and one whose entry is gone is retired only when off
    and idle. One host that cannot be driven never stops the others — nor the rebuild."""
    global managed_hosts
    managed_hosts = _load_managed_hosts()
    config_ids = {backend_id(b) for b in config_backends}
    rows = store.list_backends() if store.is_active() else []
    attached: dict = {n: [] for n in managed_hosts}
    not_att: dict = {}
    for b in backends:
        hn = str(b.get("host") or "").strip()
        if hn not in managed_hosts:
            continue
        bid = backend_id(b)
        if bid in config_ids:
            not_att.setdefault(hn, []).append(bid)
            if (hn, bid) not in _not_attach_warned:
                _not_attach_warned.add((hn, bid))
                logger.warning(f"[host {hn}] {bid} is not attached: "
                               f"{NOT_ATTACHABLE_REASON}")
            continue
        try:
            _attach_fields(b, rows)
        except Exception as e:                  # the controller shows it down (no port)
            logger.warning(f"[host {hn}] {bid}: no forward assigned: "
                           f"{type(e).__name__}: {e}")
        attached[hn].append(b)
    _not_attach_warned.intersection_update(
        {(hn, bid) for hn, bids in not_att.items() for bid in bids})
    _host_not_attachable.clear()
    _host_not_attachable.update(not_att)
    _host_attached.clear()
    _host_attached.update({n: [backend_id(b) for b in svcs] for n, svcs in attached.items()})
    for name in list(_host_errors):
        if name not in managed_hosts:
            _host_errors.pop(name, None)
    tokens: dict = {}                       # provider kind → its token, read once per sync
    for name, entry in managed_hosts.items():
        kind = str(entry.get("provider") or "")
        if kind not in tokens:
            tokens[kind] = provider_token(kind)
        # the controller reads the token as its host's `api_key` (hostctl unchanged):
        # one token per PROVIDER, handed to every host of that provider
        h = dict(entry, name=name, api_key=tokens[kind])
        try:
            _sync_one_host(name, h, attached[name])
        except Exception as e:
            # the host dict holds the provider token, and this text reaches /health and
            # the console: the token is redacted, the message clipped
            msg = hostctl._redact(str(e), tokens[kind])
            msg = msg if len(msg) <= 200 else msg[:200] + "…"
            _host_error(name, f"not driven: {type(e).__name__}: {msg}")
    for hn in [n for n in host_controllers if n not in managed_hosts]:
        c = host_controllers[hn]
        if _host_idle_off(c):
            _host_warned.discard(hn)
            _host_retire(hn)
        elif hn not in _host_warned:        # once per controller, not per rebuild
            _host_warned.add(hn)
            what = (f"instance runs ({c.state.phase})" if c.state.phase != "off" or c.op
                    else f"snapshot {c.state.pending_snapshot} is still being taken")
            logger.warning(f"[host {hn}] managed host entry removed while {what} — "
                           "controller kept" + ("" if c.state.phase == "off" and not c.op
                                                else "; stop it from the console"))


def apply_managed_hosts() -> None:
    """After a managed-host Save/Delete: re-read the entries, hand every controller its
    current entry (a new token, new options) and services."""
    sync_host_controllers()


def _hosts_boot() -> None:
    """Lifespan, after the first discovery: resume + background loop per controller."""
    global _hosts_booted
    _hosts_booted = True
    for name, c in list(host_controllers.items()):
        _host_run(name, c)
    for name, c in list(volume_controllers.items()):
        _vol_run(name, c)
    _modelsrc_prepare()


async def _hosts_shutdown() -> None:
    """Lifespan end: stop the background work, then each controller's tunnel and
    client. The INSTANCES are not touched — they keep running and resume() finds them."""
    global _hosts_booted
    _hosts_booted = False
    tasks = [t for ts in list(_host_tasks.values()) + list(_vol_tasks.values())
             for t in ts if not t.done()]
    if _modelsrc_key_task is not None and not _modelsrc_key_task.done():
        tasks.append(_modelsrc_key_task)
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    for name, c in list(host_controllers.items()):
        try:
            await c.aclose()
        except Exception as e:
            logger.warning(f"[host {name}] shutdown: {type(e).__name__}: {e}")

    for name, c in list(volume_controllers.items()):
        try:
            await c.aclose()
        except Exception as e:
            logger.warning(f"[volume {name}] shutdown: {type(e).__name__}")


# ── host views and actions (the console's card, /health) ─────────────────────────

def host_names() -> list:
    """Every managed host the console shows: each store entry (driven or not) and each
    controller kept past its entry (its instance may still bill), sorted."""
    return sorted(set(managed_hosts) | set(host_controllers), key=lambda n: str(n).lower())


def _undriven_view(name: str, h: dict) -> dict:
    """The view of a host without a controller (an unknown provider): shown `off`, its
    attached backends `down` — never a made-up lifecycle."""
    svcs = {}
    for bid in _host_attached.get(name, []):
        b = _live_backend(bid) or {}
        svcs[bid] = {"name": str(b.get("name") or bid.partition(":")[2]),
                     "type": str(b.get("type") or "openai"),
                     "local_port": b.get("local_port"), "remote_port": b.get("remote_port"),
                     "status": "down", "error": "host not driven"}
    return {"name": name, "provider": str(h.get("provider") or ""), "phase": "off",
            "error": "", "failed_phase": "", "uptime_s": 0, "long_running": False,
            "cost_per_h": None, "session_cost": None, "snapshot": {}, "log": [],
            "transfers": [], "plan": None, "ready_aliases": [], "op": None,
            "waiting_jobs": None, "services": svcs, "orphans": [],
            "unreconciled_uuids": []}


def host_view(name: str) -> Optional[dict]:
    """A managed host's view for the console: the controller's (plus `gated_only` per
    planned alias, see `_gated_only_aliases`), else the undriven one; with the host's
    options (never its provider's token — `api_key_set` only), why it is not driven
    (`error`), the config backends that name it but are not attached (`not_attachable`,
    R-K3) and, for a driven host, `start_blockers` + `checklist` (the card's Start)."""
    c = host_controllers.get(name)
    h = managed_hosts.get(name)
    if c is None and h is None:
        return None
    if c is not None:
        v = c.view()
        rows = (v.get("plan") or {}).get("aliases") or {}
        try:
            only = _gated_only_aliases([a for a, r in rows.items() if not r.get("ready")])
        except Exception as e:              # the panel note is a courtesy, never an error
            logger.warning(f"[host {name}] alias gate note unavailable: {e!r}")
            only = set()
        for a, r in rows.items():
            r["gated_only"] = a in only
    else:
        v = _undriven_view(name, h)
    src = h if h is not None else getattr(c, "host", None)
    src = src if isinstance(src, dict) else {}
    opts = src.get("options") if isinstance(src.get("options"), dict) else {}
    v["options"] = copy.deepcopy(opts)
    # the PROVIDER's token (one per provider) — whether it is set, never the value
    v["api_key_set"] = bool(provider_token(str(src.get("provider") or v.get("provider") or "")))
    if c is not None:
        # what a Start needs and why it would be refused now — the card disables Start
        # on the same list `Controller.start()` raises from
        try:
            v["start_blockers"] = c.start_blockers()
            st = c.state
            # the card shows the checklist only while startable — and its LAN item
            # reads the store, per host per tick while a host runs
            if c.op is None and (st.phase == "off" or (st.phase == "failed" and not st.uuid
                                                       and not st.index)):
                v["checklist"] = c.checklist()
        except Exception as e:              # the card's courtesy, never the card
            logger.warning(f"[host {name}] start check unavailable: {type(e).__name__}: {e}")
    v["managed"] = h is not None               # False: entry deleted, controller kept
    err = _host_errors.get(name, "")
    v["host_error"] = err
    if err and not v.get("error"):
        v["error"] = err
    v["not_attachable"] = [{"bid": bid, "reason": NOT_ATTACHABLE_REASON}
                           for bid in _host_not_attachable.get(name, [])]
    return v


def host_longrun() -> list:
    """[(host name, view)] of the controllers whose instance is up for more than 24 h —
    for the Dashboard's cost banner, polled every 4 s: `Controller.view()` alone (in
    memory), never `host_view`, whose alias-gate note reads the store per alias."""
    out = []
    for name in sorted(host_controllers, key=lambda n: str(n).lower()):
        c = host_controllers.get(name)
        if c is None:
            continue
        try:
            v = c.view()
        except Exception as e:              # a banner, never the Dashboard
            logger.warning(f"[host {name}] view failed: {type(e).__name__}: {e}")
            continue
        if isinstance(v, dict) and v.get("long_running"):
            out.append((name, v))
    return out


# action → (controller method, label, takes the service's backend id)
_HOST_OPS = {"start": ("start", "start", False), "stop": ("stop", "stop", False),
             "restart_service": ("restart_service", "restart", True),
             "resetup": ("resetup", "setup re-run", True),
             "sync": ("sync_now", "sync", False), "sync_now": ("sync_now", "sync", False)}
# How long a console "delete unknown files" waits for its answer before it says
# "still running" (the controller plans first — an ssh index of the whole disk).
_THUNDER_ANSWER_S = 30


async def _host_op(name: str, coro, label: str) -> str:
    """Run a controller op as a held background task — a start takes up to hours — and
    answer after its first step: a refusal (RuntimeError before its first await:
    "already …", "not running") comes back as the message, never as an exception."""
    answered: set = set()
    t = _host_spawn(name, label, coro, answered)
    await asyncio.sleep(0)                  # let the op run up to its first await
    if t.done() and not t.cancelled():
        # the done-callback runs after this (call_soon order): the refusal is the
        # answer here, so it is not logged a second time as a background failure
        answered.add(t)
        e = t.exception()
        if isinstance(e, RuntimeError):
            return f"{label} refused: {e}"
        if e is not None:
            return f"{label} failed: {type(e).__name__}: {e}"
        return f"{label} done"
    return f"{label} requested"


async def _host_delete_unknown(name: str, c, paths) -> str:
    """The panel's "delete unknown files": the controller re-plans and refuses the whole
    request when one path is no longer unknown (needed or synced meanwhile). The answer
    is awaited up to `_THUNDER_ANSWER_S` — the refusal is what the operator must see — and
    a delete still running then goes on as a held task whose failure is logged."""
    paths = [str(p) for p in paths or [] if p]
    if not paths:
        return "no file selected — nothing deleted"
    answered: set = set()
    t = _host_spawn(name, "delete unknown files", c.delete_unknown(paths), answered)
    answered.add(t)                         # the answer below reports it, not the log
    done, _ = await asyncio.wait({t}, timeout=_THUNDER_ANSWER_S)
    if t not in done:
        answered.discard(t)                 # finishing later: a failure goes to the log
        return f"deleting {len(paths)} file(s) — still running, see the log"
    if t.cancelled():
        return "delete cancelled"
    e = t.exception()
    if isinstance(e, (RuntimeError, ValueError)):
        return f"delete refused: {e}"
    if e is not None:
        return f"delete failed: {type(e).__name__}: {e}"
    n = t.result()
    return f"deleted {n} unknown file{'s' if n != 1 else ''}"


async def host_action(name: str, action: str, bid: Optional[str] = None,
                      paths: Optional[list] = None) -> str:
    """A console action on managed host `name`: start, stop, restart_service <bid>,
    resetup <bid>, forget_unreconciled, sync / sync_now, delete_unknown <paths>. Always
    answers with text — a refusal included — never with an exception."""
    c = host_controllers.get(name)
    if c is None:
        if name in managed_hosts:
            return (f"managed host {name!r} is not driven: "
                    f"{_host_errors.get(name) or 'no controller'}")
        return f"unknown managed host {name!r}"
    if action == "forget_unreconciled":
        c.forget_unreconciled()
        return "unreconciled instances forgotten"
    if action == "delete_unknown":
        return await _host_delete_unknown(name, c, paths)
    op = _HOST_OPS.get(action)
    if op is None:
        return f"unknown action {action!r}"
    meth, label, per_service = op
    if per_service:
        bid = str(bid or "").strip()
        if not bid:
            return f"{label} refused: no backend named"
        return await _host_op(name, getattr(c, meth)(bid), f"{label} of {bid}")
    return await _host_op(name, getattr(c, meth)(), label)


def _gated_only_aliases(aliases) -> set:
    """Aliases of a host's plan that no candidate can serve right now outside the
    model-sync gate — every candidate sits on a managed host's ComfyUI service that has
    not synced it (backend → host → controller, R-W7). Their schema, image slots and
    LoRA list read EMPTY until the sync finishes (the gate empties the candidate set),
    which the panel says instead of leaving it a mystery. Blocking (store read)."""
    out = set()
    for alias in aliases:
        cands = store.get(alias) if store.is_active() else None
        if cands is None:
            cands = image_models.get(alias, [])
        open_ = False
        for cand in cands or []:
            if not isinstance(cand, dict):
                continue
            bid = f"comfyui:{cand.get('backend')}"
            b = _live_backend(bid) if adapters.cand_kind(cand) == "comfyui" else None
            c = _host_ctl(b)
            if not (c is not None and not c.is_alias_ready(bid, alias)):
                open_ = True
                break
        if not open_:
            out.add(alias)
    return out


def modelsync_gate(backend: dict, alias: str) -> Optional[str]:
    """None = `alias` may route to `backend`; else why not (the text a client's 503
    carries). RunPod volume snapshots and managed ComfyUI services gate routing —
    backends are keyed (name, type), so a same-named LLM backend is not the box. Backend
    → `host` → controller → the service's plan (R-W7). Runs per request and per waiter ×
    backend inside a worker thread: it reads the controller's in-memory plan only
    (`is_alias_ready`/`alias_status` do no I/O), never anything slower."""
    if backend.get("type") == "runpod":
        v = backend.get("volume")
        if not v:
            return None
        c = volume_controllers.get(v)
        if c is None:
            return f"RunPod volume {v} is not set up yet"
        return None if c.is_alias_ready(alias) else c.alias_status(alias)
    if backend.get("type") != "comfyui":
        return None
    c = _host_ctl(backend)
    bid = backend_id(backend)
    if c is None or c.is_alias_ready(bid, alias):
        return None
    return c.alias_status(bid, alias)


def hosts_managed_info() -> dict:
    """/health (admin view): per managed host its provider, phase, uptime, price and
    each attached service's status — from the controllers' memory, no I/O."""
    out = {}
    for name in host_names():
        c = host_controllers.get(name)
        h = managed_hosts.get(name) or {}
        try:
            v = c.view() if c is not None else _undriven_view(name, h)
        except Exception as e:              # one host never takes /health down
            logger.warning(f"[host {name}] view failed: {type(e).__name__}: {e}")
            v = {}
        e = {"provider": str(v.get("provider") or h.get("provider") or ""),
             "phase": v.get("phase"), "uptime_s": v.get("uptime_s") or 0,
             "cost_per_h": v.get("cost_per_h"),
             "services": {bid: (s or {}).get("status")
                          for bid, s in (v.get("services") or {}).items()}}
        if _host_errors.get(name):
            e["error"] = _host_errors[name]
        out[name] = e
    return out


_HOST_NAME_RULE = ("host name: 1–40 characters of a-z, 0-9 and '-' (not at either end) — "
                   "it names the host's snapshots and state and cannot be changed later")


def _host_name_refusal(name: str, existing: Optional[dict] = None) -> Optional[str]:
    """Why `name` cannot be a NEW managed host (None = free): the name rule, then R-W6 —
    no managed host or retained controller, no key of the Hosts map, no backend's
    `backend_host()` (a URL hostname without a dot counts)."""
    if not _HOST_NAME_RE.fullmatch(name):
        return _HOST_NAME_RULE
    if existing is None:
        existing = _load_managed_hosts() if store.is_active() else dict(managed_hosts)
    if name in existing or name in host_controllers:
        return f"a managed host named {name!r} already exists"
    hosts_map = store.get_hosts() if store.is_active() else dict(hosts_meta)
    if name in hosts_map:
        return (f"{name!r} is already a host in the Hosts list — pick another name "
                "(a managed host's label and flags go there once it exists)")
    b = next((x for x in backends if backend_host(x) == name), None)
    if b is not None:
        return (f"backend {backend_id(b)} already runs on a host named {name!r} — "
                "pick another name")
    return None


def suggest_host_name(kind: str) -> str:
    """The new-host form's pre-filled name: the first free `<kind>-<n>` (n = 1, 2, …) by
    the same rules a Save applies — a suggestion the Save then refuses would be worse
    than none. "" when the kind is no plain word or nothing below 1000 is free."""
    kind = str(kind or "")
    if not _PROVIDER_KIND_RE.fullmatch(kind):
        return ""
    existing = _load_managed_hosts() if store.is_active() else dict(managed_hosts)
    for n in range(1, 1000):
        cand = f"{kind}-{n}"
        if _host_name_refusal(cand, existing) is None:
            return cand
    return ""


def managed_host_refusal(name: str, entry, new: bool = True) -> Optional[str]:
    """Why a managed-host Save must be refused (None = fine) — for the console's form.
    The name is `[a-z0-9-]` (it IS the identity: state, snapshots, socket — R-W5) and,
    for a NEW host, collides with nothing (R-W6): no managed host, no key of the Hosts
    map, no backend's `backend_host()` — a URL hostname without a dot counts. The
    provider must be known, never changed on an existing host, and its options must
    pass the provider's own `options_of`."""
    name = str(name or "")
    existing = _load_managed_hosts() if store.is_active() else dict(managed_hosts)
    why = _host_name_refusal(name, existing) if new else (
        None if _HOST_NAME_RE.fullmatch(name) else _HOST_NAME_RULE)
    if why:
        return why
    if not new and name not in existing:
        return f"unknown managed host {name!r}"
    e = entry if isinstance(entry, dict) else {}
    prov = hostapi.provider(e.get("provider"))
    if prov is None:
        return (f"unknown provider {e.get('provider')!r} (known: "
                + ", ".join(sorted(hostapi.PROVIDERS)) + ")")
    if not new:
        old = (existing.get(name) or {}).get("provider")
        c = host_controllers.get(name)
        if old != e.get("provider") or (c is not None and c.kind != prov[0].KIND):
            return ("the provider of a managed host cannot change (its state and "
                    "snapshots belong to it) — delete the host and create a new one")
    opts = e.get("options") if isinstance(e.get("options"), dict) else {}
    norm, errs, _ = prov[0].options_of({f"opt__{k}": v for k, v in opts.items()})
    if errs:
        return "; ".join(errs)
    # what the provider's spec list refuses (a vCPU count the GPU configuration does
    # not offer) — judged against the CACHED specs only; unknown specs cannot judge,
    # and the start re-checks before the create
    check = getattr(prov[0], "options_refusal", None)
    if check is not None:
        why = check(norm, provider_specs(prov[0].KIND))
        if why:
            return why
    return None


def provider_specs(kind: str):
    """The provider's spec list (`/v2/specs`) as any of its controllers last fetched it
    (the specs are the provider's, not a host's), None when none has — never a fetch:
    the Save and the host form read this, and a page view must not wait on the network."""
    for c in list(host_controllers.values()):
        if getattr(c, "kind", None) != kind:
            continue
        try:
            specs = c.specs_cached()
        except Exception as e:              # a cache read, never the Save
            logger.warning(f"[host {c.name}] specs cache unreadable: {type(e).__name__}")
            continue
        if specs is not None:
            return specs
    return None


def save_managed_host(name: str, entry: dict, new: bool) -> str:
    """Store one managed host (provider + normalized options) and apply it → the
    refusal, "" = saved. Tokens are the provider's (`save_provider_token`): an
    `api_key` in `entry` is never stored."""
    why = managed_host_refusal(name, entry, new=new)
    if why:
        return why
    if not store.is_active():
        return "the store is not active — not saved"
    # the store holds validated values only: the provider's own normalization ("auto"
    # stored as "", every key present, ints as ints) — never what the caller typed
    prov = hostapi.provider(entry.get("provider"))
    opts = entry.get("options") if isinstance(entry.get("options"), dict) else {}
    norm = prov[0].options_of({f"opt__{k}": v for k, v in opts.items()})[0]
    store.set_managed_host(name, {"provider": entry.get("provider"), "options": norm})
    apply_managed_hosts()
    return ""


# ── RunPod volumes (store entries and restart-safe controller state) ─────────────

runpod_volumes: dict = {}
volume_controllers: dict = {}
_vol_tasks: dict = {}
_vol_warned: set = set()
runpod_secret_kinds = ("runpod", "runpod_s3")
_SECRET_KINDS = set(runpod_secret_kinds)


def _load_runpod_volumes() -> dict:
    """A failed read must not look like every volume was removed."""
    if not store.is_active():
        return {}
    try:
        entries = store.get_setting("runpod_volumes")
        if entries is None:
            return {}
        if not isinstance(entries, dict):
            raise ValueError("runpod_volumes is not a dict")
        return entries
    except Exception as e:
        logger.warning(f"runpod_volumes unreadable — keeping the last read: {type(e).__name__}")
        return dict(runpod_volumes)


def _volume_load_state(name: str) -> Optional[dict]:
    d = store.get_setting("runpod_volume_state")
    if d is None:
        return None
    if not isinstance(d, dict):
        raise ValueError("runpod_volume_state is not a dict")
    return d.get(name)


def _volume_save_state(name: str, d: dict) -> None:
    """Read-modify-write without an await, on the event loop only. Moving this to
    worker threads loses updates when two volumes read the same shared setting."""
    cur = store.get_setting("runpod_volume_state")
    if cur is None:
        cur = {}
    if not isinstance(cur, dict):
        raise ValueError("runpod_volume_state is unreadable — not saved")
    cur[name] = d
    store.set_settings({"runpod_volume_state": cur})


def _volume_deps(name: str) -> "rpvolume.VolumeDeps":
    lan = modelsrc()

    def live(bid):
        b = next((b for b in backends if backend_id(b) == bid and b.get("type") == "runpod"), None)
        ad = backend_adapters.get(bid)
        if b is None or not isinstance(ad, adapters.RunpodAdapter):
            raise RuntimeError(f"RunPod backend {bid} is not available")
        return b, ad

    async def run_fetch(bid, payload, on_id, *, max_wait=None):
        b, ad = live(bid)
        budget = float(b.get("max_wait", 600)) if max_wait is None else max_wait
        return await ad.run_op(payload, budget, on_id)

    async def fetch_status(bid, rp_id):
        return await live(bid)[1].job_status(rp_id)

    async def fetch_cancel(bid, rp_id):
        return await live(bid)[1].cancel_runpod_id(rp_id)

    def creds():
        access, _, secret = provider_token("runpod_s3").partition(":")
        return provider_token("runpod"), access, secret

    return rpvolume.VolumeDeps(
        client_factory=lambda: httpx.AsyncClient(),
        load_state=_volume_load_state, save_state=_volume_save_state, creds=creds,
        backends=lambda name: sorted([b for b in backends if b.get("type") == "runpod"
            and b.get("enabled", True) and b.get("volume") == name], key=lambda b: b["name"]),
        alias_needs=service_alias_needs, alias_signature=service_alias_signature,
        source_index=lan.cached, lan=lan,
        url_catalog=lambda src: modelsync.url_catalog(_modelsync_catalog(), src, _share_sha_files()),
        run_fetch=run_fetch, fetch_status=fetch_status, fetch_cancel=fetch_cancel,
        note_fault=lambda kind, detail: _note_fault(
            {"name": f"volume:{name}", "type": "runpod"}, "volume", kind, detail),
        log=logger.info, now=time.time, hf_token=_thunder_hf_token)


def _vol_spawn(name: str, what: str, coro) -> asyncio.Task:
    t = _bg(coro)
    def done(t):
        if not t.cancelled() and t.exception() is not None:
            # Provider exceptions can contain keys or signed URLs.
            logger.warning(f"[volume {name}] {what}: {type(t.exception()).__name__}")
    t.add_done_callback(done)
    _vol_tasks[name] = [x for x in _vol_tasks.get(name, []) if not x.done()] + [t]
    return t


def _vol_run(name: str, c) -> None:
    async def run():
        try:
            await c.resume()
        except Exception as e:
            # The loop retries resume with its own backoff; boot must not strand it.
            logger.warning(f"[volume {name}] resume: {type(e).__name__}")
        await c.run_forever()
    _vol_spawn(name, "background loop", run())


def sync_volume_controllers() -> None:
    global runpod_volumes
    runpod_volumes = _load_runpod_volumes()
    for name, entry in runpod_volumes.items():
        try:
            c = volume_controllers.get(name)
            if c is None:
                c = rpvolume.VolumeController(name, entry, _volume_deps(name))
                volume_controllers[name] = c
                if _hosts_booted:
                    _vol_run(name, c)
            else:
                c.cfg = entry
            _vol_warned.discard(name)
        except Exception as e:
            logger.warning(f"[volume {name}] not driven: {type(e).__name__}")
    for name in list(volume_controllers):
        if name in runpod_volumes:
            continue
        c = volume_controllers[name]
        if c.state.get("fetch_job") or c.state.get("mpu"):
            if name not in _vol_warned:
                logger.warning(f"[volume {name}] entry removed with writers pending — controller kept")
                _vol_warned.add(name)
            continue
        volume_controllers.pop(name)
        for t in _vol_tasks.pop(name, []):
            t.cancel()
        if _hosts_booted:
            _vol_spawn(name, "retire", c.aclose())


def volume_names() -> list:
    return sorted(set(runpod_volumes) | set(volume_controllers))


def volume_view(name: str) -> Optional[dict]:
    c = volume_controllers.get(name)
    if c is not None:
        v = c.view()
        # every referencing backend, disabled ones too (delete_volume refuses on them):
        # the controller's own list exists only after a plan round with working keys
        v["backends"] = sorted(b["name"] for b in backends
                               if b.get("type") == "runpod" and b.get("volume") == name)
        return v
    entry = runpod_volumes.get(name)
    return {"name": name, **entry, "phase": "off"} if entry is not None else None


def volume_field_refusal(value) -> str:
    """Refuse a dangling backend reference before the console stores it."""
    if value == "" or (isinstance(value, str) and value in runpod_volumes):
        return ""
    return "unknown RunPod volume"


def save_volume(name: str, entry: dict, new: bool) -> str:
    if not isinstance(name, str) or not rpvolume.NAME_RE.fullmatch(name):
        return "volume name must match [a-z0-9-]{1,40}"
    entries = _load_runpod_volumes()
    if new and (name in entries or name in volume_controllers):
        return "volume name is already taken"
    if not new and name not in entries:
        return "unknown RunPod volume"
    dc = entry.get("datacenter")
    if dc not in rpvolume.DCS:
        return "unknown datacenter"
    c = volume_controllers.get(name)
    state = c.state if c else (_volume_load_state(name) or {})
    if state.get("id") and dc != state.get("dc"):
        return "a created volume cannot change datacenter"
    size, maximum = entry.get("size_gb"), entry.get("max_size_gb")
    if state.get("id") and name in entries:
        # the start size only applies to the create; afterwards the sync grows the
        # volume and the console's form shows the size read-only — compared with the
        # grown size, every later edit (a new ceiling) would be refused as a shrink
        size = entries[name].get("size_gb", size)
    if (type(size) is not int or type(maximum) is not int
            or not 10 <= size <= maximum <= 4000):
        return "sizes must satisfy 10 <= size_gb <= max_size_gb <= 4000"
    if state.get("size_gb") and maximum < state["size_gb"]:
        return (f"max size is below the volume's current {state['size_gb']} GB "
                "(a volume only grows)")
    if not store.is_active():
        return "the store is not active — not saved"
    entries[name] = {"datacenter": dc, "size_gb": size, "max_size_gb": maximum}
    store.set_settings({"runpod_volumes": entries})
    sync_volume_controllers()
    return ""


async def volume_sync_now(name: str, recreate: bool = False) -> str:
    c = volume_controllers.get(name)
    if c is None:
        return "unknown RunPod volume"
    c.sync_now(recreate=recreate)
    return ""


async def volume_delete_unknown(name: str, paths) -> str:
    c = volume_controllers.get(name)
    if c is None:
        return "unknown RunPod volume"
    try:
        await c.delete_unknown(paths)
        return ""
    except Exception as e:
        return f"volume delete failed ({type(e).__name__})"


async def delete_volume(name: str) -> str:
    c = volume_controllers.get(name)
    if c is None:
        return "unknown RunPod volume"
    try:
        states = store.get_setting("runpod_volume_state")
        if states is None:
            states = {}
        if not isinstance(states, dict):
            return "runpod_volume_state is unreadable — not deleted"
    except Exception:
        return "runpod_volume_state is unreadable — not deleted"
    # Disabled references also keep the volume: enabling them later must be safe.
    if any(b.get("type") == "runpod" and b.get("volume") == name for b in backends):
        return "Volume is referenced by a RunPod backend"
    if c.state.get("fetch_job") or c.state.get("mpu"):
        return "Volume has transfers pending"
    if not c.state.get("id") or c.state.get("missing", 0) >= rpvolume.GONE_AFTER:
        # A never-created or confirmed-gone volume has no remaining bill to cancel.
        # Remove its local identity too, otherwise reusing the name keeps the gone id.
        entries = _load_runpod_volumes()
        entries.pop(name, None)
        states.pop(name, None)
        store.set_settings({"runpod_volumes": entries, "runpod_volume_state": states})
        sync_volume_controllers()
        return ""
    try:
        why = await c.delete_volume()
    except Exception as e:
        return f"volume delete failed ({type(e).__name__})"
    if why:
        return why
    entries = _load_runpod_volumes()
    entries.pop(name, None)
    store.set_settings({"runpod_volumes": entries})
    sync_volume_controllers()
    return ""


# ── provider API tokens (one per provider kind, `store.set_provider_token`) ─────────

_PROVIDER_KIND_RE = re.compile(r"[a-z0-9_-]{1,40}")
_PROVIDER_TOKEN_MAX = 1024


def provider_token(kind: str) -> str:
    """The provider's API token ("" = none, unknown, or the store cannot answer)."""
    kind = str(kind or "")
    if not store.is_active() or not _PROVIDER_KIND_RE.fullmatch(kind):
        return ""
    try:
        return store.get_provider_token(kind)
    except Exception as e:
        logger.warning(f"provider token of {kind} unreadable: {type(e).__name__}")
        return ""


def provider_tokens_info() -> dict:
    """Provider and volume credential presence for the console; never their values."""
    return {k: bool(provider_token(k)) for k in tuple(hostapi.PROVIDERS) + runpod_secret_kinds}


def save_provider_token(kind: str, token: str) -> str:
    """The console's provider-token Save: "" removes it, anything else replaces it
    (encrypted at rest). Every controller of that provider gets it at once (a sync, no
    restart). → the refusal, "" = saved; a refusal never repeats the value."""
    kind, token = str(kind or ""), str(token or "")
    if hostapi.provider(kind) is None and kind not in _SECRET_KINDS:
        return (f"unknown provider {kind!r} (known: "
                + ", ".join(sorted(hostapi.PROVIDERS)) + ")")
    if kind == "runpod_s3" and token and (token.count(":") != 1
            or not all(token.split(":")) or any(ch.isspace() for ch in token)):
        return "expected <access key id>:<secret>"
    # it goes into an Authorization header: a space, a line break or a control
    # character would make every provider call fail — refused here, out loud
    if token and (len(token) > _PROVIDER_TOKEN_MAX
                  or any(not ch.isprintable() or ch.isspace() for ch in token)):
        return ("the API token may hold only printable characters without spaces "
                f"(at most {_PROVIDER_TOKEN_MAX}) — not saved")
    if not token:
        # clearing reaches EVERY host of this provider at once: a running one's next
        # provider call — its stop's snapshot and delete — would be a 401, leaving it
        # `failed` with the instance kept and billing. A new token (rotation) is fine.
        busy = sorted(n for n, c in host_controllers.items()
                      if getattr(c, "kind", None) == kind and not _host_idle_off(c))
        if busy:
            return (f"{_provider_display(kind)} hosts are not off ({', '.join(busy)}) — "
                    "stop them first, or enter a new token instead of clearing it")
    if not store.is_active():
        return "the store is not active — not saved"
    store.set_provider_token(kind, token)
    apply_managed_hosts()
    return ""


def _provider_display(kind: str) -> str:
    p = hostapi.provider(kind)
    return str(getattr(p[0], "NAME", kind)) if p else str(kind)


def _record_names_instance(name: str) -> bool:
    """Does the stored state record of host `name` name an instance (a uuid/index, or a
    phase other than off)? An unreadable state setting answers False (no preference)."""
    try:
        rec = _host_load_state(name)
    except Exception:
        return False
    return isinstance(rec, dict) and bool(rec.get("uuid") or rec.get("index")
                                          or (rec.get("phase") or "off") != "off")


def migrate_provider_tokens() -> None:
    """Startup, idempotent: before tokens were per provider, every managed host carried
    its own `api_key`. A provider without a token takes a READABLE one of its hosts —
    that of a host whose state record names an instance first (that instance must stay
    stoppable), else by host name — then every entry loses its copy. A provider token
    that is already set is never overwritten. A dropped per-host token that DIFFERS from
    the provider token is warned about by host name (a second account's host would
    otherwise first say so as a 401 at its stop). Never logs a token."""
    if not store.is_active():
        return
    hosts = store.get_managed_hosts()
    legacy = sorted(n for n, e in hosts.items() if "api_key" in e)
    if not legacy:
        return
    live = {n for n in legacy if _record_names_instance(n)}
    for name in sorted(legacy, key=lambda n: (n not in live, n)):
        e = hosts[name]
        kind, tok = str(e.get("provider") or ""), str(e.get("api_key") or "")
        if not tok or not _PROVIDER_KIND_RE.fullmatch(kind) or provider_token(kind):
            continue
        store.set_provider_token(kind, tok)
        logger.info(f"managed hosts: the API token of host {name} is now the {kind} "
                    "provider token (one per provider)")
    for name in legacy:
        e = hosts[name]
        kind, tok = str(e.get("provider") or ""), str(e.get("api_key") or "")
        if tok and tok != provider_token(kind):
            logger.warning(f"managed hosts: host {name}'s API token differs from the "
                           f"{kind} provider token and is dropped — if {name} runs on "
                           "another account, enter that token before starting or "
                           "stopping it")
        store.set_managed_host(name, e)                 # drops `api_key`
    logger.info(f"managed hosts: per-host API tokens removed from {len(legacy)} "
                "entr" + ("y" if len(legacy) == 1 else "ies"))


def managed_host_delete_refusal(name: str) -> Optional[str]:
    """R-W5: a host is deleted only while off, idle and without a snapshot being taken
    — its instance and snapshots would otherwise bill with nobody to stop them — and
    once no backend names it any more."""
    existing = _load_managed_hosts() if store.is_active() else dict(managed_hosts)
    if name not in existing:
        return f"unknown managed host {name!r}"
    # attached backends would be left pointing at a forward nobody opens any more, and
    # would keep the name taken (R-W6) — they move first, deliberately
    on = sorted(backend_id(b) for b in backends
                if str(b.get("host") or "").strip() == name)
    if on:
        return (f"backends still name {name} as their host ({', '.join(on)}) — move or "
                "delete them first")
    c = host_controllers.get(name)
    if c is None:
        # no controller (an unknown provider: never driven) — the stored record is the
        # ONLY pointer to an instance or a snapshot being taken; with no controller of
        # that provider left, nothing could even list them as orphans afterwards
        try:
            rec = _host_load_state(name)
        except Exception as e:
            return (f"the state record of {name} is unreadable ({type(e).__name__}) — "
                    "an instance may still run; not deleted")
        if rec is None:
            return None
        if not isinstance(rec, dict):
            return f"the state record of {name} is unreadable — not deleted"
        if (rec.get("phase") != "off" or rec.get("uuid") or rec.get("index")
                or rec.get("pending_snapshot")):
            what = (f"snapshot {rec.get('pending_snapshot')} is still being taken"
                    if rec.get("phase") == "off" and not (rec.get("uuid") or rec.get("index"))
                    else f"its record names an instance ({rec.get('phase')}, "
                         f"{rec.get('uuid') or rec.get('index')})")
            return (f"{name} is not driven and {what} — make the host drivable and stop "
                    "it first; not deleted")
        return None
    if c.op is not None:
        return f"{name} is busy ({c.op}) — delete it once it is off"
    if c.state.phase != "off":
        return f"{name} is {c.state.phase} — stop it first; only an off host can be deleted"
    if c.state.pending_snapshot:
        return (f"snapshot {c.state.pending_snapshot} of {name} is still being taken — "
                "delete the host once it is done")
    return None


def delete_managed_host(name: str) -> str:
    """Delete a managed host → the refusal, "" = deleted. Its state record and its Hosts
    map entry go too: a later host of the same name must not adopt this one's snapshot
    — they show as "foreign" from now on (R-W5). Its controller is retired by the sync
    (off and idle)."""
    why = managed_host_delete_refusal(name)
    if why:
        return why
    store.set_managed_host(name, None)
    cur = store.get_setting(_HOST_STATE_KEY)
    if isinstance(cur, dict) and name in cur:
        cur.pop(name)
        store.set_settings({_HOST_STATE_KEY: cur})
    # its label and coordination flags in the Hosts map were the managed host's too
    # (same key) — left behind they would refuse a new host of this name (R-W6)
    store.set_host(name, None)
    apply_hosts()
    apply_managed_hosts()
    return ""


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
    global park_timeout_s, async_park_timeout_s, park_health_grace_s, max_parked, max_queued_gen
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
    if "max_queued_gen" in s:
        try:
            max_queued_gen = int(s["max_queued_gen"])
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
                       ("stats_db_path", "db_path"), ("stats_retention_days", "retention_days"),
                       ("stats_body_retention_days", "body_retention_days")):
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
            "max_queued_gen": max_queued_gen,
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
            "stats_body_retention_days": stats_cfg.get("body_retention_days",
                                                       stats.BODY_RETENTION_DAYS_DEFAULT),
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
             "models": sorted(backend_models.get(backend_id(b), set())),
             # `/running` answers → the alias editor also offers `current` here
             "current": _is_current(b, adapters.CURRENT_MODEL)}
            for b in backends if not _is_gen(b)]


# Wire the UI to the generation core, the ComfyUI backends, the status snapshot,
# and the backend-change hook.
admin.bind(runpod_probe=runpod_probe, runpod_object_info=runpod_object_info,
           comfy_backends=lambda: [b for b in backends if b.get("type") == "comfyui"],
           gen_backends=lambda: [b for b in backends if _is_gen(b)],
           gateway_info=gateway_info,
           gen_speed_info=gen_speed_info,
           faults_info=faults_info,
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
           admin_session_tag=admin_session_tag,
           admin_credential_exists=admin_credential_exists,
           admin_change_refusal=admin_change_refusal,
           backend_api_key=lambda name, typ: next(
               (b.get("api_key") for b in backends
                if b["name"] == name and b.get("type", "openai") == typ), None),
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
           parse_voice_target=parse_voice_target, voice_dir_ok=_voice_dir_ok,
           apply_hosts=apply_hosts,
           # managed hosts: the Backends tab's host cards, form and actions
           volume_names=volume_names, volume_view=volume_view, save_volume=save_volume,
           delete_volume=delete_volume, volume_sync_now=volume_sync_now,
           volume_delete_unknown=volume_delete_unknown, volume_field_refusal=volume_field_refusal,
           host_names=host_names, host_view=host_view, host_action=host_action,
           host_longrun=host_longrun, save_managed_host=save_managed_host,
           assign_local_port=assign_local_port,
           delete_managed_host=delete_managed_host,
           managed_host_delete_refusal=managed_host_delete_refusal,
           suggest_host_name=suggest_host_name, provider_specs=provider_specs,
           provider_tokens=provider_tokens_info, save_provider_token=save_provider_token,
           thunder_default_nodes=_thunder_default_nodes,
           modelsrc_view=modelsrc_view, modelsrc_scan=modelsrc_scan,
           modelsrc_pin=modelsrc_pin, modelsrc_list=modelsrc_list,
           save_modelsrc_host=save_modelsrc_host,
           save_hf_token=save_hf_token, hf_token_set=hf_token_set,
           thunder_orphan_snapshots=thunder_orphan_snapshots,
           # views read the catalog WITHOUT seeding it (a GET never writes the store)
           modelsync_catalog=_modelsync_catalog_view,
           save_modelsync_catalog=save_modelsync_catalog,
           # model sources: the editor's stale-form guard, the overview, the actions
           modelsync_catalog_hash=modelsync_catalog_hash, catalog_stale=CATALOG_STALE,
           source_checks_pending=source_checks_pending,
           model_sources=model_sources_overview, model_source_kinds=model_source_kinds,
           check_source=check_source, check_dir_source=check_dir_source,
           remove_source=remove_source,
           backend_loras=lambda: {b["name"]: sorted(backend_loras.get(backend_id(b), set()))
                                  for b in backends if b.get("type") in ("comfyui", "runpod")},
           lora_meta_view=lora_meta_view, lora_curate=lora_curate,
           lora_refresh=lora_refresh, lora_refresh_all=lora_refresh_all)


def _health_full_allowed(request: Request, authorization: Optional[str],
                         x_api_key: Optional[str]) -> bool:
    """The full /health snapshot is for admins: bootstrap-open (everything is open
    anyway), an admin credential as Bearer or x-api-key, or a valid /ui session."""
    if not users and not api_key:
        return True
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
    if resolve_admin(token) or resolve_admin(x_api_key):
        return True
    return bool(admin._session_user(request))


@app.get("/health")
async def health_endpoint(request: Request, authorization: Optional[str] = Header(None),
                          x_api_key: Optional[str] = Header(None)):
    """Liveness for everyone, the inventory for admins (see _health_full_allowed).
    Model ids only with `?verbose=1` — every backend listing every model made this the
    largest unasked-for response the gateway sends."""
    if not _health_full_allowed(request, authorization, x_api_key):
        en = [b for b in backends if is_enabled(b)]
        return {"status": "ok", "backends_total": len(en),
                "backends_healthy": sum(1 for b in en if backend_healthy.get(backend_id(b), False))}
    verbose = request.query_params.get("verbose", "") not in ("", "0", "false", "no")
    return await health(verbose=verbose)


async def health(verbose: bool = True) -> dict:
    """The full health snapshot (admin view of /health; tests read it directly)."""
    fmap = {s["bid"]: s for s in (await asyncio.to_thread(faults_info))["backends"]}
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
                "models_count": len(backend_models.get(backend_id(b), set())) if is_enabled(b) else 0,
                **({"models": sorted(backend_models.get(backend_id(b), set())) if is_enabled(b) else []}
                   if verbose else {}),
                # What the fault log holds for the last 24h (faults.py) — the current
                # `error` above is gone the moment the next poll succeeds.
                "faults_24h": {k: (fmap.get(backend_id(b)) or {}).get(k, 0)
                               for k in ("faults", "outages", "downtime_s")},
                **_comfy_watch_info(b), **_cloud_info(b), **_runpod_info(b), **_model_filter_info(b), **_loaded_info(b),
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
        # Managed hosts (hostctl.py): lifecycle and each attached service's status.
        "hosts_managed": hosts_managed_info(),
        **({"runpod_volumes": {name: {k: c.view().get(k) for k in
            ("id", "size_gb", "phase")} | {"ready_aliases": sorted(c.ready_aliases)}
            for name, c in volume_controllers.items()}} if verbose else {}),
    }
