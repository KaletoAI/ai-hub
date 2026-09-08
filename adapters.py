"""Backend adapters — the pluggable per-backend protocol layer.

Phase 0 of the multimodal-gateway plan (see docs/archive/multimodal-gateway-plan.md):
introduce the seam *without any behaviour change*. `OpenAIAdapter` is a 1:1 move
of the former module-level `proxy()` (dispatch) and the `/v1/models` discovery
helpers (`extract_models` / `extract_pricing` / …). New protocols (ComfyUI, …)
implement `BackendAdapter` and register in `ADAPTERS`.

Design notes:
- Adapters never import `main` (avoids an import cycle). Everything they need
  from the app — the in-flight counter, stats sink, pricing, log flag — is
  injected via `AdapterContext`, with hot-reloadable values read through
  callables so nothing is cached at construction time.
- The discovery helpers below are pure functions of (payload, backend dict);
  they live here because they are OpenAI-`/v1/models`-payload-specific.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import fnmatch
import json
import logging
import os
import random
import re
import struct
import time
import uuid
import zlib
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional
from urllib.parse import quote

import httpx

import scheduler
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import anthropic_bridge
import cloudtask
import meshy
import tripo

logger = logging.getLogger(__name__)

# Outbound timeouts (seconds): discovery is short (health tick must stay snappy),
# chat generous (long completions), ComfyUI uploads sized for LAN image posts.
_CHAT_TIMEOUT = 300.0
_DISCOVERY_TIMEOUT = 5.0
_COMFY_DISCOVERY_TIMEOUT = 8.0
_UPLOAD_TIMEOUT = 20.0             # floor; the real budget scales with the file (_upload_timeout_for)
_COMFY_STUCK_AFTER_S = 90.0        # default: pending-with-idle-executor this long → stuck
_COMFY_RESTART_WAIT_S = 120.0      # restart(): max wait for the server to come back


def _upload_timeout_for(nbytes: int) -> float:
    """Seconds to allow one ComfyUI `/upload/image` POST of `nbytes`.

    A flat 20 s was sized for LAN image posts, but the chain relays MESHES through the
    same endpoint: measured 2026-09-02 on prod, a no-remesh Meshy humanoid came back at
    70 MB — on a 20 s budget that upload is a coin flip, and a lost mesh fails the whole
    (already paid for) chain. Budget a conservative 1 MiB/s on top of the flat floor."""
    return max(_UPLOAD_TIMEOUT, 20.0 + max(0, nbytes) / (1024 * 1024))


class ComfyExecutorStuck(Exception):
    """ComfyUI answers HTTP but its prompt executor is not draining the queue
    (queue_pending non-empty, queue_running empty, same head across checks)."""


class CloudNoCredits(ConnectionError):
    """A cloud task account has no credits (balance 0 on discovery; Meshy 402 / Tripo
    403+2010 on submit). A ConnectionError on purpose: _GEN_FAILOVER_ERRORS moves the job
    to the next candidate without touching that tuple; _fault_label names it apart and
    `vendor` says whose account."""

    def __init__(self, msg: str = "", vendor: str = "cloud"):
        super().__init__(msg)
        self.vendor = vendor


class CloudBusy(ConnectionError):
    """The cloud refused the task with 429 (Meshy NoMoreConcurrentTasks /
    RateLimitExceeded): the account's concurrency limit is full — other API keys of the
    same account fill it too, so the gateway's own max_concurrent cannot rule it out.
    Failover-class."""

    def __init__(self, msg: str = "", vendor: str = "cloud"):
        super().__init__(msg)
        self.vendor = vendor


class CloudTaskRetryable(ConnectionError):
    """The vendor failed the task on ITS side and says so in a machine-readable field
    (Meshy `task_error.type` = server_error / timeout / service_unavailable — the docs
    answer each with "retry the request"). Failover-class for the same reason CloudBusy
    is: the fault is the service's, so another attempt is worth making, and the gateway
    already owns that machinery (`self_retries`, then the next candidate). The narrow
    part is what does NOT come here — a permanent `invalid_input` stays a final
    RuntimeError, because a rejected input is rejected on every candidate alike, and a
    task the vendor already BILLED is never re-run whatever it says (CloudTaskAdapter.
    _poll checks the credits): retrying that would buy the same mesh twice."""

    def __init__(self, msg: str = "", vendor: str = "cloud"):
        super().__init__(msg)
        self.vendor = vendor


MeshyNoCredits, MeshyBusy = CloudNoCredits, CloudBusy   # pre-Tripo names (main._fault_label, tests)


# ── OpenAI /v1/models discovery helpers (moved verbatim from main.py) ──────────

def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def normalize_pricing(m: dict) -> Optional[dict[str, float]]:
    """Per-model pricing normalized to USD per **million** tokens, or None.

    Handles the two upstream schemas we see in the wild:
      - Together:   pricing.input / pricing.output  — numbers, already per-million
      - OpenRouter: pricing.prompt / pricing.completion — strings, per *single*
                    token, so multiplied by 1e6 to match the per-million convention
    Local backends (llama-swap / vLLM) carry no pricing → None.
    """
    p = m.get("pricing")
    if not isinstance(p, dict):
        return None
    if "input" in p or "output" in p:            # Together-style (per-million)
        return {"input": _to_float(p.get("input")), "output": _to_float(p.get("output"))}
    if "prompt" in p or "completion" in p:        # OpenRouter-style (per-token)
        return {"input": _to_float(p.get("prompt")) * 1_000_000,
                "output": _to_float(p.get("completion")) * 1_000_000}
    return None


def _is_priced(m: dict) -> bool:
    pr = normalize_pricing(m)
    return bool(pr and (pr["input"] or pr["output"]))


def extract_pricing(payload) -> dict[str, dict[str, float]]:
    """Per-model pricing from a /v1/models response, keyed by model id. Values
    are USD per million tokens (see normalize_pricing). Together + OpenRouter
    expose pricing; OpenAI/llama-swap don't → empty dict."""
    data = payload["data"] if isinstance(payload, dict) else payload
    out: dict[str, dict[str, float]] = {}
    for m in data or []:
        if "id" not in m:
            continue
        if _is_priced(m):
            out[m["id"]] = normalize_pricing(m)  # type: ignore[assignment]
    return out


def extract_context(payload) -> dict[str, int]:
    """Per-model context window (tokens) from a /v1/models response, keyed by id —
    the number a client sizes its prompts by. ONE reader for every listing shape:
    OpenRouter / Together `context_length`, vLLM `max_model_len`, llama-server
    `meta.n_ctx` (the SERVING size — `n_ctx_train` is what the weights could do,
    not what this server accepts). Models without a usable number are absent, never
    0: a client treats 0 as "unknown" at best and as "no room" at worst."""
    data = payload["data"] if isinstance(payload, dict) else payload
    out: dict[str, int] = {}
    for m in data or []:
        if not isinstance(m, dict) or "id" not in m:
            continue
        meta = m.get("meta") if isinstance(m.get("meta"), dict) else {}
        for v in (m.get("context_length"), m.get("max_model_len"), meta.get("n_ctx")):
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                out[m["id"]] = int(v)
                break
    return out


def parse_model_context(text) -> list[tuple[str, int]]:
    """The backend's `model_context` rules: one `glob=tokens` per line (blank lines and
    `#` comments skipped). A line that does not parse is DROPPED, not fatal — a typo
    must not take the whole backend's rules with it. Order is kept: the first
    matching rule wins in model_context_for()."""
    rules: list[tuple[str, int]] = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        pat, _, num = line.partition("=")
        pat, num = pat.strip(), num.strip()
        if not pat or not num.isdigit() or int(num) <= 0:
            continue
        rules.append((pat, int(num)))
    return rules


def model_context_for(backend: dict, model: str, learned: dict) -> Optional[int]:
    """Context window for `model` on `backend`: the admin's `model_context` rule
    (first glob match) beats the discovery-learned value — the rule exists for the
    llama-swap model that is not loaded right now and never reported its n_ctx, and
    for the operator who knows better than a listing. None = unknown; the catalog
    then omits `context_length` rather than invent one."""
    for pat, n in parse_model_context(backend.get("model_context")):
        if fnmatch.fnmatchcase(model, pat):
            return n
    v = (learned or {}).get(model)
    return int(v) if isinstance(v, (int, float)) and v > 0 else None


def _is_chat_model(m: dict) -> bool:
    """True unless the model is clearly not chat-completions routable.

    - Together tags models with `type`; keep only type=="chat".
    - OpenRouter has no `type` but exposes architecture.output_modalities;
      drop models that can't emit text (image-/audio-only output).
    - Backends without either field (llama-swap, vLLM) always pass.
    """
    if m.get("type", "chat") != "chat":
        return False
    om = (m.get("architecture") or {}).get("output_modalities")
    if om is not None and "text" not in om:
        return False
    return True


def extract_models(payload, backend: dict) -> set[str]:
    """Return model IDs from a /v1/models response, applying per-backend filters.

    Accepts both `{"data": [...]}` (OpenAI / llama-swap) and a bare `[...]`
    (e.g. together.ai). Optional per-backend filters:
      - chat_only:       keep only models with type=="chat" (skip image/video/embedding/…)
      - serverless_only: keep only models with non-zero token pricing
                         (excludes Together's dedicated-only endpoints)
    Filters are skipped silently for backends that don't expose the relevant
    metadata field — llama-swap etc. stay unaffected.
    """
    data = payload["data"] if isinstance(payload, dict) else payload
    if backend.get("chat_only"):
        data = [m for m in data if _is_chat_model(m)]
    if backend.get("serverless_only"):
        data = [m for m in data if _is_priced(m)]
    return {m["id"] for m in data if "id" in m}


# ── Adapter contracts ─────────────────────────────────────────────────────────

@dataclass
class Capabilities:
    """What a backend exposes, as seen by a discovery poll. Phase 0 carries the
    OpenAI shape (models + pricing); later phases add tasks/workflows."""
    models: set[str]
    pricing: dict[str, dict[str, float]]
    loras: set = field(default_factory=set)         # ComfyUI: installed LoRA filenames
    context: dict = field(default_factory=dict)     # model → context window (tokens), where the listing says


@dataclass
class NormalizedRequest:
    """Container threaded from input adapter → router → backend adapter.

    Chat/LLM path (Phase 0) uses path/alias/real_model/body/raw/stream and the
    OpenAI adapter forwards `body` verbatim. The generation path (Phase 1+) adds
    task/inputs/params/output/workflow, populated by the native input adapter and
    rendered into a backend protocol by e.g. ComfyUIAdapter.
    """
    path: str = ""
    alias: str = ""
    real_model: Optional[str] = None
    body: dict = field(default_factory=dict)
    raw: Optional[Request] = None
    stream: bool = False
    stats_endpoint: Optional[str] = None            # stats label when it differs from path (e.g. /v1/responses)
    # CONVENTION: body keys starting with "_" are gateway-private control fields
    # (e.g. route() stashes body["_reasoning"]); _prepare() strips them all before
    # forwarding, so they never reach a backend.
    reasoning: Optional[str] = None                 # normalized reasoning control: "off" | "on" | None(auto)
    # ── generation extension ──
    task: str = "chat"                              # text2img | img2video | tts | …
    inputs: dict = field(default_factory=dict)      # prompt, negative_prompt, …
    params: dict = field(default_factory=dict)      # width, height, steps, cfg, seed, …
    output: dict = field(default_factory=dict)      # n, format, mode, ttl_s
    workflow: Optional[str] = None                  # workflow file path (share/legacy)
    workflow_json: Optional[dict] = None            # gateway-owned API workflow (preferred)
    node_mapping: dict = field(default_factory=dict)  # {param: {node, field}} request-time
    fixed: list = field(default_factory=list)       # [{node, field, value}] admin-pinned (models, …)
    upload_image: Optional[bytes] = None            # legacy single reference image (back-compat)
    upload_images: dict = field(default_factory=dict)  # {param: bytes} per request-field image uploads
    upload_files: dict = field(default_factory=dict)  # {param: (suggested filename, bytes)} client-supplied
                                                    # non-image files (meshes) — uploaded PER dispatch, so
                                                    # parking/failover re-uploads onto the running backend.
                                                    # The filename is only a hint: the extension is kept, the
                                                    # NAME is rebuilt from `upload_prefix` (see below).
    upload_prefix: str = ""                         # job-unique namespace for EVERY input file this request
                                                    # uploads (`<prefix>_<param>.<ext>`). Filled from the job
                                                    # id by run_generation/_run_chain; blank → the adapter
                                                    # mints a random one. NEVER a name shared between jobs:
                                                    # ComfyUI opens an input file at EXECUTION time, so a
                                                    # queued prompt reads whatever bytes sit under that name
                                                    # by then — a shared slot silently swaps one job's
                                                    # reference image for another's.
    loras: Optional[list] = None                    # [{name, strength}] — pairs resolved server-side
    output_node: Optional[str] = None               # alias setting: ONLY this node's artifacts count
    output_ext: Optional[str] = None                # alias setting: fetch the sibling with THIS extension
    output_globs: Optional[list] = None             # alias setting: deliver every file matching these globs
    output_cases: Optional[list] = None             # alias setting: [{rig, globs}] — first detected case wins
    texture_format: Optional[str] = None            # alias setting: "jpeg" transcodes generic texture PNGs
    dummy_check: bool = True                         # alias setting: flat-mode 2x2-dummy safety net (opt-out
                                                    # for workflows that legitimately export a 1x1/2x2 texture)
    bypass: list = field(default_factory=list)      # per-backend node ids to bypass (ComfyUI mode-4: remove
                                                    # the node, rewire consumers to its same-typed input)
    cloud: Optional[dict] = None                    # cloud alias candidate block {endpoint, options} of
                                                    # whatever kind — `cloud_block(cand)`; None on ComfyUI
                                                    # candidates
    meshy: Optional[dict] = None                    # the pre-Tripo name of `cloud`, kept so existing
                                                    # callers keep constructing a Meshy request the old
                                                    # way; folded into `cloud` below, nothing reads it
    slot_held: bool = False                         # caller already holds the in-flight slot (chain) —
                                                    # generate() must not inc/dec it a second time
    job_id: str = ""                                # the job row this request runs for. Only live progress
                                                    # uses it (ctx.note_progress keys on it); blank simply
                                                    # means "report nothing", never an error.
    cloud_trace: dict = field(default_factory=dict)  # what a cloud adapter has BILLABLY done so far, written
                                                    # the moment the vendor's task exists rather than on the
                                                    # way out: a failed run returns no GenOutput, so without
                                                    # this the job row keeps no task id, endpoint or request
                                                    # and the one question a failure raises ("which task, and
                                                    # what did we send?") is answerable only from the vendor's
                                                    # dashboard. Per-REQUEST, so concurrent jobs on one
                                                    # adapter cannot overwrite each other's facts.

    def __post_init__(self):
        if self.cloud is None and self.meshy is not None:
            self.cloud = self.meshy


# ── cloud task kinds (Meshy, Tripo): the one seam main/admin ask "which kind?" ──
CLOUD_MODULES: dict = {meshy.KIND: meshy, tripo.KIND: tripo}     # kind → pure module


def cloud_kind(cand: dict) -> Optional[str]:
    """The cloud kind of a generation alias candidate (the key its block sits under),
    None for a ComfyUI (workflow) candidate."""
    for k in CLOUD_MODULES:
        if (cand or {}).get(k) is not None:
            return k
    return None


def cand_kind(cand: dict) -> str:
    return cloud_kind(cand) or "comfyui"


def backend_kind(b: dict) -> str:
    """A generation backend's kind: its type for a cloud backend, else comfyui. A
    candidate may run on a backend iff cand_kind(cand) == backend_kind(b) — backends are
    keyed (name, type), so a bare-name match could route a cloud alias onto a GPU box."""
    t = (b or {}).get("type")
    return t if t in CLOUD_MODULES else "comfyui"


def cloud_module(kind: str):
    return CLOUD_MODULES[kind]


def cloud_block(cand: dict) -> Optional[dict]:
    """The candidate's cloud block, whatever kind it is — what NormalizedRequest.cloud
    carries. A copy: the request must not write through into the stored candidate."""
    k = cloud_kind(cand)
    return dict(cand.get(k) or {}) if k else None


@dataclass
class GenBlob:
    """One produced artifact (image/video/audio/file) as raw bytes + type hints."""
    data: bytes
    mime: str
    kind: str                                       # "image" | "video" | "audio" | "file"
    name: Optional[str] = None                      # original ComfyUI filename (display/download)


@dataclass
class GenOutput:
    """Result of a generation dispatch: the artifacts plus run metadata."""
    blobs: list[GenBlob]
    meta: dict = field(default_factory=dict)


@dataclass
class ChainExport:
    """What a stage-1 adapter contributes before the chain runs: the name the mesh
    will have (`mesh_name`), pins the stage-1 request needs (`extra_fixed`) and, when
    the candidate cannot be a stage 1 as configured, a NAMED `error` — the chain fails
    the job with it before any GPU-minutes or credits are spent."""
    mesh_name: str
    extra_fixed: list = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class AdapterContext:
    """App services injected into every adapter. Keeps adapters import-cycle-free
    and hot-reload-safe: `log_enabled` is a callable so the live flag value is
    read per-call, never cached."""
    auth_headers: Callable[[dict], dict]
    inflight_inc: Callable[[str], None]
    inflight_dec: Callable[[str], None]
    cost_usd: Callable[[str, Optional[str], int, int], float]
    source_of: Callable[[Request], str]
    record_call: Callable[..., Any]
    log_enabled: Callable[[], bool]
    # Live LLM-call registry (dashboard "running calls"). Same lifecycle as the
    # in-flight counter — register at dispatch start, drop on completion incl. the
    # streamed-finally. Default no-ops so non-main constructions stay valid.
    active_register: Callable[[dict], Any] = lambda meta: None
    active_done: Callable[[Any], None] = lambda token: None
    # Speed-routing signal: fold a completed call's throughput into the backend's
    # tok/s EWMA — (bid, out_tok, duration_ms, status). Default no-op keeps non-main
    # constructions valid; the guard (200-only, min tokens) lives in main._note_speed.
    note_speed: Callable[[str, int, int, int], None] = \
        lambda bid, out_tok, dur_ms, status: None
    # Live generation progress: (job_id, info | None) — info None clears the entry.
    # Fed by ComfyUI's /ws feed (see _ws_progress), read by the job view. A no-op
    # default keeps every non-main construction (and every test) valid.
    note_progress: Callable[[str, Optional[dict]], None] = lambda job_id, info: None
    # Normalized reasoning toggle: (backend, model, requested, payload) -> (payload, control).
    # Default no-op (auto) so non-main constructions stay valid.
    apply_reasoning: Callable[[dict, Optional[str], Optional[str], dict], Any] = \
        lambda backend, model, requested, payload: (payload, None)
    # App-shared pooled httpx client (connection reuse / keep-alive across requests).
    # Callable so hot paths always see the live instance; None → adapters fall back
    # to a transient per-call client, keeping non-main constructions valid.
    http_client: Callable[[], Optional[httpx.AsyncClient]] = lambda: None
    # Installed LoRAs of a backend (discovery) — the `loras:[…]` pair resolution
    # validates against this. Default empty = no validation (non-main constructions).
    loras_of: Callable[[str], set] = lambda bid: set()


@asynccontextmanager
async def _pooled_client(ctx: AdapterContext):
    """The app-shared pooled client when injected (NOT closed here — main owns its
    lifecycle), else a transient one closed on exit. The shared client carries no
    per-client timeout, so every call site passes its own `timeout=`."""
    shared = ctx.http_client()
    if shared is not None:
        yield shared
    else:
        async with httpx.AsyncClient() as client:
            yield client


class BackendAdapter(ABC):
    """One instance per configured backend. Owns protocol-specific discovery and
    dispatch; the router (main.py) stays protocol-agnostic."""

    type: str = "base"
    serves_generation: bool = False   # True → the backend is a POST /v1/generations candidate

    def __init__(self, backend: dict, ctx: AdapterContext):
        self.backend = backend
        self.ctx = ctx
        self.name = backend["name"]                 # display label (stats / logs)
        # stable id = type:name — keys in-flight + pricing so an LLM and a ComfyUI
        # backend can share a name without colliding.
        self.bid = f'{backend.get("type", "openai")}:{backend["name"]}'

    @abstractmethod
    async def discover(self, client: httpx.AsyncClient) -> Capabilities:
        """Poll liveness + capabilities. Raise on failure → caller marks DOWN."""
        raise NotImplementedError

    async def dispatch(self, req: NormalizedRequest):
        """Forward an OpenAI-shaped (chat/completions/embeddings) request and
        return a Starlette Response. Owns the in-flight counter lifecycle —
        critically including decrementing when a *streamed* response finishes (in
        the generator's `finally`), not when headers are sent. May raise
        httpx.ConnectError/TimeoutException for the router to fail over.

        Default: unsupported (e.g. a pure image backend has no chat path)."""
        raise NotImplementedError(f"{self.type} adapter has no chat dispatch")

    async def generate(self, req: NormalizedRequest) -> GenOutput:
        """Run a generation task (image/video/audio) and return its artifacts.
        Owns the in-flight counter lifecycle. Default: unsupported."""
        raise NotImplementedError(f"{self.type} adapter has no generate path")

    async def cancel(self) -> None:
        """Best-effort: stop whatever this backend is running for the gateway (a
        cancelled job). Default no-op — a cloud task API has nothing to interrupt."""
        return None

    # ── workflow chains (main._run_chain) — the three places a stage is backend-specific ──
    def chain_export(self, cand: dict, succ: dict, params: dict, prefix: str) -> ChainExport:
        """Stage 1: how this backend will name/export the mesh. Default: not a stage 1."""
        return ChainExport("", error=f"a {self.type} backend cannot be a chain stage 1")

    async def chain_take_mesh(self, out: GenOutput, export: ChainExport, want_bytes: bool) -> Optional[bytes]:
        """Stage 1: the produced mesh (bytes; b'' when only existence is asked for), None if absent."""
        return None

    async def chain_feed_mesh(self, req2: NormalizedRequest, backend2: dict, mesh_param: str,
                              mesh_name: str, mesh_bytes: Optional[bytes], outdir: str) -> str:
        """Stage 2: put the mesh where THIS backend reads it and return the `mesh_ref`
        recorded on the job (a path, or a marker for an embedded upload)."""
        raise RuntimeError(f"a {self.type} backend cannot be a chain stage 2")


# ── OpenAI-compatible adapter (the only one in Phase 0) ───────────────────────

def _estimate_prompt_tokens(fwd: dict) -> int:
    """~chars/4 over the request's text parts — the fallback when a backend
    streams its usage block as all zeros (LocalAI does; measured 2026-07-09,
    non-streaming reports real numbers). Image parts are skipped: a base64
    image would explode a char-based estimate."""
    texts = []
    if isinstance(fwd.get("prompt"), str):
        texts.append(fwd["prompt"])
    for m in fwd.get("messages") or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            texts.extend(p["text"] for p in c
                         if isinstance(p, dict) and isinstance(p.get("text"), str))
    chars = sum(len(t) for t in texts)
    return (chars // 4 + 1) if chars else 0


class _StreamNormalizer:
    """Rewrites a backend's chat SSE into strict OpenAI shape. Strict clients
    (Hermes) abort mid-stream on the deviations LocalAI ships, so:

    - `delta` never carries null-valued keys (LocalAI puts `content:null` on
      the role/finish chunks) or an empty-string `content` — the keys are
      dropped; the finish chunk becomes the spec's `"delta":{}`.
    - the terminal usage chunk (`"choices":[]`) reaches the client only if it
      asked via `stream_options.include_usage` (WE always ask upstream, for
      stats); without the flag a stray `usage` key on delta chunks is dropped
      too. A client that did ask gets the chunk exactly once, directly before
      [DONE] — synthesized from the last chunk's id/model if the backend never
      sent one.
    - usage numbers are the backend's; if it only ever reported zeros, the
      fallback is counted content/reasoning deltas (llama.cpp-family streams
      ≈ one token per delta) + the caller's prompt estimate.

    Non-`data:` lines, [DONE], unparseable JSON and non-chunk payloads pass
    through verbatim; buffering is bytes-based so multi-byte UTF-8 split
    across network chunks survives. `flush()` returns the unterminated tail
    (error bodies aren't newline-framed)."""

    def __init__(self, want_usage: bool, prompt_estimate: int,
                 reasoning_as_content: bool = False):
        self.want_usage = want_usage
        self.prompt_est = prompt_estimate
        # LocalAI mislabels EVERY stream delta as `reasoning` when the rendered
        # prompt contains a thinking marker (e.g. Gemma-4's pre-closed
        # `<|channel>thought\n<channel|>` tail) — even though the model, with its
        # thought channel pre-closed, can only produce plain answer text. For
        # (backend, model)s flagged via the backend's `stream_reasoning_as_content`
        # globs those deltas ARE the content, so they are relabeled here.
        self.reasoning_as_content = reasoning_as_content
        self.buf = b""
        self.in_tok = self.out_tok = 0      # best backend-reported usage so far
        self.cache_read = self.cache_write = 0   # prompt-cache share of in_tok (stats only)
        self.pieces = 0                     # content-bearing deltas ≈ completion tokens
        self.meta: dict = {}                # id/model/created of the last chunk seen
        self.usage_sent = False

    def feed(self, chunk: bytes) -> bytes:
        self.buf += chunk
        out = []
        while b"\n" in self.buf:
            raw, self.buf = self.buf.split(b"\n", 1)
            out.append(self._line(raw.decode("utf-8", "ignore")))
        return "".join(out).encode()

    def flush(self) -> bytes:
        tail, self.buf = self.buf, b""
        return tail

    def tokens(self) -> tuple:
        """(prompt, completion) for stats + the client chunk — backend numbers,
        zeros replaced by the estimates."""
        return (self.in_tok or self.prompt_est, self.out_tok or self.pieces)

    def _usage_json(self) -> str:
        it, ot = self.tokens()
        return json.dumps({**self.meta, "object": "chat.completion.chunk",
                           "choices": [],
                           "usage": {"prompt_tokens": it, "completion_tokens": ot,
                                     "total_tokens": it + ot}},
                          ensure_ascii=False, separators=(",", ":"))

    def _line(self, line: str) -> str:
        data = line.strip()
        if not data.startswith("data:"):
            return line + "\n"
        payload = data[5:].strip()
        if payload == "[DONE]":
            if self.want_usage and not self.usage_sent:      # backend never sent one
                self.usage_sent = True
                return f"data: {self._usage_json()}\n\ndata: [DONE]\n"
            return line + "\n"
        try:
            obj = json.loads(payload)
        except Exception:
            return line + "\n"
        if not isinstance(obj, dict) or "choices" not in obj:
            return line + "\n"
        for k in ("id", "model", "created", "system_fingerprint"):
            if k in obj and obj[k] is not None:
                self.meta[k] = obj[k]
        u = obj.get("usage")
        if isinstance(u, dict):
            self.in_tok = max(self.in_tok, int(u.get("prompt_tokens") or 0))
            self.out_tok = max(self.out_tok, int(u.get("completion_tokens") or 0))
            # Prompt-cache share, for stats only — never forwarded, the client's
            # usage chunk keeps the strict OpenAI shape.
            det = u.get("prompt_tokens_details") or {}
            self.cache_read = max(self.cache_read,
                                  int(det.get("cached_tokens") or u.get("cached_tokens") or 0))
            self.cache_write = max(self.cache_write,
                                   int(u.get("cache_creation_input_tokens") or 0))
        if not obj.get("choices"):                            # terminal usage chunk
            if u is None:
                return line + "\n"                            # odd keepalive — leave it
            if self.want_usage and not self.usage_sent:
                self.usage_sent = True
                return f"data: {self._usage_json()}\n"        # own blank line follows
            return ""                                         # client didn't ask → swallow
        changed = False
        for ch in obj["choices"]:
            d = ch.get("delta") if isinstance(ch, dict) else None
            if isinstance(d, dict):
                if self.reasoning_as_content:
                    moved = ""
                    for k in ("reasoning", "reasoning_content"):
                        if k in d:
                            v = d.pop(k)
                            changed = True
                            if isinstance(v, str):
                                moved += v
                    if moved:
                        c = d.get("content")
                        d["content"] = (c + moved) if isinstance(c, str) else moved
                for k in [k for k, v in d.items()
                          if v is None or (k == "content" and v == "")]:
                    del d[k]
                    changed = True
                if any(isinstance(d.get(k), str) and d[k]
                       for k in ("content", "reasoning", "reasoning_content")):
                    self.pieces += 1
            if isinstance(ch, dict) and isinstance(ch.get("text"), str) and ch["text"]:
                self.pieces += 1                              # legacy /v1/completions
        if "usage" in obj and not self.want_usage:            # no flag → no usage anywhere
            del obj["usage"]
            changed = True
        if not changed:
            return line + "\n"
        return "data: " + json.dumps(obj, ensure_ascii=False,
                                     separators=(",", ":")) + "\n"


@dataclass
class _Call:
    """Per-dispatch bookkeeping shared by the stream and non-stream paths —
    built once in OpenAIAdapter._prepare()."""
    url: str
    headers: dict                       # outgoing headers (client's, minus hop-by-hop, plus backend auth)
    fwd: dict                           # outgoing payload (private keys stripped, reasoning applied)
    real_model: Optional[str]
    reasoning_ctl: Optional[str]        # x-reasoning-control value ("off:prefill", …) or None
    rheaders: dict                      # gateway response headers (serving backend + reasoning control)
    source: str
    req_text: str
    started: float
    log_on: bool
    act: Any                            # live-call registry token


# Anthropic error `type` per HTTP status — a client steers its retry off the status,
# but shows the type, so a 429 must not read as a generic api_error.
_ANTHROPIC_ERROR_TYPE = {400: "invalid_request_error", 401: "authentication_error",
                         403: "permission_error", 404: "not_found_error",
                         413: "request_too_large", 429: "rate_limit_error",
                         529: "overloaded_error"}


def _ratelimit_headers(headers) -> dict:
    """The upstream's `retry-after`, lowercased — the ONE response header of a
    backend error that is not diagnostics but an instruction to the caller.

    Everything else a backend sends is either the gateway's own business or
    describes a body the gateway is about to re-serialize (`content-length`), so
    the response builders below keep only `x-gateway*`/`x-reasoning*`. Dropping
    this one with them is silent: the client still gets its 429 and simply retries
    on a blind 1-2-4-8 s backoff. Measured 2026-09-01 on prod — one Claude Code
    request turned into ~10 upstream calls inside 20 s, against a rate limit that
    had told the gateway exactly how long to wait."""
    val = None
    for k, v in (headers or {}).items():
        if k.lower() == "retry-after":
            val = v
    return {"retry-after": val} if val else {}


def _anthropic_error(status: int, etype: str, message: str, headers=None) -> JSONResponse:
    """An error in the shape Claude Code expects — it parses `error.message` and
    shows it; an OpenAI-shaped error body would surface as an unhelpful blank."""
    keep = {k: v for k, v in (headers or {}).items()
            if k.lower().startswith(("x-gateway", "x-reasoning"))}
    keep.update(_ratelimit_headers(headers))
    return JSONResponse({"type": "error", "error": {"type": etype, "message": message[:2000]}},
                        status_code=status, headers=keep)


def _error_text(resp) -> str:
    """Readable message out of a failed upstream response (OpenAI error shape,
    Anthropic error shape, or raw body)."""
    parsed = getattr(resp, "parsed_json", None)
    if not isinstance(parsed, dict):
        try:
            parsed = json.loads(bytes(getattr(resp, "body", b"") or b"{}"))
        except Exception:
            parsed = {}
    err = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(err, dict) and err.get("message"):
        return str(err["message"])
    if isinstance(err, str):
        return err
    try:
        return bytes(getattr(resp, "body", b"") or b"").decode("utf-8", "ignore") or "upstream error"
    except Exception:
        return "upstream error"


# Client headers that must NEVER be forwarded to a backend. `authorization` AND
# `x-api-key` are both gateway credentials (Claude Code authenticates with the
# latter) — forwarding either would hand the caller's gateway key to OpenRouter,
# Anthropic or whoever else serves the call. The adapter adds the backend's own
# credential afterwards.
_HOP_BY_HOP = ("host", "content-length", "authorization", "x-api-key", "x-park-mode")


class OpenAIAdapter(BackendAdapter):
    """llama.cpp / llama-swap / vLLM / Ollama / Together / OpenRouter / OpenAI —
    anything that speaks `/v1/models` + `/v1/chat|completions|embeddings`.

    Also serves `/v1/messages` (Anthropic) for these backends by translating both
    directions through `anthropic_bridge` — so one alias can list an Anthropic
    backend AND an OpenRouter model, and the router's failover works across them
    without knowing anything about protocols."""

    type = "openai"

    async def discover(self, client: httpx.AsyncClient) -> Capabilities:
        b = self.backend
        resp = await client.get(
            f"{b['url']}/v1/models", headers=self.ctx.auth_headers(b), timeout=_DISCOVERY_TIMEOUT
        )
        resp.raise_for_status()
        payload = resp.json()
        context = extract_context(payload)
        context.update(await self._llamaswap_context(client, b))
        return Capabilities(models=extract_models(payload, b),
                            pricing=extract_pricing(payload), context=context)

    async def _llamaswap_context(self, client: httpx.AsyncClient, b: dict) -> dict[str, int]:
        """llama-swap's own /v1/models carries no n_ctx (only the llama-server behind
        each model knows it), so ask `/running` which models are loaded and read each
        one's `/upstream/<model>/v1/models` through the same extract_context(). ONLY
        loaded (`state == ready`) models are asked: `/upstream/…` on an unloaded model
        makes llama-swap LOAD it, and a discovery poll must never swap a model in.
        Every failure is silent — a server without `/running` (vLLM, OpenRouter,
        LocalAI: 404) is the normal case, and the listing's own values still count."""
        out: dict[str, int] = {}
        try:
            r = await client.get(f"{b['url']}/running", headers=self.ctx.auth_headers(b),
                                 timeout=_DISCOVERY_TIMEOUT)
            if r.status_code != 200:
                return out
            running = r.json().get("running") or []
        except Exception:
            return out
        for ent in running:
            if not isinstance(ent, dict) or ent.get("state") != "ready" or not ent.get("model"):
                continue
            mid = str(ent["model"])
            try:
                r = await client.get(f"{b['url']}/upstream/{mid}/v1/models",
                                     headers=self.ctx.auth_headers(b), timeout=_DISCOVERY_TIMEOUT)
                if r.status_code != 200:
                    continue
                found = extract_context(r.json())
            except Exception:
                continue
            # The upstream answers with ITS id (the --alias), which llama-swap keys the
            # same way; if it differs, the value still belongs to the model we asked for.
            if mid in found:
                out[mid] = found[mid]
            elif len(found) == 1:
                out[mid] = next(iter(found.values()))
        return out

    async def dispatch(self, req: NormalizedRequest):
        if req.path.startswith("/v1/messages"):
            return await self._dispatch_messages(req)
        call = self._prepare(req)                 # claims the in-flight slot
        if req.body.get("stream"):
            return await self._dispatch_stream(req, call)
        return await self._dispatch_once(req, call)

    async def _dispatch_messages(self, req: NormalizedRequest):
        """Serve an Anthropic Messages request off a chat-completions backend.

        The request is translated, dispatched down the NORMAL chat path (so
        in-flight accounting, sampling defaults, reasoning and stats behave exactly
        as they do for any other call), and the answer is translated back. The
        recorded endpoint stays `/v1/messages` so the console shows what the client
        actually called."""
        estimate = anthropic_bridge.estimate_input_tokens(req.body)
        if req.path.startswith("/v1/messages/count_tokens"):
            # No chat backend has this endpoint; answering from the estimate is the
            # whole point (Claude Code uses it to size the context, not to bill).
            return JSONResponse({"input_tokens": estimate})
        try:
            # `prompt_cache` (per backend): keep Claude Code's cache breakpoints in the
            # translated body. OpenRouter forwards them to Anthropic/Gemini models, where
            # dropping them means paying full price for the entire context every turn.
            # Opt-in, because a strict server may reject the part-list shape it produces.
            chat_body = anthropic_bridge.messages_to_chat(
                req.body, keep_cache_control=bool(self.backend.get("prompt_cache")))
        except anthropic_bridge.UnsupportedContent as e:
            return _anthropic_error(400, "invalid_request_error", str(e))
        chat_body["model"] = req.real_model
        chat_req = replace(req, path="/v1/chat/completions", body=chat_body,
                           stats_endpoint=req.stats_endpoint or "/v1/messages",
                           stream=bool(chat_body.get("stream")))
        if chat_body.get("stream"):
            # The bridge reads the chat usage chunk for message_delta; ask for it
            # explicitly — the client never sees the raw chunk.
            chat_body["stream_options"] = {"include_usage": True}
        resp = await self.dispatch(chat_req)
        status = getattr(resp, "status_code", 200)
        if status >= 400:
            # A backend-local load failure (llama-swap's "unable to start process"
            # 502) must reach the router UNTOUCHED: it recognises that body and fails
            # over to the next candidate. Re-shaping it here would leave that
            # detection depending on where the marker happens to land in the new body.
            if status == 502 and b"unable to start process" in (getattr(resp, "body", b"") or b""):
                return resp
            return _anthropic_error(status, _ANTHROPIC_ERROR_TYPE.get(status, "api_error"),
                                    _error_text(resp), headers=getattr(resp, "headers", None))
        if chat_body.get("stream") and isinstance(resp, StreamingResponse):
            return StreamingResponse(
                anthropic_bridge.messages_stream(resp, req.alias, input_tokens=estimate),
                media_type="text/event-stream",
                headers={k: v for k, v in (resp.headers or {}).items()
                         if k.lower().startswith(("x-gateway", "x-reasoning"))})
        chat_json = getattr(resp, "parsed_json", None)
        if not isinstance(chat_json, dict):
            try:
                chat_json = json.loads(bytes(resp.body))
            except Exception:
                chat_json = {}
        return JSONResponse(anthropic_bridge.chat_to_messages(chat_json, req.alias),
                            headers={k: v for k, v in (resp.headers or {}).items()
                                     if k.lower().startswith(("x-gateway", "x-reasoning"))})

    def _prepare(self, req: NormalizedRequest) -> _Call:
        """Shared per-dispatch setup: outgoing headers + payload (gateway-private
        `_` keys stripped, normalized reasoning applied for THIS backend — a
        per-backend, failover-safe copy), stats fields, live-call registry entry.
        Claims the in-flight slot LAST (after anything that could raise), with no
        await before dispatch uses it — the dispatching path's `finally` MUST
        release it via _finish()."""
        b = self.backend
        ctx = self.ctx
        headers = {
            k: v for k, v in req.raw.headers.items()
            if k.lower() not in _HOP_BY_HOP
        }
        headers.update(self._backend_auth())
        real_model = req.body.get("model")
        fwd, reasoning_ctl = self._payload(req)
        ctx.inflight_inc(self.bid)        # released on completion (stream close / return / error)
        act = ctx.active_register({       # dropped on completion (both finally paths)
            "alias": req.alias, "model": real_model, "backend": self.name,
            "source": ctx.source_of(req.raw), "endpoint": (req.stats_endpoint or req.path),
            "stream": bool(req.body.get("stream")),
        })
        # Response headers: which backend served the call (so API clients — incl. the
        # /ui playgrounds, which are plain clients — can show routing), plus the
        # reasoning control actually applied.
        rheaders = {"x-gateway-backend": self.name}
        if reasoning_ctl:
            rheaders["x-reasoning-control"] = reasoning_ctl
        return _Call(
            url=f"{b['url']}{req.path}", headers=headers, fwd=fwd, real_model=real_model,
            reasoning_ctl=reasoning_ctl, rheaders=rheaders,
            source=ctx.source_of(req.raw), req_text=json.dumps(fwd, ensure_ascii=False),
            started=time.monotonic(), log_on=ctx.log_enabled(), act=act,
        )

    def _backend_auth(self) -> dict:
        """Credential headers for THIS backend. Overridden where the protocol wants
        a different scheme (Anthropic: x-api-key / OAuth bearer)."""
        return self.ctx.auth_headers(self.backend)

    def _payload(self, req: NormalizedRequest) -> tuple[dict, Optional[str]]:
        """(outgoing body, reasoning-control label). Gateway-private `_` keys are
        stripped, the backend's sampling defaults fill only keys the request does
        NOT carry (client > alias > backend), and the normalized reasoning toggle is
        applied to a COPY — all of it derived per backend, so a failover re-derives
        it from the backend actually serving the call. Text endpoints only:
        embeddings have no sampling and /v1/audio is a binary passthrough.

        Overridden by adapters that must forward the client's body verbatim."""
        b = self.backend
        fwd = {k: v for k, v in req.body.items() if not k.startswith("_")}
        sd = b.get("sampling_defaults")
        if sd and not (req.path.startswith("/v1/embeddings") or req.path.startswith("/v1/audio/")):
            if isinstance(sd, dict):
                for k, v in sd.items():
                    if k not in fwd:
                        fwd[k] = v
            else:
                logger.warning(f"backend '{self.name}': sampling_defaults is not a dict — ignored")
        return self.ctx.apply_reasoning(b, req.body.get("model"), req.reasoning, fwd)

    def _usage_of(self, resp_json: dict) -> tuple[int, int]:
        """(input, output) tokens out of a non-streamed response body. Overridden
        where the protocol names them differently (Anthropic)."""
        usage = (resp_json.get("usage") or {}) if isinstance(resp_json, dict) else {}
        return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)

    def _cache_of(self, resp_json: dict) -> tuple[int, int]:
        """(cache_read, cache_write) — how much of the input came out of the
        backend's prompt cache, and how much was written into it. Both are SUBSETS
        of the input figure, so the console can show what was actually paid fresh.

        OpenAI-shaped backends report reads under `prompt_tokens_details.cached_
        tokens` (OpenAI, vLLM, OpenRouter) and have no write counter — except
        OpenRouter serving an Anthropic model, which passes Anthropic's through."""
        usage = (resp_json.get("usage") or {}) if isinstance(resp_json, dict) else {}
        details = usage.get("prompt_tokens_details") or {}
        read = int(details.get("cached_tokens") or usage.get("cached_tokens") or 0)
        write = int(usage.get("cache_creation_input_tokens") or 0)
        return read, write

    def _finish(self, call: _Call) -> None:
        """Release the in-flight slot + live-registry entry (every path's finally)."""
        self.ctx.inflight_dec(self.bid)
        self.ctx.active_done(call.act)

    def _record(self, req: NormalizedRequest, call: _Call, status: int,
                in_tok: int, out_tok: int, response_text: Optional[str] = None,
                response_audio: Optional[tuple] = None,
                cache: tuple[int, int] = (0, 0)) -> int:
        """Fire-and-forget stats row for this dispatch; returns the elapsed ms.
        `cache` is (read, write) — the prompt-cache share of `in_tok`."""
        ctx = self.ctx
        elapsed_ms = int((time.monotonic() - call.started) * 1000)
        ctx.note_speed(self.bid, out_tok, elapsed_ms, status)   # speed-routing EWMA (guarded in main)
        asyncio.create_task(ctx.record_call(
            duration_ms=elapsed_ms, backend=self.name, source=call.source,
            alias=req.alias, model=call.real_model, endpoint=(req.stats_endpoint or req.path),
            status=status, input_tokens=in_tok, output_tokens=out_tok,
            cost_usd=ctx.cost_usd(self.bid, call.real_model, in_tok, out_tok),
            request_text=call.req_text, response_text=response_text,
            response_audio=response_audio,
            reasoning=call.reasoning_ctl,
            cache_read=cache[0], cache_write=cache[1],
        ))
        return elapsed_ms

    async def _dispatch_stream(self, req: NormalizedRequest, call: _Call) -> Response:
        ctx = self.ctx
        # Ask the backend to emit a final usage chunk so streamed calls record
        # real tokens/cost (graceful: backends that ignore the field just yield
        # no usage line → the normalizer's estimates kick in). The client only
        # sees that chunk if IT asked (call.fwd still holds its stream_options).
        body = {**call.fwd,
                "stream_options": {**(call.fwd.get("stream_options") or {}), "include_usage": True}}
        want_usage = bool((call.fwd.get("stream_options") or {}).get("include_usage"))
        r2c = any(fnmatch.fnmatch(call.real_model or "", g)
                  for g in (self.backend.get("stream_reasoning_as_content") or []))
        norm = _StreamNormalizer(want_usage, _estimate_prompt_tokens(call.fwd),
                                 reasoning_as_content=r2c)

        # Open the upstream stream BEFORE building the client response: status +
        # headers arrive first, so a backend that answers with an error (llama-swap
        # "unable to start process" 502 when the model can't load) returns a plain
        # Response with its REAL status — previously the SSE shell went out as
        # HTTP 200 before upstream ever replied — and _dispatch_over can fail over
        # on it. Connect/timeout raise into _dispatch_over's failover exactly like
        # the non-stream path; every error exit releases the in-flight slot.
        client_cm = _pooled_client(ctx)
        client = None
        try:
            client = await client_cm.__aenter__()
            stream_cm = client.stream("POST", call.url, json=body,
                                      headers=call.headers, timeout=_CHAT_TIMEOUT)
            resp = await stream_cm.__aenter__()
        except BaseException:
            if client is not None:
                await client_cm.__aexit__(None, None, None)
            self._finish(call)
            raise
        if call.log_on:
            logger.info(f"← [{self.name}] {req.path} HTTP {resp.status_code} "
                        f"(stream open, {(time.monotonic() - call.started):.2f}s)")
        if resp.status_code >= 400:               # error body, not SSE — real status through
            try:
                err = await resp.aread()
            finally:
                await stream_cm.__aexit__(None, None, None)
                await client_cm.__aexit__(None, None, None)
                self._finish(call)
            self._record(req, call, resp.status_code, 0, 0,
                         response_text=err.decode("utf-8", "ignore"))
            return Response(content=err, status_code=resp.status_code,
                            media_type=resp.headers.get("content-type"),
                            headers={**call.rheaders, **_ratelimit_headers(resp.headers)})

        async def generate():
            try:
                async for chunk in resp.aiter_bytes():
                    out = norm.feed(chunk)
                    if out:
                        yield out
                tail = norm.flush()
                if tail:
                    yield tail
            finally:
                await stream_cm.__aexit__(None, None, None)
                await client_cm.__aexit__(None, None, None)
                self._finish(call)
            in_tok, out_tok = norm.tokens()
            self._record(req, call, resp.status_code, in_tok, out_tok,
                         cache=(norm.cache_read, norm.cache_write))

        return StreamingResponse(generate(), media_type="text/event-stream", headers=call.rheaders)

    async def _dispatch_once(self, req: NormalizedRequest, call: _Call) -> Response:
        try:
            async with _pooled_client(self.ctx) as client:
                resp = await client.post(call.url, json=call.fwd,
                                         headers=call.headers, timeout=_CHAT_TIMEOUT)
        finally:
            self._finish(call)
        # Parse ONCE (usage → tokens/cost); the body itself passes through as the
        # backend's raw bytes — the gateway doesn't transform it, so the old
        # parse → re-serialize (stats) → re-serialize (JSONResponse) round-trip
        # is gone. Internal callers (Responses bridge) still read resp.body.
        # Binary bodies (e.g. /v1/audio/speech WAV) are neither parsed nor stored
        # in the stats blob — charset-decoding audio bytes is garbage + waste.
        ct = resp.headers.get("content-type", "")
        is_texty = ct.startswith(("application/json", "text/"))
        resp_json = {}
        if is_texty:
            try:
                resp_json = resp.json()
            except Exception:
                pass
        in_tok, out_tok = self._usage_of(resp_json if isinstance(resp_json, dict) else {})
        # Successful binary audio (TTS) → stored as its own stats blob so the call
        # view can play it back; other binary bodies stay unstored.
        resp_audio = ((resp.content, ct) if ct.startswith("audio/") and resp.status_code == 200
                      else None)
        elapsed_ms = self._record(req, call, resp.status_code, in_tok, out_tok,
                                  response_text=(resp.text if is_texty else None),
                                  response_audio=resp_audio,
                                  cache=self._cache_of(resp_json if isinstance(resp_json, dict) else {}))
        if call.log_on:
            logger.info(f"← [{self.name}] {req.path} HTTP {resp.status_code} ({elapsed_ms} ms)")
        out = Response(resp.content, status_code=resp.status_code,
                       media_type=resp.headers.get("content-type", "application/json"),
                       headers={**call.rheaders, **_ratelimit_headers(resp.headers)})
        out.parsed_json = resp_json    # internal callers (Responses bridge) reuse this — no re-parse
        return out


# ── Anthropic adapter ─────────────────────────────────────────────────────────
# Claude Code's own protocol, forwarded VERBATIM. Nothing on this path may rewrite
# the body or the stream: `cache_control` breakpoints, thinking signatures and
# fine-grained tool streaming only survive untouched, and without cache
# breakpoints Claude Code re-reads the whole context at full price every turn.
# Concretely, three things that run for every other backend must NOT run here:
# `_StreamNormalizer` (OpenAI-SSE-specific), `apply_reasoning()` (Claude Code sets
# `thinking` itself) and `sampling_defaults`.

_ANTHROPIC_VERSION = "2023-06-01"           # sent only when the client didn't
_OAUTH_BETA = "oauth-2025-04-20"            # required for subscription (setup-token) auth


class AnthropicAdapter(OpenAIAdapter):
    """`type: anthropic` — api.anthropic.com (or a compatible endpoint).

    Reachable ONLY through `/v1/messages`: the credential is normally a personal
    Claude subscription token, and a subscription is licensed for Claude Code, not
    for serving a general-purpose API. Keeping the other endpoints shut is the
    enforcement; the README and the console say the same thing in words.
    Inherits the OpenAI adapter's in-flight/stats/failover bookkeeping and replaces
    exactly what is protocol-specific: auth headers, payload (verbatim), usage
    field names, and the streamed passthrough."""

    type = "anthropic"

    def _subscription(self) -> bool:
        """A subscription (OAuth setup-token) credential rather than an API key."""
        return (self.backend.get("auth_mode") or "subscription") == "subscription"

    def _backend_auth(self) -> dict:
        key = self.backend.get("api_key") or ""
        if not key:
            return {}
        if self._subscription():
            # setup-token → OAuth bearer + the beta that unlocks it. The client's own
            # anthropic-beta list is preserved by _prepare and merged in dispatch().
            return {"authorization": f"Bearer {key}"}
        return {"x-api-key": key}

    def _payload(self, req: NormalizedRequest) -> tuple[dict, Optional[str]]:
        """Verbatim, minus the gateway's private `_` keys. No sampling defaults, no
        reasoning rewrite — see the module note above."""
        return {k: v for k, v in req.body.items() if not k.startswith("_")}, None

    def _usage_of(self, resp_json: dict) -> tuple[int, int]:
        """Anthropic names them input_tokens/output_tokens; cache reads and writes
        are input the model processed, so they count toward the input figure the
        console shows (with `_cache_of` breaking that total back down)."""
        u = (resp_json.get("usage") or {}) if isinstance(resp_json, dict) else {}
        in_tok = (int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0)
                  + int(u.get("cache_creation_input_tokens") or 0))
        return in_tok, int(u.get("output_tokens") or 0)

    def _cache_of(self, resp_json: dict) -> tuple[int, int]:
        """Anthropic reports the cache explicitly — a read costs a tenth of a fresh
        token, a write a quarter more, so the split is what makes a session's cost
        readable."""
        u = (resp_json.get("usage") or {}) if isinstance(resp_json, dict) else {}
        return (int(u.get("cache_read_input_tokens") or 0),
                int(u.get("cache_creation_input_tokens") or 0))

    async def discover(self, client: httpx.AsyncClient) -> Capabilities:
        """Anthropic's GET /v1/models, falling back to the backend's configured
        `models` list — a subscription token is not guaranteed to be allowed on the
        models endpoint, and a 401 there must not take the backend down when the
        admin has said which models to offer."""
        b = self.backend
        configured = b.get("models") or []
        if isinstance(configured, str):
            configured = [m.strip() for m in configured.split(",") if m.strip()]
        headers = {"anthropic-version": _ANTHROPIC_VERSION, **self._backend_auth()}
        if self._subscription():
            headers["anthropic-beta"] = _OAUTH_BETA
        try:
            resp = await client.get(f"{b['url']}/v1/models", headers=headers,
                                    timeout=_DISCOVERY_TIMEOUT)
            resp.raise_for_status()
            models = extract_models(resp.json(), b) | set(configured)
            return Capabilities(models=models, pricing={})
        except Exception as e:
            if configured:
                logger.info(f"[{self.name}] /v1/models unavailable ({type(e).__name__}) — "
                            f"using the {len(configured)} configured model(s)")
                return Capabilities(models=set(configured), pricing={})
            raise

    async def dispatch(self, req: NormalizedRequest):
        if not req.path.startswith("/v1/messages"):
            # Unreachable through the router (resolve_routes filters this backend out
            # for every other path); explicit so a future caller can't slip past it.
            return _anthropic_error(404, "not_found_error",
                                    f"backend '{self.name}' serves /v1/messages only")
        call = self._prepare(req)                  # claims the in-flight slot
        # The client's anthropic-beta list must survive, and subscription auth needs
        # its own beta appended to it.
        if self._subscription():
            betas = [b for b in (call.headers.get("anthropic-beta") or "").split(",") if b.strip()]
            if _OAUTH_BETA not in betas:
                betas.append(_OAUTH_BETA)
            call.headers["anthropic-beta"] = ",".join(betas)
        call.headers.setdefault("anthropic-version", _ANTHROPIC_VERSION)
        call.headers.pop("accept-encoding", None)  # httpx re-negotiates its own
        if req.body.get("stream"):
            return await self._dispatch_stream_raw(req, call)
        return await self._dispatch_once(req, call)

    async def _dispatch_stream_raw(self, req: NormalizedRequest, call: _Call) -> Response:
        """Byte-for-byte SSE passthrough. The gateway only READS along (usage for
        stats); it never rewrites an event, so thinking signatures and partial tool
        JSON reach Claude Code exactly as Anthropic sent them."""
        ctx = self.ctx
        client_cm = _pooled_client(ctx)
        client = None
        try:
            client = await client_cm.__aenter__()
            stream_cm = client.stream("POST", call.url, json=call.fwd,
                                      headers=call.headers, timeout=_CHAT_TIMEOUT)
            resp = await stream_cm.__aenter__()
        except BaseException:
            if client is not None:
                await client_cm.__aexit__(None, None, None)
            self._finish(call)
            raise
        if call.log_on:
            logger.info(f"← [{self.name}] {req.path} HTTP {resp.status_code} "
                        f"(stream open, {(time.monotonic() - call.started):.2f}s)")
        if resp.status_code >= 400:
            try:
                err = await resp.aread()
            finally:
                await stream_cm.__aexit__(None, None, None)
                await client_cm.__aexit__(None, None, None)
                self._finish(call)
            self._record(req, call, resp.status_code, 0, 0,
                         response_text=err.decode("utf-8", "ignore"))
            return Response(content=err, status_code=resp.status_code,
                            media_type=resp.headers.get("content-type"),
                            headers={**call.rheaders, **_ratelimit_headers(resp.headers)})

        counted = {"in": 0, "out": 0, "read": 0, "write": 0, "text": []}

        def sniff(raw: str) -> None:
            """Read usage + text off the passing events (stats only). The cache
            figures ride in `message_start`'s usage — the very numbers that say
            whether a long Claude Code session is still being served cheaply."""
            for line in raw.split("\n"):
                if not line.startswith("data:"):
                    continue
                try:
                    obj = json.loads(line[5:].strip())
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                u = obj.get("usage") or (obj.get("message") or {}).get("usage") or {}
                if u:
                    counted["in"] = max(counted["in"], self._usage_of({"usage": u})[0])
                    counted["out"] = max(counted["out"], int(u.get("output_tokens") or 0))
                    read, write = self._cache_of({"usage": u})
                    counted["read"] = max(counted["read"], read)
                    counted["write"] = max(counted["write"], write)
                d = obj.get("delta") or {}
                if d.get("type") == "text_delta" and d.get("text"):
                    counted["text"].append(d["text"])

        async def generate():
            buf = ""
            try:
                async for chunk in resp.aiter_bytes():
                    buf += chunk.decode("utf-8", "ignore")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        sniff(line)
                    yield chunk                       # verbatim, always
            finally:
                await stream_cm.__aexit__(None, None, None)
                await client_cm.__aexit__(None, None, None)
                self._finish(call)
            self._record(req, call, resp.status_code, counted["in"], counted["out"],
                         response_text="".join(counted["text"]) or None,
                         cache=(counted["read"], counted["write"]))

        return StreamingResponse(generate(), media_type="text/event-stream", headers=call.rheaders)


# ── ComfyUI adapter ───────────────────────────────────────────────────────────
# A focused async port of anima-verse's ComfyUIBackend (text2img path). Workflows
# are stored API-format JSON whose injectable knobs are *named Primitive nodes*
# (by `_meta.title`) with class-based fallbacks. Dispatch is submit → poll → view.

_MIME_BY_EXT = {
    ".png": ("image/png", "image"), ".jpg": ("image/jpeg", "image"),
    ".jpeg": ("image/jpeg", "image"), ".webp": ("image/webp", "image"),
    ".gif": ("image/gif", "image"), ".mp4": ("video/mp4", "video"),
    ".webm": ("video/webm", "video"), ".wav": ("audio/wav", "audio"),
    ".mp3": ("audio/mpeg", "audio"), ".flac": ("audio/flac", "audio"),
    # 3D artifacts (Trellis/rigging workflows) — kind "file" renders as download
    ".glb": ("model/gltf-binary", "file"), ".gltf": ("model/gltf+json", "file"),
    ".obj": ("model/obj", "file"), ".fbx": ("application/octet-stream", "file"),
    ".ply": ("application/octet-stream", "file"), ".stl": ("model/stl", "file"),
    # Meshy target formats beyond the ComfyUI set — without them a delivered mesh
    # falls through to the ("application/octet-stream", "image") default and would be
    # stored/rendered as an IMAGE.
    ".usdz": ("model/vnd.usdz+zip", "file"), ".3mf": ("model/3mf", "file"),
}


def _mime_and_kind(filename: str) -> tuple[str, str]:
    return _MIME_BY_EXT.get(os.path.splitext(filename)[1].lower(),
                            ("application/octet-stream", "image"))


def _image_dims(b: bytes) -> Optional[tuple]:
    """(width, height) from a PNG, baseline-JPEG or WebP header, else None. Enough to
    catch the 2x2 dummy — no image library needed. None means "format not recognised"
    (unknown ≠ absent — the caller tracks image PRESENCE separately)."""
    if b[:8] == b"\x89PNG\r\n\x1a\n" and len(b) >= 24:
        w, h = struct.unpack(">II", b[16:24])
        return int(w), int(h)
    if b[:2] == b"\xff\xd8":                          # JPEG: scan for a SOF marker
        i = 2
        while i + 9 < len(b):
            if b[i] != 0xFF:
                i += 1
                continue
            marker = b[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3):
                h, w = struct.unpack(">HH", b[i + 5:i + 9])
                return int(w), int(h)
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            i += 2 + struct.unpack(">H", b[i + 2:i + 4])[0]
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP" and len(b) >= 20:   # WebP (VP8/VP8L/VP8X)
        fmt = b[12:16]
        if fmt == b"VP8X" and len(b) >= 30:          # extended: 24-bit width-1/height-1
            w = 1 + int.from_bytes(b[24:27], "little")
            h = 1 + int.from_bytes(b[27:30], "little")
            return w, h
        if fmt == b"VP8 " and len(b) >= 30:          # lossy: 14-bit dims after the start code
            return (int.from_bytes(b[26:28], "little") & 0x3FFF,
                    int.from_bytes(b[28:30], "little") & 0x3FFF)
        if fmt == b"VP8L" and len(b) >= 25:          # lossless: 14-bit dims packed after the 0x2f sig
            bits = int.from_bytes(b[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


def glb_chunks(data: bytes) -> Optional[tuple]:
    """Walk a glTF-binary into `(json-dict, BIN-bytes-or-None)`, or None if `data`
    isn't a parseable GLB. `bin` is None when the file carries no BIN chunk (buffer 0
    is external). The shared pure leaf under `_glb_info` (delivery validation) and
    `previewanim` (idle preview) — no gateway state, importable by either."""
    try:
        if data[:4] != b"glTF" or len(data) < 20:
            return None
        off, json_chunk, bin_chunk = 12, None, None
        while off + 8 <= len(data):
            clen, ctype = struct.unpack_from("<I4s", data, off)
            body = data[off + 8: off + 8 + clen]
            if ctype == b"JSON":
                json_chunk = body
            elif ctype == b"BIN\x00":
                bin_chunk = body
            off += 8 + clen                          # chunkLength already includes padding
        if json_chunk is None:
            return None
        return json.loads(json_chunk), bin_chunk
    except Exception:
        return None


def _glb_info(data: bytes) -> Optional[dict]:
    """Parse a GLB (via `glb_chunks`) into `{texture_dims:[(w,h),…], embedded_images:int,
    mixamorig_joints:int, skins:int, meshes:int}`, or None if `data` isn't a parseable
    glTF-binary. Backs the delivery validation (2x2 dummy, 52-joint mixamorig skin,
    embedded texture present)."""
    parsed = glb_chunks(data)
    if parsed is None:
        return None
    try:
        gltf, bin_chunk = parsed
        bufviews = gltf.get("bufferViews") or []
        dims, embedded_images = [], 0
        for im in (gltf.get("images") or []):
            bv_i = im.get("bufferView")
            if bv_i is None or bin_chunk is None or bv_i >= len(bufviews):
                continue
            embedded_images += 1                     # an embedded texture IS present…
            bv = bufviews[bv_i]
            start = bv.get("byteOffset", 0)
            # …readable dims are a bonus (2x2-dummy check): a 64 KB window clears a
            # large EXIF/ICC preamble before a JPEG SOF; an unrecognised format (KTX2,
            # …) yields no dims but still counts as present above.
            head = bin_chunk[start: start + min(bv.get("byteLength", 0), 65536)]
            wh = _image_dims(head)
            if wh:
                dims.append(wh)
        joints = sum(1 for n in (gltf.get("nodes") or [])
                     if str(n.get("name", "")).lower().startswith("mixamorig"))
        return {"texture_dims": dims, "embedded_images": embedded_images,
                "mixamorig_joints": joints, "skins": len(gltf.get("skins") or []),
                "meshes": len(gltf.get("meshes") or [])}
    except Exception:
        return None


def _is_dummy(dims) -> bool:
    """True when a GLB's readable embedded-texture dims are ALL the tell-tale ≤2 px
    export-node stub (the known bug). Empty/None dims (no readable image) → not a
    positive dummy verdict — presence is judged separately (`embedded_images`)."""
    return bool(dims) and max(max(w, h) for w, h in dims) <= 2


def _check_glb_not_dummy(blobs) -> None:
    """Safety net for the flat/single-file modes: a GLB whose ONLY embedded texture
    is the 2x2 dummy is the known texture-export node bug, not a result — fail the
    job clearly (raises → final content error, never retried across backends).
    Case mode runs the fuller _validate_delivery instead."""
    for b in blobs:
        if b.mime != "model/gltf-binary":
            continue
        info = _glb_info(b.data)
        dims = info["texture_dims"] if info else None
        if _is_dummy(dims):
            w, h = dims[0]
            raise RuntimeError(f"GLB '{b.name or 'output'}' carries only a {w}x{h} dummy texture "
                               "(known texture-export node bug) — no real texture was embedded; "
                               "treated as a failed generation")


_SIZE_LIMIT_MB = 30                                  # web-suitability guideline (warn, not fail)


def normalize_delivery(blobs: list, rig: Optional[str],
                       texture_format: Optional[str] = None) -> None:
    """Fix known mesh-pipeline artifacts IN PLACE, before validation.

    generic (FBX + separate texture PNGs): the bake writes the textures
    V-flipped relative to the FBX UVs — a native consumer (the /ui preview,
    the 3D client, a DCC import) sees the mapping mirrored; only GL-style
    loaders that flip images on upload happen to render it right. The gateway
    is the ONE place that corrects this, so every consumer renders the pair
    as delivered, without client-side compensation. All PNGs of a generic
    delivery share the FBX's UV set (basecolor, metallic, …), so all are
    flipped. A blob flag guards against double application (the adapter
    normalizes case deliveries, the chain level normalizes the combined one).
    A failed normalize delivers the original bytes instead of failing the job.

    `texture_format="jpeg"` (alias Output option) additionally transcodes the
    flipped texture PNGs to JPEG (quality 90) in the same decode pass —
    ComfyUI has no JPEG export, so the gateway is the one place to shrink a
    multi-MB bake for delivery. A texture with a REAL alpha channel keeps PNG
    (JPEG has none). The FBX references its texture by a temp path anyway
    (clients re-bind by delivered name), so the extension change is safe.
    """
    if rig != "generic":
        return
    for b in blobs:
        if getattr(b, "_normalized", False):
            continue
        if not (b.name or "").lower().endswith(".png"):
            continue
        try:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(b.data)).transpose(Image.FLIP_TOP_BOTTOM)
            fmt = "PNG"
            if (texture_format or "").lower() in ("jpg", "jpeg"):
                if img.mode == "P" and "transparency" in img.info:
                    img = img.convert("RGBA")
                translucent = ("A" in img.getbands()
                               and img.getchannel("A").getextrema()[0] < 255)
                if not translucent:
                    img = img.convert("RGB")
                    fmt = "JPEG"
            buf = io.BytesIO()
            img.save(buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
            b.data = buf.getvalue()
            if fmt == "JPEG":
                b.name = b.name[:-len(".png")] + ".jpg"
                b.mime = "image/jpeg"
            b._normalized = True
        except Exception as e:
            logger.warning(f"normalize_delivery: texture normalize failed for "
                           f"'{b.name}' — delivering as produced: {e}")


def validate_delivery(blobs: list, rig: Optional[str]) -> list:
    """Gate a case-mode delivery before reporting success (returns warnings; raises
    on hard failure). Per the character-model spec:
    - mixamo (humanoid): a valid GLB with a mixamorig skin and a real embedded
      texture; a 2x2 texture is the known node bug → fail.
    - generic (non-humanoid): a rigged FBX AND its basecolor PNG (the FBX only
      references the texture by a temp path, so the PNG must ship with it).
    - both: files over ~30 MB warn (web-suitability guideline, not a hard fail).
    Orientation/scale are the client's to normalise — not checked."""
    warnings = []
    for b in blobs:
        mb = len(b.data) / (1024 * 1024)
        if mb > _SIZE_LIMIT_MB:
            warnings.append(f"{b.name} is {mb:.0f} MB (> {_SIZE_LIMIT_MB} MB guideline)")
    if rig == "mixamo":
        glb = next((b for b in blobs if b.mime == "model/gltf-binary"), None)
        if glb is None:
            raise RuntimeError("humanoid (mixamo) result: no GLB was delivered")
        info = _glb_info(glb.data)
        if info is None:
            raise RuntimeError(f"humanoid result '{glb.name}': not a valid GLB")
        if info["skins"] == 0 or info["mixamorig_joints"] == 0:
            raise RuntimeError(f"humanoid result '{glb.name}': no mixamorig skin "
                               f"(skins={info['skins']}, mixamorig joints={info['mixamorig_joints']})")
        if info["embedded_images"] == 0:             # presence is separate from readable dims:
            raise RuntimeError(f"humanoid result '{glb.name}': no embedded texture")
        if _is_dummy(info["texture_dims"]):          # a WebP/KTX2 texture is present but may be unreadable
            raise RuntimeError(f"humanoid result '{glb.name}': embedded texture is a 2x2 dummy "
                               "(known node bug) — no result")
        if info["mixamorig_joints"] < 50:
            warnings.append(f"{glb.name}: {info['mixamorig_joints']} mixamorig joints (expected ~52)")
    elif rig == "generic":
        fbx = next((b for b in blobs if (b.name or "").lower().endswith(".fbx")), None)
        png = next((b for b in blobs if (b.mime or "").startswith("image/")), None)
        if fbx is None:
            raise RuntimeError("non-humanoid (generic) result: no rigged FBX was delivered")
        if not (fbx.data[:23].find(b"Kaydara FBX") != -1 or fbx.data[:4] == b"; FB"):
            warnings.append(f"{fbx.name}: does not look like an FBX (magic missing)")
        if png is None:
            raise RuntimeError("non-humanoid (generic) result: no basecolor PNG — the FBX only "
                               "references its texture by a temp path, so the PNG must ship with it")
    return warnings


def _view_params(item) -> Optional[tuple]:
    """Map ONE item from a ComfyUI output list to (filename, /view params), or
    None when it isn't a fetchable file. Two shapes seen in the wild:

    - dict `{filename, subfolder?, type?}` — core save nodes (SaveImage/Video/…).
    - str absolute/relative PATH — custom nodes emit the on-disk path instead
      (UniRig's `fbx_file`: '/…/ComfyUI/output/x_mia.fbx'; Preview3D's `result`).
      ComfyUI still serves it via /view by basename; the dir before an
      output/temp/input segment becomes the subfolder + type. Only paths with a
      known artifact extension are taken, so booleans/status strings are ignored."""
    if isinstance(item, dict):
        fn = item.get("filename")
        if not fn:
            return None
        view = {"filename": fn, "type": item.get("type", "output")}
        if item.get("subfolder"):
            view["subfolder"] = item["subfolder"]
        return fn, view
    if isinstance(item, str) and item:
        p = item.replace("\\", "/")
        if os.path.splitext(p)[1].lower() not in _MIME_BY_EXT:   # not a recognised artifact
            return None
        # Default when no output/temp/input segment names the dir type: an absolute
        # path is reduced to its basename (best guess — /view can't take its dirs),
        # but a RELATIVE path is already output-relative, so its dirs are the subfolder.
        typ = "output"
        rel = p.rsplit("/", 1)[-1] if (p.startswith("/") or ":" in p.split("/", 1)[0]) else p
        for t in ("output", "temp", "input"):
            i = p.rfind(f"/{t}/")
            if i != -1:
                typ, rel = t, p[i + len(t) + 2:]
                break
        fn = rel.rsplit("/", 1)[-1]
        sub = rel[:len(rel) - len(fn)].strip("/")
        view = {"filename": fn, "type": typ}
        if sub:
            view["subfolder"] = sub
        return fn, view
    return None


def _comfy_prompt_error(status: int, text: str, wf: dict, mapping: dict) -> str:
    """Human-readable ComfyUI /prompt rejection. Raw node_errors name only node ids —
    meaningless in the console, where everything is wired by node TITLE. Each error
    line therefore carries the node's title, class, field, the REQUEST PARAM mapped
    onto that field (the thing the user actually typed), and the received value.
    Falls back to the raw body when the payload shape is unknown."""
    try:
        err = json.loads(text or "")
        node_errors = err.get("node_errors") or {}
    except (ValueError, AttributeError):
        return f"ComfyUI /prompt HTTP {status}: {text}"
    lines = []
    for nid, ne in node_errors.items():
        node = (wf or {}).get(str(nid)) or {}
        title = (node.get("_meta") or {}).get("title") or ""
        cls = ne.get("class_type") or node.get("class_type") or "?"
        where = f"node {nid} “{title}”" if title else f"node {nid}"
        for e in ne.get("errors") or []:
            extra = e.get("extra_info") or {}
            fldn = extra.get("input_name") or ""
            param = next((p for p, m in (mapping or {}).items()
                          if str((m or {}).get("node")) == str(nid) and (m or {}).get("field") == fldn), None)
            via = f" ← request param '{param}'" if param else ""
            recv = extra.get("received_value")
            recv_s = f" (received {recv!r})" if recv is not None else ""
            lines.append(f"{where} ({cls}).{fldn}{via}: {e.get('message', 'error')}{recv_s}")
    # status 200 = ComfyUI ACCEPTED the prompt and only flagged individual nodes, so it
    # is not a rejection: saying "HTTP 200" or "rejected" would both mislead.
    accepted = status == 200
    where = "queued with unrunnable nodes" if accepted else f"HTTP {status}"
    if not lines:
        msg = (err.get("error") or {}).get("message") or text
        return f"ComfyUI /prompt {where}: {msg}"
    head = (err.get("error") or {}).get("message") or ("workflow incomplete" if accepted
                                                       else "prompt rejected")
    return f"ComfyUI {head} ({where}): " + " · ".join(lines)


def _node_by_title(wf: dict, title: str) -> Optional[str]:
    t = title.lower()
    for nid, node in wf.items():
        if node.get("_meta", {}).get("title", "").lower() == t:
            return nid
    return None


_FINAL_TITLES = ("output_final", "Output")           # image workflows / mesh convention


def _preview_consumer(wf: dict, nid: str) -> Optional[str]:
    """The `Preview*` node (Preview3D, …) that consumes node `nid`'s output, if any.
    Links are `[<source-node-id>, <slot>]` in an input value."""
    for cid in sorted(wf, key=lambda n: (len(n), n)):
        if not str((wf[cid] or {}).get("class_type") or "").startswith("Preview"):
            continue
        for v in ((wf[cid] or {}).get("inputs") or {}).values():
            if isinstance(v, list) and v and str(v[0]) == str(nid):
                return cid
    return None


def final_output_node(wf: dict) -> Optional[str]:
    """The node whose `/history` outputs ARE the workflow's main result — the
    delivery node. Found by title: `output_final` (image workflows) or `Output`
    (the mesh convention, see sample_comfyui_workflows/README.md — the export that
    writes `<name>.<ext>` next to intermediates like `<name>_whitemesh.glb`).

    One redirect on top: the mesh export classes (`Trellis2ExportMesh`,
    `Hy3D21ExportMesh`) return a bare tuple and NO `ui` dict, so despite
    OUTPUT_NODE=True they never appear in `/history.outputs` — measured on
    k12-gpu. Their file reaches the client through the `Preview3D` node that
    consumes them, so a Preview consumer of the titled node becomes the delivery
    node (trellis2_high 82→83, low 100→101, Pixal3D 312→317, hunyuan3d 50→51,
    mesh-shrink 30→31). Core save nodes carry their own `ui` dict and have no
    such consumer, so they stay themselves (triposplat's SaveGLB 107).

    NOTE this is the DELIVERY node only. A chain's `successor.export_node` stays
    the EXPORT node (82/100/312/50) — that is where the gateway pins
    `filename_prefix`, which a preview node has no field for.

    Backs the auto delivery mode and the registration pre-fill; None when the
    workflow declares neither title."""
    for t in _FINAL_TITLES:
        nid = _node_by_title(wf, t)
        if nid is not None:
            return _preview_consumer(wf, nid) or nid
    return None


def _node_by_class(wf: dict, class_type: str) -> Optional[str]:
    for nid, node in wf.items():
        if node.get("class_type") == class_type:
            return nid
    return None


def _clip_nodes(wf: dict, want: str) -> Optional[str]:
    """Positive/negative CLIPTextEncode by title hint, falling back to order
    (first = positive, second = negative)."""
    found = []
    for nid, node in wf.items():
        if node.get("class_type") == "CLIPTextEncode":
            title = node.get("_meta", {}).get("title", "").lower()
            if want == "positive" and ("positive" in title or "prompt" in title):
                return nid
            if want == "negative" and "negative" in title:
                return nid
            found.append(nid)
    if want == "positive":
        return found[0] if found else None
    return found[1] if len(found) > 1 else None


_IMG_LOADER_CLASSES = ("LoadImage", "LoadAndResizeImage", "LoadImageMask")
PLACEHOLDER_SENTINEL = "__gw_placeholder__"          # fixed-binding value → upload + use an 8×8 image
UPLOAD_SENTINEL = "__gw_upload__"                    # fixed-binding value → use the playground upload


def is_img_loader_class(cls) -> bool:
    """Image-loader detection: the known core classes plus any class whose name
    carries 'LoadImage' — custom node packs bring their own loaders (Trellis2's
    Trellis2LoadImageWithTransparency) that must behave like LoadImage: upload
    slot in playground/mapping, placeholder/required empty handling."""
    return cls in _IMG_LOADER_CLASSES or "loadimage" in str(cls or "").lower()


def is_image_field(wf: dict, node: str) -> bool:
    """True if a mapped node is an image-loader (its request field takes an uploaded
    image, not a scalar). Drives the per-field file inputs in the playground."""
    return is_img_loader_class((wf or {}).get(node, {}).get("class_type"))


def image_params(wf: dict, mapping: dict) -> list:
    """Request params whose target node is an image loader → rendered as uploads and
    filled per-field (uploaded file, else an 8×8 placeholder)."""
    return [p for p, m in (mapping or {}).items() if is_image_field(wf, (m or {}).get("node"))]


# Workflow fields that take a path to a client file. A mapped node is a plain
# Primitive/string node either way, so the workflow itself never says "this is a
# file" — the NAME is the only signal there is.
_FILE_FIELDS = ("file_path", "mesh_path", "path", "filename", "file")


def is_file_param(param: str, m: Optional[dict]) -> bool:
    """A mapped NON-image param that takes a client file (a mesh): its public name or the
    workflow field says so. Heuristic on purpose — the mapping has no kind flag, and every
    rig/shrink alias in the wild names it `input_mesh_path`. Never true for image loaders
    (the caller checks is_image_field first)."""
    names = {(param or "").lower(), ((m or {}).get("label") or "").lower()}
    if any(("path" in n or "mesh" in n or n.endswith("_file")) for n in names if n):
        return True
    return ((m or {}).get("field") or "").lower() in _FILE_FIELDS


def file_params(wf: dict, mapping: dict) -> list:
    """Mapped params rendered as file uploads: not an image loader, and is_file_param.
    Deliberately NOT part of public_fields' `images` list — that schema means images
    (the API carries these under `files`, with no placeholder/empty-mode semantics)."""
    return [p for p, m in (mapping or {}).items()
            if not is_image_field(wf, (m or {}).get("node")) and is_file_param(p, m)]


def slot_empty_mode(m: dict) -> str:
    """What a mapped image slot does when the request sends no image for it:
    'placeholder' (8×8 black, the default) · 'required' (no fallback → ComfyUI errors
    if it is missing) · 'disable' (drop the loader node, any link to it, and the dead
    branch behind it — every node that REQUIRED that input, see _prune_branch — so an
    OPTIONAL downstream input runs without this image). Reads the `on_empty` field,
    falling back to the legacy `no_placeholder` boolean."""
    mode = (m or {}).get("on_empty")
    if mode in ("placeholder", "required", "disable"):
        return mode
    return "required" if (m or {}).get("no_placeholder") else "placeholder"


def public_fields(cand: dict) -> tuple[list, list, list]:
    """The public request fields of a generation alias candidate — what the schema
    endpoint advertises, the playground renders and the OpenAI shims map reference
    images onto. ONE seam for both candidate kinds: a ComfyUI candidate derives them
    from workflow + mapping (labels are the external names), a cloud candidate from
    its fixed label table (<kind>.public_fields).

    Three lists: params (scalars), images (loader slots) and files (uploads that are
    not images — a mesh). A ComfyUI file param ALSO stays in `params`: there it is the
    backend-side path, which is the second, upload-free way to name the same input."""
    k = cloud_kind(cand)
    if k:
        return cloud_module(k).public_fields(cand)
    wf = cand.get("workflow_json") or {}
    mapping = cand.get("mapping") or {}
    files = [{"name": ((mapping.get(p) or {}).get("label") or "").strip() or p, "required": False}
             for p in file_params(wf, mapping)]
    params, images = [], []
    for p, m in mapping.items():
        m = m or {}
        name = (m.get("label") or "").strip() or p
        node, fld = m.get("node"), m.get("field")
        if is_image_field(wf, node):
            mode = slot_empty_mode(m)
            images.append({"name": name, "on_empty": mode, "required": mode == "required"})
            continue
        cur = (wf.get(node, {}).get("inputs") or {}).get(fld)
        if isinstance(cur, list):                # linked to another node — no scalar default
            cur = None
        typ = ("bool" if isinstance(cur, bool) else
               "int" if isinstance(cur, int) else
               "float" if isinstance(cur, float) else "string")
        entry: dict = {"name": name, "type": typ}
        # Advertise ONLY the LABEL (`name`). The internal param key can be node-based
        # (`value_307`) and changes when the workflow is rebuilt, so it must never be a
        # public field — the adapter still accepts it on input, but clients bind to the
        # stable label. Set a label on every param for a fully node-id-free schema.
        if cur is not None:
            entry["default"] = cur
        if name == "seed" or (p == "seed" and name == p):
            entry["auto"] = "random unless sent"
        params.append(entry)
    return params, images, files


def slot_empty_bypass(m: dict) -> list:
    """Extra node ids to BYPASS when this slot's image is missing — the `on_empty_bypass`
    companion to `on_empty: disable`. Pruning the loader only kills what REQUIRES the
    image; a node in the main path that merely passes something through (an apply/switch
    node whose image socket is optional) survives the cascade and would still run on
    nothing. Listing it here removes it ComfyUI mode-4 style instead: its consumers are
    rewired to its same-typed input, so the main path stays connected.

    Bypass, not prune, on purpose — pruning such a node would cut the path behind it.
    The ids join the backend's own `bypass` list, so ONE `_apply_bypass` pass handles
    both and the job summary reports them together under `bypassed`.

    Only meaningful for `on_empty: disable`; any other mode returns []."""
    if slot_empty_mode(m) != "disable":
        return []
    ids = (m or {}).get("on_empty_bypass") or []
    if isinstance(ids, str):                       # tolerate a comma/space separated string
        ids = re.split(r"[,\s]+", ids)
    return list(dict.fromkeys(s for s in (str(x).strip() for x in ids) if s))


def _prune_branch(wf: dict, nid: str, node_types: dict) -> list:
    """Deactivate a node AND the dead branch it leaves behind. Removes `nid`, drops every
    input link pointing at it (`[nid, slot]`), and CASCADES onto any consumer that thereby
    lost a REQUIRED input — ComfyUI aborts the whole prompt on a missing required input,
    so such a consumer cannot run either and is part of the same dead branch. A consumer
    that only lost an OPTIONAL socket stays: running without that image is the entire
    point of `on_empty: disable`.

    Measured on img2mesh-trellis2_multiview (2026-08-30): dropping the `back` loader
    leaves its Trellis2PreProcessImage with no `image` — required, so it dies too — while
    the generator's `back_image` is optional and the mesh is built from the other views.
    Pruning only the loader (the pre-cascade behaviour) failed the job at /prompt.

    `node_types` is the /object_info-derived map (`req` per class). A class missing from
    it stops the cascade there — the old single-node behaviour, never a guess about what
    a node needs. Returns the ids removed, in removal order."""
    removed: list = []
    queue = [str(nid)]
    while queue:
        cur = queue.pop(0)
        if wf.pop(cur, None) is None:            # absent, or already taken by this cascade
            continue
        removed.append(cur)
        for cid, n in list(wf.items()):
            inp = n.get("inputs")
            if not isinstance(inp, dict):
                continue
            lost = [k for k, v in inp.items()
                    if isinstance(v, list) and v and str(v[0]) == cur]
            if not lost:
                continue
            for fld in lost:
                inp.pop(fld, None)
            req = (node_types.get(n.get("class_type")) or {}).get("req")
            if req is None:
                logger.warning(f"node {cid} ({n.get('class_type')}) lost input(s) "
                               f"{lost} to a disabled node, but /object_info for that "
                               "class is unavailable — not cascading")
                continue
            if any(fld in req for fld in lost):
                queue.append(cid)
    return removed


def _is_link(v) -> bool:
    """An input value that is a wired link `[src_node_id, src_slot]` (vs a literal)."""
    return isinstance(v, list) and len(v) == 2 and not isinstance(v[0], (list, dict))


def _bypass_passthrough(wf: dict, nid: str, out_slot: int, node_types: dict):
    """The link `[src, slot]` that ComfyUI mode-4 passes through a bypassed node for its
    output slot `out_slot`, or None (→ consumer disconnected). Rule: the out_slot is the
    k-th output of its type T; it maps to the k-th LINK-valued input of type T (both in
    the node's declared socket order). Fallback when the class has no known types: a node
    with exactly one link input passes it through type-free; otherwise None."""
    node = wf.get(nid) or {}
    inputs = node.get("inputs") or {}
    types = node_types.get(node.get("class_type"))
    link_fields = [f for f, v in inputs.items() if _is_link(v)]
    if not types:                                    # unknown class → single-link passthrough only
        return inputs[link_fields[0]] if len(link_fields) == 1 else None
    outs = types.get("out") or []
    if out_slot >= len(outs):
        return None
    T = outs[out_slot]
    if not T:
        return None
    k = sum(1 for s in range(out_slot) if outs[s] == T)   # this output's index among type-T outputs
    same_type_link_inputs = [f for f in types.get("in", {})
                             if types["in"][f] == T and f in inputs and _is_link(inputs[f])]
    return inputs[same_type_link_inputs[k]] if k < len(same_type_link_inputs) else None


def _apply_bypass(wf: dict, bypass_ids: list, node_types: dict) -> list:
    """Bypass nodes (ComfyUI mode 4): remove each and rewire its consumers to the input
    that passes through (type-matched), following chains of bypassed nodes. Stale/absent
    ids are skipped. An input that resolves to nothing is dropped (the consuming socket
    must be optional, else ComfyUI errors — bypass does NOT cascade the way _prune_branch
    does: a bypassed node is deliberately transparent, not dead). Returns the
    ids actually applied. Pure: `node_types` is injected (adapter owns the cache/fetch)."""
    byp = [n for n in (str(x) for x in bypass_ids) if n in wf]
    if not byp:
        return []
    byp_set = set(byp)

    def resolve(nid: str, slot: int, seen: frozenset):
        if nid not in byp_set:
            return [nid, slot]
        if (nid, slot) in seen:                      # bypass cycle → give up (disconnect)
            return None
        src = _bypass_passthrough(wf, nid, slot, node_types)
        if not _is_link(src):
            return None
        return resolve(str(src[0]), src[1], seen | {(nid, slot)})

    for mid, node in wf.items():
        if mid in byp_set:
            continue
        inp = node.get("inputs")
        if not isinstance(inp, dict):
            continue
        for fld in [k for k, v in inp.items() if _is_link(v) and str(v[0]) in byp_set]:
            r = resolve(str(inp[fld][0]), inp[fld][1], frozenset())
            if r is None:
                inp.pop(fld, None)                   # unresolved → drop (ComfyUI parity)
            else:
                inp[fld] = r
    for nid in byp:
        wf.pop(nid, None)
    return byp


def upload_prefix_for(prefix: Optional[str]) -> str:
    """The per-JOB namespace every input upload of one request is named under.

    A blank prefix (playground/mapping probes, any caller without a job) gets a
    random time-based one — there is deliberately NO shared fallback name: input
    files are read by ComfyUI when the prompt EXECUTES, not when it is submitted,
    so any name two jobs can both write is a data-corruption window (measured: a
    client job delivered another subject's mesh)."""
    p = "".join(ch if ch.isalnum() else "_" for ch in (prefix or "").strip())[:60]
    return p or f"gw_{int(time.time() * 1000):x}_{random.randint(0, 0xFFFFFF):06x}"


def upload_slot_name(prefix: str, param: str, ext: str = "png") -> str:
    """`<job prefix>_<param>.<ext>` — the input filename ONE upload slot of ONE job
    uses. Job-unique by construction, so concurrent jobs (same alias or not, same
    backend or not) never share input state."""
    keep = "".join(ch if ch.isalnum() else "_" for ch in (param or "img"))[:40] or "img"
    e = "".join(ch for ch in (ext or "").lstrip(".").lower() if ch.isalnum()) or "bin"
    return f"{prefix}_{keep}.{e}"


def _placeholder_png(size: int = 8) -> bytes:
    """A tiny valid gray PNG, so an admin can pin an unused reference-image node to
    a real (but empty) image instead of letting ComfyUI choke on an empty input."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x7f\x7f\x7f" * size for _ in range(size))   # filter byte + RGB row
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


_PLACEHOLDER_PNG = _placeholder_png(8)
# The ONE input filename that is deliberately shared across jobs, aliases and clients:
# its content is a CONSTANT 8×8 grey PNG, so two jobs overwriting each other's copy
# cannot mix any client data up — the file is byte-identical either way. Every other
# upload name carries the job prefix (see upload_slot_name).
_PLACEHOLDER_NAME = "gw_placeholder.png"


def _coerce(value, current):
    """Coerce a string form value to the type of the field it overrides, so a
    pinned Switch boolean / int stays the right JSON type for ComfyUI."""
    if isinstance(current, bool):
        return str(value).strip().lower() in ("true", "1", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if isinstance(current, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def _apply_fixed(wf: dict, fixed: list) -> dict:
    """Apply admin-pinned constant bindings: each `{node, field, value}` sets
    `workflow[node].inputs[field] = value` (type-coerced to the field's current
    type). This is the general mechanism for pre-setting nodes the admin chose —
    model/clip/vae loader slots, a Switch's boolean, a reference image, etc. — so
    a registered workflow runs without the gateway guessing."""
    applied = {}
    for b in fixed or []:
        node, fieldn, value = b.get("node"), b.get("field"), b.get("value")
        if node in wf and fieldn and value is not None and value != "":
            inputs = wf[node].setdefault("inputs", {})
            inputs[fieldn] = _coerce(value, inputs.get(fieldn))
            applied[f"{node}.{fieldn}"] = inputs[fieldn]
    return applied


def _apply_mapping(wf: dict, mapping: dict, values: dict, protected: Optional[set] = None) -> dict:
    """The runtime injector: dumb and convention-free. For each logical param
    bound to `{node, field}`, set `workflow[node].inputs[field] = values[param]`.

    `mapping` is an explicit per-workflow binding table (authored in config or,
    later, in the registration UI). `values` is the merged logical params
    (prompt, negative_prompt, width, height, steps, cfg, seed, sampler, model, …
    plus any `params.extra`). Missing values are skipped. Returns what was set.

    `protected` is the set of admin-pinned `(node, field)` pairs: a request param
    that maps onto a pinned node/field is **ignored** — a pin is authoritative and
    the API must not override it."""
    protected = protected or set()
    applied = {}
    for param, binding in mapping.items():
        if values.get(param) is None:
            continue
        node, field = binding.get("node"), binding.get("field")
        if node in wf and field and (node, field) not in protected:
            wf[node].setdefault("inputs", {})[field] = values[param]
            applied[param] = values[param]
    return applied


def _format_comfy_error(messages) -> str:
    """Readable one-liner from ComfyUI /history error `messages` (a list of
    [type, data] pairs): the failing node + exception, else a trimmed raw fallback."""
    for m in (messages or []):
        if isinstance(m, (list, tuple)) and len(m) == 2 and m[0] == "execution_error":
            d = m[1] or {}
            where = " ".join(str(x) for x in (d.get("node_id"), d.get("node_type")) if x)
            msg = d.get("exception_message") or "execution error"
            return f"node {where}: {msg}" if where else msg
    try:
        return json.dumps(messages)[:600]
    except Exception:
        return str(messages)[:600]


def _prompt_in_queue(queue: dict, prompt_id: str) -> bool:
    """Is `prompt_id` still running or pending in a ComfyUI /queue response? Each entry
    is [number, prompt_id, prompt, extra, …]; the id sits at index 1."""
    for key in ("queue_running", "queue_pending"):
        for item in (queue.get(key) or []):
            if isinstance(item, (list, tuple)) and len(item) > 1 and item[1] == prompt_id:
                return True
    return False


_LORA_SLOT_RE = re.compile(r"^lora_0*(\d+)$")          # rgthree stack: lora_01..lora_NN
_STR_SLOT_RE = re.compile(r"^strength_0*(\d+)$")
# standalone high/low token in a LoRA filename — '-HIGH.', '_High_', '-low-' …
# but NOT substrings inside words ('Thigh Gap' carries no token).
_LORA_TOKEN_RE = re.compile(r"(?i)(?<![a-z0-9])(high|low)(?![a-z0-9])")


def lora_counterpart(name: str) -> Optional[str]:
    """The other half of a high/low LoRA pair: the standalone high/low token in
    the filename swapped case-preservingly (…-HIGH.safetensors ↔ …-LOW.…,
    …-High-i2v_e70 ↔ …-Low-i2v_e70). None when the name carries no token."""
    def swap(m):
        t = m.group(1)
        other = {"high": "low", "low": "high"}[t.lower()]
        return other.upper() if t.isupper() else other.capitalize() if t[0].isupper() else other
    new, n = _LORA_TOKEN_RE.subn(swap, name or "")
    return new if n else None


def _lora_kind(name: str) -> Optional[str]:
    m = _LORA_TOKEN_RE.search(name or "")
    return m.group(1).lower() if m else None


def lora_groups(wf: dict, mapping: dict) -> list:
    """The workflow's LoRA stack nodes as placement groups [(node_id, kind)].
    kind 'high'/'low'/None comes from the node title (Wan: input_lora_high/_low),
    else from a mapping label bound to that node ('lora1_high'), else None."""
    groups = []
    for nid, n in wf.items():
        inp = n.get("inputs") or {}
        if not any(_LORA_SLOT_RE.match(f) for f in inp):
            continue
        kind = _lora_kind(((n.get("_meta") or {}).get("title") or ""))
        if kind is None:
            for p, m in (mapping or {}).items():
                if str((m or {}).get("node")) == str(nid) and _LORA_SLOT_RE.match(str((m or {}).get("field") or "")):
                    kind = _lora_kind(((m or {}).get("label") or "")) or _lora_kind(p)
                    if kind:
                        break
        groups.append((nid, kind))
    return sorted(groups, key=lambda g: (len(g[0]), g[0]))


def _free_stack_slots(wf: dict, nid: str) -> list:
    """(lora_field, strength_field) pairs currently empty/'None' on one stack node."""
    inp = wf.get(nid, {}).get("inputs") or {}
    out = []
    for f in sorted((f for f in inp if _LORA_SLOT_RE.match(f)),
                    key=lambda f: int(_LORA_SLOT_RE.match(f).group(1))):
        if inp.get(f) in (None, "", "None"):
            idx = int(_LORA_SLOT_RE.match(f).group(1))
            sf = next((s for s in (f"strength_{idx:02d}", f"strength_{idx}") if s in inp), None)
            out.append((f, sf))
    return out


def _apply_lora_list(wf: dict, mapping: dict, loras: list, avail: set) -> list:
    """The `loras: [{name, strength}]` request shape: place each logical LoRA into
    EVERY stack group, resolving high/low pairs server-side — the client sends ONE
    name, the group whose kind differs gets the token-swapped counterpart (checked
    against the backend's installed set). Clients never learn the pairing. Raises
    RuntimeError (→ job fails with the message) when a name/counterpart is not
    installed or the stacks run out of free slots. Returns [(node, field, value)]."""
    if not loras:
        return []
    groups = lora_groups(wf, mapping)
    if not groups:
        raise RuntimeError("this workflow has no LoRA stack")
    free = {nid: _free_stack_slots(wf, nid) for nid, _ in groups}
    placed = []
    for entry in loras:
        if isinstance(entry, str):
            entry = {"name": entry}
        name = str((entry or {}).get("name") or "").strip()
        if not name or name == "None":
            continue
        strength = (entry or {}).get("strength", 1.0)
        base_kind = _lora_kind(name)
        counterpart = lora_counterpart(name)
        for nid, kind in groups:
            if kind is None or base_kind is None or kind == base_kind:
                val = name                       # kindless file goes into every group as-is
            else:
                val = counterpart
                if val is None or (avail and val not in avail):
                    raise RuntimeError(f"LoRA pair incomplete: '{name}' needs a {kind} "
                                       f"counterpart ('{counterpart or '?'}' is not installed)")
            if avail and val not in avail:
                raise RuntimeError(f"LoRA '{val}' is not installed on this backend")
            slots = free.get(nid) or []
            if not slots:
                raise RuntimeError(f"no free LoRA slot left on stack node {nid}")
            lf, sf = slots.pop(0)
            inp = wf[nid].setdefault("inputs", {})
            inp[lf] = val
            if sf:
                try:
                    inp[sf] = float(strength)
                except (TypeError, ValueError):
                    inp[sf] = 1.0
            placed.append((nid, lf, val))
    return placed


def _apply_lora_cascade(wf: dict, values: dict) -> list:
    """Place client-supplied LoRAs into the next FREE stack slots, never overwriting an
    occupied (pinned/baked) one.

    A client numbers its loras from 1 and cannot know which physical slot is reserved,
    so each client `lora_N` (with optional `strength_N`) is dropped onto the next slot
    whose current value is empty/'None'; occupied slots are skipped. Consumes the
    matched `lora_*`/`strength_*` keys from `values` so the normal mapping doesn't also
    place them at fixed slots. Returns [(node, field, value)] placed."""
    client, cstr = [], {}
    for k in [k for k in values if _LORA_SLOT_RE.match(k)]:
        idx, val = int(_LORA_SLOT_RE.match(k).group(1)), values.pop(k)
        if val is not None and str(val) not in ("", "None"):
            client.append((idx, val))
    for k in [k for k in values if _STR_SLOT_RE.match(k)]:
        cstr[int(_STR_SLOT_RE.match(k).group(1))] = values.pop(k)
    if not client:
        return []
    client.sort(key=lambda t: t[0])
    free = []                                          # (node, lora_field, strength_field) currently empty
    for nid, n in wf.items():
        inp = n.get("inputs") or {}
        for f in sorted((f for f in inp if _LORA_SLOT_RE.match(f)),
                        key=lambda f: int(_LORA_SLOT_RE.match(f).group(1))):
            if inp.get(f) in (None, "", "None"):
                idx = int(_LORA_SLOT_RE.match(f).group(1))
                sf = next((s for s in (f"strength_{idx:02d}", f"strength_{idx}") if s in inp), None)
                free.append((nid, f, sf))
    placed = []
    for (cidx, val), (nid, lf, sf) in zip(client, free):
        inp = wf[nid].setdefault("inputs", {})
        inp[lf] = val
        if sf:
            inp[sf] = cstr.get(cidx, 1.0)
        placed.append((nid, lf, val))
    return placed


def suggest_mapping(wf: dict) -> dict:
    """Convention-based *suggestion* of a {param: {node, field}} binding table —
    the seed for a "register workflow" UI to pre-fill, NOT the runtime mechanism.
    Also used as a zero-config fallback when a candidate omits an explicit mapping.

    Heuristics mirror common ComfyUI templates (named `input_*` Primitive nodes
    by `_meta.title`, with class-based fallbacks). Anything it gets wrong, the
    explicit mapping overrides."""
    m: dict = {}

    def _prompt_binding(titles, want):
        for title in titles:
            nid = _node_by_title(wf, title)
            if nid is not None:
                inputs = wf[nid].get("inputs", {})
                field = "value" if "value" in inputs else ("prompt" if "prompt" in inputs else "text")
                return {"node": nid, "field": field}
        nid = _clip_nodes(wf, want)
        return {"node": nid, "field": "text"} if nid else None

    if (b := _prompt_binding(["input - prompt - positive", "input_prompt_positiv"], "positive")):
        m["prompt"] = b
    if (b := _prompt_binding(["input - prompt - negative", "input_prompt_negativ"], "negative")):
        m["negative_prompt"] = b

    for which in ("width", "height"):
        nid = _node_by_title(wf, f"input_{which}")
        if nid is not None:
            m[which] = {"node": nid, "field": "value"}
        else:
            nid = _node_by_class(wf, "EmptyLatentImage")
            if nid is not None:
                m[which] = {"node": nid, "field": which}

    seed_node = _node_by_title(wf, "input_seed")
    if seed_node is not None:
        m["seed"] = {"node": seed_node, "field": "value"}
    elif (ks := _node_by_class(wf, "KSampler")) is not None:
        m["seed"] = {"node": ks, "field": "seed"}
    elif (rn := _node_by_class(wf, "RandomNoise")) is not None:
        m["seed"] = {"node": rn, "field": "noise_seed"}

    if (ks := _node_by_class(wf, "KSampler")) is not None:
        for param, field in (("steps", "steps"), ("cfg", "cfg"),
                             ("sampler", "sampler_name"), ("scheduler", "scheduler")):
            m[param] = {"node": ks, "field": field}

    # Every REMAINING node titled `input_<name>` is a declared bind point (the mesh
    # workflows follow this strictly — see sample_comfyui_workflows/README.md): the
    # title IS the param's public name, so bind it under the title AND set it as the
    # label — labels are the stable cross-stage names a chain threads params by
    # (raw `value` collides between stages). Nodes already bound above (prompt /
    # width / seed / …) keep their conventional param name; multi-field nodes (a
    # LoRA stack) are skipped — those are placed by _apply_lora_*.
    bound = {(b or {}).get("node") for b in m.values()}
    for nid, node in sorted(wf.items(), key=lambda kv: (len(kv[0]), kv[0])):
        title = ((node.get("_meta") or {}).get("title") or "").strip()
        if not title.lower().startswith("input_") or nid in bound or title in m:
            continue
        scalars = [f for f, v in (node.get("inputs") or {}).items() if not isinstance(v, list)]
        field = ("value" if "value" in scalars else "image" if "image" in scalars else
                 scalars[0] if len(scalars) == 1 else None)
        if field and not field.endswith("_name"):   # *_name = model/file picker → pinned, see below
            m[title] = {"node": nid, "field": field, "label": title}

    # Note: the model loader is NOT mapped as a request param here — it's an
    # admin-fixed binding (detected separately) / Pinned values dropdown, since the
    # model is a per-alias choice, not something varied per request.
    return m


def _comfy_models(object_info: dict) -> set[str]:
    """Best-effort: pull available checkpoint/unet filenames out of /object_info
    so routing can validate a workflow's model binding exists on the backend."""
    out: set[str] = set()
    for cls, fieldn in (("CheckpointLoaderSimple", "ckpt_name"), ("UNETLoader", "unet_name"),
                        ("UnetLoaderGGUF", "unet_name"), ("VAELoader", "vae_name")):
        spec = (object_info.get(cls, {}).get("input", {}).get("required", {}).get(fieldn))
        if isinstance(spec, list) and spec and isinstance(spec[0], list):
            out.update(str(x) for x in spec[0])
    return out


def _comfy_loras(object_info: dict) -> set[str]:
    """Installed LoRA filenames from /object_info — the combo options of any LoRA
    loader's `lora_name` / `lora_NN` field ('None' sentinel excluded)."""
    out: set[str] = set()
    for cls, spec in (object_info or {}).items():
        if "lora" not in str(cls).lower():
            continue
        inp = spec.get("input", {}) or {}
        for fn, fspec in {**(inp.get("required") or {}), **(inp.get("optional") or {})}.items():
            if not (fn == "lora_name" or _LORA_SLOT_RE.match(fn)):
                continue
            if isinstance(fspec, list) and fspec and isinstance(fspec[0], list):
                out.update(str(x) for x in fspec[0] if x and x != "None")
    return out


def _node_type_entry(info: dict) -> dict:
    """Reduce ONE /object_info class entry to
    `{"out":[type|None,…], "in":{field:type}, "req":[field,…]}`.
    Only string type tokens (data links: IMAGE/LATENT/MODEL/…) are kept in `in` —
    combo/enum specs (a list, or "COMBO") are never wired links, so they carry no
    passthrough type. Input order is required-then-optional = the node's socket order
    (dict order preserved), which the bypass rewiring relies on for the k-th-same-type
    pairing. `req` names EVERY field of the required section, typed or not: it answers
    "does losing this input kill the node?" for _prune_branch, and a required combo
    can be link-fed too."""
    outs = [t if isinstance(t, str) else None for t in (info.get("output") or [])]
    ins: dict = {}
    req: list = []
    inp = info.get("input") or {}
    for section in ("required", "optional"):
        for field_name, spec in (inp.get(section) or {}).items():
            if section == "required":
                req.append(field_name)
            t = spec[0] if isinstance(spec, list) and spec and isinstance(spec[0], str) else None
            if t and t != "COMBO":
                ins[field_name] = t
    return {"out": outs, "in": ins, "req": req}


def _comfy_node_types(object_info: dict) -> dict:
    """`{class: {"out":[type],"in":{field:type}}}` for every class in /object_info —
    the slot-type map the bypass rewiring needs. Built once at discovery from the same
    fetch as _comfy_models/_comfy_loras (no extra request)."""
    return {cls: _node_type_entry(info) for cls, info in (object_info or {}).items()
            if isinstance(info, dict)}


def _gen_values(req: NormalizedRequest) -> dict:
    """Flatten a request's logical generation values into one {param: value} dict
    the mapping can draw from: inputs (prompt/negative), params (width/steps/…),
    `params.extra` (workflow-specific knobs), and the resolved model."""
    values = {k: v for k, v in req.inputs.items() if v is not None}
    params = dict(req.params)
    extra = params.pop("extra", None) or {}
    values.update({k: v for k, v in params.items() if v is not None})
    values.update(extra)
    if req.real_model:
        values["model"] = req.real_model
    return values


def comfy_input_dir(backend: dict) -> str:
    """The absolute path of a ComfyUI backend's input directory (as the ComfyUI
    process sees it): explicit `comfy_input_dir`, else derived as the sibling of
    `comfy_output_dir` (…/output → …/input — ComfyUI's standard layout).
    '' when neither is known."""
    d = (backend.get("comfy_input_dir") or "").rstrip("/")
    if d:
        return d
    out = (backend.get("comfy_output_dir") or "").rstrip("/")
    if out.endswith("/output"):
        return out[:-len("output")] + "input"
    return ""


def input_path_ref(backend: dict, stored: str) -> str:
    """What a mesh param receives after an upload hand-off: the uploaded file's
    ABSOLUTE path in that backend's input dir, so the workflow loads it exactly like
    a shared-disk path (no load-from-input node needed). Falls back to the bare
    stored name when no input dir is known — only a load-from-input node can
    resolve that."""
    indir = comfy_input_dir(backend)
    return f"{indir}/{stored}" if indir else stored


class ComfyUIAdapter(BackendAdapter):
    """ComfyUI image/video/audio backend. discover() via /object_info; generate()
    submits a parametrized workflow and polls /history, then fetches /view."""

    type = "comfyui"
    serves_generation = True

    def __init__(self, backend: dict, ctx: AdapterContext):
        super().__init__(backend, ctx)
        self._node_types: dict = {}      # {class: {"out":[type],"in":{field:type}}} — bypass slot types
        # executor watchdog (discover-driven): pending head seen while nothing ran
        self._stuck_head: Optional[str] = None
        self._stuck_since: float = 0.0
        self._stuck_checks: int = 0
        self.exec_stuck: bool = False
        self.last_restart: float = 0.0        # ts of the last restart() call (cooldown)
        self.last_restart_result: str = ""    # "" | running | ok | timeout | no-manager

    async def discover(self, client: httpx.AsyncClient) -> Capabilities:
        url = self.backend["url"].rstrip("/")
        resp = await client.get(f"{url}/object_info", timeout=_COMFY_DISCOVERY_TIMEOUT)
        resp.raise_for_status()
        oi = resp.json()
        self._node_types = _comfy_node_types(oi)      # cache slot types for bypass (free — same fetch)
        caps = Capabilities(models=_comfy_models(oi), loras=_comfy_loras(oi), pricing={})
        qr = await client.get(f"{url}/queue", timeout=_COMFY_DISCOVERY_TIMEOUT)
        qr.raise_for_status()
        self._check_executor(qr.json())               # raises ComfyExecutorStuck → DOWN path
        return caps

    def _check_executor(self, queue: dict) -> None:
        """Watchdog: prompts waiting while nothing runs, same head prompt across
        ≥2 consecutive checks AND ≥stuck_after_s → ComfyExecutorStuck. A transient
        idle moment between dequeues changes the head / empties pending → reset."""
        pending = queue.get("queue_pending") or []
        running = queue.get("queue_running") or []
        head = None
        if pending and not running:
            entry = pending[0]
            head = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else str(entry)
        if head is None or head != self._stuck_head:
            self._stuck_head = head               # None (healthy) or new tracking baseline
            self._stuck_since = time.time()
            self._stuck_checks = 0
            self.exec_stuck = False
            return
        self._stuck_checks += 1                   # same head again, still nothing running
        after = float(self.backend.get("stuck_after_s") or _COMFY_STUCK_AFTER_S)
        if time.time() - self._stuck_since >= after:
            self.exec_stuck = True
            raise ComfyExecutorStuck(
                f"executor stuck: {len(pending)} prompt(s) pending, none running for "
                f"{int(time.time() - self._stuck_since)}s (head {head})")

    async def cancel(self) -> None:
        """POST /interrupt — frees the GPU of the running prompt (main.cancel_generation)."""
        try:
            async with _pooled_client(self.ctx) as client:
                await client.post(f"{self.backend['url'].rstrip('/')}/interrupt", timeout=5.0)
        except Exception:
            pass

    async def restart(self) -> str:
        """Restart the ComfyUI service via the ComfyUI-Manager reboot endpoint.
        The process kills itself mid-response, so a transport error on the POST is
        the EXPECTED success signal; systemd (Restart=always) brings it back. A 404
        means the Manager extension is missing → RuntimeError, no wait loop."""
        url = self.backend["url"].rstrip("/")
        self.last_restart = time.time()
        self.last_restart_result = "running"
        try:
            async with _pooled_client(self.ctx) as client:
                r = await client.post(f"{url}/manager/reboot", timeout=5.0)
            if r.status_code == 404:
                self.last_restart_result = "no-manager"
                raise RuntimeError("ComfyUI-Manager not installed (/manager/reboot → 404)")
        except (httpx.TransportError, httpx.TimeoutException):
            pass                                   # process died mid-response — expected
        deadline = time.time() + _COMFY_RESTART_WAIT_S
        while time.time() < deadline:
            await asyncio.sleep(3.0)
            try:
                async with _pooled_client(self.ctx) as client:
                    r = await client.get(f"{url}/object_info", timeout=_COMFY_DISCOVERY_TIMEOUT)
                if r.status_code == 200:
                    self._stuck_head, self._stuck_checks = None, 0
                    self.exec_stuck = False
                    self.last_restart_result = "ok"
                    logger.info(f"[{self.name}] ComfyUI back up after restart")
                    return "ok"
            except Exception:
                pass                               # still rebooting
        self.last_restart_result = "timeout"
        logger.warning(f"[{self.name}] ComfyUI not back {_COMFY_RESTART_WAIT_S:.0f}s after restart")
        return "timeout"

    async def _node_types_for(self, wf: dict, bypass_ids: list) -> dict:
        """Slot types for the classes of the nodes about to be bypassed. Uses the
        discovery cache; lazily fetches per-class /object_info/{class} for any class the
        cache lacks (a generate() before the first discovery). A failed fetch just leaves
        that class absent → _apply_bypass falls back to its single-link heuristic."""
        need = {(wf.get(str(n)) or {}).get("class_type") for n in bypass_ids}
        need = {c for c in need if c and c not in self._node_types}
        if not need:
            return self._node_types
        url = self.backend["url"].rstrip("/")
        async with _pooled_client(self.ctx) as c:
            for cls in need:
                try:
                    r = await c.get(f"{url}/object_info/{cls}", timeout=_COMFY_DISCOVERY_TIMEOUT)
                    if r.status_code == 200:
                        info = (r.json() or {}).get(cls)
                        if isinstance(info, dict):
                            self._node_types[cls] = _node_type_entry(info)
                except Exception as e:
                    logger.warning(f"[{self.name}] object_info for bypass class '{cls}' "
                                   f"unavailable ({type(e).__name__}) — single-link fallback")
        return self._node_types

    def _load_workflow(self, path: Optional[str]) -> dict:
        if not path:
            raise FileNotFoundError(f"{self.name}: no workflow configured for this alias")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{self.name}: workflow file not found: {path}")
        with open(path) as f:
            return json.load(f)

    def _workflow_for(self, req: NormalizedRequest) -> dict:
        """A fresh, mutable copy of the workflow to inject into — preferring the
        gateway-owned JSON (registered via the UI), else a file path (share)."""
        if req.workflow_json is not None:
            return copy.deepcopy(req.workflow_json)            # never mutate the stored one
        return self._load_workflow(req.workflow)

    async def _post_upload(self, client: httpx.AsyncClient, data: bytes, name: str,
                           content_type: str):
        """The one ComfyUI `/upload/image` POST (overwrite=true; ComfyUI has no
        delete-input API, so overwriting is also how we release the space again —
        see _cleanup_uploads). Returns the raw response; callers apply their own
        success/subfolder + error policy."""
        url = self.backend["url"].rstrip("/")
        return await client.post(f"{url}/upload/image",
                                 files={"image": (name, data, content_type)},
                                 data={"overwrite": "true"},
                                 timeout=_upload_timeout_for(len(data)))

    async def _upload_image(self, client: httpx.AsyncClient, data: bytes, name: str) -> str:
        """Upload one input image into this backend's ComfyUI input dir; returns the
        stored name. RAISES on failure, like upload_input(): a swallowed error used to
        leave the workflow pointing at a name whose file was never written — ComfyUI
        would then run against whatever bytes already sat there (another job's image).
        Failing the job is the only honest outcome."""
        r = await self._post_upload(client, data, name, "image/png")
        if r.status_code != 200:
            raise RuntimeError(f"image upload '{name}' to '{self.backend.get('name')}' failed "
                               f"(HTTP {r.status_code}: {r.text[:200]})")
        try:
            return (r.json() or {}).get("name", name)
        except Exception:
            return name                                  # 200 without JSON: the name we asked for

    async def _upload_placeholder(self, client: httpx.AsyncClient) -> str:
        """The 8×8 grey filler for empty image slots. Best-effort ON PURPOSE (the one
        exception to the rule above): the file is a shared CONSTANT, so a failed upload
        can only mean the already-present placeholder stays — never someone else's data."""
        try:
            return await self._upload_image(client, _PLACEHOLDER_PNG, _PLACEHOLDER_NAME)
        except Exception as e:
            logger.debug(f"[{self.name}] placeholder upload failed ({e}) — "
                         f"using '{_PLACEHOLDER_NAME}' as-is")
            return _PLACEHOLDER_NAME

    async def _cleanup_uploads(self, names: list) -> None:
        """Best-effort garbage control for job-unique input files: overwrite each with
        the 72-byte placeholder, so a finished job leaves a stub instead of a 40 MB mesh
        (ComfyUI has no delete-input API). Call ONLY after a clean success — after a
        timeout/interrupt the ComfyUI prompt may still be running and about to read the
        file. Never raises: cleanup failure is cosmetic, the job is already done."""
        if not names:
            return
        try:
            async with _pooled_client(self.ctx) as c:
                for n in dict.fromkeys(names):           # dedupe, keep order
                    try:
                        await self._post_upload(c, _PLACEHOLDER_PNG, n, "image/png")
                    except Exception as e:
                        logger.debug(f"[{self.name}] input cleanup '{n}' failed: {e}")
        except Exception as e:                           # client/pool trouble — still not the job's problem
            logger.debug(f"[{self.name}] input cleanup skipped: {e}")

    async def upload_input(self, data: bytes, name: str,
                           content_type: str = "application/octet-stream") -> str:
        """Upload an arbitrary file (a chain's intermediate mesh) into this backend's
        ComfyUI input dir and return the stored name. Lets a two-stage chain relay
        stage 1's mesh to a stage-2 backend that does NOT share disk — the successor
        workflow then loads it from input/ by this name (overwrite=true, one slot).
        Raises on failure (a lost mesh must fail the chain, not run stage 2 blind)."""
        async with _pooled_client(self.ctx) as c:
            r = await self._post_upload(c, data, name, content_type)
        if r.status_code != 200:
            raise RuntimeError(f"mesh upload to '{self.backend.get('name')}' failed "
                               f"(HTTP {r.status_code}: {r.text[:200]})")
        j = r.json() or {}
        sub = (j.get("subfolder") or "").strip("/")
        nm = j.get("name", name)
        return f"{sub}/{nm}" if sub else nm

    # ── Chain hand-off primitives (keep ComfyUI's filename/output conventions out of
    #    main._run_chain so chains stay adapter-agnostic; siblings of upload_input) ──
    @staticmethod
    def export_pin(node: str, prefix: str) -> dict:
        """The `fixed` pin that makes an export/save node write under a known prefix."""
        return {"node": node, "field": "filename_prefix", "value": prefix}

    @staticmethod
    def export_node_error(wf: dict, node: Optional[str]) -> Optional[str]:
        """Why `node` cannot serve as a chain's stage-1 export node — None if it can.

        The pin from `export_pin` is what makes stage 1 write under OUR filename, and
        `_apply_fixed` silently drops a binding whose `(node, field)` isn't there. So a
        node without a `filename_prefix` input keeps its own name, stage 1 runs to
        completion (tens of GPU-minutes for a mesh) and only the /view fetch afterwards
        reports 'produced no mesh' — pointing at the export node instead of at the whole
        wasted run. Node ids drift when a workflow is rebuilt (measured 2026-08-31: a
        multi-view Trellis2 alias inherited `export_node: 82` from the revision where 82
        WAS the export node; there it is the model loader), so name the node's real class
        and list the workflow's actual export candidates."""
        n = wf.get(node) if node else None
        if not isinstance(n, dict):
            return f"chain export node '{node}' is not in the mesh workflow"
        if "filename_prefix" in (n.get("inputs") or {}):
            return None
        cls = n.get("class_type") or "?"
        title = ((n.get("_meta") or {}).get("title") or "").strip()
        cands = [f"{nid} ({(nn.get('class_type') or '?')}"
                 + (f" '{t}'" if (t := ((nn.get("_meta") or {}).get("title") or "").strip()) else "")
                 + ")"
                 for nid, nn in sorted(wf.items(), key=lambda kv: (not str(kv[0]).isdigit(),
                                                                   int(kv[0]) if str(kv[0]).isdigit() else 0,
                                                                   str(kv[0])))
                 if isinstance(nn, dict) and "filename_prefix" in (nn.get("inputs") or {})]
        hint = ("; the workflow's export/save nodes are " + ", ".join(cands[:4])
                if cands else "; this workflow has no node with a 'filename_prefix' input")
        return (f"chain export node '{node}' is a {cls}"
                + (f" '{title}'" if title else "")
                + " with no 'filename_prefix' input — it cannot be pinned, so stage 1 "
                  "would keep its own output name" + hint)

    @staticmethod
    def pinned_output_name(prefix: str, ext: str) -> str:
        """The file an export node writes for `filename_prefix=prefix`: ComfyUI's %05d
        counter, and a fresh prefix always yields `_00001_`."""
        return f"{prefix}_00001_.{ext}"

    async def fetch_output(self, name: str, *, want_bytes: bool = True) -> Optional[bytes]:
        """Fetch a produced output file from ComfyUI /view by name. Returns its bytes
        (want_bytes), or b"" for an existence-only check (a cheap 1-byte Range GET — the
        caller only needs to know the file is there); None if absent (404); raises on any
        other status (a backend error must not read as 'file missing')."""
        url = self.backend["url"].rstrip("/")
        headers = None if want_bytes else {"Range": "bytes=0-0"}
        async with _pooled_client(self.ctx) as c:
            r = await c.get(f"{url}/view", params={"filename": name, "type": "output"},
                            headers=headers, timeout=30.0)
        if r.status_code in (200, 206):
            return r.content if want_bytes else b""
        if r.status_code == 404:
            return None
        raise RuntimeError(f"/view '{name}' → HTTP {r.status_code}")

    # ── the chain roles, built from the primitives above (main._run_chain calls these) ──
    def chain_export(self, cand: dict, succ: dict, params: dict, prefix: str) -> ChainExport:
        """Stage 1 on ComfyUI: pin the export node's `filename_prefix` and predict the
        name it writes. A node that cannot be pinned is refused HERE — `_apply_fixed`
        drops such a binding silently, so stage 1 would run to completion (tens of
        GPU-minutes) under its own name and only the /view fetch would notice."""
        wf = cand.get("workflow_json") or {}
        node = str(succ.get("export_node") or "").strip()
        why = self.export_node_error(wf, node)
        if why:
            return ChainExport("", error=why)
        # The extension is what ComfyUI will WRITE: the node's file_format as applied —
        # a mapped request param overrides the workflow value, an admin pin beats both.
        ext = str(((wf.get(node) or {}).get("inputs") or {}).get("file_format") or "glb")
        for p, m in (cand.get("mapping") or {}).items():
            m = m or {}
            if str(m.get("node")) == node and m.get("field") == "file_format":
                v = params.get(p)
                if v in (None, "") and (m.get("label") or "").strip():
                    v = params.get((m.get("label") or "").strip())
                if v not in (None, ""):
                    ext = str(v)
        for fx in (cand.get("fixed") or []):
            if (str(fx.get("node")) == node and fx.get("field") == "file_format"
                    and fx.get("value") not in (None, "")):
                ext = str(fx["value"])
        return ChainExport(self.pinned_output_name(prefix, ext), [self.export_pin(node, prefix)])

    async def chain_take_mesh(self, out: GenOutput, export: ChainExport, want_bytes: bool) -> Optional[bytes]:
        """Stage 1's mesh comes off the backend's disk, not out of the response: the
        export node wrote it under our pinned name (existence-only = cheap Range GET)."""
        return await self.fetch_output(export.mesh_name, want_bytes=want_bytes)

    async def chain_feed_mesh(self, req2, backend2, mesh_param, mesh_name, mesh_bytes, outdir) -> str:
        """Stage 2 on ComfyUI reads a FILE: either the shared-disk output path (path
        relay, no bytes travelled) or the mesh uploaded into this backend's input dir."""
        if mesh_bytes is None:                       # path relay: shared disk, absolute path
            return f"{outdir}/{mesh_name}"
        return input_path_ref(backend2, await self.upload_input(mesh_bytes, mesh_name))

    async def _resolve_image_sentinels(self, fixed: list, upload: Optional[bytes],
                                       prefix: str, used: list) -> list:
        """Replace image sentinels in fixed bindings with real uploaded names.
        A node pinned to the playground upload uses the uploaded image, or falls
        back to the 8×8 placeholder when nothing was uploaded — so the placeholder
        is simply the 'no reference' alternative right at the upload step."""
        has_ph = any(b.get("value") == PLACEHOLDER_SENTINEL for b in fixed)
        has_up = any(b.get("value") == UPLOAD_SENTINEL for b in fixed)
        if not (has_ph or has_up):
            return fixed
        need_ph = has_ph or (has_up and not upload)        # placeholder also backs an empty upload
        up_name = upload_slot_name(prefix, "upload")       # job-unique, never a shared 'gw_upload.png'
        async with _pooled_client(self.ctx) as c:
            ph = await self._upload_placeholder(c) if need_ph else None
            up = await self._upload_image(c, bytes(upload), up_name) if (has_up and upload) else None
        if up:
            used.append(up_name)
        def sub(b):
            v = b.get("value")
            if v == PLACEHOLDER_SENTINEL:
                return {**b, "value": ph} if ph else b
            if v == UPLOAD_SENTINEL:
                name = up or ph                             # uploaded image, else 8×8 placeholder
                return {**b, "value": name} if name else b
            return b
        return [sub(b) for b in fixed]

    async def _apply_image_params(self, wf: dict, mapping: dict, params: list,
                                  uploads: dict, prefix: str, used: list,
                                  node_types: dict, pruned: dict,
                                  extra_bypass: list) -> list:
        """Per request-field image params: upload that field's image (or the 8×8
        placeholder when none was provided) and set the loader node's field to the
        stored name. Each param gets its own JOB-UNIQUE input file (`<prefix>_<param>`),
        so several images coexist within a job and no concurrent job can overwrite one
        of them between /prompt and execution. `used` collects the names for the
        post-success cleanup; `pruned` collects `{param: [node ids]}` for the ones an
        `on_empty: disable` slot took out (job summary + the output-node check) and
        `extra_bypass` the same slot's `on_empty_bypass` ids, handed to the one
        `_apply_bypass` pass at the end of `generate` (see slot_empty_bypass)."""
        if not params:
            return []
        applied = []
        async with _pooled_client(self.ctx) as c:
            for p in params:
                m = mapping.get(p) or {}
                nid, fld = m.get("node"), m.get("field")
                if not (nid and fld):
                    continue
                data = uploads.get(p)
                if data:
                    slot = upload_slot_name(prefix, p)
                    name = await self._upload_image(c, bytes(data), slot)   # raises → job fails
                    used.append(slot)
                else:
                    mode = slot_empty_mode(m)
                    if mode == "required":
                        continue              # no 8×8 fallback (e.g. inpaint image/mask) →
                                              # keep the workflow's own value / error if empty
                    if mode == "disable":     # drop the loader AND whatever cannot run
                        gone = _prune_branch(wf, nid, node_types)   # without it (dead branch)
                        if gone:
                            pruned[p] = gone
                        # …plus the slot's opt-in extras, bypassed (not pruned) so a node
                        # in the main path is skipped without cutting the path behind it
                        extra_bypass.extend(slot_empty_bypass(m))
                        applied.append(p)
                        continue
                    name = await self._upload_placeholder(c)
                wf.setdefault(nid, {}).setdefault("inputs", {})[fld] = name
                applied.append(p)
        return applied

    async def _apply_file_params(self, wf: dict, mapping: dict, files: dict,
                                 protected: set, prefix: str, used: list) -> list:
        """Client-supplied non-image files (`files:{param: …}`, e.g. the mesh a shrink
        alias works on): upload each into THIS backend's input dir under a JOB-UNIQUE
        name (`<prefix>_<param>.<ext>`; the client's filename only contributes the
        extension) and inject the file's ABSOLUTE path into the mapped param. Runs per
        dispatch, so parking and failover simply re-upload onto the backend that
        actually runs the job — the client never deals with backend paths.

        Absolute, because the consumers are plain path loaders (`PrimitiveString`/
        `GeomPackLoadMeshPath`): a bare input name would not resolve, so a backend whose
        input dir is unknown fails the job instead of running it against a phantom path."""
        if not files:
            return []
        indir = comfy_input_dir(self.backend)
        if not indir:
            raise RuntimeError(f"backend '{self.name}' has no comfy_input_dir/comfy_output_dir "
                               "— cannot resolve uploaded file to an absolute path")
        applied = []
        for p, (name, data) in files.items():
            m = mapping.get(p) or {}                    # keyed by param — the endpoint already
            nid, fld = m.get("node"), m.get("field")    # resolved label→param against the mapping
            if not (nid and fld):
                raise RuntimeError(f"uploaded file for '{p}' has no mapping on this backend's "
                                   "workflow — cannot place it")
            if (nid, fld) in protected:                 # admin pin stays authoritative (repo-wide rule)
                logger.warning(f"[{self.name}] uploaded file for '{p}' ignored — "
                               f"node {nid}.{fld} is pinned by the alias")
                continue
            slot = upload_slot_name(prefix, p, os.path.splitext(name or "")[1] or "bin")
            stored = await self.upload_input(bytes(data), slot)   # raises → job fails with the reply
            used.append(slot)
            wf.setdefault(nid, {}).setdefault("inputs", {})[fld] = input_path_ref(self.backend, stored)
            applied.append(p)
        return applied

    async def _autofill_empty_images(self, wf: dict, mapping: dict) -> list:
        """Any image-loader node still left with an empty `image` (not a mapped
        request field) → fill it with the 8×8 placeholder, so ComfyUI never tries to
        open the input/ directory. Mapped image fields are handled per-field above.

        Slots a `required`/`disable` request-field maps to are never auto-filled:
        `required` is left empty so ComfyUI errors clearly, `disable` was already pruned."""
        skip = {(m or {}).get("node") for m in (mapping or {}).values()
                if slot_empty_mode(m) in ("required", "disable")}
        empty = [nid for nid, n in wf.items()
                 if is_img_loader_class(n.get("class_type"))
                 and not (n.get("inputs") or {}).get("image")
                 and nid not in skip]
        if not empty:
            return []
        async with _pooled_client(self.ctx) as c:
            name = await self._upload_placeholder(c)
        for nid in empty:
            wf[nid].setdefault("inputs", {})["image"] = name
        return empty

    def model_set_key(self, req: NormalizedRequest) -> Optional[str]:
        """What THIS request will load, as `scheduler.model_set_key` sees it — the
        workflow after the same injections `generate()` applies to its loaders (pins,
        the LoRA list/cascade, mapped params, label aliasing), minus the uploads and
        the bypass resolution, which touch no weight set. Pure: no I/O, and the stored
        workflow is copied first. None when nothing recognisable is loaded (the caller
        keys on the alias then)."""
        try:
            wf = self._workflow_for(req)
            fixed = [f for f in (req.fixed or []) if isinstance(f, dict)]
            _apply_fixed(wf, fixed)
            protected = {(f.get("node"), f.get("field")) for f in fixed if f.get("node") and f.get("field")}
            values = _gen_values(req)
            mapping = req.node_mapping or suggest_mapping(wf)
            for prm, m in mapping.items():
                lbl = ((m or {}).get("label") or "").strip()
                if lbl and lbl != prm and lbl in values and prm not in values:
                    values[prm] = values.pop(lbl)
            _apply_lora_list(wf, mapping, req.loras or [], self.ctx.loras_of(self.bid))
            _apply_lora_cascade(wf, values)
            _apply_mapping(wf, mapping, values, protected)
            return scheduler.model_set_key(wf, req.bypass or [])
        except Exception as e:                      # a key is a hint, never a failure
            logger.debug(f"model_set_key on [{self.name}] failed: {e}")
            return None

    async def generate(self, req: NormalizedRequest) -> GenOutput:
        b = self.backend
        url = b["url"].rstrip("/")
        bname = self.name
        wf = self._workflow_for(req)
        # Every input file this job uploads lives under ONE job-unique prefix, and the
        # names it actually wrote are collected in `uploaded` for the post-success
        # cleanup. No two jobs ever address the same input file (see upload_slot_name).
        prefix = upload_prefix_for(req.upload_prefix)
        uploaded: list = []
        fixed = await self._resolve_image_sentinels(list(req.fixed or []), req.upload_image,
                                                    prefix, uploaded)
        fixed_applied = _apply_fixed(wf, fixed)                # pin models / switches / ref-images
        protected = {(b.get("node"), b.get("field")) for b in fixed   # pins the API cannot override
                     if b.get("node") and b.get("field")}
        values = _gen_values(req)
        mapping = req.node_mapping or suggest_mapping(wf)      # explicit wins, convention is fallback
        # Labels are the EXTERNAL field names (mapping editor: "label overrides the
        # Playground label / API field name") — honor them as aliases: a client may
        # send {"seed": …} when the param is wired as e.g. `value` labelled "seed".
        for p, m in mapping.items():
            lbl = ((m or {}).get("label") or "").strip()
            if lbl and lbl != p and lbl in values and p not in values:
                values[p] = values.pop(lbl)
        # Auto-seed keys on the EFFECTIVE name (param or label), so a PrimitiveInt
        # wired as `value` + label "seed" gets a random seed instead of int('') 400s.
        seed_param = next((p for p, m in mapping.items()
                           if p == "seed" or ((m or {}).get("label") or "").strip().lower() == "seed"),
                          None)
        if seed_param and values.get(seed_param) in (None, ""):
            values[seed_param] = random.randint(0, 2**63 - 1)
        # image request-fields are filled per-field (upload or 8×8 placeholder), not
        # via the scalar mapping — keep them out of _apply_mapping.
        img_params = image_params(wf, mapping)
        uploads = dict(req.upload_images or {})
        # Label aliasing for image slots (same as scalar params above): the schema
        # advertises a slot under its LABEL, so a client sends the image under the
        # label (`input_image`); remap it to the param the upload path keys on
        # (`image`), else the loader keeps its baked-in value.
        for p in img_params:
            lbl = ((mapping.get(p) or {}).get("label") or "").strip()
            if lbl and lbl != p and lbl in uploads and p not in uploads:
                uploads[p] = uploads.pop(lbl)
        if req.upload_image and len(img_params) == 1 and img_params[0] not in uploads:
            uploads[img_params[0]] = req.upload_image          # back-compat single upload
        for p in img_params:
            values.pop(p, None)
        for p in (req.upload_files or {}):
            values.pop(p, None)              # the uploaded file feeds this param, not a stale scalar
        # `loras:[{name,strength}]` first (pairs resolved per group), then the flat
        # legacy lora_N params → remaining free slots.
        lora_placed = _apply_lora_list(wf, mapping, req.loras or [], self.ctx.loras_of(self.bid))
        lora_placed += _apply_lora_cascade(wf, values)    # client loras → next free stack slots
        applied = _apply_mapping(wf, mapping, values, protected)
        # An `on_empty: disable` slot left empty prunes a whole dead branch, which needs
        # each candidate class's required-input set. Loaded ONLY when such a slot is
        # actually empty; after discovery the cache already holds every class, so this
        # normally costs no request at all (and a cold cache stops the cascade safely).
        pruned: dict = {}
        extra_bypass: list = []          # the empty slots' `on_empty_bypass` ids (mode-4)
        prune_types = ({} if not any(not uploads.get(p)
                                     and slot_empty_mode(mapping.get(p)) == "disable"
                                     for p in img_params)
                       else await self._node_types_for(wf, list(wf)))
        img_applied = await self._apply_image_params(wf, mapping, img_params, uploads,
                                                     prefix, uploaded, prune_types, pruned,
                                                     extra_bypass)
        # A disabled slot that takes the alias's output node with it would submit a
        # workflow that cannot deliver anything — name the slot instead of letting the
        # job fail later as "produced no output".
        if pruned and req.output_node and req.output_node not in wf:
            culprit = next((p for p, ids in pruned.items() if req.output_node in ids),
                           next(iter(pruned)))
            raise RuntimeError(
                f"no image for '{culprit}' disabled node "
                f"{(mapping.get(culprit) or {}).get('node')} and everything downstream that "
                f"requires it, including the alias's output node {req.output_node} — "
                f"this slot is not optional in this workflow")
        files_applied = await self._apply_file_params(wf, mapping, req.upload_files or {},
                                                      protected, prefix, uploaded)
        autofilled = await self._autofill_empty_images(wf, mapping)
        # Bypass LAST — after every injection, so it rewires the FINAL link graph
        # (remove each bypassed node, reconnect consumers to its same-typed input).
        # The backend's own `bypass` and the empty image slots' `on_empty_bypass` run in
        # ONE pass: same mechanics, one `bypassed` entry in the summary, and a chain of
        # bypassed nodes resolves across both sources instead of only within one.
        # de-duplicated: a node may sit in the backend's list AND in an empty slot's
        # extras, and `_apply_bypass` reports what it was given — a doubled id in the
        # summary reads like the node was skipped twice.
        byp_ids = list(dict.fromkeys(str(x) for x in (list(req.bypass or []) + extra_bypass)))
        bypassed = _apply_bypass(wf, byp_ids,
                                 await self._node_types_for(wf, byp_ids))
        summary = {"applied": sorted(applied.keys()),
                   "seed": values.get(seed_param) if seed_param else None,
                   "fixed": sorted(fixed_applied.keys()),
                   "loras": [f"{n}.{f}={v}" for n, f, v in lora_placed],
                   "images": sorted(img_applied), "autofilled_images": autofilled,
                   **({"disabled_nodes": pruned} if pruned else {}),
                   **({"files": sorted(files_applied)} if files_applied else {}),
                   **({"bypassed": bypassed} if bypassed else {})}

        poll_interval = float(b.get("poll_interval", 1.0))
        max_wait = float(b.get("max_wait", 600))
        if not req.slot_held:            # a chain claims the slot itself and holds it across stages
            self.ctx.inflight_inc(self.bid)
        started = time.monotonic()
        log_on = self.ctx.log_enabled()
        try:
            # short per-request read timeout (NOT max_wait): a hung /history or /view read
            # then fails fast → the disconnect-grace/failover logic kicks in instead of
            # blocking the whole job for max_wait. The overall budget is the poll deadline.
            # Deliberately a per-job client (not the shared pool): the client-level
            # timeout shapes every submit/poll/fetch call below, and one client per
            # multi-second generation job costs nothing.
            timeout = httpx.Timeout(30.0, read=float(b.get("read_timeout", 60)))
            async with httpx.AsyncClient(timeout=timeout) as client:
                # A client_id ties this prompt to our websocket listener below. ComfyUI
                # 0.30 broadcasts progress to everyone, 0.34 measurably does not — sending
                # the id is what makes the feed work on both, and it is inert otherwise.
                client_id = f"gw-{req.job_id or uuid.uuid4().hex[:12]}"
                pr = await client.post(f"{url}/prompt",
                                       json={"prompt": wf, "client_id": client_id})
                if pr.status_code != 200:
                    raise RuntimeError(_comfy_prompt_error(pr.status_code, pr.text, wf, mapping))
                submitted = pr.json() or {}
                # A PARTLY invalid workflow still returns 200 AND a prompt_id: ComfyUI
                # queues the branches it can run and reports the rest in `node_errors`
                # (measured — one wrong link type). Letting that run would deliver
                # silently less than the alias configures, so a non-empty node_errors
                # fails the job here, before anything occupies the GPU.
                if submitted.get("node_errors"):
                    raise RuntimeError(_comfy_prompt_error(pr.status_code, pr.text, wf, mapping))
                prompt_id = submitted.get("prompt_id")
                if not prompt_id:
                    raise RuntimeError("ComfyUI returned no prompt_id")
                if log_on:
                    logger.info(f"→ [{bname}] queued {prompt_id} (workflow {os.path.basename(req.workflow or '?')})")
                # Live progress runs BESIDE the /history poll, never instead of it: the
                # poll owns the job's outcome (and its timeout, disconnect grace and
                # failover), the socket only says how far along it is. If the socket
                # never connects the job is completely unaffected.
                ws_state = {"prompt_id": prompt_id, "began": time.monotonic()}
                ws_task = asyncio.create_task(
                    self._ws_progress(url, client_id, req.job_id, ws_state))
                try:
                    outputs = await self._poll(client, url, prompt_id, poll_interval, max_wait, started)
                finally:
                    ws_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await ws_task
                    if req.job_id:
                        self.ctx.note_progress(req.job_id, None)   # the row owns it now
                rig, warnings = None, []
                if req.output_cases:                 # conditional case delivery (+ rig + validation)
                    blobs, rig = await self._fetch_by_cases(client, url, outputs, req.output_cases)
                    if not blobs:
                        raise RuntimeError(f"no output case matched (rigs: "
                                           f"{[c.get('rig') for c in req.output_cases]}; nodes with "
                                           f"outputs: {', '.join(sorted(outputs)) or 'none'})")
                    if req.output_globs:             # plain glob lines mixed with cases: unconditional extras
                        have = {b.name for b in blobs}
                        blobs += [b for b in await self._fetch_by_globs(client, url, outputs, req.output_globs)
                                  if b.name not in have]
                    normalize_delivery(blobs, rig, req.texture_format)   # V-flip (+ optional jpeg) textures
                    warnings = validate_delivery(blobs, rig)
                else:
                    blobs = await self._fetch_outputs(client, url, wf, outputs, req.output_node,
                                                      req.output_ext, req.output_globs)
                    if req.output_globs and not req.output_node and not blobs:
                        raise RuntimeError(f"no output file matched {req.output_globs} "
                                           f"(nodes with outputs: {', '.join(sorted(outputs)) or 'none'})")
                    if req.output_node and not blobs:
                        extra = (f" as a '.{req.output_ext}' sibling" if req.output_ext else "")
                        raise RuntimeError(f"configured output node {req.output_node} produced "
                                           f"no fetchable artifact{extra} (no matching file in its outputs)")
                    if req.dummy_check:          # 2x2-dummy safety net (case mode does it in validate);
                        _check_glb_not_dummy(blobs)  # opt-out for legit 1x1/2x2 constant-colour exports
                    warnings = validate_delivery(blobs, None)   # rig-less: the >30 MB size guideline
        finally:
            if not req.slot_held:
                self.ctx.inflight_dec(self.bid)

        # Clean success only (any raise above skipped this): the prompt has finished, so
        # nothing will read these inputs again — shrink them to the 72-byte placeholder.
        # After a timeout/interrupt the ComfyUI prompt may STILL run and open the file,
        # so the failure paths deliberately leave the inputs alone.
        await self._cleanup_uploads(uploaded)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if log_on:
            logger.info(f"← [{bname}] {len(blobs)} artifact(s) in {elapsed_ms} ms")
        return GenOutput(blobs=blobs, meta={
            "backend": bname, "workflow": req.workflow,
            "elapsed_ms": elapsed_ms, **summary,
            **({"rig": rig} if rig else {}),
            **({"warnings": warnings} if warnings else {}),
        })

    async def _ws_progress(self, url: str, client_id: str, job_id: str, state: dict) -> None:
        """Follow ComfyUI's /ws feed and report this job's real step progress.

        REST cannot answer "how far along is it?": `/queue` says only WHICH prompt runs
        and `/history` appears once it is over, which is why the job view had to
        estimate from the median of past runs. The sampler's per-step counter — the
        `25/35` in ComfyUI's own log — is published over the websocket only:
          progress:       {"value": 25, "max": 35, "node": "34", "prompt_id": …}
          progress_state: {"prompt_id": …, "nodes": {"34": {value, max, state}, …}}
        We submit with a `client_id` and connect with the SAME id: ComfyUI 0.30 happens
        to broadcast progress to every listener, but that is not something to rely on —
        measured 2026-09-03, a 0.34 box sent a fresh listener nothing at all while a job
        was running. Messages for another prompt are ignored either way (`prompt_id` is
        on every one), so one stray broadcast cannot mislabel this job's progress.

        Best-effort by construction: any failure — no `websockets` module, a refused
        connection, a message shape we do not know — just leaves the estimate as it was.
        It never raises into the job, and it is never the reason a generation fails.
        `state` is the caller's dict, so the poll loop can read what arrived (and notice
        that nothing has arrived for a while) while this task runs.
        """
        try:
            import websockets                       # ships with uvicorn[standard], like watchfiles
        except ImportError:
            return
        ws_url = re.sub(r"^http", "ws", url) + f"/ws?clientId={quote(client_id)}"
        try:
            async with websockets.connect(ws_url, max_size=None,
                                          open_timeout=10) as ws:
                while True:
                    raw = await ws.recv()
                    if isinstance(raw, (bytes, bytearray)):
                        continue                    # binary = a preview image frame
                    try:
                        msg = json.loads(raw)
                    except ValueError:
                        continue
                    view = self._progress_apply(state, msg, time.monotonic())
                    if view is not None:
                        self.ctx.note_progress(job_id, view)
        except asyncio.CancelledError:
            raise
        except Exception as e:                      # never a reason for a job to fail
            logger.debug(f"[{self.backend.get('name')}] ws progress unavailable: {e}")

    def _progress_apply(self, state: dict, msg: dict, now: float) -> Optional[dict]:
        """Fold one /ws message into `state`; return the view to report, or None to
        ignore the message. Pure apart from `state` — this is where every rule that
        would fail SILENTLY lives, so it is unit-tested (`test_ws_progress.py`).

        The `prompt_id` gate is the important one: ComfyUI 0.30 broadcasts progress for
        EVERY prompt on the box to every listener, so without it a job would happily
        display the step count of somebody else's render."""
        data = msg.get("data") or {}
        if data.get("prompt_id") != state.get("prompt_id"):
            return None                             # another job on the same backend
        kind = msg.get("type")
        if kind == "progress":
            val, mx = data.get("value"), data.get("max")
            if not (isinstance(val, (int, float)) and isinstance(mx, (int, float)) and mx > 0):
                return None
            node = str(data.get("node") or "")
            # Time the steps from the FIRST one, not from the submit: everything before
            # it is model loading, and charging that to step one predicted 250 s for a
            # job that took 15 (measured 2026-09-04). A new node — or a counter that
            # went backwards — starts a fresh measurement; a graph can hold several
            # samplers, and each has its own pace.
            if node != state.get("node") or val < state.get("value", 0):
                state.update({"first_at": now, "first_val": float(val)})
            state.update({"value": float(val), "max": float(mx), "node": node, "at": now})
            return self._progress_view(state)
        if kind == "progress_state":
            nodes = data.get("nodes") or {}
            if not nodes:
                return None
            done = sum(1 for n in nodes.values()
                       if isinstance(n, dict) and n.get("state") == "finished")
            state.update({"nodes_done": done, "nodes_total": len(nodes), "at": now})
            return self._progress_view(state)
        return None

    def _progress_view(self, state: dict) -> dict:
        """The public shape of live progress: fraction + an ETA derived from the
        MEASURED seconds per step of THIS run — far better than a median over past
        jobs, which cannot know that this one is a 1536² render on a busy box."""
        val, mx = state.get("value"), state.get("max")
        out = {"basis": "live", "backend": self.backend.get("name"),
               "node": state.get("node") or None,
               "nodes_done": state.get("nodes_done"), "nodes_total": state.get("nodes_total"),
               "updated_ago_s": 0}
        if isinstance(val, (int, float)) and isinstance(mx, (int, float)) and mx > 0:
            out.update({"step": int(val), "steps": int(mx),
                        "fraction": round(min(val / mx, 1.0), 3)})
            # Needs at least one step-to-step interval; before that there is no rate to
            # extrapolate from and NO eta is far better than a wrong one.
            first_at, first_val, t0 = state.get("first_at"), state.get("first_val"), state.get("at")
            if first_at and t0 and val > (first_val or 0):
                per_step = (t0 - first_at) / (val - first_val)
                out["eta_s"] = max(0, int(per_step * (mx - val)))
        return out

    async def _poll(self, client, url, prompt_id, poll_interval, max_wait, started) -> dict:
        # If /history stops responding *after* the run was reachable for a while,
        # ComfyUI most likely crashed ("Reconnecting"). Fail fast with a clear
        # ConnectionError (→ the router can fail over) instead of waiting out the
        # full timeout. `disconnect_grace` = seconds of continuous unreachability
        # tolerated before declaring the backend gone.
        grace = float(self.backend.get("disconnect_grace", 30))
        deadline = time.monotonic() + max_wait
        last_ok = time.monotonic()
        gone_since = None
        last_exc = None
        # The vanished-prompt /queue probe only needs to fire a few times within the
        # grace window (not every 1 s tick) — checking ~3× per grace still detects a
        # restart within grace + one interval, at a fraction of the request load.
        queue_every = max(poll_interval, grace / 3)
        last_queue_check = 0.0
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval)
            try:
                hr = await client.get(f"{url}/history/{prompt_id}")
                last_ok = time.monotonic()
                if hr.status_code != 200:
                    continue
                hist = hr.json()
                if prompt_id in hist:
                    entry = hist[prompt_id]
                    status = entry.get("status", {})
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"ComfyUI: {_format_comfy_error(status.get('messages'))}")
                    return entry.get("outputs", {})
                # Not done yet. Confirm it's still queued/running — if ComfyUI restarted,
                # the prompt is gone from BOTH history and queue, yet /history keeps
                # answering 200 (so the disconnect-grace never fires). Detecting the
                # vanished prompt fails fast instead of polling out the full max_wait.
                # Throttled: a skipped tick leaves `still = None` (unknown), so the
                # gone-timer carries over exactly as an errored probe would.
                still = None
                now = time.monotonic()
                if now - last_queue_check >= queue_every:
                    last_queue_check = now
                    try:
                        qr = await client.get(f"{url}/queue")
                        if qr.status_code == 200:
                            still = _prompt_in_queue(qr.json(), prompt_id)
                    except Exception:
                        still = None            # unknown → leave the gone-timer as is
                if still is True:
                    gone_since = None
                elif still is False:
                    gone_since = gone_since or time.monotonic()
                    if time.monotonic() - gone_since > grace:
                        raise ConnectionError(
                            f"ComfyUI prompt {prompt_id} gone from history AND queue for "
                            f">{grace:.0f}s — ComfyUI likely restarted mid-job")
                continue
            except (RuntimeError, ConnectionError):
                raise
            except Exception as e:
                last_exc = e
                if time.monotonic() - last_ok > grace:
                    raise ConnectionError(
                        f"ComfyUI unreachable for >{grace:.0f}s during execution "
                        f"(likely crashed/restarting): {type(e).__name__}: {e}")
                continue
        try:                                    # free the GPU: stop the still-running prompt
            await client.post(f"{url}/interrupt")
        except Exception:
            pass
        raise TimeoutError(f"ComfyUI timeout after {max_wait:.0f}s (prompt {prompt_id}); "
                           f"last poll error: {last_exc}")

    async def _fetch_outputs(self, client, url, wf, outputs,
                             output_node: Optional[str] = None,
                             output_ext: Optional[str] = None,
                             output_globs: Optional[list] = None) -> list[GenBlob]:
        # Glob mode (multi-file): deliver EVERY file matching the alias's patterns,
        # gathered across ALL output nodes. A workflow that rigs twice + bakes
        # textures registers several files (…_articulationxl.fbx, …_basecolor_*.png,
        # …_mia.fbx); the client wants a specific subset AND a sibling (…_mia.glb,
        # written next to …_mia.fbx but not registered). For each reported file we
        # also try its stem swapped to each glob's extension, so a glob picks up an
        # on-disk sibling via /view. Unmatched globs are simply absent (a later
        # split workflow producing only some files still works). Without an
        # output_node the globs ARE the delivery; with one, the node stays
        # authoritative and the globs ship as unconditional extras (appended
        # below) — same semantics as globs mixed with output_cases.
        if output_globs and not output_node:
            return await self._fetch_by_globs(client, url, outputs, output_globs)
        # Single-node mode: the alias's explicit output node (mapping editor
        # "Output" section) is authoritative — a workflow may export intermediate
        # files from several nodes, and only the configured one is the result. It
        # producing nothing is an ERROR, not a fallback. Without the setting: the
        # node the workflow titles as its main export (`output_final` / `Output`),
        # else everything (legacy behaviour).
        if output_node:
            if output_node not in outputs:
                raise RuntimeError(f"configured output node {output_node} produced no output "
                                   f"(nodes with outputs: {', '.join(sorted(outputs)) or 'none'})")
            targets = {output_node: outputs[output_node]}
        else:
            final = final_output_node(wf)
            targets = {final: outputs[final]} if (final and final in outputs) else outputs
        blobs: list[GenBlob] = []
        seen: set = set()
        for _nid, out in targets.items():
            # Scan every list-valued output key (images / gifs / videos / audio /
            # fbx_file / result / …), not just images+gifs, so SaveVideo/SaveAudio
            # AND custom nodes that report an on-disk PATH (UniRig's `fbx_file`,
            # Preview3D's `result`) are all caught regardless of the key. Two item
            # shapes: a {filename, subfolder?, type?} dict (core save nodes) or a
            # bare path string (custom nodes) — _view_params handles both and
            # ignores non-file items (booleans, status strings). Deduped by
            # (filename, subfolder).
            for _key, items in out.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    parsed = _view_params(item)
                    if parsed is None:
                        continue
                    fn, view = parsed
                    # With output_ext, prefer the SIBLING with that extension over the
                    # reported file: UniRig registers only `fbx_file` but writes
                    # <stem>.fbx AND <stem>.glb (+ a .fbm folder); the client wants
                    # the textured .glb. A reported file WITHOUT such a sibling (the
                    # basecolor PNG next to the mesh) still ships itself — swapping
                    # it away would silently drop it from the delivery.
                    tries = [(fn, view)]
                    if output_ext:
                        stem = fn[:-(len(fn.split(".")[-1]) + 1)] if "." in fn else fn
                        sib = f"{stem}.{output_ext}"
                        if sib != fn:
                            tries.insert(0, (sib, {**view, "filename": sib}))
                    r = None
                    for fn, view in tries:
                        key = (fn, view.get("subfolder", ""))
                        if key in seen:          # already delivered/probed for an earlier item
                            r = None
                            break
                        seen.add(key)
                        r = await client.get(f"{url}/view", params=view)
                        if r.status_code == 200:
                            break
                        if r.status_code != 404:   # only "no such file" may probe on — a backend
                            raise RuntimeError(    # error must not silently shrink the delivery
                                f"/view '{fn}' → HTTP {r.status_code}")
                        r = None
                    if r is None:
                        continue
                    mime, kind = _mime_and_kind(fn)
                    if mime == "application/octet-stream" and isinstance(item, dict):
                        fmt = item.get("format")               # dict items carry a format hint
                        if isinstance(fmt, str) and "/" in fmt:
                            top = fmt.split("/", 1)[0].lower()
                            if top in ("video", "audio"):
                                kind = top
                                mime = {"video": "video/mp4", "audio": "audio/mpeg"}[top]
                    blobs.append(GenBlob(data=r.content, mime=mime, kind=kind, name=fn))
        # output_node + output_globs: glob extras on top of the node's result (e.g.
        # the metallic PNG a SaveImage bakes next to a Preview3D-delivered GLB).
        # Only on a delivered node result — an empty one stays empty so the caller's
        # "output node produced no fetchable artifact" error fires instead of a
        # partial extras-only delivery.
        if output_globs and blobs:
            have = {b.name for b in blobs}
            blobs += [b for b in await self._fetch_by_globs(client, url, outputs, output_globs)
                      if b.name not in have]
        return blobs

    async def _fetch_by_cases(self, client, url, outputs, cases) -> tuple:
        """Conditional delivery: pick the FIRST case whose detect file (its first
        glob) actually exists in this run, then deliver all of that case's globs.
        Two riggers may share a workflow (humanoid MIA vs non-humanoid UniRig) or
        live in split workflows — either way exactly one case's detect file is
        present, so the client gets that case's files + its rig type. Returns
        (blobs, rig).

        A case is fetched ONCE and the detect condition checked on the result —
        a separate detect probe would download the (often large) detect file
        twice per job. A /view failure that is not a plain 404 raises inside
        _fetch_by_globs, so a transiently unreachable file fails the job loudly
        instead of silently selecting the wrong case."""
        for case in cases:
            globs = case.get("globs") or []
            if not globs:
                continue
            blobs = await self._fetch_by_globs(client, url, outputs, globs)
            if not any(fnmatch.fnmatch((b.name or "").lower(), globs[0].lower()) for b in blobs):
                continue                              # detect file absent → this case didn't run
            return blobs, case.get("rig")
        return [], None

    async def _fetch_by_globs(self, client, url, outputs, globs) -> list[GenBlob]:
        """Deliver every registered file (or a same-stem sibling) matching a glob.
        Order follows the globs, so results come out predictable for the client."""
        glob_exts = {g.rsplit(".", 1)[-1].lower() for g in globs if "." in g}
        # collect (filename, view-params) candidates across all output nodes: the
        # reported file plus its stem swapped to each requested extension (siblings).
        cands: list = []
        seen_cand: set = set()
        for out in outputs.values():
            for items in out.values():
                if not isinstance(items, list):
                    continue
                for item in items:
                    parsed = _view_params(item)
                    if parsed is None:
                        continue
                    fn, view = parsed
                    stem = fn[:-(len(fn.rsplit(".", 1)[-1]) + 1)] if "." in fn else fn
                    # reported file first, then its siblings in sorted-extension order —
                    # a DETERMINISTIC candidate order (a set literal would hash-shuffle,
                    # so a wildcard-ext glob like *_mia.??? could pick a different
                    # "result 0" per process).
                    for cfn in [fn, *(f"{stem}.{e}" for e in sorted(glob_exts))]:
                        key = (cfn, view.get("subfolder", ""))
                        if key not in seen_cand:
                            seen_cand.add(key)
                            cands.append((cfn, {**view, "filename": cfn}))
        blobs, fetched = [], set()
        for g in globs:                                  # glob order → stable result order
            for cfn, view in cands:
                if (cfn, view.get("subfolder", "")) in fetched:
                    continue
                if not fnmatch.fnmatch(cfn.lower(), g.lower()):
                    continue
                r = await client.get(f"{url}/view", params=view)
                if r.status_code == 404:                 # sibling for this glob doesn't exist
                    continue
                if r.status_code != 200:                 # a backend error is not "file absent" —
                    raise RuntimeError(                  # it must not shrink or mis-detect a delivery
                        f"/view '{cfn}' → HTTP {r.status_code}")
                fetched.add((cfn, view.get("subfolder", "")))
                mime, kind = _mime_and_kind(cfn)
                blobs.append(GenBlob(data=r.content, mime=mime, kind=kind, name=cfn))
        return blobs


# ── Cloud task backends (Meshy, Tripo): POST a task, poll, download ────────────

_CLOUD_DISCOVERY_TIMEOUT = 8.0
_CLOUD_HTTP_TIMEOUT = 30.0
_CLOUD_DOWNLOAD_TIMEOUT = 120.0
_MESHY_HTTP_TIMEOUT = _CLOUD_HTTP_TIMEOUT    # pre-Tripo name (test_meshy_adapter reads it)


class _TaskVerdict(RuntimeError):
    """A 200 poll whose body says the task cannot be read (Tripo `code != 0`) — counted
    like a 4xx by CloudTaskAdapter._poll (three in a row = final)."""


@dataclass
class RunResult:
    """What a vendor's `_run` hands back: the PRIMARY task (its id, endpoint and the body
    sent) and the final TaskState whose `downloads` already include every follow-up task's
    file (converts, clips). `extra_meta` is merged into the job meta."""
    task_id: str
    endpoint: str
    body: dict
    state: "cloudtask.TaskState"
    extra_meta: dict = field(default_factory=dict)


class CloudTaskAdapter(BackendAdapter):
    """Everything a cloud task API shares: the in-flight slot, the create/poll/download
    loop with its grace rules, the job meta and the chain roles. A vendor subclass
    supplies the URLs, the response envelope and the run order (see the hooks below).

    The request/response SHAPE lives in the pure module (`mod`: meshy.py / tripo.py);
    this class owns the HTTP, the in-flight slot and the credit balance seen at
    discovery."""

    cloud = True
    serves_generation = True
    mod = meshy                      # the pure module; every subclass overrides it
    type = mod.KIND

    def __init__(self, backend: dict, ctx: AdapterContext):
        super().__init__(backend, ctx)
        self.credits: Optional[float] = None
        self.credits_at: float = 0.0
        self.vendor: str = self.mod.VENDOR

    def _headers(self) -> dict:
        key = (self.backend.get("api_key") or "").strip()
        return {"Authorization": f"Bearer {key}"} if key else {}

    # ── vendor hooks ──────────────────────────────────────────────────────────
    async def discover(self, client: httpx.AsyncClient) -> Capabilities:
        """The balance call: models + the credits shown in /health. 0 → CloudNoCredits."""
        raise NotImplementedError

    async def _run(self, client, req: NormalizedRequest, cand: dict, opts: dict,
                   poll_interval: float, max_wait: float) -> RunResult:
        """Everything between the request values and the downloads: build the body,
        `_create` it, `_poll` it — plus whatever follow-up tasks the vendor needs."""
        raise NotImplementedError

    async def _task_request(self, client, endpoint: str, task_id: str):
        """One poll of a task (the HTTP response, not the body)."""
        raise NotImplementedError

    def _task_body(self, r) -> dict:
        """The task object out of a 200 poll response; raise _TaskVerdict for a body that
        is a verdict about the task (Tripo's code != 0)."""
        return r.json() or {}

    def _classify_create(self, r) -> Optional[str]:
        """'nocredits' | 'busy' | 'server' | 'rejected' | None (= accepted) for a create
        response — every vendor spells the same four verdicts differently."""
        raise NotImplementedError

    def _task_id_of(self, js: dict) -> str:
        """The new task's id out of an accepted create response."""
        raise NotImplementedError

    def _msg(self, r) -> str:
        try:
            return str((r.json() or {}).get("message") or r.text[:200])
        except Exception:
            return r.text[:200]

    def _thumb_name(self) -> str:
        return "preview.png"

    # ── the shared run ────────────────────────────────────────────────────────
    async def generate(self, req: NormalizedRequest) -> GenOutput:
        b, mod = self.backend, self.mod
        cand = {"model": req.real_model, mod.KIND: req.cloud or {}}
        opts = mod.options_of(cand)
        poll_interval = float(b.get("poll_interval", mod.POLL_INTERVAL_DEFAULT))
        max_wait = float(b.get("max_wait", mod.MAX_WAIT_DEFAULT))
        if not req.slot_held:
            self.ctx.inflight_inc(self.bid)
        started = time.monotonic()
        log_on = self.ctx.log_enabled()
        try:
            # The client default stays SHORT (30 s): every poll runs on it, and the
            # disconnect_grace logic in _poll only reacts as fast as a poll gives up.
            # The create POST carries the whole input and gets its own budget (_create).
            async with httpx.AsyncClient(timeout=_CLOUD_HTTP_TIMEOUT) as client:
                run = await self._run(client, req, cand, opts, poll_interval, max_wait)
                # `run.endpoint`, never the alias's: a vendor whose `_run` chases several
                # tasks names the PRIMARY one there, and the rig/thumbnail decisions and
                # the job meta must all describe the task that was actually run.
                state, endpoint = run.state, run.endpoint
                blobs = []
                for name, url in state.downloads:      # the pure module named them
                    try:
                        data = await self._download(client, url)
                    except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
                        # The task is DONE and BILLED by now, so a blip fetching its
                        # artifact must not raise a member of main._GEN_FAILOVER_ERRORS
                        # (httpx's transport errors are listed there literally): _run_job
                        # would re-run the whole task on the next candidate and pay again.
                        # Final, naming the paid task so the file can be fetched by hand.
                        raise RuntimeError(f"{self.vendor} artifact download failed after "
                                           f"the paid task {run.task_id}: {name}: {e}") from e
                    mime, kind = _mime_and_kind(name)
                    blobs.append(GenBlob(data=data, mime=mime, kind=kind, name=name))
                # the rig endpoint has no thumbnail — its input already had a preview
                if endpoint != mod.RIG_ENDPOINT and opts.get("thumbnail") and state.thumbnail:
                    try:
                        thumb = await self._download(client, state.thumbnail)
                        blobs.append(GenBlob(data=thumb, mime="image/png", kind="image",
                                             name=self._thumb_name()))
                    except Exception as e:           # a preview is a courtesy, the mesh is the job
                        logger.warning(f"[{self.name}] {mod.KIND} thumbnail download failed: {e}")
        finally:
            if not req.slot_held:
                self.ctx.inflight_dec(self.bid)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if log_on:
            logger.info(f"← [{self.name}] {len(blobs)} artifact(s) in {elapsed_ms} ms, "
                        f"{state.credits} credits")
        meta = {"backend": self.name, "cloud": mod.KIND, "cloud_task_id": run.task_id,
                "endpoint": endpoint,
                "ai_model": run.body.get("ai_model") or run.body.get("model"),
                "request": mod.request_summary(run.body), "consumed_credits": state.credits,
                "elapsed_ms": elapsed_ms, **run.extra_meta}
        if endpoint == mod.RIG_ENDPOINT:
            # A standalone rig job is a rig delivery like a chain's stage 2 is — `main._job_view`
            # reads meta["rig"], so name it here too or the job view shows no rig type at all
            # for the one endpoint that always produces one.
            meta.setdefault("rig", mod.KIND)
        return GenOutput(blobs=blobs, meta=meta)

    def _note_task(self, req: Optional[NormalizedRequest], task_id: str, endpoint: str,
                   body: Optional[dict] = None, role: Optional[str] = None) -> None:
        """Record a vendor task on the request the moment it EXISTS, under the same keys
        the success meta uses — so `admin._cloud_table` renders a failed run with no
        special case, and `main` can put the facts on the failed job row.

        Deliberately called from `_create` rather than from each vendor's `_run`: every
        task a job creates passes through there, so none can be forgotten (Tripo's
        converts and clips included), and the LAST one to be created is the one the job
        died on."""
        if req is None:
            return                              # a direct _create caller (tests) traces nothing
        tr = req.cloud_trace
        tr.setdefault("backend", self.name)
        tr.setdefault("cloud", self.mod.KIND)
        # `role` given = a SIDE task (Tripo's rig-check, convert:<fmt>, clip:<preset>): it
        # joins the task list but must not claim `cloud_task_id`/`endpoint`, which name the
        # PRIMARY task in the success meta and mean the same thing here.
        if role is None:
            tr.update({"cloud_task_id": task_id, "endpoint": endpoint})
            if body is not None:
                tr["request"] = self.mod.request_summary(body)
        # No `credits` here: what a task consumed is only known from its POLL, and a
        # guess on a failed row is worse than the silence (the vendor bills what it bills).
        tr.setdefault("tasks", []).append({"role": role or endpoint, "task_id": task_id})

    async def _create(self, client, url: str, body: dict, endpoint: str,
                      req: Optional[NormalizedRequest] = None,
                      role: Optional[str] = None) -> str:
        """POST the task and return its id, or raise the verdict `_classify_create` gives."""
        # The create POST is the one call that carries the whole INPUT in its body — image
        # data URIs, and for rigging the entire mesh as base64. Measured 2026-09-02 on
        # prod: a no-remesh 70 MB GLB became a ~93 MB JSON body, the POST hit the 30 s
        # client timeout, and httpx's WriteTimeout has an EMPTY str() — the job died as
        # "chain failed: " with nothing after the colon. So the create gets its OWN
        # size-scaled budget (~4 s per MiB ≈ a 256 KiB/s floor) without slowing the polls.
        raw = json.dumps(body)                  # serialised ONCE, sent as content=
        mb = len(raw) / (1024 * 1024)           # ASCII JSON: chars == bytes
        pr = await client.post(
            url, content=raw,
            headers={**self._headers(), "Content-Type": "application/json"},
            timeout=httpx.Timeout(connect=30.0, read=max(120.0, mb * 4),
                                  write=max(60.0, mb * 4), pool=30.0))
        verdict = self._classify_create(pr)
        if verdict == "nocredits":
            raise CloudNoCredits(f"{self.vendor}: {self._msg(pr)}", vendor=self.vendor)
        if verdict == "busy":
            raise CloudBusy(f"{self.vendor} queue full: {self._msg(pr)}", vendor=self.vendor)
        if verdict == "server":
            raise ConnectionError(f"{self.vendor} {pr.status_code}: {self._msg(pr)}")
        if verdict == "rejected":
            raise RuntimeError(f"{self.vendor} rejected the task ({pr.status_code}): {self._msg(pr)}")
        task_id = self._task_id_of(pr.json() or {})
        if not task_id:
            raise RuntimeError(f"{self.vendor} returned no task id")
        if self.ctx.log_enabled():
            logger.info(f"→ [{self.name}] {self.mod.KIND} {endpoint} task {task_id}")
        self._note_task(req, task_id, endpoint, body, role)
        return task_id

    async def _poll(self, client, endpoint: str, task_id: str, formats: list, opts: dict,
                    poll_interval: float, max_wait: float) -> "cloudtask.TaskState":
        # A poll that answers is not proof the task is running. A revoked key (401) or a
        # task that is gone (404) would otherwise be reported as a max_wait TimeoutError —
        # the wrong cause, AND only after holding the in-flight slot for the full wait.
        # So: a 4xx is a verdict about the TASK (three in a row, so one edge hiccup does
        # not end a job) → final RuntimeError; transport errors, 5xx and a 429 are about
        # the SERVICE and get `disconnect_grace` seconds of CONTINUOUS failure before the
        # failover-class ConnectionError — same key and default as ComfyUIAdapter._poll.
        # 429 sits on the service side deliberately: a poll-rate limit says nothing about
        # the task, which is running and already paid for — ending it would burn credits.
        # Any 200 resets both counters: only an unbroken run of failures means anything.
        # `max_wait` is a budget from HERE, so a vendor that runs several tasks for one
        # job passes the REMAINDER of the job's budget to each poll.
        grace = float(self.backend.get("disconnect_grace", 30))
        deadline = time.monotonic() + max_wait
        client_errs, gone_since = 0, None
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval)
            try:
                r = await self._task_request(client, endpoint, task_id)
            except httpx.HTTPError as e:            # one failed poll is not a verdict
                gone_since = gone_since or time.monotonic()
                if time.monotonic() - gone_since > grace:
                    raise ConnectionError(
                        f"{self.vendor} unreachable for >{grace:.0f}s while polling task {task_id}"
                        f": {type(e).__name__}: {e}")
                continue
            if 400 <= r.status_code < 500 and r.status_code != 429:
                client_errs += 1
                if client_errs >= 3:
                    raise RuntimeError(f"{self.vendor} task {task_id}: poll answered "
                                       f"{r.status_code} — {self._msg(r)}")
                continue
            if r.status_code != 200:                # 5xx / 429 — the service, not this task
                gone_since = gone_since or time.monotonic()
                if time.monotonic() - gone_since > grace:
                    raise ConnectionError(
                        f"{self.vendor} unreachable for >{grace:.0f}s while polling task {task_id}"
                        f": HTTP {r.status_code}")
                continue
            gone_since = None                       # the service answered — it is reachable
            try:
                task = self._task_body(r)
            except _TaskVerdict as e:               # a 200 whose BODY refuses the task: the
                client_errs += 1                    # same class of verdict as a 4xx, counted alike
                if client_errs >= 3:
                    raise RuntimeError(f"{self.vendor} task {task_id}: {e}")
                continue
            client_errs = 0
            # `options` by KEYWORD: meshy.parse_task takes the legacy `animations` bool in
            # the 4th positional slot, so a positional options dict would land there.
            state = self.mod.parse_task(task, formats, endpoint, options=opts)
            if state.error:            # failed/cancelled — or a status this gateway does not know
                detail = f"{self.vendor} task {task_id} {state.status.lower()}: {state.error}"
                # A failure the vendor blames on itself is the one kind worth attempting
                # again — UNLESS it was already billed, in which case a retry would buy the
                # same result twice (the paid-task rule the convert/clip paths follow too).
                if state.retryable and not (state.credits or 0):
                    raise CloudTaskRetryable(detail, vendor=self.vendor)
                raise RuntimeError(detail)
            if state.status == self.mod.SUCCESS_STATUS:
                return state
        raise TimeoutError(f"{self.vendor} task {task_id} not finished within max_wait={max_wait:.0f}s "
                           f"(still running at {self.vendor} — fetch it by id from the "
                           f"{self.vendor} dashboard)")

    @staticmethod
    async def _download(client, url: str) -> bytes:
        # NO auth header: signed asset URLs on a CDN host; the bearer must not leak there.
        r = await client.get(url, timeout=_CLOUD_DOWNLOAD_TIMEOUT, follow_redirects=True)
        if r.status_code != 200:
            raise RuntimeError(f"asset download failed ({r.status_code}) for {url.split('?')[0]}")
        return r.content

    # ── the chain roles (main._run_chain calls these) ─────────────────────────
    def chain_export(self, cand: dict, succ: dict, params: dict, prefix: str) -> ChainExport:
        """Stage 1 in the cloud: nothing to pin — the mesh comes back as a RESULT BLOB, not
        off a disk, so the name is ours to choose (it only labels the hand-off upload).
        A task that does not deliver glb is refused here, before credits are spent: the
        successor is fed a glb, and the vendor bills the task either way."""
        mod = self.mod
        if mod.endpoint_of(cand) == mod.RIG_ENDPOINT:
            # A rigging alias is a stage-2 product: it delivers `rigged.glb`, and its
            # request needs a mesh it has no way to get here. Refused by name — the
            # generic path would spend credits and then fail on "produced no mesh"
            # (chain_take_mesh looks for `model.glb`, which a rig task never returns).
            return ChainExport("", error=f"chain: a {self.vendor} rigging alias cannot be stage 1 — "
                                         f"it rigs an existing mesh; use an image-to-3d alias")
        opts = mod.options_of({mod.KIND: cand.get(mod.KIND) or {}, "model": cand.get("model")})
        if "glb" not in opts["target_formats"]:
            return ChainExport("", error=f"chain: {self.vendor} stage 1 must deliver glb — add it to "
                                         f"target_formats (now {opts['target_formats']})")
        return ChainExport(f"{prefix}.glb")             # no pins: the mesh is a result blob

    async def chain_take_mesh(self, out: GenOutput, export: ChainExport, want_bytes: bool) -> Optional[bytes]:
        """Stage 1's mesh is already in hand: the `model.glb` blob `generate()` downloaded
        (b'' when only its existence is asked for — nothing to fetch either way). Both
        kinds name it that: the pure module's `parse_task` decides the stem."""
        blob = next((b for b in (out.blobs or []) if (b.name or "") == "model.glb"), None)
        if blob is None:
            return None
        return blob.data if want_bytes else b""

    async def chain_feed_mesh(self, req2, backend2, mesh_param, mesh_name, mesh_bytes, outdir) -> str:
        """Stage 2 in the cloud reads the mesh off the REQUEST — `build_request` embeds
        `upload_files[mesh_param]` (Meshy as the `model_url` data URI, Tripo as an uploaded
        file token). There is no disk to upload to and no path the vendor could resolve, so
        the bytes are mandatory (a `path` relay is forced to `upload` in _run_chain whenever
        either stage is a cloud kind)."""
        if mesh_bytes is None:
            raise RuntimeError(f"chain: a {self.vendor} stage 2 needs the mesh bytes (upload relay)")
        req2.upload_files[mesh_param] = (mesh_name, mesh_bytes)
        return f"<upload:{mesh_name} ({len(mesh_bytes) / (1024 * 1024):.1f} MB)>"


# ── Meshy.ai (cloud image → 3D) ───────────────────────────────────────────────

_MESHY_API = "/openapi/v1"


class MeshyAdapter(CloudTaskAdapter):
    """Meshy.ai: POST /openapi/v1/<endpoint>, poll GET …/<endpoint>/{id}, download the
    model urls. Everything vendor-neutral is CloudTaskAdapter's — this class only says
    where the URLs are and how Meshy spells its answers."""

    mod = meshy
    type = mod.KIND

    def _api(self, path: str) -> str:
        return f"{self.backend['url'].rstrip('/')}{_MESHY_API}{path}"

    async def discover(self, client: httpx.AsyncClient) -> Capabilities:
        r = await client.get(self._api("/balance"), headers=self._headers(),
                             timeout=_CLOUD_DISCOVERY_TIMEOUT)
        r.raise_for_status()                       # 401 → auth kind in main._classify_error
        bal = int((r.json() or {}).get("balance") or 0)
        self.credits, self.credits_at = bal, time.time()
        if bal <= 0:
            raise CloudNoCredits("no credits left on the Meshy account", vendor=self.vendor)
        return Capabilities(models=set(meshy.AI_MODELS), pricing={}, loras=set())

    async def _run(self, client, req: NormalizedRequest, cand: dict, opts: dict,
                   poll_interval: float, max_wait: float) -> RunResult:
        """One task, one delivery: Meshy answers every requested format off the same
        task, so there is nothing to chase after the poll."""
        endpoint = meshy.endpoint_of(cand)
        body = meshy.build_request(cand, _gen_values(req), req.upload_images or {},
                                   req.upload_files or {})            # MeshyInput → final
        task_id = await self._create(client, self._api(f"/{endpoint}"), body, endpoint, req)
        req.cloud_trace["meshy_task_id"] = task_id      # the name existing rows/views read
        state = await self._poll(client, endpoint, task_id, opts["target_formats"], opts,
                                 poll_interval, max_wait)
        # `meshy_task_id`, not the neutral key: existing job rows and the job view read
        # it by that name.
        return RunResult(task_id, endpoint, body, state, {"meshy_task_id": task_id})

    async def _task_request(self, client, endpoint: str, task_id: str):
        return await client.get(self._api(f"/{endpoint}/{task_id}"), headers=self._headers())

    def _classify_create(self, r) -> Optional[str]:
        if r.status_code == 402:
            return "nocredits"
        if r.status_code == 429:                   # NoMoreConcurrentTasks / RateLimitExceeded
            return "busy"
        if r.status_code >= 500:
            return "server"
        # Meshy ACCEPTS a task with 202 (measured 2026-09-02 on prod: the first live job
        # died as "rejected (202)" while Meshy had created and billed it) — any 2xx that
        # carries a task id is a success, not just 200.
        return None if 200 <= r.status_code < 300 else "rejected"

    def _task_id_of(self, js: dict) -> str:
        return str(js.get("result") or "")


# ── Tripo3D (cloud image → 3D, and its rigger) ────────────────────────────────


class TripoAdapter(CloudTaskAdapter):
    """Tripo3D V3: every input is uploaded to `/v3/files` FIRST (the API takes no inline
    bytes — no data URI, no base64), then one task per endpoint under
    `/v3/generation|animations|models`, all polled at `/v3/tasks/{id}`.

    Where Meshy answers one job with one task, a Tripo job is SEVERAL: a rig runs
    rig-check → rig → one convert per extra format → one retarget per clip. They share
    the backend's `max_wait` (each poll gets the REMAINDER), their ids are recorded under
    `meta.tasks` so the job view can name every billed task, and their credits are summed."""

    mod = tripo
    type = mod.KIND

    # The endpoints an ALIAS can be configured on → the path that creates their task. The
    # follow-up tasks (rig-check, convert, retarget) are created by _run itself.
    _CREATE_PATH = {"image-to-model": "/generation/image-to-model",
                    "multiview-to-model": "/generation/multiview-to-model",
                    tripo.RIG_ENDPOINT: "/animations/rig"}

    def _api(self, path: str) -> str:
        return f"{self.backend['url'].rstrip('/')}{tripo.API}{path}"

    async def discover(self, client: httpx.AsyncClient) -> Capabilities:
        r = await client.get(self._api("/account/balance"), headers=self._headers(),
                             timeout=_CLOUD_DISCOVERY_TIMEOUT)
        r.raise_for_status()                       # 401 → auth kind in main._classify_error
        js = r.json() or {}
        if js.get("code", 0) != 0:                 # a 200 that carries a refusal
            raise RuntimeError(f"{self.vendor} balance: {js.get('message') or js.get('code')}")
        bal = _to_float((js.get("data") or {}).get("balance"))
        self.credits, self.credits_at = bal, time.time()
        if bal <= 0:
            raise CloudNoCredits("no credits left on the Tripo account", vendor=self.vendor)
        return Capabilities(models=set(tripo.AI_MODELS), pricing={}, loras=set())

    # ── the response envelope ────────────────────────────────────────────────
    def _msg(self, r) -> str:
        """Tripo's error body is `{code, message, suggestion}` — the suggestion is half
        the answer ("Check whether the API Key is correct"), so it travels with it."""
        try:
            js = r.json() or {}
            m = str(js.get("message") or "")
            s = str(js.get("suggestion") or "")
            return (m + (f" — {s}" if s else "")) or r.text[:200]
        except Exception:
            return r.text[:200]

    @staticmethod
    def _code_of(r):
        try:
            return (r.json() or {}).get("code")
        except Exception:
            return None

    def _classify_create(self, r) -> Optional[str]:
        code = self._code_of(r)
        if r.status_code == 403 and code == 2010:  # Tripo has no 402: 403 + 2010 = broke
            return "nocredits"
        if r.status_code == 429:                   # 1007 request rate / 2000 concurrency
            return "busy"
        if r.status_code >= 500:
            return "server"
        # A non-zero `code` is a refusal even under HTTP 200 — the envelope is the truth.
        if not 200 <= r.status_code < 300 or code not in (None, 0):
            return "rejected"
        return None

    def _task_id_of(self, js: dict) -> str:
        return str((js.get("data") or {}).get("task_id") or "")

    async def _task_request(self, client, endpoint: str, task_id: str):
        # ONE poll url for every task kind — the endpoint is not part of it.
        return await client.get(self._api(f"/tasks/{task_id}"), headers=self._headers())

    def _task_body(self, r) -> dict:
        js = r.json() or {}
        if js.get("code", 0) != 0:
            # HTTP 200 with a non-zero code says the task cannot be READ (gone, wrong
            # key): a verdict, which _poll counts like a 4xx (three in a row = final).
            raise _TaskVerdict(str(js.get("message") or f"code {js.get('code')}"))
        return js.get("data") or {}

    # ── uploads ──────────────────────────────────────────────────────────────
    async def _upload(self, client, name: str, data: bytes) -> str:
        """One `POST /v3/files` → file token. Every image and every mesh goes this way.

        A transport error or a 5xx is re-raised as ConnectionError so the job FAILS OVER
        to the next candidate: httpx's own transport exceptions are not ConnectionErrors,
        and the message names the step — an upload failure and a create failure send you
        looking in different places. A 4xx (or a non-zero envelope code) is Tripo's
        verdict on the FILE and is final."""
        try:
            r = await client.post(self._api("/files"),
                                  files={"file": (name, data, _mime_and_kind(name)[0])},
                                  headers=self._headers(),
                                  timeout=_upload_timeout_for(len(data)))
        except httpx.HTTPError as e:
            raise ConnectionError(f"{self.vendor} upload of {name} failed: "
                                  f"{type(e).__name__}: {e}")
        if r.status_code >= 500:
            raise ConnectionError(f"{self.vendor} upload of {name} failed "
                                  f"({r.status_code}): {self._msg(r)}")
        if not 200 <= r.status_code < 300 or self._code_of(r) not in (None, 0):
            raise RuntimeError(f"{self.vendor} refused upload {name}: {self._msg(r)}")
        token = str(((r.json() or {}).get("data") or {}).get("file_token") or "")
        if not token:
            raise RuntimeError(f"{self.vendor} returned no file_token for {name}")
        return token

    def _collect_inputs(self, req: NormalizedRequest, endpoint: str,
                        cand: dict, values: dict) -> tuple[dict, dict]:
        """({slot: (filename, bytes)}, {file field: (filename, bytes)}) for the endpoint —
        and every reason to refuse the request, checked BEFORE the first upload.

        The pure module refuses the same requests when it builds the body, but by then the
        files are already in the Tripo account: an upload for a request that can never run
        is litter nobody cleans up. So the slot count, the format sniff and the client's
        rig type (a 150 MB mesh pushed for a rig the alias's model cannot do) happen
        here — the rig-type RULE itself stays in the pure module, which build_request
        reads too."""
        got = req.upload_images or {}
        images = {s: got[s] for s in tripo.SLOTS[endpoint] if got.get(s)}
        if endpoint == "multiview-to-model" and ("input_image_front" not in images
                                                 or len(images) < 2):
            raise tripo.TripoInput("Tripo multiview needs `input_image_front` plus at "
                                   "least one more view")
        if endpoint == "image-to-model" and "input_image" not in images:
            raise tripo.TripoInput("`images.input_image` is required")
        named = {s: (f"{s}.{tripo.image_ext(b)}", b) for s, b in images.items()}
        files = {}
        for fname in tripo.FILES[endpoint]:
            f = (req.upload_files or {}).get(fname)
            data = f[1] if isinstance(f, tuple) else f
            if not data:
                raise tripo.TripoInput(f"`files.{fname}` is required")
            files[fname] = (f"{fname}.{tripo.mesh_ext(data)}", data)
        if endpoint == tripo.RIG_ENDPOINT:
            tripo.check_rig_type(cand, values)
        return named, files

    # ── the run ──────────────────────────────────────────────────────────────
    async def _run(self, client, req: NormalizedRequest, cand: dict, opts: dict,
                   poll_interval: float, max_wait: float) -> RunResult:
        endpoint = tripo.endpoint_of(cand)
        deadline = time.monotonic() + max_wait
        tasks: list = []
        credits = 0.0

        def left() -> float:
            """`max_wait` caps the WHOLE job, not one task: every follow-up poll gets
            what is left of the budget (the backend form says so)."""
            return max(0.0, deadline - time.monotonic())

        def took(role: str, tid: str, st) -> None:
            nonlocal credits
            tasks.append({"role": role, "task_id": tid, "credits": st.credits})
            credits += _to_float(st.credits)

        # 1. inputs → file tokens (validated first, see _collect_inputs)
        values = _gen_values(req)
        images, files = self._collect_inputs(req, endpoint, cand, values)
        tokens = {s: await self._upload(client, n, b) for s, (n, b) in images.items()}
        ftokens = {f: await self._upload(client, n, b) for f, (n, b) in files.items()}
        body = tripo.build_request(cand, values, tokens, ftokens)
        # The format the primary task delivers ITSELF leads the list: generation is always
        # GLB, a rig its `out_format`. parse_task names that download after formats[0], so
        # an alias asking for obj alone would otherwise ship a GLB called `model.obj`;
        # every OTHER requested format is a convert task below.
        native = opts["target_formats"][0] if endpoint == tripo.RIG_ENDPOINT else tripo.NATIVE_FORMAT
        formats = [native] + [f for f in opts["target_formats"] if f != native]
        extra: dict = {}
        # 2. the free rig-check: a verdict about the MESH, before the rig's 25 credits
        if endpoint == tripo.RIG_ENDPOINT and opts.get("rig_check"):
            cid = await self._create(client, self._api("/animations/rig-check"),
                                     tripo.build_rig_check(body["input"]),
                                     tripo.RIG_CHECK_ENDPOINT, req,
                                     role=tripo.RIG_CHECK_ENDPOINT)
            cst = await self._poll(client, tripo.RIG_CHECK_ENDPOINT, cid, formats, opts,
                                   poll_interval, left())
            took(tripo.RIG_CHECK_ENDPOINT, cid, cst)
            if not cst.riggable:
                raise RuntimeError(f"{self.vendor} rig-check: mesh is not riggable "
                                   f"(detected rig_type={cst.rig_type or '?'})")
            if cst.rig_type and cst.rig_type != body.get("rig_type"):
                # Tripo rigs what the REQUEST asked for; the check only reports what it saw.
                # A mismatch is a silently wrong skeleton (a quadruped rigged as a biped),
                # so it is named here — meta.rig_type records the DETECTED type.
                logger.warning(f"[{self.name}] tripo rig-check for {req.alias}: mesh looks "
                               f"like {cst.rig_type}, but the request rigs as "
                               f"{body.get('rig_type')}")
            extra["rig_type"] = cst.rig_type or body.get("rig_type")
        # 3. the primary task
        task_id = await self._create(client, self._api(self._CREATE_PATH[endpoint]), body,
                                     endpoint, req)
        state = await self._poll(client, endpoint, task_id, formats, opts, poll_interval, left())
        took(endpoint, task_id, state)
        stem = "rigged" if endpoint == tripo.RIG_ENDPOINT else "model"
        # 4. extra formats — each its own paid convert task. A failed one FAILS the job:
        #    a requested delivery must never silently shrink (same rule as Meshy's
        #    missing url), and the caller asked for that format on purpose.
        #    From HERE on the job is FINAL, whatever goes wrong: the primary task is done
        #    and billed (30 credits for a generation), and every failover-class error —
        #    CloudBusy from a 429 (converts run in Tripo's separate model-processing pool,
        #    so a rate limit there is routine), ConnectionError from a 5xx, TimeoutError
        #    from the job's max_wait running out, and the RAW httpx errors a create's POST
        #    can raise (ConnectError/WriteTimeout/ReadError are subclasses of NEITHER, and
        #    main._GEN_FAILOVER_ERRORS lists them literally) — would otherwise make
        #    _run_job re-run the WHOLE image-to-model task on the next Tripo candidate,
        #    paying for it again.
        #    The message names the format AND the paid task id, so the operator can fetch
        #    the finished mesh from Tripo's dashboard instead of buying it twice.
        for fmt in formats[1:]:
            cbody = tripo.build_convert(task_id, fmt, endpoint == tripo.RIG_ENDPOINT)
            try:
                cid = await self._create(client, self._api("/models/convert"), cbody,
                                         f"convert:{fmt}", req, role=f"convert:{fmt}")
                cst = await self._poll(client, "convert", cid, [fmt], opts, poll_interval, left())
            except (httpx.HTTPError, ConnectionError, TimeoutError, RuntimeError) as e:
                raise RuntimeError(f"{self.vendor} convert to {fmt} failed after the paid "
                                   f"task {task_id}: {e}") from e
            took(f"convert:{fmt}", cid, cst)
            state.downloads.append((f"{stem}.{fmt}", cst.downloads[0][1]))
        # 5. animation clips (rig only) — a courtesy, like Meshy's: a clip that fails is
        #    skipped with a warning, never the rigged mesh the job is actually about.
        if endpoint == tripo.RIG_ENDPOINT:
            extra["rig_spec"] = body.get("spec")
            extra.setdefault("rig_type", body.get("rig_type"))
            for preset in opts.get("animations") or []:
                rbody = tripo.build_retarget(task_id, preset, native)
                try:
                    cid = await self._create(client, self._api("/animations/retarget"),
                                             rbody, f"clip:{preset}", req,
                                             role=f"clip:{preset}")
                    cst = await self._poll(client, "retarget", cid, [native], opts,
                                           poll_interval, left())
                except (httpx.HTTPError, RuntimeError, ConnectionError, TimeoutError) as e:
                    # Every class, create and poll alike — a 429 on the create is a
                    # ConnectionError-derived CloudBusy, and a raw httpx transport error is
                    # neither a ConnectionError nor a TimeoutError but IS in
                    # main._GEN_FAILOVER_ERRORS. The job's max_wait running out, the clip
                    # pool being busy or the connection dropping must not throw away the
                    # rigged mesh, which is finished, paid for and in hand — and must not
                    # fail the job into a failover that would rig and bill it a second time.
                    logger.warning(f"[{self.name}] tripo clip {preset} skipped: {e}")
                    continue
                took(f"clip:{preset}", cid, cst)
                state.downloads.append((f"{tripo.clip_name(preset)}.{native}",
                                        cst.downloads[0][1]))
        state.credits = credits
        extra["tasks"] = tasks
        return RunResult(task_id, endpoint, body, state, extra)


# ── Registry ──────────────────────────────────────────────────────────────────

ADAPTERS: dict[str, type[BackendAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "comfyui": ComfyUIAdapter,
    "meshy": MeshyAdapter,
    "tripo": TripoAdapter,
}

# Backend types that take POST /v1/generations work (main routes generation to these
# and keeps them OUT of the chat catalogs). Derived from the classes, not listed twice.
GEN_TYPES: frozenset = frozenset(t for t, cls in ADAPTERS.items() if cls.serves_generation)

# The cloud task backends among them (Meshy, Tripo) — a candidate's kind must match its
# backend's (see backend_kind). Derived from the classes for the same reason as above.
CLOUD_TYPES: frozenset = frozenset(t for t, cls in ADAPTERS.items() if getattr(cls, "cloud", False))


def make_adapter(backend: dict, ctx: AdapterContext) -> BackendAdapter:
    """Instantiate the adapter for a backend, dispatched on its `type` field
    (default `openai` → unchanged behaviour for every existing backend)."""
    cls = ADAPTERS.get(backend.get("type", "openai"), OpenAIAdapter)
    return cls(backend, ctx)
