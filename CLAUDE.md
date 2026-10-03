# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

Renamed from llm-gateway on 2026-09-08 (prod: /opt/ai-hub, unit ai-hub.service).

## What this is

An OpenAI-compatible reverse proxy that fans one endpoint out across many backends:
local LLM servers (llama.cpp / llama-swap / vLLM / Ollama), cloud APIs (Together.ai /
OpenAI / OpenRouter / Anthropic), ComfyUI image/video/3D servers and cloud 3D services
(Meshy, Tripo). It adds per-backend discovery, one queued scheduler with failover,
virtual aliases, concurrency caps, multi-user auth + quotas, call parking, a
media-generation subsystem (workflow mapping, LoRAs, chains, jobs), a normalized
reasoning toggle, **managed hosts** (GPU machines rented on demand from Thunder Compute,
with tunnels, service setup and a model sync) and a server-rendered `/ui` console.

## Where things are documented

- **`README.md`** — every config knob, endpoint and routing rule (operator view).
- **`docs/architecture.md`** — the detailed design reference: per module what it does,
  the rules it keeps and WHY (most name the measurement that found them), plus request
  flow, routing rules, auth, stats and what each test guards. **Grep it for the symbol
  before changing a mechanism**, and update it when a mechanism changes.
- **Test docstrings** (`tests/test_*.py`) — each says which silent failure it pins.
- `docs/superpowers/` — specs/plans, **local only** (gitignored; the repo is public —
  never stage them). Managed-host rulings `R-K*`/`R-W*`/`M*` cited in code live there.

## Run / develop

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml          # then edit backends + api_key
venv/bin/uvicorn main:app --host 0.0.0.0 --port 4000   # add --reload for dev
```

- `config.yaml` is gitignored and **hot-reloaded** on save (watchfiles; the watch is on
  the directory, so rename-replace saves and a symlinked config work). Read **only at
  startup**: `stats.enabled` and the stats/jobs/faults DB paths.
- `requirements.txt` omits `watchfiles` and `websockets`; both ship with
  `uvicorn[standard]`. Keep it that way.
- **No linter, no build step, no blanket test suite** — only targeted stdlib `unittest`
  files for mechanisms that fail SILENTLY:
  `venv/bin/python -m unittest discover -s tests -t .`
  Everything else is verified by running the server and hitting endpoints with `curl`
  (README "Try it"), or `curl -H "Authorization: Bearer <admin key>" localhost:4000/health`
  for a routing snapshot (`?verbose=1` lists model ids; without an admin key a locked
  gateway answers only status + counts).
- Compile-gate before every deploy: `venv/bin/python -m py_compile *.py` — a broken file
  fails the restart silently.
- `/ui` changes: verify in a real browser (headless Chromium via CDP against a test
  instance), since most console failures are silent in the HTML.

## Deploy

`DEPLOY_HOST=root@host ./deploy.sh` (rsync — present on both ends — or tar over SSH,
remote venv install, systemd sync, restart). `ai-hub.service` runs as root inside a
root-compatible sandbox (NoNewPrivileges, PrivateTmp, ProtectSystem=full, kernel/cgroup
protections, AF_NETLINK kept for Scan network); ProtectHome / ProtectSystem=strict would
silently break voice shipping and DB writes — `test_service_unit.py` pins both halves.

`deploy.sh` has TWO exclude lists (`RSYNC_EXCLUDES`/`TAR_EXCLUDES`). Every runtime file
must be in both AND in `.gitignore`, or `rsync --delete` wipes it on prod: `config.yaml`,
`store.db` + `secret.key`, `stats.db*`, `jobs.db*`, `faults.db*`, `jobs/`, `calls/`,
`voiceref/`, `thunder.key*`, `thunder-known_hosts/`, `thunder-ctl/`,
`modelsrc.key*`, `modelsrc-known_hosts`. Never commit any of them.

## Architecture map

Twenty-four Python files at the top level (`ls *.py` is the count of record), tests in
`tests/`, scripts that run on OTHER boxes in `ops/`. `main.py` owns all app state as
module globals; **no other module imports `main`** — they get what they need through
injected callables (`AdapterContext`, `admin.bind(...)`, hostctl `Deps`) and stay
hot-reload-safe. Modules marked *pure* do no I/O and import neither `main` nor
`adapters`.

| Module | Role |
|---|---|
| `main.py` | Config load/hot reload, discovery loop, routing, every HTTP endpoint, auth/quotas, parking, generation orchestration, Responses bridge endpoints. `load_config` → `rebuild_backends` / `rebuild_virtual_models` → `rebuild_route_index`; `refresh_backend` = discovery; `build_backend_adapters` keeps unchanged adapter instances (`adopt_state`). |
| `adapters.py` | Per-backend protocol seam: `OpenAIAdapter` (chat/completions/embeddings/TTS, stream normalizer), `AnthropicAdapter` (verbatim `/v1/messages` passthrough), `ComfyUIAdapter` (workflow mapping, bypass/prune, watchdog, ws progress, targeted stop), `CloudTaskAdapter` + `MeshyAdapter`/`TripoAdapter`. |
| `meshy.py`, `tripo.py` | *Pure* halves of the cloud 3D vendors; same duck-typed module interface (`KIND`, `ENDPOINTS`, `OPTION_FIELDS`, `build_request`, `parse_task`, …). |
| `cloudtask.py` | *Pure* leaf for both: `TaskState`, the `opt__<key>` option-form reader/writer. |
| `scheduler.py` | *Pure* ordering: fastest free unpaid backend, freed-backend type affinity, overdue guard, exec-fault quarantine, VRAM-free decisions, host flag table. |
| `reasoning.py` | *Pure* `reasoning: off\|on\|auto` → per-(model, backend) mechanism. |
| `responses_bridge.py`, `anthropic_bridge.py`, `openai_image_bridge.py` | *Pure* protocol translation (Responses↔Chat, Messages↔Chat, OpenAI image shims). |
| `jobs.py` | Generation job store (SQLite + `jobs/<id>/` artifacts); terminal states are final. |
| `stats.py` | Optional call log (SQLite WAL + gzip body files), aggregates, month-cost quota. Data only — `admin` renders. |
| `faults.py` | Backend fault log (memory ring + `faults.db`), always on. |
| `store.py` | Writable SQLite store — the console's source of truth (seeded once from config); secrets encrypted via `secret.key`. |
| `admin.py` | The `/ui` console (tabs, live morph, forms). |
| `previewanim.py` | Idle animation injected into a rigged GLB for the `/ui` inspection view only. |
| `netscan.py` | *Pure* LAN scan behind Backends → Scan network. |
| `thunder.py` / `hostapi.py` | Thunder Compute provider: *pure* half / HTTP half + the `PROVIDERS` registry. |
| `hostctl.py` | One lifecycle `Controller` per managed host: start/stop via snapshots, tunnel, services, model sync. |
| `services.py` | What runs a backend on a managed VM, by type (ComfyUI loop, generic command service). |
| `sshrun.py` | System-`ssh` argv builders, ControlMaster tunnel `Supervisor`, streams. |
| `modelsync.py` | *Pure*: which model files a ComfyUI alias needs, where they come from (LAN share, catalog URL, derived HF URL). |
| `loratags.py` | *Pure* half of LoRA trigger words (stored and delivered, never put into a prompt); metadata keyed on the share file's sha256, Civitai by hash. |

Requests: chat/completions/embeddings/speech and `/v1/messages` all funnel through
`_dispatch_or_park()` (resolve → ready/busy split → dispatch with failover → park FIFO
when all busy). Generation (`/v1/generations` + OpenAI image shims) runs as a job via
`get_gen_routes` → `_run_job` / `_run_chain`. Details: `docs/architecture.md`.

## Invariants — the rules that break silently

Each of these once failed without an error; the full story is in `docs/architecture.md`.

**Routing & dispatch**
- Never mutate `backends` / `virtual_models` outside their rebuild functions — the
  precomputed route index goes stale.
- No `await` between `resolve_routes`/busy check and `_inflight_inc`, and the releasing
  `try` starts right after the inc (else caps overrun or slots leak).
- Failover on transport errors only; a `ReadTimeout` on a `paid`/`anthropic` backend is a
  504, never a failover (it would buy the answer twice). `PoolTimeout` is the gateway's
  own 503 — no failover, no fault row.
- A billed cloud task never fails over or self-retries (only `CloudTaskRetryable`);
  after the primary task is billed, follow-up failures are final `RuntimeError`s.
- Sampling precedence is client > alias > backend; backend defaults and reasoning are
  derived per backend inside the adapter so a failover re-derives them.
- `<backend>/current` never loads a model.

**Anthropic / protocols**
- An `anthropic` backend serves `/v1/messages` ONLY (licence boundary, `serves_path`)
  and is a verbatim passthrough: no `_StreamNormalizer`, `apply_reasoning` or
  `sampling_defaults` (they destroy cache breakpoints and thinking signatures).
- `_forward_headers` is a DENYLIST on purpose; `authorization` and `x-api-key` are
  gateway credentials and never forwarded. Response builders keep only their own
  headers, except upstream `retry-after`, which must be passed through everywhere.
- Serialize an outgoing body ONCE (`_encode`); big JSON/PIL/`/object_info` work runs in
  a worker thread, never on the event loop.

**Generation**
- Every uploaded input is named `gw_<job id>…` — never a shared name or fallback.
- Pinned `(node, field)` values and per-backend `bypass` beat any client param; client
  values that are lists, or paths in file fields, are refused unless trusted.
- Job terminal states are final (first terminal write wins); `prune_once` removes
  finished jobs only. Once a job row exists, the JOB owns the outcome, not the call log.
- ComfyUI stops are targeted (`_stop_prompt`) — never a bare `/interrupt`.

**Console (`/ui`)**
- Every action is a POST (`_POST_ACTIONS`); GET routes never write. Query values go
  through `_q`, never `_esc`.
- Live pages morph `<main>` and never insert `<script>`s: a live page must already
  contain every script any later state renders (hoist them / `window.gwLiveHooks`).
  List rows carry keys (`data-k`); JS is ES5 *syntax*.
- Values into JavaScript only via `data-*` attributes or `_js_json` — never
  `html.escape` into an inline handler.
- Refused saves answer 400 with the form re-rendered as typed; blank is the only
  "unset" for numbers; taken names are refused, never merged.
- A form switches panes by `display` only — a field not rendered is read as CLEARED.
- Secrets are never rendered back (backend keys, provider/HF tokens); provider tokens
  are not part of `store.get_settings()`.

**Managed hosts** (billing money — read `docs/architecture.md` first)
- A mutating provider call only on an item found BY UUID in a fresh list; an instance
  is `off` only after two consecutive lists without it.
- Nothing secret in argv (world-readable); admin text goes over stdin, never into a
  command line; every ssh argv ends `-- <host>`.
- A backend's `enabled` belongs to the host it is on NOW (R-K2): a detach never disables.
- Tunnel forwards change via ControlMaster `-O forward|cancel`, never a respawn.

**Every new silently-failing mechanism gets a test** in `tests/` whose docstring says
what it guards, and a line in `docs/architecture.md`.

## Conventions

- Discovery handles two pricing schemas and two `/v1/models` shapes (`{"data":[…]}` vs
  bare list) defensively (`extract_models`/`extract_pricing`/`_is_chat_model`). Keep it.
- Keep `stats.py`/`jobs.py`/`store.py`/`reasoning.py`/`responses_bridge.py`
  dependency-free and hot-reload-safe (no config values cached at import time).
- Generation: workflow + mapping are shared across an alias's candidates; `fixed` pins
  and `bypass` are per backend. A cloud alias has an option block instead, and an alias
  is homogeneous in kind (`cand_kind == backend_kind`).
- The Responses bridge supports `stream` and `background`; not built: stream reconnect
  for a background response (poll only).
- Voice cloning: TTS backends read `voice` as a file on THEIR host, so the voice library
  ships references via scp to every `voice_ref_hosts` target.
- Single prod instance. Verify with compile + a route/render check; restart only when
  the user says the gateway is idle.
