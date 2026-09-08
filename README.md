# AI-Hub

*(formerly `llm-gateway` — renamed 2026-09-08; the old GitHub URL redirects.)*

An OpenAI-compatible reverse proxy that fans one endpoint out across many
backends — local LLM servers (llama.cpp / llama-swap / vLLM / Ollama …), cloud
APIs (together.ai, OpenAI, OpenRouter …), **and ComfyUI image-generation
servers**. Callers see a single OpenAI endpoint; the gateway handles discovery,
one queue with a unified scheduler, failover, virtual aliases, per-backend
concurrency, an optional multi-user layer, call parking, and a built-in
management console.

It sits between OpenAI-compatible clients (N8N, LibreChat, Open WebUI, LangChain
code, image clients like anima-verse, …) and a fleet of backends.

---

## Contents

- [Why](#why)
- [Quick start](#quick-start)
- [Configuration](#configuration) — backends, aliases, knobs
- [Authentication & multi-user](#authentication--multi-user) — keys, allow-lists, quotas
- [Routing](#routing) — the scheduler, prefixing, `local`, concurrency, parking
- [Call parking](#call-parking) — queue instead of `503` when busy
- [Reasoning control](#reasoning-control) — thinking on/off per request, per alias, per model×backend
- [Claude Code / Anthropic Messages](#claude-code--anthropic-messages) — `/v1/messages`, mixed Anthropic + open-weight
- [Media generation](#media-generation) — ComfyUI image/video/audio + Meshy.ai and Tripo3D cloud meshes & rigging, aliases, mapping, chains, LoRA, jobs
- [The `/ui` console](#the-ui-console)
- [Stats & routing dashboard](#stats--routing-dashboard)
- [Endpoint reference](#endpoint-reference)
- [Try it](#try-it)
- [Running & deploying](#running--deploying)

---

## Why

- **One endpoint for many backends.** Point your tools at one URL; add/remove
  backends without touching clients. Chat, embeddings, the Responses API, *and*
  image generation all go through the same gateway.
- **Auto-discovery.** Each backend's catalog is polled — `/v1/models` for LLMs,
  `/object_info` for ComfyUI (models + installed LoRAs). No manual registry.
- **One queue, one scheduling rule + failover.** Every request queues; the
  **fastest free unpaid** backend that can serve it takes it, a backend that
  frees up prefers the request type it just ran (no model reload), and nothing
  waits longer than `affinity_max_wait_s` for that preference.
- **Live progress for generation jobs.** ComfyUI's own step counter (`25/35`) is read
  off its websocket, so a running job shows `running 25/35` with a bar and an ETA
  computed from the seconds per step measured in that very run — not a guess from past
  jobs. Backends that do not report it simply fall back to the old estimate.
- **A backend that burns jobs is taken out of rotation.** A generation backend can
  answer every health check and still fail every prompt (a broken driver update, a
  missing custom node). Such a job now moves on to the next backend instead of dying,
  and a backend that fails twice where another one succeeded is skipped for that alias
  for 15 minutes — shown as `quarantined` in `/health` and the Backends tab. If they
  all fail alike the request is the suspect, so nothing is quarantined.
- **Virtual aliases.** `fast`, `vision`, `translator` map to different real model
  IDs per backend.
- **Cloud-as-backend.** A per-backend `api_key` wires in any OpenAI-compatible
  provider as just another backend; mark it `paid: true` and it is used only
  when no unpaid backend is free.
- **Multi-user.** Optional per-user API keys with model/alias/backend allow-lists
  (which also filter what each key sees in `/v1/models`) and monthly cost quotas.
- **Call parking (default).** When every matching backend is busy, the call
  queues until one frees instead of returning `503` — no client change needed.
  Park time is per-alias; async is the standard Responses background mode.
- **Media generation.** ComfyUI workflows exposed as OpenAI image endpoints + a
  native job API — **image, video, and audio** outputs — with a convention-free
  node mapping, dynamic LoRAs, and LoRA-aware backend routing.
- **Built-in console at `/ui`.** Manage backends, aliases, workflow mappings,
  users, server settings; run a chat/media playground; watch jobs, stats, parked
  calls, and the live routing map. Server-rendered, zero JS framework.
- **Hot config reload.** `config.yaml` changes apply live; most management also
  lives in a writable store edited from the console.

---

## Quick start

```bash
git clone https://github.com/KaletoAI/ai-hub.git
cd ai-hub
python3 -m venv venv && venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml
$EDITOR config.yaml                    # set backends + api_key
venv/bin/uvicorn main:app --host 0.0.0.0 --port 4000   # add --reload for dev
```

Point any OpenAI-compatible client at `http://<host>:4000/v1` with the `api_key`
you set. Open `http://<host>:4000/ui` for the management console.

> `requirements.txt` omits `watchfiles`; it ships transitively with
> `uvicorn[standard]` and powers the hot-reload of `config.yaml`. Keep that extra.

---

## Configuration

`config.example.yaml` is the documented template. Copy to `config.yaml`
(gitignored, hot-reloaded on save). Two things read **only at startup**:
`stats.enabled` and the jobs/stats DB paths.

**Config vs. store.** `config.yaml` is the bootstrap source. Once the console is
used, most state (backends, chat aliases, generation aliases + mappings, users,
server settings) lives in a writable SQLite **store** (`store.db`) which then
becomes the source of truth. The store is seeded once from config and merged
over it; you can run almost entirely from config or almost entirely from the
console — both work.

```yaml
api_key: "sk-change-me"                # master/admin key (see Authentication)
health_check_interval: 30              # seconds between backend liveness polls
log_per_call: true                     # one log line per forwarded request
model_prefix: true                     # list models as <backend>/<model>
# max_concurrent: 1                    # global default in-flight cap per backend

backends:
  - name: local-gpu
    url: http://192.168.1.10:8080      # llama-swap / llama.cpp / vLLM / …
    max_concurrent: 1                  # single-slot llama.cpp → one at a time
    # local: true                      # ALSO list its models bare (see below)
  - name: together                     # cloud fallback (OpenAI-compatible)
    url: https://api.together.xyz
    paid: true                         # only used when no unpaid backend is free
    api_key: "tgp_v1_…"                # sent as Bearer to this backend
    chat_only: true                    # drop non-chat models at discovery
    serverless_only: true              # drop dedicated-endpoint-only models

virtual_models:                        # chat aliases
  "translator": "Aya-Expanse-8B"       # same model on every backend
  "fast":                              # per-backend mapping
    local-gpu: "Qwen3.5-9B"
    together:  "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
```

A backend's value under an alias is just the model name to call there.

---

## Authentication & multi-user

Two layers, both optional:

- **Master key** — the top-level `api_key`. Clients send
  `Authorization: Bearer <key>`. Also unlocks the `/ui` console (sign in with an
  admin key). Leave empty to run fully open (bootstrap mode).
- **Per-user keys** — created in the **Users** tab. Each user has its own API
  key (generate one in the form, or paste your own), a role (`user` / `admin`),
  an enabled flag, an optional model **allow-list**, and an optional **monthly
  cost quota**. Calls are attributed to the user (stats source, job owner).
  An existing user's key can be **copied again** later: the editor pre-fills it
  (masked; 📋 Copy reveals and copies). Keys are stored encrypted, and the
  pre-fill is a console convenience you can switch off — clear
  `show_user_keys` in the Server tab and only a key generated right there in the
  form is ever shown, as before.

**Bootstrap-open → locked.** With no users *and* no master key, the gateway and
console are fully open. Add an admin user (or set a master key) to lock it down.

**Job ownership.** Generation jobs and background responses are owner-gated:
`GET`/cancel of a job (and its result/input artifacts) is allowed only for its
owner; admin/master see all. Authenticated calls are owned by the user. In
bootstrap-open mode an anonymous caller is owned by its **client IP** (`ip:<addr>`),
so keyless LAN services get a best-effort separation — each sees only its own
jobs. This is convenience, **not** a security boundary (NAT/spoofing); for a real
boundary give each service its own user + key. Legacy/`default`-owned jobs stay
visible to everyone.

### Allow-list (what a key may use — and see)

A user's allow-list can contain any mix of:

| Entry kind | Grants |
|---|---|
| **chat alias** (`fast`) | that chat alias |
| **image alias** (`Qwen`) | that generation alias |
| **backend name** (`together`) | **all** of that backend's models |
| model id (`together/llama-3…` or bare) | that specific model |

An **empty** allow-list = everything allowed (the default). A non-empty list both
**restricts usage** (a disallowed model → `403`) **and filters `/v1/models`** so
the key only sees what it's allowed. This is how you point an image client at the
gateway and have it see just the image aliases instead of the whole 400-model
catalog: give it a key whose allow-list is the image alias(es) (or the ComfyUI
backend), and `GET /v1/models` returns only those. `?type=image` / `?type=chat`
narrows by namespace too.

### Quotas

- **`quota_req_day`** — requests per day (in-memory counter) → `429` when exceeded.
- **`quota_cost_month`** — summed USD cost for the month (from the stats log) →
  blocked when exceeded. Needs stats enabled and priced backends; streaming calls
  are costed from the backend's usage chunk (the gateway always requests
  `stream_options.include_usage`) and fall back to gateway estimates only on a
  backend that reports nothing.

---

## Routing

A backend is a **candidate** for a request when it is (1) enabled, (2) healthy
(last discovery poll ok), (3) mapped for the alias (or exposes the bare/real
model), and (4) actually has the resolved model. Every request then goes through
one queue, and one rule set decides who runs where — LLM calls and media jobs
alike:

- **Fastest free unpaid backend first.** Ready (not busy) candidates are ordered
  by `(paid, speed)`: unpaid before paid, then fastest first. Speed is measured —
  tok/s per LLM backend, seconds per media job per alias+backend — and a backend
  that has never been measured sorts first, so it gets probed once. `paid: true`
  is the cost guard: a paid backend (a cloud API) is used **only when no unpaid
  candidate is free**, which is the old "spill to the cloud" behaviour without
  per-backend priority numbers.
- **A freed backend prefers what it just ran.** When a backend frees, it takes
  the oldest waiting request whose *type key* matches what it last ran — the
  generation alias for media (same workflow = same loaded models), the real model
  id for LLMs (no llama-swap model reload). Only the request the scheduler
  designates may claim that backend; the others stay queued.
- **Nobody waits on that preference forever.** A request queued longer than
  `affinity_max_wait_s` (default **120 s**, Server tab, hot-reloaded) counts as
  *overdue* and is served strictly oldest-first by the next free backend that can
  run it.

If the chosen backend errors on the forward, the remaining candidates are tried
in the same order. When every candidate is busy the call **parks** (queues) by
default until one frees, and only `503`s if the park time runs out (below).
`priority` is no longer a routing input — the key may stay in a config for
listing order, but nothing routes by it.

### Provider-prefixed model names

With `model_prefix: true` (default), `/v1/models` lists every model as
`<backend>/<model>` so the provider is visible. Input is liberal: a prefixed id
routes to exactly that backend; a bare id or an alias goes through the
scheduler. Backend
names never collide with vendor prefixes (`moonshotai/…`), so the leading segment
disambiguates. `model_prefix: false` → legacy bare, de-duplicated listing.

**`local: true`** on a backend *additionally* lists its models bare (alongside
the prefixed id). A bare request then routes across every `local`
backend that serves it — same failover/busy-spill as a virtual alias; shared ids
collapse to one entry. Independent of `model_prefix`.

### Per-backend concurrency cap (`max_concurrent`)

A live per-backend in-flight counter; at/above the cap the backend is **busy** and
skipped, so the request spills to the next backend instead of overloading a slow
one. Match it to real parallelism (`1` for `llama.cpp --parallel 1`; unset for a
cloud API). Missing/`0` = unlimited. The counter is released when the response
**completes** — including when a streamed response finishes, not when headers are
sent. Busy state shows in `/health` and the **Input & Routing → Chat aliases** tab.

### Per-backend model filters

| Flag | Effect (at discovery) |
|---|---|
| `chat_only` | keep only `type == "chat"` models (drops image/video/embedding). Understands Together's `type` and OpenRouter's `architecture.output_modalities`. Backends without those fields (llama-swap, vLLM) are unaffected — so **don't** set it on a backend whose embedding models you want routable. |
| `serverless_only` | keep only models with non-zero pricing (Together's dedicated-only models are `0/0`; on OpenRouter this also drops `:free`). |

### Sampling defaults (`sampling_defaults`)

Some backends sample with bare server defaults when a request carries no sampling
parameters. vLLM without a truncation sampler (`top_p=1`, `top_k=-1`, `min_p=0`)
at temperature ≈ 1 will emit token salad — foreign-script and code fragments
spliced into the text (measured on an FP8 70B: 1.6–3.5 % non-Latin characters,
0 % with `min_p: 0.05` or `temperature: 0.85`). Clients that send nothing —
OpenWebUI, the `/ui` Chat Playground with its fields left blank — hit exactly
that.

Both a **backend** (Backends tab) and a **chat alias** (chat-alias editor) can
carry defaults, filled into a request only for keys the caller did **not** send.
The editors show one input per common sampler — `temperature`, `top_p`, `top_k`,
`min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty` — plus a
**more (JSON)** box for anything else a backend understands (`typical_p`,
`stop`, `logit_bias`, …). A decimal comma is accepted; a key that has its own
input is rejected in the JSON box, so a value can never be set twice.

Stored shape (this is also the `config.yaml` form on a backend):

```yaml
# Backend "Infermatic"
sampling_defaults: {temperature: 0.85, min_p: 0.05}
```

**Precedence: client > alias > backend.** With the backend above and an alias
carrying `temperature: 1.0`, a request that sends no sampling parameters runs at
`temperature` 1.0 (alias) plus `min_p` 0.05 (backend); a client sending
`temperature: 0.2` keeps 0.2.

Any key is allowed except `model`, `messages`, `stream`, `stream_options` and
anything starting with `_` — those drive routing, streaming and the reasoning
hand-off, and are rejected when you save. Values may be scalars, lists (`stop`)
or objects (`logit_bias`).

Applies to `/v1/chat/completions`, `/v1/completions` and `/v1/responses` — plus a
`/v1/messages` request served by a **translated** (`type: openai`) backend, where
only the backend stage runs (the alias stage is deliberately skipped, see
[Claude Code](#claude-code--anthropic-messages)). Never `/v1/embeddings`, never
`/v1/audio/*`, never generation, and never the Anthropic passthrough. The backend stage is
derived **per backend**, so a failover uses the values of the backend that
actually serves the call. The forwarded body is what gets logged, so the
**LLM Calls** tab shows exactly which values went out.

### Alias / model-name collisions

Naming an alias the same as a real model id *shadows* that model. The
**Input & Routing → Chat aliases** tab flags every collision, split into
**covered** (in the mapping → still routable) and **shadowed** (hosts the model
but isn't mapped → unreachable by that name); `/health`'s `alias_model_conflicts`
carries only the entries that actually shadow a backend — the actionable case.

---

## Call parking

When **all** backends that map an alias are busy (at their in-flight cap), the
call is **held in a FIFO queue** until a mapping backend frees (then dispatched)
instead of returning `503`. This is the **default** — no client field needed, so
callers stay plain-OpenAI. A standard request just sees a slightly slower `200`,
or a `503` (with `Retry-After`) if the wait runs out.

- **Park time is per-alias** (`park_s` in the chat-alias editor, or config
  `alias_park`): blank = the global default (`park_timeout_s`, **60 s**, Server
  tab), `0` = parking off for that alias (immediate `503` when busy). `max_parked`
  caps the queue.
- **Fair:** when a slot frees, the scheduler designates one waiter for it —
  overdue first, else the one whose type key the backend just ran, else the
  oldest it can serve (no head-of-line blocking across aliases). Live queue is
  visible in the **Parked calls** panel on the Dashboard.
- **A backend reserved for a designated waiter is not overtaken.** With parking
  off for an alias (`park_s: 0`) or a full queue, a fresh request that finds only
  such a backend free gets a `503` + `Retry-After` instead of jumping the queue.
- **On timeout** the call leaves the queue with a `503` + `Retry-After`.
- **A backend that comes back is picked up immediately.** Waiting work is never
  pinned to the backend it was queued for: a parked call re-evaluates its routes
  whenever a backend goes healthy or gains models, and a parked generation job
  re-resolves its candidates every 2 s — so a returning or newly added backend is
  used the moment it is available. To keep *noticing* it
  fast, unhealthy backends are re-polled every `fast_probe_interval_s` (**3 s**,
  Server tab; `0` = off) for as long as something is waiting, instead of only once
  per `health_check_interval`. Nothing waiting → no extra polling.

Applies to `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
`/v1/responses`, and the generation path (a busy ComfyUI backend queues the job
rather than `503`-ing). The distinction "all busy" vs. "no backend at all" is
explicit — a genuine no-backend still `503`s.

**Async LLM requests** don't use a custom field: they follow the official OpenAI
**Responses background mode** — `POST /v1/responses` with `background:true`
returns immediately with `{id, status:"queued"}`; poll `GET /v1/responses/{id}`
(→ `in_progress`/`completed`/`failed`/`cancelled`) and cancel via
`POST /v1/responses/{id}/cancel`. The background worker parks in the same queue.

---

## Reasoning control

One normalized switch turns a thinking model's reasoning **off/on** — regardless
of which mechanism the model actually needs. Clients send a single field:

```jsonc
{ "model": "tool", "reasoning": "off", "messages": [...] }   // "off" | "on" | "auto"
```

`"auto"` (or omitting the field) leaves the request untouched. The OpenAI
`reasoning_effort` field works as an alias (`minimal` → off, anything else → on),
and `/v1/responses` also accepts the `reasoning: {effort}` object shape.

**Rules decide the mechanism.** The switch is translated per (model × backend)
by an ordered rule list (UI → **Reasoning** tab; stored, hot). The first enabled
rule whose **model glob** matches the real model and whose **backend set**
contains the serving backend wins; its *adapter* does the work:

| Adapter | What it does |
|---|---|
| `enable_thinking` | sets `chat_template_kwargs: {enable_thinking: bool}` (vLLM; llama.cpp with `--jinja`) |
| `reasoning_effort` | sets `reasoning_effort` (off → `minimal`, on → `high`; overridable per rule) |
| `nothink_token` | appends a token (default `/nothink`) to the last user message |
| `prefill` | appends a closed `<think>…</think>` assistant turn |
| `none` | no mechanism — reported as `unsupported` |

No matching rule → the request is forwarded unchanged and the control is
reported as `unsupported` — **it never fails a call**. What was actually applied
comes back in the **`x-reasoning-control`** response header (e.g. `off:prefill`,
`on:noop`, `unsupported`) and is logged per call in the **LLM Calls** tab.

**Per-alias default.** A chat alias can carry a reasoning default (chat-alias
editor → `reasoning: auto|on|off`), applied when the client sends nothing — so
`tool` (off) and `tool-thinking` (auto) can point at the **same backend and
model**. An explicit client `reasoning` field always wins.

**Thinking output on `/v1/responses`.** Models that stream their thinking in the
`reasoning` delta channel are translated to Responses-API reasoning events
(`response.reasoning_summary_text.delta`) and a `reasoning` output item —
`output_text` stays answer-only; clients that don't know reasoning events simply
ignore them.

---

## Claude Code / Anthropic Messages

The gateway speaks Anthropic's Messages protocol as a **frontdoor**, so
[Claude Code](https://claude.com/claude-code) can run through it — mixing Claude
models with open-weight models behind gateway aliases, while keeping routing,
parking, failover, quotas and stats:

```bash
export ANTHROPIC_BASE_URL=http://gateway:4000
export ANTHROPIC_AUTH_TOKEN=<your gateway key>   # or ANTHROPIC_API_KEY (sent as x-api-key)
export ANTHROPIC_MODEL=claude-sub                # any gateway alias
claude
```

Two kinds of backend can serve `POST /v1/messages`:

| Backend | What happens | Why |
|---|---|---|
| `type: anthropic` | **verbatim passthrough** to `api.anthropic.com` | `cache_control` breakpoints, thinking signatures and fine-grained tool streaming survive untouched — without cache breakpoints, Claude Code re-reads the whole context at full price every turn |
| `type: openai` (OpenRouter, LocalAI, vLLM …) | **translated** by `anthropic_bridge.py` — Messages → chat and back, streaming included | one alias can list both, so a failing Anthropic backend fails over to an open-weight model |

Nothing on the passthrough path rewrites body or stream: the SSE normalizer,
the reasoning rewrite and `sampling_defaults` are all skipped there. On the
translated path they apply as usual, and Claude Code's `thinking: {type:
enabled}` maps onto the gateway's [reasoning control](#reasoning-control) so an
open-weight model thinks when asked to.

### Prompt caching and cost

Claude Code marks cache breakpoints with `cache_control`, and they are what keeps
a long session cheap — without them the whole context is billed again every turn.

- **Passthrough** (`type: anthropic`): the request body reaches Anthropic
  byte-for-byte, with the alias resolved to the real model id as the only change.
  Breakpoints, thinking signatures and tool streaming all survive.
- **Translated** (`type: openai`): breakpoints are dropped by default, because a
  chat backend has no field for them. That costs nothing for local models (no
  token billing) or OpenAI models (they cache automatically) — but it *does* cost
  money on **OpenRouter**, which forwards `cache_control` to Anthropic and Gemini
  models. Set `prompt_cache: true` on such a backend and the breakpoints are
  carried into the translated body. It stays opt-in because they turn a message
  into a content-part list, which a strict server may reject.

`POST /v1/messages/count_tokens` is passed through to Anthropic and answered from
an estimate for chat backends (they have no such endpoint).

**One caveat for mixed aliases with thinking on:** a translated backend produces
thinking blocks without Anthropic's cryptographic `signature`. Claude Code sends
those blocks back in the next turn, and if that turn lands on the Anthropic
backend, Anthropic rejects unsigned thinking blocks with a `400`. Failover
between the two is fine as long as thinking is off; with thinking on, give
Claude and the open-weight model **separate aliases** and switch with `/model`.

**Point Claude Code's background model somewhere cheap** — it runs a small model
for titles and summaries:

```bash
export ANTHROPIC_SMALL_FAST_MODEL=cheap          # a gateway alias on a local/OpenRouter model
```

Recent Claude Code versions carry one slot per model class, each taking any
gateway alias or bare model id — so you can mix providers inside a single
session and switch with `/model`:

```bash
export ANTHROPIC_DEFAULT_OPUS_MODEL=opus         # alias → your Anthropic backend
export ANTHROPIC_DEFAULT_SONNET_MODEL=sonnet
export ANTHROPIC_DEFAULT_FABLE_MODEL=fable
export ANTHROPIC_DEFAULT_HAIKU_MODEL=glm         # background model → OpenRouter/local
```

`ANTHROPIC_CUSTOM_MODEL_OPTION` (plus `…_NAME` / `…_DESCRIPTION`) adds an entry of
your own to the `/model` menu instead of overwriting one of the built-in slots.
Note that a *reasoning* model in the Haiku slot spends its budget thinking about
titles and summaries — a small local model is usually the better fit there.

### Licence boundary (please keep it)

A Claude **subscription** token (`claude setup-token`) is licensed for *your own*
use of Claude Code — not for re-serving Claude as a general-purpose API to other
clients or other people. This is enforced in the gateway, not just documented:

- an `anthropic` backend is routable **only** on `/v1/messages` (`serves_path()`);
  it can never be reached through `/v1/chat/completions`, `/v1/responses`,
  `/v1/embeddings` or the console's Playground, whatever alias points at it,
- an alias served exclusively by Anthropic backends is hidden from the Playground
  and answers other endpoints with `404 … reachable through POST /v1/messages only`,
- the Backends tab states the same rule at the credential field.

The reverse direction (Claude models behind `/v1/chat/completions`) is
deliberately **not** built — that is precisely the path that would turn a
subscription into an API. If you have a paid API key from console.anthropic.com,
set the backend's auth mode to `api key`; the same endpoint restriction still
applies.

### Getting the subscription token

`claude setup-token` mints a **long-lived (1-year)** token from your Claude
subscription. It works fine on a headless server — including the gateway box
itself — because the flow does **not** use a localhost callback: it prints a URL,
you sign in wherever you have a browser, and paste the code it shows back into the
CLI. No X server, no SSH tunnel.

On the gateway machine (Debian/Ubuntu container, root, no Node required):

```bash
# 1. install the CLI once — native binary, lands in ~/.local/bin
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"

# 2. mint the token
claude setup-token
#    "Opening browser to sign in…"
#    "Browser didn't open? Use the url below to sign in (c to copy)"
#    → copy that URL into any browser, sign in with the subscription account,
#      then paste the code it displays back into the waiting CLI
#    → prints a token starting with sk-ant-oat01-…
```

The token belongs to the *account*, not the machine — minting it on your laptop
and pasting it into the console works exactly as well. Either way it goes into the
backend's **api key** field (stored encrypted in `store.db`).

Do **not** use the `accessToken` out of `~/.claude/.credentials.json`: that one is
the short-lived session token Claude Code refreshes for itself (hours), and the
gateway has no refresh mechanism — the backend would go DOWN when it expires.

### Backend setup

In the console: **Backends → + Add backend**, type `anthropic`, url
`https://api.anthropic.com` (no `/v1` — the gateway appends the path), the token in
**api key**, auth mode `subscription`. Or in `config.yaml`:

```yaml
backends:
  - name: anthropic-sub
    type: anthropic
    url: https://api.anthropic.com
    api_key: <output of `claude setup-token`>   # sk-ant-oat01-…
    auth_mode: subscription        # → Authorization: Bearer + OAuth beta header
    # auth_mode: api_key           # → x-api-key (console.anthropic.com key)
    models: [claude-sonnet-5]      # fallback list; discovery tries GET /v1/models first
    paid: true                     # a subscription/API backend — used when no free unpaid one
```

Then check the Backends tab: the row should read **UP** with the model count
(a subscription token is served on `GET /v1/models`, so discovery finds them all).
`401 Unauthorized` in the log means the credential is wrong — a real one always
starts with `sk-ant-oat01-` (subscription) or `sk-ant-api03-` (API key).

Discovery asks Anthropic's `GET /v1/models` and falls back to `models:` when the
token isn't allowed there — so a 401 on that endpoint doesn't take the backend
down. Then map an alias to it (console → **Input & Routing → Chat aliases**, or
`virtual_models`) and hand that alias to Claude Code as `ANTHROPIC_MODEL`. A bare
model id works too: `ANTHROPIC_MODEL=claude-sonnet-5` routes straight to whichever
backend serves it.

Smoke-test it from the gateway box before pointing Claude Code at it:

```bash
curl -s localhost:4000/v1/messages -H "x-api-key: $GATEWAY_KEY" \
  -H 'anthropic-version: 2023-06-01' -H 'content-type: application/json' \
  -d '{"model":"claude-sonnet-5","max_tokens":16,
       "system":"You are Claude Code, Anthropic'\''s official CLI for Claude.",
       "messages":[{"role":"user","content":"Reply with: gateway works"}]}'
```

**That `system` line is not decoration.** With a subscription token, Anthropic
serves the stronger models only when the request identifies itself as Claude Code:
without it, `claude-sonnet-5` and `claude-opus-5` answer `429 rate_limit_error`
while the rate-limit headers report the account barely used (measured 2026-08-19:
5h utilization 0.1 with Haiku going through fine). Haiku is exempt; Sonnet and
Opus are not. So a bare curl failing with 429 is expected and says nothing about
your quota.

This matters not at all for normal use — Claude Code sends that system prompt
itself and the passthrough forwards it untouched. And the gateway deliberately
does **not** inject it on your behalf: doing so would disguise arbitrary clients
as Claude Code, which is exactly the licence boundary this backend type exists to
respect.

---

## Media generation

Generation runs on three backend types: **`type: comfyui`** — a ComfyUI server on
your own GPU — and the two cloud mesh APIs **`type: meshy`** (Meshy.ai) and
**`type: tripo`** (Tripo3D), both further down. A ComfyUI
backend speaks a different protocol, so it declares `type: comfyui`.
Discovery is via `/object_info` (checkpoints/UNETs/VAEs **and** installed LoRAs);
dispatch submits a parametrised workflow, polls `/history`, and fetches whatever
artifacts it produced — **image, video (e.g. SaveVideo), or audio** — with each
artifact's kind/mime carried through the API and rendered in the console.

```yaml
backends:
  - name: gpu-3090
    type: comfyui
    url: http://192.168.1.20:8188
    max_concurrent: 1        # one generation at a time on this GPU
    # poll_interval: 1.0     # seconds between /history polls (Backends tab)
    # max_wait: 600          # hard cap for a single generation (Backends tab)
    # read_timeout: 60       # per-HTTP-request read timeout (hung read → failover)
    # disconnect_grace: 30   # tolerated unreachability before failing over
    # stuck_after_s: 90      # executor watchdog: pending prompts + idle executor
    #                        # this long → backend goes down (see below)
    # auto_restart: true     # opt-in: restart the ComfyUI service when stuck
    # restart_cooldown_s: 600  # at most one auto-restart per this window
```

**How long a generation may take** is the gateway's call, not ComfyUI's: `max_wait`
(default **600 s**) caps one generation — the span from submitting the prompt until
its result shows up in `/history`, polled every `poll_interval` (default 1 s). On
expiry the gateway sends `/interrupt` (freeing the GPU) and fails the attempt over to
the next candidate, so the cap is spent **per candidate backend** — with two
candidates a client can wait twice that long. Raise it for slow workflows (video,
mesh, rigging). Both fields are editable in the **Backends** tab; note that a backend
managed there overrides a same-named `config.yaml` entry *wholesale*, so for
store-managed backends the tab is the only place that takes effect.

**Executor watchdog.** ComfyUI's HTTP server keeps answering even when its
prompt executor has died (e.g. after a CUDA fault) — prompts then pile up in
`queue_pending` while nothing runs, and every generation runs into its poll
timeout. The health loop therefore also checks `/queue`: the same head prompt
pending with an idle executor across ≥2 checks and ≥`stuck_after_s` (default
90 s) marks the backend **down** (`exec_stuck: true` in `/health`, "executor
stuck" badge in the Backends tab). The ⟳ action there — or `auto_restart` —
restarts the service via the **ComfyUI-Manager** reboot endpoint (requires the
Manager extension and a systemd unit with `Restart=always`); auto-restart fires
at most once per `restart_cooldown_s` (default 600 s). A GPU that fell off the
bus needs a host reboot instead — the backend then simply stays down.

### Meshy.ai (cloud mesh generation)

A Meshy backend (`type: meshy`, <https://docs.meshy.ai>) serves **image → 3D**,
**multi-image → 3D** and **rigging** through the same `POST /v1/generations` API as a
ComfyUI mesh alias — same `input_*` labels, same image slots, same job endpoints. It is
always a **paid** backend: the scheduler reaches for it only when no unpaid backend is
free.

```yaml
backends:
  - name: meshy
    type: meshy
    url: https://api.meshy.ai
    api_key: msy_…              # Meshy dashboard → API
    max_concurrent: 4           # keep it ≤ your tier's concurrent-task limit (Pro: 10)
    # poll_interval: 5          # seconds between task polls
    # max_wait: 900             # cap for one task incl. Meshy's own queue
    # disconnect_grace: 30      # unreachability tolerated while polling a task
```

Register an alias on the Meshy backend in **Mapping › Media** (no workflow JSON) and
pick the endpoint: `image-to-3d` takes `images.input_image`; `multi-image-to-3d` takes
`images.input_image_front` (required) plus optional `input_image_back`,
`input_image_left`, `input_image_right` — the slot names of the Trellis2 multiview
alias, so a client can switch by changing `model` only. On those two endpoints an
upload under `files` is refused with a `400` (images belong under `images`) — `rigging`
is the endpoint that takes a file, see below.

Client params: `input_name`, `input_face_num`, `input_texture_resolution` (pixels →
2k/4k/8k), `input_texture_prompt`, `input_pose` (`a-pose`/`t-pose`);
`input_remove_background` and `input_no_fingers` are accepted and ignored.
`input_face_num` becomes Meshy's `target_polycount` **and turns the remesh pass on**
for that request. Its default is the alias's **target polycount** option (100–300000,
blank = none): set, it is applied to every request that omits the label (and shows up
as the param's `default` in the schema); blank, Meshy's own per-model default decides
the face count. Set it on any alias that **chains into a rigger** — Meshy's rigging
endpoint refuses a mesh above 300k faces, and a no-remesh humanoid came back at 70 MB
(measured 2026-09-02), which the hand-off then has to push through as base64. Textures,
PBR, texture resolution, topology, ultra mode, delivered formats and the preview
thumbnail are alias defaults set by the admin too.

The job records what was sent (`meta.request`), the Meshy task id and
`consumed_credits`; `/health` and the Backends tab show the credit balance together
with the age of that reading (`credits 120 (3m ago)`) and the same rolling fail-rate
the ComfyUI backends carry — a balance of 0 takes the backend **down** with that
reason. A failed Meshy task is final (Meshy refunds the credits); `402` (out of
credits) and `429` (tier's concurrent-task limit) fail over to the next candidate.
While a task is polled, a persistent `4xx` (three in a row, `429` excepted) fails the
job with that status named, while transport errors, `5xx` and `429` are tolerated for
`disconnect_grace` seconds (default 30) and then fail over. Cancelling stops the
gateway's job only — Meshy finishes the task and bills it.

**Cloud rigging (`Meshy-Rig`).**
The third endpoint, `rigging`, takes a **mesh** instead of an image: a `.glb` biped
under `files.input_mesh_path` (5 credits), and gives it Meshy's own skeleton. Register
it as its own alias — `Meshy-Rig`, task `mesh2rig` — and it works standalone on any GLB,
including one a local ComfyUI pipeline produced:

```json
{"model": "Meshy-Rig", "mode": "async",
 "params": {"input_name": "Held", "input_height_m": 1.8},
 "files":  {"input_mesh_path": "data:model/gltf-binary;base64,…"}}
```

Client params are `input_name` and `input_height_m` (float, default `1.7`);
`input_no_fingers` is accepted and ignored, and **none** of the image-to-3D options
apply. Delivered artifacts are `rigged.glb` — plus `rigged.fbx` when the alias's
*deliver formats* include `fbx`, the only two formats rigging knows — and, with the
alias option **animations**, Meshy's `walking.*` / `running.*` clips as extra results.
A missing file, or one that is not a binary glTF (sniffed by magic, so a renamed OBJ
is caught), is refused *before* the task is created and costs nothing.

A Meshy alias can also be **either stage of a workflow chain** — a Meshy mesh rigged by
a local ComfyUI rigger, or a locally generated mesh rigged by `Meshy-Rig` (or by
`Tripo-Rig`, see below); see [Workflow chains](#workflow-chains-successor-aliases).

*Console note:* since Tripo joined, **one** schema-driven editor serves every cloud
alias, and its form posts to `/ui/mapping/cloud-update`. The Meshy-only
`POST /ui/mapping/meshy-update` route is **gone** — a saved script or bookmark against
it now answers `404`. Nothing changed for the form's field names, for a stored alias,
or for the public API.

### Tripo3D (cloud mesh generation + Mixamo rigging)

A Tripo backend (`type: tripo`, <https://developers.tripo3d.ai/en/docs>, **API V3**) is
the second cloud mesh backend and serves **image → 3D**, **multiview → 3D** and
**auto-rigging** through the same `POST /v1/generations` API as a Meshy or a ComfyUI
mesh alias — same `input_*` labels, same image slots, same job endpoints. Like Meshy it
is always **paid**: the scheduler reaches for it only when no unpaid backend is free.

```yaml
backends:
  - name: tripo
    type: tripo
    url: https://openapi.tripo3d.ai   # V3 only (V2 ends 2026-11-01) — the form pre-fills it
    api_key: tsk_…                    # Tripo console → API Keys
    max_concurrent: 4                 # keep it ≤ Tripo's per-account pool: 10 for the
                                      # H-series (v2.5/v3.0/v3.1) and for animation
                                      # tasks, 5 for the P-series — beyond it: 429
    # poll_interval: 2                # seconds between task polls (the docs' own advice)
    # max_wait: 900                   # cap for the WHOLE job — every task shares it
    # disconnect_grace: 30            # unreachability tolerated while polling
```

What is different from Meshy (none of it visible to a client):

- **Uploads instead of base64.** Tripo takes no inline bytes at all. Every image and
  every mesh is `POST`ed to `/v3/files` first and travels as a file token. Clients keep
  sending `images` / `files` exactly as before; the upload is the gateway's business.
- **Only GLB is native.** A generation task delivers `model.glb`, a rig task
  `rigged.<first deliver format>`. **Every other ticked format is its own convert task
  at 5 credits**, delivered as `model.<fmt>` / `rigged.<fmt>`. A convert that fails
  fails the job — a requested delivery must never silently shrink.
- **A free rig-check before every rig.** The `rig` endpoint first runs Tripo's
  0-credit rig-check; a mesh it calls unriggable ends the job *before* the rig's 25
  credits are spent, naming the rig type it did detect. The alias option **rig check**
  turns it off.
- **Mixamo-compatible skeletons.** The rig alias's **skeleton** option (`spec`) decides
  the bone names — `mixamo` by default, `tripo` for Tripo's own — and the job records
  it as `meta.rig_spec`, lifted to the job object's top-level `rig_spec` next to
  `rig: "tripo"` (the client spec's §3.1 field).
- **Animation clips by preset.** The rig alias's **animations** option is a list of
  Tripo preset names (`preset:walk`, …); each is a retarget task at 10 credits,
  delivered as its own artifact (`walk.glb`). A clip that fails or runs out of time is
  skipped with a log warning — never the rigged mesh, which is finished and paid for.
- **One `max_wait` for the whole job.** Rig-check, the main task, every convert and
  every clip share the backend's budget, so an alias with extra formats or clips needs
  a bigger one than a plain single-task alias.
- **Two rig models.** `v1.0-20240301` rigs **bipeds only** (90+ animation presets),
  `v2.5-20260210` all seven rig types (`biped`, `quadruped`, `hexapod`, `octopod`,
  `avian`, `serpentine`, `aquatic`; 16 presets). On a v1.0 alias the schema narrows
  `input_rig_type` to `["biped"]`, and a client that asks for another type has its job
  refused *before* the rig task is created (so no credits) instead of quietly receiving
  a biped skeleton.

Register a Tripo alias in **Mapping › Media** (no workflow JSON — endpoint plus admin
option defaults) and pick the endpoint: `image-to-model` takes `images.input_image`;
`multiview-to-model` takes `images.input_image_front` **plus at least one** of
`input_image_back` / `input_image_left` / `input_image_right` — Tripo refuses a
multiview job with fewer than two views, and the gateway refuses it before the upload;
`rig` takes no image at all, only the file `files.input_mesh_path`.

Client params:

| Param | Endpoint | Effect |
|---|---|---|
| `input_image` | image-to-model | the source image (required) |
| `input_image_front` (+ `_back` / `_left` / `_right`) | multiview-to-model | `front` required, at least two views in total |
| `input_mesh_path` | rig | the mesh as a **file** (`files`, not `params`) — binary glTF only, sniffed by magic |
| `input_face_num` | the two generation ones | Tripo's `face_limit`, clamped to 100 … the model's maximum (v3.1 1.5M, v3.0 1M, v2.5 500k, P-series 50k; 150k with `quad`) |
| `input_texture_resolution` | the two generation ones | pixels → Tripo's texture quality: ≤2048 `standard`, ≤4096 `detailed`, else `extreme` |
| `input_rig_type` | rig | overrides the alias's rig-type default — but only a type the **rig model** supports; anything else fails the job before the rig task is created (no credits), never a silent fallback to `biped` |
| `input_name`, `input_remove_background`, `input_no_fingers` | all | accepted and ignored (Tripo has no such field) |

`input_texture_prompt`, `input_pose` and `input_height_m` are **not** advertised for a
Tripo alias and are ignored like any unknown param — Tripo has no field for them, and
listing them would promise something the request builder never sends. Textures, PBR,
geometry quality, the face budget, quads, parts, orientation, compression, the
delivered formats and the preview thumbnail are alias defaults set by the admin.

**Cloud rigging (`Tripo-Rig`).** The `rig` endpoint takes a `.glb` under
`files.input_mesh_path` (25 credits after the free rig-check) and returns
`rigged.<format>` with the texture embedded. Register it as its own alias — `Tripo-Rig`,
task `mesh2rig` — and it works standalone on any GLB, including one a local ComfyUI
pipeline produced:

```bash
MESH="data:model/gltf-binary;base64,$(base64 -w0 Held.glb)"
jq -n --arg m "$MESH" '{model:"Tripo-Rig", mode:"async",
                        params:{input_rig_type:"biped"},
                        files:{input_mesh_path:$m}}' \
  | curl -s "$B/v1/generations" -H "Authorization: Bearer $KEY" \
         -H "Content-Type: application/json" -d @-
```

The job records the kind (`meta.cloud: "tripo"`), the primary task id
(`meta.cloud_task_id`), the endpoint, the body actually sent (`meta.request`), the
credits **summed over every task**, and `meta.tasks` — one row per billed task with its
`role` (`rig-check`, the endpoint, `convert:<fmt>`, `clip:<preset>`), id and credits, so
each one can be looked up in Tripo's own dashboard. A rig job additionally carries
`rig: "tripo"`, `rig_spec` (`mixamo` | `tripo`) and `rig_type` (what the rig-check
detected, else what was submitted). `/health` and the Backends tab show the credit
balance with the age of that reading and the same rolling fail-rate the ComfyUI backends
carry — a balance of 0 takes the backend **down** with that reason.

Errors: `403` + code `2010` (out of credits) and `429` (concurrency or request rate)
fail over to the next candidate, as do `5xx` and transport errors — including a failed
upload. Any other `4xx`, and a non-zero `code` in the response envelope, is Tripo's
verdict on the request and ends the job. While a task is polled, a persistent `4xx` or a
non-zero `code` (three in a row, `429` excepted) fails the job with that message named,
while transport errors, `5xx` and `429` are tolerated for `disconnect_grace` seconds and
then fail over. Cancelling stops the gateway's job only — Tripo has no cancel endpoint in
V3, so the task finishes and is billed.

The Tripo alias set this is built for (register them in **Mapping › Media**):

| Alias | Task | Tripo endpoint | Successor |
|---|---|---|---|
| `Tripo-Object` | `img2mesh` | image-to-model | – |
| `Tripo-Multiview` | `img2mesh` | multiview-to-model (front + ≥1 more view) | – |
| `Tripo-Humanoid` | `img2mesh` | image-to-model, `face_limit: 150000` | `Tripo-Rig` · `input_mesh_path` · `rig: tripo` |
| `Tripo-Rig` | `mesh2rig` | rig (`spec: mixamo`) | – |

### Generation aliases + mapping

A **generation alias** (the `model` of a generation request) maps to an ordered
list of candidate backends. Each candidate carries the **workflow** (a ComfyUI
**API-format JSON**) and a **mapping** that binds logical params to concrete
workflow nodes+fields:

```yaml
image_models:
  "flux":
    - backend: gpu-3090
      task: text2img
      workflow_json: { … }          # the ComfyUI API JSON (owned by the gateway)
      mapping:
        prompt:          { node: "6", field: "text" }
        width:           { node: "5", field: "width" }
        seed:            { node: "3", field: "seed" }
      fixed:                          # pinned node values (models, switches, …)
        - { node: "4", field: "unet_name", value: "flux1-dev.safetensors" }
```

The mapping is **convention-free** — it works with any workflow regardless of node
naming. (An auto-detect heuristic pre-fills it for templated workflows; the
explicit mapping always wins.) In practice you author all this in the **Mapping**
tab of the console rather than by hand: paste the ComfyUI API JSON, the gateway
owns it, auto-suggests the mapping, and gives you discovery-fed dropdowns.

Key mapping concepts:

- **Workflow + mapping are backend-independent** (shared across an alias's
  candidates). Only **Pinned values** are per-backend (one tab per backend), so
  the same alias can use a different checkpoint on each GPU while looking
  identical from outside. A request param that targets a pinned node/field is
  **ignored** — a pin is authoritative; the API can't override it.
- **Image input slots** (a `LoadImage` / `LoadImageMask` node) become file-upload
  request fields. The Mapping editor picks one of three behaviours per slot for a
  request that sends no image:
  - **`8×8 if empty`** (default) — the loader gets a black 8×8 placeholder.
  - **`required`** — the slot is left empty so ComfyUI errors clearly when a needed
    image/mask is missing (inpaint).
  - **`disable branch if empty`** — the loader node is removed together with the
    **dead branch behind it**: every node that declares that input **required** in
    `/object_info` cannot run and is removed too, transitively; a node whose socket
    is **optional** keeps running without the image. Nothing to configure — it
    follows the workflow. Example (`img2mesh-trellis2_multiview`): leaving
    `input_image_back` empty drops the back loader *and* its
    `Trellis2PreProcessImage` (whose `image` is required), while the multi-view
    generator's optional `back_image` is simply unwired and the mesh is built from
    the remaining views. If the branch would take the alias's **output node** with
    it (e.g. the *front* view, which the generator requires), the job is refused up
    front naming that slot, instead of submitting a workflow that cannot deliver.
    Removed nodes are listed as `disabled_nodes` in the job's parameter summary.

    Next to the dropdown, **`also bypass`** takes extra node ids (comma separated)
    for the same empty slot. The cascade only takes what *requires* the image, so a
    node sitting in the **main path** with an optional image socket survives it and
    then runs on nothing — an apply/switch node that exists solely for that image.
    Listed here, such a node is **bypassed** instead (ComfyUI mode 4: its consumers
    reconnect to its same-typed input), so the path behind it stays connected —
    pruning it would cut that path, which is why this is bypass and not prune. The
    ids join the backend's own **Bypass** list for one pass and show up together
    under `bypassed` in the job summary. Only stored for `disable branch if empty`.
- **Numeric fields** (strength, steps, cfg) render with `min`/`max`/`step` pulled
  live from `/object_info`.

### Workflow chains (successor aliases)

A generation alias can carry a **`successor`**: a second alias the gateway runs on the
first stage's mesh, delivering only the second stage's result (plus any
`keep_from_mesh` files of the first). Both stages may be either backend kind — the
stage-specific parts (how the mesh is exported, taken and fed) are adapter hooks:

| stage 1 | stage 2 | hand-off |
|---|---|---|
| ComfyUI | ComfyUI | the mesh's absolute path on a shared disk (`relay: path`), or an upload into the stage-2 backend's input dir (`relay: upload`) |
| ComfyUI | Meshy (`Meshy-Rig`) | the mesh bytes ride in the rigging request as its `model_url` |
| ComfyUI | Tripo (`Tripo-Rig`) | the mesh bytes are uploaded to `/v3/files` and ride in the rig request as their file token |
| Meshy | ComfyUI (`mesh-mia`, `mesh-rig-unirig`) | the mesh comes back as a result blob and is uploaded into the rigger's input dir |
| Tripo | ComfyUI (`mesh-mia`, `mesh-rig-unirig`) | as above |
| Meshy | Meshy, or Tripo (`Tripo-Rig`) | as above, bytes in the second stage's request |
| Tripo | Tripo (`Tripo-Rig`), or Meshy | as above |

A **cloud stage (Meshy, Tripo) shares no disk with anything**, so with a cloud alias on
**either** side the gateway forces `relay: upload` whatever the alias stored (the editor
hides the field for a cloud stage 1). A cloud stage 1 must deliver `glb` — otherwise the
job is refused up front, before credits are spent — and a rigging alias (Meshy's
`rigging`, Tripo's `rig`) cannot be stage 1 at all (it rigs an existing mesh and would
have no way to obtain one). `successor.mesh_param` must be a request field of the
successor: a mapped param or label on a ComfyUI alias, a **file field** on a cloud one
(`input_mesh_path` for both kinds). Stage-1 params are threaded on, so a
`Meshy-Humanoid-Cloud` request can carry `input_height_m` for the rigging stage, and a
`Tripo-Humanoid` one an `input_rig_type`.

`successor.rig` tags the delivery for the client. `mixamo`/`generic` are additionally
normalized (texture V-flip, optional JPEG) and validated at chain level; the cloud
values **`meshy`** and **`tripo`** are only tagged — a cloud rig follows its vendor's
own conventions, and re-flipping or validating it against ComfyUI-shaped rules would
only break it. (Which bone names a `tripo` delivery carries is `meta.rig_spec`.) A cloud
stage of a chain keeps its own task id, request, sub-tasks and credits on the job
(`meta.chain_stage1` for stage 1, the top-level meta for stage 2), so the Media Jobs
view shows one table per cloud stage — and the two stages may be different vendors.

The Meshy alias set this is built for (register them in **Mapping › Media**; the
successor column is the chain config above):

| Alias | Task | Meshy endpoint | Successor |
|---|---|---|---|
| `Meshy-Object`, `Meshy-Multiview` | `img2mesh` | image-to-3d / multi-image-to-3d | – |
| `Meshy-Humanoid` | `img2mesh` | image-to-3d, `pose_mode: t-pose` | `mesh-mia` · `input_mesh_path` · `rig: mixamo` |
| `Meshy-Humanoid-Multiview` | `img2mesh` | multi-image-to-3d, `pose_mode: t-pose` | as above |
| `Meshy-Humanoid-Cloud` | `img2mesh` | image-to-3d, `pose_mode: t-pose` | `Meshy-Rig` · `input_mesh_path` · `rig: meshy` |
| `Meshy-Rig` | `mesh2rig` | rigging | – |

### LoRAs

LoRAs are first-class:

- **Pinned LoRA** — a `fixed` binding on a LoRA-loader slot; the API can't change it.
- **Dynamic LoRAs** — the client sends `lora_1`, `lora_2`, … (+ optional
  `strength_N`). The gateway **cascades** them into the next *free* slots of the
  workflow's LoRA stack, never overwriting a pinned/occupied slot — so a client
  needn't know which slot is reserved.
- **LoRA-aware routing** — a backend that lacks a requested LoRA is dropped from
  the candidate set (decided over all candidates incl. busy, so the request parks
  for the backend that has it rather than spilling to one that doesn't). A LoRA
  installed on no backend is ignored (the normal ordering decides). An explicit `backend`
  pin is never overridden.
- **`GET /v1/generations/{alias}/loras`** returns the LoRA filenames valid for an
  alias (the union installed across its backends) — for building a correct picker.

### Jobs & TTL

Every generation is a **job**: SQLite metadata + on-disk artifacts under
`jobs/<id>/<n>.<ext>` (**image, video, or audio** — the artifact's kind/mime flow
through the API and the console), lifecycle `queued → running → done|failed`,
retrievable by id until its TTL (default 24 h), then pruned. The job also keeps
its **inputs** (prompt, params, reference images) so it stays inspectable in the
**Media Jobs** tab. A running job can be cancelled (`POST /v1/jobs/{id}/cancel` or the ✕ button),
which interrupts the ComfyUI prompt to free the GPU. On a restart, any job left
`running`/`queued` is reconciled to `failed`.

### Hosts & VRAM policy

Backends group by the physical box they run on (`host` field, else the URL's
host/IP — shown in `/health` and the Backends tab's **Hosts · GPU policy** panel).
Per host, four flags decide who may use its GPU and when ComfyUI's VRAM is freed.
ComfyUI never releases its model cache by itself, so the gateway does:

| Flag | Default | What |
|---|---|---|
| `comfy_free_before_job` | **on** (every ComfyUI box) | At claim time, when the job's **model set** differs from what the box holds, `POST /free` is awaited **and watched** (`/system_stats` VRAM, re-posted every 2 s) before the prompt goes out. The model set is what the workflow's loader nodes name after pins/mapping/LoRAs — two aliases on one model keep the cache, a mapped model change under one alias frees. Needed even on a dedicated box: a node that runs in its own process (rigging, Make-It-Animatable) cannot share ComfyUI's cache and OOMs on it. |
| `comfy_free_after_job` | on **iff shared** (an LLM and a ComfyUI backend on one GPU) | `POST /free` when a job ends, so the next llama-swap load does not abort on VRAM ComfyUI still holds. Skipped while another job runs there, and when the queued job the scheduler will hand this box next wants the model set it just ran. |
| `avoid_llm_during_media` | on | Chat candidates on a box with a running media job sort **last** (never dropped). |
| `llm_unload_before_media` | off | `GET /unload` on the box's llama-swap backends before a media job. |

Defaults come from one table (`scheduler.HOST_FLAGS`); only a non-default value is
stored per host. A gateway restart counts the VRAM as unknown (it never empties
ComfyUI's cache), so the first job after one frees.

### Two ways to call it

- **OpenAI Images API** (for OpenAI image clients):
  - `POST /v1/images/generations` — JSON, text→image. Bonus: LocalAI-style
    `ref_images` (base64/URL list). Extra keys pass through as workflow params.
  - `POST /v1/images/edits` — multipart; `image` file(s) + the OpenAI `mask` field
    map positionally onto the workflow's image slots. `response_format` = `url`
    (job-result URL, needs the Bearer key to fetch) or `b64_json` (inline).
  These are **synchronous** and block until the image is ready (and **park** if
  the backend is busy rather than `503`-ing).
- **Native job API** — `POST /v1/generations` with `{model, prompt, mode, params}`.
  `mode: "async"` returns `202 {job_id}`; poll `GET /v1/jobs/{id}` and fetch
  `GET /v1/jobs/{id}/result/{n}`. `mode: "sync"` blocks and returns inline.
  Files ride along in their own fields, keyed by param or label: `images:
  {param: base64|data-URI|URL}` for image slots, `files: {param: …}` for any
  other file input (a mesh to shrink/rig). A `files` entry is uploaded into the
  input dir of whichever backend runs the job — after parking and across
  failover — and the param gets that file's absolute path, so a client never
  needs a path on a backend; on a cloud backend it rides in the request
  instead (Meshy embeds it as a `model_url` data URI, Tripo uploads it to `/v3/files`
  and sends the token), so no path exists. The bytes are not kept as a job input.
  Unlike `params`, `files` is strict: unknown key or unreadable value → `400`,
  over 64 MB → `413`.
- **`GET /v1/generations/{alias}/schema`** self-describes an alias in three lists:
  `params`, `images` (loader slots with their empty behaviour) and **`files`** — the
  uploads that are not images. A ComfyUI alias lists its mapped mesh params there
  (`required: false`; the same input can also be named as a backend-side path in
  `params`), a Meshy or Tripo rigging alias lists
  `{"name": "input_mesh_path", "required": true, "accept": ["glb"]}`. It is the
  machine-readable source of truth — a client builds a valid request from it without
  out-of-band docs.

**Playground.** In the console's **Media** playground a mesh parameter takes a real
file (glb/gltf/obj/fbx/stl/ply — sent as `files`) or, as before, a path that already
exists on the backend; when both are filled the upload wins. Every upload field —
image slot or mesh param — can alternatively take an **artifact of an earlier job**
from a dropdown (results and stored reference images of the 60 most recent media
jobs), so rigging the mesh a previous job produced needs no download/upload detour.

**Input isolation (guarantee).** Every file the gateway uploads into a backend
(`images`, `files`, chain hand-off meshes) is named per job —
`gw_<job id>_<param>.<ext>` — so **no two jobs ever share input state**, not
across aliases, not across clients, not on the same backend. This is a
correctness requirement, not tidiness: ComfyUI opens an input file when the
prompt *executes*, not when it is submitted, so a name two jobs can both write
is a window in which one job's reference image is silently swapped for
another's. An upload that fails now **fails the job** instead of running on
whatever bytes were already there. After a clean success the job's input files
are overwritten with a 72-byte placeholder (ComfyUI has no delete-input API), so
they do not accumulate; after a timeout or cancel they are left alone, because
the prompt may still be running. The one deliberately shared input file is
`gw_placeholder.png`, the 8×8 filler for empty slots — its content is constant,
so overwriting it can mix nothing up. Job inputs are recorded with their
**sha256** (`input_images[].sha256`, same as `results[]`), so a client can prove
which bytes a delivered artifact was made from.

See [docs/anima-versa-integration.md](docs/anima-versa-integration.md) for a full
client-integration walkthrough.

---

## The `/ui` console

A server-rendered console mounted at `/ui` (sign in with an admin key once
locked). Tabs:

| Tab | What |
|---|---|
| **Dashboard** | live per-backend status + in-flight, parked calls, media-job counts/recent, recent LLM calls |
| **Server** | runtime + restart-required settings (API key, caps, park time/queue, `affinity_max_wait_s`, stats/jobs, TTL/prune) |
| **Backends** | add/edit/remove backends (LLM, ComfyUI, Meshy, Tripo), incl. the `paid` cost tier; the **Hosts · GPU policy** panel below the list edits the per-box VRAM flags (see [Hosts & VRAM policy](#hosts--vram-policy)) |
| **Input & Routing** | sub-tabs **Input** (what clients can call — chat aliases, generation models, endpoints), **Chat aliases** (the live alias→backend map + alias/model collisions), **LLM models**, **Media aliases**, **Image models**, **LoRAs** — all searchable |
| **Mapping** | register a ComfyUI workflow, wire its node mapping, pin values (a cloud alias — Meshy, Tripo — needs no workflow: one schema-driven editor renders its endpoint + option defaults instead); chat-alias editor (per-alias `park_s` + reasoning default) |
| **Reasoning** | the normalized-thinking rule list (model glob × backend set → adapter) + test resolver |
| **Playground** | one tab, sub-tabs **Chat** (default — chat completion through `/v1/chat/completions`), **Media** (generation via `POST /v1/generations` — image/video/audio, upload refs + mesh files, or an earlier job's artifact) and **Voice** (TTS via `POST /v1/audio/speech`, inline player + download) — all as **real API clients** (auth, routing, parking, stats all apply) |
| **Jobs & Calls** | sub-tabs **LLM Calls** (per-call history with stored request/response bodies — LLM endpoints only), **Media Jobs** (list + detail of generation jobs, inputs + outputs, within TTL, plus the media requests that were refused before they became a job) and **Voice Calls** |
| **Statistic** | the call-stats dashboard (search, aggregates, drilldown) |
| **Users** | multi-user keys, allow-lists, quotas, IP aliases |

**Live views update in place — an update never reloads the page.** Anything that
moves on its own (the Dashboard, Media Jobs, a running job's detail page, the
Backends tab while a backend drains, the Media Playground while a job generates,
the Voice sub-tab while a reference uploads) re-fetches its own URL every few
seconds and patches only the parts of the page that actually changed. So an update
never interrupts you: your scroll position, a sort order you clicked, half-typed
text in a filter or a form field, an open `<details>`, playing audio/video and the
3D viewer's camera angle all stay exactly where they were, and you can keep editing
the playground form while the job you just started renders into the column beside
it. The one case that does navigate for real is a redirect to a different page —
that is how an expired session takes you to the login form instead of pasting it
into the view you were on. Updating stops on its own once there is nothing live
left to watch (the job finished, the drain completed) — no timer keeps running in
the background, except that a server answering non-200 is retried with a doubling
backoff up to every 30 s rather than given up on. Tabs in the background are
skipped entirely and catch up the moment you switch back.

---

## Stats & routing dashboard

Opt-in SQLite call log, surfaced in the **Statistic** and **Input & Routing** tabs of the
console (no separate port — the old standalone dashboard was folded into `/ui`).
Every call records timestamp, duration, backend, source, alias, model, endpoint,
HTTP status, tokens, and USD cost.

```yaml
stats:
  enabled: false        # read at startup only — toggling needs a restart
  db_path: stats.db
  retention_days: 0     # 0 = keep forever; else prune older rows hourly
```

- **Cost** comes from each backend's pricing (cached at discovery, normalised to
  USD/million tokens — Together's per-million and OpenRouter's per-token schemas).
  Local backends → 0.
- **Source** is the authenticated user, else the `X-Source` header, else client IP
  (IP aliases give those friendly names; reverse-DNS is auto-resolved).
- **Streaming** calls record real tokens when the backend honors
  `stream_options.include_usage` (requested automatically); a backend that reports
  nothing — or all-zero usage, as LocalAI does — is replaced by gateway estimates
  (content-bearing deltas ≈ completion tokens, ~chars/4 for the prompt), never by
  `0`, so a priced backend still books a cost.
- The applied **reasoning control** is logged per call (LLM Calls tab column).
- **Refused calls are logged too** — a request turned away before any backend saw
  it (no healthy backend, park timeout, quota exceeded, unknown alias, bad key)
  appears with backend `(refused)`, its status, and the reason in the stored body:
  in **LLM Calls** for the chat endpoints, in **Media Jobs** for the image ones.
  Without that, the calls you most want to investigate were the only ones missing
  from the log. A generation that *ran* and failed is **not** in this list — once a
  job exists the job owns the outcome, with the real backend, duration and error.
- **Prompt cache** — the *By backend* table breaks the input tokens down into
  `cached` (served out of the backend's prompt cache, at a fraction of the fresh
  price), `written` (stored into it, a one-off surcharge) and `fresh` (billed in
  full), plus a **24h trend** sparkline of the hit rate. This is what tells you a
  long [Claude Code](#claude-code--anthropic-messages) session is still cheap — a
  cache that stops being hit (changed prefix, expired window) shows up as the
  trend falling to zero while input keeps climbing. Anthropic reports the split
  natively (`cache_read_input_tokens` / `cache_creation_input_tokens`);
  OpenAI-shaped backends report reads via `prompt_tokens_details.cached_tokens`.
  A backend that reports nothing shows `—` rather than zeros — "no cache
  reporting" is not the same statement as "the cache missed everything".
- Recent calls store the full request/response body (large/binary bodies on disk),
  viewable per-call, pruned with the same retention.

---

## Endpoint reference

### OpenAI-compatible

| Method | Path | Notes |
|---|---|---|
| `GET` | `/v1/models` | catalog filtered by the caller's allow-list; `?type=chat\|image` |
| `GET` | `/v1/models/{id}` | single-model lookup |
| `POST` | `/v1/chat/completions` | chat; scheduled dispatch + failover; streaming; parking; `reasoning: off\|on\|auto` |
| `POST` | `/v1/completions` | completions; same routing |
| `POST` | `/v1/embeddings` | embeddings; same routing |
| `POST` | `/v1/audio/speech` | TTS / voice cloning; same routing; binary audio passthrough (WAV …); `voice` + `params.ref_text` forward verbatim, per-alias voice defaults fill them in |
| `POST` | `/v1/responses` | Responses API ↔ chat bridge; streaming; parking; `background:true` (async) |
| `GET` | `/v1/responses/{id}` | poll a background response (queued→…→completed/failed/cancelled) |
| `POST` | `/v1/responses/{id}/cancel` | cancel a background response |
| `POST` | `/v1/images/generations` | text→image (sync); may return a video/audio URL for such aliases |
| `POST` | `/v1/images/edits` | multipart image+mask edit (sync) |

### Anthropic-compatible

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/messages` | [Claude Code frontdoor](#claude-code--anthropic-messages); verbatim on `anthropic` backends, translated on chat backends; streaming; parking; auth via `x-api-key` **or** `Authorization: Bearer` |
| `POST` | `/v1/messages/count_tokens` | native upstream count, estimated for chat backends |

### Voice cloning & the reference library

**Backend constraint (measured, not assumed):** LocalAI has **no upload API**, and
cloning-capable TTS models (e.g. `qwen3-tts-cpp-customvoice`) read `voice` strictly
as a **local file on the backend host** — base64/data-URIs are ignored and URLs are
treated as file names. (`omnivoice-cpp` ignores `voice` entirely — it never clones.)

The gateway therefore keeps a **voice reference library** (Playground → Voice):

- **Upload a WAV once** — the master copy lives on the gateway (`voiceref/`, kept
  out of git and deploys). Listen, re-ship or delete entries from the panel.
- An empty *ref text* is **auto-transcribed** by the gateway's local faster-whisper
  (CPU; `whisper_model` setting, default `small`). A backend serving a `whisper*`
  model is used as fallback.
- The file is **shipped via scp to every configured target** (one per LocalAI host
  that serves a cloning model — routing/failover may pick any of them). A target is
  `user@host:/abs/host/dir` — the **host-side** dir, e.g. the source of the docker
  bind mount (`/root/localai/models/voices`). Separately, *voice dir (model view)*
  is the path **as the model sees it** (the container path, e.g. `/models/voices`);
  that single path goes into `voice`, so it must be identical on every host.
  One-time setup per host: `ssh-copy-id` from the gateway host (root login via
  password is usually disabled — `PermitRootLogin prohibit-password`; append the
  gateway's `/root/.ssh/id_ed25519.pub` to the host's `authorized_keys` instead).
- Use an entry as `voice: "lib:<name>"` — API body, playground picker, or an
  alias's voice default; the gateway substitutes the shipped path + ref text
  (explicit client fields always win). Not-yet-shipped entries return a clear 409.

### Native generation + jobs

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/generations` | run a generation alias (sync or `mode:"async"`); per-field reference images via `images: {param: base64\|URL}`, other file inputs (meshes) via `files: {param: base64\|URL}` |
| `GET` | `/v1/generations/{alias}/schema` | alias self-description: `params`, `images`, `files` (mesh uploads), LoRA/fps info |
| `GET` | `/v1/generations/{alias}/loras` | LoRAs valid for an alias |
| `GET` | `/v1/jobs/{id}` | job status + results |
| `GET` | `/v1/jobs/{id}/result/{n}` | a result artifact (owner-gated) |
| `GET` | `/v1/jobs/{id}/input/{n}` | a stored reference image (owner-gated) |
| `POST` | `/v1/jobs/{id}/cancel` | cancel a queued/running job (interrupts ComfyUI; a cloud task — Meshy, Tripo — keeps running and is billed) |

### Other

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | per-backend health/models/`paid`/tok-s + busy/inflight + conflicts |
| `*` | `/ui/**` | the management console |

Every proxied LLM response carries **`x-gateway-backend`** (which backend served
the call) and, when a reasoning switch was requested, **`x-reasoning-control`**
(what was actually applied).

### Responses API bridge

Clients on LangChain.js (N8N's AI Agent, …) call `/v1/responses`; most backends
only speak `/v1/chat/completions`. The gateway translates request
(`input`/`instructions`/`tools` → `messages`/system/tool schema) and response
(`choices[0].message` → `output[…]`, token field renames) transparently, and
routes through the same dispatch/parking path as chat. `stream: true` is
supported (chat SSE → Responses SSE events). **`background: true`** runs the
request asynchronously per the official OpenAI pattern: it returns immediately
with a `queued` response object; poll `GET /v1/responses/{id}` until a terminal
state and cancel via `POST /v1/responses/{id}/cancel`. The background worker
parks in the shared queue (longer async window) — so long-running or busy
requests never time out the client connection. Thinking-model output arrives as
Responses-API **reasoning items/events** (see [Reasoning control](#reasoning-control)).

---

## Try it

```bash
KEY=sk-change-me ; B=http://localhost:4000

# List models (filtered by your key's allow-list)
curl $B/v1/models -H "Authorization: Bearer $KEY"

# Chat through an alias
curl $B/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"fast","messages":[{"role":"user","content":"hi"}]}'

# Embeddings
curl $B/v1/embeddings -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"embedding","input":["hallo welt"]}'

# Text→image (sync, inline base64)
curl $B/v1/images/generations -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"flux","prompt":"a red apple","size":"1024x1024","response_format":"b64_json"}'

# LoRAs valid for an alias
curl $B/v1/generations/flux/loras -H "Authorization: Bearer $KEY"

# Backend health snapshot
curl $B/health
```

---

## Running & deploying

`ai-hub.service` is an example systemd unit (assumes `/opt/ai-hub` with
`venv/` next to `main.py`):

```bash
sudo install -m 0644 ai-hub.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-hub
journalctl -u ai-hub -f
```

`deploy.sh` is an rsync-over-SSH helper (`DEPLOY_HOST=root@host ./deploy.sh`):
syncs code (excluding `config.yaml`, `venv/`), installs requirements in a remote
venv, syncs the systemd unit, restarts.

> **Secrets & data never to commit:** `config.yaml`, `store.db` (+ `secret.key` —
> they travel together, keys encrypted at rest), `stats.db*`, `jobs.db*`,
> `jobs/`, `*.key`. All gitignored.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Copyright 2026 Kai.
