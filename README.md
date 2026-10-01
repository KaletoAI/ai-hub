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
- [Managed hosts](#managed-hosts-thunder-compute-runpod-later) — on-demand GPU machines (Thunder Compute) the gateway starts and stops; ComfyUI and OpenAI-compatible backends attach to them, with model sync
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

Prerequisites: Python 3.10+ with the `venv` module, `git`, and — only for `deploy.sh` —
`rsync`. A minimal Debian/Ubuntu has none of the three:
`apt install -y git python3 python3-venv rsync` (verified 2026-09-08 on a fresh Debian 13
container with 512 MB RAM and 2 GB disk — the install needs ~450 MB of it, most of that
`faster-whisper`/`ctranslate2`/`onnxruntime` for the voice library).

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
  `show_user_keys` in Server → Runtime and only a key generated right there in the
  form is ever shown, as before.

**Bootstrap-open → locked.** With no users *and* no master key, the gateway and
console are fully open. Add an admin user (or set a master key — Server → API Keys)
to lock it down:
from the first user or master key on, the API needs a key AND the console a login.
The Users tab therefore refuses a first user that is not an admin, and refuses
deleting, demoting or disabling the last admin while no master key is set — either
would leave nobody able to sign in, or silently open the console again. A gateway
already in that state (users, but no enabled admin with a key and no master key)
stays locked; set `api_key:` in `config.yaml` (hot-reloaded) and sign in with it.

**Limits on what clients send.** Request bodies are capped at `max_body_mb`
(config.yaml, default 200, hot-reloaded; `0` = off) → `413`. A reference image or
`files` entry given as a URL is fetched by the gateway only from a PUBLIC address:
every address the host resolves to is checked, the connection goes to the checked
address, redirects are not followed and the body is capped at 64 MB. A host that
resolves to loopback/private/link-local/multicast is refused with `400` — list the
ranges you trust (a LAN NAS) in `ref_url_allow_cidrs`. `images` keys that are not an
image slot of the alias (read from its stored workflow, or its `workflow:` file; a file
the gateway cannot read filters nothing) are ignored without being fetched, and the OpenAI shims'
`ref_images` beyond the alias's slot count are never downloaded.

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
| `<backend>/current` | whatever that llama-swap backend has loaded (a backend-name grant covers it too) |

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
  backend that reports nothing or all zeros (as LocalAI does).

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
  `affinity_max_wait_s` (default **120 s**, Server → Runtime, hot-reloaded) counts as
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

### Whatever is loaded (`<backend>/current`, llama-swap)

llama-swap needs a model name on every call, and naming one swaps it in. For calls
where the model does not matter, send **`<backend>/current`**: the gateway asks
llama-swap's `GET /running` right before routing and rewrites `model` to what is
loaded **now** — it never loads anything.

- **Which loaded model**: the endpoint decides the kind, read from the llama-server
  flags in `/running`'s `cmd` — `/v1/embeddings` takes a model started with
  `--embedding`, every other endpoint one started without it (and without
  `--reranking`). So an embedding model llama-swap keeps loaded beside a chat model
  never receives a chat call. `ready` beats `starting` (a starting model still counts —
  llama-swap queues the call, no swap); among several, the one the gateway last sent
  there wins. `models_allow`/`models_deny` apply.
- **Nothing suitable loaded** → that backend is not a candidate: a busy backend with
  the right model loaded **parks** the call, otherwise `503 "no chat model loaded on
  <backend> — 'current' never loads one"`.
- **Across backends**: put `current` as the per-backend model of a chat alias
  (`egal: {llamaswap-strix: current, llamaswap-phoenix: current}`) — the scheduler
  picks a backend, each resolves to its own loaded model. A bare `current` does not
  exist (it would be ambiguous); name an alias `current` if you want one.
- Available only on backends whose `/running` answers (llama-swap); a backend that
  really lists a model named `current` keeps it. `/v1/models` lists `<backend>/current`
  without `context_length` (it changes with every swap). Allowed by a whole-backend
  grant or the exact entry `<backend>/current` — a grant for one model is not enough,
  since `current` may land on any of them.
- The **Backends** tab and the Dashboard's Backends panel show the loaded model(s) per
  llama-swap backend (`loaded` in `/health`), kind and state included.

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
| `models_allow` | keep only models matching one of these globs (comma-separated, or a YAML list): `models_allow: "gpt-*, claude-*"`. Empty/unset = no filter. |
| `models_deny` | drop every model matching one of these globs, **after** `models_allow` — so deny wins, and `gpt-*` allow + `gpt-*-embed` deny is "all the GPTs except the embedders". Empty/unset = no filter. |
| `models_extra` | **exact** ids (no globs) this backend serves but does not list, added to the discovered set: `models_extra: "Whisper-V3-Turbo-NPU2"`. Applied last, so a narrow `models_allow` can't take them back out. Empty/unset = nothing added. |

`models_allow`/`models_deny` are glob-matched case-sensitively (an exact id is just a
pattern without wildcards) and, unlike the two flags above, apply to **every** backend
type — `openai`, `anthropic`, `comfyui`, `meshy`, `tripo` — because they narrow the
discovered set itself. One source: `/v1/models`, routing, alias candidates and the
Aliases editor's checkpoint dropdowns all see the filtered set. Edit them in the Backends
tab under **Models**.

A backend that filters reports `models_filtered: {kept, total}` in `/health`, and the
Backends tab badges it — `filtered 3/6`, or the red `0 models — filter matches nothing`
when the globs match none of the discovered ids. Without that badge a whitelist typo
would leave the backend **healthy, discovered and routing nothing**, with every symptom
pointing somewhere else. The numbers are what a discovery poll measured: a backend that
has not polled successfully reports none at all, so an unreachable host is never blamed
on its filter.

`models_extra` is the one knob that goes the other way. Some servers serve more than
they publish: FastFlowLM (an AMD NPU box) answers `/v1/embeddings` and
`/v1/audio/transcriptions` while `/v1/models` lists its chat models only — so its
embedding and whisper models are reachable but, to the gateway, do not exist, and no
whitelist can bring them back because a filter only ever subtracts. List the exact ids
here and they join the discovered set: routable by alias, by bare id and by
`backend/model`, and eligible as the whisper fallback for voice-reference
transcription. They are added only once discovery has SUCCEEDED, so an unreachable
backend never advertises a model. `/health` reports them as `models_added: [ids]` and
the Backends tab badges `+2 listed manually` — a typo'd id is otherwise silent: it
routes, and the backend rejects the call.

### Context windows (`context_length` in `/v1/models`)

Every `/v1/models` entry (and `GET /v1/models/{id}`) carries `context_length` when
the gateway knows it — the number agent clients (Oh My Pi, Hermes, OpenCode) size
their prompts by. Without it a client assumes a default (Oh My Pi: 128k) and a
33k prompt to a 32k model comes back as a `400 exceed_context_size_error`. Two
sources, in this order:

1. **`model_context`** on the backend (Backends tab → *Models* → *context windows*, or
   config): one `model-glob=tokens` per line, first match wins.
   ```yaml
   model_context: |
     glm-*=32768
     qwen3.8-flash-next-ple4=131072
   ```
2. **Discovery** — read from the listing where it carries the value: OpenRouter /
   Together `context_length`, vLLM `max_model_len`, llama-server `meta.n_ctx`. A
   **llama-swap** listing has none, so the gateway also asks `/running` and reads
   each **loaded** model's `/upstream/<model>/v1/models` (never an unloaded one —
   that would load it). Learned values are remembered across unloads and restarts.

A bare id served by several backends, and a chat alias, publish the **smallest**
window among them — the client cannot pick the backend. Unknown stays absent,
never `0`. The **Input & Routing → LLM models** tab shows the value on each
backend chip (`ctx 32k`), so "why does my client think 128k?" is answerable there.

### Scan network (find backends on the LAN)

Backends tab → **Scan network** sweeps this host's own subnet(s) for servers the
gateway can register — llama-swap / llama.cpp / vLLM / Ollama (`openai`) and
ComfyUI (`comfyui`) — and lists each find with an **Add** link that opens the
backend form pre-filled (name from reverse DNS, type, url; `local` ticked). A find
already registered says *registered as `name`* instead. Nothing is ever added by
itself, and nothing is scanned unless someone clicks.

| Setting (Server → Runtime) | Meaning |
|---|---|
| `scan_cidrs` | comma-separated ranges; blank = the /24 of every IPv4 address of this host. Capped at **1024** hosts per scan. |
| `scan_ports` | ports tried on every host; blank = `8080, 8000, 11434, 8188, 1234, 5000`. |

Detection per open port, first hit wins: `GET /v1/models` (a `data` list → openai,
flavor from `owned_by`; 401/403 → openai that *needs api key*), `GET /system_stats`
(`comfyui_version` → comfyui), `GET /api/tags` (Ollama's native API). TCP connect
0.5 s, 256 in parallel, so a /24 with six ports takes a few seconds. Progress and
the result table update live.

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
  caps the queue. Async generation jobs have their own cap, `max_queued_gen`
  (Server → Runtime, default **200** queued or running; `0` = none): beyond it a new
  `mode: async` job gets `503` + `Retry-After`. A client's job `ttl_s` is capped at
  `jobs.max_ttl_s` (config, default 7 days).
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
  Server → Runtime; `0` = off) for as long as something is waiting, instead of only once
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
your own GPU, or on a GPU rented on demand as a
[managed host](#managed-hosts-thunder-compute-runpod-later) — and the two cloud mesh APIs
**`type: meshy`** (Meshy.ai) and **`type: tripo`** (Tripo3D), both further down. A ComfyUI
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

Register an alias on the Meshy backend in **Aliases › Media** (no workflow JSON) and
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

Register a Tripo alias in **Aliases › Media** (no workflow JSON — endpoint plus admin
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

The Tripo alias set this is built for (register them in **Aliases › Media**):

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
explicit mapping always wins.) In practice you author all this in the **Aliases**
tab of the console rather than by hand: paste the ComfyUI API JSON, the gateway
owns it, auto-suggests the mapping, and gives you discovery-fed dropdowns.

Key mapping concepts:

- **Workflow + mapping are backend-independent** (shared across an alias's
  candidates). Only **Pinned values** are per-backend (one tab per backend), so
  the same alias can use a different checkpoint on each GPU while looking
  identical from outside. A request param that targets a pinned node/field is
  **ignored** — a pin is authoritative; the API can't override it.
- **Image input slots** (a `LoadImage` / `LoadImageMask` node) become file-upload
  request fields. The Aliases media editor picks one of three behaviours per slot for a
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

The Meshy alias set this is built for (register them in **Aliases › Media**; the
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
  over 64 MB → `413`. Naming a backend PATH for such a file field in `params`
  instead is admin-only (`400` for a user key, unless the admin ticked *client may send
  a backend path* on that field in the Aliases media editor — `client_path: true`). A file
  field is one whose name ends in `path` (`input_mesh_path`) or whose workflow field is
  a file field, and only a value that names a file (`/`, `\`, `~` or an extension) is
  judged — `mesh_format: glb` is a setting, not a path. And a list or object is never accepted as a mapped
  `params` value, nor as `prompt`/`negative_prompt` (`400`; in ComfyUI's API format a
  list is a link between nodes).
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

## Managed hosts (Thunder Compute; RunPod later)

A **managed host** is a GPU machine rented on demand that the gateway **starts and
stops** for you — image, 3D mesh, video or LLM work on a big GPU for an evening, without
paying for it around the clock. The machine and what runs on it are two levels:

- the **host** carries the **provider** — the company whose API creates, snapshots and
  deletes the machine (today **Thunder Compute**, <https://www.thundercompute.com>;
  RunPod is planned and plugs into the same seam) — and its options (what to rent);
  the provider's **API token** is entered once per provider, not per host;
- **backends attach to it** by naming it as their `host`. The backend keeps its own
  type: a `comfyui` backend is set up, started and model-synced fully automatically; an
  `openai` backend (vLLM, llama-swap, any OpenAI-compatible server) brings a **setup
  script** and a **start command**. **One VM can carry several services** — e.g. one
  ComfyUI plus a vLLM.

For every attached backend the gateway then:

- reaches the service **only through one SSH tunnel** it supervises — an ssh
  ControlMaster with one forward per service
  (`127.0.0.1:<local port>` → `127.0.0.1:<remote port>` inside the VM), so the backend's
  URL is `http://127.0.0.1:<local port>` and the ordinary adapter does everything else
  (discovery, jobs, watchdog, VRAM policy, chat routing);
- enables it at Start and disables it at Stop (nothing polls a dead tunnel port);
- for ComfyUI, copies onto the machine **exactly the model files** the aliases that name
  this backend need — every GB on that disk bills, running or snapshotted.

**Setting one up — four steps** (the order the *Managed hosts* section shows on top):

1. **Enter the provider's API token** — the *Thunder Compute API token* row in
   **Server → API Keys** (the guide's step 1 links there). Once per provider: every host
   of that provider uses it.
2. **+ Managed host** — what to rent (GPU, vCPUs, …). The name is pre-filled
   (`thunder-1`, `thunder-2`, …) and is a label inside AI-Hub only.
3. **Add a backend on the host's card** — *+ ComfyUI on this host* /
   *+ OpenAI-compatible service on this host* open the backend form with the host
   selected, the port and a name filled in.
4. **Start** — the card's checklist says what is still missing; Start stays disabled,
   naming the reason, until nothing is.

**Never create the instance in the provider's console.** AI-Hub rents the machine
itself: Start creates the instance (with AI-Hub's ssh key), Stop takes a snapshot and
deletes it. An instance made by hand in the Thunder console cannot be reached by AI-Hub
(no ssh key) and bills on its own — the card lists it as *not managed by AI-Hub* with
its $/h; delete it in the Thunder console.

**Why SSH only.** Thunder's own port forwarding (`https://<uuid>-<port>.thundercompute.net`)
is public **without authentication** — ComfyUI behind it is code execution (the Manager)
and file read (`/view`) for anyone who finds the URL, and an LLM server is free compute
for strangers. So the gateway never opens one: after every create and restore, and
before a service is attached to a running host, it reads the instance's forwarded HTTP
ports and removes any it finds, and it never starts a service while one is open. Every
service listens on `127.0.0.1` inside the VM (the host bootstrap switches a template's
own autostart off). No new dependency — the system `ssh`, `ssh-keygen` and
`ssh-keyscan`.

**Prerequisites.**

- A Thunder Compute account and an **API token** (Thunder console → API tokens; Thunder
  shows it only once). It goes into the **Thunder Compute API token** row in
  **Server → API Keys** — **one token per provider**, used by every host of that provider;
  stored encrypted (store setting `provider_token_thunder`), never rendered back (the row
  says *set* / *not set*), never sent to a service, never in `/health`. Blank keeps it,
  *clear* removes it (refused while a host of that provider is not off — its stop would
  fail without a token; entering a new token is always possible), and a Save reaches
  running hosts at once. (Until 2026-10-01 this row and the HF token sat in the Managed
  hosts section, `POST /ui/hosts/managed/provider-token|hf-token`; those routes are gone —
  use `/ui/server/provider-token|hf-token`.) A store from before this change is migrated at startup: a
  readable per-host token becomes the provider token (unless one is set) — that of a
  host with a running instance first — and every per-host copy is removed; a dropped
  token that differs is logged as a warning naming the host.
- `ssh` / `ssh-keygen` / `ssh-keyscan` on the gateway host. The gateway generates its
  instance key `thunder.key` (ed25519, one per provider kind: `<kind>.key`) next to
  `store.db` on first need and hands the public half to every create — there is nothing
  to install on Thunder's side.
- For ComfyUI models that have no public download URL: the **LAN model source** below
  (set up in **Server → Models** — one for the whole gateway, every host uses it).

### Creating a managed host

*Backends* tab → **Managed hosts** (below the backend list) → **+ Managed host**. The form
opens by saying what happens: Start creates the instance at the provider, Stop
snapshots and deletes it — do not create one in the provider's console.

- **name** — pre-filled with the first free `<provider>-<n>` (`thunder-1`, …); a label
  inside AI-Hub only (you never enter it at the provider). `a-z`, `0-9` and `-`. It is
  the host's identity: its snapshots
  (`aihub-<name>-<stamp>`), its state record and its tunnel socket are named after it, so
  it **cannot be renamed**. A name that is already a managed host, a key of the
  *Hosts · GPU policy* table, or the host any backend derives today (from its `host`
  field or its URL — `http://gpu-a:8188` counts as host `gpu-a`) is refused, or two boxes
  would share one host policy.
- **Provider** — today only *Thunder Compute*. Fixed once saved: the
  host's state and snapshots belong to it.
- **What to rent at Start** — the provider's **options** (Thunder):

  | Field | Default | What |
  |---|---|---|
  | gpu | `a6000` | Thunder GPU type (`a6000`, `l40`, `a100xl`, `h100`) |
  | gpus | `1` | GPUs per instance; each includes 100 GB of disk |
  | vcpus | *included* (blank) | blank or `included` = the GPU configuration's included vCPUs — the smallest `vcpuOptions` entry of Thunder's `/v2/specs`, resolved at Start (the form shows it, e.g. `included (6 for l40 ×1)`, once the price list was fetched). Every vCPU above it bills extra (`additional_vcpus`), so a typed count costs more; it must be one the configuration offers |
  | template | `auto` | Thunder's image for the **first** start: `auto` = `comfy-ui` when a ComfyUI backend is attached at that start, else `base` (so a vLLM-only host does not inherit the template's ComfyUI and its bundled models) |
  | disk reserve GB | `20` | free space kept on top of the models and the install when the disk is sized |
  | ComfyUI commit | `1d61dcc3…` | the full 40-hex sha the ComfyUI bootstrap pins ComfyUI to |
  | custom nodes | `ops/thunder-nodes.default.txt` | one pack per line: `<git-url>@<commit>` or `registry:<id>@<version>`; `#` comments allowed; empty = the default list (a new host's field is pre-filled with it) |

An invalid value (a GPU the provider does not know, `1.5` vCPUs, a commit that is no
full sha) is refused on Save — a `400` with the form as typed, nothing stored — before
anything can bill. So is a vCPU count the GPU configuration does not offer
(`l40 ×1 offers vCPUs 6, 8, 12`), as far as the cached spec list knows it; with no
spec list cached yet the Save accepts it and the Start checks it before the create.
A Start never guesses a vCPU count: a failing fetch falls back to the last fetched
spec list (Thunder's own, refreshed hourly), and when there is none (or it names no
option for that GPU configuration) a blank vcpus ends the Start in `off` with
*cannot read Thunder's vCPU options for … — set vcpus explicitly or try again*, and no
instance is created. Hosts saved with an explicit count (the old default was `8`) keep
it — clear the field to switch to the included count. The host's label and GPU flags live in its row of
*Hosts · GPU policy*, like any other box; the attached backends group under the host's
name there. Managed hosts are console-only (store setting `managed_hosts`; the token is
the provider's, see Prerequisites).

### Attaching backends

From the host's card: **+ ComfyUI on this host** (shown while no ComfyUI is attached —
one per host) or **+ OpenAI-compatible service on this host** open the new-backend form
with the type set, the host selected, the remote port at the type's default and a free
name (`<host>-comfy` / `<host>-llm`, then `-2`, `-3` …). Or, in any backend form
(*Backends → Add/Edit*, General tab), choose the host in
**managed host** (the free-text **host** field is for ordinary boxes: pick
*(none / free text)* to use it). A **Service on the managed host** block appears:

- **remote port** — the port the service listens on **inside the VM**, on
  `127.0.0.1`; required, unique per host. Pre-filled with the default of the type the
  form opened with (ComfyUI `8188`, OpenAI-compatible `8000`) — switching the type in the
  open form does not change it, so check it.
- **url** becomes read-only and is **derived on Save**: `http://127.0.0.1:<local port>`,
  where the local port is the gateway's end of the tunnel, picked from `18100–18999`,
  unique across all hosts and **stable** over Saves, renames and a move to another host
  (a URL that moved would pull the backend out from under running jobs).
- **ComfyUI** (`type: comfyui`) needs nothing else: it is installed and started by the
  host's ComfyUI bootstrap. Blank output/input dirs (ComfyUI tab) become
  `/home/ubuntu/ComfyUI/output|input`.
- **OpenAI-compatible** (`type: openai`) needs:
  - **setup script** (optional) — a shell script run once per host disk and again
    whenever it changes (see below), e.g. `pip install -U vllm`. A command in it that
    reads stdin consumes the rest of the script: give it `-y` or `</dev/null`.
  - **start command** (required) — e.g.
    `vllm serve <model> --host 127.0.0.1 --port 8000`. It must listen on
    `127.0.0.1:<remote port>`.
  - **health path** (default `/v1/models`) — probed through the tunnel; `200`, `401` or
    `403` count as up (a server with its own API key answers `401` and is running).

  Once a command service is up, the gateway lists the VM's listeners (`ss -ltnH`); one on
  the service's remote port bound to anything but loopback (`--host 0.0.0.0` copied from
  a tutorial) puts a **warning** on its row — *listening on all interfaces — reachable
  from outside the VM; bind to 127.0.0.1* — because through the tunnel it looks exactly
  like a loopback bind. Nothing is stopped; fix the start command.

  **No tokens in the setup script or start command** — both are stored in plain text.
  The Hugging Face token belongs in **Server → API Keys** (*Hugging Face token*).

Refused on Save (`400`, the form as typed): a type that cannot run on a managed host
(`meshy`, `tripo`, `anthropic`), a missing or out-of-range remote port, a remote port
another service of the host already uses, a **second ComfyUI** on one host, a command
service whose file-name slug (the name reduced to `[a-z0-9-]`) another command service
of the host already uses, a missing start command, a health path with odd characters.
A **`config.yaml` backend** naming a managed host is **not attached** — its card says
*not attachable (config-defined backend — create it in the console)*; the tunnel
fields only exist on console-made backends.

**Detaching**: set the backend's host back to *(none / free text)* and type its real URL
(the old tunnel URL is refused — nothing would forward it any more), or move it to
another managed host. A running host then ends that service's process on the VM and
removes its forward. **A detach never disables the backend** — `enabled` belongs to the
host it is attached to now: Start enables what is attached, Stop disables what is
attached at that moment.

**Changes while the host runs** are applied within ~5 s, without restarting the tunnel:
a new service gets its forward added to the running ssh master (`ssh -O forward`), the
port check, its setup if needed, its start and its probe; a detached one gets its
process ended and its forward cancelled; a changed start command, setup script or
remote port restarts that one service — **after its requests in flight finished**: new
ones go elsewhere meanwhile, and its row shows `restart pending` (at most 10 min, then it
restarts anyway). A changed health path only re-probes. Other services keep serving
throughout — a
stream in flight on the vLLM is not cut because a ComfyUI was attached. A forward that
cannot be added (the local port is busy) marks only that service `down` with ssh's
reason and is retried with a backoff. Changes made while a Stop runs wait until the host
is `off` — except that a backend moved to another host (or detached) leaves the stop's
drain at once: it is not waited for and is never disabled by the old host.

**Per-service setup (command services).** The setup script is streamed over SSH on
stdin (`bash -s`, never on a command line), its output goes to the card log and to
`~/gw-svc-<slug>.setup.log` on the VM. Success records the script's **sha256** for the
host's disk — a start from a snapshot taken after it runs nothing, a start from the
template or from an older snapshot runs it again, and an edited script runs again at the
next start, or at once when the host is running. A failing script (exit ≠ 0) marks only
that service **`setup failed`** (naming its last output line and the log) — the host
stays `ready` and the other services keep running. **Re-run setup** in the service
table forces it, whatever the hash says, then restarts the service. The start command
runs inside a gateway-written wrapper `~/gw-svc-<slug>.sh` (a `flock`ed lock so a
second start is a no-op, a `while :` restart loop, log `~/gw-svc-<slug>.log`), started
with `setsid nohup`.

**Separate Hugging Face caches.** A command service runs with
`HF_HOME=~/hf-cache-<slug>` — its own cache. `~/hf-cache` belongs to ComfyUI and the
model sync alone; LLM weights never appear there, so the sync never lists them as
*unknown* files or prunes them.

### Start / Stop

The host's **card** (Backends tab → Managed hosts; live while an instance runs) shows the
phase, the provider, costs, the snapshot, the log — and the **service table**: one row
per attached backend with its type, `VM :<remote> → local :<local>`, status
(`starting`, `up`, `restart pending`, `setup failed`, `down`), the last error or
warning, and per service **Restart**
and **Re-run setup** (while the host runs and no operation is in flight).

- While the host is off, the card shows a **checklist** — ✓ / ✗ for the *Thunder Compute
  API token* (a missing one links to Server → API Keys) and an attached backend that can
  run, – for the optional *LAN model source*
  (only with a ComfyUI attached; needed only for model files no URL/catalog entry
  provides; while it is not usable the item links to Server → Models). **Start** is rendered disabled, its first reason as tooltip and as a line
  under the buttons, whenever the controller would refuse it; the refusal itself stays in
  the controller (one list — `start_blockers()` — feeds both the card and `start()`).
- **Start** (asks first) is refused without the provider's API token ("no Thunder
  Compute API token set — enter it under Server → API Keys"),
  without an attached backend, or when no attached backend can run — before any
  API call; the host stays `off`. Otherwise it enables every attached backend that can run, then
  creates an instance from the host's **newest READY snapshot**
  (`aihub-<host name>-<YYYYmmdd>t<HHMMSS>z`, all lowercase). With none yet, the **first
  start bootstraps** a fresh instance from the template (`auto`: see the options): first
  the **host bootstrap** (`ops/host-bootstrap.sh`, whatever is attached: the template's
  own autostart off, the models and node packs the template brought reported, `flock`
  and `uv` installed when missing; log `~/gw-host-bootstrap.log`), then — only with a
  ComfyUI backend attached — the **ComfyUI bootstrap** (`ops/thunder-bootstrap.sh`: pins
  ComfyUI to the commit, builds its own venv — Python 3.13, torch 2.11.0+cu130, unless
  the template's already matches — installs the node packs, ComfyUI-Manager and a
  looping `~/start-comfy.sh <port>`, and ends with a CUDA smoke test; up to a few hours;
  log `~/gw-bootstrap.log`), then each command service's setup. A ComfyUI attached later
  is bootstrapped then (at once on a running host, else at the next start). The disk is
  sized from the models the aliases need + the base install + the reserve (never below
  the snapshot's minimum or 100 GB per GPU; Thunder disks only grow). A restore from a
  snapshot takes up to **~8 min per 100 GB** (Thunder's figure). Then: port check,
  tunnel, every service started and probed, model sync, `ready`.
  The host is `ready` once the VM runs and the tunnel stands; each service has its own
  status, and routing follows each backend's normal health poll. A service whose setup
  or start fails is that service's `setup failed` / `down` — the host stays up. A
  failure of the machine itself before the instance exists ends in `off`; after it in
  `failed (<phase>)` with the instance **kept** for diagnosis — it bills until you press
  Stop. A failed host bootstrap is `failed (bootstrapping)`.
  **While a bootstrap or a service's setup runs, the card shows it live**: a step strip
  for the whole start (create → connect → set up → start services → sync models →
  ready), and for the running script its step in the script's fixed phase order (the
  node pack *i/N* during `nodes`), a progress bar, the time so far and the log's last
  line — read from the log on the instance every 15 s. The first ComfyUI setup on a
  template usually takes 15–30 min; **do not Stop it** — a Stop throws the half-done
  setup away and the next start does it all again. The red "The ComfyUI setup did not
  finish" note appears only when nothing runs any more (interrupted or failed): Re-run
  setup on the ComfyUI service, or Stop (that snapshot is marked incomplete).
- **Stop** (asks first) drains **every** attached backend (running jobs finish, new ones
  go elsewhere; the card says how many jobs it still waits for, per service), ends
  running transfers, deletes the synced files no selected alias needs any more (see
  below), takes a snapshot, deletes the instance and disables the attached backends →
  `off`. The instance only counts as gone once two fresh instance lists in a row no
  longer show it; one that stays is `failed (deleting)` — press Stop again, every step
  resumes where it stopped. The snapshot settles in the background: READY → this host's
  older snapshots are deleted (never the newest READY one); FAILED → a fault entry, and
  the previous READY snapshot stays the one the next start uses. A Stop during a Start
  **aborts** the start and stops from wherever it got (before the create: just `off`;
  before any bootstrap ran: deleted without a snapshot). A snapshot taken before a
  bootstrap finished is marked, and the next start from it runs that bootstrap again.
- **Restart** (service table) restarts one service on the running instance: its loop is
  ended and started fresh on its port (so a changed start command takes effect) at once,
  without waiting for requests in flight. For ComfyUI this is also the way out of a
  failed ComfyUI service start once the cause is fixed on the box;
  the ⟳ restart via ComfyUI-Manager works too (`start-comfy.sh` is a loop).
- There is **no auto-stop**: an instance runs, and bills, until Stop. A gateway restart
  does not touch it — the state (`host_state` store setting, one record per host name)
  is persisted on every phase change and reconciled with Thunder's instance list on
  boot: the tunnel comes back with every forward, each service is probed on its own
  forward and only one that does not answer is restarted; an interrupted stop runs on.
  The node list and the ComfyUI commit are applied by the bootstrap only — a start from
  a snapshot keeps what the snapshot holds.

**Deleting a host** (*Delete* on its card, asks first) is allowed only while it is `off`
with no operation in flight, no snapshot still settling, and **no backend naming it**
(re-point or delete those first). Its state goes; its READY snapshots **bill on** at the
provider until you delete them there by hand — another host's card lists them as
*foreign* (with their $/month), but nothing deletes them automatically, and with the last
host gone nothing lists them at all.

### Model sync (ComfyUI services)

**Selecting aliases.** An alias runs on the managed host when one of its candidates
names the ComfyUI backend attached to it — in *Aliases → Media*, add that backend as a
candidate (with its own pins and bypassed nodes, like any other). There is no second
list: every alias that names the backend is synced, every other one is not, and removing
the candidate frees its files at the next Stop. There is no model sync for command
services yet — their setup script or start command fetches what they need (into their
own HF cache).

**Model sync rules.** The sync runs when the instance comes up, when an alias's
candidates change (checked every 5 s), when the LAN source changes, and on *Sync now*.

- **What a candidate needs** is its workflow after **that candidate's** `fixed` pins and
  without its `bypass` nodes — the same two per-backend rules the adapter applies —
  so bypassing the unused one of two loader branches on the managed candidate saves that
  download. Counted are the weight inputs of loader nodes (not the per-job image, mask,
  mesh, path, video, audio loaders; not inputs that only say HOW to load —
  dtype/format/quant/mode…), plus any string input of any node ending in
  `.safetensors`, `.gguf`, `.ckpt`, `.pt`, `.pth`, `.bin`, `.onnx` or `.sft`. Empty LoRA
  slots do not count. For a loader field clients may choose (mapped), only the default is
  synced — the card says so.
- **Resolution never guesses:** a file name is looked up in the loader's own folders
  (ComfyUI's order), then as a unique suffix anywhere in the source; two hits are
  *ambiguous*, none is *missing* — the alias is blocked with the reason.
- **Hub ids and bare names.** A value like `org/repo` (a node that fetches its own
  model) is only covered by a catalog entry matching the node **class and value**;
  without one the alias is blocked and the reason names the entry it needs. A bare name
  (a variant name a node resolves itself) counts only when a catalog entry matches it.
  An alias with **no** model reference at all is `blocked: no model references known`
  until a catalog alias entry says what it needs (`"paths": []` = nothing).
- **Two roots:** `models/…` ↔ `~/ComfyUI/models/…` and `hf-cache/…` ↔ `~/hf-cache/…`
  (ComfyUI runs with `HF_HOME=~/hf-cache`, so nodes that load through the Hugging Face
  cache read the synced copy; its `snapshots/` symlinks are recreated). `hf-cache/token`
  and anything with a dot segment are never synced.
- **Where files come from:** a catalog `file` + `url` entry → the instance downloads it
  itself (`curl`, ≤ 3 at a time, resumable, sha256-checked when the entry has one, else
  by size; the HF token is attached only for `huggingface.co`/`hf.co`, and only via
  stdin, never on a command line). Everything else streams from the LAN source through
  the gateway — one stream per host, resumed from the partial file, sha256-verified on
  both sides. A file is *present* when the instance holds it at the source's size;
  partial `.part` files never are. Aliases with the fewest missing bytes go first.
- **Disk:** when the downloads do not fit, the gateway grows the instance's disk through
  the API; beyond the GPU's maximum the alias is `blocked: disk`.
- **Failures:** three attempts with backoff, then the alias is blocked with the file and
  the error, and a `sync` fault is logged; *Sync now* tries again.
- **Nothing is deleted during a session.** At **Stop**, files the gateway synced that no
  selected alias needs any more are deleted before the snapshot (the card previews
  "deleted at stop: N files, X GB"). A **blocked** alias's files are *held*, never
  deleted, until the block is fixed. Files the gateway did not put there and nobody needs
  — models the template brought, a node's own downloads — are listed as **unknown** with
  their size and deleted only when you tick them in the card. Delete template models
  before the first Stop, or every snapshot carries them. What ComfyUI ships itself is
  not listed (nor deletable there): its `put_…_here` placeholders, any empty file, and
  the stock `models/configs/*.yaml` under 1 MB.
- **Routing waits for the sync.** Until all of an alias's files are present and nothing
  blocks it, the managed candidate is out of routing **and** of the queue — other
  candidates of the alias are unaffected. An alias that runs **only** there answers at
  once with a `503` that says why:
  `models for <alias> are syncing on <backend> (12.3 of 31.0 GB)` or
  `models for <alias> are blocked on <backend>: <reason>`. Meanwhile its schema, image
  slots and LoRA list read empty (the card notes it).

**Catalog.** *Server → Models → Model-sync catalog* (the Managed hosts section of the
Backends tab only points there — it is the gateway's, not one host's):
one JSON list for every managed host (store setting `modelsync_catalog`), validated as a
whole on Save — a refused Save comes back with the text as typed and saves nothing.
Entries:

```json
[
 {"match": {"class": "SomeModelLoader", "value": "org/model"}, "paths": ["models/org/model/"]},
 {"match": {"alias": "my-rig-alias"}, "paths": []},
 {"file": "models/vae/x.safetensors", "url": "https://huggingface.co/…/x.safetensors", "sha256": "…"}
]
```

Paths start with `models/` or `hf-cache/`; a trailing `/` is a whole directory. The
catalog is seeded **once** with entries for public hub models some 3D nodes load
themselves (TRELLIS.2, Pixal3D, StableX normals, Hunyuan3D-2.1's texture stage); after
that it is yours — emptied, it stays empty. The **Hugging Face token** (for gated
Hugging Face downloads; sent only to `huggingface.co` / `hf.co`) is entered in
**Server → API Keys**, stored encrypted, never shown again: blank keeps it, *clear*
removes it.

**Public download sources.** A file the instance can download from a public URL is
fetched there, ON the instance — hundreds of MB/s instead of your uplink — and only
the rest streams from the LAN share. Where the URL comes from:

- **The share's Hugging Face cache, automatically.** A file under
  `hf-cache/hub/models--<org>--<name>/snapshots/<commit>/…` is downloaded from
  `huggingface.co/<org>/<name>/resolve/<commit>/…` (checked against its content
  sha256 when the blob name is one). Nothing to enter.
- **A URL you enter once — Check & save.** For a share file under `models/…` that is
  public somewhere (Hugging Face, a GitHub release, your own mirror), name its URL; for a
  share **directory** that mirrors a Hugging Face repo, name the repo (`org/name`). The
  gateway checks before it saves, in the background and one check at a time: it HEADs
  the URL (each redirect followed by hand, at most 5, every hop refused when it leads
  to a private address; your Hugging Face token goes only to `huggingface.co`/`hf.co`
  and only on the first request), compares the size with the share's file, and compares
  the content sha256 Hugging Face names (`X-Linked-Etag`, LFS files) with the share's
  own sha256 — hashed on the share once and remembered (`modelsrc_sha`; every LAN
  transfer stores it too). Accepted entries store the **share's** sha256, so every
  download is verified against your bytes. A URL that names no sha256 is accepted on
  its size and marked *size only*. A refusal says why: `size differs: share 7.70 GB
  (…), URL 7.50 GB (…)`, `hash differs: …`, `HTTP 404`, `redirect to a private
  address`, … (never the server's answer text). A **directory** is checked file by file
  at ONE commit (the repo's branch is resolved to its commit and stored, so the URLs
  never move): each file is accepted on its size, a file the repo lacks or holds at
  another size is left out (it keeps syncing from the LAN, and the check names why), and
  the directory is refused only when no file verifies. Its large files carry Hugging
  Face's sha256 until the share's own hash is known — hashed in the background, after
  any transfer waiting for one — and a file whose share copy then differs from Hugging
  Face's is marked *outdated* and syncs from the LAN.
- **Precedence**: a per-file entry beats a directory entry beats the automatic
  Hugging Face derivation (a mirror URL you entered wins). An entry whose stored size no
  longer matches the share's listing (the file was replaced) is *outdated* and the file
  syncs from the LAN until you check it again.
- **On the instance**: a download whose size or sha256 does not match is not retried
  (the same URL serves the same bytes), nor is an HTTP 4xx other than 408/429; for a
  file the share also
  holds, the gateway then ends that download, discards its partial file and syncs the
  share's copy instead (a `url_fallback` entry in the fault log). A URL given up on
  network trouble (5xx, 429, a dead connection — three attempts) is tried again by the
  next instance; a mismatch or a 4xx stays given up until the entry changes, you press
  *Sync now*, or a new *Check & save* (or a *remove*) of that file clears it — on every
  host, a stopped one included.

**Filling in the sources — Server → Models → Model sources.** Below the LAN model
source and the catalog, the console lists every model file a media alias needs (over
every ComfyUI backend, blocked aliases included; no running host needed) and where an
instance would get it:

- a summary — `N files · X GB public (HF auto Y GB · URL Z GB) · W GB LAN only · V GB
  outdated` — and one row per file: path, size, source badge, the aliases that need it
  (a blocked one is marked, its reason on hover), the URL (as text, never a link);
- badges: **HF auto** (from the share's Hugging Face cache), **URL ✓** / **URL ✓ size
  only** (a catalog source, sha256- or size-verified), **outdated — re-check** (the
  share's file changed since the check), **LAN only**, **URL failed — LAN** (an
  instance found the URL's bytes wrong or got a 4xx, and synced the share's copy; it
  stays so until *Sync now* or a new *Check & save*);
- sorted as a worklist: LAN only (largest first), outdated, failed, public; two or
  more LAN-only files under 1 MB in one directory collapse into one line; the filter boxes narrow it to
  one or more sources;
- **how to fill it**: enter a URL in a LAN-only (or outdated) row and press *Check &
  save*; for a `models/…` directory with several LAN-only files, name its Hugging Face
  repo (`org/name`) in the row's directory form instead — edit the directory first if
  the repo's root sits higher. *remove* drops a stored source (it asks first) — also
  one whose URL failed on an instance. A refused URL is answered with a fixed text,
  never echoed (a URL may carry a token). The
  section follows a running check or share hash live (state, progress, refusal reason,
  the files a directory check left out) and is static otherwise. A share that has not
  been listed yet shows only catalog URLs — press *List now* above.

Each host card's model-sync file list carries the same badge per file (`HF auto`,
`URL ✓`, `outdated`, `LAN`, `URL failed — LAN`), as that host's plan decided it: the
CURRENT source — where the plan would fetch the file now — not where a file already on
the disk came from (one LAN-synced before its URL entry existed reads `URL ✓`/`HF
auto`). A file that turns *outdated* after it was downloaded is not fetched again while
it is present at the same size; the badge tells you the entry needs a re-check, it does
not change what a running host holds.

The entries Check & save writes are ordinary catalog entries (you may also write them
by hand):

```json
[
 {"file": "models/vae/x.safetensors", "url": "https://example.org/x.safetensors",
  "size": 334643268, "sha256": "…", "verified": "sha256"},
 {"dir": "models/org/repo/", "repo": "org/repo", "rev": "<40-hex commit>",
  "files": {"model.safetensors": [7700000000, "…", false], "config.json": [512, null, false]}}
]
```

`size` is the share file's, `sha256` the share file's (or, with `provisional: true` as
the third value of a directory row, Hugging Face's until the share's is known),
`verified` what Check & save proved (`sha256` or `size`). The catalog editor refuses a
Save when the catalog changed since it was opened (a Check & save meanwhile) — your
text stays in the form, with the catalog as stored now shown read-only beside it to
merge from. Viewing the tab never writes the catalog: until the first Save (or a
host's first plan) the default is shown from memory.

**LAN model source.** Files without a public URL come from a model share on the LAN,
served read-only by the SSH forced command `ops/modelsrc-serve.sh` (verbs `list`,
`cat <rel> <offset>`, `sha256 <rel>`; no absolute paths, no `..`, no dot files, no
`*.log`, of the HF cache only `hf-cache/hub/…`, no symlink on the path — everything
else is refused). The
share root is env `MODELSRC_ROOT` (set in the key's `command=`, see below); share
path `<x>` is `models/<x>`, and the Hugging Face cache must be a **real directory**
`hf-cache/` inside the share (a symlinked one makes `list` fail with exit 1 — it
would otherwise silently lack the HF half of the share). The LAN card lives in **Server → Models** (one LAN source for every managed host of
every provider; the Backends tab's Managed hosts section only points there, a host card's
checklist links there, and so does the sync table's *waiting for LAN source* badge). The
share host is set in `modelsrc_host` (field at the end of the LAN card). There is
**no default**: while it is blank the LAN source is simply not configured (the card
says *LAN model source not configured — enter the share host under Server → Models*) and the gateway
never opens an ssh connection for it. **Upgrading from a build that had the
`modelsrc@…` default:** enter the host explicitly — blank is now *not configured*, and
aliases that need LAN files just wait (a pin left from the old default is ignored until
a host is set; the card then names it). The LAN card shows the gateway's public key
(`modelsrc.key.pub`, generated by the gateway) and the install instructions with that
key filled in.

**Recommended: a VM (or container) that already mounts the model share.** It needs no
new user and no root: the user that can read the share gets the script and ONE line in
its `authorized_keys`. Copy the script over first
(`scp ops/modelsrc-serve.sh <user>@<vm>:/tmp/`), then on the VM, as that user:

```bash
install -D -m 0755 /tmp/modelsrc-serve.sh ~/bin/modelsrc-serve
mkdir -p -m 0700 ~/.ssh
echo 'restrict,command="MODELSRC_ROOT=<share mount path> /home/<user>/bin/modelsrc-serve" <public key from the LAN card>' \
  >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

`restrict` (OpenSSH ≥ 7.2) turns off pty, port/agent/X11 forwarding and `~/.ssh/rc`;
with the forced `command=` the key can run only the read-only script, whatever the
client asks for. Set `MODELSRC_ROOT` to where the share is mounted on that machine (a
path with spaces is single-quoted inside the command: `MODELSRC_ROOT='/mnt/my share'`). Then set the share
host to `<user>@<vm>`.

**Alternative — only if no VM mounts the share** (not on a hypervisor if avoidable,
e.g. a Proxmox host that exports the share): a dedicated system user on the share host
itself, read access by ACL. As root there, after copying the script to `/tmp`:

```bash
install -m 0755 /tmp/modelsrc-serve.sh /usr/local/bin/modelsrc-serve
useradd --system --home /var/lib/modelsrc --shell /bin/bash modelsrc
setfacl -R -m u:modelsrc:rX <share path>
setfacl -R -d -m u:modelsrc:rX <share path>
install -d -m 0700 -o modelsrc /var/lib/modelsrc/.ssh
echo 'restrict,command="MODELSRC_ROOT=<share path> /usr/local/bin/modelsrc-serve" <public key from the LAN card>' \
  > /var/lib/modelsrc/.ssh/authorized_keys
chown modelsrc /var/lib/modelsrc/.ssh/authorized_keys && chmod 600 /var/lib/modelsrc/.ssh/authorized_keys
```

Either way the share user needs a **real login shell** (`/bin/bash`): sshd runs the
forced command through the user's shell, and with `/usr/sbin/nologin` it runs nothing —
every listing then fails as "unreachable". (`setfacl` comes with the `acl` package.)

Then **pin the host key** in the LAN card: *Fetch host key* runs `ssh-keyscan` and shows
the fingerprint (nothing is trusted yet); compare it with
`ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` on the share machine, then
*Confirm fingerprint* pins exactly that key (`modelsrc-known_hosts`; every later
connection checks it strictly). Until then aliases that need LAN files show
`waiting for LAN source (…)` and fetch nothing yet — not even their URL
files: nothing that cannot make an alias ready this session goes onto the paid disk
(aliases that need only URL files sync normally). A changed
share host drops the old listing and needs a new pin. *List now* lists the share at
once — it needs only the share machine, not a running instance — and the card then
shows `listed N files, M links · 3 min ago` (before the first listing: *not listed yet
— press List now or start a host*). The listing is cached for 10 min and re-read at
every start and on *Sync now*.

### Costs, warnings and orphans

The card shows GPU × count, vCPUs (the running instance's, else what a Start would
use; *vCPUs included* while the count is not known yet), disk, uptime, **$/h** (from
Thunder's public price list: the GPU rate + the extra vCPUs + disk beyond the included
100 GB per GPU), the
session total so far, and the snapshot (size, **$/month** while off). Snapshots and the
account's instance list are re-read every 10 minutes, the price list hourly.
An instance up for more than **24 h** puts a banner on its card **and** on the
Dashboard. Instances in the account that no managed host owns are listed on the card
with their $/h — *not managed by AI-Hub; if you created it by hand, delete it in the
Thunder Compute console* — and `aihub-…` snapshots no host owns (a deleted host leaves them behind)
with their $/month — **neither is ever deleted automatically**.
`/health` (full view) carries
`hosts_managed: {<host>: {provider, phase, uptime_s, cost_per_h, services: {<backend id>: <status>}}}`
— never the token. Failures land in the [fault log](#backend-fault-log): the machine's
(create, snapshot, delete, an instance that vanished) under the pseudo backend
`managed-host:<host>`, a service's (its setup, its start, a model transfer) under that
backend — sources `lifecycle` and `sync`.

### Known limits — verify on the first live run

- Not yet checked against the live API: the exact instance status values; whether the
  listed `port` is the SSH port; which id form `/instances/{id}/delete|modify` and the
  port removal accept (the gateway tries the uuid first and the index only on a `404` —
  a `400`/`422` for the wrong form would not fall back); that deleting an already-gone
  snapshot answers `404` (treated as "gone"); the `additional_vcpus` share of the $/h
  against a real invoice; and whether the 3D CUDA extensions run on Thunder's GPUs (the
  ComfyUI bootstrap's smoke test says so).
- `flock` is required on the instance (the service loops, the lock-based stop and the
  LAN append lock use it); the host bootstrap installs it via `sudo -n apt-get` when the
  image lacks it — an image without passwordless sudo and without `flock` fails the host
  bootstrap.
- On the first sync, check `~/comfy.log` on the instance: the 3D nodes should read the
  synced `models/…` / `hf-cache/…` copies and download nothing from Hugging Face. A node
  that does download puts its files on the disk as *unknown*.
- A catalog directory entry syncs the whole directory: `facebook/dinov2-giant` brings
  its weights twice (`.bin` and `.safetensors`, ~4.5 GB extra), and the TRELLIS.2
  directory ~1.5 GB of checkpoints only another pipeline loads. The catalog cannot
  exclude single files inside a directory yet.
- While a bootstrap or setup script runs, the card shows its phase and LAST log line
  (polled every 15 s); the whole output reaches the card's log only when it ends.
- One LAN stream per host: the first sync of a large alias set is bounded by the share
  host's uplink.
- The tunnel's control socket path must fit a Unix socket (at most 86 bytes). It lives in
  `<datadir>/<kind>-ctl/` next to `store.db`; a data directory too long for that moves it
  to `/tmp/ai-hub-<uid>-ctl/` (0700, private to the service under `PrivateTmp`). A path
  that fits neither leaves the tunnel down, and the card says so ("Tunnel will not come
  up: …").
- A running service that a later edit makes unrunnable (e.g. its remote port now collides) shows `down` but keeps
  running on the VM until the next Stop.
- A failed setup script's last output line is shown on the card and in the fault log —
  one more reason to keep secrets out of it.
- No model sync for LLM services; one ComfyUI per host; RunPod and a provider "stop" that
  keeps the volume are not built yet.

The instance key, the LAN key, the known_hosts files and the tunnel sockets
(`thunder.key*`, `thunder-known_hosts/`, `thunder-ctl/`, `modelsrc.key*`,
`modelsrc-known_hosts`) live next to `store.db`; they are gitignored and excluded by
`deploy.sh`, so a deploy never deletes them.

---

## The `/ui` console

A server-rendered console mounted at `/ui` (sign in with an admin key once
locked). Ten failed sign-ins from one address within five minutes block the form for
that address until the window has passed (429); behind a reverse proxy that address is
the proxy's. Served over HTTPS (directly or with `X-Forwarded-Proto: https`), the
session cookie is marked `Secure`. Tabs:

| Tab | What |
|---|---|
| **Dashboard** | live per-backend status (a down backend names its cause) + in-flight, a **backend faults · 24h** card, column and panel (see [Backend fault log](#backend-fault-log)), parked calls, media-job counts/recent, recent LLM calls |
| **Backends** | add/edit/remove backends (LLM, ComfyUI, Meshy, Tripo), incl. the `paid` cost tier; the editor is split into **General** (name, type, url, host, cost tier, concurrency, credential — never shown again once stored: blank keeps it, *clear* removes it), **Models** (whitelist/blacklist, discovery filters, bare-id listing, context windows), **Behavior** (prompt-cache passthrough, sampling defaults, self-retries) and one tab named after the type (**ComfyUI** / **Cloud task API** / **Anthropic**); the **Hosts · GPU policy** panel below the list edits the per-box VRAM flags (see [Hosts & VRAM policy](#hosts--vram-policy)); the **Managed hosts** section below the list adds a rented GPU machine (the setup guide — step 1 links to **Server → API Keys** for the provider token — then **+ Managed host**: name, **Provider**, what to rent) and carries one lifecycle card per host (a what-Start-needs checklist, Start/Stop, *+ ComfyUI / + OpenAI-compatible service on this host*, costs, service table with Restart / Re-run setup, model sync, log) — per-host things only: a line points to **Server → Models** for the LAN model source and the model-sync catalog; a backend attaches through the **managed host** select in its General tab (see [Managed hosts](#managed-hosts-thunder-compute-runpod-later)) |
| **Input & Routing** | sub-tabs **Input** (what clients can call — chat aliases, generation models, endpoints), **LLM models**, **Image models**, **LoRAs** — all searchable |
| **Aliases** | sub-tabs **Chat** and **Media** — the alias list on the left; with nothing picked the right column is the LIVE overview (chat: alias → backend · model · status + alias/model collisions; media: alias → backends, or pick a backend to see everything mapped onto it); pick an alias for its editor. Chat editor: per-alias `park_s`, reasoning/voice/sampling defaults, backends — plus that alias's live routes. Media editor: register a ComfyUI workflow, wire its node mapping, pin values (a cloud alias — Meshy, Tripo — needs no workflow: one schema-driven editor renders its endpoint + option defaults instead). Old `/ui/mapping?…` and `/ui/routing?sub=chat|gen` links redirect here. |
| **Reasoning** | the normalized-thinking rule list (model glob × backend set → adapter) + test resolver |
| **Playground** | one tab, sub-tabs **Chat** (default — chat completion through `/v1/chat/completions`), **Media** (generation via `POST /v1/generations` — image/video/audio, upload refs + mesh files, or an earlier job's artifact) and **Voice** (TTS via `POST /v1/audio/speech`, inline player + download) — all as **real API clients** (auth, routing, parking, stats all apply) |
| **Jobs & Calls** | sub-tabs **LLM Calls** (per-call history with stored request/response bodies — LLM endpoints only), **Media Jobs** (list + detail of generation jobs, inputs + outputs, within TTL, plus the media requests that were refused before they became a job) and **Voice Calls** |
| **Statistics** | **Backend faults · last 24h** (per backend + every message, bundled — shown even with stats off), then the call-stats dashboard (search, aggregates, drilldown) — empty until `stats.enabled: true` (the example config ships it `false`; set it in `config.yaml` or Server → Restart, then restart — the flag is read at startup only). The same switch feeds the LLM Calls list under Jobs & Calls. |
| **Server** | sub-tabs **Runtime** (default — applied on Save: caps, park time/queue, `affinity_max_wait_s`, probe interval, scan ranges, flags), **Restart** (port, stats/jobs, TTL/prune — take effect on the next restart; a pending restart is badged on the Restart sub-tab from every sub-tab), **API Keys** (every secret the server holds, each its own Save, never shown again — blank keeps it: the master API key, one API token per managed-host provider, the Hugging Face token; user keys live in Users) and **Models** (what the model sync of every managed host reads: the LAN model source — share host, public key, install instructions, host-key pin, *List now* — and the model-sync catalog — whose Save is refused, your text kept, when the catalog changed since the form was opened — and below them **Model sources**, the per-file worklist of public URL vs LAN only with *Check & save* / *remove*, live while a check runs; each action comes back to this sub-tab with its answer, a refused Save with the value as typed). A number that does not parse (`1.5` in a whole-number field, `abc`) is refused with the form as typed — blank is the only "default" |
| **Users** | multi-user keys, allow-lists, quotas, IP aliases |

**Live views update in place — an update never reloads the page.** Anything that
moves on its own (the Dashboard, Media Jobs, a running job's detail page, the
Backends tab while a backend drains or a managed host runs, the Media Playground while a job generates,
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
skipped entirely and catch up the moment you switch back. A chip at the right of
the tab bar says what the page shows: **live · 4s**, or **stale since 14:35:07**
(the server answered with an error) / **offline since 14:35:07** (it could not be
reached) — the time of the last good update, i.e. how old the numbers are. The
Media Jobs list is always live (every 15 s when idle, so a job started from another
client shows up without F5), sortable like the other lists, and pages back through
older jobs 100 at a time (`older →`).

---

## Backend fault log

A backend's live status only says what it is doing NOW: the moment the next health
poll succeeds its error is gone, a chat call that failed over to another backend is
logged as a 200, and a media job that crashed on one ComfyUI and then finished after a
retry is a clean `done`. So a box that fell over five times in twenty minutes looks
perfectly healthy by the time anyone opens the console. The fault log keeps those
failures — always on, independent of `stats.enabled`:

| source | recorded when |
|---|---|
| `health` | a discovery poll sees the backend FAIL (once per outage, with the cause: `timeout`, `auth`, `not_found`, `upstream`, `stuck`, …); coming back UP closes the outage and stores its length |
| `call` | a chat dispatch fails over (connect error/timeout, llama-swap "unable to start process") or returns a 5xx to the client — with the backend's own error text |
| `job` | a generation attempt fails on the backend (`connection_lost` mid-job, `max_wait`, execution error) — also when a self-retry or another backend then completed the job |
| `watchdog` | a ComfyUI service restart (auto or manual) and a failed restart |
| `lifecycle` | a managed host: a snapshot that FAILED (`snapshot_failed`), an instance that vanished outside a stop (`instance_vanished`), a failed start/stop step (booked on the pseudo backend `managed-host:<host>`), a service's failed setup or start (booked on that backend) |
| `sync` | a managed host's model transfer that gave up after three attempts (`transfer`) |

**A backend that is switched off is not a fault.** `unreachable` — no connection at all:
host powered off, service stopped — is left out everywhere: no outage, no downtime, no
failed-over call, no job attempt that never connected. The live status still shows it.
A connection that drops WHILE a job runs is booked as `connection_lost` — the box died
mid-work, which is a real error.

The **Dashboard** shows a *backend faults · 24h* card, a per-backend column and a panel
listing every backend that failed in the last 24h (faults, outages, downtime incl. an
outage still open, the last error). **Statistics → Backend faults** adds every message of
the window, bundled: messages that differ only in ids and numbers are one line with a
count, first and last time. Hosts appear under their label from the Hosts panel.
`/health` carries `faults_24h: {faults, outages, downtime_s}` per backend.

```yaml
faults:                 # optional — these are the defaults; read at startup only
  db_path: faults.db    # SQLite; if it cannot be opened the log stays in memory
  retention_days: 7     # pruned hourly
```

---

## Stats & routing dashboard

Opt-in SQLite call log, surfaced in the **Statistics** and **Input & Routing** tabs of the
console (no separate port — the old standalone dashboard was folded into `/ui`).
Every call records timestamp, duration, backend, source, alias, model, endpoint,
HTTP status, tokens, and USD cost.

```yaml
stats:
  enabled: false        # read at startup only — toggling needs a restart
  db_path: stats.db
  retention_days: 0     # 0 = keep forever; else prune older rows hourly
  body_retention_days: 14   # request/response bodies go after 14 days, rows stay (0 = keep)
  body_max_kb: 256      # per side; larger bodies keep their first and last 128 KB
```

- **Cost** comes from each backend's pricing (cached at discovery, normalised to
  USD/million tokens — Together's per-million and OpenRouter's per-token schemas).
  Local backends → 0.
- **Source** is the authenticated user, else the `X-Source` header, else client IP
  (IP aliases give those friendly names; the Users page offers reverse-DNS names and
  stores them on *Save resolved names*).
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
- Recent calls store the request/response body on disk (gzip, `calls/<id>.json.gz`),
  viewable per-call. A body side larger than `body_max_kb` keeps its head and tail
  only — a Claude Code turn re-sends the whole context, ~1 MB per call — and bodies
  are deleted after `body_retention_days` (default 14) while the row, and with it every
  aggregate and the monthly cost quota, stays. A **refused** call stores its reason but
  not its request (the list's preview column still shows what was asked). All stats
  settings are read at startup only.

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
| `GET` | `/health` | liveness (`status` + backend counts) for anyone; with an admin key (Bearer or `x-api-key`), a `/ui` session, or in bootstrap-open mode the full snapshot: per-backend health/`models_count`/`paid`/tok-s + busy/inflight + hosts + conflicts; `?verbose=1` adds every backend's model ids |
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

# Backend health snapshot (full view needs an admin key once the gateway is locked;
# add ?verbose=1 for every backend's model ids)
curl $B/health -H "Authorization: Bearer $KEY"
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

The unit runs as `root` (the voice-reference ship uses root's SSH key) inside a systemd
sandbox that works for root: `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=full`
(/usr, /boot, /etc read-only), the kernel/cgroup/clock protections, `RestrictSUIDSGID`,
`RestrictNamespaces` and a socket-family allow-list. It deliberately leaves `ProtectHome`
off (`~/.ssh`, the faster-whisper cache in `~/.cache`) and uses `full`, not `strict`
(the DBs, `jobs/`, `voiceref/` live in `/opt/ai-hub`). Running as a dedicated user is
possible but needs that user to own `/opt/ai-hub` and to hold its own SSH key on every
voice host — a decision for the operator, not the unit file.

`deploy.sh` is an rsync-over-SSH helper (`DEPLOY_HOST=root@host ./deploy.sh`):
syncs code (excluding `config.yaml`, `venv/`, the databases, `jobs/`, the managed-host/LAN
ssh keys, known_hosts files and tunnel sockets), installs requirements in a remote venv, syncs the
systemd unit, restarts.

**Upgrading an install from before 2026-09-08** (the rename) — do this BEFORE the
first `deploy.sh`, because a deploy into a fresh `/opt/ai-hub` brings none of the
excluded state with it (`config.yaml`, `store.db`, `secret.key`, `jobs/` are never
synced), and the `secret.key` a fresh install generates cannot decrypt the old
store's API keys. So MOVE the install, never re-create it beside the old one:

```bash
sudo systemctl stop llm-gateway.service
sudo mv /opt/llm-gateway /opt/ai-hub
sudo rm -rf /opt/ai-hub/venv          # its shebangs carry the old absolute path
DEPLOY_HOST=root@host ./deploy.sh     # rebuilds the venv, installs ai-hub.service
sudo systemctl disable llm-gateway.service
sudo rm /etc/systemd/system/llm-gateway.service
```

**Upgrading to the 2026-09-23 review release.** No manual migration: `stats.db` gets a
new index (`idx_calls_ts_backend`, the old `idx_calls_ts` is dropped) and `faults.db` a
`bkey` column on first start, both in place. What an operator will notice:

- **Console sign-in.** Every `/ui` session ends once — sign in again. The console now
  locks as soon as ANY user or a master `api_key` exists (before: only with an admin);
  the Users tab refuses a first user that is not an admin, and refuses removing,
  demoting or disabling the last admin while no master key is set.
- **Console actions are POST-only.** Old bookmarks or scripts that fired an action by
  GET (`/ui/backends/delete?…`, `/ui/job/<id>/cancel`, …) no longer run it: they get a
  `405` console page. A link into `/ui` from another site (or another port on the same
  host) lands on an intermediate page with a *Continue* link first. The Users page no
  longer stores reverse-DNS names by itself — they are offered as *(resolved)*, and
  *Save resolved names* stores them.
- **`/health` without an admin credential** answers `status` + backend counts only; the
  full snapshot needs the master or an admin key (or a console session), and model ids
  appear only with `?verbose=1` — adjust monitoring that parsed the old body.
- **Generation requests:** a reference image or `files` URL pointing at a private /
  loopback / link-local address is `400` until its range is listed in
  `ref_url_allow_cidrs`; a backend PATH in a file field's `params` is admin-only unless
  the field ticks *client may send a backend path* (`client_path`) in the Aliases media editor;
  a list or object in `params`, `prompt` or `negative_prompt` is `400`; an unreadable
  reference image is `400` instead of a silent placeholder.
- **Limits:** request bodies capped at `max_body_mb` (default 200 → `413`), async
  generation jobs at `max_queued_gen` (default 200 queued/running → `503`), a client's
  `ttl_s` at `jobs.max_ttl_s` (7 days).
- **Stats:** stored request/response bodies are now gzipped, capped at `body_max_kb`
  (256 per side, head + tail kept) and deleted after `body_retention_days` (14; the call
  rows stay) — the first start prunes older bodies. Statistic aggregates may be up to
  30 s old, the fault summary up to 5 s; at most 60 `401` rows per minute are logged.
- **Chat dispatch:** a `ReadTimeout` on a `paid` backend answers `504` instead of failing
  over (it was buying the answer twice); the connect timeout is 10 s (was 300 s), and
  every transport error (a reset, a closed keep-alive) now fails over. Client headers
  such as cookies, `x-forwarded-*`, `origin`/`referer`/`sec-*` and `accept-encoding` are
  no longer forwarded to backends. A stream the upstream breaks mid-answer ends with an
  in-band error event instead of a cut connection; aborted and dropped streams are
  booked (499/502) and count toward the cost quota.
- **Generation jobs:** a cloud task (Meshy/Tripo) is never created a second time once it
  exists — no failover or self-retry after that point, timeouts and outages end the job;
  a sync generation keeps running when the client disconnects (it is a job; poll it);
  a cancel stops only that job's own ComfyUI prompt and may take up to 15 s to return.
- **Service unit:** `ai-hub.service` gains a systemd sandbox (still `root`, see above);
  `deploy.sh` installs it — after the restart, check the voice-reference ship, *Scan
  network* and whisper transcription once.

> **Secrets & data never to commit:** `config.yaml`, `store.db` (+ `secret.key` —
> they travel together, keys encrypted at rest), `stats.db*`, `jobs.db*`,
> `jobs/`, `*.key`, and the managed-host files `thunder.key*`, `thunder-known_hosts/`,
> `thunder-ctl/`, `modelsrc.key*`, `modelsrc-known_hosts`. All gitignored.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Copyright 2026 Kai.
