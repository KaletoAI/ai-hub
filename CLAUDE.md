# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

Renamed from llm-gateway on 2026-09-08 (prod: /opt/ai-hub, unit ai-hub.service).

## What this is

An OpenAI-compatible reverse proxy that fans one endpoint out across many
backends: local LLM servers (llama.cpp / llama-swap / vLLM / Ollama), cloud APIs
(Together.ai / OpenAI / OpenRouter), **and ComfyUI image-generation servers**. It
does per-backend auto-discovery, one queued scheduler (fastest free unpaid backend,
freed-backend type affinity, overdue guard — `scheduler.py`) with failover, virtual aliases,
per-backend concurrency caps, an optional multi-user auth layer, **call parking**
(a default FIFO queue instead of 503 when busy; per-alias park time; async via the
Responses background mode), a full **media-generation subsystem** (image/video/
audio; workflow mapping, LoRAs, jobs), a **normalized reasoning toggle** (one
`reasoning: off|on|auto` control mapped to the right per-(model,backend)
mechanism, plus per-alias defaults), and a server-rendered `/ui` console. Read
`README.md` first — it documents every config knob, endpoint, and routing rule.

## Run / develop

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml          # then edit backends + api_key
venv/bin/uvicorn main:app --host 0.0.0.0 --port 4000   # add --reload for dev
```

- `config.yaml` is gitignored and **hot-reloaded** on save (watchfiles) — no
  restart for backend/alias changes. Read **only at startup**:
  `stats.enabled` and the stats/jobs DB paths.
- **No linter or build step, and no blanket test suite** — only targeted stdlib
  `unittest` files for the mechanisms that fail SILENTLY (see the six listed under
  `anthropic_bridge.py`): `venv/bin/python -m unittest discover -s tests -t .`.
  Everything else is verified by running the server and hitting endpoints with
  `curl` (README "Try it"), `curl localhost:4000/health` for a routing snapshot, or
  compile-gating (`venv/bin/python -m py_compile *.py`) before deploy.
- `requirements.txt` omits `watchfiles`; it ships with `uvicorn[standard]`. Keep it.
- Deploy with `DEPLOY_HOST=root@host ./deploy.sh` (rsync/tar over SSH, remote venv
  install, systemd sync, restart). rsync IS present on both dev and the prod box
  (re-checked 2026-08-18; an older note claiming otherwise was stale). Always
  compile-gate first — a broken file fails the restart silently.

## Architecture

Fifteen self-contained Python files hold everything (`ls *.py` is the count of
record; the tests live in `tests/`). `main.py` owns app state; the others
(`adapters`, `meshy`, `tripo`, `cloudtask`, `jobs`, `store`, `stats`, `admin`,
`reasoning`, `scheduler`, `responses_bridge`, `anthropic_bridge`,
`openai_image_bridge`, `previewanim`) never import `main` — they receive what
they need via injected callables, staying hot-reload-safe.

- **`main.py`** — config loading, health/discovery loop, routing, all HTTP
  endpoints, auth/quotas, call parking, generation orchestration, the Responses
  bridge. Module-level globals are the entire app state: config-bound
  (`backends`, `virtual_models`), discovery-bound (`backend_models`,
  `backend_pricing`, `backend_loras`, `backend_healthy`), live (`backend_inflight`,
  `_gen_tasks`, `users`/`_users_by_key`), and `backend_adapters`. `load_config()`
  rebinds config globals, `refresh_backend()` populates discovery ones, and
  `build_backend_adapters()` (re)binds one adapter per backend.
- **`adapters.py`** — the pluggable per-backend protocol seam. `BackendAdapter`
  ABC; `OpenAIAdapter` (`dispatch()` forwards chat/completions/embeddings,
  owns the in-flight counter incl. the streamed-`finally` decrement; streamed
  chat SSE is rewritten to strict OpenAI shape by `_StreamNormalizer` — no
  null-valued delta keys, the terminal usage chunk only for clients that sent
  `stream_options.include_usage`; strict clients like Hermes abort on the raw
  LocalAI shape; a backend's `stream_reasoning_as_content` model-globs relabel
  stream `reasoning` deltas as `content` — LocalAI marks EVERY delta as
  reasoning when the rendered prompt contains a thinking marker, e.g. Gemma-4's
  pre-closed `<|channel>thought` tail, although the model can only emit plain
  answer text then) and
  `ComfyUIAdapter` (`type: comfyui`; `discover()` via `/object_info` →
  models + **installed LoRAs**, plus an **executor watchdog** via `/queue`:
  same head prompt pending with an idle executor across ≥2 checks and
  ≥`stuck_after_s` (default 90) → `ComfyExecutorStuck` → the normal DOWN path
  in `refresh_backend` (ComfyUI answers HTTP even when its prompt worker died —
  measured 2026-07-30, CUDA fault); `restart()` = ComfyUI-Manager
  `/manager/reboot` (transport error on the POST is the expected success
  signal, 404 = Manager missing, then ≤120 s wait for `/object_info` to
  return); `refresh_backend` triggers opt-in `auto_restart` per backend
  (`restart_cooldown_s`, default 600, one attempt per cooldown,
  `_comfy_restarting` guard; deliberately not gated on inflight — stuck means
  nothing executes). `exec_stuck`/`last_restart*` surface in `/health` + the
  Backends tab (⟳ restart action). `generate()` submits a parametrised
  workflow, polls `/history` every `poll_interval` (default 1) until the backend's
  `max_wait` (default 600) — the gateway's cap on ONE generation; ComfyUI itself has
  none — then `/interrupt`s and raises `TimeoutError`, and fetches `/view`. Both
  fields are edited in the Backends tab: a store backend replaces a same-named config
  entry WHOLESALE (`rebuild_backends`), so config.yaml cannot supply them for a
  UI-managed backend. `TimeoutError` sits in `_GEN_FAILOVER_ERRORS` so it fails over,
  but `_fault_label`/`_gen_exhausted_msg` keep it NAMED apart from a real connection
  fault and from an httpx transport timeout — reporting a `max_wait` expiry as
  "unreachable (connection)" sends you diagnosing the network instead of the
  workflow, and the cap is spent PER CANDIDATE.) `AdapterContext` injects app services.
  `AnthropicAdapter` (`type: anthropic`, subclasses `OpenAIAdapter`) serves
  **`/v1/messages` only** — a licence boundary enforced in routing
  (`main.serves_path`), not just documented: a Claude subscription token covers
  Claude Code, not re-serving Claude as an API. It is a VERBATIM passthrough, so
  three things that run for every other backend must not run there —
  `_StreamNormalizer`, `apply_reasoning()` and `sampling_defaults` — else
  `cache_control` breakpoints (full-price context re-reads without them), thinking
  signatures and fine-grained tool streaming are lost. The three seams that make
  this a subclass instead of a fork: `_payload()` (verbatim vs. sampling+reasoning),
  `_backend_auth()` (`x-api-key` vs. OAuth bearer + `oauth-2025-04-20` beta appended
  to the CLIENT's beta list) and `_usage_of()` (`input_tokens`/`output_tokens`
  incl. cache reads, else every subscription call books 0 tokens). The same
  `OpenAIAdapter` also serves `/v1/messages` for chat backends by calling
  `anthropic_bridge` in both directions, which is what lets ONE alias hold an
  Anthropic backend AND an OpenRouter model with normal failover between them.
  `_HOP_BY_HOP` drops `x-api-key` alongside `authorization` — both are GATEWAY
  credentials (Claude Code sends the former), and forwarding either would hand the
  caller's key to the backend.
  Going the other way, every response builder keeps only its OWN headers
  (`call.rheaders` = `x-gateway-backend` + `x-reasoning-control`) — an upstream
  `content-length` would describe a body the gateway re-serializes — with ONE
  exception: `_ratelimit_headers()` carries the upstream's `retry-after` through,
  because that header is not diagnostics but an instruction to the caller. It is
  merged in at all four places that rebuild a response from an upstream one
  (`_dispatch_once`, both stream error paths, `_anthropic_error`), covered by
  `test_ratelimit_headers.py`. Dropping it fails silently — the client still gets
  its 429 and just retries blind: measured 2026-09-01 on prod, one Claude Code
  request became ~10 upstream calls in 20 s against `api.anthropic.com`, which had
  said exactly how long to wait.
  Workflow injection is **mapping-driven, convention-free** (`_apply_mapping`
  sets `workflow[node].inputs[field]`); a mapping `label` is the param's public
  API name — incoming values are accepted under label OR param, and the
  auto-random seed keys on that effective name (`''` counts as unset);
  `_apply_lora_cascade` drops client LoRAs into free stack slots; `_apply_fixed`
  applies admin pins (the API can't override a pinned `(node,field)`);
  `_apply_bypass` runs LAST (per-backend `bypass` node ids): ComfyUI mode-4 —
  remove each node and reconnect its consumers to the same-typed input (k-th
  same-typed output → k-th same-typed link input, via cached `/object_info`
  slot types; single-link fallback when types are unknown);
  an image slot mapped `on_empty: disable` and left empty prunes a whole DEAD
  BRANCH (`_prune_branch`, not just the loader): the cascade follows consumers that
  lose a **required** input — ComfyUI aborts the prompt on one, so they cannot run
  either — and stops at an **optional** socket, which is the point of the mode
  (measured 2026-08-30 on trellis2-multiview: the back loader's
  `Trellis2PreProcessImage.image` is required and must go, the generator's
  `back_image` is optional and stays). `required` per class comes from the same
  `/object_info` cache the bypass rewiring uses (`_node_type_entry`'s `req`), loaded
  only when such a slot is actually empty; an unknown class stops the cascade
  (pre-cascade behaviour — never guess what a node needs). A branch that would take
  the alias's `output_node` with it fails the job UP FRONT naming the slot, instead
  of submitting a workflow that cannot deliver. The same slot's `on_empty_bypass`
  (`slot_empty_bypass`, Mapping's *also bypass* field) names extra ids the cascade
  CANNOT take — a main-path node whose image socket is optional, which would then run
  on nothing; those are BYPASSED, not pruned (pruning cuts the path behind them), by
  joining `req.bypass` for the single `_apply_bypass` pass, so both sources dedupe,
  chain-resolve and report together under `bypassed`;
  **Live progress** (`_ws_progress` + the pure `_progress_apply`/`_progress_view`):
  REST cannot say how far along a run is — `/queue` names the running prompt and
  `/history` appears only once it is over — so the job view had to estimate from the
  median of past runs. The sampler's own step counter (the `25/35` ComfyUI prints) is
  published on the **websocket** only: `progress` `{value,max,node,prompt_id}` and
  `progress_state` `{nodes:{id:{value,max,state}}}`. `generate()` submits with a
  `client_id` and a task connects to `/ws?clientId=` with the SAME id — 0.30 broadcasts
  progress to every listener, 0.34 measurably does not (2026-09-03), and the id is what
  works on both. Every message carries a `prompt_id` and one that is not ours is
  DROPPED: under 0.30's broadcast a job would otherwise display a stranger's steps,
  which is worse than showing none. It runs BESIDE the `/history` poll, never instead
  of it (the poll still owns outcome, timeout, disconnect grace and failover), is
  cancelled in a `finally`, and reports `None` at the end so a finished row cannot keep
  a stale "25/35". Any failure — no `websockets` (it ships with `uvicorn[standard]`,
  like watchfiles), a refused socket, an unknown shape — silently leaves the median
  estimate in place. The ETA is derived from the seconds per step measured in THIS run,
  timed from the FIRST step and restarted whenever the node changes or the counter goes
  backwards: timing from submit charges the model load to step one and predicted 250 s
  for a job that took 15 (measured 2026-09-04). Fed to `main.gen_progress` via
  `ctx.note_progress(job_id, …)` (in memory — it updates several times a second and a
  restart forgets it, which is correct); `_job_view` prefers it (`progress_basis:
  "live"`) over the median, and the console shows `running 25/35`, `~8.9 min left` and
  a bar on the job page. Covered by `test_ws_progress.py`.
  `suggest_mapping()` is only an auto-detect pre-fill. ComfyUI `/prompt`
  rejections are translated to readable per-node errors (node title, class,
  field, offending request param) via `_comfy_prompt_error`; raw body stays the
  fallback. Artifact delivery: a `/view` 404 means "file absent" (probe on), any
  OTHER status raises — a backend error must never silently shrink a delivery;
  `output_ext` prefers the same-stem sibling but ships the reported file when the
  sibling is missing; `output_cases` fetches a case ONCE and checks the detect
  glob on the result; plain `output_globs` may accompany cases OR an
  `output_node` as unconditional extras (the node's result stays authoritative:
  an empty node still errors, never an extras-only delivery; globs WITHOUT a
  node are the whole delivery). A relative `/view` path keeps its dirs as the
  subfolder (`_view_params`).
  `normalize_delivery` (case mode + chain level, normalize-once flagged) V-flips
  generic texture PNGs and — alias Output option `texture_format: jpeg` —
  transcodes them to JPEG q90 (real alpha keeps PNG; ComfyUI has no JPEG export).
  **Input isolation** (`upload_prefix` on `NormalizedRequest`, `upload_prefix_for`
  + `upload_slot_name`): EVERY uploaded input is named `gw_<job id>[_s1|_s2]_<param>
  .<ext>` — no two jobs share input state, ever. ComfyUI reads an input file at
  EXECUTION time, so a shared name is a corruption window the gateway's one-slot
  cap does NOT close (a poll timeout frees the slot while the prompt runs on) —
  measured 2026-08: a client job was delivered another subject's mesh. A blank
  prefix mints a random one; never add a shared fallback. `_upload_image` RAISES
  on failure (silently keeping the intended name meant running on foreign bytes);
  only `_upload_placeholder` is best-effort, because `gw_placeholder.png` is a
  shared CONSTANT. `_cleanup_uploads` overwrites the job's inputs with that 72-byte
  placeholder after a CLEAN success only (a timed-out prompt may still read them);
  it never raises.
  `CloudTaskAdapter` (`cloud = True`, `serves_generation = True`) is the vendor-NEUTRAL
  half of every cloud task backend — `MeshyAdapter` and `TripoAdapter` are subclasses,
  and a third vendor is a subclass plus a pure module, not a second copy of the
  ~60 sites Meshy used to be wired into. It owns the in-flight slot (incl. the `finally`
  decrement and `slot_held` for chain stages), `generate()` (run → download every
  `state.downloads` in order → thumbnail outside the rig endpoint → the job meta),
  `_create` (serialise ONCE, size-scaled timeout ≈ 4 s/MiB — a 93 MB body once died on
  the 30 s client timeout with an EMPTY `str(e)`), `_poll` (a 4xx, or a 200 whose BODY
  refuses the task, is a verdict about the TASK: three IN A ROW → final; transport
  errors, 5xx and 429 are about the SERVICE and get `disconnect_grace` — a poll-rate
  429 must not end a task that is running and already paid for), `_download` (NO auth
  header: signed CDN urls, the bearer must not leak there) and the three chain hooks.
  A subclass supplies only `discover`, `_run` (build → create → poll, plus whatever
  follow-up tasks the vendor needs, returning `RunResult`), `_task_request`,
  `_task_body`, `_classify_create` (→ `nocredits`/`busy`/`server`/`rejected`/None),
  `_task_id_of` and `_msg`. The kind seam `main`/`admin` ask instead of `== "meshy"`:
  `cloud_kind(cand)` (the key its block sits under, None = ComfyUI candidate),
  `cand_kind`/`backend_kind` (a candidate may run on a backend iff they are EQUAL —
  backends are keyed `(name, type)`, so a bare-name match could hand a Meshy alias the
  GPU box), `cloud_module(kind)`, `cloud_block(cand)` (a COPY — the request must not
  write through into the stored candidate) and the derived `CLOUD_TYPES`/
  `CLOUD_MODULES`. `NormalizedRequest.cloud` carries that block (`.meshy` stays as the
  pre-Tripo name, folded in by `__post_init__`), and `MeshyNoCredits`/`MeshyBusy` are
  now `CloudNoCredits`/`CloudBusy` with a `vendor` attribute that `main._fault_label`/
  `_gen_exhausted_msg` name (the old names stay as aliases).
- **`meshy.py`** — the PURE half of the Meshy.ai backend (`type: meshy`; the HTTP half
  is `adapters.MeshyAdapter`): the fixed `input_*` label table → Meshy request body
  (`build_request`), the recorded request (`request_summary`, image data → byte size),
  task-object parsing (`parse_task` — a SUCCEEDED task missing a requested format
  raises, never a silently smaller delivery), the advertised schema (`public_fields`)
  and `default_candidate`/`options_of` (admin option defaults, deep-copied — a shallow
  copy would hand every alias the same `target_formats` list). No `main`/`adapters`
  imports; covered by `test_meshy.py` + `test_meshy_adapter.py` (HTTP stub).
  Three `ENDPOINTS`: the two image ones and **`rigging`** (`POST /openapi/v1/rigging`,
  5 credits, bipeds) — a different request entirely (`_build_rigging`: the mesh as a
  `model_url` data URI + `height_meters` + `name`, none of the image-to-3d options),
  so its inputs are FILES, not image slots: `FILES` names them per endpoint
  (`rigging: ["input_mesh_path"]`) and `public_fields` is a TRIPLE
  `(params, images, files)` — `files` entries carry `{name, required, accept}` and are
  what the schema endpoint advertises, what the playground renders as an upload and
  what `main._decode_upload_files` checks a Meshy `files` key against (an image
  endpoint has none → 400). `glb_data_uri` sniffs the `glTF` magic, so a renamed OBJ
  is refused before 5 credits are spent, and `options_of` narrows `target_formats` to
  `RIG_FORMATS` (glb/fbx) for the endpoint. `parse_task(task, formats, endpoint,
  animations)`: `rigging` reads the urls off `task["result"]`
  (`rigged_character_<fmt>_url`, `basic_animations.<clip>_<fmt>_url`) instead of
  `model_urls`, and `TaskState.downloads` is `[(filename, url)]` — the FILENAME is
  decided here (`rigged.glb` vs `model.glb`, `walking`/`running` clips only with the
  `animations` option), the adapter just downloads them in order. A missing FORMAT
  still raises; a missing CLIP is skipped (a courtesy, not the delivery). A Meshy
  alias candidate has no workflow — `cand["meshy"] = {endpoint, options}`; the client
  may set only what the label table names (`input_name`, `input_face_num` →
  `target_polycount` + `should_remesh`, `input_texture_resolution` px → 2k/4k/8k,
  `input_texture_prompt`, `input_pose`; `input_remove_background`/`input_no_fingers`
  are accepted and inert), the rest are admin options — the counterpart of a ComfyUI
  `fixed` pin. `input_face_num` is advertised WITHOUT a default: `build_request` only
  sets a polycount when the client sends one, and that also forces the remesh pass.
  `adapters.public_fields(cand)` is the ONE seam schema/playground/shims read for both
  candidate kinds, and `adapters.GEN_TYPES` (types whose adapter sets
  `serves_generation`) replaces `type == "comfyui"` in `main` wherever "generation
  backend" is meant; the GPU-host sites (`/free`, `/interrupt` via `adapter.cancel()`,
  restart, watchdog) stay ComfyUI — a cloud task has nothing to interrupt, Meshy
  finishes and bills it. A Meshy backend is always `paid` (it bills per task);
  discovery = `GET /openapi/v1/balance` (0 → DOWN "no credits", balance + its age and
  the rolling gen fail-rate in `/health` + the Backends tab); 402/429 raise
  `MeshyNoCredits`/`MeshyBusy` (ConnectionError subclasses → failover, named by
  `_fault_label`). A FAILED task is final. While polling, a 4xx is a verdict about the
  TASK (3 in a row → final `RuntimeError` naming the status) while transport errors,
  5xx and 429 are about the SERVICE and get `disconnect_grace` seconds (default 30,
  same key as ComfyUI) of CONTINUOUS failure before the failover-class error — a
  poll-rate 429 must not end a task that is running and already paid for. A
  standalone rigging job tags its own meta `rig: "meshy"` (a CHAIN's `rig` comes from
  the successor config instead), so a rigged delivery is recognisable by that field on
  either path; `normalize_delivery`/`validate_delivery` never run for `meshy` — the
  cloud rigs to its own conventions, so the files ship exactly as they arrive.
  Since Tripo joined, everything above is ONE half of a two-vendor interface: `meshy.py`
  and `tripo.py` export the same names and nothing but those names is read by the
  adapter, the console editor and `main` (duck-typed, no ABC — they are modules):
  `KIND`/`VENDOR`/`URL` (kind key, display name, the backend form's fixed URL),
  `ENDPOINTS`/`RIG_ENDPOINT`/`SUCCESS_STATUS`, `POLL_INTERVAL_DEFAULT`/
  `MAX_WAIT_DEFAULT`, `AI_MODELS`/`FORMATS`/`RIG_FORMATS`, `OPTION_DEFAULTS` +
  `OPTION_FIELDS` (the console's option FORM as data), `SLOTS`/`FILES`/
  `IGNORED_PARAMS`, `endpoint_of`/`options_of`/`default_candidate`/`public_fields`/
  `build_request`/`request_summary`/`parse_task`, a `<Kind>Input(RuntimeError)` for
  final content errors, and the three help texts `BACKEND_HINT`/`ENDPOINT_HINT`/
  `CHAIN_HINT`. `parse_task(task, formats, endpoint, options=…)` takes the whole option
  block so the signature is identical for both (Meshy reads `options["animations"]`,
  and keeps accepting the legacy 4th positional bool).
- **`cloudtask.py`** — the pure leaf both cloud modules import (no `main`/`adapters`
  imports, no I/O): `TaskState` (status/progress/error/downloads/thumbnail/credits,
  plus `riggable`/`rig_type` for a task that answers a QUESTION instead of delivering a
  file — Tripo's rig-check), and `parse_options(fields, form, defaults)` +
  `field_value_str`, the reader and writer of the `opt__<key>` form the ONE console
  editor renders from a module's `OPTION_FIELDS`. `parse_options` never raises on a
  form — an unknown value falls back to that key's default, because the module's
  `options_of` is the validator of record and runs on every read anyway. `meshy.py`
  re-exports `TaskState` from here so `test_meshy.py` stayed unchanged. Covered by
  `test_cloudtask.py`.
- **`tripo.py`** — the PURE half of the Tripo3D backend (`type: tripo`, API **V3**; the
  HTTP half is `adapters.TripoAdapter`), declaring the module interface above. The one
  structural difference from Meshy: **Tripo takes no inline bytes at all**, so
  `build_request` receives `{label: file_token}` where `meshy.build_request` receives
  bytes — the adapter `POST`s every image and mesh to `/v3/files` first
  (`_collect_inputs` validates slot count and sniffs the magic BEFORE the first upload:
  an upload for a request that can never run is litter nobody cleans up). Three
  `ENDPOINTS` — `image-to-model`, `multiview-to-model` and **`rig`** (`RIG_ENDPOINT`,
  25 credits, `input_mesh_path` as a FILE like Meshy's `rigging`). `rig-check` is
  deliberately NOT among them: no alias can be configured on it, the adapter runs it as
  a free step of the rig. Multiview needs `input_image_front` plus at least one more
  view (Tripo refuses fewer than two) — checked in `_view_inputs` AND in
  `_collect_inputs`. Every response is a `{code, data}` envelope, so `_task_body`
  unwraps `data` and raises `_TaskVerdict` on a non-zero `code` under HTTP 200 (`_poll`
  counts it exactly like a 4xx: three in a row = final), and `_classify_create` reads
  **403 + code 2010** as out-of-credits (Tripo has no 402) → `CloudNoCredits`, 429 →
  `CloudBusy`, 5xx/transport (uploads included) → `ConnectionError`, anything else
  non-2xx or `code != 0` → final `rejected`. `parse_task` is TOTAL over the documented
  statuses: one outside `TASK_STATUSES` is terminal with an explaining error rather
  than polled to `max_wait` holding the slot (V2 also knew `banned`/`expired`).
  One job is SEVERAL tasks — rig-check → the endpoint → one `/models/convert` per extra
  format (5 credits) → one `/animations/retarget` per `animations` preset (10 credits) —
  and they share ONE `max_wait`: every follow-up poll gets the REMAINDER of the budget
  (the backend form says so). Only the NATIVE format is delivered by the primary task
  (`model.glb`, or `rigged.<out_format>` = the FIRST ticked deliver format), so
  `formats[0]` leads the list; a failed CONVERT fails the job (a requested delivery must
  never silently shrink) while a failed or timed-out CLIP is skipped with a warning
  (`clip_name`: `preset:walk` → `walk.glb`) — the rigged mesh is finished and paid for.
  Once the PRIMARY task is billed, nothing after it may FAIL OVER: a convert failure of
  EVERY class (429 → `CloudBusy`, 5xx → `ConnectionError`, `max_wait` → `TimeoutError`,
  and the raw `httpx.HTTPError` a create's POST raises — a subclass of NEITHER, but named
  literally in `main._GEN_FAILOVER_ERRORS`) is re-raised as a plain `RuntimeError` naming
  the format and the paid task id, a CLIP failure of those same classes is skipped with a
  warning, and an artifact DOWNLOAD failure (`CloudTaskAdapter.generate`, Meshy included)
  becomes the same final `RuntimeError` — otherwise `_run_job` re-runs and re-bills the
  whole image-to-model task on the next candidate over a blip on the asset host. Only the
  rig-check, which runs BEFORE the paid task, keeps normal failover semantics. A client's
  `input_rig_type` is checked (`tripo.check_rig_type`, the same `_rig_types_for` rule)
  BEFORE the uploads, so a refused rig job never pushes its mesh.
  Meta: `cloud`/`cloud_task_id`/`endpoint`/`ai_model`/`request`/`consumed_credits`
  (SUMMED over every task)/`elapsed_ms`, plus `tasks: [{role, task_id, credits}]` (roles
  `rig-check`, the endpoint, `convert:<fmt>`, `clip:<preset>`) so the job view can name
  every BILLED task, and on the rig endpoint `rig: "tripo"`, `rig_spec` (`mixamo`
  default — the Mixamo-compatible bone names Meshy has no answer for) and `rig_type`
  (what the rig-check DETECTED, else what was submitted). Two cross-field rules live in
  ONE place each so the builder, `options_of` and the advertised schema cannot diverge:
  `_rig_types_for` — rig model `v1.0-20240301` rigs BIPEDS only, so `options_of`
  normalizes a stored `rig_type` to `biped`, `public_fields` narrows
  `input_rig_type.choices` to `["biped"]`, and a client value outside the set is
  REFUSED (`TripoInput` — a final content error that ends the job before the rig task
  exists, never quietly bent to biped: a caller asking for a quadruped must learn the
  alias cannot do it); and `generate_parts`, which Tripo
  rejects together with texture/pbr/quad/smart_low_poly, so `options_of` forces those
  four off and a stored alias can never build a request that comes back 400. Client
  labels: `input_face_num` → `face_limit` CLAMPED to 100…the model's cap (v3.1 1.5M,
  v3.0 1M, v2.5 500k, P-series 50k; 150k with `quad`), `input_texture_resolution` px →
  the `standard`/`detailed`/`extreme` bucket, `input_rig_type`; `input_name`/
  `input_remove_background`/`input_no_fingers` are accepted and inert, and
  `input_texture_prompt`/`input_pose`/`input_height_m` are NOT advertised at all —
  Tripo has no field for them, and listing them would promise what the builder never
  sends. The admin `face_limit` option is validated-or-`None`, NOT clamped (same
  reasoning as `meshy.opt_polycount`: a stored value out of range means the candidate
  is broken, and silently rewriting an admin's number is worse than falling back).
  Covered by `test_tripo.py` + `test_tripo_adapter.py` (HTTP stub).
- **`jobs.py`** — generation job store: SQLite metadata + on-disk artifacts under
  `jobs/<id>/<n>.<ext>` (image/video/audio; manifest carries `kind`+`mime`),
  lifecycle `queued→running→done|failed`, TTL pruning. Also persists job **inputs**
  (`set_inputs`: prompt/params/reference images, each with `sha256`+`bytes` like a
  result entry — the job view proves WHICH bytes ran; JSON in meta, no migration)
  and `reconcile_orphans()` (startup:
  mark interrupted `running`/`queued` as failed). Carries `owner`. Reused for
  **background Responses** jobs (task type `response`, result via `complete_json`).
- **`store.py`** — writable SQLite store, the console's source of truth: backends,
  chat aliases, generation aliases (+ workflow_json/mapping/fixed), users (api keys
  **encrypted** via `secret.key`, and therefore re-readable: the user editor
  pre-fills an existing key so it can be copied again — masked, gated by the
  `show_user_keys` setting, default ON, `admin._show_user_keys()`), IP aliases,
  server settings. Seeded once from
  config, then authoritative.
- **`admin.py`** — the `/ui` console (mounted via `admin.register(app)` +
  `add_api_route`, *not* `include_router` — broken in this starlette build;
  callbacks injected via `admin.bind(...)`). Session-gated by `_ui_guard` once
  locked. Tabs in `TABS`; a top tab can group child views via `SUBTABS` +
  `_with_subnav()` (`?sub=` on the parent route, first child = default —
  Playground: Chat | Media | Voice, Jobs & Calls: LLM | Media | Voice,
  Mapping: Chat | Media, Input & Routing: Input | Chat aliases | LLM models |
  Media aliases | Image models | LoRAs); the workflow Mapping editor owns a pasted
  ComfyUI API JSON and offers discovery-fed dropdowns. A CLOUD alias has no workflow at
  all: `_cloud_editor(kind, …)` + `cloud_update` (`POST /ui/mapping/cloud-update`, which
  REPLACED the Meshy-only `/ui/mapping/meshy-update`) render the vendor's option block
  from `mod.OPTION_FIELDS` and read it back through `cloudtask.parse_options` →
  `mod.options_of` — the same normalization the request builder applies, so editor and
  builder cannot drift; the structural fields (alias, task, endpoint, model, deliver
  formats, retries, the chain block, the request-field table) stay the editor's and the
  vendor supplies only the hints. The backend form's `#cloudopts` block carries
  `cloud_max_wait`/`cloud_poll_interval` (named `cloud_*` because `#comfyopts` already
  spends `max_wait`/`poll_interval`, and one form may carry each name only once), and
  `_type_select` fills in the chosen kind's fixed URL when the field is blank or still
  holds ANOTHER cloud kind's fixed URL — switching meshy → tripo would otherwise store a
  Tripo backend pointing at api.meshy.ai, which surfaces only as an auth error at
  discovery; a URL the operator typed themselves is never overwritten.
  Mapping's sub-tab is derived
  when `?sub=` is absent (`?edit=`/`?new=` → media, else chat), so the dozens of
  existing action links keep working unchanged and still land in the right tab.
  The Media list groups by `task` in `_TASK_OPTIONS` order (unknown tasks trail
  alphabetically, so a typo stays visible) with a per-group count; the task is the
  header, so the row shows what differs WITHIN a group (backends · mapped params).
  **Auto-update is one mechanism, not six** (`_LIVE_JS`): `_page(refresh=N)` marks
  `<main data-live="N">` and a global poller re-fetches the SAME url and MORPHS the
  response's `<main>` into the live one — nodes matched by `id`, `data-k` or `data-sk`
  (`keyOf`), falling back to position+tag. An UPDATE never reloads, so scroll, sort
  order, a focused filter, an open form, playing media and the model-viewer camera
  survive it; a response without `data-live` stops the poller (what the meta tag's
  absence used to mean). The one deliberate real navigation is the escape hatch:
  a re-fetch that REDIRECTS to a different path does `location.href = r.url`, which
  is how an expired session lands on `/ui/login` instead of having a login form
  morphed into `<main>` — do not "simplify" it away. The three live row templates
  carry a key themselves — `_job_row` → `job-<job id>`, `_call_row` → `call-<call
  id>`, `_dash_backends`' row → `bk-<_bid>` (`type:name`, because an LLM and a
  ComfyUI backend may share a name) — because those lists are newest-first (and the
  backends panel re-sorts ready→busy→off): without a key the reconciler matches
  POSITIONALLY and a list that gains one row rewrites EVERY row, which is the one
  thing the mechanism exists to avoid. `data-sk` counts as a key for the same reason
  one level up — the dashboard renders three of its four tables conditionally, and an
  unkeyed table node would be reused for a DIFFERENT logical table when a panel comes
  or goes. Never touched: `<script>`, `[data-live-skip]` subtrees (the `.fbxview`
  container, which the server renders EMPTY and the client fills), focused/dirty form
  controls, media with an unchanged `src`, and `<details open>`.
  **The `<script>` rule cuts both ways, and the second way is the trap**: `adopt()`
  drops a script it would otherwise INSERT, not just leave existing ones alone. So
  markup that appears only in a LATER state of a live page arrives without its script
  and silently never initialises — and `data-live` is typically gone in that same
  response, so the poller stops and it cannot self-heal (measured: a finishing job's
  `<model-viewer>`/`.fbxview` preview, a permanently black box until F5). Hence the
  invariant every live page owes: **it must already contain every `<script>` any later
  state of it can render** — hoist them (`job_detail_page`, `_playground_body`) and,
  where the script must act on nodes arriving later, register the action in
  `window.gwLiveHooks` (`gwFbxScan`). `test_admin_live.py` pins this, plus `data-live`
  on `<main>` and the ES5 validity of every JS constant — all three fail silently.
  Post-morph hooks in `window.gwLiveHooks`: the SORT hook because the server always
  renders insertion order and the morph re-imposes it (a clicked sort would be undone
  every tick), the FILTER hook because rows the morph brings in FRESH carry no
  `display` style and would ignore an active filter; the sort hook also `wire()`s
  every sortable table, since a table the morph INSERTS mid-session never went through
  the load-time binding and would otherwise ignore header clicks until a
  real reload. This replaced four `<meta http-equiv="refresh">` pages (Dashboard,
  Media Jobs, Job detail, Backends while draining) and both hand-rolled fragment
  pollers: `_PG_POLL_JS` + `/ui/playground/status/{job_id}` (the media playground's
  result column — the dirty-input rule is what keeps the form editable now) and
  `_VU_POLL_JS` + `/ui/playground/voice-upload-status`, whose terminating
  `location.replace(…&vu=done)` reload, the `vu=done` parameter and its branch left
  with it (one tick now refreshes the progress column AND the voice-library table).
  `voice_upload` therefore 303-redirects to the GET view: the poller re-fetches
  `location.href`, and a POST-only URL answers a GET with 405; the per-user progress
  entry is CONSUMED once its terminal state has been rendered (`prog["seen"]` → popped
  on the next GET), because `_voice_upload_prog` lives for the whole process and the
  `vu=done` reload that used to make it moot is gone. Of the state-restore
  hacks the reload needed, only `_FILTER_JS`'s focus/caret save+restore and the
  `blur`+`beforeunload` listeners that saved them are gone; `_SCROLL_JS`'s `<main>`
  restore STAYS, because
  `<main>` is the page's scroll container (`body{overflow:hidden}` +
  `main{overflow-y:auto}`) and that restore serves REAL navigation (F5, a nav link
  back to a long list, a POST's 303), which the morph does not cover — it is merely
  inert from the first tick onward. POST bodies parsed by
  hand (`parse_qs`) to stay `python-multipart`-free.
- **Media has no call log — it has the job store.** A ComfyUI/cloud generation is a
  JOB, not a forwarded call, so it never reaches `stats.calls` and every table in the
  Statistic tab was blind to it (it looked unmeasured; it never was). `jobs.gen_stats_rows()`
  aggregates done/failed/avg per alias+backend and `admin._media_gen_panel()` renders it
  as **Media generation** — including the `routing` column, which is the live
  `gen_speed` EMA the scheduler actually orders on (`main.gen_speed_info()`, bound into
  admin), so "why did it pick THAT backend?" is answerable in the console instead of
  from the source. `avg` counts DONE jobs only: averaging a three-second failure into a
  two-minute render would make a broken backend look like the fastest one. The panel
  shows even with `stats.enabled` off — its data does not come from stats. The job
  LISTS carry the same estimate per row (`_job_dur_cell` → `_expected_dur_s`, the same
  `jobs.median_duration` basis as the job view's ETA, memoised 60 s): a running row
  reads `12.0 s / ~1.7 min`, so "stuck or just slow?" needs no click. The estimate is a
  SIBLING of `.jdur` — `_JOB_TICK` overwrites that element's text every second.
- **`stats.py`** — optional SQLite (WAL) call log + body store. The dashboard is
  **in the `/ui` Statistic/Routing tabs** (no separate port — the old standalone
  :4001 server was folded into the console). Zero new dependencies — keep it.
  The `calls` row carries the applied `reasoning` control (shown in LLM Calls) and
  the prompt-cache split `cache_read`/`cache_write` — both SUBSETS of
  `input_tokens` (which stays the total the model processed), so
  `input - read - write` is what was billed fresh. Fed by the adapters'
  `_cache_of()` hook (Anthropic: `cache_read_input_tokens` /
  `cache_creation_input_tokens`; OpenAI-shaped: `prompt_tokens_details.
  cached_tokens`) on all four paths — streamed and not, both protocols.
  `cache_trend()` buckets that per backend over 24h and drives the Statistic tab's
  sparkline; a hit rate collapsing while input keeps rising is the signal that a
  long Claude Code session started paying full price again. New columns are added
  by the same `ALTER TABLE` migration list as the earlier ones (an existing prod
  stats.db migrates in place; pre-existing rows read 0, never NULL).
- **`responses_bridge.py`** — pure Responses↔Chat translation functions (no
  gateway state): `responses_to_chat` / `chat_to_responses` / `responses_stream`
  (chat SSE → Responses SSE) and `response_shell()`, the ONE Responses-object
  skeleton every state (completed / stream events / background queued-failed)
  builds on. `main.py` keeps the endpoints, dispatch/parking, and background
  mode; the adapter attaches `resp.parsed_json` so the bridge never re-parses
  the raw body.
- **`anthropic_bridge.py`** — pure Messages↔Chat translation (no gateway state,
  no `main`/`adapters` imports): `messages_to_chat` / `chat_to_messages` /
  `messages_stream` (chat SSE → Anthropic SSE) / `estimate_input_tokens` and
  `message_shell()`. Used ONLY when a non-Anthropic backend serves `/v1/messages`;
  an `anthropic` backend forwards verbatim (see `AnthropicAdapter`). Translation
  policy: drop what is inert (`cache_control`, history `thinking` blocks,
  server-side tools), raise `UnsupportedContent` → 400 where dropping would
  silently answer about content the model never saw (documents/PDFs). Covered by
  `test_anthropic_bridge.py` (stdlib `unittest` — a streaming tool-call bridge fails
  silently rather than crashing). `ls tests/test_*.py` is the count of record —
  **twenty** files today — and each exists for that same reason: the mechanism it
  guards fails SILENTLY, so it is named next to that mechanism above.
  `test_anthropic_bridge.py`, `test_prune_branch.py` (a
  dead-branch prune that cascades one node too far or too few surfaces as an aborted
  generation, not an exception), `test_chain_export_node.py`, `test_chain_hooks.py`,
  `test_chain_mesh_param.py`, `test_ratelimit_headers.py`, `test_scheduler.py`,
  `test_admin_live.py`, `test_err_text.py`, `test_gen_backend_for.py`,
  `test_playground_files.py`, `test_meshy.py`, `test_meshy_adapter.py` — plus the four
  the Tripo backend brought:
  `test_cloudtask.py` (the option-form reader: a field the schema-driven editor loses —
  a missing `opt__<key>`, a type parsed wrong — does not error, it silently saves that
  key's DEFAULT, so an alias's textures quietly turn off on the next Save);
  `test_tripo.py` (the pure builder/parser: a body Tripo REJECTS costs a round trip, but
  one it accepts with the wrong face cap, texture bucket or rig type delivers a
  plausible WRONG mesh — and a task status treated as "not finished yet" instead of
  terminal polls to `max_wait` holding the slot);
  `test_tripo_adapter.py` (the run ORDER and the error classes: a rig-check that does
  not gate the rig spends 25 credits on an unriggable mesh, a 403/2010 classed as final
  stops the failover that would have found the next candidate, and a convert or clip
  that quietly goes missing is a smaller delivery than the client asked for);
  `test_cloud_editor.py` (the console's kind-neutrality — a job view that renders
  nothing for a Tripo run, an editor offering the wrong backends, a type select whose JS
  never reveals the cloud option block: all three are silent in the BROWSER).
  Newest are the two the execution-fault quarantine brought:
  `test_gen_quarantine.py` (the pure fault bookkeeping — a quarantine that never
  triggers leaves a broken backend eating every job, one that triggers on an UNPROVEN
  fault lets a single bad workflow disable an alias's whole fleet, and a candidate that
  gets its probe-once head start back on expiry walks straight to the front again);
  `test_run_job_failover.py` (what `_run_job` does with an execution error: every case
  ends in a plausible-looking job row — a job dead on the first candidate reads like a
  workflow error, a fault charged to the winner idles a healthy backend for 15 minutes,
  and a cloud candidate that fails over re-runs a task the vendor already BILLED);
  `test_ws_progress.py` (folding ComfyUI's /ws messages into a job's progress: 0.30
  BROADCASTS every prompt's progress to every listener, so a missing prompt_id gate
  shows a job a stranger's step count — a lying feature, not an absent one — and an ETA
  looks plausible whatever it says, which is how timing the steps from job start
  predicted 250 s for a 15 s job).
  Run them all with `python -m unittest discover -s tests -t .` (no runner dependency).
- **`openai_image_bridge.py`** — pure request/response plumbing for the OpenAI
  image shims (`multipart_list`, `parse_size`, `coerce_scalar`, `images_uploads`
  slot mapping, `images_response`); imports only the leaf `jobs`. `main.py`
  keeps the endpoints and passes the alias's image `slots` in (one
  `_gen_image_slots` lookup per request).
- **`reasoning.py`** — pure functions for the normalized thinking toggle, no
  `main`/`adapters` imports (hot-reload/test-friendly). Rules are an ordered list
  of `{match(model-glob), backends[], adapter, param}`; `resolve()` picks the
  first rule whose glob matches AND whose backend-set contains the dispatch
  backend, `apply()` rewrites a **copy** of the outgoing chat body per the chosen
  `adapter` (`enable_thinking` / `reasoning_effort` / `nothink_token` / `prefill`
  / `none`) and returns the `x-reasoning-control` string. `none`/no-match →
  `unsupported` (never fails). Rules live in `store` (settings key
  `reasoning_rules`), are cached in `main.reasoning_rules` (refreshed on save via
  `apply_reasoning_rules()`), and are edited in the `/ui` **Reasoning** tab.
  Additionally a **per-alias default** (`alias_reasoning`, store settings; edited
  in the chat-alias editor) supplies off/on when the client sends nothing — so
  `tool`/`tool-thinking` can share one backend+model; an explicit client
  `reasoning` always wins.
- **`previewanim.py`** — injects a short looping idle animation into a rigged GLB for
  the `/ui` inspection view ONLY (`add_idle(glb) → glb`), so bad skin weights show as
  spikes/rings once it deforms. Pure struct/json on the glTF binary, append-only; the
  result route applies it on `?anim=idle`, never to the delivered file.

### Request flow

- **Chat/LLM** (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  and `/v1/audio/speech` — a binary TTS passthrough: `route()` strips `stream`
  on `/v1/audio/*`, fills per-alias `alias_voice` defaults, and the adapter
  skips body parsing/stats blobs for non-text responses — via `route()`): all
  funnel through **`_dispatch_or_park()`** — `resolve_routes()` →
  ready vs busy split → `backend_adapters[bid].dispatch(NormalizedRequest)` to the
  first ready, failing over on connection/timeout errors and on llama-swap's
  "unable to start process" 502 (backend-local load failure, `_retryable_upstream_error`);
  other HTTP error statuses return as-is. The adapter opens the upstream stream
  BEFORE answering, so streamed upstream errors carry their real status too. All busy → **park by default** (FIFO queue, per-alias `park_s`)
  until a backend frees, else 503; no client field. Before dispatch,
  **sampling defaults** are folded in two stages whose ORDER is the precedence
  (client > alias > backend; each stage only fills keys that are still absent):
  `_apply_alias_sampling()` in `route()`/`/v1/responses` applies the alias's
  `alias_sampling` (store settings, cached in `main.alias_sampling`), then
  `adapters._prepare()` applies the backend's `sampling_defaults` — there, so it
  is derived PER BACKEND and a failover re-derives it. Text endpoints only (never
  embeddings/audio). Rationale: a backend whose server samples with bare defaults
  (vLLM, no truncation sampler, temp ≈ 1) degenerates into token salad when a
  client sends no sampling params — measured 2026-08-10 on Infermatic/Anubis-70B.
  Also before dispatch, `_normalize_reasoning()` folds the client `reasoning`/`reasoning_effort` control
  to `off|on|None` and stashes it in `body["_reasoning"]`; the adapter strips all
  `_`-prefixed keys and runs `apply_reasoning` **per backend** on a copy of the
  body (so failover re-derives it). `/v1/responses` translates bodies and shares
  `_dispatch_or_park()` too (also parks, and normalizes reasoning incl. the
  Responses `{effort}` shape); `background:true` runs it async via the official
  Responses background mode (see below).
- **Messages** (`POST /v1/messages`, `POST /v1/messages/count_tokens` — the Claude
  Code frontdoor): `_messages_route()` authenticates (`x-api-key` OR `Authorization:
  Bearer` — Claude Code uses the former; both carry a GATEWAY key), folds
  `thinking:{type:enabled|disabled}` into `body["_reasoning"]` (an explicit client
  control beats the per-alias default, as on the chat path; honoured by translated
  backends only) and hands over to the SAME `_dispatch_or_park()`.
  `count_tokens` for a chat backend is answered from the bridge's estimate BEFORE
  dispatch, so sizing a context never queues for an in-flight slot. Deliberately no
  `_apply_alias_sampling()` here: Claude Code sends a complete request, and a
  chat-shaped `min_p` would 400 against Anthropic. Errors are re-shaped to
  `{"type":"error","error":{…}}` — Claude Code reads `error.message` and renders a
  FastAPI `detail` body as blank. Routing filters candidates by `serves_path()`, so
  an Anthropic backend is invisible everywhere else and an alias served only by
  such backends answers `404 … reachable through POST /v1/messages only` instead of
  a misleading 503.
- **Generation** (`POST /v1/generations`, and the OpenAI shims
  `/v1/images/generations` + `/v1/images/edits`): `get_gen_routes(alias)` resolves
  the alias via the **separate** generation store (`image_models`/store), filtered
  to enabled+healthy generation backends of the candidate's own kind
  (`adapters.cand_kind` == `adapters.backend_kind`); LoRA-aware preference +
  busy→park; a
  `jobs.py` job runs via `adapter.generate()` (sync inline or async job-id). A
  running job can be cancelled (`cancel_generation` → ComfyUI `/interrupt` + task
  cancel).
- **Workflow chains** (`_run_chain`): a gen alias's stage-1 config carries a
  `successor` (`{alias, export_node, mesh_param, relay?, keep_from_mesh?, rig?}`);
  stage 1 exports a mesh under a gateway-pinned filename (`gwchain_<jobid>`) and
  ONLY stage 2's result is delivered (+ any `keep_from_mesh` files). The three
  backend-specific spots are **adapter hooks**, not branches in the router
  (`ChainExport` + `chain_export`/`chain_take_mesh`/`chain_feed_mesh`; the base class
  refuses both roles, so a new backend type is opt-in): ComfyUI pins the export node's
  `filename_prefix` (extension from pins/mapped params on `file_format`), takes the
  mesh off `/view` and feeds a PATH (shared disk) or an upload into the stage-2 input
  dir; a **cloud stage** (`CloudTaskAdapter`, so Meshy AND Tripo) names it
  `gwchain_<jobid>.glb` with NO pins, takes the `model.glb`
  RESULT BLOB and feeds it as `req2.upload_files[mesh_param]` (Meshy embeds it as the
  rigging `model_url` data URI, Tripo uploads it to `/v3/files` and sends the token;
  `mesh_ref` = `<upload:… (N MB)>` either way). So EITHER stage may be a cloud one, and all nine
  ComfyUI/Meshy/Tripo × ComfyUI/Meshy/Tripo combinations run: `Meshy-Humanoid` →
  `mesh-mia`, `Trellis2-…` → `Meshy-Rig` and `Tripo-Humanoid` → `Tripo-Rig` are the
  same code path. A cloud stage
  has no shared disk, so `_kind_cloud` on either side FORCES
  `relay: upload` whatever the alias stored (the editor hides the field); a cloud
  stage 1 without `glb` in `target_formats`, and a rigging alias (`mod.RIG_ENDPOINT`)
  as stage 1 at all,
  are refused by `chain_export` BEFORE credits are spent. `mesh_param` is validated
  against the successor's mapping (param or label) or — a cloud successor has no
  mapping — against its file fields (`public_fields()[2]`). A paid stage-1 cloud task
  keeps its own kind/task id/endpoint/request/sub-tasks/credits under
  `meta.chain_stage1` (keys `backend, cloud, cloud_task_id, meshy_task_id, endpoint,
  consumed_credits, request, tasks`; stage 2's
  are the top-level meta, which would otherwise overwrite them in the merge), and
  `admin._cloud_table` renders one table per cloud stage — naming the VENDOR from that
  stage's own meta, because the two stages may be different vendors. `rig` accepts the
  cloud values `meshy` and `tripo`: tagged on the delivery like the others, but NEVER
  normalized or validated —
  `normalize_delivery`/`validate_delivery` run for `generic`/`mixamo` only, a cloud rig
  follows its vendor's conventions (which bone names a `tripo` one carries is
  `meta.rig_spec`). Stage 1 gets
  the normal routing guarantees: candidates re-resolved while parked (force pin +
  LoRA eligibility kept), misconfigured candidates skipped, connection errors fail
  over (stage-2/hand-off errors are FINAL). The job row's `backend` is re-pointed
  at claim and hand-off (`jobs.set_backend`) so cancel interrupts the LIVE backend.
  Stage-1 params are threaded to the successor by mapping **label** (never the raw
  node-based name), and `_apply_mapping` SILENTLY skips a name the successor does not
  bind — so what stage 2 was handed is recorded on the job (`meta.chain_stage2`:
  alias/backend/relay/mesh_param/mesh_ref/params, plus stage 2's `applied` on success)
  and rendered by `admin._stage2_section` as handed / applied / dropped. Recorded at
  run time, never re-derived from config (the mapping may have moved since); written
  into `jobs.complete`'s meta AND, via `fail_meta()`, onto a failed row — a stage-2
  failure is when the hand-off matters most. Without it, "did my param reach the
  rigger?" was only answerable from the backend's own ComfyUI history.
  Chain stages run with `slot_held` (the chain claims the one slot itself — no
  double count). Two hand-offs (ComfyUI↔ComfyUI; with a cloud stage the second is
  forced):
  `relay: path` (default) keeps both stages on ONE backend (shared disk, one slot
  held across both — queue-isolated) and passes the mesh's absolute output path;
  `relay: upload` lets the successor run on a **different** backend — the gateway
  fetches the mesh (`/view`), uploads it into the stage-2 backend's input dir
  (`adapter.upload_input` → ComfyUI `/upload/image`) and passes the file's
  **absolute input-dir path** (backend `comfy_input_dir`, blank = derived from
  `comfy_output_dir`'s `…/input` sibling) — the successor consumes it exactly like
  a path hand-off; only with no input dir known does the bare stored name go over
  (then a load-from-input node is required). Cross-backend releases the stage-1
  slot AND frees its ComfyUI VRAM once the mesh is in hand, then claims the
  stage-2 slot.

### Routing rules (`get_routes_for`/`get_gen_routes` + `alias_entry`)

A backend is a candidate only if enabled, healthy, **not busy** (in-flight cap),
maps the alias, and exposes the resolved model. Recurring concepts:

- **Route index** (`_route_index` + `rebuild_route_index()`): alias→candidates and
  bare-model-id pass-through are **precomputed** (insertion order, no sort);
  `resolve_routes()` evaluates the live flags (healthy/busy/draining) per request and
  applies the scheduler ordering — unpaid before paid, then fastest (`backend_tps` /
  `gen_speed`), unmeasured first. A freed backend takes the waiter the scheduler
  designates for it (`scheduler.designated_taker`: overdue > same type key > oldest;
  `affinity_max_wait_s`, default 120, Server tab). `priority` routes nothing any more —
  it only keeps the backend LIST order stable. Rebuilt by `rebuild_backends()`/`rebuild_virtual_models()` and on every
  discovery model-set change — never mutate `backends`/`virtual_models` outside
  those functions or the index goes stale.

- **Aliases** (`virtual_models`): string (same everywhere) or per-backend dict whose
  value is a model id (old entries may still be `{model, priority}` — `alias_entry()`
  parses that shape, the priority is ignored).
- **Backend prefixing** (`split_backend_prefix`): `<backend>/<model>` pins that
  backend; a bare id/alias goes through the scheduler. `local: true` *also* lists models
  bare (cross-backend implicit alias). `model_prefix` toggles prefixed listing.
- **Concurrency/busy** (`backend_inflight`, `backend_busy`): incremented in
  `dispatch()`/`generate()`, decremented on completion incl. the streamed `finally`.
- **Re-routing onto a returning backend**: waiting work is never pinned to the
  backend it queued for. `refresh_backend` calls `_notify_slot_free()` on DOWN→UP
  and on a model-set change (parked calls re-evaluate); `apply_backend_change` and
  `cancel_drain` do the same; parked gen jobs re-resolve routes every 2 s. The slow
  part was NOTICING, so `health_loop` polls backends concurrently (a sequential loop
  added every dead backend's connect timeout to the cycle) and `fast_probe_loop`
  re-polls only UNHEALTHY backends every `fast_probe_interval_s` (default 3, Server
  tab, 0 = off) while `_capacity_wanted()` — `_parked` non-empty, or a gen park loop
  pinged `_gen_wait_ping()` within 5 s. A TIMESTAMP, not a counter: a cancelled job
  task cannot leave a phantom waiter. `_probing` guards against two concurrent polls
  of one backend. `_run_job` re-points the job row (`jobs.set_backend`) at the
  backend that actually claims it — a parked job routinely lands elsewhere, and a row
  naming the wrong backend sends you reading the wrong ComfyUI's log.
- **Parking** vs 503: "all busy" is distinguished from "no backend" — only the
  former parks (the default). The queue is `_parked` (rich entries: alias, source,
  deadline, `asyncio.Event`); `_inflight_dec`→`_notify_slot_free` wakes all in FIFO
  order so the oldest eligible claims the freed slot (the invariant: dispatch's
  `inflight_inc` runs with no `await` between it and `resolve_routes`). Park time
  per alias via `alias_park_s` (store `alias_park` + config), else `park_timeout_s`
  (default 60); `0` disables. Timeout → 503 + `Retry-After`. "Parked calls" panel
  on the Dashboard. **Async chat has no OpenAI spec** — async lives on the Responses
  background mode: `POST /v1/responses {background:true}` → `resp_<jobid>` queued →
  `GET /v1/responses/{id}` poll → `POST …/cancel`; the worker (`_run_bg_response`)
  parks in the same queue (jobs.py task `response`).
- **LoRA-aware generation routing**: a backend lacking a requested LoRA is dropped
  from candidates (decided over all candidates incl. busy → parks for the
  LoRA-backend rather than spilling); a LoRA on no backend is ignored (the normal
  ordering wins); an explicit `backend` force is never overridden. Per-backend LoRA sets
  come from discovery (`backend_loras`).
- **Execution-fault quarantine** (`scheduler.exec_fault_*`, state `main.gen_exec_faults`
  keyed `alias|bid`): a generation backend that ANSWERS but cannot EXECUTE is invisible
  to every other signal — discovery only calls `/object_info` so `backend_healthy` stays
  True, and the executor watchdog only sees a STUCK queue while this one drains fine,
  turning every prompt into an error in seconds. It was also self-reinforcing: a
  candidate that never succeeds never gets a `gen_speed` sample, so it kept the
  unmeasured **probe-once** head start and won the ordering again on the very next
  retry (measured 2026-09-03 on comfyui-strix — a torch/ROCm update broke
  `flux_time_shift`, so every Flux/Krea2 model load died while two healthy backends sat
  idle and four consecutive user retries all landed on the broken one). So: an
  execution error now **fails over to the next candidate** (never a repeat on the same
  backend — it reproduces; `self_retries` stays a connection-path thing), and a fault is
  charged ONLY once a later candidate succeeded on that same job — the difference
  between "this backend is broken" and "this request is broken". When they all fail
  alike nobody is charged and the report names the execution error plus the backends
  that shared it, never `_gen_exhausted_msg`'s "unreachable". Two consecutive charged
  faults quarantine that `alias|backend` for `EXEC_QUARANTINE_S` (900 s): it leaves
  `ready` but stays at the END of `allc`, so the job PARKS for a healthy backend
  instead of 503-ing and a failover can still reach it as the last resort — and if
  EVERY candidate is quarantined the quarantine is ignored entirely (a blocked alias is
  worse than a slow one). One success clears the record; `exec_probed` outlives the
  window on purpose, or the expiry would hand the head start straight back. Surfaced as
  `quarantined` in `/health` + the Backends tab, which it must be: unlike the fail rates
  this one really does change routing. **A CLOUD candidate is excluded from the failover**
  — a billed task may have failed AFTER creation, and re-running the job would buy the
  same mesh twice (the invariant `tripo.py` protects). Covered by
  `test_gen_quarantine.py` + `test_run_job_failover.py`.
- **Allow-list filtering**: `/v1/models` authenticates the caller and filters by
  their allow-list (entries may be aliases, model ids, or **backend names** =
  all that backend's models); image aliases are included; `?type=chat|image`.
- **Alias/model-name collisions** (`alias_model_conflicts`): surfaced in `/health`
  + Routing tab, split `covered` vs actionable `shadowed`.
- **Host coordination** (every box with a ComfyUI backend — the LLM-vs-media policies
  matter on SHARED boxes, the VRAM policy on all; `docs/host-coordination-plan.md`):
  backends group by physical box (`backend_host`: explicit `host` field, else URL
  IP → `backend_hosts`/`host_backends`, shown in `/health` and the Backends tab's
  Hosts panel). Per-host policies (store settings `hosts`, cached `hosts_meta`):
  chat candidates on a host with a RUNNING media job sort LAST in `resolve_routes`
  (never dropped; flag `avoid_llm_during_media`, default on);
  opt-in `llm_unload_before_media` GETs llama-swap `/unload` first.
  **VRAM is freed BEFORE a job, not after it** — ComfyUI never releases its model
  cache by itself, and at job END nobody knows yet what comes next, so the answer
  was always guessed. At CLAIM it is known: `_claim_gen_backend(backend, key)`
  records the affinity key `backend_last_key` (media: the alias) and, when what the GPU
  HOLDS (`backend_vram_key`) differs from what the request will LOAD, AWAITS the free
  before the prompt goes out (`scheduler.free_vram_before_job`, pure +
  `test_scheduler.py`; host flag `comfy_free_before_job`, default on for EVERY ComfyUI
  box). What it loads is the **model-set key** — `scheduler.model_set_key` over the
  workflow's loader nodes (class /load/ minus image/mask/mesh/path loaders; only their
  STRING-valued weight inputs — name/model/ckpt/unet/clip/vae/lora/gguf — never node
  ids, links, dtype/device/`keep_models_loaded`), computed by
  `ComfyUIAdapter.model_set_key(req)` AFTER pins, the LoRA list/cascade and the mapped
  params, so the request is built before the claim. The alias was the key at first and
  is wrong both ways: `img2mesh-trellis2_high`/`_low` both load `microsoft/TRELLIS.2-4B`
  and were freed+reloaded on every switch, while a mapped model choice or a LoRA under
  ONE alias never freed. Alias stays the FALLBACK (no loader recognised) and the
  affinity key. The two records are separate on purpose: `backend_vram_key` is written from
  the free's VERDICT (`_comfy_free` returns one), never from the attempt — a failed or
  skipped free leaves it `None`, and `None` (unknown, or two aliases' sets mixed by a
  job in flight, or a gateway restart — which never empties ComfyUI's VRAM) counts as
  a change. And "awaited" means watched: ComfyUI's `POST /free` only sets two flags
  and answers 200; its single worker reads them after `q.get()`, and `set_flag`'s
  `notify` is LOST while the worker sits in its post-prompt `gc.collect()` — exactly
  when the gateway's /history poll posts the free — so the flag would fire only after
  the NEXT prompt ran. `_comfy_free(settle_s=…)` therefore polls `/system_stats`'
  `torch_vram_total` (torch's reserved pool, what other processes cannot use) and
  re-posts every 2 s until it dropped to ≤ 20 %, else reports False after 30 s
  (`test_run_job_failover.py::ComfyFreeSettles`). It is also the only hook that sits between two CHAIN stages — two
  workflows on one backend with no job end in between (measured 2026-09-05, job
  `cc604da29e0e`, Meshy→`mesh-mia` on k12-gpu: stage 2 died on
  `CUDA out of memory. Tried to allocate 20.00 MiB` with 21.4 of 23.5 GiB held by
  the ComfyUI process, and Make-It-Animatable runs in its OWN venv/process, so it can
  never share that cache). The after-job free (`_free_comfy_vram`, flag
  `comfy_free_after_job`, default on for shared hosts only — all four host flags and
  their defaults live in ONE table, `scheduler.HOST_FLAGS`, read by `main._host_flag`,
  the console's host form/panel and `host_save`, and `host_save` derives "shared" from
  the LIVE backend list, never from a hidden form field) survives for its one
  remaining purpose — a shared box must free even when NO media job follows, or the
  next llama-swap load aborts — and skips when the waiter the scheduler will hand
  this backend next (`_designated_gen_waiter`, so `exclude`/force/LoRA eligibility
  apply) wants the alias its VRAM holds. Asking the scheduler, not scanning the queue
  by alias, is what keeps a waiter that can never run here (a chain that excluded the
  backend after a failed stage 1) from suppressing the free forever
  (`test_run_job_failover.py::FreeAfterJob`).

### Auth / multi-user

`authenticate()` resolves a Bearer token to a user (`_users_by_key`) or the
master `_MASTER_ADMIN` (the top-level `api_key`); `gate_request()` enforces the
allow-list (`_model_allowed`, incl. whole-backend grants) + quotas and attributes
the call. Bootstrap-open with no users and no master key. The `/ui` console
session is gated by `_ui_guard` once locked.

### Stats recording

Every forward calls `stats.record_call(...)` fire-and-forget via
`asyncio.create_task` — never raises into the request path. **Refused** calls are
recorded too, by the app-wide `HTTPException` handler (`_rejected_call` →
`_record_rejected`): anything turned away before a backend saw it (no healthy
backend, park timeout, quota, unknown alias, bad key) used to leave NO trace at
all, which is precisely the call you go looking for in LLM Calls. Such rows carry
backend `stats.REFUSED_BACKEND` = `(refused)` (defined THERE, not in `main`, because
the aggregates are what must exclude it), an empty `model` (none was ever resolved),
the status, and the reason as the stored response body. That marker is a call-log
entry only: `summary()` keeps it out of `by_backend`/`by_model` — a pseudo-backend
with 0 tokens/0 cost/0 ms says nothing about any backend, and its empty model splits
the alias into two rows — and reports it as `refused_count`/`refused_24h`, which the
Statistic tab shows as its own card (totals and `by_source` still count it: a refused
call is real traffic from that user). `request.state.gw_dispatched` (set in
`_dispatch_over` once an adapter answered, and in `run_generation` right after
`jobs.create`) prevents a second row when an endpoint re-raises an upstream error —
and `gw_alias`/`gw_body`/`gw_endpoint` carry the context the handler cannot see.
The generation arm of that flag is not cosmetic: `gen_done_or_502` raises AFTER a
job ran, so without it a **failed media job** was logged a second time as
`(refused)`, 0 ms, no backend — a request that was in fact served and failed
(measured 2026-08-25). The invariant: once a job row exists, the JOB owns the
outcome; only refusals BEFORE it (no eligible backend, quota, malformed request)
belong in the call log. `admin._call_kind()` partitions that log into
`voice`/`media`/`llm` so each row has exactly one home — media refusals show under
Media Jobs, not LLM Calls. The same handler renders `/v1/messages` errors in
Anthropic shape, so that form lives in ONE place. Cost from pricing
cached at discovery (`normalize_pricing`: Together per-million, OpenRouter
per-token). Streaming records the backend's usage chunk (the adapter always
requests `include_usage` upstream); a backend that reports zeros/nothing
(LocalAI streams all-zero usage — measured) gets gateway estimates instead
(content-delta count ≈ completion tokens, ~chars/4 for the prompt).

## Conventions

- Two pricing schemas and two `/v1/models` payload shapes (`{"data":[…]}` vs bare
  list) are handled defensively in discovery (`extract_models`/`extract_pricing`/
  `_is_chat_model`). Preserve that.
- The Responses↔Chat bridge supports `stream:true` (chat SSE → Responses SSE) and
  `background:true` (async). What's **not** built yet: streaming-reconnect for a
  background response (poll-only). Keep in sync when touching `/v1/responses`.
- Keep `stats.py`/`jobs.py`/`store.py`/`reasoning.py`/`responses_bridge.py`
  dependency-free and hot-reload-safe (no caching config values at import time);
  `reasoning.py` and `responses_bridge.py` must stay pure (no `main`/`adapters`
  imports) — they're called on the request path.
- Generation: workflow + mapping are **backend-independent** (shared across an
  alias's candidates); **pinned values** (`fixed`) AND **per-node bypass**
  (`bypass`: node ids) are per-backend. A pinned `(node,field)` is authoritative —
  never overridden by an API request param. A **cloud** alias (Meshy, Tripo) has
  neither workflow nor mapping: endpoint + an admin option block, and the options are
  what `fixed` is to a ComfyUI alias — the client may set only what the kind's fixed
  label table names. An alias stays homogeneous (one kind), so a Meshy alias can never
  be given a Tripo backend or vice versa (`cand_kind` == `backend_kind`).
- Single instance only; verify with compile + a route/render check; restart the
  one instance when the user says idle. Never commit `config.yaml`, `store.db`
  (+ `secret.key`), `stats.db*`, `jobs.db*`, `jobs/`, `voiceref/`, `*.key`.
- Voice cloning (`/v1/audio/speech`): TTS backends read `voice` strictly as a
  file on THEIR host (no base64/URL/upload API — measured). The voice library
  (`voiceref/` blobs + store `voice_library`, UI in the Voice sub-tab) therefore
  ships references via scp to EVERY target in `voice_ref_hosts` (settings;
  comma-separated `user@host:/abs/host/dir`, host-side dirs may differ — docker
  mounts), while `voice_ref_dir` is the single model-visible path written into
  `voice`; `voice:"lib:<name>"` resolves to the shipped path + ref_text in
  `route()`. An empty ref_text is auto-transcribed: local faster-whisper first
  (lazy CPU import; the one heavyweight entry in `requirements.txt`), a backend
  whisper model as fallback.
