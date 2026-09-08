# Generation Backend Fault-Handling (Runbook Strix) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Model directive:** Dispatch implementer subagents with `model: opus`.

**Goal:** Implement runbook `quartermaster/runbooks/llm-gateway-fehlerhandling-strix.md` items **B** (per-backend `self_retries` — retry a sporadically crashing backend on ITSELF before failing over) and **C** (per-backend rolling generation fail-rate, display-only in `/health` + Backends tab), plus the **A** configuration for the new `comfyui-strix` backend.

**Architecture:** All code changes live in `main.py` (retry loop in `_run_job` + `_run_chain` stage 1, rolling attempt window as module globals), `jobs.py` (optional meta on `fail()`), and `admin.py` (one form field, one status line). No new dependencies, no schema migrations — backends are free-form JSON in the store, so `self_retries` is just a new key. Item **D** (fail-rate as routing criterion) is explicitly OUT of scope (runbook: only after C proves out).

**Tech Stack:** Python 3 / FastAPI / httpx / SQLite — existing gateway stack, zero additions.

## Global Constraints

- **No test suite exists in this repo** (CLAUDE.md). Verification per task = `venv/bin/python -m py_compile <files>` + the task's runtime check. Do NOT introduce pytest or a tests/ directory.
- **Never retry content errors** (`RuntimeError` from `_format_comfy_error`) or `ComfyExecutorStuck` — only `_GEN_FAILOVER_ERRORS` (`httpx.ConnectError`, `httpx.TimeoutException`, `httpx.ReadError`, `ConnectionError`, `TimeoutError`) are retried/failed-over (runbook "Was ausdruecklich NICHT gebaut werden sollte").
- **The fail-rate is display-only.** Never drop or de-prioritize a backend from it (that is item D, out of scope).
- **The job slot (`inflight`) is held across self-retries** — no double counting, no window where a parked job can steal the slot mid-retry.
- **Every retry is visible**: server log line per attempt + `attempts: n` in job meta — the feature must not mask the fault rate.
- `jobs.py` stays dependency-free and hot-reload-safe (no config caching at import).
- Default `self_retries` = 0 = today's behavior, byte-for-byte.
- Commit messages: imperative prefix style used in this repo (`gen: …`, `ui: …`, `jobs: …`).
- The dev checkout has a valid gitignored `config.yaml`; `venv/bin/uvicorn main:app --port 4599` boots against it for runtime checks.

---

### Task 1: Retry + fail-window infrastructure

**Files:**
- Modify: `main.py:8` (imports), `main.py:1859` (after `_GEN_FAILOVER_ERRORS`)
- Modify: `jobs.py:235-238` (`fail()`)

**Interfaces:**
- Produces: `backend_gen_window: dict` (module global), `_record_gen_attempt(bid: str, conn_fail: bool) -> None`, `_gen_fail_stats(bid: str) -> Optional[dict]` (returns `{"fail_rate": float, "gen_fails": int, "gen_attempts": int}` or `None`), `async _wait_backend_up(backend: dict, timeout_s: float = 30.0) -> None`, `jobs.fail(job_id, error, meta: Optional[dict] = None)`. Tasks 2–4 call exactly these names.

- [ ] **Step 1: Add the deque import to `main.py`**

After `from contextlib import asynccontextmanager` (main.py:10) region, add alongside the stdlib imports:

```python
from collections import deque
```

- [ ] **Step 2: Add window + helpers to `main.py`**

Insert directly AFTER the `_GEN_FAILOVER_ERRORS` tuple (main.py:1858-1859):

```python
# ── Per-backend rolling generation fail-rate (runbook C) ────────────────────────
# bid → deque[(ts, conn_fail)] of the last generate() attempts. In-memory on
# purpose (a gateway restart resets the sample — fine) and module-global so
# adapter rebinds on config hot-reload don't lose it. Display-only: NEVER used
# to drop a backend from rotation (routing control stays priority / runbook A1).
backend_gen_window: dict = {}
_GEN_WINDOW_N = 50            # last N attempts …
_GEN_WINDOW_S = 86400         # … no older than 24 h


def _record_gen_attempt(bid: str, conn_fail: bool) -> None:
    """Count one generate() attempt: every attempt lands in the window;
    `conn_fail` marks connection-type aborts (_GEN_FAILOVER_ERRORS). Content
    errors count as attempts, not as faults."""
    dq = backend_gen_window.setdefault(bid, deque(maxlen=_GEN_WINDOW_N))
    dq.append((time.time(), bool(conn_fail)))


def _gen_fail_stats(bid: str) -> Optional[dict]:
    """{fail_rate, gen_fails, gen_attempts} over the window, or None without data."""
    dq = backend_gen_window.get(bid)
    if not dq:
        return None
    cutoff = time.time() - _GEN_WINDOW_S
    total = fails = 0
    for ts, cf in dq:
        if ts >= cutoff:
            total += 1
            fails += 1 if cf else 0
    if not total:
        return None
    return {"fail_rate": round(fails / total, 2), "gen_fails": fails,
            "gen_attempts": total}


async def _wait_backend_up(backend: dict, timeout_s: float = 30.0) -> None:
    """Give a crashed-and-systemd-restarting ComfyUI time to come back before a
    self-retry: poll /system_stats until it answers (or timeout_s passes). Never
    raises — a still-down backend just makes the retry fail fast into failover."""
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
```

- [ ] **Step 3: Let `jobs.fail()` carry meta**

Replace `jobs.py:235-238`:

```python
def fail(job_id: str, error: str) -> None:
    with _conn() as c:
        c.execute("UPDATE jobs SET status='failed', error=?, stage=NULL, updated=? WHERE id=?",
                  (str(error), int(time.time()), job_id))
```

with:

```python
def fail(job_id: str, error: str, meta: Optional[dict] = None) -> None:
    """Mark failed. Optional `meta` (e.g. {"attempts": 2} from a self-retried
    generation) is merged into meta_json — a retried-and-still-failed job must
    show its attempt count, or retries would mask the fault rate."""
    with _conn() as c:
        sets = "status='failed', error=?, stage=NULL, updated=?"
        args: list = [str(error), int(time.time())]
        if meta:
            sets += ", meta_json=?"
            args.append(json.dumps({**_read_meta(c, job_id), **meta}))
        c.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*args, job_id))
```

(All existing callers pass two positional args — signature stays backward-compatible.)

- [ ] **Step 4: Compile-gate**

Run: `cd /home/dev/projekte/llm-gateway && venv/bin/python -m py_compile main.py jobs.py`
Expected: exit 0, no output.

- [ ] **Step 5: Runtime sanity of the pure helpers**

Run from the repo root (imports `main`, which loads `config.yaml` at import — that file exists here):

```bash
venv/bin/python - <<'EOF'
import main
main._record_gen_attempt("comfyui:x", conn_fail=True)
main._record_gen_attempt("comfyui:x", conn_fail=False)
s = main._gen_fail_stats("comfyui:x")
assert s == {"fail_rate": 0.5, "gen_fails": 1, "gen_attempts": 2}, s
assert main._gen_fail_stats("comfyui:none") is None
print("OK", s)
EOF
```

Expected: `OK {'fail_rate': 0.5, 'gen_fails': 1, 'gen_attempts': 2}`

- [ ] **Step 6: Commit**

```bash
git add main.py jobs.py
git commit -m "gen: fault-handling groundwork — attempt window, /system_stats wait, jobs.fail meta"
```

---

### Task 2: `self_retries` in `_run_job`

**Files:**
- Modify: `main.py:1916-1944` (`_run_job`, entire function body)

**Interfaces:**
- Consumes: `_record_gen_attempt`, `_wait_backend_up`, `jobs.fail(…, meta=…)` from Task 1; existing `_inflight_inc`/`_inflight_dec`, `backend_id`, `backend_adapters`, `_unload_host_llms`, `_free_comfy_vram`, `log_per_call`.
- Produces: backend key `self_retries` (int ≥ 0, default 0) is now honored by every non-chain generation; `NormalizedRequest.slot_held=True` is set by `_run_job` itself (it owns the slot now).

- [ ] **Step 1: Replace `_run_job`**

Replace the whole function (main.py:1916-1944) with:

```python
async def _run_job(job_id: str, candidates: list, build_req) -> None:
    """Run a generation job, failing over to the next candidate on connection-type
    errors. A backend with `self_retries: n` gets n extra attempts on ITSELF first
    (runbook B: for sporadic driver faults the same host is the cheapest second
    try — no model re-load elsewhere, same success odds as attempt one). Its ONE
    job slot is held across the repeats, so no parked job slips in between.
    Content errors are final (not retried). Stops at the first success."""
    await asyncio.to_thread(jobs.set_status, job_id, "running")
    last = None
    attempts = 0
    for backend, cand in candidates:
        bid = backend_id(backend)
        adapter = backend_adapters.get(bid)
        if adapter is None:
            continue
        tries = 1 + max(0, int(backend.get("self_retries") or 0))
        _inflight_inc(bid)                     # hold ONE slot across all self-retries
        try:
            for attempt in range(1, tries + 1):
                attempts += 1
                try:
                    await _unload_host_llms(backend)   # opt-in host policy, no-op by default
                    req = build_req(backend, cand)
                    req.slot_held = True               # we hold it — generate() must not double-count
                    out = await adapter.generate(req)
                    _record_gen_attempt(bid, conn_fail=False)
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
                    if attempt < tries:
                        logger.warning(f"✗ job {job_id} [{backend['name']}] connection issue "
                                       f"({type(e).__name__}: {e}) — retrying same backend "
                                       f"(self-retry {attempt}/{tries - 1})")
                        await _wait_backend_up(backend)
                        continue
                    logger.warning(f"✗ job {job_id} [{backend['name']}] connection issue "
                                   f"({type(e).__name__}: {e}) — failing over")
                except Exception as e:
                    # content error (ComfyUI validation/execution) — final, never retried:
                    # it would fail identically on any attempt and any backend.
                    _record_gen_attempt(bid, conn_fail=False)
                    logger.warning(f"✗ job {job_id} [{backend['name']}] failed: {e}")
                    await asyncio.to_thread(jobs.fail, job_id, str(e),
                                            {"attempts": attempts} if attempts > 1 else None)
                    asyncio.create_task(_free_comfy_vram(backend, "job failure"))
                    return
        finally:
            _inflight_dec(bid)
    await asyncio.to_thread(jobs.fail, job_id,
                            f"all candidate backends unreachable (connection): {last}",
                            {"attempts": attempts} if attempts > 1 else None)
```

Notes for the implementer:
- `NormalizedRequest` is a mutable dataclass — `req.slot_held = True` is fine. `adapters.py:2024/2081` already skip inc/dec when `slot_held` (the chain uses the same contract).
- `asyncio.CancelledError` (job cancel) inherits `BaseException`, so the generic `except Exception` does NOT swallow it; the `finally` still releases the slot. Do not add a CancelledError handler.
- `build_req` is called fresh per attempt — a retry re-uploads its inputs under the SAME job-unique `gw_<jobid>_…` names (input isolation holds; same job, so no cross-job collision).

- [ ] **Step 2: Compile-gate**

Run: `venv/bin/python -m py_compile main.py`
Expected: exit 0.

- [ ] **Step 3: Smoke-test the retry loop with stub adapters**

Write `/home/dev/projekte/llm-gateway/.smoke_run_job.py` (temp, deleted in Step 5):

```python
"""Ad-hoc smoke test for _run_job self-retry (no test framework in this repo)."""
import asyncio
import jobs
import main
from adapters import NormalizedRequest

calls = {"attempts": 0, "completed": None, "failed": None}


class StubOut:
    blobs = []
    meta = {"backend": "fake"}


class StubAdapter:
    def __init__(self, fail_first_n):
        self.fail_first_n = fail_first_n

    async def generate(self, req):
        assert req.slot_held is True, "slot must be pre-held by _run_job"
        calls["attempts"] += 1
        if calls["attempts"] <= self.fail_first_n:
            raise ConnectionError("boom")
        return StubOut()


async def no_wait(backend, timeout_s=30.0):
    return


async def no_op(*a, **k):
    return


main._wait_backend_up = no_wait
main._unload_host_llms = no_op
main._free_comfy_vram = no_op
jobs.set_status = lambda job_id, s: None
jobs.complete = lambda job_id, blobs, meta=None: calls.__setitem__("completed", meta)
jobs.fail = lambda job_id, err, meta=None: calls.__setitem__("failed", (err, meta))

backend = {"name": "fake", "type": "comfyui", "url": "http://127.0.0.1:9",
           "priority": 1, "self_retries": 1}
bid = main.backend_id(backend)
main.backend_adapters[bid] = StubAdapter(fail_first_n=1)


def build_req(b, c):
    return NormalizedRequest(alias="t", task="text2img")


# scenario 1: fault once, succeed on the self-retry
asyncio.run(main._run_job("job-a", [(backend, {})], build_req))
assert calls["attempts"] == 2, calls
assert calls["completed"] is not None and calls["completed"].get("attempts") == 2, calls
assert calls["failed"] is None, calls
assert main.backend_inflight.get(bid, 0) == 0, "slot leaked"
assert len(main.backend_gen_window[bid]) == 2   # 1 fail + 1 ok recorded

# scenario 2: every attempt faults → 2 self-attempts, then final fail with meta
calls.update(attempts=0, completed=None, failed=None)
main.backend_adapters[bid] = StubAdapter(fail_first_n=99)
asyncio.run(main._run_job("job-b", [(backend, {})], build_req))
assert calls["attempts"] == 2, calls
assert calls["completed"] is None, calls
assert calls["failed"] is not None and calls["failed"][1] == {"attempts": 2}, calls
assert main.backend_inflight.get(bid, 0) == 0, "slot leaked"

print("OK: self-retry, attempts meta, slot release, window recording")
```

Run: `venv/bin/python .smoke_run_job.py`
Expected: `OK: self-retry, attempts meta, slot release, window recording`

- [ ] **Step 4: Verify default behavior unchanged**

In the same file style, quick check: with `self_retries` absent, one fault must fail over immediately (attempts == 1 per backend). Append to the smoke file and re-run:

```python
# scenario 3: no self_retries → exactly one attempt, then failover exhausts
calls.update(attempts=0, completed=None, failed=None)
b2 = {"name": "fake2", "type": "comfyui", "url": "http://127.0.0.1:9", "priority": 1}
bid2 = main.backend_id(b2)
main.backend_adapters[bid2] = StubAdapter(fail_first_n=99)
asyncio.run(main._run_job("job-c", [(b2, {})], build_req))
assert calls["attempts"] == 1, calls
assert calls["failed"] is not None and calls["failed"][1] is None, calls
print("OK: default (self_retries unset) = today's behavior")
```

Expected: both OK lines.

- [ ] **Step 5: Remove the smoke file**

```bash
rm /home/dev/projekte/llm-gateway/.smoke_run_job.py
```

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "gen: self_retries — retry a faulted backend on itself before failing over (runbook B)"
```

---

### Task 3: Stage-1 self-retry in `_run_chain`

**Files:**
- Modify: `main.py` inside `_run_chain` (function starts at main.py:1963): counter init near `tried`/`skip_reason` (main.py:2028-2029), the `out1 = await adapter.generate(req1)` call (main.py:2165), the stage-2 `out2 = await adapter2.generate(req2)` call (main.py:2231), the meta build (main.py:2233), and both `except` blocks (main.py:2251-2266).

**Interfaces:**
- Consumes: `_record_gen_attempt`, `_wait_backend_up`, `jobs.fail(…, meta=…)` from Task 1. In-scope locals: `backend`, `bid`, `active`, `adapter`, `req1`, `s1_done`, `job_id`, `alias`, `succ_alias`.
- Produces: chain jobs honor stage-1 `self_retries`; chain meta carries `attempts` when > 1.

- [ ] **Step 1: Init the attempt counter**

After `skip_reason = None` (main.py:2029), add:

```python
    gen_attempts = 0                         # generate() calls across candidates + self-retries
```

- [ ] **Step 2: Wrap the stage-1 generate in the self-retry loop**

Replace the single line `out1 = await adapter.generate(req1)` (main.py:2165) with:

```python
            # runbook B: retry a sporadic fault on the SAME backend first — the held
            # slot (`held`) spans the repeats; the last attempt re-raises into the
            # existing stage-1 failover (next candidate via `tried`).
            s1_tries = 1 + max(0, int(backend.get("self_retries") or 0))
            for s1_attempt in range(1, s1_tries + 1):
                gen_attempts += 1
                try:
                    out1 = await adapter.generate(req1)
                    _record_gen_attempt(bid, conn_fail=False)
                    break
                except _GEN_FAILOVER_ERRORS as e:
                    if s1_attempt >= s1_tries:
                        raise                # outer handler records + fails over
                    _record_gen_attempt(bid, conn_fail=True)
                    logger.warning(f"✗ chain job {job_id} stage 1 [{backend['name']}] "
                                   f"connection issue ({type(e).__name__}: {e}) — retrying "
                                   f"same backend (self-retry {s1_attempt}/{s1_tries - 1})")
                    await _wait_backend_up(backend)
```

(Re-running `adapter.generate(req1)` re-uploads the job's inputs under the same `gw_<jobid>_s1_…` names — same job, so input isolation holds.)

- [ ] **Step 3: Count + record stage 2**

Directly BEFORE `out2 = await adapter2.generate(req2)` (main.py:2231) add the counter bump, and record the success after it:

```python
            gen_attempts += 1
            out2 = await adapter2.generate(req2)
            _record_gen_attempt(bid2, conn_fail=False)
```

- [ ] **Step 4: Surface attempts in chain meta and fails**

Meta build (main.py:2233) — extend:

```python
            meta = {**out2.meta, "backend": backend2["name"], "chain": [alias, succ_alias]}
            if gen_attempts > 2:             # a clean chain is exactly 2 generate() calls
                meta["attempts"] = gen_attempts
```

Outer `except _GEN_FAILOVER_ERRORS as e:` block (main.py:2251) — first line inside, record the fault against the live backend, and give the two `jobs.fail` calls in the connection branch the meta:

```python
        except _GEN_FAILOVER_ERRORS as e:
            _record_gen_attempt(backend_id(active), conn_fail=True)
            if s1_done:                              # mesh already relayed — a stage-2 loss is final
                logger.warning(f"✗ chain job {job_id} [{active['name']}] ({alias}→{succ_alias}) failed: {e}")
                await asyncio.to_thread(jobs.fail, job_id, f"chain failed: {e}",
                                        {"attempts": gen_attempts} if gen_attempts > 2 else None)
                asyncio.create_task(_free_comfy_vram(active, "chain failure"))
                return
            logger.warning(f"✗ chain job {job_id} stage 1 [{backend['name']}] connection issue "
                           f"({type(e).__name__}: {e}) — failing over")
            tried.add(backend["name"])
            asyncio.create_task(_free_comfy_vram(backend, "chain stage-1 failure"))
            continue
```

The generic `except Exception as e:` (main.py:2262) keeps its body but its `jobs.fail` also gets the meta argument:

```python
            await asyncio.to_thread(jobs.fail, job_id, f"chain failed: {e}",
                                    {"attempts": gen_attempts} if gen_attempts > 2 else None)
```

- [ ] **Step 5: Compile-gate + boot check**

```bash
venv/bin/python -m py_compile main.py
timeout 12 venv/bin/uvicorn main:app --port 4599 & sleep 6 && curl -s localhost:4599/health | head -c 400; wait
```

Expected: compile exit 0; `/health` returns the JSON snapshot (status ok).

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "gen: chain stage-1 honors self_retries; attempts surface in chain meta"
```

---

### Task 4: Surface `fail_rate` in `/health` and the Backends tab

**Files:**
- Modify: `main.py:2886-2895` (`_comfy_watch_info`)
- Modify: `admin.py:945-950` (Backends tab `render()` sub-line)

**Interfaces:**
- Consumes: `_gen_fail_stats` (Task 1). `_comfy_watch_info` is merged into BOTH `/health` (main.py:3147) and `gateway_info()` (main.py:2913) — one edit covers both surfaces.
- Produces: `/health` + UI backend dicts gain `fail_rate` / `gen_fails` / `gen_attempts` (only when data exists).

- [ ] **Step 1: Extend `_comfy_watch_info`**

Replace main.py:2886-2895 with:

```python
def _comfy_watch_info(b: dict) -> dict:
    """Executor-watchdog + rolling fail-rate fields for comfy backends (merged
    into /health + UI snapshot). fail_rate is display-only (runbook C): the
    operator decides — routing control stays priority (A1)."""
    if b.get("type") != "comfyui":
        return {}
    info: dict = {}
    fs = _gen_fail_stats(backend_id(b))
    if fs:
        info.update(fs)
    ad = backend_adapters.get(backend_id(b))
    if ad is not None:
        info.update({"exec_stuck": bool(getattr(ad, "exec_stuck", False)),
                     "last_restart": int(ad.last_restart) if getattr(ad, "last_restart", 0.0) else None,
                     "last_restart_result": getattr(ad, "last_restart_result", "") or None})
    return info
```

- [ ] **Step 2: Show it in the Backends tab**

In `admin.py` `render()` (the Backends tab item renderer), before the `sub = …` line (admin.py:949), add and use a fail-rate fragment:

```python
        fr = (f" · fail_rate {b['fail_rate']:.2f} ({b['gen_fails']}/{b['gen_attempts']})"
              if b.get("fail_rate") is not None else "")
        sub = f"{b['url']}{host} · prio {b['priority']} · {b['models']} models{flags}{fr}{rst}{src}"
```

- [ ] **Step 3: Compile-gate + render check**

```bash
venv/bin/python -m py_compile main.py admin.py
timeout 12 venv/bin/uvicorn main:app --port 4599 & sleep 6 \
  && curl -s localhost:4599/health | python3 -m json.tool | head -40 \
  && curl -s -o /dev/null -w "backends tab: %{http_code}\n" localhost:4599/ui/backends; wait
```

Expected: compile exit 0; `/health` JSON renders; `/ui/backends` answers 200 (or 303 to the unlock page if this dev store is locked — both prove the route renders without a 500).

- [ ] **Step 4: Commit**

```bash
git add main.py admin.py
git commit -m "gen: rolling per-backend fail_rate in /health + Backends tab (runbook C, display-only)"
```

---

### Task 5: `self_retries` in the backend editor

**Files:**
- Modify: `admin.py:850-856` (comfy watchdog form block)
- Modify: `admin.py:1120` (numeric parse tuple)

**Interfaces:**
- Consumes: nothing new — backends are free-form JSON in the store; unknown keys already round-trip through the editor (`b` is the original dict, mutated).
- Produces: operators can set/clear `self_retries` per backend in `/ui/backends`.

- [ ] **Step 1: Add the form field**

In the comfy section of the backend form, after the `stuck after s` field (admin.py:850-851) and before its hint paragraph, add:

```python
            + _field("self retries", _inp("self_retries", g("self_retries"),
                     placeholder="0", typ="number"))
```

And extend that hint paragraph (admin.py:852-856) with one sentence before the closing `</p>`:

```
 <b>self retries</b>: a connection-type fault mid-job retries the <b>same</b> backend
 this many times (after waiting for <code>/system_stats</code>) before failing over —
 for hosts with sporadic driver faults. Blank/0 = fail over immediately; content
 errors are never retried.
```

- [ ] **Step 2: Parse it**

Extend the numeric loop tuple (admin.py:1120):

```python
    for nkey in ("restart_cooldown_s", "stuck_after_s", "self_retries"):
```

(The existing `int(v) > 0`-else-pop semantics are correct here: 0/blank = key absent = default 0.)

- [ ] **Step 3: Compile-gate + render check**

```bash
venv/bin/python -m py_compile admin.py
timeout 12 venv/bin/uvicorn main:app --port 4599 & sleep 6 \
  && curl -s -o /dev/null -w "backends tab: %{http_code}\n" localhost:4599/ui/backends; wait
```

Expected: compile exit 0; route answers without 500.

- [ ] **Step 4: Commit**

```bash
git add admin.py
git commit -m "ui: self_retries field on the backend editor (comfy watchdog block)"
```

---

### Task 6: Strix registration config (runbook A) — quartermaster repo

**Files:**
- Modify: `/home/dev/projekte/quartermaster/hosts/ct452/gw-register-strix.py` (`BACKEND` dict + A2 check in `main()`)

**Interfaces:**
- Consumes: gateway backend keys `priority`, `disconnect_grace`, `stuck_after_s`, `auto_restart`, `restart_cooldown_s`, `self_retries` (Task 2/adapters); store tables `backends`, `gen_aliases` (`candidates_json`).
- Produces: a registration run that satisfies runbook A1/A2/A3 + B for `comfyui-strix`.

- [ ] **Step 1: Fix the priority (A1) and add the fault-handling params (A3 + B)**

Replace the `BACKEND` dict in `gw-register-strix.py`:

```python
BACKEND = {
    "name": "comfyui-strix",
    "type": "comfyui",
    "url": "http://192.168.8.228:8188",
    # Runbook A1: NICHT letzte Wahl! Bestand k12-gpu=5, evo-x2-gpu=8, dx10-02=10 —
    # Strix auf 9, damit dx10-02 hinter ihm steht und einen Fault auffangen kann.
    # (Prio 90 = "gewinnt nur, wenn sonst nichts frei ist" waere genau der Fall
    # OHNE Failover-Ziel.)
    "priority": 9,
    "max_concurrent": 1,
    "enabled": True,
    "comfy_output_dir": "/opt/ComfyUI/output",
    "host": "192.168.8.228",
    # Runbook A3 + B (Fehlerhandling fuer sporadische gfx1151-Faults):
    "disconnect_grace": 15,      # Default 30 — halbiert die pro Fault verlorene Wartezeit
    "stuck_after_s": 60,         # Default 90
    "auto_restart": True,        # ComfyUI-Manager-Reboot bei wedged Executor
    "restart_cooldown_s": 300,
    "self_retries": 1,           # 1 Wiederholung auf Strix selbst, dann Failover
}
```

- [ ] **Step 2: A2 guard — warn on per-alias `retries`**

`retries` lives on every candidate dict in `gen_aliases.candidates_json` (the alias editor mirrors it onto all candidates). In `main()`, inside the existing `for alias, spec in PLAN.items():` loop right after `row = cur.execute(…).fetchone()` and its existing missing-row handling, add:

```python
        cands_now = json.loads(row[0]) if row and row[0] else []
        capped = next((c.get("retries") for c in cands_now
                       if c.get("retries") not in (None, "")), None)
        if capped is not None:
            print(f"  WARNUNG {alias}: retries={capped} gesetzt — kappt die "
                  f"Failover-Kette (Runbook A2: Feld leeren!)")
```

- [ ] **Step 3: Verify**

```bash
python3 -m py_compile /home/dev/projekte/quartermaster/hosts/ct452/gw-register-strix.py
```

Expected: exit 0. (A `--dry` run needs the prod store on .10 — that happens in Task 7.)

- [ ] **Step 4: Commit (quartermaster repo)**

```bash
cd /home/dev/projekte/quartermaster
git add hosts/ct452/gw-register-strix.py
git commit -m "ct452: strix backend per Runbook — prio 9 (A1), Fault-Params (A3), self_retries 1 (B), A2-Warnung"
```

---

### Task 7: Deploy to the gateway (.10) and verify — **gated on the user**

⚠️ **Do not restart the prod gateway without the user's go-ahead** (project rule: restart the one instance when the user says idle). Everything before the restart is safe to prepare.

**Files:** none (ops task). Prod topology: gateway runs on `root@192.168.8.10:/opt/llm-gateway`, no rsync on the box → scp per file (see project memory).

- [ ] **Step 1: Compile-gate everything locally**

```bash
cd /home/dev/projekte/llm-gateway && venv/bin/python -m py_compile *.py
```

Expected: exit 0. Never scp without this — a broken file silently fails the restart.

- [ ] **Step 2: Ship the three changed files**

```bash
scp main.py jobs.py admin.py root@192.168.8.10:/opt/llm-gateway/
```

- [ ] **Step 3: Ask the user for an idle window, then restart**

```bash
ssh root@192.168.8.10 "systemctl restart llm-gateway"
```

- [ ] **Step 4: Verify on prod**

```bash
ssh root@192.168.8.10 "curl -s localhost:4000/health" | python3 -m json.tool | head -60
```

Expected: status ok; comfy backends present. (`fail_rate` appears only once generation attempts happened.)

- [ ] **Step 5: Register/refresh strix (when CT 452 is being attached)**

On .10: `python3 gw-register-strix.py --dry` (review, incl. A2 warnings) then `--apply`, then restart per the script's instruction and check `/health` shows `comfyui-strix` with `priority: 9`. In `/ui/backends`, confirm the strix editor shows `self retries = 1`.

- [ ] **Step 6: Live fault check (optional but recommended)**

Start a generation on strix, `systemctl restart comfyui` on CT 452 mid-job, and confirm in the gateway log: one `retrying same backend (self-retry 1/1)` line, then either a completed job with `"attempts": 2` in its meta (Jobs tab) or a failover to `dx10-02`; `/health` now shows a `fail_rate` for strix.

---

## Out of scope

- **D (fail-rate as routing criterion)** — runbook: only after C proves stable in practice.
- No retry on content errors, no health-exclusion after a crash, no automatic `max_concurrent` reduction (runbook "NICHT bauen").
- LLM/chat path (`_dispatch_or_park`) — the runbook targets generation only.
