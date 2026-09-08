# Unified Queue Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One scheduling rule set for ALL gateway requests: always queue, fastest free
backend wins, freed backends prefer the type they just ran, an overdue timer guards
fairness — priority and the speed toggle are removed.

**Architecture:** A new pure-logic `scheduler.py` (no `main` import, unit-tested) holds
the selection functions; `main.py` wires them into the existing dispatch/park loops
without changing task lifecycle, cancel or timeout plumbing. The wake mechanism stays a
broadcast; every waiter self-checks whether it is the designated taker.

**Tech Stack:** Python 3 stdlib, FastAPI app state in `main.py`, unittest test files run
directly with the venv interpreter (pattern: `test_prune_branch.py`). No pytest.

**Spec:** `docs/superpowers/plans/2026-09-01-unified-queue-scheduler-spec.md` — read it
first; every rule referenced below (type key, paid tier, overdue, removals) is defined there.

## Global Constraints

- Repo conventions: `CLAUDE.md` — modules other than `main.py` NEVER import `main`;
  compile-gate `venv/bin/python -m py_compile *.py` before every commit; no pytest.
- Work in `/home/dev/projekte/llm-gateway`, branch `master`, commit per task with
  `git commit -- <paths>` (never a bare `git commit -a`).
- Commit trailer (exact):
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- All UI strings English. Comments English.
- Config default: `affinity_max_wait_s = 120.0` (float seconds).
- EMA alpha for gen speed: `0.3`.
- Unmeasured speed sorts FIRST (probe-once convention, spec "Speed").
- Do NOT touch: auth, quotas, LoRA library, drain, health probing, VRAM freeing,
  chain hand-off semantics from commit 6965382.

---

### Task 1: scheduler.py — pure selection logic

**Files:**
- Create: `scheduler.py`
- Create: `test_scheduler.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces (used verbatim by Tasks 2–5):
  - `scheduler.EMA_ALPHA: float = 0.3`
  - `scheduler.ema(prev: Optional[float], sample: float, alpha: float = EMA_ALPHA) -> float`
  - `scheduler.order_ready(cands: list, speed_of, paid_of) -> list` — sort
    `(backend, x)` pairs: unpaid before paid, then faster first. `speed_of(backend, x)`
    returns higher-is-better float, `float("inf")` = unmeasured/probe-first.
    `paid_of(backend)` returns bool. Stable for equal keys.
  - `scheduler.designated_taker(pool, can_serve, type_key, last_key, now, max_wait_s)`
    — returns the entry a freed backend should take, or `None`. `pool` is an iterable
    of dict-like entries in enqueue order (oldest first), each with key
    `"enqueued_at"` (monotonic seconds). `can_serve(entry) -> bool`,
    `type_key(entry) -> Optional[str]` (the entry's type key ON this backend),
    `last_key: Optional[str]` (what the backend ran last). Rules, in order:
    overdue (`now - enqueued_at > max_wait_s`) oldest first → same-type oldest first
    → oldest servable.

- [ ] **Step 1: Write the failing tests** — `test_scheduler.py`:

```python
"""Unit tests for scheduler.py — run: venv/bin/python test_scheduler.py"""
import unittest

import scheduler


def _e(name, age, key):
    return {"name": name, "enqueued_at": 1000.0 - age, "key": key}


class TestEma(unittest.TestCase):
    def test_first_sample_is_taken_verbatim(self):
        self.assertEqual(scheduler.ema(None, 12.0), 12.0)

    def test_ema_blends_with_alpha(self):
        self.assertAlmostEqual(scheduler.ema(10.0, 20.0, alpha=0.3), 13.0)


class TestOrderReady(unittest.TestCase):
    B = [({"name": "slow", "paid": False}, "m"),
         ({"name": "fast", "paid": False}, "m"),
         ({"name": "cloud", "paid": True}, "m"),
         ({"name": "new", "paid": False}, "m")]

    def _speed(self, b, x):
        return {"slow": 5.0, "fast": 50.0, "cloud": 500.0,
                "new": float("inf")}[b["name"]]

    def test_unpaid_beats_paid_and_speed_orders_within_tier(self):
        got = scheduler.order_ready(list(self.B), self._speed, lambda b: b["paid"])
        self.assertEqual([b["name"] for b, _ in got],
                         ["new", "fast", "slow", "cloud"])

    def test_stable_for_equal_keys(self):
        pair = [({"name": "a", "paid": False}, 1), ({"name": "b", "paid": False}, 2)]
        got = scheduler.order_ready(pair, lambda b, x: 1.0, lambda b: False)
        self.assertEqual([b["name"] for b, _ in got], ["a", "b"])


class TestDesignatedTaker(unittest.TestCase):
    def _pick(self, pool, last_key, now=1000.0, max_wait=120.0,
              unservable=()):
        return scheduler.designated_taker(
            pool,
            can_serve=lambda e: e["name"] not in unservable,
            type_key=lambda e: e["key"],
            last_key=last_key, now=now, max_wait_s=max_wait)

    def test_empty_pool_returns_none(self):
        self.assertIsNone(self._pick([], "x"))

    def test_oldest_wins_without_affinity(self):
        pool = [_e("old", 50, "a"), _e("young", 10, "b")]
        self.assertEqual(self._pick(pool, last_key=None)["name"], "old")

    def test_same_type_beats_older_other_type(self):
        pool = [_e("old-a", 50, "a"), _e("young-b", 10, "b")]
        self.assertEqual(self._pick(pool, last_key="b")["name"], "young-b")

    def test_overdue_beats_affinity(self):
        pool = [_e("overdue-a", 200, "a"), _e("young-b", 10, "b")]
        self.assertEqual(self._pick(pool, last_key="b")["name"], "overdue-a")

    def test_oldest_overdue_wins_among_overdue(self):
        pool = [_e("older", 300, "a"), _e("newer", 200, "b")]
        self.assertEqual(self._pick(pool, last_key="b")["name"], "older")

    def test_unservable_entries_are_skipped(self):
        pool = [_e("cant", 300, "a"), _e("can", 10, "b")]
        self.assertEqual(self._pick(pool, last_key=None,
                                    unservable={"cant"})["name"], "can")

    def test_all_unservable_returns_none(self):
        pool = [_e("cant", 300, "a")]
        self.assertIsNone(self._pick(pool, last_key="a", unservable={"cant"}))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure** — `venv/bin/python test_scheduler.py` must fail
  with `ModuleNotFoundError: No module named 'scheduler'`.

- [ ] **Step 3: Implement `scheduler.py`:**

```python
"""Unified queue scheduler — pure selection logic (spec:
docs/superpowers/plans/2026-09-01-unified-queue-scheduler-spec.md).

All state comes in as arguments; this module never imports main (hot-reload-safe,
unit-testable). Three rules replace priority/speed routing everywhere:
unpaid-then-fastest ordering, freed-backend type affinity, and an overdue guard.
"""
from typing import Callable, Iterable, Optional

EMA_ALPHA = 0.3


def ema(prev: Optional[float], sample: float, alpha: float = EMA_ALPHA) -> float:
    """Exponential moving average; the first sample is taken verbatim."""
    return sample if prev is None else alpha * sample + (1 - alpha) * prev


def order_ready(cands: list, speed_of: Callable, paid_of: Callable) -> list:
    """Order ready (backend, x) candidates for dispatch: unpaid before paid, then
    fastest first. speed_of(backend, x) is higher-is-better; float('inf') marks an
    unmeasured candidate, which therefore sorts first within its tier (probe-once).
    Stable, so equal candidates keep their incoming order."""
    return sorted(cands, key=lambda bx: (bool(paid_of(bx[0])), -speed_of(bx[0], bx[1])))


def designated_taker(pool: Iterable, can_serve: Callable, type_key: Callable,
                     last_key: Optional[str], now: float, max_wait_s: float):
    """The waiting entry a freed backend should take, or None.

    pool iterates in enqueue order (oldest first); entries expose ["enqueued_at"]
    (monotonic seconds). Rules, first match wins:
      1. overdue entries (waited > max_wait_s) the backend can serve — oldest first;
      2. entries whose type key equals what the backend last ran — oldest first;
      3. the oldest entry the backend can serve.
    """
    servable = [e for e in pool if can_serve(e)]
    if not servable:
        return None
    overdue = [e for e in servable if now - e["enqueued_at"] > max_wait_s]
    if overdue:
        return overdue[0]
    if last_key:
        same = [e for e in servable if type_key(e) == last_key]
        if same:
            return same[0]
    return servable[0]
```

- [ ] **Step 4: Run tests** — `venv/bin/python test_scheduler.py` → all pass.
- [ ] **Step 5: Compile-gate + commit**

```bash
venv/bin/python -m py_compile *.py
git add scheduler.py test_scheduler.py
git commit -m "sched: pure selection logic for the unified queue scheduler" -- scheduler.py test_scheduler.py
```

---

### Task 2: media speed metric + backend_last_key state

**Files:**
- Modify: `main.py` (globals near `backend_tps` ≈ line 290; `_run_job`; `_run_chain`;
  startup/lifespan where backends are first refreshed)
- Modify: `stats.py` (one new read helper)

**Interfaces:**
- Consumes: `scheduler.ema`.
- Produces (used by Tasks 3–5):
  - `gen_speed: dict = {}` in main — key `f"{alias}|{bid}"` → EMA duration seconds.
  - `backend_last_key: dict = {}` in main — `bid` → type key str (LLM real model or
    media alias), set at DISPATCH time.
  - `def _note_gen_speed(alias: str, bid: str, seconds: float) -> None` in main.
  - `def _gen_speed_of(alias: str)` in main — returns a `speed_of(backend, cand)`
    callable for `scheduler.order_ready`: `float("inf")` when unmeasured, else
    `1.0 / max(seconds, 0.001)`.
  - `stats.gen_speed_rows() -> list[tuple[str, str, float]]` — (alias, backend_name,
    avg_duration_ms) of successful media calls.

- [ ] **Step 1: main.py state + helpers.** Next to `backend_tps` add:

```python
gen_speed: dict = {}        # "alias|bid" → EMA duration seconds of successful media jobs
backend_last_key: dict = {} # bid → type key last DISPATCHED (media: alias, LLM: real model)


def _note_gen_speed(alias: str, bid: str, seconds: float) -> None:
    if seconds <= 0:
        return
    k = f"{alias}|{bid}"
    gen_speed[k] = scheduler.ema(gen_speed.get(k), seconds)


def _gen_speed_of(alias: str):
    """speed_of callable for scheduler.order_ready over media candidates."""
    def speed(backend: dict, _cand) -> float:
        s = gen_speed.get(f"{alias}|{backend_id(backend)}")
        return float("inf") if s is None else 1.0 / max(s, 0.001)
    return speed
```

Add `import scheduler` next to the other local-module imports (`import store`).

- [ ] **Step 2: record durations.** Wrap the successful `generate()` calls with a
  monotonic clock and feed `_note_gen_speed`:
  - `_run_job`: `t0 = time.monotonic()` immediately before `out = await
    adapter.generate(req)`; after `_record_gen_attempt(bid, conn_fail=False)` add
    `_note_gen_speed(alias, bid, time.monotonic() - t0)`. `_run_job` does not know the
    alias yet → change its signature to `async def _run_job(job_id, alias, candidates,
    build_req)` and update ALL call sites (grep `_run_job(` — sync gen path, async gen
    path, `_run_gen_parked`).
  - `_run_chain` stage 1: around `out1 = await adapter.generate(req1)` (inside the
    self-retry loop, success branch only) → `_note_gen_speed(alias, bid, dt)`.
  - `_run_chain` stage 2: around `out2 = await adapter2.generate(req2)` →
    `_note_gen_speed(succ_alias, bid2, dt)`.

- [ ] **Step 3: stats seed.** In `stats.py` add (guard `_enabled`/connection the same way
  the neighbouring read helpers do — copy their pattern):

```python
def gen_speed_rows() -> list:
    """(alias, backend, avg_duration_ms) of successful media calls — boot seed for
    the scheduler's gen-speed EMA."""
    return _q("SELECT alias, backend, AVG(duration_ms) FROM calls "
              "WHERE status='ok' AND alias!='' AND backend!='' "
              "AND endpoint LIKE '%/generations%' "
              "GROUP BY alias, backend")
```

First INSPECT actual media rows (`sqlite3 stats.db "SELECT DISTINCT endpoint, source
FROM calls LIMIT 30"`) and adjust the WHERE to whatever tags media generations (e.g.
`source='media'` or the images endpoints) — the filter above is the starting guess,
the real one comes from the data. In `main.py` startup (where the first
`refresh_backend` sweep completes / lifespan start), seed best-effort:

```python
try:
    for alias, bname, avg_ms in stats.gen_speed_rows():
        b = next((b for b in backends if b["name"] == bname
                  and b.get("type") == "comfyui"), None)
        if b is not None and avg_ms:
            gen_speed.setdefault(f"{alias}|{backend_id(b)}", float(avg_ms) / 1000.0)
except Exception as e:
    logger.info(f"gen-speed seed skipped: {e}")
```

- [ ] **Step 4: compile-gate + unit tests still green** —
  `venv/bin/python -m py_compile *.py && venv/bin/python test_scheduler.py`
- [ ] **Step 5: commit** — `git commit -m "sched: measure media job speed per
  alias+backend, seed from stats" -- main.py stats.py`

---

### Task 3: LLM routing — fastest-free ordering, paid tier, speed/priority removal

**Files:**
- Modify: `main.py` — `resolve_routes` (≈984), `rebuild_route_index` (≈937),
  `load_config`/settings loader (route_mode block ≈176–191), globals (`route_mode`
  ≈338, `_speed_keys` ≈935), `/health` builder (≈1121, drops `priority`, adds `paid`),
  `alias_entry` docstring (priority now ignored), `_DEFAULT_PRIORITY` **stays for now**
  (media path, removed in Task 5).

**Interfaces:**
- Consumes: `scheduler.order_ready`, `backend_tps`, `backend_last_key` (set only).
- Produces: `resolve_routes` returns ready ALREADY ordered by the new rule; a helper
  `def _llm_speed_of(backend, real) -> float` (tps, `inf` unmeasured); backends carry
  a normalized `paid: bool` (config + store backends: `b["paid"] =
  bool(b.get("paid"))` wherever backends are normalized on load).

- [ ] **Step 1: ordering.** In `resolve_routes`, replace the `_speed_keys` block:

```python
    if len(ready) > 1:
        # Unified scheduling (spec 2026-09-01): unpaid before paid, then fastest
        # first by measured tok/s; unmeasured backends sort first so each gets
        # probed once. Priority and the per-key speed switch are gone.
        ready = scheduler.order_ready(
            ready,
            lambda b, real: backend_tps.get(backend_id(b), float("inf")),
            lambda b: bool(b.get("paid")))
        # Shared-GPU consideration: unchanged, stays dominant (after the sort).
        mb = _media_busy_hosts()
        if mb:
            ready.sort(key=lambda br: backend_hosts.get(backend_id(br[0]), "") in mb)
    return ready, busy
```

- [ ] **Step 2: removals.** Delete: `route_mode` global + its load block, `_speed_keys`
  global + every read, the `route_speed` handling inside `rebuild_route_index` (the
  index keeps per-alias candidate lists but no longer pre-sorts by priority — keep
  insertion order; per-alias priority overrides from `alias_entry` are ignored, note it
  in the docstring, keep the tuple shape `(real, prio)` so store data still parses).
  `resolve_routes` docstring: "priority order" wording → describe the new order.
  Grep check: `grep -n "route_mode\|_speed_keys\|route_speed" main.py` must show ZERO
  functional hits after this task (store/admin cleanup follows in Task 6 — main.py must
  no longer call `store.get_route_mode`).
- [ ] **Step 3: paid flag + health.** Normalize `paid` where backends load; in the
  `/health` backends dict swap `"priority": …` for `"paid": bool(b.get("paid"))`.
- [ ] **Step 4: last_key at dispatch.** In the chat dispatch path (the function that
  walks `get_routes_for`/`resolve_routes` candidates and calls the adapter — find via
  `grep -n "get_routes_for\|resolve_routes" main.py` call sites), set
  `backend_last_key[backend_id(b)] = real` right where the winning backend is chosen
  (immediately before the adapter call), for BOTH the direct and the parked-resume
  path.
- [ ] **Step 5: compile-gate + tests + quick boot check** — `venv/bin/python -m
  py_compile *.py && venv/bin/python test_scheduler.py`; then
  `venv/bin/python -c "import main"` must import cleanly (config.yaml present).
- [ ] **Step 6: commit** — `git commit -m "sched: LLM routing goes unpaid-then-fastest,
  speed toggle and priority sort removed" -- main.py`

---

### Task 4: LLM parking — affinity + overdue guard + config knob

**Files:**
- Modify: `main.py` — `_park_and_dispatch` (≈1548), the parked-entry constructor
  (≈1491/1554), settings loader (≈3527), the `/ui/input` field list (≈5388),
  globals (new `affinity_max_wait_s`).

**Interfaces:**
- Consumes: `scheduler.designated_taker`, `backend_last_key`, `affinity_max_wait_s`.
- Produces: `affinity_max_wait_s: float = 120.0` global, hot-reloaded from
  `settings.affinity_max_wait_s`, editable on /ui/input.

- [ ] **Step 1: config knob.** Global `affinity_max_wait_s: float = 120.0`; parse in the
  settings loader exactly like `async_park_timeout_s` (float, try/except); add an
  input-page row to the field list: `("affinity_max_wait_s", "float", "affinity max
  wait", "seconds — a queued request older than this beats the same-type preference
  and takes the next free backend")`.
- [ ] **Step 2: entry fields.** Every `_parked` entry gets `"enqueued_at":
  time.monotonic()` at append time (keep existing fields; the entry already carries its
  alias — verify the key name by reading the constructor).
- [ ] **Step 3: the gate.** In `_park_and_dispatch`'s wake loop, when the entry finds
  `ready` candidates for its alias: order them (Task 3 already returns them ordered);
  for the best candidate backend `B` compute:

```python
            chosen = scheduler.designated_taker(
                _parked,
                can_serve=lambda e: any(backend_id(rb) == backend_id(B)
                                        for rb, _ in resolve_routes(e["alias"], e.get("path") or "/v1/chat/completions")[0]),
                type_key=lambda e: (alias_entry(e["alias"], B["name"])[0] or e["alias"]),
                last_key=backend_last_key.get(backend_id(B)),
                now=time.monotonic(), max_wait_s=affinity_max_wait_s)
            if chosen is not entry:
                # someone else is designated for this slot — wait for the next wake
                continue-or-resleep  # ← adapt to the loop's actual control flow
```

  Read the real loop first: keep its timeout/cancel/removal handling EXACTLY, only add
  the gate between "routes found" and "claim". `e["alias"]`/`e["path"]` key names must
  match the real entry dict (adjust after reading). If the pool has only one entry the
  gate must trivially pass (designated_taker returns it).
- [ ] **Step 4: compile-gate + tests** (as Task 3 Step 5).
- [ ] **Step 5: commit** — `git commit -m "sched: parked LLM calls resume by type
  affinity with an overdue guard" -- main.py`

---

### Task 5: media queue — waiting registry, gated claims, priority removal

**Files:**
- Modify: `main.py` — `_gen_routes` (≈2030), `_gen_pick` (≈2900), `_run_gen_parked`
  (≈2820), `_run_job` (≈2320, sets last_key), `_run_chain` (stage-1 park loop ≈2480 and
  both claim points), `_DEFAULT_PRIORITY` (delete).

**Interfaces:**
- Consumes: `scheduler.order_ready`, `scheduler.designated_taker`, `_gen_speed_of`,
  `backend_last_key`, `affinity_max_wait_s`.
- Produces: `_gen_waiting: list = []` — entries
  `{"job_id", "alias", "enqueued_at", "eligible", "force"}` in enqueue order.

- [ ] **Step 1: ordering in `_gen_routes`.** Replace the priority sort
  (`allc.sort(key=lambda bc: bc[0].get("priority", _DEFAULT_PRIORITY))`) with:

```python
    allc = scheduler.order_ready(allc, _gen_speed_of(alias),
                                 lambda b: bool(b.get("paid")))
```

  Delete `_DEFAULT_PRIORITY`. The `retries` cap keeps operating on the ordered lists
  (unchanged code below the sort). Update the docstring ("backend priority order" →
  the new rule).
- [ ] **Step 2: registry.** Global `_gen_waiting: list = []`. `_run_gen_parked`
  registers `entry = {"job_id": job_id, "alias": alias, "enqueued_at":
  time.monotonic(), "eligible": eligible, "force": force}` before its while loop,
  removes it in a `finally:`. The chain's stage-1 park path (`_run_chain`, the
  `picked is None` branch and the successor pre-check park added in commit 6965382)
  registers/removes an equivalent entry around its while loop (`"job_id": job_id,
  "alias": alias`).
- [ ] **Step 3: the gate in `_run_gen_parked`.** Where `if ready:` currently dispatches,
  gate first — for the best ready backend `B = ready[0][0]`:

```python
        if ready:
            B = ready[0][0]
            chosen = scheduler.designated_taker(
                _gen_waiting,
                can_serve=lambda e: _entry_can_use(e, B),
                type_key=lambda e: e["alias"],
                last_key=backend_last_key.get(backend_id(B)),
                now=time.monotonic(), max_wait_s=affinity_max_wait_s)
            if chosen is None or chosen["job_id"] == job_id:
                await _run_job(job_id, alias, ready, build_req)
                return
            # a different waiter is designated for this slot — keep parking
```

  with a module-level helper:

```python
def _entry_can_use(entry: dict, backend: dict) -> bool:
    """May this waiting generation entry run on `backend` right now? Mirrors the
    entry's own filters: alias candidates, force pin, LoRA eligibility."""
    if entry.get("force") and backend.get("name") != entry["force"]:
        return False
    if entry.get("eligible") is not None and backend.get("name") not in entry["eligible"]:
        return False
    ready, _ = _gen_routes(entry["alias"])
    return any(backend_id(b) == backend_id(backend) for b, _ in ready)
```

  (`_gen_routes` is blocking-ish but store-read only; it already runs inside
  `asyncio.to_thread` in the poll — call `_entry_can_use` via the same
  `asyncio.to_thread` wrapping as the surrounding code if the loop does so.)
- [ ] **Step 4: fresh arrivals.** `_gen_pick` keeps its shape, but "immer in die Queue":
  when it returns `parked=False` (something free) the caller dispatches immediately —
  that stays; add the same designated gate there ONLY if `_gen_waiting` is non-empty
  (a fresh request must not overtake designated waiters):

```python
    if ready and (not _gen_waiting or scheduler.designated_taker(...) is None):
        return ready, False, eligible
    return allc, True, eligible
```

  (i.e. compute the taker for `ready[0][0]`; if some WAITING entry is designated, the
  fresh request parks instead of dispatching.)
- [ ] **Step 5: last_key.** `_run_job` sets `backend_last_key[bid] = alias` right after
  `_inflight_inc(bid)`. `_run_chain` sets it at the stage-1 claim
  (`backend_last_key[bid] = alias` after its `_inflight_inc(bid)`) and at the stage-2
  claim (`backend_last_key[bid2] = succ_alias` after `_wait_and_hold` succeeds).
- [ ] **Step 6: compile-gate + tests + import check** (as Task 3 Step 5).
- [ ] **Step 7: commit** — `git commit -m "sched: media jobs queue backend-independent,
  freed backends pull same-type first" -- main.py`

---

### Task 6: UI, store and docs cleanup

**Files:**
- Modify: `admin.py` — backends form (`priority` input ≈904, save ≈1339), backend
  badge/labels (≈1104, 1620), models-table routing chip + `rmode` toggle (≈1624–1658),
  chat-alias page `rmode_field` (≈1948–1992) and save (≈2035–2058), routing-page
  explanatory copy (≈963/976), the `/ui/routing/rmode` route registration in `main.py`
  (`rmode_toggle`).
- Modify: `store.py` — delete `get_route_mode`/`set_route_mode` (and their table
  bootstrap if dedicated; leave existing rows in place, they are simply unread).
- Modify: `README.md` (routing section, settings table), `CLAUDE.md` (architecture
  paragraph on routing), `config.example.yaml` (drop `priority`/`route_speed`/
  `route_mode` examples; add `paid: true` example on a cloud backend and
  `affinity_max_wait_s` under `settings:`).

**Interfaces:**
- Consumes: the `paid` flag from Task 3.
- Produces: backends form saves `paid` as checkbox → bool.

- [ ] **Step 1: backends page.** Replace the `priority` number input with a `paid`
  checkbox ("paid — used only when no unpaid backend is free"); save handler:
  `"paid": bool(f.get("paid"))` instead of the priority int. Remove `prio` from the
  backend subtitle line.
- [ ] **Step 2: speed-toggle removal.** Delete the routing-page priority⇄speed chip and
  its `rmode_toggle` handler + route registration; delete the chat-alias `rmode_field`
  select and its save handling; rewrite the two explanatory copy blocks (≈963/976) to
  describe the unified rule (one sentence each: "requests go to the fastest free
  unpaid backend; a freed backend prefers the type it just ran; requests waiting
  longer than affinity_max_wait_s take the next free backend").
- [ ] **Step 3: store cleanup.** Remove `get_route_mode`/`set_route_mode`; grep
  `route_mode` across `*.py` → zero hits.
- [ ] **Step 4: docs.** README: replace the priority/speed routing description with the
  spec's Dispatch rules (compact), document `paid` and `affinity_max_wait_s`;
  CLAUDE.md: update the one-line routing description; config.example.yaml as above.
- [ ] **Step 5: compile-gate + tests + import check; commit** — `git commit -m "sched:
  drop the speed toggle and priority UI, document the unified scheduler" -- admin.py
  store.py README.md CLAUDE.md config.example.yaml main.py`

---

### Task 7 (session lead, not a subagent): review, fix wave, deploy

- [ ] Full Fable review of `git diff <pre-task-1>..HEAD` against the spec (correctness,
  races around the no-await claim invariant, removed-symbol leftovers, UI strings).
- [ ] Fix findings, `venv/bin/python -m py_compile *.py`, `venv/bin/python
  test_scheduler.py`, `venv/bin/python test_prune_branch.py`.
- [ ] Deploy `DEPLOY_HOST=root@<prod-host> ./deploy.sh`; verify `/health` (backends
  show `paid`, no `priority`), journal clean, one real generation through the queue.
- [ ] Post-deploy: set `paid: true` on claude/openrouter/together in the prod config
  (UI or config.yaml — hot-reloaded), set `affinity_max_wait_s` if 120 is not wanted.
- [ ] Push `master`.

## Self-Review (done at plan time)

- Spec coverage: queue-always (T5 S2/S4, LLM parking pre-exists), fastest-free (T3 S1,
  T5 S1), single-free/none-free (unchanged park paths), freed-backend affinity (T4 S3,
  T5 S3), overdue guard (T1 designated_taker + knob T4 S1), removals (T3 S2, T5 S1,
  T6). Type keys per spec (T4 type_key = real model; T5 = alias). Paid tier (T3 S3,
  T6 S1). Seeding (T2 S3).
- Placeholders: none — every step names exact anchors or carries code; two spots
  direct the implementer to read the real control flow first (T4 S3, T2 S3) because
  line-exact code would be wrong by the time they run; the required behaviour is
  stated precisely.
- Type consistency: `order_ready(cands, speed_of, paid_of)`, `designated_taker(pool,
  can_serve, type_key, last_key, now, max_wait_s)`, `gen_speed` key `"alias|bid"`,
  `backend_last_key[bid]`, `affinity_max_wait_s` — spelled identically in Tasks 1–6.
