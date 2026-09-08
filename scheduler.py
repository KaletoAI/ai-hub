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


def free_vram_before_job(last_key: Optional[str], next_key: str,
                         others_inflight: int, enabled: bool = True) -> bool:
    """Must a ComfyUI backend's VRAM be freed BEFORE the job that is about to run?

    ComfyUI never releases its model cache by itself, and freeing it AFTER a job is
    the wrong moment twice over: it throws away a cache the very next job may want,
    and it cannot know what that job will be. Once a job is CLAIMED the answer is
    known, so the decision moves here:

      * the backend last ran the SAME type key (media: the alias = one workflow, one
        model set) → keep the cache. This is the payoff of the freed-backend type
        affinity in `designated_taker`, which steers a same-alias job here for
        exactly that reason;
      * anything else — a different alias, or nothing recorded (a gateway restart
        does not empty ComfyUI's VRAM: measured 2026-09-05, 21.4 of 23.5 GiB still
        held) → free, so the run starts against an empty GPU;
      * another job is in flight on that backend → never (the free would drop the
        cache under a RUNNING prompt). The caller's own slot is not counted.

    Pure — `enabled` carries the host policy, `others_inflight` the live counter.
    """
    if not enabled or others_inflight > 0:
        return False
    return last_key != next_key


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


# ── Execution-fault quarantine (per alias|backend) ─────────────────────────────
# A generation backend that ANSWERS but cannot EXECUTE is invisible to every other
# signal: discovery only calls /object_info so `backend_healthy` stays True, and the
# executor watchdog only sees a stuck queue — this one drains fine, it just turns
# every prompt into an error in three seconds. Worse, it is SELF-REINFORCING: a
# candidate that never succeeds never gets a gen_speed sample, so it keeps the
# unmeasured "probe-once" head start and wins the ordering again on the next retry
# (measured 2026-09-03: four consecutive retries all landed on the same broken box
# while two healthy ones sat idle).
#
# The signal used here is deliberately narrow: a fault counts ONLY when the same job
# then succeeded on another candidate. That is the difference between "this backend
# is broken" and "this request is broken" — without it, one bad model name would
# quarantine every backend an alias has. `main` therefore notes faults only after a
# later candidate returns artifacts, and notes NONE when they all failed alike.
EXEC_FAULT_THRESHOLD = 2       # consecutive proven faults before the candidate is held
EXEC_QUARANTINE_S = 900        # ... and for how long (15 min — long enough to matter,
                               # short enough that a fixed backend returns on its own)


def exec_fault_note(state: dict, key: str, now: float, error: str = "",
                    threshold: int = EXEC_FAULT_THRESHOLD,
                    quarantine_s: float = EXEC_QUARANTINE_S) -> dict:
    """Record one PROVEN execution fault for `key` (an "alias|backend id"), returning
    its record. At `threshold` consecutive faults the candidate is quarantined for
    `quarantine_s`. The record survives the quarantine window on purpose — see
    `exec_probed`."""
    e = state.get(key) or {"fails": 0, "until": 0.0, "error": "", "at": 0.0}
    e["fails"] += 1
    e["error"] = error
    e["at"] = now
    if e["fails"] >= threshold:
        e["until"] = now + quarantine_s
    state[key] = e
    return e


def exec_fault_clear(state: dict, key: str) -> None:
    """Forget `key` — called on every SUCCESS, so the count means *consecutive*
    faults and a backend that recovers is immediately first-class again."""
    state.pop(key, None)


def exec_quarantined(state: dict, key: str, now: float) -> bool:
    """Is this candidate currently held out of rotation?"""
    e = state.get(key)
    return bool(e) and now < e.get("until", 0.0)


def exec_probed(state: dict, key: str) -> bool:
    """Has this candidate already spent its probe (i.e. produced a proven fault)?

    Separate from `exec_quarantined` because it outlives the window: an unmeasured
    candidate sorts FIRST (probe-once), and a candidate that has failed has had that
    probe. Letting the head start come back when the quarantine expires would send
    the next job straight back to the broken backend."""
    return bool(state.get(key))


def split_quarantined(cands: list, key_of: Callable, state: dict, now: float) -> tuple:
    """(usable, held) over ordered candidates, preserving relative order.

    `held` is never ALL of them: if every candidate is quarantined the quarantine is
    ignored entirely and the caller gets its full list back. A blocked alias is worse
    than a slow one — the quarantine exists to prefer a working backend, not to refuse
    service when none is known to work."""
    usable, held = [], []
    for c in cands:
        (held if exec_quarantined(state, key_of(c), now) else usable).append(c)
    if not usable:
        return list(cands), []
    return usable, held
