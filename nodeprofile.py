"""Per-node timing profile of a ComfyUI workflow — pure, no I/O.

The live feed (`ComfyUIAdapter._ws_progress`) knows the step counter of the ONE node
that is sampling right now ("KSampler 25/35"). For a workflow whose sampler is a
fifth of the run — a mesh workflow that decodes, remeshes, textures and rigs after
it — that bar reads 100 % long before the job is anywhere near done, and its ETA
covers the sampler only.

So every finished run RECORDS how long each node took (`record` → the job's
`meta.node_times`), and the next run of the same alias builds a PROFILE from the
recent records (`build`: median seconds per node id) and reads its live state against
it (`overall`): done nodes + the running node's share, over the expected total of the
nodes that will execute this time. The ETA comes from the remaining expected seconds,
corrected by how fast THIS run has been so far.

Rules that would fail silently (pinned in tests/test_node_profile.py):
  - Nodes ComfyUI serves from its cache (`execution_cached`) are out of the plan: a
    warm second run skips the loaders, and counting them would hold the bar at 60 %.
  - A node this run's workflow does not contain is out of the plan (an edited alias
    keeps old records for a while).
  - A profile from ANOTHER backend of the alias (`foreign`) supplies the node SHARES
    only — its seconds belong to another GPU, so it gives no ETA until this run has
    measured its own pace.
  - Time is attributed to `display_node` when ComfyUI reports one: an expanded node
    runs as internal children, and the record must name the node of the workflow.
"""
from __future__ import annotations

from typing import Iterable, Optional

VERSION = 1
MAX_RECORDS = 10                 # the last N done runs of an alias(+backend) form a profile
FRACTION_CAP = 0.99              # only completion says 100 %
TIME_FRACTION_CAP = 0.95         # a node without a step counter: elapsed/expected, never "done"
SPEED_CLAMP = (0.33, 3.0)        # one odd node must not scale the whole ETA wildly
SPEED_MIN_EVIDENCE_S = 2.0       # expected seconds of finished nodes before the pace counts
SPEED_MIN_EVIDENCE_SHARE = 0.10  # … or this share of the expected total
LAZY_VISIT_S = 1.0               # a finished visit shorter than this AND than …
LAZY_VISIT_SHARE = 0.10          # … this share of the node's expected time is a lazy
                                 # node's first pass, not the node being done


# ── folding the live feed ────────────────────────────────────────────────────────────

def fold(run: dict, kind: str, data: dict, now: float) -> bool:
    """Fold one ComfyUI /ws message (already gated on prompt_id) into `run`, the
    per-run timing state. Returns True when the message changed what we know.

    `run` keys: done {node: seconds}, order [node…], cached set, cur, cur_at,
    exec_at, steps {node: max}."""
    if kind == "execution_start":
        run.setdefault("exec_at", now)
        return True
    if kind == "execution_cached":
        nodes = data.get("nodes") or []
        run.setdefault("cached", set()).update(str(n) for n in nodes)
        return bool(nodes)
    if kind == "executing":
        node = data.get("display_node") or data.get("node")
        run.setdefault("exec_at", now)
        if node is None:                       # pre-0.3.4x end-of-prompt marker
            close(run, now)
            return True
        node = str(node)
        if node == run.get("cur"):
            return False                       # an expanded node's next child
        close(run, now)
        run["cur"], run["cur_at"] = node, now
        return True
    if kind in ("execution_success", "execution_interrupted", "execution_error"):
        close(run, now)
        return True
    return False


def note_steps(run: dict, node: str, val: float, mx: float) -> None:
    """Remember the step count a node ran with — a profile scales that node's
    expected seconds when the next run asks for a different count.

    Some nodes count SEVERAL phases (a sampler stage, then a tiled decode; mesh nodes
    with chunked passes): a counter that restarts or changes its max opens a new
    phase. Such a node's counter says nothing about the node as a whole — neither its
    fraction nor a step-count scale — so it is counted (`phases`) and the node falls
    back to elapsed-vs-expected."""
    owner = _owner(run, node)
    if not owner:
        return
    phases = run.setdefault("phases", {})
    ph = phases.get(owner)
    if ph is None:
        ph = [1, val, mx]
    elif val < ph[1] or mx != ph[2]:
        ph = [ph[0] + 1, val, mx]
    else:
        ph[1] = val
    phases[owner] = ph
    run.setdefault("steps", {})[owner] = int(mx)


def _phases(run: dict, node: str) -> int:
    return ((run.get("phases") or {}).get(node) or [1])[0]


def close(run: dict, now: float) -> None:
    """End the running node (if any) and book its time."""
    cur = run.get("cur")
    if cur is None:
        return
    at = run.get("cur_at")
    secs = max(0.0, now - float(now if at is None else at))
    done = run.setdefault("done", {})
    if cur not in done:
        run.setdefault("order", []).append(cur)
    done[cur] = done.get(cur, 0.0) + secs
    run["cur"], run["cur_at"] = None, None


def _owner(run: dict, node: str) -> Optional[str]:
    """The workflow node a progress message belongs to: the running node itself or,
    for an expanded node, the display node its internal child id starts with."""
    cur = run.get("cur")
    if not node:
        return cur
    if cur and (node == cur or node.startswith(cur + ".")):
        return cur
    return node


# ── record (end of a successful run) ────────────────────────────────────────────────

def record(run: dict, titles: dict, now: float) -> Optional[dict]:
    """The `meta.node_times` of a finished run, or None when the feed told us nothing
    (no websocket, an all-cached rerun). Execution order is kept: the job page lists
    the nodes the way they ran."""
    close(run, now)
    done = run.get("done") or {}
    if not done or run.get("broken"):
        # a feed that dropped mid-run would book the whole remaining runtime on the
        # node that happened to be current — a skewed record poisons the median
        return None
    steps = run.get("steps") or {}

    def _steps(n):
        if _phases(run, n) > 1:
            return {"phases": _phases(run, n)}
        return {"steps": steps[n]} if steps.get(n) else {}
    nodes = [{"id": n, "title": str(titles.get(n) or n), "s": round(done[n], 2), **_steps(n)}
             for n in run.get("order") or [] if n in done]
    out = {"v": VERSION, "nodes": nodes,
           "cached": sorted(run.get("cached") or [], key=_id_key),
           "total_s": round(sum(done.values()), 2)}
    if run.get("exec_at") is not None and run.get("submit_at") is not None:
        out["wait_s"] = round(max(0.0, run["exec_at"] - run["submit_at"]), 2)
    return out


def _id_key(n: str):
    return (0, int(n), "") if str(n).isdigit() else (1, 0, str(n))


# ── profile (from the recent records) ───────────────────────────────────────────────

def build(records: Iterable[dict], foreign: bool = False) -> Optional[dict]:
    """Median seconds (and step count) per node over the given records. A node counts
    only from the runs it EXECUTED in — a cached run says nothing about its cost."""
    secs: dict = {}
    steps: dict = {}
    phases: dict = {}
    titles: dict = {}
    n = 0
    for rec in records:
        if not isinstance(rec, dict) or rec.get("v") != VERSION:
            continue
        n += 1
        for node in rec.get("nodes") or []:
            nid, s = str(node.get("id")), node.get("s")
            if not isinstance(s, (int, float)) or s < 0:
                continue
            secs.setdefault(nid, []).append(float(s))
            if isinstance(node.get("steps"), int) and node["steps"] > 0:
                steps.setdefault(nid, []).append(node["steps"])
            if isinstance(node.get("phases"), int) and node["phases"] > 1:
                phases[nid] = max(phases.get(nid, 1), node["phases"])
            titles.setdefault(nid, node.get("title"))
        if n >= MAX_RECORDS:
            break
    if not secs:
        return None
    nodes = {nid: {"s": _median(v), "title": titles.get(nid),
                   **({"phases": phases[nid]} if nid in phases
                      else ({"steps": int(_median(steps[nid]))} if nid in steps else {}))}
             for nid, v in secs.items()}
    return {"nodes": nodes, "runs": n, "foreign": bool(foreign)}


def _median(vals: list) -> float:
    v = sorted(vals)
    k = len(v)
    return float(v[k // 2] if k % 2 else (v[k // 2 - 1] + v[k // 2]) / 2)


# ── overall progress (live) ─────────────────────────────────────────────────────────

def overall(profile: dict, run: dict, plan: Iterable[str], now: float,
            step: Optional[tuple] = None) -> Optional[dict]:
    """Whole-workflow fraction + ETA for a running job, or None when the profile
    covers none of the nodes that will execute.

    `plan` = node ids of the workflow as submitted. `step` = (node, value, max,
    node_eta_s | None) from the sampler counter, when one is counting."""
    pnodes = (profile or {}).get("nodes") or {}
    cached = run.get("cached") or set()
    done = run.get("done") or {}
    rsteps = run.get("steps") or {}
    cur = run.get("cur")
    cur_s = max(0.0, now - float(run["cur_at"])) if cur and run.get("cur_at") is not None else 0.0

    def single(nid):          # one counting phase, in the profile AND in this run
        return (pnodes[nid].get("phases") or 1) <= 1 and _phases(run, nid) <= 1

    step_frac = step_eta = None
    if step and cur in pnodes and _owner(run, step[0]) == cur and single(cur) and step[2] > 0:
        step_frac = min(max(step[1] / step[2], 0.0), 1.0)
        step_eta = step[3]
    exp = {}
    for nid in plan:
        nid = str(nid)
        if nid in cached or nid not in pnodes:
            continue
        e = float(pnodes[nid]["s"])
        # this run's step count — live for the running node, recorded for a finished
        # one: a finished 40-step sampler scaled back to its 20-step profile would
        # read as "twice as slow" and double the whole ETA
        st = step[2] if (step_frac is not None and nid == cur) else rsteps.get(nid)
        if st and pnodes[nid].get("steps") and single(nid):
            e *= st / pnodes[nid]["steps"]
        exp[nid] = e
    total = sum(exp.values())
    if total <= 0:
        return None

    def finished(n):
        # A LAZY node runs a near-zero first visit (it asks for its inputs, which run
        # next, then ComfyUI sends `executing` for it again): counting that visit as
        # "done" puts its whole share on the bar early and drives the pace to 0.33.
        return n in exp and n != cur and done[n] >= min(LAZY_VISIT_S, LAZY_VISIT_SHARE * exp[n])

    # this run's pace: actual vs expected seconds of the nodes already finished
    fin = [n for n in done if finished(n)]
    se = sum(exp[n] for n in fin)
    sa = sum(done[n] for n in fin)
    if se >= SPEED_MIN_EVIDENCE_S or (se > 0 and se >= SPEED_MIN_EVIDENCE_SHARE * total):
        k = min(max(sa / se, SPEED_CLAMP[0]), SPEED_CLAMP[1])
    else:
        k = None if profile.get("foreign") else 1.0

    part = se
    cur_rem = None
    if cur in exp:
        e_cur = exp[cur]
        if step_frac is not None:
            f = step_frac
            cur_rem = step_eta
        else:
            f = min(cur_s / (e_cur * (k or 1.0)), TIME_FRACTION_CAP) if e_cur > 0 else 0.0
        part += e_cur * f
        if cur_rem is None and k is not None:
            cur_rem = max(0.0, e_cur * k - cur_s)
    fset = set(fin)
    rest = sum(e for n, e in exp.items() if n not in fset and n != cur)
    out = {"fraction": round(min(part / total, FRACTION_CAP), 3),
           "expected_s": round(total, 1)}
    if k is not None:
        out["eta_s"] = max(0, int(rest * k + (cur_rem or 0.0)))
    return out
