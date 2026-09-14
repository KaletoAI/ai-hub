"""Backend fault log — every time a backend FAILED, kept long enough to be seen.

Why it exists: the gateway already knows the moment a backend fails, and it said so
in exactly one place — the journal. `main.backend_error` is the CURRENT state only
(popped the instant the next poll succeeds), a chat call that failed over to another
backend books as a 200 in the call log, and a media job that crashed on one ComfyUI
and was retried or moved on ends as a clean `done` row. So a box that fell over five
times in twenty minutes (measured 2026-09-13, comfyui-strix on the Evo-X2: 22:03 –
22:22, every job "ComfyUI unreachable … likely crashed/restarting") looked perfectly
healthy in the console by the time anyone opened it.

Pure and dependency-free (stdlib only, no `main`/`adapters` imports) like `jobs.py`
and `stats.py`: `main` records, `admin` renders what `main.faults_info()` derives.

An event is one row: `ts, bid, backend, type, host, source, kind, status, detail,
dur_s`. `source` says which path noticed (`health` = a discovery poll, `call` = a
chat dispatch, `job` = a media generation, `watchdog` = a ComfyUI restart), `kind`
says what it was (`unreachable`, `timeout`, `upstream`, `stuck`, `execution`, …).
A `health` event with kind `recovered` is not a fault — it closes an outage and
carries its length in `dur_s`, which is what downtime is summed from.

Always on, independent of `stats.enabled`: every event goes into a bounded in-memory
ring, and — once `init()` ran — into SQLite as well, so a gateway restart (which is
often what follows a bad evening) does not wipe the evidence. Reads use the DB when
it is active, else the ring.
"""
import asyncio
import logging
import re
import sqlite3
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

RECOVERED = "recovered"          # kind of the event that CLOSES an outage (not a fault)
WINDOW_S = 86400                 # what the console shows: the last 24h
_MEM_MAX = 5000
_DETAIL_MAX = 500

_DB_PATH: Optional[str] = None
_MEM: deque = deque(maxlen=_MEM_MAX)
_lock = threading.Lock()

_COLS = ("ts", "bid", "backend", "type", "host", "source", "kind", "status", "detail", "dur_s")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS faults (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      INTEGER NOT NULL,
    bid     TEXT    NOT NULL,
    backend TEXT    NOT NULL,
    type    TEXT,
    host    TEXT,
    source  TEXT    NOT NULL,
    kind    TEXT    NOT NULL,
    status  INTEGER,
    detail  TEXT,
    dur_s   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_faults_ts ON faults(ts);
"""


def init(db_path: str) -> None:
    """Open (create) the fault DB. A failure leaves the log memory-only — the fault
    log must never be the reason the gateway does not boot."""
    global _DB_PATH
    try:
        with sqlite3.connect(db_path, timeout=10) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(_SCHEMA)
        _DB_PATH = db_path
    except Exception as e:
        _DB_PATH = None
        logger.warning(f"faults: could not open {db_path} — keeping faults in memory only: {e}")


def is_persistent() -> bool:
    return _DB_PATH is not None


def _insert(ev: dict) -> None:
    if _DB_PATH is None:
        return
    try:
        with sqlite3.connect(_DB_PATH, timeout=10) as c:
            c.execute(f"INSERT INTO faults ({', '.join(_COLS)}) VALUES ({', '.join('?' * len(_COLS))})",
                      tuple(ev[k] for k in _COLS))
    except Exception as e:
        logger.warning(f"faults: insert failed: {e}")


def record(*, bid: str, backend: str, source: str, kind: str, detail: str = "",
           type: str = "openai", host: str = "", status: Optional[int] = None,
           dur_s: Optional[int] = None, ts: Optional[int] = None) -> dict:
    """Record one event. Never raises; on an event loop the DB write runs off-loop."""
    ev = {"ts": int(ts if ts is not None else time.time()), "bid": str(bid),
          "backend": str(backend), "type": str(type or "openai"), "host": str(host or ""),
          "source": str(source), "kind": str(kind or "error"),
          "status": int(status) if isinstance(status, int) else None,
          "detail": " ".join(str(detail or "").split())[:_DETAIL_MAX],
          "dur_s": int(dur_s) if dur_s is not None else None}
    with _lock:
        _MEM.append(ev)
    if _DB_PATH is not None:
        try:
            asyncio.get_running_loop().run_in_executor(None, _insert, ev)
        except RuntimeError:                  # no running loop (tests, sync callers)
            _insert(ev)
    return ev


def events_since(since: int, limit: int = 20000) -> list:
    """Events with ts > since, oldest first."""
    if _DB_PATH is not None:
        try:
            with sqlite3.connect(_DB_PATH, timeout=10) as c:
                rows = c.execute(f"SELECT {', '.join(_COLS)} FROM faults WHERE ts > ? "
                                 f"ORDER BY ts, id LIMIT ?", (int(since), int(limit))).fetchall()
            return [dict(zip(_COLS, r)) for r in rows]
        except Exception as e:
            logger.warning(f"faults: read failed, falling back to memory: {e}")
    with _lock:
        return [dict(ev) for ev in _MEM if ev["ts"] > since][-limit:]


def prune(retention_s: int) -> int:
    if _DB_PATH is None or retention_s <= 0:
        return 0
    with sqlite3.connect(_DB_PATH, timeout=10) as c:
        return c.execute("DELETE FROM faults WHERE ts < ?", (int(time.time()) - retention_s,)).rowcount


async def prune_loop(retention_days: float, interval_s: int = 3600) -> None:
    while True:
        try:
            await asyncio.to_thread(prune, int(float(retention_days) * 86400))
        except Exception as e:
            logger.warning(f"faults: prune failed: {e}")
        await asyncio.sleep(interval_s)


# ── derivation (pure) ─────────────────────────────────────────────────────────

_HEX = re.compile(r"\b[0-9a-fA-F]{8,}\b")
_NUM = re.compile(r"\d+(?:\.\d+)*")


def bundle_key(detail: str) -> str:
    """The message with what varies per occurrence taken out — job ids, byte counts,
    ports, node numbers — so "unreachable for >15s" at 22:03 and at 22:19 bundle
    into ONE line with a count instead of twenty lines nobody reads."""
    s = _HEX.sub("…", detail or "")
    s = _NUM.sub("#", s)
    return " ".join(s.split()).lower()


def bundles(events: list) -> list:
    """Faults (never `recovered`) grouped by backend + source + kind + status + the
    normalized message. `message` is the LATEST verbatim text of the group — a real
    example beats the normalized key. Newest group first."""
    groups: dict = {}
    for ev in events:
        if ev["kind"] == RECOVERED:
            continue
        key = (ev["bid"], ev["source"], ev["kind"], ev.get("status"), bundle_key(ev.get("detail") or ""))
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"bid": ev["bid"], "backend": ev["backend"], "type": ev.get("type"),
                               "host": ev.get("host") or "", "source": ev["source"],
                               "kind": ev["kind"], "status": ev.get("status"),
                               "message": ev.get("detail") or "", "count": 0,
                               "first": ev["ts"], "last": ev["ts"]}
        g["count"] += 1
        g["first"] = min(g["first"], ev["ts"])
        if ev["ts"] >= g["last"]:
            g["last"] = ev["ts"]
            g["message"] = ev.get("detail") or g["message"]
    return sorted(groups.values(), key=lambda g: (-g["last"], g["backend"]))


def per_backend(events: list, since: int, now: int, down_since: Optional[dict] = None) -> dict:
    """bid → {backend, type, host, faults, outages, downtime_s, last_ts, last_kind,
    last_detail, last_source}.

    Downtime is clipped to the window: an outage that began before `since` counts from
    `since`. `down_since` (bid → ts) are outages STILL OPEN — they have no `recovered`
    event yet and count up to `now`; a backend in it appears even with no fault event
    inside the window (it went down earlier and never came back)."""
    out: dict = {}

    def slot(ev_or_bid, backend="", typ=None, host=""):
        bid = ev_or_bid
        if bid not in out:
            out[bid] = {"bid": bid, "backend": backend, "type": typ, "host": host, "faults": 0,
                        "outages": 0, "downtime_s": 0, "last_ts": None, "last_kind": None,
                        "last_detail": "", "last_source": None}
        return out[bid]

    for ev in events:
        s = slot(ev["bid"], ev["backend"], ev.get("type"), ev.get("host") or "")
        if ev["kind"] == RECOVERED:
            dur = int(ev.get("dur_s") or 0)
            start = max(since, ev["ts"] - dur)
            s["downtime_s"] += max(0, ev["ts"] - start)
            continue
        s["faults"] += 1
        if ev["source"] == "health":
            s["outages"] += 1
        if s["last_ts"] is None or ev["ts"] >= s["last_ts"]:
            s.update(last_ts=ev["ts"], last_kind=ev["kind"], last_detail=ev.get("detail") or "",
                     last_source=ev["source"])
    for bid, t0 in (down_since or {}).items():
        s = slot(bid)
        s["downtime_s"] += max(0, now - max(since, int(t0 or now)))
    return out
