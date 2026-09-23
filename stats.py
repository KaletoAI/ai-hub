"""Call stats: SQLite-backed per-request log (+ on-disk request/response bodies).

Data only — every view of it is rendered by the /ui console (admin.py: Statistic,
Jobs & Calls, Dashboard). Zero dependencies beyond the stdlib; keep it that way.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH: Optional[Path] = None
_BLOB_DIR: str = "calls"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            INTEGER NOT NULL,
    duration_ms   INTEGER NOT NULL,
    backend       TEXT    NOT NULL,
    source        TEXT,
    alias         TEXT,
    model         TEXT,
    endpoint      TEXT,
    status        INTEGER,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd      REAL    DEFAULT 0,
    req_preview   TEXT,
    reasoning     TEXT,
    -- Prompt-cache breakdown of input_tokens (which stays the TOTAL the model
    -- processed): cache_read = served from cache, cache_write = written into it.
    -- input_tokens - cache_read - cache_write = paid at the full fresh rate.
    cache_read    INTEGER DEFAULT 0,
    cache_write   INTEGER DEFAULT 0
);
-- (ts, backend) serves every time-window read AND the Dashboard's per-backend count
-- as a covering index; it replaced a plain idx_calls_ts (a prefix of it, dropped in init).
CREATE INDEX IF NOT EXISTS idx_calls_ts_backend ON calls(ts, backend);
CREATE INDEX IF NOT EXISTS idx_calls_backend ON calls(backend);
CREATE INDEX IF NOT EXISTS idx_calls_source  ON calls(source, ts);  -- month_cost quota scan
"""

# Backend column of a call NO backend ever saw (park timeout, quota, unknown alias,
# bad key — written by main._record_rejected). A marker, not a backend: it is excluded
# from every per-backend / per-model aggregate below, because a pseudo-backend with 0
# tokens, 0 cost and 0 ms in the "By backend" table is not a fact about any backend,
# and its empty model splits the alias's row in two. summary() reports it as its own
# figure instead, so the refusals stay visible where they mean something.
REFUSED_BACKEND = "(refused)"


def is_active() -> bool:
    return _DB_PATH is not None


# The column shape of every per-call list (the console's _call_row template).
_CALL_COLS = ("id, ts, duration_ms, backend, source, alias, model, endpoint, status, "
              "input_tokens, output_tokens, cost_usd, req_preview, has_body, reasoning")

# Both Dashboard-tick queries order/group so that SQLite reads the ts WINDOW through
# idx_calls_ts_backend. `ORDER BY id DESC` walked the whole table backwards, and a bare
# `GROUP BY backend` made the planner prefer idx_calls_backend (ordered for the grouping)
# over the ts range — a full index scan per tick (274 ms at 300k rows, measured). The
# unary `+` takes the column out of index consideration for the grouping only.
# test_stats_store.py pins both plans.
_SQL_RECENT_SINCE = (f"SELECT {_CALL_COLS} FROM calls WHERE ts > ? "
                     f"ORDER BY ts DESC, id DESC LIMIT ?")
_SQL_COUNT_BY_BACKEND_SINCE = "SELECT backend, COUNT(*) FROM calls WHERE ts > ? GROUP BY +backend"


def recent_since(ts: int, limit: int = 100) -> list:
    """Calls completed since `ts` (unix secs), newest first — the dashboard's
    'last N minutes' window. Same column shape as summary()['recent'] so it
    renders with the identical row template."""
    if _DB_PATH is None:
        return []
    return _q(_SQL_RECENT_SINCE, ts, limit)


def count_since(ts: int) -> int:
    if _DB_PATH is None:
        return 0
    return _q("SELECT COUNT(*) FROM calls WHERE ts > ?", ts)[0][0]


def count_by_backend_since(ts: int) -> dict:
    """Calls per backend (display name) completed since `ts` — the dashboard's
    per-backend request rate. Empty if stats are off."""
    if _DB_PATH is None:
        return {}
    return {r[0]: r[1] for r in
            _q(_SQL_COUNT_BY_BACKEND_SINCE, ts)}


def summary(recent_limit: int = 50, model_limit: int = 30, source_limit: int = 20,
            user: Optional[str] = None) -> dict:
    """Aggregated call stats for the in-UI dashboard (data only, no HTML). If
    `user` is given, every figure is scoped to that source (per-user drilldown);
    `by_source` stays unscoped so it can drive the user picker.

    Refused calls (see REFUSED_BACKEND) are traffic, so they count in the totals
    and in by_source — but they are kept out of by_backend/by_model, which are
    about work a backend actually did, and reported separately as `refused_*`."""
    if _DB_PATH is None:
        return {"active": False}
    now = int(time.time())
    flt = " WHERE source = ?" if user else ""
    ua = [user] if user else []
    served = (flt + " AND" if flt else " WHERE") + " backend <> ?"   # forwarded calls only
    sa = [*ua, REFUSED_BACKEND]
    total = _q(f"SELECT COUNT(*), COALESCE(SUM(cost_usd),0) FROM calls{flt}", *ua)[0]
    h24flt = " WHERE ts > ?" + (" AND source = ?" if user else "")
    h24 = _q(f"SELECT COUNT(*), COALESCE(SUM(cost_usd),0) FROM calls{h24flt}", now - 86400, *ua)[0]
    # Params bind by position across the WHOLE statement, so the SELECT's ts window
    # comes before the WHERE's backend/source.
    ref = _q(f"SELECT COUNT(*), COALESCE(SUM(CASE WHEN ts > ? THEN 1 ELSE 0 END),0) "
             f"FROM calls WHERE backend = ?" + (" AND source = ?" if user else ""),
             now - 86400, REFUSED_BACKEND, *ua)[0]
    return {
        "active": True, "user": user,
        "total_count": total[0], "total_cost": total[1],
        "h24_count": h24[0], "h24_cost": h24[1],
        "refused_count": ref[0], "refused_24h": ref[1],
        # by_backend carries the prompt-cache breakdown too: cache_read/cache_write
        # are subsets of input_tokens, so "fresh" is the remainder.
        "by_backend": _q(
            f"SELECT backend, COUNT(*), COALESCE(SUM(input_tokens),0), "
            f"COALESCE(SUM(output_tokens),0), COALESCE(SUM(cost_usd),0), COALESCE(AVG(duration_ms),0), "
            f"COALESCE(SUM(cache_read),0), COALESCE(SUM(cache_write),0) "
            f"FROM calls{served} GROUP BY backend ORDER BY COUNT(*) DESC", *sa),
        "by_model": _q(
            f"SELECT COALESCE(alias,''), COALESCE(model,''), COUNT(*), "
            f"COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(SUM(cost_usd),0) "
            f"FROM calls{served} GROUP BY alias, model ORDER BY COUNT(*) DESC LIMIT ?", *sa, model_limit),
        "by_source": _q(
            "SELECT COALESCE(source,'unknown'), COUNT(*), COALESCE(SUM(cost_usd),0) "
            "FROM calls GROUP BY source ORDER BY COUNT(*) DESC LIMIT ?", source_limit),
        "recent": _q(
            f"SELECT id, ts, duration_ms, backend, source, alias, model, endpoint, status, "
            f"input_tokens, output_tokens, cost_usd, req_preview, has_body, reasoning "
            f"FROM calls{flt} ORDER BY id DESC LIMIT ?", *ua, recent_limit),
    }


def cache_trend(hours: int = 24, buckets: int = 24, user: Optional[str] = None) -> dict:
    """Prompt-cache history per backend: `buckets` equal slices over the last
    `hours`, each carrying (input, cache_read, cache_write).

    A single total says whether caching works at all; the trend says whether it
    still works — a cache that stopped being hit (a changed prefix, an expired
    window) shows up here as a hit rate falling to zero while input keeps rising,
    which is exactly the moment a long session starts costing full price again.
    Only backends that reported cache numbers at all are included."""
    if _DB_PATH is None or hours <= 0 or buckets <= 0:
        return {}
    now = int(time.time())
    start = now - hours * 3600
    span = max(1, (hours * 3600) // buckets)
    flt = " AND source = ?" if user else ""
    ua = [user] if user else []
    rows = _q(
        f"SELECT backend, (ts - ?) / ? AS bucket, COALESCE(SUM(input_tokens),0), "
        f"COALESCE(SUM(cache_read),0), COALESCE(SUM(cache_write),0) "
        f"FROM calls WHERE ts > ?{flt} GROUP BY backend, bucket ORDER BY bucket",
        start, span, start, *ua)
    out: dict = {}
    for backend, bucket, in_tok, read, write in rows:
        series = out.setdefault(backend, [(0, 0, 0)] * buckets)
        idx = min(buckets - 1, max(0, int(bucket)))
        i, r, w = series[idx]
        series[idx] = (i + int(in_tok), r + int(read), w + int(write))
    return {b: s for b, s in out.items() if any(r or w for _, r, w in s)}


def month_cost(user: str, month_start_ts: int) -> float:
    """Total cost_usd for a user since a UTC timestamp — drives the monthly
    cost quota (E1). 0 when stats are off (quota simply can't bind then)."""
    if _DB_PATH is None:
        return 0.0
    r = _q("SELECT COALESCE(SUM(cost_usd),0) FROM calls WHERE source = ? AND ts >= ?",
           user, month_start_ts)
    return float(r[0][0]) if r else 0.0


def init(db_path: str, blob_dir: str = "calls") -> None:
    """Open / create the stats DB, set WAL, ensure schema. Full call bodies
    (request + response) live on disk under `blob_dir` (never in the DB), keyed
    by call id, and are pruned together with their row."""
    global _DB_PATH, _BLOB_DIR
    _DB_PATH = Path(db_path)
    _BLOB_DIR = blob_dir
    os.makedirs(_BLOB_DIR, exist_ok=True)
    with _conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(_SCHEMA)
        c.execute("DROP INDEX IF EXISTS idx_calls_ts")    # a prefix of idx_calls_ts_backend
        # Migrate older DBs that predate later columns.
        cols = {r[1] for r in c.execute("PRAGMA table_info(calls)").fetchall()}
        for col, ddl in (("req_preview", "TEXT"),
                         ("has_body", "INTEGER DEFAULT 0"),
                         ("reasoning", "TEXT"),
                         ("cache_read", "INTEGER DEFAULT 0"),
                         ("cache_write", "INTEGER DEFAULT 0")):
            if col not in cols:
                c.execute(f"ALTER TABLE calls ADD COLUMN {col} {ddl}")
    logger.info(f"stats: SQLite at {_DB_PATH} (WAL), call bodies in {_BLOB_DIR}/")


@contextmanager
def _conn():
    conn = sqlite3.connect(_DB_PATH, isolation_level=None, timeout=10.0)
    try:
        yield conn
    finally:
        conn.close()


def _body_path(call_id: int) -> str:
    return os.path.join(_BLOB_DIR, f"{call_id}.json")


def _audio_path(call_id: int) -> str:
    return os.path.join(_BLOB_DIR, f"{call_id}.audio")


def _as_obj(text):
    """Parse a JSON string back to an object for tidy storage; keep raw if not JSON."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def get_body(call_id: int) -> Optional[dict]:
    """Full {request, response} for a call from its on-disk blob (or None)."""
    try:
        with open(_body_path(int(call_id)), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def call_neighbors(call_id: int, voice: bool) -> tuple:
    """(newer_id, older_id) around a call within its list partition — Voice Calls
    (endpoint /v1/audio/*) vs LLM Calls (the rest) — for the detail page's
    prev/next navigation. None at the list ends."""
    if _DB_PATH is None:
        return None, None
    flt = ("endpoint LIKE '/v1/audio/%'" if voice
           else "(endpoint IS NULL OR endpoint NOT LIKE '/v1/audio/%')")
    newer = _q(f"SELECT id FROM calls WHERE id > ? AND {flt} ORDER BY id ASC LIMIT 1", int(call_id))
    older = _q(f"SELECT id FROM calls WHERE id < ? AND {flt} ORDER BY id DESC LIMIT 1", int(call_id))
    return (newer[0][0] if newer else None, older[0][0] if older else None)


def get_audio(call_id: int) -> Optional[tuple]:
    """(path, mime) of a call's stored binary audio response, or None. The mime
    lives in the JSON blob's response marker ({'_audio': mime, 'bytes': n})."""
    p = _audio_path(int(call_id))
    if not os.path.isfile(p):
        return None
    body = get_body(call_id) or {}
    resp = body.get("response")
    mime = resp.get("_audio") if isinstance(resp, dict) else None
    return p, (mime or "audio/wav")


def _record_sync(row: tuple, request_text=None, response_text=None, response_audio=None) -> None:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO calls (ts, duration_ms, backend, source, alias, model, "
            "endpoint, status, input_tokens, output_tokens, cost_usd, req_preview, reasoning, "
            "cache_read, cache_write) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )
        if request_text is not None or response_text is not None or response_audio is not None:
            try:
                payload = {"request": _as_obj(request_text), "response": _as_obj(response_text)}
                if response_audio:                  # binary body (TTS WAV): own file + JSON marker
                    data, mime = response_audio
                    with open(_audio_path(cur.lastrowid), "wb") as af:
                        af.write(data)
                    payload["response"] = {"_audio": mime or "audio/wav", "bytes": len(data)}
                with open(_body_path(cur.lastrowid), "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                c.execute("UPDATE calls SET has_body=1 WHERE id=?", (cur.lastrowid,))
            except Exception as e:                  # row stays valid, body view just absent
                logger.warning(f"stats: body blob write failed for call {cur.lastrowid}: {e}")


def _preview(text: Optional[str], head: int = 50, tail: int = 50) -> Optional[str]:
    """First `head` + last `tail` chars of the request, ellipsis in between."""
    if not text:
        return None
    text = " ".join(text.split())  # collapse whitespace/newlines for a compact preview
    if len(text) <= head + tail:
        return text
    return f"{text[:head]} … {text[-tail:]}"


async def record_call(
    *,
    duration_ms: int,
    backend: str,
    source: Optional[str],
    alias: Optional[str],
    model: Optional[str],
    endpoint: Optional[str],
    status: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    request_text: Optional[str] = None,
    response_text: Optional[str] = None,
    response_audio: Optional[tuple] = None,
    reasoning: Optional[str] = None,
    cache_read: int = 0,
    cache_write: int = 0,
) -> None:
    """Async-safe insert. Never raises into the request path. Full request/response
    bodies (when given) are written to an on-disk blob, not the DB row;
    `response_audio` = (bytes, mime) for binary TTS replies (own blob + player).

    `cache_read`/`cache_write` break `input_tokens` down: how much of it the
    backend served from its prompt cache and how much it wrote into the cache.
    Both are SUBSETS of input_tokens, never additions — the rest was processed
    fresh at full price."""
    if _DB_PATH is None:
        return
    row = (
        int(time.time()),
        duration_ms,
        backend,
        source,
        alias,
        model,
        endpoint,
        status,
        input_tokens,
        output_tokens,
        cost_usd,
        _preview(request_text),
        reasoning,
        max(0, int(cache_read or 0)),
        max(0, int(cache_write or 0)),
    )
    try:
        await asyncio.to_thread(_record_sync, row, request_text, response_text, response_audio)
    except Exception as e:
        logger.warning(f"stats: insert failed: {e}")


async def prune_loop(retention_days: int, interval_s: int = 3600) -> None:
    """Periodically delete rows older than retention_days. 0 = disabled."""
    if retention_days <= 0:
        return
    while True:
        try:
            cutoff = int(time.time()) - retention_days * 86400
            def _prune():
                with _conn() as c:
                    ids = [r[0] for r in c.execute("SELECT id FROM calls WHERE ts < ?", (cutoff,)).fetchall()]
                    for cid in ids:
                        for p in (_body_path(cid), _audio_path(cid)):
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                    c.execute("DELETE FROM calls WHERE ts < ?", (cutoff,))
                    return len(ids)
            n = await asyncio.to_thread(_prune)
            if n:
                logger.info(f"stats: pruned {n} rows older than {retention_days} days")
        except Exception as e:
            logger.warning(f"stats: prune failed: {e}")
        await asyncio.sleep(interval_s)


# ── Query helper ─────────────────────────────────────────────────────────────

def _q(sql: str, *params) -> list[tuple]:
    with _conn() as c:
        return c.execute(sql, params).fetchall()
