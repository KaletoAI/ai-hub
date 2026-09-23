"""Call stats: SQLite-backed per-request log (+ on-disk request/response bodies).

Data only — every view of it is rendered by the /ui console (admin.py: Statistic,
Jobs & Calls, Dashboard). Zero dependencies beyond the stdlib; keep it that way.
"""

from __future__ import annotations

import asyncio
import gzip
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
# A stored body keeps at most this many characters per side (request / response): a
# Claude Code turn re-sends its whole context, so uncapped blobs cost ~1 MB per CALL.
# Beyond the cap the head and the tail are kept — where the system prompt and the
# newest turn sit. `stats.body_max_kb` in config.
_BODY_MAX_CHARS: int = 256 * 1024
BODY_RETENTION_DAYS_DEFAULT = 14     # bodies are pruned after this many days; the ROW stays
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


def init(db_path: str, blob_dir: str = "calls", body_max_kb: Optional[int] = None) -> None:
    """Open / create the stats DB, set WAL, ensure schema. Full call bodies
    (request + response) live on disk under `blob_dir` (never in the DB), keyed
    by call id, gzip-compressed and capped at `body_max_kb` per side; they are
    pruned with their row, or earlier by `body_retention_days` (see prune_once)."""
    global _DB_PATH, _BLOB_DIR, _BODY_MAX_CHARS
    _DB_PATH = Path(db_path)
    _BLOB_DIR = blob_dir
    if body_max_kb:
        try:
            _BODY_MAX_CHARS = max(1024, int(body_max_kb) * 1024)
        except (TypeError, ValueError):
            logger.warning(f"stats: ignoring invalid body_max_kb {body_max_kb!r}")
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
    # Per connection (not persisted). In WAL mode NORMAL only gives up durability of
    # the LAST commits on a power cut — never consistency — and saves an fsync per
    # recorded call; a call-log row is worth less than that fsync on the request path.
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
    finally:
        conn.close()


def _body_path(call_id: int) -> str:
    """Where a call's body is WRITTEN (gzip). Reads also accept the legacy plain
    `<id>.json` of blobs stored before compression (_body_paths)."""
    return os.path.join(_BLOB_DIR, f"{call_id}.json.gz")


def _body_paths(call_id: int) -> tuple:
    return (_body_path(call_id), os.path.join(_BLOB_DIR, f"{call_id}.json"))


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


def _capped(text):
    """A body side for storage: parsed JSON when it fits the cap, else head + tail
    of the raw text with a marker saying how much was dropped."""
    if text is None or len(text) <= _BODY_MAX_CHARS:
        return _as_obj(text)
    keep = _BODY_MAX_CHARS // 2
    return {"_truncated": f"{len(text)} chars — only the first and last {keep} are stored "
                          f"(stats.body_max_kb)",
            "head": text[:keep], "tail": text[-keep:]}


def get_body(call_id: int) -> Optional[dict]:
    """Full {request, response} for a call from its on-disk blob (or None). Reads the
    gzip blob, or the plain JSON one written before bodies were compressed."""
    gz, plain = _body_paths(int(call_id))
    try:
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, EOFError):
        return None
    try:
        with open(plain, "r", encoding="utf-8") as f:
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


_REQUEST_NOT_STORED = "(not stored — the call was refused before any backend saw it)"


def _write_body(call_id: int, request_text, response_text, response_audio,
                store_request: bool) -> None:
    payload = {"request": _capped(request_text) if store_request else _REQUEST_NOT_STORED,
               "response": _capped(response_text)}
    if response_audio:                          # binary body (TTS WAV): own file + JSON marker
        data, mime = response_audio
        with open(_audio_path(call_id), "wb") as af:
            af.write(data)
        payload["response"] = {"_audio": mime or "audio/wav", "bytes": len(data)}
    with gzip.open(_body_path(call_id), "wt", encoding="utf-8", compresslevel=5) as f:
        json.dump(payload, f, ensure_ascii=False)


def _record_sync(row: tuple, request_text=None, response_text=None, response_audio=None,
                 store_request: bool = True) -> None:
    want_body = request_text is not None or response_text is not None or response_audio is not None
    with _conn() as c:
        # ONE autocommitted statement: has_body goes in with the row instead of a second
        # UPDATE (and commit) after the blob write. The rare failed blob write takes it
        # back; a reader racing the write for a few ms at worst sees "no stored body".
        cur = c.execute(
            "INSERT INTO calls (ts, duration_ms, backend, source, alias, model, "
            "endpoint, status, input_tokens, output_tokens, cost_usd, req_preview, reasoning, "
            "cache_read, cache_write, has_body) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (*row, 1 if want_body else 0),
        )
        if want_body:
            try:
                _write_body(cur.lastrowid, request_text, response_text, response_audio, store_request)
            except Exception as e:                  # row stays valid, body view just absent
                logger.warning(f"stats: body blob write failed for call {cur.lastrowid}: {e}")
                c.execute("UPDATE calls SET has_body=0 WHERE id=?", (cur.lastrowid,))


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
    store_request: bool = True,
) -> None:
    """Async-safe insert. Never raises into the request path. Full request/response
    bodies (when given) are written to an on-disk blob, not the DB row;
    `response_audio` = (bytes, mime) for binary TTS replies (own blob + player).
    `store_request=False` keeps the request out of the blob (the preview column still
    describes it) — for refused calls, whose body would be the one nobody reads,
    repeated as often as a misconfigured client retries.

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
        await asyncio.to_thread(_record_sync, row, request_text, response_text, response_audio,
                                store_request)
    except Exception as e:
        logger.warning(f"stats: insert failed: {e}")


def _drop_blobs(call_id: int) -> None:
    for p in (*_body_paths(call_id), _audio_path(call_id)):
        try:
            os.remove(p)
        except OSError:
            pass


def prune_once(retention_days: float = 0, body_retention_days: float = BODY_RETENTION_DAYS_DEFAULT) -> tuple:
    """Delete rows older than `retention_days` (0 = keep forever) with their blobs, and
    the BODY blobs of rows older than `body_retention_days` (0 = keep forever) — the row
    itself stays for the aggregates and the monthly cost quota. (rows, bodies) removed."""
    now = int(time.time())
    rows = bodies = 0
    with _conn() as c:
        if body_retention_days and float(body_retention_days) > 0:
            cutoff = now - int(float(body_retention_days) * 86400)
            ids = [r[0] for r in c.execute(
                "SELECT id FROM calls WHERE ts < ? AND has_body = 1", (cutoff,)).fetchall()]
            for cid in ids:
                _drop_blobs(cid)
            c.execute("UPDATE calls SET has_body = 0 WHERE ts < ? AND has_body = 1", (cutoff,))
            bodies = len(ids)
        if retention_days and float(retention_days) > 0:
            cutoff = now - int(float(retention_days) * 86400)
            ids = [r[0] for r in c.execute("SELECT id FROM calls WHERE ts < ?", (cutoff,)).fetchall()]
            for cid in ids:
                _drop_blobs(cid)
            c.execute("DELETE FROM calls WHERE ts < ?", (cutoff,))
            rows = len(ids)
    return rows, bodies


async def prune_loop(retention_days: float, body_retention_days: float = BODY_RETENTION_DAYS_DEFAULT,
                     interval_s: int = 3600) -> None:
    """Hourly prune_once(). Returns at once when both retentions are 0 (keep forever).
    A blank/None value means the default (rows: forever, bodies: 14 days) — the Server
    tab stores an emptied field as ""."""
    def days(v, default):
        if v is None or (isinstance(v, str) and not v.strip()):
            return float(default)
        return float(v)
    try:
        retention_days = days(retention_days, 0)
        body_retention_days = days(body_retention_days, BODY_RETENTION_DAYS_DEFAULT)
    except (TypeError, ValueError):
        logger.warning("stats: invalid retention setting — pruning disabled")
        return
    if retention_days <= 0 and body_retention_days <= 0:
        return
    while True:
        try:
            rows, bodies = await asyncio.to_thread(prune_once, retention_days, body_retention_days)
            if rows or bodies:
                logger.info(f"stats: pruned {rows} rows older than {retention_days:g} days, "
                            f"{bodies} bodies older than {body_retention_days:g} days")
        except Exception as e:
            logger.warning(f"stats: prune failed: {e}")
        await asyncio.sleep(interval_s)


# ── Query helper ─────────────────────────────────────────────────────────────

def _q(sql: str, *params) -> list[tuple]:
    with _conn() as c:
        return c.execute(sql, params).fetchall()
