"""Call stats: SQLite-backed per-request log + minimal HTML dashboard.

Zero new dependencies — sqlite3 is stdlib, FastAPI/uvicorn are already
required by the gateway. The dashboard is plain HTML rendered from
f-strings; no template engine, no JS, no chart libs. Auto-refresh via
<meta http-equiv="refresh">.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse

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
CREATE INDEX IF NOT EXISTS idx_calls_ts      ON calls(ts);
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


def recent(limit: int = 15) -> list:
    """Just the latest calls (one indexed query) — for the live dashboard."""
    if _DB_PATH is None:
        return []
    return _q("SELECT ts, duration_ms, backend, source, alias, model, endpoint, status, "
              "input_tokens, output_tokens, cost_usd FROM calls ORDER BY id DESC LIMIT ?", limit)


def recent_since(ts: int, limit: int = 100) -> list:
    """Calls completed since `ts` (unix secs), newest first — the dashboard's
    'last N minutes' window. Same column shape as summary()['recent'] so it
    renders with the identical row template."""
    if _DB_PATH is None:
        return []
    return _q("SELECT id, ts, duration_ms, backend, source, alias, model, endpoint, status, "
              "input_tokens, output_tokens, cost_usd, req_preview, has_body, reasoning "
              "FROM calls WHERE ts > ? ORDER BY id DESC LIMIT ?", ts, limit)


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
            _q("SELECT backend, COUNT(*) FROM calls WHERE ts > ? GROUP BY backend", ts)}


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


# ── Dashboard ────────────────────────────────────────────────────────────────

stats_app = FastAPI(title="AI-Hub Stats", docs_url=None, redoc_url=None)


def _q(sql: str, *params) -> list[tuple]:
    with _conn() as c:
        return c.execute(sql, params).fetchall()


@stats_app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"


@stats_app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    if _DB_PATH is None:
        return "<h1>Stats not enabled</h1>"

    total_count, total_cost = _q(
        "SELECT COUNT(*), COALESCE(SUM(cost_usd),0) FROM calls"
    )[0]
    h24 = int(time.time()) - 86400
    h24_count, h24_cost = _q(
        "SELECT COUNT(*), COALESCE(SUM(cost_usd),0) FROM calls WHERE ts > ?", h24
    )[0]
    by_backend = _q(
        "SELECT backend, COUNT(*), COALESCE(SUM(input_tokens),0), "
        "COALESCE(SUM(output_tokens),0), COALESCE(SUM(cost_usd),0), "
        "COALESCE(AVG(duration_ms),0) "
        "FROM calls GROUP BY backend ORDER BY COUNT(*) DESC"
    )
    by_model = _q(
        "SELECT COALESCE(alias,''), COALESCE(model,''), COUNT(*), "
        "COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
        "COALESCE(SUM(cost_usd),0) "
        "FROM calls GROUP BY alias, model ORDER BY COUNT(*) DESC LIMIT 30"
    )
    by_source = _q(
        "SELECT COALESCE(source,'unknown'), COUNT(*), COALESCE(SUM(cost_usd),0) "
        "FROM calls GROUP BY source ORDER BY COUNT(*) DESC LIMIT 20"
    )
    recent = _q(
        "SELECT ts, duration_ms, backend, source, alias, model, endpoint, "
        "status, input_tokens, output_tokens, cost_usd, req_preview "
        "FROM calls ORDER BY id DESC LIMIT 50"
    )
    return _render(
        (total_count, total_cost),
        (h24_count, h24_cost),
        by_backend, by_model, by_source, recent,
    )


def _fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_time(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%H:%M:%S")


def _esc(s) -> str:
    if s is None:
        return ""
    return (str(s)
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


# Shared chrome: CSS + nav tabs + optional search JS, wrapped by _doc(). Kept as
# plain (non-f) strings so braces don't need escaping.
_CSS = """
* { box-sizing: border-box; }
body { font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 1.5rem; color: #1c1c1c; background: #f4f5f7; }
h1 { margin: 0 0 .5rem; }
h2 { margin: 0 0 .75rem; font-size: 1.05rem; color: #555; }
.muted { color: #888; font-size: .85rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
         gap: 1rem; margin: 1rem 0 2rem; }
.card { background: #fff; border-radius: 8px; padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.metric { font-size: 1.6rem; font-weight: 600; margin-top: .25rem; }
.panel { background: #fff; border-radius: 8px; padding: 1rem 1.25rem;
         margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
table { width: 100%; border-collapse: collapse; }
th, td { padding: .4rem .5rem; text-align: left;
         border-bottom: 1px solid #eee; vertical-align: top; }
th { background: #fafafa; font-weight: 600; color: #555; }
tr:hover td { background: #fcfcfc; }
td.ok { color: #1a7f37; font-weight: 600; }
td.err { color: #cf222e; font-weight: 600; }
code { background: #f1f1f1; padding: 1px 4px; border-radius: 3px; }
.tabs { display: flex; gap: .25rem; margin: .25rem 0 0; border-bottom: 2px solid #e0e2e8; }
.tabs a { padding: .45rem 1rem; border-radius: 6px 6px 0 0; text-decoration: none;
          color: #555; background: #e7e9ee; font-weight: 600; }
.tabs a.active { background: #fff; color: #1c1c1c; box-shadow: 0 -1px 3px rgba(0,0,0,.06); }
.search { margin: 1rem 0 0; }
.search input { width: 100%; max-width: 460px; padding: .55rem .75rem; font-size: 1rem;
                border: 1px solid #ccc; border-radius: 6px; }
.count { color: #888; font-size: .8rem; margin-left: .5rem; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: .75rem;
         font-weight: 600; }
.badge.ok   { background: #e6f4ea; color: #1a7f37; }
.badge.warn { background: #fff4e5; color: #b35900; }
.badge.err  { background: #fde8e8; color: #cf222e; }
.badge.info { background: #e8eefc; color: #1a56c4; }
.badge.off  { background: #eceef1; color: #888; }
.badge.busy { background: #f3e8fd; color: #8250df; }
tr.off td { color: #aaa; }
.dim { color: #aaa; }
.host { white-space: nowrap; }
.hidden { display: none; }
"""

# Vanilla filter for the routing tab. Matches each row's text plus its
# data-search attribute (carries the alias name onto continuation rows), keeps
# the query in the URL hash so it survives a reload.
_SEARCH_JS = """
<script>
(function () {
  var box = document.getElementById('q');
  if (!box) return;
  var rows = Array.prototype.slice.call(document.querySelectorAll('tr.f'));
  var counter = document.getElementById('count');
  function apply(q) {
    q = q.trim().toLowerCase();
    var shown = 0;
    rows.forEach(function (r) {
      var hay = (r.textContent + ' ' + (r.dataset.search || '')).toLowerCase();
      var match = !q || hay.indexOf(q) !== -1;
      r.classList.toggle('hidden', !match);
      if (match) shown++;
    });
    if (counter) counter.textContent = q ? (shown + ' match' + (shown === 1 ? '' : 'es')) : '';
    history.replaceState(null, '', q ? '#' + encodeURIComponent(q) : location.pathname);
  }
  box.addEventListener('input', function () { apply(box.value); });
  var initial = decodeURIComponent(location.hash.slice(1));
  if (initial) box.value = initial;
  apply(box.value);
  box.focus();
})();
</script>
"""


def _nav(active: str) -> str:
    def cls(name: str) -> str:
        return ' class="active"' if name == active else ""
    return (f'<div class="tabs"><a href="/"{cls("stats")}>Stats</a>'
            f'<a href="/routing"{cls("routing")}>Routing</a></div>')


def _doc(title: str, active: str, body: str, *, refresh: Optional[int] = None,
         search: bool = False) -> str:
    refresh_tag = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'{refresh_tag}\n<title>{title}</title>\n<style>{_CSS}</style>\n</head>\n'
            f'<body>\n<h1>ai-hub</h1>\n{_nav(active)}\n{body}\n'
            f'{_SEARCH_JS if search else ""}\n</body>\n</html>')


def _render(total, h24, by_backend, by_model, by_source, recent) -> str:
    rows_backend = "".join(
        f"<tr><td>{_esc(b)}</td><td>{c:,}</td><td>{int(it):,}</td>"
        f"<td>{int(ot):,}</td><td>${cu:.4f}</td><td>{int(dm):,} ms</td></tr>"
        for (b, c, it, ot, cu, dm) in by_backend
    ) or '<tr><td colspan="6" class="muted">no data yet</td></tr>'

    rows_model = "".join(
        f"<tr><td>{_esc(a)}</td><td>{_esc(m)}</td><td>{c:,}</td>"
        f"<td>{int(it):,}</td><td>{int(ot):,}</td><td>${cu:.4f}</td></tr>"
        for (a, m, c, it, ot, cu) in by_model
    ) or '<tr><td colspan="6" class="muted">no data yet</td></tr>'

    rows_source = "".join(
        f"<tr><td>{_esc(s)}</td><td>{c:,}</td><td>${cu:.4f}</td></tr>"
        for (s, c, cu) in by_source
    ) or '<tr><td colspan="3" class="muted">no data yet</td></tr>'

    rows_recent = "".join(
        f"<tr><td>{_fmt_ts(ts - round((dm or 0) / 1000))}</td><td>{_fmt_time(ts)}</td>"
        f"<td><code>{_esc(rp)}</code></td><td>{_esc(b)}</td><td>{_esc(src)}</td>"
        f"<td>{_esc(a)}</td><td>{_esc(m)}</td><td>{_esc(ep)}</td>"
        f"<td class='{ 'ok' if 200 <= (st or 0) < 400 else 'err' }'>{st}</td>"
        f"<td>{int(dm):,} ms</td><td>{int(it or 0)}/{int(ot or 0)}</td>"
        f"<td>${(cu or 0):.5f}</td></tr>"
        for (ts, dm, b, src, a, m, ep, st, it, ot, cu, rp) in recent
    ) or '<tr><td colspan="12" class="muted">no calls recorded yet</td></tr>'

    body = f"""<p class="muted">Auto-refreshes every 30 s. Source override: send header
<code>X-Source: my-workflow</code> from clients to tag calls.</p>

<div class="cards">
  <div class="card"><div class="muted">Total calls</div><div class="metric">{total[0]:,}</div></div>
  <div class="card"><div class="muted">Total cost</div><div class="metric">${total[1]:.4f}</div></div>
  <div class="card"><div class="muted">Last 24h calls</div><div class="metric">{h24[0]:,}</div></div>
  <div class="card"><div class="muted">Last 24h cost</div><div class="metric">${h24[1]:.4f}</div></div>
</div>

<div class="panel">
<h2>By backend</h2>
<table><thead><tr><th>Backend</th><th>Calls</th><th>Input tokens</th>
<th>Output tokens</th><th>Cost</th><th>Avg duration</th></tr></thead>
<tbody>{rows_backend}</tbody></table>
</div>

<div class="panel">
<h2>By model (top 30)</h2>
<table><thead><tr><th>Alias</th><th>Real model</th><th>Calls</th>
<th>Input</th><th>Output</th><th>Cost</th></tr></thead>
<tbody>{rows_model}</tbody></table>
</div>

<div class="panel">
<h2>By source (top 20)</h2>
<table><thead><tr><th>Source</th><th>Calls</th><th>Cost</th></tr></thead>
<tbody>{rows_source}</tbody></table>
</div>

<div class="panel">
<h2>Recent calls (last 50)</h2>
<table><thead><tr><th>Start</th><th>End</th><th>Request</th><th>Backend</th><th>Source</th><th>Alias</th>
<th>Model</th><th>Endpoint</th><th>Status</th><th>Duration</th>
<th>In/Out</th><th>Cost</th></tr></thead>
<tbody>{rows_recent}</tbody></table>
</div>"""
    return _doc("ai-hub stats", "stats", body, refresh=30)


@stats_app.get("/routing", response_class=HTMLResponse)
def routing() -> str:
    import main  # lazy import: main imports stats at load, so defer to call time
    return _render_routing(main.routing_snapshot())


def _render_routing(snap: dict) -> str:
    aliases = snap["aliases"]
    models = snap["models"]
    conflicts = snap["conflicts"]

    # Collisions panel — only shown when an alias name equals a real model id.
    conflict_html = ""
    if conflicts:
        crows = []
        for c in conflicts:
            shadow = ", ".join(_esc(b) for b in c["shadowed"])
            covered = ", ".join(_esc(b) for b in c["covered"]) or "—"
            if c["shadowed"]:
                kind = '<span class="badge err">shadows real model</span>'
                shadow_cell = f'<td class="err">{shadow}</td>'
            else:
                kind = '<span class="badge info">covered</span>'
                shadow_cell = '<td class="dim">—</td>'
            crows.append(
                f'<tr class="f"><td><code>{_esc(c["name"])}</code></td>'
                f"<td>{kind}</td><td>{covered}</td>{shadow_cell}</tr>"
            )
        has_actionable = any(c["shadowed"] for c in conflicts)
        note = ('<p class="muted">⚠ <b>Shadowed</b> hosts serve a real model whose id '
                "equals the alias but aren't in its mapping — that model is unreachable "
                "by its bare name (the alias overrides the pass-through). Add them to the "
                "alias mapping, or rename the alias.</p>") if has_actionable else (
                '<p class="muted">These aliases intentionally shadow a real model id and '
                "cover every host that serves it — no backend is left unreachable.</p>")
        conflict_html = (
            '<div class="panel"><h2>Alias / model-name collisions</h2>' + note +
            "<table><thead><tr><th>Alias</th><th>Kind</th><th>Covered hosts</th>"
            "<th>Shadowed hosts</th></tr></thead><tbody>" + "".join(crows) +
            "</tbody></table></div>"
        )

    # Aliases — one row per route, in the backend list order.
    arows = []
    for a in aliases:
        name = a["alias"]
        routes = a["routes"]
        if not routes:
            arows.append(f'<tr class="f"><td><code>{_esc(name)}</code></td>'
                         '<td colspan="3" class="muted">no enabled backend maps this alias</td></tr>')
            continue
        for i, r in enumerate(routes):
            label = f'<code>{_esc(name)}</code>' if i == 0 else '<span class="dim">↳</span>'
            if not r.get("enabled", True):
                status = '<span class="badge off">disabled</span>'
            elif not r["healthy"]:
                status = '<span class="badge warn">backend down</span>'
            elif not r["present"]:
                status = '<span class="badge err">model missing</span>'
            elif r.get("busy"):
                status = '<span class="badge busy">busy</span>'
            else:
                status = '<span class="badge ok">routable</span>'
            row_cls = "f off" if not r.get("enabled", True) else "f"
            arows.append(
                f'<tr class="{row_cls}" data-search="{_esc(name)}"><td>{label}</td>'
                f'<td class="host">{_esc(r["backend"])}</td>'
                f'<td><code>{_esc(r["model"])}</code></td>'
                f"<td>{status}</td></tr>"
            )
    alias_html = (
        '<div class="panel"><h2>Aliases <span class="muted">'
        "(a call takes the fastest free unpaid host)</span></h2>"
        "<table><thead><tr><th>Alias</th><th>Host</th><th>Real model</th>"
        "<th>Status</th></tr></thead><tbody>" +
        ("".join(arows) or '<tr><td colspan="4" class="muted">no virtual_models configured</td></tr>') +
        "</tbody></table></div>"
    )

    # Discovered models — which backends a bare/direct model id can route to.
    mrows = []
    for m in models:
        parts = []
        for h in m["hosts"]:
            down = "" if h["healthy"] else ' <span class="badge warn">down</span>'
            busy = ' <span class="badge busy">busy</span>' if h.get("busy") else ""
            parts.append(f'<span class="host">{_esc(h["backend"])}{down}{busy}</span>')
        shadow = ' <span class="badge info">alias exists</span>' if m["shadowed_by_alias"] else ""
        mrows.append(f'<tr class="f"><td><code>{_esc(m["model"])}</code>{shadow}</td>'
                     f'<td>{" &nbsp; ".join(parts)}</td></tr>')
    model_html = (
        '<div class="panel"><h2>Discovered models <span class="muted">'
        "(where a bare / direct model id can route)</span></h2>"
        "<table><thead><tr><th>Model id</th><th>Hosts</th></tr></thead><tbody>" +
        ("".join(mrows) or '<tr><td colspan="2" class="muted">nothing discovered yet</td></tr>') +
        "</tbody></table></div>"
    )

    search = ('<div class="search"><input id="q" type="search" autocomplete="off" '
              'placeholder="filter aliases, models, hosts…">'
              '<span id="count" class="count"></span></div>'
              '<p class="muted">Reflects live health &amp; discovery — reload to refresh.</p>')

    body = search + conflict_html + alias_html + model_html
    return _doc("ai-hub routing", "routing", body, search=True)
