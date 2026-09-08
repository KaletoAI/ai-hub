"""Generation-alias store — the writable source of truth for the UI.

Per the multimodal-gateway plan (§13b): the store is authoritative, YAML is a
one-way bootstrap. On init we import the config's `image_models` into an empty
store; thereafter the UI reads/writes here and the router resolves from here.
Self-contained stdlib sqlite3 (WAL), in the spirit of stats.py / jobs.py.

An alias maps to an ordered list of candidates, each:
    {backend, task, workflow, model?, mapping?}
where `mapping` is the explicit {param: {node, field}} binding table.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import struct
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = "store.db"
_active = False


# ── Secret encryption (stdlib-only, zero-dep) ────────────────────────────────────
# Backend api_keys are stored encrypted, not plaintext. Construction: per-secret
# random nonce → HMAC-SHA256 keystream (CTR-style) XOR plaintext, authenticated with
# a separate-subkey HMAC-SHA256 (encrypt-then-MAC). Master key from $GATEWAY_SECRET_KEY
# or an auto-generated `secret.key` (chmod 600) next to the DB. Protects the DB file
# at rest (leak/backup/commit); not against an attacker who already has the key file.

_ENC_PREFIX = "enc:v1:"
_MASTER_KEY: Optional[bytes] = None


def _master_key() -> bytes:
    global _MASTER_KEY
    if _MASTER_KEY is not None:
        return _MASTER_KEY
    env = os.environ.get("GATEWAY_SECRET_KEY")
    if env:
        _MASTER_KEY = hashlib.sha256(env.encode()).digest()
        return _MASTER_KEY
    path = os.path.join(os.path.dirname(_DB_PATH) or ".", "secret.key")
    try:
        with open(path, "rb") as fh:
            _MASTER_KEY = fh.read()
    except FileNotFoundError:
        _MASTER_KEY = os.urandom(32)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(_MASTER_KEY)
        os.chmod(path, 0o600)
        logger.info(f"store: generated secret key at {path} (chmod 600)")
    return _MASTER_KEY


def _subkey(label: bytes) -> bytes:
    return hmac.new(_master_key(), label, hashlib.sha256).digest()


def _keystream(enc_key: bytes, nonce: bytes, n: int) -> bytes:
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hmac.new(enc_key, nonce + struct.pack(">I", i), hashlib.sha256).digest()
        i += 1
    return bytes(out[:n])


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret string → 'enc:v1:<b64>'. Idempotent (already-encrypted and
    empty values pass through)."""
    if not plaintext or plaintext.startswith(_ENC_PREFIX):
        return plaintext
    enc_key, mac_key = _subkey(b"gw-enc-v1"), _subkey(b"gw-mac-v1")
    nonce = os.urandom(16)
    pt = plaintext.encode("utf-8")
    ct = bytes(a ^ b for a, b in zip(pt, _keystream(enc_key, nonce, len(pt))))
    tag = hmac.new(mac_key, nonce + ct, hashlib.sha256).digest()
    return _ENC_PREFIX + base64.b64encode(nonce + ct + tag).decode()


def decrypt_secret(token: str) -> str:
    """Inverse of encrypt_secret. Legacy plaintext (no prefix) passes through, so old
    rows keep working until re-saved. Returns '' on tamper/wrong-key."""
    if not token or not token.startswith(_ENC_PREFIX):
        return token
    try:
        raw = base64.b64decode(token[len(_ENC_PREFIX):])
        nonce, ct, tag = raw[:16], raw[16:-32], raw[-32:]
        enc_key, mac_key = _subkey(b"gw-enc-v1"), _subkey(b"gw-mac-v1")
        if not hmac.compare_digest(hmac.new(mac_key, nonce + ct, hashlib.sha256).digest(), tag):
            logger.warning("store: secret MAC mismatch (wrong key / tampered)")
            return ""
        pt = bytes(a ^ b for a, b in zip(ct, _keystream(enc_key, nonce, len(ct))))
        return pt.decode("utf-8")
    except Exception:
        logger.warning("store: secret decrypt failed")
        return ""


def init(db_path: str = "store.db") -> None:
    global _DB_PATH, _active
    _DB_PATH = db_path
    with _conn() as c:
        c.execute("PRAGMA journal_mode=WAL")   # persistent DB property — set once here
        c.execute("""
            CREATE TABLE IF NOT EXISTS gen_aliases (
                alias          TEXT PRIMARY KEY,
                candidates_json TEXT NOT NULL,
                updated        INTEGER NOT NULL
            )
        """)
        # Backends are keyed by (name, type), so an LLM and a ComfyUI backend may share
        # a name. Migrate the old single-PK (name) table if present.
        bcols = {r[1] for r in c.execute("PRAGMA table_info(backends)").fetchall()}
        old_rows: list = []
        if bcols and "type" not in bcols:
            old_rows = c.execute("SELECT name, json, updated FROM backends").fetchall()
            c.execute("DROP TABLE backends")
            bcols = set()
        if not bcols:
            c.execute("""
                CREATE TABLE backends (
                    name    TEXT NOT NULL,
                    type    TEXT NOT NULL DEFAULT 'openai',
                    json    TEXT NOT NULL,
                    updated INTEGER NOT NULL,
                    PRIMARY KEY (name, type)
                )
            """)
            for r in old_rows:
                b = json.loads(r["json"])
                c.execute("INSERT INTO backends (name, type, json, updated) VALUES (?,?,?,?)",
                          (r["name"], b.get("type", "openai"), r["json"], r["updated"]))
        c.execute("""
            CREATE TABLE IF NOT EXISTS chat_aliases (
                alias      TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated    INTEGER NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key        TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated    INTEGER NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                name    TEXT PRIMARY KEY,
                json    TEXT NOT NULL,
                updated INTEGER NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS backend_models (
                bid         TEXT PRIMARY KEY,
                models_json TEXT NOT NULL,
                updated     INTEGER NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS backend_context (
                bid          TEXT PRIMARY KEY,
                context_json TEXT NOT NULL,
                updated      INTEGER NOT NULL
            )
        """)
    _active = True
    logger.info(f"store: generation aliases at {_DB_PATH}")


def is_active() -> bool:
    return _active


@contextmanager
def _conn():
    conn = sqlite3.connect(_DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Generic single-JSON-column table access ──────────────────────────────────────
# Every store entity shares one shape: PK column(s) + one JSON value column +
# `updated`. These four helpers hold the ONE select/upsert/delete implementation;
# the public per-entity functions below keep their exact signatures, so callers in
# main/admin stay untouched. Table/column names are module constants, never input.

def _row_get(table: str, key_cols: tuple, key_vals: tuple, val_col: str) -> Optional[str]:
    where = " AND ".join(f"{c}=?" for c in key_cols)
    with _conn() as c:
        row = c.execute(f"SELECT {val_col} FROM {table} WHERE {where}", key_vals).fetchone()
    return row[val_col] if row else None


def _row_upsert(table: str, key_cols: tuple, key_vals: tuple, val_col: str, value: str) -> None:
    cols = ", ".join((*key_cols, val_col, "updated"))
    marks = ",".join("?" * (len(key_cols) + 2))
    with _conn() as c:
        c.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({marks}) "
            f"ON CONFLICT({', '.join(key_cols)}) DO UPDATE "
            f"SET {val_col}=excluded.{val_col}, updated=excluded.updated",
            (*key_vals, value, int(time.time())),
        )


def _row_delete(table: str, key_cols: tuple, key_vals: tuple) -> None:
    where = " AND ".join(f"{c}=?" for c in key_cols)
    with _conn() as c:
        c.execute(f"DELETE FROM {table} WHERE {where}", key_vals)


def _rows_all(table: str, cols: tuple, order: str) -> list:
    with _conn() as c:
        return c.execute(f"SELECT {', '.join(cols)} FROM {table} ORDER BY {order}").fetchall()


def _decode_secret_json(raw: str) -> dict:
    """JSON row → dict with `api_key` decrypted (backends and users share this shape)."""
    d = json.loads(raw)
    if d.get("api_key"):
        d["api_key"] = decrypt_secret(d["api_key"])
    return d


def _encode_secret_json(entity: dict) -> str:
    """dict → JSON row with `api_key` encrypted — plaintext is never persisted."""
    d = dict(entity)
    if d.get("api_key"):
        d["api_key"] = encrypt_secret(d["api_key"])
    return json.dumps(d)


def bootstrap(image_models_cfg: dict) -> None:
    """Seed the store from config `image_models` for aliases not already present
    (one-way import; the store wins once populated)."""
    existing = set(list_aliases().keys())
    seeded = 0
    for alias, candidates in (image_models_cfg or {}).items():
        if alias not in existing:
            upsert(alias, candidates)
            seeded += 1
    if seeded:
        logger.info(f"store: bootstrapped {seeded} generation alias(es) from config")


def list_aliases() -> dict:
    return {r["alias"]: json.loads(r["candidates_json"])
            for r in _rows_all("gen_aliases", ("alias", "candidates_json"), "alias")}


def get(alias: str) -> Optional[list]:
    raw = _row_get("gen_aliases", ("alias",), (alias,), "candidates_json")
    return json.loads(raw) if raw else None


def upsert(alias: str, candidates: list) -> None:
    _row_upsert("gen_aliases", ("alias",), (alias,), "candidates_json", json.dumps(candidates))


def delete(alias: str) -> None:
    _row_delete("gen_aliases", ("alias",), (alias,))


# ── Backends (UI-added, additive to config) ─────────────────────────────────────
# api_key is stored encrypted (encrypt_secret) and decrypted on read, so callers
# always see/store plaintext while the DB never holds it.

def list_backends() -> list:
    return [_decode_secret_json(r["json"]) for r in _rows_all("backends", ("json",), "name")]


def get_backend(name: str, btype: str = "openai") -> Optional[dict]:
    raw = _row_get("backends", ("name", "type"), (name, btype), "json")
    return _decode_secret_json(raw) if raw else None


def upsert_backend(backend: dict) -> None:
    _row_upsert("backends", ("name", "type"), (backend["name"], backend.get("type", "openai")),
                "json", _encode_secret_json(backend))


def delete_backend(name: str, btype: str = "openai") -> None:
    _row_delete("backends", ("name", "type"), (name, btype))


def rename_backend_references(old: str, new: str) -> int:
    """Re-point every alias that names `old` backend to `new` — generation-alias
    candidates and per-backend chat-alias entries. Returns how many aliases changed,
    so a UI rename doesn't leave dangling 'down' references."""
    changed = 0
    for alias, cands in list_aliases().items():
        hit = False
        for c in cands:
            if c.get("backend") == old:
                c["backend"], hit = new, True
        if hit:
            upsert(alias, cands)
            changed += 1
    for alias, value in list_chat_aliases().items():
        if isinstance(value, dict) and old in value:
            value[new] = value.pop(old)        # keep model/priority, swap the key
            upsert_chat_alias(alias, value)
            changed += 1
    return changed


def backend_references(name: str) -> dict:
    """Everything in the store that names `name` as a backend, WITHOUT touching it:
    generation-alias candidates, per-backend chat-alias entries, reasoning rules and
    whole-backend user grants. Same buckets as delete_backend_references() so the
    UI can show beforehand exactly what a delete would do.

    Three buckets are "would not be removed", because removing them destroys or
    widens something rather than just dropping a dangling name:
      · `gen_last` — the alias's LAST candidate: `workflow_json` lives inside the
        candidate, so dropping it would leave the alias standing without its
        workflow (the editor enforces the same rule: "an alias must keep at least one");
      · `rule_last` — a rule's LAST backend: an empty list means ALL backends in
        reasoning.resolve(), so removing it would silently widen the rule;
      · `user_alias` — an allow-list entry that is also an alias name, which then
        grants that alias rather than the backend.
    """
    out = {k: [] for k in ("gen", "gen_last", "chat", "chat_empty",
                           "rule", "rule_last", "user", "user_alias")}
    gen = list_aliases()
    alias_names = set(gen) | set(list_chat_aliases())
    for alias, cands in sorted(gen.items()):
        if not any((c.get("backend") or "").strip() == name for c in cands):
            continue
        kept = [c for c in cands if (c.get("backend") or "").strip() != name]
        out["gen" if kept else "gen_last"].append(alias)
    for alias, value in sorted(list_chat_aliases().items()):
        if isinstance(value, dict) and name in value:
            out["chat_empty" if len(value) == 1 else "chat"].append(alias)
    for r in get_reasoning_rules() or []:
        bks = r.get("backends") or []
        if name not in bks:
            continue
        label = r.get("match") or "*"
        out["rule_last" if len(bks) == 1 else "rule"].append(label)
    for u in list_users():
        if name in (u.get("models") or []):
            out["user_alias" if name in alias_names else "user"].append(u.get("name") or "?")
    return out


def delete_backend_references(name: str) -> dict:
    """Drop every reference to a backend being removed — the counterpart to
    rename_backend_references(), which is what a delete had been missing: a deleted
    backend used to stay behind in every alias that named it.

    REFERENCES only. No alias, rule or user is ever deleted here, and the three
    buckets documented in backend_references() are left untouched. Returns the same
    dict, now describing what actually happened."""
    found = backend_references(name)
    for alias in found["gen"]:
        cands = get(alias) or []
        kept = [c for c in cands if (c.get("backend") or "").strip() != name]
        if kept and len(kept) != len(cands):
            upsert(alias, kept)
    for alias in found["chat"] + found["chat_empty"]:
        value = get_chat_alias(alias)
        if isinstance(value, dict) and name in value:
            value.pop(name)
            upsert_chat_alias(alias, value)          # the alias itself stays, empty or not
    if found["rule"]:
        rules = get_reasoning_rules() or []
        for r in rules:
            bks = r.get("backends") or []
            if name in bks and len(bks) > 1:
                r["backends"] = [b for b in bks if b != name]
        set_reasoning_rules(rules)
    for uname in found["user"]:
        u = get_user(uname)
        if u and name in (u.get("models") or []):
            u["models"] = [m for m in u["models"] if m != name]
            upsert_user(u)
    return found


# ── Server settings (UI-managed overrides of config.yaml) ────────────────────────
# Secret-valued keys (api_key) are encrypted at rest like backend keys.

_SECRET_SETTINGS = {"api_key"}


def save_backend_models(bid: str, models) -> None:
    """Persist a backend's discovered model ids so they survive a restart — lets the
    gateway still resolve a bare model id to its (now offline) backend, returning a
    truthful 503 instead of 403, even if the backend isn't reachable at startup."""
    _row_upsert("backend_models", ("bid",), (bid,), "models_json", json.dumps(sorted(models)))


def load_backend_models() -> dict:
    """{bid: set(model_ids)} from each backend's last successful discovery."""
    if not _active:
        return {}
    with _conn() as c:
        rows = c.execute("SELECT bid, models_json FROM backend_models").fetchall()
    return {r["bid"]: set(json.loads(r["models_json"] or "[]")) for r in rows}


def save_backend_context(bid: str, context: dict) -> None:
    """Persist a backend's LEARNED context windows {model: tokens}. A llama-swap model
    reports its n_ctx only while loaded, so what one poll learned must outlive the
    unload and a gateway restart — that is what makes the value dependable."""
    _row_upsert("backend_context", ("bid",), (bid,), "context_json",
                json.dumps({str(k): int(v) for k, v in (context or {}).items()}, sort_keys=True))


def load_backend_context() -> dict:
    """{bid: {model: tokens}} as last persisted by save_backend_context()."""
    if not _active:
        return {}
    with _conn() as c:
        rows = c.execute("SELECT bid, context_json FROM backend_context").fetchall()
    return {r["bid"]: {k: int(v) for k, v in json.loads(r["context_json"] or "{}").items()}
            for r in rows}


def get_settings() -> dict:
    """ALL settings (server tab / startup overlay). Per-key readers use get_setting()."""
    with _conn() as c:
        rows = c.execute("SELECT key, value_json FROM settings").fetchall()
    out = {}
    for r in rows:
        v = json.loads(r["value_json"])
        if r["key"] in _SECRET_SETTINGS and isinstance(v, str):
            v = decrypt_secret(v)
        out[r["key"]] = v
    return out


def get_setting(key: str, default=None):
    """One settings value (single-row read — the per-key helpers below use this
    instead of parsing the whole settings table)."""
    raw = _row_get("settings", ("key",), (key,), "value_json")
    if raw is None:
        return default
    v = json.loads(raw)
    if key in _SECRET_SETTINGS and isinstance(v, str):
        v = decrypt_secret(v)
    return v


def set_settings(values: dict) -> None:
    now = int(time.time())
    with _conn() as c:
        for k, v in values.items():
            if k in _SECRET_SETTINGS and isinstance(v, str) and v:
                v = encrypt_secret(v)
            c.execute(
                "INSERT INTO settings (key, value_json, updated) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated=excluded.updated",
                (k, json.dumps(v), now),
            )


# ── IP aliases (friendly names for caller IPs in stats; auto reverse-DNS) ────────
# Stored as a single settings dict {ip: name}. An empty name means "auto-resolve
# was attempted but found no hostname" — kept so we don't retry every page load.

def get_ip_aliases() -> dict:
    return get_setting("ip_aliases") or {}


def save_ip_aliases(aliases: dict) -> None:
    set_settings({"ip_aliases": dict(aliases)})


def set_ip_alias(ip: str, name: str) -> None:
    a = get_ip_aliases()
    a[ip] = name
    save_ip_aliases(a)


def delete_ip_alias(ip: str) -> None:
    a = get_ip_aliases()
    if ip in a:
        del a[ip]
        save_ip_aliases(a)


# ── Per-alias sync-park time (seconds a call may wait for a free backend) ────────
# Stored as one settings dict {alias: seconds}. Absent → the global default; 0 → no
# parking for that alias (immediate 503 when all its backends are busy).

def get_alias_park() -> dict:
    return get_setting("alias_park") or {}


def set_alias_park(alias: str, park_s) -> None:
    """Set/clear an alias's park seconds. None/'' removes the override (→ global default)."""
    m = get_alias_park()
    if park_s in (None, ""):
        m.pop(alias, None)
    else:
        try:
            m[alias] = max(0, int(park_s))
        except (TypeError, ValueError):
            return
    set_settings({"alias_park": m})


def rename_alias_park(old: str, new: str) -> None:
    if old == new:
        return
    m = get_alias_park()
    if old in m:
        m[new] = m.pop(old)
        set_settings({"alias_park": m})


# ── Per-alias reasoning default (normalized thinking toggle) ─────────────────────
# One settings dict {alias: "off"|"on"}; absent = auto. Lets two aliases point at the
# same backend+model with different thinking behavior (e.g. `tool` off /
# `tool-thinking` auto); an explicit client `reasoning` field always wins.

def get_alias_reasoning() -> dict:
    return get_setting("alias_reasoning") or {}


def set_alias_reasoning(alias: str, mode) -> None:
    """Set/clear an alias's reasoning default. Anything but 'on'/'off' clears (→ auto)."""
    m = get_alias_reasoning()
    if mode in ("on", "off"):
        m[alias] = mode
    elif alias in m:
        del m[alias]
    else:
        return
    set_settings({"alias_reasoning": m})


# ── Per-alias sampling defaults ──────────────────────────────────────────────────
# One settings dict {alias: {param: value}} filled into chat bodies when the CLIENT
# omits the key (client always wins). Backends carry their own `sampling_defaults`,
# which apply after these — precedence is client > alias > backend.

def get_alias_sampling() -> dict:
    return get_setting("alias_sampling") or {}


def set_alias_sampling(alias: str, params) -> None:
    """Set/clear an alias's sampling defaults. Empty/None/non-dict clears."""
    m = get_alias_sampling()
    d = dict(params) if isinstance(params, dict) else {}
    if d:
        m[alias] = d
    elif alias in m:
        del m[alias]
    else:
        return
    set_settings({"alias_sampling": m})


def get_alias_voice() -> dict:
    return get_setting("alias_voice") or {}


def set_alias_voice(alias: str, defaults) -> None:
    """Set/clear an alias's TTS voice defaults ({voice, ref_text} — filled into
    /v1/audio/speech bodies when the client omits them). Empty/None clears."""
    m = get_alias_voice()
    d = {k: str(v).strip() for k, v in (defaults or {}).items()
         if k in ("voice", "ref_text") and str(v or "").strip()}
    if d:
        m[alias] = d
    elif alias in m:
        del m[alias]
    else:
        return
    set_settings({"alias_voice": m})


# ── Voice reference library ──────────────────────────────────────────────────────
# Named voice-cloning references for /v1/audio/speech: the WAV blob lives on the
# GATEWAY host (voiceref/<name>.wav — TTS backends read `voice` only as a local
# file, so the blob must additionally be shipped to the backend host via scp).
# Entry: {ref_text, file (gateway path), remote (backend-side abs path), shipped}.

def get_voice_library() -> dict:
    return get_setting("voice_library") or {}


def set_voice_entry(name: str, entry) -> None:
    """Upsert (dict) or delete (None) one library entry."""
    m = get_voice_library()
    if entry:
        m[name] = entry
    elif name in m:
        del m[name]
    else:
        return
    set_settings({"voice_library": m})


# ── Hosts (physical boxes backends run on) ─────────────────────────────────────
# Backends group by host (explicit `host` field on the backend, else URL IP —
# derived in main.backend_host). This map holds the per-host extras: a display
# label now, the shared-GPU coordination flags later (host-coordination plan).

def get_hosts() -> dict:
    return get_setting("hosts") or {}


def set_host(name: str, entry) -> None:
    """Upsert (dict) or delete (None/empty) one host entry."""
    m = get_hosts()
    if entry:
        m[name] = entry
    elif name in m:
        del m[name]
    else:
        return
    set_settings({"hosts": m})


# ── Reasoning rules (normalized thinking toggle) ────────────────────────────────
# Ordered list of {match, backends[], adapter, param} — see reasoning.py. Stored as
# one settings entry; first matching rule (model-glob × backend-set) wins.

def get_reasoning_rules() -> list:
    v = get_setting("reasoning_rules")
    return v if isinstance(v, list) else []


def set_reasoning_rules(rules: list) -> None:
    set_settings({"reasoning_rules": list(rules or [])})


# ── Users (multi-user: api_key → identity, role, quota, model access) ────────────
# Each user: {name, api_key(enc), role, enabled, models[], quota_req_day, quota_cost_month}.
# api_key encrypted at rest; the empty `models` list means "all models allowed".

def list_users() -> list:
    return [_decode_secret_json(r["json"]) for r in _rows_all("users", ("json",), "name")]


def get_user(name: str) -> Optional[dict]:
    raw = _row_get("users", ("name",), (name,), "json")
    return _decode_secret_json(raw) if raw else None


def upsert_user(user: dict) -> None:
    _row_upsert("users", ("name",), (user["name"],), "json", _encode_secret_json(user))


def delete_user(name: str) -> None:
    _row_delete("users", ("name",), (name,))


# ── Chat aliases (UI-added/overridden, merged over config `virtual_models`) ──────
# A value is either a model-id string (same model on every backend) or a per-backend
# dict {backend: model-id | {model, priority}} — exactly the config `virtual_models`
# shapes, so the merged result drops straight into the LLM router.

def list_chat_aliases() -> dict:
    return {r["alias"]: json.loads(r["value_json"])
            for r in _rows_all("chat_aliases", ("alias", "value_json"), "alias")}


def get_chat_alias(alias: str) -> Optional[object]:
    raw = _row_get("chat_aliases", ("alias",), (alias,), "value_json")
    return json.loads(raw) if raw else None


def upsert_chat_alias(alias: str, value) -> None:
    _row_upsert("chat_aliases", ("alias",), (alias,), "value_json", json.dumps(value))


def delete_chat_alias(alias: str) -> None:
    _row_delete("chat_aliases", ("alias",), (alias,))
