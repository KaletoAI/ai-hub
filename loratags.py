"""LoRA trigger words — the PURE half (spec 2026-10-02-lora-trigger-words-design).

AI-Hub stores and delivers a LoRA's trigger words; it NEVER changes a prompt. The
metadata hangs on the sha256 of the file on the LAN model share (the share's files are
renamed, so a name is no identity); Civitai answers by that hash. Every rule the
worker, the API and the console share lives here: cleaning Civitai's raw
`trainedWords`, parsing its by-hash answer, mapping a ComfyUI LoRA name onto a share
path, the ONE status rule and the item a client gets. No I/O; imports neither `main`
nor `adapters` (only `modelsync.MODEL_EXT`).
"""
import posixpath
import re
import time
from typing import Callable, Optional

from modelsync import MODEL_EXT

LORA_ROOT = "models/loras/"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
STATUSES = ("unavailable", "pending", "not_on_share", "curated", "civitai", "not_on_civitai")
_WS = re.compile(r"\s+")
_COMMAS = re.compile(r",(\s*,)+")
_SPACE_COMMA = re.compile(r"\s+,")


def valid_sha(s) -> bool:
    return isinstance(s, str) and bool(SHA_RE.match(s))


def clean_words(raw) -> list:
    """Civitai's `trainedWords` (or the console's lines) → a clean list. Only the FORM
    is cleaned — whitespace runs, `,,` runs, spaces before commas, leading/trailing
    commas, empties, exact duplicates; the order is kept and an entry is NEVER split: a
    tag chain stays one entry (requirement F4 — taking it apart is the operator's call,
    in the curated list). Non-strings are dropped; anything but a list is []."""
    if not isinstance(raw, list):
        return []
    out: list = []
    for w in raw:
        if not isinstance(w, str):
            continue
        w = _SPACE_COMMA.sub(",", _COMMAS.sub(",", _WS.sub(" ", w))).strip(" ,")
        if w and w not in out:
            out.append(w)
    return out


def _int(v) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _str(v) -> Optional[str]:
    return v.strip() if isinstance(v, str) and v.strip() else None


def parse_civitai(body, now: float) -> dict:
    """Civitai's `GET /api/v1/model-versions/by-hash/<sha256>` answer → the `found`
    record kept in the store. Odd fields become None/[]; ValueError when the body is no
    model version at all (an error object, a challenge page) — the worker treats that
    as a transient failure, never as data."""
    if not isinstance(body, dict):
        raise ValueError("not an object")
    version_id, model_id = _int(body.get("id")), _int(body.get("modelId"))
    if version_id is None and model_id is None:
        raise ValueError("not a model version")
    model = body.get("model") if isinstance(body.get("model"), dict) else {}
    return {"status": "found", "model_id": model_id, "version_id": version_id,
            "model_name": _str(model.get("name")), "version_name": _str(body.get("name")),
            "base_model": _str(body.get("baseModel")),
            "trained_words": clean_words(body.get("trainedWords")),
            "fetched_at": float(now)}


def not_found_record(now: float) -> dict:
    """Civitai's 404: kept, so the next pass does not ask again (F2)."""
    return {"status": "not_found", "fetched_at": float(now)}


def civitai_url(civ) -> Optional[str]:
    """The model page, built from the two integer ids ONLY — never a URL out of an
    answer (it would be a link the console renders)."""
    if not isinstance(civ, dict):
        return None
    mid, vid = _int(civ.get("model_id")), _int(civ.get("version_id"))
    if mid is None:
        return None
    if vid is None:
        return f"https://civitai.com/models/{mid}"
    return f"https://civitai.com/models/{mid}?modelVersionId={vid}"


def retry_after_s(value) -> Optional[float]:
    """A 429's `Retry-After` in seconds (digits only, capped at one hour); None for an
    HTTP date or anything else — the caller then doubles its own pause."""
    if isinstance(value, str) and value.strip().isdigit():
        return float(min(int(value.strip()), 3600))
    return None


def _is_lora_file(path: str) -> bool:
    return (path.startswith(LORA_ROOT) and not path.endswith(".part")
            and path.lower().endswith(MODEL_EXT))


def share_loras(listing) -> dict:
    """The share's LoRAs from a LanSource listing (`{path: size | {"link": target}}`) →
    `{listed path: (real path, size)}`. A file maps to itself; a symlink to its target
    when that is a listed LoRA FILE — the share hashes the target (the serve script
    refuses a link path); anything else is left out."""
    out: dict = {}
    if not isinstance(listing, dict):
        return out
    for p, v in listing.items():
        if isinstance(v, int) and not isinstance(v, bool) and _is_lora_file(p):
            out[p] = (p, v)
    for p, v in listing.items():
        if (isinstance(v, dict) and isinstance(v.get("link"), str)
                and p.startswith(LORA_ROOT) and p.lower().endswith(MODEL_EXT)):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(p), v["link"]))
            hit = out.get(target)
            if hit is not None and hit[0] == target:
                out[p] = hit
    return out


def share_path(name, loras: dict) -> tuple:
    """A ComfyUI LoRA name → `(listed share path, "")`, or `(None, why)`:
    `models/loras/<name>` when listed, else the ONE listed LoRA whose path ends in
    `/<name>`; two such = "ambiguous", none = "missing". Never a guess — the wrong one of
    two same-named files would deliver another LoRA's words."""
    if (not isinstance(name, str) or not name or name.startswith("/")
            or ".." in name.split("/")):
        return None, "missing"
    direct = LORA_ROOT + name
    if direct in loras:
        return direct, ""
    hits = [p for p in loras if p.endswith("/" + name)]
    if len(hits) == 1:
        return hits[0], ""
    return None, ("ambiguous" if hits else "missing")


def status_of(configured: bool, usable: bool, path, sha, rec) -> str:
    """The ONE status rule (first match wins) — client, console and counts all read it,
    so "not determined yet" can never read as a verdict somewhere."""
    if not configured:
        return "unavailable"
    if not usable:
        return "pending"
    if path is None:
        return "not_on_share"
    if not sha:
        return "pending"
    rec = rec if isinstance(rec, dict) else {}
    if rec.get("curated") is not None:
        return "curated"
    st = (rec.get("civitai") or {}).get("status")
    if st == "found":
        return "civitai"
    if st == "not_found":
        return "not_on_civitai"
    return "pending"


def effective_words(rec) -> list:
    """What a client should put into the prompt: the curated list when set (also []),
    else Civitai's cleaned list, else []."""
    if not isinstance(rec, dict):
        return []
    if rec.get("curated") is not None:
        return list(rec["curated"])
    civ = rec.get("civitai") or {}
    return list(civ.get("trained_words") or []) if civ.get("status") == "found" else []


def merge_pair(own: list, other: list) -> list:
    out = list(own)
    for w in other:
        if w not in out:
            out.append(w)
    return out


def _iso(t) -> Optional[str]:
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def civitai_public(civ) -> Optional[dict]:
    """A `found` record as a client sees it (the link built from ids); None otherwise —
    a `not_found` one is told by the item's status."""
    if not isinstance(civ, dict) or civ.get("status") != "found":
        return None
    return {"trained_words": list(civ.get("trained_words") or []),
            "model_id": civ.get("model_id"), "version_id": civ.get("version_id"),
            "model_name": civ.get("model_name"), "version_name": civ.get("version_name"),
            "base_model": civ.get("base_model"), "url": civitai_url(civ),
            "fetched_at": _iso(civ.get("fetched_at"))}


def lookup(name: str, snap: dict) -> dict:
    """One LoRA name's item from a snapshot `{configured, problem, share, shas, meta,
    errors}` (share = `share_loras(…)`, shas = {real path: sha}, meta = {sha: record},
    errors = {sha or real path: {"error": …}}) — memory only (N1)."""
    configured = bool(snap.get("configured"))
    usable = configured and not snap.get("problem")
    share = snap.get("share") or {}
    path = real = sha = rec = None
    if usable:
        path, _ = share_path(name, share)
        if path is not None:
            real = share[path][0]
            sha = (snap.get("shas") or {}).get(real)
            rec = (snap.get("meta") or {}).get(sha) if sha else None
    st = status_of(configured, usable, path, sha, rec)
    cur = rec.get("curated") if isinstance(rec, dict) else None
    out = {"name": name, "status": st, "trigger_words": effective_words(rec),
           "curated": list(cur) if cur is not None else None,
           "civitai": civitai_public((rec or {}).get("civitai")),
           "sha256": sha, "pair": None}
    if st == "pending":
        errs = snap.get("errors") or {}
        err = (errs.get(sha) if sha else None) or (errs.get(real) if real else None)
        if err and err.get("error"):
            out["error"] = str(err["error"])
    return out


def items(names: list, snap: dict, counterpart: Optional[Callable] = None) -> list:
    """The client items for an alias's LoRA names, in their order. `counterpart` (main
    passes `adapters.lora_counterpart` ONLY when the alias's workflow has both high and
    low stacks — exactly when the gateway loads the other half itself, F7): a name
    whose counterpart is in the set gets `pair` and both halves' words, own first."""
    base = {n: lookup(n, snap) for n in names}
    out = []
    for n in names:
        it = dict(base[n])
        cp = counterpart(n) if counterpart else None
        if cp and cp != n and cp in base:
            it["pair"] = {"name": cp, "status": base[cp]["status"]}
            it["trigger_words"] = merge_pair(it["trigger_words"], base[cp]["trigger_words"])
        out.append(it)
    return out
