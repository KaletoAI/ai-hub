"""Shared pure pieces of the cloud task backends (Meshy, Tripo): the task state the
adapters poll towards, and the admin-option form schema every cloud kind declares
(`OPTION_FIELDS`) so ONE console editor serves all of them. No main/adapters imports."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskState:
    status: str
    progress: int = 0
    error: Optional[str] = None
    downloads: list = field(default_factory=list)      # [(filename, url)] in delivery order
    thumbnail: Optional[str] = None
    credits: Optional[float] = None
    # A failed task the VENDOR blames on itself and invites a retry for (Meshy's
    # `task_error.type` server_error/timeout/service_unavailable). The distinction is the
    # vendor's own, never a guess from the message text: a rejected input fails the same
    # way on every candidate, while the service's own hiccup is exactly what the gateway's
    # self-retry and failover exist for. Default False — a module that cannot tell says so.
    retryable: bool = False
    # A task that answers a QUESTION instead of delivering a file (Tripo's free rig-check:
    # "can this mesh be rigged, and as what?"). Optional because every other task leaves
    # them None — the adapter reads them only for the endpoint that sets them.
    riggable: Optional[bool] = None
    rig_type: Optional[str] = None


# ── the option form schema ────────────────────────────────────────────────────
# A field: {key, label, type: bool|select|tristate|int|text|list, choices?, placeholder?,
# hint?, rig_only?, checkbox_text?}. Form names are `opt__<key>` (Meshy's existing names).

_TRISTATE = {"true": True, "false": False}


def _choice_values(fld: dict) -> list:
    """A `select` field's accepted values — its choices may be (value, text) or plain
    strings, because a value that reads well needs no second label."""
    return [c[0] if isinstance(c, (tuple, list)) else c for c in (fld.get("choices") or [])]


def parse_options(fields: list, form: dict, defaults: dict) -> dict:
    """Read `opt__<key>` form values by schema. Unknown/garbage values fall back to the
    DEFAULT for that key (never raise on a form) — the module's `options_of` is the
    validator of record and runs on every read anyway. Returns a fresh dict."""
    out = dict(defaults)
    for fld in fields:
        k, t = fld["key"], fld["type"]
        raw = form.get(f"opt__{k}")
        if t == "bool":
            out[k] = bool(raw)                          # an unchecked box is absent
            continue
        s = (raw or "").strip() if isinstance(raw, str) else ""
        if t == "select":
            out[k] = s if s in _choice_values(fld) else defaults.get(k)
        elif t == "tristate":
            out[k] = _TRISTATE.get(s)                   # "" / unknown → None
        elif t == "int":
            try:
                out[k] = int(float(s)) if s else None
            except ValueError:
                out[k] = None
        elif t == "list":
            out[k] = [x.strip() for x in re.split(r"[\s,]+", s) if x.strip()]
        else:                                           # text
            out[k] = s
    return out


def field_value_str(fld: dict, value) -> str:
    """A stored option value as the form control's string (the inverse of parse_options)."""
    t = fld.get("type")
    if t == "tristate":
        return {True: "true", False: "false"}.get(value, "")
    if t == "list":
        return ", ".join(str(x) for x in (value or []))
    if value is None:
        return ""
    return str(value)


# ── artifact sniffing (a vendor answer or an upload says nothing about the type) ──
def sniff(data):
    """(extension, mime, kind) from the first bytes, or None."""
    head = data[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png", "image"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpg", "image/jpeg", "image"
    if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp", "image"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif", "image/gif", "image"
    if head[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav", "audio/wav", "audio"
    if head.startswith(b"fLaC"):
        return "flac", "audio/flac", "audio"
    if head.startswith(b"OggS"):
        return "ogg", "audio/ogg", "audio"
    if head.startswith(b"ID3") or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return "mp3", "audio/mpeg", "audio"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm", "video/webm", "video"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"M4A ", b"M4B "):
            return "m4a", "audio/mp4", "audio"
        if brand == b"qt  ":
            return "mov", "video/quicktime", "video"
        return "mp4", "video/mp4", "video"
    return None
