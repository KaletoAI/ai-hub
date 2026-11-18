"""Shared pure pieces of the cloud task backends (Meshy, Tripo): the task state the
adapters poll towards, and the admin-option form schema every cloud kind declares
(`OPTION_FIELDS`) so ONE console editor serves all of them. No main/adapters imports."""
from __future__ import annotations

import base64
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Iterator, Optional


class DataUri:
    """`data:<mime>;base64,<data>` that is NOT built until it is sent.

    A rigging request carries the whole mesh this way. As a plain str a 70 MB GLB was a
    93 MB string, then a 93 MB `json.dumps` text, then its 93 MB encoded copy — and
    `json.dumps`/base64 hold the GIL for the whole call, so the event loop stood still
    ~0.2 s per job even from a worker thread (measured 2026-11-18). This object keeps
    the caller's bytes (no copy) and `json_stream` encodes them in slices.

    Deliberately NOT a str and not JSON-serialisable: a body holding one must go out
    through `json_stream`; anything else fails loudly instead of quietly rebuilding
    the 93 MB."""
    __slots__ = ("mime", "data")

    def __init__(self, mime: str, data: bytes):
        self.mime, self.data = mime, data

    @property
    def prefix(self) -> str:
        return f"data:{self.mime};base64,"

    @property
    def size(self) -> int:                    # bytes of the payload itself
        return len(self.data)

    def __len__(self) -> int:                 # characters of the URI text
        return len(self.prefix) + 4 * ((len(self.data) + 2) // 3)

    def __str__(self) -> str:                 # the whole text — small inputs and tests only
        return self.prefix + base64.b64encode(self.data).decode()

    def chunks(self, raw: int = 1572864) -> Iterator[bytes]:
        """The URI text as ASCII bytes, `raw` payload bytes (a multiple of 3, so the
        base64 of the slices concatenates to the base64 of the whole) at a time."""
        raw = max(3, raw - raw % 3)
        yield self.prefix.encode()
        view = memoryview(self.data)
        for i in range(0, len(view), raw):
            yield base64.b64encode(view[i:i + raw])


def json_stream(body, raw: int = 1572864) -> tuple:
    """(content length, [byte chunks…] lazily, number of DataUri values) — the exact
    bytes of `json.dumps(body)` with every DataUri written as its text, without ever
    holding that text. The skeleton is dumped with a random token in each DataUri's
    place and cut there; a string from a client cannot pose as the token (it carries a
    fresh uuid), and base64 needs no JSON escaping."""
    tag = uuid.uuid4().hex
    blobs: list = []

    def default(o):
        if isinstance(o, DataUri):
            blobs.append(o)
            return f"@@gwblob:{tag}:{len(blobs) - 1}@@"
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    parts = re.split(rf'"@@gwblob:{tag}:(\d+)@@"', json.dumps(body, default=default))
    texts = [p.encode() for p in parts[0::2]]
    order = [blobs[int(i)] for i in parts[1::2]]
    length = sum(map(len, texts)) + sum(len(b) + 2 for b in order)

    def chunks() -> Iterator[bytes]:
        for i, text in enumerate(texts):
            if text:
                yield text
            if i < len(order):
                yield b'"'
                yield from order[i].chunks(raw)
                yield b'"'
    return length, chunks(), len(order)


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
