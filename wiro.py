"""Pure wiro.ai catalog, request and task helpers; no HTTP, main or adapters imports.

One wiro account is two APIs: an OpenAI-compatible LLM endpoint (llm.wiro.ai) and
the task API (api.wiro.ai). This module is the pure half of the SECOND one plus
the catalog shaping both adapters share: Tool/List row → one media candidate
(category mapping, W3), the parameter items → public fields / file slots (W5),
value precedence with the sample-default refusal (W6) and the task-detail
statuses (only task_postprocess_end / task_cancel are terminal, W8).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from urllib.parse import urlsplit

from cloudtask import TaskState, sniff

KIND, VENDOR = "wiro", "wiro"
URL = "https://llm.wiro.ai"                       # LLM endpoint base; the adapter appends /v1/...
MEDIA_URL = "https://api.wiro.ai/v1"              # task API base (backend key `media_url` overrides)
# `speech`, not `tts`: main's text side owns the catalog key `tts` (NanoGPT's chat-side
# speech); a media catalog under that key became LLM-routable (review 2026-11-10).
ENDPOINTS = ("image", "video", "audio", "speech", "asr")
AI_MODELS, FORMATS, RIG_FORMATS = (), (), ()
DYNAMIC_MODELS = True
RIG_ENDPOINT = None
MAX_WAIT_DEFAULT, POLL_INTERVAL_DEFAULT = 1800, 3.0
SUCCESS_STATUS = "TASK_POSTPROCESS_END"
BACKEND_HINT = ("One wiro backend serves chat (llm.wiro.ai) and task-API media (api.wiro.ai). "
                "api_key is KEY for a key-only project or KEY:SECRET for a signature project. "
                "Generation is paid; an accepted or uncertain submit is never re-created.")
ENDPOINT_HINT = ("Choose an exact discovered owner/project model and its endpoint. "
                 "Controls are the model's own parameters; file and text sample defaults are "
                 "never sent, so required inputs must be set explicitly.")
OPTION_DEFAULTS = {"options_json": ""}
OPTION_FIELDS = [{"key": "options_json", "label": "Model-specific defaults (JSON)", "type": "text",
                  "textarea": True,
                  "hint": "A JSON object of the model's parameters (playground names, or wiro's "
                          "own ids); model and credentials are refused."}]
NONCE_COUNTER = 0                                  # same-second nonce uniqueness (module-level, pure)

_RESERVED = {"model", "endpoint", "provider", "api_key", "apikey", "authorization", "x-api-key",
             "headers", "options_json", "taskid", "task_id", "socketaccesstoken", "tasktoken"}
_FILE_TYPES = frozenset(("fileinput", "combinefileinput", "multifileinput"))
_MULTI_TYPES = frozenset(("combinefileinput", "multifileinput"))
_DEFAULTABLE = frozenset(("select", "selectwithcover", "number", "float", "checkbox"))
_STATUS_ORDER = ("task_queue", "task_accept", "task_preprocess_start", "task_preprocess_end",
                 "task_assign", "task_start", "task_output", "task_output_full", "task_end",
                 "task_postprocess_start", "task_postprocess_end", "task_cancel",
                 "task_error", "task_error_full")
_TERMINAL = frozenset(("task_postprocess_end", "task_cancel"))
_TYPE_MAP = {"number": "int", "float": "float", "checkbox": "bool", "text": "string",
             "textarea": "string", "json": "string", "json-array": "string",
             "select": "string", "selectwithcover": "string", "multiselect": "string"}

# W3, first match wins. realtime-conversation is rejected by Run, llm/chat rows run
# through the LLM endpoint — neither may become a paid task candidate.
_CATEGORY_SKIP = ("realtime-conversation",)
_CATEGORY_MAP = (
    (("speech-to-text",), "asr"),
    (("text-to-speech",), "speech"),
    (("music-generation", "text-to-music", "text-to-song", "image-to-song",
      "speech-to-speech", "voice-clone"), "audio"),
    (("text-to-video", "image-to-video", "video-to-video", "talking-head"), "video"),
    (("text-to-image", "image-to-image", "image-editing", "upscale", "background-removal"), "image"),
)


class WiroInput(RuntimeError):
    """Invalid wiro media input is final; it must never initiate paid failover."""


INPUT_ERROR = WiroInput


def _metadata_guard(fn):
    """Vendor catalog data shapes the paid request: a malformed row (a parameter list
    where an object belongs, a non-numeric bound) must surface as WiroInput — a 400
    before any paid POST — never as a TypeError the preflight lets through (a 500)."""
    import functools

    @functools.wraps(fn)
    def guarded(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except WiroInput:
            raise
        except (TypeError, ValueError, ArithmeticError, AttributeError, KeyError, IndexError) as error:
            raise WiroInput(f"wiro model metadata or input is malformed "
                            f"({type(error).__name__}: {error})") from error
    return guarded


# ── auth (W1) ─────────────────────────────────────────────────────────────────
def split_key(api_key):
    """KEY (key-only project) or (KEY, SECRET) for a signature project; split at the
    FIRST ':' — the LLM endpoint sends the whole api_key verbatim either way."""
    key = (api_key or "").strip()
    if ":" in key:
        first, second = key.split(":", 1)
        return first.strip(), second.strip()
    return key, ""


def signature_headers(api_key, now=None):
    """Task API auth: `x-api-key` always; a signature project adds a fresh nonce and
    hex(HMAC_SHA256(key=KEY, msg=SECRET+NONCE)) per request. Two calls in one second
    differ via a module counter, so a retry never replays a stale signature."""
    global NONCE_COUNTER
    key, secret = split_key(api_key)
    if not key:
        return {}
    headers = {"x-api-key": key}
    if not secret:
        return headers
    if now is None:
        import time
        now = time.time()
    NONCE_COUNTER = (NONCE_COUNTER + 1) % 1000
    nonce = str(int(now) * 1000 + NONCE_COUNTER)    # an integer, as the docs require
    headers["x-nonce"] = nonce
    headers["x-signature"] = hmac.new(key.encode(), (secret + nonce).encode(),
                                      hashlib.sha256).hexdigest()
    return headers


def redact(text, api_key):
    """Both halves of the credential disappear from every error/log text."""
    if not isinstance(text, str):
        return text
    key, secret = split_key(api_key)
    for part in ((api_key or "").strip(), key, secret):
        if part:
            text = text.replace(part, "[redacted]")
    return text


# ── candidate seam ────────────────────────────────────────────────────────────
def endpoint_of(cand):
    endpoint = (cand.get(KIND) or {}).get("endpoint") or "image"
    if endpoint not in ENDPOINTS:
        raise WiroInput("invalid wiro media endpoint")
    return endpoint


def generation_type(endpoint):
    if endpoint not in ENDPOINTS:
        raise WiroInput("invalid wiro media endpoint")
    return "tts" if endpoint == "speech" else endpoint


def task_for(endpoint, previous_task=""):
    return {"image": "text2img", "video": "text2vid", "audio": "text2audio",
            "speech": "text2audio", "asr": "audio2text"}[endpoint_of({"wiro": {"endpoint": endpoint}})]


def candidate_for(backend, model, endpoint, model_meta=None):
    cand = {"backend": backend, "model": model, "task": task_for(endpoint),
            KIND: {"endpoint": endpoint, "options": dict(OPTION_DEFAULTS)}}
    options_of(cand)                            # validate the pinned defaults eagerly
    if model_meta is not None:
        if not isinstance(model_meta, dict):
            raise WiroInput("wiro model metadata must be an object")
        cand["model_metadata"] = model_meta
    return cand


def default_candidate(backend):
    return candidate_for(backend, "", "image")


def options_of(cand):
    stored = (cand.get(KIND) or {}).get("options") or {}
    if not isinstance(stored, dict):
        raise WiroInput("wiro options must be an object")
    unknown = set(stored) - set(OPTION_DEFAULTS)
    if unknown:
        raise WiroInput("unsupported wiro alias option: " + str(sorted(unknown)[0]))
    result = {key: stored.get(key, value) for key, value in OPTION_DEFAULTS.items()}
    _advanced(result["options_json"])
    return result


def read_options(form):
    values = {key: form.get("opt__" + key) for key in OPTION_DEFAULTS}
    values["options_json"] = values["options_json"] or ""
    return options_of({KIND: {"options": values}})


def _advanced(raw):
    if raw in (None, ""):
        return {}
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise WiroInput("model-specific defaults must be valid finite JSON") from exc
    if not isinstance(obj, dict):
        raise WiroInput("model-specific defaults must be a JSON object")
    for key, value in obj.items():
        if str(key).lower() in _RESERVED:
            raise WiroInput("model-specific defaults cannot override model, endpoint or credentials")
        _finite_safe(value)
    return obj


def _finite_safe(value):
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WiroInput("model-specific defaults must be finite JSON")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _RESERVED:
                raise WiroInput("model-specific defaults cannot override model, endpoint or credentials")
            _finite_safe(item)
        return
    if isinstance(value, list):
        for item in value:
            _finite_safe(item)
        return
    raise WiroInput("model-specific defaults must be JSON values")


# ── catalog rows (W3) ─────────────────────────────────────────────────────────
def catalog_endpoint(categories):
    """One row → its single media endpoint, or None to skip it (realtime models are
    rejected by Run; llm/chat/uncategorized rows belong to the LLM endpoint)."""
    if not isinstance(categories, list):
        return None
    lowered = [str(cat).strip().lower() for cat in categories]
    if any(cat in lowered for cat in _CATEGORY_SKIP):
        return None
    for names, endpoint in _CATEGORY_MAP:
        if any(cat in lowered for cat in names):
            return endpoint
    return None


def model_id_of(row):
    """`owner/project` from the CLEAN slugs — `slugproject` may contain spaces."""
    owner = str((row or {}).get("cleanslugowner") or "").strip().strip("/")
    project = str((row or {}).get("cleanslugproject") or "").strip().strip("/")
    if not owner or not project or "/" in owner or "/" in project:
        return ""
    return f"{owner}/{project}"


def tool_metadata(row):
    """The stored media-catalog entry for one Tool/List row; None when the row is not
    a callable media model (skipped category or unusable id)."""
    endpoint = catalog_endpoint((row or {}).get("categories"))
    model = model_id_of(row)
    if endpoint is None or not model:
        return None
    return model, endpoint, {"name": (row or {}).get("title") or model, "parameters": row.get("parameters"),
                             "dynamicprice": row.get("dynamicprice"), "cps": row.get("cps"),
                             "approximatelycost": row.get("approximatelycost"),
                             "description": row.get("description")}


def dynamic_prices(meta):
    """`dynamicprice` is a JSON-encoded STRING in Tool/List and an array in Detail."""
    value = (meta or {}).get("dynamicprice")
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    return value if isinstance(value, list) else []


def price_from(meta):
    """The cheapest advertised run price for the console/`pricing.request` preview;
    per-request (cpr) and per-output (cpo) USD prices only — output/token-metered
    prices (cp-readoutput, per-second) have no request-time USD and stay unknown
    (None, never 0)."""
    prices = []
    for row in dynamic_prices(meta):
        if not isinstance(row, dict) or row.get("priceMethod") not in ("cpr", "cpo"):
            continue
        price = row.get("price")
        if isinstance(price, bool):
            continue
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        if math.isfinite(price) and price > 0:
            prices.append(price)
    return {"usd": min(prices)} if prices else None


def _items(meta):
    """Every parameter item of the stored row, group-tolerant; vendor data is guarded
    at the boundary, so a malformed group raises WiroInput, not a TypeError."""
    groups = (meta or {}).get("parameters")
    if groups in (None, ""):
        return []
    if not isinstance(groups, list):
        raise WiroInput("wiro model parameters must be a list")
    items = []
    for group in groups:
        entries = (group or {}).get("items") if isinstance(group, dict) else None
        if not isinstance(entries, list):
            raise WiroInput("wiro parameter group has no item list")
        items.extend(item for item in entries if isinstance(item, dict))
    return items


def _param_items(cand):
    return _items(cand.get("model_metadata") or {})


def _is_file(item):
    return str(item.get("type") or "").lower() in _FILE_TYPES


def _multi_count(item):
    """File params repeat: multifileinput always (wiro repeats the field), other file
    inputs when maxinputlenght says >1 (cap 16)."""
    if str(item.get("type") or "").lower() == "multifileinput":
        return 16
    try:
        count = int(str(item.get("maxinputlenght") or "1"))
    except (TypeError, ValueError):
        count = 1
    return min(max(count, 1), 16)


def _is_image_file(item):
    fid = str(item.get("id") or "").lower()
    return "image" in fid or item.get("id") == "lastFrameImage"


def _accept(item):
    fid = str(item.get("id") or "").lower()
    if "video" in fid:
        return ["mp4", "webm", "mov"]
    if "audio" in fid or "speaker" in fid:
        return ["wav", "mp3", "flac", "ogg", "m4a"]
    if "document" in fid:
        return ["pdf", "txt", "md"]
    if fid == "inputall":
        return ["png", "jpg", "jpeg", "webp", "gif", "mp4", "webm", "mov",
                "wav", "mp3", "flac", "ogg", "m4a", "pdf", "txt", "md"]
    return []


# Public names follow the gateway's convention (snake_case, `input_image`,
# `input_image_2`, `input_audio`, …) like Venice / NanoGPT / Meshy, so a value or a kept
# upload carries across providers; the wire keeps wiro's own ids. `inputImage2` →
# `input_image_2`, `aspectRatio` → `aspect_ratio`; the few end-frame spellings share
# one name. Measured on the live Tool/List (577 rows, 2026-11-11): no collisions.
_NAME_ALIASES = {"lastFrameImage": "input_image_last", "inputImageEnd": "input_image_last",
                 "inputImageFirst": "input_image"}


def _snake(fid):
    name = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", fid)
    name = re.sub(r"(?<=[A-Za-z])(\d)", r"_\1", name)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _expand(base, count):
    return [base] + [f"{base}_{i}" for i in range(2, count + 1)]


def _visible(cand):
    """{wiro id: item} of the settable items, catalog order."""
    return {str(item["id"]): item for item in _param_items(cand)
            if item.get("id") not in (None, "") and item.get("hidden") is not True}


def _public_names(items):
    """{wiro id: public base name}. An item whose name (or an expanded slot name)
    would be taken twice keeps its wiro id, so two ids never share a public name."""
    def names(fid, base):
        item = items[fid]
        return _expand(base, _multi_count(item)) if _is_file(item) else [base]
    pub = {fid: (_NAME_ALIASES.get(fid) if _is_file(item) else None) or _snake(fid) or fid
           for fid, item in items.items()}
    for _round in range(2):
        seen = {}
        for fid, base in pub.items():
            for name in names(fid, base):
                seen.setdefault(name, set()).add(fid)
        clash = {fid for owners in seen.values() if len(owners) > 1 for fid in owners}
        if not clash:
            return pub
        pub = {fid: (fid if fid in clash else base) for fid, base in pub.items()}
    raise WiroInput("wiro model parameter ids collide")


def _slots(cand):
    """Every upload slot in catalog order: {name (public), vendor (wiro slot name),
    id, image, required, accept}; multi-entry items expand to name, name_2 … (cap 16)."""
    items = _visible(cand)
    pub = _public_names(items)
    slots = []
    for fid, item in items.items():
        if not _is_file(item):
            continue
        required = item.get("required") is True
        image = _is_image_file(item)
        for i, (name, vendor) in enumerate(zip(_expand(pub[fid], _multi_count(item)),
                                               _expand(fid, _multi_count(item)))):
            slots.append({"name": name, "vendor": vendor, "id": fid, "image": image,
                          "required": required and i == 0,
                          "accept": [] if image else _accept(item)})
    return slots


def _to_vendor(values, pub):
    """Client / pinned values → wiro ids. Public names win over a wiro id given for the
    same item; keys naming no item (n, size, …) pass through."""
    rev = {name: fid for fid, name in pub.items()}
    out = {}
    for key in sorted(values, key=lambda k: k not in rev):          # public first
        out.setdefault(rev.get(key, key), values[key])
    return out


def slot_names(cand):
    """Image upload slots (public names): `input_image`, `input_image_2`, …"""
    return [s["name"] for s in _slots(cand) if s["image"]]


def _file_fields(cand):
    """Non-image upload slots, expanded the same way; `required` from the item."""
    return [{"name": s["name"], "required": s["required"], "accept": s["accept"]}
            for s in _slots(cand) if not s["image"]]


def public_fields(cand):
    """(params, images, files): the schema endpoint, playground and upload checks —
    public names (see _public_names); wiro's own label rides along as `label`."""
    items = _visible(cand)
    pub = _public_names(items)
    params = []
    for fid, item in items.items():
        typ = str(item.get("type") or "").lower()
        if _is_file(item):
            continue
        field = {"name": pub[fid], "type": _TYPE_MAP.get(typ, "string")}
        if item.get("required") is True:
            field["required"] = True
        default = _vendor_default(item)        # what the run sends when the client is silent
        if default is not None:
            if typ == "checkbox":
                field["default"] = str(default).strip().lower() in ("true", "1", "yes")
            elif isinstance(default, list):
                field["default"] = ", ".join(str(v) for v in default)
            else:
                field["default"] = default
        options = _choices(item)
        if options:
            field["choices"] = options
        bounds = {}
        for key, target in (("minvalue", "min"), ("maxvalue", "max")):
            raw = item.get(key)
            if raw in (None, ""):
                continue
            try:
                bounds[target] = float(str(raw))
            except (TypeError, ValueError):
                continue
        field.update(bounds)
        label = str(item.get("label") or "").strip()
        if label and label.lower() not in (fid.lower(), pub[fid].lower()):
            field["label"] = label
        note = str(item.get("note") or "").strip()
        if note:
            field["hint"] = note
        params.append(field)
    images = [{"name": s["name"], "required": s["required"],
               "on_empty": "required" if s["required"] else "none"}
              for s in _slots(cand) if s["image"]]
    return params, images, _file_fields(cand)


def _number(key, value, integer=False):
    try:
        number = float(str(value).strip())
        if str(value).strip() == "" or not math.isfinite(number) or (integer and number != int(number)):
            raise ValueError
        return int(number) if integer else number
    except (TypeError, ValueError, OverflowError) as exc:
        raise WiroInput(f"wiro {key} must be a finite {'integer' if integer else 'number'}") from exc


# ── request values (W6) ───────────────────────────────────────────────────────
def _coerce(item, value, name=None):
    typ = str(item.get("type") or "").lower()
    fid = name or str(item.get("id") or "")
    if typ == "checkbox":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
            return value.strip().lower() in ("true", "1", "yes")
        raise WiroInput(f"wiro {fid} must be true or false")
    if typ in ("number", "float"):
        number = _number(fid, value, typ == "number")
        for key, op in (("minvalue", lambda n, bound: n >= bound), ("maxvalue", lambda n, bound: n <= bound)):
            raw = item.get(key)
            if raw in (None, ""):
                continue
            try:
                bound = float(str(raw))
            except (TypeError, ValueError):
                continue
            if not op(number, bound):
                raise WiroInput(f"wiro {fid} is outside the model's bounds")
        return number
    if typ in ("select", "selectwithcover"):
        text = str(value)
        options = _choices(item)
        if options and text not in options:
            raise WiroInput(f"wiro {fid} must be one of the model's choices")
        return text
    if typ == "multiselect":
        picked = value if isinstance(value, list) else [v.strip() for v in str(value).split(",")]
        options = _choices(item)
        if options and any(str(v) not in options for v in picked):
            raise WiroInput(f"wiro {fid} must be among the model's choices")
        return [str(v) for v in picked if str(v)]
    if typ in ("json", "json-array"):
        if isinstance(value, (dict, list)):
            _finite_safe(value)
            return json.dumps(value)
        text = str(value)
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise WiroInput(f"wiro {fid} must be valid JSON") from exc
        if typ == "json-array" and not isinstance(parsed, list):
            raise WiroInput(f"wiro {fid} must be a JSON array")
        if typ == "json" and not isinstance(parsed, (dict, list)):
            raise WiroInput(f"wiro {fid} must be a JSON object or array")
        return text
    if isinstance(value, (dict, list)):
        raise WiroInput(f"wiro {fid} must be a string")
    return str(value)


def _choices(item):
    """Option values; wiro spells them {text, value} or (multiselect) {id, name}."""
    out = []
    for opt in item.get("options") or []:
        if isinstance(opt, dict):
            value = opt.get("value", opt.get("id"))
            if value not in (None, "") and not isinstance(value, bool):
                out.append(str(value))
    return out


def _vendor_default(item):
    """The model's own default for a CONTROL, sent verbatim (it is wiro's value, so it
    is not re-validated: a default outside its own options or a checkbox default like
    '--use_pe' is how wiro's catalog spells them). None = do not send. Never for file
    inputs (sample URLs) or prose (sample prompts); a `text` default only when it is a
    number AND the item is required (seed '0' — a value, not a sample; optional seeds
    stay unset so results vary)."""
    typ = str(item.get("type") or "").lower()
    default = item.get("defaultvalue")
    if default in (None, "", []):
        return None
    if typ in ("number", "float"):
        try:
            return _number(str(item.get("id")), default, typ == "number")
        except WiroInput:
            return default
    if typ in _DEFAULTABLE or typ == "multiselect":
        return default
    # A numeric `text` default only for a REQUIRED item: an optional seed "0" sent on
    # every run would pin every result to one seed (the vendor picks when it is absent).
    if typ == "text" and item.get("required") is True and isinstance(default, (str, int, float)) \
            and not isinstance(default, bool):
        try:
            float(str(default))
            return str(default)
        except ValueError:
            return None
    return None


# What the OpenAI Images shims pass for EVERY client (Open WebUI sends size=1024x1024 by
# default): translated onto the model's own controls where it has them, else dropped —
# refusing them made every OpenAI-image client a 400 on most wiro models.
_OPENAI_GENERIC = ("size", "quality", "style", "background", "output_format",
                   "response_format", "user")


def _openai_generics(client, items):
    size = client.get("size")
    if "size" not in items and isinstance(size, str) and "x" in size.lower():
        try:
            width, height = (int(part) for part in size.lower().split("x", 1))
        except ValueError:
            width = height = 0
        if width > 0 and height > 0:
            if "width" in items and "height" in items:
                client.setdefault("width", width)
                client.setdefault("height", height)
            else:
                ratio_id = next((fid for fid in ("aspectRatio", "aspect_ratio", "ratio")
                                 if fid in items), None)
                if ratio_id:
                    import math as _m
                    g = _m.gcd(width, height)
                    want = f"{width // g}:{height // g}"
                    if want in _choices(items[ratio_id]):
                        client.setdefault(ratio_id, want)
    for key in _OPENAI_GENERIC:
        if key not in items:
            client.pop(key, None)
    n = client.get("n")
    if n not in (None, "") and "n" not in items:
        count_id = next((fid for fid in ("samples", "numberOfImages", "num_images", "numImages")
                         if fid in items), None)
        try:
            count = int(float(n))
        except (TypeError, ValueError):
            count = 0
        if count_id and count > 1:
            client.setdefault(count_id, count)


def _merged_values(cand, values):
    """client value > alias option pin > wiro defaultvalue — but defaults fill ONLY
    controls (_vendor_default): file defaults are wiro SAMPLE URLs and text defaults
    are sample prompts, and sending either silently generates from the vendor's
    samples instead of the client's request. Keys are public names (a wiro id is
    accepted too, for clients and pins written against wiro's own spelling)."""
    opts = options_of(cand)
    # The gateway always passes `prompt` / `negative_prompt` (blank when the client
    # sent none): blank is "not set", never an unknown parameter.
    items = _visible(cand)
    pub = _public_names(items)
    client = _to_vendor({k: v for k, v in (values or {}).items() if v not in (None, "")}, pub)
    if "negative_prompt" in client and "negative_prompt" not in items:
        native = next((fid for fid in ("negativePrompt", "negative") if fid in items), None)
        if native:
            client.setdefault(native, client.pop("negative_prompt"))
    _openai_generics(client, items)
    merged = {**_to_vendor(_advanced(opts.get("options_json")), pub), **client}
    body = {}
    for fid, item in items.items():
        if _is_file(item):
            continue
        value = merged.pop(fid, None)
        if value in (None, ""):
            default = _vendor_default(item)
            if default is not None:
                body[fid] = default
            elif item.get("required") is True:
                raise WiroInput(f"wiro {pub[fid]} is required")
            continue
        body[fid] = _coerce(item, value, pub[fid])
    # Gateway-generic keys that never name a wiro parameter (a model's own "prompt"
    # or "seed" was already consumed from `merged` above; `n` is validated apart).
    unknown = set(merged) - {"n"}
    if unknown:
        raise WiroInput("unsupported wiro parameter: " + str(sorted(unknown)[0]))
    return body, items


def validate_request(cand, values, image_keys, file_keys, model_meta=None):
    if model_meta is not None:
        cand = {**cand, "model_metadata": model_meta}
    endpoint_of(cand)
    body, items = _merged_values(cand, values)
    slots = _slots(cand)
    image_keys, file_keys = set(image_keys), set(file_keys)
    if image_keys - {s["name"] for s in slots if s["image"]} or \
            file_keys - {s["name"] for s in slots if not s["image"]}:
        raise WiroInput("unsupported wiro upload slot")
    n = (values or {}).get("n")
    if n not in (None, ""):
        count = _number("n", n, integer=True)
        if count > 1 and not any(key in items for key in ("samples", "numberOfImages", "num_images",
                                                              "numImages")):
            raise WiroInput("wiro model has no sample-count parameter for n > 1")
    for slot in slots:
        if slot["required"] and slot["name"] not in (image_keys if slot["image"] else file_keys):
            raise WiroInput(f"wiro {slot['name']} is required")
    return body


def build_request(cand, values, images, files=None):
    """(body, multipart) — None for a JSON run; otherwise (text fields, file parts).
    File parts are (wiro param id, filename, bytes): fileinput sends the file under its
    id plus an empty {id}Url; combine/multi repeat the id per file, expanded slots in
    order. Uploads arrive under public slot names. The sample-URL defaults of file
    inputs are NEVER sent (W6)."""
    images, files = images or {}, files or {}
    body = validate_request(cand, values, images.keys(), files.keys())
    items = _visible(cand)
    slots = _slots(cand)
    uploads = [(s, images[s["name"]], None) for s in slots if s["image"] and images.get(s["name"])]
    uploads += [(s, files[s["name"]][1], files[s["name"]][0]) for s in slots
                if not s["image"] and files.get(s["name"])]
    grouped = {}
    for slot, data, filename in uploads:
        fid = slot["id"]
        if not data:
            raise WiroInput(f"wiro upload {slot['name']} is empty")
        if not filename:
            found = sniff(data)                 # an image slot carries bytes only
            filename = f"{fid}.{found[0]}" if found else fid + ".bin"
        grouped.setdefault(fid, []).append((filename, data))
    if not grouped:
        return body, None
    parts = []
    for fid, entries in grouped.items():
        for filename, data in entries:
            parts.append((fid, filename, data))
        if str(items[fid].get("type") or "").lower() == "fileinput":
            parts.append((fid + "Url", None, b""))   # filename None = a plain (empty) form field
    return body, parts


def request_summary(body, parts=None):
    """No file bytes and no credentials on the job row."""
    def redact(value):
        if isinstance(value, dict):
            return {k: redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value
    summary = redact(dict(body or {}))
    if parts:
        counts = {}
        for fid, _filename, data in parts:
            counts[fid] = counts.get(fid, 0) + (len(data) if data else 0)
        summary["<files>"] = {fid: f"{size} bytes" for fid, size in counts.items()}
    return summary


# ── run / poll / stop envelopes (W7/W8) ───────────────────────────────────────
def run_created(payload):
    """(taskid, socketaccesstoken) of an accepted run, or None to classify the refusal.
    Anything else — malformed success included — is a final RuntimeError."""
    if not isinstance(payload, dict):
        raise WiroInput("wiro run response must be an object")
    if payload.get("result") is not True:
        return None
    taskid, token = payload.get("taskid"), payload.get("socketaccesstoken")
    if not str(taskid or "").strip() or not str(token or "").strip():
        raise WiroInput("wiro run response has no task id or task token")
    return str(taskid), str(token)


def error_code(payload):
    """The first refusal code of a `result:false` envelope; None when absent."""
    if not isinstance(payload, dict) or payload.get("result") is not False:
        return None
    errors = payload.get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        try:
            return int(errors[0].get("code"))
        except (TypeError, ValueError):
            return None
    return None


def error_message(payload):
    errors = (payload or {}).get("errors") if isinstance(payload, dict) else None
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        message = errors[0].get("message")
        if message:
            return str(message)
    return "refused"


def parse_task(task, formats=None, endpoint="image", options=None):
    """task_detail → TaskState. Only task_postprocess_end / task_cancel are terminal:
    task_end comes BEFORE post-processing (outputs not final) and task_error /
    task_error_full are transient vendor states, never a verdict. Success ⇔ terminal
    task_postprocess_end AND pexit == "0"."""
    if not isinstance(task, dict):
        return TaskState(status="UNKNOWN", error="wiro task detail is not an object")
    status = str(task.get("status") or "").strip().lower()
    if status not in _STATUS_ORDER:
        return TaskState(status="UNKNOWN", error=f"unknown wiro task status: {status or 'missing'}")
    try:
        progress = min(99, 5 + _STATUS_ORDER.index(status) * 9)
    except ValueError:
        progress = 0
    state = TaskState(status=status.upper(), progress=progress)
    cost = task.get("totalcost")
    if cost not in (None, ""):
        try:
            cost = float(str(cost))
            if math.isfinite(cost):
                state.credits = cost                # wiro's totalcost is plain USD (G5)
        except (TypeError, ValueError):
            pass
    if status == "task_cancel":
        state.error = "task canceled"
        return state
    if status != "task_postprocess_end":
        return state
    state.progress = 100
    if str(task.get("pexit") or "") != "0":
        detail = str(task.get("debugerror") or "generation failed")[:300]
        state.error = detail
        return state
    downloads, texts = [], []
    outputs = task.get("outputs")
    if isinstance(outputs, list):
        for index, output in enumerate(outputs, 1):
            if not isinstance(output, dict):
                continue
            url = output.get("url")
            if not isinstance(url, str) or not url.strip():
                text = _output_text(output.get("content"))
                if text:
                    texts.append(text)
                continue
            name = str(output.get("name") or "").strip()
            if not name or "/" in name:
                name = urlsplit(url).path.rsplit("/", 1)[-1] or f"output_{index}"
            downloads.append((name, url.strip()))
    if not downloads and not texts and endpoint == "asr":
        debug = str(task.get("debugoutput") or "").strip()
        if debug:
            texts.append(debug)
    if not downloads and not texts:
        state.error = "wiro task finished without outputs"
    state.downloads = downloads
    state.texts = texts                  # transcripts / raw outputs: the adapter files them
    return state


def _output_text(content):
    """A `raw` output's text (wiro: `{raw, segments[], …}` or a plain string)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        for key in ("raw", "text", "answer", "transcript"):
            if isinstance(content.get(key), str) and content[key].strip():
                return content[key].strip()
        segments = content.get("segments")
        if isinstance(segments, list):
            return "".join(str(s.get("content") or s.get("text") or "") for s in segments
                           if isinstance(s, dict) and s.get("type") in (None, "answer")).strip()
    return ""


def task_of_detail(payload):
    """The tasklist[0] object of a Task/Detail answer; raises _TaskVerdict-style via
    WiroInput for a body that refuses the task."""
    if not isinstance(payload, dict) or payload.get("result") is not True:
        raise WiroInput("wiro task detail refused: " + error_message(payload if isinstance(payload, dict) else {}))
    tasks = payload.get("tasklist")
    if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
        raise WiroInput("wiro task detail has no task")
    return tasks[0]
