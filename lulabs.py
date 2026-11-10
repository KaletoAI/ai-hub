"""Pure LU Labs (lu-labs.ai, "LU Cloud") catalog, request and job helpers; no HTTP,
main or adapters imports.

One LU account is one credit wallet behind one base URL: OpenAI-compatible chat
(`/v1/chat/completions`, streaming BY DEFAULT) and the asynchronous create API
(`/v1/jobs`, `/v1/jobs/quote`, `/v1/files`). The create catalog's exact JSON is not
published, so every reader here is defensive: an unexpected row is skipped, never a
crash of the whole discovery.
"""
from __future__ import annotations

import json
import math

from cloudtask import TaskState, sniff as cloudtask_sniff

KIND, VENDOR, URL = "lulabs", "LU Labs", "https://lu-labs.ai/api/inference"
# endpoint → (job kind, ops in preference order). `image` and `video` hold two ops each:
# a request WITH a source image is an edit / animate, without one a generate — one id per
# model, so the OpenAI edits shim and `backend/model` reach the edit too. LU's
# upscale/extend of an own video result, lora-train and studio are not offered.
OPS = {
    "image": ("image", ("generate", "edit")),
    "image-removebg": ("image", ("removebg",)),
    "image-upscale": ("image", ("upscale",)),
    "video": ("video", ("generate", "animate")),
    "video-lipsync": ("video", ("lipsync",)),
    "video-motion": ("video", ("motion",)),
    "speech": ("audio", ("tts",)),          # not `tts`: main's text side owns that key
    "music": ("audio", ("music",)),
}
ENDPOINTS = tuple(OPS)
TOOL_ENDPOINTS = {"removebg": "image-removebg", "upscale": "image-upscale"}
AI_MODELS, FORMATS, RIG_FORMATS = (), (), ()
DYNAMIC_MODELS = True
RIG_ENDPOINT = None
MAX_WAIT_DEFAULT, POLL_INTERVAL_DEFAULT = 1800, 3.0
SUCCESS_STATUS = "SUCCEEDED"
SUBMIT_ATTEMPTS = 3          # same client_request_id: LU returns the job already booked
BACKEND_HINT = ("One LU Labs backend serves chat and media (image, video, speech, music) "
                "from one credit wallet at https://lu-labs.ai/api/inference. API keys always "
                "spend credits. Every job is quoted first and booked with that quote as its cap.")
ENDPOINT_HINT = ("Choose an exact discovered model and operation. Inputs are uploaded to LU "
                 "first; model-specific defaults accept a JSON object of LU job params.")
OPTION_DEFAULTS = {"options_json": ""}
OPTION_FIELDS = [{"key": "options_json", "label": "Model-specific defaults (JSON)", "type": "text",
                  "textarea": True,
                  "hint": "A JSON object merged into the job's params; op, model, credentials, "
                          "max_credits, client_request_id and *_path keys are refused."}]
_RESERVED = {"op", "model", "kind", "max_credits", "client_request_id", "api_key", "apikey",
             "authorization", "options_json", "source_path", "mask_path", "audio_path",
             "video_path", "image_paths", "source_url"}
# Upload slot → (role on POST /files, job param that receives the returned path)
SLOT_ROLES = {"source_image": ("source", "source_path"), "mask_image": ("mask", "mask_path"),
              "audio": ("audio", "audio_path"), "video": ("video", "video_path")}
_TERMINAL_FAIL = {"failed", "canceled", "cancelled", "error", "expired"}


class LuLabsInput(RuntimeError):
    """Invalid LU media input is final; it must never initiate a paid submit."""


INPUT_ERROR = LuLabsInput


def endpoint_of(cand):
    endpoint = (cand.get(KIND) or {}).get("endpoint") or "image"
    if endpoint not in OPS:
        raise LuLabsInput("invalid LU Labs media endpoint")
    return endpoint


def generation_type(endpoint):
    if endpoint not in OPS:
        raise LuLabsInput("invalid LU Labs media endpoint")
    return {"speech": "tts", "music": "audio"}.get(endpoint, OPS[endpoint][0])


def task_for(endpoint, previous_task=""):
    return {"image": "text2img", "image-removebg": "img2img", "image-upscale": "img2img",
            "video": "text2vid", "video-lipsync": "img2vid", "video-motion": "img2vid",
            "speech": "text2audio", "music": "text2audio"}[endpoint_of({KIND: {"endpoint": endpoint}})]


def candidate_for(backend, model, endpoint, model_meta=None):
    cand = {"backend": backend, "model": model, "task": task_for(endpoint),
            KIND: {"endpoint": endpoint, "options": dict(OPTION_DEFAULTS)}}
    if model_meta is not None:
        if not isinstance(model_meta, dict):
            raise LuLabsInput("LU Labs model metadata must be an object")
        cand["model_metadata"] = model_meta
    return cand


def default_candidate(backend):
    return candidate_for(backend, "", "image")


def _advanced(raw):
    if raw in (None, ""):
        return {}

    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite")
        return number
    try:
        obj = json.loads(raw, parse_float=finite, parse_constant=finite)
    except (ValueError, TypeError) as exc:
        raise LuLabsInput("model-specific defaults must be valid finite JSON") from exc
    if not isinstance(obj, dict):
        raise LuLabsInput("model-specific defaults must be a JSON object")
    if any(str(key).lower() in _RESERVED for key in obj):
        raise LuLabsInput("model-specific defaults cannot override op, model, paths, "
                          "price caps or credentials")
    return obj


def options_of(cand):
    stored = (cand.get(KIND) or {}).get("options") or {}
    if not isinstance(stored, dict):
        raise LuLabsInput("LU Labs options must be an object")
    result = {key: stored.get(key, value) for key, value in OPTION_DEFAULTS.items()}
    _advanced(result["options_json"])
    return result


def read_options(form):
    return options_of({KIND: {"options": {"options_json": form.get("opt__options_json") or ""}}})


# ── catalog (defensive) ───────────────────────────────────────────────────────
def _rows(payload):
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if rows is None and isinstance(payload, dict):
        rows = payload.get("models")
    return rows if isinstance(rows, list) else None


def _ops(row):
    ops = row.get("ops")
    if isinstance(ops, str):
        ops = [part.strip() for part in ops.split(",")]
    if isinstance(ops, dict):
        ops = list(ops)
    if not isinstance(ops, list):
        op = row.get("op")
        ops = [op] if isinstance(op, str) else ["generate"]
    return {str(op).strip().lower() for op in ops if isinstance(op, str) and op.strip()}


def parse_create_catalog(payload):
    """`GET /v1/models?type=create` → {endpoint: {model id: row}}. A row needs an id
    (`id`/`model`) and a `kind` (image|video|audio); it yields one entry per op it
    lists that the gateway supports. A `tools` block naming removebg/upscale publishes
    model-less ids `removebg` / `upscale`. Malformed rows are skipped, never fatal —
    but a payload without any list is (the snapshot would be silently empty)."""
    rows = _rows(payload)
    if rows is None:
        raise ValueError("LU Labs create catalog has no model list")
    catalogs = {endpoint: {} for endpoint in ENDPOINTS}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("type") not in (None, "create"):          # ?type=all mixes chat rows in
            continue
        mid = row.get("id") or row.get("model")
        kind = str(row.get("kind") or "").strip().lower()
        if not isinstance(mid, str) or not mid.strip() or kind not in ("image", "video", "audio"):
            continue
        ops = _ops(row)
        for endpoint, (ekind, eops) in OPS.items():
            supported = [op for op in eops if op in ops]
            if ekind == kind and supported:
                catalogs[endpoint][mid.strip()] = {**row, "_ops": supported}
    tools = payload.get("tools") if isinstance(payload, dict) else None
    names = set(tools) if isinstance(tools, dict) else {
        str((t or {}).get("id") or (t or {}).get("op") or (t or {}).get("name") or "")
        for t in tools if isinstance(t, dict)} if isinstance(tools, list) else set()
    for name, endpoint in TOOL_ENDPOINTS.items():
        if name in names:
            meta = tools.get(name) if isinstance(tools, dict) else next(
                t for t in tools if isinstance(t, dict) and name in (t.get("id"), t.get("op"), t.get("name")))
            catalogs[endpoint][name] = {"id": name, "kind": "image", "tool": True,
                                        "_ops": list(OPS[endpoint][1]),
                                        "credits": meta if not isinstance(meta, dict) else
                                        meta.get("credits", meta)}
    return catalogs


def credits_from(meta):
    """Smallest credit figure a row publishes (`credits` number or nested object), or None."""
    found = []

    def walk(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            if math.isfinite(value) and value > 0:
                found.append(float(value))
        elif isinstance(value, str):
            try:
                number = float(value)
            except ValueError:
                return
            if math.isfinite(number) and number > 0:
                found.append(number)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk((meta or {}).get("credits"))
    return min(found) if found else None


def _durations(meta):
    durations = (meta or {}).get("durations")
    out = []
    if isinstance(durations, list):
        for value in durations:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out.append(value)
            elif isinstance(value, dict) and isinstance(value.get("seconds"), (int, float)):
                out.append(value["seconds"])
    return out


# ── fields / validation ───────────────────────────────────────────────────────
_PROMPT_OPS = {"generate", "edit", "animate", "tts", "music"}
_OPENAI_GENERIC = ("quality", "style", "background", "output_format", "response_format", "user")


def model_ops(cand):
    """The ops this model offers on its endpoint (catalog `_ops`); without metadata the
    endpoint's full list, so a stored alias still renders before discovery."""
    endpoint = endpoint_of(cand)
    ops = (cand.get("model_metadata") or {}).get("_ops")
    allowed = OPS[endpoint][1]
    picked = [op for op in ops if op in allowed] if isinstance(ops, list) else list(allowed)
    return picked or list(allowed)


def op_for(cand, image_keys=(), file_keys=()):
    """generate vs edit / animate: a source image makes it the image-taking op."""
    ops = model_ops(cand)
    if len(ops) == 1:
        return ops[0]
    return ops[1] if "source_image" in set(image_keys or ()) else ops[0]


def public_fields(cand):
    """(params, images, files) for the endpoint — the union over the model's ops, the
    source image required only when every op needs one."""
    endpoint = endpoint_of(cand)
    kind = OPS[endpoint][0]
    ops = set(model_ops(cand))
    meta = cand.get("model_metadata") or {}
    params = []
    if ops & (_PROMPT_OPS | {"lipsync"}):
        params.append({"name": "prompt", "type": "string", "default": "",
                       **({"required": True} if ops <= _PROMPT_OPS else {})})
    if kind == "image" and ops & {"generate", "edit"}:
        params += [{"name": "negative_prompt", "type": "string", "default": ""},
                   {"name": "size", "type": "string", "hint": "WIDTHxHEIGHT"},
                   {"name": "seed", "type": "int"}]
    if "upscale" in ops:
        params.append({"name": "target_resolution", "type": "string",
                       "choices": ["2k", "4k", "8k"], "default": "2k"})
    if ops & {"generate", "animate"} and kind == "video":
        durations = _durations(meta)
        field = {"name": "duration", "type": "int"}
        if durations:
            field.update(choices=[str(d) for d in durations], default=durations[0])
        params += [field, {"name": "seed", "type": "int"}]
    if "music" in ops:
        params.append({"name": "duration", "type": "int", "hint": "seconds"})
    needs_image = {"edit", "removebg", "upscale", "animate", "motion", "lipsync"}
    images = []
    if ops & needs_image:
        required = ops <= (needs_image - {"lipsync"})
        images.append({"name": "source_image", "required": required,
                       "on_empty": "required" if required else "none"})
    if "edit" in ops:
        images.append({"name": "mask_image", "required": False, "on_empty": "none"})
    files = []
    if ops & {"lipsync", "tts"}:
        files.append({"name": "audio", "required": "lipsync" in ops,
                      "accept": ["wav", "mp3", "flac", "ogg", "m4a"]})
    if ops & {"lipsync", "motion"}:
        files.append({"name": "video", "required": "motion" in ops, "accept": ["mp4", "webm", "mov"]})
    return params, images, files


def slot_names(cand):
    return [image["name"] for image in public_fields(cand)[1]]


def _int(key, value):
    try:
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or number != int(number):
            raise ValueError
        return int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LuLabsInput(f"LU Labs {key} must be an integer") from exc


def validate_request(cand, values, image_keys, file_keys, model_meta=None):
    """Everything checkable before any upload, quote or paid POST → the job params
    (without paths, op included). Unknown keys, n > 1, a duration the model does not
    publish and missing required inputs are the client's mistakes (400). Blank values
    (the gateway always passes prompt / negative_prompt) are "not set"."""
    if model_meta is not None:
        cand = {**cand, "model_metadata": model_meta}
    endpoint = endpoint_of(cand)
    kind = OPS[endpoint][0]
    image_keys, file_keys = set(image_keys or ()), set(file_keys or ())
    op = op_for(cand, image_keys, file_keys)
    params_spec, images_spec, files_spec = public_fields(cand)
    allowed = {field["name"] for field in params_spec} | {"n"}
    values = {k: v for k, v in (values or {}).items() if v not in (None, "")}
    for key in _OPENAI_GENERIC:          # what the OpenAI Images shims pass for every client
        values.pop(key, None)
    unknown = set(values) - allowed
    if unknown:
        raise LuLabsInput("unsupported LU Labs parameter: " + sorted(unknown)[0])
    if "n" in values and _int("n", values["n"]) != 1:
        raise LuLabsInput("LU Labs creates one result per job (n must be 1)")
    if image_keys - {i["name"] for i in images_spec} or file_keys - {f["name"] for f in files_spec}:
        raise LuLabsInput("unsupported LU Labs upload slot")
    for spec, keys in ((images_spec, image_keys), (files_spec, file_keys)):
        for slot in spec:
            if slot["required"] and slot["name"] not in keys:
                raise LuLabsInput(f"LU Labs {endpoint} requires {slot['name']}")
    if op == "lipsync" and not ({"source_image", "video"} & (image_keys | file_keys)):
        raise LuLabsInput("LU Labs lipsync needs source_image or video")
    if "mask_image" in image_keys and op != "edit":
        raise LuLabsInput("LU Labs mask_image needs a source_image to edit")
    if op in _PROMPT_OPS and "prompt" not in values:
        raise LuLabsInput(f"LU Labs {endpoint} requires prompt")
    params = {**_advanced(options_of(cand)["options_json"])}
    for key, value in values.items():
        if key in ("n", "prompt"):
            continue
        if key in ("seed", "duration"):
            value = _int(key, value)
        params[key] = value
    durations = _durations(cand.get("model_metadata"))
    if kind == "video" and op in ("generate", "animate") and durations:
        if "duration" not in params:
            params["duration"] = durations[0]
        elif params["duration"] not in durations:
            raise LuLabsInput(f"LU Labs duration must be one of {durations}")
    if op == "upscale":
        params.setdefault("target_resolution", "2k")
        if params["target_resolution"] not in ("2k", "4k", "8k"):
            raise LuLabsInput("LU Labs target_resolution must be 2k, 4k or 8k")
    size = params.get("size")
    if size is not None:
        parts = str(size).lower().split("x")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            raise LuLabsInput("LU Labs size must be WIDTHxHEIGHT")
    params["op"] = op
    return params


def build_job(cand, params, prompt, paths, client_request_id, max_credits=None):
    """The `POST /v1/jobs` body from validate_request's params, the prompt and the
    uploaded paths (job param → LU file path). The same body, client_request_id
    included, is what a re-POST after an uncertain answer sends (adapter rule)."""
    kind = OPS[endpoint_of(cand)][0]
    params = {**params, **paths, "client_request_id": client_request_id}
    if max_credits is not None:
        params["max_credits"] = max_credits
    body = {"kind": kind, "params": params}
    model = cand.get("model")
    if not (cand.get("model_metadata") or {}).get("tool"):
        if not isinstance(model, str) or not model.strip():
            raise LuLabsInput("an exact LU Labs model id is required")
        body["model"] = model
    if prompt not in (None, ""):
        body["prompt"] = str(prompt)
    return body


def request_summary(body):
    """No credentials, ids only; uploaded inputs appear as their LU paths."""
    summary = {k: v for k, v in (body or {}).items() if k != "params"}
    params = dict((body or {}).get("params") or {})
    params.pop("client_request_id", None)
    summary["params"] = params
    return summary


# ── job envelopes ─────────────────────────────────────────────────────────────
def job_id_of(payload):
    """The booked job's id out of a `POST /jobs` answer (`{"object":"job","id":…}`,
    or a `job`/`data` wrapper)."""
    if isinstance(payload, dict):
        for candidate in (payload, payload.get("job"), payload.get("data")):
            if isinstance(candidate, dict) and isinstance(candidate.get("id"), (str, int)) \
                    and str(candidate["id"]).strip():
                return str(candidate["id"]).strip()
    return ""


def quote_credits(payload):
    """The quoted credit figure (`credits`, also as a numeric string or under `quote`/
    `data`), or None — the adapter then refuses to book: an uncapped job is exactly
    what the quote exists to prevent."""
    if isinstance(payload, dict):
        for holder in (payload, payload.get("quote"), payload.get("data")):
            if not isinstance(holder, dict):
                continue
            value = holder.get("credits")
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    continue
            if isinstance(value, (int, float)) and not isinstance(value, bool) \
                    and math.isfinite(value) and value >= 0:
                return int(value) if float(value).is_integer() else value
    return None


def error_code(payload):
    if isinstance(payload, dict):
        code = payload.get("code")
        error = payload.get("error")
        if code is None and isinstance(error, dict):
            code = error.get("code")
        return str(code) if code not in (None, "") else ""
    return ""


def error_text(payload):
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            error = error.get("message")
        if error:
            return str(error)[:300]
    return ""


def _result_urls(job):
    urls = []

    def take(value):
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            urls.append(value)
        elif isinstance(value, dict):
            for key in ("url", "urls", "file", "files", "output", "outputs"):
                take(value.get(key))
        elif isinstance(value, list):
            for item in value:
                take(item)
    for key in ("result", "output", "outputs", "data", "url", "urls"):
        take(job.get(key))
    seen, out = set(), []
    for url in urls:
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def parse_job(job, formats=None, endpoint="image", options=None):
    """`GET /jobs/{id}` → TaskState. succeeded → downloads = the result URLs the job
    names (the adapter prefers `/jobs/{id}/content`); failed/canceled → error."""
    if not isinstance(job, dict):
        return TaskState(status="UNKNOWN", error="LU Labs job is not an object")
    if isinstance(job.get("job"), dict):
        job = job["job"]
    status = str(job.get("status") or "").strip().lower()
    state = TaskState(status=status.upper() or "UNKNOWN")
    credits = job.get("credits")
    if isinstance(credits, (int, float)) and not isinstance(credits, bool) and math.isfinite(credits):
        state.credits = credits
    progress = job.get("progress")
    if isinstance(progress, (int, float)) and not isinstance(progress, bool) and math.isfinite(progress):
        state.progress = int(progress * 100 if progress <= 1 else progress)
    if status in _TERMINAL_FAIL:
        state.error = error_text(job) or f"job {status}"
        state.credits = None                      # failed and canceled jobs are refunded
        return state
    if status in ("succeeded", "completed", "done", "success"):
        state.status, state.progress = SUCCESS_STATUS, 100
        state.downloads = [(f"result-{i}", url) for i, url in enumerate(_result_urls(job), 1)]
    return state


# ── artifact sniffing: cloudtask.sniff (shared with wiro uploads) ──────────────
sniff = cloudtask_sniff


parse_task = parse_job          # the CloudTaskAdapter._poll seam name
