"""Pure NanoGPT media request/result helpers; HTTP and artifact bytes live in adapters.

3D uses the shared video transport. Its asset fields are published by NanoGPT's
frontend model3d utilities, not inferred from filenames or hardcoded model IDs.
"""
from __future__ import annotations

import base64
import json
import math
import mimetypes
from urllib.parse import urlsplit, unquote

from cloudtask import TaskState

KIND, VENDOR, URL = "nanogpt", "NanoGPT", "https://api.nano-gpt.com/api"
ENDPOINTS = ("images", "video", "3d")
CATALOG_PATHS = {"images": "/v1/images/models", "video": "/v1/video-models", "3d": "/v1/3d-models"}
SUBMIT_PATHS = {"images": "/v1/images/generations", "video": "/generate-video", "3d": "/generate-video"}
AI_MODELS, FORMATS, RIG_FORMATS = (), (), ()
DYNAMIC_MODELS = True
RIG_ENDPOINT = None
MAX_WAIT_DEFAULT, POLL_INTERVAL_DEFAULT, SUCCESS_STATUS = 1800, 5.0, "COMPLETED"
BACKEND_HINT = ("One NanoGPT backend serves text, TTS, embeddings, images, video and 3D "
                "at the same /api base URL. Generation is paid; an accepted or uncertain "
                "submit is never re-created. Gateway cancellation does not stop the vendor.")
ENDPOINT_HINT = ("Select images, video or 3D and an exact model ID from that host's catalog. "
                 "Set only controls supported by the model; blank leaves its provider default. "
                 "Model-specific defaults accept a JSON object, without model or credentials.")
_INT_FIELDS = {"seed", "n", "steps", "width", "height"}
OPTION_DEFAULTS = {k: None for k in ("n", "size", "resolution", "aspect_ratio", "quality", "output_format",
                                    "seed", "duration", "seconds", "steps", "cfg")}
OPTION_DEFAULTS["options_json"] = ""
OPTION_FIELDS = [{"key": k, "label": k.replace("_", " "), "type": "int" if k in _INT_FIELDS else "text"}
                 for k in OPTION_DEFAULTS]
OPTION_FIELDS[-1].update(label="Model-specific defaults (JSON)", textarea=True,
                        hint="A JSON object of model-supported controls; no model, credentials or endpoint overrides.")
_RESERVED = {"model", "endpoint", "provider", "api_key", "authorization", "x-api-key", "headers", "options_json"}
_IMAGE_SLOTS = ["input_image"] + [f"input_image_{i}" for i in range(2, 17)]


class NanoGPTInput(RuntimeError):
    """An invalid or incomplete media request/result is final, never another paid submit."""


INPUT_ERROR = NanoGPTInput


def endpoint_of(cand):
    endpoint = (cand.get(KIND) or {}).get("endpoint") or "images"
    if endpoint not in ENDPOINTS:
        raise NanoGPTInput("NanoGPT endpoint must be images, video or 3d")
    return endpoint


def task_for(endpoint, previous_task=""):
    if endpoint not in ENDPOINTS:
        raise NanoGPTInput("invalid NanoGPT media endpoint")
    image = str(previous_task).startswith("img")
    return ("img2" if image else "text2") + {"images": "img", "video": "vid", "3d": "mesh"}[endpoint]


def candidate_for(backend: str, model: str, endpoint: str, model_meta: dict | None = None) -> dict:
    """Make an ephemeral catalog candidate; no stored alias or model enum is required."""
    cand = {"backend": backend, "model": model, "task": task_for(endpoint),
            KIND: {"endpoint": endpoint, "options": dict(OPTION_DEFAULTS)}}
    if model_meta is not None:
        if not isinstance(model_meta, dict):
            raise NanoGPTInput("NanoGPT model metadata must be an object")
        cand["model_metadata"] = model_meta
    return cand


def default_candidate(backend):
    return candidate_for(backend, "", "images")


def _number(key, value, integer=False):
    try:
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or (integer and number != int(number)):
            raise ValueError
        return int(number) if integer else number
    except (TypeError, ValueError, OverflowError) as exc:
        raise NanoGPTInput(f"NanoGPT {key} must be a finite {'integer' if integer else 'number'}") from exc


def _advanced(raw):
    if raw in (None, ""):
        return {}

    def finite_number(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON number")
        return number

    try:
        obj = json.loads(raw, parse_float=finite_number, parse_constant=finite_number)
    except (ValueError, TypeError) as exc:
        raise NanoGPTInput("model-specific defaults must be valid finite JSON") from exc
    if not isinstance(obj, dict):
        raise NanoGPTInput("model-specific defaults must be a JSON object")
    if any(str(key).lower() in _RESERVED for key in obj):
        raise NanoGPTInput("model-specific defaults cannot override model, endpoint or credentials")
    return obj


def options_of(cand):
    stored = (cand.get(KIND) or {}).get("options") or {}
    result = {k: stored.get(k, value) for k, value in OPTION_DEFAULTS.items()}
    _advanced(result["options_json"])
    for key in _INT_FIELDS | {"cfg"}:
        if result.get(key) not in (None, ""):
            result[key] = _number(key, result[key], key in _INT_FIELDS)
    return result


def read_options(form):
    candidate = {KIND: {"options": {k: form.get(f"opt__{k}") or None for k in OPTION_DEFAULTS}}}
    result = options_of(candidate)
    for key in ("n", "steps"):
        if result.get(key) is not None and result[key] <= 0:
            raise NanoGPTInput(f"NanoGPT {key} must be positive")
    return result


def _metadata(cand):
    meta = cand.get("model_metadata")
    return meta if isinstance(meta, dict) else None


def _input_supported(cand, modality):
    meta = _metadata(cand)
    if meta is None:
        return modality in {"text", "image"} or endpoint_of(cand) == "video"
    ep = endpoint_of(cand)
    caps = meta.get("capabilities") or {}
    flag = {"text": {"images": "text_to_image", "video": "text_to_video", "3d": "text_to_3d"},
            "image": {"images": "image_to_image", "video": "image_to_video", "3d": "image_to_3d"},
            "video": {"video": "video_to_video"}, "audio": {"video": "audio_input"}}[modality].get(ep)
    if flag in caps:
        return caps[flag] is True
    modalities = (meta.get("architecture") or {}).get("input_modalities")
    if isinstance(modalities, list):
        return modality in modalities
    supported = meta.get("supported_parameters") or {}
    if modality == "image":
        return isinstance(supported.get("max_input_images"), int) and supported["max_input_images"] > 0
    # Images use a prompt by default; absence of catalog input information does
    # not invent video/audio/reference capabilities.
    return modality == "text" and ep == "images"


def _parameter_specs(cand):
    """Normalize flat catalog arrays and documented parameter descriptors."""
    meta = _metadata(cand)
    if meta is None:
        return {}
    supported = meta.get("supported_parameters") or {}
    aliases = {"resolutions": "resolution", "sizes": "size", "aspect_ratios": "aspect_ratio",
               "qualities": "quality", "output_formats": "output_format", "durations": "duration"}
    descriptors = {k: v for k, v in supported.items()
                   if k not in {"parameters", "defaults", "prompt_length", "input_image_constraints"}
                   and not k.startswith(("max_", "min_"))}
    descriptors.update(supported.get("parameters") or {})
    defaults = supported.get("defaults") or {}
    specs = {}
    for key, raw in descriptors.items():
        key = aliases.get(key, key)
        if key.lower() in _RESERVED or raw is None or raw is False:
            continue
        if isinstance(raw, list):
            raw = {"enum": raw}
        elif raw is True:
            raw = {}
        if not isinstance(raw, dict):
            continue
        choices = raw.get("enum", raw.get("options", raw.get("values")))
        if isinstance(choices, list):
            choices = [v.get("value") if isinstance(v, dict) else v for v in choices]
            choices = [v for v in choices if isinstance(v, (str, int, float, bool))]
        typ = raw.get("type")
        if typ in {"array", "object", "json"} or any(k in raw for k in
                ("item", "items", "properties", "max_items", "min_items")):
            # Structured native controls remain JSON, not guessed scalar text.
            continue
        if typ is None and (raw.get("range") or any(k in raw for k in ("min", "max", "minimum", "maximum"))):
            typ = "number"
        bounds = raw.get("range") or {}
        if isinstance(bounds, list) and len(bounds) == 2:
            bounds = {"min": bounds[0], "max": bounds[1]}
        if not isinstance(bounds, dict):
            bounds = {}
        integer = typ in {"integer", "int"} or key in _INT_FIELDS | {"num_inference_steps"}
        if typ == "number" and raw.get("step") == 1:
            integer = True
        field = {"name": key, "type": "bool" if typ in {"switch", "boolean", "bool"} else
                 "int" if integer else "float" if typ in {"number", "float", "range"} else "string"}
        if isinstance(choices, list):
            field["choices"] = choices
        for dest, names in (("min", ("min", "minimum")), ("max", ("max", "maximum")), ("step", ("step",))):
            value = next((source[name] for source in (raw, bounds) for name in names if name in source), None)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                field[dest] = value
        if "default" in raw or key in defaults:
            field["default"] = raw.get("default", defaults.get(key))
        specs[key] = field
    if endpoint_of(cand) == "images":
        limit = supported.get("max_output_images", supported.get("max_images"))
        if "n" in specs or isinstance(limit, int) and not isinstance(limit, bool):
            count = specs.setdefault("n", {"name": "n", "type": "int"})
            count["min"] = max(1, count.get("min", 1))
            if isinstance(limit, int) and not isinstance(limit, bool):
                count["max"] = min(limit, count.get("max", limit))
    return specs


def slot_names(cand):
    if not _input_supported(cand, "image"):
        return []
    if _metadata(cand) is None:
        return _IMAGE_SLOTS.copy() if endpoint_of(cand) == "images" else ["input_image"]
    supported = _metadata(cand).get("supported_parameters") or {}
    limit = supported.get("max_input_images", (supported.get("input_image_constraints") or {}).get("max_items", 1))
    if not isinstance(limit, int) or isinstance(limit, bool):
        limit = 1
    return ["input_image" if i == 1 else f"input_image_{i}" for i in range(1, max(0, limit) + 1)]


def _file_fields(cand):
    if endpoint_of(cand) != "video":
        return []
    return [{"name": name, "required": False, "accept": accept}
            for modality, name, accept in (
                ("video", "input_video", ["mp4", "webm", "mov"]),
                ("audio", "input_audio", ["wav", "mp3", "flac", "ogg", "m4a"]))
            if _input_supported(cand, modality)]


def public_fields(cand):
    ep, opts = endpoint_of(cand), options_of(cand)
    if _metadata(cand) is None:
        names = (["size", "resolution", "aspect_ratio", "quality", "output_format", "seed", "n", "steps", "cfg"]
                 if ep == "images" else
                 ["duration", "seconds", "resolution", "aspect_ratio", "seed"] if ep == "video" else ["seed"])
        params = [{"name": "prompt", "type": "string", "default": ""},
                  {"name": "negative_prompt", "type": "string", "default": ""}]
        params += [{"name": k, "type": "int" if k in _INT_FIELDS else "float" if k == "cfg" else "string",
                    **({"default": opts[k]} if opts[k] not in (None, "") else {})} for k in names]
    else:
        specs = _parameter_specs(cand)
        params = [{"name": "prompt", "type": "string", "default": ""}] if _input_supported(cand, "text") else []
        pinned = {**_advanced(opts.pop("options_json")), **{k: v for k, v in opts.items() if v not in (None, "")}}
        for key, field in specs.items():
            if key == "prompt" and params:
                params[0].update(field)
                continue
            field = dict(field)
            default_key = {"guidance_scale": "cfg", "num_inference_steps": "steps"}.get(key)
            default = pinned.get(key, pinned.get(default_key))
            if default not in (None, ""):
                field["default"] = default
            params.append(field)
    images = [{"name": name, "required": False, "on_empty": "none"} for name in slot_names(cand)]
    return params, images, _file_fields(cand)


def _values(cand, values):
    ep = endpoint_of(cand)
    model = cand.get("model")
    if not isinstance(model, str) or not model.strip():
        raise NanoGPTInput("an exact NanoGPT model ID is required")
    opts = options_of(cand)
    body = _advanced(opts.pop("options_json"))
    body.update({k: v for k, v in opts.items() if v not in (None, "")})
    extra = values.get("extra") or {}
    if not isinstance(extra, dict):
        raise NanoGPTInput("NanoGPT extra parameters must be an object")
    client = {**extra, **{k: v for k, v in values.items() if k != "extra" and v not in (None, "")}}
    for key in client:
        if key != "model" and str(key).lower() in _RESERVED:
            raise NanoGPTInput(f"NanoGPT client cannot override {key}")
    body.update({k: v for k, v in client.items() if k != "model"})
    for short, native in (("cfg", "guidance_scale"), ("steps", "num_inference_steps")):
        if short in body:
            if short in client and native not in client or native not in body:
                body[native] = body[short]
            del body[short]
    for key in _INT_FIELDS | {"num_inference_steps"}:
        if key in body:
            body[key] = _number(key, body[key], True)
            if key not in {"seed"} and body[key] <= 0:
                raise NanoGPTInput(f"NanoGPT {key} must be positive")
    if "guidance_scale" in body:
        body["guidance_scale"] = _number("guidance_scale", body["guidance_scale"])
    for key in ("prompt", "negative_prompt"):
        if key in body and not isinstance(body[key], str):
            raise NanoGPTInput(f"NanoGPT {key} must be text")
    if ep == "images":
        # Width/height are gateway conveniences, not vendor-native resolution fields.
        explicit = client.get("size") or client.get("resolution")
        dims = (client.get("width"), client.get("height"))
        size = explicit or (f"{body['width']}x{body['height']}" if all(v is not None for v in dims) else None)
        size = size or body.get("size") or body.get("resolution")
        if size:
            body["size"] = str(size)
        body.pop("resolution", None)
        body.pop("width", None)
        body.pop("height", None)
        body.setdefault("n", 1)
        body["response_format"] = "url"
    else:
        if body.pop("n", 1) != 1:
            raise NanoGPTInput("NanoGPT video/3D supports one generation per request")
        for key in ("duration", "seconds"):
            if key in body:
                value = _number(key, body[key])
                if value <= 0:
                    raise NanoGPTInput(f"NanoGPT {key} must be positive")
                body[key] = str(int(value) if value.is_integer() else value)
    body["model"] = model.strip()
    return body


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise NanoGPTInput("NanoGPT parameters must be finite JSON")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_json(item)


def _validate_controls(cand, values, body):
    specs = _parameter_specs(cand)
    for key, value in list(body.items()):
        short = {"guidance_scale": "cfg", "num_inference_steps": "steps"}.get(key)
        field = specs.get(key) or specs.get(short)
        if key == "size" and endpoint_of(cand) == "images":
            field = field or specs.get("resolution")
            client = {**(values.get("extra") or {}), **{k: v for k, v in values.items()
                                                     if k != "extra" and v not in (None, "")}}
            opts = options_of(cand)
            pinned = {**_advanced(opts.pop("options_json")), **{k: v for k, v in opts.items() if v not in (None, "")}}
            explicit_size = client.get("size") if client.get("size") or client.get("resolution") else pinned.get("size")
            # OpenAI size=auto is a documented provider-selection override, even
            # when a model lists only concrete resolutions; native resolution is not.
            if value == "auto" and explicit_size == "auto":
                continue
        if field is None:
            continue
        typ = field["type"]
        if typ in {"int", "float"}:
            value = _number(key, value, typ == "int")
            for bound, comparison in (("min", value < field.get("min", -math.inf)),
                                      ("max", value > field.get("max", math.inf))):
                if comparison:
                    raise NanoGPTInput(f"NanoGPT {key} is outside the model's {bound} bound")
        elif typ == "bool":
            if not isinstance(value, bool):
                raise NanoGPTInput(f"NanoGPT {key} must be a boolean")
        elif "choices" not in field and not isinstance(value, str):
            raise NanoGPTInput(f"NanoGPT {key} must be text")
        if "choices" in field and not any(value == choice or str(value) == str(choice) for choice in field["choices"]):
            raise NanoGPTInput(f"NanoGPT {key} must be one of the model's advertised choices")
        body[key] = str(value) if key in {"duration", "seconds"} else value


def validate_request(cand, values, image_keys, file_keys, model_meta=None):
    if model_meta is not None:
        cand = {**cand, "model_metadata": model_meta}
    body = _values(cand, values)
    _finite_json(body)
    image_keys, file_keys = set(image_keys), set(file_keys)
    unknown_images = image_keys - set(slot_names(cand))
    unknown_files = file_keys - {field["name"] for field in _file_fields(cand)}
    if unknown_images or unknown_files:
        raise NanoGPTInput("unsupported NanoGPT upload slot")
    if _metadata(cand) is not None:
        _validate_controls(cand, values, body)
        urls = body.get("imageDataUrls", body.get("imageUrls", []))
        if not isinstance(urls, list):
            raise NanoGPTInput("NanoGPT image URL inputs must be a list")
        image_count = len(image_keys) + len(urls) + int(bool(body.get("imageDataUrl") or body.get("imageUrl")))
        if image_count > len(slot_names(cand)):
            raise NanoGPTInput(f"NanoGPT model accepts at most {len(slot_names(cand))} reference images")
        for modality in ("video", "audio"):
            if (body.get(modality + "Url") or body.get(modality + "DataUrl")) and not _input_supported(cand, modality):
                raise NanoGPTInput(f"NanoGPT model does not support {modality} input")
    return body


def _data_url(data, mime):
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _image_mime(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if data[4:8] == b"ftyp" and data[8:12] in (b"avif", b"avis"):
        return "image/avif"
    raise NanoGPTInput("NanoGPT reference input must contain image bytes")


def build_request(cand, values, images, files=None):
    files = files or {}
    body = validate_request(cand, values, images.keys(), files.keys())
    ep = endpoint_of(cand)
    image_values = [_data_url(images[name], _image_mime(images[name]))
                    for name in slot_names(cand) if images.get(name)]
    if image_values:
        key = "imageDataUrls" if len(image_values) > 1 else "imageDataUrl"
        if key in body or "imageUrl" in body or "imageDataUrl" in body or "imageDataUrls" in body:
            raise NanoGPTInput("choose uploaded images or URL/data-URL parameters, not both")
        body[key] = image_values if len(image_values) > 1 else image_values[0]
    for name, (filename, data) in files.items():
        key = {"input_video": "videoDataUrl", "input_audio": "audioDataUrl"}[name]
        if key in body or key.replace("DataUrl", "Url") in body:
            raise NanoGPTInput("choose an uploaded file or its URL parameter, not both")
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body[key] = _data_url(data, mime)
    return body


def request_summary(body):
    def redact(value):
        if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
            encoded = value.split(",", 1)[1]
            return f"<media:{len(encoded.rstrip('=')) * 3 // 4} bytes>"
        if isinstance(value, dict):
            return {k: redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value
    return redact(body)


def status_request(endpoint, task_id):
    if endpoint == "images":
        return "POST", "/check-midjourney-status", {"task_id": task_id}
    return "GET", "/video/status", {"requestId": task_id}


def task_id_of(payload, endpoint):
    if not isinstance(payload, dict):
        raise NanoGPTInput("NanoGPT submit response must be an object")
    for key in (("task_id", "taskId") if endpoint == "images" else ("runId", "id")):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _url(value):
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str) or not value:
        raise NanoGPTInput("NanoGPT asset URL is missing")
    parts = urlsplit(value)
    if parts.scheme not in {"https", "http"} or not parts.hostname or parts.username or parts.password:
        raise NanoGPTInput("NanoGPT asset URL must be HTTP(S) without credentials")
    return value


def _asset(value, default):
    url = _url(value)
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    return name if "." in name and "/" not in name else default, url


def image_outputs(payload):
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        raise NanoGPTInput("NanoGPT image result contains no outputs")
    result = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise NanoGPTInput("NanoGPT image result contains an invalid output")
        encoded, url = entry.get("b64_json"), entry.get("url")
        if bool(encoded) == bool(url):
            raise NanoGPTInput("NanoGPT image output must contain exactly one URL or base64 image")
        if encoded:
            if not isinstance(encoded, str):
                raise NanoGPTInput("NanoGPT image output must contain base64 text")
            result.append((f"image_{i}.png", encoded, True))
        else:
            name, url = _asset(url, f"image_{i}.png")
            result.append((name, url, False))
    return result


def _mesh_assets(output):
    assets = []
    def add(value, name):
        if value:
            item = _asset(value, name)
            if item[1] not in {url for _, url in assets}:
                assets.append(item)
    for ext, value in (output.get("model_urls") or {}).items():
        if ext in {"glb", "fbx", "obj", "usdz"}:
            add(value, f"model.{ext}")
    for key in ("model_url", "model_glb", "video_url", "rigged_character_glb", "rigged_character_fbx",
                "animation_glb", "animation_fbx"):
        add(output.get(key), key + (".fbx" if key.endswith("fbx") else ".glb"))
    for key, value in (output.get("basic_animations") or {}).items():
        if key in {"walking_glb", "walking_fbx", "running_glb", "running_fbx"}:
            add(value, key + (".fbx" if key.endswith("fbx") else ".glb"))
    for i, textures in enumerate(output.get("texture_urls") or []):
        for key in ("base_color", "metallic", "normal", "roughness"):
            add(textures.get(key), f"texture_{i}_{key}.png")
    for i, item in enumerate(output.get("model3d_assets") or []):
        add(item.get("url"), str(item.get("label") or f"asset_{i}") + ".glb")
    return assets


def parse_task(payload, formats=None, endpoint="video", options=None):
    if not isinstance(payload, dict):
        return TaskState(status="UNKNOWN", error="NanoGPT status response is not an object")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    status = str(data.get("status") or "UNKNOWN").upper()
    status = {"SUCCESS": "COMPLETED", "COMPLETED": "COMPLETED", "SUCCEEDED": "COMPLETED",
              "CANCELLED": "CANCELED"}.get(status, status)
    try:
        progress = max(0, min(100, int(float(str(data.get("progress") or 0).rstrip("%")))))
    except (TypeError, ValueError, OverflowError):
        progress = 0
    state = TaskState(status=status, progress=progress)
    if status in {"FAILED", "CANCELED", "ERROR"}:
        detail = data.get("error") or data.get("failReason") or "generation failed or canceled"
        state.error = str(detail.get("message") or detail) if isinstance(detail, dict) else str(detail)
    elif status == "COMPLETED":
        output = data.get("output") or {}
        try:
            if endpoint == "images":
                state.downloads = [_asset(data.get("imageUrl"), "image.png")]
            elif endpoint == "3d":
                state.downloads = _mesh_assets(output)
                if not state.downloads:
                    state.downloads = [_asset(data.get("videoUrl") or output.get("video", {}).get("url"), "model.glb")]
            else:
                urls = output.get("videoUrls") or data.get("videoUrls")
                if urls:
                    state.downloads = [_asset(url, f"video_{i}.mp4") for i, url in enumerate(urls)]
                else:
                    value = output.get("video") or output.get("url") or data.get("videoUrl") or output.get("video_url")
                    state.downloads = [_asset(value, "video.mp4")]
            if not state.downloads:
                raise NanoGPTInput("completed without an artifact")
            state.progress = 100
        except (NanoGPTInput, TypeError, AttributeError) as exc:
            state.error = f"NanoGPT completed without valid artifacts: {exc}"
    elif status not in {"IN_QUEUE", "IN_PROGRESS", "PENDING", "QUEUED", "PROCESSING", "GENERATING",
                        "DELIVERING", "RUNNING", "SUBMITTED", "NOT_START"}:
        state.error = f"unknown NanoGPT task status: {status}"
    return state
