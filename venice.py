"""Pure Venice media schemas and native request construction; no HTTP or job I/O.

Static controls follow docs.venice.ai/api-reference/endpoint/{image,video,audio};
model choices, limits and upload requirements come from the live /models metadata.
Multipart audio bytes remain in the adapter's upload_files, never in JSON.
"""
from __future__ import annotations

import base64
import functools
import json
import math
import mimetypes
import re
from urllib.parse import urlsplit, urlunsplit

KIND, VENDOR, URL = "venice", "Venice", "https://api.venice.ai/api"
ENDPOINTS = ("images", "edit", "upscale", "background-remove", "video", "music",
             "voice-changer", "speech", "transcription")
CATALOG_PATHS = {ep: "/v1/models?type=" + typ for ep, typ in
                 (("images", "image"), ("edit", "inpaint"), ("upscale", "upscale"),
                  ("video", "video"), ("music", "music"), ("voice-changer", "music"),
                  ("speech", "tts"), ("transcription", "asr"))}
SUBMIT_PATHS = dict(zip(ENDPOINTS, ("/v1/image/generate", "/v1/image/edit", "/v1/image/upscale",
    "/v1/image/background-remove", "/v1/video/queue", "/v1/audio/queue",
    "/v1/audio/voice-changer/queue", "/v1/audio/speech", "/v1/audio/transcriptions")))
AI_MODELS, FORMATS, RIG_FORMATS = (), (), ()
DYNAMIC_MODELS, RIG_ENDPOINT = True, None
MAX_WAIT_DEFAULT, POLL_INTERVAL_DEFAULT, SUCCESS_STATUS = 1800, 5.0, "COMPLETED"
BACKEND_HINT = ("One Venice backend serves chat, embeddings and all media operations at the same /api URL. "
                "Generation is paid; uncertain submissions are never repeated. Cancellation is local only.")
ENDPOINT_HINT = ("Choose an exact discovered model and endpoint. Controls and reference inputs are model-specific. "
                 "Background removal is a fixed operation, not a vendor model. Model-specific defaults are native JSON.")
_BOOL = {"safe_mode", "hide_watermark", "embed_exif_metadata", "return_binary", "enhance_prompt",
         "enable_web_search", "disable_prompt_optimization_thinking", "transparent_background",
         "audio", "force_instrumental", "lyrics_optimizer", "loop", "remove_background_noise",
         "timestamps", "streaming", "h264_output"}
_INT = {"variants", "n", "seed", "steps", "width", "height", "lora_strength", "duration_seconds",
        "target_fps", "upscale_factor", "slowdown_factor"}
_OPTION_KEYS = ("variants", "format", "output_format", "response_format", "resolution", "aspect_ratio",
                "quality", "width", "height", "seed", "steps", "cfg_scale", "duration", "duration_seconds",
                "scale", "creativity", "voice", "speed", "language", "language_code", "temperature", "top_p",
                "safe_mode", "hide_watermark", "embed_exif_metadata", "enhance_prompt", "enable_web_search",
                "disable_prompt_optimization_thinking", "transparent_background", "audio", "force_instrumental",
                "lyrics_optimizer", "loop", "remove_background_noise", "timestamps", "options_json")
OPTION_DEFAULTS = {k: "" if k == "options_json" else None for k in _OPTION_KEYS}
OPTION_FIELDS = [{"key": k, "label": k.replace("_", " "),
                  "type": "tristate" if k in _BOOL else "int" if k in _INT else "text"}
                 for k in _OPTION_KEYS]
OPTION_FIELDS[-1].update(label="Model-specific defaults (JSON)", textarea=True,
                        hint="Native supported controls only; no model, credentials, queue IDs or endpoint overrides.")
_RESERVED = {"model", "modelid", "endpoint", "provider", "api_key", "apikey", "authorization", "x-api-key",
             "headers", "options_json", "queue_id", "download_url", "cancel", "cancel_url", "submit_path",
             "delete_media_on_completion", "file", "input_audio", "input_video", "token", "access_token"}
_AUDIO_FORMATS = ["wav", "wave", "flac", "m4a", "aac", "mp4", "mp3", "ogg", "oga", "webm"]


def normalize_account_info(payload, checked_at=None):
    """Only account charge/access flags establish API inclusion; balances stay USD.

    The model catalog and Pro website entitlements cannot answer this question.
    Never retain arbitrary response fields or coerce strings/bools into balances.
    """
    def finite_nonnegative(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value) and value >= 0
        except OverflowError:
            return False

    result = {"state": "unknown", "access_permitted": None, "balances": {}}
    if finite_nonnegative(checked_at):
        result["checked_at"] = checked_at
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return result
    access = data.get("accessPermitted")
    result["access_permitted"] = access if isinstance(access, bool) else None
    tier = data.get("apiTier")
    if isinstance(tier, dict):
        charged = tier.get("isCharged")
        if charged is True:
            result["state"] = "credits"
        elif charged is False and access is True:
            result["state"] = "included"
        tier_id = tier.get("id")
        if isinstance(tier_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", tier_id):
            result["api_tier"] = tier_id
    balances = data.get("balances")
    if isinstance(balances, dict):
        result["balances"] = {key: balances[key] for key in ("USD", "DIEM", "BUNDLED_CREDITS")
                              if finite_nonnegative(balances.get(key))}
    return result


class VeniceInput(RuntimeError):
    """Invalid Venice media input is final; it must never initiate paid failover."""


INPUT_ERROR = VeniceInput


def _metadata_guard(fn):
    """The live /models metadata is vendor data: a malformed constraint (a None or 0
    divisor, a list where a dict belongs, a non-numeric limit) must surface as
    VeniceInput — a 400 before any paid POST — never as a TypeError the preflight
    does not catch (a 500)."""
    @functools.wraps(fn)
    def guarded(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except VeniceInput:
            raise
        except (TypeError, ValueError, ArithmeticError, AttributeError, KeyError, IndexError) as error:
            raise VeniceInput(f"Venice model metadata or input is malformed "
                              f"({type(error).__name__}: {error})") from error
    return guarded


def endpoint_of(cand):
    endpoint = (cand.get(KIND) or {}).get("endpoint") or "images"
    if endpoint not in ENDPOINTS:
        raise VeniceInput("invalid Venice media endpoint")
    return endpoint


def generation_type(endpoint):
    if endpoint not in ENDPOINTS:
        raise VeniceInput("invalid Venice media endpoint")
    return {"video": "video", "music": "audio", "voice-changer": "audio",
            "speech": "tts", "transcription": "asr"}.get(endpoint, "image")


def music_endpoint(model_meta):
    return "voice-changer" if (model_meta.get("model_spec") or {}).get("voice_changer") is True else "music"


def task_for(endpoint, previous_task=""):
    generation_type(endpoint)
    if endpoint in {"edit", "upscale", "background-remove"}:
        return "img2img"
    if endpoint == "images":
        return "text2img"
    if endpoint == "video":
        return "img2vid" if str(previous_task).startswith("img") else "text2vid"
    return {"music": "text2audio", "speech": "text2audio", "voice-changer": "audio2audio",
            "transcription": "audio2text"}[endpoint]


def candidate_for(backend, model, endpoint, model_meta=None):
    cand = {"backend": backend, "model": model, "task": task_for(endpoint),
            KIND: {"endpoint": endpoint, "options": dict(OPTION_DEFAULTS)}}
    if model_meta is not None:
        if not isinstance(model_meta, dict):
            raise VeniceInput("Venice model metadata must be an object")
        cand["model_metadata"] = model_meta
        if endpoint == "video" and _constraints(cand).get("model_type") == "image-to-video":
            cand["task"] = "img2vid"
    return cand


def default_candidate(backend):
    return candidate_for(backend, "", "images")


def _spec(cand):
    return (cand.get("model_metadata") or {}).get("model_spec") or {}


def _constraints(cand):
    return _spec(cand).get("constraints") or {}


def _number(key, value, integer=False):
    try:
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or integer and number != int(number):
            raise ValueError
        return int(number) if integer else number
    except (ValueError, TypeError, OverflowError) as exc:
        raise VeniceInput(f"Venice {key} must be a finite {'integer' if integer else 'number'}") from exc


def _finite_safe(value, nested=False):
    if isinstance(value, float) and not math.isfinite(value):
        raise VeniceInput("Venice controls must be finite JSON")
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _RESERVED:
                raise VeniceInput("Venice controls cannot override model, credentials, endpoint or queue state")
            _finite_safe(item, True)
    elif isinstance(value, list):
        for item in value:
            _finite_safe(item, True)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise VeniceInput("Venice controls must be JSON values")


def _advanced(raw):
    if raw in (None, ""):
        return {}
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise VeniceInput("model-specific defaults must be valid finite JSON") from exc
    if not isinstance(obj, dict):
        raise VeniceInput("model-specific defaults must be a JSON object")
    _finite_safe(obj)
    return obj


def options_of(cand):
    stored = (cand.get(KIND) or {}).get("options") or {}
    if not isinstance(stored, dict):
        raise VeniceInput("Venice options must be an object")
    unknown = set(stored) - set(OPTION_DEFAULTS)
    if unknown:
        raise VeniceInput("unsupported Venice alias option: " + str(sorted(unknown)[0]))
    result = {k: stored.get(k, v) for k, v in OPTION_DEFAULTS.items()}
    _advanced(result["options_json"])
    for key in _INT & result.keys():
        if result[key] not in (None, ""):
            result[key] = _number(key, result[key], True)
    for key in _BOOL & result.keys():
        if result[key] is not None and not isinstance(result[key], bool):
            raise VeniceInput(f"Venice {key} must be a boolean or unset")
    return result


def read_options(form):
    values = {}
    for key in OPTION_DEFAULTS:
        value = form.get("opt__" + key)
        if key in _BOOL:
            if value in (None, ""):
                value = None
            elif value in ("true", "false"):
                value = value == "true"
            else:
                raise VeniceInput(f"Venice {key} must be true, false or blank")
        values[key] = value if value != "" else None
    values["options_json"] = values["options_json"] or ""
    return options_of({KIND: {"options": values}})


def _field(name, typ="string", **kw):
    return {"name": name, "type": typ, **kw}


def _parameter_specs(cand):
    ep, spec, c = endpoint_of(cand), _spec(cand), _constraints(cand)
    fields = {}

    def add(name, typ="string", **kw):
        fields[name] = _field(name, typ, **kw)

    def choices(name, values, default=None):
        if values:
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
            typ = ("int" if all(isinstance(v, int) for v in values) else "float") if numeric else "string"
            add(name, typ, choices=list(values), **({"default": default} if default is not None else {}))

    if ep in {"images", "edit"}:
        add("prompt", required=True, min_length=1, max_length=c.get("promptCharacterLimit", 7500 if ep == "images" else 32768))
        add("safe_mode", "bool")
        add("enhance_prompt", "bool")
        add("anon_user_id", min_length=1, max_length=128)
        choices("aspect_ratio", c.get("aspectRatios"), c.get("defaultAspectRatio"))
        choices("resolution", c.get("resolutions"), c.get("defaultResolution"))
        choices("quality", c.get("qualities"), c.get("defaultQuality"))
        if c.get("supportsTransparentBackground"):
            add("transparent_background", "bool")
        if spec.get("supportsOptimizePromptThinking"):
            add("disable_prompt_optimization_thinking", "bool")
        if ep == "edit":
            choices("output_format", ["jpeg", "png", "webp"])
            add("image")
            add("images", "array", max_items=_image_limit(cand))
        else:
            add("negative_prompt", max_length=c.get("promptCharacterLimit", 7500))
            choices("format", ["jpeg", "png", "webp"])
            add("variants", "int", min=1, max=4)
            add("seed", "int", min=-999999999, max=999999999)
            add("cfg_scale", "float", min=0, exclusive_min=True, max=20)
            for name in ("hide_watermark", "embed_exif_metadata", "return_binary"):
                add(name, "bool")
            if not c.get("aspectRatios"):
                for name in ("width", "height"):
                    add(name, "int", min=1, max=1280)
            if c.get("steps"):
                add("steps", "int", **{"min": 1, **{k: v for k, v in c["steps"].items() if k in {"min", "max", "default"}}})
            add("style_preset")
            add("lora_strength", "int", min=0, max=100)
            if spec.get("supportsWebSearch"):
                add("enable_web_search", "bool")
            if spec.get("supportsStyleReferences"):
                add("style_references", "array", max_items=c.get("maxStyleReferences", spec.get("maxStyleReferences", 4)))
    elif ep in {"upscale", "background-remove"}:
        add("image")
        if ep == "upscale":
            add("scale", "int", choices=[2, 4], default=2)
            add("creativity", "float", min=0, max=0.02, default=0.01)
        else:
            add("image_url")
            add("anon_user_id", min_length=1, max_length=128)
    elif ep == "video":
        model = cand.get("model", "")
        reference = "reference-to-video" in model
        seedance = model.startswith(("seedance-2-0", "seedance-2-5"))
        limit = c.get("prompt_character_limit", 2500)
        add("prompt", required=not (model.startswith("topaz-") or model == "minimax-h3-max-multi-angle"),
            min_length=1, max_length=limit)
        add("negative_prompt", max_length=limit)
        durations = list(c.get("durations") or [])
        ratios = list(c.get("aspect_ratios") or [])
        if reference and seedance:
            ratios = list(dict.fromkeys(ratios + ["adaptive", "auto"]))
            if model.startswith("seedance-2-5"):
                durations = list(dict.fromkeys(durations + ["-1", "auto"]))
        choices("duration", durations)
        choices("resolution", c.get("resolutions"))
        choices("aspect_ratio", ratios)
        if not durations:
            add("duration")
        if c.get("audio_configurable"):
            add("audio", "bool")
        if c.get("model_type") == "image-to-video":
            if not reference or model.startswith("kling-o3"):
                add("image_url")
            if _end_image_supported(cand):
                add("end_image_url")
            if reference or c.get("reference_images"):
                add("reference_image_urls", "array", max_items=_reference_image_limit(cand))
            if reference and model.startswith("kling-o3") or c.get("elements"):
                add("elements", "array", max_items=4)
                add("scene_image_urls", "array", max_items=4)
            if c.get("keyframes"):
                add("keyframes", "array", max_items=10)
            if c.get("image_strength"):
                add("image_strength", "float", min=0, max=1)
            if model == "minimax-h3-max-multi-angle" or c.get("camera_trajectory"):
                add("camera_trajectory", "array", min_items=2, max_items=12)
        if c.get("audio_input"):
            add("audio_url")
        if c.get("video_input"):
            add("video_url")
        if reference and (seedance or model.startswith("wan-3-0")):
            reference_limit = 3 if model.startswith("seedance-2-0") else 10
            add("reference_video_urls", "array", max_items=reference_limit)
            add("reference_audio_urls", "array", max_items=reference_limit)
            if model.startswith("wan-3-0"):
                add("reference_document_urls", "array", max_items=1)
            if model.startswith("seedance-2-5"):
                choices("omni_reference_task_type", ["auto", "reference", "edit", "extend"])
        if seedance:
            choices("bitrate_mode", ["standard", "high"])
            add("consents", "object")
        if model == "topaz-video-upscale":
            fields = {}
            add("video_url")
            choices("upscale_factor", [1, 2, 4], 2)
        elif c.get("topaz"):
            choices("upscale_factor", [1, 2, 4])
            fields.pop("resolution", None)
            add("enhancement_model")
            add("target_fps", "int", min=16, max=120)
            for name in ("creativity", "realism", "sharp", "compression", "noise", "halo", "recover_detail"):
                add(name, "float", min=0, max=1)
            add("softness", "float", min=1, max=5)
            add("grain", "float", min=0, max=0.1)
            add("h264_output", "bool")
            choices("output_format", ["mp4", "prores"])
            choices("slowdown_factor", [1, 2, 4, 8])
    elif ep == "music":
        add("prompt", required=True, min_length=spec.get("min_prompt_length", 1),
            **({"max_length": spec["prompt_character_limit"]} if "prompt_character_limit" in spec else {}))
        if spec.get("supports_lyrics"):
            add("lyrics_prompt", required=spec.get("lyrics_required", False),
                **({"max_length": spec["lyrics_character_limit"]} if "lyrics_character_limit" in spec else {}))
        if any(k in spec for k in ("duration_options", "min_duration", "max_duration", "default_duration")):
            add("duration_seconds", "int", min=spec.get("min_duration", 1),
                **({"max": spec["max_duration"]} if "max_duration" in spec else {}),
                **({"choices": spec["duration_options"]} if spec.get("duration_options") else {}),
                **({"default": spec["default_duration"]} if "default_duration" in spec else {}))
        for name in ("force_instrumental", "lyrics_optimizer", "loop"):
            if spec.get("supports_" + name):
                add(name, "bool")
        if spec.get("supports_language_code"):
            add("language_code")
        if spec.get("supports_speed"):
            add("speed", "float", min=spec.get("min_speed", 0.25), max=spec.get("max_speed", 4),
                default=spec.get("default_speed", 1))
    elif ep == "voice-changer":
        add("audio_url")
        add("remove_background_noise", "bool")
        add("seed", "int", min=0)
    elif ep == "speech":
        add("input", required=True, min_length=1, max_length=4096)
        add("language", min_length=2, max_length=32)
        add("speed", "float", min=0.25, max=4)
        choices("response_format", spec.get("supported_formats", ["mp3", "opus", "aac", "flac", "wav", "pcm"]),
                spec.get("default_format"))
        # These are endpoint-documented capabilities. The catalog currently omits
        # supports*Param, so do not infer them from an arbitrary future model ID.
        model = cand.get("model", "")
        if spec.get("supportsPromptParam", model.startswith("tts-qwen3-")):
            add("prompt", max_length=500)
        if spec.get("supportsTemperatureParam", model.startswith("tts-qwen3-") or model in {"tts-orpheus", "tts-chatterbox-hd"}):
            add("temperature", "float", min=0, max=2)
        if spec.get("supportsTopPParam", model.startswith("tts-qwen3-")):
            add("top_p", "float", min=0, max=1)
        add("streaming", "bool")
    elif ep == "transcription":
        choices("response_format", ["json", "text"])
        add("timestamps", "bool")
        add("language")
    if ep in {"speech", "music", "voice-changer"} and (spec.get("voices") or spec.get("supports_custom_voice_id") or spec.get("voice_cloning")):
        add("voice", max_length=512, **({"default": spec["default_voice"]} if "default_voice" in spec else {}))
        if spec.get("voices") and not (spec.get("supports_custom_voice_id") or spec.get("voice_cloning")):
            fields["voice"]["choices"] = spec["voices"]
    return fields


def _reference_image_limit(cand):
    model, c = cand.get("model", ""), _constraints(cand)
    documented = 9 if model.startswith("seedance-2-0") else 7 if model.startswith("grok-imagine") else 30
    advertised = c.get("max_reference_images", documented)
    return max(0, min(documented, _number("max_reference_images", advertised, True)))


def _end_image_supported(cand):
    model, c = cand.get("model", ""), _constraints(cand)
    explicit = c.get("end_image") or c.get("supportsEndImage")
    if "reference-to-video" in model:
        return bool(explicit or model.startswith("kling-o3"))
    return bool(explicit or "transition" in model or "first-last-frame" in model or
                model.startswith(("seedance-2-0", "seedance-2-5", "kling-o3")))


def _image_limit(cand):
    ep, c = endpoint_of(cand), _constraints(cand)
    if ep == "edit":
        return max(1, int(c.get("maxInputImages", 3 if c.get("combineImages") else 1))) if c.get("combineImages", True) else 1
    if ep in {"upscale", "background-remove"}:
        return 1
    if ep == "images" and _spec(cand).get("supportsStyleReferences"):
        return int(c.get("maxStyleReferences", _spec(cand).get("maxStyleReferences", 4)))
    if ep == "video" and c.get("model_type") == "image-to-video":
        if ("reference-to-video" in cand.get("model", "") or c.get("reference_images")) and not _end_image_supported(cand):
            return _reference_image_limit(cand)
        return 2 if _end_image_supported(cand) else 1
    return 0


def slot_names(cand):
    return ["input_image" if i == 1 else f"input_image_{i}" for i in range(1, _image_limit(cand) + 1)]


def _file_fields(cand):
    ep, spec, c = endpoint_of(cand), _spec(cand), _constraints(cand)
    fields = []
    if ep in {"transcription", "voice-changer"} or ep == "speech" and spec.get("voice_cloning"):
        clone = spec.get("voice_cloning") or {}
        fields.append({"name": "input_audio", "required": ep != "speech",
                       "accept": clone.get("accepted_formats", spec.get("accepted_audio_formats", _AUDIO_FORMATS))})
    if ep == "video":
        for modality in ("audio", "video"):
            if c.get(modality + "_input"):
                fields.append({"name": "input_" + modality,
                               "required": modality == "video" and c.get("model_type") == "video",
                               "accept": ["mp4", "mov", "webm"] if modality == "video" else ["wav", "mp3"]})
    return fields


def public_fields(cand):
    opts = options_of(cand)
    defaults = {**_advanced(opts.pop("options_json")), **{k: v for k, v in opts.items() if v not in (None, "")}}
    defaults = _translate(cand, defaults)
    params = []
    for key, field in _parameter_specs(cand).items():
        if field["type"] in {"array", "object"}:
            continue  # Native structured controls are supplied through extra/options_json.
        field = dict(field)
        if key in defaults:
            field["default"] = defaults[key]
        params.append(field)
    ep = endpoint_of(cand)
    images = [{"name": name, "required": i == 0 and ep in {"edit", "upscale", "background-remove", "video"},
               "on_empty": "none"} for i, name in enumerate(slot_names(cand))]
    return params, images, _file_fields(cand)


def _translate(cand, values):
    body, ep = dict(values), endpoint_of(cand)
    if ep != "images" and "n" in body:
        if _number("n", body.pop("n"), True) != 1:
            raise VeniceInput("Venice endpoint supports one output per request")
    aliases = {"n": "variants", "cfg": "cfg_scale"} if ep == "images" else {}
    if ep == "images":
        aliases.update(output_format="format")
    elif ep == "edit":
        aliases.update(format="output_format")
    elif ep in {"speech", "transcription"}:
        aliases.update(format="response_format", output_format="response_format")
    if ep == "video":
        aliases.update(seconds="duration", duration_seconds="duration")
    if ep == "music":
        aliases.update(seconds="duration_seconds", duration="duration_seconds")
    for short, native in aliases.items():
        if short in body:
            body.setdefault(native, body.pop(short))
    if ep in {"images", "edit"} and "size" in body:
        size = body.pop("size")
        if size == "auto":
            pass
        elif isinstance(size, str) and re.fullmatch(r"[1-9]\d*x[1-9]\d*", size):
            width, height = map(int, size.split("x"))
            if _constraints(cand).get("aspectRatios"):
                divisor = math.gcd(width, height)
                body.setdefault("aspect_ratio", f"{width // divisor}:{height // divisor}")
            else:
                body.setdefault("width", width)
                body.setdefault("height", height)
        else:
            raise VeniceInput("Venice size must be auto or WIDTHxHEIGHT")
    if ep == "video" and "duration" in body:
        value = body["duration"]
        if isinstance(value, (int, float)) or isinstance(value, str) and re.fullmatch(r"\d+(\.\d+)?", value):
            number = _number("duration", value)
            body["duration"] = f"{int(number) if number.is_integer() else number}s"
    if ep == "images" and body.get("format") == "jpg" or ep == "edit" and body.get("output_format") == "jpg":
        body["format" if ep == "images" else "output_format"] = "jpeg"
    return body


def _values(cand, values):
    if not isinstance(values, dict):
        raise VeniceInput("Venice parameters must be an object")
    model, ep = cand.get("model"), endpoint_of(cand)
    if not isinstance(model, str) or not model.strip():
        raise VeniceInput("an exact discovered Venice model ID is required")
    extra = values.get("extra") or {}
    if not isinstance(extra, dict):
        raise VeniceInput("Venice extra must be an object")
    client = {**extra, **{k: v for k, v in values.items() if k not in {"extra", "model"} and v not in (None, "")}}
    _finite_safe(client)
    opts = options_of(cand)
    alias = {**_advanced(opts.pop("options_json")), **{k: v for k, v in opts.items() if v not in (None, "")}}
    if ep == "speech" and "input" not in client and "prompt" in client and "input" not in alias:
        client["input"] = client.pop("prompt")
    specs = _parameter_specs(cand)
    body = {k: f["default"] for k, f in specs.items() if "default" in f}
    body.update(_translate(cand, alias))
    body.update(_translate(cand, client))
    if ep in {"music", "voice-changer"} and cand.get("model_metadata") and music_endpoint(cand["model_metadata"]) != ep:
        raise VeniceInput("Venice music and voice-changer models use different endpoints")
    unknown = set(body) - set(specs)
    if unknown:
        raise VeniceInput("unsupported Venice model control: " + str(sorted(unknown)[0]))
    for key, field in specs.items():
        if field.get("required") and (key not in body or isinstance(body[key], str) and not body[key].strip()):
            raise VeniceInput(f"Venice {key} is required")
        if key not in body:
            continue
        value, typ = body[key], field["type"]
        if typ in {"int", "float"}:
            value = _number(key, value, typ == "int")
            if value < field.get("min", -math.inf) or value > field.get("max", math.inf) or field.get("exclusive_min") and value == field.get("min"):
                raise VeniceInput(f"Venice {key} is outside the model's bounds")
        elif typ == "bool":
            if not isinstance(value, bool):
                raise VeniceInput(f"Venice {key} must be a boolean")
        elif typ in {"array", "object"}:
            if not isinstance(value, list if typ == "array" else dict):
                raise VeniceInput(f"Venice {key} must be a JSON {typ}")
            if len(value) > field.get("max_items", math.inf) or len(value) < field.get("min_items", 0):
                raise VeniceInput(f"Venice {key} exceeds the model's item bounds")
        elif not isinstance(value, str):
            raise VeniceInput(f"Venice {key} must be text")
        if isinstance(value, str) and (len(value) < field.get("min_length", 0) or len(value) > field.get("max_length", math.inf)):
            raise VeniceInput(f"Venice {key} is outside the model's text length bounds")
        if "choices" in field and value not in field["choices"]:
            raise VeniceInput(f"Venice {key} must be one of the model's advertised choices")
        body[key] = value
    if ep == "images":
        body.setdefault("variants", 1)
        if body.get("return_binary") and body["variants"] != 1:
            raise VeniceInput("Venice binary image output supports one variant only")
        divisor = _constraints(cand).get("widthHeightDivisor")
        if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor < 1:
            divisor = 1
        if any(body.get(key, divisor) % divisor for key in ("width", "height")):
            raise VeniceInput("Venice image dimensions must match the model divisor")
    if body.get("transparent_background") and body.get("format" if ep == "images" else "output_format") not in (None, "png", "webp"):
        raise VeniceInput("transparent images require png or webp output")
    if body.get("lyrics_optimizer") and body.get("lyrics_prompt"):
        raise VeniceInput("lyrics_optimizer requires empty lyrics_prompt")
    if ep == "video" and model != "topaz-video-upscale" and "duration" not in body:
        raise VeniceInput("Venice video duration is required (native seconds, not frames)")
    if ep == "voice-changer" and body.get("audio_url"):
        _url(body["audio_url"], http_only=True)
    if "voice" in body and _spec(cand).get("voices"):
        voice, spec = body["voice"], _spec(cand)
        if voice not in spec["voices"] and not spec.get("supports_custom_voice_id") and not (spec.get("voice_cloning") and voice.startswith("vv_")):
            raise VeniceInput("Venice voice must belong to the selected model")
    if "anon_user_id" in body and ("||" in body["anon_user_id"] or any(not 32 <= ord(ch) <= 126 for ch in body["anon_user_id"])):
        raise VeniceInput("Venice anon_user_id must be printable ASCII without ||")
    if ep not in {"upscale", "background-remove"}:
        body["model"] = model.strip()
    return body


def _url(value, http_only=False):
    if not isinstance(value, str) or not value.strip():
        raise VeniceInput("Venice media references must be non-empty strings")
    if value.startswith(("http://", "https://")):
        parsed = urlsplit(value)
        if not parsed.hostname or parsed.username or parsed.password:
            raise VeniceInput("Venice media URLs must not contain credentials")
        return
    if not http_only and value.startswith("data:") and ";base64," in value:
        encoded = value.split(";base64,", 1)[1]
        if not encoded or len(encoded) % 4 or not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", encoded):
            raise VeniceInput("Venice inline media must contain valid base64")
        return
    raise VeniceInput("Venice media reference must be an HTTP(S) URL" + (" or data URL" if not http_only else ""))


def _object_keys(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise VeniceInput("invalid Venice structured control fields")


def _structured_controls(cand, body, image_uploaded=False):
    for ref in body.get("style_references", []):
        _object_keys(ref, {"image", "strength"}, {"image"})
        _image_reference(ref["image"])
        if "strength" in ref and not 0.1 <= _number("style strength", ref["strength"]) <= 1:
            raise VeniceInput("Venice style reference strength must be between 0.1 and 1")
        if "strength" in ref and _spec(cand).get("supportsStyleReferenceStrength") is False:
            raise VeniceInput("Venice model does not support style reference strength")
    for element in body.get("elements", []):
        _object_keys(element, {"frontal_image_url", "reference_image_urls", "video_url"})
        if not (element.get("frontal_image_url") or element.get("video_url")):
            raise VeniceInput("Venice element requires a frontal image or source video")
        for key in ("frontal_image_url", "video_url"):
            if element.get(key):
                _url(element[key])
        refs = element.get("reference_image_urls", [])
        if not isinstance(refs, list) or len(refs) > 3:
            raise VeniceInput("Venice element supports at most three reference images")
        for ref in refs:
            _url(ref)
    indices = set()
    for frame in body.get("keyframes", []):
        _object_keys(frame, {"image_url", "frame_index"}, {"image_url", "frame_index"})
        _url(frame["image_url"])
        index = _number("frame_index", frame["frame_index"], True)
        duration = body.get("duration", "")
        if index < 0 or index in indices or str(duration).endswith("s") and index > float(duration[:-1]) * 24:
            raise VeniceInput("Venice keyframe indices must be unique and within the video duration")
        indices.add(index)
    previous, travel, azimuth = -1, 0, None
    for point in body.get("camera_trajectory", []):
        _object_keys(point, {"time", "azimuth", "elevation", "distance"},
                     {"time", "azimuth", "elevation", "distance"})
        time = _number("camera time", point["time"])
        angle = _number("camera azimuth", point["azimuth"])
        if not previous < time <= 1 or time < 0 or not -90 <= _number("camera elevation", point["elevation"]) <= 90 or _number("camera distance", point["distance"]) <= 0:
            raise VeniceInput("invalid Venice camera trajectory bounds or ordering")
        if azimuth is not None:
            travel += abs(angle - azimuth)
        previous, azimuth = time, angle
    if travel > 32 * 360:
        raise VeniceInput("Venice camera trajectory exceeds 32 turns")
    if body.get("camera_trajectory") and not (body.get("image_url") or image_uploaded):
        raise VeniceInput("Venice camera trajectory requires image_url")
    if "consents" in body:
        _object_keys(body["consents"], {"seedance"})
        consent = body["consents"].get("seedance")
        if consent is not None:
            names = {"confirmed_terms_and_privacy", "confirmed_legal_right", "confirmed_screening_acknowledged"}
            _object_keys(consent, names, names)
            if any(value is not True for value in consent.values()):
                raise VeniceInput("Venice consent attestations must each be true")


def _image_reference(value):
    if isinstance(value, str) and value.startswith(("http://", "https://", "data:")):
        _url(value)
        return
    try:
        if not isinstance(value, str) or not base64.b64decode(value, validate=True):
            raise ValueError
    except (ValueError, TypeError) as exc:
        raise VeniceInput("Venice image reference must be base64 or an HTTP(S)/data URL") from exc


@_metadata_guard
def validate_request(cand, values, image_keys, file_keys, model_meta=None):
    if model_meta is not None:
        cand = {**cand, "model_metadata": model_meta}
    body = _values(cand, values)
    ep, c = endpoint_of(cand), _constraints(cand)
    image_keys, file_keys = set(image_keys), set(file_keys)
    if image_keys - set(slot_names(cand)) or file_keys - {f["name"] for f in _file_fields(cand)}:
        raise VeniceInput("unsupported Venice upload slot")
    urls = body.get("images", [])
    if urls and any(not isinstance(value, str) or not value.strip() for value in urls):
        raise VeniceInput("Venice image references must be non-empty strings")
    if ep in {"edit", "upscale", "background-remove"}:
        references = len(urls) + int(bool(body.get("image"))) + int(bool(body.get("image_url")))
        if image_keys and references:
            raise VeniceInput("choose uploaded images or native image parameters, not both")
        if references + len(image_keys) < 1 or references + len(image_keys) > _image_limit(cand):
            raise VeniceInput("Venice image reference count is outside the model's bounds")
        for ref in urls + ([body["image"]] if body.get("image") else []):
            _image_reference(ref)
        if body.get("image_url"):
            _url(body["image_url"], http_only=True)
    if ep == "video":
        visual = image_keys or any(body.get(k) for k in ("image_url", "reference_image_urls", "elements", "scene_image_urls", "keyframes", "reference_video_urls"))
        if c.get("model_type") == "image-to-video" and not visual:
            raise VeniceInput("Venice image-to-video requires a visual reference")
        if c.get("model_type") == "video" and not (body.get("video_url") or "input_video" in file_keys):
            raise VeniceInput("Venice video-to-video requires a source video")
        if "input_image" in image_keys and body.get("image_url") or "input_image_2" in image_keys and body.get("end_image_url"):
            raise VeniceInput("choose an uploaded image or URL, not both")
        if image_keys and body.get("reference_image_urls"):
            raise VeniceInput("choose uploaded reference images or reference_image_urls, not both")
        if "reference_image_urls" in body and len(body["reference_image_urls"]) + len(image_keys) > _reference_image_limit(cand):
            raise VeniceInput("Venice reference image count exceeds the model's bound")
        for key in ("image_url", "end_image_url", "audio_url", "video_url"):
            if body.get(key):
                _url(body[key])
        for key in ("reference_image_urls", "reference_audio_urls", "reference_video_urls", "reference_document_urls", "scene_image_urls"):
            for url in body.get(key, []):
                _url(url)
        if body.get("reference_audio_urls") and not (visual or body.get("reference_video_urls")):
            raise VeniceInput("reference audio requires visual references")
        model = cand.get("model", "")
        if body.get("duration") in {"auto", "-1"} and model.startswith("seedance-2-5") and not body.get("reference_video_urls"):
            raise VeniceInput("source-matched duration requires reference_video_urls")
        if body.get("aspect_ratio") in {"adaptive", "auto"} and "reference-to-video" in model and model.startswith(("seedance-2-0", "seedance-2-5")) and not body.get("reference_video_urls"):
            raise VeniceInput("source-matched aspect ratio requires reference_video_urls")
        if ("transition" in model or "first-last-frame" in model) and not (
                (body.get("image_url") or "input_image" in image_keys) and
                (body.get("end_image_url") or "input_image_2" in image_keys)):
            raise VeniceInput("Venice transition requires both start and end images")
    if ep in {"transcription", "voice-changer"} and not ("input_audio" in file_keys or ep == "voice-changer" and body.get("audio_url")):
        raise VeniceInput("Venice source audio is required")
    for modality in ("audio", "video"):
        if "input_" + modality in file_keys and body.get(modality + "_url"):
            raise VeniceInput("choose an uploaded file or URL, not both")
    if ep == "speech" and "input_audio" in file_keys and (values.get("voice") or (values.get("extra") or {}).get("voice")):
        raise VeniceInput("choose a voice handle or uploaded reference, not both")
    _structured_controls(cand, body, "input_image" in image_keys)
    return body


def _data_url(data, mime):
    return "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")


def _image_mime(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    raise VeniceInput("Venice image reference must contain image bytes")


_IMAGE_CONSTRAINTS = ("reference_image_min_short_side_pixels",
                      "reference_image_min_aspect_ratio", "reference_image_max_aspect_ratio")


def image_constraints(cand):
    """Decoded-image restrictions; the adapter performs decoding in its worker."""
    constraints = _constraints(cand)
    return {key: constraints[key] for key in _IMAGE_CONSTRAINTS if constraints.get(key) is not None}


@_metadata_guard
def validate_image_dimensions(cand, dimensions):
    """Validate dimensions supplied by a trusted decoder, never by client controls."""
    limits = image_constraints(cand)
    if not limits:
        return
    if not isinstance(dimensions, (tuple, list)) or len(dimensions) != 2:
        raise VeniceInput("decoded Venice reference image dimensions are required")
    width = _number("reference image width", dimensions[0], True)
    height = _number("reference image height", dimensions[1], True)
    if min(width, height) <= 0 or min(width, height) < limits.get("reference_image_min_short_side_pixels", 0):
        raise VeniceInput("Venice reference image is below the model's minimum short side")
    ratio = width / height
    lower = limits.get("reference_image_min_aspect_ratio", -math.inf)
    upper = limits.get("reference_image_max_aspect_ratio", math.inf)
    exclusive = cand.get("model", "").startswith(("seedance-2-0", "seedance-2-5"))
    if ratio < lower or ratio > upper or exclusive and (ratio == lower or ratio == upper):
        raise VeniceInput("Venice reference image is outside the model's aspect-ratio bounds")


@_metadata_guard
def build_request(cand, values, images, files=None, image_dimensions=None):
    files = files or {}
    body = validate_request(cand, values, images.keys(), files.keys())
    ep = endpoint_of(cand)
    for name, data in images.items():
        if not isinstance(data, bytes) or not data:
            raise VeniceInput("Venice uploaded images must contain bytes")
        limit = 8 * 1024 * 1024 if ep == "images" else 25 * 1024 * 1024
        if len(data) >= limit:
            raise VeniceInput("Venice uploaded image exceeds the endpoint byte limit")
        _image_mime(data)
        if image_constraints(cand):
            validate_image_dimensions(cand, (image_dimensions or {}).get(name))
    encoded = {name: _data_url(images[name], _image_mime(images[name]))
               for name in slot_names(cand) if images.get(name) and ep not in {"upscale", "background-remove"}}
    refs = list(encoded.values())
    if ep == "edit":
        native_refs, native_image = body.pop("images", []), body.pop("image", None)
        refs = refs or native_refs or ([native_image] if native_image else [])
        if len(refs) > 1:
            body["images"], body["modelId"] = refs, body.pop("model")
        else:
            body["image"] = refs[0]
    elif ep in {"upscale", "background-remove"} and images:
        body["image"] = base64.b64encode(images["input_image"]).decode("ascii")
    elif ep == "images" and refs:
        if body.get("style_references"):
            raise VeniceInput("choose uploaded style references or style_references, not both")
        body["style_references"] = [{"image": ref} for ref in refs]
    elif ep == "video":
        if ("reference-to-video" in cand.get("model", "") or _constraints(cand).get("reference_images")) and not _end_image_supported(cand):
            if refs:
                body["reference_image_urls"] = refs
        else:
            for name, key in (("input_image", "image_url"), ("input_image_2", "end_image_url")):
                if images.get(name):
                    body[key] = encoded[name]
    for name, (filename, data) in files.items():
        if not data:
            raise VeniceInput("Venice uploaded media must not be empty")
        if ep == "video":
            body[name.removeprefix("input_") + "_url"] = _data_url(data, mimetypes.guess_type(filename)[0] or "application/octet-stream")
        else:
            field = next(f for f in _file_fields(cand) if f["name"] == name)
            suffix = filename.rsplit(".", 1)[-1].lower()
            if suffix not in field["accept"]:
                raise VeniceInput("Venice uploaded audio format is not supported by the model")
            limit = _spec(cand).get("max_source_audio_size_bytes")
            if isinstance(limit, int) and len(data) > limit:
                raise VeniceInput("Venice source audio exceeds the model's byte limit")
    return body


def submit_path(endpoint, body=None):
    if endpoint not in ENDPOINTS:
        raise VeniceInput("invalid Venice media endpoint")
    if endpoint == "edit" and (body or {}).get("images"):
        return "/v1/image/multi-edit"
    return SUBMIT_PATHS[endpoint]


def request_summary(body):
    """Never retain audio/image payloads, credentials or signed delivery links."""
    media_keys = {"image", "images", "style_references", "image_url", "end_image_url", "audio_url", "video_url",
                  "reference_image_urls", "reference_audio_urls", "reference_video_urls", "scene_image_urls",
                  "elements", "keyframes", "download_url", "reference_document_urls"}

    def redact(value, key=""):
        if key.lower() in _RESERVED - {"model", "modelid"} or any(
                word in key.lower() for word in ("password", "secret", "credential")):
            return "<redacted>"
        if isinstance(value, (bytes, bytearray)):
            return f"<media:{len(value)} bytes>"
        if isinstance(value, str):
            if value.startswith("data:") or key in media_keys and not value.startswith(("http://", "https://")):
                return "<media>"
            if value.startswith(("http://", "https://")):
                url = urlsplit(value)
                return urlunsplit((url.scheme, url.hostname or "", url.path, "", ""))
            return value
        if isinstance(value, dict):
            return {k: redact(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v, key) for v in value]
        return value

    return redact(body)
