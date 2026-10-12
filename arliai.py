"""Pure ArliAI SDNext text-to-image request/schema helpers; HTTP lives in adapters."""
from __future__ import annotations

import math

KIND, VENDOR, URL = "arliai", "ArliAI", "https://api.arliai.com"
ENDPOINTS = ("txt2img",)
RIG_ENDPOINT = None
AI_MODELS = ()                         # checkpoint names are account-specific, never invented
FORMATS, RIG_FORMATS = (), ()
MAX_WAIT_DEFAULT = 600
POLL_INTERVAL_DEFAULT = 5.0            # interface compatibility; synchronous API never polls
MAX_IMAGE_PIXELS_DEFAULT = 3_000_000
OPTION_DEFAULTS = {
    "width": 1024, "height": 1024, "steps": 20, "cfg_scale": 6.0,
    "sampler_name": "Euler", "seed": -1, "clip_skip": None,
}
_INT_FIELDS = {"width", "height", "steps", "seed", "clip_skip"}
_TEXT_FIELDS = {"sampler_name"}
OPTION_FIELDS = [
    {"key": k, "label": k.replace("_", " "),
     "type": "int" if k in _INT_FIELDS else "text"}
    for k in OPTION_DEFAULTS
]
BACKEND_HINT = (
    "ArliAI Core shares one simultaneous account request across LLM and image backends; "
    "use the same concurrency group and max_concurrent=1 for both. "
    "max_image_pixels defaults to 3000000. max wait defaults to 600 seconds. "
    "A submitted request with an uncertain answer is never retried.")
ENDPOINT_HINT = (
    "Text-to-image only. Set the exact checkpoint title from your account's model listing. "
    "Client controls override these defaults, never the checkpoint. One image per request.")


class ArliAIInput(RuntimeError):
    """Invalid generation input: final, never fail over to another checkpoint."""


def endpoint_of(cand: dict) -> str:
    ep = (cand.get(KIND) or {}).get("endpoint") or ENDPOINTS[0]
    if ep not in ENDPOINTS:
        raise ArliAIInput("ArliAI supports only txt2img")
    return ep


def options_of(cand: dict) -> dict:
    stored = (cand.get(KIND) or {}).get("options") or {}
    return {k: stored.get(k, v) if stored.get(k, v) not in (None, "") else v
            for k, v in OPTION_DEFAULTS.items()}


def default_candidate(backend: str) -> dict:
    return {"backend": backend, "task": "text2img", "model": "",
            KIND: {"endpoint": "txt2img", "options": dict(OPTION_DEFAULTS)}}


def slot_names(cand: dict) -> list:
    return []


def read_options(form: dict) -> dict:
    """Blank uses the alias default; nonblank malformed numbers never silently truncate."""
    opts = dict(OPTION_DEFAULTS)
    for k in OPTION_DEFAULTS:
        raw = form.get(f"opt__{k}")
        if raw in (None, ""):
            continue
        if k in _TEXT_FIELDS:
            if not isinstance(raw, str) or not raw.strip():
                raise ArliAIInput(f"ArliAI {k} must be a nonempty string")
            opts[k] = raw.strip()
        else:
            opts[k] = _number(k, raw, k in _INT_FIELDS)
    for k in ("width", "height", "steps"):
        if opts[k] <= 0:
            raise ArliAIInput(f"ArliAI {k} must be positive")
    return opts


def public_fields(cand: dict) -> tuple[list, list, list]:
    opts = options_of(cand)
    params = [{"name": "prompt", "type": "string", "default": ""},
              {"name": "negative_prompt", "type": "string", "default": ""}]
    for k in OPTION_DEFAULTS:
        if k in ("cfg_scale", "sampler_name"):
            continue                    # one control per setting; native aliases still accepted by API
        typ = "int" if k in _INT_FIELDS else "float"
        params.append({"name": k, "type": typ,
                       **({"default": opts[k]} if opts[k] is not None else {})})
    params.extend([{"name": "cfg", "type": "float", "default": opts["cfg_scale"]},
                   {"name": "sampler", "type": "string", "default": opts["sampler_name"]}])
    return params, [], []


def _number(k, v, integer=False):
    try:
        n = float(v)
        if isinstance(v, bool) or not math.isfinite(n) or (integer and n != int(n)):
            raise ValueError
        return int(n) if integer else n
    except (TypeError, ValueError, OverflowError) as e:
        raise ArliAIInput(f"ArliAI {k} must be a finite {'integer' if integer else 'number'}") from e


def build_request(cand: dict, values: dict, images: dict, files=None,
                  max_image_pixels=MAX_IMAGE_PIXELS_DEFAULT) -> dict:
    endpoint_of(cand)
    model = cand.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ArliAIInput("ArliAI requires an exact checkpoint name")
    if images or files:
        raise ArliAIInput("ArliAI txt2img accepts no image or file inputs")
    for k in ("batch_size", "n_iter"):
        if k in values and _number(k, values[k], True) != 1:
            raise ArliAIInput(f"ArliAI {k} must be 1; one image per generation request")
    body = options_of(cand)
    for k in body:
        if k in values:
            body[k] = values[k]
    # Gateway short names are aliases; explicit SDNext names take precedence.
    for alias, native in (("cfg", "cfg_scale"), ("sampler", "sampler_name")):
        if alias in values and native not in values:
            body[native] = values[alias]
    for k in list(body):
        v = body[k]
        if v in (None, "") and k not in _TEXT_FIELDS:
            del body[k]
        elif k in _TEXT_FIELDS:
            if not isinstance(v, str) or not v.strip():
                raise ArliAIInput(f"ArliAI {k} must be a nonempty string")
        else:
            body[k] = _number(k, v, k in _INT_FIELDS)
    for k in ("width", "height", "steps"):
        if body.get(k, 0) <= 0:
            raise ArliAIInput(f"ArliAI {k} must be positive")
    budget = _number("max_image_pixels", max_image_pixels, True)
    if budget <= 0 or body["width"] * body["height"] > budget:
        raise ArliAIInput(f"ArliAI image exceeds max_image_pixels ({budget})")
    for k in ("prompt", "negative_prompt"):
        v = values.get(k, "")
        if not isinstance(v, str):
            raise ArliAIInput(f"ArliAI {k} must be a string")
        body[k] = v
    body.update(sd_model_checkpoint=model, batch_size=1, n_iter=1)
    return body


def request_summary(body: dict) -> dict:
    return dict(body)
