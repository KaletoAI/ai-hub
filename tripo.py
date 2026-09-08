"""Tripo3D (https://developers.tripo3d.ai/en/docs, API V3) as a generation backend — the PURE half.

Same shape as meshy.py, one difference that matters: Tripo takes NO inline bytes. Every
image and mesh is uploaded first (POST /v3/files → file_token, done by the adapter), so
`build_request` receives TOKENS where meshy.build_request receives bytes. No main/adapters
imports, no I/O. Covered by test_tripo.py.
"""
from __future__ import annotations

import copy
from typing import Optional

from cloudtask import TaskState

# The cloud-kind interface every cloud task module declares (design spec 2026-09-03 §3.2):
# kind key, display name, default backend URL, API prefix, and which ENDPOINT rigs. The
# adapter, the console editor and `main` read these instead of hard-coding "tripo" again.
KIND, VENDOR, URL, API = "tripo", "Tripo", "https://openapi.tripo3d.ai", "/v3"
POLL_INTERVAL_DEFAULT, MAX_WAIT_DEFAULT = 2.0, 900     # the docs' own poll recommendation
ENDPOINTS = ("image-to-model", "multiview-to-model", "rig")
RIG_ENDPOINT = "rig"
# The free verdict task the docs ask for before every rig. Not in ENDPOINTS: no alias
# can be configured on it — the adapter runs it as a step of the rig endpoint.
RIG_CHECK_ENDPOINT = "rig-check"
SUCCESS_STATUS = "success"              # the one `TaskState.status` that means delivered
AI_MODELS = ("v3.1-20260211", "v3.0-20250812", "v2.5-20250123", "P1-20260311", "P2-20260801")
RIG_MODELS = ("v1.0-20240301", "v2.5-20260210")        # the rig endpoint has its OWN series
RIG_TYPES = ("biped", "quadruped", "hexapod", "octopod", "avian", "serpentine", "aquatic")
SPECS = ("mixamo", "tripo")
FORMATS = ("glb", "fbx", "obj", "stl", "usdz", "3mf", "gltf")
RIG_FORMATS = ("glb", "fbx")            # the only two `out_format` values the rig takes
NATIVE_FORMAT = "glb"                   # what every generation task delivers itself
TEXTURE_QUALITIES = ("standard", "detailed", "extreme")
TEXTURE_ALIGNMENTS = ("original_image", "geometry")
GEOMETRY_QUALITIES = ("standard", "detailed")
ORIENTATIONS = ("default", "align_image")
COMPRESS = ("", "geometry")             # "" = Tripo's own default (meshopt), key left out
FACE_MAX = {"v3.1-20260211": 1_500_000, "v3.0-20250812": 1_000_000, "v2.5-20250123": 500_000,
            "P1-20260311": 50_000, "P2-20260801": 50_000}
FACE_MAX_QUAD, _FACE_MIN, _FACE_MAX_UNKNOWN = 150_000, 100, 2_000_000
_RES_PX = {"standard": 2048, "detailed": 4096, "extreme": 8192}

# Admin defaults of a Tripo alias candidate (`cand["tripo"]["options"]`). Which of these a
# CLIENT may override is decided by the label table in build_request — the rest is
# admin-only, the counterpart of a ComfyUI `fixed` pin.
OPTION_DEFAULTS: dict = {
    "texture": True, "pbr": True, "texture_quality": "standard",
    "texture_alignment": "original_image", "geometry_quality": "standard",
    # Admin face budget: None = Tripo's adaptive default. Set → sent with every request,
    # but a client `input_face_num` still wins (a DEFAULT, not a pin — as with Meshy's
    # target_polycount, which a chained alias needs so the rigger gets a sane mesh).
    "face_limit": None, "quad": False, "smart_low_poly": False, "generate_parts": False,
    "auto_size": False, "orientation": "default", "enable_image_autofix": False, "compress": "",
    "target_formats": ["glb"], "thumbnail": True,
    "rig_check": True, "rig_model": "v1.0-20240301", "rig_type": "biped", "spec": "mixamo",
    "animations": [],
}
# Options copied into the generation body verbatim.
_PASSTHROUGH = ("texture", "pbr", "texture_quality", "texture_alignment", "geometry_quality",
                "quad", "smart_low_poly", "generate_parts", "auto_size", "orientation",
                "enable_image_autofix")
# Enum options and the tuple each is validated against (unknown value → the default).
_ENUMS = {"texture_quality": TEXTURE_QUALITIES, "texture_alignment": TEXTURE_ALIGNMENTS,
          "geometry_quality": GEOMETRY_QUALITIES, "orientation": ORIENTATIONS,
          "compress": COMPRESS, "spec": SPECS, "rig_type": RIG_TYPES, "rig_model": RIG_MODELS}

SLOTS: dict[str, list[str]] = {
    "image-to-model": ["input_image"],
    "multiview-to-model": ["input_image_front", "input_image_back",
                           "input_image_left", "input_image_right"],
    "rig": [],                          # takes a MESH, not images — see FILES
}
_VIEW_OF = {"input_image_front": "front", "input_image_back": "back",
            "input_image_left": "left", "input_image_right": "right"}
# File inputs per endpoint — the `files` half of the public fields. A file is NOT an image
# slot: no placeholder, no empty-mode; it is sent or the request is refused.
FILES: dict[str, list[str]] = {"image-to-model": [], "multiview-to-model": [],
                               "rig": ["input_mesh_path"]}
# Accepted, no effect: Tripo has no name field, and background/finger handling is its own.
IGNORED_PARAMS = ("input_name", "input_remove_background", "input_no_fingers")


class TripoInput(RuntimeError):
    """A request Tripo cannot run (missing image, unsupported format, bad enum, fewer
    than two views) — a content error, final, never failed over."""


def endpoint_of(cand: dict) -> str:
    ep = ((cand.get("tripo") or {}).get("endpoint") or "").strip()
    return ep if ep in ENDPOINTS else ENDPOINTS[0]


def _rig_types_for(rig_model: str) -> tuple:
    """Which rig types a RIG MODEL can actually do (API notes §7.2): `v1.0-20240301` rigs
    BIPEDS only, `v2.5-20260210` every creature. A cross-field rule, so it cannot live in
    the option form — that renders one field at a time — and must be enforced here, where
    the request builder, `options_of` and the advertised schema all read the same answer."""
    return ("biped",) if rig_model == RIG_MODELS[0] else RIG_TYPES


def _check_rig_type(opts: dict, values: dict) -> str:
    """The rig type a rig request will run with — or TripoInput naming the allowed set.

    REFUSED, not clamped to biped: a caller asking for a quadruped rig on a v1.0 alias
    must learn that the alias cannot do it, not receive a biped skeleton."""
    rt = values.get("input_rig_type")
    if rt in (None, ""):
        rt = opts["rig_type"]
    allowed = _rig_types_for(opts["rig_model"])
    if rt not in allowed:
        raise TripoInput(f"`input_rig_type` must be one of {', '.join(allowed)} "
                         f"(rig model {opts['rig_model']})")
    return rt


def check_rig_type(cand: dict, values: dict) -> str:
    """The same rule, reachable from the adapter BEFORE it uploads anything. Without it a
    request that can never run still pushes its mesh (up to 150 MB) into the Tripo
    account, where nobody cleans it up — build_request only refuses after the upload."""
    return _check_rig_type(options_of(cand), values)


def options_of(cand: dict) -> dict:
    """OPTION_DEFAULTS overlaid with the candidate's stored options (unknown keys dropped),
    then normalized. The console editor runs the SAME function on save, so a stored alias
    can never build a request the request builder would not have built itself."""
    stored = (cand.get("tripo") or {}).get("options") or {}
    out = dict(OPTION_DEFAULTS)
    for k in OPTION_DEFAULTS:
        if k in stored:
            out[k] = stored[k]
    for k, allowed in _ENUMS.items():
        if out[k] not in allowed:
            out[k] = OPTION_DEFAULTS[k]
    fmts = out["target_formats"] if isinstance(out["target_formats"], list) else []
    allowed_fmts = RIG_FORMATS if endpoint_of(cand) == RIG_ENDPOINT else FORMATS
    out["target_formats"] = [f for f in fmts if f in allowed_fmts] or ["glb"]
    for k in ("texture", "pbr", "quad", "smart_low_poly", "generate_parts", "auto_size",
              "enable_image_autofix", "thumbnail", "rig_check"):
        out[k] = bool(out[k])
    if out["rig_type"] not in _rig_types_for(out["rig_model"]):
        # Same class of documented incompatibility as generate_parts below: rig model v1.0
        # answers 400 for anything but a biped, and a two-click console configuration must
        # not be able to store one.
        out["rig_type"] = "biped"
    if out["generate_parts"]:
        # Tripo REJECTS the combination (segmented parts carry no texture, no quads), so a
        # stored alias must not be able to build a request that comes back 400.
        out["texture"] = out["pbr"] = out["quad"] = out["smart_low_poly"] = False
    anims = out["animations"] if isinstance(out["animations"], list) else []
    out["animations"] = [str(a).strip() for a in anims if str(a).strip()]
    out["face_limit"] = opt_face_limit(cand.get("model"), out["quad"], out.get("face_limit"))
    return out


def _face_cap(model, quad: bool) -> int:
    """The `face_limit` ceiling: per generation model, or the flat quad ceiling. An unknown
    model gets the widest documented cap — refusing a budget because the model list has
    aged would be worse than sending one Tripo itself clamps."""
    return FACE_MAX_QUAD if quad else FACE_MAX.get(model, _FACE_MAX_UNKNOWN)


def face_limit_for(model, quad: bool, v) -> Optional[int]:
    """The CLIENT's `input_face_num` → a `face_limit` in [100, cap], or None when the
    client sent nothing usable (then Tripo's adaptive default decides). Clamped, not
    refused: a caller asking for more faces than the model can make wants the maximum."""
    if v is None or v == "":
        return None
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return None
    return max(_FACE_MIN, min(_face_cap(model, quad), n))


def opt_face_limit(model, quad: bool, v) -> Optional[int]:
    """The ADMIN `face_limit` option, validated: an int in [100, cap] or None.

    Deliberately NOT clamped the way the client's `input_face_num` is (same reasoning as
    `meshy.opt_polycount`): a stored option is admin input that Save already vetted, so a
    value outside the range means the stored candidate is broken — falling back to Tripo's
    adaptive default is honest, silently rewriting an admin's number is not."""
    if v is None or v == "":
        return None
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return None
    return n if _FACE_MIN <= n <= _face_cap(model, quad) else None


def default_candidate(backend: str) -> dict:
    """The candidate the console creates when registering an alias on a Tripo backend."""
    # deepcopy, not dict(): a shallow copy hands every candidate the SAME `target_formats`
    # / `animations` list object as the module constant, so one in-place edit on a stored
    # candidate would rewrite the default for every future alias.
    return {"backend": backend, "task": "img2mesh", "model": AI_MODELS[0],
            "tripo": {"endpoint": ENDPOINTS[0], "options": copy.deepcopy(OPTION_DEFAULTS)}}


def texture_quality(px) -> str:
    """Public `input_texture_resolution` (pixels, as the ComfyUI aliases take it) → Tripo
    bucket: ≤2048 → standard, ≤4096 → detailed, else extreme. A bucket name passes through."""
    if isinstance(px, str) and px.strip().lower() in TEXTURE_QUALITIES:
        return px.strip().lower()
    try:
        n = int(float(px))
    except (TypeError, ValueError):
        return "standard"
    return "standard" if n <= 2048 else "detailed" if n <= 4096 else "extreme"


def image_ext(data: bytes) -> str:
    """The upload filename's extension for image bytes, sniffed by magic. Tripo's file
    endpoint takes PNG and JPEG; anything else is refused HERE, before the upload."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    raise TripoInput("Tripo accepts PNG or JPEG images only")


def mesh_ext(data: bytes) -> str:
    """Same for a mesh upload. Binary glTF is the only container the rig endpoint takes, so
    a mislabelled .glb (an OBJ someone renamed) is refused before 25 credits are spent."""
    if data[:4] == b"glTF":
        return "glb"
    raise TripoInput("Tripo rigging takes a binary glTF (.glb) mesh")


def _view_inputs(endpoint: str, images: dict) -> list:
    """`inputs` for the multiview endpoint: view-key objects in slot order. `front` is
    mandatory and Tripo refuses fewer than two views — both checked here, so the caller
    learns it from the gateway instead of from a rejected task."""
    slots = SLOTS[endpoint]
    if not images.get(slots[0]):
        raise TripoInput(f"`images.{slots[0]}` is required")
    views = [{_VIEW_OF[s]: images[s]} for s in slots if images.get(s)]
    if len(views) < 2:
        raise TripoInput("Tripo multiview needs at least two views (front plus one more)")
    return views


def build_request(cand: dict, values: dict, images: dict,
                  files: Optional[dict] = None) -> dict:
    """The JSON body for POST /v3/generation/<endpoint> or /v3/animations/rig.

    `values` is the flattened request (params + inputs, public labels as keys); `images`
    and `files` are {label: file_token} — the adapter uploads the bytes first (Tripo takes
    no base64). Admin options come from the candidate; the client may set only what the
    label table below names. Unknown params are ignored, as on every generation alias."""
    ep = endpoint_of(cand)
    opts = options_of(cand)
    if ep == RIG_ENDPOINT:
        # A different request entirely: mesh in, rig out. None of the generation options
        # apply, and the rig model is its own series (the option, not the alias's `model`).
        tok = (files or {}).get("input_mesh_path")
        if not tok:
            raise TripoInput("`files.input_mesh_path` is required")
        rt = _check_rig_type(opts, values)      # the adapter ran this before the upload
        return {"input": tok, "model": opts["rig_model"], "rig_type": rt,
                "spec": opts["spec"], "out_format": opts["target_formats"][0]}
    m = cand.get("model")
    body: dict = {"model": m if m in AI_MODELS else AI_MODELS[0]}
    if ep == "image-to-model":
        tok = images.get("input_image")
        if not tok:
            raise TripoInput("`images.input_image` is required")
        body["input"] = tok
    else:
        body["inputs"] = _view_inputs(ep, images)
    for k in _PASSTHROUGH:
        body[k] = opts[k]
    if opts["compress"]:                        # "" = Tripo's own default, key left out
        body["compress"] = opts["compress"]
    if opts["face_limit"] is not None:
        # Admin face budget — applied BEFORE the client label on purpose: a client
        # `input_face_num` below overwrites it, so the option is a DEFAULT, not a pin.
        body["face_limit"] = opts["face_limit"]
    # ── client-settable labels ──
    fn = face_limit_for(body["model"], opts["quad"], values.get("input_face_num"))
    if fn is not None:
        body["face_limit"] = fn
    tr = values.get("input_texture_resolution")
    if tr not in (None, ""):
        body["texture_quality"] = texture_quality(tr)
    return body


def build_rig_check(token: str) -> dict:
    """POST /v3/animations/rig-check — free, and the docs ask for it before every rig."""
    return {"input": token}


def build_convert(task_id: str, fmt: str, animated: bool) -> dict:
    """POST /v3/models/convert — every format but the native GLB is a paid task of its own.
    `format` is upper-case (the API rejects lower); `with_animation` carries the skeleton
    of a rigged model over, which is only ever wanted on the rig endpoint's extras."""
    return {"input": task_id, "format": str(fmt).upper(), "with_animation": bool(animated)}


def build_retarget(task_id: str, preset: str, fmt: str) -> dict:
    """POST /v3/animations/retarget — one clip, 10 credits. `input` must be the task id of
    a RIGGED model (a file token is not accepted here)."""
    return {"input": task_id, "animation": preset, "out_format": fmt}


def clip_name(preset: str) -> str:
    """A retarget preset → the delivered file's stem: `preset:walk` → `walk`,
    `preset:quadruped:walk` → `quadruped_walk` (a `:` is no filename character)."""
    name = str(preset).strip()
    if name.startswith("preset:"):
        name = name[len("preset:"):]
    return name.replace(":", "_")


def request_summary(body: dict) -> dict:
    """The request as recorded on the job. Nothing to redact — a Tripo body carries file
    TOKENS, not bytes — but it is a deep copy so the recorded meta cannot be changed by a
    later edit of the live body (`inputs` is a nested list)."""
    return copy.deepcopy(body)


def public_fields(cand: dict) -> tuple[list, list, list]:
    """(params, images, files) in the shape GET /v1/generations/{alias}/schema advertises.
    Image entries: {name, on_empty, required}; file entries: {name, required, accept};
    params: {name, type, default?, choices?}."""
    ep = endpoint_of(cand)
    opts = options_of(cand)
    images = [{"name": s, "on_empty": "required" if i == 0 else "skip", "required": i == 0}
              for i, s in enumerate(SLOTS[ep])]
    files = [{"name": n, "required": True, "accept": ["glb"]} for n in FILES[ep]]
    if ep == RIG_ENDPOINT:
        # A rig job is the mesh plus one knob. None of the generation labels exist here,
        # and advertising them would promise settings the request builder drops.
        return ([{"name": "input_rig_type", "type": "string", "default": opts["rig_type"],
                  "choices": list(_rig_types_for(opts["rig_model"]))},
                 {"name": "input_name", "type": "string", "default": ""},
                 {"name": "input_no_fingers", "type": "bool", "default": False}],
                images, files)
    params = [
        {"name": "input_name", "type": "string", "default": ""},
        # A `default` ONLY when the admin option is set: build_request then really does
        # apply that budget to a request that omits the label, so the schema may say so.
        # With the option blank Tripo's adaptive default decides, and advertising any
        # number would be a promise the request builder never keeps.
        {"name": "input_face_num", "type": "int",
         **({"default": opts["face_limit"]} if opts.get("face_limit") is not None else {})},
        {"name": "input_texture_resolution", "type": "int",
         "default": _RES_PX[opts["texture_quality"]]},
        {"name": "input_remove_background", "type": "bool", "default": True},
        {"name": "input_no_fingers", "type": "bool", "default": False},
    ]
    return params, images, files


TASK_STATUSES = ("queued", "running", "success", "failed", "cancelled")


def parse_task(task: dict, formats: list, endpoint: str = "image-to-model",
               options: Optional[dict] = None) -> TaskState:
    """Read a finished or running task object — `task` is the `data` OBJECT of
    `GET /v3/tasks/{id}` (the `{code, data}` envelope is the adapter's business).

    A generation task delivers exactly one file (`output.model_url`, always GLB); the
    extra formats and the animation clips are separate tasks the adapter appends. Hence
    `formats[0]` names the one download here — for the rig endpoint that is its
    `out_format`, and the stem says which it is (`model.glb` vs `rigged.glb`). The free
    `rig-check` is the one task that succeeds with NO file: it answers `riggable` /
    `rig_type` instead, and the adapter reads them off the same TaskState.

    TOTAL over what the API may answer: `progress` that is not an integer counts as 0 (a
    poll must not die on a cosmetic field), and any status outside TASK_STATUSES is
    terminal with an explaining error — V2 also knew `banned`/`expired`/`unknown` and the
    V3 docs do not say they are gone. Falling through as "not finished yet" would poll an
    unknown state until `max_wait`, holding the backend slot to learn nothing.

    `options` is the whole admin option block, which is what the shared cloud adapter
    hands every kind (the signature is the same for Meshy and Tripo). Tripo needs none of
    it today: its clips and extra formats are their own tasks, not fields of this one."""
    status = str(task.get("status") or "").lower()
    out = task.get("output") or {}
    try:
        progress = int(task.get("progress") or 0)
    except (TypeError, ValueError):
        progress = 0
    st = TaskState(status=status, progress=progress,
                   credits=task.get("credits_consumed"),
                   thumbnail=out.get("rendered_image_url") or None)
    if status not in TASK_STATUSES:
        st.error = f"unknown task status {status!r}"
        return st
    if status in ("queued", "running"):
        return st
    if status == "failed":
        msg = str(task.get("error_message") or "failed")
        code = task.get("error_code")
        st.error = f"{msg} (code {code})" if code not in (None, "") else msg
        # `st.retryable` stays False: unlike Meshy's documented `task_error.type`, Tripo's
        # V3 docs give no field separating its own faults from a rejected input, and
        # guessing one from `error_code` would risk re-submitting a task the vendor already
        # judged. Left final on purpose — fill this in when Tripo documents the split.
        return st
    if status == "cancelled":
        st.error = "cancelled"
        return st
    # ── success: what the task delivered depends on WHICH task it was ──
    if endpoint == RIG_CHECK_ENDPOINT:
        # A verdict, not a delivery: the rig-check succeeds WITHOUT a model_url, and the
        # answer the adapter needs (rig this mesh, yes/no, and as what) is the output.
        st.riggable = bool(out.get("riggable"))
        st.rig_type = out.get("rig_type") or None
        return st
    url = out.get("model_url")
    if not url:
        raise TripoInput("Tripo task succeeded but has no model_url")
    stem = "rigged" if endpoint == RIG_ENDPOINT else "model"
    st.downloads = [(f"{stem}.{formats[0] if formats else NATIVE_FORMAT}", url)]
    return st


# The console's option form for a Tripo alias — rendered and parsed by admin's cloud
# editor from THIS table (cloudtask.parse_options). Order = form order. An empty `label`
# means "same row as the bool field above".
OPTION_FIELDS: list = [
    {"key": "texture", "label": "texture", "type": "bool"},
    {"key": "pbr", "label": "", "type": "bool", "checkbox_text": "pbr (PBR maps)"},
    {"key": "texture_quality", "label": "texture quality", "type": "select", "choices": list(TEXTURE_QUALITIES),
     "hint": "Default when the client sends no <code>input_texture_resolution</code> (≤2048 → standard, "
             "≤4096 → detailed, else extreme). detailed +10, extreme +20 credits."},
    {"key": "texture_alignment", "label": "texture alignment", "type": "select", "choices": list(TEXTURE_ALIGNMENTS)},
    {"key": "geometry_quality", "label": "geometry quality", "type": "select", "choices": list(GEOMETRY_QUALITIES),
     "hint": "detailed: +20 credits, v3.0+ only."},
    {"key": "face_limit", "label": "face limit", "type": "int", "placeholder": "blank = Tripo adaptive default",
     "hint": "Face budget applied when the client sends no <code>input_face_num</code> (100 … the model's "
             "maximum: v3.1 1.5M, v3.0 1M, v2.5 500k, P-series 50k; 150k with quad). A client value still wins."},
    {"key": "quad", "label": "topology", "type": "bool", "checkbox_text": "quad (+5 credits; quads only travel well as FBX)"},
    {"key": "smart_low_poly", "label": "", "type": "bool", "checkbox_text": "smart_low_poly (+10 credits)"},
    {"key": "generate_parts", "label": "parts", "type": "bool",
     "checkbox_text": "generate_parts (+20 credits; forces texture/pbr/quad/smart_low_poly off — Tripo rejects the combination)"},
    {"key": "auto_size", "label": "input", "type": "bool", "checkbox_text": "auto_size (real-world scale)"},
    {"key": "enable_image_autofix", "label": "", "type": "bool"},
    {"key": "orientation", "label": "orientation", "type": "select", "choices": list(ORIENTATIONS)},
    {"key": "compress", "label": "compress", "type": "select", "choices": [("", "none"), ("geometry", "geometry")]},
    {"key": "thumbnail", "label": "thumbnail", "type": "bool",
     "checkbox_text": "deliver Tripo's rendered preview as an extra image artifact"},
    {"key": "rig_check", "label": "rig check", "type": "bool", "rig_only": True,
     "checkbox_text": "run the free rig-check first and refuse an unriggable mesh before the 25 credits"},
    {"key": "rig_model", "label": "rig model", "type": "select", "rig_only": True, "choices": list(RIG_MODELS),
     "hint": "v1.0: biped only, 90+ animation presets · v2.5: every rig type, 16 presets. "
             "Picking v1.0 forces <b>rig type</b> to biped — Tripo refuses any other, and a "
             "client <code>input_rig_type</code> that is not biped is then refused too."},
    {"key": "rig_type", "label": "rig type", "type": "select", "rig_only": True, "choices": list(RIG_TYPES),
     "hint": "Default; a client <code>input_rig_type</code> wins — but only a type the rig "
             "model above supports."},
    {"key": "spec", "label": "skeleton", "type": "select", "rig_only": True,
     "choices": [("mixamo", "mixamo — Mixamo-compatible bone names"), ("tripo", "tripo — Tripo's own skeleton")]},
    {"key": "animations", "label": "animations", "type": "list", "rig_only": True,
     "placeholder": "preset:walk, preset:run",
     "hint": "Retarget clips delivered as extra files (<code>walk.glb</code>, …), 10 credits each; "
             "the preset catalogue depends on the rig model — see the Tripo docs."},
]

BACKEND_HINT = (
    "<b>api key</b> (above) is the Tripo key (Tripo console → <b>API Keys</b>, "
    "<code>platform.tripo3d.ai/api-keys</code>). <b>max_concurrent</b> should stay at or below "
    "Tripo's per-account concurrency pool — <b>10</b> for the H-series models (v2.5/v3.0/v3.1) and "
    "for animation tasks, <b>5</b> for the P-series; beyond it Tripo answers 429 and the job fails "
    "over. <b>max wait s</b> caps the WHOLE generation, not one task: rig-check, the main task, "
    "every convert and every animation clip share it (blank = 900), so an alias with extra formats "
    "or clips needs a bigger one. <b>poll interval s</b> is the gap between task polls "
    "(blank = 2, the docs' recommendation). Credits: 30 per image/multiview task (20 without "
    "texture), 25 per rig, 10 per animation clip, 5 per format convert; 1 credit = 0.01 USD. The "
    "current balance shows in the backend list after the next health poll.")
ENDPOINT_HINT = (
    "<b>image-to-model</b> takes <code>input_image</code>; <b>multiview-to-model</b> takes "
    "<code>input_image_front</code> (required) plus at least ONE of "
    "<code>_back/_left/_right</code> — Tripo refuses a multiview job with fewer than two views. "
    "<b>rig</b> takes no image at all: it rigs an uploaded <code>input_mesh_path</code> (a "
    "<code>.glb</code>, 25 credits; the rig-check before it is free) and ignores every generation "
    "option below. <b>deliver formats</b>: Tripo generates <b>glb</b> and nothing else (the rig "
    "endpoint delivers the FIRST ticked format directly), so every other format is a separate "
    "convert task at 5 credits.")
CHAIN_HINT = (
    "The successor may itself be a <b>Tripo alias</b> (e.g. <code>Tripo-Rig</code>, endpoint "
    "<code>rig</code>) — it then takes the mesh as its file field <code>input_mesh_path</code>, the "
    "<b>delivered rig type</b> is <code>tripo</code>, and the bone names follow the rig alias's "
    "<b>skeleton</b> option (<code>mixamo</code> by default). Tripo embeds its texture in the GLB, "
    "so <b>keep from this stage</b> is usually empty — <code>preview.png</code> is the one "
    "candidate. Any OTHER format in <b>deliver formats</b> is wasted on a chained alias: each one "
    "is a paid convert task, but only the successor's result (plus what <b>keep from this "
    "stage</b> matches) is delivered.")
