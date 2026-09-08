"""Meshy.ai (https://docs.meshy.ai) as a generation backend — the PURE half.

Everything here is a function of dicts and bytes: the public `input_*` label table
(spec 2026-09-02 §5), the request body Meshy's image-to-3d / multi-image-to-3d /
rigging endpoints take, the task object they hand back, and the schema the gateway
advertises for a Meshy alias. No `main`/`adapters` imports, no I/O — the adapter in
`adapters.py` does the HTTP, this module decides WHAT to send and what came back.
Covered by test_meshy.py (stdlib unittest).
"""
from __future__ import annotations

import base64
import copy
from typing import Optional

from cloudtask import TaskState                # shared with tripo.py; re-exported for callers

# The cloud-kind interface every cloud task module declares (see the design spec
# 2026-09-03 §3.2): kind key, display name, default backend URL, and which of its
# ENDPOINTS rigs. Adapter, console editor and `main` read these instead of
# hard-coding "meshy" a second time for the next vendor.
KIND, VENDOR, URL = "meshy", "Meshy", "https://api.meshy.ai"
RIG_ENDPOINT = "rigging"
SUCCESS_STATUS = "SUCCEEDED"            # the one `TaskState.status` that means delivered
POLL_INTERVAL_DEFAULT = 5.0             # backend defaults when the form leaves them blank
MAX_WAIT_DEFAULT = 900

ENDPOINTS = ("image-to-3d", "multi-image-to-3d", "rigging")
AI_MODELS = ("latest", "meshy-7", "meshy-6", "meshy-5")
FORMATS = ("glb", "obj", "fbx", "stl", "usdz", "3mf")
RIG_FORMATS = ("glb", "fbx")            # the only two the rigging endpoint delivers
TOPOLOGIES = ("triangle", "quad")
TEXTURE_RES = ("2k", "4k", "8k")
POSES = ("", "a-pose", "t-pose")

# Admin defaults of a Meshy alias candidate (`cand["meshy"]["options"]`). Which of
# these a CLIENT may override is decided by the label table in build_request — the
# rest is admin-only, the counterpart of a ComfyUI `fixed` pin.
OPTION_DEFAULTS: dict = {
    "should_texture": True, "enable_pbr": False, "texture_resolution": "2k",
    "topology": "triangle", "should_remesh": None,          # None = Meshy's per-model default
    "ultra_mode": False, "pose_mode": "", "image_enhancement": True,
    "remove_lighting": True, "moderation": False,
    # Admin face budget: None = don't ask for one (Meshy's per-model default, i.e. no
    # remesh). Measured 2026-09-02 on prod: a humanoid generated WITHOUT a polycount
    # came back at 70 MB, which then blew up the chain's rigging hand-off — and Meshy's
    # rigging endpoint refuses more than 300k faces anyway. So a chained alias needs a
    # cap even when the client sends no `input_face_num`. Set → also forces should_remesh.
    "target_polycount": None,
    "target_formats": ["glb"], "thumbnail": True,
    "animations": False,            # rigging: also deliver Meshy's walking/running clips
}
# Options that are copied into the request body verbatim (None = leave it out).
_PASSTHROUGH = ("should_texture", "enable_pbr", "topology", "should_remesh", "ultra_mode",
                "image_enhancement", "remove_lighting", "moderation", "target_formats")

SLOTS: dict[str, list[str]] = {
    "image-to-3d": ["input_image"],
    "multi-image-to-3d": ["input_image_front", "input_image_back",
                          "input_image_left", "input_image_right"],
    "rigging": [],                  # takes a MESH, not images — see FILES
}
# File inputs per endpoint — the `files` half of the public fields. A file is NOT an
# image slot: no placeholder, no empty-mode; it is sent or the request is refused.
FILES: dict[str, list[str]] = {"image-to-3d": [], "multi-image-to-3d": [],
                               "rigging": ["input_mesh_path"]}
IGNORED_PARAMS = ("input_remove_background", "input_no_fingers")   # accepted, no effect

_POLY_MIN, _POLY_MAX = 100, 300_000
_HEIGHT_DEFAULT = 1.7               # Meshy's own default for `height_meters`
_NAME_MAX, _TEXTURE_PROMPT_MAX = 100, 800
_RES_PX = {"2k": 2048, "4k": 4096, "8k": 8192}


class MeshyInput(RuntimeError):
    """A request Meshy cannot run (missing front image, unsupported image format,
    bad enum) — a content error, final, never failed over."""


def endpoint_of(cand: dict) -> str:
    ep = ((cand.get("meshy") or {}).get("endpoint") or "").strip()
    return ep if ep in ENDPOINTS else ENDPOINTS[0]


def options_of(cand: dict) -> dict:
    """OPTION_DEFAULTS overlaid with the candidate's stored options (unknown keys dropped)."""
    stored = (cand.get("meshy") or {}).get("options") or {}
    out = dict(OPTION_DEFAULTS)
    for k in OPTION_DEFAULTS:
        if k in stored:
            out[k] = stored[k]
    if not isinstance(out["target_formats"], list) or not out["target_formats"]:
        out["target_formats"] = ["glb"]
    out["target_formats"] = [f for f in out["target_formats"] if f in FORMATS] or ["glb"]
    if endpoint_of(cand) == "rigging":       # rigging answers with glb/fbx and nothing else
        out["target_formats"] = [f for f in out["target_formats"] if f in RIG_FORMATS] or ["glb"]
    if out["texture_resolution"] not in TEXTURE_RES:
        out["texture_resolution"] = "2k"
    out["target_polycount"] = opt_polycount(out.get("target_polycount"))
    return out


def opt_polycount(v) -> Optional[int]:
    """The admin `target_polycount` option, validated: an int in [100, 300000] or None.

    Deliberately NOT clamped the way the client's `input_face_num` is: a stored option
    is admin input that Save already vetted, so a value outside the range here means the
    stored candidate is broken — ignoring it (falling back to Meshy's default) is honest,
    silently rewriting an admin's number to 300000 is not."""
    if v is None or v == "":
        return None
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return None
    return n if _POLY_MIN <= n <= _POLY_MAX else None


def default_candidate(backend: str) -> dict:
    """The candidate the console creates when registering an alias on a Meshy backend."""
    # deepcopy, not dict(): a shallow copy hands every candidate the SAME
    # `target_formats` list object as the module constant, so one in-place edit on a
    # stored candidate would rewrite the default for every future alias.
    return {"backend": backend, "task": "img2mesh", "model": "latest",
            "meshy": {"endpoint": ENDPOINTS[0], "options": copy.deepcopy(OPTION_DEFAULTS)}}


def texture_res(px) -> str:
    """Public `input_texture_resolution` (pixels, as the ComfyUI aliases take it) →
    Meshy bucket: ≤2048 → 2k, ≤4096 → 4k, else 8k. A bucket name passes through."""
    if isinstance(px, str) and px.strip().lower() in TEXTURE_RES:
        return px.strip().lower()
    try:
        n = int(float(px))
    except (TypeError, ValueError):
        return "2k"
    return "2k" if n <= 2048 else "4k" if n <= 4096 else "8k"


def data_uri(data: bytes) -> str:
    """Bytes → base64 data URI. Meshy accepts JPG/JPEG/PNG only; sniffed by magic."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif data[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    else:
        raise MeshyInput("Meshy accepts PNG or JPEG images only")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _slot_images(endpoint: str, images: dict) -> list[str]:
    """Data URIs in SLOT order (front first); empty slots dropped; first slot required."""
    slots = SLOTS[endpoint]
    present = [(s, images.get(s)) for s in slots if images.get(s)]
    if not present or present[0][0] != slots[0]:
        raise MeshyInput(f"`images.{slots[0]}` is required")
    return [data_uri(b) for _, b in present]


def glb_data_uri(data: bytes) -> str:
    """Mesh bytes → base64 data URI for `model_url`. A binary glTF is the only container
    the rigging endpoint takes, sniffed by magic — so a mislabelled .glb (an OBJ someone
    renamed) is refused HERE, not after 5 credits are spent."""
    if data[:4] != b"glTF":
        raise MeshyInput("Meshy rigging takes a binary glTF (.glb) mesh")
    return f"data:model/gltf-binary;base64,{base64.b64encode(data).decode()}"


def _build_rigging(cand: dict, values: dict, files: dict) -> dict:
    """POST /openapi/v1/rigging: the mesh, a height and an optional name. NONE of the
    image-to-3d options apply — no ai_model, no texturing, no target_formats (the
    formats are picked at DOWNLOAD time off the finished task)."""
    f = (files or {}).get("input_mesh_path")
    data = f[1] if isinstance(f, tuple) else f
    if not data:
        raise MeshyInput("`files.input_mesh_path` is required")
    body: dict = {"model_url": glb_data_uri(data)}
    h = values.get("input_height_m")
    try:
        body["height_meters"] = float(h) if h not in (None, "") else _HEIGHT_DEFAULT
    except (TypeError, ValueError):
        body["height_meters"] = _HEIGHT_DEFAULT
    name = values.get("input_name")
    if name not in (None, ""):
        body["name"] = str(name)[:_NAME_MAX]
    return body


def build_request(cand: dict, values: dict, images: dict, files: Optional[dict] = None) -> dict:
    """The JSON body for POST /openapi/v1/<endpoint>.

    `values` is the flattened request (params + inputs, public labels as keys);
    `images` is {label: bytes}. Admin options come from the candidate; the client may
    set only what the label table below names. Unknown params are ignored, as on
    every generation alias."""
    ep = endpoint_of(cand)
    if ep == "rigging":                 # a different request entirely: mesh in, rig out
        return _build_rigging(cand, values, files or {})
    opts = options_of(cand)
    m = cand.get("model")
    body: dict = {"ai_model": m if m in AI_MODELS else "latest"}
    uris = _slot_images(ep, images)
    if ep == "image-to-3d":
        body["image_url"] = uris[0]
    else:
        body["image_urls"] = uris
    for k in _PASSTHROUGH:
        if opts.get(k) is not None:
            body[k] = opts[k]
    body["texture_resolution"] = opts["texture_resolution"]
    if opts.get("pose_mode"):
        body["pose_mode"] = opts["pose_mode"]
    if opts.get("target_polycount") is not None:
        # Admin face budget — applied BEFORE the client block on purpose: a client
        # `input_face_num` below overwrites both keys, so the option is a DEFAULT, not a
        # pin. Forces the remesh pass, same as the client label (a polycount without one
        # is silently ignored by Meshy).
        body["target_polycount"] = opts["target_polycount"]
        body["should_remesh"] = True
    # ── client-settable labels ──
    name = values.get("input_name")
    if name not in (None, ""):
        body["name"] = str(name)[:_NAME_MAX]
    fn = values.get("input_face_num")
    if fn not in (None, ""):
        try:
            body["target_polycount"] = max(_POLY_MIN, min(_POLY_MAX, int(float(fn))))
            body["should_remesh"] = True          # a polycount only applies with a remesh pass
        except (TypeError, ValueError):
            pass
    tr = values.get("input_texture_resolution")
    if tr not in (None, ""):
        body["texture_resolution"] = texture_res(tr)
    tp = values.get("input_texture_prompt")
    if tp not in (None, ""):
        body["texture_prompt"] = str(tp)[:_TEXTURE_PROMPT_MAX]
    pose = values.get("input_pose")
    if pose not in (None, ""):
        if pose not in POSES:
            raise MeshyInput("`input_pose` must be one of " + ", ".join(p or "none" for p in POSES))
        body["pose_mode"] = pose
    return body


def request_summary(body: dict) -> dict:
    """The request as recorded on the job: image data replaced by its byte size."""
    def _sz(uri: str) -> str:
        try:
            raw = uri.split(",", 1)[1]
            return f"<{len(base64.b64decode(raw))} bytes>"
        except Exception:
            return "<image>"
    out = dict(body)
    if "image_url" in out:
        out["image_url"] = _sz(out["image_url"])
    if "image_urls" in out:
        out["image_urls"] = [_sz(u) for u in out["image_urls"]]
    if "model_url" in out:              # rigging: the whole mesh rides in the body
        out["model_url"] = _sz(out["model_url"])
    return out


def public_fields(cand: dict) -> tuple[list, list, list]:
    """(params, images, files) in the shape GET /v1/generations/{alias}/schema advertises.
    Image entries: {name, on_empty, required}; file entries: {name, required, accept};
    params: {name, type, default?, choices?}."""
    ep = endpoint_of(cand)
    opts = options_of(cand)
    images = [{"name": s, "on_empty": "required" if i == 0 else "skip", "required": i == 0}
              for i, s in enumerate(SLOTS[ep])]
    files = [{"name": n, "required": True, "accept": ["glb"]} for n in FILES[ep]]
    if ep == "rigging":
        # A rigging job is the mesh plus two knobs. None of the image-to-3d labels exist
        # here, and advertising them would promise settings the request builder drops.
        return ([{"name": "input_name", "type": "string", "default": ""},
                 {"name": "input_height_m", "type": "float", "default": _HEIGHT_DEFAULT},
                 {"name": "input_no_fingers", "type": "bool", "default": False}],
                images, files)
    params = [
        {"name": "input_name", "type": "string", "default": ""},
        # A `default` ONLY when the admin option is set: build_request then really does
        # apply that polycount (plus should_remesh) to a request that omits the label, so
        # the schema may say so. With the option blank nothing is applied — Meshy's own
        # per-model default decides — and advertising any number would be a promise the
        # request builder never keeps.
        {"name": "input_face_num", "type": "int",
         **({"default": opts["target_polycount"]} if opts.get("target_polycount") is not None else {})},
        {"name": "input_texture_resolution", "type": "int", "default": _RES_PX[opts["texture_resolution"]]},
        {"name": "input_texture_prompt", "type": "string", "default": ""},
        {"name": "input_pose", "type": "string", "default": opts.get("pose_mode") or "",
         "choices": list(POSES)},
        {"name": "input_remove_background", "type": "bool", "default": True},
        {"name": "input_no_fingers", "type": "bool", "default": False},
    ]
    return params, images, files


TASK_STATUSES = ("PENDING", "IN_PROGRESS", "SUCCEEDED", "FAILED", "CANCELED")

# `task_error.type` on a FAILED task — Meshy's own verdict on whether trying again can
# help (docs.meshy.ai/en/api/errors). `invalid_input` is permanent ("something is wrong
# with the input you provided"); these three are the service's own trouble and the docs
# say "retry the request" for each. Read the TYPE, never the message: the message is
# prose that changes, the type is the contract.
RETRYABLE_ERROR_TYPES = ("timeout", "service_unavailable", "server_error")


def parse_task(task: dict, formats: list, endpoint: str = "image-to-3d",
               animations: bool = False, options: Optional[dict] = None) -> TaskState:
    """Read a task object (GET …/{id}). On SUCCEEDED every requested format must have
    a URL — a missing one raises, never a silently smaller delivery.

    `endpoint` decides WHERE the urls sit (image-to-3d: `model_urls`; rigging: the
    `result` object's `rigged_character_<fmt>_url`) and what the delivered files are
    called (`model.glb` vs `rigged.glb`).

    TOTAL over what the API may answer: `progress` that is not an integer counts as 0
    (a poll must not die on a cosmetic field), and any status outside TASK_STATUSES is
    terminal-FAILED with an explaining error. Falling through as "not finished yet"
    would poll an unknown state until `max_wait` — holding the backend slot for the
    full wait to learn nothing.

    `options` is the whole admin option block, which is what the shared cloud adapter
    hands every kind (the signature is the same for Meshy and Tripo). Given, it decides
    the clips; the older `animations` bool stays for direct callers."""
    if options is not None:
        animations = bool(options.get("animations"))
    status = str(task.get("status") or "").upper()
    try:
        progress = int(task.get("progress") or 0)
    except (TypeError, ValueError):
        progress = 0
    st = TaskState(status=status, progress=progress,
                   credits=task.get("consumed_credits"),
                   thumbnail=task.get("thumbnail_url") or None)
    if status not in TASK_STATUSES:
        st.error = f"unknown task status {status!r}"
        return st
    if status in ("FAILED", "CANCELED"):
        err = task.get("task_error") or {}
        st.error = err.get("message") or status.lower()
        # CANCELED is a decision, not a fault — re-running it would defy whoever cancelled.
        st.retryable = (status == "FAILED"
                        and str(err.get("type") or "").lower() in RETRYABLE_ERROR_TYPES)
        return st
    if status == "SUCCEEDED":
        anim: dict = {}
        if endpoint == "rigging":           # the rigged character lives under `result`
            res = task.get("result") or {}
            urls = {f: res.get(f"rigged_character_{f}_url") for f in formats}
            anim = res.get("basic_animations") or {}
        else:
            urls = task.get("model_urls") or {}
        missing = [f for f in formats if not urls.get(f)]
        if missing:
            raise MeshyInput(f"Meshy task succeeded but has no url for {', '.join(missing)}")
        stem = "rigged" if endpoint == "rigging" else "model"
        st.downloads = [(f"{stem}.{f}", urls[f]) for f in formats]
        if endpoint == "rigging" and animations:
            # A courtesy, not the delivery: a clip Meshy did not produce is skipped —
            # unlike a missing FORMAT above, which IS the result and must not shrink.
            for f in formats:
                for clip in ("walking", "running"):
                    u = anim.get(f"{clip}_{f}_url")
                    if u:
                        st.downloads.append((f"{clip}.{f}", u))
    return st


# The console's option form for a Meshy alias — rendered and parsed by admin's cloud
# editor from THIS table (cloudtask.parse_options). Order = form order. An empty `label`
# means "same row as the bool field above", which is what keeps Meshy's grouped rows
# (`texture` = should_texture + enable_pbr, `input` = three boxes) as they are today.
OPTION_FIELDS: list = [
    {"key": "should_texture", "label": "texture", "type": "bool"},
    {"key": "enable_pbr", "label": "", "type": "bool", "checkbox_text": "enable_pbr (PBR maps)"},
    {"key": "texture_resolution", "label": "texture resolution", "type": "select",
     "choices": list(TEXTURE_RES),
     "hint": "Default when the client sends no <code>input_texture_resolution</code> "
             "(≤2048 → 2k, ≤4096 → 4k, else 8k). 4k/8k need Meshy-6+."},
    {"key": "topology", "label": "topology", "type": "select", "choices": list(TOPOLOGIES)},
    {"key": "should_remesh", "label": "remesh", "type": "tristate",
     "hint": "A client <code>input_face_num</code> always turns remesh on for that request "
             "(a polycount needs the remesh pass)."},
    {"key": "target_polycount", "label": "target polycount", "type": "int",
     "placeholder": "blank = Meshy default / no remesh",
     "hint": "Face budget applied when the client sends no <code>input_face_num</code> "
             "(100–300000; turns remesh on). A client value still wins. An alias that "
             "<b>chains into a rigger</b> should stay ≤ 300000: Meshy's rigging endpoint refuses "
             "more, and a no-remesh humanoid came back at <b>70 MB</b> (measured 2026-09-02), "
             "which is a mesh the hand-off then has to push through as base64."},
    {"key": "pose_mode", "label": "pose", "type": "select", "choices": [(p, p or "none") for p in POSES]},
    {"key": "image_enhancement", "label": "input", "type": "bool"},
    {"key": "remove_lighting", "label": "", "type": "bool"},
    {"key": "moderation", "label": "", "type": "bool"},
    {"key": "ultra_mode", "label": "ultra", "type": "bool", "checkbox_text": "ultra_mode (+5 credits, Meshy-7 only)"},
    {"key": "animations", "label": "animations", "type": "bool", "rig_only": True,
     "checkbox_text": "rigging only: also deliver walking/running clips"},
    {"key": "thumbnail", "label": "thumbnail", "type": "bool",
     "checkbox_text": "deliver Meshy's preview.png as an extra image artifact"},
]

BACKEND_HINT = (
    "<b>api key</b> (above) is the Meshy key (<code>msy_…</code>, Meshy dashboard → API). "
    "<b>max_concurrent</b> should stay at or below your Meshy tier's concurrent-task limit — "
    "this account is on <b>Pro (10)</b> (Studio 20 · Premium 30 · Ultra 100; the limit is shared "
    "by every key of the account) — beyond it Meshy answers 429 and the job fails over. "
    "<b>max wait s</b> caps one task incl. Meshy's own queue (blank = 900); <b>poll interval s</b> "
    "is the gap between task polls (blank = 5). Credits: 20 (no texture) / 30 (textured) / "
    "35 (8K) per Meshy-6/7 task, +5 ultra; refunded when a task fails. The current balance shows "
    "in the backend list after the next health poll.")
ENDPOINT_HINT = (
    "<b>image-to-3d</b> takes <code>input_image</code>; <b>multi-image-to-3d</b> takes "
    "<code>input_image_front</code> (required) plus optional <code>_back/_left/_right</code> — the "
    "same slot names as the Trellis2 multiview alias. <b>rigging</b> takes no image at all: it rigs "
    "an uploaded <code>input_mesh_path</code> (a <code>.glb</code> biped, 5 credits) and ignores every "
    "option below except <b>deliver formats</b> (glb/fbx) and <b>animations</b>.")
CHAIN_HINT = (
    "The successor may itself be a <b>Meshy alias</b> (e.g. <code>Meshy-Rig</code>, endpoint "
    "<code>rigging</code>) — it then takes the mesh as its file field <code>input_mesh_path</code>, "
    "and the <b>delivered rig type</b> is <code>meshy</code>. Meshy embeds its texture in the GLB, so "
    "<b>keep from this stage</b> is usually empty — <code>preview.png</code> is the one candidate. "
    "Any OTHER format in <b>deliver formats</b> is wasted on a chained alias: Meshy bills every one "
    "of them, but only the successor's result (plus what <b>keep from this stage</b> matches) is "
    "delivered.")
