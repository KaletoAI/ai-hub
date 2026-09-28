"""Which model files an alias candidate needs, and where they are in the source tree —
the pure half of the Thunder backend's model sync (the plan is built on these
functions, the controller runs it).

A synced backend is a rented GPU box with an empty disk: it gets exactly the files the
aliases that name it load, copied from the LAN model box (or a public URL). Everything
here guards a failure that looks like a working sync:

- **What a candidate loads** is its workflow AFTER this candidate's own `fixed` pins and
  WITHOUT its `bypass` nodes (`effective_workflow`) — the same two per-backend rules the
  adapter applies before submitting. Reading the raw workflow syncs the weights a pin
  replaced (tens of GB billed per hour) and misses the ones it put in.
- **Which inputs are references** (`refs_for`): loader detection as in
  `scheduler.model_set_key` (class /load/ minus the per-job input loaders; inputs that
  NAME a weight set), with a wider "how" exclusion — `format|quant|mode|type|scheme` on
  top of the scheduler's `dtype|precision|device|backend|attn`. Without it
  `Trellis2LoadModel_GGUF.model_format = "GGUF Q4_K_M"` became a reference no source can
  ever satisfy, and every alias on that node was blocked for good. Plus every string
  input of ANY class whose value ends in a model extension (`MODEL_EXT`): nodes that load
  weights themselves (`Hy3D21MeshGenerator.model`) are no "loader" by name.
- **What a reference can do** is its `kind`: `file` (a model extension or any `.xxx`
  ending) resolves against the source index; `hub` (an `org/repo` id) and `name` (a bare
  word a node resolves itself, e.g. a GGUF variant name) are catalog material only —
  they resolve solely through an exact file in the loader's own folder, never by suffix.
  The plan treats an unmatched `hub` ref as blocking and an unmatched `name` ref as
  nothing (it may just be an option value).
- **Resolution never guesses** (`resolve`): the loader class's folders in ComfyUI's own
  order, then a UNIQUE `/<value>` suffix anywhere under the two roots; two hits are
  `Ambiguous` (the panel lists them), none is `Missing`. Copying the wrong one of two
  same-named files gives a plausible, wrong result.
- **Two roots** (`models/…` ↔ `~/ComfyUI/models/…`, `hf-cache/…` ↔ `~/hf-cache/…`), and
  every path carries its root prefix. `hf-cache/token` (the Hugging Face token the
  source shares for its own downloads) and every dot segment are never resolved,
  expanded or accepted in the catalog: a path onto the token ships a credential to a
  machine we rent.
- **The catalog** (`modelsync_catalog` store setting) supplies what no workflow names:
  class+value entries for nodes that fetch their own model (hub ids), alias entries for
  aliases without any loader reference (`paths: []` says explicitly "needs nothing" —
  an EMPTY reference set is otherwise not "complete", it is "unknown"), and
  `file`+`url` entries for a public download source. `validate_catalog` refuses it out
  loud: a typo'd key would otherwise be ignored, and the alias it was meant for stays
  blocked with no hint why. `catalog_paths`/`url_catalog` drop every entry the
  validator would refuse, so an unvalidated catalog can never expand a whole root.
- **Per-job assets never block**: an asset extension (`.glb`, `.png`, `.mp4` …) is no
  model file on any node, and core `Load3D` is an input loader — `Load3D.model_file =
  "x.glb"` otherwise blocked the alias on a file the client uploads.
- **The plan** (`plan`, `ready`, `status_text`): present = the destination holds the
  file at the SOURCE's size (the manifest's, where the source does not list it), a
  `.part` never is; `prune` is manifest files only (never a file we did not put there),
  `unknown` everything else nobody needs — listed, never deleted automatically. A
  BLOCKED alias fetches nothing (it cannot become ready) but HOLDS the manifest files
  recorded for it (`held`, never pruned) until the block is fixed. An alias is ready only with nothing missing and nothing blocking it; an empty reference
  set without an explicit alias entry is blocked, never "complete".

Pure: stdlib only (plus `sshrun.safe_rel`, itself pure), no `main`/`adapters` imports,
no I/O, no module-level config. Covered by tests/test_modelsync.py.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Union

from sshrun import safe_rel

MODEL_EXT = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".sft")

ROOTS = ("models/", "hf-cache/")
# The source's Hugging Face token share: never a sync path, never a resolution.
TOKEN_PATH = "hf-cache/token"
# The Hugging Face cache's own tree: as a catalog directory it is every Hub model the
# source ever downloaded — a whole root in all but name.
_HF_HUB = "hf-cache/hub/"

# Loader detection — the same three rules as scheduler.model_set_key, kept here (not
# imported) because the "how" list below is deliberately WIDER than the scheduler's: a
# missed "how" there costs one skipped VRAM free, here it blocks an alias forever.
_LOADER_CLASS = re.compile(r"load", re.I)
# `load_?\s*3d`: core `Load3D.model_file = "x.glb"` is a per-job asset, not a weight. Not a
# bare `3d` — `Hy3D21VAELoader` is a model loader whose name merely contains it.
_INPUT_LOADER = re.compile(r"image|mask|mesh|path|video|audio|load_?\s*3d", re.I)
_WEIGHT_INPUT = re.compile(r"name|model|ckpt|unet|clip|vae|lora|gguf|weight", re.I)
# `mode(?!l)`: `attention_mode` is a how, `model`/`modelname` are exactly what we want.
_HOW_INPUT = re.compile(r"dtype|precision|device|backend|attn|format|quant|mode(?!l)|type|scheme", re.I)
# A file ending on the LAST segment: a letter first, so a version (`Model2.5`,
# `org/Model-2.5-VL`) is no extension.
_FILE_ENDING = re.compile(r"\.[A-Za-z][A-Za-z0-9_]{0,7}$")

# Per-job assets (a mesh, an image, a clip) that a loader-named node may take: never a
# model file, whatever the field is called — `Load3D.model_file = "x.glb"` blocked an
# alias with "not in source" for a file the CLIENT supplies.
ASSET_EXT = (".glb", ".gltf", ".obj", ".fbx", ".ply", ".stl", ".png", ".jpg", ".jpeg", ".webp",
             ".gif", ".mp4", ".webm", ".wav", ".mp3", ".flac")

KINDS = ("file", "hub", "name")

# Loader class → folders under models/, in the order ComfyUI searches them (the first
# is where current ComfyUI puts new files, the second the legacy name it still reads).
FOLDERS: dict[str, tuple[str, ...]] = {
    "UNETLoader": ("diffusion_models", "unet"),
    "UnetLoaderGGUF": ("diffusion_models", "unet"),
    "UnetLoaderGGUFAdvanced": ("diffusion_models", "unet"),
    "LoaderGGUF": ("diffusion_models", "unet"),
    "CLIPLoader": ("text_encoders", "clip"),
    "DualCLIPLoader": ("text_encoders", "clip"),
    "TripleCLIPLoader": ("text_encoders", "clip"),
    "QuadrupleCLIPLoader": ("text_encoders", "clip"),
    "CLIPLoaderGGUF": ("text_encoders", "clip"),
    "ClipLoaderGGUF": ("text_encoders", "clip"),
    "DualCLIPLoaderGGUF": ("text_encoders", "clip"),
    "TripleCLIPLoaderGGUF": ("text_encoders", "clip"),
    "QuadrupleCLIPLoaderGGUF": ("text_encoders", "clip"),
    "VAELoader": ("vae",),
    "CheckpointLoaderSimple": ("checkpoints",),
    "CheckpointLoader": ("checkpoints",),
    "ImageOnlyCheckpointLoader": ("checkpoints",),
    "unCLIPCheckpointLoader": ("checkpoints",),
    "UpscaleModelLoader": ("upscale_models",),
    "CLIPVisionLoader": ("clip_vision",),
    "ControlNetLoader": ("controlnet",),
    "StyleModelLoader": ("style_models",),
    "LoadBackgroundRemovalModel": ("background_removal", "rembg"),
}
_LORA_FOLDERS = ("loras",)

# Pre-filled catalog for a fresh install — copied into the `modelsync_catalog` setting on
# first read, after which the setting is authoritative (edited in the Backends tab). Base
# and hub models ONLY: the repo is public, a private model or LoRA name never goes here.
# Derived from the 3D node packs' own loaders (read 2026-09-28 on the LAN model box, which
# holds the tree these paths name): what each loader value makes its node read at run
# time, beside the one directory the value names. Every entry is class+value — a hub id is
# covered by nothing less (Ruling 17). Directories end in `/` (recursive; `.cache` and
# other dot segments never sync). Must always pass validate_catalog (pinned by a test).
#
# ComfyUI-Trellis2-vb `Trellis2LoadModel`: refuses to run without DINOv3 under
# models/facebook/, always fetches TRELLIS-image-large's sparse-structure decoder into
# models/microsoft/TRELLIS-image-large/ckpts/, and copies reconviagen_pipeline.json INTO
# models/microsoft/TRELLIS.2-4B/ whatever model is chosen (that directory must exist);
# Pixal3D models also load MoGe (models/Ruicheng/moge-2-vitl/) for the camera estimate.
_DINOV3 = "models/facebook/dinov3-vitl16-pretrain-lvd1689m/"
_MOGE = "models/Ruicheng/moge-2-vitl/"
_TRELLIS1_SS_DEC = "models/microsoft/TRELLIS-image-large/ckpts/"
# ComfyUI-Trellis2-GGUF `Trellis2LoadModel_GGUF` = "Pixal3D-GGUF": models/Pixal3D-GGUF/
# holds every quantisation; the aliases pin `GGUF Q4_K_M`, so only that one syncs (the
# other two are ~9 GB more). A different model_format needs its files added here.
_PIXAL_GGUF = "models/Pixal3D-GGUF/"
_PIXAL_GGUF_Q4 = [_PIXAL_GGUF + f"{d}/{n}{x}"
                  for d, n in (("Sparse", "ss_flow_img_dit_1_3B_64_bf16"),
                               ("shape", "slat_flow_img2shape_dit_1_3B_512_bf16"),
                               ("shape", "slat_flow_img2shape_dit_1_3B_1024_bf16"),
                               ("texture", "slat_flow_imgshape2tex_dit_1_3B_1024_bf16"))
                  for x in (".json", "_Q4_K_M.gguf")]
DEFAULT_CATALOG: list[dict] = [
    {"match": {"class": "Trellis2LoadModel", "value": "microsoft/TRELLIS.2-4B"},
     "paths": ["models/microsoft/TRELLIS.2-4B/", _TRELLIS1_SS_DEC, _DINOV3]},
    {"match": {"class": "Trellis2LoadModel", "value": "TencentARC/Pixal3D-T"},
     "paths": ["models/TencentARC/Pixal3D-T/", _TRELLIS1_SS_DEC, _DINOV3, _MOGE,
               "models/microsoft/TRELLIS.2-4B/reconviagen_pipeline.json"]},
    {"match": {"class": "Trellis2LoadModel_GGUF", "value": "Pixal3D-GGUF"},
     "paths": [_PIXAL_GGUF + "pipeline.json", _PIXAL_GGUF + "decoder/",
               _PIXAL_GGUF + "encoders/", *_PIXAL_GGUF_Q4, _DINOV3, _MOGE]},
    # ComfyUI-StableXWrapper: models/diffusers/<value>/ (snapshot of Stable-X/<value>)
    {"match": {"class": "DownloadAndLoadStableXModel", "value": "yoso-normal-v1-8-1"},
     "paths": ["models/diffusers/yoso-normal-v1-8-1/"]},
    # ComfyUI-Hunyuan3d-2-1: the texture stage (Hy3DMultiViewsGenerator, no loader input
    # of its own) loads tencent/Hunyuan3D-2.1's paint model and facebook/dinov2-giant from
    # the Hugging Face cache (HF_HOME) — anchored on the shape model every such workflow
    # names. Whole repo dirs: blobs + snapshots + refs are what the HF loader reads.
    {"match": {"class": "Hy3D21MeshGenerator", "value": "hunyuan3D-dit-v2-1-fp16.ckpt"},
     "paths": ["hf-cache/hub/models--tencent--Hunyuan3D-2.1/",
               "hf-cache/hub/models--facebook--dinov2-giant/"]},
]


def ref_kind(value: str) -> str:
    """`file` | `hub` | `name` for a reference value (see the module docstring). A model
    extension is always a file; an asset extension never is (`name`: ignored unless a
    catalog entry names it); with a `/` it is a hub id — `org/repo.v2` included — and
    only without one does any other `.xxx` ending make a file."""
    v = str(value or "")
    low = v.lower()
    if low.endswith(MODEL_EXT):
        return "file"
    if low.endswith(ASSET_EXT):
        return "name"
    if "/" in v:
        return "hub"
    if _FILE_ENDING.search(v):
        return "file"
    return "name"


@dataclass(frozen=True)
class Ref:
    """One model reference: which node/field names which value. `selectable` = the
    field is client-choosable in the mapping (only its default is synced). `kind` is
    derived from the value when not given."""
    node: str
    cls: str
    field: str
    value: str
    selectable: bool = False
    kind: str = ""

    def __post_init__(self):
        if not self.kind:
            object.__setattr__(self, "kind", ref_kind(self.value))
        elif self.kind not in KINDS:
            raise ValueError(f"unknown ref kind {self.kind!r}")


@dataclass
class Ambiguous:
    options: list[str] = field(default_factory=list)


@dataclass
class Missing:
    pass


# --- the effective workflow -------------------------------------------------------------

def _coerce(value, current):
    """adapters._coerce: a pinned form string takes the type of the field it replaces."""
    if isinstance(current, bool):
        return str(value).strip().lower() in ("true", "1", "yes", "on")
    if isinstance(current, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if isinstance(current, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return value


def _pins(fixed) -> list[dict]:
    """The candidate's pins as `[{node, field, value}]`. The store and config write the
    list form; a dict — `{"<node>.<field>": value}` or `{node: {field: value}}` — is
    read too. Non-dict list entries are skipped, as the adapter's callers do."""
    if isinstance(fixed, dict):
        out = []
        for k, v in fixed.items():
            if isinstance(v, dict):
                out.extend({"node": str(k), "field": f, "value": fv} for f, fv in v.items())
            elif "." in str(k):
                n, f = str(k).split(".", 1)
                out.append({"node": n, "field": f, "value": v})
        return out
    return [b for b in (fixed or []) if isinstance(b, dict)]


def effective_workflow(workflow: dict, fixed, bypass) -> dict:
    """What this candidate actually submits, as far as WHAT it loads goes: a deep copy
    with its pins applied (exactly `adapters._apply_fixed`'s rule — a pin with a blank
    value or an unknown node is dropped there, so it is dropped here) and its bypassed
    nodes removed (a bypassed loader loads nothing; the rewiring is irrelevant here)."""
    wf = copy.deepcopy(workflow or {})
    for b in _pins(fixed):
        node, fieldn, value = b.get("node"), b.get("field"), b.get("value")
        if node in wf and fieldn and value is not None and value != "" and isinstance(wf[node], dict):
            inputs = wf[node].setdefault("inputs", {})
            inputs[fieldn] = _coerce(value, inputs.get(fieldn))
    for nid in {str(x) for x in (bypass or [])}:
        wf.pop(nid, None)
    return wf


def refs_for(candidate: dict, workflow, mapping_fields) -> list[Ref]:
    """The model references of one alias candidate. `workflow` is the caller's
    `adapters.cand_workflow` result (None = unknown → no references; the plan then
    reports the alias as blocked, never as complete). `mapping_fields` are the
    `(node, field)` pairs the mapping lets a client choose — their Ref is `selectable`."""
    if not isinstance(workflow, dict):
        return []
    cand = candidate or {}
    wf = effective_workflow(workflow, cand.get("fixed"), cand.get("bypass"))
    sel = {(str(n), str(f)) for n, f in (mapping_fields or ())}
    out: list[Ref] = []
    for nid, node in wf.items():
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type") or "")
        loader = bool(_LOADER_CLASS.search(cls)) and not _INPUT_LOADER.search(cls)
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for k, v in inputs.items():
            # links are lists, strengths/flags are numbers/bools: never a weight name
            if not isinstance(v, str) or not v.strip() or v.strip().lower() == "none":
                continue
            if v.lower().endswith(MODEL_EXT):
                kind = "file"                        # any class, any field
            elif loader and _WEIGHT_INPUT.search(k) and not _HOW_INPUT.search(k):
                kind = ref_kind(v)
            else:
                continue
            out.append(Ref(str(nid), cls, str(k), v, (str(nid), str(k)) in sel, kind))
    return out


# --- the catalog ---------------------------------------------------------------------------

_MATCH_KEYS = ("alias", "class", "value")


def _match_entries(catalog):
    """The catalog's VALID match+paths entries. Anything `validate_catalog` would refuse
    is dropped whole — a catalog that reached us unvalidated (an old store row, a
    hand-edited setting) must never expand `models/` or `hf-cache/hub/` into a sync of
    the entire tree, billed per GB on a rented disk."""
    for e in catalog if isinstance(catalog, list) else []:
        if isinstance(e, dict) and "match" in e and not validate_catalog([e]):
            yield e


def _matched_refs(m: dict, refs) -> list[Ref]:
    return [r for r in refs or () if ("class" not in m or r.cls == m["class"])
            and ("value" not in m or r.value == m["value"])]


def catalog_paths(refs, alias: str, catalog) -> tuple[list[str], bool]:
    """The catalog paths this alias needs beyond its resolved references, and whether an
    `alias` entry names it (`explicit_alias_entry` — with `paths: []` that is the
    declaration "needs nothing"). Every key of an entry's `match` must hold: `alias`
    against the alias, `class`/`value` against ONE of the refs. Invalid entries are
    skipped here; `validate_catalog` is what reports them."""
    paths: list[str] = []
    explicit = False
    for e in _match_entries(catalog):
        m = e["match"]
        if "alias" in m and m["alias"] != alias:
            continue
        if ("class" in m or "value" in m) and not _matched_refs(m, refs):
            continue
        explicit = explicit or "alias" in m
        for p in e["paths"]:
            if p not in paths:
                paths.append(p)
    return paths, explicit


def url_catalog(catalog) -> dict:
    """`{path: {"url", "sha256"?}}` from the catalog's valid `file`+`url` entries — the
    public download sources the plan prefers over the LAN (a later entry for the same
    file wins, as a later line in the editor would be expected to)."""
    out: dict = {}
    for e in catalog if isinstance(catalog, list) else []:
        if isinstance(e, dict) and "file" in e and not validate_catalog([e]):
            out[e["file"]] = {"url": e["url"], **({"sha256": e["sha256"]} if "sha256" in e else {})}
    return out


def _root_path_error(p, want_dir) -> str:
    """Why `p` is no valid catalog path ('' = valid). `want_dir`: True = any path (a
    trailing `/` means a directory), False = a file (no trailing `/`)."""
    if not isinstance(p, str):
        return f"path {p!r} is not a string"
    try:
        safe_rel(p)
    except ValueError as e:
        return str(e)
    if not p.startswith(ROOTS):
        return f"path {p!r} must start with models/ or hf-cache/"
    if p in ROOTS or p == _HF_HUB:
        return f"path {p!r} is a whole root"
    if p == TOKEN_PATH or p.startswith(TOKEN_PATH + "/"):
        return f"path {p!r} is the token share"
    if want_dir is False and p.endswith("/"):
        return f"file {p!r} ends in / (a directory)"
    return ""


def _url_error(u) -> str:
    if not isinstance(u, str) or not u.startswith("https://") or len(u) <= len("https://"):
        return f"url {u!r} must be https://"
    # the URL reaches curl through a --config stdin: a newline or quote there would
    # become a config line of its own (a header, an output path)
    if any(ord(c) <= 32 or ord(c) == 127 or c in "\"\\" for c in u):
        return f"url {u!r} contains whitespace, a quote or a control character"
    return ""


def validate_catalog(obj) -> list[str]:
    """Every problem of a catalog, as readable lines (empty = valid)."""
    if not isinstance(obj, list):
        return ["catalog must be a JSON list"]
    errs: list[str] = []
    for i, e in enumerate(obj, 1):
        where = f"entry {i}"
        if not isinstance(e, dict):
            errs.append(f"{where}: not an object")
            continue
        keys = set(e)
        if keys & {"match", "paths"} and keys & {"file", "url", "sha256"}:
            errs.append(f"{where}: either match+paths or file+url, not both")
            continue
        if keys & {"file", "url", "sha256"}:
            extra = keys - {"file", "url", "sha256"}
            if extra:
                errs.append(f"{where}: unknown key(s) {sorted(extra)}")
            if "file" not in e or "url" not in e:
                errs.append(f"{where}: a source entry needs file and url")
            if "file" in e:
                msg = _root_path_error(e["file"], False)
                if msg:
                    errs.append(f"{where}: {msg}")
            if "url" in e:
                msg = _url_error(e["url"])
                if msg:
                    errs.append(f"{where}: {msg}")
            if "sha256" in e and not (isinstance(e["sha256"], str)
                                      and re.fullmatch(r"[0-9a-fA-F]{64}", e["sha256"])):
                errs.append(f"{where}: sha256 must be 64 hex characters")
            continue
        extra = keys - {"match", "paths"}
        if extra:
            errs.append(f"{where}: unknown key(s) {sorted(extra)}")
        m = e.get("match")
        if not isinstance(m, dict) or not m:
            errs.append(f"{where}: match must be a non-empty object")
        else:
            bad = set(m) - set(_MATCH_KEYS)
            if bad:
                errs.append(f"{where}: unknown match key(s) {sorted(bad)}")
            for k in _MATCH_KEYS:
                if k in m and (not isinstance(m[k], str) or not m[k]):
                    errs.append(f"{where}: match.{k} must be a non-empty string")
        ps = e.get("paths")
        if not isinstance(ps, list):
            errs.append(f"{where}: paths must be a list")
        else:
            for p in ps:
                msg = _root_path_error(p, True)
                if msg:
                    errs.append(f"{where}: {msg}")
    return errs


# --- resolution against the source index ------------------------------------------------

def _usable(key: str) -> bool:
    """An index key that may ever be synced: under a root, no dot segment, not the
    token share."""
    if not isinstance(key, str) or not key.startswith(ROOTS) or key.endswith("/"):
        return False
    if key == TOKEN_PATH or key.startswith(TOKEN_PATH + "/"):
        return False
    try:
        safe_rel(key)
    except ValueError:
        return False
    return True


def folders_for(cls: str) -> tuple[str, ...]:
    """The `models/` subfolders a loader class reads, in ComfyUI's order (() = unknown)."""
    if cls in FOLDERS:
        return FOLDERS[cls]
    if "lora" in str(cls or "").lower():
        return _LORA_FOLDERS
    return ()


def resolve(ref: Ref, source_index: dict) -> Union[str, Ambiguous, Missing]:
    """The source path of a reference: the first `models/<folder>/<value>` of the
    loader's folders that exists, else — for a `file` ref only — the ONE index key
    ending in `/<value>`; several are `Ambiguous`, none is `Missing`. A value that is no
    safe relative path (`..`, absolute, backslash) is `Missing`: ComfyUI's own lookup
    would not reach outside its folder either."""
    v = ref.value
    try:
        safe_rel(v)
    except ValueError:
        return Missing()
    if v.endswith("/"):
        return Missing()
    idx = source_index or {}
    for f in folders_for(ref.cls):
        k = f"models/{f}/{v}"
        if k in idx and _usable(k):
            return k
    if ref.kind != "file":
        return Missing()                     # hub ids and bare names are catalog-only
    hits = sorted(k for k in idx if k.endswith("/" + v) and _usable(k))
    if len(hits) == 1:
        return hits[0]
    if hits:
        return Ambiguous(hits)
    return Missing()


def expand_dir(prefix: str, source_index: dict) -> dict:
    """A catalog path's files with their sizes: a path ending in `/` is a directory
    (every file under it, recursively), anything else the one file (or nothing). The
    token share and dot segments never come out, whatever the prefix."""
    try:
        safe_rel(prefix)
    except ValueError:
        return {}
    idx = source_index or {}
    if prefix.endswith("/"):
        return {k: s for k, s in idx.items() if k.startswith(prefix) and _usable(k)}
    return {prefix: idx[prefix]} if prefix in idx and _usable(prefix) else {}



# --- the plan ---------------------------------------------------------------------------

# Transfer artefacts next to a target (`<path>.part` + its `.lock`/`.log`, see the
# controller): never present, never "unknown" — a half-downloaded file is neither a
# model nor a stranger to delete.
_PART_SUFFIXES = (".part", ".part.lock", ".part.log")


@dataclass
class AliasNeed:
    """What one alias needs on one backend: its candidate's refs, the catalog paths
    (`catalog_paths`), whether an alias entry names it (`explicit`), and `covered` — the
    `(node, field)` of refs a class/value catalog entry matched (their files come from
    that entry, so an unresolved one blocks nothing). Build it with `alias_need`."""
    alias: str
    refs: list
    catalog: list
    explicit: bool
    covered: frozenset = frozenset()


def alias_need(alias: str, refs, catalog) -> AliasNeed:
    """An `AliasNeed` from a candidate's refs and the whole catalog."""
    refs = list(refs or ())
    paths, explicit = catalog_paths(refs, alias, catalog)
    covered = set()
    for e in _match_entries(catalog):
        m = e["match"]
        if ("class" in m or "value" in m) and ("alias" not in m or m["alias"] == alias):
            # A hub id is covered only by an entry naming its class AND its value: a
            # class-only entry would also "cover" a hub id the loader is later switched
            # to, whose weights that entry does not sync — ready, and wrong (Ruling 17).
            full = "class" in m and "value" in m
            covered.update((r.node, r.field) for r in _matched_refs(m, refs)
                           if full or r.kind != "hub")
    return AliasNeed(alias, refs, paths, explicit, frozenset(covered))


def _is_part(p: str) -> bool:
    return p.endswith(_PART_SUFFIXES)


def _msize(entry):
    s = entry.get("size") if isinstance(entry, dict) else None
    return s if isinstance(s, int) and not isinstance(s, bool) and s >= 0 else None


def _merge_needs(needs) -> list:
    """One AliasNeed per alias name, sorted: the same alias twice (two candidates on this
    backend, a caller bug) would otherwise overwrite its own per_alias row while both
    counted into need_total."""
    by: dict = {}
    for n in needs or ():
        m = by.get(n.alias)
        if m is None:
            by[n.alias] = AliasNeed(n.alias, list(n.refs or ()), list(n.catalog or ()),
                                    bool(n.explicit), frozenset(n.covered or ()))
            continue
        m.refs.extend(n.refs or ())
        m.catalog.extend(p for p in n.catalog or () if p not in m.catalog)
        m.explicit = m.explicit or bool(n.explicit)
        m.covered = m.covered | frozenset(n.covered or ())
    return [by[a] for a in sorted(by)]


def plan(needs, source_index: dict, dest_index: dict, manifest: dict, url_catalog: dict) -> dict:
    """The sync plan of one backend (see the module docstring for the rules):

    `per_alias[alias]` — `files` (`path`, `size`, `node`/`cls` of the ref that needs it,
    None for a catalog path, `present`), `need_bytes`/`have_bytes` (known sizes),
    `missing` (not present), `blocked` (why the alias cannot become ready by fetching),
    `hints` (uncovered loader values beside real files: synced nothing, may need
    weights), `selectable` (`node.field` whose value a client may change — only the
    default syncs).
    `fetch` — what to transfer, in order: UNBLOCKED aliases by missing bytes ascending,
    large files first within one (unknown sizes last), each file once with every alias
    it is fetched for; `source` is `url` when the catalog has a public URL, else `lan`.
    A blocked alias fetches nothing — it cannot become ready, and its bytes cost disk.
    `prune` — manifest files no alias needs (only what WE synced is ever deleted), minus
    `held`: `[path, size, alias]` of manifest files recorded for an alias that is
    selected but BLOCKED — a block (a new same-named copy in the source, a removed
    catalog entry) must not make the stop delete what the alias already synced.
    `unknown` — destination files neither in the manifest nor needed, `[path, size]`.

    Resolution runs over the source index plus the URL catalog and the manifest: a file
    the manifest verified still counts when the source does not list it (spec: "bzw. im
    Manifest verifiziert") — a LAN box that is down must not turn every synced file into
    one the stop prunes. Its size is the source's, else the manifest's; a URL file never
    downloaded has size None until the manifest records one. Deterministic; the inputs
    are not modified."""
    src = {k: v for k, v in (source_index or {}).items() if _usable(k) and not _is_part(k)}
    urls = {k: v for k, v in (url_catalog or {}).items()
            if _usable(k) and not _is_part(k) and isinstance(v, dict) and v.get("url")}
    man = {k: v for k, v in (manifest or {}).items() if _usable(k) and not _is_part(k)}
    dest = dest_index or {}
    # the resolution index: every path we could name, the source's size where it has one
    index = {k: None for k in list(urls) + list(man)}
    index.update(src)

    def size_of(p):
        return src[p] if p in src else _msize(man.get(p))

    def present(p, size):
        return size is not None and not _is_part(p) and dest.get(p) == size

    per_alias: dict = {}
    wanted: dict = {}                       # unblocked alias -> [path] it must fetch
    needed: set = set()
    for n in _merge_needs(needs):
        files: dict = {}
        blocked: list = []
        loose: list = []                    # uncovered name refs: nothing synced for them
        for r in n.refs:
            got = resolve(r, index)
            if isinstance(got, str):
                files.setdefault(got, (r.node, r.cls))
            elif isinstance(got, Ambiguous):
                blocked.append(f"ambiguous {r.value}: {', '.join(got.options)}")
            elif (r.node, r.field) in n.covered:
                continue                    # its files come from the matching catalog entry
            elif r.kind == "name":
                if not r.value.lower().endswith(ASSET_EXT):
                    loose.append(r)         # an asset is a per-job input, not a weight
            elif r.kind == "hub":
                blocked.append(f"unknown hub model {r.value} — add a catalog entry")
            else:
                blocked.append(f"not in source: {r.value}")
        for cp in n.catalog:
            err = _root_path_error(cp, True)
            if err:
                blocked.append(f"invalid catalog path: {err}")
                continue
            got = expand_dir(cp, index)
            if not got:
                blocked.append(f"not in source: {cp}")
            for p in got:
                files.setdefault(p, (None, None))
        rows = [{"path": p, "size": size_of(p), "node": files[p][0], "cls": files[p][1],
                 "present": present(p, size_of(p))} for p in sorted(files)]
        missing = [f["path"] for f in rows if not f["present"]]
        blocked += [f"not in source: {p}" for p in missing if p not in urls and p not in src]
        hints: list = []
        if not rows and not blocked and not n.explicit:
            if loose:
                vals = ", ".join(sorted({r.value for r in loose}))
                blocked.append(f"no model files known (loader values: {vals} — add a catalog "
                               f"class+value entry)")
            else:
                blocked.append("no model references known")
        else:
            hints = sorted({f"{r.cls}={r.value} is not synced — add a catalog entry if it "
                            f"needs weights" for r in loose})
        needed.update(files)
        if not blocked:
            wanted[n.alias] = missing
        per_alias[n.alias] = {
            "need_bytes": sum(f["size"] or 0 for f in rows),
            "have_bytes": sum(f["size"] or 0 for f in rows if f["present"]),
            "missing": missing,
            "blocked": sorted(set(blocked)),
            "hints": hints,
            "selectable": sorted({f"{r.node}.{r.field}" for r in n.refs if r.selectable}),
            "files": rows,
        }

    # fetch: the fewest missing bytes first; within one alias the large files first
    # (unknown sizes last), each file once
    fetch: list = []
    by_path: dict = {}
    for a in sorted(wanted, key=lambda a: (sum(size_of(p) or 0 for p in wanted[a]), a)):
        for p in sorted(wanted[a], key=lambda p: (size_of(p) is None, -(size_of(p) or 0), p)):
            if p in by_path:
                by_path[p]["aliases"].append(a)
                continue
            e = {"path": p, "size": size_of(p), "source": "url" if p in urls else "lan"}
            if p in urls:
                e["url"] = urls[p]["url"]
                if urls[p].get("sha256"):
                    e["sha256"] = urls[p]["sha256"]
            e["aliases"] = [a]
            by_path[p] = e
            fetch.append(e)

    blocked_aliases = {a for a, v in per_alias.items() if v["blocked"]}
    held, prune = [], []
    for p in sorted(man):
        if p in needed:
            continue
        rec = man[p].get("aliases") if isinstance(man[p], dict) else None
        # only a list names owners: a string would be iterated character by character
        # (`"AB"` → owners A and B), anything else is no record at all
        rec = rec if isinstance(rec, (list, tuple)) else ()
        owners = sorted(blocked_aliases & {x for x in rec if isinstance(x, str)})
        if owners:
            size = _msize(man[p])
            held.append([p, size if size is not None else dest.get(p), owners[0]])
        else:
            prune.append(p)

    sizes = {p: size_of(p) for p in needed}
    return {
        "per_alias": per_alias,
        "fetch": fetch,
        "prune": prune,
        "held": held,
        "unknown": sorted([p, s] for p, s in dest.items()
                          if _usable(p) and not _is_part(p) and p not in man and p not in needed),
        "need_total": sum(s or 0 for s in sizes.values()),
        "have_total": sum(s or 0 for p, s in sizes.items() if present(p, s)),
    }


def ready(plan_: dict, alias: str) -> bool:
    """The alias may route to this backend: it is in the plan, nothing blocks it and
    every file it needs is present. No plan yet = not ready."""
    a = (plan_ or {}).get("per_alias", {}).get(alias)
    return bool(a) and not a["blocked"] and not a["missing"]


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f}"


def status_text(plan_: dict, alias: str, backend_name: str) -> str:
    """One line for the 503 a client gets while this backend cannot serve the alias."""
    a = (plan_ or {}).get("per_alias", {}).get(alias)
    if not a:
        return f"models for {alias} are not planned on {backend_name} yet"
    if a["blocked"]:
        return f"models for {alias} are blocked on {backend_name}: {'; '.join(a['blocked'])}"
    if not a["missing"]:
        return f"models for {alias} are ready on {backend_name}"
    unsized = sum(1 for f in a["files"] if not f["present"] and f["size"] is None)
    extra = f" + {unsized} file{'s' if unsized != 1 else ''} of unknown size" if unsized else ""
    return (f"models for {alias} are syncing on {backend_name} "
            f"({_gb(a['have_bytes'])} of {_gb(a['need_bytes'])} GB{extra})")
