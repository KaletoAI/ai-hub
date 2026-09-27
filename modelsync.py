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
  blocked with no hint why.

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

# Loader detection — the same three rules as scheduler.model_set_key, kept here (not
# imported) because the "how" list below is deliberately WIDER than the scheduler's: a
# missed "how" there costs one skipped VRAM free, here it blocks an alias forever.
_LOADER_CLASS = re.compile(r"load", re.I)
_INPUT_LOADER = re.compile(r"image|mask|mesh|path|video|audio", re.I)
_WEIGHT_INPUT = re.compile(r"name|model|ckpt|unet|clip|vae|lora|gguf|weight", re.I)
# `mode(?!l)`: `attention_mode` is a how, `model`/`modelname` are exactly what we want.
_HOW_INPUT = re.compile(r"dtype|precision|device|backend|attn|format|quant|mode(?!l)|type|scheme", re.I)
# A file ending on the LAST segment: a letter first, so a version (`Model2.5`,
# `org/Model-2.5-VL`) is no extension.
_FILE_ENDING = re.compile(r"\.[A-Za-z][A-Za-z0-9_]{0,7}$")

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

# Pre-filled catalog for a fresh install. Empty in P2 on purpose: its entries (base and
# hub models only — never private model/LoRA names, the repo is public) come with the
# catalog editor (Task 13). Must always pass validate_catalog (pinned by a test).
DEFAULT_CATALOG: list[dict] = []


def ref_kind(value: str) -> str:
    """`file` | `hub` | `name` for a reference value (see the module docstring)."""
    v = str(value or "")
    last = v.rsplit("/", 1)[-1]
    if v.lower().endswith(MODEL_EXT) or _FILE_ENDING.search(last):
        return "file"
    if "/" in v:
        return "hub"
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


def catalog_paths(refs, alias: str, catalog) -> tuple[list[str], bool]:
    """The catalog paths this alias needs beyond its resolved references, and whether an
    `alias` entry names it (`explicit_alias_entry` — with `paths: []` that is the
    declaration "needs nothing"). Every key of an entry's `match` must hold: `alias`
    against the alias, `class`/`value` against ONE of the refs. Invalid entries are
    skipped here; `validate_catalog` is what reports them."""
    paths: list[str] = []
    explicit = False
    for e in catalog if isinstance(catalog, list) else []:
        if not isinstance(e, dict):
            continue
        m, ps = e.get("match"), e.get("paths")
        if not isinstance(m, dict) or not m or set(m) - set(_MATCH_KEYS) or not isinstance(ps, list):
            continue
        if "alias" in m and m["alias"] != alias:
            continue
        if ("class" in m or "value" in m) and not any(
                ("class" not in m or r.cls == m["class"]) and ("value" not in m or r.value == m["value"])
                for r in refs or ()):
            continue
        explicit = explicit or "alias" in m
        for p in ps:
            if isinstance(p, str) and p not in paths:
                paths.append(p)
    return paths, explicit


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
    if p in ROOTS:
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
