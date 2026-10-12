"""Pure half of ComfyUI-KimodoHub: no ComfyUI, torch or FBX SDK import, so the
rules the alias contract depends on are unit-testable (tests/test_kimodo_hub.py)."""

import os
import re

MAX_VARIANTS = 4
# Bones every Mixamo rig has, fingers or not — matched the way kimodo_retarget_fbx's
# get_bone does (case-insensitive, prefix-tolerant).
MIXAMO_CORE_BONES = ("mixamorig:Hips", "mixamorig:Spine", "mixamorig:Head",
                     "mixamorig:LeftArm", "mixamorig:RightArm",
                     "mixamorig:LeftUpLeg", "mixamorig:RightUpLeg")
_FBX_MAGIC = (b"Kaydara FBX Binary", b"; FBX")


def prompt_segments(prompt: str) -> list:
    """One motion segment per non-empty line, in order. Upstream Kimodo_TextEncode
    splits on PERIODS instead, which turns one line with two sentences into two
    segments (and a client's line count into a different clip length). A line keeps
    its inner punctuation; one without a closing period gets one, as upstream gives
    every segment."""
    segs = []
    for line in (prompt or "").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        segs.append(line if line.endswith((".", "!", "?")) else line + ".")
    if not segs:
        raise ValueError("Kimodo prompt is empty: send one motion description per line")
    return segs


def looks_like_fbx(head: bytes) -> bool:
    return any(head.startswith(m) for m in _FBX_MAGIC)


def resolve_rig(path_str: str, input_dir: str) -> str:
    """The target rig as an existing FBX file, or ValueError. Absolute paths (the
    AI-Hub injects one per upload) are taken as they are; anything else is relative to
    ComfyUI's input dir. Upstream Kimodo_ExportFBX returns an empty string here and the
    job ends 'successfully' with no file — a missing rig must fail the run."""
    p = (path_str or "").strip().replace("\\", "/")
    if not p:
        raise ValueError("Kimodo export: no target rig given (input_rig_path is empty)")
    full = p if os.path.isabs(p) else os.path.normpath(os.path.join(input_dir, p))
    if not os.path.isfile(full):
        raise ValueError(f"Kimodo export: target rig not found: {p}")
    with open(full, "rb") as fh:
        head = fh.read(32)
    if not looks_like_fbx(head):
        raise ValueError(f"Kimodo export: target rig is not an FBX file: {os.path.basename(full)}")
    return full


def output_paths(output_dir: str, filename_prefix: str, count: int, uid: str) -> list:
    """One output path per sample, index-suffixed so the delivery order (= the order
    reported to /history) is the sample order. `filename_prefix` may carry a
    subfolder (`3D/kimodo`), like ComfyUI's save nodes; it must stay inside the
    output dir."""
    prefix = (filename_prefix or "kimodo").strip().replace("\\", "/").strip("/") or "kimodo"
    base = os.path.normpath(os.path.join(output_dir, prefix))
    root = os.path.normpath(output_dir)
    if os.path.commonpath([root, base]) != root:
        raise ValueError(f"Kimodo export: filename_prefix leaves the output dir: {filename_prefix}")
    return [f"{base}_{uid}_{i}.fbx" for i in range(count)]


def sampler_call(texts: list, frames: int, n: int, steps: int, root_margin: float) -> tuple:
    """(args, kwargs) for the Kimodo model call. Upstream Kimodo_Sampler always passes
    the texts as a LIST, and for ONE segment the model then takes the batch size from
    that list (`num_samples = len(prompts)`) — `num_samples` is dropped and a request
    for 3 variants silently yields 1. One segment therefore goes in as a plain string
    (the path that honours num_samples), several as the multi-prompt sequence.

    Post-processing (foot-skate / root correction) runs INSIDE this call: the model
    corrects on its 30-joint SOMA skeleton and only then converts to the 77-joint
    output. Upstream Kimodo_PostProcess corrects that 77-joint output against the
    30-joint skeleton — "jointRotations has the wrong size" on every SOMA run, which
    it swallowed and passed the motion on uncorrected."""
    kw = {"num_denoising_steps": steps, "num_samples": check_variants(n),
          "constraint_lst": [], "post_processing": True, "root_margin": float(root_margin),
          "return_numpy": True}
    if len(texts) == 1:
        return (texts[0], frames), kw
    return (list(texts), [frames] * len(texts)), {**kw, "multi_prompt": True}


def check_variants(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_VARIANTS:
        raise ValueError(f"Kimodo: number of variants must be 1–{MAX_VARIANTS}, got {n!r}")
    return n
