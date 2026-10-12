"""ComfyUI-KimodoHub — the AI-Hub's four nodes around ComfyUI-Kimodo (alias
`motion-kimodo`). Install next to it as custom_nodes/ComfyUI-KimodoHub; it uses
Kimodo's own retargeting (`kimodo_retarget_fbx`, needs fbxsdkpy) and changes nothing
in that package.

Why not the upstream nodes:
- Kimodo_TextEncode splits segments on periods; the alias contract is one per LINE.
- Kimodo_Sampler drops num_samples for a one-segment prompt (core.sampler_call).
- Kimodo_PostProcess swallows every error and passes the motion on uncorrected.
- Kimodo_ExportFBX returns a bare STRING (no `ui` dict), so its file never shows up in
  /history and the AI-Hub cannot find the result; it also swallows every error into
  that string, and exports ONE sample index (clamped, so index 3 of a 1-sample run is
  sample 0 again).
"""

import os
import shutil
import sys
import tempfile
import uuid

import folder_paths

from . import kimodo_hub_core as core

_KIMODO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "ComfyUI-Kimodo")


def _kimodo_imports():
    # ComfyUI-Kimodo puts itself on sys.path when it loads (it sorts before us);
    # repeated here so a different load order cannot break the lazy imports.
    for d in (_KIMODO_DIR, os.path.join(_KIMODO_DIR, "kimodo")):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)


class _Cond:
    """Kimodo_Sampler reads only `.texts` from its conditioning (it encodes the texts
    itself); the other two attributes keep the upstream KimodoCondData shape."""
    def __init__(self, texts):
        self.texts = texts
        self.text_feat = None
        self.text_pad_mask = None


class KimodoHub_PromptLines:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt": ("STRING", {
            "default": "A person walks forward.", "multiline": True,
            "tooltip": "One motion segment per line; several lines are played in sequence."})}}

    RETURN_TYPES = ("KIMODO_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Kimodo/Hub"

    def encode(self, prompt):
        _kimodo_imports()
        from kimodo.sanitize import sanitize_texts
        texts = sanitize_texts(core.prompt_segments(prompt))
        print(f"[KimodoHub] {len(texts)} segment(s): {texts}", flush=True)
        return (_Cond(texts),)


def _upstream(name: str):
    """A loaded ComfyUI-Kimodo node class (ComfyUI's global registry) — reused for its
    helpers, never patched."""
    import nodes
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"KimodoHub: ComfyUI-Kimodo node {name} is not loaded")
    return cls


class KimodoHub_Sampler:
    """Kimodo_Sampler with num_samples honoured (core.sampler_call) and range-checked
    1–4 by ComfyUI when the prompt is queued — a request for 5 is refused before
    anything loads."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("KIMODO_MODEL",),
            "conditioning": ("KIMODO_COND",),
            "duration": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 30.0, "step": 0.5,
                                   "tooltip": "Seconds per segment"}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            "num_samples": ("INT", {"default": 1, "min": 1, "max": core.MAX_VARIANTS}),
            "diffusion_steps": ("INT", {"default": 100, "min": 10, "max": 500}),
        }}

    RETURN_TYPES = ("KIMODO_MOTION",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "sample"
    CATEGORY = "Kimodo/Hub"

    def sample(self, model, conditioning, duration, seed, num_samples, diffusion_steps):
        _kimodo_imports()
        from kimodo.tools import seed_everything
        seed_everything(seed)
        texts = list(conditioning.texts)
        frames = int(duration * model.fps)
        args, kw = core.sampler_call(texts, frames, num_samples, diffusion_steps)
        print(f"[KimodoHub] sampling {len(texts)} segment(s) × {frames} frames, "
              f"{num_samples} sample(s), {diffusion_steps} steps", flush=True)
        output = model(*args, **kw)
        got = int(output["posed_joints"].shape[0])
        if got != num_samples:
            raise RuntimeError(f"KimodoHub sampler: asked for {num_samples} samples, model returned {got}")
        return (_upstream("Kimodo_Sampler")._wrap_output(
            model, output, texts, [frames] * len(texts), num_samples, []),)


class KimodoHub_PostProcess:
    """Kimodo_PostProcess (foot-skate / root correction), but a failure FAILS the run.
    Upstream catches every error and returns the uncorrected motion — with the
    motion_correction extension missing, every run skipped the correction unnoticed."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "motion": ("KIMODO_MOTION",),
            "root_margin": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("KIMODO_MOTION",)
    RETURN_NAMES = ("motion",)
    FUNCTION = "process"
    CATEGORY = "Kimodo/Hub"

    def process(self, motion, root_margin):
        _kimodo_imports()
        import numpy as np
        import torch
        from kimodo.postprocess import post_process_motion
        if motion.skeleton is None:
            raise RuntimeError("KimodoHub post-process: the motion carries no skeleton")
        out = motion.output_dict
        t = {k: (torch.from_numpy(out[k]) if isinstance(out[k], np.ndarray) else out[k])
             for k in ("local_rot_mats", "root_positions", "foot_contacts")}
        corrected = post_process_motion(t["local_rot_mats"], t["root_positions"],
                                        t["foot_contacts"], motion.skeleton,
                                        motion.constraint_lst or [], root_margin=root_margin)
        new_out = dict(out)
        for k, v in corrected.items():
            new_out[k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v
        print(f"[KimodoHub] post-process done (root_margin={root_margin})", flush=True)
        return (type(motion)(
            output_dict=new_out, model_name=motion.model_name, skeleton_name=motion.skeleton_name,
            fps=motion.fps, texts=motion.texts, num_frames=motion.num_frames,
            num_samples=motion.num_samples, joint_parents=motion.joint_parents,
            joint_names=motion.joint_names, neutral_joints=motion.neutral_joints,
            skeleton=motion.skeleton, constraint_lst=motion.constraint_lst),)


def _check_mixamo(rig: str) -> None:
    """Upstream retargeting only LOGS "No bone pairs matched" and writes an FBX
    without animation — a wrong rig would come back as a `done` job with still files.
    Refuse it before the export instead."""
    from kimodo_retarget_fbx import load_target_fbx
    manager, _scene, tgt = load_target_fbx(rig)
    try:
        missing = [b for b in core.MIXAMO_CORE_BONES if not tgt.get_bone(b)]
    finally:
        manager.Destroy()
    if missing:
        raise ValueError(f"KimodoHub export: the target rig is not a Mixamo skeleton "
                         f"(missing {', '.join(missing)})")


class KimodoHub_ExportFBX:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "motion": ("KIMODO_MOTION",),
            "rig_path": ("STRING", {"default": "", "tooltip": "Mixamo-rigged FBX (absolute, or relative to input/)."}),
            "filename_prefix": ("STRING", {"default": "3D/kimodo"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export"
    CATEGORY = "Kimodo/Hub"
    OUTPUT_NODE = True

    def export(self, motion, rig_path, filename_prefix):
        _kimodo_imports()
        from kimodo_retarget_fbx import HAS_FBX_SDK, export_kimodo_fbx
        if not HAS_FBX_SDK:
            raise RuntimeError("KimodoHub export: FBX SDK (fbxsdkpy) is not installed")
        n = core.check_variants(int(motion.batch_size))
        rig = core.resolve_rig(rig_path, folder_paths.get_input_directory())
        work = None
        if not rig.lower().endswith(".fbx"):     # the FBX SDK picks its reader by extension
            work = tempfile.mkdtemp(prefix="kimodo_hub_", dir=folder_paths.get_temp_directory())
            shutil.copyfile(rig, os.path.join(work, "rig.fbx"))
            rig = os.path.join(work, "rig.fbx")
        paths = core.output_paths(folder_paths.get_output_directory(), filename_prefix, n,
                                  uuid.uuid4().hex[:8])
        os.makedirs(os.path.dirname(paths[0]), exist_ok=True)
        try:
            _check_mixamo(rig)
            for i, out in enumerate(paths):
                export_kimodo_fbx(motion_data=motion, target_fbx_path=rig, output_path=out,
                                  sample_index=i)
                with open(out, "rb") as fh:
                    if not core.looks_like_fbx(fh.read(32)):
                        raise RuntimeError(f"KimodoHub export: sample {i} wrote no FBX ({out})")
        finally:
            if work:
                shutil.rmtree(work, ignore_errors=True)
        print(f"[KimodoHub] exported {n} FBX: {paths}", flush=True)
        # `ui` is what lands in /history.outputs — the AI-Hub delivers these paths in
        # this order (= sample index).
        return {"ui": {"fbx_file": paths}, "result": (paths[0],)}


NODE_CLASS_MAPPINGS = {
    "KimodoHub_PromptLines": KimodoHub_PromptLines,
    "KimodoHub_Sampler": KimodoHub_Sampler,
    "KimodoHub_PostProcess": KimodoHub_PostProcess,
    "KimodoHub_ExportFBX": KimodoHub_ExportFBX,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KimodoHub_PromptLines": "Kimodo Prompt (one segment per line)",
    "KimodoHub_Sampler": "Kimodo Sampler (1–4 variants, AI-Hub)",
    "KimodoHub_PostProcess": "Kimodo Post-Process (fails loudly, AI-Hub)",
    "KimodoHub_ExportFBX": "Kimodo Export FBX (all samples, AI-Hub)",
}
