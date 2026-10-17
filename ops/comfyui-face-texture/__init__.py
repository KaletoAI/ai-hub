"""ComfyUI bridge for face-only texture projection on an existing static GLB."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import folder_paths
import numpy as np
from PIL import Image
import torch
from comfy_api.latest import Types


PACKAGE_DIR = Path(__file__).resolve().parent


class FaceTextureOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_glb": ("FILE_3D", {"tooltip": "Original static textured Trellis GLB. Geometry is preserved, not regenerated."}),
                "reference_image": ("IMAGE", {"tooltip": "One matching RGB face photo. Face detection and registration run anew."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "feather": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 0.5, "step": 0.01}),
                "render_resolution": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "yaw": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0, "tooltip": "Source viewing yaw in degrees for face registration."}),
            }
        }

    RETURN_TYPES = ("FILE_3D", "IMAGE", "STRING")
    RETURN_NAMES = ("corrected_glb", "comparison", "proof_metrics")
    FUNCTION = "retexture"
    CATEGORY = "3d/texture"
    DESCRIPTION = "Project a matching photo onto only the visible face texture; preserve the source GLB geometry and original assets. CPU worker runs in a separate process."

    def retexture(self, source_glb, reference_image, strength, feather, render_resolution, yaw):
        installer = PACKAGE_DIR / "install-worker.sh"
        if not isinstance(source_glb, Types.File3D) or source_glb.format != "glb":
            raise ValueError("FaceTextureOnly requires a static textured GLB from the native Load 3D node.")
        if reference_image.ndim != 4 or reference_image.shape[0] != 1 or reference_image.shape[-1] != 3:
            raise ValueError("FaceTextureOnly requires exactly one RGB reference image, not an image batch.")

        local_python = PACKAGE_DIR / ".venv" / "bin" / "python"
        # Prefer the portable isolated CPU environment. An already-equipped Comfy
        # interpreter is also safe: the core is imported only by this subprocess.
        python = str(local_python) if local_python.is_file() else sys.executable
        model_dir = PACKAGE_DIR / "models"
        for filename in ("face_landmarker.task", "selfie_multiclass.tflite"):
            if not (model_dir / filename).is_file():
                raise RuntimeError(f"FaceTextureOnly is missing {model_dir / filename}. Run: bash {installer}")

        folder_paths_temp = Path(folder_paths.get_temp_directory())
        folder_paths_temp.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="face_texture_only_", dir=folder_paths_temp))
        source = source_glb.get_source()
        if isinstance(source, str):
            source_path = Path(source).resolve()
        else:
            source_path = work_dir / "source.glb"
            source_glb.save_to(str(source_path))
        if not source_path.is_file():
            raise ValueError(f"FaceTextureOnly source GLB does not exist: {source_path}")

        reference_path = work_dir / "reference.png"
        pixels = reference_image[0].detach().cpu().numpy()
        Image.fromarray(np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)).save(reference_path)
        output_path = work_dir / "FaceTextureOnly.glb"
        comparison_path = work_dir / "comparison.png"
        proof_path = work_dir / "proof.json"
        command = [
            python, str(PACKAGE_DIR / "worker.py"),
            "--source-glb", str(source_path), "--reference-image", str(reference_path),
            "--output-glb", str(output_path), "--comparison-png", str(comparison_path),
            "--proof-json", str(proof_path), "--model-dir", str(model_dir),
            "--strength", str(strength), "--feather", str(feather),
            "--render-resolution", str(render_resolution), "--yaw", str(yaw),
        ]
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        try:
            completed = subprocess.run(command, cwd=PACKAGE_DIR, env=environment, capture_output=True, text=True)
        except OSError as exc:
            raise RuntimeError(f"FaceTextureOnly could not start the CPU worker ({python}): {exc}. Run: bash {installer}") from exc
        if completed.returncode:
            details = completed.stderr.strip() or completed.stdout.strip() or "Worker produced no error details."
            raise RuntimeError(f"FaceTextureOnly CPU worker failed (exit {completed.returncode}):\n{details}")
        for output in (output_path, comparison_path, proof_path):
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"FaceTextureOnly worker did not produce its required output: {output}")
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        with Image.open(comparison_path) as comparison_image:
            comparison = np.asarray(comparison_image.convert("RGB"), dtype=np.float32) / 255.0
        comparison_tensor = torch.from_numpy(comparison).unsqueeze(0)
        proof_text = json.dumps(proof, ensure_ascii=False, indent=2)
        return {"ui": {"text": [proof_text]}, "result": (Types.File3D(str(output_path)), comparison_tensor, proof_text)}


NODE_CLASS_MAPPINGS = {"FaceTextureOnly": FaceTextureOnly}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceTextureOnly": "Face Texture Only (Preserve Geometry)"}
