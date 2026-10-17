"""CPU-only command line entrypoint; never imported into ComfyUI's process."""

import argparse
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description="Face-only texture projection with unchanged source geometry.")
    parser.add_argument("--source-glb", required=True, type=Path)
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--output-glb", required=True, type=Path)
    parser.add_argument("--comparison-png", required=True, type=Path)
    parser.add_argument("--proof-json", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--feather", type=float, default=0.18)
    parser.add_argument("--render-resolution", type=int, default=768)
    parser.add_argument("--yaw", type=float, default=0.0)
    arguments = parser.parse_args()
    installer = Path(__file__).resolve().parent / "install-worker.sh"
    try:
        # This import belongs here, not in the Comfy node's module. MediaPipe and
        # its CPU runtime do not share state with ComfyUI's torch interpreter.
        from face_core import run_retexture
        run_retexture(
            source_glb=arguments.source_glb,
            reference_image=arguments.reference_image,
            output_glb=arguments.output_glb,
            comparison_png=arguments.comparison_png,
            proof_json=arguments.proof_json,
            model_dir=arguments.model_dir,
            strength=arguments.strength,
            feather=arguments.feather,
            render_resolution=arguments.render_resolution,
            yaw=arguments.yaw,
        )
    except ImportError as exc:
        print(f"FaceTextureOnly CPU dependency unavailable in {sys.executable}: {exc}.\nRun: bash {installer}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
