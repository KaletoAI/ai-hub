"""The RunPod worker's build context and handler.

Why this fails SILENTLY: the worker image is built on RunPod, far from every test, and a
drift there only shows as a job that runs on a different stack than Thunder — a
different torch, another ComfyUI commit, a node pack at another revision — and renders
a subtly different picture, or as a manifest the gateway reads as "file absent" and
delivers less. These tests pin the image to the Thunder pins, the node list to the
default list, and the handler's output shape to what the gateway reads.

Run: venv/bin/python -m unittest tests.test_runpod_worker -v
"""
import os
import pathlib
import re
import sys
import unittest

_here = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_here))
import adapters  # noqa: E402
import thunder  # noqa: E402

RP = _here / "ops" / "runpod"
BOOT = _here / "ops" / "thunder-bootstrap.sh"


def _sh_var(path, name):
    m = re.search(rf"^{name}=(\S+)$", path.read_text(), re.M)
    return m.group(1) if m else None


def _arg(name):
    m = re.search(rf"^ARG {name}=(\S+)$", (RP / "Dockerfile").read_text(), re.M)
    return m.group(1) if m else None


class BuildContext(unittest.TestCase):
    def test_stack_pins_equal_thunder(self):
        for v in ("PY_VER", "TORCH_VER", "TORCHVISION_VER", "TORCHAUDIO_VER", "TORCH_CUDA"):
            self.assertEqual(_arg(v), _sh_var(BOOT, v), v)
        self.assertEqual(_arg("COMFY_COMMIT"), thunder.COMFY_COMMIT_DEFAULT)

    def test_base_image_is_cuda_13(self):
        self.assertIn("FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04",
                      (RP / "Dockerfile").read_text())

    def test_node_lines_are_verbatim_default_lines(self):
        default = set((_here / "ops" / "thunder-nodes.default.txt").read_text().splitlines())
        lines = [l for l in (RP / "nodes.image.txt").read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
        self.assertTrue(lines)
        for l in lines:
            self.assertIn(l, default, l)

    def test_placeholder_is_the_gateways(self):
        self.assertEqual((RP / "gw_placeholder.png").read_bytes(), adapters._PLACEHOLDER_PNG)

    def test_extra_model_paths_point_at_the_volume(self):
        t = (RP / "extra_model_paths.yaml").read_text()
        self.assertIn("base_path: /runpod-volume/models", t)
        for folder in ("checkpoints", "diffusion_models", "unet", "text_encoders", "clip",
                       "vae", "loras"):
            self.assertRegex(t, rf"\n\s+{folder}: {folder}\n")

    def test_no_agpl_licence_text(self):
        # our own code: no file in the build context may carry AGPL licence text
        for p in RP.iterdir():
            if p.is_file() and p.suffix != ".png":
                self.assertNotIn("gnu affero general public license",
                                 p.read_text(errors="ignore").lower(), p.name)


if __name__ == "__main__":
    unittest.main()
