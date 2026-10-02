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



import base64  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

sys.path.insert(0, str(RP))
import handler  # noqa: E402


class _ComfyStub(BaseHTTPRequestHandler):
    """POST /prompt → prompt id (or node_errors); GET /history/<id> → `history`."""
    history: dict = {}
    prompt_reply: dict = {"prompt_id": "p1"}
    prompts: list = []

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        _ComfyStub.prompts.append(json.loads(self.rfile.read(n)))
        self._json(200, _ComfyStub.prompt_reply)

    def do_GET(self):
        if self.path.startswith("/history/"):
            return self._json(200, _ComfyStub.history)
        if self.path == "/object_info":
            return self._json(200, {"KSampler": {"input": {}, "output": ["LATENT"]}})
        self._json(404, {})


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _ComfyStub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


class Handler(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.t.name)
        self.dirs = {k: root / k for k in ("input", "output", "temp")}
        for d in self.dirs.values():
            d.mkdir()
        self.srv, self.base = _serve()
        _ComfyStub.prompts = []
        _ComfyStub.prompt_reply = {"prompt_id": "p1"}

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.t.cleanup()

    def _run(self, job_input):
        return handler.run_prompt(job_input, self.base, {k: str(v) for k, v in self.dirs.items()},
                                  report=lambda p: None, poll_s=0.01)

    def test_view_key_matches_gateway_view_params(self):
        items = [{"filename": "a.png", "subfolder": "", "type": "output"},
                 {"filename": "b.png", "subfolder": "x/y", "type": "temp"},
                 "/comfyui/output/sub/m.glb", "/comfyui/temp/p.glb", "rel/dir/q.fbx",
                 "/abs/elsewhere/r.glb", True, "not a file"]
        for it in items:
            mine = handler.view_params(it)
            theirs = adapters._view_params(it)
            self.assertEqual(mine, theirs, it)

    def test_inputs_written_and_names_held_to_plain_characters(self):
        handler.write_inputs([{"name": "gw_j_image.png", "b64": base64.b64encode(b"x").decode()}],
                             str(self.dirs["input"]))
        self.assertEqual((self.dirs["input"] / "gw_j_image.png").read_bytes(), b"x")
        with self.assertRaises(handler.HandlerError):
            handler.write_inputs([{"name": "../etc/x", "b64": ""}], str(self.dirs["input"]))

    def test_manifest_has_files_siblings_temp_and_null_for_missing(self):
        (self.dirs["output"] / "o_00001_.png").write_bytes(b"png")
        (self.dirs["output"] / "m.fbx").write_bytes(b"fbx")
        (self.dirs["output"] / "m.glb").write_bytes(b"glb")
        (self.dirs["temp"] / "prev.glb").write_bytes(b"tmp")
        _ComfyStub.history = {"p1": {"status": {"status_str": "success"}, "outputs": {
            "9": {"images": [{"filename": "o_00001_.png", "subfolder": "", "type": "output"},
                             {"filename": "gone.png", "subfolder": "", "type": "output"}]},
            "10": {"result": [str(self.dirs["temp"] / "prev.glb"), "m.fbx"]}}}}
        out = self._run({"op": "prompt", "workflow": {"1": {}}, "inputs": [],
                         "deliver": {"sibling_exts": ["glb"]}})
        man = out["manifest"]
        self.assertEqual(base64.b64decode(man["output/o_00001_.png"]["b64"]), b"png")
        self.assertEqual(man["output/o_00001_.png"]["sha256"], hashlib.sha256(b"png").hexdigest())
        self.assertIsNone(man["output/gone.png"])
        self.assertEqual(base64.b64decode(man["temp/prev.glb"]["b64"]), b"tmp")
        self.assertEqual(base64.b64decode(man["output/m.glb"]["b64"]), b"glb")
        self.assertIn("outputs", out)
        self.assertEqual(_ComfyStub.prompts[0]["prompt"], {"1": {}})

    def test_output_over_the_cap_is_an_error(self):
        (self.dirs["output"] / "big.png").write_bytes(b"x" * 300)
        _ComfyStub.history = {"p1": {"status": {"status_str": "success"}, "outputs": {
            "9": {"images": [{"filename": "big.png", "type": "output"}]}}}}
        old = handler.OUT_MAX_B64
        handler.OUT_MAX_B64 = 100
        try:
            out = self._run({"op": "prompt", "workflow": {}, "inputs": []})
        finally:
            handler.OUT_MAX_B64 = old
        self.assertIn("output too large", out["error"])

    def test_over_budget_file_is_never_read(self):
        import builtins
        (self.dirs["output"] / "huge.png").write_bytes(b"x" * 300)
        outputs = {"9": {"images": [{"filename": "huge.png", "type": "output"}]}}
        real, opened = builtins.open, []

        def spy(f, *a, **k):
            opened.append(str(f))
            return real(f, *a, **k)
        builtins.open = spy
        try:
            with self.assertRaises(handler.HandlerError):
                handler.build_manifest(outputs, [], {k: str(v) for k, v in self.dirs.items()}, 100)
        finally:
            builtins.open = real
        self.assertFalse([o for o in opened if o.endswith("huge.png")])

    def test_dead_comfyui_is_an_error_not_a_hang(self):
        _ComfyStub.history = {}
        out = handler.run_prompt({"op": "prompt", "workflow": {}, "inputs": []}, self.base,
                                 {k: str(v) for k, v in self.dirs.items()},
                                 report=lambda p: None, poll_s=0.01, alive=lambda: False)
        self.assertIn("died", out["error"])

    def test_comfyui_gone_after_submit_is_an_error(self):
        _ComfyStub.history = {}
        calls = []

        def alive():
            calls.append(1)
            if len(calls) == 2:
                self.srv.shutdown()
                self.srv.server_close()
            return True
        out = handler.run_prompt({"op": "prompt", "workflow": {}, "inputs": []}, self.base,
                                 {k: str(v) for k, v in self.dirs.items()},
                                 report=lambda p: None, poll_s=0.01, alive=alive)
        self.assertIn("error", out)
        self.srv = _serve()[0]          # tearDown shuts a live one down

    def test_malformed_inputs_are_errors(self):
        for bad in ([{"name": "ok.png", "b64": "***not base64***"}], ["notadict"]):
            out = self._run({"op": "prompt", "workflow": {}, "inputs": bad})
            self.assertIn("error", out, bad)

    def test_node_errors_and_execution_errors_pass_through(self):
        _ComfyStub.prompt_reply = {"error": {"message": "bad"}, "node_errors": {"4": {"x": 1}}}
        self.assertIn("node_errors", self._run({"op": "prompt", "workflow": {}, "inputs": []})["error"])
        _ComfyStub.prompt_reply = {"prompt_id": "p1"}
        _ComfyStub.history = {"p1": {"status": {"status_str": "error", "messages": [
            ["execution_error", {"node_id": "4", "exception_message": "CUDA out of memory"}]]},
            "outputs": {}}}
        self.assertIn("CUDA out of memory",
                      self._run({"op": "prompt", "workflow": {}, "inputs": []})["error"])

    def test_info_returns_gzipped_object_info_and_model_index(self):
        m = pathlib.Path(self.t.name, "models", "unet")
        m.mkdir(parents=True)
        (m / "q.gguf").write_bytes(b"1234")
        out = handler.run_info(self.base, {"image": str(pathlib.Path(self.t.name, "models")),
                                           "volume": str(pathlib.Path(self.t.name, "none"))})
        import gzip
        oi = json.loads(gzip.decompress(base64.b64decode(out["object_info_gz"])))
        self.assertIn("KSampler", oi)
        self.assertEqual(out["models"]["image"], {"models/unet/q.gguf": 4})
        self.assertEqual(out["models"]["volume"], {})

    def test_artifact_extensions_equal_the_gateways(self):
        self.assertEqual(set(handler.ARTIFACT_EXTS), set(adapters._MIME_BY_EXT))

    def test_manifest_never_reads_outside_its_type_dir(self):
        """Item 11a (final review): the manifest named files by joining what /history
        said onto the type dir — a `..` subfolder, a relative bare path or a sibling ext
        holding `/` walked out of it, and the file shipped to the gateway as a result."""
        import builtins
        root = pathlib.Path(self.t.name)
        (root / "secret.png").write_bytes(b"model weights")
        (self.dirs["output"] / "ok.png").write_bytes(b"png")
        (self.dirs["output"] / "link.png").symlink_to(root / "secret.png")
        outputs = {"9": {"images": [
            {"filename": "secret.png", "subfolder": "..", "type": "output"},
            {"filename": "secret.png", "subfolder": "../output/..", "type": "temp"},
            "../secret.png",
            {"filename": "link.png", "type": "output"},
            {"filename": "ok.png", "type": "output"}]}}
        real, opened = builtins.open, []

        def spy(f, *a, **k):
            opened.append(str(f))
            return real(f, *a, **k)
        builtins.open = spy
        try:
            man = handler.build_manifest(outputs, ["/../../secret.png", "png"],
                                         {k: str(v) for k, v in self.dirs.items()}, 10 ** 6)
        finally:
            builtins.open = real
        self.assertFalse([o for o in opened if "secret" in o or o.endswith("link.png")], opened)
        self.assertEqual(base64.b64decode(man["output/ok.png"]["b64"]), b"png")
        self.assertIsNone(man["output/link.png"])
        others = {k: v for k, v in man.items() if k not in ("output/ok.png", "output/link.png")}
        self.assertTrue(others)
        self.assertTrue(all(v is None for v in others.values()), others)


    def test_gateway_refuses_exactly_what_the_worker_refuses(self):
        self.assertEqual(adapters._RP_INPUT_NAME_RE.pattern, handler.NAME_RE.pattern)


if __name__ == "__main__":
    unittest.main()
