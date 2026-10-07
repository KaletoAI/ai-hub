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


class FetchOps(unittest.TestCase):
    """The fetch job writes straight onto the network volume every RunPod worker mounts.
    A path escape writes outside models/, a rename before the hash check publishes a
    truncated model that ComfyUI loads as garbage, and a token read from the job input
    lands in RunPod's stored job history — all silent."""

    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        self.seen = []
        # Temp directories stand in for the real mount for existing path/download tests.
        from functools import partial
        from unittest.mock import patch
        fetch = patch.object(handler, 'run_fetch', partial(handler.run_fetch, ismount=lambda root: True))
        link = patch.object(handler, 'run_link', partial(handler.run_link, ismount=lambda root: True))
        fetch.start(); link.start()
        self.addCleanup(fetch.stop); self.addCleanup(link.stop)

    def opener(self, data, status=200):
        import io

        def op(req, timeout=None):
            self.seen.append(req)
            r = io.BytesIO(data)
            r.status, r.headers = status, {}
            return r
        return op

    def test_fetch_writes_after_check(self):
        data = b"model-bytes"
        sha = hashlib.sha256(data).hexdigest()
        out = handler.run_fetch({"items": [{"path": "models/loras/a.safetensors",
                                            "url": "https://example.com/a", "size": len(data),
                                            "sha256": sha}]}, self.root, opener=self.opener(data), env={})
        self.assertEqual(out["results"][0]["ok"], True)
        p = pathlib.Path(self.root, "models/loras/a.safetensors")
        self.assertEqual(p.read_bytes(), data)
        self.assertFalse(pathlib.Path(str(p) + ".gw-part").exists())

    def test_fetch_sha_mismatch_publishes_nothing(self):
        out = handler.run_fetch({"items": [{"path": "models/x", "url": "https://e/x", "size": 3,
                                            "sha256": "0" * 64}]}, self.root,
                                opener=self.opener(b"abc"), env={})
        self.assertFalse(out["results"][0]["ok"])
        self.assertTrue(out["results"][0]["error"].startswith("final: "))
        self.assertEqual([p for p in pathlib.Path(self.root).rglob("*") if p.is_file()], [])

    def test_fetch_refuses_bad_paths(self):
        items = [{"path": p, "url": "https://e/x", "size": 1} for p in
                 ("models/../../etc/x", "/abs", "other/x", "models/a\x00b", "hf-cache/../x")]
        items.append({"path": "models/ok", "url": "https://e/x", "size": 1})
        out = handler.run_fetch({"items": items}, self.root, opener=self.opener(b"z"), env={})
        self.assertEqual([r["ok"] for r in out["results"]], [False] * 5 + [True])

    def test_fetch_refuses_http_url(self):
        out = handler.run_fetch({"items": [{"path": "models/a", "url": "http://e/a", "size": 1}]},
                                self.root, opener=self.opener(b"z"), env={})
        self.assertFalse(out["results"][0]["ok"])

    def test_hf_token_only_from_env_and_only_to_hf(self):
        env = {"HF_TOKEN": "hf_secret"}
        handler.run_fetch({"items": [{"path": "models/a", "url": "https://huggingface.co/r/a",
                                      "size": 1}], "hf_token": "from_input"},
                          self.root, opener=self.opener(b"z"), env=env)
        handler.run_fetch({"items": [{"path": "models/b", "url": "https://huggingface.co.evil.example/b",
                                      "size": 1}]}, self.root, opener=self.opener(b"z"), env=env)
        auth = [r.get_header("Authorization") for r in self.seen]
        self.assertEqual(auth, ["Bearer hf_secret", None])

    def test_link_stays_inside_hf_cache(self):
        pathlib.Path(self.root, "hf-cache/hub/m/blobs").mkdir(parents=True)
        pathlib.Path(self.root, "hf-cache/hub/m/blobs/abc").write_bytes(b"x")
        out = handler.run_link({"links": [
            {"path": "hf-cache/hub/m/snapshots/r/f.bin", "target": "../../blobs/abc"},
            {"path": "hf-cache/hub/m/snapshots/r/g.bin", "target": "../../../../../../etc/passwd"}]},
            self.root)
        self.assertEqual([r["ok"] for r in out["results"]], [True, False])
        self.assertTrue(pathlib.Path(self.root, "hf-cache/hub/m/snapshots/r/f.bin").is_symlink())
    def test_fetch_size_mismatch_is_final(self):
        """A short body must never publish a model or consume three URL attempts."""
        out = handler.run_fetch({"items": [{"path": "models/a", "url": "https://e/a",
                                            "size": 4}]}, self.root,
                                opener=self.opener(b"abc"), env={})
        self.assertTrue(out["results"][0]["error"].startswith("final: "))
        self.assertFalse(pathlib.Path(self.root, "models/a.gw-part").exists())

    def test_fetch_resume_and_range_ignored(self):
        """Hash the prefix on 206; a server ignoring Range must not duplicate it."""
        data = b"abcdef"
        for status, body in ((206, b"def"), (200, data)):
            with self.subTest(status=status):
                p = pathlib.Path(self.root, "models/a")
                p.parent.mkdir(exist_ok=True)
                pathlib.Path(str(p) + ".gw-part").write_bytes(b"abc")
                out = handler.run_fetch({"items": [{"path": "models/a", "url": "https://e/a",
                                                    "size": 6,
                                                    "sha256": hashlib.sha256(data).hexdigest()}]},
                                        self.root, opener=self.opener(body, status), env={})
                self.assertTrue(out["results"][0]["ok"], out)
                self.assertEqual(p.read_bytes(), data)
                self.assertEqual(self.seen[-1].get_header("Range"), "bytes=3-")

    def test_fetch_resumed_mismatch_is_retryable(self):
        """A stale .gw-part (an older file's head) + a correct tail fails the hash; that
        is no reason to give the URL up for good — the part is gone, the next try
        starts at byte 0. Marked final, the sync would fall back to the LAN at once."""
        data = b"abcdef"
        pathlib.Path(self.root, "models").mkdir(exist_ok=True)
        pathlib.Path(self.root, "models/a.gw-part").write_bytes(b"XYZ")
        out = handler.run_fetch({"items": [{"path": "models/a", "url": "https://e/a", "size": 6,
                                            "sha256": hashlib.sha256(data).hexdigest()}]},
                                self.root, opener=self.opener(b"def", 206), env={})
        r = out["results"][0]
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"], "sha256 mismatch")
        self.assertFalse(pathlib.Path(self.root, "models/a.gw-part").exists())

    def test_fetch_http_errors_and_transport_are_per_item(self):
        """Only 4xx are final; an unavailable URL must not stop the batch."""
        import urllib.error
        for code in (403, 404, 429, 500, None):
            def fail(req, timeout=None):
                if code is None:
                    raise urllib.error.URLError("offline")
                raise urllib.error.HTTPError(req.full_url, code, "refused", {}, None)
            out = handler.run_fetch({"items": [{"path": "models/a", "url": "https://e/a",
                                                "size": 1}]}, self.root, opener=fail, env={})
            r = out["results"][0]
            self.assertFalse(r["ok"])
            self.assertEqual(r["error"].startswith("final: "), code is not None and code < 500)
            self.assertLessEqual(len(r["error"]), 300)

    def test_safe_rel_boundaries(self):
        """Odd path spellings and oversized keys must not bypass the two roots."""
        for p in (None, "", "models", "models/./a", "models//a", "models/a/..",
                  "models/a\\b", "models/" + "a" * 506):
            self.assertIsNone(handler.safe_rel(p), p)
        self.assertEqual(handler.safe_rel("models/a/"), "models/a")

    def test_symlink_escapes_refused(self):
        """Lexical containment alone lets existing symlinks write outside the volume."""
        outside = pathlib.Path(self.root, "outside")
        outside.mkdir()
        pathlib.Path(self.root, "models").symlink_to(outside)
        out = handler.run_fetch({"items": [{"path": "models/a", "url": "https://e/a",
                                            "size": 1}]}, self.root,
                                opener=self.opener(b"z"), env={})
        self.assertFalse(out["results"][0]["ok"])
        cache = pathlib.Path(self.root, "hf-cache")
        cache.mkdir()
        (cache / "escape").symlink_to(outside)
        out = handler.run_link({"links": [{"path": "hf-cache/a", "target": "escape/x"},
                                          {"path": "hf-cache/escape/a", "target": "../x"}]},
                               self.root)
        self.assertEqual([r["ok"] for r in out["results"]], [False, False])
        self.assertEqual(list(outside.iterdir()), [])

    def test_link_replaces_and_is_idempotent(self):
        """Repeated syncs preserve the correct link and replace stale files atomically."""
        p = pathlib.Path(self.root, "hf-cache/a")
        p.parent.mkdir()
        p.write_bytes(b"stale")
        payload = {"links": [{"path": "hf-cache/a", "target": "blobs/x"},
                             {"path": "models/a", "target": "blobs/x"},
                             {"path": "hf-cache/b", "target": "/etc/passwd"}]}
        for _ in range(2):
            out = handler.run_link(payload, self.root)
            self.assertEqual([r["ok"] for r in out["results"]], [True, False, False])
            self.assertEqual(p.readlink(), pathlib.Path("blobs/x"))

    def test_volume_space_and_info(self):
        """A missing mount reports None; info exposes filesystem capacity in bytes."""
        from unittest.mock import patch
        from types import SimpleNamespace
        with patch.object(handler.os, "statvfs", return_value=SimpleNamespace(
                f_blocks=100, f_frsize=4096, f_bfree=40, f_bavail=30)):
            self.assertEqual(handler.volume_space(self.root),
                             {"total": 409600, "used": 245760, "free": 122880})
            with patch.object(handler, "_http", return_value=(200, b"{}")):
                self.assertEqual(handler.run_info("http://e", {})["volume"]["free"], 122880)
        self.assertIsNone(handler.volume_space(self.root + "/absent"))

    def test_fetch_link_before_comfy_readiness(self):
        """Volume ops must work even when ComfyUI failed to start."""
        from unittest.mock import patch, Mock
        from types import SimpleNamespace
        sdk = SimpleNamespace(serverless=SimpleNamespace(start=Mock()))
        with patch.dict(sys.modules, {"runpod": sdk}), \
                patch.object(handler.subprocess, "Popen"), \
                patch.object(handler, "wait_ready", return_value=False), \
                patch.object(handler, "VOLUME", self.root):
            handler.main()
        handle = sdk.serverless.start.call_args[0][0]["handler"]
        self.assertEqual(handle({"input": {"op": "fetch", "items": []}}), {"results": []})
        self.assertEqual(handle({"input": {"op": "link", "links": []}}), {"results": []})
        self.assertIn("error", handle({"input": {"op": "prompt"}}))

    def test_token_not_forwarded_on_redirect(self):
        """urllib normally copies headers to a CDN redirect, leaking the HF secret."""
        import urllib.request
        handler.run_fetch({"items": [{"path": "models/a", "url": "https://huggingface.co/a",
                                      "size": 1}]}, self.root,
                          opener=self.opener(b"z"), env={"HF_TOKEN": "hf_secret"})
        redirected = urllib.request.HTTPRedirectHandler().redirect_request(
            self.seen[0], None, 302, "Found", {}, "https://cdn.example/a")
        self.assertIsNone(redirected.get_header("Authorization"))

    def test_hf_hosts_and_bad_items_continue(self):
        """Subdomains and hf.co get env auth; malformed items do not lose later results."""
        items = [None, {"path": "models/a", "url": "https://e/a", "size": -1}]
        items += [{"path": "models/" + str(i), "url": url, "size": 1}
                  for i, url in enumerate(("https://hf.co/a", "https://sub.huggingface.co/a",
                                           "https://hf.co.evil.example/a"))]
        out = handler.run_fetch({"items": items}, self.root,
                                opener=self.opener(b"z"), env={"HF_TOKEN": "secret"})
        self.assertEqual([r["ok"] for r in out["results"]], [False, False, True, True, True])
        self.assertEqual([r.get_header("Authorization") for r in self.seen],
                         ["Bearer secret", "Bearer secret", None])

    def test_mismatch_preserves_previous_final(self):
        """A failed refresh must leave the last verified model available."""
        p = pathlib.Path(self.root, "models/a")
        p.parent.mkdir()
        p.write_bytes(b"good")
        out = handler.run_fetch({"items": [{"path": "models/a", "url": "https://e/a", "size": 3,
                                            "sha256": "0" * 64}]}, self.root,
                                opener=self.opener(b"bad"), env={})
        self.assertFalse(out["results"][0]["ok"])
        self.assertEqual(p.read_bytes(), b"good")

    def test_complete_partial_restarts(self):
        """An already full partial must not append another copy on retry."""
        p = pathlib.Path(self.root, "models/a.gw-part")
        p.parent.mkdir()
        p.write_bytes(b"old")
        out = handler.run_fetch({"items": [{"path": "models/a", "url": "https://e/a", "size": 3}]},
                                self.root, opener=self.opener(b"new"), env={})
        self.assertTrue(out["results"][0]["ok"])
        self.assertEqual(pathlib.Path(self.root, "models/a").read_bytes(), b"new")
        self.assertIsNone(self.seen[-1].get_header("Range"))

    def test_link_target_resolves_symlinks_before_parent_segments(self):
        """Collapsing '..' before symlinks hides a target escaping the cache."""
        cache = pathlib.Path(self.root, "hf-cache")
        (cache / "parent").mkdir(parents=True)
        (cache / "sub").mkdir()
        (cache / "parent/deep").symlink_to(cache / "sub")
        out = handler.run_link({"links": [{"path": "hf-cache/parent/deep/a",
                                          "target": "../../secret"}]}, self.root)
        self.assertFalse(out["results"][0]["ok"])
        self.assertFalse((cache / "sub/a").is_symlink())


    def test_sizeless_fetch_checks_hash_and_reports_actual_size(self):
        """HEAD-refusing URLs must still download safely with only the optional hash check."""
        data = b'abc'
        out = handler.run_fetch({'items': [dict(path='models/a', url='https://e/a', size=None,
                    sha256=hashlib.sha256(data).hexdigest())]}, self.root,
                    opener=self.opener(data), env={}, ismount=lambda root: True)
        self.assertTrue(out['results'][0]['ok'])
        self.assertEqual(out['results'][0]['size'], 3)
        out = handler.run_fetch({'items': [dict(path='models/b', url='https://e/b', size=None,
                    sha256='0' * 64)]}, self.root,
                    opener=self.opener(data), env={}, ismount=lambda root: True)
        self.assertFalse(out['results'][0]['ok'])
        self.assertFalse(pathlib.Path(self.root, 'models/b').exists())

    def test_unmounted_fetch_and_link_refused_before_writing(self):
        """A worker without its network mount must not publish files into the container."""
        out = handler.run_fetch({'items': [dict(path='models/a', url='https://e/a', size=3)]},
                                self.root, opener=self.opener(b'abc'), ismount=lambda root: False)
        self.assertEqual(out, {'error': 'network volume not mounted at /runpod-volume'})
        self.assertEqual(handler.run_link({'links': []}, self.root, ismount=lambda root: False), out)
        self.assertEqual(self.seen, [])


if __name__ == "__main__":
    unittest.main()
