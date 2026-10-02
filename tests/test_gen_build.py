"""What `_build_prompt` hands to a backend — pinned so the build/execute/deliver split
(and later the RunPod adapter, which reuses the build with another GenIO) cannot change
it silently.

Why this fails SILENTLY: the built workflow is never shown anywhere. A refactor that
drops the label→param aliasing, prunes one node too many or forgets a pin still submits
a VALID prompt — ComfyUI runs it and delivers a plausible picture of the wrong thing.
Only a byte-for-byte comparison of the submitted workflow catches that.

Run: venv/bin/python -m unittest tests.test_gen_build -v
"""
import asyncio
import copy
import json
import os
import sys
import unittest
import unittest.mock

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import httpx  # noqa: E402
import adapters  # noqa: E402

# One request exercising every build step: a label-aliased image slot with an upload,
# an empty `disable` slot whose dead branch is pruned, a mapped scalar (+ auto seed),
# an admin pin the client tries to override, a client file param, an empty unmapped
# loader (autofill) and a backend bypass.
WF = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
    "2": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "3": {"class_type": "ImageScale", "inputs": {"image": ["2", 0], "width": 64}},
    "4": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 20, "cfg": 4.0}},
    "5": {"class_type": "PrimitiveString", "inputs": {"value": ""}},
    "6": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "7": {"class_type": "ImageInvert", "inputs": {"image": ["1", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "o"}},
}
MAPPING = {
    "image": {"node": "1", "field": "image", "label": "input_image"},
    "image2": {"node": "2", "field": "image", "on_empty": "disable"},
    "seed": {"node": "4", "field": "seed"},
    "steps": {"node": "4", "field": "steps"},
    "mesh": {"node": "5", "field": "value"},
}
OBJECT_INFO = {
    "ImageScale": {"input": {"required": {"image": ["IMAGE"], "width": ["INT", {}]}},
                   "output": ["IMAGE"]},
    "ImageInvert": {"input": {"required": {"image": ["IMAGE"]}}, "output": ["IMAGE"]},
    "LoadImage": {"input": {"required": {"image": [["x.png"], {}]}}, "output": ["IMAGE", "MASK"]},
    "KSampler": {"input": {"required": {"seed": ["INT", {}]}}, "output": ["LATENT"]},
    "PrimitiveString": {"input": {"required": {"value": ["STRING", {}]}}, "output": ["STRING"]},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"]}}, "output": []},
}
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40


def _req():
    return adapters.NormalizedRequest(
        alias="a", job_id="job1", upload_prefix="gw_job1",
        workflow_json=copy.deepcopy(WF), node_mapping=copy.deepcopy(MAPPING),
        params={"steps": 30, "seed": 7},
        fixed=[{"node": "4", "field": "cfg", "value": 2.5},
               {"node": "4", "field": "steps", "value": 12}],
        upload_images={"input_image": PNG},
        upload_files={"mesh": ("m.glb", b"glTF" + b"\x00" * 20)},
        bypass=["7"])


def _ctx():
    return adapters.AdapterContext(
        auth_headers=lambda b: {}, inflight_inc=lambda b: None, inflight_dec=lambda b: None,
        cost_usd=lambda *a: 0.0, source_of=lambda r: "t", record_call=lambda *a, **k: None,
        log_enabled=lambda: False)


class _Comfy:
    """A ComfyUI stub: /upload/image echoes the name, /object_info/<cls> answers from
    OBJECT_INFO, /prompt records the submitted workflow."""

    def __init__(self):
        self.prompts, self.uploads = [], []

    def handler(self, request):
        p = request.url.path
        if p == "/upload/image":
            body = request.read()
            name = body.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
            self.uploads.append(name)
            return httpx.Response(200, json={"name": name, "subfolder": ""})
        if p.startswith("/object_info/"):
            cls = p.rsplit("/", 1)[1]
            return httpx.Response(200, json={cls: OBJECT_INFO[cls]} if cls in OBJECT_INFO else {})
        if p == "/prompt":
            self.prompts.append(json.loads(request.read())["prompt"])
            return httpx.Response(400, json={"error": {"message": "stop here"}})   # the build is all we want
        return httpx.Response(404, json={})


def _adapter():
    b = {"name": "gpu", "type": "comfyui", "url": "http://comfy",
         "comfy_input_dir": "/srv/comfy/input"}
    return adapters.ComfyUIAdapter(b, _ctx())


def _submitted(req):
    """The workflow a ComfyUI backend receives for `req` — via the public generate()."""
    comfy = _Comfy()
    real = httpx.AsyncClient
    with unittest.mock.patch.object(
            adapters.httpx, "AsyncClient",
            lambda *a, **kw: real(transport=httpx.MockTransport(comfy.handler),
                                  **{k: v for k, v in kw.items() if k != "transport"})):
        with unittest.mock.patch.object(adapters.random, "randint", lambda a, b: 42):
            try:
                asyncio.run(_adapter().generate(req))
            except RuntimeError:
                pass
    return comfy


EXPECTED = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "gw_job1_image.png"}},
    "4": {"class_type": "KSampler", "inputs": {"seed": 7, "steps": 12, "cfg": 2.5}},
    "5": {"class_type": "PrimitiveString",
          "inputs": {"value": "/srv/comfy/input/gw_job1_mesh.glb"}},
    "6": {"class_type": "LoadImage", "inputs": {"image": "gw_placeholder.png"}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "o"}},
}


class Characterization(unittest.TestCase):
    def test_the_submitted_workflow_is_exactly_this(self):
        comfy = _submitted(_req())
        self.assertEqual(len(comfy.prompts), 1)
        self.assertEqual(comfy.prompts[0], EXPECTED)
        self.assertIn("gw_job1_image.png", comfy.uploads)
        self.assertIn("gw_job1_mesh.glb", comfy.uploads)


class _FakeIO(adapters.GenIO):
    """No HTTP at all: names in, names out — what a RunPod-style io does."""

    def __init__(self):
        self.puts = []

    async def put_image(self, name, data):
        self.puts.append(name)
        return name

    async def put_file(self, name, data):
        self.puts.append(name)
        return name

    async def placeholder(self):
        return "gw_placeholder.png"

    def input_ref(self, stored):
        return f"/srv/comfy/input/{stored}"

    async def node_types(self, wf, ids):
        return {cls: adapters._node_type_entry(info) for cls, info in OBJECT_INFO.items()}


class IoEquivalence(unittest.TestCase):
    def test_another_io_builds_the_same_workflow(self):
        with unittest.mock.patch.object(adapters.random, "randint", lambda a, b: 42):
            built = asyncio.run(_adapter()._build_prompt(_req(), _FakeIO()))
        self.assertEqual(built.wf, EXPECTED)
        self.assertEqual(sorted(built.io.puts), ["gw_job1_image.png", "gw_job1_mesh.glb"])
        self.assertEqual(built.uploaded, ["gw_job1_image.png", "gw_job1_mesh.glb"])


class FetchSeam(unittest.TestCase):
    """Delivery reads files through ONE callable — a 404 probes on, anything else
    raises, exactly as with /view."""

    def _fetch(self, files, calls):
        async def fetch(params):
            calls.append(dict(params))
            key = params.get("filename")
            if key in files:
                return files[key]
            return 404, b""
        return fetch

    def test_sibling_preferred_and_404_probes_on(self):
        outputs = {"9": {"result": [{"filename": "m.fbx", "type": "output"}]}}
        calls = []
        fetch = self._fetch({"m.glb": (200, b"glb")}, calls)
        blobs = asyncio.run(_adapter()._fetch_outputs(fetch, {}, outputs, "9", "glb", None))
        self.assertEqual([(b.name, b.data) for b in blobs], [("m.glb", b"glb")])

    def test_a_non_404_raises(self):
        outputs = {"9": {"images": [{"filename": "a.png", "type": "output"}]}}
        fetch = self._fetch({"a.png": (500, b"")}, [])
        with self.assertRaises(RuntimeError):
            asyncio.run(_adapter()._fetch_outputs(fetch, {}, outputs, "9", None, None))

    def test_globs_through_fetch(self):
        outputs = {"9": {"images": [{"filename": "x_mia.fbx", "type": "output"}]}}
        fetch = self._fetch({"x_mia.glb": (200, b"g")}, [])
        blobs = asyncio.run(_adapter()._fetch_by_globs(fetch, outputs, ["*_mia.glb"]))
        self.assertEqual([b.name for b in blobs], ["x_mia.glb"])


if __name__ == "__main__":
    unittest.main()
