"""Playground file uploads: mesh-param detection, the job-artifact picker's data
source, and the multipart parser's filename.

Worth a test file for the same reason test_prune_branch.py is: every failure here is
SILENT. A mesh param mis-detected as a scalar renders a text box again (the bug this
feature fixes); a dropped upload filename turns a .fbx into `application/octet-stream`
and the API can no longer recover the extension, so ComfyUI's loader refuses a file
that looks fine in the UI.

    venv/bin/python -m unittest tests.test_playground_files -v
"""
import asyncio
import os
import tempfile
import unittest

import adapters
import admin
import jobs


# ── adapters.is_file_param / file_params ────────────────────────────────────────

class FileParamDetection(unittest.TestCase):
    def test_param_name_says_mesh(self):
        self.assertTrue(adapters.is_file_param("input_mesh_path",
                                               {"node": "9", "field": "value"}))

    def test_plain_scalar_is_not_a_file(self):
        self.assertFalse(adapters.is_file_param("steps", {"field": "steps"}))

    def test_workflow_field_says_file(self):
        # node-based param name (value_9-style) with a telling workflow field
        self.assertTrue(adapters.is_file_param("x", {"field": "file_path"}))

    def test_image_slot_name_is_not_a_file(self):
        self.assertFalse(adapters.is_file_param("input_image", {"label": "input_image"}))

    def test_label_says_mesh(self):
        # the public label is what a client binds to; the raw param may be node-based
        self.assertTrue(adapters.is_file_param("value_9", {"label": "input_mesh_path"}))

    def test_prompt_is_not_a_file(self):
        self.assertFalse(adapters.is_file_param("prompt", {"label": "prompt"}))


WF = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}},
    "3": {"class_type": "KSampler", "inputs": {"steps": 20}},
    "9": {"class_type": "PrimitiveString", "inputs": {"value": "/mnt/meshes/a.glb"},
          "_meta": {"title": "input_mesh_path"}},
}
MAPPING = {
    "input_image": {"node": "1", "field": "image"},
    "steps": {"node": "3", "field": "steps"},
    "value_9": {"node": "9", "field": "value", "label": "input_mesh_path"},
}


class FileParams(unittest.TestCase):
    def test_only_the_mesh_param(self):
        self.assertEqual(adapters.file_params(WF, MAPPING), ["value_9"])

    def test_image_params_untouched(self):
        self.assertEqual(adapters.image_params(WF, MAPPING), ["input_image"])


# ── jobs.recent_artifacts ───────────────────────────────────────────────────────

class Blob:
    """Duck-type of adapters' GenBlob (the only shape jobs.complete reads)."""
    def __init__(self, data, mime, kind, name=None):
        self.data, self.mime, self.kind, self.name = data, mime, kind, name


PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + b"\x00\x00\x00\x08\x00\x00\x00\x08")


class RecentArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        jobs.init(os.path.join(self.tmp.name, "jobs.db"),
                  os.path.join(self.tmp.name, "blobs"), 3600)

    def test_results_and_inputs(self):
        jid = jobs.create("img2mesh", "mesh-mia", "comfy-a")
        jobs.set_inputs(jid, {"params": {"steps": 20}}, [("input_image", PNG)])
        jobs.complete(jid, [Blob(b"glTF-ish", "model/gltf-binary", "file", "hero.glb")],
                      {"backend": "comfy-a"})
        rows = jobs.recent_artifacts(limit=60)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["id"], jid)
        self.assertEqual(row["alias"], "mesh-mia")
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["results"][0]["kind"], "file")
        self.assertEqual(row["results"][0]["name"], "hero.glb")
        self.assertEqual(row["inputs"][0]["slot"], "input_image")
        self.assertEqual(row["inputs"][0]["n"], 0)

    def test_unnamed_result_falls_back_to_the_stored_filename(self):
        jid = jobs.create("text2img", "sdxl", "comfy-a")
        jobs.complete(jid, [Blob(PNG, "image/png", "image")], None)
        row = jobs.recent_artifacts()[0]
        self.assertEqual(row["results"][0]["kind"], "image")
        self.assertEqual(row["results"][0]["name"], "0.png")

    def test_chat_rows_stay_out(self):
        jobs.create("response", "gpt-alias", "openai")
        self.assertEqual(jobs.recent_artifacts(), [])


# ── admin._multipart keeps the upload filename ──────────────────────────────────

class FakeRequest:
    def __init__(self, body: bytes, ctype: str):
        self._body, self.headers = body, {"content-type": ctype}

    async def body(self) -> bytes:
        return self._body



# ── generate(): what the playground actually POSTs ──────────────────────────────
# The stash survives a model switch on purpose (same-named slots carry over), which
# makes "an alias that cannot consume this input" the interesting case: a mesh left
# behind by a rig alias must not be handed to a text2img alias as a reference IMAGE.
# Reproduced before the fix: body carried images={"input_mesh_path": <glb bytes>}.

MESH_ALIAS = [{"backend": "comfy-a", "task": "mesh2rig",
               "workflow_json": {"9": {"class_type": "PrimitiveString",
                                       "inputs": {"value": "/mnt/m.glb"}}},
               "mapping": {"input_mesh_path": {"node": "9", "field": "value",
                                               "label": "input_mesh_path"}}}]
TXT_ALIAS = [{"backend": "comfy-a", "task": "text2img",
              "workflow_json": {"6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                                "3": {"class_type": "KSampler", "inputs": {"steps": 20}}},
              "mapping": {"prompt": {"node": "6", "field": "text"},
                          "steps": {"node": "3", "field": "steps"}}}]


class FakeGenRequest:
    """Just enough Request for admin.generate (multipart body, no session cookie)."""
    base_url = "http://gw/"
    cookies: dict = {}

    def __init__(self, parts):
        b = "----gwtest"
        raw = b""
        for name, val, *fn in parts:
            disp = f'; filename="{fn[0]}"' if fn else ""
            raw += (f'--{b}\r\nContent-Disposition: form-data; name="{name}"{disp}\r\n\r\n'
                    ).encode() + (val if isinstance(val, bytes) else val.encode()) + b"\r\n"
        self._body = raw + f"--{b}--\r\n".encode()
        self.headers = {"content-type": f"multipart/form-data; boundary={b}"}

    async def body(self):
        return self._body


class GenerateBody(unittest.TestCase):
    def setUp(self):
        import store
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        store.init(os.path.join(self.tmp.name, "store.db"))
        jobs.init(os.path.join(self.tmp.name, "jobs.db"),
                  os.path.join(self.tmp.name, "blobs"), 3600)
        store.upsert("mesh-rig", MESH_ALIAS)
        store.upsert("sdxl", TXT_ALIAS)
        self.sent = {}

        class Resp:
            status_code = 200
            text = ""

            def json(self):
                return {"job_id": "job-1"}

        async def fake_self_api(request, method, path, **kw):
            self.sent.clear()
            self.sent.update(kw.get("json") or {})
            return Resp()

        real = admin._self_api
        admin._self_api = fake_self_api
        self.addCleanup(lambda: setattr(admin, "_self_api", real))
        admin._pg_images.clear()
        self.addCleanup(admin._pg_images.clear)

    def _post(self, parts):
        self.sent.clear()                       # so "no API call" stays observable
        return asyncio.run(admin.generate(FakeGenRequest(parts)))

    def test_mesh_upload_rides_as_a_file_not_an_image(self):
        self._post([("model", "mesh-rig"), ("p__input_mesh_path", "/backend/old.glb"),
                    ("file__input_mesh_path", b"GLBBYTES", "hero.glb")])
        self.assertTrue(self.sent["files"]["input_mesh_path"]
                        .startswith("data:model/gltf-binary;base64,"))
        self.assertNotIn("images", self.sent)
        # the upload wins over the typed path — binding the node twice would 400
        self.assertNotIn("input_mesh_path", self.sent.get("params", {}))

    def test_stashed_mesh_never_leaks_into_an_alias_without_upload_fields(self):
        self._post([("model", "mesh-rig"),
                    ("file__input_mesh_path", b"GLBBYTES", "hero.glb")])
        self._post([("model", "sdxl"), ("p__prompt", "a cat"), ("p__steps", "25")])
        self.assertNotIn("images", self.sent)   # ← the bug: images={"input_mesh_path": glb}
        self.assertNotIn("files", self.sent)
        self.assertEqual(self.sent["prompt"], "a cat")
        self.assertEqual(self.sent["params"]["steps"], 25)

    def test_unnamed_mesh_upload_still_carries_its_type(self):
        # a client may send a file part with an EMPTY filename (a part with no
        # filename at all is not a file part for _multipart — it arrives as text)
        self._post([("model", "mesh-rig"), ("file__input_mesh_path", b"GLBBYTES", "")])
        self.assertTrue(self.sent["files"]["input_mesh_path"]
                        .startswith("data:model/gltf-binary;base64,"))


class MultipartFilename(unittest.TestCase):
    def test_file_part_carries_its_name(self):
        b = "----gwtest"
        body = (
            f"--{b}\r\n"
            'Content-Disposition: form-data; name="model"\r\n\r\n'
            "mesh-mia\r\n"
            f"--{b}\r\n"
            'Content-Disposition: form-data; name="file__input_mesh_path"; filename="hero.glb"\r\n'
            "Content-Type: model/gltf-binary\r\n\r\n"
            "GLBBYTES\r\n"
            f"--{b}--\r\n"
        ).encode()
        f = asyncio.run(admin._multipart(FakeRequest(body, f'multipart/form-data; boundary={b}')))
        self.assertEqual(f["model"], "mesh-mia")
        self.assertEqual(f["file__input_mesh_path"], b"GLBBYTES")
        self.assertEqual(f["file__input_mesh_path__filename"], "hero.glb")
        self.assertNotIn("model__filename", f)      # scalar parts stay single keys


if __name__ == "__main__":
    unittest.main()
