"""Chain hooks: what a stage-1 / stage-2 adapter contributes to a workflow chain.
run: venv/bin/python -m unittest tests.test_chain_hooks -v"""
import asyncio
import unittest

import adapters
from adapters import ComfyUIAdapter, GenBlob, GenOutput, NormalizedRequest

WF = {
    "47": {"inputs": {"value": "Kai"}, "class_type": "PrimitiveString", "_meta": {"title": "input_name"}},
    "100": {"inputs": {"filename_prefix": ["47", 0], "file_format": "glb", "trimesh": ["107", 0]},
            "class_type": "Trellis2ExportMesh", "_meta": {"title": "Output"}},
}


def _ctx():
    return adapters.AdapterContext(
        auth_headers=lambda b: {}, inflight_inc=lambda bid: None, inflight_dec=lambda bid: None,
        cost_usd=lambda *a: 0.0, source_of=lambda r: "test", record_call=lambda *a, **k: None,
        log_enabled=lambda: False)


def _comfy():
    return ComfyUIAdapter({"name": "gpu", "type": "comfyui", "url": "http://127.0.0.1:1",
                           "comfy_input_dir": "/srv/comfy/input"}, _ctx())


class ComfyChainExport(unittest.TestCase):
    def test_pins_export_node_and_names_mesh(self):
        ex = _comfy().chain_export({"workflow_json": WF}, {"export_node": "100"}, {}, "gwchain_j1")
        self.assertIsNone(ex.error)
        self.assertEqual(ex.mesh_name, "gwchain_j1_00001_.glb")
        self.assertEqual(ex.extra_fixed, [{"node": "100", "field": "filename_prefix", "value": "gwchain_j1"}])

    def test_mapped_file_format_overrides_ext(self):
        cand = {"workflow_json": WF, "mapping": {"fmt": {"node": "100", "field": "file_format", "label": "input_format"}}}
        ex = _comfy().chain_export(cand, {"export_node": "100"}, {"input_format": "obj"}, "gwchain_j1")
        self.assertEqual(ex.mesh_name, "gwchain_j1_00001_.obj")

    def test_pinned_file_format_beats_mapping(self):
        cand = {"workflow_json": WF,
                "mapping": {"fmt": {"node": "100", "field": "file_format"}},
                "fixed": [{"node": "100", "field": "file_format", "value": "fbx"}]}
        ex = _comfy().chain_export(cand, {"export_node": "100"}, {"fmt": "obj"}, "gwchain_j1")
        self.assertEqual(ex.mesh_name, "gwchain_j1_00001_.fbx")

    def test_bad_export_node_is_an_error_not_a_crash(self):
        ex = _comfy().chain_export({"workflow_json": WF}, {"export_node": "47"}, {}, "gwchain_j1")
        self.assertIsNotNone(ex.error)
        self.assertIn("filename_prefix", ex.error)

    def test_no_workflow_is_an_error(self):
        ex = _comfy().chain_export({}, {"export_node": "100"}, {}, "gwchain_j1")
        self.assertIsNotNone(ex.error)


class ComfyChainFeed(unittest.TestCase):
    def test_path_relay_returns_shared_disk_path(self):
        req2 = NormalizedRequest(alias="rig")
        ref = asyncio.run(_comfy().chain_feed_mesh(req2, {"name": "gpu"}, "input_mesh_path",
                                                   "gwchain_j1_00001_.glb", None, "/srv/comfy/output"))
        self.assertEqual(ref, "/srv/comfy/output/gwchain_j1_00001_.glb")
        self.assertEqual(req2.upload_files, {})


class ComfyChainFeedUpload(unittest.TestCase):
    def test_upload_relay_returns_the_input_dir_path(self):
        """The bytes go into the stage-2 backend's input dir; what stage 2 is handed is
        that file's ABSOLUTE path there (a bare stored name only loads via a
        load-from-input node)."""
        ad = _comfy()
        backend2 = {"comfy_input_dir": "/srv/comfy/input"}
        seen = {}

        async def fake_upload(data, name, content_type="application/octet-stream"):
            seen["args"] = (data, name)
            return "sub/gwchain_j1_00001_.glb"

        ad.upload_input = fake_upload
        req2 = NormalizedRequest(alias="rig")
        ref = asyncio.run(ad.chain_feed_mesh(req2, backend2, "input_mesh_path",
                                             "gwchain_j1_00001_.glb", b"glTFxxxx", "/srv/comfy/output"))
        self.assertEqual(ref, adapters.input_path_ref(backend2, "sub/gwchain_j1_00001_.glb"))
        self.assertEqual(seen["args"], (b"glTFxxxx", "gwchain_j1_00001_.glb"))


class MeshyChainStage1(unittest.TestCase):
    def _ad(self):
        return adapters.MeshyAdapter({"name": "meshy", "type": "meshy", "url": "http://127.0.0.1:1"}, _ctx())

    def test_export_names_glb_without_pins(self):
        cand = {"meshy": {"endpoint": "image-to-3d", "options": {"target_formats": ["glb"]}}}
        ex = self._ad().chain_export(cand, {"alias": "mesh-mia"}, {}, "gwchain_j1")
        self.assertIsNone(ex.error)
        self.assertEqual(ex.mesh_name, "gwchain_j1.glb")
        self.assertEqual(ex.extra_fixed, [])

    def test_export_requires_glb_format(self):
        cand = {"meshy": {"endpoint": "image-to-3d", "options": {"target_formats": ["fbx"]}}}
        ex = self._ad().chain_export(cand, {"alias": "mesh-mia"}, {}, "gwchain_j1")
        self.assertIn("glb", ex.error)

    def test_rigging_cannot_be_stage_1(self):
        """A rigging alias delivers `rigged.glb` — a stage-2 product, not a stage-1 mesh.
        Refused by NAME up front: it would otherwise spend credits and then die on the
        generic "stage-1 produced no mesh" (chain_take_mesh looks for `model.glb`)."""
        cand = {"meshy": {"endpoint": "rigging", "options": {"target_formats": ["glb"]}}}
        ex = self._ad().chain_export(cand, {"alias": "mesh-mia"}, {}, "gwchain_j1")
        self.assertIsNotNone(ex.error)
        self.assertIn("rigging", ex.error)
        self.assertEqual(ex.mesh_name, "")

    def test_take_mesh_from_blobs(self):
        out = GenOutput(blobs=[GenBlob(b"glTFxxxx", "model/gltf-binary", "file", "model.glb"),
                               GenBlob(b"png", "image/png", "image", "preview.png")])
        ex = adapters.ChainExport("gwchain_j1.glb")
        self.assertEqual(asyncio.run(self._ad().chain_take_mesh(out, ex, True)), b"glTFxxxx")
        self.assertEqual(asyncio.run(self._ad().chain_take_mesh(out, ex, False)), b"")
        self.assertIsNone(asyncio.run(self._ad().chain_take_mesh(GenOutput(blobs=[]), ex, True)))


class MeshyChainStage2(unittest.TestCase):
    """A Meshy successor reads the mesh off the REQUEST (embedded as a model_url data
    URI by meshy.build_request) — there is no backend disk to upload it to."""

    def _ad(self):
        return adapters.MeshyAdapter({"name": "meshy", "type": "meshy", "url": "http://127.0.0.1:1"}, _ctx())

    def test_feed_embeds_upload(self):
        req2 = NormalizedRequest(alias="Meshy-Rig", meshy={"endpoint": "rigging", "options": {}})
        ref = asyncio.run(self._ad().chain_feed_mesh(req2, {"name": "meshy"}, "input_mesh_path",
                                                     "gwchain_j1.glb", b"glTF" + b"\0" * 1_048_576, ""))
        self.assertEqual(req2.upload_files["input_mesh_path"][0], "gwchain_j1.glb")
        self.assertEqual(len(req2.upload_files["input_mesh_path"][1]), 1_048_580)
        self.assertEqual(ref, "<upload:gwchain_j1.glb (1.0 MB)>")

    def test_feed_needs_bytes(self):
        """No path relay to a cloud backend: without the bytes there is nothing to send,
        and a path from someone else's disk would silently mean nothing to Meshy."""
        with self.assertRaises(RuntimeError):
            asyncio.run(self._ad().chain_feed_mesh(NormalizedRequest(), {}, "input_mesh_path",
                                                   "n", None, "/out"))


class UploadTimeout(unittest.TestCase):
    """The chain relays MESHES through ComfyUI's /upload/image, not LAN-sized images:
    measured 2026-09-02, a no-remesh Meshy humanoid came back at 70 MB, which the flat
    20 s budget could not push through. The budget must grow with the file."""

    def test_small_upload_keeps_the_flat_floor(self):
        self.assertEqual(adapters._upload_timeout_for(0), adapters._UPLOAD_TIMEOUT)
        # a LAN-sized image is still essentially the old flat budget
        self.assertLess(adapters._upload_timeout_for(64 * 1024),
                        adapters._UPLOAD_TIMEOUT + 0.1)

    def test_large_upload_gets_at_least_one_mib_per_second(self):
        mb = 1024 * 1024
        self.assertAlmostEqual(adapters._upload_timeout_for(70 * mb), 90.0)
        self.assertGreaterEqual(adapters._upload_timeout_for(70 * mb), 70.0)   # ≥1 MiB/s
        # monotonic: a bigger file never gets a smaller budget
        self.assertGreater(adapters._upload_timeout_for(100 * mb),
                           adapters._upload_timeout_for(50 * mb))


class BaseDefaults(unittest.TestCase):
    def test_base_adapter_refuses_chain_roles(self):
        base = adapters.OpenAIAdapter({"name": "llm", "type": "openai", "url": "http://x"}, _ctx())
        ex = base.chain_export({}, {}, {}, "p")
        self.assertIsNotNone(ex.error)
        self.assertIn("openai", ex.error)
        with self.assertRaises(RuntimeError):
            asyncio.run(base.chain_feed_mesh(NormalizedRequest(), {}, "m", "n", b"x", ""))
        self.assertIsNone(asyncio.run(base.chain_take_mesh(GenOutput(blobs=[]), ex, True)))


if __name__ == "__main__":
    unittest.main()
