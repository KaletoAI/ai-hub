"""A chain's `mesh_param` names the successor's request field that carries the mesh.
Both adapters SILENTLY drop a param they don't know, so a wrong name does not raise:
stage 2 would run on its workflow's baked-in mesh path (or, on Meshy, on no mesh at
all) and deliver a stale/WRONG mesh as a "done" job. main._chain_mesh_param_error is
the guard that turns that into an up-front failure, and it is pure over the successor
candidate — these tests pin it for both successor kinds.

Run: python -m unittest tests.test_chain_mesh_param -v
"""
import os
import sys
import tempfile
import unittest

# `import main` reads ./config.yaml at import time — give it a minimal one in a temp cwd.
# The dir is needed ONLY for that import, so it is removed in the same finally that
# restores the cwd; leaving it to the finalizer raises a ResourceWarning under -W error.
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (tests/ is one level down)
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


# A Meshy rigging alias: no mapping at all — its request fields are the fixed label
# table, and the mesh is the FILE field `input_mesh_path`.
MESHY_RIG = {"backend": "meshy", "meshy": {"endpoint": "rigging"}}
# A Meshy image-to-3d alias takes no file at all (it cannot be a rigging successor).
MESHY_I23 = {"backend": "meshy", "meshy": {"endpoint": "image-to-3d"}}
# The same two shapes on the OTHER cloud kind: the guard must read the candidate's KIND,
# not the literal `meshy` key (Tripo's rig endpoint is `rig`, its file field the same name).
TRIPO_RIG = {"backend": "tripo", "tripo": {"endpoint": "rig"}}
TRIPO_I2M = {"backend": "tripo", "tripo": {"endpoint": "image-to-model"}}
# A ComfyUI successor: the mesh-load input is mapped under the label `input_mesh_path`,
# while its raw param name is node-based (`value_12`) and must not be needed.
COMFY_RIG = {"backend": "gpu", "workflow_json": {"12": {"inputs": {"mesh": ""}}},
             "mapping": {"value_12": {"node": "12", "field": "mesh",
                                      "label": "input_mesh_path"}}}


class TestChainMeshParam(unittest.TestCase):
    def test_meshy_successor_accepts_its_file_field(self):
        self.assertIsNone(main._chain_mesh_param_error(MESHY_RIG, "input_mesh_path", "rig"))

    def test_meshy_successor_rejects_a_comfy_style_name(self):
        why = main._chain_mesh_param_error(MESHY_RIG, "mesh_path", "rig")
        self.assertIsNotNone(why)
        self.assertIn("mesh_path", why)
        self.assertIn("input_mesh_path", why)      # names what it DOES take
        self.assertIn("rig", why)

    def test_meshy_successor_without_any_file_input(self):
        why = main._chain_mesh_param_error(MESHY_I23, "input_mesh_path", "img23")
        self.assertIsNotNone(why)
        self.assertIn("no file input at all", why)

    def test_tripo_successor_accepts_its_file_field(self):
        self.assertIsNone(main._chain_mesh_param_error(TRIPO_RIG, "input_mesh_path", "rig"))

    def test_tripo_successor_rejects_a_comfy_style_name(self):
        why = main._chain_mesh_param_error(TRIPO_RIG, "mesh_path", "rig")
        self.assertIsNotNone(why)
        self.assertIn("Tripo", why)                # names the VENDOR, never a hardcoded "Meshy"
        self.assertIn("input_mesh_path", why)      # names what it DOES take
        self.assertIn("rig", why)

    def test_tripo_successor_without_any_file_input(self):
        why = main._chain_mesh_param_error(TRIPO_I2M, "input_mesh_path", "img2m")
        self.assertIsNotNone(why)
        self.assertIn("no file input at all", why)

    def test_comfy_successor_accepts_a_mapping_label(self):
        self.assertIsNone(main._chain_mesh_param_error(COMFY_RIG, "input_mesh_path", "unirig"))

    def test_comfy_successor_accepts_the_raw_param_name(self):
        self.assertIsNone(main._chain_mesh_param_error(COMFY_RIG, "value_12", "unirig"))

    def test_comfy_successor_rejects_an_unmapped_name(self):
        why = main._chain_mesh_param_error(COMFY_RIG, "mesh_path", "unirig")
        self.assertIsNotNone(why)
        self.assertIn("request field", why)
        self.assertIn("unirig", why)


if __name__ == "__main__":
    unittest.main()
