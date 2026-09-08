"""Chain export-node validation (stdlib unittest, like test_anthropic_bridge.py).

Worth a test file for the same reason the bridge is: the failure is SILENT and
expensive. A chain pins `filename_prefix` on the successor's `export_node`; if
that node has no such input the pin is dropped without a word, stage 1 runs to
completion (tens of GPU-minutes for a mesh), and only the /view fetch afterwards
reports "produced no mesh". Node-id drift between two revisions of the same
workflow is the everyday cause.

Run: venv/bin/python -m unittest tests.test_chain_export_node -v
"""
import unittest

from adapters import ComfyUIAdapter


# Shaped like the real Trellis2 mesh workflows: the export node's filename_prefix
# arrives as a LINK (from a string node), which is still a pinnable input.
WF = {
    "47": {"inputs": {"value": "Kai"}, "class_type": "PrimitiveString",
           "_meta": {"title": "input_name"}},
    "82": {"inputs": {"modelname": "microsoft/TRELLIS.2-4B", "low_vram": True},
           "class_type": "Trellis2LoadModel", "_meta": {"title": "Trellis2 - LoadModel"}},
    "100": {"inputs": {"filename_prefix": ["47", 0], "file_format": "glb",
                       "trimesh": ["107", 0]},
            "class_type": "Trellis2ExportMesh", "_meta": {"title": "Output"}},
    "101": {"inputs": {"image": ""}, "class_type": "Preview3D",
            "_meta": {"title": "Preview 3D & Animation"}},
}


class ExportNodeValidation(unittest.TestCase):
    def test_real_export_node_passes(self):
        self.assertIsNone(ComfyUIAdapter.export_node_error(WF, "100"))

    def test_missing_node_is_named(self):
        err = ComfyUIAdapter.export_node_error(WF, "999")
        self.assertIsNotNone(err)
        self.assertIn("999", err)
        self.assertIn("not in the mesh workflow", err)

    def test_node_without_filename_prefix_is_rejected(self):
        # The live bug: export_node "82" was carried over from another revision of
        # the workflow, where 82 WAS the export node; here it is the model loader.
        err = ComfyUIAdapter.export_node_error(WF, "82")
        self.assertIsNotNone(err)
        self.assertIn("filename_prefix", err)
        self.assertIn("Trellis2LoadModel", err)      # says WHAT the node actually is

    def test_rejection_suggests_the_workflow_export_nodes(self):
        err = ComfyUIAdapter.export_node_error(WF, "82")
        self.assertIn("100", err)                    # the node it should have been
        self.assertIn("Trellis2ExportMesh", err)

    def test_blank_and_none_node_are_rejected_not_crashed(self):
        for node in ("", None):
            self.assertIsNotNone(ComfyUIAdapter.export_node_error(WF, node))


if __name__ == "__main__":
    unittest.main()
