"""The rules of ComfyUI-KimodoHub (ops/comfyui-kimodo-hub) that motion-kimodo's
contract rests on — each one replaced an upstream ComfyUI-Kimodo behaviour that fails
SILENTLY:

- Segments: upstream splits the prompt on PERIODS. A client's line "Walk forward. Then
  wave." became two segments, i.e. a clip twice as long as the client computed.
- Variants: upstream Kimodo_Sampler hands ONE segment to the model as a list, and the
  model then sizes the batch from that list — a request for 3 variants came back with
  one result, `done` (measured 2026-10-12, job 2ee76afbb2ec).
- Post-process: upstream Kimodo_PostProcess corrects the 77-joint output against the
  30-joint skeleton, fails on every SOMA run and passes the motion on uncorrected
  (measured 2026-10-12, job 33c944a55735 once the failure was made loud).
- Rig: upstream returns "" for a missing rig path and the run ends without a file.
- Delivery order: one file per sample, index-suffixed, reported in sample order — the
  client maps results[n] to sample n. Upstream exported ONE clamped index (index 3 of a
  one-sample run was sample 0 again: a duplicate presented as a variant).
"""
import importlib.util
import os
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "kimodo_hub_core", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "ops", "comfyui-kimodo-hub", "kimodo_hub_core.py"))
core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(core)


class Segments(unittest.TestCase):
    def test_one_segment_per_line(self):
        self.assertEqual(core.prompt_segments("Walk forward. Then wave\n\n  jump  high \n"),
                         ["Walk forward. Then wave.", "jump high."])

    def test_empty_prompt_is_refused(self):
        with self.assertRaises(ValueError):
            core.prompt_segments(" \n\n")


class Sampler(unittest.TestCase):
    def test_one_segment_goes_in_as_a_string(self):
        args, kw = core.sampler_call(["Walk."], 120, 3, 100, 0.04)
        self.assertEqual(args, ("Walk.", 120))
        self.assertEqual(kw["num_samples"], 3)
        self.assertNotIn("multi_prompt", kw)

    def test_post_processing_runs_inside_the_model_call(self):
        _args, kw = core.sampler_call(["Walk."], 120, 1, 100, 0.04)
        self.assertTrue(kw["post_processing"])
        self.assertEqual(kw["root_margin"], 0.04)

    def test_several_segments_are_a_sequence(self):
        args, kw = core.sampler_call(["Walk.", "Wave."], 120, 2, 100, 0.04)
        self.assertEqual(args, (["Walk.", "Wave."], [120, 120]))
        self.assertTrue(kw["multi_prompt"])
        self.assertEqual(kw["num_samples"], 2)

    def test_variant_count_is_checked(self):
        with self.assertRaises(ValueError):
            core.sampler_call(["Walk."], 120, 5, 100, 0.04)


class Rig(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)

    def _file(self, name, data):
        p = os.path.join(self.d.name, name)
        with open(p, "wb") as fh:
            fh.write(data)
        return p

    def test_absolute_and_input_relative(self):
        p = self._file("gw_1_input_rig_path.bin", b"Kaydara FBX Binary  \x00")
        self.assertEqual(core.resolve_rig(p, "/nowhere"), p)
        self.assertEqual(core.resolve_rig("gw_1_input_rig_path.bin", self.d.name), p)

    def test_empty_missing_or_not_fbx_is_refused(self):
        self._file("x.glb", b"glTF\x02\x00\x00\x00")
        for v in ("", "  ", "nope.fbx", "x.glb"):
            with self.assertRaises(ValueError):
                core.resolve_rig(v, self.d.name)


class Outputs(unittest.TestCase):
    def test_one_path_per_sample_in_order(self):
        ps = core.output_paths("/c/output", "3D/kimodo", 3, "ab12")
        self.assertEqual(ps, [f"/c/output/3D/kimodo_ab12_{i}.fbx" for i in range(3)])

    def test_prefix_cannot_leave_the_output_dir(self):
        with self.assertRaises(ValueError):
            core.output_paths("/c/output", "../../etc/x", 1, "ab12")

    def test_variant_range(self):
        for n in (1, 4):
            self.assertEqual(core.check_variants(n), n)
        for n in (0, 5, True, "2"):
            with self.assertRaises(ValueError):
                core.check_variants(n)


if __name__ == "__main__":
    unittest.main()
