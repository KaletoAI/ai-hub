"""GPU load and VRAM in the job page's "Time per node" table (admin._node_times_table).

Why this file exists (it fails SILENTLY — the job page just loses a section, or the
whole page answers 500 for one odd row): the table reads `meta.node_times` as it was
written by whatever version ran the job, and by a backend-fed recorder. Pinned here:
  - a record WITHOUT device readings (every job before 2026-11-18, every box without
    a monitor) renders exactly the old table — no empty GPU/VRAM columns, no curve;
  - only the columns that have data appear (VRAM without Crystools, no GPU column);
  - a malformed `hw` block or node value never raises and is never echoed unescaped;
  - the section never contains a <script> (live pages morph and insert none).

Run: venv/bin/python -m unittest tests.test_node_times_table -v
"""
import os
import sys
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import admin  # noqa: E402


def _rec(hw=None, a=None, b=None):
    rec = {"v": 1, "total_s": 10.0,
           "nodes": [{"id": "4", "title": "Loader", "s": 2.0, **(a or {})},
                     {"id": "34", "title": "KSampler", "s": 8.0, "steps": 35, **(b or {})}],
           "cached": []}
    if hw is not None:
        rec["hw"] = hw
    return {"node_times": rec}


FULL = _rec({"gpu": [10, 50, 100, 90], "vram": [1024, 2048, 3072, 3072],
             "gpu_avg": 75, "vram_max": 3072, "vram_total": 8192},
            {"gpu": 12, "vram": 2048}, {"gpu": 98, "vram": 3072})


class NodeTimesHw(unittest.TestCase):

    def test_without_hw_is_the_old_table(self):
        out = admin._node_times_table(_rec())
        self.assertIn("Time per node", out)
        for s in ("GPU", "VRAM", "<svg"):
            self.assertNotIn(s, out)

    def test_full_record(self):
        out = admin._node_times_table(FULL)
        self.assertEqual(out.count("<svg"), 2)
        self.assertIn("⌀ 75%", out)
        self.assertIn("3.0 / 8.0 GB", out)
        self.assertIn("98%", out)
        self.assertIn("2.0 GB", out)
        self.assertIn("whole device", out)
        self.assertNotIn("<script", out)

    def test_only_vram(self):
        out = admin._node_times_table(_rec({"vram": [512, 900], "vram_max": 900},
                                           {"vram": 512}, {"vram": 900}))
        self.assertIn("VRAM", out)
        self.assertNotIn("GPU", out)
        self.assertIn("900 MB", out)
        self.assertEqual(out.count("<svg"), 1)

    def test_node_values_without_run_block(self):
        out = admin._node_times_table(_rec(None, {"gpu": 40}, {}))
        self.assertIn("GPU", out)
        self.assertIn("40%", out)
        self.assertNotIn("<svg", out)

    def test_malformed_is_harmless(self):
        for hw in ("x", 7, [], {"gpu": ["a", None], "vram": "zz", "gpu_avg": "<b>",
                                "vram_max": None, "vram_total": "<i>"},
                   {"gpu": [5], "gpu_avg": True}, {"vram": [1, 2], "vram_max": -5}):
            out = admin._node_times_table(_rec(hw, {"gpu": "<b>", "vram": "<b>"},
                                               {"gpu": None, "vram": -1}))
            self.assertIn("Time per node", out)
            self.assertNotIn("<b>", out)
            self.assertNotIn("<i>", out)

    def test_spark(self):
        self.assertEqual(admin._spark([5], 100, "#fff"), "")
        self.assertEqual(admin._spark(["a", None, True], 100, "#fff"), "")
        svg = admin._spark([0, 50, 100], 100, "#fff")
        self.assertTrue(svg.startswith("<svg") and svg.endswith("</svg>"))
        self.assertNotIn("nan", admin._spark([0, 0, 0], 0, "#fff"))


if __name__ == "__main__":
    unittest.main()
