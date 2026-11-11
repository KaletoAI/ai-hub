"""The cost next to a cloud job's Output heading names its source and unit.

Why this fails SILENTLY: every vendor records its bill under a different key and unit
— wiro's `cost_usd` is USD (its `consumed_credits` stays None), Meshy/Tripo/LU book
`consumed_credits`, RunPod only an execution-time estimate, and Venice/NanoGPT nothing.
Reading the wrong key shows a plausible but wrong figure (wiro USD as "credits", or a
catalog list price passed off as the bill); a failed job must not show a list price
it may never have paid. Local (ComfyUI) jobs carry no line at all.

Run: venv/bin/python -m unittest tests.test_job_cost_line -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import admin  # noqa: E402


class CostLine(unittest.TestCase):
    def test_recorded_cost_wins_with_its_own_unit(self):
        self.assertEqual(admin._job_cost_text(
            {"cloud": "wiro", "cost_usd": 0.035, "consumed_credits": None}, "wiro/x"), "cost $0.035")
        self.assertEqual(admin._job_cost_text(
            {"cloud": "meshy", "consumed_credits": 20}, "Meshy-Low"), "cost 20 credits")
        self.assertEqual(admin._job_cost_text(
            {"runpod_job_id": "r1", "cost_est_usd": 0.0123}, "rp"), "cost ≈ $0.0123 (estimate)")

    def test_unrecorded_cost_falls_back_to_the_labelled_list_price(self):
        with mock.patch.object(admin, "_media_price",
                               lambda model, aliases=None: {"usd": 0.04, "unit": "image"}):
            self.assertEqual(admin._job_cost_text({"cloud": "venice"}, "venice.ai/m"),
                             "price list from $0.04/image")
            self.assertEqual(admin._job_cost_text({"cloud": "venice"}, "venice.ai/m", "failed"), "")
        with mock.patch.object(admin, "_media_price", lambda model, aliases=None: None):
            self.assertEqual(admin._job_cost_text({"cloud": "venice"}, "venice.ai/m"), "cost unknown")

    def test_local_job_has_no_cost_line(self):
        self.assertEqual(admin._job_cost_text({"inputs": {}}, "Flux-Dev"), "")


if __name__ == "__main__":
    unittest.main()
