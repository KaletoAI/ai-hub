"""🎲 random: a pinned value or a request-field default that is drawn fresh per job.

Why this fails SILENTLY: a seed baked into the workflow (or pinned) makes every job of
an alias the SAME job — Trellis2-Humanoid-Multiview delivered the identical mesh for
the same three images, run after run, and nothing says why. The obvious fix, mapping
the seed as a request field, ALSO failed: the auto-seed drew up to 2**63-1 while the
Trellis2 nodes declare `max: 0x7fffffff`, so ComfyUI refused the prompt.

So: one draw range every common seed widget accepts (0..0x7fffffff, also numpy's
32-bit limit), a `random: true` flag on a pin (always drawn, the API cannot override
it) and on a request field (drawn unless the request sends a value), both set by a 🎲
checkbox in the alias editor, and every drawn value named in the job summary so a
good result can be reproduced.

Run: venv/bin/python -m unittest tests.test_random_values -v
"""
import copy
from html.parser import HTMLParser
import unittest
from unittest.mock import patch

import admin
import adapters
from tests.test_gen_build import _req, _submitted


class _Inputs(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.inputs = {}
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "input" and a.get("name"):
            self.inputs[a["name"]] = a


class DrawRange(unittest.TestCase):
    def test_draws_stay_inside_the_32bit_signed_range(self):
        """2**63-1 is rejected by every node that declares max 0x7fffffff (Trellis2)."""
        with patch.object(adapters.random, "randint", lambda a, b: (a, b)):
            self.assertEqual(adapters.random_seed(), (0, 0x7fffffff))

    def test_the_auto_seed_uses_the_same_range(self):
        req = _req()
        req.params.pop("seed")
        calls = []
        orig = adapters.random_seed

        def spy():
            calls.append(1)
            return orig()
        with patch.object(adapters, "random_seed", spy):
            comfy = _submitted(req)
        self.assertTrue(calls)
        self.assertEqual(comfy.prompts[0]["4"]["inputs"]["seed"], 42)


class PinnedRandom(unittest.TestCase):
    def test_apply_fixed_draws_instead_of_the_stored_value(self):
        wf = {"34": {"class_type": "Gen", "inputs": {"seed": 123239}}}
        with patch.object(adapters.random, "randint", lambda a, b: 777):
            applied = adapters._apply_fixed(wf, [{"node": "34", "field": "seed",
                                                  "value": 123239, "random": True}])
        self.assertEqual(wf["34"]["inputs"]["seed"], 777)
        self.assertEqual(applied, {"34.seed": 777})

    def test_a_random_pin_reaches_comfyui_and_the_client_cannot_override_it(self):
        req = _req()
        req.fixed.append({"node": "4", "field": "seed", "value": 5, "random": True})
        comfy = _submitted(req)          # randint is patched to 42; the client sent seed=7
        self.assertEqual(comfy.prompts[0]["4"]["inputs"]["seed"], 42)


class RequestFieldRandom(unittest.TestCase):
    def _req(self, send):
        req = _req()
        req.node_mapping["steps"]["random"] = True
        req.fixed = [b for b in req.fixed if b["field"] != "steps"]
        if not send:
            req.params.pop("steps")
        return req

    def test_drawn_when_the_request_omits_it(self):
        self.assertEqual(_submitted(self._req(False)).prompts[0]["4"]["inputs"]["steps"], 42)

    def test_a_sent_value_wins(self):
        self.assertEqual(_submitted(self._req(True)).prompts[0]["4"]["inputs"]["steps"], 30)

    def test_the_schema_says_it_is_random_unless_sent(self):
        cand = {"workflow_json": {"34": {"class_type": "Gen", "inputs": {"seed": 1}}},
                "mapping": {"seed_34": {"node": "34", "field": "seed",
                                        "label": "input_seed", "random": True}}}
        params, _, _ = adapters.public_fields(cand)
        self.assertEqual(params[0]["auto"], "random unless sent")


class Summary(unittest.TestCase):
    def test_every_drawn_value_is_named(self):
        req = _req()
        req.fixed.append({"node": "4", "field": "seed", "value": 5, "random": True})
        req.node_mapping["steps"]["random"] = True
        req.fixed = [b for b in req.fixed if b["field"] != "steps"]
        req.params.pop("steps")
        seen = {}
        orig = adapters.ComfyUIAdapter._build_prompt

        async def spy(self, r, io):
            built = await orig(self, r, io)
            seen.update(built.summary)
            return built
        with patch.object(adapters.ComfyUIAdapter, "_build_prompt", spy):
            _submitted(req)
        self.assertEqual(seen.get("random"), {"4.seed": 42, "4.steps": 42})


class Editor(unittest.TestCase):
    WF = {"34": {"class_type": "Gen", "inputs": {"seed": 1, "sampler": "euler"}}}

    def test_pinned_int_gets_a_dice_box_strings_do_not(self):
        fixed = [{"node": "34", "field": "seed", "value": 1, "random": True},
                 {"node": "34", "field": "sampler", "value": "euler"}]
        html = admin._pin_tab_rows("a", {"backend": "g"}, True, fixed, self.WF, {})
        inputs = _Inputs(html).inputs
        self.assertIn("checked", inputs["fixedrnd__34__seed"])
        self.assertNotIn("fixedrnd__34__sampler", inputs)

    def test_request_field_int_gets_a_dice_box(self):
        mapping = {"seed": {"node": "34", "field": "seed", "random": True},
                   "sampler": {"node": "34", "field": "sampler"}}
        inputs = _Inputs(admin._req_fields_rows("a", self.WF, mapping, {})).inputs
        self.assertIn("checked", inputs["random__seed"])
        self.assertNotIn("random__sampler", inputs)

    def test_save_stores_the_flags_on_every_candidate(self):
        cands = [{"backend": "a", "workflow_json": copy.deepcopy(self.WF)},
                 {"backend": "b", "workflow_json": copy.deepcopy(self.WF)}]
        admin._apply_update_form(cands, {
            "node__seed": "34", "field__seed": "seed", "random__seed": "on",
            "fixed__34__sampler": "euler",
            "fixed__34__seed": "5", "fixedrnd__34__seed": "on"})
        for c in cands:
            self.assertTrue(c["mapping"]["seed"]["random"])
            pins = {b["field"]: b for b in c["fixed"]}
            self.assertTrue(pins["seed"]["random"])
            self.assertNotIn("random", pins["sampler"])

    def test_an_unticked_box_clears_the_flag(self):
        cands = [{"backend": "a", "workflow_json": copy.deepcopy(self.WF),
                  "fixed": [{"node": "34", "field": "seed", "value": 5, "random": True}],
                  "mapping": {"seed": {"node": "34", "field": "seed", "random": True}}}]
        admin._apply_update_form(cands, {"node__seed": "34", "field__seed": "seed",
                                         "fixed__34__seed": "5"})
        self.assertNotIn("random", cands[0]["mapping"]["seed"])
        self.assertNotIn("random", cands[0]["fixed"][0])


if __name__ == "__main__":
    unittest.main()
