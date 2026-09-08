"""Dead-branch pruning for `on_empty: disable` image slots (adapters._prune_branch).

The cascade rule is what ComfyUI itself enforces: a node whose REQUIRED input link
disappears cannot run, so it dies with the branch; a node that only lost an OPTIONAL
one keeps running without it. Getting that backwards is invisible until a real
generation aborts at /prompt, so it is worth a test.

Fixture mirrors the shape of img2mesh-trellis2_multiview_api.json (verified against
the live /object_info of a Trellis2 backend, 2026-08-30):
    loader → PreProcess(image required) → MultiView(front required, back/left/right optional)

    venv/bin/python -m unittest tests.test_prune_branch -v
"""
import unittest

import adapters


TYPES = {
    "Loader":    {"out": ["IMAGE"], "in": {}, "req": ["image"]},
    "PreProcess": {"out": ["IMAGE"], "in": {"image": "IMAGE"},
                   "req": ["image", "padding", "remove_background"]},
    "MultiView": {"out": ["MESH"], "in": {"front_image": "IMAGE", "back_image": "IMAGE"},
                  "req": ["pipeline", "front_image"]},
    "Post":      {"out": ["MESH"], "in": {"mesh": "MESH"}, "req": ["mesh"]},
}


def _wf():
    """front + back through a PreProcess each, left wired straight into MultiView."""
    return {
        "58":  {"class_type": "Loader", "inputs": {"image": "front.png"}},
        "108": {"class_type": "Loader", "inputs": {"image": "back.png"}},
        "109": {"class_type": "Loader", "inputs": {"image": "left.png"}},
        "61":  {"class_type": "PrimitiveBoolean", "inputs": {"value": True}},
        "105": {"class_type": "PreProcess",
                "inputs": {"padding": 25, "remove_background": ["61", 0], "image": ["58", 2]}},
        "104": {"class_type": "PreProcess",
                "inputs": {"padding": 25, "remove_background": ["61", 0], "image": ["108", 2]}},
        "106": {"class_type": "MultiView",
                "inputs": {"seed": 1, "pipeline": ["82", 0], "front_image": ["105", 0],
                           "back_image": ["104", 0], "left_image": ["109", 2]}},
        "107": {"class_type": "Post", "inputs": {"mesh": ["106", 0]}},
        "82":  {"class_type": "LoadModel", "inputs": {"modelname": "x"}},
    }


class PruneBranch(unittest.TestCase):

    def test_required_consumer_dies_with_the_loader(self):
        """back loader gone → its PreProcess cannot run (image is required) → both go,
        and MultiView merely loses its OPTIONAL back_image."""
        wf = _wf()
        removed = adapters._prune_branch(wf, "108", TYPES)
        self.assertEqual(set(removed), {"108", "104"})
        self.assertNotIn("104", wf)
        self.assertIn("106", wf)
        self.assertNotIn("back_image", wf["106"]["inputs"])
        self.assertEqual(wf["106"]["inputs"]["front_image"], ["105", 0])

    def test_optional_consumer_survives(self):
        """left is wired straight into an OPTIONAL socket → only the loader goes."""
        wf = _wf()
        removed = adapters._prune_branch(wf, "109", TYPES)
        self.assertEqual(removed, ["109"])
        self.assertIn("106", wf)
        self.assertNotIn("left_image", wf["106"]["inputs"])

    def test_cascade_follows_a_required_chain_to_the_end(self):
        """front feeds a REQUIRED socket → the whole downstream chain is dead."""
        wf = _wf()
        removed = adapters._prune_branch(wf, "58", TYPES)
        self.assertEqual(set(removed), {"58", "105", "106", "107"})
        self.assertEqual(set(wf), {"108", "109", "61", "104", "82"})

    def test_sibling_link_from_a_surviving_node_is_kept(self):
        """The PreProcess also reads a boolean from 61 — pruning must not touch it."""
        wf = _wf()
        adapters._prune_branch(wf, "108", TYPES)
        self.assertIn("61", wf)
        self.assertEqual(wf["105"]["inputs"]["remove_background"], ["61", 0])

    def test_unknown_class_stops_the_cascade(self):
        """No /object_info for the consumer → fall back to removing just the node
        (the pre-cascade behaviour), never a guess about what it needs."""
        wf = _wf()
        removed = adapters._prune_branch(wf, "108", {})
        self.assertEqual(removed, ["108"])
        self.assertIn("104", wf)
        self.assertNotIn("image", wf["104"]["inputs"])   # link still dropped

    def test_absent_node_is_a_noop(self):
        wf = _wf()
        self.assertEqual(adapters._prune_branch(wf, "999", TYPES), [])
        self.assertEqual(len(wf), 9)

    def test_second_required_input_kills_a_shared_consumer(self):
        """A node with two REQUIRED image links dies when either one goes."""
        wf = {"1": {"class_type": "Loader", "inputs": {"image": "a.png"}},
              "2": {"class_type": "Loader", "inputs": {"image": "b.png"}},
              "3": {"class_type": "Batch", "inputs": {"image1": ["1", 0], "image2": ["2", 0]}}}
        types = {"Batch": {"out": ["IMAGE"], "in": {}, "req": ["image1", "image2"]}}
        removed = adapters._prune_branch(wf, "2", types)
        self.assertEqual(set(removed), {"2", "3"})
        self.assertEqual(set(wf), {"1"})


class SlotEmptyBypass(unittest.TestCase):
    """The `on_empty_bypass` companion: extra ids the empty slot bypasses (mode 4) on top
    of the pruned branch. Reading it under the WRONG mode would silently skip nodes on a
    slot that is merely placeholder-filled, so the mode gate is the point of these."""

    def test_ids_are_read_for_the_disable_mode(self):
        self.assertEqual(
            adapters.slot_empty_bypass({"on_empty": "disable", "on_empty_bypass": ["58", "61"]}),
            ["58", "61"])

    def test_other_modes_yield_nothing(self):
        for mode in ("placeholder", "required", None):
            m = {"on_empty_bypass": ["58"]}
            if mode:
                m["on_empty"] = mode
            self.assertEqual(adapters.slot_empty_bypass(m), [], mode)

    def test_values_are_normalized_to_stripped_strings(self):
        """Ints from a hand-edited store row and a comma string from a form both work."""
        self.assertEqual(
            adapters.slot_empty_bypass({"on_empty": "disable", "on_empty_bypass": [58, " 61 ", ""]}),
            ["58", "61"])
        self.assertEqual(
            adapters.slot_empty_bypass({"on_empty": "disable", "on_empty_bypass": "58, 61"}),
            ["58", "61"])

    def test_duplicates_collapse(self):
        """`_apply_bypass` returns what it was handed — a doubled id would be reported
        twice in the job summary as if the node had been skipped twice."""
        self.assertEqual(
            adapters.slot_empty_bypass({"on_empty": "disable",
                                        "on_empty_bypass": ["58", "61", "61", "58"]}),
            ["58", "61"])

    def test_missing_field_is_empty(self):
        self.assertEqual(adapters.slot_empty_bypass({"on_empty": "disable"}), [])
        self.assertEqual(adapters.slot_empty_bypass({}), [])
        self.assertEqual(adapters.slot_empty_bypass(None), [])


class EmptySlotBypassApplied(unittest.TestCase):
    """What the two mechanisms do TOGETHER for one empty slot: the loader's branch is
    pruned, the slot's extra ids are bypassed. The extra is for a node the cascade
    cannot take — its image socket is optional — but which only exists for that image.
    Bypassing keeps the path behind it; pruning would cut it, which is the whole reason
    this is a separate field."""

    def _wf_with_apply(self):
        """loader 58 → Apply 70 (image optional, model passthrough) → Sampler 71."""
        return {
            "58": {"class_type": "Loader", "inputs": {"image": "ref.png"}},
            "60": {"class_type": "LoadModel", "inputs": {"name": "m"}},
            "70": {"class_type": "Apply", "inputs": {"model": ["60", 0], "image": ["58", 0]}},
            "71": {"class_type": "Sampler", "inputs": {"model": ["70", 0]}},
        }

    TYPES = {"Apply": {"out": ["MODEL"], "in": {"model": "MODEL", "image": "IMAGE"},
                       "req": ["model"]},                      # image OPTIONAL → survives prune
             "Sampler": {"out": ["LATENT"], "in": {"model": "MODEL"}, "req": ["model"]},
             "Loader": {"out": ["IMAGE"], "in": {}, "req": ["image"]},
             "LoadModel": {"out": ["MODEL"], "in": {}, "req": ["name"]}}

    def test_prune_alone_leaves_the_apply_node_running_on_nothing(self):
        wf = self._wf_with_apply()
        adapters._prune_branch(wf, "58", self.TYPES)
        self.assertIn("70", wf)                                # optional socket → cascade stops
        self.assertNotIn("image", wf["70"]["inputs"])          # …but it has no image any more

    def test_extra_bypass_removes_it_and_keeps_the_path(self):
        wf = self._wf_with_apply()
        m = {"node": "58", "field": "image", "on_empty": "disable", "on_empty_bypass": ["70"]}
        adapters._prune_branch(wf, "58", self.TYPES)
        applied = adapters._apply_bypass(wf, adapters.slot_empty_bypass(m), self.TYPES)
        self.assertEqual(applied, ["70"])
        self.assertNotIn("70", wf)
        self.assertEqual(wf["71"]["inputs"]["model"], ["60", 0])   # rewired past it, not cut


class NodeTypeEntry(unittest.TestCase):

    def test_required_field_names_are_captured(self):
        """`req` carries EVERY required field (a required COMBO is still required),
        while `in` keeps only the link-typed ones as before."""
        e = adapters._node_type_entry({
            "output": ["IMAGE"],
            "input": {"required": {"image": ["IMAGE"], "mode": [["a", "b"]], "n": ["INT", {}]},
                      "optional": {"mask": ["MASK"]}}})
        self.assertEqual(set(e["req"]), {"image", "mode", "n"})
        self.assertEqual(e["in"], {"image": "IMAGE", "n": "INT", "mask": "MASK"})
        self.assertEqual(e["out"], ["IMAGE"])


if __name__ == "__main__":
    unittest.main()
