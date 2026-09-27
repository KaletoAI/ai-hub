"""Model-set derivation and resolution for synced backends (modelsync.py).
run: venv/bin/python -m unittest tests.test_modelsync -v

Every case here fails SILENTLY in production: a reference derived from an input that
only says HOW a model loads (`model_format = "GGUF Q4_K_M"`) blocks an alias forever, a
pin or bypass ignored syncs the wrong weights (tens of GB, billed per hour), a guessed
resolution copies the wrong file, and a catalog path onto `hf-cache/token` ships the
Hugging Face token to a rented machine."""
import unittest

import modelsync as ms

WF = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "A-nvfp4.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "LoaderGGUF", "inputs": {"gguf_name": "A-Q8.gguf"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
    "4": {"class_type": "Lora Loader Stack (rgthree)", "inputs": {"lora_01": "None", "lora_02": "x.safetensors"}},
    "5": {"class_type": "Trellis2LoadModel_GGUF", "inputs": {"modelname": "Pixal3D-GGUF", "model_format": "GGUF Q4_K_M"}},
    "6": {"class_type": "Hy3D21MeshGenerator", "inputs": {"model": "hunyuan3D-dit-v2-1-fp16.ckpt"}},
    "7": {"class_type": "LoadImage", "inputs": {"image": "in.png"}},
    "8": {"class_type": "Trellis2LoadModel", "inputs": {"modelname": "microsoft/TRELLIS.2-4B"}},
}


def vals(refs):
    return sorted(r.value for r in refs)


class Refs(unittest.TestCase):
    def test_pins_override_and_bypass_drops(self):
        cand = {"fixed": [{"node": "1", "field": "unet_name", "value": "A-fp8.safetensors"}], "bypass": ["2"]}
        v = vals(ms.refs_for(cand, WF, set()))
        self.assertIn("A-fp8.safetensors", v)
        self.assertNotIn("A-nvfp4.safetensors", v)
        self.assertNotIn("A-Q8.gguf", v)

    def test_none_lora_skipped_real_lora_kept(self):
        v = vals(ms.refs_for({}, WF, set()))
        self.assertNotIn("None", v)
        self.assertIn("x.safetensors", v)

    def test_model_format_is_not_a_ref(self):
        self.assertNotIn("GGUF Q4_K_M", vals(ms.refs_for({}, WF, set())))

    def test_file_extension_anywhere_counts(self):
        self.assertIn("hunyuan3D-dit-v2-1-fp16.ckpt", vals(ms.refs_for({}, WF, set())))

    def test_input_loader_and_dtype_ignored(self):
        v = vals(ms.refs_for({}, WF, set()))
        self.assertNotIn("in.png", v)
        self.assertNotIn("default", v)

    def test_hub_id_is_a_ref(self):
        self.assertIn("microsoft/TRELLIS.2-4B", vals(ms.refs_for({}, WF, set())))

    def test_selectable_flag(self):
        refs = ms.refs_for({}, WF, {("3", "vae_name")})
        self.assertTrue(next(r for r in refs if r.value == "flux2-vae.safetensors").selectable)

    # -- beyond the brief ------------------------------------------------------------

    def test_kinds(self):
        """Ruling 2: every weight-input value is a Ref, its kind says what it can do."""
        kinds = {r.value: r.kind for r in ms.refs_for({}, WF, set())}
        self.assertEqual(kinds["flux2-vae.safetensors"], "file")
        self.assertEqual(kinds["hunyuan3D-dit-v2-1-fp16.ckpt"], "file")
        self.assertEqual(kinds["microsoft/TRELLIS.2-4B"], "hub")
        self.assertEqual(kinds["Pixal3D-GGUF"], "name")

    def test_kind_derivation(self):
        self.assertEqual(ms.ref_kind("sub/x.safetensors"), "file")    # a subfolder is still a file
        self.assertEqual(ms.ref_kind("v1-inference.yaml"), "file")    # any .xxx ending
        self.assertEqual(ms.ref_kind("org/Model-2.5-VL"), "hub")      # a dot inside is no ending
        self.assertEqual(ms.ref_kind("Model2.5"), "name")             # ".5" is a version, not an ending
        self.assertEqual(ms.ref_kind("yoso-normal-v1-8-1"), "name")
        # the constructor derives it, so a hand-built Ref is never kind-less
        self.assertEqual(ms.Ref("1", "C", "f", "a/b").kind, "hub")
        self.assertEqual(ms.Ref("1", "C", "f", "a.gguf").kind, "file")

    def test_model_field_not_eaten_by_mode_rule(self):
        """`mode` is a HOW input, but it must not match `model`/`modelname`."""
        wf = {"1": {"class_type": "SomeModelLoader",
                    "inputs": {"model": "org/repo", "attention_mode": "sdpa", "model_type": "flux",
                               "quantization": "int8", "precision": "fp16", "weight_scheme": "x"}}}
        self.assertEqual(vals(ms.refs_for({}, wf, set())), ["org/repo"])

    def test_how_input_with_file_value_still_counts(self):
        """The any-class file rule is about VALUES: a model file is a model file."""
        wf = {"1": {"class_type": "Custom", "inputs": {"model_type": "y.safetensors"}}}
        self.assertEqual(vals(ms.refs_for({}, wf, set())), ["y.safetensors"])

    def test_links_numbers_and_empty_values_ignored(self):
        wf = {"1": {"class_type": "UNETLoader",
                    "inputs": {"unet_name": ["9", 0], "lora_name": "", "clip_name": "none",
                               "model_strength": 1.0, "vae_name": 3}}}
        self.assertEqual(ms.refs_for({}, wf, set()), [])

    def test_none_workflow_has_no_refs(self):
        self.assertEqual(ms.refs_for({"fixed": [{"node": "1", "field": "f", "value": "x.gguf"}]},
                                     None, set()), [])

    def test_one_ref_per_node_field(self):
        refs = ms.refs_for({}, WF, set())
        keys = [(r.node, r.field) for r in refs]
        self.assertEqual(len(keys), len(set(keys)))
        r = next(r for r in refs if r.value == "A-Q8.gguf")
        self.assertEqual((r.node, r.cls, r.field), ("2", "LoaderGGUF", "gguf_name"))

    def test_dict_form_pins(self):
        cand = {"fixed": {"1.unet_name": "A-fp8.safetensors"}}
        self.assertIn("A-fp8.safetensors", vals(ms.refs_for(cand, WF, set())))
        cand = {"fixed": {"1": {"unet_name": "A-fp8.safetensors"}}}
        self.assertIn("A-fp8.safetensors", vals(ms.refs_for(cand, WF, set())))

    def test_pin_rules_mirror_the_adapter(self):
        """A pin the adapter would not apply (blank value, unknown node, not a dict)
        must not change what gets synced either."""
        cand = {"fixed": [{"node": "1", "field": "unet_name", "value": ""},
                          {"node": "99", "field": "unet_name", "value": "ghost.safetensors"},
                          "garbage"]}
        v = vals(ms.refs_for(cand, WF, set()))
        self.assertIn("A-nvfp4.safetensors", v)
        self.assertNotIn("ghost.safetensors", v)

    def test_effective_workflow_is_a_copy(self):
        eff = ms.effective_workflow(WF, [{"node": "1", "field": "unet_name", "value": "B.safetensors"}], [3])
        self.assertEqual(eff["1"]["inputs"]["unet_name"], "B.safetensors")
        self.assertNotIn("3", eff)                                     # bypass ids compared as strings
        self.assertEqual(WF["1"]["inputs"]["unet_name"], "A-nvfp4.safetensors")
        self.assertIn("3", WF)

    def test_pin_coerced_like_the_adapter(self):
        wf = {"1": {"class_type": "Switch", "inputs": {"on": False, "n": 1}}}
        eff = ms.effective_workflow(wf, [{"node": "1", "field": "on", "value": "true"},
                                         {"node": "1", "field": "n", "value": "7"}], [])
        self.assertIs(eff["1"]["inputs"]["on"], True)
        self.assertEqual(eff["1"]["inputs"]["n"], 7)


IDX = {"models/vae/flux2-vae.safetensors": 300, "models/diffusion_models/A-fp8.safetensors": 9000,
       "models/loras/x.safetensors": 50, "models/diffusion_models/hunyuan3D-dit-v2-1-fp16.ckpt": 7000,
       "models/vae/dup.safetensors": 1, "models/checkpoints/dup.safetensors": 2,
       "models/microsoft/TRELLIS.2-4B/a.bin": 10, "models/microsoft/TRELLIS.2-4B/sub/b.bin": 20,
       "hf-cache/token": 1}


class Resolve(unittest.TestCase):
    def test_folder_candidates(self):
        r = ms.Ref("3", "VAELoader", "vae_name", "flux2-vae.safetensors")
        self.assertEqual(ms.resolve(r, IDX), "models/vae/flux2-vae.safetensors")

    def test_unique_basename_fallback(self):
        r = ms.Ref("6", "Hy3D21MeshGenerator", "model", "hunyuan3D-dit-v2-1-fp16.ckpt")
        self.assertEqual(ms.resolve(r, IDX), "models/diffusion_models/hunyuan3D-dit-v2-1-fp16.ckpt")

    def test_ambiguous_never_guesses(self):
        r = ms.Ref("9", "SomeCustomLoader", "file", "dup.safetensors")
        self.assertIsInstance(ms.resolve(r, IDX), ms.Ambiguous)

    def test_missing(self):
        r = ms.Ref("9", "VAELoader", "vae_name", "nope.safetensors")
        self.assertIsInstance(ms.resolve(r, IDX), ms.Missing)

    def test_expand_dir(self):
        self.assertEqual(sorted(ms.expand_dir("models/microsoft/TRELLIS.2-4B/", IDX)),
                         ["models/microsoft/TRELLIS.2-4B/a.bin", "models/microsoft/TRELLIS.2-4B/sub/b.bin"])

    def test_token_never_resolvable(self):
        # Ruling 1: "token" (not "token.bin") — the basename that WOULD hit the share
        r = ms.Ref("9", "X", "model", "token")
        self.assertIsInstance(ms.resolve(r, {"hf-cache/token": 1}), ms.Missing)

    # -- beyond the brief ------------------------------------------------------------

    def test_ambiguous_lists_options(self):
        r = ms.Ref("9", "SomeCustomLoader", "file", "dup.safetensors")
        self.assertEqual(ms.resolve(r, IDX).options,
                         ["models/checkpoints/dup.safetensors", "models/vae/dup.safetensors"])

    def test_folder_order_wins_over_ambiguity(self):
        """A loader whose folders pin the file resolves there, even though the same
        basename exists elsewhere — that is not a guess, it is where ComfyUI looks."""
        r = ms.Ref("3", "VAELoader", "vae_name", "dup.safetensors")
        self.assertEqual(ms.resolve(r, IDX), "models/vae/dup.safetensors")
        idx = {"models/unet/u.gguf": 1, "models/diffusion_models/u.gguf": 2}
        self.assertEqual(ms.resolve(ms.Ref("1", "UnetLoaderGGUF", "unet_name", "u.gguf"), idx),
                         "models/diffusion_models/u.gguf")

    def test_lora_classes_and_subfolder_values(self):
        idx = {"models/loras/style/y.safetensors": 5, "models/loras/other/y.safetensors": 6}
        r = ms.Ref("4", "Lora Loader Stack (rgthree)", "lora_02", "style/y.safetensors")
        self.assertEqual(ms.resolve(r, idx), "models/loras/style/y.safetensors")
        self.assertEqual(ms.folders_for("LoraLoaderModelOnly"), ("loras",))
        self.assertEqual(ms.folders_for("NoSuchClass"), ())

    def test_suffix_match_respects_segments(self):
        """`/x.safetensors` — never a file that merely ENDS in the same characters."""
        r = ms.Ref("9", "Custom", "f", "x.safetensors")
        self.assertIsInstance(ms.resolve(r, {"models/foo/prefix-x.safetensors": 1}), ms.Missing)

    def test_hf_cache_root_is_searched(self):
        r = ms.Ref("9", "Custom", "f", "w.safetensors")
        self.assertEqual(ms.resolve(r, {"hf-cache/hub/m/w.safetensors": 1}), "hf-cache/hub/m/w.safetensors")

    def test_hidden_and_foreign_keys_never_resolve(self):
        idx = {"models/.cache/w.safetensors": 1, "hf-cache/token/w.safetensors": 2,
               "other/w.safetensors": 3}
        self.assertIsInstance(ms.resolve(ms.Ref("9", "Custom", "f", "w.safetensors"), idx), ms.Missing)

    def test_unsafe_value_is_missing(self):
        idx = {"models/x.safetensors": 1, "models/vae/x.safetensors": 1}
        for bad in ("../x.safetensors", "/abs/x.safetensors", "sub\\x.safetensors"):
            self.assertIsInstance(ms.resolve(ms.Ref("3", "VAELoader", "vae_name", bad), idx), ms.Missing, bad)

    def test_hub_and_name_refs_are_catalog_only(self):
        """Ruling 2: a hub id / bare name never resolves by suffix — only an exact file
        in the loader's folder counts."""
        self.assertIsInstance(ms.resolve(ms.Ref("8", "Trellis2LoadModel", "modelname",
                                                "microsoft/TRELLIS.2-4B"), IDX), ms.Missing)
        idx = {"models/somewhere/Pixal3D-GGUF": 1}
        self.assertIsInstance(ms.resolve(ms.Ref("5", "Trellis2LoadModel_GGUF", "modelname",
                                                "Pixal3D-GGUF"), idx), ms.Missing)
        idx = {"models/checkpoints/plainname": 1}
        self.assertEqual(ms.resolve(ms.Ref("1", "CheckpointLoaderSimple", "ckpt_name", "plainname"), idx),
                         "models/checkpoints/plainname")

    def test_expand_dir_edges(self):
        idx = dict(IDX, **{"models/microsoft/TRELLIS.2-4B/.git/x": 1,
                           "models/microsoft/TRELLIS.2-4B-other/c.bin": 1})
        self.assertEqual(sorted(ms.expand_dir("models/microsoft/TRELLIS.2-4B/", idx)),
                         ["models/microsoft/TRELLIS.2-4B/a.bin", "models/microsoft/TRELLIS.2-4B/sub/b.bin"])
        self.assertEqual(ms.expand_dir("models/vae/flux2-vae.safetensors", IDX),
                         {"models/vae/flux2-vae.safetensors": 300})
        self.assertEqual(ms.expand_dir("hf-cache/", IDX), {})       # the token share never expands
        self.assertEqual(ms.expand_dir("hf-cache/token", IDX), {})
        self.assertEqual(ms.expand_dir("../", IDX), {})


class Catalog(unittest.TestCase):
    CAT = [{"match": {"class": "Trellis2LoadModel", "value": "microsoft/TRELLIS.2-4B"},
            "paths": ["models/microsoft/TRELLIS.2-4B/"]},
           {"match": {"alias": "mesh-shrink"}, "paths": []}]

    def test_class_value_match(self):
        refs = ms.refs_for({}, WF, set())
        paths, explicit = ms.catalog_paths(refs, "X", self.CAT)
        self.assertEqual(paths, ["models/microsoft/TRELLIS.2-4B/"])
        self.assertFalse(explicit)

    def test_alias_entry_empty_paths_is_explicit(self):
        self.assertEqual(ms.catalog_paths([], "mesh-shrink", self.CAT), ([], True))

    def test_validate(self):
        self.assertEqual(ms.validate_catalog(self.CAT), [])
        self.assertTrue(ms.validate_catalog([{"paths": ["../x"]}]))
        self.assertTrue(ms.validate_catalog([{"file": "models/x", "url": "ftp://x"}]))
        self.assertTrue(ms.validate_catalog([{"match": {"alias": "a"}, "paths": ["hf-cache/token"]}]))

    # -- beyond the brief ------------------------------------------------------------

    def test_name_ref_matches_catalog(self):
        """Ruling 2: a bare name is exactly what a class+value entry exists for."""
        cat = [{"match": {"class": "Trellis2LoadModel_GGUF", "value": "Pixal3D-GGUF"},
                "paths": ["models/pixal/", "hf-cache/hub/pixal/"]}]
        self.assertEqual(ms.catalog_paths(ms.refs_for({}, WF, set()), "X", cat),
                         (["models/pixal/", "hf-cache/hub/pixal/"], False))

    def test_all_match_keys_must_hold_and_paths_dedupe(self):
        cat = [{"match": {"class": "Trellis2LoadModel", "value": "other/repo"}, "paths": ["models/a/"]},
               {"match": {"value": "microsoft/TRELLIS.2-4B"}, "paths": ["models/b/"]},
               {"match": {"alias": "X"}, "paths": ["models/b/", "models/c/"]},
               {"match": {"alias": "Y"}, "paths": ["models/d/"]},
               {"file": "models/vae/flux2-vae.safetensors", "url": "https://example.com/v"}]
        self.assertEqual(ms.catalog_paths(ms.refs_for({}, WF, set()), "X", cat),
                         (["models/b/", "models/c/"], True))

    def test_catalog_paths_tolerates_junk(self):
        self.assertEqual(ms.catalog_paths([], "X", None), ([], False))
        self.assertEqual(ms.catalog_paths([], "X", ["junk", {"match": "x"}, {"match": {}}]), ([], False))

    def test_validate_details(self):
        ok = [{"file": "models/vae/v.safetensors", "url": "https://example.com/v",
               "sha256": "a" * 64},
              {"match": {"alias": "a"}, "paths": ["hf-cache/hub/models--org--repo/"]}]
        self.assertEqual(ms.validate_catalog(ok), [])
        bad = [
            "not a list",
            ["not a dict"],
            [{"match": {"alias": "a"}}],                                     # no paths
            [{"match": {"alias": "a"}, "paths": "models/x/"}],               # paths not a list
            [{"match": {}, "paths": []}],                                    # empty match
            [{"match": {"klass": "C"}, "paths": []}],                        # unknown match key
            [{"match": {"alias": 3}, "paths": []}],                          # non-string match
            [{"match": {"alias": "a"}, "paths": ["vae/x.safetensors"]}],     # no root prefix
            [{"match": {"alias": "a"}, "paths": ["models/"]}],               # a whole root
            [{"match": {"alias": "a"}, "paths": ["hf-cache/token/"]}],
            [{"match": {"alias": "a"}, "paths": ["hf-cache/"]}],             # would expand onto the token
            [{"match": {"alias": "a"}, "paths": ["models/.hidden/x"]}],
            [{"match": {"alias": "a"}, "paths": ["models/x/"], "path": ["models/y/"]}],  # typo'd key
            [{"file": "models/x/", "url": "https://e/x"}],                   # a file, not a dir
            [{"file": "models/x", "url": "https://e/x", "sha256": "abc"}],
            [{"file": "models/x", "url": "http://e/x"}],
            [{"file": "models/x", "url": "https://e/x\nheader = \"X: y\""}],   # curl config injection
            [{"file": "models/x", "url": "https://e/x\"y"}],                  # ends the quoted value
            [{"file": "models/x", "url": "https://e/x", "match": {"alias": "a"}, "paths": []}],
            [{"file": "hf-cache/token", "url": "https://e/x"}],
        ]
        for b in bad:
            self.assertTrue(ms.validate_catalog(b), b)

    def test_default_catalog_is_valid(self):
        self.assertEqual(ms.validate_catalog(ms.DEFAULT_CATALOG), [])


if __name__ == "__main__":
    unittest.main()
