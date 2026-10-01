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



# --- Task 10: the sync plan ------------------------------------------------------------

GB = 10 ** 9
SRC = {"models/diffusion_models/A.safetensors": 1000, "models/diffusion_models/A-Q8.gguf": 800,
       "models/vae/v.safetensors": 100, "models/loras/L.safetensors": 50,
       "models/vae/dup.safetensors": 1, "models/checkpoints/dup.safetensors": 2,
       "models/microsoft/TRELLIS.2-4B/a.bin": 10, "models/microsoft/TRELLIS.2-4B/sub/b.bin": 20,
       "hf-cache/token": 1}
R_UNET = ms.Ref("1", "UNETLoader", "unet_name", "A.safetensors")
R_GGUF = ms.Ref("2", "LoaderGGUF", "gguf_name", "A-Q8.gguf")
R_VAE = ms.Ref("3", "VAELoader", "vae_name", "v.safetensors")
R_LORA = ms.Ref("4", "LoraLoaderModelOnly", "lora_name", "L.safetensors")
R_HUB = ms.Ref("8", "Trellis2LoadModel", "modelname", "microsoft/TRELLIS.2-4B")
R_NAME = ms.Ref("5", "Trellis2LoadModel_GGUF", "modelname", "Pixal3D-GGUF")


def need(alias, refs, catalog=(), explicit=False, covered=frozenset()):
    return ms.AliasNeed(alias, list(refs), list(catalog), explicit, frozenset(covered))


def mk(needs, src=None, dest=None, manifest=None, urls=None):
    return ms.plan(needs, SRC if src is None else src, dest or {}, manifest or {}, urls or {})


def fetched(p):
    return [f["path"] for f in p["fetch"]]


class Plan(unittest.TestCase):
    def test_present_requires_same_size(self):
        p = mk([need("x", [R_VAE])], dest={"models/vae/v.safetensors": 99})
        a = p["per_alias"]["x"]
        self.assertEqual(a["missing"], ["models/vae/v.safetensors"])
        self.assertEqual(fetched(p), ["models/vae/v.safetensors"])
        self.assertFalse(ms.ready(p, "x"))
        p = mk([need("x", [R_VAE])], dest={"models/vae/v.safetensors": 100})
        self.assertEqual(p["per_alias"]["x"]["missing"], [])
        self.assertEqual(p["fetch"], [])
        self.assertTrue(ms.ready(p, "x"))
        self.assertEqual((p["need_total"], p["have_total"]), (100, 100))

    def test_part_file_not_present(self):
        dest = {"models/vae/v.safetensors.part": 100, "models/vae/v.safetensors.part.lock": 5,
                "models/vae/v.safetensors.part.log": 7}
        p = mk([need("x", [R_VAE])], dest=dest)
        self.assertEqual(p["per_alias"]["x"]["missing"], ["models/vae/v.safetensors"])
        self.assertEqual(p["unknown"], [])                 # transfer artefacts are no strangers
        self.assertEqual(p["prune"], [])

    def test_prune_only_manifest_files(self):
        dest = {"models/vae/v.safetensors": 100, "models/vae/old.safetensors": 7,
                "models/x/stranger.bin": 5}
        manifest = {"models/vae/v.safetensors": {"size": 100}, "models/vae/old.safetensors": {"size": 7},
                    "models/vae/gone.safetensors": {"size": 3}}
        p = mk([need("x", [R_VAE])], dest=dest, manifest=manifest)
        self.assertEqual(p["prune"], ["models/vae/gone.safetensors", "models/vae/old.safetensors"])

    def test_comfyui_stock_files_are_no_unknowns(self):
        """Addendum 2 (thunder-1 on the `comfy-ui` template, 2026-10-01): 36 of 37
        "unknown files" were ComfyUI's own zero-byte `put_..._here` placeholders and its
        stock `models/configs/*.yaml` — the one real 2.1 GB model drowned in them. They
        cost nothing and are ComfyUI's: not listed, so never offered for deletion
        either (`delete_unknown` judges against this list) and never pruned."""
        dest = {"models/checkpoints/put_checkpoints_here": 0,
                "models/vae/put_vae_here": 0,
                "models/custom/deep/put_anything_here": 0,          # any dir
                "hf-cache/empty.lock": 0,                           # any 0-byte file
                "models/configs/v1-inference.yaml": 1_947,
                "models/configs/anything_v3.yaml": 1_024 * 1024 - 1,
                "models/configs/huge.yaml": 1_024 * 1024,           # not a stock config
                "models/other/put_me_here_too.yaml": 5,             # not `put_*_here`
                "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors": 2_132_696_762}
        p = mk([], dest=dest)
        self.assertEqual(p["unknown"], [
            ["models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors", 2_132_696_762],
            ["models/configs/huge.yaml", 1_024 * 1024],
            ["models/other/put_me_here_too.yaml", 5]])
        self.assertEqual(p["prune"], [])
        # the name alone decides for a placeholder (ComfyUI ships them; whatever size)
        p = mk([], dest={"models/vae/put_vae_here": 12})
        self.assertEqual(p["unknown"], [])
        # a 0-byte file a manifest names is still ours to prune (only `unknown` filters)
        p = mk([], dest={"models/x.bin": 0}, manifest={"models/x.bin": {"size": 0}})
        self.assertEqual(p["prune"], ["models/x.bin"])

    def test_unknown_never_in_prune(self):
        dest = {"models/vae/v.safetensors": 100, "models/vae/old.safetensors": 7,
                "models/x/stranger.bin": 5, "hf-cache/hub/models--o--r/blobs/abc": 9}
        manifest = {"models/vae/old.safetensors": {"size": 7}}
        p = mk([need("x", [R_VAE])], dest=dest, manifest=manifest)
        self.assertEqual(p["unknown"], [["hf-cache/hub/models--o--r/blobs/abc", 9],
                                        ["models/x/stranger.bin", 5]])
        self.assertFalse({u for u, _ in p["unknown"]} & set(p["prune"]))
        self.assertNotIn("models/vae/v.safetensors", {u for u, _ in p["unknown"]})   # needed

    def test_empty_refs_blocked_unless_explicit(self):
        p = mk([need("mesh-mia", []), need("mesh-shrink", [], explicit=True)])
        self.assertEqual(p["per_alias"]["mesh-mia"]["blocked"], ["no model references known"])
        self.assertFalse(ms.ready(p, "mesh-mia"))
        self.assertEqual(p["per_alias"]["mesh-shrink"]["blocked"], [])
        self.assertTrue(ms.ready(p, "mesh-shrink"))

    def test_ambiguous_blocks_alias(self):
        r = ms.Ref("9", "SomeCustomLoader", "file", "dup.safetensors")
        p = mk([need("x", [r, R_VAE])])
        a = p["per_alias"]["x"]
        self.assertEqual(a["blocked"], ["ambiguous dup.safetensors: models/checkpoints/dup.safetensors, "
                                        "models/vae/dup.safetensors"])
        self.assertFalse(ms.ready(p, "x"))
        self.assertNotIn("models/vae/dup.safetensors", fetched(p))       # never guessed

    def test_url_source_preferred_over_lan(self):
        urls = {"models/vae/v.safetensors": {"url": "https://e/v", "sha256": "a" * 64},
                "models/upscale_models/u.pth": {"url": "https://e/u"}}
        ru = ms.Ref("6", "UpscaleModelLoader", "model_name", "u.pth")
        p = mk([need("x", [R_VAE, ru, R_LORA])], urls=urls)
        by = {f["path"]: f for f in p["fetch"]}
        self.assertEqual(by["models/vae/v.safetensors"],
                         {"path": "models/vae/v.safetensors", "size": 100, "source": "url",
                          "origin": "catalog", "url": "https://e/v", "sha256": "a" * 64,
                          "aliases": ["x"]})
        # a URL-only file resolves too; its size is unknown until it was downloaded once
        self.assertEqual(by["models/upscale_models/u.pth"],
                         {"path": "models/upscale_models/u.pth", "size": None, "source": "url",
                          "origin": "catalog", "url": "https://e/u", "aliases": ["x"]})
        self.assertEqual(by["models/loras/L.safetensors"]["source"], "lan")
        self.assertNotIn("url", by["models/loras/L.safetensors"])
        # once downloaded, the manifest's size is what "present" is measured against
        p = mk([need("x", [ru])], urls=urls, dest={"models/upscale_models/u.pth": 64},
               manifest={"models/upscale_models/u.pth": {"size": 64, "source": "url"}})
        self.assertTrue(ms.ready(p, "x"))
        p = mk([need("x", [ru])], urls=urls, dest={"models/upscale_models/u.pth": 63},
               manifest={"models/upscale_models/u.pth": {"size": 64, "source": "url"}})
        self.assertFalse(ms.ready(p, "x"))

    def test_fetch_order_smallest_alias_first_and_dedup(self):
        big = need("big", [R_UNET, R_VAE])           # 1100 missing
        small = need("small", [R_VAE, R_LORA])        # 150 missing
        p = mk([big, small])
        self.assertEqual(fetched(p), ["models/vae/v.safetensors", "models/loras/L.safetensors",
                                      "models/diffusion_models/A.safetensors"])
        by = {f["path"]: f for f in p["fetch"]}
        self.assertEqual(by["models/vae/v.safetensors"]["aliases"], ["small", "big"])
        self.assertEqual(by["models/diffusion_models/A.safetensors"]["aliases"], ["big"])
        self.assertEqual((p["need_total"], p["have_total"]), (1150, 0))   # v counted once

    def test_ready_and_status_text(self):
        src = {"models/diffusion_models/A.safetensors": int(18.7 * GB), "models/vae/v.safetensors": int(12.3 * GB)}
        p = mk([need("img", [R_UNET, R_VAE]), need("rig", [])], src=src,
               dest={"models/vae/v.safetensors": int(12.3 * GB)})
        self.assertFalse(ms.ready(p, "img"))
        self.assertEqual(ms.status_text(p, "img", "thunder-a6000"),
                         "models for img are syncing on thunder-a6000 (12.3 of 31.0 GB)")
        self.assertEqual(ms.status_text(p, "rig", "thunder-a6000"),
                         "models for rig are blocked on thunder-a6000: no model references known")
        p = mk([need("img", [R_UNET, R_VAE])], src=src,
               dest={"models/vae/v.safetensors": int(12.3 * GB),
                     "models/diffusion_models/A.safetensors": int(18.7 * GB)})
        self.assertTrue(ms.ready(p, "img"))
        self.assertEqual(ms.status_text(p, "img", "b"), "models for img are ready on b")
        self.assertFalse(ms.ready(p, "nope"))
        self.assertFalse(ms.ready(None, "img"))
        self.assertIn("not planned", ms.status_text(p, "nope", "b"))
        self.assertIn("not planned", ms.status_text(None, "img", "b"))

    def test_dual_loader_both_needed_with_node_info(self):
        wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "A.safetensors", "weight_dtype": "default"}},
              "2": {"class_type": "LoaderGGUF", "inputs": {"gguf_name": "A-Q8.gguf"}},
              "3": {"class_type": "SwitchAny", "inputs": {"on_true": ["1", 0], "on_false": ["2", 0], "boolean": True}}}
        p = mk([ms.alias_need("x", ms.refs_for({}, wf, set()), [])])
        self.assertEqual(p["per_alias"]["x"]["files"], [
            {"path": "models/diffusion_models/A-Q8.gguf", "size": 800, "node": "2", "cls": "LoaderGGUF", "present": False},
            {"path": "models/diffusion_models/A.safetensors", "size": 1000, "node": "1", "cls": "UNETLoader", "present": False}])
        # the bypass on this candidate is what drops the unused branch
        p = mk([ms.alias_need("x", ms.refs_for({"bypass": ["2"]}, wf, set()), [])])
        self.assertEqual([f["path"] for f in p["per_alias"]["x"]["files"]], ["models/diffusion_models/A.safetensors"])

    # -- beyond the brief ------------------------------------------------------------

    def test_missing_file_ref_blocks(self):
        r = ms.Ref("3", "VAELoader", "vae_name", "nope.safetensors")
        p = mk([need("x", [r, R_VAE])])
        self.assertEqual(p["per_alias"]["x"]["blocked"], ["not in source: nope.safetensors"])
        self.assertEqual(fetched(p), [])                 # Ruling 16: a blocked alias fetches nothing
        self.assertEqual(p["per_alias"]["x"]["missing"], ["models/vae/v.safetensors"])

    def test_hub_ref_needs_a_catalog_entry(self):
        """Ruling 2: an unmatched hub id blocks, a matched one syncs the entry's paths."""
        p = mk([need("x", [R_HUB])])
        self.assertEqual(p["per_alias"]["x"]["blocked"],
                         ["unknown hub model microsoft/TRELLIS.2-4B — add a catalog entry"])
        cat = [{"match": {"class": "Trellis2LoadModel", "value": "microsoft/TRELLIS.2-4B"},
                "paths": ["models/microsoft/TRELLIS.2-4B/"]}]
        n = ms.alias_need("x", [R_HUB], cat)
        self.assertEqual(n.covered, frozenset({("8", "modelname")}))
        p = mk([n])
        a = p["per_alias"]["x"]
        self.assertEqual(a["blocked"], [])
        self.assertEqual([(f["path"], f["node"], f["cls"]) for f in a["files"]],
                         [("models/microsoft/TRELLIS.2-4B/a.bin", None, None),
                          ("models/microsoft/TRELLIS.2-4B/sub/b.bin", None, None)])
        self.assertEqual(a["need_bytes"], 30)
        # Ruling 15: an ALIAS entry never covers a hub id — only class+value does
        for paths in ([], ["models/microsoft/TRELLIS.2-4B/"]):
            n = ms.alias_need("x", [R_HUB], [{"match": {"alias": "x"}, "paths": paths}])
            self.assertEqual(n.covered, frozenset())
            p = mk([n])
            self.assertEqual(p["per_alias"]["x"]["blocked"],
                             ["unknown hub model microsoft/TRELLIS.2-4B — add a catalog entry"], paths)
            self.assertFalse(ms.ready(p, "x"))

    def test_name_ref_alone_is_nothing(self):
        """Ruling 2: an unmatched bare name syncs nothing — an alias with nothing else is
        blocked (never ready by accident), and the text names the loader values a
        class+value entry would need; beside real files it is a hint, not a block."""
        r2 = ms.Ref("7", "DownloadAndLoadStableXModel", "model", "yoso-normal-v1-8-1")
        p = mk([need("x", [R_NAME, r2, R_NAME]), need("y", [R_NAME, R_VAE])])
        self.assertEqual(p["per_alias"]["x"]["blocked"],
                         ["no model files known (loader values: Pixal3D-GGUF, yoso-normal-v1-8-1"
                          " — add a catalog class+value entry)"])
        self.assertEqual(p["per_alias"]["x"]["hints"], [])
        self.assertEqual(p["per_alias"]["y"]["blocked"], [])
        self.assertTrue(ms.ready(p, "y") is False and p["per_alias"]["y"]["missing"])
        self.assertEqual(p["per_alias"]["y"]["hints"],
                         ["Trellis2LoadModel_GGUF=Pixal3D-GGUF is not synced — add a catalog entry "
                          "if it needs weights"])
        # a covered name ref is neither
        n = ms.alias_need("y", [R_NAME, R_VAE],
                          [{"match": {"class": "Trellis2LoadModel_GGUF", "value": "Pixal3D-GGUF"},
                            "paths": ["models/vae/v.safetensors"]}])
        self.assertEqual(mk([n])["per_alias"]["y"]["hints"], [])
        # no refs at all keeps the plain text; an asset value is no "loader value"
        self.assertEqual(mk([need("z", [])])["per_alias"]["z"]["blocked"], ["no model references known"])
        self.assertEqual(mk([need("z", [ms.Ref("1", "ModelLoaderX", "model_name", "a.glb")])])
                         ["per_alias"]["z"]["blocked"], ["no model references known"])

    def test_catalog_path_not_in_source_blocks(self):
        p = mk([need("x", [], catalog=["models/mia/"], explicit=True)])
        self.assertEqual(p["per_alias"]["x"]["blocked"], ["not in source: models/mia/"])
        self.assertFalse(ms.ready(p, "x"))

    def test_token_and_unsafe_paths_never_in_the_plan(self):
        dest = {"hf-cache/token": 1, "models/.cache/x": 2, "models/ok.bin": 3}
        manifest = {"hf-cache/token": {"size": 1}, "../etc/passwd": {"size": 1}, "models/-rf": {"size": 1}}
        p = mk([need("x", [ms.Ref("9", "X", "model", "token")], catalog=["hf-cache/", "hf-cache/token", "../x/"],
                     explicit=True)], dest=dest, manifest=manifest)
        flat = repr(p)
        self.assertNotIn("hf-cache/token", flat.replace("path 'hf-cache/token'", ""))
        self.assertEqual(p["prune"], ["models/-rf"])      # safe_rel guards only the first segment
        self.assertEqual(p["unknown"], [["models/ok.bin", 3]])
        self.assertEqual(len(p["per_alias"]["x"]["blocked"]), 3)       # each bad catalog path named
        self.assertEqual(p["fetch"], [])

    def test_manifest_verifies_a_file_the_source_no_longer_lists(self):
        """Spec: present = same size as the source, 'or verified in the manifest'. A LAN
        box that is down (empty source index) must not make every synced file an
        unneeded one — the stop's prune would wipe the snapshot."""
        man = {"models/vae/v.safetensors": {"size": 100, "source": "lan"},
               "models/microsoft/TRELLIS.2-4B/a.bin": {"size": 10, "source": "lan"}}
        dest = {"models/vae/v.safetensors": 100, "models/microsoft/TRELLIS.2-4B/a.bin": 10}
        cat = [{"match": {"alias": "x"}, "paths": ["models/microsoft/TRELLIS.2-4B/"]}]
        p = mk([ms.alias_need("x", [R_VAE], cat)], src={}, dest=dest, manifest=man)
        self.assertTrue(ms.ready(p, "x"))
        self.assertEqual(p["prune"], [])
        # gone from the VM and from the source: nothing can fetch it
        p = mk([need("x", [R_VAE])], src={}, dest={}, manifest=man)
        a = p["per_alias"]["x"]
        self.assertEqual(a["missing"], ["models/vae/v.safetensors"])
        self.assertEqual(a["blocked"], ["not in source: models/vae/v.safetensors"])
        self.assertEqual(p["fetch"], [])

    def test_source_size_beats_manifest_size(self):
        p = mk([need("x", [R_VAE])], dest={"models/vae/v.safetensors": 90},
               manifest={"models/vae/v.safetensors": {"size": 90}})
        self.assertFalse(ms.ready(p, "x"))                  # the source file changed
        self.assertEqual(fetched(p), ["models/vae/v.safetensors"])

    def test_blocked_alias_fetches_nothing(self):
        """Ruling 16: a blocked alias cannot become ready, so its files are not pulled
        onto the paid disk — a file another (unblocked) alias needs still is."""
        amb = ms.Ref("9", "SomeCustomLoader", "file", "dup.safetensors")
        p = mk([need("blocked", [amb, R_LORA, R_VAE]), need("fine", [R_UNET, R_VAE])])
        self.assertEqual(fetched(p), ["models/diffusion_models/A.safetensors", "models/vae/v.safetensors"])
        self.assertEqual({f["path"]: f["aliases"] for f in p["fetch"]}["models/vae/v.safetensors"], ["fine"])
        self.assertEqual(p["per_alias"]["blocked"]["missing"],
                         ["models/loras/L.safetensors", "models/vae/v.safetensors"])

    def test_blocked_alias_holds_its_manifest_files(self):
        """Ruling 16: a block (a new same-basename copy in the source, a removed catalog
        entry, an emptied workflow) must not make the stop prune what the alias synced."""
        man = {"models/vae/v.safetensors": {"size": 100, "aliases": ["A"]},
               "models/loras/L.safetensors": {"size": 50, "aliases": ["A", "B"]},
               "models/old.bin": {"size": 9, "aliases": ["gone"]}}
        dest = {"models/vae/v.safetensors": 100, "models/loras/L.safetensors": 50, "models/old.bin": 9}
        src = dict(SRC, **{"models/other/v.safetensors": 100})           # v is ambiguous now
        amb = ms.Ref("9", "SomeCustomLoader", "file", "v.safetensors")
        cases = {
            "ambiguous": ([need("A", [amb])], src),
            "catalog entry removed": ([need("A", [R_HUB])], SRC),
            "empty ref set": ([need("A", [])], SRC),
        }
        for why, (needs, s) in cases.items():
            p = mk(needs, src=s, dest=dest, manifest=man)
            self.assertTrue(p["per_alias"]["A"]["blocked"], why)
            self.assertEqual(p["prune"], ["models/old.bin"], why)
            self.assertEqual(p["held"], [["models/loras/L.safetensors", 50, "A"],
                                         ["models/vae/v.safetensors", 100, "A"]], why)
            self.assertEqual(p["unknown"], [], why)
        # once the alias is fine again nothing is held; an unselected alias holds nothing
        p = mk([need("A", [R_VAE])], dest=dest, manifest=man)
        self.assertEqual(p["held"], [])
        self.assertEqual(p["prune"], ["models/loras/L.safetensors", "models/old.bin"])

    def test_duplicate_alias_needs_merge(self):
        cat_hub = [{"match": {"class": "Trellis2LoadModel", "value": "microsoft/TRELLIS.2-4B"},
                    "paths": ["models/microsoft/TRELLIS.2-4B/"]}]
        a1 = need("x", [R_VAE])
        a2 = ms.alias_need("x", [R_HUB], cat_hub)
        p = mk([a1, a2, need("x", [], explicit=True)])
        self.assertEqual(list(p["per_alias"]), ["x"])
        a = p["per_alias"]["x"]
        self.assertEqual(a["blocked"], [])
        self.assertEqual(a["need_bytes"], 130)
        self.assertEqual(p["need_total"], a["need_bytes"])
        self.assertEqual(len(a["files"]), 3)

    def test_selectable_and_have_bytes(self):
        r = ms.Ref("3", "VAELoader", "vae_name", "v.safetensors", selectable=True)
        p = mk([need("x", [r, R_LORA])], dest={"models/vae/v.safetensors": 100})
        a = p["per_alias"]["x"]
        self.assertEqual(a["selectable"], ["3.vae_name"])
        self.assertEqual((a["need_bytes"], a["have_bytes"]), (150, 100))
        self.assertEqual(a["missing"], ["models/loras/L.safetensors"])

    def test_unknown_size_in_status_text(self):
        urls = {"models/upscale_models/u.pth": {"url": "https://e/u"}}
        ru = ms.Ref("6", "UpscaleModelLoader", "model_name", "u.pth")
        p = mk([need("x", [ru, R_VAE])], urls=urls)
        self.assertEqual(ms.status_text(p, "x", "b"),
                         "models for x are syncing on b (0.0 of 0.0 GB + 1 file of unknown size)")

    def test_deterministic_and_inputs_untouched(self):
        needs = [need("b", [R_UNET, R_VAE]), need("a", [R_VAE, R_LORA])]
        dest = {"models/x/s.bin": 1}
        man = {"models/vae/old.safetensors": {"size": 1}}
        before = (repr(needs), repr(dest), repr(man))
        self.assertEqual(mk(needs, dest=dest, manifest=man), mk(needs, dest=dest, manifest=man))
        self.assertEqual(before, (repr(needs), repr(dest), repr(man)))
        self.assertEqual(list(mk(needs)["per_alias"]), ["a", "b"])

    def test_url_catalog_helper(self):
        cat = [{"file": "models/vae/v.safetensors", "url": "https://e/v", "sha256": "b" * 64},
               {"file": "models/u.pth", "url": "https://e/u"},
               {"file": "hf-cache/token", "url": "https://e/t"},              # refused
               {"file": "models/w", "url": "http://e/w"},                    # refused
               {"match": {"alias": "a"}, "paths": []}, "junk"]
        self.assertEqual(ms.url_catalog(cat), {
            "models/vae/v.safetensors": {"url": "https://e/v", "sha256": "b" * 64},
            "models/u.pth": {"url": "https://e/u"}})
        self.assertEqual(ms.url_catalog(None), {})


class Hardening(unittest.TestCase):
    """Ruling 14 — per-job assets never block, catalogs never expand a whole tree, and
    the pin coercion cannot drift from the adapter's."""

    def test_asset_values_are_no_model_files(self):
        for v in ("x.glb", "out/3D/x.glb", "a.gltf", "a.obj", "a.fbx", "a.ply", "a.stl", "a.png",
                  "a.JPG", "a.jpeg", "a.webp", "a.gif", "a.mp4", "a.webm", "a.wav", "a.mp3", "a.flac"):
            self.assertEqual(ms.ref_kind(v), "name", v)

    def test_hub_id_with_dot_stays_hub(self):
        self.assertEqual(ms.ref_kind("org/repo.v2"), "hub")
        self.assertEqual(ms.ref_kind("TencentARC/Pixal3D-T.v1"), "hub")
        self.assertEqual(ms.ref_kind("org/repo/model.safetensors"), "file")
        self.assertEqual(ms.ref_kind("org/repo/w.gguf"), "file")
        self.assertEqual(ms.ref_kind("v1-inference.yaml"), "file")        # no slash: still a file

    def test_load3d_is_an_input_loader(self):
        wf = {"1": {"class_type": "Load3D", "inputs": {"model_file": "x.glb", "image": "i.png"}},
              "2": {"class_type": "Load3DAnimation", "inputs": {"model_file": "org/y"}},
              # a model loader whose name merely CONTAINS "3D" keeps working
              "3": {"class_type": "Hy3D21VAELoader", "inputs": {"model_name": "hunyuan3d-vae-v2-1"}}}
        refs = ms.refs_for({}, wf, set())
        self.assertEqual([(r.node, r.value, r.kind) for r in refs], [("3", "hunyuan3d-vae-v2-1", "name")])
        # and even on a node the loader rule catches, an asset value can never block
        wf = {"1": {"class_type": "ModelLoaderX", "inputs": {"model_name": "out/3D/x.glb"}}}
        p = ms.plan([ms.alias_need("x", ms.refs_for({}, wf, set()), [])], {}, {}, {}, {})
        self.assertEqual(p["per_alias"]["x"]["blocked"], ["no model references known"])

    def test_catalog_paths_never_expand_a_whole_tree(self):
        refs = [R_HUB]
        cat = [{"match": {"alias": "x"}, "paths": ["models/"]},
               {"match": {"alias": "x"}, "paths": ["hf-cache/"]},
               {"match": {"alias": "x"}, "paths": ["hf-cache/hub/"]},
               {"match": {"alias": "x"}, "paths": ["models/ok/", "../evil/"]},     # one bad path spoils it
               {"match": {"alias": "x", "value": 3}, "paths": ["models/v/"]},      # refused match value
               {"match": {"alias": "x"}, "paths": ["models/good/"], "pathz": []},  # typo'd key
               {"match": {"value": "microsoft/TRELLIS.2-4B"}, "paths": ["hf-cache/hub/models--m--t/"]}]
        self.assertEqual(ms.catalog_paths(refs, "x", cat), (["hf-cache/hub/models--m--t/"], False))
        self.assertTrue(ms.validate_catalog([{"match": {"alias": "x"}, "paths": ["hf-cache/hub/"]}]))

    def test_coerce_matches_the_adapter(self):
        import adapters
        cases = [("true", False), ("0", True), ("on", False), (" Yes ", True), (1, False),
                 ("7", 1), ("x", 1), (None, 1), ("2.5", 1), (3.0, 1),
                 ("2.5", 1.0), ("x", 1.0), (None, 1.0), ("7", 0.0),
                 ("abc", "s"), (5, "s"), ("v", None), ([1], None), ("1", True), ("1", 0)]
        for value, current in cases:
            a, m = adapters._coerce(value, current), ms._coerce(value, current)
            self.assertEqual((type(a), a), (type(m), m), (value, current))



class Ruling17(unittest.TestCase):
    """Task 13 hardening: a hub id is covered only by an entry naming BOTH its class and
    its value (a class-only entry would cover every hub id the loader might ever be
    set to — including one whose weights the entry does not sync), and a manifest whose
    `aliases` is not a list never iterates a string character by character."""

    def test_class_only_or_value_only_entry_never_covers_a_hub_ref(self):
        for m in ({"class": "Trellis2LoadModel"}, {"value": "microsoft/TRELLIS.2-4B"},
                  {"class": "Trellis2LoadModel", "alias": "x"}):
            cat = [{"match": m, "paths": ["models/microsoft/TRELLIS.2-4B/"]}]
            n = ms.alias_need("x", [R_HUB], cat)
            self.assertEqual(n.covered, frozenset(), m)
            p = mk([n])
            self.assertIn("unknown hub model microsoft/TRELLIS.2-4B — add a catalog entry",
                          p["per_alias"]["x"]["blocked"], m)
            self.assertFalse(ms.ready(p, "x"), m)
        cat = [{"match": {"class": "Trellis2LoadModel", "value": "microsoft/TRELLIS.2-4B"},
                "paths": ["models/microsoft/TRELLIS.2-4B/"]}]
        self.assertEqual(ms.alias_need("x", [R_HUB], cat).covered, frozenset({("8", "modelname")}))

    def test_class_only_entry_still_covers_a_name_ref(self):
        # a bare name is catalog material by nature (Ruling 2) — only hub ids got stricter
        cat = [{"match": {"class": "Trellis2LoadModel_GGUF"}, "paths": ["models/microsoft/TRELLIS.2-4B/"]}]
        self.assertEqual(ms.alias_need("x", [R_NAME], cat).covered, frozenset({("5", "modelname")}))

    def test_manifest_aliases_not_a_list_is_empty(self):
        dest = {"models/vae/v.safetensors": 100, "models/old.bin": 9}
        for rec in ("A", "AB", 5, {"A": 1}, None):
            man = {"models/vae/v.safetensors": {"size": 100, "aliases": rec},
                   "models/old.bin": {"size": 9, "aliases": rec}}
            # alias "A" is blocked (empty ref set): a string "A" must not make it an owner
            p = mk([need("A", [])], dest=dest, manifest=man)
            self.assertEqual(p["held"], [], rec)
            self.assertEqual(p["prune"], ["models/old.bin", "models/vae/v.safetensors"], rec)
        man = {"models/old.bin": {"size": 9, "aliases": ("A",)}}
        p = mk([need("A", [])], dest=dest, manifest=man)
        self.assertEqual(p["held"], [["models/old.bin", 9, "A"]])


class DefaultCatalog(unittest.TestCase):
    """The seed a fresh install starts with (Task 13): public hub/base models only —
    the repo is public, so a private model or LoRA name must never land in it."""

    def test_default_catalog_validates(self):
        self.assertEqual(ms.validate_catalog(ms.DEFAULT_CATALOG), [])
        self.assertTrue(ms.DEFAULT_CATALOG)

    def test_default_catalog_has_no_lora_or_private_names(self):
        import json
        blob = json.dumps(ms.DEFAULT_CATALOG).lower()
        self.assertNotIn("lora", blob)
        for e in ms.DEFAULT_CATALOG:
            for p in list(e.get("paths") or ()) + [e.get("file") or "", e["match"].get("value", "")]:
                self.assertFalse(p.lower().endswith(".safetensors"), p)
            self.assertEqual(set(e["match"]) - {"class", "value"}, set(), e)   # no alias names

    def test_default_catalog_covers_the_k12_loaders(self):
        want = {("Trellis2LoadModel", "microsoft/TRELLIS.2-4B"),
                ("Trellis2LoadModel", "TencentARC/Pixal3D-T"),
                ("Trellis2LoadModel_GGUF", "Pixal3D-GGUF"),
                ("DownloadAndLoadStableXModel", "yoso-normal-v1-8-1")}
        have = {(e["match"].get("class"), e["match"].get("value")) for e in ms.DEFAULT_CATALOG}
        self.assertLessEqual(want, have)
        # every hub id of the seed is class+value (Ruling 17) and so actually covers
        for e in ms.DEFAULT_CATALOG:
            self.assertIn("class", e["match"])
            self.assertIn("value", e["match"])

    def test_trellis_alias_ready_from_the_seed(self):
        refs = ms.refs_for({}, {"8": WF["8"]}, set())
        n = ms.alias_need("t", refs, ms.DEFAULT_CATALOG)
        self.assertEqual(n.covered, frozenset({("8", "modelname")}))
        src = {p.rstrip("/") + "/f.bin" if p.endswith("/") else p: 5 for p in n.catalog}
        p = ms.plan([n], src, dict(src), {}, {})
        self.assertTrue(ms.ready(p, "t"), p["per_alias"]["t"])



# --- Hugging Face cache symlinks (Task 15) -------------------------------------------------
_HF = "hf-cache/hub/models--o--n/"
_BLOB = _HF + "blobs/abc123"
_SNAP = _HF + "snapshots/r1/model.safetensors"
_REF = _HF + "refs/main"


class Links(unittest.TestCase):
    """An HF cache repo is blobs/<sha> FILES + snapshots/<rev>/<name> SYMLINKS + refs/.
    Syncing only the files leaves a cache the HF loader cannot read (it opens the
    snapshot path) — it re-downloads or fails, while the plan said "ready"."""

    def hf(self, extra_src=None, dest=None, manifest=None):
        src = {_BLOB: 50, _REF: 1, _SNAP: {"link": "../../blobs/abc123"}}
        src.update(extra_src or {})
        n = need("hf", [], [_HF])
        return ms.plan([n], src, dest or {}, manifest or {}, {})

    def test_resolve_link_grammar(self):
        self.assertEqual(ms.resolve_link(_SNAP, "../../blobs/abc123"), _BLOB)
        for bad in ("/abs/x", "../../.hidden/x", "../../blobs/../x", "", "a\\b",
                    "../../../../../../x", "x/", "a//b", "./x"):
            with self.subTest(bad=bad):
                self.assertIsNone(ms.resolve_link(_SNAP, bad))

    def test_catalog_dir_syncs_its_links(self):
        p = self.hf()
        row = p["per_alias"]["hf"]
        self.assertEqual(row["blocked"], [])
        links = [f for f in row["files"] if f.get("link")]
        self.assertEqual([(f["path"], f["link"], f["size"]) for f in links],
                         [(_SNAP, "../../blobs/abc123", 0)])
        e = next(e for e in p["fetch"] if e["path"] == _SNAP)
        self.assertEqual((e["source"], e["target"], e["size"]), ("link", "../../blobs/abc123", 0))
        self.assertIn(_SNAP, row["missing"])
        self.assertFalse(ms.ready(p, "hf"))

    def test_present_link_is_not_fetched_again_and_counts_ready(self):
        dest = {_BLOB: 50, _REF: 1, _SNAP: {"link": "../../blobs/abc123"}}
        p = self.hf(dest=dest)
        self.assertEqual(p["fetch"], [])
        self.assertTrue(ms.ready(p, "hf"))
        self.assertEqual(p["unknown"], [])
        # the same path with another target text is not the link the source has
        p = self.hf(dest=dict(dest, **{_SNAP: {"link": "../../blobs/zzz"}}))
        self.assertEqual(fetched(p), [_SNAP])

    def test_link_to_a_file_the_alias_does_not_sync_is_dropped(self):
        # the target is outside the catalog dir: syncing the link would dangle
        src = {_BLOB: 50, _REF: 1, "hf-cache/hub/models--x--y/snapshots/r/w.bin":
               {"link": "../../../models--o--n/blobs/abc123"}}
        n = need("hf", [], ["hf-cache/hub/models--x--y/", _HF])
        p = ms.plan([n], src, {}, {}, {})
        self.assertEqual([f["path"] for f in p["per_alias"]["hf"]["files"] if f.get("link")],
                         ["hf-cache/hub/models--x--y/snapshots/r/w.bin"])
        n = need("hf", [], ["hf-cache/hub/models--x--y/"])
        p = ms.plan([n], dict(src, **{"hf-cache/hub/models--x--y/f.bin": 1}), {}, {}, {})
        self.assertFalse([f for f in p["per_alias"]["hf"]["files"] if f.get("link")])

    def test_cross_root_link_is_dropped(self):
        # models/… and hf-cache/… do not sit side by side on the instance
        src = {"models/vae/a.safetensors": 5,
               "models/vae/l.safetensors": {"link": "../../hf-cache/hub/x"},
               "hf-cache/hub/x": 3}
        p = ms.plan([need("v", [], ["models/vae/"])], src, {}, {}, {})
        self.assertEqual([f["path"] for f in p["per_alias"]["v"]["files"]],
                         ["models/vae/a.safetensors"])

    def test_single_file_catalog_path_that_is_a_link_brings_its_target(self):
        p = ms.plan([need("s", [], [_SNAP])], {_BLOB: 50, _SNAP: {"link": "../../blobs/abc123"}},
                    {}, {}, {})
        self.assertEqual(sorted(f["path"] for f in p["per_alias"]["s"]["files"]), [_BLOB, _SNAP])
        self.assertEqual(p["per_alias"]["s"]["blocked"], [])

    def test_manifest_link_is_pruned_with_its_dir_and_verifies_while_source_down(self):
        man = {_BLOB: {"size": 50, "aliases": ["hf"]}, _REF: {"size": 1, "aliases": ["hf"]},
               _SNAP: {"size": 0, "source": "link", "target": "../../blobs/abc123",
                       "aliases": ["hf"]}}
        dest = {_BLOB: 50, _REF: 1, _SNAP: {"link": "../../blobs/abc123"}}
        # the LAN box is down (empty index): the manifest still verifies files AND link
        p = ms.plan([need("hf", [], [_HF])], {}, dest, man, {})
        self.assertTrue(ms.ready(p, "hf"), p["per_alias"]["hf"])
        self.assertEqual(p["prune"], [])
        # the alias is gone: the link is pruned with the files it belongs to
        p = ms.plan([], {}, dest, man, {})
        self.assertEqual(sorted(p["prune"]), sorted([_BLOB, _REF, _SNAP]))
        self.assertEqual(p["unknown"], [])

    def test_dest_links_nobody_synced_are_unknown(self):
        p = ms.plan([], {}, {"hf-cache/hub/m/snapshots/r/x": {"link": "../../blobs/b"}}, {}, {})
        self.assertEqual(p["unknown"], [["hf-cache/hub/m/snapshots/r/x", 0]])

    def test_a_file_in_the_source_beats_a_manifest_link(self):
        man = {_SNAP: {"size": 0, "source": "link", "target": "../../blobs/abc123",
                       "aliases": ["hf"]}}
        p = ms.plan([need("hf", [], [_HF])], {_BLOB: 50, _SNAP: 7}, {}, man, {})
        rows = {f["path"]: f for f in p["per_alias"]["hf"]["files"]}
        self.assertEqual(rows[_SNAP]["size"], 7)
        self.assertNotIn("link", rows[_SNAP])


# --- model sources: catalog shapes, "outdated", Stage 1 derivation ----------------------

REV = "0123456789abcdef0123456789abcdef01234567"
REV2 = "fedcba9876543210fedcba9876543210fedcba98"
SHA = "ab" * 32
SHA_B = "cd" * 32
SHA1 = "ef" * 20
HFR = "hf-cache/hub/models--org--repo/"
HF_URL = "https://huggingface.co/org/repo/resolve/" + REV + "/"


def snap(rel, rev=REV, repo=HFR):
    return f"{repo}snapshots/{rev}/{rel}"


class DerivedUrls(unittest.TestCase):
    """Stage 1: an HF-cache file on the share is a public download by its path alone.
    Every miss here is silent — a wrong URL or a sha that is no content hash makes the
    instance download, fail the check and fall back to the uplink; a derivation for a
    layout we do not understand downloads the wrong bytes."""

    def test_top_level_snapshot_link(self):
        src = {HFR + "blobs/" + SHA: 100, snap("model.safetensors"): {"link": "../../blobs/" + SHA}}
        self.assertEqual(ms.derived_urls(src), {
            HFR + "blobs/" + SHA: {"url": HF_URL + "model.safetensors", "sha256": SHA}})

    def test_nested_snapshot_link_goes_through_link_target(self):
        src = {HFR + "blobs/" + SHA: 100,
               snap("text_encoder/model.safetensors"): {"link": "../../../blobs/" + SHA}}
        self.assertEqual(ms.derived_urls(src), {
            HFR + "blobs/" + SHA: {"url": HF_URL + "text_encoder/model.safetensors",
                                   "sha256": SHA}})
        # the two-level target text of a top-level file, written under a nested path,
        # names snapshots/<rev>/blobs/… — no blob of this repo: nothing derived
        src = {HFR + "blobs/" + SHA: 100,
               snap("text_encoder/model.safetensors"): {"link": "../../blobs/" + SHA}}
        self.assertEqual(ms.derived_urls(src), {})

    def test_git_sha1_blob_is_size_only(self):
        src = {HFR + "blobs/" + SHA1: 3, snap("config.json"): {"link": "../../blobs/" + SHA1}}
        self.assertEqual(ms.derived_urls(src), {HFR + "blobs/" + SHA1: {"url": HF_URL + "config.json"}})

    def test_blob_name_neither_sha1_nor_sha256_is_no_layout_we_know(self):
        for oid in ("abc123", "AB" * 32, SHA + "0", "ab" * 20 + "x"):
            with self.subTest(oid=oid):
                src = {HFR + "blobs/" + oid: 3, snap("x.bin"): {"link": "../../blobs/" + oid}}
                self.assertEqual(ms.derived_urls(src), {})

    def test_revision_must_be_a_40_hex_commit(self):
        for rev in ("main", REV[:39], REV + "0", REV.upper(), "refs"):
            with self.subTest(rev=rev):
                src = {HFR + "blobs/" + SHA: 1, snap("x.bin", rev=rev): {"link": "../../blobs/" + SHA},
                       snap("y.bin", rev=rev): 5}
                self.assertEqual(ms.derived_urls(src), {})

    def test_datasets_and_spaces_are_skipped(self):
        for kind in ("datasets", "spaces"):
            with self.subTest(kind=kind):
                r = f"hf-cache/hub/{kind}--org--repo/"
                src = {r + "blobs/" + SHA: 1, snap("x.bin", repo=r): {"link": "../../blobs/" + SHA},
                       snap("y.bin", repo=r): 5}
                self.assertEqual(ms.derived_urls(src), {})

    def test_repo_dir_name_must_split_into_org_and_name(self):
        for d in ("models--repo", "models--a--b--c", "models--org--re..po", "models--org--",
                  "models----repo", "models--org--re po", "models--.org--repo", "models--org--repo-",
                  "model--org--repo"):
            with self.subTest(d=d):
                r = f"hf-cache/hub/{d}/"
                src = {r + "blobs/" + SHA: 1, snap("x.bin", repo=r): {"link": "../../blobs/" + SHA},
                       snap("y.bin", repo=r): 5}
                self.assertEqual(ms.derived_urls(src), {})
        # dots, dashes and underscores inside are fine
        r = "hf-cache/hub/models--Org_1--Re.po-2/"
        self.assertEqual(ms.derived_urls({snap("y.bin", repo=r): 5}), {
            snap("y.bin", repo=r): {"url": f"https://huggingface.co/Org_1/Re.po-2/resolve/{REV}/y.bin"}})

    def test_path_is_percent_encoded_and_passes_url_error(self):
        rel = "sub dir/my model (v2)#1.safetensors"
        src = {HFR + "blobs/" + SHA: 9, snap(rel): {"link": "../../../blobs/" + SHA}}
        got = ms.derived_urls(src)[HFR + "blobs/" + SHA]["url"]
        self.assertEqual(got, HF_URL + "sub%20dir/my%20model%20%28v2%29%231.safetensors")
        self.assertEqual(ms._url_error(got), "")

    def test_no_symlink_cache_layout(self):
        """HF_HUB_DISABLE_SYMLINKS: the snapshot entry is the file itself — URL from the
        path, no oid to take a sha from."""
        src = {snap("unet/w.safetensors"): 7, HFR + "refs/main": 40}
        self.assertEqual(ms.derived_urls(src), {
            snap("unet/w.safetensors"): {"url": HF_URL + "unet/w.safetensors"}})

    def test_a_link_leaving_its_repo_is_ignored(self):
        other = "hf-cache/hub/models--o--n/blobs/" + SHA
        src = {other: 1, snap("x.bin"): {"link": "../../../models--o--n/blobs/" + SHA},
               HFR + "refs/main": 40, snap("y.bin"): {"link": "../../refs/main"}}
        self.assertEqual(ms.derived_urls(src), {})

    def test_a_dangling_link_derives_nothing(self):
        self.assertEqual(ms.derived_urls({snap("x.bin"): {"link": "../../blobs/" + SHA}}), {})
        # the blob listed as a LINK is no file either
        self.assertEqual(ms.derived_urls({snap("x.bin"): {"link": "../../blobs/" + SHA},
                                          HFR + "blobs/" + SHA: {"link": "x"}}), {})

    def test_two_links_onto_one_blob_take_the_first_path(self):
        src = {HFR + "blobs/" + SHA: 1,
               snap("b.bin"): {"link": "../../blobs/" + SHA},
               snap("a.bin", rev=REV2): {"link": "../../blobs/" + SHA},
               snap("a.bin"): {"link": "../../blobs/" + SHA}}
        self.assertEqual(ms.derived_urls(src)[HFR + "blobs/" + SHA]["url"], HF_URL + "a.bin")

    def test_only_the_hub_cache_and_no_unusable_paths(self):
        src = {"models/models--org--repo/snapshots/" + REV + "/x.bin": 5,      # not the HF cache
               "hf-cache/models--org--repo/snapshots/" + REV + "/x.bin": 5,
               snap(".cache/x.bin"): 5, snap("x.bin.part"): 5,
               HFR + "snapshots/" + REV: 5}                                    # no file under rev
        self.assertEqual(ms.derived_urls(src), {})

    def test_junk_inputs(self):
        self.assertEqual(ms.derived_urls(None), {})
        self.assertEqual(ms.derived_urls({snap("x.bin"): "big", snap("y.bin"): True,
                                          snap("z.bin"): None}), {})


DIR = "models/org/thing/"


def dir_entry(files, rev=REV, d=DIR, repo="org/thing"):
    return {"dir": d, "repo": repo, "rev": rev, "files": files}


class UrlCatalogSources(unittest.TestCase):
    """`url_catalog(catalog, source_index)` is what the plan downloads from. An entry
    the share has outgrown that still counts makes the instance download the OLD file
    and fail the check; one dropped by mistake syncs gigabytes over the uplink."""

    F = "models/vae/v.safetensors"

    def test_outdated_by_size_is_dropped(self):
        cat = [{"file": self.F, "url": "https://e/v", "sha256": SHA, "size": 100}]
        self.assertEqual(ms.url_catalog(cat, {self.F: 100}),
                         {self.F: {"url": "https://e/v", "sha256": SHA}})
        self.assertEqual(ms.url_catalog(cat, {self.F: 90}), {})
        k = ms.source_kinds(cat, {self.F: 90})[self.F]
        self.assertEqual(k["kind"], "outdated")
        self.assertIn("size differs", k["reason"])

    def test_no_listing_size_keeps_the_entry(self):
        cat = [{"file": self.F, "url": "https://e/v", "size": 100}]
        for idx in ({}, None, {self.F: {"link": "x"}}):
            with self.subTest(idx=idx):
                self.assertEqual(ms.url_catalog(cat, idx), {self.F: {"url": "https://e/v"}})

    def test_old_entries_without_size_are_never_outdated(self):
        cat = [{"file": self.F, "url": "https://e/v", "sha256": SHA}]
        self.assertEqual(ms.url_catalog(cat, {self.F: 1}), {self.F: {"url": "https://e/v", "sha256": SHA}})
        self.assertEqual(ms.url_catalog(cat, {self.F: 1}, {self.F: [1, SHA_B]}),
                         {self.F: {"url": "https://e/v", "sha256": SHA}})
        self.assertEqual(ms.source_kinds(cat, {self.F: 1})[self.F]["kind"], "url")

    def test_one_arg_call_keeps_the_old_behaviour(self):
        cat = [{"file": self.F, "url": "https://e/v", "size": 5},
               dir_entry({"a.bin": [10, None, False]})]
        self.assertEqual(ms.url_catalog(cat), {
            self.F: {"url": "https://e/v"},
            DIR + "a.bin": {"url": f"https://huggingface.co/org/thing/resolve/{REV}/a.bin"}})

    def test_dir_entry_expands_per_file(self):
        cat = [dir_entry({"a.bin": [10, SHA, False], "sub/b c.json": [5, None, False],
                          "c.safetensors": [7, SHA_B, True]})]
        base = f"https://huggingface.co/org/thing/resolve/{REV}/"
        idx = {DIR + "a.bin": 10, DIR + "sub/b c.json": 5, DIR + "c.safetensors": 7}
        self.assertEqual(ms.url_catalog(cat, idx), {
            DIR + "a.bin": {"url": base + "a.bin", "sha256": SHA},
            DIR + "sub/b c.json": {"url": base + "sub/b%20c.json"},
            DIR + "c.safetensors": {"url": base + "c.safetensors", "sha256": SHA_B}})
        kinds = ms.source_kinds(cat, idx)
        self.assertEqual(kinds[DIR + "c.safetensors"]["provisional"], True)
        self.assertEqual(kinds[DIR + "a.bin"]["provisional"], False)
        self.assertEqual({k["origin"] for k in kinds.values()}, {"dir"})
        # one row outdated: only that file drops out
        idx[DIR + "a.bin"] = 11
        got = ms.url_catalog(cat, idx)
        self.assertNotIn(DIR + "a.bin", got)
        self.assertEqual(len(got), 2)
        # a share file under the dir that `files` does not name is no catalog file (lan)
        idx[DIR + "new.bin"] = 3
        self.assertNotIn(DIR + "new.bin", ms.source_kinds(cat, idx))

    def test_file_entry_beats_dir_entry_and_falls_back_to_it_when_outdated(self):
        p = DIR + "a.bin"
        cat = [{"file": p, "url": "https://mirror/a", "size": 10},
               dir_entry({"a.bin": [10, SHA, False]})]
        self.assertEqual(ms.url_catalog(cat, {p: 10}), {p: {"url": "https://mirror/a"}})
        self.assertEqual(ms.source_kinds(cat, {p: 10})[p]["origin"], "file")
        cat[0]["size"] = 9                                   # the mirror entry is outdated
        self.assertEqual(ms.url_catalog(cat, {p: 10})[p]["url"],
                         f"https://huggingface.co/org/thing/resolve/{REV}/a.bin")

    def test_later_entry_wins_within_a_shape(self):
        cat = [{"file": self.F, "url": "https://e/1"}, {"file": self.F, "url": "https://e/2"}]
        self.assertEqual(ms.url_catalog(cat, {})[self.F]["url"], "https://e/2")

    def test_derived_urls_merged_under_explicit_entries(self):
        blob = HFR + "blobs/" + SHA
        idx = {blob: 100, snap("m.safetensors"): {"link": "../../blobs/" + SHA},
               HFR + "blobs/" + SHA1: 3, snap("c.json"): {"link": "../../blobs/" + SHA1}}
        got = ms.url_catalog([], idx)
        self.assertEqual(got[blob], {"url": HF_URL + "m.safetensors", "sha256": SHA, "origin": "hf-auto"})
        self.assertEqual(ms.source_kinds([], idx)[blob]["kind"], "hf-auto")
        # an operator's mirror beats the derivation
        cat = [{"file": blob, "url": "https://mirror/m", "size": 100}]
        self.assertEqual(ms.url_catalog(cat, idx)[blob], {"url": "https://mirror/m"})
        self.assertEqual(ms.source_kinds(cat, idx)[blob]["kind"], "url")
        # ... unless it is outdated: then the derivation serves, and says what it replaced
        cat[0]["size"] = 99
        self.assertEqual(ms.url_catalog(cat, idx)[blob]["origin"], "hf-auto")
        k = ms.source_kinds(cat, idx)[blob]
        self.assertEqual(k["kind"], "hf-auto")
        self.assertIn("size differs", k["outdated_entry"])

    def test_share_sha_turns_a_differing_entry_outdated(self):
        p = DIR + "c.safetensors"
        cat = [dir_entry({"c.safetensors": [7, SHA_B, True]})]
        self.assertIn(p, ms.url_catalog(cat, {p: 7}, {p: [7, SHA_B]}))          # confirmed
        self.assertIn(p, ms.url_catalog(cat, {p: 7}, {p: [7, SHA_B.upper()]}))  # case-blind
        self.assertNotIn(p, ms.url_catalog(cat, {p: 7}, {p: [7, SHA]}))         # share differs
        k = ms.source_kinds(cat, {p: 7}, {p: [7, SHA]})[p]
        self.assertEqual(k["kind"], "outdated")
        self.assertIn("hash differs", k["reason"])
        # a cached hash of another size describes another file: ignored
        self.assertIn(p, ms.url_catalog(cat, {p: 7}, {p: [8, SHA]}))
        # a size-only row has nothing to compare; junk caches are ignored
        cat = [dir_entry({"c.safetensors": [7, None, False]})]
        self.assertIn(p, ms.url_catalog(cat, {p: 7}, {p: [7, SHA]}))
        for junk in (None, [], {p: "x"}, {p: [7]}, {p: [7, 3]}):
            self.assertIn(p, ms.url_catalog(cat, {p: 7}, junk))

    def test_invalid_entries_are_dropped(self):
        cat = [{"file": self.F, "url": "https://e/v", "size": 0},
               dir_entry({"../x": [1, None, False]}),
               dir_entry({"a.bin": [1, None, False]}, rev="main"), "junk", None]
        self.assertEqual(ms.url_catalog(cat, {}), {})
        self.assertEqual(ms.source_kinds(cat, {}), {})
        self.assertEqual(ms.url_catalog(None, None), {})

    def test_plan_labels_the_origin(self):
        blob = HFR + "blobs/" + SHA
        idx = {blob: 100, snap("m.safetensors"): {"link": "../../blobs/" + SHA},
               HFR + "refs/main": 40, "models/vae/v.safetensors": 5}
        cat = [{"file": "models/vae/v.safetensors", "url": "https://e/v", "size": 5}]
        urls = ms.url_catalog(cat, idx)
        n = ms.alias_need("x", [ms.Ref("3", "VAELoader", "vae_name", "v.safetensors")],
                          [{"match": {"alias": "x"}, "paths": [HFR]}])
        p = ms.plan([n], idx, {}, {}, urls)
        by = {e["path"]: e for e in p["fetch"]}
        self.assertEqual(by[blob], {"path": blob, "size": 100, "source": "url", "origin": "hf-auto",
                                    "url": HF_URL + "m.safetensors", "sha256": SHA, "aliases": ["x"]})
        self.assertEqual(by["models/vae/v.safetensors"]["origin"], "catalog")
        self.assertEqual(by[HFR + "refs/main"]["source"], "lan")
        self.assertNotIn("origin", by[HFR + "refs/main"])
        self.assertEqual(by[snap("m.safetensors")]["source"], "link")
        self.assertTrue(ms.ready(ms.plan([n], idx, {blob: 100, HFR + "refs/main": 40,
                                                    "models/vae/v.safetensors": 5,
                                                    snap("m.safetensors"): {"link": "../../blobs/" + SHA}},
                                         {}, urls), "x"))


class CatalogShapes(unittest.TestCase):
    """The validator is the gate for both writers of the catalog: a shape it lets through
    wrongly reaches curl and the plan; one it refuses wrongly blocks a Check & save."""

    def test_new_shapes_accepted(self):
        ok = [{"file": "models/vae/v.safetensors", "url": "https://e/v", "sha256": SHA, "size": 1},
              {"file": "models/vae/w.safetensors", "url": "https://e/w", "size": 10 ** 12},
              dir_entry({"a.bin": [1, None, False], "sub/b.json": [2, SHA, True],
                         "c d.bin": [3, SHA.upper(), False]}),
              dir_entry({"x": [1, None, False]}, d="models/a/b/c/", repo="O-1/n_2.v3")]
        self.assertEqual(ms.validate_catalog(ok), [])

    def test_bad_size(self):
        for size in (0, -1, "5", 1.0, True, None):
            with self.subTest(size=size):
                self.assertTrue(ms.validate_catalog(
                    [{"file": "models/x", "url": "https://e/x", "size": size}]))

    def test_bad_dir_entries(self):
        good = {"a.bin": [1, None, False]}
        bad = {
            "no files": {"dir": DIR, "repo": "org/thing", "rev": REV},
            "empty files": dir_entry({}),
            "files not a map": dir_entry([["a.bin", 1, None, False]]),
            "no repo": {"dir": DIR, "rev": REV, "files": good},
            "no rev": {"dir": DIR, "repo": "org/thing", "files": good},
            "no trailing slash": dir_entry(good, d="models/org/thing"),
            "hf-cache dir": dir_entry(good, d="hf-cache/hub/models--org--thing/"),
            "a whole root": dir_entry(good, d="models/"),
            "escaping dir": dir_entry(good, d="models/../x/"),
            "dot dir": dir_entry(good, d="models/.cache/"),
            "token": dir_entry(good, d="hf-cache/token/"),
            "rev 39 hex": dir_entry(good, rev=REV[:39]),
            "rev main": dir_entry(good, rev="main"),
            "rev upper": dir_entry(good, rev=REV.upper()),
            "unknown key": dict(dir_entry(good), size=5),
            "mixed with file": dict(dir_entry(good), file="models/x", url="https://e/x"),
            "mixed with match": dict(dir_entry(good), match={"alias": "a"}, paths=[]),
        }
        for repo in ("org", "org/na--me", "org/../x", "-org/x", "org/x.", "org/x/y", "org/x y",
                     "/x", "org/", 5, "org/x.git", "o" * 97 + "/x"):
            bad[f"repo {repo!r}"] = dir_entry(good, repo=repo)
        for rel in ("../x", "/x", "a/", ".hidden/x", "a/.cache/x", "a//b", "", "a\\b", "a\nb"):
            bad[f"relpath {rel!r}"] = dir_entry({rel: [1, None, False]})
        for row in ([1, None], [1, None, False, 0], [0, None, False], [1, "abc", False],
                    [1, None, "no"], [1, None, True], ["1", None, False], [True, None, False],
                    (1, None, False), {"size": 1}, None):
            bad[f"row {row!r}"] = dir_entry({"a.bin": row})
        for why, e in bad.items():
            with self.subTest(why=why):
                self.assertTrue(ms.validate_catalog([e]), why)

    def test_dir_entry_errors_name_the_problem(self):
        errs = ms.validate_catalog([dir_entry({"../x": [1, None, False]}, rev="main")])
        self.assertTrue(any("rev" in e for e in errs), errs)
        self.assertTrue(any("../x" in e for e in errs), errs)


if __name__ == "__main__":
    unittest.main()
