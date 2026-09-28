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
                          "url": "https://e/v", "sha256": "a" * 64, "aliases": ["x"]})
        # a URL-only file resolves too; its size is unknown until it was downloaded once
        self.assertEqual(by["models/upscale_models/u.pth"],
                         {"path": "models/upscale_models/u.pth", "size": None, "source": "url",
                          "url": "https://e/u", "aliases": ["x"]})
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


if __name__ == "__main__":
    unittest.main()
