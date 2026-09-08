"""The console is kind-neutral: every place that used to ask `== "meshy"` now asks the
adapters' kind helpers, so a second cloud backend (Tripo) needs no new branch. These
three fail SILENTLY in the browser — a job view that renders nothing for a Tripo run, an
editor that offers the wrong backends, a type select whose JS never reveals the cloud
option block — so they are pinned here.

Run: python -m unittest tests.test_cloud_editor -v
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
    import admin
    import adapters
    import meshy
    import tripo
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


class AdminKindNeutral(unittest.TestCase):
    def test_cloud_table_reads_both_id_keys(self):
        import admin
        html = admin._cloud_table("Cloud", {"meshy_task_id": "m1", "request": {"a": 1},
                                            "endpoint": "image-to-3d"})
        self.assertIn("m1", html); self.assertIn("Meshy", html)
        html = admin._cloud_table("Cloud", {"cloud": "tripo", "cloud_task_id": "t1",
                                            "request": {"input": "tok"}, "endpoint": "rig",
                                            "tasks": [{"role": "rig-check", "task_id": "t0",
                                                       "credits": 0}]})
        self.assertIn("t1", html); self.assertIn("Tripo", html)
        self.assertIn("rig-check", html); self.assertIn("t0", html)
        self.assertEqual(admin._cloud_table("x", {"request": {}}), "")

    def test_same_kind_matches_backend_type(self):
        import admin, tripo
        # restored: _gen_backends is a module global the whole console reads, and a test
        # that leaves a stub behind decides what LATER tests see (order-dependent failures).
        self.addCleanup(setattr, admin, "_gen_backends", admin._gen_backends)
        admin._gen_backends = lambda: [{"name": "x", "type": "comfyui"}, {"name": "x", "type": "tripo"},
                                       {"name": "m", "type": "meshy"}]
        self.assertTrue(admin._same_kind([tripo.default_candidate("x")], "x"))
        self.assertFalse(admin._same_kind([tripo.default_candidate("x")], "m"))
        self.assertTrue(admin._same_kind([{"workflow_json": {}}], "x"))
        self.assertFalse(admin._same_kind([meshy.default_candidate("m")], "x"))

    def test_type_select_knows_every_cloud_url(self):
        import admin, tripo
        html = admin._type_select("tripo")
        self.assertIn(tripo.URL, html); self.assertIn(meshy.URL, html)
        self.assertIn('value="tripo" selected', html)
        self.assertIn("cloudopts", html)

    def test_type_select_replaces_another_kinds_url(self):
        """Switching meshy → tripo must not leave api.meshy.ai in the url field: the
        backend would be saved pointing at the wrong vendor and only fail at discovery,
        with an auth error that names the wrong service. The handler therefore overwrites
        a url that is blank OR equals ANOTHER kind's fixed URL — never a typed one."""
        import admin
        js = admin._type_select("meshy")
        self.assertIn("for(var k in cloudUrls)", js)          # every other kind is compared
        self.assertIn("u.value===cloudUrls[k]", js)           # …against the url field
        self.assertIn("u.value=cloudUrls[t]", js)             # …and replaced by the new kind's
        for url in (meshy.URL, tripo.URL):
            self.assertIn(url, js)
        self.assertNotIn("if(u&&!u.value)", js)               # the old fill-only rule is gone


class CloudBackendUrl(unittest.TestCase):
    """`backend_save`'s url rule, as the pure helper it delegates to. It fails SILENTLY:
    a Tripo backend saved with api.meshy.ai only shows up as an auth error at discovery,
    naming the wrong vendor."""

    def test_cloud_url_for(self):
        import admin, meshy, tripo
        self.assertEqual(admin._cloud_url_for("tripo", meshy.URL), tripo.URL)   # kind switched
        self.assertEqual(admin._cloud_url_for("tripo", ""), tripo.URL)          # blank → filled
        self.assertEqual(admin._cloud_url_for("tripo", "https://my.proxy"),
                         "https://my.proxy")                                    # typed → kept
        self.assertEqual(admin._cloud_url_for("meshy", tripo.URL), meshy.URL)
        self.assertEqual(admin._cloud_url_for("comfyui", "http://gpu:8188"), "http://gpu:8188")
        self.assertEqual(admin._cloud_url_for("comfyui", ""), "")               # never touched
        self.assertEqual(admin._cloud_url_for("tripo", tripo.URL), tripo.URL)   # already right


class CloudEditor(unittest.TestCase):
    """The schema-driven cloud alias editor: ONE form rendered from `mod.OPTION_FIELDS`
    serves every cloud kind. A field the editor forgets fails SILENTLY — the option keeps
    whatever is stored and the admin never sees it — so every field of every kind and
    endpoint is pinned here."""

    def _render(self, mod, endpoint):
        import admin
        c = mod.default_candidate("b")
        c[mod.KIND]["endpoint"] = endpoint
        self.addCleanup(setattr, admin, "_gen_backends", admin._gen_backends)
        admin._gen_backends = lambda: [{"name": "b", "type": mod.KIND}]
        html = admin._cloud_editor(mod.KIND, "A-1", [c])
        for fld in mod.OPTION_FIELDS:
            self.assertIn(f'name="opt__{fld["key"]}"', html, fld["key"])
        for f in (mod.RIG_FORMATS if endpoint == mod.RIG_ENDPOINT else mod.FORMATS):
            self.assertIn(f'name="fmt__{f}"', html)
        self.assertIn('name="cloud_endpoint"', html)
        self.assertIn('name="cloud_model"', html)
        self.assertIn('action="/ui/mapping/cloud-update"', html)
        self.assertIn('name="chain_rig"', html)
        self.assertIn('value="tripo"', html)
        self.assertIn(mod.VENDOR, html)
        return html

    def test_renders_meshy_and_tripo_every_endpoint(self):
        import tripo
        for mod in (meshy, tripo):
            for ep in mod.ENDPOINTS:
                self._render(mod, ep)

    def test_bool_fields_with_blank_label_share_a_row(self):
        import tripo
        html = self._render(tripo, "image-to-model")
        # `pbr` (label "") rides in the `texture` row: exactly one field row carries both boxes
        row = html[html.index('name="opt__texture"'):html.index('name="opt__texture_quality"')]
        self.assertIn('name="opt__pbr"', row)

    def test_tripo_defaults_show_mixamo(self):
        import tripo
        html = self._render(tripo, "rig")
        self.assertIn('value="mixamo" selected', html)

    def test_meshy_form_names_are_unchanged(self):
        """Stored Meshy aliases keep working: the schema editor posts exactly the field
        names the hand-written one did (only endpoint/model were renamed to the kind-neutral
        pair) — a renamed `opt__` field would silently save the DEFAULT on the next save."""
        html = self._render(meshy, "image-to-3d")
        for n in ("opt__should_texture", "opt__enable_pbr", "opt__texture_resolution", "opt__topology",
                  "opt__should_remesh", "opt__target_polycount", "opt__pose_mode", "opt__ultra_mode",
                  "opt__image_enhancement", "opt__remove_lighting", "opt__moderation",
                  "opt__animations", "opt__thumbnail", "new_alias", "retries", "successor",
                  "chain_mesh_param", "chain_keep", "chain_rig", "task"):
            self.assertIn(f'name="{n}"', html)

    def test_cloud_update_apply_normalizes_and_copies(self):
        import admin, tripo
        c = tripo.default_candidate("b"); cands = [c, dict(c, backend="b2")]
        form = {"cloud_endpoint": "rig", "cloud_model": "v3.0-20250812", "opt__spec": "tripo",
                # v2.5 is the rig model that can do a non-biped; with the default (v1.0)
                # `avian` is normalized away, which the next test pins
                "opt__rig_model": "v2.5-20260210",
                "opt__rig_type": "avian", "opt__animations": "preset:walk preset:run", "opt__face_limit": "9",
                "fmt__fbx": "on", "task": "mesh2rig", "retries": "1",
                "successor": "", "chain_mesh_param": "", "chain_keep": "", "chain_rig": ""}
        admin._cloud_update_apply("tripo", cands, form)
        for x in cands:
            self.assertEqual(x["tripo"]["endpoint"], "rig")
            self.assertEqual(x["model"], "v3.0-20250812")
            self.assertEqual(x["tripo"]["options"]["spec"], "tripo")
            self.assertEqual(x["tripo"]["options"]["rig_type"], "avian")
            self.assertEqual(x["tripo"]["options"]["animations"], ["preset:walk", "preset:run"])
            self.assertIsNone(x["tripo"]["options"]["face_limit"])      # 9 < 100 → ignored by options_of
            self.assertEqual(x["tripo"]["options"]["target_formats"], ["fbx"])
            self.assertEqual(x["task"], "mesh2rig"); self.assertEqual(x["retries"], "1")
            self.assertNotIn("successor", x)
        self.assertIsNot(cands[0]["tripo"]["options"], cands[1]["tripo"]["options"])
        form.update({"successor": "mesh-mia", "chain_rig": "mixamo", "chain_keep": "preview.png"})
        admin._cloud_update_apply("tripo", cands, form)
        self.assertEqual(cands[0]["successor"], {"alias": "mesh-mia", "mesh_param": "input_mesh_path",
                                                 "keep_from_mesh": ["preview.png"], "rig": "mixamo"})
        self.assertIsNot(cands[0]["successor"], cands[1]["successor"])

    def test_cloud_update_apply_runs_the_modules_normalizer(self):
        """Save stores what the request builder would send: the editor offers all seven rig
        types (rendering one field cannot know another's value), and a combination the vendor
        refuses is normalized away at SAVE time, not discovered on a paid request."""
        import admin, tripo
        cands = [tripo.default_candidate("b")]
        admin._cloud_update_apply("tripo", cands, {"cloud_endpoint": "rig", "cloud_model": "v3.1-20260211",
                                                   "opt__rig_model": "v1.0-20240301", "opt__rig_type": "avian",
                                                   "opt__generate_parts": "1", "opt__texture": "1",
                                                   "fmt__glb": "on"})
        o = cands[0]["tripo"]["options"]
        self.assertEqual(o["rig_type"], "biped")        # rig model v1.0 rigs bipeds only
        self.assertFalse(o["texture"])                  # generate_parts forbids texture/pbr/quad

    def test_cloud_update_apply_keeps_meshy_semantics(self):
        import admin
        cands = [meshy.default_candidate("b")]
        admin._cloud_update_apply("meshy", cands, {"cloud_endpoint": "rigging", "cloud_model": "meshy-6",
                                                   "opt__should_remesh": "false", "opt__target_polycount": "42",
                                                   "opt__should_texture": "1", "fmt__glb": "on", "fmt__obj": "on"})
        o = cands[0]["meshy"]["options"]
        self.assertIs(o["should_remesh"], False)
        self.assertIsNone(o["target_polycount"])        # 42 < 100 → meshy.opt_polycount drops it
        self.assertEqual(o["target_formats"], ["glb"])  # rigging delivers glb/fbx only
        self.assertEqual(cands[0]["meshy"]["endpoint"], "rigging")
        self.assertEqual(cands[0]["model"], "meshy-6")


if __name__ == "__main__":
    unittest.main()
