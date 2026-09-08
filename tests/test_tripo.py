"""Unit tests for tripo.py — run: venv/bin/python -m unittest tests.test_tripo -v"""
import unittest

import cloudtask
import tripo

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
GLB = b"glTF" + b"\x00" * 60


def cand(endpoint="image-to-model", model="v3.1-20260211", **opts):
    c = tripo.default_candidate("tripo")
    c["model"] = model
    c["tripo"]["endpoint"] = endpoint
    c["tripo"]["options"].update(opts)
    return c


class Sniff(unittest.TestCase):
    def test_image_ext(self):
        self.assertEqual(tripo.image_ext(PNG), "png")
        self.assertEqual(tripo.image_ext(JPG), "jpg")
        with self.assertRaises(tripo.TripoInput):
            tripo.image_ext(b"RIFF....WEBPVP8 ")

    def test_mesh_ext(self):
        self.assertEqual(tripo.mesh_ext(GLB), "glb")
        with self.assertRaises(tripo.TripoInput):
            tripo.mesh_ext(b"# obj\nv 0 0 0\n")


class Options(unittest.TestCase):
    def test_defaults_are_deep_copied(self):
        a, b = tripo.default_candidate("t"), tripo.default_candidate("t")
        a["tripo"]["options"]["target_formats"].append("fbx")
        self.assertEqual(b["tripo"]["options"]["target_formats"], ["glb"])
        self.assertEqual(tripo.OPTION_DEFAULTS["target_formats"], ["glb"])
        self.assertEqual(a["task"], "img2mesh")
        self.assertEqual(a["model"], "v3.1-20260211")

    def test_options_of_normalizes(self):
        o = tripo.options_of(cand(texture_quality="ultra", target_formats=["xyz", "fbx"],
                                  face_limit="999999999", spec="nope", rig_type="dragon",
                                  compress="zip"))
        self.assertEqual(o["texture_quality"], "standard")
        self.assertEqual(o["target_formats"], ["fbx"])
        self.assertIsNone(o["face_limit"])              # out of range → ignored, not clamped
        self.assertEqual(o["spec"], "mixamo")
        self.assertEqual(o["rig_type"], "biped")
        self.assertEqual(o["compress"], "")

    def test_rig_endpoint_narrows_formats(self):
        o = tripo.options_of(cand("rig", target_formats=["obj", "fbx", "glb"]))
        self.assertEqual(o["target_formats"], ["fbx", "glb"])   # stored order, filtered to RIG_FORMATS
        o = tripo.options_of(cand("rig", target_formats=["obj"]))
        self.assertEqual(o["target_formats"], ["glb"])

    def test_generate_parts_excludes_texture_pbr_quad_lowpoly(self):
        o = tripo.options_of(cand(generate_parts=True, texture=True, pbr=True, quad=True, smart_low_poly=True))
        self.assertTrue(o["generate_parts"])
        self.assertFalse(o["texture"]); self.assertFalse(o["pbr"])
        self.assertFalse(o["quad"]); self.assertFalse(o["smart_low_poly"])

    def test_rig_model_v1_rigs_bipeds_only(self):
        o = tripo.options_of(cand("rig", rig_model="v1.0-20240301", rig_type="quadruped"))
        self.assertEqual(o["rig_type"], "biped")            # admin option normalized
        o = tripo.options_of(cand("rig", rig_model="v2.5-20260210", rig_type="quadruped"))
        self.assertEqual(o["rig_type"], "quadruped")        # v2.5 does every creature

    def test_face_limit_for(self):
        self.assertEqual(tripo.face_limit_for("v3.1-20260211", False, "2500000"), 1_500_000)
        self.assertEqual(tripo.face_limit_for("v3.1-20260211", True, "2500000"), 150_000)
        self.assertEqual(tripo.face_limit_for("P2-20260801", False, 10), 100)
        self.assertIsNone(tripo.face_limit_for("v3.1-20260211", False, ""))
        self.assertIsNone(tripo.face_limit_for("v3.1-20260211", False, "abc"))
        self.assertEqual(tripo.face_limit_for("unknown-model", False, 5000), 5000)

    def test_option_fields_cover_options(self):
        self.assertEqual({f["key"] for f in tripo.OPTION_FIELDS}, set(tripo.OPTION_DEFAULTS) - {"target_formats"})
        form = {}
        for fld in tripo.OPTION_FIELDS:
            v = tripo.OPTION_DEFAULTS[fld["key"]]
            if fld["type"] == "bool":
                if v:
                    form[f"opt__{fld['key']}"] = "on"
            else:
                form[f"opt__{fld['key']}"] = cloudtask.field_value_str(fld, v)
        out = cloudtask.parse_options(tripo.OPTION_FIELDS, form, tripo.OPTION_DEFAULTS)
        self.assertEqual(out, tripo.OPTION_DEFAULTS)


class BuildRequest(unittest.TestCase):
    def test_image_to_model_defaults(self):
        body = tripo.build_request(cand(), {}, {"input_image": "tok1"}, {})
        self.assertEqual(body["input"], "tok1")
        self.assertEqual(body["model"], "v3.1-20260211")
        self.assertTrue(body["texture"]); self.assertTrue(body["pbr"])
        self.assertEqual(body["texture_quality"], "standard")
        self.assertNotIn("face_limit", body)              # adaptive default: not sent
        self.assertNotIn("compress", body)                # "" = leave it out
        self.assertNotIn("inputs", body)

    def test_image_required(self):
        with self.assertRaises(tripo.TripoInput):
            tripo.build_request(cand(), {}, {}, {})

    def test_client_labels(self):
        body = tripo.build_request(cand(), {"input_face_num": "20000", "input_texture_resolution": 4096,
                                            "input_name": "x", "input_texture_prompt": "shiny"},
                                   {"input_image": "tok"}, {})
        self.assertEqual(body["face_limit"], 20000)
        self.assertEqual(body["texture_quality"], "detailed")
        self.assertNotIn("name", body)
        self.assertNotIn("texture_prompt", body)

    def test_admin_face_limit_is_a_default_not_a_pin(self):
        body = tripo.build_request(cand(face_limit=30000), {}, {"input_image": "t"}, {})
        self.assertEqual(body["face_limit"], 30000)
        body = tripo.build_request(cand(face_limit=30000), {"input_face_num": 500}, {"input_image": "t"}, {})
        self.assertEqual(body["face_limit"], 500)

    def test_texture_quality_buckets(self):
        self.assertEqual(tripo.texture_quality(1024), "standard")
        self.assertEqual(tripo.texture_quality(2048), "standard")
        self.assertEqual(tripo.texture_quality(4096), "detailed")
        self.assertEqual(tripo.texture_quality(8192), "extreme")
        self.assertEqual(tripo.texture_quality("extreme"), "extreme")
        self.assertEqual(tripo.texture_quality("zzz"), "standard")

    def test_multiview_shape_and_minimum(self):
        body = tripo.build_request(cand("multiview-to-model"), {},
                                   {"input_image_front": "f", "input_image_back": "b"}, {})
        self.assertEqual(body["inputs"], [{"front": "f"}, {"back": "b"}])
        self.assertNotIn("input", body)
        with self.assertRaises(tripo.TripoInput):          # fewer than two views
            tripo.build_request(cand("multiview-to-model"), {}, {"input_image_front": "f"}, {})
        with self.assertRaises(tripo.TripoInput):          # front missing
            tripo.build_request(cand("multiview-to-model"), {}, {"input_image_back": "b", "input_image_left": "l"}, {})

    def test_rig_body(self):
        body = tripo.build_request(cand("rig"), {}, {}, {"input_mesh_path": "mtok"})
        self.assertEqual(body, {"input": "mtok", "model": "v1.0-20240301", "rig_type": "biped",
                                "spec": "mixamo", "out_format": "glb"})
        body = tripo.build_request(cand("rig", target_formats=["fbx"], rig_model="v2.5-20260210", spec="tripo"),
                                   {"input_rig_type": "quadruped"}, {}, {"input_mesh_path": "mtok"})
        self.assertEqual((body["out_format"], body["model"], body["spec"], body["rig_type"]),
                         ("fbx", "v2.5-20260210", "tripo", "quadruped"))
        with self.assertRaises(tripo.TripoInput):
            tripo.build_request(cand("rig"), {"input_rig_type": "dragon"}, {}, {"input_mesh_path": "m"})
        with self.assertRaises(tripo.TripoInput):
            tripo.build_request(cand("rig"), {}, {}, {})

    def test_client_rig_type_must_suit_the_rig_model(self):
        with self.assertRaises(tripo.TripoInput) as e:      # refused, not silently biped
            tripo.build_request(cand("rig"), {"input_rig_type": "quadruped"}, {}, {"input_mesh_path": "m"})
        self.assertIn("v1.0-20240301", str(e.exception))
        self.assertIn("biped", str(e.exception))
        body = tripo.build_request(cand("rig", rig_model="v2.5-20260210"),
                                   {"input_rig_type": "quadruped"}, {}, {"input_mesh_path": "m"})
        self.assertEqual(body["rig_type"], "quadruped")

    def test_follow_up_bodies(self):
        self.assertEqual(tripo.build_rig_check("t"), {"input": "t"})
        self.assertEqual(tripo.build_convert("task_1", "fbx", False), {"input": "task_1", "format": "FBX", "with_animation": False})
        self.assertEqual(tripo.build_convert("task_1", "gltf", True), {"input": "task_1", "format": "GLTF", "with_animation": True})
        self.assertEqual(tripo.build_retarget("task_r", "preset:walk", "glb"),
                         {"input": "task_r", "animation": "preset:walk", "out_format": "glb"})
        self.assertEqual(tripo.clip_name("preset:walk"), "walk")
        self.assertEqual(tripo.clip_name("preset:quadruped:walk"), "quadruped_walk")

    def test_request_summary_is_a_copy(self):
        body = {"input": "tok", "model": "m"}
        s = tripo.request_summary(body)
        self.assertEqual(s, body)
        self.assertIsNot(s, body)


class PublicFields(unittest.TestCase):
    def test_image_endpoint(self):
        params, images, files = tripo.public_fields(cand())
        self.assertEqual([i["name"] for i in images], ["input_image"])
        self.assertEqual(images[0]["on_empty"], "required")
        self.assertEqual(files, [])
        names = {p["name"] for p in params}
        self.assertEqual(names, {"input_name", "input_face_num", "input_texture_resolution",
                                 "input_remove_background", "input_no_fingers"})
        tr = next(p for p in params if p["name"] == "input_texture_resolution")
        self.assertEqual(tr["default"], 2048)
        fn = next(p for p in params if p["name"] == "input_face_num")
        self.assertNotIn("default", fn)
        fn = next(p for p in tripo.public_fields(cand(face_limit=30000))[0] if p["name"] == "input_face_num")
        self.assertEqual(fn["default"], 30000)

    def test_multiview_endpoint(self):
        _, images, _ = tripo.public_fields(cand("multiview-to-model"))
        self.assertEqual([i["name"] for i in images],
                         ["input_image_front", "input_image_back", "input_image_left", "input_image_right"])
        self.assertEqual([i["on_empty"] for i in images], ["required", "skip", "skip", "skip"])

    def test_rig_endpoint(self):
        params, images, files = tripo.public_fields(cand("rig"))
        self.assertEqual(images, [])
        self.assertEqual(files, [{"name": "input_mesh_path", "required": True, "accept": ["glb"]}])
        rt = next(p for p in params if p["name"] == "input_rig_type")
        self.assertEqual(rt["default"], "biped")
        self.assertEqual(rt["choices"], ["biped"])          # default rig model is v1.0
        rt = next(p for p in tripo.public_fields(cand("rig", rig_model="v2.5-20260210"))[0]
                  if p["name"] == "input_rig_type")
        self.assertEqual(rt["choices"], list(tripo.RIG_TYPES))
        self.assertEqual({p["name"] for p in params}, {"input_rig_type", "input_name", "input_no_fingers"})


class ParseTask(unittest.TestCase):
    def test_running(self):
        st = tripo.parse_task({"status": "running", "progress": 40}, ["glb"])
        self.assertEqual((st.status, st.progress, st.error, st.downloads), ("running", 40, None, []))

    def test_success_generation(self):
        st = tripo.parse_task({"status": "success", "progress": 100,
                               "output": {"model_url": "u/m.glb", "rendered_image_url": "u/p.png"},
                               "credits_consumed": 30.0}, ["glb"])
        self.assertEqual(st.downloads, [("model.glb", "u/m.glb")])
        self.assertEqual(st.thumbnail, "u/p.png")
        self.assertEqual(st.credits, 30.0)

    def test_success_rig_names_by_out_format(self):
        st = tripo.parse_task({"status": "success", "output": {"model_url": "u/r.fbx"}}, ["fbx"], "rig")
        self.assertEqual(st.downloads, [("rigged.fbx", "u/r.fbx")])

    def test_rig_check_success_has_no_model_url(self):
        """The rig-check task delivers a VERDICT, not a mesh — so `model_url` is not
        required there, and its answer travels on the TaskState the adapter polls to."""
        st = tripo.parse_task({"status": "success", "output": {"riggable": False,
                                                               "rig_type": "avian"}},
                              ["glb"], "rig-check")
        self.assertEqual((st.riggable, st.rig_type, st.downloads), (False, "avian", []))
        st = tripo.parse_task({"status": "success", "output": {"riggable": True,
                                                               "rig_type": "biped"}},
                              ["glb"], "rig-check")
        self.assertIs(st.riggable, True)

    def test_success_without_model_url_raises(self):
        with self.assertRaises(tripo.TripoInput):
            tripo.parse_task({"status": "success", "output": {}}, ["glb"])

    def test_failed_cancelled_unknown_are_terminal(self):
        st = tripo.parse_task({"status": "failed", "error_code": 2018, "error_message": "too complex"}, ["glb"])
        self.assertIn("too complex", st.error)
        self.assertIn("2018", st.error)
        st = tripo.parse_task({"status": "cancelled"}, ["glb"])
        self.assertEqual(st.error, "cancelled")
        st = tripo.parse_task({"status": "banned"}, ["glb"])
        self.assertIn("banned", st.error)
        st = tripo.parse_task({"status": "success", "progress": "x", "output": {"model_url": "u"}}, ["glb"])
        self.assertEqual(st.progress, 0)


if __name__ == "__main__":
    unittest.main()
