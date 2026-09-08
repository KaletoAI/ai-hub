"""Unit tests for meshy.py — run: venv/bin/python -m unittest tests.test_meshy -v"""
import base64
import unittest

import meshy

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 16
GLB = b"glTF" + b"\x00" * 60


def _cand(endpoint="image-to-3d", **opts):
    return {"backend": "meshy", "task": "img2mesh", "model": "latest",
            "meshy": {"endpoint": endpoint, "options": opts}}


class TestTextureRes(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(meshy.texture_res(1024), "2k")
        self.assertEqual(meshy.texture_res(2048), "2k")
        self.assertEqual(meshy.texture_res(4096), "4k")
        self.assertEqual(meshy.texture_res(8192), "8k")
        self.assertEqual(meshy.texture_res("4k"), "4k")      # already a bucket → verbatim
        self.assertEqual(meshy.texture_res("garbage"), "2k")  # unknown → default


class TestDataUri(unittest.TestCase):
    def test_png_and_jpeg(self):
        self.assertTrue(meshy.data_uri(PNG).startswith("data:image/png;base64,"))
        self.assertTrue(meshy.data_uri(JPG).startswith("data:image/jpeg;base64,"))
        self.assertEqual(base64.b64decode(meshy.data_uri(PNG).split(",", 1)[1]), PNG)

    def test_webp_rejected(self):
        with self.assertRaises(meshy.MeshyInput):
            meshy.data_uri(WEBP)


class TestBuildRequestSingle(unittest.TestCase):
    def test_minimal(self):
        body = meshy.build_request(_cand(), {}, {"input_image": PNG})
        self.assertTrue(body["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(body["ai_model"], "latest")
        self.assertTrue(body["should_texture"])
        self.assertEqual(body["texture_resolution"], "2k")
        self.assertEqual(body["target_formats"], ["glb"])
        self.assertNotIn("image_urls", body)
        self.assertNotIn("should_remesh", body)        # untouched → Meshy's model default

    def test_missing_image_is_input_error(self):
        with self.assertRaises(meshy.MeshyInput):
            meshy.build_request(_cand(), {}, {})

    def test_face_num_sets_polycount_and_remesh(self):
        body = meshy.build_request(_cand(), {"input_face_num": 20000}, {"input_image": PNG})
        self.assertEqual(body["target_polycount"], 20000)
        self.assertTrue(body["should_remesh"])

    def test_face_num_clamped(self):
        lo = meshy.build_request(_cand(), {"input_face_num": 5}, {"input_image": PNG})
        hi = meshy.build_request(_cand(), {"input_face_num": "999999"}, {"input_image": PNG})
        self.assertEqual(lo["target_polycount"], 100)
        self.assertEqual(hi["target_polycount"], 300000)

    def test_admin_target_polycount_applies_without_a_client_value(self):
        """The admin face budget is what keeps a CHAINED alias shippable: measured
        2026-09-02, a no-remesh humanoid came back at 70 MB and the rigging hand-off
        died on it (and Meshy's rigger refuses >300k faces anyway)."""
        body = meshy.build_request(_cand(target_polycount=150000), {}, {"input_image": PNG})
        self.assertEqual(body["target_polycount"], 150000)
        self.assertTrue(body["should_remesh"])      # a polycount is inert without the pass

    def test_client_face_num_beats_the_admin_option(self):
        body = meshy.build_request(_cand(target_polycount=150000),
                                   {"input_face_num": 20000}, {"input_image": PNG})
        self.assertEqual(body["target_polycount"], 20000)
        self.assertTrue(body["should_remesh"])

    def test_out_of_range_admin_option_is_ignored(self):
        for bad in (5, 400000, "abc", "", None):
            body = meshy.build_request(_cand(target_polycount=bad), {}, {"input_image": PNG})
            self.assertNotIn("target_polycount", body, f"option {bad!r} leaked")

    def test_options_of_coerces_target_polycount(self):
        self.assertEqual(meshy.options_of(_cand(target_polycount="20000"))["target_polycount"], 20000)
        self.assertIsNone(meshy.options_of(_cand())["target_polycount"])
        self.assertIsNone(meshy.options_of(_cand(target_polycount=99))["target_polycount"])
        self.assertIsNone(meshy.options_of(_cand(target_polycount=300001))["target_polycount"])
        self.assertEqual(meshy.options_of(_cand(target_polycount=300000))["target_polycount"], 300000)

    def test_client_params_and_ignored(self):
        body = meshy.build_request(_cand(), {
            "input_name": "x" * 150, "input_texture_resolution": 4096,
            "input_texture_prompt": "rusty metal", "input_pose": "t-pose",
            "input_remove_background": False, "input_no_fingers": True,
            "input_bogus": 1, "prompt": ""}, {"input_image": JPG})
        self.assertEqual(len(body["name"]), 100)
        self.assertEqual(body["texture_resolution"], "4k")
        self.assertEqual(body["texture_prompt"], "rusty metal")
        self.assertEqual(body["pose_mode"], "t-pose")
        for k in ("input_bogus", "input_remove_background", "input_no_fingers", "prompt"):
            self.assertNotIn(k, body)

    def test_bad_pose_is_input_error(self):
        with self.assertRaises(meshy.MeshyInput):
            meshy.build_request(_cand(), {"input_pose": "sitting"}, {"input_image": PNG})

    def test_admin_options_and_model(self):
        c = _cand(enable_pbr=True, topology="quad", ultra_mode=True, target_formats=["glb", "fbx"],
                  texture_resolution="8k")
        c["model"] = "meshy-7"
        body = meshy.build_request(c, {}, {"input_image": PNG})
        self.assertEqual(body["ai_model"], "meshy-7")
        self.assertTrue(body["enable_pbr"])
        self.assertEqual(body["topology"], "quad")
        self.assertTrue(body["ultra_mode"])
        self.assertEqual(body["target_formats"], ["glb", "fbx"])
        self.assertEqual(body["texture_resolution"], "8k")          # admin default …
        body2 = meshy.build_request(c, {"input_texture_resolution": 1024}, {"input_image": PNG})
        self.assertEqual(body2["texture_resolution"], "2k")         # … the client may override

    def test_thumbnail_option_not_sent(self):
        body = meshy.build_request(_cand(thumbnail=False), {}, {"input_image": PNG})
        self.assertNotIn("thumbnail", body)


class TestBuildRequestMulti(unittest.TestCase):
    C = _cand("multi-image-to-3d")

    def test_front_first_optional_dropped(self):
        body = meshy.build_request(self.C, {}, {"input_image_left": JPG, "input_image_front": PNG})
        self.assertEqual(len(body["image_urls"]), 2)
        self.assertTrue(body["image_urls"][0].startswith("data:image/png"))
        self.assertTrue(body["image_urls"][1].startswith("data:image/jpeg"))
        self.assertNotIn("image_url", body)

    def test_all_four_in_order(self):
        imgs = {"input_image_right": PNG, "input_image_back": PNG,
                "input_image_left": PNG, "input_image_front": JPG}
        body = meshy.build_request(self.C, {}, imgs)
        self.assertEqual(len(body["image_urls"]), 4)
        self.assertTrue(body["image_urls"][0].startswith("data:image/jpeg"))

    def test_missing_front_is_input_error(self):
        with self.assertRaises(meshy.MeshyInput):
            meshy.build_request(self.C, {}, {"input_image_back": PNG})

    def test_empty_bytes_count_as_absent(self):
        body = meshy.build_request(self.C, {}, {"input_image_front": PNG, "input_image_back": b""})
        self.assertEqual(len(body["image_urls"]), 1)


class TestPublicFields(unittest.TestCase):
    def test_single(self):
        params, images, files = meshy.public_fields(_cand())
        self.assertEqual(files, [])
        self.assertEqual([i["name"] for i in images], ["input_image"])
        self.assertEqual(images[0]["on_empty"], "required")
        self.assertTrue(images[0]["required"])
        names = [p["name"] for p in params]
        for n in ("input_name", "input_face_num", "input_texture_resolution",
                  "input_texture_prompt", "input_pose", "input_remove_background", "input_no_fingers"):
            self.assertIn(n, names)
        pose = next(p for p in params if p["name"] == "input_pose")
        self.assertEqual(pose["choices"], ["", "a-pose", "t-pose"])
        tex = next(p for p in params if p["name"] == "input_texture_resolution")
        self.assertEqual(tex["default"], 2048)
        self.assertEqual(tex["type"], "int")
        # no advertised default — build_request never applies one, and sending it
        # would force should_remesh (see meshy.public_fields)
        face = next(p for p in params if p["name"] == "input_face_num")
        self.assertNotIn("default", face)

    def test_multi(self):
        _, images, _ = meshy.public_fields(_cand("multi-image-to-3d"))
        self.assertEqual([i["name"] for i in images],
                         ["input_image_front", "input_image_back", "input_image_left", "input_image_right"])
        self.assertEqual([i["on_empty"] for i in images], ["required", "skip", "skip", "skip"])

    def test_default_from_admin_option(self):
        params, _, _ = meshy.public_fields(_cand(texture_resolution="4k"))
        tex = next(p for p in params if p["name"] == "input_texture_resolution")
        self.assertEqual(tex["default"], 4096)

    def test_face_num_default_iff_the_admin_option_is_set(self):
        """With the option set the schema MAY promise a default — build_request really
        applies it now. With it blank there is still nothing to promise."""
        params, _, _ = meshy.public_fields(_cand(target_polycount=150000))
        face = next(p for p in params if p["name"] == "input_face_num")
        self.assertEqual(face["default"], 150000)
        params, _, _ = meshy.public_fields(_cand(target_polycount=400000))   # invalid → ignored
        face = next(p for p in params if p["name"] == "input_face_num")
        self.assertNotIn("default", face)


class TestParseTask(unittest.TestCase):
    OK = {"id": "t1", "status": "SUCCEEDED", "progress": 100, "consumed_credits": 30,
          "model_urls": {"glb": "https://a/x.glb?e=1", "fbx": "https://a/x.fbx"},
          "thumbnail_url": "https://a/p.png", "task_error": {"message": ""}}

    def test_succeeded(self):
        st = meshy.parse_task(self.OK, ["glb"])
        self.assertEqual(st.status, "SUCCEEDED")
        self.assertEqual(st.downloads, [("model.glb", "https://a/x.glb?e=1")])
        self.assertEqual(st.thumbnail, "https://a/p.png")
        self.assertEqual(st.credits, 30)
        self.assertIsNone(st.error)

    def test_succeeded_missing_format_is_error(self):
        with self.assertRaises(meshy.MeshyInput):
            meshy.parse_task(self.OK, ["glb", "usdz"])

    def test_failed(self):
        st = meshy.parse_task({"status": "FAILED", "progress": 30,
                               "task_error": {"message": "bad image"}}, ["glb"])
        self.assertEqual(st.status, "FAILED")
        self.assertEqual(st.error, "bad image")
        self.assertEqual(st.downloads, [])

    def test_pending(self):
        st = meshy.parse_task({"status": "PENDING", "progress": 0, "preceding_tasks": 3}, ["glb"])
        self.assertEqual(st.status, "PENDING")
        self.assertEqual(st.downloads, [])
        self.assertIsNone(st.error)

    def test_unknown_status_is_terminal_failed(self):
        # Not falling through as "still running": an unknown state would be polled until
        # max_wait, holding the backend slot for the whole wait to learn nothing.
        st = meshy.parse_task({"status": "EXPIRED", "progress": 10,
                               "model_urls": {"glb": "https://a/x.glb"}}, ["glb"])
        self.assertEqual(st.status, "EXPIRED")
        self.assertEqual(st.error, "unknown task status 'EXPIRED'")
        self.assertEqual(st.downloads, [])

    def test_missing_status_is_terminal_failed(self):
        st = meshy.parse_task({"progress": 0}, ["glb"])
        self.assertIsNotNone(st.error)
        self.assertEqual(st.downloads, [])

    def test_non_integer_progress_is_zero(self):
        st = meshy.parse_task({"status": "IN_PROGRESS", "progress": "abc"}, ["glb"])
        self.assertEqual(st.progress, 0)
        self.assertIsNone(st.error)


class TestRequestSummary(unittest.TestCase):
    def test_images_replaced(self):
        body = meshy.build_request(_cand("multi-image-to-3d"), {"input_name": "n"},
                                   {"input_image_front": PNG, "input_image_back": JPG})
        s = meshy.request_summary(body)
        self.assertEqual(s["name"], "n")
        self.assertEqual(s["image_urls"], [f"<{len(PNG)} bytes>", f"<{len(JPG)} bytes>"])
        self.assertNotIn("data:", str(s))


class TestRigging(unittest.TestCase):
    C = {"backend": "meshy", "task": "mesh2rig", "model": "latest",
         "meshy": {"endpoint": "rigging", "options": {"target_formats": ["glb"]}}}

    def test_build_request(self):
        body = meshy.build_request(self.C, {"input_name": "Kai", "input_height_m": 1.8},
                                   {}, {"input_mesh_path": ("hero.glb", GLB)})
        self.assertTrue(body["model_url"].startswith("data:model/gltf-binary;base64,"))
        self.assertEqual(body["height_meters"], 1.8)
        self.assertEqual(body["name"], "Kai")
        for k in ("ai_model", "image_url", "should_texture", "target_formats"):
            self.assertNotIn(k, body)

    def test_default_height(self):
        body = meshy.build_request(self.C, {}, {}, {"input_mesh_path": ("h.glb", GLB)})
        self.assertEqual(body["height_meters"], 1.7)

    def test_missing_mesh_and_non_glb(self):
        with self.assertRaises(meshy.MeshyInput):
            meshy.build_request(self.C, {}, {}, {})
        with self.assertRaises(meshy.MeshyInput):
            meshy.build_request(self.C, {}, {}, {"input_mesh_path": ("h.obj", b"v 0 0 0")})

    def test_public_fields_triple(self):
        params, images, files = meshy.public_fields(self.C)
        self.assertEqual(images, [])
        self.assertEqual(files, [{"name": "input_mesh_path", "required": True, "accept": ["glb"]}])
        names = [p["name"] for p in params]
        self.assertEqual(names, ["input_name", "input_height_m", "input_no_fingers"])
        p2, i2, f2 = meshy.public_fields(_cand())
        self.assertEqual(f2, [])
        self.assertEqual([i["name"] for i in i2], ["input_image"])

    def test_parse_rigging_result(self):
        task = {"status": "SUCCEEDED", "progress": 100, "consumed_credits": 5,
                "result": {"rigged_character_glb_url": "https://a/r.glb", "rigged_character_fbx_url": "https://a/r.fbx",
                           "basic_animations": {"walking_glb_url": "https://a/w.glb", "running_glb_url": "https://a/x.glb",
                                                "walking_fbx_url": "https://a/w.fbx", "running_fbx_url": "https://a/x.fbx"}}}
        st = meshy.parse_task(task, ["glb"], "rigging")
        self.assertEqual(st.downloads, [("rigged.glb", "https://a/r.glb")])
        st = meshy.parse_task(task, ["glb", "fbx"], "rigging", animations=True)
        self.assertEqual([n for n, _ in st.downloads],
                         ["rigged.glb", "rigged.fbx", "walking.glb", "running.glb", "walking.fbx", "running.fbx"])
        with self.assertRaises(meshy.MeshyInput):
            meshy.parse_task({"status": "SUCCEEDED", "result": {}}, ["glb"], "rigging")

    def test_image_endpoint_downloads_are_named(self):
        st = meshy.parse_task({"status": "SUCCEEDED", "model_urls": {"glb": "https://a/m.glb"}}, ["glb"])
        self.assertEqual(st.downloads, [("model.glb", "https://a/m.glb")])

    def test_request_summary_hides_mesh(self):
        body = meshy.build_request(self.C, {}, {}, {"input_mesh_path": ("h.glb", GLB)})
        self.assertEqual(meshy.request_summary(body)["model_url"], f"<{len(GLB)} bytes>")

    def test_formats_restricted_to_glb_fbx(self):
        c = {"meshy": {"endpoint": "rigging", "options": {"target_formats": ["obj", "usdz"]}}}
        self.assertEqual(meshy.options_of(c)["target_formats"], ["glb"])
        c["meshy"]["options"]["target_formats"] = ["fbx", "obj"]
        self.assertEqual(meshy.options_of(c)["target_formats"], ["fbx"])


class TestDefaultCandidate(unittest.TestCase):
    def test_shape(self):
        c = meshy.default_candidate("meshy-cloud")
        self.assertEqual(c["backend"], "meshy-cloud")
        self.assertEqual(c["task"], "img2mesh")
        self.assertEqual(c["model"], "latest")
        self.assertEqual(c["meshy"]["endpoint"], "image-to-3d")
        self.assertEqual(c["meshy"]["options"], meshy.OPTION_DEFAULTS)

    def test_options_are_a_deep_copy(self):
        """A stored candidate must not share the module constant's nested list —
        one in-place edit would otherwise rewrite the default for every alias."""
        c = meshy.default_candidate("meshy-cloud")
        c["meshy"]["options"]["target_formats"].append("fbx")
        c["meshy"]["options"]["texture_resolution"] = "8k"
        self.assertEqual(meshy.OPTION_DEFAULTS["target_formats"], ["glb"])
        self.assertEqual(meshy.OPTION_DEFAULTS["texture_resolution"], "2k")
        fresh = meshy.default_candidate("meshy-cloud")
        self.assertEqual(fresh["meshy"]["options"]["target_formats"], ["glb"])
        self.assertEqual(fresh["meshy"]["options"], meshy.OPTION_DEFAULTS)


if __name__ == "__main__":
    unittest.main()
