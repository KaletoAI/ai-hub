"""Guard ArliAI checkpoint saves and input refusals: silently replacing an account's
checkpoint with an enum default, truncating dimensions, or dropping image inputs
would generate a plausible image with settings the caller never requested.
"""
import unittest
from unittest.mock import AsyncMock, patch

import arliai
from tests.test_ui_form_validation import _Fixture
import admin
import main
import store


class ArliAliases(_Fixture):
    def setUp(self):
        super().setUp()
        self.backend = {"name": "arli-images", "type": "arliai", "url": arliai.URL,
                        "max_image_pixels": 3_000_000}
        self.cand = arliai.default_candidate(self.backend["name"])
        self.cand["model"] = "Original Checkpoint [ab12]"
        store.upsert("arli-image", [self.cand])
        p = patch.object(admin, "_gen_backends", lambda: [self.backend])
        p.start()
        self.addCleanup(p.stop)

    def test_exact_account_checkpoint_and_fractional_cfg_saved(self):
        form = {"alias": "arli-image", "cloud_model": "Private Model v2 [ef34]",
                "cloud_endpoint": "txt2img", "opt__cfg_scale": "7.5"}
        response = self.post("/ui/mapping/cloud-update", form)
        self.assertEqual(response.status_code, 303)
        saved = store.get("arli-image")[0]
        self.assertEqual(saved["model"], "Private Model v2 [ef34]")
        self.assertEqual(saved["arliai"]["options"]["cfg_scale"], 7.5)
        self.assertEqual(saved["task"], "text2img")

    def test_refused_save_preserves_store_and_typed_settings(self):
        for bad in ({"cloud_model": ""}, {"opt__width": "1024.5"},
                    {"opt__cfg_scale": "nan"}):
            with self.subTest(bad=bad):
                form = {"alias": "arli-image", "cloud_model": "New Model",
                        "opt__sampler_name": "DPM++ 2M", **bad}
                response = self.post("/ui/mapping/cloud-update", form)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(store.get("arli-image"), [self.cand])
                self.assertIn('value="DPM++ 2M"', response.text)
                key = next(iter(bad))
                self.assertIn(f'name="{key}" value="{bad[key]}"', response.text)

    def test_generation_rejects_unsupported_or_overbudget_input_before_job(self):
        routes = [(self.backend, self.cand)]
        with patch.object(main, "_gen_pick", AsyncMock(return_value=(routes, False, None))), \
                patch.object(main, "get_gen_routes", return_value=routes):
            for extra in ({"size": "2001x1500"}, {"n": 2},
                          {"ref_images": ["not-needed-base64"]}):
                with self.subTest(extra=extra):
                    response = self.c.post("/v1/images/generations", json={
                        "model": "arli-image", "prompt": "bird", **extra})
                    self.assertEqual(response.status_code, 400, response.text)
            response = self.c.post("/v1/images/edits", data={"model": "arli-image"},
                                   files={"image": ("image.png", b"image", "image/png")})
            self.assertEqual(response.status_code, 400, response.text)


if __name__ == "__main__":
    unittest.main()
