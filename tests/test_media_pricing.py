"""Media "from" prices: what the cheapest request on a media model costs at least, read
off the provider's own catalog (NanoGPT `pricing`, Venice `model_spec.pricing`, Tripo /
Meshy credit tables, RunPod `cost_per_hour`).

Why this fails SILENTLY: every wrong reading is still a plausible small number. A
setting read as a price (`default_frames_per_second: 24` → "$24/s"), a surcharge read
as the base (an upscale or a reference-video rate), a paid model with no price shown
as free, or an alias published at a price one of its backends does not have — none
of it errors; the bill disagrees later. Each shape seen in the real catalogs
(2026-11-04) is pinned here with the value it must give.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    import admin
    import mediaprice as mp
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


def nano(pricing):
    return mp.from_price("nanogpt", {"model_metadata": {"pricing": pricing}}, {"paid": True})


def venice(pricing):
    return mp.from_price("venice", {"model_metadata": {"model_spec": {"pricing": pricing}}}, {"paid": True})


class NanoGPT(unittest.TestCase):
    def test_image_by_size_and_rendering_speed(self):
        self.assertEqual(nano({"per_image": {"768x768": 0.008, "1024x1024": 0.009}, "currency": "USD"}),
                         {"usd": 0.008, "unit": "image"})
        # TURBO is as callable as the default size table, and cheaper
        self.assertEqual(nano({"per_image": {"1024x1024": 0.0105},
                               "by_rendering_speed": {"TURBO": {"1024x1024": 0.00525}}})["usd"], 0.00525)

    def test_reference_surcharge_is_not_the_base(self):
        p = nano({"per_image": {"1024*1024": 0.04}, "per_image_with_references": 0.048,
                  "extra_reference_image_per_output": 0.008})
        self.assertEqual(p, {"usd": 0.04, "unit": "image"})

    def test_empty_size_table_falls_back_to_starting_at(self):
        self.assertEqual(nano({"per_image": {}, "starting_at": 0.08, "currency": "USD"}),
                         {"usd": 0.08, "unit": "run"})

    def test_3d_variants(self):
        self.assertEqual(nano({"per_run": 0.48, "per_run_by_variant": {"text:none": 0.36}})["usd"], 0.36)

    def test_video_per_second_with_floor(self):
        p = nano({"per_second": 0.07, "supported_durations": [3, 4, 5]})
        self.assertEqual(p, {"usd": 0.07, "unit": "second", "min_s": 3.0})
        self.assertAlmostEqual(mp.request_usd(p), 0.21)
        # a rate without a known minimum has no per-request floor
        self.assertIsNone(mp.request_usd(nano({"per_second_by_resolution": {"540p": 0.1}})))

    def test_video_long_tail(self):
        self.assertEqual(nano({"with_audio": 4.8, "without_audio": 3.2, "fixed_duration_seconds": 8}),
                         {"usd": 3.2, "unit": "video"})
        self.assertEqual(nano({"base_price": 3.4, "base_duration": 5, "per_extra_second": 0.68}),
                         {"usd": 3.4, "unit": "video"})
        self.assertEqual(nano({"text_to_video_per_second": {"720p": 0.14},
                               "text_to_video_with_reference_video_input_per_second": {"720p": 0.075}}),
                         {"usd": 0.14, "unit": "second"})

    def test_a_setting_is_never_a_price(self):
        # ltx-2.3-quality: per_megapixel billing + default_frames_per_second 24 → unknown, not $24/s
        self.assertIsNone(nano({"per_megapixel": 0.0024, "default_frames_per_second": 24,
                                "default_duration": 5, "currency": "USD"}))

    def test_tts_and_minimum_only(self):
        self.assertEqual(nano({"per_thousand_chars": 0.0221}), {"usd": 0.0221, "unit": "1k chars"})
        self.assertEqual(nano({"minimum": 0.408, "note": "…"}), {"usd": 0.408, "unit": "run"})

    def test_other_currency_is_unknown(self):
        self.assertIsNone(nano({"per_image": {"a": 1}, "currency": "EUR"}))


class Venice(unittest.TestCase):
    def test_flat_generation_ignores_upscale(self):
        self.assertEqual(venice({"generation": {"usd": 0.01}, "upscale": {"4x": {"usd": 0.08}}}),
                         {"usd": 0.01, "unit": "run"})

    def test_flat_price_on_an_image_endpoint_is_per_image(self):
        cand = {"venice": {"endpoint": "images"},
                "model_metadata": {"model_spec": {"pricing": {"generation": {"usd": 0.01}}}}}
        self.assertEqual(mp.from_price("venice", cand, {"paid": True}), {"usd": 0.01, "unit": "image"})

    def test_cheapest_resolution_or_quality(self):
        p = venice({"resolutions": {"1K": {"usd": 0.07}}, "quality": {"1K": {"low": {"usd": 0.05}}}})
        self.assertEqual(p, {"usd": 0.05, "unit": "image"})

    def test_audio_and_unpriced_video(self):
        self.assertEqual(venice({"input": {"usd": 3.5}}), {"usd": 3.5, "unit": "1M chars"})
        self.assertEqual(venice({"durations": {"60": {"usd": 0.03}, "90": {"usd": 0.04}}})["usd"], 0.03)
        self.assertIsNone(venice(None))            # Venice video lists no price


class Kinds(unittest.TestCase):
    def test_credit_vendors_and_runpod(self):
        self.assertEqual(mp.from_price("tripo", {"tripo": {"endpoint": "rig"}}, {"paid": True}),
                         {"usd": 0.25, "unit": "run"})
        self.assertEqual(mp.from_price("meshy", {}, {"paid": True}), {"credits": 20, "unit": "run"})
        self.assertIsNone(mp.request_usd({"credits": 20, "unit": "run"}))
        self.assertEqual(mp.from_price("runpod", {}, {"paid": True, "cost_per_hour": 1.75}),
                         {"usd": 1.75, "unit": "hour"})

    def test_unpaid_is_free_paid_unknown_is_none(self):
        self.assertTrue(mp.from_price("comfyui", {}, {})["free"])
        self.assertIsNone(mp.from_price("runpod", {}, {"paid": True}))

    def test_cheapest_over_an_alias(self):
        best = mp.cheapest([{"usd": 0.04, "unit": "image"}, {"usd": 0.01, "unit": "image"}, None])
        self.assertEqual((best["usd"], best["unknown"]), (0.01, 1))
        self.assertEqual(mp.cheapest([None]), {"unknown": 1})
        self.assertIsNone(mp.cheapest([]))

    def test_usd_str_has_no_exponent(self):
        self.assertEqual(mp.usd_str(0.00525), "0.00525")
        self.assertEqual(mp.usd_str(0.0), "0")


class Catalog(unittest.TestCase):
    def _pairs(self, *prices):
        return [({"name": f"b{i}", "type": "nanogpt", "paid": True},
                 {"backend": f"b{i}", "nanogpt": {"endpoint": "images"},
                  "model_metadata": {"pricing": p}}) for i, p in enumerate(prices)]

    def test_request_price_is_the_cheapest_floor(self):
        with patch.object(main, "_media_price_pairs", lambda m: self._pairs(
                {"per_image": {"a": 0.04}}, {"per_image": {"a": 0.009}})):
            self.assertEqual(main._media_request_pricing("x"), {"request": "0.009"})

    def test_any_candidate_without_a_floor_omits_it(self):
        with patch.object(main, "_media_price_pairs", lambda m: self._pairs(
                {"per_image": {"a": 0.04}}, {"per_second_by_resolution": {"720p": 0.1}})):
            self.assertIsNone(main._media_request_pricing("x"))
        with patch.object(main, "_media_price_pairs", lambda m: []):
            self.assertIsNone(main._media_request_pricing("x"))


class Console(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(admin._media_price_short({"usd": 0.008, "unit": "image"}), "from $0.008/image")
        self.assertEqual(admin._media_price_short({"usd": 0.07, "unit": "second", "unknown": 1}),
                         "from $0.07/s +?")
        self.assertEqual(admin._media_price_short({"credits": 20, "unit": "run"}), "from 20 credits")
        self.assertEqual(admin._media_price_short({"usd": 0.0, "unit": "run", "free": True}), "free")
        self.assertEqual(admin._media_price_short({"unknown": 2}), "price ?")
        self.assertIn("≥ $0.21 at the shortest 3 s",
                      admin._media_price_long({"usd": 0.07, "unit": "second", "min_s": 3.0}))
        self.assertIn("$1.75/h", admin._media_price_long({"usd": 1.75, "unit": "hour"}))


if __name__ == "__main__":
    unittest.main()
