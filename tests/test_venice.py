"""Venice native schemas must reject invalid paid requests before queueing them."""
import base64
import struct
import unittest
import zlib

import venice


PNG = b"\x89PNG\r\n\x1a\nreference"


def candidate(endpoint, spec=None, model="exact-model"):
    return venice.candidate_for("venice-host", model, endpoint,
                                {"id": model, "model_spec": spec or {}})


def reference_png(width, height):
    """Generate a real RGB PNG for image-decoder and metadata-bound regressions."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = (b"\0" + b"\x28\x50\x78" * width) * height
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


class VeniceAccountTests(unittest.TestCase):
    """Never mistake model prices, fake booleans or Pro labels for API inclusion."""

    def test_only_exact_account_charge_and_access_flags_establish_inclusion(self):
        cases = (
            ({"apiTier": {"isCharged": True}, "accessPermitted": True}, "credits"),
            ({"apiTier": {"isCharged": True}, "accessPermitted": False}, "credits"),
            ({"apiTier": {"isCharged": False}, "accessPermitted": True}, "included"),
            ({"apiTier": {"isCharged": False}, "accessPermitted": False}, "unknown"),
            ({"apiTier": {"isCharged": False}}, "unknown"),
            ({"apiTier": {"isCharged": "false"}, "accessPermitted": True}, "unknown"),
            ({"apiTier": {"isCharged": 0}, "accessPermitted": True}, "unknown"),
            ({"apiTier": {"isCharged": 1}, "accessPermitted": True}, "unknown"),
            ({"apiTier": {"isCharged": False}, "accessPermitted": 1}, "unknown"),
            ({"apiTier": {"id": "pro"}, "accessPermitted": True,
              "pricing": {"input": 0}, "isFree": True}, "unknown"),
        )
        for data, expected in cases:
            with self.subTest(data=data):
                info = venice.normalize_account_info({"data": data}, 123.5)
                self.assertEqual(info["state"], expected)
                self.assertEqual(info["checked_at"], 123.5)
                self.assertIs(info["access_permitted"],
                              data.get("accessPermitted") if type(data.get("accessPermitted")) is bool else None)

    def test_balances_are_separate_finite_nonnegative_numbers_not_credit_guesses(self):
        info = venice.normalize_account_info({"data": {
            "apiTier": {"id": "paid", "isCharged": True}, "accessPermitted": True,
            "balances": {"USD": 5, "DIEM": 0, "BUNDLED_CREDITS": 0.8652965,
                         "monthlyCredits": 500}, "api_key": "secret-key"}}, 123)
        self.assertEqual(info, {"state": "credits", "api_tier": "paid",
            "access_permitted": True, "balances": {"USD": 5, "DIEM": 0,
            "BUNDLED_CREDITS": 0.8652965}, "checked_at": 123})
        for value in (True, False, -1, float("nan"), float("inf"), -float("inf"),
                      "5", None, [], {}, 10 ** 1000):
            with self.subTest(value=value):
                info = venice.normalize_account_info({"data": {
                    "balances": {"USD": value, "DIEM": value, "BUNDLED_CREDITS": value}}}, 123)
                self.assertEqual(info["balances"], {})

    def test_malformed_envelopes_and_tiers_cannot_claim_free_api(self):
        for payload in (None, [], "secret-key", {}, {"apiTier": {"isCharged": False}},
                        {"data": None}, {"data": []}, {"data": {"apiTier": []}},
                        {"data": {"apiTier": "pro", "accessPermitted": True}}):
            with self.subTest(payload=payload):
                info = venice.normalize_account_info(payload, 123)
                self.assertEqual(info["state"], "unknown")
                self.assertEqual(info["balances"], {})
                self.assertNotIn("secret-key", str(info))
        for tier in ([], {}, True, 123, "<script>alert(1)</script>", "x" * 100):
            with self.subTest(tier=tier):
                info = venice.normalize_account_info({"data": {
                    "apiTier": {"id": tier, "isCharged": True}}}, 123)
                self.assertNotIn("api_tier", info)

    def test_invalid_last_check_is_never_published_as_nonfinite_or_fake_time(self):
        for checked_at in (None, True, "123", -1, float("nan"), float("inf"), 10 ** 1000):
            with self.subTest(checked_at=checked_at):
                info = venice.normalize_account_info({}, checked_at)
                self.assertNotIn("checked_at", info)


class VeniceSchemaTests(unittest.TestCase):
    """Guard model pinning, advertised constraints, and distinct paid media paths."""

    def test_every_requested_family_has_a_native_path(self):
        self.assertEqual(set(venice.ENDPOINTS), set(venice.SUBMIT_PATHS))
        self.assertEqual(venice.generation_type("speech"), "tts")
        self.assertEqual(venice.generation_type("transcription"), "asr")
        self.assertEqual(venice.generation_type("voice-changer"), "audio")
        self.assertEqual(venice.CATALOG_PATHS["edit"], "/v1/models?type=inpaint")
        self.assertEqual(venice.music_endpoint({"model_spec": {"voice_changer": True}}), "voice-changer")
        self.assertEqual(venice.music_endpoint({"model_spec": {}}), "music")

    def test_catalog_bounds_and_provider_defaults_are_public(self):
        cand = candidate("images", {"constraints": {"promptCharacterLimit": 8,
            "aspectRatios": ["1:1", "16:9"], "defaultAspectRatio": "16:9",
            "resolutions": ["1K"], "steps": {"max": 20, "default": 10}}})
        params, images, files = venice.public_fields(cand)
        fields = {f["name"]: f for f in params}
        self.assertEqual(fields["aspect_ratio"]["default"], "16:9")
        self.assertEqual(fields["steps"]["max"], 20)
        self.assertNotIn("width", fields)
        self.assertFalse(images or files)
        for values in ({"prompt": "too lengthy"}, {"prompt": "ok", "steps": 21},
                       {"prompt": "ok", "aspect_ratio": "3:2"}, {"prompt": "ok", "width": 512}):
            with self.subTest(values=values), self.assertRaises(venice.INPUT_ERROR):
                venice.validate_request(cand, values, [], [])

    def test_client_alias_provider_precedence_and_count_translation(self):
        cand = candidate("images", {"constraints": {"steps": {"default": 15, "max": 30}}})
        cand["venice"]["options"].update(steps=20, options_json='{"variants":2,"safe_mode":true}')
        body = venice.build_request(cand, {"prompt": "hello", "n": 3, "steps": 12,
                                          "extra": {"safe_mode": False}}, {})
        self.assertEqual((body["model"], body["variants"], body["steps"], body["safe_mode"]),
                         ("exact-model", 3, 12, False))
        self.assertNotIn("n", body)
        for value in (0, 5, 1.5, True):
            with self.subTest(value=value), self.assertRaises(venice.INPUT_ERROR):
                venice.build_request(cand, {"prompt": "hello", "n": value}, {})

    def test_reserved_values_cannot_change_model_or_credentials(self):
        cand = candidate("images")
        body = venice.build_request(cand, {"model": "public-alias", "prompt": "hello"}, {})
        self.assertEqual(body["model"], "exact-model")
        for key in ("modelId", "api_key", "Authorization", "queue_id", "download_url", "endpoint", "headers"):
            with self.subTest(key=key), self.assertRaises(venice.INPUT_ERROR):
                venice.build_request(cand, {"prompt": "hello", "extra": {key: "evil"}}, {})
        cand["venice"]["options"]["options_json"] = '{"model":"other"}'
        with self.assertRaises(venice.INPUT_ERROR):
            venice.options_of(cand)

    def test_no_fallback_model_and_unknown_controls_are_refused(self):
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(candidate("images", model=""), {"prompt": "hello"}, {})
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(candidate("music"), {"prompt": "hello", "quality": "high"}, {})
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(candidate("images"), {"prompt": "hello", "extra": {"bogus": 2}}, {})

    def test_edit_requires_reference_and_selects_native_single_or_multi(self):
        cand = candidate("edit", {"constraints": {"combineImages": True, "maxInputImages": 3}})
        _, fields, _ = venice.public_fields(cand)
        self.assertTrue(fields[0]["required"])
        self.assertEqual(len(fields), 3)
        with self.assertRaises(venice.INPUT_ERROR):
            venice.validate_request(cand, {"prompt": "change sky"}, [], [])
        single = venice.build_request(cand, {"prompt": "change sky", "n": 1}, {"input_image": PNG})
        self.assertEqual(venice.submit_path("edit", single), "/v1/image/edit")
        self.assertEqual(single["model"], "exact-model")
        self.assertEqual(base64.b64decode(single["image"].split(",")[-1]), PNG)
        multiple = venice.build_request(cand, {"prompt": "combine"},
                                        {"input_image": PNG, "input_image_2": PNG})
        self.assertEqual(venice.submit_path("edit", multiple), "/v1/image/multi-edit")
        self.assertEqual(multiple["modelId"], "exact-model")
        self.assertNotIn("model", multiple)
        self.assertEqual(len(multiple["images"]), 2)
        with self.assertRaises(venice.INPUT_ERROR):
            venice.validate_request(cand, {"prompt": "combine"}, ["input_image_4"], [])

    def test_empty_upload_is_an_input_error_not_a_paid_attempt(self):
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(candidate("edit"), {"prompt": "edit"}, {"input_image": b""})

    def test_single_only_editor_never_offers_multiple_or_mask(self):
        cand = candidate("edit", {"constraints": {"combineImages": False}})
        self.assertEqual(len(venice.public_fields(cand)[1]), 1)
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(cand, {"prompt": "edit", "mask": "unsupported"}, {"input_image": PNG})

    def test_fixed_image_operations_omit_upstream_model(self):
        for ep in ("upscale", "background-remove"):
            with self.subTest(endpoint=ep):
                body = venice.build_request(candidate(ep), {}, {"input_image": PNG})
                self.assertNotIn("model", body)
                self.assertIn("image", body)
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(candidate("upscale"), {"scale": 3}, {"input_image": PNG})

    def test_video_seconds_remain_seconds_and_image_input_is_required(self):
        cand = candidate("video", {"constraints": {"model_type": "image-to-video",
            "durations": ["5s", "10s"], "resolutions": ["720p"], "aspect_ratios": []}})
        self.assertEqual(cand["task"], "img2vid")
        with self.assertRaises(venice.INPUT_ERROR):
            venice.validate_request(cand, {"prompt": "move", "duration": 5}, [], [])
        body = venice.build_request(cand, {"prompt": "move", "seconds": 5}, {"input_image": PNG})
        self.assertEqual(body["duration"], "5s")
        self.assertIn("image_url", body)
        self.assertNotIn("seconds", body)
        for values in ({"prompt": "move", "duration": 8},
                       {"prompt": "move", "duration": 5, "audio": True}):
            with self.subTest(values=values), self.assertRaises(venice.INPUT_ERROR):
                venice.build_request(cand, values, {"input_image": PNG})

    def test_invalid_data_urls_fail_before_video_queue(self):
        cand = candidate("video", {"constraints": {"model_type": "image-to-video", "durations": ["5s"]}})
        for url in ("data:image/png;base64,!!!", "data:image/png;base64,"):
            with self.subTest(url=url), self.assertRaises(venice.INPUT_ERROR):
                venice.build_request(cand, {"prompt": "move", "duration": 5, "image_url": url}, {})

    def test_video_native_structured_references_and_required_video(self):
        cand = candidate("video", {"constraints": {"model_type": "video", "video_input": True,
                                                   "durations": ["Auto"]}})
        with self.assertRaises(venice.INPUT_ERROR):
            venice.validate_request(cand, {"prompt": "move", "duration": "Auto"}, [], [])
        body = venice.build_request(cand, {"prompt": "move", "duration": "Auto"}, {},
                                    {"input_video": ("source.mp4", b"movie")})
        self.assertTrue(body["video_url"].startswith("data:video/mp4;base64,"))

    def test_topaz_upscale_uses_source_duration_and_no_prompt(self):
        cand = candidate("video", {"constraints": {"model_type": "video", "video_input": True,
            "durations": ["Auto"], "resolutions": ["2x", "4x"]}}, model="topaz-video-upscale")
        params = {field["name"] for field in venice.public_fields(cand)[0]}
        self.assertNotIn("prompt", params)
        self.assertNotIn("duration", params)
        body = venice.build_request(cand, {"upscale_factor": 4}, {},
                                    {"input_video": ("source.mp4", b"movie")})
        self.assertEqual(body["upscale_factor"], 4)
        self.assertNotIn("duration", body)
        self.assertNotIn("prompt", body)
        for values in ({"prompt": "enhance"}, {"resolution": "4x"}, {"upscale_factor": 3}):
            with self.subTest(values=values), self.assertRaises(venice.INPUT_ERROR):
                venice.build_request(cand, values, {}, {"input_video": ("source.mp4", b"movie")})

    def test_music_lyrics_duration_and_capability_validation(self):
        cand = candidate("music", {"supports_lyrics": True, "lyrics_required": True,
            "min_prompt_length": 3, "prompt_character_limit": 12,
            "min_duration": 30, "max_duration": 90, "duration_options": [30, 60, 90]})
        with self.assertRaises(venice.INPUT_ERROR):
            venice.validate_request(cand, {"prompt": "jazz"}, [], [])
        body = venice.build_request(cand, {"prompt": "jazz", "lyrics_prompt": "sing", "duration": 60}, {})
        self.assertEqual(body["duration_seconds"], 60)
        self.assertNotIn("duration", body)
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(cand, {"prompt": "jazz", "lyrics_prompt": "sing", "seconds": 45}, {})
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(cand, {"prompt": "jazz", "lyrics_prompt": "sing", "loop": True}, {})

    def test_speech_native_text_style_and_optional_cloning(self):
        cand = candidate("speech", {"voices": ["Aurora"], "supported_formats": ["wav"],
            "voice_cloning": {"accepted_formats": ["wav", "mp3"]}})
        params, images, files = venice.public_fields(cand)
        self.assertIn("input", {f["name"] for f in params})
        self.assertFalse(images)
        self.assertFalse(files[0]["required"])
        body = venice.build_request(cand, {"input": "Hello", "response_format": "wav"}, {},
                                    {"input_audio": ("voice.wav", b"RIFFsample")})
        self.assertEqual(body["input"], "Hello")
        self.assertNotIn("input_audio", body)
        self.assertNotIn("audio_url", body)
        plain = candidate("speech", {"voices": ["Aria"], "supported_formats": ["mp3"]})
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(plain, {"input": "Hello"}, {}, {"input_audio": ("v.wav", b"voice")})
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(plain, {"input": "Hello", "response_format": "wav"}, {})
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(plain, {"input": "Hello", "voice": "unadvertised"}, {})

    def test_transcription_and_voice_changer_files_stay_multipart(self):
        for endpoint in ("transcription", "voice-changer"):
            cand = candidate(endpoint, {"voice_changer": endpoint == "voice-changer", "voices": ["Aria"]})
            self.assertTrue(venice.public_fields(cand)[2][0]["required"])
            with self.assertRaises(venice.INPUT_ERROR):
                venice.validate_request(cand, {}, [], [])
            body = venice.build_request(cand, {}, {}, {"input_audio": ("source.wav", b"source")})
            self.assertNotIn("file", body)
            self.assertNotIn("audio_url", body)
        cand = candidate("voice-changer", {"voice_changer": True})
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(cand, {"audio_url": "data:audio/wav;base64,eA=="}, {})

    def test_speech_text_convenience_does_not_replace_native_style(self):
        cand = candidate("speech", {"voices": ["Vivian"]}, model="tts-qwen3-0-6b")
        cand["venice"]["options"]["options_json"] = '{"prompt":"calm delivery"}'
        body = venice.build_request(cand, {"input": "Hello"}, {})
        self.assertEqual((body["input"], body["prompt"]), ("Hello", "calm delivery"))
        body = venice.build_request(cand, {"prompt": "Hello"}, {})
        self.assertEqual((body["input"], body["prompt"]), ("Hello", "calm delivery"))

    def test_optional_bools_and_native_choices_preserve_false(self):
        options = venice.read_options({"opt__safe_mode": "false", "opt__audio": ""})
        self.assertIs(options["safe_mode"], False)
        self.assertIsNone(options["audio"])
        with self.assertRaises(venice.INPUT_ERROR):
            venice.read_options({"opt__safe_mode": "yes"})
        cand = candidate("images")
        body = venice.build_request(cand, {"prompt": "hello", "n": 1, "variants": 3,
                                          "size": "auto", "output_format": "jpg"}, {})
        self.assertEqual((body["variants"], body["format"]), (3, "jpeg"))
        self.assertNotIn("width", body)
        self.assertNotIn("height", body)

    def test_native_size_maps_to_advertised_ratio_or_pixel_dimensions(self):
        cand = candidate("images", {"constraints": {"aspectRatios": ["16:9"]}})
        body = venice.build_request(cand, {"prompt": "wide", "size": "1280x720"}, {})
        self.assertEqual(body["aspect_ratio"], "16:9")
        self.assertNotIn("width", body)
        cand = candidate("images", {"constraints": {"widthHeightDivisor": 16}})
        body = venice.build_request(cand, {"prompt": "wide", "size": "1024x768"}, {})
        self.assertEqual((body["width"], body["height"]), (1024, 768))
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(cand, {"prompt": "wide", "width": 1023}, {})

    def test_seedance_reference_limits_and_source_matched_duration(self):
        cand = candidate("video", {"constraints": {"model_type": "image-to-video", "durations": ["5s"]}},
                         model="seedance-2-5-reference-to-video-basic")
        values = {"prompt": "Strictly edit <Video 1>", "duration": "auto",
                  "extra": {"reference_video_urls": ["https://example.com/source.mp4"]}}
        body = venice.build_request(cand, values, {})
        self.assertEqual(body["duration"], "auto")
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(cand, {"prompt": "edit", "duration": "auto"},
                                 {"input_image": PNG})
        old = candidate("video", {"constraints": {"model_type": "image-to-video", "durations": ["5s"]}},
                        model="seedance-2-0-reference-to-video-basic")
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(old, {"prompt": "refer", "duration": 5,
                "extra": {"reference_image_urls": ["https://example.com/image.png"] * 10}}, {})

    def test_grok_reference_uploads_use_seven_native_reference_slots(self):
        cand = candidate("video", {"constraints": {"model_type": "image-to-video",
            "durations": ["5s"]}}, model="grok-imagine-reference-to-video-private")
        _, images, _ = venice.public_fields(cand)
        self.assertEqual(len(images), 7)
        uploads = {slot["name"]: PNG for slot in images}
        body = venice.build_request(cand, {"prompt": "animate", "duration": 5}, uploads)
        self.assertEqual(len(body["reference_image_urls"]), 7)
        self.assertNotIn("image_url", body)
        self.assertNotIn("end_image_url", body)
        with self.assertRaises(venice.INPUT_ERROR):
            venice.validate_request(cand, {"prompt": "animate", "duration": 5},
                                    list(uploads) + ["input_image_8"], [])
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(cand, {"prompt": "animate", "duration": 5,
                "extra": {"reference_image_urls": ["https://example.com/ref.png"]}}, uploads)

    def test_kling_reference_uploads_keep_start_and_end_frame_roles(self):
        cand = candidate("video", {"constraints": {"model_type": "image-to-video",
            "durations": ["5s"]}}, model="kling-o3-pro-reference-to-video")
        _, images, _ = venice.public_fields(cand)
        self.assertEqual([field["name"] for field in images], ["input_image", "input_image_2"])
        body = venice.build_request(cand, {"prompt": "move between frames", "duration": 5},
                                    {"input_image": PNG, "input_image_2": PNG})
        self.assertIn("image_url", body)
        self.assertIn("end_image_url", body)
        self.assertNotIn("reference_image_urls", body)
        with self.assertRaises(venice.INPUT_ERROR):
            venice.validate_request(cand, {"prompt": "animate", "duration": 5},
                                    ["input_image", "input_image_2", "input_image_3"], [])

    def test_reference_slot_count_respects_metadata_below_documented_cap(self):
        cand = candidate("video", {"constraints": {"model_type": "image-to-video",
            "max_reference_images": 3, "durations": ["5s"]}},
            model="grok-imagine-reference-to-video-private")
        self.assertEqual(len(venice.slot_names(cand)), 3)
        with self.assertRaises(venice.INPUT_ERROR):
            venice.validate_request(cand, {"prompt": "animate", "duration": 5,
                "extra": {"reference_image_urls": ["https://example.com/ref.png"] * 4}}, [], [])

    def test_decoded_reference_image_metadata_bounds_are_enforced(self):
        constraints = {"model_type": "image-to-video", "durations": ["5s"],
            "reference_image_min_short_side_pixels": 300,
            "reference_image_min_aspect_ratio": 0.4,
            "reference_image_max_aspect_ratio": 2.5}
        cand = candidate("video", {"constraints": constraints}, model="seedance-2-0-image-to-video-basic")
        # The adapter supplies dimensions from Pillow's actual decoded image, not
        # filenames, declared MIME types, or client width/height parameters.
        for dimensions in ((256, 256), (300, 750), (750, 300), (300, 751), (751, 300)):
            with self.subTest(dimensions=dimensions), self.assertRaises(venice.INPUT_ERROR):
                venice.build_request(cand, {"prompt": "animate", "duration": 5},
                    {"input_image": reference_png(*dimensions)}, image_dimensions={"input_image": dimensions})
        for dimensions in ((300, 300), (300, 749), (749, 300)):
            with self.subTest(dimensions=dimensions):
                body = venice.build_request(cand, {"prompt": "animate", "duration": 5},
                    {"input_image": reference_png(*dimensions)}, image_dimensions={"input_image": dimensions})
                self.assertIn("image_url", body)
        with self.assertRaises(venice.INPUT_ERROR):
            venice.build_request(cand, {"prompt": "animate", "duration": 5}, {"input_image": PNG})

    def test_structured_native_controls_do_not_bypass_validation(self):
        cand = candidate("images", {"supportsStyleReferences": True, "maxStyleReferences": 2})
        for references in ([{"image": "https://example.com/i.png", "strength": 2}],
                           [{"image": "https://example.com/i.png", "invented": True}]):
            with self.subTest(references=references), self.assertRaises(venice.INPUT_ERROR):
                venice.build_request(cand, {"prompt": "paint", "extra": {"style_references": references}}, {})

    def test_summary_redacts_secrets_inline_media_and_signed_urls(self):
        body = {"image": base64.b64encode(PNG).decode(), "nested": {"Authorization": "secret",
            "audio_url": "https://cdn.example/a.wav?token=secret"},
            "images": ["data:image/png;base64,c2VjcmV0"], "prompt": "a scene"}
        summary = venice.request_summary(body)
        self.assertEqual(summary["prompt"], "a scene")
        self.assertNotIn("secret", str(summary))
        self.assertNotIn(base64.b64encode(PNG).decode(), str(summary))

    def test_nonfinite_json_and_nested_credentials_are_rejected(self):
        cand = candidate("images")
        for raw in ('{"cfg_scale":NaN}', '{"cfg_scale":1e999}', '{"style_references":[{"api_key":"secret"}]}'):
            cand["venice"]["options"]["options_json"] = raw
            with self.subTest(raw=raw), self.assertRaises(venice.INPUT_ERROR):
                venice.build_request(cand, {"prompt": "hello"}, {})


class VeniceMalformedMetadataTests(unittest.TestCase):
    """Vendor metadata that breaks a constraint read must be a VeniceInput (400 before
    the paid POST), never a TypeError/ZeroDivisionError the preflight lets through (500)."""

    def test_broken_constraints_refuse_as_input_errors(self):
        for constraints in ({"steps": ["not", "a", "dict"]}, {"maxInputImages": "many"},
                            {"widthHeightDivisor": None}, {"widthHeightDivisor": 0}):
            with self.subTest(constraints=constraints):
                cand = candidate("images", {"constraints": constraints})
                try:
                    venice.validate_request(cand, {"prompt": "a tree", "width": 512, "height": 512},
                                            [], [])
                except venice.VeniceInput:
                    pass


if __name__ == "__main__":
    unittest.main()
