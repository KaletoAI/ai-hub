"""Guard NanoGPT paid media requests: silently replacing pinned models, dropping
image counts/inputs or treating incomplete provider results as success buys the
wrong output and cannot safely be corrected by resubmitting the generation.
"""
import base64
import json
import unittest

import nanogpt
from tests.test_nanogpt_media_adapter import _png


class MediaRequests(unittest.TestCase):
    def candidate(self, endpoint="images", **options):
        return {"model": "provider/exact-model", "task": nanogpt.task_for(endpoint),
                nanogpt.KIND: {"endpoint": endpoint, "options": options}}

    def test_client_controls_override_defaults_without_changing_model(self):
        cand = self.candidate(quality="high", resolution="2k", options_json='{"guidance_scale":4,"seed":12}')
        result = nanogpt.build_request(cand, {"quality": "low", "size": "1024x1536", "cfg": 7.5,
                                                "model": "another-model", "prompt": "apple"}, {})
        self.assertEqual(result["model"], "provider/exact-model")
        self.assertEqual((result["quality"], result["size"], result["guidance_scale"]),
                         ("low", "1024x1536", 7.5))
        self.assertEqual(result["seed"], 12)
        self.assertEqual(cand[nanogpt.KIND]["options"]["quality"], "high")

    def test_multiple_outputs_and_reference_bytes_are_preserved(self):
        raw = _png()
        result = nanogpt.build_request(self.candidate(), {"prompt": "apple", "n": 2},
                                      {"input_image": raw, "input_image_2": raw})
        self.assertEqual(result["n"], 2)
        self.assertEqual(len(result["imageDataUrls"]), 2)
        self.assertEqual(base64.b64decode(result["imageDataUrls"][0].split(",", 1)[1]), raw)

    def test_explicit_native_control_beats_short_alias(self):
        result = nanogpt.build_request(self.candidate(), {"cfg": 2, "guidance_scale": 8,
                                                          "steps": 10, "num_inference_steps": 30}, {})
        self.assertEqual((result["guidance_scale"], result["num_inference_steps"]), (8, 30))

    def test_video_seconds_and_media_inputs_are_not_workflow_frames(self):
        result = nanogpt.build_request(self.candidate("video"), {"prompt": "sunrise", "seconds": 8},
                                      {"input_image": _png()},
                                      {"input_video": ("source.mp4", b"video"),
                                       "input_audio": ("source.wav", b"audio")})
        self.assertEqual(result["seconds"], "8")
        self.assertNotIn("frames", result)
        self.assertTrue(result["imageDataUrl"].startswith("data:image/"))
        self.assertEqual(base64.b64decode(result["videoDataUrl"].split(",", 1)[1]), b"video")
        self.assertEqual(base64.b64decode(result["audioDataUrl"].split(",", 1)[1]), b"audio")

    def test_text_and_image_mesh_inputs_use_shared_video_transport(self):
        text = nanogpt.build_request(self.candidate("3d", options_json='{"face_count":500000,"enable_pbr":false}'),
                                     {"prompt": "ceramic lamp"}, {})
        self.assertEqual((text["prompt"], text["face_count"], text["enable_pbr"]),
                         ("ceramic lamp", 500000, False))
        image = nanogpt.build_request(self.candidate("3d"), {}, {"input_image": _png()})
        self.assertIn("imageDataUrl", image)
        self.assertNotIn("imageDataUrls", image)

    def test_bad_options_and_counts_refuse_instead_of_silently_normalizing(self):
        for options in ({"options_json": "[]"}, {"options_json": '{"api_key":"secret"}'},
                        {"options_json": '{"model":"replacement"}'}, {"options_json": '{"cfg":NaN}'},
                        {"options_json": '{"controls":{"scale":1e999}}'},
                        {"n": 1.5}, {"n": True}, {"n": 0}, {"cfg": float("inf")}):
            with self.subTest(options=options):
                with self.assertRaises(nanogpt.NanoGPTInput):
                    nanogpt.build_request(self.candidate(**options), {}, {})
        with self.assertRaises(nanogpt.NanoGPTInput):
            nanogpt.build_request(self.candidate("video"), {"n": 2}, {})

    def test_form_keeps_zero_seed_and_refuses_fractional_count(self):
        options = nanogpt.read_options({"opt__seed": "0", "opt__n": "2", "opt__cfg": "7.5"})
        self.assertEqual((options["seed"], options["n"], options["cfg"]), (0, 2, 7.5))
        with self.assertRaises(nanogpt.NanoGPTInput):
            nanogpt.read_options({"opt__n": "2.5"})

    def test_summary_does_not_retain_uploaded_media(self):
        body = nanogpt.build_request(self.candidate(), {}, {"input_image": _png()})
        summary = nanogpt.request_summary(body)
        self.assertNotIn("data:image", str(summary))
        self.assertIn("data:image", str(body))


class CatalogSchemas(unittest.TestCase):
    """Catalog consumers must not offer/refuse unsupported inputs or buy excess outputs."""

    def qwen(self):
        return nanogpt.candidate_for("nano", "qwen-image-2.1-turbo", "images", {
            "architecture": {"input_modalities": ["text", "image"]},
            "capabilities": {"image_generation": True, "image_to_image": True},
            "supported_parameters": {
                "resolutions": ["1k", "2k"],
                "aspect_ratio": ["1:1", "3:2", "16:9"],
                "max_input_images": 3, "max_output_images": 4, "max_images": 4,
            },
        })

    def test_direct_catalog_candidate_needs_no_alias_and_keeps_model_pinned(self):
        cand = self.qwen()
        body = nanogpt.build_request(cand, {"prompt": "pear", "model": "replacement"}, {})
        self.assertEqual((body["model"], body["n"]), ("qwen-image-2.1-turbo", 1))
        self.assertEqual((cand["backend"], cand["task"]), ("nano", "text2img"))
        with self.assertRaises(nanogpt.NanoGPTInput):
            nanogpt.candidate_for("nano", "text-model", "text")

    def test_qwen_flat_metadata_exposes_three_references_choices_and_count_bounds(self):
        params, images, files = nanogpt.public_fields(self.qwen())
        fields = {p["name"]: p for p in params}
        self.assertEqual(set(fields), {"prompt", "resolution", "aspect_ratio", "n"})
        self.assertEqual(fields["resolution"]["choices"], ["1k", "2k"])
        self.assertEqual(fields["aspect_ratio"]["choices"], ["1:1", "3:2", "16:9"])
        self.assertEqual((fields["n"]["type"], fields["n"]["min"], fields["n"]["max"]), ("int", 1, 4))
        self.assertEqual([p["name"] for p in images], ["input_image", "input_image_2", "input_image_3"])
        self.assertEqual(files, [])

    def test_known_text_only_model_has_no_reference_slots_or_generic_controls(self):
        cand = nanogpt.candidate_for("nano", "text-only", "images", {
            "architecture": {"input_modalities": ["text"]},
            "capabilities": {"image_to_image": False},
            "supported_parameters": {},
        })
        params, images, files = nanogpt.public_fields(cand)
        self.assertEqual([p["name"] for p in params], ["prompt"])
        self.assertEqual((images, files, nanogpt.slot_names(cand)), ([], [], []))
        self.assertEqual(nanogpt.build_request(cand, {"prompt": "pear"}, {})["prompt"], "pear")
        with self.assertRaises(nanogpt.NanoGPTInput):
            nanogpt.build_request(cand, {}, {"input_image": _png()})

    def test_unknown_metadata_retains_handwritten_alias_schema(self):
        cand = nanogpt.candidate_for("nano", "old-image-model", "images")
        params, images, files = nanogpt.public_fields(cand)
        self.assertIn("negative_prompt", [p["name"] for p in params])
        self.assertIn("cfg", [p["name"] for p in params])
        self.assertEqual(len(images), 16)
        self.assertEqual(files, [])
        known = nanogpt.candidate_for("nano", "unknown-capabilities", "images", {})
        self.assertEqual(nanogpt.slot_names(known), [])

    def test_explicit_choices_and_output_limits_refuse_before_submission(self):
        for values in ({"resolution": "4k"}, {"aspect_ratio": "7:1"}, {"n": 5},
                       {"n": 0}, {"n": 1.5}, {"n": True}, {"resolution": "auto"}):
            with self.subTest(values=values), self.assertRaises(nanogpt.NanoGPTInput):
                nanogpt.validate_request(self.qwen(), values, [], [])
        with self.assertRaises(nanogpt.NanoGPTInput):
            nanogpt.build_request(self.qwen(), {}, {"input_image_4": _png()})
        body = nanogpt.build_request(self.qwen(), {"n": 4, "resolution": "2k"},
                                    {name: _png() for name in ("input_image", "input_image_2", "input_image_3")})
        self.assertEqual((body["n"], body["size"], len(body["imageDataUrls"])), (4, "2k", 3))

    def test_auto_and_explicit_native_overrides_preserve_openai_compatibility(self):
        cand = self.qwen()
        cand[nanogpt.KIND]["options"].update(resolution="2k")
        body = nanogpt.build_request(cand, {"size": "auto", "n": 2, "cfg": 3, "guidance_scale": 7}, {})
        self.assertEqual((body["size"], body["n"], body["guidance_scale"]), ("auto", 2, 7))
        body = nanogpt.build_request(cand, {"size": "1k", "resolution": "invalid"}, {})
        self.assertEqual(body["size"], "1k")
        # Unknown vendor-native controls remain available, not guessed unsupported.
        self.assertEqual(nanogpt.build_request(cand, {"extra": {"new_native_control": 3}}, {})["new_native_control"], 3)

    def test_documented_enum_range_and_quality_fields_validate_native_values(self):
        cand = nanogpt.candidate_for("nano", "bounded", "images", {
            "supported_parameters": {
                "quality": ["low", "high"],
                "n": {"type": "integer", "minimum": 1, "maximum": 2},
                "parameters": {
                    "seed": {"type": "integer", "min": -1, "max": 10},
                    "guidance_scale": {"type": "number", "range": {"min": 1, "max": 8}},
                    "output_format": {"type": "string", "enum": ["png", "webp"]},
                },
            },
        })
        fields = {p["name"]: p for p in nanogpt.public_fields(cand)[0]}
        self.assertEqual(fields["quality"]["choices"], ["low", "high"])
        self.assertEqual((fields["guidance_scale"]["min"], fields["guidance_scale"]["max"]), (1, 8))
        self.assertEqual((fields["n"]["min"], fields["n"]["max"]), (1, 2))
        for values in ({"quality": "medium"}, {"seed": 11}, {"cfg": 9}, {"output_format": "jpeg"}, {"n": 3}):
            with self.subTest(values=values), self.assertRaises(nanogpt.NanoGPTInput):
                nanogpt.build_request(cand, values, {})
        body = nanogpt.build_request(cand, {"quality": "high", "seed": 0, "cfg": 2, "n": 2}, {})
        self.assertEqual((body["seed"], body["guidance_scale"], body["n"]), (0, 2, 2))

    def test_structured_native_controls_are_not_rejected_as_scalar_text(self):
        """A catalog LoRA array must remain usable via native params or alias defaults."""
        loras = [{"path": "https://example.com/style.safetensors", "scale": 1}]
        cand = nanogpt.candidate_for("nano", "minimax-h3/text-to-image", "images", {
            "supported_parameters": {"loras": {
                "max_items": 3, "item": {"path": "HTTPS URL", "scale": "number"},
                "url_fields": ["lora_url_1", "lora_url_2", "lora_url_3"],
            }},
        })
        cand[nanogpt.KIND]["options"] = {"options_json": json.dumps({"loras": loras})}
        self.assertEqual(nanogpt.build_request(cand, {}, {})["loras"], loras)
        self.assertEqual(nanogpt.build_request(cand, {"loras": []}, {})["loras"], [])
        self.assertNotIn("loras", [p["name"] for p in nanogpt.public_fields(cand)[0]])

    def test_video_and_mesh_fields_follow_actual_capabilities_and_parameter_descriptors(self):
        video = nanogpt.candidate_for("nano", "video-editor", "video", {
            "architecture": {"input_modalities": ["video", "audio"]},
            "capabilities": {"text_to_video": False, "image_to_video": False,
                             "video_to_video": True, "audio_input": True},
            "supported_parameters": {"parameters": {
                "duration": {"type": "select", "options": [{"value": "5"}, {"value": "10"}]},
                "generate_audio": {"type": "switch", "default": False},
            }},
        })
        params, images, files = nanogpt.public_fields(video)
        self.assertEqual([p["name"] for p in params], ["duration", "generate_audio"])
        self.assertEqual(params[0]["choices"], ["5", "10"])
        self.assertEqual(params[1]["type"], "bool")
        self.assertEqual(images, [])
        self.assertEqual([p["name"] for p in files], ["input_video", "input_audio"])
        self.assertEqual(nanogpt.build_request(video, {"duration": 5, "generate_audio": True}, {})["duration"], "5")
        with self.assertRaises(nanogpt.NanoGPTInput):
            nanogpt.build_request(video, {"duration": 6}, {})
        with self.assertRaises(nanogpt.NanoGPTInput):
            nanogpt.build_request(video, {"generate_audio": "not-a-bool"}, {})
        mesh = nanogpt.candidate_for("nano", "image-mesh", "3d", {
            "architecture": {"input_modalities": ["image"]},
            "capabilities": {"text_to_3d": False, "image_to_3d": True},
            "supported_parameters": {"parameters": {
                "target_face_count": {"type": "number", "min": 10000, "max": 1000000,
                                      "step": 10000, "default": 200000},
                "remove_background": {"type": "switch", "default": True},
            }},
        })
        params, images, files = nanogpt.public_fields(mesh)
        self.assertEqual([p["name"] for p in params], ["target_face_count", "remove_background"])
        self.assertEqual((params[0]["min"], params[0]["max"], params[0]["step"]), (10000, 1000000, 10000))
        self.assertEqual([p["name"] for p in images], ["input_image"])
        self.assertEqual(files, [])
        body = nanogpt.build_request(mesh, {"target_face_count": 40000}, {"input_image": _png()})
        self.assertEqual(body["target_face_count"], 40000)
        with self.assertRaises(nanogpt.NanoGPTInput):
            nanogpt.build_request(mesh, {"target_face_count": 9999}, {})

    def test_current_metadata_overrides_candidate_snapshot_in_preflight(self):
        cand = self.qwen()
        with self.assertRaises(nanogpt.NanoGPTInput):
            nanogpt.validate_request(cand, {"n": 4}, [], [],
                                     {"supported_parameters": {"max_output_images": 2}})
        with self.assertRaises(nanogpt.NanoGPTInput):
            nanogpt.validate_request(cand, {}, ["input_image"], [],
                                     {"capabilities": {"image_to_image": False}})


    def test_url_inputs_obey_reference_and_file_capabilities(self):
        cand = self.qwen()
        with self.assertRaises(nanogpt.NanoGPTInput):
            nanogpt.build_request(cand, {"imageDataUrls": ["https://cdn/a.png"] * 4}, {})
        cand = nanogpt.candidate_for("nano", "text-video", "video", {
            "architecture": {"input_modalities": ["text"]},
            "capabilities": {"text_to_video": True, "image_to_video": False,
                             "video_to_video": False, "audio_input": False},
            "supported_parameters": {},
        })
        self.assertEqual([p["name"] for p in nanogpt.public_fields(cand)[0]], ["prompt"])
        self.assertEqual(nanogpt.public_fields(cand)[1:], ([], []))
        for key in ("imageUrl", "videoDataUrl", "audioUrl"):
            with self.subTest(key=key), self.assertRaises(nanogpt.NanoGPTInput):
                nanogpt.build_request(cand, {key: "https://cdn/input"}, {})

    def test_short_control_metadata_bounds_apply_after_native_alias_conversion(self):
        cand = nanogpt.candidate_for("nano", "bounded-native", "images", {
            "supported_parameters": {
                "cfg": {"range": [1, 8]},
                "steps": {"type": "integer", "min": 1, "max": 30},
            },
        })
        fields = {p["name"]: p for p in nanogpt.public_fields(cand)[0]}
        self.assertEqual((fields["cfg"]["type"], fields["cfg"]["min"], fields["cfg"]["max"]), ("float", 1, 8))
        for values in ({"guidance_scale": 9}, {"num_inference_steps": 31}):
            with self.subTest(values=values), self.assertRaises(nanogpt.NanoGPTInput):
                nanogpt.build_request(cand, values, {})
        body = nanogpt.build_request(cand, {"cfg": 20, "guidance_scale": 3,
                                         "steps": 40, "num_inference_steps": 10}, {})
        self.assertEqual((body["guidance_scale"], body["num_inference_steps"]), (3, 10))

    def test_nested_client_native_controls_remain_finite_json(self):
        for value in (float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(nanogpt.NanoGPTInput):
                nanogpt.build_request(self.qwen(), {"extra": {"native": [{"scale": value}]}}, {})


class MediaStates(unittest.TestCase):
    def test_both_documented_video_envelopes_preserve_downloads(self):
        nested = {"data": {"status": "COMPLETED", "output": {"video": {"url": "https://cdn.example/video.mp4"}}}}
        flat = {"status": "completed", "videoUrl": "https://cdn.example/video.mp4"}
        for payload in (nested, flat):
            state = nanogpt.parse_task(payload, endpoint="video")
            self.assertEqual(state.status, "COMPLETED")
            self.assertEqual(state.downloads, [("video.mp4", "https://cdn.example/video.mp4")])
            self.assertIsNone(state.error)

    def test_progress_failure_and_empty_completed_results_are_distinct(self):
        state = nanogpt.parse_task({"data": {"status": "IN_PROGRESS", "progress": "37%"}}, endpoint="video")
        self.assertEqual((state.status, state.progress, state.error), ("IN_PROGRESS", 37, None))
        failure = nanogpt.parse_task({"data": {"status": "FAILED", "error": "content policy"}}, endpoint="video")
        self.assertEqual(failure.error, "content policy")
        incomplete = nanogpt.parse_task({"data": {"status": "COMPLETED", "output": {}}}, endpoint="video")
        self.assertIsNotNone(incomplete.error)
        self.assertFalse(incomplete.retryable)
        unknown = nanogpt.parse_task({"status": "unrecognized"}, endpoint="video")
        self.assertIsNotNone(unknown.error)

    def test_invalid_image_row_cannot_be_dropped_from_paid_output(self):
        for payload in ({"data": []}, {"data": [{"url": "https://cdn.example/a.png"}, {}]},
                        {"data": [{"url": "file:///etc/passwd"}]}):
            with self.subTest(payload=payload):
                with self.assertRaises(nanogpt.NanoGPTInput):
                    nanogpt.image_outputs(payload)

    def test_midjourney_completion_maps_real_image_and_failure(self):
        state = nanogpt.parse_task({"status": "SUCCESS", "imageUrl": "https://cdn.example/a.png"}, endpoint="images")
        self.assertEqual(state.downloads, [("a.png", "https://cdn.example/a.png")])
        self.assertEqual(state.status, "COMPLETED")
        failed = nanogpt.parse_task({"status": "FAILED", "failReason": "rejected"}, endpoint="images")
        self.assertEqual(failed.error, "rejected")


if __name__ == "__main__":
    unittest.main()
