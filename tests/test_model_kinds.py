"""Capabilities come from provider metadata; names only cover untyped legacy servers."""
import unittest

from model_kinds import generation_types, model_types


class ModelKinds(unittest.TestCase):
    def test_known_metadata_beats_name_fallback(self):
        cases = [
            ("kokoro", {"type": "chat"}, {"chat"}),
            ("opaque", {"type": "embedding"}, {"embedding"}),
            ("opaque", {"architecture": {"input_modalities": ["text", "image"],
                                         "output_modalities": ["text"]}}, {"chat"}),
            ("opaque", {"architecture": {"output_modalities": ["text", "image"]}},
             {"chat", "image"}),
            ("opaque", {"type": "chat", "architecture": {"output_modalities": ["image"]}},
             {"image"}),
            ("opaque", {"architecture": {"output_modalities": ["audio"]},
                        "capabilities": {"text_to_speech": True}}, {"tts"}),
            ("opaque", {"type": "asr", "architecture": {"output_modalities": ["text"]}}, {"asr"}),
            ("opaque", {"type": "embedding", "architecture": {"output_modalities": ["text"]}},
             {"embedding"}),
            ("opaque", {"capabilities": {"text_to_speech": True}}, {"tts"}),
            ("opaque", {"model_spec": {"capabilities": {"vision": True}}}, {"chat"}),
        ]
        for model, meta, expected in cases:
            with self.subTest(model=model, meta=meta):
                self.assertEqual(model_types(model, meta), expected)

    def test_untyped_legacy_model_names_have_conservative_defaults(self):
        cases = {"old-custom-2026": {"chat"}, "qwen3-vl-32b-q8": {"chat"},
                 "Kokoro-82M": {"tts"}, "Qwen3-TTS-12Hz-1.7B": {"tts"},
                 "flm/whisper-v3": {"asr"}, "nomic-embed-text-v1.5": {"embedding"},
                 "BAAI/bge-m3": {"embedding"}, "BAAI/bge-reranker-v2-m3": {"rerank"},
                 "black-forest-labs/FLUX.1-dev": {"image"}, "gpt-image-1": {"image"},
                 "audiophile-chat": {"chat"}, "reflection-chat": {"chat"}}
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(model_types(model), expected)

    def test_generic_text_output_does_not_override_known_non_chat_models(self):
        metadata = {"architecture": {"input_modalities": ["text"],
                                     "output_modalities": ["text"]},
                    "meta": {"llamaswap": {"type": "model"}}}
        for model, expected in (("bge-m3", {"embedding"}), ("kokoro", {"tts"}),
                                ("whisper-large-v3", {"asr"}), ("legacy-chat", {"chat"})):
            with self.subTest(model=model):
                self.assertEqual(model_types(model, metadata), expected)

    def test_generation_outputs_include_mesh_and_distinguish_audio_tasks(self):
        cases = [({"task": "text2mesh"}, {"3d"}),
                 ({"task": "image2mesh"}, {"3d"}),
                 ({"task": "text2audio"}, {"tts"}),
                 ({"task": "audio2text"}, {"asr"}),
                 ({"task": "audio2audio"}, {"audio"}),
                 ({"task": "img2vid"}, {"video"}),
                 ({"venice": {"endpoint": "music"}}, {"audio"}),
                 ({"venice": {"endpoint": "speech"}}, {"tts"}),
                 ({"nanogpt": {"endpoint": "3d"}}, {"3d"}),
                 ({"meshy": {"operation": "text-to-3d"}}, {"3d"})]
        for candidate, expected in cases:
            with self.subTest(candidate=candidate):
                self.assertEqual(generation_types(candidate), expected)
