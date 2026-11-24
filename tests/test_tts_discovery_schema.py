"""TTS discovery (`/v1/models`), schema enrichment (`/v1/generations/{alias}/schema`)
and reference-audio upload / voice-override behaviour for ComfyUI and cloud TTS.

Why this fails SILENTLY:
- Legacy ComfyUI audio aliases all share `task: "text2audio"`, so music generators like
  `ACE-Step-1.5-XL` showed up under `owned_by: "ai-hub (tts)"` and `?type=tts` without
  a `task` or `local` discriminator.
- ComfyUI `public_fields` ignored `/object_info` combo choices and numeric bounds, so
  TTS aliases exposed only `default` (no `voice` or `language` `choices`, no `min`/`max`)
  and omitted `files[].accept` and the standardized `tts` block — while `LoadAudio`
  combos must NEVER leak `ComfyUI/input/` filenames into `params[].choices`.
- `ComfyUIAdapter._apply_file_params` always prepended `comfy_input_dir` and failed
  without it, breaking `LoadAudio` nodes (which validate bare filenames in `input/`),
  and did not neutralize preset `voice` when `files.reference_audio` was uploaded.

Run: venv/bin/python -m unittest tests.test_tts_discovery_schema -v
"""
import asyncio
import base64
import copy
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)

from tests.test_ui_form_validation import _Fixture  # noqa: E402
import adapters  # noqa: E402
import lulabs  # noqa: E402
import main  # noqa: E402
import model_kinds  # noqa: E402
import store  # noqa: E402
import venice  # noqa: E402
import wiro  # noqa: E402


WAV_BYTES = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
WAV_DATA_URI = "data:audio/wav;base64," + base64.b64encode(WAV_BYTES).decode()

ZONOS_WF = {
    "1": {"class_type": "LoadAudio", "inputs": {"audio": ""}},
    "2": {"class_type": "ZonosTTS", "inputs": {
        "text": "Hallo", "language": "de", "voice": "Wolfgang",
        "speaking_rate": 0.0, "reference_audio": ["1", 0],
    }},
    "3": {"class_type": "SaveAudio", "inputs": {"audio": ["2", 0], "filename_prefix": "tts"}},
}
ZONOS_MAP = {
    "text": {"node": "2", "field": "text"},
    "language": {"node": "2", "field": "language"},
    "voice": {"node": "2", "field": "voice"},
    "speaking_rate": {"node": "2", "field": "speaking_rate"},
    "reference_audio": {"node": "1", "field": "audio"},
}
ZONOS_OI = {
    "LoadAudio": {
        "input": {"required": {"audio": [["gw_old_upload.wav", "sample.wav"], {"audio_upload": True}]}},
        "output": ["AUDIO"],
    },
    "ZonosTTS": {
        "input": {
            "required": {
                "text": ["STRING", {"multiline": True}],
                "language": [["de", "en", "fr", "es", "ja"]],
                "voice": [["None", "Wolfgang", "Serena"]],
                "speaking_rate": ["FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}],
            },
            "optional": {
                "reference_audio": ["AUDIO"],
            },
        },
        "output": ["AUDIO"],
    },
    "SaveAudio": {"input": {"required": {"audio": ["AUDIO"]}}, "output": []},
}

MING_WF = {
    "1": {"class_type": "LoadAudio", "inputs": {"audio": ""}},
    "2": {"class_type": "MingTTSNode", "inputs": {
        "text": "Hallo", "language": "German", "reference_text": "",
        "reference_audio": ["1", 0],
    }},
    "3": {"class_type": "SaveAudio", "inputs": {"audio": ["2", 0], "filename_prefix": "ming"}},
}
MING_MAP = {
    "text": {"node": "2", "field": "text"},
    "language": {"node": "2", "field": "language"},
    "reference_text": {"node": "2", "field": "reference_text"},
    "reference_audio": {"node": "1", "field": "audio", "required": True},
}
MING_OI = {
    "LoadAudio": ZONOS_OI["LoadAudio"],
    "MingTTSNode": {
        "input": {
            "required": {
                "text": ["STRING", {}],
                "language": ["COMBO", {"options": ["German", "English", "Chinese"]}],
                "reference_text": ["STRING", {}],
                "reference_audio": ["AUDIO"],
            },
        },
        "output": ["AUDIO"],
    },
    "SaveAudio": ZONOS_OI["SaveAudio"],
}

ACESTEP_WF = {
    "1": {"class_type": "TextEncodeAceStepAudio", "inputs": {"tags": "rock", "lyrics": "la la"}},
    "2": {"class_type": "SaveAudio", "inputs": {"audio": ["1", 0]}},
}
ACESTEP_MAP = {
    "tags": {"node": "1", "field": "tags"},
    "lyrics": {"node": "1", "field": "lyrics"},
}


class _TtsFixture(_Fixture):
    def setUp(self):
        super().setUp()
        gpu = {"name": "gpu-3090", "type": "comfyui", "url": "http://127.0.0.1:1"}
        rp = {"name": "rp-tts", "type": "runpod", "url": "https://api.runpod.ai/v2/ep1"}
        ven = {"name": "venice.ai", "type": "venice", "url": venice.URL, "paid": True}
        wir = {"name": "wiro", "type": "wiro", "url": wiro.URL, "paid": True}
        lu = {"name": "lulabs", "type": "lulabs", "url": lulabs.URL, "paid": True}
        loc_llm = {"name": "localai", "type": "openai", "url": "http://127.0.0.1:2", "local": True}
        cld_llm = {"name": "cloud-oai", "type": "openai", "url": "https://api.example.com", "paid": True}

        ven_cats = {
            "speech": {"tts-qwen3-1.7b": {"model_spec": {
                "voices": ["alloy", "nova"], "voice_cloning": {"max_duration_s": 12},
                "supportsPromptParam": True, "default_format": "mp3",
            }}},
            "music": {"ace-step-cloud": {"model_spec": {"supports_lyrics": True}},
                      "elevenlabs-sfx": {"model_spec": {}}},
            "transcription": {"whisper-v3": {"model_spec": {}}},
        }
        wiro_cats = {
            "speech": {"xiaomi/mimo-v2-tts": {"parameters": [{"items": [
                {"id": "text", "type": "textarea", "required": True},
                {"id": "instruction", "type": "text"},
                {"id": "speed", "type": "float", "defaultvalue": "1.0", "minvalue": "0.5", "maxvalue": "2.0"},
                {"id": "language", "type": "select", "options": [
                    {"value": "de-DE"}, {"value": "en-US"}]},
            ]}]}},
            "audio": {"suno/music-v1": {"categories": ["music-generation"], "parameters": []}},
            "asr": {"openai/whisper-large": {"categories": ["speech-to-text"], "parameters": []}},
        }
        lu_cats = {
            "speech": {"lu-tts-v1": {"_ops": ["tts"]}},
            "music": {"lu-music-v1": {"_ops": ["music"]}},
        }

        store.upsert("ZONOS3-Mini", [{"backend": "gpu-3090", "task": "text2audio",
                                      "workflow_json": ZONOS_WF, "mapping": ZONOS_MAP}])
        store.upsert("ming-3.0-tts-0.5b", [{"backend": "gpu-3090", "task": "text2audio",
                                            "workflow_json": MING_WF, "mapping": MING_MAP}])
        store.upsert("ACE-Step-1.5-XL", [{"backend": "gpu-3090", "task": "text2audio",
                                          "workflow_json": ACESTEP_WF, "mapping": ACESTEP_MAP}])
        store.upsert("Qwen-Image", [{"backend": "gpu-3090", "task": "text2img",
                                     "workflow_json": {}, "mapping": {}}])
        store.upsert("Cloud-Runpod-TTS", [{"backend": "rp-tts", "task": "tts",
                                           "workflow_json": ZONOS_WF, "mapping": ZONOS_MAP}])

        comfy_ad = adapters.ComfyUIAdapter(gpu, main.adapter_ctx)
        raw_oi = json.dumps({**ZONOS_OI, **MING_OI}).encode()
        parsed = adapters._parse_object_info(raw_oi)
        comfy_ad._oi_cache = (b"test", parsed)
        comfy_ad._node_types, _, _, comfy_ad._field_specs = parsed

        state = {
            "backends": [gpu, rp, ven, wir, lu, loc_llm, cld_llm],
            "config_backends": [],
            "backend_adapters": {
                "comfyui:gpu-3090": comfy_ad,
                "venice:venice.ai": SimpleNamespace(model_catalogs=ven_cats),
                "wiro:wiro": SimpleNamespace(model_catalogs=wiro_cats),
                "lulabs:lulabs": SimpleNamespace(model_catalogs=lu_cats),
            },
            "backend_models": {
                "openai:localai": {"kokoro-local"},
                "openai:cloud-oai": {"tts-1"},
            },
            "backend_model_metadata": {
                "openai:localai": {"kokoro-local": {"type": "tts"}},
                "openai:cloud-oai": {"tts-1": {"type": "tts"}},
            },
            "backend_healthy": {
                "comfyui:gpu-3090": True, "runpod:rp-tts": True,
                "venice:venice.ai": True, "wiro:wiro": True, "lulabs:lulabs": True,
                "openai:localai": True, "openai:cloud-oai": True,
            },
            "backend_pricing": {}, "backend_loras": {}, "backend_context": {},
            "backend_running": {}, "virtual_models": {}, "image_models": {},
            "model_prefix": True,
            "_backend_names": {b["name"] for b in (gpu, rp, ven, wir, lu, loc_llm, cld_llm)},
            "_llm_backends": [], "_gen_backends": [], "_route_index": {},
        }
        for name, value in state.items():
            p = patch.object(main, name, value)
            p.start()
            self.addCleanup(p.stop)
        main.rebuild_route_index()


class ModelsTaskAndLocal(_TtsFixture):

    def _models(self, query=""):
        r = self.c.get("/v1/models" + query)
        self.assertEqual(r.status_code, 200, r.text)
        return {row["id"]: row for row in r.json()["data"]}

    def test_models_carry_task_and_local_flags_and_classify_acestep_as_music(self):
        rows = self._models()
        self.assertEqual((rows["ZONOS3-Mini"]["task"], rows["ZONOS3-Mini"]["local"],
                          rows["ZONOS3-Mini"]["owned_by"]), ("tts", True, "ai-hub (tts)"))
        self.assertEqual((rows["ming-3.0-tts-0.5b"]["task"], rows["ming-3.0-tts-0.5b"]["local"],
                          rows["ming-3.0-tts-0.5b"]["owned_by"]), ("tts", True, "ai-hub (tts)"))
        self.assertEqual((rows["ACE-Step-1.5-XL"]["task"], rows["ACE-Step-1.5-XL"]["local"],
                          rows["ACE-Step-1.5-XL"]["owned_by"]), ("music", True, "ai-hub (audio)"))
        self.assertNotIn("task", rows["Qwen-Image"])
        self.assertIs(rows["Qwen-Image"]["local"], True)
        self.assertEqual((rows["Cloud-Runpod-TTS"]["task"], rows["Cloud-Runpod-TTS"]["local"]),
                         ("tts", False))

        # Cloud media catalog entries
        self.assertEqual((rows["venice.ai/tts-qwen3-1.7b"]["task"],
                          rows["venice.ai/tts-qwen3-1.7b"]["local"]), ("tts", False))
        self.assertEqual((rows["venice.ai/ace-step-cloud"]["task"],
                          rows["venice.ai/ace-step-cloud"]["local"]), ("music", False))
        self.assertEqual((rows["venice.ai/elevenlabs-sfx"]["task"],
                          rows["venice.ai/elevenlabs-sfx"]["local"]), ("sfx", False))
        self.assertEqual((rows["venice.ai/whisper-v3"]["task"],
                          rows["venice.ai/whisper-v3"]["local"]), ("stt", False))
        self.assertEqual((rows["wiro/xiaomi/mimo-v2-tts"]["task"],
                          rows["wiro/xiaomi/mimo-v2-tts"]["local"]), ("tts", False))
        self.assertEqual((rows["wiro/suno/music-v1"]["task"],
                          rows["wiro/suno/music-v1"]["local"]), ("music", False))
        self.assertEqual((rows["wiro/openai/whisper-large"]["task"],
                          rows["wiro/openai/whisper-large"]["local"]), ("stt", False))
        self.assertEqual((rows["lulabs/lu-tts-v1"]["task"],
                          rows["lulabs/lu-tts-v1"]["local"]), ("tts", False))
        self.assertEqual((rows["lulabs/lu-music-v1"]["task"],
                          rows["lulabs/lu-music-v1"]["local"]), ("music", False))

        # LLM-side TTS entries
        self.assertEqual((rows["localai/kokoro-local"]["task"],
                          rows["localai/kokoro-local"]["local"]), ("tts", True))
        self.assertEqual((rows["cloud-oai/tts-1"]["task"],
                          rows["cloud-oai/tts-1"]["local"]), ("tts", False))

        # Type filter excludes ACE-Step-1.5-XL from ?type=tts and keeps it in ?type=audio
        tts_ids = set(self._models("?type=tts"))
        audio_ids = set(self._models("?type=audio"))
        self.assertIn("ZONOS3-Mini", tts_ids)
        self.assertIn("ming-3.0-tts-0.5b", tts_ids)
        self.assertNotIn("ACE-Step-1.5-XL", tts_ids)
        self.assertIn("ACE-Step-1.5-XL", audio_ids)

    def test_single_model_lookup_mirrors_task_and_local(self):
        for mid, want_task, want_local in (
            ("ZONOS3-Mini", "tts", True),
            ("ACE-Step-1.5-XL", "music", True),
            ("venice.ai/tts-qwen3-1.7b", "tts", False),
            ("localai/kokoro-local", "tts", True),
        ):
            r = self.c.get(f"/v1/models/{mid}")
            self.assertEqual(r.status_code, 200, (mid, r.text))
            d = r.json()
            self.assertEqual((d.get("task"), d.get("local")), (want_task, want_local), mid)


class SchemaChoicesAndTtsBlock(_TtsFixture):
    def _schema(self, alias):
        r = self.c.get(f"/v1/generations/{alias}/schema")
        self.assertEqual(r.status_code, 200, (alias, r.text))
        return r.json()

    def test_zonos_schema_extracts_voice_choices_accept_and_tts_block(self):
        s = self._schema("ZONOS3-Mini")
        params = {p["name"]: p for p in s["params"]}
        files = {f["name"]: f for f in s["files"]}

        self.assertEqual(params["voice"]["choices"], ["None", "Wolfgang", "Serena"])
        self.assertEqual(params["language"]["choices"], ["de", "en", "fr", "es", "ja"])
        self.assertEqual(params["speaking_rate"]["min"], -1.0)
        self.assertEqual(params["speaking_rate"]["max"], 1.0)
        # LoadAudio combo must NEVER leak ComfyUI/input filenames into params[].choices
        self.assertNotIn("choices", params["reference_audio"])
        self.assertEqual(files["reference_audio"]["accept"], ["wav", "flac", "mp3", "ogg"])
        self.assertFalse(files["reference_audio"]["required"])

        tts = s["tts"]
        self.assertEqual(tts["task"], "tts")
        self.assertEqual(tts["text_param"], "text")
        self.assertEqual(tts["language_param"], "language")
        self.assertEqual(tts["language_format"], "iso639_1")
        self.assertEqual(tts["language_choices"], ["de", "en", "fr", "es", "ja"])
        self.assertEqual(tts["voice_param"], "voice")
        self.assertEqual([v["id"] for v in tts["voices"]], ["Wolfgang", "Serena"])
        self.assertEqual(tts["clone"], {
            "supported": True,
            "required": False,
            "file_slot": "reference_audio",
            "accept": ["wav", "flac", "mp3", "ogg"],
            "max_duration_s": 15,
            "reference_text_param": None,
            "overrides_voice": True,
        })
        self.assertEqual(tts["controls"], {
            "speed": {"param": "speaking_rate", "default": 0.0, "min": -1.0, "max": 1.0, "neutral": 0.0},
            "style": None,
        })
        self.assertEqual(tts["preferred_mode"], "sync")
        self.assertEqual(tts["output_format"], "wav")

    def test_primitive_string_inherits_downstream_choices(self):
        wf = copy.deepcopy(ZONOS_WF)
        wf["10"] = {"class_type": "PrimitiveString", "inputs": {"value": "Wolfgang"}}
        wf["2"]["inputs"]["voice"] = ["10", 0]
        mp = {**ZONOS_MAP, "voice": {"node": "10", "field": "value", "label": "voice"}}
        store.upsert("ZONOS-Prim", [{"backend": "gpu-3090", "task": "tts",
                                     "workflow_json": wf, "mapping": mp}])
        s = self._schema("ZONOS-Prim")
        params = {p["name"]: p for p in s["params"]}
        self.assertEqual(params["voice"]["choices"], ["None", "Wolfgang", "Serena"])
        self.assertEqual([v["id"] for v in s["tts"]["voices"]], ["Wolfgang", "Serena"])

    def test_ming_schema_has_language_choices_accept_and_clone_required(self):
        s = self._schema("ming-3.0-tts-0.5b")
        params = {p["name"]: p for p in s["params"]}
        files = {f["name"]: f for f in s["files"]}
        self.assertEqual(params["language"]["choices"], ["German", "English", "Chinese"])
        self.assertEqual(files["reference_audio"]["accept"], ["wav", "flac", "mp3", "ogg"])
        self.assertTrue(files["reference_audio"]["required"])

        tts = s["tts"]
        self.assertEqual(tts["task"], "tts")
        self.assertEqual(tts["text_param"], "text")
        self.assertEqual(tts["language_param"], "language")
        self.assertEqual(tts["language_format"], "english_name")
        self.assertEqual(tts["language_choices"], ["German", "English", "Chinese"])
        self.assertIsNone(tts["voice_param"])
        self.assertEqual(tts["voices"], [])
        self.assertEqual(tts["clone"], {
            "supported": True,
            "required": True,
            "file_slot": "reference_audio",
            "accept": ["wav", "flac", "mp3", "ogg"],
            "max_duration_s": 10,
            "reference_text_param": "reference_text",
            "overrides_voice": True,
        })
        self.assertEqual(tts["controls"], {"speed": None, "style": None})
        self.assertEqual(tts["preferred_mode"], "sync")
        self.assertEqual(tts["output_format"], "wav")

    def test_ming_language_fallback_when_node_uses_plain_string(self):
        # Clear cached field specs for MingTTSNode (plain STRING language input).
        main.backend_adapters["comfyui:gpu-3090"]._field_specs.pop("MingTTSNode", None)
        s = self._schema("ming-3.0-tts-0.5b")
        params = {p["name"]: p for p in s["params"]}
        self.assertIn("German", params["language"]["choices"])
        self.assertIn("English", params["language"]["choices"])
        self.assertEqual(s["tts"]["language_format"], "english_name")

    def test_acestep_music_alias_has_no_tts_block(self):
        s = self._schema("ACE-Step-1.5-XL")
        self.assertNotIn("tts", s)

    def test_cloud_tts_schemas_populate_style_speed_and_voices(self):
        vs = self._schema("venice.ai/tts-qwen3-1.7b")
        vtts = vs["tts"]
        self.assertEqual(vtts["text_param"], "input")
        self.assertEqual(vtts["controls"]["style"], {"param": "prompt"})
        self.assertEqual(vtts["controls"]["speed"],
                         {"param": "speed", "default": 1.0, "min": 0.25, "max": 4, "neutral": 1.0})
        self.assertEqual([v["id"] for v in vtts["voices"]], ["alloy", "nova"])
        self.assertEqual(vtts["clone"]["file_slot"], "input_audio")
        self.assertFalse(vtts["clone"]["required"])
        self.assertEqual(vtts["clone"]["max_duration_s"], 12)
        self.assertEqual(vtts["output_format"], "mp3")

        ws = self._schema("wiro/xiaomi/mimo-v2-tts")
        wtts = ws["tts"]
        self.assertEqual(wtts["text_param"], "text")
        self.assertEqual(wtts["language_format"], "bcp47")
        self.assertEqual(wtts["controls"]["style"], {"param": "instruction"})
        self.assertEqual(wtts["controls"]["speed"],
                         {"param": "speed", "default": 1.0, "min": 0.5, "max": 2.0, "neutral": 1.0})

        ls = self._schema("lulabs/lu-tts-v1")
        ltts = ls["tts"]
        self.assertEqual(ltts["text_param"], "prompt")
        self.assertIsNone(ltts["controls"]["style"])
        self.assertEqual(ltts["clone"]["file_slot"], "input_audio")
        self.assertFalse(ltts["clone"]["required"])


class TtsBuildAndUpload(unittest.TestCase):
    def _adapter(self, oi, comfy_input_dir=""):
        ctx = adapters.AdapterContext(
            auth_headers=lambda b: {}, inflight_inc=lambda b: None, inflight_dec=lambda b: None,
            cost_usd=lambda *a: 0.0, source_of=lambda r: "t", record_call=lambda *a, **k: None,
            log_enabled=lambda: False)
        b = {"name": "gpu", "type": "comfyui", "url": "http://comfy"}
        if comfy_input_dir:
            b["comfy_input_dir"] = comfy_input_dir
        ad = adapters.ComfyUIAdapter(b, ctx)
        nt, _, _, fs = adapters._parse_object_info(json.dumps(oi).encode())
        ad._node_types, ad._field_specs = nt, fs
        return ad

    def _fake_io(self, ad):
        class _IO(adapters.GenIO):
            def __init__(self):
                self.puts = []

            async def put_image(self, name, data):
                self.puts.append((name, data))
                return name

            async def put_file(self, name, data):
                self.puts.append((name, data))
                return name

            async def placeholder(self):
                return "gw_placeholder.png"

            def files_unsupported(self):
                return (None if adapters.comfy_input_dir(ad.backend)
                        else "no comfy_input_dir")

            def input_ref(self, stored):
                return adapters.input_path_ref(ad.backend, stored)

            async def node_types(self, wf, ids):
                return ad._node_types

            async def field_specs(self, classes):
                return ad._field_specs

        return _IO()

    def test_loadaudio_gets_bare_filename_without_comfy_input_dir_and_overrides_voice_to_none(self):
        ad = self._adapter(ZONOS_OI)  # NO comfy_input_dir!
        io = self._fake_io(ad)
        req = adapters.NormalizedRequest(
            alias="ZONOS3-Mini", job_id="job1", upload_prefix="gw_job1",
            workflow_json=copy.deepcopy(ZONOS_WF), node_mapping=copy.deepcopy(ZONOS_MAP),
            params={"text": "Hallo", "language": "de", "voice": "Wolfgang"},
            upload_files={"reference_audio": ("gwup_reference_audio.wav", WAV_BYTES)},
        )
        built = asyncio.run(ad._build_prompt(req, io))
        # Bare stored filename for LoadAudio (never absolute path)
        self.assertEqual(built.wf["1"]["inputs"]["audio"], "gw_job1_reference_audio.wav")
        self.assertEqual(io.puts, [("gw_job1_reference_audio.wav", WAV_BYTES)])
        # Reference audio overrides preset voice → "None" sentinel from /object_info
        self.assertEqual(built.wf["2"]["inputs"]["voice"], "None")
        self.assertEqual(built.summary["voice_overridden"], ["voice"])

    def test_voice_override_handles_optional_and_required_no_sentinel_and_string(self):
        # 1) Optional combo without "None" → popped
        oi_opt = copy.deepcopy(ZONOS_OI)
        oi_opt["ZonosTTS"]["input"]["required"].pop("voice")
        oi_opt["ZonosTTS"]["input"]["optional"]["voice"] = [["Wolfgang", "Serena"]]
        ad = self._adapter(oi_opt)
        req = adapters.NormalizedRequest(
            alias="ZONOS3-Mini", job_id="j2", upload_prefix="gw_j2",
            workflow_json=copy.deepcopy(ZONOS_WF), node_mapping=copy.deepcopy(ZONOS_MAP),
            params={"text": "Hallo", "voice": "Serena"},
            upload_files={"reference_audio": ("ref.wav", WAV_BYTES)},
        )
        built = asyncio.run(ad._build_prompt(req, self._fake_io(ad)))
        self.assertNotIn("voice", built.wf["2"]["inputs"])

        # 2) Required combo without "None" → client voice ignored, valid default kept
        oi_req = copy.deepcopy(ZONOS_OI)
        oi_req["ZonosTTS"]["input"]["required"]["voice"] = [["Wolfgang", "Serena"]]
        ad = self._adapter(oi_req)
        built = asyncio.run(ad._build_prompt(req, self._fake_io(ad)))
        self.assertEqual(built.wf["2"]["inputs"]["voice"], "Wolfgang")

        # 3) Plain STRING voice → cleared to ""
        oi_str = copy.deepcopy(ZONOS_OI)
        oi_str["ZonosTTS"]["input"]["required"]["voice"] = ["STRING", {}]
        ad = self._adapter(oi_str)
        built = asyncio.run(ad._build_prompt(req, self._fake_io(ad)))
        self.assertEqual(built.wf["2"]["inputs"]["voice"], "")

    def test_path_node_still_gets_absolute_path_and_requires_comfy_input_dir(self):
        wf = {"1": {"class_type": "PrimitiveString", "inputs": {"value": ""}},
              "2": {"class_type": "MingTTSNode", "inputs": {"reference_audio": ["1", 0]}}}
        mp = {"reference_audio": {"node": "1", "field": "value"}}
        req = adapters.NormalizedRequest(
            alias="ming", job_id="j3", upload_prefix="gw_j3",
            workflow_json=wf, node_mapping=mp,
            upload_files={"reference_audio": ("ref.wav", WAV_BYTES)},
        )
        ad_no_dir = self._adapter(MING_OI)
        with self.assertRaises(RuntimeError):
            asyncio.run(ad_no_dir._build_prompt(req, self._fake_io(ad_no_dir)))

        ad_with_dir = self._adapter(MING_OI, comfy_input_dir="/srv/comfy/input")
        built = asyncio.run(ad_with_dir._build_prompt(req, self._fake_io(ad_with_dir)))
        self.assertEqual(built.wf["1"]["inputs"]["value"],
                         "/srv/comfy/input/gw_j3_reference_audio.wav")


class TtsGenerationsEndpoint(_TtsFixture):
    def test_sync_generation_decodes_data_uri_and_returns_audio_result(self):
        captured = {}

        async def fake_run(body, request, upload_images=None, upload_files=None):
            captured["body"] = body
            captured["upload_files"] = upload_files
            return {
                "job_id": "ttsjob1", "status": "done", "task": "text2audio",
                "alias": body["model"], "backend": "gpu-3090", "error": None,
                "results": [{"n": 0, "mime": "audio/wav", "kind": "audio",
                             "name": "out.wav", "url": "http://testserver/v1/jobs/ttsjob1/result/0"}],
            }

        with patch.object(main, "run_generation", fake_run):
            r = self.c.post("/v1/generations", json={
                "model": "ZONOS3-Mini",
                "mode": "sync",
                "params": {"text": "Hallo Welt", "language": "de", "voice": "Wolfgang"},
                "files": {"reference_audio": WAV_DATA_URI},
            })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["status"], "done")
        self.assertEqual(body["results"][0]["mime"], "audio/wav")
        self.assertEqual(captured["upload_files"],
                         {"reference_audio": ("gwup_reference_audio.wav", WAV_BYTES)})

    def test_empty_optional_file_slot_is_ignored_cleanly(self):
        captured = {}

        async def fake_run(body, request, upload_images=None, upload_files=None):
            captured["upload_files"] = upload_files
            return {"job_id": "ttsjob2", "status": "done", "results": []}

        with patch.object(main, "run_generation", fake_run):
            r = self.c.post("/v1/generations", json={
                "model": "ZONOS3-Mini",
                "mode": "sync",
                "params": {"text": "Hallo", "voice": "Wolfgang"},
                "files": {"reference_audio": None},
            })
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(captured["upload_files"], {})


if __name__ == "__main__":
    unittest.main()
