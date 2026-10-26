"""Speech aliases must not execute non-speech workflows or silently ignore voice references.

These checks pin HTTP refusal boundaries before a GPU job exists, including offline
aliases and attempts to run image workflows through the speech endpoint.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request

_here = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _here)
_previous = os.getcwd()
with tempfile.TemporaryDirectory() as _tmp:
    Path(_tmp, 'config.yaml').write_text('api_key: ""\nbackends: []\n')
    os.chdir(_tmp)
    try:
        import main
    finally:
        os.chdir(_previous)


def request(body):
    async def receive():
        return {'type': 'http.request', 'body': json.dumps(body).encode()}
    return Request({'type': 'http', 'method': 'POST', 'path': '/v1/audio/speech',
                    'headers': [], 'scheme': 'http', 'server': ('test', 80),
                    'query_string': b'', 'client': ('127.0.0.1', 1)}, receive)


class ComfySpeech(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cand = {'backend': 'gpu', 'task': 'tts', 'workflow_json': {
            '1': {'class_type': 'LoadAudio', 'inputs': {'audio': ''}}},
            'mapping': {'reference_audio': {'node': '1', 'field': 'audio'}}}
        self.stack = __import__('contextlib').ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(main.store, 'is_active', return_value=False))
        self.stack.enter_context(patch.object(main, 'image_models', {'ming': [self.cand]}))

    async def test_offline_speech_alias_does_not_fall_through_to_chat(self):
        async def forbidden(*args, **kw):
            self.fail('workflow alias was sent to native chat routing')
        with patch.object(main, 'route', forbidden):
            with self.assertRaises(main.HTTPException) as error:
                await main.audio_speech(request({'model': 'ming', 'input': 'Hallo'}), None)
        self.assertEqual(error.exception.status_code, 503)

    async def test_invalid_text_refused_before_job(self):
        with self.assertRaises(main.HTTPException) as error:
            await main.audio_speech(request({'model': 'ming', 'input': ['wrong']}), None)
        self.assertEqual(error.exception.status_code, 400)

    async def test_unmapped_reference_is_refused_instead_of_using_default_voice(self):
        self.cand["mapping"] = {}
        with self.assertRaises(main.HTTPException) as error:
            await main.audio_speech(request({"model": "ming", "input": "Hallo",
                "input_audio": "data:audio/wav;base64,UklGRg=="}), None)
        self.assertEqual(error.exception.status_code, 400)

    async def test_image_alias_cannot_be_executed_as_speech(self):
        self.cand["task"] = "text2img"
        with self.assertRaises(main.HTTPException) as error:
            await main.audio_speech(request({"model": "ming", "input": "Hallo"}), None)
        self.assertEqual(error.exception.status_code, 400)

    async def test_explicit_audio_overrides_alias_default_voice(self):
        with patch.object(main, "alias_voice", {"ming": {"voice": "lib:default"}}):
            with self.assertRaises(main.HTTPException) as error:
                await main.audio_speech(request({"model": "ming", "input": "Hallo",
                    "input_audio": "data:audio/wav;base64,UklGRg=="}), None)
        # The upload is accepted; only the absence of a live GPU prevents dispatch.
        self.assertEqual(error.exception.status_code, 503)

    async def test_inconsistent_reference_keys_are_refused_before_dispatch(self):
        import copy
        other = copy.deepcopy(self.cand)
        other["mapping"] = {"other_ref": {"node": "1", "field": "audio",
                                         "label": "reference_audio"}}
        main.image_models["ming"].append(other)
        with self.assertRaises(main.HTTPException) as error:
            await main.audio_speech(request({"model": "ming", "input": "Hallo",
                "input_audio": "data:audio/wav;base64,UklGRg=="}), None)
        self.assertEqual(error.exception.status_code, 400)

    async def test_falsy_nonstring_voice_cannot_select_a_default(self):
        for voice in (False, 0, [], {}):
            with self.subTest(voice=voice), self.assertRaises(main.HTTPException) as error:
                await main.audio_speech(request({"model": "ming", "input": "Hallo",
                                                "voice": voice}), None)
            self.assertEqual(error.exception.status_code, 400)



if __name__ == '__main__':
    unittest.main()
