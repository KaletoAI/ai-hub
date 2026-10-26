"""Configured speech aliases must remain selectable while their GPU is busy.

The healthy-only media catalog hides queued workflows. The rendered form must
include stored speech aliases without offering unrelated music/cloud workflows.
"""
from html.parser import HTMLParser
from unittest.mock import patch

from tests.test_ui_form_validation import _Fixture
import admin
import store


class _VoiceControls(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.models = []
        self.fields = {}
        self.datalist = ""
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "datalist":
            self.datalist = attrs.get("id", "")
        if tag == "option" and self.datalist == "vpmodels":
            self.models.append(attrs.get("value"))
        if tag in ("input", "select", "textarea") and attrs.get("name"):
            self.fields[attrs["name"]] = attrs

    def handle_endtag(self, tag):
        if tag == "datalist":
            self.datalist = ""


class VoicePlayground(_Fixture):
    def test_busy_workflow_aliases_join_native_models_without_cloud_or_music(self):
        store.upsert("workflow-speech", [{"backend": "gpu", "task": "tts"}])
        store.upsert("workflow-text2audio", [{"backend": "gpu", "task": "text2audio"}])
        store.upsert("workflow-music", [{"backend": "gpu", "task": "audio"}])
        store.upsert("cloud-speech", [{"backend": "cloud", "task": "tts",
                                        "nanogpt": {"endpoint": "tts"}}])
        # No healthy catalog row: stored aliases must still be selectable in a queue.
        with patch.object(admin, "_generation_models", lambda: []), \
                patch.object(admin, "_llm_backends", lambda: [{
                    "name": "native", "models": ["native-tts"],
                    "model_types": {"native-tts": ["tts"]}}]):
            response = self.c.get("/ui/playground?sub=voice")
        self.assertEqual(response.status_code, 200)
        controls = _VoiceControls(response.text)
        self.assertEqual(set(controls.models), {
            "native-tts", "native/native-tts", "workflow-speech", "workflow-text2audio"})
        self.assertEqual(controls.models.count("workflow-speech"), 1)
        for name in ("model", "input", "voice_sel", "voice", "ref_text", "language"):
            self.assertIn(name, controls.fields)

