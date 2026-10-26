"""Task pickers use catalog capabilities, not alias names or provider families."""
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from unittest.mock import patch
import httpx

from tests.test_ui_form_validation import _Fixture
import admin
import store


class Controls(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.options = {}
        self.values = {}
        self.selected = {}
        self.current = None
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("select", "datalist"):
            self.current = attrs.get("name") or attrs.get("id")
            self.options.setdefault(self.current, [])
        elif tag == "option" and self.current:
            value = attrs.get("value", "")
            self.options[self.current].append(value)
            if "selected" in attrs:
                self.selected[self.current] = value
        elif tag == "input" and attrs.get("name"):
            self.values[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag in ("select", "datalist"):
            self.current = None


def run_picker_js(case, markup, marker, setup, action):
    if not shutil.which("node"):
        case.skipTest("node not installed")
    script = next(s for s in re.findall(r"<script>(.*?)</script>", markup, re.S) if marker in s)
    result = subprocess.run(["node", "-e", setup + script + action], capture_output=True, text=True)
    case.assertEqual(result.returncode, 0, result.stderr)
    return json.loads(result.stdout)


class PickerTypes(_Fixture):
    def setUp(self):
        super().setUp()
        self.backends = [
            {"name": "mixed", "type": "openai", "current": True,
             "models": ["plain", "camera", "opaque-speech", "vectors", "painting", "both"],
             "model_types": {"plain": ["chat"], "camera": ["chat"], "opaque-speech": ["tts"],
                             "vectors": ["embedding"], "painting": ["image"], "both": ["chat", "tts"]}},
            {"name": "voice-only", "type": "openai", "models": ["another-voice"],
             "model_types": {"another-voice": ["tts"]}},
        ]
        # Alias names deliberately suggest the WRONG task; actual targets win.
        self.config_chat.update({"embedding": {"mixed": "plain"},
                                 "helper": {"mixed": "opaque-speech"},
                                 "painting": {"mixed": "vectors"},
                                 "shared": {"mixed": "both"}})
        for name, callback in {
            "_llm_backends": lambda: self.backends,
            "_llm_backend_names": lambda: [b["name"] for b in self.backends],
            "_gateway_info": lambda: {"backends": [], "virtual_models": list(self.config_chat),
                                      "alias_types": {"embedding": ["chat"], "helper": ["tts"],
                                                      "painting": ["embedding"], "shared": ["chat", "tts"]}},
            "_generation_models": lambda: [],
            "_generation_candidate": lambda model: None,
        }.items():
            p = patch.object(admin, name, callback)
            p.start()
            self.addCleanup(p.stop)

    def test_chat_aliases_bare_and_prefixed_models_keep_only_chat_and_vision_chat(self):
        markup = self.c.get("/ui/playground?sub=chat").text
        choices = Controls(markup).options["cpmodels"]
        self.assertIn("embedding", choices)
        self.assertIn("camera", choices)
        self.assertIn("mixed/camera", choices)
        self.assertIn("shared", choices)
        for wrong in ("helper", "painting", "opaque-speech", "vectors", "mixed/vectors", "another-voice"):
            self.assertNotIn(wrong, choices)

    def test_backend_specific_chat_and_reasoning_do_not_fall_back_from_empty_list(self):
        chat = Controls(self.c.get("/ui/playground?sub=chat&backend=voice-only").text)
        self.assertEqual(chat.options["cpmodels"], [])
        reasoning = Controls(admin._reasoning_form({"backends": ["voice-only"]}, None))
        self.assertEqual(reasoning.options["rtmodels"], [])

    def test_voice_uses_tts_targets_and_preserves_busy_speech_workflows(self):
        store.upsert("queued", [{"backend": "gpu", "task": "text2audio"}])
        store.upsert("music", [{"backend": "gpu", "task": "audio"}])
        store.upsert("cloud-alias", [{"backend": "cloud", "task": "tts", "nanogpt": {"endpoint": "tts"}}])
        rows = [{"id": "cloud/direct-voice", "endpoint": "tts",
                 "candidate": {"backend": "cloud", "model": "direct-voice", "nanogpt": {"endpoint": "tts"}}},
                {"id": "painting", "endpoint": "tts",
                 "candidate": {"backend": "cloud", "model": "other-voice", "nanogpt": {"endpoint": "tts"}}}]
        with patch.object(admin, "_generation_models", lambda: rows):
            choices = Controls(self.c.get("/ui/playground?sub=voice").text).options["vpmodels"]
        for good in ("helper", "shared", "opaque-speech", "mixed/opaque-speech", "queued", "cloud/direct-voice"):
            self.assertIn(good, choices)
        for wrong in ("embedding", "plain", "vectors", "music", "cloud-alias", "painting"):
            self.assertNotIn(wrong, choices)

    def test_legacy_alias_fallback_follows_store_override_instead_of_alias_name(self):
        store.upsert_chat_alias("embedding", {"mixed": "opaque-speech"})
        with patch.object(admin, "_gateway_info", lambda: {"virtual_models": list(self.config_chat)}):
            self.assertNotIn("embedding", admin._chat_models())
            self.assertIn("embedding", admin._voice_models())

    def test_prefixed_task_choices_resolve_aliases_but_target_editors_keep_literal_models(self):
        self.config_chat.update({"plain": {"mixed": "opaque-speech"},
                                 "opaque-speech": {"mixed": "plain"}})
        for mapped in (False, True):
            with self.subTest(callable_metadata=mapped):
                if mapped:
                    self.backends[0]["callable_model_types"] = {
                        **self.backends[0]["model_types"], "plain": ["tts"], "opaque-speech": ["chat"]}
                chat = Controls(self.c.get("/ui/playground?sub=chat&backend=mixed").text)
                all_chat = Controls(self.c.get("/ui/playground?sub=chat").text)
                with patch.object(admin, "_generation_models", lambda: [{
                        "id": "mixed/opaque-speech", "endpoint": "tts", "candidate": {
                            "backend": "mixed", "model": "opaque-speech", "nanogpt": {"endpoint": "tts"}}}]):
                    voice = Controls(self.c.get("/ui/playground?sub=voice").text)
                reasoning = Controls(admin._reasoning_form({"backends": ["mixed"]}, None))
                literal = Controls(admin._chat_new_form({"backend": "mixed", "type": "chat"}))
                self.assertIn("opaque-speech", chat.options["cpmodels"])
                self.assertNotIn("plain", chat.options["cpmodels"])
                self.assertIn("opaque-speech", reasoning.options["rtmodels"])
                self.assertIn("mixed/opaque-speech", all_chat.options["cpmodels"])
                self.assertNotIn("mixed/plain", all_chat.options["cpmodels"])
                self.assertNotIn("plain", reasoning.options["rtmodels"])
                self.assertIn("mixed/plain", voice.options["vpmodels"])
                self.assertNotIn("mixed/opaque-speech", voice.options["vpmodels"])
                self.assertIn("plain", literal.options["cm_all"])
                self.assertNotIn("opaque-speech", literal.options["cm_all"])

    def test_empty_callable_metadata_does_not_fall_back_to_raw_chat_classification(self):
        self.backends[0]["callable_model_types"] = {
            **self.backends[0]["model_types"], "plain": []}
        choices = Controls(self.c.get("/ui/playground?sub=chat&backend=mixed").text).options["cpmodels"]
        self.assertNotIn("plain", choices)

    def test_voice_omits_display_only_direct_tts_snapshot_but_keeps_queued_workflows(self):
        store.upsert("queued", [{"backend": "gpu", "task": "tts"}])
        rows = [{"id": "cloud/offline-voice", "endpoint": "tts",
                 "candidate": {"backend": "cloud", "model": "offline-voice", "unavailable": True,
                               "venice": {"endpoint": "tts"}}}]
        with patch.object(admin, "_generation_models", lambda: rows):
            choices = Controls(self.c.get("/ui/playground?sub=voice").text).options["vpmodels"]
        self.assertIn("queued", choices)
        self.assertNotIn("cloud/offline-voice", choices)

    def test_alias_editor_retains_embedding_targets_with_explicit_type_selection(self):
        new = Controls(admin._chat_new_form())
        self.assertEqual(new.selected["type"], "chat")
        self.assertNotIn("vectors", new.options["cm_all"])
        editor = Controls(admin._chat_editor("painting"))
        self.assertEqual(editor.selected["type"], "embedding")
        self.assertEqual(editor.values["model__mixed"], "vectors")
        self.assertEqual(editor.options["cm_0"], ["vectors"])

    def test_multipurpose_alias_management_keeps_all_target_capabilities_editable(self):
        controls = Controls(admin._chat_editor("shared"))
        self.assertEqual(controls.selected["type"], "all")
        self.assertIn("both", controls.options["cm_0"])
        self.assertEqual(controls.values["model__mixed"], "both")

    def test_chat_backend_switch_rebuilds_task_choices_and_clears_stale_model(self):
        markup = admin._chatplay_form({})
        result = run_picker_js(self, markup, "var CP_BK=", """
var model={value:'plain'}, backend={value:'voice-only'}, out={textContent:''};
var dl={items:[],set innerHTML(v){this.items=[];},appendChild(o){this.items.push(o.value);}};
var document={getElementById:function(id){return id==='cpmodels'?dl:out;},
 querySelector:function(q){return q.indexOf('[name="model"]')>=0?model:backend;},
 createElement:function(){return {};}};
""", "cpFilterModels('voice-only');console.log(JSON.stringify({choices:dl.items,value:model.value}));")
        self.assertEqual(result, {"choices": [], "value": ""})

    def test_alias_type_and_backend_switch_rebuild_suggestions(self):
        markup = admin._chat_new_form()
        result = run_picker_js(self, markup, "var AM_BK=", """
var dl={items:[],set innerHTML(v){this.items=[];},appendChild(o){this.items.push(o.value);}};
var controls={type:{value:'tts'},backend:{value:'mixed'}};
var document={getElementById:function(){return dl;},createElement:function(){return {};}};
var form={elements:controls};
""", "amFilter(form);console.log(JSON.stringify(dl.items));")
        self.assertEqual(result, ["both", "opaque-speech"])


class MediaTypes(_Fixture):
    def setUp(self):
        super().setUp()
        self.rows = [
            {"id": "cloud/picture", "name": "Picture", "backend": "cloud", "endpoint": "images",
             "candidate": {"backend": "cloud", "model": "picture", "nanogpt": {"endpoint": "images"}}},
            {"id": "cloud/movie", "name": "Movie", "backend": "cloud", "endpoint": "video",
             "candidate": {"backend": "cloud", "model": "movie", "nanogpt": {"endpoint": "video"}}},
            {"id": "cloud/transcribe", "backend": "cloud", "endpoint": "asr",
             "candidate": {"backend": "cloud", "model": "transcribe", "nanogpt": {"endpoint": "asr"}}},
        ]
        for name, callback in {"_generation_models": lambda: self.rows,
                               "_generation_candidate": lambda model: None}.items():
            p = patch.object(admin, name, callback)
            p.start()
            self.addCleanup(p.stop)
        store.upsert("picture-alias", [{"backend": "gpu", "task": "text2img", "workflow_json": {}}])
        store.upsert("movie-alias", [{"backend": "gpu", "task": "text2video", "workflow_json": {}}])
        store.upsert("speech-alias", [{"backend": "gpu", "task": "text2audio", "workflow_json": {}}])

    def test_category_filters_aliases_and_catalog_and_replaces_invalid_selected_model(self):
        page = self.c.get("/ui/playground?sub=media&type=video&model=picture-alias").text
        controls = Controls(page)
        self.assertEqual(controls.options["model"], ["movie-alias", "cloud/movie"])
        self.assertEqual(controls.selected["type"], "video")
        self.assertEqual(controls.selected["model"], "movie-alias")
        self.assertNotIn("picture-alias", controls.options["model"])

    def test_all_media_includes_speech_and_transcription(self):
        controls = Controls(self.c.get("/ui/playground?sub=media&type=media").text)
        self.assertIn("speech-alias", controls.options["model"])
        self.assertIn("cloud/transcribe", controls.options["model"])

    def test_empty_category_still_offers_category_control_and_no_generation(self):
        controls = Controls(self.c.get("/ui/playground?sub=media&type=3d").text)
        self.assertEqual(controls.selected["type"], "3d")
        self.assertEqual(controls.options["model"], [])

    def test_category_switch_carries_type_and_scalars_but_not_invalid_model(self):
        markup = admin._playground_form(admin._media_models(), {"model": "picture-alias", "type": "media"}, None)
        url = run_picker_js(self, markup, "function pgSwitch", """
var location={href:''},form={elements:{type:{value:'video'},model:{value:'picture-alias'}},
 querySelectorAll:function(){return [{name:'p__prompt',value:'a cat'}];}};
""", "pgSwitch({form:form,name:'type',value:'video'});console.log(JSON.stringify(location.href));")
        self.assertEqual(url, "/ui/playground?sub=media&type=video&p__prompt=a%20cat")

    def test_cloud_alias_model_suggestions_follow_endpoint_initially_and_on_change(self):
        with patch.object(admin, "_generation_models", lambda: self.rows):
            markup = admin._cloud_editor("nanogpt", "picture-alias", [self.rows[0]["candidate"]])
        controls = Controls(markup)
        self.assertEqual(controls.options["nano-alias-models"], ["picture"])
        result = run_picker_js(self, markup, "var CA_MODELS=", """
var input={value:'picture'};
var dl={items:[],set innerHTML(v){this.items=[];},appendChild(o){this.items.push(o.value);}};
var document={getElementById:function(){return dl;},createElement:function(){return {};}};
var form={elements:{cloud_endpoint:{value:'video'},cloud_model:input}};
""", "caFilter(form);console.log(JSON.stringify({choices:dl.items,value:input.value}));")
        self.assertEqual(result, {"choices": ["movie"], "value": ""})

    def test_alias_collision_is_not_reintroduced_by_catalog_after_category_filtering(self):
        self.rows.append({"id": "picture-alias", "backend": "cloud", "endpoint": "video",
                          "candidate": self.rows[1]["candidate"]})
        choices = Controls(self.c.get("/ui/playground?sub=media&type=video").text).options["model"]
        self.assertEqual(choices, ["movie-alias", "cloud/movie"])

    def test_multi_endpoint_direct_id_keeps_all_memberships_and_canonical_candidate(self):
        self.rows.append({"id": "cloud/picture", "name": "Picture video", "backend": "cloud",
                          "endpoint": "video", "candidate": {
                              "backend": "cloud", "model": "picture", "nanogpt": {"endpoint": "video"}}})
        for kind in ("image", "video", "media"):
            with self.subTest(category=kind):
                choices = Controls(self.c.get(
                    "/ui/playground?sub=media&type=" + kind + "&model=cloud/picture").text)
                self.assertEqual(choices.options["model"].count("cloud/picture"), 1)
                self.assertEqual(choices.selected["model"], "cloud/picture")
                selected = next(row for row in admin._media_models(kind) if row["id"] == "cloud/picture")
                self.assertEqual(selected["candidate"]["nanogpt"]["endpoint"], "images")
                self.assertEqual(admin._media_candidate("cloud/picture", [selected])["nanogpt"]["endpoint"], "images")

    def test_category_is_preserved_after_generation_without_changing_api_payload(self):
        sent = []

        async def api(request, method, path, **kwargs):
            sent.append((method, path, kwargs["json"]))
            return httpx.Response(202, json={"job_id": "category-job"})

        with patch.object(admin, "_self_api", api):
            response = self.post("/ui/playground/generate", {
                "model": "picture-alias", "type": "image", "p__prompt": "a cat"})
        self.assertEqual(response.status_code, 303)
        self.assertIn("type=image", response.headers["location"])
        self.assertEqual(sent, [("POST", "/v1/generations", {
            "model": "picture-alias", "mode": "async", "params": {}, "prompt": "a cat"})])
