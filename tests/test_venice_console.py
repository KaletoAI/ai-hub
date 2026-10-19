"""Guard composite Venice console edits and native media controls against silent
paid requests with lost false values, missing uploads or stale catalog availability.
"""
from html.parser import HTMLParser
import json
import re
import shutil
import subprocess
from unittest.mock import patch

import httpx

from tests.test_ui_form_validation import _Fixture
import admin
import adapters
import store
import venice


class Controls(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.names, self.fields, self.selected, self.options, self.groups = [], {}, {}, [], []
        self.select = ""
        self.downloads = []
        self.disabled_buttons = 0
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "select":
            self.select = attrs.get("name", "")
        if tag in ("input", "select", "textarea") and attrs.get("name"):
            self.names.append(attrs["name"])
            self.fields[attrs["name"]] = attrs
        if tag == "option":
            self.options.append(attrs.get("value"))
            if "selected" in attrs:
                self.selected[self.select] = attrs.get("value")
        if tag == "optgroup":
            self.groups.append(attrs.get("label"))
        if tag == "button" and "disabled" in attrs:
            self.disabled_buttons += 1
        if tag == "a" and "download" in attrs:
            self.downloads.append(attrs.get("href"))

    def handle_endtag(self, tag):
        if tag == "select":
            self.select = ""


class VeniceBackend(_Fixture):
    def test_one_account_preserves_text_defaults_and_forces_paid(self):
        response = self.post("/ui/backends/save", {
            "name": "venice", "type": "venice", "url": "", "api_key": "private-key",
            "chat_only": "1", "models_allow": "llama-*", "smp_temperature": "0.7",
            "cloud_max_wait": "120", "cloud_poll_interval": "0.5"})
        self.assertEqual(response.status_code, 303, response.text)
        saved = store.get_backend("venice", "venice")
        self.assertEqual(saved["url"], venice.URL)
        self.assertTrue(saved["paid"])
        self.assertTrue(saved["chat_only"])
        self.assertEqual(saved["sampling_defaults"]["temperature"], 0.7)
        self.assertEqual(saved["poll_interval"], 0.5)
        page = self.c.get("/ui/backends?edit=venice:venice")
        controls = Controls(page.text)
        self.assertEqual(controls.names.count("api_key"), 1)
        for name in ("chat_only", "smp_temperature", "cloud_max_wait", "cloud_poll_interval"):
            self.assertIn(name, controls.fields)
        self.assertNotIn("private-key", page.text)

    def test_refused_timing_retains_typed_controls_without_key(self):
        for field, value in (("cloud_max_wait", "1.5"), ("cloud_poll_interval", "broken")):
            response = self.post("/ui/backends/save", {
                "name": "venice", "type": "venice", "url": "", "api_key": "private-key",
                field: value})
            self.assertEqual(response.status_code, 400, response.text)
            self.assertEqual(Controls(response.text).fields[field]["value"], value)
            self.assertNotIn("private-key", response.text)
            self.assertIsNone(store.get_backend("venice", "venice"))


class VeniceMedia(_Fixture):
    def setUp(self):
        super().setUp()
        self.cand = venice.candidate_for("venice", "native-model", "transcription")
        self.rows = [{"id": "venice/native-model", "name": "Native model", "backend": "venice",
                      "endpoint": "transcription", "candidate": self.cand}]
        for name, callback in (("_generation_models", lambda: self.rows),
                               ("_generation_candidate", lambda model: self.cand
                                if model == self.rows[0]["id"] else None)):
            p = patch.object(admin, name, callback)
            p.start()
            self.addCleanup(p.stop)
        old = dict(admin._pg_images)
        admin._pg_images.clear()
        self.addCleanup(lambda: (admin._pg_images.clear(), admin._pg_images.update(old)))

    def test_every_endpoint_catalog_groups_without_rewriting_ids(self):
        self.rows[:] = [{"id": "venice/" + endpoint, "backend": "venice", "endpoint": endpoint,
                        "candidate": venice.candidate_for("venice", "native-" + endpoint, endpoint)}
                       for endpoint in venice.ENDPOINTS]
        page = self.c.get("/ui/playground?sub=media")
        controls = Controls(page.text)
        for row in self.rows:
            self.assertIn(row["id"], controls.options)
        for group in ("Images", "Video", "Audio", "Speech", "Transcription"):
            self.assertIn("venice · " + group, controls.groups)

    def test_transcription_uses_schema_uploads_without_alias(self):
        page = self.c.get("/ui/playground?sub=media")
        controls = Controls(page.text)
        _, _, files = adapters.public_fields(self.cand)
        self.assertTrue(files)
        for field in files:
            self.assertEqual(controls.fields["file__" + field["name"]]["type"], "file")

    def test_optional_false_is_selected_and_sent_as_boolean(self):
        self.cand = venice.candidate_for("venice", "native-image", "images")
        self.rows[0].update(candidate=self.cand, endpoint="images")
        sent = []

        async def api(request, method, path, **kwargs):
            sent.append(kwargs["json"])
            return httpx.Response(202, json={"job_id": "venice-job"})

        page = self.c.get("/ui/playground?sub=media&p__safe_mode=false")
        self.assertEqual(Controls(page.text).selected["p__safe_mode"], "false")
        with patch.object(admin, "_self_api", api):
            response = self.post("/ui/playground/generate", {
                "model": self.rows[0]["id"], "p__safe_mode": "false"})
        self.assertEqual(response.status_code, 303, response.text)
        self.assertIs(sent[0]["params"]["safe_mode"], False)

    def test_numeric_speech_text_survives_native_validation(self):
        """Numeric-looking text/voice IDs are strings, not numeric generation controls."""
        self.cand = venice.candidate_for("venice", "tts-custom", "speech",
            {"model_spec": {"supports_custom_voice_id": True}})
        self.rows[0].update(candidate=self.cand, endpoint="speech")

        async def api(request, method, path, **kwargs):
            try:
                venice.validate_request(self.cand, kwargs["json"]["params"], (), ())
            except venice.INPUT_ERROR as error:
                return httpx.Response(400, json={"detail": str(error)})
            return httpx.Response(202, json={"job_id": "speech-job"})

        with patch.object(admin, "_self_api", api):
            response = self.post("/ui/playground/generate", {
                "model": self.rows[0]["id"], "p__input": "12345", "p__voice": "67890"})
        self.assertEqual(response.status_code, 303, response.text)

    def test_unavailable_snapshot_stays_visible_but_generation_disabled(self):
        self.cand["unavailable"] = True
        page = self.c.get("/ui/playground?sub=media")
        controls = Controls(page.text)
        self.assertIn(self.rows[0]["id"], controls.options)
        self.assertGreater(controls.disabled_buttons, 0)

    def test_native_metadata_bounds_choices_and_alias_false_controls(self):
        metadata = {"id": "native-image", "model_spec": {"constraints": {
            "aspectRatios": ["1:1", "16:9"], "defaultAspectRatio": "1:1",
            "resolutions": ["1K", "2K"], "defaultResolution": "1K",
            "steps": {"default": 20, "max": 30}}}}
        self.cand = venice.candidate_for("venice", "native-image", "images", metadata)
        self.rows[0].update(candidate=self.cand, endpoint="images")
        page = self.c.get("/ui/playground?sub=media")
        controls = Controls(page.text)
        self.assertIn("16:9", controls.options)
        self.assertIn("2K", controls.options)
        self.assertEqual(controls.fields["p__steps"]["max"], "30")
        self.assertNotIn("p__width", controls.fields)
        self.cand["venice"]["options"]["safe_mode"] = False
        editor = admin._cloud_editor("venice", "native-alias", [self.cand])
        controls = Controls(editor)
        self.assertEqual(controls.selected["opt__safe_mode"], "false")
        self.assertEqual(controls.fields["opt__steps"]["max"], "30")
        self.assertIn("native-image", controls.options)

    def test_json_and_text_artifacts_are_downloads_not_images(self):
        for mime in ("application/json", "text/plain"):
            job = {"status": "succeeded", "results": [{"n": 0, "mime": mime}], "meta": {}}
            rendered, refresh = admin._job_result_html("venice-result", job)
            self.assertIsNone(refresh)
            self.assertEqual(Controls(rendered).downloads, ["/ui/playground/result/venice-result/0"])
            gallery = admin._job_thumbs("venice-result", "result", job["results"])
            self.assertEqual(Controls(gallery).downloads, ["/ui/playground/result/venice-result/0"])

class VeniceAliases(_Fixture):
    def setUp(self):
        super().setUp()
        self.cand = venice.candidate_for("venice", "native-image", "images")
        store.upsert("venice-alias", [self.cand])

    def test_every_endpoint_saves_exact_model_without_workflow(self):
        for endpoint in venice.ENDPOINTS:
            with self.subTest(endpoint=endpoint):
                response = self.post("/ui/mapping/cloud-update", {
                    "alias": "venice-alias", "cloud_endpoint": endpoint,
                    "cloud_model": "native-model:version", "task": "text2img"})
                self.assertEqual(response.status_code, 303, response.text)
                saved = store.get("venice-alias")[0]
                self.assertEqual(saved["model"], "native-model:version")
                self.assertEqual(saved["venice"]["endpoint"], endpoint)
                self.assertEqual(saved["task"], venice.task_for(endpoint, "text2img"))
                self.assertNotIn("workflow_json", saved)

    def test_empty_model_refusal_keeps_form_and_stored_alias(self):
        response = self.post("/ui/mapping/cloud-update", {
            "alias": "venice-alias", "cloud_endpoint": "video", "cloud_model": "",
            "task": "text2vid", "retries": "3"})
        self.assertEqual(response.status_code, 400, response.text)
        controls = Controls(response.text)
        self.assertEqual(controls.selected["cloud_endpoint"], "video")
        self.assertEqual(controls.fields["cloud_model"]["value"], "")
        self.assertEqual(controls.fields["retries"]["value"], "3")
        self.assertEqual(store.get("venice-alias"), [self.cand])


class BillingHTML(HTMLParser):
    """Read rendered labels and form values rather than implementation source."""
    def __init__(self, source):
        super().__init__()
        self.options, self.text, self.context = {}, [], {}
        self.option = None
        self.current = None
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "option":
            self.option = attrs.get("value")
            if self.option is not None:
                self.options[self.option] = attrs.get("label", "")
        if attrs.get("id") == "cpbilling":
            self.current = "cpbilling"
            self.context[self.current] = ""

    def handle_data(self, data):
        self.text.append(data)
        if self.option is not None:
            self.options[self.option] += data
        if self.current:
            self.context[self.current] += data

    def handle_endtag(self, tag):
        if tag == "option":
            self.option = None
        if tag == "span":
            self.current = None


class VeniceBilling(_Fixture):
    """Guard account billing signs: labels never become routing IDs or free claims."""
    def setUp(self):
        super().setUp()
        self.llm = [
            {"name": "charged", "type": "venice", "models": ["shared", "credit-only"],
             "billing": {"state": "credits"}},
            {"name": "included", "type": "venice", "models": ["shared", "included-only"],
             "billing": {"state": "included"}},
            {"name": "unknown", "type": "venice", "models": ["unknown-only"],
             "billing": {"state": "unknown", "error": "HTTP 401"}},
            {"name": "other", "type": "openai", "models": ["shared"]}]
        for name, callback in (
                ("_llm_backends", lambda: self.llm),
                ("_gateway_info", lambda: {"backends": [
                    {**b, "url": "https://api.example", "enabled": True,
                     "healthy": True, "models": len(b["models"])}
                    for b in self.llm], "virtual_models": ["mixed-alias", "included-alias"]})):
            p = patch.object(admin, name, callback)
            p.start()
            self.addCleanup(p.stop)
        store.upsert_chat_alias("mixed-alias", {"included": "shared", "other": "shared"})
        store.upsert_chat_alias("included-alias", {"included": "included-only"})

    def test_chat_labels_keep_actual_ids_and_mixed_routes_are_not_included(self):
        page = self.c.get("/ui/playground?sub=chat&model=included/included-only")
        rendered = BillingHTML(page.text)
        for value, marker in (
                ("charged/credit-only", "[C]"), ("included/included-only", "[I]"),
                ("unknown/unknown-only", "[?]"), ("shared", "[mixed]"),
                ("mixed-alias", "[mixed]"), ("included-alias", "[I]")):
            self.assertIn(marker, rendered.options[value], value)
        self.assertEqual(Controls(page.text).fields["model"]["value"], "included/included-only")
        self.assertIn("[I]", rendered.context["cpbilling"])
        text = " ".join(rendered.text)
        self.assertIn("Pro web", text)
        self.assertIn("free API", text)

    def test_backend_filtered_chat_labels_keep_values_and_selected_context(self):
        """Execute the shipped filtering handler; re-filtering cannot drop billing labels."""
        if not shutil.which("node"):
            self.skipTest("node not installed")
        markup = admin._chatplay_form({"model": "shared"})
        scripts = "\n".join(re.findall(r"<script>(.*?)</script>", markup, re.S))
        program = """
var dl={children:[],set innerHTML(v){this.children=[];},appendChild:function(o){this.children.push(o);}};
var model={value:'shared'},backend={value:''},context={textContent:''};
var document={
 getElementById:function(id){return id==='cpmodels'?dl:id==='cpbilling'?context:null;},
 querySelector:function(q){return q.indexOf('backend')>=0?backend:model;},
 createElement:function(){return {value:'',label:'',textContent:''};}
};
""" + scripts + """
var result=[];
['included','charged','unknown',''].forEach(function(bk){
 backend.value=bk;cpFilterModels(bk);
 result.push({options:dl.children,context:context.textContent,model:model.value});
});
console.log(JSON.stringify(result));
"""
        result = subprocess.run(["node", "-e", program], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        filtered = json.loads(result.stdout)
        for i, marker in ((0, "[I]"), (1, "[C]"), (2, "[?]")):
            options = filtered[i]["options"]
            expected = ("included-only", "credit-only", "unknown-only")[i]
            option = next(o for o in options if o["value"] == expected)
            self.assertIn(marker, option["label"])
            self.assertEqual(filtered[i]["model"], "shared")
        self.assertIn("[I]", filtered[0]["context"])
        self.assertIn("[C]", filtered[1]["context"])
        self.assertIn("[?]", filtered[2]["context"])
        self.assertIn("[mixed]", filtered[3]["context"])

    def test_media_labels_preserve_direct_and_alias_ids(self):
        rows = [{"id": "included/native-image", "name": "Native image", "backend": "included",
                 "endpoint": "images", "billing": {"state": "included"},
                 "candidate": venice.candidate_for("included", "native-image", "images")},
                {"id": "charged/native-image", "name": "Native image", "backend": "charged",
                 "endpoint": "images", "billing": {"state": "credits"},
                 "candidate": venice.candidate_for("charged", "native-image", "images")},
                {"id": "unknown/native-image", "backend": "unknown", "endpoint": "images",
                 "billing": {"state": "unknown"},
                 "candidate": venice.candidate_for("unknown", "native-image", "images")}]
        store.upsert("mixed-media", [
            venice.candidate_for("included", "native-image", "images"),
            {"backend": "other", "workflow_json": {}, "mapping": {}}])
        with patch.object(admin, "_generation_models", lambda: rows):
            markup = admin._playground_form(admin._media_models(),
                {"model": "charged/native-image"}, rows[1]["candidate"])
        rendered = BillingHTML(markup)
        for value, marker in (("included/native-image", "[I]"), ("charged/native-image", "[C]"),
                              ("unknown/native-image", "[?]"), ("mixed-media", "[mixed]")):
            self.assertIn(marker, rendered.options[value])
        self.assertEqual(Controls(markup).selected["model"], "charged/native-image")

    def test_media_alias_billing_keeps_same_named_backend_kinds_separate(self):
        """A local GPU sharing a Venice account's name must never inherit included API usage."""
        self.llm.extend([
            {"name": "same", "type": "comfyui", "models": []},
            {"name": "same", "type": "venice", "models": ["native-image"],
             "billing": {"state": "included"}}])
        gpu = {"backend": "same", "model": "checkpoint", "workflow_json": {}, "mapping": {}}
        cloud = venice.candidate_for("same", "native-image", "images")
        store.upsert("gpu-only", [gpu])
        store.upsert("cloud-only", [cloud])
        store.upsert("same-name-mixed", [gpu, cloud])
        markup = admin._playground_form(admin._media_models(), {"model": "cloud-only"}, cloud)
        rendered = BillingHTML(markup)
        self.assertNotIn("[I]", rendered.options["gpu-only"])
        self.assertIn("[I]", rendered.options["cloud-only"])
        self.assertIn("[mixed]", rendered.options["same-name-mixed"])

    def test_balances_are_available_units_not_initial_allowance_or_unknown_zero(self):
        self.llm[0]["billing"].update(
            balances={"BUNDLED_CREDITS": 0.8652965, "USD": 5, "DIEM": 0})
        rendered = BillingHTML(self.c.get("/ui/backends").text)
        text = " ".join(rendered.text)
        self.assertIn("86.52965 credits", text)
        self.assertIn("cash USD 5", text)
        self.assertIn("DIEM 0", text)
        self.assertNotIn("500 credits", text)
        self.assertIn("bundled credits unavailable", text)
        self.assertIn("API billing unknown", text)

    def test_failed_account_refresh_removes_previous_included_assurance(self):
        """A failed optional account query must not leave last poll's free API sign."""
        first = BillingHTML(self.c.get(
            "/ui/playground?sub=chat&backend=included&model=included-only").text)
        self.assertIn("[I]", first.options["included-only"])
        self.llm[1]["billing"] = {"state": "unknown", "balances": {}, "error": "HTTP 401"}
        second = BillingHTML(self.c.get(
            "/ui/playground?sub=chat&backend=included&model=included-only").text)
        self.assertIn("[?]", second.options["included-only"])
        self.assertNotIn("[I]", second.options["included-only"])
        self.assertIn("API billing unknown", second.context["cpbilling"])
        self.llm[:] = [self.llm[1]]
        text = " ".join(BillingHTML(self.c.get("/ui/backends").text).text)
        self.assertIn("bundled credits unavailable", text)
        self.assertIn("cash USD unavailable", text)
        self.assertIn("DIEM unavailable", text)
        self.assertNotIn("0 credits", text)
        self.assertNotIn("cash USD 0", text)


