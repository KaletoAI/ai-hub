"""Guard NanoGPT console saves against lost exact model IDs, malformed defaults,
wrong modality after an endpoint switch, and secrets reflected by refused forms.
These failures otherwise dispatch plausible paid generations with unintended settings.
"""
import html
import base64
import io
import json
import os
from html.parser import HTMLParser
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image

from tests.test_ui_form_validation import _Fixture
import admin
import main
import adapters
import nanogpt
import store
import jobs


class NanoBackends(_Fixture):
    def test_unified_provider_autofills_and_forces_paid_without_reflecting_key(self):
        form = {"name": "nano", "type": "nanogpt", "url": "", "api_key": "provider-secret"}
        response = self.post("/ui/backends/save", form)
        self.assertEqual(response.status_code, 303, response.text)
        saved = store.get_backend("nano", "nanogpt")
        self.assertEqual(saved["url"], nanogpt.URL)
        self.assertTrue(saved["paid"])
        response = self.post("/ui/backends/save", {
            **form, "orig": "nanogpt:nano", "max_concurrent": "1.5"})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("provider-secret", response.text)
        self.assertEqual(store.get_backend("nano", "nanogpt"), saved)

    def test_switching_from_provider_autofills_but_arbitrary_openai_url_is_untouched(self):
        response = self.post("/ui/backends/save", {
            "name": "nano", "type": "nanogpt", "url": "https://api.meshy.ai"})
        self.assertEqual(response.status_code, 303)
        self.assertEqual(store.get_backend("nano", "nanogpt")["url"], nanogpt.URL)
        response = self.post("/ui/backends/save", {
            "name": "custom", "type": "openai", "url": "https://custom.example/v1"})
        self.assertEqual(response.status_code, 303)
        self.assertEqual(store.get_backend("custom", "openai")["url"], "https://custom.example/v1")

    def test_text_filter_and_sampling_options_survive_save(self):
        response = self.post("/ui/backends/save", {
            "name": "nano-text", "type": "nanogpt", "url": "",
            "models_allow": "gpt-*", "chat_only": "1", "smp_temperature": "0.7"})
        self.assertEqual(response.status_code, 303, response.text)
        saved = store.get_backend("nano-text", "nanogpt")
        self.assertEqual(saved["models_allow"], "gpt-*")
        self.assertTrue(saved["chat_only"])
        self.assertEqual(saved["sampling_defaults"]["temperature"], 0.7)

    def test_media_timing_numbers_are_not_silently_truncated_or_unset(self):
        for field, value in (("cloud_max_wait", "10.5"), ("cloud_poll_interval", "broken")):
            with self.subTest(field=field):
                response = self.post("/ui/backends/save", {
                    "name": "timing", "type": "nanogpt", "url": "", field: value})
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(f'name="{field}" value="{value}"', response.text)
                self.assertIsNone(store.get_backend("timing", "nanogpt"))

    def test_one_backend_editor_exposes_text_and_media_with_one_key(self):
        response = self.c.get("/ui/backends?new=1&type=nanogpt&url=" + nanogpt.URL)
        controls = _Controls(response.text)
        self.assertEqual(controls.types.count("nanogpt"), 1)
        self.assertNotIn("nanogpt_media", controls.types)
        for name in ("chat_only", "smp_temperature", "cloud_max_wait", "cloud_poll_interval"):
            self.assertIn(name, controls.fields)
        self.assertEqual(controls.names.count("api_key"), 1)


class _Controls(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.names, self.fields, self.options, self.groups, self.types = [], {}, [], [], []
        self.selected = {}
        self.select = ""
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
            if self.select == "type":
                self.types.append(attrs.get("value"))
        if tag == "optgroup":
            self.groups.append(attrs.get("label"))

    def handle_endtag(self, tag):
        if tag == "select":
            self.select = ""


class NanoDirectPlayground(_Fixture):
    """Catalog-only media must remain usable and obey the selected model's real schema."""

    def setUp(self):
        super().setUp()
        self.cand = {
            "backend": "nano", "model": "qwen/image-turbo", "task": "text2img",
            "nanogpt": {"endpoint": "images", "options": {}},
            "model_metadata": {"capabilities": {"image_to_image": True},
                               "supported_parameters": {"max_input_images": 3, "max_output_images": 4,
                                                        "resolutions": ["1k", "2k"],
                                                        "quality": ["standard", "hd"]}},
        }
        self.rows = [{"id": "nano/qwen/image-turbo", "name": "Qwen Turbo", "backend": "nano",
                      "endpoint": "images", "candidate": self.cand}]
        for name, fn in (("_generation_models", lambda: self.rows),
                         ("_generation_candidate", lambda model: self.cand
                          if model == "nano/qwen/image-turbo" else None)):
            p = patch.object(admin, name, fn, create=True)
            p.start()
            self.addCleanup(p.stop)
        self.stash = dict(admin._pg_images)
        admin._pg_images.clear()
        self.addCleanup(lambda: (admin._pg_images.clear(), admin._pg_images.update(self.stash)))

    def test_empty_aliases_offer_catalog_model_and_bounded_form(self):
        response = self.c.get("/ui/playground?sub=media")
        self.assertEqual(response.status_code, 200)
        controls = _Controls(response.text)
        self.assertIn("nano/qwen/image-turbo", controls.options)
        self.assertIn("nano · Images", controls.groups)
        self.assertIn("p__prompt", controls.fields)
        refs = [name for name in controls.names if name.startswith("img__")]
        self.assertEqual(len(refs), 3)
        self.assertEqual(controls.fields["p__n"]["min"], "1")
        self.assertEqual(controls.fields["p__n"]["max"], "4")
        self.assertEqual(controls.fields["p__resolution"]["name"], "p__resolution")
        self.assertIn("2k", controls.options)
        self.assertIn("hd", controls.options)
        self.assertNotIn("Register an alias", response.text)

    def test_model_change_drops_choices_and_counts_outside_new_schema(self):
        """A model switch must not offer the old model's invalid paid controls."""
        cand = {**self.cand, "model": "text-only",
                "model_metadata": {"capabilities": {"image_to_image": False},
                                   "supported_parameters": {"resolutions": ["1k"], "max_output_images": 1}}}
        self.rows.append({"id": "nano/text-only", "name": "Text only", "backend": "nano",
                          "endpoint": "images", "candidate": cand})
        page = self.c.get("/ui/playground?sub=media&model=nano/text-only"
                          "&p__prompt=keep+the+prompt&p__resolution=2k&p__n=4")
        controls = _Controls(page.text)
        self.assertNotIn("2k", controls.options)
        self.assertNotEqual(controls.fields["p__n"].get("value"), "4")
        self.assertIn("keep the prompt", page.text)

    def test_text_only_model_has_no_refs_and_drops_previous_uploads(self):
        self.cand["model_metadata"] = {"capabilities": {"image_to_image": False},
                                      "supported_parameters": {"max_input_images": 0}}
        admin._pg_images["default"] = {"input_image": ("old.png", b"old-image")}
        page = self.c.get("/ui/playground?sub=media")
        self.assertFalse(any(n.startswith("img__") for n in _Controls(page.text).names))
        sent = []

        async def api(request, method, path, **kwargs):
            self.assertEqual((method, path), ("POST", "/v1/generations"))
            sent.append(kwargs["json"])
            return httpx.Response(202, json={"job_id": "direct-job"})

        with patch.object(admin, "_self_api", api):
            response = self.post("/ui/playground/generate", {
                "model": "nano/qwen/image-turbo", "p__prompt": "a bird"})
        self.assertEqual(response.status_code, 303)
        self.assertEqual(sent[0]["model"], "nano/qwen/image-turbo")
        self.assertEqual(sent[0]["prompt"], "a bird")
        self.assertNotIn("images", sent[0])
        self.assertIn("job=direct-job", response.headers["location"])

    def test_native_boolean_false_is_not_replaced_by_provider_default(self):
        """Unchecked/default controls must not turn an explicit false into an omitted paid option."""
        self.cand["model_metadata"] = {
            "max_input_images": 0,
            "supported_parameters": {"enhance_prompt": {"type": "boolean", "enum": [True, False],
                                                        "default": True}},
        }
        page = self.c.get("/ui/playground?sub=media&p__enhance_prompt=false")
        self.assertEqual(_Controls(page.text).selected["p__enhance_prompt"], "false")
        sent = []

        async def api(request, method, path, **kwargs):
            sent.append(kwargs["json"])
            return httpx.Response(202, json={"job_id": "boolean-job"})

        with patch.object(admin, "_self_api", api):
            response = self.post("/ui/playground/generate", {
                "model": "nano/qwen/image-turbo", "p__prompt": "a bird",
                "p__enhance_prompt": "false"})
        self.assertEqual(response.status_code, 303)
        self.assertIs(sent[0]["params"]["enhance_prompt"], False)

    def test_aliases_and_direct_models_coexist_and_use_current_metadata(self):
        store.upsert("saved-image", [{**self.cand, "model_metadata": {"max_input_images": 16}}])
        with patch.object(admin, "_generation_candidate", lambda model: self.cand):
            response = self.c.get("/ui/playground?sub=media&model=saved-image")
        controls = _Controls(response.text)
        self.assertIn("saved-image", controls.options)
        self.assertIn("nano/qwen/image-turbo", controls.options)
        self.assertIn("Aliases", controls.groups)
        self.assertEqual(len([n for n in controls.names if n.startswith("img__")]), 3)

    def test_direct_error_keeps_picker_prompt_and_uploads(self):
        async def api(request, method, path, **kwargs):
            return httpx.Response(400, json={"detail": "invalid native choice"})

        with patch.object(admin, "_self_api", api):
            response = self.post("/ui/playground/generate", {
                "model": "nano/qwen/image-turbo", "p__prompt": "keep this prompt"},
                files={"img__input_image": ("ref.png", b"image-bytes", "image/png")})
        self.assertIn("nano/qwen/image-turbo", _Controls(response.text).options)
        self.assertIn("keep this prompt", response.text)
        self.assertIn("✓ kept", response.text)
        self.assertIn("invalid native choice", response.text)


class NanoAliases(_Fixture):
    def setUp(self):
        super().setUp()
        self.backend = {"name": "nano-media", "type": nanogpt.KIND, "url": nanogpt.URL}
        p = patch.object(admin, "_gen_backends", lambda: [self.backend])
        p.start()
        self.addCleanup(p.stop)
        self.cand = nanogpt.default_candidate(self.backend["name"])
        self.cand["model"] = "original-account-model"
        store.upsert("nano-alias", [self.cand])

    def test_alias_editor_suggests_discovered_ids_and_current_schema_without_changing_alias(self):
        """A stored alias must use catalog suggestions and limits, not stale generic refs."""
        metadata = {"capabilities": {"image_to_image": True},
                    "supported_parameters": {"max_input_images": 3, "max_output_images": 4,
                                             "resolutions": ["1k", "2k"]}}
        catalog_cand = {**self.cand, "model_metadata": metadata}
        row = {"id": "nano-media/original-account-model", "name": "Account image model",
               "backend": "nano-media", "endpoint": "images", "candidate": catalog_cand}
        with patch.object(admin, "_generation_models", lambda: [row], create=True):
            response = self.c.get("/ui/aliases?sub=media&edit=nano-alias")
        self.assertEqual(response.status_code, 200)
        controls = _Controls(response.text)
        self.assertIn("original-account-model", controls.options)
        self.assertEqual(controls.fields["cloud_model"]["value"], "original-account-model")
        self.assertEqual(controls.fields["opt__n"]["max"], "4")
        self.assertIn("2k", controls.options)
        self.assertEqual(store.get("nano-alias"), [self.cand])

    def test_register_chooses_all_modalities_without_a_workflow(self):
        for task, endpoint in (("text2img", "images"), ("img2vid", "video"),
                               ("text2mesh", "3d"), ("img2mesh", "3d")):
            with self.subTest(task=task):
                alias = "nano-" + task
                response = self.post("/ui/mapping/register", {
                    "alias": alias, "backend": self.backend["name"], "task": task})
                self.assertEqual(response.status_code, 303, response.text)
                saved = store.get(alias)[0]
                self.assertEqual(saved[nanogpt.KIND]["endpoint"], endpoint)
                self.assertEqual(saved["task"], task)
                self.assertNotIn("workflow_json", saved)

    def test_endpoint_change_recomputes_task_and_preserves_exact_model_and_defaults(self):
        for endpoint, task in (("video", "text2vid"), ("3d", "text2mesh"),
                               ("images", "text2img")):
            with self.subTest(endpoint=endpoint):
                response = self.post("/ui/mapping/cloud-update", {
                    "alias": "nano-alias", "task": "text2img", "cloud_endpoint": endpoint,
                    "cloud_model": "actual-model:version", "opt__seed": "42",
                    "opt__options_json": '{"model_control":"exact-value"}'})
                self.assertEqual(response.status_code, 303, response.text)
                saved = store.get("nano-alias")[0]
                self.assertEqual(saved["model"], "actual-model:version")
                self.assertEqual(saved["task"], task)
                self.assertEqual(saved[nanogpt.KIND]["endpoint"], endpoint)
                self.assertEqual(saved[nanogpt.KIND]["options"]["seed"], 42)
                self.assertIn("exact-value", saved[nanogpt.KIND]["options"]["options_json"])

    def test_refused_defaults_preserve_store_and_every_typed_control(self):
        for bad in ({"cloud_model": ""}, {"opt__seed": "42.5"},
                    {"opt__n": "not-a-number"}, {"opt__options_json": "{broken"},
                    {"opt__options_json": "[]"}, {"retries": "2.5"},
                    {"opt__options_json": '{"model":"unwanted-override"}'}):
            with self.subTest(bad=bad):
                form = {"alias": "nano-alias", "cloud_endpoint": "video",
                        "task": "img2vid", "cloud_model": "typed-model", "opt__seed": "7",
                        "opt__options_json": '{"typed_control":"keep me"}', **bad}
                response = self.post("/ui/mapping/cloud-update", form)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(store.get("nano-alias"), [self.cand])
                self.assertIn('value="video" selected', response.text)
                self.assertIn('value="img2vid" selected', response.text)
                self.assertIn('name="cloud_model" value="' + html.escape(form["cloud_model"]) + '"', response.text)
                self.assertIn(html.escape(form["opt__options_json"]), response.text)
                self.assertIn('name="opt__seed" value="' + form["opt__seed"] + '"', response.text)

    def test_openai_images_refuses_invalid_count_and_excess_refs(self):
        """A malformed count must not be truncated into a different paid generation."""
        routes = [(self.backend, self.cand)]
        with patch.object(main, "_gen_pick", AsyncMock(return_value=(routes, False, None))), \
                patch.object(main, "get_gen_routes", return_value=routes):
            for extra in ({"n": 1.5}, {"n": 0}, {"n": True},
                          {"ref_images": ["not-fetched"] * 17}):
                with self.subTest(extra=extra):
                    response = self.c.post("/v1/images/generations", json={
                        "model": "nano-alias", "prompt": "a bird", **extra})
                    self.assertEqual(response.status_code, 400, response.text)

    def test_discovered_output_count_limit_refuses_before_job_creation(self):
        """Model-specific image limits must be enforced before submitting a paid job."""
        ad = adapters.NanoGPTMediaAdapter(self.backend, main.adapter_ctx)
        ad.model_catalogs = {"images": {self.cand["model"]: {
            "supported_parameters": {"max_output_images": 2}}}}
        routes = [(self.backend, self.cand)]
        with patch.object(main, "_gen_pick", AsyncMock(return_value=(routes, False, None))), \
                patch.object(main, "get_gen_routes", return_value=routes), \
                patch.dict(main.backend_adapters, {main.backend_id(self.backend): ad}):
            response = self.c.post("/v1/images/generations", json={
                "model": "nano-alias", "prompt": "a bird", "n": 3})
            self.assertEqual(response.status_code, 400, response.text)

    def test_image_defaults_and_explicit_auto_produce_requested_outputs(self):
        """The image shim must not buy one image or force alias size when the caller requested otherwise."""
        saved_jobs = {key: getattr(jobs, key) for key in ("_DB_PATH", "_BLOB_DIR", "_DEFAULT_TTL", "_active")}
        self.addCleanup(lambda: [setattr(jobs, key, value) for key, value in saved_jobs.items()])
        jobs.init(os.path.join(self.tmp.name, "jobs.db"), os.path.join(self.tmp.name, "blobs"))
        self.cand[nanogpt.KIND]["options"] = {"n": 2, "resolution": "2k"}
        backend = {**self.backend, "url": "https://provider.test/api"}
        ad = adapters.NanoGPTMediaAdapter(backend, main.adapter_ctx)
        routes = [(backend, self.cand)]

        def provider(request):
            body = json.loads(request.content)
            buf = io.BytesIO()
            Image.new("RGB", (6, 4) if body["size"] == "auto" else (3, 3)).save(buf, format="PNG")
            return httpx.Response(200, json={"data": [
                {"b64_json": base64.b64encode(buf.getvalue()).decode()} for _ in range(body["n"])]})

        client = httpx.AsyncClient
        factory = lambda **kwargs: client(transport=httpx.MockTransport(provider), **kwargs)
        with patch.object(main, "_gen_pick", AsyncMock(return_value=(routes, False, None))), \
                patch.object(main, "get_gen_routes", return_value=routes), \
                patch.dict(main.backend_adapters, {main.backend_id(backend): ad}), \
                patch.object(main, "backend_inflight", {}), \
                patch.object(main, "gen_speed", {}), \
                patch.object(main, "gen_exec_faults", {}), \
                patch.object(main, "gen_progress", {}), \
                patch("adapters.httpx.AsyncClient", factory):
            for controls, count, dimensions in (({"size": "auto"}, 2, (6, 4)),
                                                ({"n": 1}, 1, (3, 3)),
                                                ({"n": 1, "params": {"extra": {"n": 3}}}, 1, (3, 3)),
                                                ({"params": {"extra": {"n": 3}}}, 3, (3, 3))):
                with self.subTest(controls=controls):
                    response = self.c.post("/v1/images/generations", json={
                        "model": "nano-alias", "prompt": "a bird", "response_format": "b64_json",
                        **controls})
                    self.assertEqual(response.status_code, 200, response.text)
                    images = response.json()["data"]
                    self.assertEqual(len(images), count)
                    for image in images:
                        with Image.open(io.BytesIO(base64.b64decode(image["b64_json"]))) as decoded:
                            self.assertEqual(decoded.size, dimensions)

    def test_playground_preserves_video_audio_upload_mime_and_filters_other_slots(self):
        self.cand[nanogpt.KIND]["endpoint"] = "video"
        self.cand["task"] = "img2vid"
        store.upsert("nano-alias", [self.cand])
        _, _, files = nanogpt.public_fields(self.cand)
        video = next(x["name"] for x in files if "video" in x["name"].lower())
        audio = next(x["name"] for x in files if "audio" in x["name"].lower())
        sent = []

        async def capture(request, method, path, **kwargs):
            sent.append(kwargs["json"])
            return httpx.Response(202, json={"job_id": "accepted"})

        old = dict(admin._pg_images)
        admin._pg_images.clear()
        self.addCleanup(lambda: (admin._pg_images.clear(), admin._pg_images.update(old)))
        with patch.object(admin, "_self_api", capture):
            response = self.post("/ui/playground/generate", {
                "model": "nano-alias", "p__prompt": "animate the subject"},
                files={f"file__{video}": ("source.mp4", b"video-bytes", "video/mp4"),
                       f"file__{audio}": ("sound.mp3", b"audio-bytes", "audio/mpeg"),
                       "img__irrelevant_old_slot": ("other.png", b"other-bytes", "image/png")})
        self.assertEqual(response.status_code, 303, response.text)
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0]["files"][video].startswith("data:video/mp4;base64,"))
        self.assertTrue(sent[0]["files"][audio].startswith("data:audio/mpeg;base64,"))
        self.assertNotIn("irrelevant_old_slot", sent[0].get("images", {}))
