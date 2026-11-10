"""wiro / LU Labs in the /ui console: a saved account gets its vendor URL, is forced
paid and never renders its key back; an alias saves every endpoint with the exact
model and no workflow; the playground form renders a catalog model's own controls
(no sample default as a value) — every one of these used to fail silently for a new
composite kind that the console did not know.
"""
from unittest.mock import patch

from tests.test_ui_form_validation import _Fixture
from tests.test_venice_console import Controls
import admin
import adapters
import lulabs
import main
import store
import wiro

SAMPLE = "https://cdn.wiro.ai/uploads/sampleinputs/sample.png"
WIRO_META = {"name": "nano", "parameters": [{"items": [
    {"id": "inputImage", "type": "combinefileinput", "defaultvalue": [SAMPLE], "maxinputlenght": 2},
    {"id": "prompt", "type": "textarea", "required": True, "defaultvalue": "sample prompt text"},
    {"id": "aspectRatio", "type": "select", "defaultvalue": "1:1",
     "options": [{"text": "1:1", "value": "1:1"}, {"text": "16:9", "value": "16:9"}]}]}]}


class Accounts(_Fixture):
    def test_saved_accounts_get_vendor_url_paid_and_hide_key(self):
        for typ, mod, key in (("wiro", wiro, "KEYPART:SECRETPART"), ("lulabs", lulabs, "lu_private")):
            with self.subTest(type=typ):
                response = self.post("/ui/backends/save", {
                    "name": typ + "-acct", "type": typ, "url": "", "api_key": key})
                self.assertEqual(response.status_code, 303, response.text)
                saved = store.get_backend(typ + "-acct", typ)
                self.assertEqual(saved["url"], mod.URL)
                self.assertTrue(saved["paid"])
                page = self.c.get(f"/ui/backends?edit={typ}:{typ}-acct")
                self.assertEqual(page.status_code, 200)
                self.assertEqual(Controls(page.text).names.count("api_key"), 1)
                for part in key.split(":"):
                    self.assertNotIn(part, page.text)

    def test_type_select_offers_both_kinds(self):
        page = self.c.get("/ui/backends?new=1")
        options = Controls(page.text).options
        self.assertIn("wiro", options)
        self.assertIn("lulabs", options)


class Aliases(_Fixture):
    def test_every_endpoint_saves_exact_model_without_workflow(self):
        for mod, model in ((wiro, "google/nano-banana"), (lulabs, "flux-dev")):
            store.upsert(mod.KIND + "-alias", [mod.candidate_for(mod.KIND, model, mod.ENDPOINTS[0])])
            for endpoint in mod.ENDPOINTS:
                with self.subTest(kind=mod.KIND, endpoint=endpoint):
                    response = self.post("/ui/mapping/cloud-update", {
                        "alias": mod.KIND + "-alias", "cloud_endpoint": endpoint,
                        "cloud_model": model, "task": "text2img"})
                    self.assertEqual(response.status_code, 303, response.text)
                    saved = store.get(mod.KIND + "-alias")[0]
                    self.assertEqual(saved["model"], model)
                    self.assertEqual(saved[mod.KIND]["endpoint"], endpoint)
                    self.assertNotIn("workflow_json", saved)

    def test_playground_renders_native_controls_without_sample_values(self):
        cand = wiro.candidate_for("wiro", "google/nano-banana", "image", WIRO_META)
        params, images, files = adapters.public_fields(cand)
        self.assertEqual([i["name"] for i in images], ["inputImage", "inputImage_2"])
        names = {p["name"]: p for p in params}
        self.assertEqual(names["aspectRatio"].get("choices"), ["1:1", "16:9"])
        self.assertNotIn("sample prompt text", str(names["prompt"].get("default", "")))
        lu = lulabs.candidate_for("lulabs", "flux-dev", "image",
                                  {"id": "flux-dev", "kind": "image", "_ops": ["generate", "edit"]})
        _, lu_images, _ = adapters.public_fields(lu)
        self.assertEqual([i["name"] for i in lu_images], ["source_image", "mask_image"])
        self.assertFalse(lu_images[0]["required"])
