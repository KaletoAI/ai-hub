"""Update workflow shows what changes before it replaces the alias's workflow.

Why this fails SILENTLY: "Update workflow" replaced `workflow_json` on the click, and a
re-export from the wrong ComfyUI tab (or an old file) just ran from then on — other
models, other sampler settings, request fields and pins pointing at nodes that no
longer exist (flagged only afterwards, and only in the editor). So the button first
asks `/ui/mapping/workflow-diff` for a node-level diff and shows it in a dialog; this
pins that the diff names what really changes, that it never writes the store, and that
the editor carries the dialog and the script without a later insert.
"""
import copy
import json
import os
import sys
import tempfile
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    import admin
    import store
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

from fastapi.testclient import TestClient  # noqa: E402

OLD = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"},
             "_meta": {"title": "Load Checkpoint"}},
       "2": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 7,
                                                   "model": ["1", 0]}},
       "3": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
       "4": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}}}
MAPPING = {"steps": {"node": "2", "field": "steps"}, "image": {"node": "3", "field": "image"},
           "cfg": {"node": "2", "field": "cfg"}}
FIXED = [{"node": "1", "field": "ckpt_name", "value": "a.safetensors"}]


class DiffTest(unittest.TestCase):
    def test_identical_workflows_have_no_changes(self):
        d = admin._workflow_diff(OLD, copy.deepcopy(OLD), MAPPING, FIXED, ["3"])
        self.assertTrue(d["same"])
        self.assertIn("No differences", admin._workflow_diff_html(d))

    def test_added_removed_and_changed_nodes(self):
        new = copy.deepcopy(OLD)
        del new["3"]                                                   # removed (bound + bypassed)
        new["5"] = {"class_type": "VAEDecode", "inputs": {}, "_meta": {"title": "Decode"}}
        new["2"]["inputs"]["steps"] = 30                               # widget value
        new["2"]["inputs"]["model"] = ["5", 0]                         # link
        del new["2"]["inputs"]["cfg"]                                  # input gone → cfg binding stale
        new["2"]["inputs"]["denoise"] = 0.5                            # input added
        new["1"]["_meta"]["title"] = "Base model"                      # title
        d = admin._workflow_diff(OLD, new, MAPPING, FIXED, ["3"])
        self.assertFalse(d["same"])
        self.assertEqual([a["id"] for a in d["added"]], ["5"])
        self.assertEqual(d["added"][0]["class"], "VAEDecode")
        self.assertEqual([r["id"] for r in d["removed"]], ["3"])
        self.assertEqual(sorted(d["removed"][0]["used_by"]), ["bypass", "request field image"])
        ch = {c["id"]: c for c in d["changed"]}
        self.assertEqual(set(ch), {"1", "2"})
        self.assertEqual(ch["1"]["title_from"], "Load Checkpoint")
        rows = {r["field"]: r for r in ch["2"]["fields"]}
        self.assertEqual((rows["steps"]["kind"], rows["steps"]["old"], rows["steps"]["new"]),
                         ("value", "20", "30"))
        self.assertEqual((rows["model"]["kind"], rows["model"]["old"], rows["model"]["new"]),
                         ("link", "1:0", "5:0"))
        self.assertEqual(rows["cfg"]["kind"], "removed")
        self.assertEqual(rows["denoise"]["kind"], "added")
        self.assertIn("request field image → node 3 (removed)", d["stale"])
        self.assertIn("request field cfg → 2.cfg (input removed)", d["stale"])
        self.assertIn("bypass → node 3 (removed)", d["stale"])
        self.assertNotIn("steps", " ".join(d["stale"]))                # still there: not stale

    def test_class_change_and_pinned_value_on_a_vanished_input(self):
        new = copy.deepcopy(OLD)
        new["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "b.safetensors"}}
        d = admin._workflow_diff(OLD, new, MAPPING, FIXED, [])
        ch = {c["id"]: c for c in d["changed"]}
        self.assertEqual(ch["1"]["class_from"], "CheckpointLoaderSimple")
        self.assertEqual(ch["1"]["class"], "UNETLoader")
        self.assertIn("pinned value → 1.ckpt_name (input removed)", d["stale"])

    def test_html_escapes_and_shortens_values(self):
        new = copy.deepcopy(OLD)
        new["2"]["inputs"]["steps"] = "<b>" + "x" * 500
        page = admin._workflow_diff_html(admin._workflow_diff(OLD, new, MAPPING, FIXED, []))
        self.assertNotIn("<b>x", page)
        self.assertIn("&lt;b&gt;", page)
        self.assertNotIn("x" * 300, page.split('title="')[0])           # cell text is shortened …
        self.assertIn("x" * 500, page)                                 # … the tooltip keeps it all


class EndpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        saved_store = (store._DB_PATH, store._active)
        self.addCleanup(lambda: (setattr(store, "_DB_PATH", saved_store[0]),
                                 setattr(store, "_active", saved_store[1])))
        store.init(os.path.join(self.tmp.name, "store.db"))
        saved_auth = (main.api_key, main.users, main._users_by_key)
        main.api_key, main.users, main._users_by_key = "", [], {}
        self.addCleanup(lambda: (setattr(main, "api_key", saved_auth[0]),
                                 setattr(main, "users", saved_auth[1]),
                                 setattr(main, "_users_by_key", saved_auth[2])))
        store.upsert("al", [{"backend": "b1", "task": "text2img", "workflow_json": OLD,
                             "mapping": MAPPING, "fixed": FIXED, "bypass": ["3"]}])
        self.client = TestClient(main.app)

    def post(self, data, files=None):
        return self.client.post("/ui/mapping/workflow-diff", data=data, files=files,
                                headers={"sec-fetch-site": "same-origin"})

    def test_diff_is_shown_and_nothing_is_written(self):
        new = copy.deepcopy(OLD)
        new["2"]["inputs"]["steps"] = 30
        # the editor form rides along (the button submits the whole form): its typed
        # steps label must NOT be saved by the preview
        r = self.post({"alias": "al", "new_alias": "al", "node__steps": "2", "field__steps": "steps",
                       "label__steps": "typed"},
                      files={"workflow_file": ("wf.json", json.dumps(new).encode(), "application/json")})
        self.assertEqual(r.status_code, 200)
        self.assertIn("KSampler", r.text)
        self.assertIn("30", r.text)
        c = store.get("al")[0]
        self.assertEqual(c["workflow_json"], OLD)
        self.assertNotIn("label", c["mapping"]["steps"])

    def test_unsaved_editor_bindings_count_for_stale(self):
        # A request field re-pointed in the form (not yet saved) is judged as typed.
        new = copy.deepcopy(OLD)
        del new["3"]
        r = self.post({"alias": "al", "new_alias": "al", "node__image": "4", "field__image": "images",
                       "node__steps": "2", "field__steps": "steps"},
                      files={"workflow_file": ("wf.json", json.dumps(new).encode(), "application/json")})
        self.assertEqual(r.status_code, 200)
        self.assertNotIn("request field image", r.text)

    def test_bad_input_is_refused_with_the_reason(self):
        r = self.post({"alias": "al"},
                      files={"workflow_file": ("wf.json", b"{nope", "application/json")})
        self.assertEqual(r.status_code, 400)
        self.assertIn("invalid workflow JSON", r.text)
        self.assertEqual(self.post({"alias": "al"}).status_code, 400)      # no file, no path
        self.assertEqual(self.post({"alias": "zz"}).status_code, 404)

    def test_editor_carries_dialog_and_script(self):
        r = self.client.get("/ui/aliases?edit=al", headers={"sec-fetch-site": "same-origin"})
        self.assertEqual(r.status_code, 200)
        self.assertIn('id="gw-wfdiff"', r.text)
        self.assertIn("data-wfdiff", r.text)
        self.assertIn("/ui/mapping/workflow-diff", r.text)


if __name__ == "__main__":
    unittest.main()
