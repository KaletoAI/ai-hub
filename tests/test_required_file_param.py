"""A file field marked `required` refuses a request without the file — with a 400.

Why this fails SILENTLY: a ComfyUI alias's file field (`input_rig_path` of
motion-kimodo, `input_mesh_path` of a rig alias) simply kept the workflow's baked-in
value when a request sent no file. The job was accepted, queued, maybe ran for a minute
and then failed on an empty path — or, with a default path baked into the workflow,
came back `done` with a result built on a file the client never sent. The alias
contract with Anima Verse (motion-kimodo, 2026-10-12) demands a 400 instead.

So the mapping entry carries `required: true` (Mapping editor checkbox on file rows,
cleared by an unticked box like `client_path`), the schema advertises it under
`files[].required`, and run_generation refuses before any job row exists.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    import adapters
    import admin
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

WF = {"2": {"class_type": "KimodoHub_PromptLines", "inputs": {"prompt": "walk"}},
      "6": {"class_type": "KimodoHub_ExportFBX", "inputs": {"rig_path": "", "filename_prefix": "3D/k"}}}
MAP = {"prompt": {"node": "2", "field": "prompt"},
       "input_rig_path": {"node": "6", "field": "rig_path", "label": "input_rig_path",
                          "required": True}}


def cand(mapping=MAP):
    return {"backend": "k12-gpu", "workflow_json": WF, "mapping": mapping}


class Refusal(unittest.TestCase):
    def test_missing_file_is_refused(self):
        why = main._missing_required_file(cand(), {}, None)
        self.assertIn("files.input_rig_path", why)

    def test_uploaded_file_satisfies(self):
        self.assertIsNone(main._missing_required_file(
            cand(), {}, {"input_rig_path": ("rig.fbx", b"x")}))

    def test_backend_path_in_params_satisfies(self):
        # whether this caller may NAME a path is _client_param_refusal's call, not ours
        self.assertIsNone(main._missing_required_file(
            cand(), {"input_rig_path": "/in/rig.fbx"}, None))
        self.assertIsNotNone(main._missing_required_file(
            cand(), {"input_rig_path": "  "}, None))

    def test_optional_file_is_not_refused(self):
        m = {**MAP, "input_rig_path": {k: v for k, v in MAP["input_rig_path"].items()
                                       if k != "required"}}
        self.assertIsNone(main._missing_required_file(cand(m), {}, None))

    def test_schema_advertises_it(self):
        _params, _images, files = adapters.public_fields(cand())
        self.assertEqual(files, [{"name": "input_rig_path", "required": True}])


class Editor(unittest.TestCase):
    """The flag lives in the mapping, which the editor REBUILDS from the form on Save —
    a key the form does not carry is dropped. So it must be a checkbox on the row."""

    def test_file_row_offers_the_checkbox(self):
        rows = admin._req_fields_rows("a", WF, MAP, {})
        self.assertIn('name="required__input_rig_path"', rows)
        self.assertNotIn('name="required__prompt"', rows)
        self.assertIn("checked", rows.split('name="required__input_rig_path"')[1].split(">")[0])

    def _save(self, form):
        c = cand({**MAP})
        base = {"node__input_rig_path": "6", "field__input_rig_path": "rig_path",
                "label__input_rig_path": "input_rig_path"}
        with mock.patch.object(admin.store, "get", lambda alias: None):
            admin._apply_update_form([c], {**base, **form})
        return c["mapping"]["input_rig_path"]

    def test_save_reads_the_checkbox(self):
        self.assertTrue(self._save({"required__input_rig_path": "on"}).get("required"))
        self.assertNotIn("required", self._save({}))             # unticked = cleared


if __name__ == "__main__":
    unittest.main()
