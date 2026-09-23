"""What a client value may become inside a ComfyUI workflow.

Why this fails SILENTLY: `_apply_mapping` wrote whatever the client sent into
`workflow[node].inputs[field]`. In ComfyUI's API format a LIST there is not a value but
a link — `["12", 0]` rewires the input to another node's output — so a client could
re-plumb the admin's workflow, and the job just runs, differently (review 2026-09-23,
S13). And a mapped file field (`input_mesh_path` of a rig/shrink alias) took any string
as a path on the backend box: another job's output mesh, or any file the ComfyUI
process can read, delivered back as the job's result — while the job looks like an
ordinary rig of an ordinary mesh.

So: a list or object is never a mapped value (400 up front; the injector itself also
skips it, so no other path can smuggle one in), and a file field takes a client string
only from an admin (the documented "a backend path in params is for server admins"),
from the console, in bootstrap-open mode, or when the mapping entry says `client_path:
true`. Everyone else sends the file under `files`, which the gateway uploads itself.
The gateway's own chain hand-off (the stage-2 mesh path) is not a client value and is
untouched.
"""
import os
import sys
import tempfile
import types
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
    import adapters
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

WF = {"1": {"class_type": "PrimitiveString", "inputs": {"value": "/in/default.glb"}},
      "2": {"class_type": "KSampler", "inputs": {"steps": 20, "model": ["4", 0]}},
      "3": {"class_type": "Cfg", "inputs": {"opts": {"a": 1}}}}
MAP = {"value": {"node": "1", "field": "value", "label": "input_mesh_path"},
       "steps": {"node": "2", "field": "steps"},
       "opts": {"node": "3", "field": "opts"}}


class Injector(unittest.TestCase):
    def _apply(self, values):
        import copy
        wf = copy.deepcopy(WF)
        return wf, adapters._apply_mapping(wf, MAP, values)

    def test_scalars_are_applied(self):
        wf, applied = self._apply({"steps": 30})
        self.assertEqual(wf["2"]["inputs"]["steps"], 30)
        self.assertEqual(applied, {"steps": 30})

    def test_a_list_never_becomes_a_link(self):
        wf, applied = self._apply({"steps": ["4", 0]})
        self.assertEqual(wf["2"]["inputs"]["steps"], 20)
        self.assertNotIn("steps", applied)

    def test_an_object_only_where_the_workflow_holds_one(self):
        wf, applied = self._apply({"steps": {"x": 1}, "opts": {"b": 2}})
        self.assertEqual(wf["2"]["inputs"]["steps"], 20)
        self.assertEqual(wf["3"]["inputs"]["opts"], {"b": 2})


class ClientRefusal(unittest.TestCase):
    pairs = [(WF, MAP)]

    def test_list_value_is_refused(self):
        self.assertIn("single value", main._client_param_refusal({"steps": ["4", 0]}, self.pairs, True))
        self.assertIn("single value",
                      main._client_param_refusal({"extra": {"steps": [1]}}, self.pairs, True))

    def test_backend_path_is_admin_only(self):
        for key in ("input_mesh_path", "value"):
            msg = main._client_param_refusal({key: "/srv/comfy/output/other.glb"}, self.pairs, False)
            self.assertIn("files", msg, key)
        self.assertIsNone(main._client_param_refusal({"input_mesh_path": "/x.glb"}, self.pairs, True))
        self.assertIsNone(main._client_param_refusal({"steps": 12, "unknown": [1]}, self.pairs, False))
        # the file heuristic is by NAME — a number sent as text is no path
        self.assertIsNone(main._client_param_refusal({"input_mesh_path": "5000"}, self.pairs, False))
        # a bare name is refused too: ComfyUI resolves it in its input dir, where every
        # other job's uploads live
        self.assertIn("files", main._client_param_refusal({"value": "gw_abc_input.glb"},
                                                          self.pairs, False))

    def test_a_setting_is_not_a_path(self):
        # a file-NAMED param judged by its VALUE: an enum word or a model tag is no path
        for v in ("quad", "glb", "v1.0-20240301"):
            self.assertIsNone(main._client_param_refusal({"input_mesh_path": v}, self.pairs, False), v)
        for v in ("../x.glb", "C:\\x\\y.glb", "~/m.obj", "other.glb"):
            self.assertIsNotNone(main._client_param_refusal({"input_mesh_path": v}, self.pairs, False), v)

    def test_refusal_names_the_way_out(self):
        msg = main._client_param_refusal({"input_mesh_path": "/srv/x.glb"}, self.pairs, False)
        self.assertIn("files.input_mesh_path", msg)
        self.assertIn("client may send a backend path", msg)

    def test_mapping_can_allow_client_paths(self):
        m = {**MAP, "value": {**MAP["value"], "client_path": True}}
        self.assertIsNone(main._client_param_refusal({"input_mesh_path": "/x.glb"}, [(WF, m)], False))

    def test_trust_rule(self):
        saved = (main.api_key, main.users)
        try:
            main.api_key, main.users = "k", []
            req = lambda path, admin=None: types.SimpleNamespace(
                url=types.SimpleNamespace(path=path),
                state=types.SimpleNamespace(**({"gw_admin": admin} if admin is not None else {})))
            self.assertFalse(main._params_trusted(req("/v1/generations")))
            self.assertFalse(main._params_trusted(req("/v1/generations", False)))
            self.assertTrue(main._params_trusted(req("/v1/generations", True)))
            self.assertTrue(main._params_trusted(req("/ui/playground/media")))
            main.api_key = ""
            self.assertTrue(main._params_trusted(req("/v1/generations")))     # bootstrap-open
        finally:
            main.api_key, main.users = saved


class EditorCheckbox(unittest.TestCase):
    """`client_path` is set in the Mapping editor, and survives a Save both ways."""

    def setUp(self):
        import admin
        self.admin = admin

    def test_file_row_offers_the_checkbox(self):
        rows = self.admin._req_fields_rows("a", WF, MAP, {})
        self.assertIn('name="clientpath__value"', rows)
        self.assertNotIn('name="clientpath__steps"', rows)
        self.assertNotIn("checked", rows.split('name="clientpath__value"')[1].split(">")[0])
        m = {**MAP, "value": {**MAP["value"], "client_path": True}}
        rows = self.admin._req_fields_rows("a", WF, m, {})
        self.assertIn("checked", rows.split('name="clientpath__value"')[1].split(">")[0])

    def _save(self, form, stored_cp):
        cand = {"backend": "b", "workflow_json": WF,
                "mapping": {"value": {**MAP["value"], **({"client_path": True} if stored_cp else {})}}}
        base = {"node__value": "1", "field__value": "value", "label__value": "input_mesh_path"}
        from unittest import mock
        with mock.patch.object(self.admin.store, "get", lambda alias: None):
            self.admin._apply_update_form([cand], {**base, **form})
        return cand["mapping"]["value"]

    def test_save_reads_the_checkbox(self):
        self.assertTrue(self._save({"clientpath__value": "on"}, False).get("client_path"))
        self.assertNotIn("client_path", self._save({}, True))     # unticked = cleared


if __name__ == "__main__":
    unittest.main()
