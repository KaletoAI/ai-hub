"""Backend names are unique only PER TYPE (the store keys backends by (name, type)),
so a ComfyUI backend and a Meshy backend may both be called "gpu". Generation now spans
both types, and a bare-name lookup would happily hand a Meshy alias the GPU box — the
job then fails opaquely inside the wrong adapter. These tests pin the two lookups that
decide the kind: main._gen_backend_for (routing/cancel) and admin._same_kind (the editor's
"which backends may I add" filter).

Run: python -m unittest tests.test_gen_backend_for -v
"""
import os
import sys
import tempfile
import unittest

# `import main` reads ./config.yaml at import time — give it a minimal one in a temp cwd.
# The dir is needed ONLY for that import, so it is removed in the same finally that
# restores the cwd; leaving it to the finalizer raises a ResourceWarning under -W error.
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (tests/ is one level down)
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    import admin
    import adapters
    import meshy
    import tripo
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


COMFY = {"name": "gpu", "type": "comfyui"}
MESHY = {"name": "gpu", "type": "meshy"}
OTHER = {"name": "gpu2", "type": "comfyui"}
POOL = [COMFY, MESHY, OTHER]

MESHY_CAND = {"backend": "gpu", "meshy": {"task": "img2mesh"}}
WF_CAND = {"backend": "gpu", "workflow_json": {"1": {}}}


class TestGenBackendFor(unittest.TestCase):
    def test_meshy_candidate_picks_the_meshy_backend(self):
        self.assertIs(main._gen_backend_for("gpu", MESHY_CAND, POOL), MESHY)

    def test_workflow_candidate_picks_the_comfy_backend(self):
        self.assertIs(main._gen_backend_for("gpu", WF_CAND, POOL), COMFY)

    def test_order_does_not_decide(self):
        rev = list(reversed(POOL))
        self.assertIs(main._gen_backend_for("gpu", MESHY_CAND, rev), MESHY)
        self.assertIs(main._gen_backend_for("gpu", WF_CAND, rev), COMFY)

    def test_no_backend_of_that_kind_is_none(self):
        self.assertIsNone(main._gen_backend_for("gpu", MESHY_CAND, [COMFY, OTHER]))
        self.assertIsNone(main._gen_backend_for("gpu2", MESHY_CAND, POOL))

    def test_missing_candidate_means_workflow_kind(self):
        # cancel of a legacy row without a candidate: treat it as the workflow kind.
        self.assertIs(main._gen_backend_for("gpu", None, POOL), COMFY)

    def test_pool_defaults_to_the_enabled_generation_backends(self):
        prev = list(main._gen_backends)
        main._gen_backends[:] = POOL
        try:
            self.assertIs(main._gen_backend_for("gpu", MESHY_CAND), MESHY)
        finally:
            main._gen_backends[:] = prev


class TestSameKind(unittest.TestCase):
    def setUp(self):
        self._prev = admin._gen_backends
        admin._gen_backends = lambda: list(POOL)

    def tearDown(self):
        admin._gen_backends = self._prev

    def test_meshy_alias_accepts_only_the_meshy_backend(self):
        self.assertTrue(admin._same_kind([MESHY_CAND], "gpu"))
        self.assertFalse(admin._same_kind([MESHY_CAND], "gpu2"))

    def test_workflow_alias_accepts_only_the_comfy_backend(self):
        self.assertTrue(admin._same_kind([WF_CAND], "gpu"))
        self.assertTrue(admin._same_kind([WF_CAND], "gpu2"))

    def test_meshy_alias_rejects_a_same_named_comfy_only_pool(self):
        admin._gen_backends = lambda: [COMFY, OTHER]
        self.assertFalse(admin._same_kind([MESHY_CAND], "gpu"))

    def test_unknown_backend(self):
        self.assertFalse(admin._same_kind([WF_CAND], "nope"))


class MainKindNeutral(unittest.TestCase):
    """Meshy is no longer the only cloud kind: every kind decision in main must read the
    candidate's/backend's kind (adapters.cand_kind / backend_kind / cloud_kind) and name
    the vendor from the exception, never a hardcoded "meshy"/"Meshy"."""

    def test_chain_mesh_param_error_for_cloud_successors(self):
        s2 = tripo.default_candidate("tripo")
        s2["tripo"]["endpoint"] = "rig"
        self.assertIsNone(main._chain_mesh_param_error(s2, "input_mesh_path", "Tripo-Rig"))
        err = main._chain_mesh_param_error(s2, "mesh_path", "Tripo-Rig")
        self.assertIsNotNone(err)
        self.assertIn("Tripo", err)
        self.assertIn("input_mesh_path", err)
        s2m = meshy.default_candidate("meshy")
        s2m["meshy"]["endpoint"] = "rigging"
        self.assertIsNone(main._chain_mesh_param_error(s2m, "input_mesh_path", "Meshy-Rig"))
        gen = tripo.default_candidate("tripo")        # an image endpoint takes no file at all
        why = main._chain_mesh_param_error(gen, "input_mesh_path", "Tripo-Object")
        self.assertIsNotNone(why)
        self.assertIn("no file input", why)

    def test_gen_backend_for_matches_kind(self):
        pool = [{"name": "gpu", "type": "comfyui"}, {"name": "gpu", "type": "tripo"},
                {"name": "gpu", "type": "meshy"}]
        self.assertEqual(main._gen_backend_for("gpu", tripo.default_candidate("gpu"), pool)["type"],
                         "tripo")
        self.assertEqual(main._gen_backend_for("gpu", meshy.default_candidate("gpu"), pool)["type"],
                         "meshy")
        self.assertEqual(main._gen_backend_for("gpu", {"workflow_json": {}}, pool)["type"],
                         "comfyui")

    def test_fault_labels_name_the_vendor(self):
        self.assertIn("Tripo", main._gen_exhausted_msg(adapters.CloudNoCredits("x", vendor="Tripo")))
        self.assertIn("Meshy", main._gen_exhausted_msg(adapters.CloudBusy("x", vendor="Meshy")))
        self.assertEqual(main._fault_label(adapters.CloudNoCredits("x", vendor="Tripo")),
                         "no credits left")
        self.assertEqual(main._fault_label(adapters.CloudBusy("x", vendor="Tripo")),
                         "Tripo queue full")


class JobViewFields(unittest.TestCase):
    """`_job_view` is the client contract (docs/mesh-client-spec.md §3.1): a field the
    spec promises at the TOP level and only meta carries is invisible to every client
    that reads the job object — and nothing fails, it is simply never there."""

    def _view(self, meta):
        import asyncio
        import types
        job = {"status": "done", "task": "img2mesh", "alias": "Tripo-Rig",
               "backend": "tripo", "error": None, "results": [], "meta": meta}
        prev = main.jobs.get
        self.addCleanup(setattr, main.jobs, "get", prev)
        main.jobs.get = lambda job_id: job
        req = types.SimpleNamespace(base_url="http://gw/")
        return asyncio.run(main._job_view("j1", req))

    def test_rig_and_rig_spec_are_lifted(self):
        view = self._view({"rig": "tripo", "rig_spec": "mixamo"})
        self.assertEqual(view["rig"], "tripo")
        self.assertEqual(view["rig_spec"], "mixamo")

    def test_absent_stays_absent(self):
        view = self._view({"rig": "generic"})
        self.assertEqual(view["rig"], "generic")
        self.assertNotIn("rig_spec", view)          # a ComfyUI rig has no bone spec to name


if __name__ == "__main__":
    unittest.main()
