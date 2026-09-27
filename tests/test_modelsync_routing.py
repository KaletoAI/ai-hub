"""Generation routing gated on per-alias model-sync readiness (Thunder backends).

Why this file exists (it fails SILENTLY): a Thunder instance syncs exactly the models
its aliases need, and until an alias's files are all there the backend is healthy,
free and — to every other routing signal — a perfect candidate. Routed there anyway,
the job runs a workflow whose weights are missing and comes back as a plausible
ComfyUI "value not in list" error, or it parks for hours behind a sync nobody named.
So a not-ready alias must leave BOTH lists (`ready` and `allc`) of `_gen_routes`, the
waiter designation must inherit that, a force pin must not bypass it, the chain's
path-relay successor (read straight from the store, not via `_gen_routes`) must be
judged too, and the 503 must say WHY — the sync progress — instead of "no healthy
backend" about a backend that is up.

Run: python -m unittest tests.test_modelsync_routing -v
"""
import asyncio
import os
import sys
import tempfile
import unittest

# `import main` reads ./config.yaml at import time — give it a minimal one in a temp cwd.
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (tests/ is one level down)
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    from fastapi import HTTPException
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


class _FakeCtl:
    """A controller's routing face: in-memory readiness per alias, and a status text.
    Counts the calls so a test can see the gate asked the controller at all."""

    def __init__(self, name, ready=(), texts=None):
        self.name = name
        self.ready = set(ready)
        self.texts = dict(texts or {})
        self.asked = []

    def is_alias_ready(self, alias):
        self.asked.append(alias)
        return alias in self.ready

    def alias_status(self, alias):
        return self.texts.get(alias, f"models for {alias} are not planned on {self.name} yet")


THUNDER = {"name": "thunder", "type": "comfyui", "url": "http://127.0.0.1:18188", "enabled": True}
THUNDER2 = {"name": "thunder2", "type": "comfyui", "url": "http://127.0.0.1:18189", "enabled": True}
GPU = {"name": "gpu", "type": "comfyui", "url": "http://gpu:8188", "enabled": True}
SYNC_TEXT = "models for img on thunder are syncing on thunder (12.3 of 31.0 GB)"


def _wf(backend):
    return {"backend": backend, "workflow_json": {"1": {"class_type": "X", "inputs": {}}}}


class _Base(unittest.TestCase):
    def setUp(self):
        # _gen_routes reads the store when it is active — another test module may have left
        # it pointing at a temp DB that is gone; the config fallback is what we set up here
        self._store_active = main.store._active
        main.store._active = False
        self._saved = (main._gen_backends[:], dict(main.image_models), dict(main.backend_healthy),
                       dict(main.thunder_controllers), dict(main.backend_inflight),
                       main._gen_waiting[:], dict(main.gen_exec_faults))
        main._gen_backends[:] = [THUNDER, THUNDER2, GPU]
        main.image_models.clear()
        main.image_models.update({
            "img": [_wf("thunder"), _wf("gpu")],        # mixed alias
            "only": [_wf("thunder")],                   # Thunder-only alias
            "both": [_wf("thunder"), _wf("thunder2")],  # two Thunder backends
            "rig": [_wf("thunder"), _wf("gpu")],        # a chain successor
        })
        main.backend_healthy.clear()
        for b in (THUNDER, THUNDER2, GPU):
            main.backend_healthy[main.backend_id(b)] = True
        main.backend_inflight.clear()
        main._gen_waiting.clear()
        main.gen_exec_faults.clear()
        main.thunder_controllers.clear()
        self.ctl = main.thunder_controllers["thunder"] = _FakeCtl(
            "thunder", ready=(), texts={"img": SYNC_TEXT})

    def tearDown(self):
        main.store._active = self._store_active
        gb, im, bh, tc, bi, gw, ef = self._saved
        main._gen_backends[:] = gb
        main.image_models.clear(); main.image_models.update(im)
        main.backend_healthy.clear(); main.backend_healthy.update(bh)
        main.thunder_controllers.clear(); main.thunder_controllers.update(tc)
        main.backend_inflight.clear(); main.backend_inflight.update(bi)
        main._gen_waiting[:] = gw
        main.gen_exec_faults.clear(); main.gen_exec_faults.update(ef)

    @staticmethod
    def names(routes):
        return [b["name"] for b, _c in routes]

    def pick_error(self, alias, force=""):
        with self.assertRaises(HTTPException) as cm:
            asyncio.run(main._gen_pick(alias, force, {}))
        return cm.exception


class GateFunction(_Base):
    def test_backend_without_controller_never_gated(self):
        self.assertIsNone(main.modelsync_gate(GPU, "img"))
        self.assertIsNone(main.modelsync_gate(THUNDER2, "img"))

    def test_same_named_llm_backend_never_gated(self):
        # backends are keyed (name, type): an LLM backend called "thunder" is not the box
        self.assertIsNone(main.modelsync_gate({"name": "thunder", "type": "openai"}, "img"))
        self.assertEqual(self.ctl.asked, [])

    def test_not_ready_alias_gets_the_controller_text(self):
        self.assertEqual(main.modelsync_gate(THUNDER, "img"), SYNC_TEXT)

    def test_ready_alias_is_routable(self):
        self.ctl.ready.add("img")
        self.assertIsNone(main.modelsync_gate(THUNDER, "img"))


class GenRoutes(_Base):
    def test_incomplete_alias_not_routed_to_synced_backend(self):
        ready, allc = main._gen_routes("img")
        self.assertEqual(self.names(ready), ["gpu"])
        self.assertEqual(self.names(allc), ["gpu"])       # out of allc too: nothing parks for it
        ready, allc = main._gen_routes("only")
        self.assertEqual((ready, allc), ([], []))
        self.assertIn("img", self.ctl.asked)

    def test_ready_alias_routes_to_thunder(self):
        self.ctl.ready.add("img")
        ready, allc = main._gen_routes("img")
        self.assertEqual(sorted(self.names(allc)), ["gpu", "thunder"])
        self.assertEqual(sorted(self.names(ready)), ["gpu", "thunder"])

    def test_other_backends_unaffected(self):
        # the readiness is PER ALIAS: "img" ready does not open "only", and a backend
        # without a controller routes whatever the Thunder box says
        self.ctl.ready.add("img")
        self.assertEqual(main._gen_routes("only"), ([], []))
        self.ctl.ready.clear()
        _r, allc = main._gen_routes("both")
        self.assertEqual(self.names(allc), ["thunder2"])

    def test_gated_texts_are_reported(self):
        gated = []
        main._gen_routes("img", gated)
        self.assertEqual(gated, [("thunder", SYNC_TEXT)])

    def test_successor_via_get_gen_routes_is_gated(self):
        # the upload relay resolves the successor with get_gen_routes
        self.assertEqual(self.names(main.get_gen_routes("rig")), ["gpu"])
        self.ctl.ready.add("rig")
        self.assertEqual(sorted(self.names(main.get_gen_routes("rig"))), ["gpu", "thunder"])

    def test_entry_can_use_follows_gate(self):
        entry = {"alias": "img", "enqueued_at": 0.0}
        self.assertFalse(main._entry_can_use(entry, THUNDER))
        self.assertTrue(main._entry_can_use(entry, GPU))
        self.ctl.ready.add("img")
        self.assertTrue(main._entry_can_use(entry, THUNDER))


class NoBackend503(_Base):
    def test_503_text_names_sync_progress(self):
        e = self.pick_error("only")
        self.assertEqual(e.status_code, 503)
        self.assertEqual(e.detail, "models for only are not planned on thunder yet")
        # a mixed alias whose other backend is down: the gate is the reason that matters
        main.backend_healthy[main.backend_id(GPU)] = False
        e = self.pick_error("img")
        self.assertEqual(e.detail, SYNC_TEXT)

    def test_several_gates_are_joined(self):
        main.thunder_controllers["thunder2"] = _FakeCtl(
            "thunder2", texts={"both": "models for both are blocked on thunder2: no source"})
        e = self.pick_error("both")
        self.assertEqual(e.detail, "models for both are not planned on thunder yet; "
                                   "models for both are blocked on thunder2: no source")

    def test_without_a_gate_the_message_is_unchanged(self):
        main.backend_healthy[main.backend_id(GPU)] = False
        self.ctl.ready.add("img")
        main.backend_healthy[main.backend_id(THUNDER)] = False
        e = self.pick_error("img")
        self.assertEqual(e.detail, "No healthy backend for generation model 'img'")

    def test_mixed_alias_routes_to_the_other_backend(self):
        routes, parked, _elig = asyncio.run(main._gen_pick("img", "", {}))
        self.assertFalse(parked)
        self.assertEqual(self.names(routes), ["gpu"])

    def test_force_pin_does_not_bypass_the_gate(self):
        e = self.pick_error("img", force="thunder")
        self.assertEqual(e.status_code, 503)
        self.assertEqual(e.detail, SYNC_TEXT)

    def test_force_pin_elsewhere_ignores_other_gates(self):
        main.backend_healthy[main.backend_id(GPU)] = False
        e = self.pick_error("img", force="gpu")
        self.assertEqual(e.detail, "No healthy backend for generation model 'img' on backend 'gpu'")


class ChainPathRelaySuccessor(_Base):
    """The path relay pins stage 2 to stage 1's backend and reads the successor's
    candidate from the store — not through _gen_routes — so the gate is applied there
    explicitly: stage 1 ready on the Thunder box must not hand a mesh to a rigger
    whose weights are still downloading."""

    def test_successor_not_ready_on_the_stage1_backend_is_a_skip_reason(self):
        self.ctl.texts["rig"] = "models for rig are syncing on thunder (1.0 of 2.0 GB)"
        s2, why = main._chain_successor_on(THUNDER, "rig")
        self.assertIsNone(s2)
        self.assertEqual(why, "successor: models for rig are syncing on thunder (1.0 of 2.0 GB)")

    def test_successor_ready_is_resolved(self):
        self.ctl.ready.add("rig")
        s2, why = main._chain_successor_on(THUNDER, "rig")
        self.assertIsNone(why)
        self.assertEqual(s2["backend"], "thunder")

    def test_backend_without_controller_unchanged(self):
        s2, why = main._chain_successor_on(GPU, "rig")
        self.assertIsNone(why)
        self.assertEqual(s2["backend"], "gpu")

    def test_unconfigured_successor_keeps_its_message(self):
        s2, why = main._chain_successor_on(THUNDER2, "rig")
        self.assertIsNone(s2)
        self.assertEqual(why, "successor 'rig' is not configured for backend 'thunder2'")


if __name__ == "__main__":
    unittest.main()
