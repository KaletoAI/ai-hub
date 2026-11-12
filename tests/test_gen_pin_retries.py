"""A force pin must survive the per-alias `retries` cap.

Why this file exists (it fails SILENTLY until someone pins): `_gen_routes` caps an
alias's candidates to 1+retries, ordered fastest first. The pin used to be applied
AFTER that cap, so with `retries: "0"` only the fastest backend was left and a pin
to any slower one 503'd "No healthy backend for generation model X on backend Y" —
with Y healthy and idle (2026-11-12: MiniMax-H3-Image pinned to rtx-3090 right after
one slow run there). The pin has to filter first; the cap still applies unpinned. The
waiter designation (`_entry_can_use`) must agree, or a pinned job parks until timeout.

Run: python -m unittest tests.test_gen_pin_retries -v
"""
import asyncio
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
    from fastapi import HTTPException
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

FAST = {"name": "fast", "type": "comfyui", "url": "http://fast:8188", "enabled": True, "max_concurrent": 1}
SLOW = {"name": "slow", "type": "comfyui", "url": "http://slow:8188", "enabled": True, "max_concurrent": 1}


def _wf(backend, retries):
    return {"backend": backend, "retries": retries, "workflow_json": {"1": {"class_type": "X", "inputs": {}}}}


class PinBeforeCap(unittest.TestCase):
    def setUp(self):
        self._store_active = main.store._active
        main.store._active = False
        self._saved = (main._gen_backends[:], dict(main.image_models), dict(main.backend_healthy),
                       dict(main.backend_inflight), main._gen_waiting[:], dict(main.gen_exec_faults),
                       dict(main.gen_speed), dict(main.host_controllers))
        main._gen_backends[:] = [FAST, SLOW]
        main.image_models.clear()
        main.image_models.update({"img": [_wf("slow", "0"), _wf("fast", "0")],
                                  "img1": [_wf("slow", "1"), _wf("fast", "1")]})
        main.backend_healthy.clear()
        for b in (FAST, SLOW):
            main.backend_healthy[main.backend_id(b)] = True
        main.backend_inflight.clear()
        main._gen_waiting.clear()
        main.gen_exec_faults.clear()
        main.host_controllers.clear()
        main.gen_speed.clear()
        for a in ("img", "img1"):        # both measured: slow 180 s, fast 60 s
            main.gen_speed[f"{a}|{main.backend_id(SLOW)}"] = 180.0
            main.gen_speed[f"{a}|{main.backend_id(FAST)}"] = 60.0

    def tearDown(self):
        main.store._active = self._store_active
        gb, im, bh, bi, gw, ef, gs, hc = self._saved
        main._gen_backends[:] = gb
        main.image_models.clear(); main.image_models.update(im)
        main.backend_healthy.clear(); main.backend_healthy.update(bh)
        main.backend_inflight.clear(); main.backend_inflight.update(bi)
        main._gen_waiting[:] = gw
        main.gen_exec_faults.clear(); main.gen_exec_faults.update(ef)
        main.gen_speed.clear(); main.gen_speed.update(gs)
        main.host_controllers.clear(); main.host_controllers.update(hc)

    @staticmethod
    def names(routes):
        return [b["name"] for b, _c in routes]

    def test_unpinned_cap_keeps_only_the_fastest(self):
        ready, allc = main._gen_routes("img")
        self.assertEqual(self.names(allc), ["fast"])
        self.assertEqual(self.names(ready), ["fast"])

    def test_pin_to_the_slower_backend_routes_despite_retries_0(self):
        ready, allc = main._gen_routes("img", force="slow")
        self.assertEqual(self.names(allc), ["slow"])
        routes, parked, _elig = asyncio.run(main._gen_pick("img", "slow", {}))
        self.assertFalse(parked)
        self.assertEqual(self.names(routes), ["slow"])

    def test_pin_to_a_busy_slower_backend_parks_instead_of_503(self):
        main.backend_inflight[main.backend_id(SLOW)] = 1      # at its cap of 1
        try:
            routes, parked, _elig = asyncio.run(main._gen_pick("img", "slow", {}))
        except HTTPException as e:              # the old behaviour
            self.fail(f"pinned to a busy backend must park, got {e.status_code} {e.detail}")
        self.assertTrue(parked)

    def test_waiter_designation_honours_the_pin(self):
        entry = {"alias": "img", "enqueued_at": 0.0, "force": "slow"}
        self.assertTrue(main._entry_can_use(entry, SLOW))
        self.assertFalse(main._entry_can_use(entry, FAST))
        # an unpinned waiter of the same alias in the same pass still sees the cap
        memo = {}
        self.assertTrue(main._entry_can_use(entry, SLOW, memo))
        self.assertFalse(main._entry_can_use({"alias": "img", "enqueued_at": 0.0}, SLOW, memo))

    def test_unknown_pin_still_503s_with_the_backend_named(self):
        with self.assertRaises(HTTPException) as cm:
            asyncio.run(main._gen_pick("img", "nope", {}))
        self.assertEqual(cm.exception.status_code, 503)
        self.assertIn("on backend 'nope'", cm.exception.detail)


if __name__ == "__main__":
    unittest.main()
