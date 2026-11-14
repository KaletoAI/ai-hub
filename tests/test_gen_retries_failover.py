"""An alias's `retries` limits the FAILOVER, never which backend a job may start on.

Why this file exists (it fails SILENTLY — the job just waits): `_gen_routes` used to cut
an alias's candidates to the 1+retries FASTEST before anything else looked at them. A
free, slower backend therefore never existed for the routing: LoRA eligibility was
decided on the cut list (`_gen_pick`), the job parked with that `eligible` set, and the
queue (`_entry_can_use`, `power_waiting_jobs`) refused the idle backend for good.
Measured 2026-11-14 on prod: Qwen2.1 (`retries: "1"`, candidates rtx-3090 ~90 s,
dx10-02 ~120 s, rtx-4070 ~230 s) with a client LoRA — job 98ed4ce0 waited for dx10-02
while the rtx-4070 sat healthy, free and holding the LoRA. Without a LoRA the same alias
did land on the 4070, which is why it looked like a LoRA problem.

Now the lists stay whole and `_run_job` stops after 1+retries backends have RUN the
job (`tried`, which survives a lost claim and a re-park).

Run: venv/bin/python -m unittest tests.test_gen_retries_failover -v
"""
import asyncio
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
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

LORA = "Qwen2.1-thighgap.safetensors"


def _bk(name):
    return {"name": name, "type": "comfyui", "url": f"http://{name}:8188", "enabled": True,
            "max_concurrent": 1}


B3090, BDX, B4070 = _bk("rtx-3090"), _bk("dx10-02"), _bk("rtx-4070")


def _cand(backend, retries="1"):
    return {"backend": backend, "retries": retries,
            "workflow_json": {"1": {"class_type": "X", "inputs": {}}}}


class _Saved(unittest.TestCase):
    def setUp(self):
        self._store_active = main.store._active
        main.store._active = False
        self._saved = (main._gen_backends[:], dict(main.image_models), dict(main.backend_healthy),
                       dict(main.backend_inflight), main._gen_waiting[:], dict(main.gen_exec_faults),
                       dict(main.gen_speed), dict(main.host_controllers), dict(main.backend_loras))
        main._gen_backends[:] = [B3090, BDX, B4070]
        main.image_models.clear()
        main.image_models.update({"Qwen2.1": [_cand("rtx-3090"), _cand("dx10-02"), _cand("rtx-4070")]})
        main.backend_healthy.clear()
        main.backend_inflight.clear()
        main._gen_waiting.clear()
        main.gen_exec_faults.clear()
        main.host_controllers.clear()
        main.gen_speed.clear()
        main.backend_loras.clear()
        for b, secs in ((B3090, 90.0), (BDX, 120.0), (B4070, 230.0)):
            main.backend_healthy[main.backend_id(b)] = True
            main.gen_speed[f"Qwen2.1|{main.backend_id(b)}"] = secs
            main.backend_loras[main.backend_id(b)] = {LORA}

    def tearDown(self):
        main.store._active = self._store_active
        gb, im, bh, bi, gw, ef, gs, hc, bl = self._saved
        main._gen_backends[:] = gb
        main.image_models.clear(); main.image_models.update(im)
        main.backend_healthy.clear(); main.backend_healthy.update(bh)
        main.backend_inflight.clear(); main.backend_inflight.update(bi)
        main._gen_waiting[:] = gw
        main.gen_exec_faults.clear(); main.gen_exec_faults.update(ef)
        main.gen_speed.clear(); main.gen_speed.update(gs)
        main.host_controllers.clear(); main.host_controllers.update(hc)
        main.backend_loras.clear(); main.backend_loras.update(bl)

    @staticmethod
    def names(routes):
        return [b["name"] for b, _c in routes]


class RetriesDoNotNarrowTheChoice(_Saved):
    def _busy(self, *bs):
        for b in bs:
            main.backend_inflight[main.backend_id(b)] = 1

    def test_routes_keep_every_candidate_fastest_first(self):
        ready, allc = main._gen_routes("Qwen2.1")
        self.assertEqual(self.names(allc), ["rtx-3090", "dx10-02", "rtx-4070"])
        self.assertEqual(self.names(ready), ["rtx-3090", "dx10-02", "rtx-4070"])

    def test_lora_job_takes_the_free_slower_backend(self):
        """The prod case: both faster backends busy, the LoRA on all three."""
        self._busy(B3090, BDX)
        routes, parked, eligible = asyncio.run(
            main._gen_pick("Qwen2.1", "", {"params": {"lora_01": LORA}}))
        self.assertFalse(parked)                       # ← the bug: parked for dx10-02
        self.assertEqual(self.names(routes)[0], "rtx-4070")
        self.assertIn("rtx-4070", eligible)

    def test_parked_lora_job_may_be_handed_the_slower_backend(self):
        self._busy(B3090, BDX, B4070)
        _routes, parked, eligible = asyncio.run(
            main._gen_pick("Qwen2.1", "", {"params": {"lora_01": LORA}}))
        self.assertTrue(parked)
        entry = {"alias": "Qwen2.1", "enqueued_at": 0.0, "eligible": eligible}
        main.backend_inflight.pop(main.backend_id(B4070))      # the 4070 frees first
        self.assertTrue(main._entry_can_use(entry, B4070))


class _Jobs:
    def __init__(self):
        self.failed = self.completed = None

    def set_status(self, job_id, st):
        return True

    def set_backend(self, job_id, name):
        pass

    def complete(self, job_id, blobs, meta):
        self.completed = meta

    def fail(self, job_id, msg, meta=None):
        self.failed = msg


class _Adapter:
    def __init__(self, boom=None):
        self.boom, self.calls = boom, 0

    async def generate(self, req):
        self.calls += 1
        if self.boom is not None:
            raise self.boom
        return types.SimpleNamespace(blobs=[b"art"], meta={})


class RetriesLimitTheFailover(_Saved):
    def setUp(self):
        super().setUp()
        self.jobs = _Jobs()
        saved = (main.jobs, main._free_comfy_vram, main._comfy_free, main._wait_backend_up)

        async def _nop(*a, **k):
            return True

        main.jobs = self.jobs
        main._free_comfy_vram = main._comfy_free = main._wait_backend_up = _nop
        main.backend_adapters.clear()
        main.backend_last_key.clear()
        main.backend_vram_key.clear()

        def _restore():
            main.jobs, main._free_comfy_vram, main._comfy_free, main._wait_backend_up = saved
            main.backend_adapters.clear()
            main.backend_last_key.clear()
            main.backend_vram_key.clear()
        self.addCleanup(_restore)

    def _cands(self):
        return [(b, _cand(b["name"])) for b in (B3090, BDX, B4070)]

    def _run(self, adapters, state=None):
        main.backend_adapters.update({f"comfyui:{n}": a for n, a in adapters.items()})
        return asyncio.run(main._run_job("job1", "Qwen2.1", self._cands(),
                                         lambda b, c: types.SimpleNamespace(slot_held=False),
                                         state))

    def test_failover_stops_after_one_plus_retries_backends(self):
        ads = {"rtx-3090": _Adapter(RuntimeError("boom")), "dx10-02": _Adapter(RuntimeError("boom")),
               "rtx-4070": _Adapter()}
        self.assertTrue(self._run(ads))
        self.assertEqual([ads[n].calls for n in ("rtx-3090", "dx10-02", "rtx-4070")], [1, 1, 0])
        self.assertIsNotNone(self.jobs.failed)
        self.assertIsNone(self.jobs.completed)

    def test_connection_failover_is_capped_too(self):
        import httpx
        ads = {"rtx-3090": _Adapter(httpx.ConnectError("down")),
               "dx10-02": _Adapter(httpx.ConnectError("down")), "rtx-4070": _Adapter()}
        self.assertTrue(self._run(ads))
        self.assertEqual(ads["rtx-4070"].calls, 0)
        self.assertIsNotNone(self.jobs.failed)

    def test_the_cap_counts_backends_tried_before_a_re_park(self):
        """`tried` carries over a lost claim: two backends already ran → the third is
        not started, and the job ends instead of parking again."""
        st = {"tried": {main.backend_id(B3090), main.backend_id(BDX)},
              "last": RuntimeError("boom"), "attempts": 2}
        ads = {"rtx-4070": _Adapter()}
        self.assertTrue(self._run(ads, st))
        self.assertEqual(ads["rtx-4070"].calls, 0)
        self.assertIsNotNone(self.jobs.failed)

    def test_retries_blank_tries_every_backend(self):
        main.image_models["Qwen2.1"] = [_cand(n, "") for n in ("rtx-3090", "dx10-02", "rtx-4070")]
        ads = {"rtx-3090": _Adapter(RuntimeError("boom")), "dx10-02": _Adapter(RuntimeError("boom")),
               "rtx-4070": _Adapter()}
        main.backend_adapters.update({f"comfyui:{n}": a for n, a in ads.items()})
        cands = [(b, _cand(b["name"], "")) for b in (B3090, BDX, B4070)]
        self.assertTrue(asyncio.run(main._run_job(
            "job1", "Qwen2.1", cands, lambda b, c: types.SimpleNamespace(slot_held=False))))
        self.assertEqual(ads["rtx-4070"].calls, 1)
        self.assertIsNotNone(self.jobs.completed)

    def test_parked_job_with_the_cap_used_up_ends_instead_of_waiting(self):
        """Back in the park loop after a lost claim with 1+retries backends tried: a
        free untried backend must not start a third run — the job reports how it ended."""
        st = {"tried": {main.backend_id(B3090), main.backend_id(BDX)},
              "last": RuntimeError("boom"), "attempts": 2}
        ads = {"rtx-4070": _Adapter()}
        main.backend_adapters.update({f"comfyui:{n}": a for n, a in ads.items()})
        asyncio.run(asyncio.wait_for(main._run_gen_parked(
            "job1", "Qwen2.1", "", lambda b, c: types.SimpleNamespace(slot_held=False),
            None, st), timeout=5))
        self.assertEqual(ads["rtx-4070"].calls, 0)
        self.assertIsNotNone(self.jobs.failed)


if __name__ == "__main__":
    unittest.main()
