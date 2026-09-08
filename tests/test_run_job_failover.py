"""What `_run_job` does with an EXECUTION error — the arm that decides whether a user's
job dies on a broken backend or lands on a working one.

Why this file exists (it fails SILENTLY): every case here ends with a job row that
looks plausible on its own. A job that died on the first candidate reads like a genuine
workflow error; a fault charged to the wrong candidate quietly takes a healthy backend
out of rotation for 15 minutes; and a cloud candidate that fails over would re-run a
task the vendor already BILLED — none of it raises, and none of it shows up anywhere
except as a bill or an idle GPU.

Run: venv/bin/python -m unittest tests.test_run_job_failover -v
"""
import asyncio
import json
import os
import sys
import tempfile
import time
import types
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
    import scheduler
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


class _FakeJobs:
    """Stands in for the jobs module: records the outcome instead of touching SQLite."""

    def __init__(self):
        self.status = None
        self.completed = None
        self.failed = None
        self.backend = None

    def set_status(self, job_id, st):
        self.status = st

    def set_backend(self, job_id, name):
        self.backend = name

    def complete(self, job_id, blobs, meta):
        self.completed = {"blobs": blobs, "meta": meta}

    def fail(self, job_id, msg, meta=None):
        self.failed = {"msg": msg, "meta": meta}


class _Adapter:
    """generate() either raises `boom` or returns one artifact."""

    def __init__(self, boom=None):
        self.boom = boom
        self.calls = 0

    async def generate(self, req):
        self.calls += 1
        if self.boom is not None:
            raise self.boom
        return types.SimpleNamespace(blobs=[b"art"], meta={})


class RunJobExecFailover(unittest.TestCase):

    def setUp(self):
        self.jobs = _FakeJobs()
        self._orig_jobs = main.jobs
        self._orig_free = main._free_comfy_vram
        self._orig_raw_free = main._comfy_free
        self._orig_wait_up = main._wait_backend_up
        main.jobs = self.jobs
        # The claim path (_claim_gen_backend) runs FOR REAL down to the one raw HTTP
        # call, which is stubbed here and RECORDED — a fixture without a `url` would
        # otherwise pass on a swallowed KeyError, covering nothing (review 2026-09-08).
        self.frees = []
        self.free_ok = True

        async def _no_free(backend, why):        # the after-job policy: not under test here
            return None

        async def _raw_free(backend, why, settle_s=0.0):
            self.frees.append(backend["name"])
            return self.free_ok

        async def _up(backend, timeout_s=30.0):  # no /system_stats to poll between self-retries
            return None

        main._free_comfy_vram = _no_free
        main._comfy_free = _raw_free
        main._wait_backend_up = _up
        main.gen_exec_faults.clear()
        main.gen_speed.clear()
        main.backend_gen_window.clear()
        main.backend_inflight.clear()
        main.backend_adapters.clear()
        main.backend_last_key.clear()
        main.backend_vram_key.clear()

    def tearDown(self):
        main.jobs = self._orig_jobs
        main._free_comfy_vram = self._orig_free
        main._comfy_free = self._orig_raw_free
        main._wait_backend_up = self._orig_wait_up
        main.gen_exec_faults.clear()
        main.backend_adapters.clear()
        main.backend_last_key.clear()
        main.backend_vram_key.clear()

    def _run(self, cands, adapters_by_bid):
        main.backend_adapters.update(adapters_by_bid)
        asyncio.run(main._run_job("job1", "alias1", cands,
                                  lambda b, c: types.SimpleNamespace(slot_held=False)))

    @staticmethod
    def _comfy(name):
        return ({"name": name, "type": "comfyui"}, {"backend": name, "workflow_json": {}})

    @staticmethod
    def _key(name):
        return f"alias1|comfyui:{name}"

    def test_execution_error_fails_over_to_the_next_backend(self):
        """The 2026-09-03 case: candidate one runs the prompt and blows up, candidate
        two delivers. The job must be DONE, not failed."""
        bad, good = self._comfy("bad"), self._comfy("good")
        self._run([bad, good], {"comfyui:bad": _Adapter(RuntimeError("node 1 blew up")),
                                "comfyui:good": _Adapter()})
        self.assertIsNotNone(self.jobs.completed)
        self.assertIsNone(self.jobs.failed)
        self.assertEqual(self.jobs.backend, "good")     # row re-pointed at the real runner

    def test_the_failing_backend_is_charged_only_because_another_succeeded(self):
        bad, good = self._comfy("bad"), self._comfy("good")
        self._run([bad, good], {"comfyui:bad": _Adapter(RuntimeError("boom")),
                                "comfyui:good": _Adapter()})
        self.assertTrue(scheduler.exec_probed(main.gen_exec_faults, self._key("bad")))
        # the winner must stay clean, or a working backend drifts into quarantine
        self.assertFalse(scheduler.exec_probed(main.gen_exec_faults, self._key("good")))

    def test_two_proven_faults_quarantine_the_candidate(self):
        bad, good = self._comfy("bad"), self._comfy("good")
        for _ in range(2):
            self._run([bad, good], {"comfyui:bad": _Adapter(RuntimeError("boom")),
                                    "comfyui:good": _Adapter()})
        self.assertTrue(scheduler.exec_quarantined(main.gen_exec_faults,
                                                   self._key("bad"), time.time()))

    def test_when_every_candidate_fails_nobody_is_charged(self):
        """All three failed the same way → the REQUEST is the common factor. Charging
        them here would let one bad workflow quarantine an alias's whole fleet."""
        a, b = self._comfy("a"), self._comfy("b")
        self._run([a, b], {"comfyui:a": _Adapter(RuntimeError("bad model name")),
                           "comfyui:b": _Adapter(RuntimeError("bad model name"))})
        self.assertIsNotNone(self.jobs.failed)
        self.assertEqual(main.gen_exec_faults, {})

    def test_the_report_names_the_execution_error_not_the_network(self):
        a, b = self._comfy("a"), self._comfy("b")
        self._run([a, b], {"comfyui:a": _Adapter(RuntimeError("node 1 UNETLoader: nope")),
                           "comfyui:b": _Adapter(RuntimeError("node 1 UNETLoader: nope"))})
        msg = self.jobs.failed["msg"]
        self.assertIn("UNETLoader", msg)
        self.assertNotIn("unreachable", msg)
        self.assertIn("2 backends", msg)      # and it says the request is the suspect

    def test_execution_error_never_repeats_on_the_same_backend(self):
        """self_retries exists for sporadic driver faults on the connection path. An
        execution error reproduces, so re-running it just burns the user's time."""
        bad = ({"name": "bad", "type": "comfyui", "self_retries": 3},
               {"backend": "bad", "workflow_json": {}})
        ad = _Adapter(RuntimeError("boom"))
        self._run([bad], {"comfyui:bad": ad})
        self.assertEqual(ad.calls, 1)

    def test_a_cloud_candidate_never_fails_over(self):
        """A cloud task is BILLED. Whatever failed may have happened after the paid task
        was created, so re-running the job on the next candidate buys it twice."""
        cloud = ({"name": "meshy", "type": "meshy"},
                 {"backend": "meshy", "meshy": {"endpoint": "image-to-3d"}})
        good = self._comfy("good")
        good_ad = _Adapter()
        self._run([cloud, good], {"meshy:meshy": _Adapter(RuntimeError("task rejected")),
                                  "comfyui:good": good_ad})
        self.assertIsNotNone(self.jobs.failed)
        self.assertEqual(good_ad.calls, 0)           # the second candidate never ran
        self.assertIsNone(self.jobs.completed)


class _FlakyAdapter:
    """generate() raises each entry of `booms` in turn, then delivers."""

    def __init__(self, *booms):
        self.booms = list(booms)
        self.calls = 0

    async def generate(self, req):
        self.calls += 1
        if self.booms:
            raise self.booms.pop(0)
        return types.SimpleNamespace(blobs=[b"art"], meta={})


class ClaimFreesVram(RunJobExecFailover):
    """The before-job free (`_claim_gen_backend`) as `_run_job` wires it. Fails SILENTLY:
    a free that never fires shows up as `CUDA out of memory` on the NEXT alias hours
    later (job cc604da29e0e), one that fires on every job shows up as "generation got
    slower" — and a VRAM record written before the free happened makes the first look
    exactly like the second one's absence."""

    def test_an_unknown_gpu_is_freed_once_per_claimed_candidate(self):
        bad, good = self._comfy("bad"), self._comfy("good")
        self._run([bad, good], {"comfyui:bad": _Adapter(RuntimeError("boom")),
                                "comfyui:good": _Adapter()})
        self.assertEqual(self.frees, ["bad", "good"])
        self.assertEqual(main.backend_vram_key["comfyui:good"], "alias1")

    def test_a_same_alias_repeat_keeps_the_cache(self):
        good = self._comfy("good")
        self._run([good], {"comfyui:good": _Adapter()})
        self._run([good], {"comfyui:good": _Adapter()})
        self.assertEqual(self.frees, ["good"])

    def test_a_self_retry_never_frees_again(self):
        """Same key, same backend: the retry must run on the cache attempt one built."""
        flaky = ({"name": "flaky", "type": "comfyui", "self_retries": 1},
                 {"backend": "flaky", "workflow_json": {}})
        self._run([flaky], {"comfyui:flaky": _FlakyAdapter(ConnectionError("blip"))})
        self.assertIsNotNone(self.jobs.completed)
        self.assertEqual(self.frees, ["flaky"])

    def test_a_failed_free_is_not_recorded_as_done(self):
        """Review 2026-09-08 finding 1: the record used to be written BEFORE the free was
        attempted, so a timed-out /free made every later same-alias job skip it too."""
        good = self._comfy("good")
        self.free_ok = False
        self._run([good], {"comfyui:good": _Adapter()})
        self.assertIsNone(main.backend_vram_key.get("comfyui:good"))
        self.free_ok = True
        self._run([good], {"comfyui:good": _Adapter()})
        self.assertEqual(self.frees, ["good", "good"])   # the second job tried again

    def test_another_job_in_flight_blocks_the_free_and_forgets_the_key(self):
        """Our own slot is not "another job" (the `- 1`); a real one is, and afterwards the
        GPU holds TWO model sets — the next same-alias job must not trust the record."""
        good = self._comfy("good")
        main.backend_inflight["comfyui:good"] = 1          # somebody else is running there
        self._run([good], {"comfyui:good": _Adapter()})
        self.assertEqual(self.frees, [])
        self.assertIsNone(main.backend_vram_key.get("comfyui:good"))
        self.assertEqual(main.backend_last_key["comfyui:good"], "alias1")   # affinity still set
        main.backend_inflight.clear()
        self._run([good], {"comfyui:good": _Adapter()})
        self.assertEqual(self.frees, ["good"])

    def test_the_free_keys_on_the_model_set_not_the_alias(self):
        """Two aliases loading the same weights (trellis2 high/low) share the cache; one
        alias loading other weights (a mapped model, a LoRA) frees — the adapter says
        what a request loads, the alias is only the fallback."""
        class _Keyed(_Adapter):
            def __init__(self, key):
                super().__init__(); self.key = key
            def model_set_key(self, req):
                return self.key
        good = self._comfy("good")
        main.backend_adapters["comfyui:good"] = _Keyed("ms:trellis")
        asyncio.run(main._run_job("j1", "trellis-high", [good],
                                  lambda b, c: types.SimpleNamespace(slot_held=False)))
        asyncio.run(main._run_job("j2", "trellis-low", [good],
                                  lambda b, c: types.SimpleNamespace(slot_held=False)))
        self.assertEqual(self.frees, ["good"])                  # other alias, same weights → kept
        main.backend_adapters["comfyui:good"] = _Keyed("ms:other")
        asyncio.run(main._run_job("j3", "trellis-low", [good],
                                  lambda b, c: types.SimpleNamespace(slot_held=False)))
        self.assertEqual(self.frees, ["good", "good"])          # same alias, other weights → freed
        self.assertEqual(main.backend_last_key["comfyui:good"], "trellis-low")   # affinity = alias

    def test_a_cloud_backend_records_affinity_but_never_frees(self):
        cloud = ({"name": "meshy", "type": "meshy"},
                 {"backend": "meshy", "meshy": {"endpoint": "image-to-3d"}})
        self._run([cloud], {"meshy:meshy": _Adapter()})
        self.assertEqual(self.frees, [])
        self.assertEqual(main.backend_last_key["meshy:meshy"], "alias1")


class FreeAfterJob(unittest.TestCase):
    """`_free_comfy_vram`'s same-alias skip must ask the scheduler which waiter this
    backend will actually get (`_designated_gen_waiter`), not scan the global queue by
    alias: a waiter that can never run here (excluded, force-pinned elsewhere) used to
    suppress the after-job free forever — on a shared box, the next llama-swap load then
    aborts on VRAM nobody frees (review 2026-09-08 finding 2)."""

    BID = "comfyui:gpu"

    def setUp(self):
        self._orig_raw_free = main._comfy_free
        self.frees = []

        async def _raw_free(backend, why, settle_s=0.0, abort_when=None):
            self.frees.append(backend["name"])
            return True

        main._comfy_free = _raw_free
        # _gen_routes reads the store when it is active — another test module may have
        # left it pointing at a temp DB that is gone; the config fallback is what we set up
        self._store_active = main.store._active
        main.store._active = False
        self.backend = {"name": "gpu", "type": "comfyui", "url": "http://gpu:8188", "enabled": True}
        self._saved = (main._gen_backends[:], dict(main.image_models), dict(main.backend_healthy),
                       dict(main.backend_hosts), dict(main.hosts_meta), main._gen_waiting[:])
        main._gen_backends[:] = [self.backend]
        main.image_models.clear()
        main.image_models["alias1"] = [{"backend": "gpu", "workflow_json": {}}]
        main.backend_healthy.clear()
        main.backend_healthy[self.BID] = True
        main.backend_hosts.clear()
        main.backend_hosts[self.BID] = "box"
        main.hosts_meta.clear()
        main.hosts_meta["box"] = {"comfy_free_after_job": True}
        main._gen_waiting.clear()
        main.backend_inflight.clear()
        main.backend_last_key.clear()
        main.backend_vram_key.clear()
        main.backend_last_key[self.BID] = "alias1"
        main.backend_vram_key[self.BID] = "alias1"

    def tearDown(self):
        main._comfy_free = self._orig_raw_free
        main.store._active = self._store_active
        gb, im, bh, bhs, hm, gw = self._saved
        main._gen_backends[:] = gb
        main.image_models.clear(); main.image_models.update(im)
        main.backend_healthy.clear(); main.backend_healthy.update(bh)
        main.backend_hosts.clear(); main.backend_hosts.update(bhs)
        main.hosts_meta.clear(); main.hosts_meta.update(hm)
        main._gen_waiting[:] = gw
        main.backend_inflight.clear()
        main.backend_last_key.clear()
        main.backend_vram_key.clear()

    def _waiter(self, **kw):
        e = {"alias": "alias1", "enqueued_at": time.monotonic(), "claimed": False}
        e.update(kw)
        main._gen_waiting.append(e)

    def _free(self):
        asyncio.run(main._free_comfy_vram(self.backend, "test"))

    def test_a_same_alias_waiter_this_backend_will_take_keeps_the_cache(self):
        self._waiter()
        self._free()
        self.assertEqual(self.frees, [])

    def test_a_waiter_that_excluded_this_backend_does_not_block_the_free(self):
        self._waiter(exclude=["gpu"])            # the chain's failover set, see _run_chain
        self._free()
        self.assertEqual(self.frees, ["gpu"])

    def test_a_waiter_pinned_elsewhere_does_not_block_the_free(self):
        self._waiter(force="other")
        self._free()
        self.assertEqual(self.frees, ["gpu"])

    def test_no_waiter_frees_and_forgets_the_record(self):
        self._free()
        self.assertEqual(self.frees, ["gpu"])
        self.assertIsNone(main.backend_vram_key.get(self.BID))

    def test_a_running_job_is_never_freed_under(self):
        main.backend_inflight[self.BID] = 1
        self._free()
        self.assertEqual(self.frees, [])

    def test_the_flag_off_frees_nothing(self):
        main.hosts_meta["box"] = {"comfy_free_after_job": False}
        self._free()
        self.assertEqual(self.frees, [])


class _FakeHttp:
    """http_client stand-in: /free counts posts, /system_stats reports a torch pool that
    only drops once `drop_after` posts have landed (a lost notify looks like exactly that:
    the first post did nothing)."""

    def __init__(self, before, drop_after=1):
        self.before, self.drop_after, self.posts = before, drop_after, 0

    async def post(self, url, json=None, timeout=None):
        self.posts += 1
        return types.SimpleNamespace(status_code=200)

    async def get(self, url, timeout=None):
        held = 0 if self.posts >= self.drop_after else self.before
        return types.SimpleNamespace(status_code=200,
                                     json=lambda: {"devices": [{"torch_vram_total": held}]})


class ComfyFreeSettles(unittest.TestCase):
    """`_comfy_free` with `settle_s`: ComfyUI's /free only queues the unload, and the queue
    wake-up is lost when the worker is in its post-prompt gc — so the free must be
    WATCHED and re-posted, and must say when it did not happen. All of it fails silently:
    the POST answers 200 either way."""

    GIB = 1 << 30

    def setUp(self):
        self._orig = main.http_client, main._FREE_REPOST_S
        main._FREE_REPOST_S = 0.0                 # re-post on every tick → fast tests
        self.backend = {"name": "gpu", "url": "http://gpu:8188"}

    def tearDown(self):
        main.http_client, main._FREE_REPOST_S = self._orig

    def _free(self, settle_s):
        return asyncio.run(main._comfy_free(self.backend, "test", settle_s=settle_s))

    def test_a_lost_first_post_is_repeated_until_the_vram_drops(self):
        main.http_client = fake = _FakeHttp(20 * self.GIB, drop_after=2)
        self.assertTrue(self._free(settle_s=5.0))
        self.assertGreaterEqual(fake.posts, 2)

    def test_vram_that_never_drops_is_reported_as_not_freed(self):
        main.http_client = _FakeHttp(20 * self.GIB, drop_after=99)
        self.assertFalse(self._free(settle_s=0.7))

    def test_an_already_empty_gpu_needs_no_wait(self):
        main.http_client = fake = _FakeHttp(64 << 20)
        t0 = time.monotonic()
        self.assertTrue(self._free(settle_s=5.0))
        self.assertLess(time.monotonic() - t0, 0.4)
        self.assertEqual(fake.posts, 1)

    def test_a_claim_during_the_settle_stops_the_reposting(self):
        """A job that claimed the backend meanwhile is loading its models — a re-post
        would unload them under it. Verdict False: the GPU's content is now unknown."""
        main.http_client = fake = _FakeHttp(20 * self.GIB, drop_after=99)
        self.assertFalse(asyncio.run(main._comfy_free(
            self.backend, "test", settle_s=5.0, abort_when=lambda: True)))
        self.assertEqual(fake.posts, 1)          # the first post only, never a second

    def test_no_settle_keeps_the_fire_and_forget_shape(self):
        main.http_client = fake = _FakeHttp(20 * self.GIB, drop_after=99)
        self.assertTrue(self._free(settle_s=0.0))
        self.assertEqual(fake.posts, 1)


class AdapterModelSetKey(unittest.TestCase):
    """`ComfyUIAdapter.model_set_key` on the shipped samples: the key must see what the
    request will LOAD after the same injections generate() applies — a mapped model
    choice under one alias is a different set, two aliases on one model are the same."""

    def setUp(self):
        import adapters
        d = os.path.join(_here, "sample_comfyui_workflows")

        def load(name):
            with open(os.path.join(d, name)) as f:
                return json.load(f)
        self.hi, self.lo = load("img2mesh-trellis2_high_api.json"), load("img2mesh-trellis2_low_api.json")
        self.ad = object.__new__(adapters.ComfyUIAdapter)      # no ctx/HTTP needed for the key
        self.ad.backend = {"name": "gpu", "type": "comfyui", "url": "http://gpu:8188"}
        self.ad.name, self.ad.bid = "gpu", "comfyui:gpu"
        self.ad.ctx = types.SimpleNamespace(loras_of=lambda bid: set())
        self.NR = adapters.NormalizedRequest

    def _req(self, wf, mapping=None, params=None, fixed=None):
        return self.NR(alias="a", real_model=None, task="text2img", inputs={}, params=params or {},
                       output={}, workflow=None, workflow_json=wf, node_mapping=mapping or {},
                       fixed=fixed or [])

    def test_high_and_low_share_the_key(self):
        self.assertEqual(self.ad.model_set_key(self._req(self.hi)), self.ad.model_set_key(self._req(self.lo)))

    def test_a_mapped_model_choice_changes_the_key(self):
        m = {"model": {"node": "60", "field": "modelname"}}
        a = self.ad.model_set_key(self._req(self.hi, m, {"model": "microsoft/TRELLIS.2-4B"}))
        b = self.ad.model_set_key(self._req(self.hi, m, {"model": "TencentARC/Pixal3D-T"}))
        self.assertNotEqual(a, b)

    def test_a_pin_is_part_of_the_key_and_the_stored_workflow_stays_untouched(self):
        before = json.dumps(self.hi, sort_keys=True)
        a = self.ad.model_set_key(self._req(self.hi))
        b = self.ad.model_set_key(self._req(self.hi, fixed=[{"node": "60", "field": "modelname", "value": "other/model"}]))
        self.assertNotEqual(a, b)
        self.assertEqual(before, json.dumps(self.hi, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
