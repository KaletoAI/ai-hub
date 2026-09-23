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
        return True                     # the row was live (jobs.set_status's verdict)

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



class _BillingAdapter:
    """A cloud adapter stand-in: each generate() creates a vendor task (records it on the
    request's cloud_trace, exactly where CloudTaskAdapter._create does) and then fails with
    `boom` — the task exists and is billed whatever happens next. `create=False` fails
    BEFORE any task exists (a refused connect, a full queue)."""

    def __init__(self, boom=None, create=True, trace=None):
        self.boom, self.create, self.trace = boom, create, trace
        self.calls = 0

    async def generate(self, req):
        self.calls += 1
        if self.create:
            req.cloud_trace.update(self.trace or {"backend": "m", "cloud": "meshy",
                                                  "cloud_task_id": f"task-{self.calls}",
                                                  "endpoint": "image-to-3d"})
        if self.boom is not None:
            raise self.boom
        return types.SimpleNamespace(blobs=[b"art"], meta={})


class CloudBilledTaskIsFinal(unittest.TestCase):
    """K1 (review 2026-09-18): the FAILOVER arm re-ran a cloud task that already existed.
    `_poll` raises failover-class errors after the paid task was created — ConnectionError
    once the vendor is unreachable past disconnect_grace, TimeoutError at max_wait (whose
    own text says "still running") — and self_retries/the next candidate then created and
    paid for a SECOND task. Silent: the job even ends `done`, the double bill shows only at
    the vendor."""

    setUp, tearDown = RunJobExecFailover.setUp, RunJobExecFailover.tearDown   # the harness only,
    _comfy = staticmethod(RunJobExecFailover._comfy)                          # not its tests

    @staticmethod
    def _cloud(name, retries=0):
        return ({"name": name, "type": "meshy", "self_retries": retries},
                {"backend": name, "meshy": {"endpoint": "image-to-3d"}})

    def _run(self, cands, adapters_by_bid):
        main.backend_adapters.update(adapters_by_bid)
        asyncio.run(main._run_job("job1", "alias1", cands,
                                  lambda b, c: types.SimpleNamespace(slot_held=False,
                                                                     cloud_trace={})))

    def _assert_final_on_first(self, boom):
        a, b = _BillingAdapter(boom), _BillingAdapter()
        self._run([self._cloud("a", retries=2), self._cloud("b")],
                  {"meshy:a": a, "meshy:b": b})
        self.assertEqual(a.calls, 1)                 # no self-retry …
        self.assertEqual(b.calls, 0)                 # … and no second candidate
        self.assertIsNone(self.jobs.completed)
        self.assertIn("task-1", self.jobs.failed["msg"])
        self.assertIn("still be running", self.jobs.failed["msg"])
        self.assertEqual(self.jobs.failed["meta"]["cloud_task_id"], "task-1")

    def test_a_poll_timeout_never_creates_a_second_task(self):
        self._assert_final_on_first(TimeoutError("Meshy task task-1 not finished within max_wait"))

    def test_a_vendor_lost_while_polling_never_creates_a_second_task(self):
        self._assert_final_on_first(ConnectionError("Meshy unreachable for >30s while polling"))

    def test_a_create_whose_answer_was_lost_is_final_too(self):
        """A read timeout on the create POST: the body went out whole, the task may exist."""
        import httpx
        a = _BillingAdapter(httpx.ReadTimeout(""), trace={
            "backend": "a", "cloud": "meshy", "endpoint": "image-to-3d",
            "create_unconfirmed": True})
        b = _BillingAdapter()
        self._run([self._cloud("a", retries=1), self._cloud("b")], {"meshy:a": a, "meshy:b": b})
        self.assertEqual((a.calls, b.calls), (1, 0))
        self.assertIn("answer was lost", self.jobs.failed["msg"])

    def test_a_failure_before_any_task_still_fails_over(self):
        """No task, no bill: a full queue or a refused connect must keep the failover."""
        import adapters
        a = _BillingAdapter(adapters.CloudBusy("queue full", vendor="Meshy"), create=False)
        b = _BillingAdapter()
        self._run([self._cloud("a"), self._cloud("b")], {"meshy:a": a, "meshy:b": b})
        self.assertEqual((a.calls, b.calls), (1, 1))
        self.assertIsNotNone(self.jobs.completed)

    def test_an_unbilled_vendor_fault_is_still_retried(self):
        """CloudTaskRetryable is raised ONLY for a vendor-side failure that consumed no
        credits — the one case a re-run is meant for; the guard must not swallow it."""
        import adapters
        a = _FlakyAdapter(adapters.CloudTaskRetryable("server_error", vendor="Meshy"))
        self._run([self._cloud("a", retries=1)], {"meshy:a": a})
        self.assertEqual(a.calls, 2)
        self.assertIsNotNone(self.jobs.completed)

    def test_a_comfy_timeout_still_fails_over(self):
        """The guard is about BILLED tasks — a ComfyUI max_wait keeps its failover."""
        bad, good = self._comfy("bad"), self._comfy("good")
        good_ad = _Adapter()
        self._run([bad, good], {"comfyui:bad": _Adapter(TimeoutError("ComfyUI timeout")),
                                "comfyui:good": good_ad})
        self.assertEqual(good_ad.calls, 1)
        self.assertIsNotNone(self.jobs.completed)


class CancelKeepsTheBilledTask(unittest.TestCase):
    """F1: a cancel reaches `_run_job` as CancelledError — a BaseException none of the
    except arms see — so the row of a cancelled cloud job carried no task id, while the
    vendor went on to finish and bill the task."""

    setUp, tearDown = RunJobExecFailover.setUp, RunJobExecFailover.tearDown   # the harness only,
    _comfy = staticmethod(RunJobExecFailover._comfy)                          # not its tests

    def test_a_cancelled_cloud_job_keeps_its_task_id(self):
        merged = []
        self.jobs.merge_meta = lambda job_id, meta: merged.append(meta)
        started = asyncio.Event()

        class _Hang:
            async def generate(self, req):
                req.cloud_trace.update({"cloud": "meshy", "cloud_task_id": "task-9",
                                        "endpoint": "image-to-3d"})
                started.set()
                await asyncio.sleep(3600)

        main.backend_adapters["meshy:m"] = _Hang()
        cand = ({"name": "m", "type": "meshy"}, {"backend": "m", "meshy": {"endpoint": "image-to-3d"}})

        async def go():
            t = asyncio.create_task(main._run_job(
                "job1", "alias1", [cand],
                lambda b, c: types.SimpleNamespace(slot_held=False, cloud_trace={})))
            await started.wait()
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
        asyncio.run(go())
        self.assertEqual(merged[-1]["cloud_task_id"], "task-9")
        self.assertEqual(main.backend_inflight.get("meshy:m", 0), 0)    # slot released


class _ChainJobs(_FakeJobs):
    """The extra job-store calls _run_chain makes."""

    def __init__(self):
        super().__init__()
        self.merged = []

    def get(self, job_id):
        return {"status": self.failed and "failed" or "running"}

    def set_stage(self, job_id, st):
        pass

    def merge_meta(self, job_id, meta):
        self.merged.append(meta)


class ChainStage1BilledIsFinal(unittest.TestCase):
    """K1 for the chain: a stage-1 cloud task lost to a poll timeout / vendor outage was
    re-created by the stage-1 self-retry and then by the next stage-1 candidate."""

    def setUp(self):
        RunJobExecFailover.setUp(self)
        import meshy
        main.jobs = self.jobs = _ChainJobs()
        self._saved = (main._gen_backends[:], dict(main.image_models), dict(main.backend_healthy),
                       main.store._active, main._gen_waiting[:])
        main.store._active = False
        self.a = {"name": "a", "type": "meshy", "self_retries": 2, "enabled": True}
        self.b = {"name": "b", "type": "meshy", "enabled": True}
        self.r = {"name": "r", "type": "meshy", "enabled": True}
        main._gen_backends[:] = [self.a, self.b, self.r]
        s1 = meshy.default_candidate("x")["meshy"]
        s1["options"]["target_formats"] = ["glb"]
        succ = {"alias": "rig", "mesh_param": "input_mesh_path"}
        rig = meshy.default_candidate("r")
        rig["meshy"]["endpoint"] = "rigging"
        main.image_models.clear()
        main.image_models.update({
            "s1": [{"backend": "a", "meshy": dict(s1), "successor": succ},
                   {"backend": "b", "meshy": dict(s1), "successor": succ}],
            "rig": [rig]})
        self.succ = succ
        main.backend_healthy.clear()
        main.backend_healthy.update({"meshy:a": True, "meshy:b": True, "meshy:r": True})
        main._gen_waiting.clear()

    def tearDown(self):
        gb, im, bh, sa, gw = self._saved
        main._gen_backends[:] = gb
        main.image_models.clear(); main.image_models.update(im)
        main.backend_healthy.clear(); main.backend_healthy.update(bh)
        main.store._active = sa
        main._gen_waiting[:] = gw
        RunJobExecFailover.tearDown(self)

    def test_a_stage1_task_lost_to_the_vendor_is_not_bought_again(self):
        import adapters as ad_mod

        class _S1(_BillingAdapter):
            def chain_export(self, cand, succ, params, prefix):
                return ad_mod.ChainExport(f"{prefix}.glb")

        a, b = _S1(ConnectionError("Meshy unreachable while polling")), _S1()
        main.backend_adapters.update({"meshy:a": a, "meshy:b": b, "meshy:r": _Adapter()})
        asyncio.run(main._run_chain("job1", "s1", self.succ, {}, None, {}, {}, {}, {}))
        self.assertEqual((a.calls, b.calls), (1, 0))
        self.assertIn("task-1", self.jobs.failed["msg"])
        self.assertEqual(self.jobs.failed["meta"]["chain_stage1"]["cloud_task_id"], "task-1")
        self.assertEqual(main.backend_inflight.get("meshy:a", 0), 0)


    def _stage1(self, boom=None):
        import adapters as ad_mod

        class _S1(_BillingAdapter):
            def chain_export(self, cand, succ, params, prefix):
                return ad_mod.ChainExport(f"{prefix}.glb")
        return _S1(boom)

    def test_a_cancel_during_the_chain_claim_does_not_leak_the_slot(self):
        import threading
        gate, entered = threading.Event(), threading.Event()

        def slow_set_status(job_id, st):
            entered.set()
            gate.wait(5)
            return True
        self.jobs.set_status = slow_set_status
        main.backend_adapters.update({"meshy:a": self._stage1(), "meshy:b": self._stage1(),
                                      "meshy:r": _Adapter()})

        async def go():
            t = asyncio.create_task(main._run_chain("job1", "s1", self.succ, {}, None,
                                                    {}, {}, {}, {}))
            while not entered.is_set():
                await asyncio.sleep(0.005)
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
            gate.set()
        asyncio.run(go())
        self.assertEqual(main.backend_inflight.get("meshy:a", 0), 0)

    def test_a_malformed_self_retries_value_does_not_crash_the_chain(self):
        """K15: `_run_job` guards int(self_retries); the chain raised ValueError, which its
        except-Exception arm turned into a failed job that never reached any backend."""
        self.a["self_retries"] = "two"
        s1 = self._stage1(ConnectionError("gone"))
        main.backend_adapters.update({"meshy:a": s1, "meshy:b": self._stage1(ConnectionError("x")),
                                      "meshy:r": _Adapter()})
        s1.create = False                     # fails before any task → normal failover
        main.backend_adapters["meshy:b"].create = False
        asyncio.run(main._run_chain("job1", "s1", self.succ, {}, None, {}, {}, {}, {}))
        self.assertEqual(s1.calls, 1)

class CreateAnswerLost(unittest.TestCase):
    """The adapter half of K1: a create POST that went out whole but whose answer was lost
    (read timeout, dropped connection) may have created the task — the trace must say so,
    while the error class stays what main names and fault-logs by. A connect failure never
    reached the vendor and must NOT be marked."""

    def _create(self, exc):
        import httpx
        import adapters

        class _Client:
            async def post(self, *a, **k):
                raise exc

        ctx = adapters.AdapterContext(auth_headers=lambda b: {}, inflight_inc=lambda b: None,
                                      inflight_dec=lambda b: None, cost_usd=lambda *a: 0.0,
                                      source_of=lambda r: "t", record_call=lambda *a, **k: None,
                                      log_enabled=lambda: False)
        ad = adapters.MeshyAdapter({"name": "m", "type": "meshy", "url": "http://x"}, ctx)
        req = adapters.NormalizedRequest(alias="a")
        with self.assertRaises(type(exc)):
            asyncio.run(ad._create(_Client(), "http://x/openapi/v1/image-to-3d", {"a": 1},
                                   "image-to-3d", req))
        return req.cloud_trace

    def test_a_read_timeout_marks_the_create_unconfirmed(self):
        import httpx
        tr = self._create(httpx.ReadTimeout(""))
        self.assertTrue(tr.get("create_unconfirmed"))
        self.assertEqual(tr.get("endpoint"), "image-to-3d")

    def test_a_refused_connect_is_not_marked(self):
        import httpx
        self.assertNotIn("create_unconfirmed", self._create(httpx.ConnectError("refused")))


class ClaimIsAtomic(unittest.TestCase):
    """K4/K5: the job slot. A claim past `max_concurrent` (the busy check ran several awaits
    before the claim, and a failover target was never checked at all) runs two prompts on a
    backend that takes one — they simply queue inside ComfyUI, and the second job's time
    budget burns while it waits. A cancel that lands between the claim and the `try` that
    releases it leaks the slot for good: the backend then reads busy forever. Both look
    like an ordinary slow or busy backend."""

    setUp, tearDown = RunJobExecFailover.setUp, RunJobExecFailover.tearDown
    _comfy = staticmethod(RunJobExecFailover._comfy)

    @staticmethod
    def _capped(name, cap=1):
        return ({"name": name, "type": "comfyui", "max_concurrent": cap},
                {"backend": name, "workflow_json": {}})

    def _run(self, cands, state=None):
        return asyncio.run(main._run_job("job1", "alias1", cands,
                                         lambda b, c: types.SimpleNamespace(slot_held=False),
                                         state))

    def test_a_backend_at_its_cap_is_not_claimed(self):
        ad = _Adapter()
        main.backend_adapters["comfyui:gpu"] = ad
        main.backend_inflight["comfyui:gpu"] = 1          # someone claimed it after our pick
        self.assertFalse(self._run([self._capped("gpu")]))   # → park again
        self.assertEqual(ad.calls, 0)
        self.assertEqual(main.backend_inflight["comfyui:gpu"], 1)
        self.assertIsNone(self.jobs.failed)                  # not reported as exhausted

    def test_a_busy_failover_target_parks_and_the_state_carries_over(self):
        a, b = _Adapter(ConnectionError("gone")), _Adapter()
        main.backend_adapters.update({"comfyui:a": a, "comfyui:b": b})
        main.backend_inflight["comfyui:b"] = 1
        st = {}
        cands = [self._capped("a"), self._capped("b")]
        self.assertFalse(self._run(cands, st))
        self.assertEqual((a.calls, b.calls), (1, 0))
        main.backend_inflight["comfyui:b"] = 0               # b frees up
        self.assertTrue(self._run(cands, st))                # a is not re-run …
        self.assertEqual((a.calls, b.calls), (1, 1))
        self.assertEqual(self.jobs.completed["meta"]["attempts"], 2)   # … and the count holds

    def test_a_cancel_during_the_claim_does_not_leak_the_slot(self):
        import threading
        gate = threading.Event()
        entered = threading.Event()

        def slow_set_backend(job_id, name):
            entered.set()
            gate.wait(5)
        self.jobs.set_backend = slow_set_backend
        main.backend_adapters["comfyui:gpu"] = _Adapter()

        async def go():
            t = asyncio.create_task(main._run_job(
                "job1", "alias1", [self._capped("gpu")],
                lambda b, c: types.SimpleNamespace(slot_held=False)))
            while not entered.is_set():
                await asyncio.sleep(0.005)
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
            gate.set()
        asyncio.run(go())
        self.assertEqual(main.backend_inflight.get("comfyui:gpu", 0), 0)

    def test_a_lost_claim_at_request_time_queues_the_job(self):
        parked = []

        async def _park(job_id, alias, force, build_req, eligible=None, run_state=None):
            parked.append(run_state)
        orig = main._run_gen_parked
        main._run_gen_parked = _park
        try:
            main.backend_adapters["comfyui:gpu"] = _Adapter()
            main.backend_inflight["comfyui:gpu"] = 1
            asyncio.run(main._run_gen_now("job1", "alias1", "", [self._capped("gpu")],
                                          lambda b, c: types.SimpleNamespace(slot_held=False)))
        finally:
            main._run_gen_parked = orig
        self.assertEqual(len(parked), 1)

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



class DesignationReadsTheStoreOncePerAlias(unittest.TestCase):
    """P11: one designation pass asked `_gen_routes` — a store read that JSON-parses the
    alias's whole candidate list, workflow JSON included — for every waiter × every free
    backend, every 2 s per parked job. Silent: it only shows as a sluggish gateway once
    a queue builds up."""

    def test_one_route_lookup_per_alias_per_pass(self):
        calls = []
        backs = [{"name": f"g{i}", "type": "comfyui"} for i in range(4)]

        def routes(alias):
            calls.append(alias)
            return [(b, {}) for b in backs], [(b, {}) for b in backs]
        orig, saved = main._gen_routes, main._gen_waiting[:]
        main._gen_routes = routes
        main._gen_waiting[:] = [{"job_id": f"j{i}", "alias": "a", "enqueued_at": float(i)}
                                for i in range(5)]
        try:
            me = main._gen_waiting[-1]            # the youngest: never designated first
            self.assertIsNone(main._designated_gen_index(me, [(b, {}) for b in backs]))
        finally:
            main._gen_routes = orig
            main._gen_waiting[:] = saved
        self.assertEqual(calls, ["a"])

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
        return self.NR(alias="a", real_model=None, inputs={}, params=params or {},
                       workflow=None, workflow_json=wf, node_mapping=mapping or {},
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
