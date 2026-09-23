"""Cancelling a generation job — and stopping exactly that job's work, nothing else —
plus the lifecycle around it: a worker that crashes, an adapter rebuilt under a job.

Why this fails SILENTLY: ComfyUI's bare `POST /interrupt` stops whatever is executing.
A cancel of a job whose prompt was still WAITING, or a `max_wait` expiry while the box
ran another job's prompt, therefore killed a stranger's run: that job then failed over
(or was charged an execution fault and quarantined its backend), and nothing anywhere
said the gateway had done it. And a SYNC job was never in `_gen_tasks`, so its cancel
only relabelled the row while the worker went on to fail over or finish — the user saw
"cancelled", the GPU kept working.

Run: venv/bin/python -m unittest tests.test_gen_cancel -v
"""
import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
import unittest.mock

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import httpx
    import adapters
    import main
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


class _Comfy:
    """A scripted ComfyUI behind httpx.MockTransport: our prompt `p1` sits where `where`
    says, somebody else's prompt runs; every POST is recorded."""

    def __init__(self, where="pending", history=None, queue_status=200):
        self.where, self.history, self.queue_status = where, history, queue_status
        self.posts, self.history_polls = [], 0

    def handler(self, request):
        path = request.url.path
        if request.method == "POST":
            body = json.loads(request.content or b"{}")
            self.posts.append((path, body))
            if path == "/prompt":
                return httpx.Response(200, json={"prompt_id": "p1", "number": 1})
            return httpx.Response(200, json={})
        if path == "/queue":
            running = [[0, "other", {}, {}]]
            pending = []
            if self.where == "running":
                running = [[0, "p1", {}, {}]]
            elif self.where == "pending":
                pending = [[1, "p1", {}, {}]]
            return httpx.Response(self.queue_status,
                                  json={"queue_running": running, "queue_pending": pending})
        if path.startswith("/history/"):
            self.history_polls += 1
            return httpx.Response(200, json=self.history or {})
        return httpx.Response(404, json={})

    def client(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def stops(self):
        return [p for p in self.posts if p[0] in ("/interrupt", "/queue")]


def _ctx():
    return adapters.AdapterContext(
        auth_headers=lambda b: {}, inflight_inc=lambda b: None, inflight_dec=lambda b: None,
        cost_usd=lambda *a: 0.0, source_of=lambda r: "t", record_call=lambda *a, **k: None,
        log_enabled=lambda: False)


def _adapter(**extra):
    b = {"name": "gpu", "type": "comfyui", "url": "http://comfy", "poll_interval": 0.01,
         "disconnect_grace": 30, **extra}
    return adapters.ComfyUIAdapter(b, _ctx())


class StopPrompt(unittest.TestCase):
    """`_stop_prompt` — the one way the gateway stops a ComfyUI prompt."""

    def _stop(self, comfy):
        async def go():
            async with comfy.client() as c:
                return await _adapter()._stop_prompt(c, "http://comfy", "p1")
        return asyncio.run(go())

    def test_our_running_prompt_is_interrupted_by_name(self):
        comfy = _Comfy("running")
        self.assertEqual(self._stop(comfy), "running")
        self.assertEqual(comfy.stops(), [("/interrupt", {"prompt_id": "p1"})])

    def test_a_waiting_prompt_is_deleted_and_the_running_stranger_left_alone(self):
        comfy = _Comfy("pending")
        self.assertEqual(self._stop(comfy), "pending")
        self.assertEqual(comfy.stops(), [("/queue", {"delete": ["p1"]})])

    def test_a_prompt_that_is_gone_touches_nothing(self):
        comfy = _Comfy(None)
        self.assertEqual(self._stop(comfy), "gone")
        self.assertEqual(comfy.stops(), [])

    def test_an_unreadable_queue_touches_nothing(self):
        comfy = _Comfy("running", queue_status=500)
        self._stop(comfy)
        self.assertEqual(comfy.stops(), [])


class PollStops(unittest.TestCase):

    def _poll(self, comfy, max_wait):
        async def go():
            async with comfy.client() as c:
                return await _adapter()._poll(c, "http://comfy", "p1", 0.01, max_wait)
        return asyncio.run(go())

    def test_max_wait_on_a_waiting_prompt_does_not_interrupt_the_running_one(self):
        comfy = _Comfy("pending")
        with self.assertRaises(TimeoutError):
            self._poll(comfy, 0.05)
        self.assertEqual(comfy.stops(), [("/queue", {"delete": ["p1"]})])

    def test_an_interrupted_prompt_is_named_as_such(self):
        comfy = _Comfy(None, history={"p1": {"status": {
            "status_str": "error", "messages": [["execution_interrupted", {"prompt_id": "p1"}]]}}})
        with self.assertRaises(adapters.ComfyPromptInterrupted):
            self._poll(comfy, 5)


class CancelledGenerate(unittest.TestCase):
    """The normal cancel path: main cancels the worker task, and generate() stops ITS
    prompt on the way out."""

    def test_cancelling_the_worker_removes_only_its_own_prompt(self):
        comfy = _Comfy("pending")
        real = httpx.AsyncClient
        ad = _adapter()
        req = adapters.NormalizedRequest(
            alias="a", job_id="job1", upload_prefix="gw_job1",
            workflow_json={"9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}}},
            node_mapping={"prompt": {"node": "9", "field": "filename_prefix"}})

        async def go():
            t = asyncio.create_task(ad.generate(req))
            for _ in range(500):
                await asyncio.sleep(0.01)
                if comfy.history_polls:
                    break
            self.assertEqual(ad._prompts.get("job1"), "p1")     # registered while it runs
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t

        with unittest.mock.patch.object(
                adapters.httpx, "AsyncClient",
                lambda *a, **kw: real(transport=httpx.MockTransport(comfy.handler),
                                      **{k: v for k, v in kw.items() if k != "transport"})):
            asyncio.run(go())
        self.assertEqual(comfy.stops(), [("/queue", {"delete": ["p1"]})])
        self.assertNotIn("job1", ad._prompts)


class _Jobs:
    def __init__(self):
        self.failed = None
        self.status = "running"

    def get(self, job_id):
        return {"status": self.status, "alias": "gone-alias", "backend": "gpu"}

    def fail(self, job_id, msg, meta=None):
        if self.status in ("queued", "running"):
            self.failed, self.status = msg, "failed"


class CancelASyncJob(unittest.TestCase):
    """A sync job runs as a tracked task now, so a cancel ENDS it."""

    def setUp(self):
        self._orig = main.jobs
        main.jobs = self.jobs = _Jobs()

    def tearDown(self):
        main.jobs = self._orig

    def test_a_sync_job_is_stopped_by_its_cancel(self):
        ran_on = []

        async def worker():
            try:
                await asyncio.sleep(3600)
                ran_on.append("finished")               # must never happen
            except asyncio.CancelledError:
                ran_on.append("cancelled")
                raise

        async def go():
            sync = asyncio.create_task(main._run_gen_sync("jobS", worker()))
            await asyncio.sleep(0.01)
            self.assertIn("jobS", main._gen_tasks)       # visible to cancel_generation
            self.assertTrue(await main.cancel_generation("jobS"))
            # the worker has UNWOUND by the time cancel returns — its slot is released, so
            # the after-cancel VRAM free does not skip the backend as still busy
            self.assertEqual(ran_on, ["cancelled"])
            await asyncio.wait_for(sync, 2)               # the request handler returns
        asyncio.run(go())
        self.assertEqual(ran_on, ["cancelled"])
        self.assertEqual(self.jobs.failed, "cancelled by user")


class InterruptedIsNoFault(unittest.TestCase):
    """A prompt somebody stopped on the backend ends the job: no failover (that would run
    what was just stopped) and no execution fault (that could quarantine the backend)."""

    def setUp(self):
        self._orig = (main.jobs, main._free_comfy_vram, main._comfy_free)
        self.failed = []
        main.jobs = types.SimpleNamespace(
            set_status=lambda j, s: True, set_backend=lambda j, b: None,
            complete=lambda *a: self.fail("completed"),
            fail=lambda j, msg, meta=None: self.failed.append(msg))

        async def _nothing(*a, **k):
            return True
        main._free_comfy_vram = main._comfy_free = _nothing
        main.gen_exec_faults.clear()
        main.backend_adapters.clear()

    def tearDown(self):
        main.jobs, main._free_comfy_vram, main._comfy_free = self._orig
        main.backend_adapters.clear()
        main.gen_exec_faults.clear()

    def test_no_failover_and_no_fault(self):
        calls = []

        class _Ad:
            def __init__(self, boom):
                self.boom = boom

            async def generate(self, req):
                calls.append(self)
                if self.boom:
                    raise self.boom
                return types.SimpleNamespace(blobs=[], meta={})

        a = _Ad(adapters.ComfyPromptInterrupted("ComfyUI: prompt p1 was interrupted on the backend"))
        b = _Ad(None)
        main.backend_adapters.update({"comfyui:a": a, "comfyui:b": b})
        cands = [({"name": n, "type": "comfyui", "url": "http://x"}, {"backend": n}) for n in "ab"]
        asyncio.run(main._run_job("j", "alias1", cands,
                                  lambda bk, c: types.SimpleNamespace(slot_held=False)))
        self.assertEqual(calls, [a])
        self.assertIn("interrupted", self.failed[0])
        self.assertEqual(main.gen_exec_faults, {})



class WorkerCrash(unittest.TestCase):
    """K10: a generation worker that dies of an unanticipated exception (a KeyError on a
    backend deleted mid-job, a bug) left its row `queued`/`running` until the next process
    restart — a job that looks alive forever, and a sync caller that got a bare 500."""

    def setUp(self):
        self._orig = main.jobs
        main.jobs = self.jobs = _Jobs()

    def tearDown(self):
        main.jobs = self._orig

    def test_a_crashed_worker_fails_its_job(self):
        async def boom():
            raise KeyError("comfyui:gone")

        async def go():
            await main._run_gen_sync("jobK", boom())     # never raises into the handler
            await asyncio.sleep(0)                       # let the done callback run
        asyncio.run(go())
        self.assertIn("internal error", self.jobs.failed)
        self.assertIn("KeyError", self.jobs.failed)
        self.assertNotIn("jobK", main._gen_tasks)

    def test_a_clean_worker_is_left_alone(self):
        async def fine():
            return True
        asyncio.run(main._run_gen_sync("jobF", fine()))
        self.assertIsNone(self.jobs.failed)


class AdapterRebuildKeepsState(unittest.TestCase):
    """K19: every backend save rebuilt EVERY adapter, dropping its runtime state. The
    auto-restart cooldown reset (a stuck box could be restarted again right after an
    unrelated edit), the watchdog forgot what it had seen, and a job running on the old
    instance could no longer be cancelled through the new one — its prompt registry was
    gone. All of it silent: the adapter simply starts over."""

    def setUp(self):
        self._saved = (main.backends[:], dict(main.backend_adapters))
        self.comfy = {"name": "gpu", "type": "comfyui", "url": "http://gpu:8188"}
        self.cloud = {"name": "m", "type": "meshy", "url": "https://api.meshy.ai", "api_key": "k"}
        main.backends[:] = [self.comfy, self.cloud]
        main.build_backend_adapters()
        ad = main.backend_adapters["comfyui:gpu"]
        ad.last_restart, ad._node_types, ad._prompts["job1"] = 123.0, {"X": {}}, "p1"
        main.backend_adapters["meshy:m"].credits = 42

    def tearDown(self):
        main.backends[:] = self._saved[0]
        main.backend_adapters.clear()
        main.backend_adapters.update(self._saved[1])

    def _rebuild(self, comfy=None, cloud=None):
        main.backends[:] = [comfy or dict(self.comfy), cloud or dict(self.cloud)]
        main.build_backend_adapters()
        return main.backend_adapters["comfyui:gpu"], main.backend_adapters["meshy:m"]

    def test_an_unchanged_backend_keeps_its_instance(self):
        before = main.backend_adapters["comfyui:gpu"]
        ad, _ = self._rebuild()
        self.assertIs(ad, before)
        self.assertIs(ad.backend, main.backends[0])          # bound to the CURRENT dict

    def test_a_changed_backend_keeps_what_still_applies(self):
        old = main.backend_adapters["comfyui:gpu"]
        ad, cl = self._rebuild(comfy={**self.comfy, "max_wait": 900},
                               cloud={**self.cloud, "max_wait": 900})
        self.assertIsNot(ad, old)
        self.assertEqual(ad.last_restart, 123.0)
        self.assertIs(ad._prompts, old._prompts)              # the running job stays cancellable
        self.assertEqual(ad._node_types, {"X": {}})
        self.assertEqual(cl.credits, 42)

    def test_a_new_url_drops_the_host_bound_caches_but_not_the_cooldown(self):
        ad, cl = self._rebuild(comfy={**self.comfy, "url": "http://other:8188"},
                               cloud={**self.cloud, "api_key": "other"})
        self.assertEqual(ad._node_types, {})
        self.assertEqual(ad.last_restart, 123.0)
        self.assertIsNone(cl.credits)                          # another account's balance


class BackgroundTasksAreHeld(unittest.TestCase):
    """K20: `asyncio.create_task` keeps only a WEAK reference — an unreferenced fire-and-
    forget task (the after-job VRAM free, a ComfyUI restart) can be collected mid-flight.
    A lost restart never runs its `finally`, and that backend stays in
    `_comfy_restarting` — never restartable again, manually or automatically."""

    def test_a_background_task_is_referenced_until_it_finishes(self):
        async def go():
            gate = asyncio.Event()

            async def work():
                await gate.wait()
            t = main._bg(work())
            self.assertIn(t, main._bg_refs)
            gate.set()
            await t
            await asyncio.sleep(0)
            self.assertNotIn(t, main._bg_refs)
        asyncio.run(go())

    def test_a_restart_that_cannot_start_does_not_block_the_backend_forever(self):
        b = {"name": "gpu", "type": "comfyui", "url": "http://gpu"}
        orig = main._note_fault

        def boom(*a, **k):
            raise RuntimeError("fault log unavailable")
        main._note_fault = boom
        try:
            with self.assertRaises(RuntimeError):
                main._spawn_comfy_restart(b, types.SimpleNamespace(), "test")
        finally:
            main._note_fault = orig
        self.assertNotIn("comfyui:gpu", main._comfy_restarting)

if __name__ == "__main__":
    unittest.main()
