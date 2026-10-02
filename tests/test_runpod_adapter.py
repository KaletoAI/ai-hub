"""The RunPod Serverless backend (adapters.RunpodAdapter) against a scripted RunPod.

Why this fails SILENTLY: every mistake here still ends in a plausible job row. A job id
lost on a dropped /run answer becomes a second billed run on the next attempt; a poll
that gives up without /cancel leaves a GPU billing until its timeout; a queued job that
never gets a CUDA-13 worker holds the gateway slot for the whole max_wait; a manifest
read differently from /view delivers less than the alias asks for — and the workflow
RunPod receives must be exactly what a local ComfyUI would, or one alias renders two
different pictures depending on where it ran.

Run: venv/bin/python -m unittest tests.test_runpod_adapter -v
"""
import asyncio
import base64
import copy
import hashlib
import json
import os
import sys
import unittest
import unittest.mock

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import httpx  # noqa: E402
import adapters  # noqa: E402

URL = "https://api.runpod.ai/v2/ep123"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40


def _b64(d):
    return base64.b64encode(d).decode()


def _entry(d):
    return {"b64": _b64(d), "size": len(d), "sha256": hashlib.sha256(d).hexdigest()}


class _RunPod:
    """Scripted RunPod: /run → id, /status walks `statuses` (last repeats), /cancel
    answers `cancel_reply`, /health + REST endpoint for discovery."""

    def __init__(self, statuses, cancel_reply=None, run_status=200, run_exc=None):
        self.statuses = list(statuses)
        self.cancel_reply = cancel_reply or {"id": "rp1", "status": "CANCELLED"}
        self.run_status, self.run_exc = run_status, run_exc
        self.runs, self.cancels, self.paths = [], [], []
        self.health = {"jobs": {"inQueue": 0, "inProgress": 0},
                       "workers": {"idle": 0, "running": 0}}
        self.endpoint = {"id": "ep123", "workersMax": 1, "gpuTypeIds": ["NVIDIA L40"]}

    def handler(self, request):
        p = request.url.path
        self.paths.append((request.method, request.url.host, p))
        if p.endswith("/run"):
            if self.run_exc:
                raise self.run_exc
            self.runs.append(json.loads(request.read()))
            if self.run_status != 200:
                return httpx.Response(self.run_status, json={"error": "nope"})
            return httpx.Response(200, json={"id": "rp1", "status": "IN_QUEUE"})
        if "/status/" in p:
            if self.cancels and getattr(self, "after_cancel", None) is not None:
                st = self.after_cancel
            else:
                st = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            if isinstance(st, int):
                return httpx.Response(st, json={})
            if isinstance(st, bytes):                    # a body that is no JSON
                return httpx.Response(200, content=st)
            if isinstance(st, list):                     # JSON, but not an object
                return httpx.Response(200, json=st)
            return httpx.Response(200, json={"id": "rp1", **st})
        if "/cancel/" in p:
            self.cancels.append(p.rsplit("/", 1)[1])
            return httpx.Response(200, json=self.cancel_reply)
        if p.endswith("/health"):
            return httpx.Response(200, json=self.health)
        if request.url.host == "rest.runpod.io" and p == "/v1/endpoints/ep123":
            return httpx.Response(200, json=self.endpoint)
        return httpx.Response(404, json={})


def _ctx(**kw):
    return adapters.AdapterContext(
        auth_headers=lambda b: {}, inflight_inc=lambda b: None, inflight_dec=lambda b: None,
        cost_usd=lambda *a: 0.0, source_of=lambda r: "t", record_call=lambda *a, **k: None,
        log_enabled=lambda: False, **kw)


def _adapter(**extra):
    b = {"name": "rp", "type": "runpod", "url": URL, "api_key": "k",
         "poll_interval": 0.01, "disconnect_grace": 0.05, "queue_max_s": 0.05,
         "max_wait": 2, "cost_per_hour": 1.8, **extra}
    return adapters.RunpodAdapter(b, _ctx())


WF = {"1": {"class_type": "LoadImage", "inputs": {"image": ""}},
      "9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "o"}}}


def _req(**kw):
    base = dict(alias="a", job_id="job1", upload_prefix="gw_job1",
                workflow_json=copy.deepcopy(WF),
                node_mapping={"image": {"node": "1", "field": "image"}},
                upload_images={"image": PNG}, output_node="9")
    base.update(kw)
    return adapters.NormalizedRequest(**base)


DONE = {"status": "COMPLETED", "delayTime": 1200, "executionTime": 36000, "output": {
    "outputs": {"9": {"images": [{"filename": "o_00001_.png", "subfolder": "", "type": "output"}]}},
    "manifest": {"output/o_00001_.png": _entry(b"img")}, "worker_version": "abc"}}


def _run(rp, coro_fn):
    real = httpx.AsyncClient
    mk = lambda *a, **kw: real(transport=httpx.MockTransport(rp.handler),
                               **{k: v for k, v in kw.items() if k != "transport"})
    with unittest.mock.patch.object(adapters.httpx, "AsyncClient", mk):
        return asyncio.run(coro_fn())


class Generate(unittest.TestCase):
    def test_success_delivers_from_the_manifest_with_meta(self):
        rp = _RunPod([{"status": "IN_QUEUE"}, {"status": "IN_PROGRESS"}, DONE])
        ad = _adapter()
        out = _run(rp, lambda: ad.generate(_req()))
        self.assertEqual([(b.name, b.data) for b in out.blobs], [("o_00001_.png", b"img")])
        self.assertEqual(out.meta["runpod_job_id"], "rp1")
        self.assertEqual((out.meta["delay_ms"], out.meta["execution_ms"]), (1200, 36000))
        self.assertEqual(out.meta["worker_version"], "abc")
        self.assertAlmostEqual(out.meta["cost_est_usd"], 0.018, places=4)
        self.assertIn("lower bound", out.meta["cost_basis"])

    def test_the_submitted_workflow_is_what_comfyui_would_get(self):
        rp = _RunPod([DONE])
        req = _req()
        _run(rp, lambda: _adapter().generate(req))
        sent = rp.runs[0]["input"]
        self.assertEqual(sent["op"], "prompt")
        self.assertEqual(sent["workflow"]["1"]["inputs"]["image"], "gw_job1_image.png")
        self.assertEqual(sent["inputs"], [{"name": "gw_job1_image.png", "b64": _b64(PNG)}])
        self.assertEqual(rp.runs[0]["policy"], {"executionTimeout": 2000, "ttl": 2050})

    def test_failed_is_an_execution_error_settled_and_run_once(self):
        rp = _RunPod([{"status": "FAILED", "error": "node 4: CUDA out of memory"}])
        req = _req()
        with self.assertRaises(RuntimeError) as cm:
            _run(rp, lambda: _adapter().generate(req))
        self.assertIn("CUDA out of memory", str(cm.exception))
        self.assertEqual(len(rp.runs), 1)
        self.assertTrue(req.cloud_trace["runpod_settled"])

    def test_queue_cap_cancels_then_busy(self):
        rp = _RunPod([{"status": "IN_QUEUE"}])
        req = _req()
        with self.assertRaises(adapters.CloudBusy):
            _run(rp, lambda: _adapter().generate(req))
        self.assertEqual(rp.cancels, ["rp1"])
        self.assertTrue(req.cloud_trace["runpod_settled"])

    def test_timed_out_before_pickup_is_busy_after_is_final(self):
        rp = _RunPod([{"status": "TIMED_OUT", "executionTime": 0}])
        with self.assertRaises(adapters.CloudBusy):
            _run(rp, lambda: _adapter(queue_max_s=60).generate(_req()))
        rp = _RunPod([{"status": "TIMED_OUT", "executionTime": 5000}])
        with self.assertRaises(RuntimeError) as cm:
            _run(rp, lambda: _adapter(queue_max_s=60).generate(_req()))
        self.assertNotIsInstance(cm.exception, ConnectionError)

    def test_poll_grace_cancels_and_failover_only_when_confirmed(self):
        rp = _RunPod([{"status": "IN_PROGRESS"}, 503])
        req = _req()
        with self.assertRaises(ConnectionError):
            _run(rp, lambda: _adapter(queue_max_s=60).generate(req))
        self.assertEqual(rp.cancels, ["rp1"])
        self.assertTrue(req.cloud_trace["runpod_settled"])
        rp = _RunPod([{"status": "IN_PROGRESS"}, 503], cancel_reply={"status": "IN_PROGRESS"})
        req = _req()
        with self.assertRaises(ConnectionError):
            _run(rp, lambda: _adapter(queue_max_s=60).generate(req))
        self.assertFalse(req.cloud_trace.get("runpod_settled"))

    def test_max_wait_cancels_then_timeout(self):
        rp = _RunPod([{"status": "IN_PROGRESS"}])
        with self.assertRaises(TimeoutError):
            _run(rp, lambda: _adapter(queue_max_s=60, max_wait=0.05).generate(_req()))
        self.assertEqual(rp.cancels, ["rp1"])

    def test_lost_create_answer_is_unconfirmed(self):
        rp = _RunPod([DONE], run_exc=httpx.ReadTimeout("lost"))
        req = _req()
        with self.assertRaises(httpx.ReadTimeout):
            _run(rp, lambda: _adapter().generate(req))
        self.assertTrue(req.cloud_trace["create_unconfirmed"])
        self.assertTrue(req.cloud_trace["runpod"])

    def test_refused_create_is_failover_class_and_nothing_exists(self):
        rp = _RunPod([DONE], run_status=503)
        req = _req()
        with self.assertRaises(ConnectionError):
            _run(rp, lambda: _adapter().generate(req))
        self.assertNotIn("runpod_job_id", req.cloud_trace)

    def test_status_404_is_final_expired(self):
        rp = _RunPod([404])
        with self.assertRaises(RuntimeError) as cm:
            _run(rp, lambda: _adapter().generate(_req()))
        self.assertIn("expired", str(cm.exception))

    def test_cancelled_is_an_interrupt(self):
        rp = _RunPod([{"status": "CANCELLED"}])
        with self.assertRaises(adapters.ComfyPromptInterrupted):
            _run(rp, lambda: _adapter().generate(_req()))

    def test_payload_over_9mb_refused_before_any_request(self):
        rp = _RunPod([DONE])
        with self.assertRaises(RuntimeError) as cm:
            _run(rp, lambda: _adapter().generate(_req(upload_images={"image": b"x" * (7 * 1024 * 1024)})))
        self.assertIn("9 MB", str(cm.exception))
        self.assertEqual(rp.runs, [])

    def test_manifest_missing_is_404_and_bad_sha_raises(self):
        bad = copy.deepcopy(DONE)
        bad["output"]["manifest"]["output/o_00001_.png"]["sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            _run(_RunPod([bad]), lambda: _adapter().generate(_req()))
        gone = copy.deepcopy(DONE)
        gone["output"]["manifest"] = {"output/o_00001_.png": None}
        with self.assertRaises(RuntimeError) as cm:
            _run(_RunPod([gone]), lambda: _adapter().generate(_req()))
        self.assertIn("no fetchable artifact", str(cm.exception))

    def test_cancelling_the_worker_cancels_the_runpod_job(self):
        rp = _RunPod([{"status": "IN_PROGRESS"}])
        ad = _adapter(queue_max_s=60, max_wait=30)

        async def go():
            t = asyncio.create_task(ad.generate(_req()))
            for _ in range(300):
                await asyncio.sleep(0.01)
                if ad._rp_jobs.get("job1"):
                    break
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
        _run(rp, go)
        self.assertEqual(rp.cancels, ["rp1"])
        self.assertNotIn("job1", ad._rp_jobs)

    def test_cancel_while_run_answer_pending_is_unconfirmed(self):
        rp = _RunPod([DONE])
        req = _req()

        orig = rp.handler

        class Slow(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                if request.url.path.endswith("/run"):
                    await asyncio.sleep(30)
                return orig(request)
        real = httpx.AsyncClient
        mk = lambda *a, **kw: real(transport=Slow(), **{k: v for k, v in kw.items() if k != "transport"})

        async def go():
            t = asyncio.create_task(_adapter().generate(req))
            await asyncio.sleep(0.1)
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
        with unittest.mock.patch.object(adapters.httpx, "AsyncClient", mk):
            asyncio.run(go())
        self.assertTrue(req.cloud_trace["create_unconfirmed"])

    def test_cancel_right_after_the_id_exists_sends_cancel_and_leaks_nothing(self):
        rp = _RunPod([{"status": "IN_PROGRESS"}])
        ad = _adapter(queue_max_s=60, max_wait=30)
        started = []

        def block(job_id, meta):
            started.append(job_id)
            import time
            time.sleep(0.3)
        ad.ctx.note_job_meta = block

        async def go():
            t = asyncio.create_task(ad.generate(_req()))
            for _ in range(300):
                await asyncio.sleep(0.01)
                if started:
                    break
            t.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t
        _run(rp, go)
        self.assertEqual(rp.cancels, ["rp1"])
        self.assertEqual(ad._rp_jobs, {})

    def test_run_200_without_id_is_unconfirmed(self):
        rp = _RunPod([DONE])
        orig = rp.handler
        rp.handler = lambda r: (httpx.Response(200, json={}) if r.url.path.endswith("/run") else orig(r))
        req = _req()
        with self.assertRaises(RuntimeError):
            _run(rp, lambda: _adapter().generate(req))
        self.assertTrue(req.cloud_trace["create_unconfirmed"])

    def test_cancel_after_url_change_hits_the_old_endpoint(self):
        rp = _RunPod([DONE])
        old = _adapter()
        old._rp_jobs["job1"] = (URL, "rp1")
        new = adapters.RunpodAdapter({**old.backend, "url": "https://api.runpod.ai/v2/other"}, _ctx())
        new.adopt_state(old)
        _run(rp, lambda: new.cancel("job1"))
        self.assertEqual([p for m, h, p in rp.paths if "/cancel/" in p], ["/v2/ep123/cancel/rp1"])

    def test_concurrent_jobs_do_not_share_inputs(self):
        rp = _RunPod([DONE])
        ad = _adapter()

        async def go():
            await asyncio.gather(ad.generate(_req(job_id="j1", upload_prefix="gw_j1")),
                                 ad.generate(_req(job_id="j2", upload_prefix="gw_j2")))
        _run(rp, go)
        names = sorted(tuple(i["name"] for i in r["input"]["inputs"]) for r in rp.runs)
        self.assertEqual(names, [("gw_j1_image.png",), ("gw_j2_image.png",)])

    def test_registry_and_kind(self):
        self.assertIs(adapters.ADAPTERS["runpod"], adapters.RunpodAdapter)
        self.assertIn("runpod", adapters.GEN_TYPES)
        self.assertNotIn("runpod", adapters.CLOUD_TYPES)
        self.assertIn("runpod", adapters.BILLING_TYPES)
        self.assertEqual(adapters.backend_kind({"type": "runpod"}), "comfyui")
        self.assertEqual(adapters.runpod_endpoint_id(URL), "ep123")
        self.assertIsNone(adapters.runpod_endpoint_id("https://api.runpod.ai/v2/"))


import gzip  # noqa: E402

OI = {"KSampler": {"input": {"required": {"seed": ["INT", {}]}}, "output": ["LATENT"]},
      "UnetLoaderGGUF": {"input": {"required": {"unet_name": [["q.gguf"], {}]}}, "output": ["MODEL"]},
      "LoraLoader": {"input": {"required": {"lora_name": [["style.safetensors"], {}]}},
                     "output": ["MODEL"]}}
INFO = {"status": "COMPLETED", "executionTime": 900, "output": {
    "object_info_gz": _b64(gzip.compress(json.dumps(OI).encode())),
    "models": {"image": {}, "volume": {"models/unet/q.gguf": 4}}, "worker_version": "v7"}}


class DiscoveryAndProbe(unittest.TestCase):
    def _disc(self, rp, ad):
        async def go():
            async with httpx.AsyncClient() as c:
                return await ad.discover(c)
        return _run(rp, go)

    def test_unprobed_backend_has_empty_caps_and_says_so(self):
        rp = _RunPod([DONE])
        ad = _adapter()
        caps = self._disc(rp, ad)
        self.assertEqual((caps.models, caps.loras), (set(), set()))
        self.assertEqual(ad.probe_state["state"], "none")
        self.assertFalse(any(p.endswith("/run") for _m, _h, p in rp.paths))

    def test_workers_max_zero_is_down_with_the_cause(self):
        rp = _RunPod([DONE])
        rp.endpoint["workersMax"] = 0
        with self.assertRaises(RuntimeError) as cm:
            self._disc(rp, _adapter())
        self.assertIn("raise max workers", str(cm.exception))

    def test_refused_key_is_down_named(self):
        rp = _RunPod([DONE])
        orig = rp.handler
        rp.handler = lambda r: (httpx.Response(401, json={}) if r.url.path.endswith("/health")
                                else orig(r))
        with self.assertRaises(RuntimeError) as cm:
            self._disc(rp, _adapter())
        self.assertIn("API key", str(cm.exception))

    def test_probe_stores_the_snapshot_and_discovery_serves_it(self):
        saved = {}
        rp = _RunPod([INFO])
        b = {"name": "rp", "type": "runpod", "url": URL, "api_key": "k", "poll_interval": 0.01}
        ad = adapters.RunpodAdapter(b, _ctx(runpod_probe_save=lambda n, rec: saved.update({n: rec})))
        real = httpx.AsyncClient
        ad.ctx.http_client = lambda: real(transport=httpx.MockTransport(rp.handler))
        st = asyncio.run(ad.probe())
        self.assertEqual(st["state"], "ok")
        self.assertEqual(st["worker_version"], "v7")
        self.assertEqual(rp.runs[0]["input"], {"op": "info"})
        self.assertIn("rp", saved)
        caps = self._disc(_RunPod([DONE]), ad)
        self.assertIn("q.gguf", caps.models)
        self.assertIn("style.safetensors", caps.loras)
        self.assertIn("KSampler", ad._node_types)
        # a fresh adapter (gateway restart) loads it from the store on its first discovery
        ad2 = adapters.RunpodAdapter(b, _ctx(runpod_probe_load=lambda n: saved.get(n)))
        caps2 = self._disc(_RunPod([DONE]), ad2)
        self.assertIn("q.gguf", caps2.models)
        self.assertEqual(ad2.probe_state["state"], "ok")

    def test_failed_probe_says_why_and_keeps_the_old_snapshot(self):
        rp = _RunPod([{"status": "FAILED", "error": "ComfyUI did not start in the worker"}])
        b = {"name": "rp", "type": "runpod", "url": URL, "api_key": "k", "poll_interval": 0.01}
        ad = adapters.RunpodAdapter(b, _ctx())
        real = httpx.AsyncClient
        ad.ctx.http_client = lambda: real(transport=httpx.MockTransport(rp.handler))
        st = asyncio.run(ad.probe())
        self.assertEqual(st["state"], "failed")
        self.assertIn("did not start", st["error"])


import tempfile  # noqa: E402


def _main():
    """Import main inside a temp cwd with an empty config (the pattern of test_gen_cancel)."""
    prev = os.getcwd()
    t = tempfile.TemporaryDirectory()
    with open(os.path.join(t.name, "config.yaml"), "w") as f:
        f.write('api_key: ""\nbackends: []\n')
    os.chdir(t.name)
    try:
        import main
    finally:
        os.chdir(prev)
        t.cleanup()
    return main


class BilledPredicate(unittest.TestCase):
    def setUp(self):
        self.main = _main()

    def test_unsettled_runpod_job_blocks_exec_failover(self):
        m = self.main
        cand = {"backend": "rp", "workflow_json": {}}
        tr = {"runpod": True, "runpod_job_id": "rp1", "runpod_settled": False}
        self.assertIn("RunPod job rp1", m._billed_cloud_task(cand, tr, RuntimeError("x")))
        tr["runpod_settled"] = True
        self.assertIsNone(m._billed_cloud_task(cand, tr, RuntimeError("x")))

    def test_lost_create_answer_is_billed(self):
        tr = {"runpod": True, "create_unconfirmed": True}
        self.assertIn("answer was lost",
                      self.main._billed_cloud_task({"backend": "rp"}, tr, ConnectionError("x")))

    def test_stale_settled_id_does_not_hide_a_lost_answer(self):
        tr = {"runpod": True, "runpod_job_id": "rp1", "runpod_settled": True,
              "create_unconfirmed": True}
        self.assertIn("answer was lost",
                      self.main._billed_cloud_task({"backend": "rp"}, tr, ConnectionError("x")))

    def test_a_refused_create_is_not_billed(self):
        self.assertIsNone(self.main._billed_cloud_task({"backend": "rp"}, {"runpod": True},
                                                       ConnectionError("x")))

    def test_runpod_is_paid(self):
        m = self.main
        saved = (list(m.backends), dict(m.backend_adapters))
        try:
            with unittest.mock.patch.object(m, "config_backends",
                                            [{"name": "rp", "type": "runpod", "url": URL}]):
                with unittest.mock.patch.object(m.store, "is_active", lambda: False):
                    m.rebuild_backends()
            self.assertTrue(next(b for b in m.backends if b["name"] == "rp")["paid"])
        finally:
            m.backends[:] = saved[0]


class SubmitTrace(unittest.TestCase):
    def test_submit_clears_the_previous_attempts_keys(self):
        ad = _adapter()
        req = _req()
        req.cloud_trace.update({"runpod_job_id": "rp1", "runpod_settled": True,
                                "create_unconfirmed": True})
        rp = _RunPod([DONE], run_status=400)

        async def go():
            async with httpx.AsyncClient(transport=httpx.MockTransport(rp.handler)) as c:
                with self.assertRaises(RuntimeError):
                    await ad._submit(c, URL, b"{}", req)
        asyncio.run(go())
        tr = req.cloud_trace
        self.assertNotIn("runpod_job_id", tr)
        self.assertNotIn("runpod_settled", tr)
        self.assertNotIn("create_unconfirmed", tr)


class StartupCancelRobust(unittest.TestCase):
    def test_one_failing_row_does_not_abort_the_rest_and_missing_backend_is_logged(self):
        m = _main()
        cancelled = []

        class _Ad:
            async def cancel_runpod_id(self, rp_id):
                if rp_id == "bad":
                    raise RuntimeError("boom")
                cancelled.append(rp_id)
                return True
        b = {"name": "rp", "type": "runpod", "url": URL}
        with unittest.mock.patch.object(m, "backends", [b]), \
                unittest.mock.patch.dict(m.backend_adapters, {m.backend_id(b): _Ad()}), \
                unittest.mock.patch.object(m.jobs, "merge_meta", lambda *a: None), \
                unittest.mock.patch.object(m.logger, "warning") as warn:
            asyncio.run(m._cancel_orphaned_runpod(
                [("j0", "gone", {"runpod_job_id": "rp0"}), ("j1", "rp", {"runpod_job_id": "bad"}),
                 ("j2", "rp", {"runpod_job_id": "rp2"})]))
        self.assertEqual(cancelled, ["rp2"])
        text = "\n".join(str(c.args[0]) for c in warn.call_args_list)
        self.assertIn("rp0", text)
        self.assertIn("boom", text)

    def test_note_job_meta_never_raises(self):
        m = _main()
        with unittest.mock.patch.object(m.jobs, "_active", True), \
                unittest.mock.patch.object(m.jobs, "merge_meta", side_effect=RuntimeError("locked")):
            m._note_job_meta("j", {"a": 1})


class StartupCancel(unittest.TestCase):
    def test_startup_cancels_orphaned_runpod_job(self):
        m = _main()
        cancelled = []

        class _Ad:
            async def cancel_runpod_id(self, rp_id):
                cancelled.append(rp_id)
                return True
        b = {"name": "rp", "type": "runpod", "url": URL}
        with unittest.mock.patch.object(m, "backends", [b]), \
                unittest.mock.patch.dict(m.backend_adapters, {m.backend_id(b): _Ad()}), \
                unittest.mock.patch.object(m.jobs, "merge_meta", lambda *a: None):
            n = asyncio.run(m._cancel_orphaned_runpod(
                [("j1", "rp", {"runpod_job_id": "rp1"}), ("j2", "gpu", {})]))
        self.assertEqual((n, cancelled), (1, ["rp1"]))


class Jobs(unittest.TestCase):
    def test_reconcile_remembers_the_orphans_with_their_meta(self):
        import jobs
        with tempfile.TemporaryDirectory() as t:
            jobs.init(os.path.join(t, "j.db"), os.path.join(t, "b"))
            jid = jobs.create("image", "a", "rp")
            jobs.set_status(jid, "running")
            jobs.merge_meta(jid, {"runpod_job_id": "rp9"})
            jobs.reconcile_orphans()
            self.assertEqual(jobs.last_orphans(), [(jid, "rp", {"runpod_job_id": "rp9"})])


class Console(unittest.TestCase):
    def setUp(self):
        _main()
        import admin
        self.admin = admin

    def test_type_select_offers_runpod_forces_paid_and_names_the_tab(self):
        html = self.admin._type_select("runpod")
        self.assertIn('<option value="runpod" selected>runpod</option>', html)
        self.assertIn("RunPod", html)
        self.assertIn("runpod", html.split("bill=", 1)[1][:80])
        self.assertEqual(self.admin._type_tab_label("runpod"), "RunPod")

    def test_form_renders_the_runpod_block(self):
        html = self.admin._backend_form({"name": "rp", "type": "runpod", "url": URL,
                                         "queue_max_s": 120, "cost_per_hour": 1.75}, [])
        self.assertIn('data-btype="runpod"', html)
        for f in ("rp_max_wait", "rp_poll_interval", "queue_max_s", "cost_per_hour"):
            self.assertIn(f'name="{f}"', html)
        self.assertIn("paid — always", html)

    def test_list_row_badge_probe_button_and_escaped_state(self):
        a = self.admin
        self.assertIn("runpod", a._type_badge("runpod"))
        info = {"backends": [{
            "name": "rp", "type": "runpod", "url": URL, "enabled": True, "healthy": True,
            "models": 0, "source": "ui", "paid": True,
            "runpod": {"workers_idle": 1, "workers_running": 2, "in_queue": 3, "workers_max": 5,
                       "probe": {"state": "failed", "error": "<script>x</script>"}}}],
            "hosts": {}}
        with unittest.mock.patch.object(a, "_gateway_info", lambda: info), \
                unittest.mock.patch.object(a.store, "is_active", lambda: False):
            r = asyncio.run(a._backends_view({}))
        html = r.body.decode()
        self.assertIn("/ui/backends/runpod-probe?id=", html)
        self.assertIn("⚡", html)
        self.assertIn("probe failed", html)
        self.assertNotIn("<script>x</script>", html)
        self.assertIn("probe failed: &lt;script&gt;x&lt;/script&gt;", html)
        self.assertIn("workers 2 running / 1 idle, max 5 · queue 3", html)

    def test_probe_is_a_post_action(self):
        self.assertTrue(self.admin._is_post_action("/ui/backends/runpod-probe?id=runpod:rp"))

    def test_job_view_table(self):
        t = self.admin._runpod_table({"runpod_job_id": "rp1", "delay_ms": 1200,
                                      "execution_ms": 36000, "cost_est_usd": 0.018,
                                      "cost_basis": "execution only — lower bound",
                                      "worker_version": "abc"})
        for s in ("rp1", "1.2 s", "36.0 s", "0.018", "lower bound", "abc"):
            self.assertIn(s, t)
        self.assertEqual(self.admin._runpod_table({}), "")

    def test_editor_widgets_come_from_the_probe_snapshot(self):
        a = self.admin
        with unittest.mock.patch.object(a, "_runpod_object_info", lambda n: OI if n == "rp" else None):
            oi = asyncio.run(a._object_info("rp", {"1": {"class_type": "UnetLoaderGGUF",
                                                           "inputs": {}}}))
        self.assertEqual(oi["UnetLoaderGGUF"]["unet_name"], ["q.gguf"])



class NoChainRoles(unittest.TestCase):
    """Item 1 (final review): a RunpodAdapter inherited ComfyUI's chain roles — a stage 1
    ran (billed) and only then GET /view on api.runpod.ai failed; a stage 2 uploaded to
    api.runpod.ai/upload/image. Milestone 1 has no chains: refused by name, up front."""

    def test_hooks_refuse_like_the_base_class(self):
        ad = _adapter()
        ex = ad.chain_export({"workflow_json": WF}, {"export_node": "9"}, {}, "gwchain_j")
        self.assertIn("cannot be a chain stage", ex.error)
        self.assertEqual(ex.mesh_name, "")
        with self.assertRaises(RuntimeError):
            asyncio.run(ad.chain_take_mesh(adapters.GenOutput(blobs=[]), ex, True))
        with self.assertRaises(RuntimeError):
            asyncio.run(ad.chain_feed_mesh(adapters.NormalizedRequest(alias="r"), {}, "m",
                                           "x.glb", b"glTF", ""))

    def test_a_runpod_stage1_candidate_is_refused_before_any_run(self):
        import meshy
        import tests.test_run_job_failover as rjf
        m = rjf.main
        h = rjf.RunJobExecFailover("test_execution_error_fails_over_to_the_next_backend")
        h.setUp()
        fj = rjf._ChainJobs()
        m.jobs = fj
        saved = (m._gen_backends[:], dict(m.image_models), dict(m.backend_healthy),
                 m.store._active, m._gen_waiting[:])
        try:
            m.store._active = False
            rpb = {"name": "rp", "type": "runpod", "url": URL, "api_key": "k", "enabled": True,
                   "paid": True}
            rb = {"name": "r", "type": "meshy", "enabled": True}
            m._gen_backends[:] = [rpb, rb]
            succ = {"alias": "rig", "mesh_param": "input_mesh_path", "export_node": "9"}
            rig = meshy.default_candidate("r")
            rig["meshy"]["endpoint"] = "rigging"
            m.image_models.clear()
            m.image_models.update({"s1": [{"backend": "rp", "workflow_json": WF,
                                           "successor": succ}], "rig": [rig]})
            m.backend_healthy.clear()
            m.backend_healthy.update({"runpod:rp": True, "meshy:r": True})
            m._gen_waiting.clear()
            rp = _RunPod([DONE])
            ad = adapters.RunpodAdapter(rpb, _ctx())
            m.backend_adapters.update({"runpod:rp": ad, "meshy:r": rjf._Adapter()})
            _run(rp, lambda: m._run_chain("job1", "s1", succ, {}, None, {}, {}, {}, {}))
            self.assertIn("cannot be a chain stage", fj.failed["msg"])
            self.assertFalse(any(p.endswith("/run") for _m, _h, p in rp.paths))
            self.assertEqual(m.backend_inflight.get("runpod:rp", 0), 0)
        finally:
            gb, im, bh, sa, gw = saved
            m._gen_backends[:] = gb
            m.image_models.clear(); m.image_models.update(im)
            m.backend_healthy.clear(); m.backend_healthy.update(bh)
            m.store._active = sa
            m._gen_waiting[:] = gw
            h.tearDown()



class ProbeAcrossRebuild(unittest.TestCase):
    """Item 3 (final review): a backend save replaces the adapter. A copied "running"
    probe_state never ended on the replacement (every new probe refused, the Backends tab
    live for good); a snapshot load still awaited on the old instance left the new one
    "loaded" and empty; a probe finishing on the old instance was lost."""

    B = {"name": "rp", "type": "runpod", "url": URL, "api_key": "k", "poll_interval": 0.01}

    def test_a_running_probe_state_is_never_copied(self):
        old = adapters.RunpodAdapter(dict(self.B), _ctx())
        old.probe_state = {"state": "ok", "at": 5, "worker_version": "v1"}
        old._probe_prev = dict(old.probe_state)
        old.probe_state = {**old.probe_state, "state": "running", "started": 9}
        new = adapters.RunpodAdapter({**self.B, "max_wait": 900}, _ctx())
        new.adopt_state(old)
        self.assertEqual(new.probe_state, {"state": "ok", "at": 5, "worker_version": "v1"})

    def test_probe_saves_the_previous_settled_state(self):
        rp = _RunPod([{"status": "IN_PROGRESS"}, INFO])
        ad = adapters.RunpodAdapter(dict(self.B), _ctx())
        ad.probe_state = {"state": "failed", "error": "x"}
        real = httpx.AsyncClient
        ad.ctx.http_client = lambda: real(transport=httpx.MockTransport(rp.handler))
        seen = []

        async def go():
            t = asyncio.create_task(ad.probe())
            await asyncio.sleep(0)
            seen.append((ad.probe_state["state"], ad._settled_probe()["state"]))
            await t
        asyncio.run(go())
        self.assertEqual(seen, [("running", "failed")])
        self.assertEqual(ad.probe_state["state"], "ok")

    def test_snapshot_loaded_after_the_rebuild_reaches_the_replacement(self):
        import threading
        gate = threading.Event()
        rec = {"at": 1, "worker_version": "v7", "endpoint": "ep123",
               "object_info_gz": INFO["output"]["object_info_gz"], "models": {}}

        def slow_load(name):
            gate.wait(5)
            return rec
        old = adapters.RunpodAdapter(dict(self.B), _ctx(runpod_probe_load=slow_load))
        rp = _RunPod([DONE])

        async def go():
            async with httpx.AsyncClient() as c:
                t = asyncio.create_task(old.discover(c))
                await asyncio.sleep(0.05)
                self.assertFalse(old._snap_loaded)          # not before the load is over
                new = adapters.RunpodAdapter({**self.B, "max_wait": 900},
                                             _ctx(runpod_probe_load=lambda n: None))
                new.adopt_state(old)
                self.assertFalse(new._snap_loaded)          # → it would load on its own
                gate.set()
                await t
                new.adopt_discovery(old)                    # main.refresh_backend's hand-over
                return new
        new = _run(rp, go)
        self.assertTrue(new._snap_loaded)
        self.assertIn("q.gguf", new._models)
        self.assertIn("KSampler", new._node_types)
        self.assertEqual(new.probe_state["state"], "ok")

    def test_adopt_discovery_respects_the_url_and_a_newer_snapshot(self):
        old = adapters.RunpodAdapter(dict(self.B), _ctx())
        old._models, old._snap_loaded = {"old.gguf"}, True
        other = adapters.RunpodAdapter({**self.B, "url": "https://api.runpod.ai/v2/other"}, _ctx())
        other.adopt_discovery(old)
        self.assertEqual(other._models, set())
        mine = adapters.RunpodAdapter(dict(self.B), _ctx())
        mine._models, mine._snap_loaded = {"new.gguf"}, True
        mine.adopt_discovery(old)
        self.assertEqual(mine._models, {"new.gguf"})

    def test_a_probe_finishing_on_the_replaced_instance_reaches_the_current_one(self):
        m = _main()
        rp = _RunPod([INFO])
        old = adapters.RunpodAdapter(dict(self.B), _ctx(runpod_probe_save=lambda n, r: None))
        new = adapters.RunpodAdapter({**self.B, "max_wait": 900}, _ctx())
        far = adapters.RunpodAdapter({**self.B, "url": "https://api.runpod.ai/v2/other"}, _ctx())
        real = httpx.AsyncClient
        old.ctx.http_client = lambda: real(transport=httpx.MockTransport(rp.handler))
        for cur, want in ((new, "ok"), (far, "none")):
            old.probe_state = {"state": "none"}
            rp.statuses = [INFO]

            async def go():
                t = asyncio.create_task(m._runpod_probe_run("runpod:rp", old))
                await asyncio.sleep(0)
                self.assertEqual(old.probe_state["state"], "running")
                cur.adopt_state(old)                 # the save lands mid-probe
                self.assertNotEqual(cur.probe_state["state"], "running")
                m.backend_adapters["runpod:rp"] = cur
                await t
            try:
                asyncio.run(go())
            finally:
                m.backend_adapters.pop("runpod:rp", None)
            self.assertEqual(cur.probe_state["state"], want)
        self.assertIn("q.gguf", new._models)
        self.assertEqual(far._models, set())



class _RpFake:
    """A RunPod stand-in for `_run_job`: writes the trace exactly where RunpodAdapter's
    _submit/_poll_rp do, then raises `boom` (or returns one artifact)."""

    def __init__(self, boom=None, settled=False):
        self.boom, self.settled, self.calls = boom, settled, 0

    async def generate(self, req):
        self.calls += 1
        req.cloud_trace.update({"backend": "rp", "runpod": True,
                                "runpod_job_id": f"rpj{self.calls}",
                                "runpod_settled": self.settled})
        if self.boom is not None:
            raise self.boom
        import types
        return types.SimpleNamespace(blobs=[b"art"], meta={})


class RunJobBilling(unittest.TestCase):
    """Item 4 (final review): the billing invariant end to end through `_run_job`, not
    only at predicate level. An unsettled RunPod job may still run — and bill — at RunPod:
    a self-retry or the next candidate would pay for the same work twice, and the row
    would read like an ordinary failover."""

    def setUp(self):
        import tests.test_run_job_failover as rjf
        self.rjf, self.m = rjf, rjf.main
        self.h = rjf.RunJobExecFailover("test_execution_error_fails_over_to_the_next_backend")
        self.h.setUp()
        self.jobs = self.h.jobs

    def tearDown(self):
        self.h.tearDown()

    @staticmethod
    def _rp(name="rp", retries=0):
        return ({"name": name, "type": "runpod", "url": URL, "paid": True,
                 "self_retries": retries}, {"backend": name, "workflow_json": {}})

    def _run(self, cands, ads):
        import types
        self.m.backend_adapters.update(ads)
        asyncio.run(self.m._run_job("job1", "alias1", cands,
                                    lambda b, c: types.SimpleNamespace(slot_held=False,
                                                                       cloud_trace={})))

    def test_unsettled_execution_error_is_final(self):
        a, good = _RpFake(RuntimeError("worker crashed")), self.rjf._Adapter()
        self._run([self._rp(), self.h._comfy("good")], {"runpod:rp": a, "comfyui:good": good})
        self.assertEqual((a.calls, good.calls), (1, 0))
        self.assertIsNone(self.jobs.completed)
        msg = self.jobs.failed["msg"]
        self.assertIn("RunPod job rpj1", msg)
        self.assertIn("may still be running", msg)
        self.assertEqual(self.jobs.failed["meta"]["runpod_job_id"], "rpj1")

    def test_unsettled_failover_class_error_is_never_self_retried(self):
        for boom in (adapters.CloudBusy("RunPod /status 429", vendor="RunPod"),
                     ConnectionError("RunPod unreachable for >30s")):
            self.h.tearDown(); self.h.setUp(); self.jobs = self.h.jobs
            a, good = _RpFake(boom), self.rjf._Adapter()
            self._run([self._rp(retries=1), self.h._comfy("good")],
                      {"runpod:rp": a, "comfyui:good": good})
            self.assertEqual((a.calls, good.calls), (1, 0), boom)
            self.assertIn("RunPod job rpj1", self.jobs.failed["msg"])
            self.assertIn("may still be running", self.jobs.failed["msg"])

    def test_a_settled_failed_job_fails_over(self):
        a, good = _RpFake(RuntimeError("RunPod: node 4 OOM"), settled=True), self.rjf._Adapter()
        self._run([self._rp(), self.h._comfy("good")], {"runpod:rp": a, "comfyui:good": good})
        self.assertEqual((a.calls, good.calls), (1, 1))
        self.assertIsNotNone(self.jobs.completed)
        self.assertIsNone(self.jobs.failed)

    def test_runpod_never_reaches_comfy_free(self):
        """Spec §5.12: a RunPod endpoint has no /free — the claim and after-job paths are
        ComfyUI-only, or a POST /free goes to api.runpod.ai with the API key."""
        m = self.m
        rpb = self._rp()[0]
        m._free_comfy_vram = self.h._orig_free           # the REAL after-job policy
        m.backend_hosts[m.backend_id(rpb)] = "rp-host"
        # both policies switched ON for its host: only the type gate may keep it out
        m.hosts_meta["rp-host"] = {"comfy_free_after_job": True, "comfy_free_before_job": True}
        try:
            async def go():
                await m._claim_gen_backend(rpb, "alias1", "other-models")
                await m._free_comfy_vram(rpb, "job done")
                m.backend_adapters["runpod:rp"] = _RpFake()
                import types
                await m._run_job("job1", "alias1", [self._rp()],
                                 lambda b, c: types.SimpleNamespace(slot_held=False,
                                                                    cloud_trace={}))
                await asyncio.sleep(0.05)                # let the after-job _bg task run
            asyncio.run(go())
        finally:
            m.backend_hosts.pop(m.backend_id(rpb), None)
            m.hosts_meta.pop("rp-host", None)
        self.assertIsNotNone(self.jobs.completed)
        self.assertEqual(self.h.frees, [])

    def test_the_alias_editor_offers_a_runpod_backend_to_a_workflow_alias(self):
        import admin
        m = self.m
        b = [{"name": "gpu", "type": "comfyui", "url": "http://10.0.0.3:8188", "enabled": True},
             {"name": "rp", "type": "runpod", "url": URL, "enabled": True},
             {"name": "mz", "type": "meshy", "url": "https://api.meshy.ai", "enabled": True}]
        with unittest.mock.patch.object(m, "backends", b):
            self.assertIn("rp", [x["name"] for x in admin._gen_backends()])
            wf_cands = [{"backend": "gpu", "workflow_json": {}}]
            self.assertTrue(admin._same_kind(wf_cands, "rp"))
            self.assertIn("<option>rp</option>", admin._backends_section("a", wf_cands))
            cloud = [{"backend": "mz", "meshy": {"endpoint": "image-to-3d"}}]
            self.assertFalse(admin._same_kind(cloud, "rp"))
            self.assertNotIn("<option>rp</option>", admin._backends_section("a", cloud))



class PollClocksAndEnds(unittest.TestCase):
    """Items 5–7 (final review): how a poll ENDS. `max_wait` counted from submit let a
    cold start's queue + boot eat the execution budget RunPod still granted; a job that
    COMPLETED while being given up was cancelled and its paid result thrown away; a 200
    whose body was not a JSON object ended the job as unsettled; three 4xx gave up
    without the /cancel every other give-up sends."""

    def test_max_wait_counts_execution_not_the_queue(self):
        rp = _RunPod([{"status": "IN_QUEUE"}] * 12 + [{"status": "IN_PROGRESS"}, DONE])
        req = _req()
        out = _run(rp, lambda: _adapter(queue_max_s=60, max_wait=0.08).generate(req))
        self.assertEqual([b.name for b in out.blobs], ["o_00001_.png"])
        self.assertEqual(rp.cancels, [])

    def test_execution_beyond_max_wait_still_times_out_and_the_queue_stays_capped(self):
        rp = _RunPod([{"status": "IN_QUEUE"}] * 3 + [{"status": "IN_PROGRESS"}])
        with self.assertRaises(TimeoutError) as cm:
            _run(rp, lambda: _adapter(queue_max_s=60, max_wait=0.05).generate(_req()))
        self.assertIn("of execution", str(cm.exception))
        self.assertEqual(rp.cancels, ["rp1"])
        rp = _RunPod([{"status": "IN_QUEUE"}])
        with self.assertRaises(adapters.CloudBusy):
            _run(rp, lambda: _adapter(queue_max_s=0.05, max_wait=60).generate(_req()))

    def test_completed_found_by_the_grace_cancel_is_delivered(self):
        rp = _RunPod([{"status": "IN_PROGRESS"}, 503], cancel_reply={"status": "COMPLETED"})
        rp.after_cancel = DONE
        req = _req()
        out = _run(rp, lambda: _adapter(queue_max_s=60).generate(req))
        self.assertEqual([(b.name, b.data) for b in out.blobs], [("o_00001_.png", b"img")])
        self.assertTrue(req.cloud_trace["runpod_settled"])
        self.assertEqual(rp.cancels, ["rp1"])

    def test_completed_found_by_the_max_wait_cancel_is_delivered(self):
        rp = _RunPod([{"status": "IN_PROGRESS"}], cancel_reply={"status": "IN_PROGRESS"})
        rp.after_cancel = DONE                  # it finished between the last poll and /cancel
        out = _run(rp, lambda: _adapter(queue_max_s=60, max_wait=0.05).generate(_req()))
        self.assertEqual([b.name for b in out.blobs], ["o_00001_.png"])

    def test_unreadable_200_is_transient(self):
        for odd in (b"<html>bad gateway</html>", ["not", "an", "object"]):
            rp = _RunPod([{"status": "IN_PROGRESS"}, odd, DONE])
            req = _req()
            out = _run(rp, lambda: _adapter(queue_max_s=60).generate(req))
            self.assertEqual([b.name for b in out.blobs], ["o_00001_.png"], odd)
        rp = _RunPod([{"status": "IN_PROGRESS"}, b"<html>"])     # for good → grace → cancel
        req = _req()
        with self.assertRaises(ConnectionError):
            _run(rp, lambda: _adapter(queue_max_s=60).generate(req))
        self.assertEqual(rp.cancels, ["rp1"])
        self.assertTrue(req.cloud_trace["runpod_settled"])

    def test_three_4xx_go_through_cancel(self):
        rp = _RunPod([{"status": "IN_PROGRESS"}, 400])
        req = _req()
        with self.assertRaises(RuntimeError) as cm:
            _run(rp, lambda: _adapter(queue_max_s=60).generate(req))
        self.assertIn("three times", str(cm.exception))
        self.assertEqual(rp.cancels, ["rp1"])
        self.assertTrue(req.cloud_trace["runpod_settled"])
        rp = _RunPod([{"status": "IN_PROGRESS"}, 403], cancel_reply={"status": "IN_PROGRESS"})
        req = _req()
        with self.assertRaises(RuntimeError):
            _run(rp, lambda: _adapter(queue_max_s=60).generate(req))
        self.assertFalse(req.cloud_trace.get("runpod_settled"))



class EndpointKeyed(unittest.TestCase):
    """Item 8 (final review): probe records and job meta were keyed by backend NAME
    only. A url changed to another endpoint (or a reused name) served the old endpoint's
    probe — models and node types of a different image — and a restart sent the orphan's
    cancel to the CURRENT endpoint, which ends nothing while the old one bills on."""

    B = {"name": "rp", "type": "runpod", "url": URL, "api_key": "k", "poll_interval": 0.01}

    def _disc(self, ad):
        async def go():
            async with httpx.AsyncClient() as c:
                return await ad.discover(c)
        return _run(_RunPod([DONE]), go)

    def test_probe_record_carries_the_endpoint_and_a_foreign_one_is_ignored(self):
        saved = {}
        rp = _RunPod([INFO])
        ad = adapters.RunpodAdapter(dict(self.B), _ctx(runpod_probe_save=lambda n, r: saved.update({n: r})))
        real = httpx.AsyncClient
        ad.ctx.http_client = lambda: real(transport=httpx.MockTransport(rp.handler))
        asyncio.run(ad.probe())
        self.assertEqual(saved["rp"]["endpoint"], "ep123")
        mine = adapters.RunpodAdapter(dict(self.B), _ctx(runpod_probe_load=lambda n: saved.get(n)))
        self.assertIn("q.gguf", self._disc(mine).models)
        for rec in ({**saved["rp"], "endpoint": "other"}, {k: v for k, v in saved["rp"].items()
                                                            if k != "endpoint"}):
            ad2 = adapters.RunpodAdapter(dict(self.B), _ctx(runpod_probe_load=lambda n, r=rec: r))
            self.assertEqual(self._disc(ad2).models, set())
            self.assertEqual(ad2.probe_state["state"], "none")
            self.assertTrue(ad2._snap_loaded)

    def test_job_meta_names_the_endpoint(self):
        metas = []
        ad = _adapter()
        ad.ctx.note_job_meta = lambda jid, meta: metas.append(meta)
        _run(_RunPod([DONE]), lambda: ad.generate(_req()))
        self.assertEqual(metas[0]["runpod_endpoint"], "ep123")
        self.assertEqual(metas[0]["runpod_job_id"], "rp1")

    def test_orphan_cancel_goes_to_the_endpoint_the_job_ran_on(self):
        m = _main()
        sent = []

        class _Ad:
            async def cancel_runpod_id(self, rp_id, url=""):
                sent.append((rp_id, url))
                return True
        b = {"name": "rp", "type": "runpod", "url": "https://api.runpod.ai/v2/newep"}
        with unittest.mock.patch.object(m, "backends", [b]), \
                unittest.mock.patch.dict(m.backend_adapters, {m.backend_id(b): _Ad()}), \
                unittest.mock.patch.object(m.jobs, "merge_meta", lambda *a: None):
            asyncio.run(m._cancel_orphaned_runpod([
                ("j1", "rp", {"runpod_job_id": "a", "runpod_endpoint": "oldep"}),
                ("j2", "rp", {"runpod_job_id": "b", "runpod_endpoint": "newep"}),
                ("j3", "rp", {"runpod_job_id": "c"}),
                ("j4", "rp", {"runpod_job_id": "d", "runpod_endpoint": "x/../../evil"})]))
        self.assertEqual(sent, [("a", "https://api.runpod.ai/v2/oldep"), ("b", ""), ("c", ""),
                                ("d", "")])



class HealthAndLoras(unittest.TestCase):
    """Item 9 (final review): an execution fault quarantines a RunPod endpoint like any
    ComfyUI box — routing changes — yet /health and the Backends tab said nothing; and
    the LoRAs tab never listed what the probe discovered."""

    def test_health_carries_quarantine_and_fail_rate_but_no_watchdog(self):
        import time as _t
        m = _main()
        import admin
        b = {"name": "rp", "type": "runpod", "url": URL, "enabled": True, "paid": True}
        bid = m.backend_id(b)
        with unittest.mock.patch.object(m, "backends", [b]), \
                unittest.mock.patch.dict(m.gen_exec_faults,
                                         {f"img|{bid}": {"until": _t.time() + 600, "fails": 2,
                                                         "error": "node 4: OOM"}}), \
                unittest.mock.patch.dict(m.backend_loras, {bid: {"style.safetensors"}}):
            m._record_gen_attempt(bid, conn_fail=False, exec_fail=True)
            try:
                e = asyncio.run(m.health())["backends"][bid]
                loras = admin._backend_loras()
            finally:
                m.backend_gen_window.pop(bid, None)
        self.assertEqual([q["alias"] for q in e["quarantined"]], ["img"])
        self.assertIn("exec_fail_rate", e)
        self.assertNotIn("exec_stuck", e)
        self.assertNotIn("last_restart", e)
        self.assertEqual(loras.get("rp"), ["style.safetensors"])



class InputNamesAndWorkerErrors(unittest.TestCase):
    """Item 11b/c (final review): `upload_slot_name` keeps every `isalnum` character, so a
    param with a unicode letter built a name the worker's NAME_RE refuses — only AFTER
    /run was billed; and a COMPLETED job whose output was the worker's own `{"error": …}`
    surfaced as "the worker returned no outputs", hiding the diagnosis."""

    def test_a_name_the_worker_would_refuse_is_refused_before_run(self):
        rp = _RunPod([DONE])
        req = _req(node_mapping={"bild_ä": {"node": "1", "field": "image"}},
                   upload_images={"bild_ä": PNG})
        with self.assertRaises(RuntimeError) as cm:
            _run(rp, lambda: _adapter().generate(req))
        self.assertIn("parameter 'bild_ä'", str(cm.exception))
        self.assertEqual(rp.runs, [])
        self.assertNotIn("runpod_job_id", req.cloud_trace)     # nothing billed → failover ok

    def test_a_file_param_is_checked_too(self):
        rp = _RunPod([DONE])
        req = _req(node_mapping={"image": {"node": "1", "field": "image"},
                                 "mesh_ü": {"node": "1", "field": "mesh"}},
                   upload_files={"mesh_ü": ("m.glb", b"glTF")})
        with self.assertRaises(RuntimeError) as cm:
            _run(rp, lambda: _adapter().generate(req))
        self.assertIn("parameter 'mesh_ü'", str(cm.exception))
        self.assertEqual(rp.runs, [])

    def test_plain_names_still_pass(self):
        io = adapters.RunpodIO(_adapter())
        self.assertIsNone(io.name_refusal("gw_job1_image.png", "image"))
        self.assertIsNotNone(io.name_refusal("gw_job1_x..png", "x"))
        self.assertIsNone(adapters.ComfyIO.name_refusal(None, "gw_ä.png", "ä"))

    def test_completed_with_a_worker_error_keeps_its_text(self):
        bad = {"status": "COMPLETED", "executionTime": 10,
               "output": {"error": "node 4 (KSampler): CUDA out of memory"}}
        with self.assertRaises(RuntimeError) as cm:
            _run(_RunPod([bad]), lambda: _adapter().generate(_req()))
        self.assertIn("CUDA out of memory", str(cm.exception))
        self.assertIn("RunPod job rp1", str(cm.exception))



class KeyStaysAtRunPod(unittest.TestCase):
    """Item 12a (final review): only discovery checked the url. A config-defined runpod
    backend with a typo'd or foreign url sent its RunPod API key — and the job's inputs —
    to that host on a probe, a job or a cancel."""

    BAD = ("https://evil.example/v2/ep1", "http://api.runpod.ai/v2/ep1",
           "https://api.runpod.ai.evil.example/v2/ep1")

    def test_nothing_is_sent_to_a_foreign_url(self):
        for bad in self.BAD:
            rp = _RunPod([INFO])
            ad = _adapter(url=bad)
            real = httpx.AsyncClient
            ad.ctx.http_client = lambda: real(transport=httpx.MockTransport(rp.handler))
            st = asyncio.run(ad.probe())
            self.assertEqual(st["state"], "failed", bad)
            self.assertIn("api.runpod.ai only", st["error"])
            with self.assertRaises(RuntimeError):
                _run(rp, lambda: ad.generate(_req()))
            self.assertFalse(_run(rp, lambda: ad.cancel_runpod_id("rp1")))
            self.assertFalse(_run(rp, lambda: _adapter().cancel_runpod_id("rp1", url=bad)))
            self.assertEqual(rp.paths, [], bad)


if __name__ == "__main__":
    unittest.main()
