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
            st = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            if isinstance(st, int):
                return httpx.Response(st, json={})
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


if __name__ == "__main__":
    unittest.main()
