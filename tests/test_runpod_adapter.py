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


if __name__ == "__main__":
    unittest.main()
