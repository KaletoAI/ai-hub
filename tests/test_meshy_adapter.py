"""Adapter I/O tests for MeshyAdapter against a local HTTP stub.
run: venv/bin/python -m unittest tests.test_meshy_adapter -v"""
import asyncio
import json
import threading
import time
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx

import adapters
import meshy

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
GLB = b"glTF" + b"\x00" * 60


class _Stub(BaseHTTPRequestHandler):
    """Scripted Meshy: POST → id; GET polls walk `script`; assets under /asset/<fmt>."""
    script: list = []          # task objects returned by successive GETs (last one repeats)
    poll_status = 200          # status a task poll answers with (non-200 → error body)
    post_status = 202          # the real Meshy answers a task create with 202 Accepted
    posted: list = []
    balance = 120
    seen_auth: list = []

    def log_message(self, *a):  # silence
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        _Stub.seen_auth.append((self.path, self.headers.get("Authorization")))
        if self.path == "/openapi/v1/balance":
            return self._json(200, {"balance": _Stub.balance})
        if self.path.startswith("/asset/"):
            self.send_response(200)
            self.send_header("Content-Length", str(len(GLB)))
            self.end_headers()
            return self.wfile.write(GLB)
        if self.path.startswith("/openapi/v1/"):
            if _Stub.poll_status != 200:
                return self._json(_Stub.poll_status, {"message": "task not found"})
            t = _Stub.script.pop(0) if len(_Stub.script) > 1 else _Stub.script[0]
            return self._json(200, t)
        self._json(404, {"message": "nope"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        _Stub.posted.append((self.path, json.loads(self.rfile.read(n) or b"{}")))
        if _Stub.post_status >= 400:
            return self._json(_Stub.post_status, {"message": "NoMoreConcurrentTasks"
                                                  if _Stub.post_status == 429 else "no credits"})
        if self.path.endswith("/rigging"):          # the real Meshy answers rigging with 200
            return self._json(200, {"result": "rig-1"})
        self._json(_Stub.post_status, {"result": "task-1"})


def _ctx():
    counts = {"inc": 0, "dec": 0}
    return adapters.AdapterContext(
        auth_headers=lambda b: {}, inflight_inc=lambda bid: counts.__setitem__("inc", counts["inc"] + 1),
        inflight_dec=lambda bid: counts.__setitem__("dec", counts["dec"] + 1),
        cost_usd=lambda *a: 0.0, source_of=lambda r: "test", record_call=lambda *a, **k: None,
        log_enabled=lambda: False), counts


def _task(status, **extra):
    return {"id": "task-1", "status": status, "progress": 50, **extra}


class TestMeshyAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _Stub)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        _Stub.script, _Stub.posted, _Stub.seen_auth = [], [], []
        _Stub.post_status, _Stub.poll_status, _Stub.balance = 202, 200, 120
        self.ctx, self.counts = _ctx()
        self.backend = {"name": "meshy", "type": "meshy", "url": self.url, "api_key": "msy_test",
                        "poll_interval": 0.01, "max_wait": 2}
        self.ad = adapters.MeshyAdapter(self.backend, self.ctx)

    def _req(self, images=None, values=None, endpoint="image-to-3d", formats=None,
             files=None, **opts):
        cand = meshy.default_candidate("meshy")
        cand["meshy"]["endpoint"] = endpoint
        if formats:
            cand["meshy"]["options"]["target_formats"] = list(formats)
        cand["meshy"]["options"].update(opts)
        return adapters.NormalizedRequest(alias="Meshy-Object", real_model="latest", task="img2mesh",
                                          params=dict(values or {}), upload_images=dict(images or {}),
                                          upload_files=dict(files or {}),
                                          meshy=cand["meshy"], upload_prefix="gw_j1")

    def _run(self, coro):
        return asyncio.run(coro)

    def _discover(self):
        async def go():
            async with httpx.AsyncClient() as client:
                return await self.ad.discover(client)
        return self._run(go())

    def test_discover_reports_credits(self):
        caps = self._discover()
        self.assertIn("latest", caps.models)
        self.assertEqual(self.ad.credits, 120)
        self.assertEqual(_Stub.seen_auth[0][1], "Bearer msy_test")

    def test_discover_zero_credits_is_down(self):
        _Stub.balance = 0
        with self.assertRaises(adapters.MeshyNoCredits):
            self._discover()

    def test_generate_success(self):
        _Stub.script = [_task("PENDING"), _task("IN_PROGRESS"),
                        _task("SUCCEEDED", progress=100, consumed_credits=30,
                              model_urls={"glb": f"{self.url}/asset/glb"},
                              thumbnail_url=f"{self.url}/asset/png")]
        out = self._run(self.ad.generate(self._req({"input_image": PNG}, {"input_name": "hero"})))
        self.assertEqual([b.name for b in out.blobs], ["model.glb", "preview.png"])
        self.assertEqual(out.blobs[0].mime, "model/gltf-binary")
        self.assertEqual(out.blobs[0].kind, "file")
        self.assertEqual(out.blobs[1].kind, "image")
        self.assertEqual(out.meta["meshy_task_id"], "task-1")
        self.assertEqual(out.meta["consumed_credits"], 30)
        self.assertEqual(out.meta["request"]["name"], "hero")
        self.assertNotIn("rig", out.meta)        # only the rigging endpoint delivers a rig
        self.assertNotIn("data:", json.dumps(out.meta))
        self.assertEqual(_Stub.posted[0][0], "/openapi/v1/image-to-3d")
        self.assertEqual((self.counts["inc"], self.counts["dec"]), (1, 1))
        # asset downloads carry NO bearer (signed URLs on another host)
        self.assertTrue(all(a is None for p, a in _Stub.seen_auth if p.startswith("/asset/")))

    def test_slot_held_does_not_double_count(self):
        _Stub.script = [_task("SUCCEEDED", model_urls={"glb": f"{self.url}/asset/glb"})]
        req = self._req({"input_image": PNG})
        req.slot_held = True
        self._run(self.ad.generate(req))
        self.assertEqual((self.counts["inc"], self.counts["dec"]), (0, 0))

    def test_failed_task_is_final_runtime_error(self):
        _Stub.script = [_task("FAILED", task_error={"message": "bad input"})]
        with self.assertRaises(RuntimeError) as cm:
            self._run(self.ad.generate(self._req({"input_image": PNG})))
        self.assertIn("bad input", str(cm.exception))
        self.assertNotIsInstance(cm.exception, ConnectionError)

    def test_failed_invalid_input_stays_final(self):
        # The permanent half of `task_error.type`: a rejected input is the same job on any
        # candidate, so it must NOT enter the failover class and re-submit elsewhere.
        _Stub.script = [_task("FAILED", task_error={"type": "invalid_input",
                                                    "code": "moderation_blocked",
                                                    "message": "blocked by safety filters"})]
        with self.assertRaises(RuntimeError) as cm:
            self._run(self.ad.generate(self._req({"input_image": PNG})))
        self.assertNotIsInstance(cm.exception, ConnectionError)

    def test_failed_server_error_is_failover_class(self):
        # The vendor failed the task on its own side (0 credits) and says "retry": that is
        # a fault of the SERVICE, so it must reach the same failover/self-retry path a
        # transport fault does — otherwise a transient Meshy hiccup ends the job outright
        # (measured 2026-09-03, job 9cf448115b4b).
        _Stub.script = [_task("FAILED", consumed_credits=0,
                              task_error={"type": "server_error",
                                          "message": "An unexpected error occurred."})]
        with self.assertRaises(adapters.CloudTaskRetryable) as cm:
            self._run(self.ad.generate(self._req({"input_image": PNG})))
        self.assertIsInstance(cm.exception, ConnectionError)     # _GEN_FAILOVER_ERRORS
        self.assertIn("task-1", str(cm.exception))
        self.assertIn("An unexpected error occurred.", str(cm.exception))
        self.assertEqual(cm.exception.vendor, "Meshy")

    def test_failed_server_error_that_consumed_credits_stays_final(self):
        # The paid-task rule: once the vendor billed the task, nothing may re-run it —
        # a retry would buy the same mesh twice. Same reasoning as tripo's convert path.
        _Stub.script = [_task("FAILED", consumed_credits=30,
                              task_error={"type": "server_error", "message": "boom"})]
        with self.assertRaises(RuntimeError) as cm:
            self._run(self.ad.generate(self._req({"input_image": PNG})))
        self.assertNotIsInstance(cm.exception, ConnectionError)

    def test_failed_run_leaves_the_task_facts_on_the_request(self):
        # A failed run returns no GenOutput, so the task id / endpoint / request summary
        # reach the job row ONLY through the request's cloud_trace. Without it a cloud
        # failure is diagnosable only from the vendor's dashboard (job 9cf448115b4b).
        _Stub.script = [_task("FAILED", task_error={"message": "bad input"})]
        req = self._req({"input_image": PNG})
        with self.assertRaises(RuntimeError):
            self._run(self.ad.generate(req))
        tr = req.cloud_trace
        self.assertEqual(tr["cloud"], "meshy")
        self.assertEqual(tr["backend"], "meshy")
        self.assertEqual(tr["cloud_task_id"], "task-1")
        self.assertEqual(tr["meshy_task_id"], "task-1")      # the name existing views read
        self.assertEqual(tr["endpoint"], "image-to-3d")
        self.assertEqual(tr["tasks"], [{"role": "image-to-3d", "task_id": "task-1"}])
        self.assertIn("request", tr)                          # what was actually sent
        self.assertNotIn("credits", tr)                       # never guessed on a failure

    def test_402_fails_over(self):
        _Stub.post_status = 402
        with self.assertRaises(adapters.MeshyNoCredits):
            self._run(self.ad.generate(self._req({"input_image": PNG})))

    def test_429_fails_over(self):
        _Stub.post_status = 429
        with self.assertRaises(adapters.MeshyBusy):
            self._run(self.ad.generate(self._req({"input_image": PNG})))

    def test_timeout_names_task(self):
        _Stub.script = [_task("IN_PROGRESS")]
        self.backend["max_wait"] = 0.05
        with self.assertRaises(TimeoutError) as cm:
            self._run(self.ad.generate(self._req({"input_image": PNG})))
        self.assertIn("task-1", str(cm.exception))

    def test_every_model_format_is_a_file_blob(self):
        urls = {f: f"{self.url}/asset/{f}" for f in ("glb", "usdz", "3mf")}
        _Stub.script = [_task("SUCCEEDED", model_urls=urls)]
        out = self._run(self.ad.generate(
            self._req({"input_image": PNG}, formats=["glb", "usdz", "3mf"])))
        self.assertEqual([b.name for b in out.blobs], ["model.glb", "model.usdz", "model.3mf"])
        self.assertEqual({b.kind for b in out.blobs}, {"file"})
        self.assertNotIn("image", [b.mime.split("/")[0] for b in out.blobs])

    def test_persistent_4xx_poll_is_final_and_fast(self):
        _Stub.script = [_task("IN_PROGRESS")]
        _Stub.poll_status = 404
        self.backend["max_wait"] = 5
        started = time.monotonic()
        with self.assertRaises(RuntimeError) as cm:
            self._run(self.ad.generate(self._req({"input_image": PNG})))
        self.assertLess(time.monotonic() - started, 1.0)      # not polled out to max_wait
        self.assertIn("404", str(cm.exception))
        self.assertIn("task-1", str(cm.exception))
        self.assertNotIsInstance(cm.exception, ConnectionError)

    def test_persistent_5xx_poll_fails_over_after_grace(self):
        _Stub.script = [_task("IN_PROGRESS")]
        _Stub.poll_status = 503
        self.backend["disconnect_grace"] = 0.05
        with self.assertRaises(ConnectionError) as cm:
            self._run(self.ad.generate(self._req({"input_image": PNG})))
        self.assertIn("task-1", str(cm.exception))

    def test_persistent_429_poll_is_service_side(self):
        # A poll-rate limit is not a verdict on the task (which is running and paid for):
        # it must reach the grace branch, never the final 4xx one.
        _Stub.script = [_task("IN_PROGRESS")]
        _Stub.poll_status = 429
        self.backend["disconnect_grace"] = 0.05
        with self.assertRaises(ConnectionError) as cm:
            self._run(self.ad.generate(self._req({"input_image": PNG})))
        self.assertIn("task-1", str(cm.exception))

    def test_missing_image_is_input_error_before_post(self):
        with self.assertRaises(meshy.MeshyInput):
            self._run(self.ad.generate(self._req({})))
        self.assertEqual(_Stub.posted, [])

    def test_rigging_flow(self):
        _Stub.script = [{"status": "SUCCEEDED", "progress": 100, "consumed_credits": 5,
                         "result": {"rigged_character_glb_url": f"{self.url}/asset/glb"}}]
        out = self._run(self.ad.generate(self._req(
            endpoint="rigging", files={"input_mesh_path": ("h.glb", GLB)},
            values={"input_height_m": 1.8})))
        self.assertEqual([b.name for b in out.blobs], ["rigged.glb"])
        self.assertEqual(out.blobs[0].mime, "model/gltf-binary")
        self.assertEqual(out.blobs[0].kind, "file")
        self.assertEqual(_Stub.posted[0][0], "/openapi/v1/rigging")
        self.assertEqual(_Stub.posted[0][1]["height_meters"], 1.8)
        self.assertEqual(out.meta["endpoint"], "rigging")
        self.assertEqual(out.meta["meshy_task_id"], "rig-1")
        self.assertEqual(out.meta["consumed_credits"], 5)
        self.assertEqual(out.meta["rig"], "meshy")   # job view reads meta["rig"]
        self.assertTrue(out.meta["request"]["model_url"].startswith("<"))
        self.assertNotIn("data:", json.dumps(out.meta))

    def test_rigging_animations_option(self):
        _Stub.script = [{"status": "SUCCEEDED", "progress": 100, "consumed_credits": 5,
                         "result": {"rigged_character_glb_url": f"{self.url}/asset/glb",
                                    "basic_animations": {"walking_glb_url": f"{self.url}/asset/glb",
                                                         "running_glb_url": f"{self.url}/asset/glb"}}}]
        out = self._run(self.ad.generate(self._req(
            endpoint="rigging", files={"input_mesh_path": ("h.glb", GLB)}, animations=True)))
        self.assertEqual([b.name for b in out.blobs], ["rigged.glb", "walking.glb", "running.glb"])

    def test_create_post_timeout_scales_with_body(self):
        """The task-create POST carries the whole input (a rigging mesh as base64), so it
        gets its OWN size-scaled timeout while the CLIENT default stays short for polls.

        Measured 2026-09-02 on prod: a 70 MB GLB → ~93 MB body → the flat 30 s client
        timeout fired mid-upload and the chain died on an httpx WriteTimeout."""
        _Stub.script = [{"status": "SUCCEEDED", "progress": 100, "consumed_credits": 5,
                         "result": {"rigged_character_glb_url": f"{self.url}/asset/glb"}}]
        big = b"glTF" + b"\x00" * (5 * 1024 * 1024)          # ~5 MiB mesh → ~6.7 MiB body
        seen: dict = {}
        orig = httpx.AsyncClient.post

        async def spy(client, url, *a, **kw):
            if str(url).endswith("/rigging"):
                seen["timeout"] = kw.get("timeout")
                seen["client_default"] = client.timeout
            return await orig(client, url, *a, **kw)

        with unittest.mock.patch.object(httpx.AsyncClient, "post", spy):
            self._run(self.ad.generate(self._req(
                endpoint="rigging", files={"input_mesh_path": ("h.glb", big)})))
        t = seen.get("timeout")
        self.assertIsInstance(t, httpx.Timeout)              # a per-request override, not None
        self.assertGreaterEqual(t.read, 20.0)
        self.assertGreaterEqual(t.write, 20.0)
        self.assertGreater(t.write, adapters._MESHY_HTTP_TIMEOUT)   # genuinely more than the default
        # …and the polls still run on the short client default: disconnect_grace only
        # reacts as fast as a poll gives up.
        self.assertEqual(seen["client_default"].read, adapters._MESHY_HTTP_TIMEOUT)
        self.assertEqual(seen["client_default"].write, adapters._MESHY_HTTP_TIMEOUT)

    def test_rigging_missing_mesh_is_input_error(self):
        with self.assertRaises(meshy.MeshyInput):
            self._run(self.ad.generate(self._req(endpoint="rigging")))
        self.assertEqual(_Stub.posted, [])


class TestGenTypesAndFields(unittest.TestCase):
    def test_registry(self):
        self.assertIs(adapters.ADAPTERS["meshy"], adapters.MeshyAdapter)
        self.assertEqual(adapters.GEN_TYPES, frozenset({"comfyui", "meshy", "tripo"}))
        self.assertFalse(adapters.OpenAIAdapter.serves_generation)

    def test_public_fields_meshy(self):
        params, images, files = adapters.public_fields(meshy.default_candidate("m"))
        self.assertEqual([i["name"] for i in images], ["input_image"])
        self.assertEqual(files, [])
        self.assertTrue(any(p["name"] == "input_face_num" for p in params))

    def test_public_fields_meshy_rigging(self):
        cand = meshy.default_candidate("m")
        cand["meshy"]["endpoint"] = "rigging"
        params, images, files = adapters.public_fields(cand)
        self.assertEqual(images, [])
        self.assertEqual(files, [{"name": "input_mesh_path", "required": True, "accept": ["glb"]}])
        self.assertEqual([p["name"] for p in params],
                         ["input_name", "input_height_m", "input_no_fingers"])

    def test_public_fields_comfy(self):
        wf = {"1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
              "2": {"class_type": "KSampler", "inputs": {"steps": 20, "seed": 1}}}
        mapping = {"image": {"node": "1", "field": "image", "label": "input_image", "on_empty": "required"},
                   "steps": {"node": "2", "field": "steps"},
                   "seed": {"node": "2", "field": "seed"}}
        params, images, files = adapters.public_fields({"workflow_json": wf, "mapping": mapping})
        self.assertEqual(images, [{"name": "input_image", "on_empty": "required", "required": True}])
        self.assertEqual(files, [])
        self.assertEqual(params[0], {"name": "steps", "type": "int", "default": 20})
        self.assertEqual(params[1]["auto"], "random unless sent")

    def test_public_fields_comfy_files(self):
        """A ComfyUI mesh param is advertised under `files` (by its LABEL) — what a client
        uploads it as — and never as an image slot. It ALSO stays a scalar param: there it
        is the backend-side path, the second, upload-free way to name the same input."""
        wf = {"1": {"class_type": "LoadMesh", "inputs": {"mesh": "a.glb"}}}
        mapping = {"mesh_path": {"node": "1", "field": "mesh", "label": "input_mesh_path"}}
        params, images, files = adapters.public_fields({"workflow_json": wf, "mapping": mapping})
        self.assertEqual(files, [{"name": "input_mesh_path", "required": False}])
        self.assertEqual(images, [])


if __name__ == "__main__":
    unittest.main()
