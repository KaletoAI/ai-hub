"""Adapter I/O tests for TripoAdapter against a local HTTP stub.
run: venv/bin/python -m unittest tests.test_tripo_adapter -v"""
import asyncio
import json
import re
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import ANY

import httpx

import adapters
import tripo

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPG = b"\xff\xd8\xff" + b"\x00" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 16
GLB = b"glTF" + b"\x00" * 60


class _Stub(BaseHTTPRequestHandler):
    """Scripted Tripo V3: POST /v3/files → file token; POST /v3/generation|animations|models
    → task id; GET /v3/tasks/<id> walks `script[<id>]` (or `script["*"]`, last entry
    repeats); assets under /asset/."""
    script: dict = {}            # task id (or "*") → list of `data` objects for successive polls
    posted: list = []            # (path, json body) of every task create
    uploads: list = []           # (filename, nbytes, content type) seen at /v3/files
    seen_auth: list = []         # (path, Authorization header)
    balance = 500.0
    create_status = 200          # HTTP status of a task create
    create_code = 0              # envelope code of a task create
    fail_path = ""               # only creates on THIS path get create_status/create_code
    task_code = 0                # envelope code of a task POLL (non-0 = a verdict, HTTP 200)
    seq = 0

    def log_message(self, *a):   # silence
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
        if self.path == "/v3/account/balance":
            return self._json(200, {"code": 0, "data": {"balance": _Stub.balance, "frozen": 0}})
        if self.path.startswith("/asset/"):
            payload = GLB if self.path.endswith((".glb", ".fbx", ".obj")) else PNG
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            return self.wfile.write(payload)
        if self.path.startswith("/v3/tasks/"):
            if _Stub.task_code:
                return self._json(200, {"code": _Stub.task_code, "message": "gone"})
            tid = self.path.rsplit("/", 1)[1]
            seq = _Stub.script.get(tid) or _Stub.script.get("*")
            if not seq:
                return self._json(404, {"code": 2001, "message": "task not found"})
            t = seq.pop(0) if len(seq) > 1 else seq[0]
            return self._json(200, {"code": 0, "data": {"task_id": tid, **t}})
        self._json(404, {"code": 1, "message": "nope"})

    def do_POST(self):
        _Stub.seen_auth.append((self.path, self.headers.get("Authorization")))
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        if self.path == "/v3/files":
            ctype = self.headers.get("Content-Type", "")
            m = re.search(rb'filename="([^"]+)"', raw)
            _Stub.uploads.append((m.group(1).decode() if m else "", len(raw),
                                  ctype.split(";")[0]))
            _Stub.seq += 1
            return self._json(200, {"code": 0, "data": {"file_token": f"tok{_Stub.seq}"}})
        _Stub.posted.append((self.path, json.loads(raw or b"{}")))
        if ((_Stub.create_status != 200 or _Stub.create_code != 0)
                and self.path.endswith(_Stub.fail_path)):
            return self._json(_Stub.create_status,
                              {"code": _Stub.create_code, "message": "refused",
                               "suggestion": "top up"})
        _Stub.seq += 1
        return self._json(200, {"code": 0, "data": {"task_id": f"task_{_Stub.seq}"}})


def _ctx():
    counts = {"inc": 0, "dec": 0}
    return adapters.AdapterContext(
        auth_headers=lambda b: {}, inflight_inc=lambda bid: counts.__setitem__("inc", counts["inc"] + 1),
        inflight_dec=lambda bid: counts.__setitem__("dec", counts["dec"] + 1),
        cost_usd=lambda *a: 0.0, source_of=lambda r: "test", record_call=lambda *a, **k: None,
        log_enabled=lambda: False), counts


class TestTripoAdapter(unittest.TestCase):
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
        _Stub.script, _Stub.posted, _Stub.uploads, _Stub.seen_auth = {}, [], [], []
        _Stub.balance, _Stub.create_status, _Stub.create_code, _Stub.task_code = 500.0, 200, 0, 0
        _Stub.fail_path = ""                     # "" = every create path (endswith "")
        _Stub.seq = 0
        self.ctx, self.counts = _ctx()
        self.backend = self._backend(self.url)
        self.ad = adapters.TripoAdapter(self.backend, self.ctx)

    def _backend(self, url, **kw):
        b = {"name": "tripo", "type": "tripo", "url": url, "api_key": "tripo_test",
             "poll_interval": 0.01, "max_wait": 3}
        b.update(kw)
        return b

    def _req(self, endpoint="image-to-model", images=None, files=None, values=None,
             model=None, **opts):
        cand = tripo.default_candidate("tripo")
        cand["tripo"]["endpoint"] = endpoint
        cand["tripo"]["options"].update(opts)
        return adapters.NormalizedRequest(
            alias="Tripo-Object", real_model=model or tripo.AI_MODELS[0], task="img2mesh",
            params=dict(values or {}), upload_images=dict(images or {}),
            upload_files=dict(files or {}), cloud=cand["tripo"], upload_prefix="gw_j1")

    def _run(self, coro):
        return asyncio.run(coro)

    def _discover(self):
        async def go():
            async with httpx.AsyncClient() as client:
                return await self.ad.discover(client)
        return self._run(go())

    # ── discovery ────────────────────────────────────────────────────────────
    def test_discover_reads_balance_and_zero_is_no_credits(self):
        caps = self._discover()
        self.assertEqual(caps.models, set(tripo.AI_MODELS))
        self.assertEqual(self.ad.credits, 500)
        self.assertEqual(_Stub.seen_auth[0][1], "Bearer tripo_test")
        _Stub.balance = 0
        with self.assertRaises(adapters.CloudNoCredits) as cm:
            self._discover()
        self.assertEqual(cm.exception.vendor, "Tripo")

    # ── the generation flow ──────────────────────────────────────────────────
    def test_image_to_model_uploads_then_creates_then_downloads(self):
        _Stub.script = {"*": [{"status": "running", "progress": 10},
                              {"status": "success", "progress": 100, "credits_consumed": 30,
                               "output": {"model_url": f"{self.url}/asset/m.glb",
                                          "rendered_image_url": f"{self.url}/asset/p.png"}}]}
        out = self._run(self.ad.generate(self._req(images={"input_image": PNG})))
        self.assertEqual(_Stub.uploads, [("input_image.png", ANY, "multipart/form-data")])
        self.assertEqual(_Stub.posted[0][0], "/v3/generation/image-to-model")
        self.assertEqual(_Stub.posted[0][1]["input"], "tok1")
        self.assertEqual([b.name for b in out.blobs], ["model.glb", "preview.png"])
        self.assertEqual(out.blobs[0].kind, "file")
        self.assertEqual(out.blobs[1].kind, "image")
        self.assertEqual(out.meta["cloud"], "tripo")
        self.assertTrue(out.meta["cloud_task_id"])
        self.assertNotIn("meshy_task_id", out.meta)          # the neutral key only
        self.assertEqual(out.meta["consumed_credits"], 30)
        self.assertEqual(out.meta["ai_model"], tripo.AI_MODELS[0])
        self.assertNotIn("rig", out.meta)
        self.assertEqual((self.counts["inc"], self.counts["dec"]), (1, 1))
        auth = dict(_Stub.seen_auth)                          # last header seen per path
        self.assertEqual(auth["/v3/files"], "Bearer tripo_test")
        self.assertEqual(auth["/v3/generation/image-to-model"], "Bearer tripo_test")
        # signed asset urls live on a CDN host — the bearer must not travel there
        self.assertTrue(all(a is None for p, a in _Stub.seen_auth if p.startswith("/asset/")))

    def test_multiview_needs_two_views_before_any_upload(self):
        """A request the module would refuse must not leave files behind at Tripo: the
        adapter checks the occupied slots BEFORE the first upload."""
        with self.assertRaises(tripo.TripoInput):
            self._run(self.ad.generate(self._req(endpoint="multiview-to-model",
                                                 images={"input_image_front": PNG})))
        self.assertEqual(_Stub.uploads, [])
        self.assertEqual(_Stub.posted, [])
        # …and the same for an image format Tripo does not take
        with self.assertRaises(tripo.TripoInput):
            self._run(self.ad.generate(self._req(images={"input_image": WEBP})))
        self.assertEqual(_Stub.uploads, [])
        self.assertEqual(_Stub.posted, [])

    def test_multiview_uploads_every_view(self):
        _Stub.script = {"*": [{"status": "success", "credits_consumed": 30,
                               "output": {"model_url": f"{self.url}/asset/m.glb"}}]}
        out = self._run(self.ad.generate(self._req(
            endpoint="multiview-to-model",
            images={"input_image_front": PNG, "input_image_left": JPG})))
        self.assertEqual([u[0] for u in _Stub.uploads],
                         ["input_image_front.png", "input_image_left.jpg"])
        self.assertEqual(_Stub.posted[0][0], "/v3/generation/multiview-to-model")
        self.assertEqual(_Stub.posted[0][1]["inputs"], [{"front": "tok1"}, {"left": "tok2"}])
        self.assertEqual([b.name for b in out.blobs], ["model.glb"])

    # ── create verdicts ──────────────────────────────────────────────────────
    def test_create_403_2010_is_no_credits_and_429_is_busy(self):
        _Stub.create_status, _Stub.create_code = 403, 2010
        with self.assertRaises(adapters.CloudNoCredits) as cm:
            self._run(self.ad.generate(self._req(images={"input_image": PNG})))
        self.assertEqual(cm.exception.vendor, "Tripo")

        _Stub.create_status, _Stub.create_code = 429, 2000
        with self.assertRaises(adapters.CloudBusy):
            self._run(self.ad.generate(self._req(images={"input_image": PNG})))

        _Stub.create_status, _Stub.create_code = 500, 1000
        with self.assertRaises(ConnectionError) as cm:
            self._run(self.ad.generate(self._req(images={"input_image": PNG})))
        self.assertNotIsInstance(cm.exception, adapters.CloudNoCredits)

        # HTTP 200 but a non-zero envelope code: Tripo's way of refusing the request
        _Stub.create_status, _Stub.create_code = 200, 2002
        with self.assertRaises(RuntimeError) as cm:
            self._run(self.ad.generate(self._req(images={"input_image": PNG})))
        self.assertNotIsInstance(cm.exception, ConnectionError)
        self.assertIn("refused", str(cm.exception))
        self.assertIn("top up", str(cm.exception))           # `suggestion` carried through

    # ── poll verdicts ────────────────────────────────────────────────────────
    def test_failed_task_is_final(self):
        _Stub.script = {"*": [{"status": "failed", "error_code": 2018,
                               "error_message": "too complex"}]}
        with self.assertRaises(RuntimeError) as cm:
            self._run(self.ad.generate(self._req(images={"input_image": PNG})))
        self.assertIn("too complex", str(cm.exception))
        self.assertNotIsInstance(cm.exception, ConnectionError)
        self.assertEqual(self.counts["dec"], 1)

    def test_poll_code_nonzero_three_times_is_final(self):
        """A 200 whose envelope says `code != 0` is a verdict about the task, not a
        service outage — three in a row end the job instead of polling out max_wait."""
        _Stub.task_code = 2001
        with self.assertRaises(RuntimeError) as cm:
            self._run(self.ad.generate(self._req(images={"input_image": PNG})))
        self.assertNotIsInstance(cm.exception, TimeoutError)
        self.assertIn("gone", str(cm.exception))

    def test_max_wait_timeout(self):
        _Stub.script = {"*": [{"status": "running", "progress": 5}]}
        self.backend["max_wait"] = 0.05
        with self.assertRaises(TimeoutError) as cm:
            self._run(self.ad.generate(self._req(images={"input_image": PNG})))
        self.assertIn("max_wait", str(cm.exception))

    # ── rigging ──────────────────────────────────────────────────────────────
    def test_rig_check_refuses_before_rig(self):
        _Stub.script = {"*": [{"status": "success",
                               "output": {"riggable": False, "rig_type": "quadruped"}}]}
        with self.assertRaises(RuntimeError) as cm:
            self._run(self.ad.generate(self._req(endpoint="rig",
                                                 files={"input_mesh_path": ("m.glb", GLB)})))
        self.assertIn("not riggable", str(cm.exception))
        self.assertIn("quadruped", str(cm.exception))
        self.assertEqual([p for p, _ in _Stub.posted], ["/v3/animations/rig-check"])

    def test_rig_with_clips_and_extra_format(self):
        _Stub.script = {"*": [{"status": "success", "credits_consumed": 25,
                               "output": {"model_url": f"{self.url}/asset/r.glb",
                                          "riggable": True, "rig_type": "biped"}}]}
        out = self._run(self.ad.generate(self._req(
            endpoint="rig", files={"input_mesh_path": ("m.glb", GLB)},
            target_formats=["glb", "fbx"], animations=["preset:walk"], spec="mixamo")))
        self.assertEqual(_Stub.uploads, [("input_mesh_path.glb", ANY, "multipart/form-data")])
        self.assertEqual([p for p, _ in _Stub.posted],
                         ["/v3/animations/rig-check", "/v3/animations/rig",
                          "/v3/models/convert", "/v3/animations/retarget"])
        self.assertEqual(_Stub.posted[1][1], {"input": "tok1", "model": "v1.0-20240301",
                                              "rig_type": "biped", "spec": "mixamo",
                                              "out_format": "glb"})
        self.assertEqual(_Stub.posted[2][1]["format"], "FBX")
        self.assertIs(_Stub.posted[2][1]["with_animation"], True)
        self.assertEqual(_Stub.posted[3][1]["animation"], "preset:walk")
        self.assertEqual(_Stub.posted[3][1]["out_format"], "glb")
        # both follow-ups hang off the RIG task, not the rig-check
        self.assertEqual(_Stub.posted[3][1]["input"], _Stub.posted[2][1]["input"])
        self.assertEqual(_Stub.posted[2][1]["input"], out.meta["cloud_task_id"])
        self.assertEqual([b.name for b in out.blobs], ["rigged.glb", "rigged.fbx", "walk.glb"])
        self.assertEqual(out.meta["rig"], "tripo")
        self.assertEqual(out.meta["rig_spec"], "mixamo")
        self.assertEqual(out.meta["rig_type"], "biped")
        self.assertEqual(out.meta["consumed_credits"], 100)   # every task of the job counts
        self.assertEqual([t["role"] for t in out.meta["tasks"]],
                         ["rig-check", "rig", "convert:fbx", "clip:preset:walk"])

    def test_failed_clip_is_skipped_not_the_job(self):
        """A clip is a courtesy: the rigged mesh is what the job is about, it is already
        finished and paid for, so a failed retarget is a warning, not a lost delivery."""
        _Stub.script = {"task_2": [{"status": "success", "credits_consumed": 0,
                                    "output": {"riggable": True, "rig_type": "biped"}}],
                        "task_3": [{"status": "success", "credits_consumed": 25,
                                    "output": {"model_url": f"{self.url}/asset/r.glb"}}],
                        "task_4": [{"status": "failed", "error_message": "no such preset"}]}
        with self.assertLogs(adapters.logger, "WARNING") as log:   # warned, not swallowed
            out = self._run(self.ad.generate(self._req(
                endpoint="rig", files={"input_mesh_path": ("m.glb", GLB)},
                animations=["preset:nope"])))
        self.assertIn("preset:nope", log.output[0])
        self.assertEqual([b.name for b in out.blobs], ["rigged.glb"])
        self.assertEqual([t["role"] for t in out.meta["tasks"]], ["rig-check", "rig"])
        self.assertEqual(out.meta["consumed_credits"], 25)

    def test_bad_rig_type_is_refused_before_the_upload(self):
        """A rig type the alias's rig model cannot do is a final input error — and it is
        caught BEFORE the mesh (up to 150 MB) is pushed into the Tripo account, where
        nobody cleans it up. The rule itself lives in tripo._rig_types_for."""
        with self.assertRaises(tripo.TripoInput) as cm:
            self._run(self.ad.generate(self._req(
                endpoint="rig", files={"input_mesh_path": ("m.glb", GLB)},
                values={"input_rig_type": "quadruped"})))
        self.assertIn("biped", str(cm.exception))
        self.assertEqual(_Stub.uploads, [])
        self.assertEqual(_Stub.posted, [])
        # …and the rig model that CAN do it lets the same request through
        _Stub.script = {"*": [{"status": "success", "credits_consumed": 25,
                               "output": {"model_url": f"{self.url}/asset/r.glb",
                                          "riggable": True, "rig_type": "quadruped"}}]}
        out = self._run(self.ad.generate(self._req(
            endpoint="rig", files={"input_mesh_path": ("m.glb", GLB)},
            rig_model="v2.5-20260210", rig_type="quadruped",
            values={"input_rig_type": "quadruped"})))
        self.assertEqual(_Stub.posted[1][1]["rig_type"], "quadruped")
        self.assertEqual(out.meta["rig_type"], "quadruped")

    def test_rig_check_type_mismatch_is_warned(self):
        """Tripo rigs what the REQUEST asked for; the check only reports what it saw. A
        mismatch means a quadruped gets a biped skeleton — silently, unless it is named."""
        _Stub.script = {"*": [{"status": "success", "credits_consumed": 25,
                               "output": {"model_url": f"{self.url}/asset/r.glb",
                                          "riggable": True, "rig_type": "quadruped"}}]}
        with self.assertLogs(adapters.logger, "WARNING") as log:
            self._run(self.ad.generate(self._req(
                endpoint="rig", files={"input_mesh_path": ("m.glb", GLB)})))
        self.assertIn("quadruped", log.output[0])
        self.assertIn("biped", log.output[0])
        self.assertIn("Tripo-Object", log.output[0])          # the alias, so it is findable

    def test_convert_create_429_is_final_never_a_second_paid_run(self):
        """After the primary task is billed, EVERY failure of a follow-up is final. A 429
        on the convert create (Tripo's model-processing pool is its own queue) would
        otherwise be a CloudBusy — a main._GEN_FAILOVER_ERRORS member — and _run_job would
        re-run the whole 30-credit image-to-model task on the next Tripo candidate."""
        _Stub.script = {"*": [{"status": "success", "credits_consumed": 30,
                               "output": {"model_url": f"{self.url}/asset/m.glb"}}]}
        _Stub.fail_path = "/models/convert"
        _Stub.create_status, _Stub.create_code = 429, 2000
        with self.assertRaises(RuntimeError) as cm:
            self._run(self.ad.generate(self._req(images={"input_image": PNG},
                                                 target_formats=["glb", "obj"])))
        self.assertNotIsInstance(cm.exception, ConnectionError)   # …so it never fails over
        self.assertNotIsInstance(cm.exception, TimeoutError)
        self.assertIn("obj", str(cm.exception))                   # names the format…
        task_id = _Stub.posted[1][1]["input"]                     # …and the PAID task, so the
        self.assertIn(task_id, str(cm.exception))                 # mesh is fetchable by hand
        self.assertEqual([p for p, _ in _Stub.posted],
                         ["/v3/generation/image-to-model", "/v3/models/convert"])

    def _create_raising(self, exc, role_prefix):
        """Make the REAL _create raise `exc` for one role only (the create leg's bare
        client.post is what a transport fault hits), leaving every other create intact."""
        real = self.ad._create

        async def flaky(client, url, body, endpoint, *a, **kw):
            if endpoint.startswith(role_prefix):
                raise exc
            return await real(client, url, body, endpoint, *a, **kw)
        self.ad._create = flaky

    def test_side_tasks_do_not_claim_the_primary_slot_in_the_trace(self):
        """One Tripo job is several tasks. The trace a failed job leaves must still name
        the PRIMARY one under cloud_task_id/endpoint — the same meaning those keys carry in
        the success meta — while converts and clips only join the task list. Getting this
        backwards would point the operator at a convert task for a job that failed there."""
        self._create_raising(httpx.ConnectError("boom"), "convert")
        req = self._req(images={"input_image": PNG}, target_formats=["glb", "obj"])
        with self.assertRaises(RuntimeError):
            self._run(self.ad.generate(req))
        tr = req.cloud_trace
        self.assertEqual(tr["cloud"], "tripo")
        self.assertEqual(tr["endpoint"], "image-to-model")     # not "convert:obj"
        self.assertEqual([t["role"] for t in tr["tasks"]], ["image-to-model"])
        self.assertEqual(tr["cloud_task_id"], tr["tasks"][0]["task_id"])

    def test_convert_create_transport_fault_is_final(self):
        """httpx's transport errors are subclasses of NEITHER ConnectionError nor
        TimeoutError, but main._GEN_FAILOVER_ERRORS lists them literally — so a dropped
        connection on the convert CREATE would fail over and re-bill the primary task."""
        _Stub.script = {"*": [{"status": "success", "credits_consumed": 30,
                               "output": {"model_url": f"{self.url}/asset/m.glb"}}]}
        self._create_raising(httpx.ConnectError("boom"), "convert")
        with self.assertRaises(RuntimeError) as cm:
            self._run(self.ad.generate(self._req(images={"input_image": PNG},
                                                 target_formats=["glb", "obj"])))
        self.assertNotIsInstance(cm.exception, httpx.HTTPError)
        self.assertNotIsInstance(cm.exception, ConnectionError)
        self.assertIn("obj", str(cm.exception))
        self.assertIn("boom", str(cm.exception))
        self.assertEqual([p for p, _ in _Stub.posted], ["/v3/generation/image-to-model"])

    def test_clip_create_transport_fault_is_skipped(self):
        _Stub.script = {"*": [{"status": "success", "credits_consumed": 25,
                               "output": {"model_url": f"{self.url}/asset/r.glb",
                                          "riggable": True, "rig_type": "biped"}}]}
        self._create_raising(httpx.ConnectError("boom"), "clip")
        with self.assertLogs(adapters.logger, "WARNING") as log:
            out = self._run(self.ad.generate(self._req(
                endpoint="rig", files={"input_mesh_path": ("m.glb", GLB)},
                animations=["preset:walk"])))
        self.assertIn("preset:walk", log.output[0])
        self.assertEqual([b.name for b in out.blobs], ["rigged.glb"])

    def test_artifact_download_fault_after_the_paid_task_is_final(self):
        """The task is done and billed before the first byte is fetched: a blip on the
        asset host must not fail over and buy the same mesh twice."""
        _Stub.script = {"*": [{"status": "success", "credits_consumed": 30,
                               "output": {"model_url": f"{self.url}/asset/m.glb"}}]}

        async def reset(client, url):
            raise httpx.ReadError("connection reset")
        self.ad._download = reset
        with self.assertRaises(RuntimeError) as cm:
            self._run(self.ad.generate(self._req(images={"input_image": PNG})))
        self.assertNotIsInstance(cm.exception, httpx.HTTPError)
        self.assertNotIsInstance(cm.exception, ConnectionError)
        self.assertIn("model.glb", str(cm.exception))
        # the PAID task id (upload = seq 1, the primary create = seq 2), so the finished
        # mesh can be fetched from Tripo's dashboard by hand
        self.assertIn("task_2", str(cm.exception))
        self.assertEqual(self.counts["dec"], 1)                          # slot still freed

    def test_clip_create_429_is_skipped_not_the_job(self):
        """The same money rule seen from the other side: a busy retarget queue costs a
        courtesy clip, never the rigged mesh that is already finished and paid for."""
        _Stub.script = {"*": [{"status": "success", "credits_consumed": 25,
                               "output": {"model_url": f"{self.url}/asset/r.glb",
                                          "riggable": True, "rig_type": "biped"}}]}
        _Stub.fail_path = "/animations/retarget"
        _Stub.create_status, _Stub.create_code = 429, 2000
        with self.assertLogs(adapters.logger, "WARNING") as log:
            out = self._run(self.ad.generate(self._req(
                endpoint="rig", files={"input_mesh_path": ("m.glb", GLB)},
                animations=["preset:walk"])))
        self.assertIn("preset:walk", log.output[0])
        self.assertEqual([b.name for b in out.blobs], ["rigged.glb"])
        self.assertEqual([t["role"] for t in out.meta["tasks"]], ["rig-check", "rig"])

    def test_rig_missing_mesh_is_input_error(self):
        with self.assertRaises(tripo.TripoInput):
            self._run(self.ad.generate(self._req(endpoint="rig")))
        self.assertEqual(_Stub.uploads, [])
        self.assertEqual(_Stub.posted, [])

    def test_generation_extra_format_convert_failure_fails_job(self):
        """A requested delivery must never silently shrink: a failed convert fails the
        job, naming the format that could not be produced."""
        _Stub.script = {"task_2": [{"status": "success", "credits_consumed": 30,
                                    "output": {"model_url": f"{self.url}/asset/m.glb"}}],
                        "task_3": [{"status": "failed", "error_message": "convert broke"}]}
        with self.assertRaises(RuntimeError) as cm:
            self._run(self.ad.generate(self._req(images={"input_image": PNG},
                                                 target_formats=["glb", "obj"])))
        self.assertIn("convert", str(cm.exception))
        self.assertIn("obj", str(cm.exception))
        self.assertEqual([p for p, _ in _Stub.posted],
                         ["/v3/generation/image-to-model", "/v3/models/convert"])

    def test_native_format_leads_the_delivery(self):
        """A generation task delivers GLB whatever the alias asks for, so the GLB is
        named `model.glb` and every other requested format is a convert — an alias that
        ticked only obj must not receive a GLB called `model.obj`."""
        _Stub.script = {"*": [{"status": "success", "credits_consumed": 30,
                               "output": {"model_url": f"{self.url}/asset/m.glb"}}]}
        out = self._run(self.ad.generate(self._req(images={"input_image": PNG},
                                                   target_formats=["obj"])))
        self.assertEqual([b.name for b in out.blobs], ["model.glb", "model.obj"])
        self.assertEqual(_Stub.posted[1][1]["format"], "OBJ")
        self.assertIs(_Stub.posted[1][1]["with_animation"], False)

    # ── chain roles ──────────────────────────────────────────────────────────
    def test_chain_hooks(self):
        cand = tripo.default_candidate("tripo")
        rig = tripo.default_candidate("tripo")
        rig["tripo"]["endpoint"] = "rig"
        self.assertIn("cannot be stage 1", self.ad.chain_export(rig, {}, {}, "gwchain_x").error)
        noglb = tripo.default_candidate("tripo")
        noglb["tripo"]["options"]["target_formats"] = ["fbx"]
        self.assertIn("glb", self.ad.chain_export(noglb, {}, {}, "gwchain_x").error)
        exp = self.ad.chain_export(cand, {}, {}, "gwchain_x")
        self.assertEqual((exp.mesh_name, exp.error), ("gwchain_x.glb", None))

        out = adapters.GenOutput(blobs=[adapters.GenBlob(data=GLB, mime="model/gltf-binary",
                                                         kind="file", name="model.glb")], meta={})
        self.assertEqual(self._run(self.ad.chain_take_mesh(out, exp, True)), GLB)

        req2 = adapters.NormalizedRequest(alias="Tripo-Rig")
        with self.assertRaises(RuntimeError):
            self._run(self.ad.chain_feed_mesh(req2, {}, "input_mesh_path", "gwchain_x.glb",
                                              None, ""))
        ref = self._run(self.ad.chain_feed_mesh(req2, {}, "input_mesh_path", "gwchain_x.glb",
                                                GLB, ""))
        self.assertEqual(req2.upload_files["input_mesh_path"], ("gwchain_x.glb", GLB))
        self.assertIn("upload:", ref)

    # ── transport ────────────────────────────────────────────────────────────
    def test_upload_transport_error_fails_over(self):
        """An upload that cannot reach Tripo is a failover-class ConnectionError (an
        httpx.ConnectError is NOT one), and the message names the step."""
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        ad = adapters.TripoAdapter(self._backend(f"http://127.0.0.1:{port}"), self.ctx)
        with self.assertRaises(ConnectionError) as cm:
            self._run(ad.generate(self._req(images={"input_image": PNG})))
        self.assertIn("upload", str(cm.exception))
        self.assertEqual((self.counts["inc"], self.counts["dec"]), (1, 1))


class TestRegistry(unittest.TestCase):
    def test_registry(self):
        self.assertIs(adapters.ADAPTERS["tripo"], adapters.TripoAdapter)
        self.assertIn("tripo", adapters.GEN_TYPES)
        self.assertIn("tripo", adapters.CLOUD_TYPES)
        self.assertEqual(adapters.TripoAdapter.type, "tripo")


if __name__ == "__main__":
    unittest.main()
