"""Unit tests for cloudtask.py — run: venv/bin/python -m unittest tests.test_cloudtask -v"""
import asyncio
import json
import unittest

import cloudtask
import meshy

FIELDS = [
    {"key": "flag", "label": "flag", "type": "bool"},
    {"key": "mode", "label": "mode", "type": "select", "choices": [("a", "A"), ("b", "B")]},
    {"key": "tri", "label": "tri", "type": "tristate"},
    {"key": "n", "label": "n", "type": "int"},
    {"key": "txt", "label": "txt", "type": "text"},
    {"key": "lst", "label": "lst", "type": "list"},
]
DEFAULTS = {"flag": True, "mode": "a", "tri": None, "n": None, "txt": "", "lst": []}


class ParseOptions(unittest.TestCase):
    def test_reads_every_type(self):
        form = {"opt__flag": "on", "opt__mode": "b", "opt__tri": "false", "opt__n": "42",
                "opt__txt": " hi ", "opt__lst": "preset:walk, preset:run,,"}
        out = cloudtask.parse_options(FIELDS, form, DEFAULTS)
        self.assertEqual(out, {"flag": True, "mode": "b", "tri": False, "n": 42,
                               "txt": "hi", "lst": ["preset:walk", "preset:run"]})

    def test_missing_checkbox_is_false_and_blank_is_default_shape(self):
        out = cloudtask.parse_options(FIELDS, {"opt__mode": "zzz", "opt__n": "x", "opt__tri": ""}, DEFAULTS)
        self.assertFalse(out["flag"])                  # an unchecked box is not submitted
        self.assertEqual(out["mode"], "a")             # unknown choice → default
        self.assertIsNone(out["n"])                    # garbage int → None (module validates later)
        self.assertIsNone(out["tri"])
        self.assertEqual(out["lst"], [])

    def test_does_not_mutate_defaults(self):
        d = dict(DEFAULTS)
        cloudtask.parse_options(FIELDS, {"opt__lst": "x"}, d)
        self.assertEqual(d, DEFAULTS)

    def test_field_value_str(self):
        self.assertEqual(cloudtask.field_value_str({"type": "tristate"}, None), "")
        self.assertEqual(cloudtask.field_value_str({"type": "tristate"}, True), "true")
        self.assertEqual(cloudtask.field_value_str({"type": "int"}, None), "")
        self.assertEqual(cloudtask.field_value_str({"type": "int"}, 7), "7")
        self.assertEqual(cloudtask.field_value_str({"type": "list"}, ["a", "b"]), "a, b")


class MeshyFields(unittest.TestCase):
    def test_meshy_option_fields_roundtrip_defaults(self):
        """Rendering the defaults into a form and parsing them back yields the defaults —
        the editor and the request builder read the same table."""
        form = {}
        for fld in meshy.OPTION_FIELDS:
            v = meshy.OPTION_DEFAULTS[fld["key"]]
            if fld["type"] == "bool":
                if v:
                    form[f"opt__{fld['key']}"] = "on"
            else:
                form[f"opt__{fld['key']}"] = cloudtask.field_value_str(fld, v)
        out = cloudtask.parse_options(meshy.OPTION_FIELDS, form, meshy.OPTION_DEFAULTS)
        for fld in meshy.OPTION_FIELDS:
            self.assertEqual(out[fld["key"]], meshy.OPTION_DEFAULTS[fld["key"]], fld["key"])

    def test_meshy_fields_cover_every_option_except_formats(self):
        keys = {f["key"] for f in meshy.OPTION_FIELDS}
        self.assertEqual(keys, set(meshy.OPTION_DEFAULTS) - {"target_formats"})

    def test_meshy_module_constants(self):
        self.assertEqual((meshy.KIND, meshy.VENDOR, meshy.RIG_ENDPOINT), ("meshy", "Meshy", "rigging"))
        self.assertTrue(meshy.URL.startswith("https://"))
        self.assertIs(meshy.TaskState, cloudtask.TaskState)

    def test_parse_task_options_kwarg(self):
        task = {"status": "SUCCEEDED", "result": {"rigged_character_glb_url": "u",
                                                  "basic_animations": {"walking_glb_url": "w"}}}
        st = meshy.parse_task(task, ["glb"], "rigging", options={"animations": True})
        self.assertEqual([n for n, _ in st.downloads], ["rigged.glb", "walking.glb"])
        st = meshy.parse_task(task, ["glb"], "rigging", options={"animations": False})
        self.assertEqual([n for n, _ in st.downloads], ["rigged.glb"])


class AdapterHelpers(unittest.TestCase):
    """The kind seam in adapters.py — pure enough to check without a server. main and
    admin ask these four questions everywhere a candidate or backend has a kind."""

    def test_kinds(self):
        import adapters
        self.assertEqual(adapters.cloud_kind({"meshy": {}}), "meshy")
        self.assertEqual(adapters.cloud_kind({"tripo": {"endpoint": "rig"}}), "tripo")
        self.assertIsNone(adapters.cloud_kind({"workflow_json": {}}))
        self.assertEqual(adapters.cand_kind({}), "comfyui")
        self.assertEqual(adapters.backend_kind({"type": "meshy"}), "meshy")
        self.assertEqual(adapters.backend_kind({"type": "comfyui"}), "comfyui")
        self.assertEqual(adapters.backend_kind({"type": "openai"}), "comfyui")
        self.assertIn("meshy", adapters.CLOUD_TYPES)
        self.assertTrue(adapters.CLOUD_TYPES <= adapters.GEN_TYPES)
        self.assertIs(adapters.cloud_module("meshy"), meshy)
        self.assertEqual(adapters.cloud_block({"tripo": {"endpoint": "rig"}}), {"endpoint": "rig"})
        self.assertIsNone(adapters.cloud_block({}))

    def test_exception_vendor(self):
        import adapters
        e = adapters.CloudNoCredits("x", vendor="Tripo")
        self.assertIsInstance(e, ConnectionError)
        self.assertEqual(e.vendor, "Tripo")
        self.assertEqual(adapters.CloudBusy("y").vendor, "cloud")


class RunResultEndpoint(unittest.TestCase):
    """The job meta and the rig/thumbnail decisions describe the task that was actually
    run — `RunResult.endpoint` — not the endpoint the alias is configured for. They are
    the same thing for Meshy (one task per job), but a vendor whose `_run` chases several
    tasks names the primary one there, and a meta that silently kept saying "image-to-3d"
    would send you reading the wrong task's history."""

    def _adapter(self, run_endpoint: str):
        import adapters

        class _Fake(adapters.CloudTaskAdapter):
            mod = meshy
            type = mod.KIND

            async def _run(self, client, req, cand, opts, poll_interval, max_wait):
                return adapters.RunResult(
                    "task-9", run_endpoint, {"ai_model": "latest"},
                    cloudtask.TaskState(status=meshy.SUCCESS_STATUS, credits=5))

        ctx = adapters.AdapterContext(
            auth_headers=lambda b: {}, inflight_inc=lambda bid: None, inflight_dec=lambda bid: None,
            cost_usd=lambda *a: 0.0, source_of=lambda r: "test", record_call=lambda *a, **k: None,
            log_enabled=lambda: False)
        return _Fake({"name": "cloud", "type": "meshy", "url": "http://127.0.0.1:1"}, ctx)

    def _meta(self, run_endpoint: str, alias_endpoint: str) -> dict:
        import adapters
        req = adapters.NormalizedRequest(alias="a", real_model="latest",
                                         cloud={"endpoint": alias_endpoint, "options": {}})
        return asyncio.run(self._adapter(run_endpoint).generate(req)).meta

    def test_meta_and_rig_follow_the_run(self):
        meta = self._meta(meshy.RIG_ENDPOINT, "image-to-3d")
        self.assertEqual(meta["endpoint"], meshy.RIG_ENDPOINT)
        self.assertEqual(meta["cloud_task_id"], "task-9")
        self.assertEqual(meta["rig"], meshy.KIND)      # the rig branch reads the run too

    def test_no_rig_when_the_run_was_not_one(self):
        meta = self._meta("image-to-3d", meshy.RIG_ENDPOINT)
        self.assertEqual(meta["endpoint"], "image-to-3d")
        self.assertNotIn("rig", meta)


class ArtifactDownloadRetry(unittest.TestCase):
    """A finished cloud task's artifact GET is free and idempotent, so a dropped transfer
    is retried instead of failing the job. assets.meshy.ai (CloudFront) cuts a download
    mid-body now and then — measured 2026-10-15: 1 in 3 GETs of the same 9 MB GLB died
    after 1–2 MB while the next took 0.4 s. With ONE try, two paid meshes became failed
    jobs ("peer closed connection without sending complete message body"), and the
    billing rule (never re-create a paid task) left them unrecoverable from the client."""

    def setUp(self):
        import adapters
        self.adapters = adapters
        self._backoff = adapters._CLOUD_DOWNLOAD_BACKOFF_S
        adapters._CLOUD_DOWNLOAD_BACKOFF_S = 0.0

    def tearDown(self):
        self.adapters._CLOUD_DOWNLOAD_BACKOFF_S = self._backoff

    def _get(self, outcomes: list):
        import httpx
        calls = []

        def handler(request):
            calls.append(request)
            o = outcomes[min(len(calls), len(outcomes)) - 1]
            if isinstance(o, Exception):
                raise o
            return httpx.Response(o, content=b"GLB" if o == 200 else b"err")

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await self.adapters._download_asset(client, "https://cdn/x.glb")
        return run, calls

    def test_dropped_transfer_is_retried(self):
        import httpx
        run, calls = self._get([httpx.RemoteProtocolError("peer closed connection"), 200])
        self.assertEqual(asyncio.run(run()), b"GLB")
        self.assertEqual(len(calls), 2)

    def test_server_error_is_retried(self):
        run, calls = self._get([503, 200])
        self.assertEqual(asyncio.run(run()), b"GLB")
        self.assertEqual(len(calls), 2)

    def test_client_error_is_final_at_once(self):
        """A 403/404 (expired or wrong signed URL) does not get better by asking again."""
        run, calls = self._get([404, 200])
        with self.assertRaises(RuntimeError):
            asyncio.run(run())
        self.assertEqual(len(calls), 1)

    def test_gives_up_with_the_transport_error(self):
        """Still a transport error after the last try — generate() turns exactly that into
        the final, non-failover RuntimeError naming the paid task."""
        import httpx
        run, calls = self._get([httpx.ReadError("reset")])
        with self.assertRaises(httpx.TransportError):
            asyncio.run(run())
        self.assertEqual(len(calls), self.adapters._CLOUD_DOWNLOAD_ATTEMPTS)


class ArtifactUrlFence(unittest.TestCase):
    """The artifact URL is data from the vendor (or a redirect of its CDN) and the
    gateway fetches it from INSIDE the LAN, handing the body to the caller as the job's
    file. Without a fence a compromised vendor answer — or one redirect — reads
    `http://127.0.0.1:4000/…`, a backend's admin port or the metadata service through
    the gateway; and a body without end is buffered until the process dies. Silent
    both ways: the job just "succeeds" with the wrong bytes, or the box runs out of
    memory. Pinned: literal and RESOLVED private addresses are refused on every hop,
    refusals are final (never retried), and the body is capped."""

    def setUp(self):
        import adapters
        self.adapters = adapters
        self._saved = (adapters._CLOUD_DOWNLOAD_BACKOFF_S, adapters._resolve_asset_host,
                       adapters._CLOUD_DOWNLOAD_MAX_BYTES)
        adapters._CLOUD_DOWNLOAD_BACKOFF_S = 0.0
        self.dns = {"cdn.vendor.example": ["93.184.216.34"], "evil.example": ["10.0.0.7"],
                    "mapped.example": ["::ffff:127.0.0.1"],
                    "mixed.example": ["93.184.216.34", "192.168.8.10"]}

        async def resolve(host, port):
            if host not in self.dns:
                raise OSError("nxdomain")
            return self.dns[host]
        adapters._resolve_asset_host = resolve

    def tearDown(self):
        a = self.adapters
        a._CLOUD_DOWNLOAD_BACKOFF_S, a._resolve_asset_host, a._CLOUD_DOWNLOAD_MAX_BYTES = self._saved

    def _fetch(self, url, handler):
        import httpx
        calls = []

        def h(request):
            calls.append(str(request.url))
            return handler(request)

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(h)) as client:
                return await self.adapters._download_asset(client, url)
        return run, calls

    def _ok(self, request):
        import httpx
        return httpx.Response(200, content=b"GLB")

    def test_private_targets_are_refused_before_any_request(self):
        for url in ("http://127.0.0.1:4000/health", "https://192.168.8.34:8188/system_stats",
                    "http://169.254.169.254/latest/meta-data/", "https://[::1]/x",
                    "https://[::ffff:10.0.0.1]/x", "https://100.64.0.1/x",
                    "https://evil.example/x.glb", "https://mapped.example/x",
                    "https://mixed.example/x", "ftp://cdn.vendor.example/x",
                    "file:///etc/passwd", "https://user:pw@cdn.vendor.example/x", ""):
            run, calls = self._fetch(url, self._ok)
            with self.assertRaises(RuntimeError, msg=url):
                asyncio.run(run())
            self.assertEqual(calls, [], url)           # refused, and not retried

    def test_public_and_unresolvable_hosts_pass(self):
        for url in ("https://cdn.vendor.example/x.glb?sig=1", "https://93.184.216.34/x",
                    "https://no-such-host.example/x"):
            run, calls = self._fetch(url, self._ok)
            self.assertEqual(asyncio.run(run()), b"GLB")
            self.assertEqual(calls, [url])

    def test_only_the_operators_own_endpoint_is_trusted(self):
        """A vendor stub/proxy on the LAN serves its files from the configured URL —
        that host AND port, nothing else on the box."""
        a = self.adapters
        trusted = frozenset({a._asset_endpoint("http://192.168.8.50:9000/api")})
        self.assertEqual(trusted, {("192.168.8.50", 9000)})
        ok = asyncio.run(a._checked_asset_url("http://192.168.8.50:9000/files/a.glb", trusted))
        self.assertTrue(ok)
        for url in ("http://192.168.8.50:4000/ui", "http://192.168.8.51:9000/a", "http://127.0.0.1:9000/a"):
            with self.assertRaises(RuntimeError, msg=url):
                asyncio.run(a._checked_asset_url(url, trusted))
        self.assertIsNone(a._asset_endpoint(None))
        self.assertIsNone(a._asset_endpoint("not a url"))

    def test_redirect_into_the_lan_is_refused(self):
        import httpx

        def handler(request):
            return httpx.Response(302, headers={"location": "http://192.168.8.10:4000/ui"})
        run, calls = self._fetch("https://cdn.vendor.example/x.glb", handler)
        with self.assertRaises(RuntimeError):
            asyncio.run(run())
        self.assertEqual(calls, ["https://cdn.vendor.example/x.glb"])

    def test_public_redirect_is_followed(self):
        import httpx

        def handler(request):
            if request.url.path == "/a":
                return httpx.Response(302, headers={"location": "/b"})
            return httpx.Response(200, content=b"GLB")
        run, calls = self._fetch("https://cdn.vendor.example/a", handler)
        self.assertEqual(asyncio.run(run()), b"GLB")
        self.assertEqual(len(calls), 2)

    def test_redirect_loop_ends(self):
        import httpx
        run, calls = self._fetch("https://cdn.vendor.example/a",
                                 lambda r: httpx.Response(302, headers={"location": "/a"}))
        with self.assertRaises(RuntimeError):
            asyncio.run(run())
        self.assertEqual(len(calls), self.adapters._CLOUD_DOWNLOAD_REDIRECTS + 1)

    def test_body_is_capped(self):
        import httpx
        self.adapters._CLOUD_DOWNLOAD_MAX_BYTES = 10
        run, calls = self._fetch("https://cdn.vendor.example/a",
                                 lambda r: httpx.Response(200, content=b"x" * 11))
        with self.assertRaises(RuntimeError):
            asyncio.run(run())
        self.assertEqual(len(calls), 1)                # final, not retried
        run, _ = self._fetch("https://cdn.vendor.example/a",
                             lambda r: httpx.Response(200, content=b"x" * 10))
        self.assertEqual(asyncio.run(run()), b"x" * 10)


class StreamedBody(unittest.TestCase):
    """A rigging request carries the whole mesh as a base64 data URI. Built the plain
    way a 70 MB GLB became a 93 MB str, a 93 MB `json.dumps` text and its 93 MB encoded
    copy — and since `json.dumps`/base64 hold the GIL for the whole call, the event
    loop stood still ~0.2 s per job even with the work in a worker thread (measured
    2026-11-18). `cloudtask.DataUri` + `json_stream` send the same bytes in slices
    without ever holding more than the original. Silent if wrong: the vendor gets a
    corrupt body and bills a task, or the Content-Length is off and the POST hangs.
    Pinned: the stream is BYTE-IDENTICAL to json.dumps for every padding and slice
    boundary, and the announced length is exact."""

    def setUp(self):
        import cloudtask
        self.ct = cloudtask

    def _plain(self, body):
        return json.dumps(body, default=str).encode()

    def test_identical_to_json_dumps_for_every_size(self):
        ct = self.ct
        for n in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100, 1000):
            data = bytes(range(256)) * 4
            data = data[:n]
            for raw in (3, 6, 9, 999):
                body = {"a": 1, "model_url": ct.DataUri("model/gltf-binary", data),
                        "name": 'K"ä\\i', "z": [ct.DataUri("image/png", data[::-1]), None, 2.5]}
                length, chunks, blobs = ct.json_stream(body, raw=raw)
                got = b"".join(chunks)
                self.assertEqual(got, self._plain(body), (n, raw))
                self.assertEqual(length, len(got))
                self.assertEqual(blobs, 2)
                self.assertEqual(json.loads(got)["model_url"], str(body["model_url"]))

    def test_no_blob_is_plain_json(self):
        body = {"a": [1, 2, {"b": "x"}], "s": "@@gwblob:not-a-token@@"}
        length, chunks, blobs = self.ct.json_stream(body)
        self.assertEqual((b"".join(chunks), blobs), (self._plain(body), 0))
        self.assertEqual(length, len(self._plain(body)))

    def test_a_client_string_cannot_pose_as_a_blob(self):
        ct = self.ct
        body = {"evil": "@@gwblob:00000000000000000000000000000000:0@@",
                "m": ct.DataUri("x/y", b"abc")}
        _, chunks, _ = ct.json_stream(body)
        self.assertEqual(b"".join(chunks), self._plain(body))

    def test_slices_stay_small(self):
        ct = self.ct
        _, chunks, _ = ct.json_stream({"m": ct.DataUri("x/y", b"\x00" * 100000)}, raw=3000)
        self.assertLessEqual(max(len(c) for c in chunks), 4000)

    def test_data_uri_basics(self):
        d = self.ct.DataUri("model/gltf-binary", b"glTFabc")
        self.assertEqual(str(d), "data:model/gltf-binary;base64,Z2xURmFiYw==")
        self.assertEqual(len(d), len(str(d)))
        self.assertEqual(d.size, 7)
        with self.assertRaises(TypeError):
            json.dumps({"m": d})                      # never serialised by accident


if __name__ == "__main__":
    unittest.main()
