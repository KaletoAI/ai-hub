"""The provider seam: `hostapi.ProviderApi`/`ThunderApi` (the HTTP half of a managed
host's provider), the provider registry, and `thunder.OPTION_FIELDS`/`options_of` (the
host form as data). The ThunderApi cases moved here from test_hostctl.py
unchanged — every one guards a rule whose failure is silent or costs money: an index
tried before the uuid can delete a STRANGER's instance, a token echoed into an error
lands in the panel and the fault log, a price list fetched per view hammers the API.
run: venv/bin/python -m unittest tests.test_hostapi -v"""
import types
import unittest

import httpx

import hostapi
import thunder
import hostctl
from tests.fakes import FakeThunder  # the scripted Thunder REST API


def _api(fake, **kw):
    return hostapi.ThunderApi(
        httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)), "tok", **kw)


class Api(unittest.IsolatedAsyncioTestCase):
    async def test_api_delete_falls_back_to_index(self):
        # uuid first (Ruling 11: a stale index can name a stranger's instance)
        fake = FakeThunder()
        fake.delete_by = "index"
        fake.instances["0"] = {"uuid": "u0", "status": "RUNNING"}
        await _api(fake).delete({"index": "0", "uuid": "u0"})
        self.assertEqual(fake.instances, {})
        paths = [p for _, p, _ in fake.calls]
        self.assertEqual(paths, ["/instances/u0/delete", "/instances/0/delete"])

    async def test_api_delete_by_uuid_never_touches_index(self):
        fake = FakeThunder()
        fake.delete_by = "uuid"
        fake.instances["0"] = {"uuid": "u0", "status": "RUNNING"}
        await _api(fake).delete({"index": "0", "uuid": "u0"})
        self.assertEqual([p for _, p, _ in fake.calls], ["/instances/u0/delete"])

    async def test_int_index_zero_is_kept(self):
        # `item.get("index") or ""` dropped index 0
        fake = FakeThunder()
        fake.instances["0"] = {"uuid": "u0", "status": "RUNNING"}
        await _api(fake).delete({"index": 0, "uuid": ""})
        self.assertEqual([p for _, p, _ in fake.calls], ["/instances/0/delete"])

    async def test_api_delete_both_404_is_ok(self):
        fake = FakeThunder()
        await _api(fake).delete({"index": "3", "uuid": "u3"})   # already gone
        self.assertEqual(len(fake.calls), 2)

    async def test_api_non_2xx_raises_thundererror_with_status(self):
        def handler(req):
            return httpx.Response(401, text="unauthorized " + "x" * 1000)
        api = hostapi.ThunderApi(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tok")
        with self.assertRaises(thunder.ThunderError) as cm:
            await api.list_instances()
        self.assertEqual(cm.exception.status, 401)
        self.assertIn("unauthorized", str(cm.exception))
        self.assertLessEqual(len(str(cm.exception)), 300)

    async def test_transport_error_is_thundererror_without_status(self):
        def handler(req):
            raise httpx.ConnectError("")          # empty str(e), like a real one can be
        api = hostapi.ThunderApi(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tok")
        with self.assertRaises(thunder.ThunderError) as cm:
            await api.snapshots()
        self.assertIsNone(cm.exception.status)
        self.assertIn("ConnectError", str(cm.exception))

    async def test_bearer_token_sent(self):
        seen = []

        def handler(req):
            seen.append(req.headers.get("authorization"))
            return httpx.Response(200, json={})
        api = hostapi.ThunderApi(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tok")
        await api.list_instances()
        self.assertEqual(seen, ["Bearer tok"])

    async def test_create_returns_index_as_string(self):
        fake = FakeThunder()
        r = await _api(fake).create({"template": "comfy-ui"})
        self.assertEqual(r, {"index": "0", "uuid": "u0"})

    async def test_list_instances_and_snapshots_are_parsed(self):
        fake = FakeThunder()
        fake.instances["4"] = {"uuid": "u4", "status": "running", "httpPorts": [8188]}
        fake.snaps.append({"id": "s0", "name": "aihub-thunder-20260927t000000z",
                           "status": "READY", "minimumDiskSizeGb": 120, "createdAt": 5})
        api = _api(fake)
        fake.status_script = []
        items = await api.list_instances()
        self.assertEqual(items[0]["index"], "4")
        self.assertEqual(items[0]["http_ports"], [8188])
        snaps = await api.snapshots()
        self.assertEqual(snaps[0]["min_disk_gb"], 120)

    async def test_remove_ports_uuid_then_index(self):
        fake = FakeThunder()
        fake.instances["0"] = {"uuid": "u0", "httpPorts": [8188, 22]}
        await _api(fake).remove_ports({"index": "0", "uuid": "u0"}, [8188])
        self.assertEqual(fake.calls, [("PATCH", "/instances/u0/ports", {"remove_ports": [8188]}),
                                      ("PATCH", "/instances/0/ports", {"remove_ports": [8188]})])
        self.assertEqual(fake.instances["0"]["httpPorts"], [22])

    async def test_modify_falls_back_to_index_and_raises_when_both_404(self):
        calls = []

        def handler(req):
            calls.append(req.url.path)
            return httpx.Response(200, json={}) if req.url.path == "/instances/0/modify" \
                else httpx.Response(404)
        api = hostapi.ThunderApi(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tok")
        await api.modify({"index": "0", "uuid": "u0"}, {"disk_size_gb": 200})
        self.assertEqual(calls, ["/instances/u0/modify", "/instances/0/modify"])
        with self.assertRaises(thunder.ThunderError) as cm:
            await api.modify({"index": "1", "uuid": "u1"}, {"disk_size_gb": 200})
        self.assertEqual(cm.exception.status, 404)

    async def test_snapshot_create_and_delete(self):
        fake = FakeThunder()
        api = _api(fake)
        sid = await api.create_snapshot({"index": "0", "uuid": "u0"}, "aihub-thunder-x")
        self.assertEqual(sid, "s0")
        self.assertEqual(fake.calls[-1][2]["name"], "aihub-thunder-x")
        # openapi CreateSnapshotRequest.instanceId is a STRING — an int is a 400
        self.assertEqual(fake.calls[-1][2]["instanceId"], "0")
        await api.create_snapshot({"index": 0, "uuid": "u0"}, "aihub-thunder-y")
        self.assertEqual(fake.calls[-1][2]["instanceId"], "0")
        await api.delete_snapshot(sid)
        self.assertEqual([x["id"] for x in fake.snaps], ["s1"])

    async def test_pricing_and_specs_cached_one_hour(self):
        fake = FakeThunder()
        clock = [1000.0]
        api = _api(fake, clock=lambda: clock[0])
        p1 = await api.pricing()
        await api.pricing()
        await api.specs()
        await api.specs()
        self.assertEqual(p1["pricing"]["a6000_x1"], 0.35)
        self.assertEqual([p for _, p, _ in fake.calls], ["/v2/pricing", "/v2/specs"])
        clock[0] += 3601
        await api.pricing()
        self.assertEqual(len(fake.calls), 3)

    async def test_ids_are_path_quoted(self):
        # an id with a "/" must not reach a different endpoint
        seen = []

        def handler(req):
            seen.append(req.url.raw_path)
            return httpx.Response(200, json={})
        api = hostapi.ThunderApi(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tok")
        await api.delete_snapshot("a/../b")
        self.assertEqual(seen, [b"/snapshots/a%2F..%2Fb"])

    # M4 ─ a 2xx body in an error is redacted
    async def test_create_errors_redact_an_echoed_token(self):
        secret = "sekrit-thunder-token"

        def handler(req):
            return httpx.Response(200, json={"echo": req.headers.get("authorization")})
        api = hostapi.ThunderApi(
            httpx.AsyncClient(transport=httpx.MockTransport(handler)), secret)
        for call in (lambda: api.create({}),
                     lambda: api.create_snapshot({"index": "1"}, "aihub-x")):
            with self.assertRaises(thunder.ThunderError) as cm:
                await call()
            self.assertNotIn(secret, str(cm.exception))
            self.assertIn("***", str(cm.exception))

    # the API half of TokenHygiene (the controller half — view, state, log — stays there)
    async def test_token_never_in_api_errors(self):
        token = "thunder-SECRET-token-123"
        mode = ["error"]

        def handler(req):
            if mode[0] == "transport":
                raise httpx.ConnectError(f"cannot reach {req.url}")
            # an API that echoes the request back, Authorization included
            return httpx.Response(500, text=f"boom: {dict(req.headers)}")
        api = hostapi.ThunderApi(httpx.AsyncClient(transport=httpx.MockTransport(handler)), token)
        errors = []
        for m in ("error", "transport"):
            mode[0] = m
            for call in (api.list_instances, api.snapshots, api.pricing, api.specs):
                with self.assertRaises(thunder.ThunderError) as cm:
                    await call()
                errors.append(str(cm.exception))
        self.assertIn("***", errors[0])
        for e in errors:
            self.assertNotIn(token, e)

    def test_controller_uses_the_registry_class(self):
        # the re-export ended with the rename: the controller takes its API class from
        # the registry — the SAME class, or an isinstance/patch in one place misses the
        # other
        self.assertFalse(hasattr(hostctl, "ThunderApi"))
        self.assertTrue(issubclass(hostapi.ThunderApi, hostapi.ProviderApi))
        deps = types.SimpleNamespace(load_state=lambda n: None, now=lambda: 0.0,
                                     log=lambda m: None)
        c = hostctl.Controller({"name": "h", "provider": "thunder", "options": {},
                                "api_key": ""}, [], deps)
        self.assertIs(c._api_cls, hostapi.ThunderApi)
        self.assertIs(c._Error, thunder.ThunderError)
        with self.assertRaises(ValueError):             # shown, never driven
            hostctl.Controller({"name": "h", "provider": "runpod"}, [], deps)


class Base(unittest.IsolatedAsyncioTestCase):
    """A second provider needs only its names: another item path, another error class,
    no Thunder fact inherited by accident."""

    class _Err(RuntimeError):
        def __init__(self, msg, status=None):
            super().__init__(msg)
            self.status = status

    def _fake_api(self, handler):
        class PodApi(hostapi.ProviderApi):
            API = "https://pods.example"
            ITEM_PATH = "/pods/{id}/{suffix}"
            Error = Base._Err
        return PodApi(httpx.AsyncClient(transport=httpx.MockTransport(handler)), "tok")

    async def test_other_provider_gets_its_own_paths_and_error(self):
        seen = []

        def handler(req):
            seen.append((req.url.host, req.url.path))
            return httpx.Response(404 if req.url.path == "/pods/u1/stop" else 200, json={})
        api = self._fake_api(handler)
        await api._by_id("POST", {"uuid": "u1", "index": "1"}, "stop")
        self.assertEqual(seen, [("pods.example", "/pods/u1/stop"), ("pods.example", "/pods/1/stop")])

        def refuse(req):
            return httpx.Response(403, text="nope")
        with self.assertRaises(Base._Err) as cm:
            await self._fake_api(refuse)._call("GET", "/pods")
        self.assertEqual(cm.exception.status, 403)


    async def test_provider_without_item_path_is_refused(self):
        # an empty ITEM_PATH would send the call to the API root — no request at all
        seen = []

        class NoPath(hostapi.ProviderApi):
            API = "https://pods.example"
            Error = Base._Err
        api = NoPath(httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: seen.append(req) or httpx.Response(200))), "tok")
        with self.assertRaises(Base._Err) as cm:
            await api._by_id("POST", {"uuid": "u1"}, "stop")
        self.assertIn("NoPath has no ITEM_PATH", str(cm.exception))
        self.assertEqual(seen, [])


class Registry(unittest.TestCase):
    def test_registry_has_thunder(self):
        self.assertEqual(hostapi.PROVIDERS["thunder"], (thunder, hostapi.ThunderApi))
        self.assertEqual(hostapi.provider("thunder"), (thunder, hostapi.ThunderApi))
        self.assertIsNone(hostapi.provider("runpod"))
        self.assertIsNone(hostapi.provider(None))
        # the registry key IS the module's KIND — one name, not two that can drift
        for kind, (mod, api) in hostapi.PROVIDERS.items():
            self.assertEqual(mod.KIND, kind)

    def test_interface_constants(self):
        self.assertEqual((thunder.KIND, thunder.NAME, thunder.SSH_USER, thunder.STOP_MODE,
                          thunder.DEFAULT_TEMPLATE_NO_COMFY),
                         ("thunder", "Thunder Compute", "ubuntu", "snapshot", "base"))
        # the controller's ssh user is the provider's, not a second literal
        deps = types.SimpleNamespace(load_state=lambda n: None, now=lambda: 0.0,
                                     log=lambda m: None)
        c = hostctl.Controller({"name": "h", "provider": "thunder"}, [], deps)
        c.state.ip = "10.0.0.5"
        self.assertEqual(c._login(), f"{thunder.SSH_USER}@10.0.0.5")


# /v2/specs for every GPU type the form offers (a6000 ×1's options as in tests/fakes.py)
SPECS_A6000 = {"specs": {f"{g}_x1": {"vcpuOptions": [6, 8]} for g in thunder.GPU_TYPES}}


class OptionFields(unittest.TestCase):
    def _fields(self):
        return {f["key"]: f for f in thunder.OPTION_FIELDS}

    def test_field_shape(self):
        f = self._fields()
        self.assertEqual(list(f), ["gpu_type", "num_gpus", "vcpus", "bootstrap_template",
                                   "reserve_gb", "comfy_commit", "nodes"])
        for k, fld in f.items():
            self.assertIn(fld["type"], ("select", "int", "text", "textarea"), k)
            for attr in ("label", "default", "hint"):
                self.assertIn(attr, fld, k)
            if fld["type"] == "select":
                self.assertIn(fld["default"], fld["choices"], k)
        self.assertEqual(f["gpu_type"]["choices"], ["a6000", "l40", "a100xl", "h100"])
        # "" = auto (Ruling M5): comfy-ui with a ComfyUI service, else base
        self.assertEqual(f["bootstrap_template"]["choices"], ["", "comfy-ui", "base"])

    def test_option_fields_cover_create_body(self):
        # every option create_body reads must be on the form, or a new host can only
        # ever send the code's fallback for it (or KeyError in the middle of a start)
        read = set()

        class Rec(dict):
            def __getitem__(self, k):
                read.add(k)
                return super().__getitem__(k)

            def get(self, k, d=None):
                read.add(k)
                return super().get(k, d)
        opts, errs, _ = thunder.options_of({})
        self.assertEqual(errs, [])
        # the default vcpus is "included", resolved at start from the specs (the
        # smallest option — what an a6000 ×1 bills nothing extra for)
        opts, why = thunder.resolve_options(opts, SPECS_A6000)
        self.assertIsNone(why)
        body = thunder.create_body(Rec(opts), "comfy-ui", 120, "ssh-ed25519 AAAA")
        self.assertTrue(read)
        self.assertLessEqual(read, set(self._fields()))
        self.assertEqual((body["gpu_type"], body["num_gpus"], body["cpu_cores"]), ("a6000", 1, 6))
        # and what the controller reads beyond create_body
        self.assertLessEqual({"bootstrap_template", "reserve_gb", "comfy_commit", "nodes"},
                             set(self._fields()))

    def _assert_valid(self, opts):
        """Every key valid, whatever the form said: what a caller stores or hands to
        create_body must never be a typo."""
        f = self._fields()
        self.assertEqual(set(opts), set(f))
        for k, fld in f.items():
            v = opts[k]
            if fld["type"] == "int" and v == "" and fld["default"] == "":
                continue            # a blank with a meaning ("included"), resolved at start
            if fld["type"] == "int":
                self.assertIsInstance(v, int, k)
                self.assertNotIsInstance(v, bool, k)
                self.assertGreaterEqual(v, fld["min"], k)
            elif fld["type"] == "select":
                self.assertIn(v, fld["choices"], k)
            elif fld["type"] == "textarea":
                self.assertIsInstance(v, list, k)
            elif k == "comfy_commit":
                self.assertRegex(v, r"^[0-9a-fA-F]{40}$")
        res, why = thunder.resolve_options(opts, SPECS_A6000)
        self.assertIsNone(why)
        thunder.create_body(res, opts["bootstrap_template"], 100, "k")   # never raises

    def test_options_of_defaults(self):
        opts, errs, typed = thunder.options_of({})
        self.assertEqual((errs, typed), ([], {}))
        self.assertEqual(opts, {k: f["default"] for k, f in self._fields().items()})
        self.assertRegex(opts["comfy_commit"], r"^[0-9a-f]{40}$")
        # blank is the one "unset" — it takes the default like an absent field
        blank = {f"opt__{k}": "" for k in self._fields()}
        bopts, berrs, btyped = thunder.options_of(blank)
        self.assertEqual((bopts, berrs), (opts, []))
        self.assertEqual(btyped, {k: "" for k in self._fields()})
        # the default list is a COPY: a caller appending to it must not change the field
        opts["nodes"].append("x")
        self.assertEqual(thunder.options_of({})[0]["nodes"], [])

    def test_options_of_reads_values(self):
        sha = "ab" * 20
        form = {"opt__gpu_type": "h100", "opt__num_gpus": " 2 ", "opt__vcpus": "16",
                "opt__bootstrap_template": "base", "opt__reserve_gb": "0",
                "opt__comfy_commit": sha, "opt__nodes": "https://x/a.git@1\n\nregistry:b@2\n\n"}
        opts, errs, typed = thunder.options_of(form)
        self.assertEqual(errs, [])
        self.assertEqual(opts, {"gpu_type": "h100", "num_gpus": 2, "vcpus": 16,
                                "bootstrap_template": "base", "reserve_gb": 0,
                                "comfy_commit": sha,
                                "nodes": ["https://x/a.git@1", "", "registry:b@2"]})
        self.assertEqual(typed["num_gpus"], "2")
        self.assertEqual(typed["nodes"], "https://x/a.git@1\n\nregistry:b@2")

    def test_options_of_validates(self):
        opts, errs, typed = thunder.options_of({"opt__comfy_commit": "1d61dcc",
                                                "opt__vcpus": "0", "opt__gpu_type": "rtx9090"})
        self.assertEqual(len(errs), 3, errs)
        joined = " ".join(errs)
        for word in ("commit", "vcpus", "rtx9090"):
            self.assertIn(word, joined)
        # the refused fields are their DEFAULTS in options — never the typo …
        self._assert_valid(opts)
        self.assertEqual(opts, thunder.options_of({})[0])
        # … and what was typed comes back separately, for the form to re-render
        self.assertEqual(typed, {"comfy_commit": "1d61dcc", "vcpus": "0", "gpu_type": "rtx9090"})
        # a whole number only: "1.5", "-1", "1e3", "+2" are errors, never a silent int
        for bad in ("1.5", "-1", "1e3", "+2", "x", "1_000", 1.5, 2.0):
            o, errs, t = thunder.options_of({"opt__num_gpus": bad})
            self.assertEqual(len(errs), 1, bad)
            self.assertEqual(o["num_gpus"], 1, bad)
            self.assertEqual(t["num_gpus"], str(bad), bad)
        for form in ({"opt__reserve_gb": "-1"}, {"opt__num_gpus": "0"},
                     {"opt__bootstrap_template": "windows"}):
            o, errs, _ = thunder.options_of(form)
            self.assertEqual(len(errs), 1, form)
            self._assert_valid(o)

    def test_options_of_never_raises(self):
        for form in (None, [], "x", {"opt__vcpus": 8}, {"opt__nodes": ["a", "b"]},
                     {"opt__gpu_type": None}, {"opt__num_gpus": object()},
                     {"opt__nodes": 5}, {"opt__vcpus": [8]}, {"opt__gpu_type": {"a": 1}},
                     {"opt__nodes": [None, 3, "c"]}):
            opts, errs, typed = thunder.options_of(form)
            self._assert_valid(opts)
            self.assertIsInstance(errs, list)
            self.assertIsInstance(typed, dict)
        # an int (a form built by code, not a browser) is read as its string
        self.assertEqual(thunder.options_of({"opt__vcpus": 8})[:2],
                         (dict(thunder.options_of({})[0], vcpus=8), []))
        self.assertEqual(thunder.options_of({"opt__nodes": ["a", "b"]})[0]["nodes"], ["a", "b"])
        # anything else is validated as its string, never silently the default
        self.assertEqual(len(thunder.options_of({"opt__num_gpus": object()})[1]), 1)
        self.assertEqual(len(thunder.options_of({"opt__vcpus": [8]})[1]), 1)


if __name__ == "__main__":
    unittest.main()
