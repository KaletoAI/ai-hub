"""The console half of a Thunder Compute backend: the form block, its save, the panel.

Every one of these fails SILENTLY:
  · `backend_save` must derive `url`/`host` from the Thunder block — a URL typed by hand
    (or left over from before the block) points discovery at a port the tunnel does not
    serve, and a host derived from `127.0.0.1` groups every Thunder backend AND the
    gateway box into one "host" for the VRAM/LLM policies;
  · the URL input must be `readonly`, never `disabled` — a disabled input is not
    submitted, and `backend_save` reads an absent field as cleared;
  · two Thunder backends on one local port: the second tunnel cannot bind, and its
    ComfyUI probe then talks to the FIRST backend's instance — healthy, routed, wrong;
  · a bad `comfy_commit` or a missing GPU/vCPU count is only noticed after an instance
    was created and billed (the bootstrap exits 2 / `create_body` raises), so the form
    refuses them up front;
  · the panel's actions must be POSTs (a GET link fires on any prefetch — here it would
    start a billing GPU), stop/forget ask first, and the page must stay live while an
    instance runs or an op is in flight, or the phase shown is a stale snapshot;
  · the API token (the backend's `api_key`) is never rendered.

Run: python -m unittest tests.test_thunder_ui -v
"""
import asyncio
import os
import re
import sys
import tempfile
import unittest
from urllib.parse import urlencode, parse_qs, urlparse

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    import admin
    import store
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

from fastapi.testclient import TestClient     # noqa: E402

SAME = {"sec-fetch-site": "same-origin"}
SHA = "1d61dcc35c35541388c0001bacc7703db14e8bea"


class _Req:
    def __init__(self, form: dict, query: str = ""):
        self._body = urlencode(form).encode()
        self.query_params = {k: v[-1] for k, v in parse_qs(query).items()}

    async def stream(self):
        yield self._body


def _thunder_form(name="tc", **over) -> dict:
    f = {"name": name, "type": "comfyui", "url": "", "thunder_on": "1",
         "thunder_gpu": "a6000", "thunder_num_gpus": "1", "thunder_vcpus": "8",
         "thunder_template": "comfy-ui", "thunder_reserve_gb": "20",
         "thunder_local_port": "18188", "thunder_comfy_commit": SHA,
         "thunder_nodes": "# comment\nhttps://github.com/a/b.git@" + "a" * 40 + "\n"}
    f.update(over)
    return f


def _view(**over) -> dict:
    v = {"name": "tc", "phase": "off", "error": "", "failed_phase": "", "index": None,
         "uuid": "", "ip": "", "port": 0, "started_at": 0, "uptime_s": 0, "disk_gb": 0,
         "cost_per_h": None, "session_cost": None,
         "snapshot": {"id": "", "pending": "", "pending_name": "", "name": "", "status": "",
                      "gb": None, "monthly": None},
         "log": [], "transfers": {}, "persist_blocked": False, "bootstrap_unknown": {},
         "bootstrap_template_nodes": [], "bootstrap_incomplete": False, "op": None,
         "waiting_jobs": None, "unreconciled_uuids": [], "orphans": []}
    v.update(over)
    return v


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        saved = (store._DB_PATH, store._active, store._MASTER_KEY)

        def restore():
            store._DB_PATH, store._active, store._MASTER_KEY = saved
        self.addCleanup(restore)
        store._MASTER_KEY = os.urandom(32)
        store.init(os.path.join(self.tmp.name, "store.db"))
        self.live = []                 # what _gateway_info reports (config + store view)
        self.views = {}
        self.actions = []

        async def act(name, action):
            self.actions.append((name, action))
            return f"{action} requested"
        stubs = {"_apply_backends": lambda: None,
                 "_gateway_info": lambda: {"backends": self.live, "virtual_models": []},
                 "_thunder_names": lambda: list(self.views),
                 "_thunder_view": lambda n: self.views.get(n),
                 "_thunder_action": act,
                 "_thunder_default_nodes": lambda: "# defaults\nregistry:x@1.0\n"}
        for k, v in stubs.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)

    def save(self, form):
        return asyncio.run(admin.backend_save(_Req(form)))

    def page(self, qp=None) -> str:
        r = asyncio.run(admin._backends_view(qp or {}))
        return r.body.decode()


class Save(_Base):
    def test_save_derives_url_and_host(self):
        r = self.save(_thunder_form(thunder_local_port="18190"))
        self.assertEqual(r.status_code, 303)
        b = store.get_backend("tc", "comfyui")
        self.assertEqual(b["url"], "http://127.0.0.1:18190")
        self.assertEqual(b["host"], "thunder-tc")
        self.assertEqual(b["comfy_output_dir"], "/home/ubuntu/ComfyUI/output")
        self.assertEqual(b["comfy_input_dir"], "/home/ubuntu/ComfyUI/input")
        t = b["thunder"]
        self.assertEqual(t["gpu_type"], "a6000")
        self.assertEqual(t["num_gpus"], 1)
        self.assertEqual(t["vcpus"], 8)
        self.assertEqual(t["bootstrap_template"], "comfy-ui")
        self.assertEqual(t["reserve_gb"], 20)
        self.assertEqual(t["local_port"], 18190)
        self.assertEqual(t["comfy_commit"], SHA)
        # lines verbatim, comments included (the bootstrap ignores them)
        self.assertEqual(t["nodes"], ["# comment", "https://github.com/a/b.git@" + "a" * 40])

    def test_typed_url_and_host_are_overridden_but_comfy_dirs_kept(self):
        self.save(_thunder_form(url="http://10.0.0.5:8188", host="gpu9",
                                comfy_output_dir="/x/out", comfy_input_dir="/x/in"))
        b = store.get_backend("tc", "comfyui")
        self.assertEqual(b["url"], "http://127.0.0.1:18188")
        self.assertEqual(b["host"], "thunder-tc")
        self.assertEqual(b["comfy_output_dir"], "/x/out")
        self.assertEqual(b["comfy_input_dir"], "/x/in")

    def test_unticked_removes_the_block(self):
        self.save(_thunder_form())
        f = _thunder_form(orig="comfyui:tc", url="http://127.0.0.1:18188")
        del f["thunder_on"]
        self.assertEqual(self.save(f).status_code, 303)
        self.assertNotIn("thunder", store.get_backend("tc", "comfyui"))

    def test_block_only_for_comfyui(self):
        f = _thunder_form(type="openai", url="http://h:1")
        self.assertEqual(self.save(f).status_code, 303)
        b = store.get_backend("tc", "openai")
        self.assertNotIn("thunder", b)
        self.assertEqual(b["url"], "http://h:1")

    def test_blank_commit_is_the_default_bad_commit_refused(self):
        self.save(_thunder_form(thunder_comfy_commit=""))
        self.assertEqual(store.get_backend("tc", "comfyui")["thunder"]["comfy_commit"], SHA)
        for bad in ("1d61dcc", "master", "g" * 40, SHA + "0"):
            r = self.save(_thunder_form(name="t2", thunder_comfy_commit=bad,
                                        thunder_local_port="18200"))
            self.assertEqual(r.status_code, 400, bad)
            self.assertIn("40-hex", r.body.decode())
            self.assertIsNone(store.get_backend("t2", "comfyui"), bad)

    def test_gpu_and_vcpus_required(self):
        for over in ({"thunder_gpu": ""}, {"thunder_vcpus": ""}, {"thunder_vcpus": "0"},
                     {"thunder_vcpus": "8.5"}, {"thunder_num_gpus": "0"},
                     {"thunder_local_port": "70000"}, {"thunder_reserve_gb": "-1"}):
            r = self.save(_thunder_form(**over))
            self.assertEqual(r.status_code, 400, over)
            self.assertIsNone(store.get_backend("tc", "comfyui"), over)

    def test_refused_form_is_shown_as_typed(self):
        r = self.save(_thunder_form(thunder_vcpus="lots"))
        html = r.body.decode()
        self.assertEqual(r.status_code, 400)
        self.assertIn('value="lots"', html)
        self.assertRegex(html, r'name="thunder_on" value="1" checked')

    def test_port_collision_refused(self):
        self.assertEqual(self.save(_thunder_form(name="one")).status_code, 303)
        r = self.save(_thunder_form(name="two"))
        self.assertEqual(r.status_code, 400)
        self.assertIn("18188", r.body.decode())
        self.assertIsNone(store.get_backend("two", "comfyui"))
        # a config-defined Thunder backend (live summary only) counts as well
        self.live = [{"name": "cfg", "type": "comfyui", "url": "http://127.0.0.1:18300",
                      "enabled": True, "healthy": True, "models": 0, "source": "config",
                      "thunder": {"local_port": 18300}}]
        r = self.save(_thunder_form(name="three", thunder_local_port="18300"))
        self.assertEqual(r.status_code, 400)
        # re-saving the SAME backend on its own port is no collision
        f = _thunder_form(name="one", orig="comfyui:one")
        self.assertEqual(self.save(f).status_code, 303)
        self.assertEqual(self.save(_thunder_form(name="two", thunder_local_port="18189"))
                         .status_code, 303)

    def test_api_key_stays_the_token(self):
        self.save(_thunder_form(api_key="th-SECRET"))
        self.assertEqual(store.get_backend("tc", "comfyui")["api_key"], "th-SECRET")


class Form(_Base):
    def _b(self, **over):
        b = {"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18188",
             "api_key": "th-SECRET-TOKEN",
             "thunder": {"gpu_type": "h100", "num_gpus": 2, "vcpus": 16,
                         "bootstrap_template": "base", "reserve_gb": 30, "local_port": 18188,
                         "comfy_commit": SHA, "nodes": ["# c", "registry:y@2"]}}
        b.update(over)
        return b

    def test_url_readonly_not_disabled(self):
        html = admin._backend_form(self._b(), [])
        m = re.search(r'<input[^>]*name="url"[^>]*>', html)
        self.assertIsNotNone(m)
        self.assertIn("readonly", m.group(0))
        self.assertNotIn("disabled", m.group(0))
        plain = admin._backend_form({"name": "c", "type": "comfyui", "url": "http://x:1"}, [])
        self.assertNotIn("readonly", re.search(r'<input[^>]*name="url"[^>]*>', plain).group(0))

    def test_stored_block_is_rendered(self):
        html = admin._backend_form(self._b(), [])
        self.assertRegex(html, r'name="thunder_on" value="1" checked')
        self.assertRegex(html, r'<option value="h100" selected>')
        self.assertRegex(html, r'<option value="base" selected>')
        for v in ('value="2"', 'value="16"', 'value="30"', f'value="{SHA}"'):
            self.assertIn(v, html)
        self.assertIn("# c\nregistry:y@2</textarea>", html)
        self.assertIn("API key", html)                  # the hint names the token field

    def test_new_block_is_prefilled_with_the_default_nodes(self):
        html = admin._backend_form({"name": "c", "type": "comfyui", "url": "http://x:1"}, [])
        self.assertNotRegex(html, r'name="thunder_on" value="1" checked')
        self.assertIn("# defaults\nregistry:x@1.0\n</textarea>", html)
        self.assertIn(f'value="{SHA}"', html)
        # an existing block with an empty list is NOT refilled: empty = the default list
        # at bootstrap time (Ruling 12), and the operator chose that
        html = admin._backend_form(self._b(thunder={"gpu_type": "a6000", "vcpus": 8,
                                                     "nodes": []}), [])
        self.assertNotIn("# defaults", html)

    def test_token_never_rendered(self):
        html = admin._backend_form(self._b(), [])
        self.assertNotIn("th-SECRET-TOKEN", html)
        store.upsert_backend(self._b())
        self.live = [dict(self._b(), api_key_set=True, enabled=True, healthy=True, models=0,
                          source="ui")]
        self.live[0].pop("api_key")
        self.views = {"tc": _view(phase="ready")}
        page = self.page({"edit": "comfyui:tc"})
        self.assertNotIn("th-SECRET-TOKEN", page)

    def test_config_summary_carries_the_block(self):
        # The editor falls back to the live summary for a config backend; without the
        # block there, the first Save of a config Thunder backend would drop it.
        cfg = self._b()
        saved = (main.backends, main.config_backends)
        self.addCleanup(lambda: (setattr(main, "backends", saved[0]),
                                 setattr(main, "config_backends", saved[1])))
        main.backends = main.config_backends = [cfg]
        s = next(b for b in main.gateway_info()["backends"] if b["name"] == "tc")
        self.assertEqual(s["thunder"]["gpu_type"], "h100")
        self.assertNotIn("api_key", s)
        s["thunder"]["gpu_type"] = "x"                  # a copy, not the live dict
        self.assertEqual(cfg["thunder"]["gpu_type"], "h100")


class Panel(_Base):
    def setUp(self):
        super().setUp()
        self.live = [{"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18188",
                      "enabled": False, "healthy": False, "models": 0, "source": "ui",
                      "thunder": {"gpu_type": "a6000", "num_gpus": 1, "vcpus": 8,
                                  "local_port": 18188}}]

    def test_panel_buttons_are_post_actions(self):
        for p in ("/ui/thunder/start", "/ui/thunder/stop", "/ui/thunder/restart",
                  "/ui/thunder/forget"):
            self.assertIn(p, admin._POST_ACTIONS)
        self.views = {"tc": _view(phase="ready", unreconciled_uuids=["u-9"])}
        html = self.page()
        self.assertNotRegex(html, r'<a[^>]*href="/ui/thunder/')
        for p in ("stop", "restart", "forget"):
            self.assertRegex(html, rf'formaction="/ui/thunder/{p}\?name=tc"')
        self.assertNotIn("/ui/thunder/start", html)     # running: nothing to start
        self.views = {"tc": _view(phase="off")}
        html = self.page()
        self.assertNotRegex(html, r'<a[^>]*href="/ui/thunder/')
        self.assertRegex(html, r'formaction="/ui/thunder/start\?name=tc"')
        self.assertNotIn("/ui/thunder/stop", html)
        # a start in flight is still `off` until the create: Stop must be there to abort it
        self.views = {"tc": _view(phase="off", op="starting")}
        html = self.page()
        self.assertRegex(html, r'formaction="/ui/thunder/stop\?name=tc"')
        self.assertNotIn("/ui/thunder/start", html)
        # failed after the create: the instance bills — Stop stays, Start does not come back
        self.views = {"tc": _view(phase="failed", uuid="u1", failed_phase="bootstrapping")}
        html = self.page()
        self.assertRegex(html, r'formaction="/ui/thunder/stop\?name=tc"')
        self.assertNotIn("/ui/thunder/start", html)

    def test_stop_and_forget_ask_first(self):
        self.views = {"tc": _view(phase="ready", unreconciled_uuids=["u-9"])}
        html = self.page()
        for p in ("stop", "forget"):
            m = re.search(rf'<button[^>]*formaction="/ui/thunder/{p}[^"]*"[^>]*>', html)
            self.assertIsNotNone(m, p)
            self.assertIn("data-confirm=", m.group(0), p)

    def test_forget_only_with_unreconciled_uuids(self):
        self.views = {"tc": _view(phase="off")}
        self.assertNotIn("/ui/thunder/forget", self.page())

    def test_backends_tab_live_while_instance_runs(self):
        self.views = {"tc": _view(phase="ready")}
        self.assertIn('<main data-live="3">', self.page())

    def test_live_while_an_op_is_in_flight_in_phase_off(self):
        self.views = {"tc": _view(phase="off", op="starting")}
        self.assertIn('<main data-live="3">', self.page())

    def test_not_live_when_everything_is_off(self):
        self.views = {"tc": _view(phase="off")}
        self.assertIn("<main>", self.page())

    def test_card_shows_the_truth(self):
        log = [f"line {i}" for i in range(120)]
        self.views = {"tc": _view(
            phase="failed", failed_phase="bootstrapping", error="smoke <failed>",
            op="stopping", waiting_jobs=2, uptime_s=3720, disk_gb=140, cost_per_h=0.78,
            session_cost=0.81,
            snapshot={"id": "s1", "pending": "", "pending_name": "", "status": "READY",
                      "name": "aihub-tc-20260927t101010z", "gb": 90, "monthly": 6.3},
            log=log, bootstrap_unknown={"models/checkpoints/x.safetensors": 2 * 1024 ** 3},
            bootstrap_template_nodes=["ComfyUI-Manager"], bootstrap_incomplete=True,
            orphans=[{"uuid": "orph-1", "status": "RUNNING", "template": "base",
                      "index": "3", "created_at": "2026-09-27"}])}
        html = self.page()
        self.assertIn('data-k="thunder-tc"', html)
        self.assertIn("failed", html)
        self.assertIn("bootstrapping", html)
        self.assertIn("smoke &lt;failed&gt;", html)
        self.assertNotIn("smoke <failed>", html)
        self.assertIn("stopping", html)                 # the op in flight
        self.assertIn("2 job", html)                    # waiting jobs while draining
        self.assertIn("a6000", html)
        self.assertIn("0.78", html)
        self.assertIn("0.81", html)
        self.assertIn("aihub-tc-20260927t101010z", html)
        self.assertIn("6.30", html)
        self.assertIn("models/checkpoints/x.safetensors", html)
        self.assertIn("2.0 GB", html)
        self.assertIn("ComfyUI-Manager", html)
        self.assertIn("orph-1", html)
        self.assertIn("line 119", html)
        self.assertIn("line 70", html)
        self.assertNotIn("line 69\n", html)             # last 50 lines only
        self.assertRegex(html, r"<details[^>]*>\s*<summary>[^<]*log")

    def test_card_rows_are_keyed_and_add_no_script(self):
        self.views = {"tc": _view(phase="ready", orphans=[{"uuid": "o1", "status": "RUNNING"}],
                                  bootstrap_unknown={"a.bin": 1})}
        html = self.page()
        m = re.search(r'<main[^>]*>(.*)</main>', html, re.S)
        self.assertNotIn("<script", m.group(1))
        self.assertIn('data-k="thunder-tc-orphan-o1"', html)

    def test_message_is_shown_escaped(self):
        self.views = {"tc": _view()}
        html = self.page({"msg": "tc: start refused: <b>x</b>"})
        self.assertIn("start refused: &lt;b&gt;x&lt;/b&gt;", html)


class Actions(_Base):
    def setUp(self):
        super().setUp()
        saved = (main.api_key, main.users, main._users_by_key)
        main.api_key, main.users, main._users_by_key = "", [], {}
        self.addCleanup(lambda: (setattr(main, "api_key", saved[0]),
                                 setattr(main, "users", saved[1]),
                                 setattr(main, "_users_by_key", saved[2])))
        self.c = TestClient(main.app)

    def test_each_action_calls_the_controller_and_says_what_happened(self):
        for path, action in (("start", "start"), ("stop", "stop"), ("restart", "restart"),
                             ("forget", "forget_unreconciled")):
            r = self.c.post(f"/ui/thunder/{path}?name=a%26b", headers=SAME,
                            follow_redirects=False)
            self.assertEqual(r.status_code, 303, path)
            loc = urlparse(r.headers["location"])
            self.assertEqual(loc.path, "/ui/backends")
            self.assertIn(f"{action} requested", parse_qs(loc.query)["msg"][0])
            self.assertEqual(self.actions[-1], ("a&b", action))

    def test_name_as_form_field(self):
        r = self.c.post("/ui/thunder/start", data={"name": "tc"}, headers=SAME,
                        follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.actions[-1], ("tc", "start"))

    def test_get_runs_nothing(self):
        r = self.c.get("/ui/thunder/start?name=tc", headers=SAME, follow_redirects=False)
        self.assertEqual(r.status_code, 405)
        self.assertEqual(self.actions, [])


class MainWiring(unittest.TestCase):
    def test_default_nodes_are_bound_to_the_ops_file(self):
        self.assertIs(admin._thunder_default_nodes, main._thunder_default_nodes)
        with open(os.path.join(_here, "ops", "thunder-nodes.default.txt"), encoding="utf-8") as fh:
            self.assertEqual(main._thunder_default_nodes(), fh.read())


class FaultSources(unittest.TestCase):
    def test_lifecycle_and_sync_are_named(self):
        self.assertIn("lifecycle", admin._FAULT_SOURCE)
        self.assertIn("sync", admin._FAULT_SOURCE)


if __name__ == "__main__":
    unittest.main()
