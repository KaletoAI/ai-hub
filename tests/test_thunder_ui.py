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
  · the API token (the backend's `api_key`) is never rendered;
  · the model-sync panel (Task 13): rows keyed (the morph would otherwise rewrite every
    row per tick), "delete unknown" a POST whose confirm names what the TICKED boxes
    hold, a catalog the validator refuses is a 400 with the text as typed and nothing
    saved (a partial save drops the refused entry silently), the seed is copied into
    the setting once and never again, and a Thunder-only alias says why its schema is
    empty;
  · the LAN source (Task 15): its host key is pinned only by a POST that carries the
    fingerprint the operator saw (a GET, or a key fetched again in between, pins
    nothing), and the install command gives the share user a real login shell;
  · Task 16: the HF token is encrypted at rest and never rendered (blank keeps, the box
    clears), `modelsrc_host` is refused unless plain and a changed host names the stale
    pin, an instance up > 24 h puts a banner on its card AND the Dashboard, and `aihub-`
    snapshots no backend owns are listed with $/month (never deleted).

Run: python -m unittest tests.test_thunder_ui -v
"""
import asyncio
import json
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
    import thunderctl
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
        self.catalog = [{"match": {"class": "C", "value": "v"}, "paths": ["models/c/"]}]
        self.saved_catalogs = []
        self.synced = []
        self.deleted = []

        def save_cat(cat):
            errs = __import__("modelsync").validate_catalog(cat)
            if not errs:
                self.saved_catalogs.append(cat)
                self.catalog = cat
            return errs

        async def sync_now(name):
            self.synced.append(name)
            return "sync requested"

        async def delete_unknown(name, paths):
            self.deleted.append((name, list(paths)))
            return f"deleted {len(paths)} unknown file(s)"
        stubs = {"_apply_backends": lambda: None,
                 "_modelsync_catalog": lambda: self.catalog,
                 "_save_modelsync_catalog": save_cat,
                 "_thunder_sync_now": sync_now,
                 "_thunder_delete_unknown": delete_unknown,
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

    def test_url_hint_says_how_to_get_a_real_url_back(self):
        html = admin._backend_form(self._b(), [])
        self.assertIn("derived from the Thunder block", html)
        plain = admin._backend_form({"name": "c", "type": "comfyui", "url": "http://x:1"}, [])
        self.assertNotIn("derived from the Thunder block", plain)

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
        self.views = {"tc": _view(phase="ready", uuid="u1", unreconciled_uuids=["u-9"])}
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
        self.views = {"tc": _view(phase="ready", uuid="u1", unreconciled_uuids=["u-9"])}
        html = self.page()
        for p in ("stop", "forget"):
            m = re.search(rf'<button[^>]*formaction="/ui/thunder/{p}[^"]*"[^>]*>', html)
            self.assertIsNotNone(m, p)
            self.assertIn("data-confirm=", m.group(0), p)

    def test_start_asks_first_and_names_the_bill(self):
        # a Start rents a GPU by the hour: one stray click must not do that
        self.views = {"tc": _view(phase="off")}
        html = self.page()
        m = re.search(r'<button[^>]*formaction="/ui/thunder/start[^"]*"[^>]*>', html)
        self.assertIsNotNone(m)
        conf = re.search(r'data-confirm="([^"]*)"', m.group(0))
        self.assertIsNotNone(conf, m.group(0))
        self.assertIn("Start the Thunder instance tc?", conf.group(1))
        self.assertIn("bills per hour until you stop it", conf.group(1))

    def test_persist_error_is_on_the_card(self):
        self.views = {"tc": _view(phase="ready", uuid="u1",
                                  persist_error="12:00:00 OSError('disk <full>')")}
        html = self.page()
        self.assertIn('data-k="thunder-tc-persist"', html)
        self.assertIn("State not saved", html)
        self.assertIn("disk &lt;full&gt;", html)
        self.views = {"tc": _view(phase="ready", uuid="u1", persist_error="")}
        self.assertNotIn("State not saved", self.page())

    def test_infinite_numbers_do_not_break_the_tab(self):
        # int(inf) is an OverflowError, not a ValueError
        self.assertEqual(admin._nbytes(float("inf")), 0)
        self.assertEqual(admin._hms(float("inf")), "0m 00s")
        self.views = {"tc": _view(phase="ready", uuid="u1", uptime_s=float("inf"),
                                  bootstrap_unknown={"a.bin": float("inf")})}
        self.assertIn("a.bin", self.page())

    def test_forget_only_with_unreconciled_uuids(self):
        self.views = {"tc": _view(phase="off")}
        self.assertNotIn("/ui/thunder/forget", self.page())

    def test_backends_tab_live_while_instance_runs(self):
        self.views = {"tc": _view(phase="ready")}
        self.assertIn('<main data-live="3">', self.page())

    def test_live_while_an_op_is_in_flight_in_phase_off(self):
        self.views = {"tc": _view(phase="off", op="starting")}
        self.assertIn('<main data-live="3">', self.page())

    def test_editor_is_never_live(self):
        # The morph would reset the visibility the type select's JS set (switch "+ New"
        # to comfyui and 3 s later its panes vanish) — only the plain list is live.
        self.views = {"tc": _view(phase="ready")}
        store.upsert_backend({"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18188",
                              "thunder": {"gpu_type": "a6000", "vcpus": 8}})
        for qp in ({"edit": "comfyui:tc"}, {"new": "1"}, {"host": "thunder-tc"}):
            html = self.page(qp)
            self.assertIn("<main>", html, qp)
            self.assertNotIn("<main data-live", html, qp)
        self.assertIn('<main data-live="3">', self.page())

    def test_refused_save_is_never_live(self):
        self.views = {"tc": _view(phase="ready")}
        r = self.save(_thunder_form(name="x", thunder_vcpus=""))
        self.assertEqual(r.status_code, 400)
        self.assertNotIn("<main data-live", r.body.decode())

    def test_buttons_follow_the_controller_refusals(self):
        # op in flight: no Start, no Restart (the controller answers "already …")
        self.views = {"tc": _view(phase="failed", op="stopping")}
        html = self.page()
        self.assertNotIn("/ui/thunder/start", html)
        self.assertNotIn("/ui/thunder/restart", html)
        self.assertIn("/ui/thunder/stop", html)
        # failed before the create (no instance): Start yes, Restart no (needs the uuid)
        self.views = {"tc": _view(phase="failed", failed_phase="creating")}
        html = self.page()
        self.assertIn("/ui/thunder/start", html)
        self.assertNotIn("/ui/thunder/restart", html)
        self.views = {"tc": _view(phase="ready", uuid="u1")}
        self.assertIn("/ui/thunder/restart", self.page())

    def test_odd_numbers_in_the_view_do_not_break_the_tab(self):
        self.views = {"tc": _view(phase="draining", waiting_jobs="?", uptime_s="x",
                                  cost_per_h="n/a", bootstrap_unknown={"a.bin": "big"})}
        html = self.page()
        self.assertIn("a.bin", html)
        self.assertIn("0.0 GB", html)

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
        self.assertIn('<span class="badge bad">failed (bootstrapping)</span>', html)
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
        self.assertIn("line 0\n", html)                 # the whole ring (200 lines)
        self.assertRegex(html, r"<details[^>]*>\s*<summary>[^<]*log")

    def test_card_log_shows_the_whole_ring(self):
        log = [f"line {i}" for i in range(thunderctl._LOG_MAX + 30)]
        self.views = {"tc": _view(log=log)}
        html = self.page()
        self.assertEqual(admin._THUNDER_LOG_LINES, thunderctl._LOG_MAX)
        self.assertIn(f"log (last {thunderctl._LOG_MAX} lines)", html)
        self.assertIn("line 30\n", html)
        self.assertNotIn("line 29\n", html)

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


GiB = 10 ** 9


def _plan(**over) -> dict:
    """A controller `view()["plan"]` (thunderctl.Controller._plan_view) with one alias of
    each status the panel names."""
    f = lambda path, size, present, node=None, cls=None: {          # noqa: E731
        "path": path, "size": size, "node": node, "cls": cls, "present": present}
    p = {"aliases": {
        "Img": {"ready": True, "need_bytes": 2 * GiB, "have_bytes": 2 * GiB, "missing": 0,
                "blocked": [], "hints": [], "held": 0, "selectable": [],
                "files": [f("models/vae/v.safetensors", 2 * GiB, True, "3", "VAELoader")]},
        "Mesh": {"ready": False, "need_bytes": 10 * GiB, "have_bytes": 4 * GiB, "missing": 1,
                 "blocked": [], "hints": ["Hy3D21VAELoader=x is not synced — add a catalog entry "
                                          "if it needs weights"],
                 "held": 0, "selectable": ["1.unet_name"],
                 "files": [f("models/diffusion_models/u.gguf", 6 * GiB, False, "1", "UnetLoaderGGUF"),
                           f("models/vae/w.safetensors", 4 * GiB, True, "2", "VAELoader")]},
        "Lan": {"ready": False, "need_bytes": 1 * GiB, "have_bytes": 0, "missing": 1,
                "blocked": ["waiting for LAN source (not configured): models/x/lan.bin"],
                "hints": [], "held": 0, "selectable": [],
                "files": [f("models/x/lan.bin", 1 * GiB, False)]},
        "Bad": {"ready": False, "need_bytes": 0, "have_bytes": 0, "missing": 0,
                "blocked": ["unknown hub model org/<repo> — add a catalog entry"],
                "hints": [], "held": 2, "selectable": [], "files": [], "gated_only": True}},
        "fetch": [], "prune": ["models/old/a.bin", "models/old/b.bin"],
        "prune_sizes": [["models/old/a.bin", 3 * GiB], ["models/old/b.bin", None]],
        "held": [["models/h/one.bin", GiB, "Bad"], ["models/h/two.bin", 2 * GiB, "Bad"]],
        "unknown": [["models/checkpoints/stray.ckpt", 5 * GiB], ["models/u/<odd>&.bin", GiB]],
        "need_total": 13 * GiB, "have_total": 6 * GiB}
    p.update(over)
    return p


def _tx(path, done, total, rate, eta, source="url"):
    return {"file": path, "source": source, "bytes": done, "total": total, "rate": rate,
            "eta": eta, "attempt": 1}


class SyncPanel(_Base):
    """Task 13: the model-sync half of the Thunder card, its actions and the catalog."""

    def setUp(self):
        super().setUp()
        self.live = [{"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18188",
                      "enabled": True, "healthy": True, "models": 0, "source": "ui",
                      "thunder": {"gpu_type": "a6000", "num_gpus": 1, "vcpus": 8}}]
        self.views = {"tc": _view(phase="ready", uuid="u1", plan=_plan(), transfers=[
            _tx("models/diffusion_models/u.gguf", 3 * GiB, 6 * GiB, 50e6, 60)])}

    def main_html(self, qp=None) -> str:
        html = self.page(qp)
        return re.search(r"<main[^>]*>(.*)</main>", html, re.S).group(1)

    def test_sync_table_rows_keyed(self):
        html = self.main_html()
        for a in ("Img", "Mesh", "Lan", "Bad"):
            self.assertIn(f'data-k="ms-{a}"', html)
        self.assertIn('data-k="tx-models/diffusion_models/u.gguf"', html)
        self.assertNotIn("<script", html)             # the live-page invariant
        # per alias: need / have / missing, and the status in words
        row = lambda a: re.search(rf'<tr data-k="ms-{a}"[^>]*>(.*?)</tr>', html, re.S).group(1)  # noqa: E731
        self.assertIn("ready", row("Img"))
        self.assertIn("10.0", row("Mesh"))
        self.assertIn("4.0", row("Mesh"))
        self.assertIn("6.0", row("Mesh"))
        self.assertIn("syncing 40 %", row("Mesh"))
        self.assertIn("waiting for LAN source", row("Lan"))
        self.assertIn("blocked: unknown hub model org/&lt;repo&gt;", row("Bad"))
        self.assertNotIn("org/<repo>", html)
        # hints, selectable fields, held count and the Thunder-only gate note
        self.assertIn("Hy3D21VAELoader=x is not synced", html)
        self.assertIn("1.unet_name", row("Mesh"))
        self.assertIn("schema", row("Bad"))           # empty until the sync finishes
        # expandable per-file rows: path · size · node <id> (<cls>) · present
        files = re.search(r'data-k="ms-Mesh-files".*?</details>', html, re.S).group(0)
        self.assertIn("<details", files)
        self.assertIn("models/diffusion_models/u.gguf", files)
        self.assertIn("node 1 (UnetLoaderGGUF)", files)
        self.assertIn("6.0 GB", files)
        # transfers: file, source, rate, ETA
        tx = re.search(r'<tr data-k="tx-models/diffusion_models/u.gguf"[^>]*>(.*?)</tr>',
                       html, re.S).group(1)
        self.assertIn("url", tx)
        self.assertIn("50.0 MB/s", tx)
        self.assertIn("1m 00s", tx)
        # held files, the prune preview and the totals
        self.assertIn("held (alias blocked)", html)
        self.assertIn("models/h/two.bin", html)
        self.assertRegex(html, r"deleted at stop: 2 files, 3\.0 GB")
        self.assertIn("6.0 of 13.0 GB", html)

    def test_sync_rows_survive_odd_views(self):
        for plan in (None, {}, _plan(aliases={"x": {"files": "junk"}}, unknown="junk",
                                     held=[["only-path"]], prune_sizes=None)):
            self.views = {"tc": _view(phase="ready", plan=plan, transfers={"a": 1})}
            self.assertIn('data-k="thunder-tc"', self.page())

    def test_sync_button_is_a_post_and_only_while_running(self):
        self.assertIn("/ui/thunder/sync", admin._POST_ACTIONS)
        html = self.page()
        self.assertRegex(html, r'formaction="/ui/thunder/sync\?name=tc"')
        self.views = {"tc": _view(phase="off", plan=_plan())}
        self.assertNotIn("/ui/thunder/sync", self.page())

    def test_delete_unknown_is_post_with_confirm(self):
        self.assertIn("/ui/thunder/delete-unknown", admin._POST_ACTIONS)
        html = self.main_html()
        form = re.search(r'<form[^>]*action="/ui/thunder/delete-unknown"[^>]*>.*?</form>',
                         html, re.S).group(0)
        self.assertIn('method="post"', form)
        self.assertIn('name="name" value="tc"', form)
        self.assertIn('name="path" value="models/checkpoints/stray.ckpt"', form)
        self.assertIn('value="models/u/&lt;odd&gt;&amp;.bin"', form)
        btn = re.search(r"<button[^>]*data-confirm=\"([^\"]*)\"[^>]*>", form)
        self.assertIsNotNone(btn)
        self.assertIn("2 unknown files", btn.group(1))
        self.assertIn("6.0 GB", btn.group(1))
        # nothing unknown → no form; not running → no form (the controller refuses)
        self.views = {"tc": _view(phase="ready", plan=_plan(unknown=[]))}
        self.assertNotIn("/ui/thunder/delete-unknown", self.page())
        self.views = {"tc": _view(phase="off", plan=_plan())}
        self.assertNotIn("/ui/thunder/delete-unknown", self.page())

    def test_confirm_js_counts_the_selection(self):
        # the confirm text follows the ticked boxes (data-confirm-sum), in the ONE
        # confirm handler every page already carries — no script in the live <main>
        self.assertIn("data-confirm-sum", admin._CONFIRM_JS)
        html = self.main_html()
        self.assertRegex(html, r'data-confirm-sum="[^"]*\{n\}[^"]*\{gb\}')
        self.assertIn('data-bytes="5000000000"', html)

    def test_confirm_js_sum_runs(self):
        """Run the confirm handler in node against a fake DOM: the text names the TICKED
        boxes, nothing ticked submits nothing, a plain data-confirm is unchanged."""
        import shutil
        import subprocess
        if not shutil.which("node"):
            self.skipTest("node not installed")
        src = admin._CONFIRM_JS.split("<script>", 1)[1].split("</script>", 1)[0]
        prog = ("var h,asked=[],alerts=0;var document={addEventListener:function(t,f){h=f;},"
                "getElementById:function(){return null;}};"
                "var window={confirm:function(m){asked.push(m);return false;},"
                "alert:function(){alerts++;}};"
                + src.split("window.gwPost", 1)[0] +
                "function box(c,b){return {checked:c,getAttribute:function(){return b;}};}"
                "function run(t,boxes){var prev=0;var btn={getAttribute:function(k){"
                "return k==='data-confirm'?'static':(k==='data-confirm-sum'?t:null);},"
                "form:{querySelectorAll:function(){return boxes;}}};"
                "var ev={target:{closest:function(){return btn;}},prevented:false,"
                "preventDefault:function(){this.prevented=true;},stopPropagation:function(){}};"
                "h(ev);return ev.prevented;}"
                "var r1=run('Delete {n} ({gb} GB) {n}',[box(true,'1500000000'),box(false,'9'),"
                "box(true,'500000000')]);"
                "var r2=run('Delete {n}',[box(false,'1')]);"
                "var r3=run(null,[]);"
                "console.log(JSON.stringify({asked:asked,alerts:alerts,r:[r1,r2,r3]}));")
        p = subprocess.run(["node", "-e", prog], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        out = json.loads(p.stdout)
        self.assertEqual(out["asked"], ["Delete 2 (2.0 GB) 2", "static"])
        self.assertEqual(out["alerts"], 1)                 # nothing ticked
        self.assertEqual(out["r"], [True, True, True])     # every confirm answered "no"

    def test_catalog_editor_in_details_with_the_stored_catalog(self):
        html = self.main_html()
        det = re.search(r'<details[^>]*data-k="thunder-catalog"[^>]*>.*?</details>', html, re.S)
        self.assertIsNotNone(det)
        self.assertIn('action="/ui/thunder/catalog"', det.group(0))
        self.assertIn('name="catalog"', det.group(0))
        self.assertIn("models/c/", det.group(0))


class SyncActions(Actions):
    def setUp(self):
        super().setUp()
        self.views = {"tc": _view(phase="ready", uuid="u1", plan=_plan())}
        self.live = [{"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18188",
                      "enabled": True, "healthy": True, "models": 0, "source": "ui",
                      "thunder": {"gpu_type": "a6000"}}]

    def test_sync_calls_the_controller(self):
        r = self.c.post("/ui/thunder/sync?name=tc", headers=SAME, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.synced, ["tc"])
        self.assertIn("sync requested", parse_qs(urlparse(r.headers["location"]).query)["msg"][0])

    def test_delete_unknown_posts_every_ticked_path(self):
        r = self.c.post("/ui/thunder/delete-unknown", headers=SAME, follow_redirects=False,
                        data={"name": "tc", "path": ["models/a.bin", "models/b&c.bin"]})
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.deleted, [("tc", ["models/a.bin", "models/b&c.bin"])])
        self.assertIn("deleted 2", parse_qs(urlparse(r.headers["location"]).query)["msg"][0])

    def test_delete_unknown_without_selection_deletes_nothing(self):
        r = self.c.post("/ui/thunder/delete-unknown", headers=SAME, follow_redirects=False,
                        data={"name": "tc"})
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.deleted, [])

    def test_get_runs_nothing(self):
        for p in ("sync?name=tc", "delete-unknown?name=tc&path=models/a.bin", "catalog"):
            r = self.c.get(f"/ui/thunder/{p}", headers=SAME, follow_redirects=False)
            self.assertEqual(r.status_code, 405, p)
        self.assertEqual((self.synced, self.deleted, self.saved_catalogs), ([], [], []))

    def test_catalog_invalid_json_is_400_and_not_saved(self):
        typed = '[{"match": {"class": "X"}, "paths": ["models/<x>/"]'
        r = self.c.post("/ui/thunder/catalog", headers=SAME, data={"catalog": typed})
        self.assertEqual(r.status_code, 400)
        body = r.text
        self.assertIn("models/&lt;x&gt;/", body)      # the textarea as typed …
        self.assertIn("not valid JSON", body)         # … and why
        self.assertNotIn("<main data-live", body)     # a refused form is never live
        self.assertRegex(body, r'<details[^>]*data-k="thunder-catalog"[^>]*\bopen\b')
        self.assertEqual(self.saved_catalogs, [])
        # valid JSON the validator refuses: every line of its answer, nothing saved
        typed = json.dumps([{"match": {"alias": "a"}, "paths": ["models/"]}, "junk"])
        r = self.c.post("/ui/thunder/catalog", headers=SAME, data={"catalog": typed})
        self.assertEqual(r.status_code, 400)
        self.assertIn("is a whole root", r.text)
        self.assertIn("entry 2: not an object", r.text)
        self.assertEqual(self.saved_catalogs, [])

    def test_catalog_saved_and_rendered(self):
        cat = [{"match": {"class": "Trellis2LoadModel", "value": "org/repo"},
                "paths": ["models/org/repo/"]}]
        r = self.c.post("/ui/thunder/catalog", headers=SAME, follow_redirects=False,
                        data={"catalog": json.dumps(cat)})
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.saved_catalogs, [cat])
        loc = urlparse(r.headers["location"])
        self.assertEqual(loc.path, "/ui/backends")
        self.assertIn("catalog saved", parse_qs(loc.query)["msg"][0])
        page = self.c.get("/ui/backends", headers=SAME).text
        self.assertIn("models/org/repo/", page)


class MainWiring(unittest.TestCase):
    def test_default_nodes_are_bound_to_the_ops_file(self):
        self.assertIs(admin._thunder_default_nodes, main._thunder_default_nodes)
        with open(os.path.join(_here, "ops", "thunder-nodes.default.txt"), encoding="utf-8") as fh:
            self.assertEqual(main._thunder_default_nodes(), fh.read())


class CatalogWiring(unittest.TestCase):
    """main's side of the catalog: seeded once, then the setting is authoritative, and a
    save refused by the validator writes nothing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        saved = (store._DB_PATH, store._active, store._MASTER_KEY)
        self.addCleanup(lambda: (setattr(store, "_DB_PATH", saved[0]),
                                 setattr(store, "_active", saved[1]),
                                 setattr(store, "_MASTER_KEY", saved[2])))
        store._MASTER_KEY = os.urandom(32)
        store.init(os.path.join(self.tmp.name, "store.db"))

    def test_bound(self):
        self.assertIs(admin._modelsync_catalog, main._modelsync_catalog)
        self.assertIs(admin._save_modelsync_catalog, main.save_modelsync_catalog)
        self.assertIs(admin._thunder_delete_unknown, main.thunder_delete_unknown)
        self.assertIs(admin._thunder_sync_now, main.thunder_sync_now)

    def test_seed_copied_on_first_read_then_the_setting_rules(self):
        import modelsync
        self.assertIsNone(store.get_setting("modelsync_catalog"))
        self.assertEqual(main._modelsync_catalog(), modelsync.DEFAULT_CATALOG)
        self.assertEqual(store.get_setting("modelsync_catalog"), modelsync.DEFAULT_CATALOG)
        self.assertEqual(main.save_modelsync_catalog([]), [])
        self.assertEqual(main._modelsync_catalog(), [])          # an emptied catalog stays empty
        self.assertEqual(store.get_setting("modelsync_catalog"), [])

    def test_refused_save_writes_nothing(self):
        cat = [{"match": {"alias": "a"}, "paths": []}]
        self.assertEqual(main.save_modelsync_catalog(cat), [])
        errs = main.save_modelsync_catalog([{"match": {"alias": "a"}, "paths": ["hf-cache/token"]}])
        self.assertTrue(errs)
        self.assertEqual(store.get_setting("modelsync_catalog"), cat)


class _FakeCtl:
    def __init__(self, ready=(), unknown_ok=True, phase="ready"):
        self.ready, self.unknown_ok, self.phase = set(ready), unknown_ok, phase
        self.deleted, self.synced = [], 0

    def is_alias_ready(self, alias):
        return alias in self.ready

    def view(self):
        return {"phase": self.phase, "plan": {"aliases": {
            a: {"ready": a in self.ready} for a in ("solo", "mixed", "done")}}}

    async def delete_unknown(self, paths):
        await asyncio.sleep(0)
        if not self.unknown_ok:
            raise ValueError("not in the unknown list (needed, synced or gone): " + paths[0])
        self.deleted.append(list(paths))
        return len(paths)

    async def sync_now(self):
        if self.phase not in ("syncing", "ready"):
            raise RuntimeError(f"no running instance ({self.phase})")
        await asyncio.sleep(0)
        self.synced += 1


class MainSyncWiring(CatalogWiring):
    """main's side of the panel: the Thunder-only gate note, delete and sync answers."""

    def setUp(self):
        super().setUp()
        saved = dict(main.thunder_controllers), main.image_models
        self.addCleanup(lambda: (main.thunder_controllers.clear(),
                                 main.thunder_controllers.update(saved[0]),
                                 setattr(main, "image_models", saved[1])))
        main.thunder_controllers.clear()
        main.image_models = {}
        self.ctl = main.thunder_controllers["tc"] = _FakeCtl(ready={"done"})
        store.upsert("solo", [{"backend": "tc", "workflow_json": {}}])
        store.upsert("mixed", [{"backend": "tc", "workflow_json": {}},
                               {"backend": "k12", "workflow_json": {}}])
        store.upsert("done", [{"backend": "tc", "workflow_json": {}}])

    def test_gated_only_names_aliases_no_other_backend_serves(self):
        rows = main.thunder_view("tc")["plan"]["aliases"]
        self.assertEqual({a: r["gated_only"] for a, r in rows.items()},
                         {"solo": True, "mixed": False, "done": False})
        # a second Thunder backend that has not synced it either is no way out
        main.thunder_controllers["k12"] = _FakeCtl()
        self.assertTrue(main.thunder_view("tc")["plan"]["aliases"]["mixed"]["gated_only"])
        self.assertIsNone(main.thunder_view("nope"))

    def test_delete_unknown_answers(self):
        self.assertIn("deleted 2 unknown files",
                      asyncio.run(main.thunder_delete_unknown("tc", ["models/a", "models/b"])))
        self.assertEqual(self.ctl.deleted, [["models/a", "models/b"]])
        self.ctl.unknown_ok = False
        self.assertIn("delete refused: not in the unknown list",
                      asyncio.run(main.thunder_delete_unknown("tc", ["models/a"])))
        self.assertIn("nothing deleted", asyncio.run(main.thunder_delete_unknown("tc", [])))
        self.assertIn("unknown Thunder backend",
                      asyncio.run(main.thunder_delete_unknown("x", ["models/a"])))

    def test_sync_now_answers(self):
        self.assertIn("sync", asyncio.run(main.thunder_sync_now("tc")))
        self.ctl.phase = "off"
        self.assertIn("sync refused: no running instance (off)",
                      asyncio.run(main.thunder_sync_now("tc")))


class FaultSources(unittest.TestCase):
    def test_lifecycle_and_sync_are_named(self):
        self.assertIn("lifecycle", admin._FAULT_SOURCE)
        self.assertIn("sync", admin._FAULT_SOURCE)



_ED_B64 = "AAAAC3NzaC1lZDI1NTE5AAAAIHm4E0tb6VPU5qn5zKm6c1tJ4HQ1Pdu6Wf7k6o7V3tJ2"


class LanSourcePanel(Actions):
    """Task 15: the LAN model source's block and its host-key pin. A pin written by a GET
    (a prefetch), or one that trusts a key the operator never saw, is a man in the
    middle of every model transfer; an install command with a nologin shell runs
    nothing on the share host and every list then fails as "unreachable"."""

    def setUp(self):
        super().setUp()
        import thunderctl
        self.live = [{"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18188",
                      "enabled": True, "healthy": True, "models": 0, "source": "ui",
                      "thunder": {"gpu_type": "a6000"}}]
        self.views = {"tc": _view(phase="ready", uuid="u1")}
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.scans = []

        async def ssh(argv, stdin=None, timeout=60):
            self.scans.append(list(argv))
            return (0, f"192.168.8.24 ssh-ed25519 {_ED_B64}\n".encode(), b"")
        self.lan = thunderctl.LanSource(self.d.name, host=lambda: "modelsrc@192.168.8.24",
                                        ssh=ssh)
        with open(self.lan.key_path + ".pub", "w") as f:
            f.write("ssh-ed25519 AAAAgatewaykey ai-hub\n")
        saved = (main.jobs_cfg, main._modelsrc_obj)
        self.addCleanup(lambda: (setattr(main, "jobs_cfg", saved[0]),
                                 setattr(main, "_modelsrc_obj", saved[1])))
        main.jobs_cfg = dict(main.jobs_cfg, store_path=os.path.join(self.d.name, "store.db"))
        main._modelsrc_obj = self.lan
        for k, v in {"_thunder_modelsrc_view": main.modelsrc_view,
                     "_thunder_modelsrc_scan": main.modelsrc_scan,
                     "_thunder_modelsrc_pin": main.modelsrc_pin}.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)
        self.fp = thunderctl.host_key_fingerprint(_ED_B64)

    def block(self) -> str:
        m = re.search(r'<div class="tcard" data-k="thunder-modelsrc">.*?</div></div>',
                      self.page(), re.S)
        self.assertIsNotNone(m)
        return m.group(0)

    def post(self, path, **data):
        r = self.c.post(path, data=data, headers=SAME, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        return parse_qs(urlparse(r.headers["location"]).query)["msg"][0]

    def test_not_set_up_shows_key_and_install_command(self):
        b = self.block()
        self.assertIn("LAN source: not set up", b)
        self.assertIn("ssh-ed25519 AAAAgatewaykey ai-hub", b)
        self.assertIn("--shell /bin/bash modelsrc", b)
        self.assertNotIn("nologin", b)
        self.assertIn("command=&quot;/usr/local/bin/modelsrc-serve&quot;,no-port-forwarding,"
                      "no-X11-forwarding,no-agent-forwarding,no-pty ssh-ed25519 AAAAgatewaykey", b)
        self.assertRegex(b, r'formaction="/ui/thunder/modelsrc-scan"')
        self.assertNotIn("modelsrc-pin", b)               # nothing fetched: nothing to confirm
        self.assertNotRegex(b, r'<a[^>]*href="/ui/thunder/')

    def test_pin_flow_is_post_and_writes_known_hosts(self):
        for p in ("/ui/thunder/modelsrc-scan", "/ui/thunder/modelsrc-pin"):
            self.assertIn(p, admin._POST_ACTIONS)
            r = self.c.get(p + "?fp=" + self.fp, headers=SAME, follow_redirects=False)
            self.assertEqual(r.status_code, 405, p)
        self.assertEqual(self.scans, [])
        msg = self.post("/ui/thunder/modelsrc-scan")
        self.assertIn(self.fp, msg)
        self.assertEqual(self.scans, [["ssh-keyscan", "-t", "ed25519", "--", "192.168.8.24"]])
        self.assertFalse(os.path.exists(self.lan.known_hosts_path))     # not trusted yet
        b = self.block()
        self.assertIn(self.fp, b)
        btn = re.search(r'<button[^>]*formaction="/ui/thunder/modelsrc-pin\?fp=([^"]+)"[^>]*>',
                        b)
        self.assertIsNotNone(btn)
        self.assertIn(f"data-confirm=\"Pin modelsrc@192.168.8.24&#x27;s host key {self.fp}?",
                      btn.group(0))
        # a confirmation of another key pins nothing
        self.assertIn("not pinned", self.post("/ui/thunder/modelsrc-pin", fp="SHA256:other"))
        self.assertFalse(os.path.exists(self.lan.known_hosts_path))
        from urllib.parse import unquote
        msg = self.post("/ui/thunder/modelsrc-pin", fp=unquote(btn.group(1)))
        self.assertIn("pinned", msg)
        with open(self.lan.known_hosts_path) as f:
            self.assertEqual(f.read(), f"192.168.8.24 ssh-ed25519 {_ED_B64}\n")
        self.assertEqual(os.stat(self.lan.known_hosts_path).st_mode & 0o777, 0o600)
        b = self.block()
        self.assertNotIn("not set up", b)
        self.assertIn(f"host key <code>{self.fp}</code> pinned", b)

    def test_no_lan_block_without_thunder_backends(self):
        self.views = {}
        self.assertNotIn("thunder-modelsrc", self.page())


class LanSourceWiring(CatalogWiring):
    def test_bound_and_deps(self):
        import thunderctl
        self.assertIs(admin._thunder_modelsrc_view, main.modelsrc_view)
        self.assertIs(admin._thunder_modelsrc_scan, main.modelsrc_scan)
        self.assertIs(admin._thunder_modelsrc_pin, main.modelsrc_pin)
        # the host rule is main's ship-target rule (plain [user@]host characters)
        self.assertEqual(thunderctl._SRC_HOST_RE.pattern, main._VOICE_HOST_RE.pattern)
        self.assertEqual(main._modelsrc_host(), "modelsrc@192.168.8.24")
        store.set_settings({"modelsrc_host": "src@10.0.0.2"})
        self.assertEqual(main._modelsrc_host(), "src@10.0.0.2")
        saved = main.jobs_cfg
        self.addCleanup(setattr, main, "jobs_cfg", saved)
        main.jobs_cfg = dict(saved, store_path=os.path.join(self.tmp.name, "store.db"))
        deps = main._thunder_deps()
        self.assertIs(deps.lan, main.modelsrc())
        self.assertEqual(deps.lan.datadir, self.tmp.name)
        self.assertEqual(deps.lan.host(), "src@10.0.0.2")
        self.assertEqual(deps.source_index(), {})          # not pinned: no source
        self.assertEqual(deps.lan.problem(), "not configured")



class HfToken(Actions):
    """Task 16: the HF token is a cloud secret like a backend key — encrypted at rest,
    never rendered back, blank keeps it and only the box clears it. Rendered, it sits in
    every Backends page a shoulder or a screenshot sees; a blank Save that CLEARED it
    silently turns every gated HF download into a 401 at the next sync."""

    def setUp(self):
        super().setUp()
        for k, v in {"_save_hf_token": main.save_hf_token,
                     "_hf_token_set": main.hf_token_set}.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)
        self.views = {"tc": _view()}

    def post(self, status=303, **data):
        r = self.c.post("/ui/thunder/hf-token", data=data, headers=SAME, follow_redirects=False)
        self.assertEqual(r.status_code, status, r.text[:300])
        return r

    def raw(self):
        import sqlite3
        with sqlite3.connect(store._DB_PATH) as c:
            row = c.execute("SELECT value_json FROM settings WHERE key='hf_token'").fetchone()
        return None if row is None else json.loads(row[0])

    def test_bound(self):
        self.assertIs(admin._save_hf_token, main.save_hf_token)
        self.assertIs(admin._hf_token_set, main.hf_token_set)
        self.assertIn("/ui/thunder/hf-token", admin._POST_ACTIONS)
        r = self.c.get("/ui/thunder/hf-token", headers=SAME, follow_redirects=False)
        self.assertEqual(r.status_code, 405)

    def test_hf_token_encrypted_at_rest(self):
        self.assertIn("hf_token", store._SECRET_SETTINGS)
        self.post(hf_token="hf_SecretValue123")
        self.assertTrue(self.raw().startswith("enc:"), self.raw())
        self.assertNotIn("SecretValue", self.raw())
        self.assertEqual(store.get_setting("hf_token"), "hf_SecretValue123")
        self.assertEqual(main._thunder_hf_token(), "hf_SecretValue123")   # what Deps reads
        # a row written before the setting became a secret stays readable (passthrough)
        import sqlite3
        with sqlite3.connect(store._DB_PATH) as c:
            c.execute("UPDATE settings SET value_json=? WHERE key='hf_token'",
                      (json.dumps("hf_legacyPlain"),))
        self.assertEqual(main._thunder_hf_token(), "hf_legacyPlain")

    def test_hf_token_never_rendered(self):
        page = self.page()
        self.assertIn('name="hf_token"', page)
        self.assertRegex(page, r'<input type="password" name="hf_token" value=""')
        self.assertIn('name="hf_token_clear"', page)
        self.assertIn("not set", page)
        self.post(hf_token="hf_SecretValue123")
        page = self.page()
        self.assertNotIn("hf_SecretValue123", page)
        self.assertNotIn(self.raw(), page)                  # nor its ciphertext
        self.assertRegex(page, r'<input type="password" name="hf_token" value=""')
        self.assertIn("set — blank keeps it", page)
        # blank keeps it
        r = self.post(hf_token="")
        self.assertIn("unchanged", parse_qs(urlparse(r.headers["location"]).query)["msg"][0])
        self.assertEqual(main._thunder_hf_token(), "hf_SecretValue123")
        # a new value replaces it
        self.post(hf_token="hf_Other456")
        self.assertEqual(main._thunder_hf_token(), "hf_Other456")
        # clear removes it (a typed value next to a ticked box: the value wins, as for keys)
        r = self.post(hf_token="", hf_token_clear="1")
        self.assertIn("removed", parse_qs(urlparse(r.headers["location"]).query)["msg"][0])
        self.assertEqual(main._thunder_hf_token(), "")
        self.assertFalse(main.hf_token_set())

    def test_refused_token_is_400_and_never_echoed(self):
        self.post(hf_token="hf_Good1")
        r = self.post(status=400, hf_token="hf_bad value\nX-Injected: 1")
        self.assertIn("not saved", r.text)
        self.assertNotIn("hf_bad value", r.text)
        self.assertNotIn("X-Injected", r.text)
        self.assertEqual(main._thunder_hf_token(), "hf_Good1")      # nothing written
        # what the transfer would withhold is not saveable either (one rule, both ends)
        import thunderctl
        for bad in ('hf_a"b', "hf_a\\b", "hf_ä", "h" * 513):
            with self.subTest(bad=bad):
                self.assertFalse(thunderctl.hf_token_ok(bad))
                self.post(status=400, hf_token=bad)
                self.assertEqual(main._thunder_hf_token(), "hf_Good1")
        self.assertTrue(thunderctl.hf_token_ok("hf_Good1"))
        self.assertRegex(r.text, r'<details class="optblock" data-k="thunder-catalog" open>')


class ModelsrcHostField(Actions):
    """Task 16: `modelsrc_host` is set in the console, next to the LAN block. A value
    that is no plain [user@]host is refused with the form as typed (it would reach an ssh
    argv); a changed host drops the old share's listing and says the pin is for the old
    host — otherwise every list fails as "unreachable" and points at the network."""

    def setUp(self):
        super().setUp()
        self.live = [{"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18188",
                      "enabled": True, "healthy": True, "models": 0, "source": "ui",
                      "thunder": {"gpu_type": "a6000"}}]
        self.views = {"tc": _view(phase="ready", uuid="u1")}
        saved = (main.jobs_cfg, main._modelsrc_obj)
        self.addCleanup(lambda: (setattr(main, "jobs_cfg", saved[0]),
                                 setattr(main, "_modelsrc_obj", saved[1])))
        main.jobs_cfg = dict(main.jobs_cfg, store_path=os.path.join(self.tmp.name, "store.db"))
        main._modelsrc_obj = None
        for k, v in {"_thunder_modelsrc_view": main.modelsrc_view,
                     "_save_modelsrc_host": main.save_modelsrc_host}.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)

    def post(self, status=303, **data):
        r = self.c.post("/ui/thunder/modelsrc-host", data=data, headers=SAME,
                        follow_redirects=False)
        self.assertEqual(r.status_code, status, r.text[:300])
        return r

    def test_bound_and_post_only(self):
        self.assertIs(admin._save_modelsrc_host, main.save_modelsrc_host)
        self.assertIn("/ui/thunder/modelsrc-host", admin._POST_ACTIONS)
        r = self.c.get("/ui/thunder/modelsrc-host?modelsrc_host=x", headers=SAME,
                       follow_redirects=False)
        self.assertEqual(r.status_code, 405)

    def test_field_shows_the_current_host(self):
        page = self.page()
        self.assertRegex(page, r'<form method="post" action="/ui/thunder/modelsrc-host"')
        self.assertRegex(page, r'name="modelsrc_host" value="modelsrc@192.168.8.24"')

    def test_valid_host_saved(self):
        r = self.post(modelsrc_host=" src@10.0.0.2 ")
        self.assertIn("src@10.0.0.2", parse_qs(urlparse(r.headers["location"]).query)["msg"][0])
        self.assertEqual(store.get_setting("modelsrc_host"), "src@10.0.0.2")
        self.assertEqual(main.modelsrc().host(), "src@10.0.0.2")
        self.assertRegex(self.page(), r'name="modelsrc_host" value="src@10.0.0.2"')
        self.post(modelsrc_host="")                             # blank = the default
        self.assertEqual(main._modelsrc_host(), "modelsrc@192.168.8.24")

    def test_invalid_host_is_400_with_the_form_as_typed(self):
        store.set_settings({"modelsrc_host": "src@10.0.0.2"})
        for bad in ("a;touch /tmp/x", "-oProxyCommand=x", "user@host:22", "a b"):
            with self.subTest(bad=bad):
                r = self.post(status=400, modelsrc_host=bad)
                self.assertIn("not saved", r.text)
                self.assertIn(f'name="modelsrc_host" value="{admin._esc(bad)}"', r.text)
                self.assertEqual(store.get_setting("modelsrc_host"), "src@10.0.0.2")

    def test_refused_host_shows_its_reason_without_controllers(self):
        self.views = {}
        r = self.post(status=400, modelsrc_host="a;b")
        self.assertIn("not saved", r.text)
        self.assertIn('name="modelsrc_host" value="a;b"', r.text)

    def test_changed_host_names_the_stale_pin(self):
        lan = main.modelsrc()
        with open(lan.known_hosts_path, "w") as f:
            f.write(f"192.168.8.24 ssh-ed25519 {_ED_B64}\n")
        self.assertEqual(lan.problem(), "not listed yet")
        self.post(modelsrc_host="modelsrc@10.0.0.9")
        m = re.search(r'<div class="tcard" data-k="thunder-modelsrc">.*?</div></div>',
                      self.page(), re.S)
        self.assertIn("pinned for 192.168.8.24, not 10.0.0.9 — fetch its key", m.group(0))


class CostBanner(_Base):
    """Task 16, spec "Kosten-Wächter": an instance up for more than 24 h gets a banner on
    its card AND on the Dashboard — the page an operator actually looks at every day."""

    def long_view(self, **over):
        v = dict(phase="ready", uuid="u1", started_at=1.0, uptime_s=26 * 3600 + 120,
                 cost_per_h=0.5, session_cost=13.02, long_running=True)
        v.update(over)
        return _view(**v)

    def dash(self) -> str:
        def longrun():
            return [(n, v) for n, v in self.views.items() if v.get("long_running")]

        def heavy(name):
            raise AssertionError("the Dashboard must not build the full panel view")
        for k, v in {"_dashboard_snapshot": lambda: {"backends": []},
                     "_faults_info": lambda: {"backends": [], "bundles": [], "total": 0},
                     "_thunder_longrun": longrun, "_thunder_view": heavy}.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)
        r = asyncio.run(admin.dashboard_page(_Req({})))
        return r.body.decode()

    def test_card_banner(self):
        self.views = {"tc": self.long_view()}
        page = self.page()
        self.assertRegex(page, r'data-k="thunder-tc-longrun"')
        self.assertIn("Thunder tc running for 26 h (≈ $13.02)", page)
        self.views = {"tc": _view(phase="ready", uuid="u1", started_at=1.0,
                                  uptime_s=3600, long_running=False)}
        self.assertNotIn("longrun", self.page())

    def test_dashboard_banner_after_24h(self):
        self.views = {"tc": self.long_view(), "b": _view(name="b")}
        html = self.dash()
        m = re.search(r'<p class="bad" data-k="dash-longrun-tc">.*?</p>', html, re.S)
        self.assertIsNotNone(m, html[:2000])
        self.assertIn("Thunder tc running for 26 h (≈ $13.02)", m.group(0))
        self.assertIn('href="/ui/backends"', m.group(0))
        self.assertNotIn("dash-longrun-b", html)
        self.assertIn('data-live="4"', html)                 # the Dashboard stays live
        # no price list yet: still the banner, no made-up figure
        self.views = {"tc": self.long_view(session_cost=None, cost_per_h=None)}
        self.assertIn("Thunder tc running for 26 h (cost unknown", self.dash())
        # under 24 h, or nothing running: no banner at all
        self.views = {"tc": _view(phase="ready", started_at=1.0, uptime_s=3600)}
        self.assertNotIn("dash-longrun", self.dash())

    def test_dashboard_banner_reads_no_store(self):
        """main.thunder_longrun is what the Dashboard polls every 4 s: Controller.view()
        only — thunder_view's alias-gate note would read the store per not-ready alias."""
        class Ctl:
            def __init__(self, long):
                self.long = long

            def view(self):
                return {"phase": "ready", "long_running": self.long, "uptime_s": 90000,
                        "session_cost": 1.5,
                        "plan": {"aliases": {"a": {"ready": False}, "b": {"ready": False}}}}
        saved = dict(main.thunder_controllers)
        self.addCleanup(lambda: (main.thunder_controllers.clear(),
                                 main.thunder_controllers.update(saved)))
        main.thunder_controllers.clear()
        main.thunder_controllers.update({"tc": Ctl(True), "k2": Ctl(False)})

        def boom(*a, **k):
            raise AssertionError("store read on the Dashboard path")
        for fn in ("get", "list_aliases", "get_setting", "get_settings"):
            self.addCleanup(setattr, store, fn, getattr(store, fn))
            setattr(store, fn, boom)
        self.assertIs(admin._thunder_longrun, main.thunder_longrun)
        self.assertEqual([n for n, _ in main.thunder_longrun()], ["tc"])
        html = admin._dash_thunder()
        self.assertIn('data-k="dash-longrun-tc"', html)
        self.assertIn("Thunder tc running for 25 h (≈ $1.50)", html)
        self.assertNotIn("k2", html)


class OrphanSnapshots(_Base):
    """Task 16: `aihub-` snapshots no current Thunder backend owns bill $/month unseen
    (a renamed backend leaves every old one behind) — shown on the Backends tab, never
    deleted, rendered from the controllers' caches only."""

    def setUp(self):
        super().setUp()
        self.orph = [{"id": "s2", "name": "aihub-old-20260926t120000z", "status": "READY",
                      "gb": 200, "monthly": 1.0}]
        self.addCleanup(setattr, admin, "_thunder_orphan_snapshots",
                        admin._thunder_orphan_snapshots)
        admin._thunder_orphan_snapshots = lambda: self.orph

    def test_orphan_snapshot_warning(self):
        self.views = {"tc": _view()}
        page = self.page()
        m = re.search(r'<div data-k="thunder-orphan-snaps">.*?</table></div>', page, re.S)
        self.assertIsNotNone(m)
        self.assertIn("aihub-old-20260926t120000z", m.group(0))
        self.assertIn("$1.00/month", m.group(0))
        self.assertIn('data-k="thunder-osnap-s2"', m.group(0))
        self.assertNotIn("<button", m.group(0))              # display only, no delete
        self.orph = []
        self.assertNotIn("thunder-orphan-snaps", self.page())

    def test_orphan_instances_priced_on_the_card(self):
        self.orph = []
        self.views = {"tc": _view(orphans=[
            {"uuid": "x", "index": "4", "status": "RUNNING", "gpu_type": "a6000",
             "num_gpus": 1, "cost_per_h": 0.42},
            {"uuid": "y", "index": "5", "status": "RUNNING", "gpu_type": "h200",
             "num_gpus": 2, "cost_per_h": None}])}
        page = self.page()
        x = re.search(r'<tr data-k="thunder-tc-orphan-x">.*?</tr>', page).group(0)
        self.assertIn("$0.42/h", x)
        self.assertIn("a6000 ×1", x)
        y = re.search(r'<tr data-k="thunder-tc-orphan-y">.*?</tr>', page).group(0)
        self.assertIn("<td>—</td>", y)                      # no price: no made-up figure
        self.assertIn("h200 ×2", y)

    def test_main_uses_the_controllers_caches(self):
        class Ctl:
            def __init__(self, snaps, table):
                self._s, self._t = snaps, table

            def snapshots(self):
                return self._s

            def pricing_table(self):
                return self._t

        snap = lambda n, i, gb=100: {"id": i, "name": n, "status": "READY",   # noqa: E731
                                     "min_disk_gb": gb, "created_at": 1}
        saved = dict(main.thunder_controllers)
        self.addCleanup(lambda: (main.thunder_controllers.clear(),
                                 main.thunder_controllers.update(saved)))
        main.thunder_controllers.clear()
        main.thunder_controllers["tc"] = Ctl([snap("aihub-tc-20260926t120000z", "a"),
                                              snap("aihub-gone-20260926t120000z", "b")], None)
        main.thunder_controllers["k2"] = Ctl([snap("aihub-gone-20260926t120000z", "b"),
                                              snap("aihub-k2-20260926t120000z", "c")],
                                             {"snapshot_gb": 0.001})
        out = main.thunder_orphan_snapshots()
        self.assertEqual([o["id"] for o in out], ["b"])      # once, though both list it
        self.assertAlmostEqual(out[0]["monthly"], 100 * 0.001 * 730)


class Task16Wiring(unittest.TestCase):
    def test_bound(self):
        self.assertIs(admin._thunder_orphan_snapshots, main.thunder_orphan_snapshots)


if __name__ == "__main__":
    unittest.main()
