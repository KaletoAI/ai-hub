"""The console half of a managed host: the host form and its save/delete, the host card
with its service table, the model-sync/LAN-source/catalog blocks, the Dashboard banner —
and the backend form's managed-host attachment.

Every one of these fails SILENTLY:
  · the host form: the API token is never rendered (blank keeps it, the box clears it),
    each provider option comes from the provider's OWN `OPTION_FIELDS` (a hand-kept
    copy drifts: a GPU the provider knows but the form does not is unreachable, one the
    form offers but the provider refuses is only noticed at the start), a refused Save
    is a 400 with the form as typed and nothing stored, and a new host's name may not
    collide with a Hosts-map key or any backend's host — an URL hostname included
    (R-W6) — or the coordination policies of that box silently apply to both;
  · a host is deleted only while off and unnamed by any backend (R-W5): its instance
    and snapshots would otherwise bill with nobody to stop them;
  · the card's actions must be POSTs (a GET link fires on any prefetch — here it would
    start a billing GPU), start/stop/delete/forget ask first, and the page must stay
    live while an instance runs or an op is in flight, or the phase shown is a stale
    snapshot; the service table's Restart / Re-run setup carry the BACKEND id, or the
    wrong service restarts;
  · every managed host is listed in the Hosts panel, also one without ComfyUI;
  · the backend form attaches a backend to a managed host: `backend_save` derives `url`
    and `local_port` (stable over Saves, renames and a move to another host — a port
    that moves takes the URL from under running jobs), the URL input is `readonly`,
    never `disabled` (a disabled input is not submitted, and absent reads as cleared),
    and whatever the host could not run is refused up front with the form as typed: a
    type without a service profile, a remote port that is none or already taken on that
    host, a second ComfyUI, a file-name slug another command service uses (R-W8), a
    config-defined backend (R-K3). Detaching drops the tunnel URL — a backend left on a
    127.0.0.1 port nothing forwards any more is healthy-looking and dead. The old
    Thunder block is gone from form, save and summary;
  · the model-sync panel: rows keyed (the morph would otherwise rewrite every row per
    tick), "delete unknown" a POST whose confirm names what the TICKED boxes hold, a
    catalog the validator refuses is a 400 with the text as typed and nothing saved,
    the seed is copied into the setting once and never again, and a host-only alias
    says why its schema is empty;
  · the LAN source: its host key is pinned only by a POST that carries the fingerprint
    the operator saw, and the install command gives the share user a real login shell;
  · the HF token is encrypted at rest and never rendered, `modelsrc_host` is refused
    unless plain and a changed host names the stale pin, an instance up > 24 h puts a
    banner on its card AND the Dashboard, and `aihub-` snapshots no host owns are
    listed with $/month (never deleted).

Run: python -m unittest tests.test_hosts_ui -v
"""
import asyncio
import json
import os
import re
import sys
import tempfile
import html as html_mod
import unittest
from unittest import mock
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
    import hostctl
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


def _mform(name="vllm", **over) -> dict:
    """A backend form attaching an OpenAI-compatible service to managed host vm1."""
    f = {"name": name, "type": "openai", "url": "", "host_managed": "vm1", "host": "",
         "remote_port": "8000", "svc_start": "vllm serve m --port 8000", "svc_setup": "",
         "svc_health": ""}
    f.update(over)
    return f


def _svcs(**over) -> dict:
    """A view's `services`: the ComfyUI backend `tc` (+ `over`)."""
    d = {"comfyui:tc": {"name": "tc", "type": "comfyui", "local_port": 18100,
                        "remote_port": 8188, "status": "up", "error": ""}}
    d.update(over)
    return d


def _view(**over) -> dict:
    """A host view (main.host_view): the controller's view plus the host's options,
    `api_key_set`, `managed` and `not_attachable`."""
    v = {"name": "tc", "provider": "thunder", "phase": "off", "error": "", "failed_phase": "", "index": None,
         "uuid": "", "ip": "", "port": 0, "started_at": 0, "uptime_s": 0, "disk_gb": 0,
         "cost_per_h": None, "session_cost": None,
         "snapshot": {"id": "", "pending": "", "pending_name": "", "name": "", "status": "",
                      "gb": None, "monthly": None},
         "log": [], "transfers": {}, "persist_blocked": False, "bootstrap_unknown": {},
         "bootstrap_template_nodes": [], "bootstrap_incomplete": False, "op": None,
         "waiting_jobs": None, "unreconciled_uuids": [], "orphans": [],
         "services": {}, "tunnel_error": "", "not_attachable": [], "host_error": "",
         "managed": True, "api_key_set": False,
         "options": {"gpu_type": "a6000", "num_gpus": 1, "vcpus": 8,
                     "bootstrap_template": "", "reserve_gb": 20, "comfy_commit": SHA,
                     "nodes": []}}
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
        self.catalog = [{"match": {"class": "C", "value": "v"}, "paths": ["models/c/"]}]
        self.saved_catalogs = []
        self.synced = []
        self.deleted = []
        self.delete_refusals = {}      # host → why Delete is refused (absent = allowed)
        self.saved_hosts = []          # (name, entry, new) main.save_managed_host got
        self.save_refusal = ""
        self.deleted_hosts = []

        def save_cat(cat):
            errs = __import__("modelsync").validate_catalog(cat)
            if not errs:
                self.saved_catalogs.append(cat)
                self.catalog = cat
            return errs

        async def act(name, action, bid=None, paths=None):
            # main.host_action's contract: always text, a refusal included
            self.actions.append((name, action) if bid is None else (name, action, bid))
            if action == "sync":
                self.synced.append(name)
                return "sync requested"
            if action == "delete_unknown":
                self.deleted.append((name, list(paths or [])))
                return f"deleted {len(paths or [])} unknown file(s)"
            return f"{action} requested"

        def save_host(name, entry, new):
            if self.save_refusal:
                return self.save_refusal
            self.saved_hosts.append((name, entry, new))
            return ""

        def delete_host(name):
            why = self.delete_refusals.get(name)
            if why:
                return why
            self.deleted_hosts.append(name)
            return ""
        stubs = {"_apply_backends": lambda: None,
                 "_modelsync_catalog": lambda: self.catalog,
                 "_save_modelsync_catalog": save_cat,
                 "_gateway_info": lambda: {"backends": self.live, "virtual_models": []},
                 "_host_names": lambda: list(self.views),
                 "_host_view": lambda n: self.views.get(n),
                 "_host_action": act,
                 "_save_managed_host": save_host,
                 "_delete_managed_host": delete_host,
                 "_managed_host_delete_refusal": lambda n: self.delete_refusals.get(n),
                 "_thunder_default_nodes": lambda: "# defaults\nregistry:x@1.0\n"}
        for k, v in stubs.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)

    def save(self, form):
        return asyncio.run(admin.backend_save(_Req(form)))

    def page(self, qp=None) -> str:
        r = asyncio.run(admin._backends_view(qp or {}))
        return r.body.decode()


class ManagedSave(_Base):
    """backend_save with a managed host: url/local_port derived, the type and port rules
    refused up front (400, the form as typed, nothing stored)."""

    def setUp(self):
        super().setUp()
        for n in ("vm1", "vm2"):
            store.set_managed_host(n, {"provider": "thunder", "options": {}, "api_key": "tok"})

    def row(self, name="vllm", t="openai"):
        return store.get_backend(name, t)

    def refused(self, form, *needles):
        r = self.save(form)
        self.assertEqual(r.status_code, 400, form)
        html = r.body.decode()
        for n in needles:
            self.assertIn(n, html)
        self.assertNotIn("<main data-live", html)
        return html

    def test_save_derives_url_and_local_port(self):
        r = self.save(_mform(url="http://10.0.0.5:9999", host="gpu9"))
        self.assertEqual(r.status_code, 303)
        b = self.row()
        lp = b["local_port"]
        self.assertTrue(main.LOCAL_PORT_MIN <= lp <= main.LOCAL_PORT_MAX, lp)
        self.assertEqual(b["url"], f"http://127.0.0.1:{lp}")      # the typed url is ignored
        self.assertEqual(b["host"], "vm1")                        # the managed one wins
        self.assertEqual(b["remote_port"], 8000)
        self.assertEqual(b["svc_start"], "vllm serve m --port 8000")
        self.assertNotIn("svc_setup", b)                          # blank = no setup
        self.assertNotIn("svc_health", b)                         # blank = the default
        self.assertNotIn("thunder", b)

    def test_local_port_stable_across_resaves_and_a_rename(self):
        self.assertEqual(self.save(_mform()).status_code, 303)
        lp = self.row()["local_port"]
        self.assertEqual(self.save(_mform("other", remote_port="8001")).status_code, 303)
        self.assertNotEqual(self.row("other")["local_port"], lp)   # unique over hosts
        self.assertEqual(self.save(_mform(orig="openai:vllm")).status_code, 303)
        self.assertEqual(self.row()["local_port"], lp)
        # a rename keeps it: the old row (still in the store until the Save writes)
        # is this backend, not another one holding the port
        self.assertEqual(self.save(_mform("vllm2", orig="openai:vllm")).status_code, 303)
        self.assertIsNone(self.row("vllm"))
        self.assertEqual(self.row("vllm2")["local_port"], lp)
        self.assertEqual(self.row("vllm2")["url"], f"http://127.0.0.1:{lp}")
        # moving to another managed host keeps it too (unique over ALL hosts)
        self.assertEqual(self.save(_mform("vllm2", orig="openai:vllm2", host_managed="vm2"))
                         .status_code, 303)
        self.assertEqual(self.row("vllm2")["local_port"], lp)

    def test_comfy_dirs_default_when_blank(self):
        self.save(_mform("c", type="comfyui", remote_port="8188", svc_start=""))
        b = self.row("c", "comfyui")
        self.assertEqual(b["comfy_output_dir"], "/home/ubuntu/ComfyUI/output")
        self.assertEqual(b["comfy_input_dir"], "/home/ubuntu/ComfyUI/input")
        self.assertEqual(b["remote_port"], 8188)
        for k in ("svc_start", "svc_setup", "svc_health"):
            self.assertNotIn(k, b)                   # command fields belong to openai only
        self.save(_mform("c", type="comfyui", remote_port="8188", orig="comfyui:c",
                         comfy_output_dir="/x/out", comfy_input_dir="/x/in"))
        b = self.row("c", "comfyui")
        self.assertEqual((b["comfy_output_dir"], b["comfy_input_dir"]), ("/x/out", "/x/in"))

    def test_type_without_a_profile_is_refused(self):
        for t in ("meshy", "tripo", "anthropic"):
            self.refused(_mform(f"x-{t}", type=t), "cannot run on a managed host")
            self.assertIsNone(self.row(f"x-{t}", t), t)

    def test_unknown_managed_host_is_refused(self):
        self.refused(_mform(host_managed="nope"), "nope")
        self.assertIsNone(self.row())

    def test_remote_port_required_and_a_port(self):
        for bad in ("", "0", "70000", "8.5", "x", "-1"):
            self.refused(_mform(remote_port=bad), "remote port")
            self.assertIsNone(self.row(), bad)

    def test_remote_port_unique_per_host(self):
        self.assertEqual(self.save(_mform("a")).status_code, 303)
        self.refused(_mform("b"), "8000", "a")
        self.assertIsNone(self.row("b"))
        self.assertEqual(self.save(_mform("b", host_managed="vm2")).status_code, 303)
        self.assertEqual(self.save(_mform("a", orig="openai:a")).status_code, 303)  # itself

    def test_second_comfyui_on_a_host_is_refused(self):
        c = dict(type="comfyui", svc_start="")
        self.assertEqual(self.save(_mform("c1", remote_port="8188", **c)).status_code, 303)
        self.refused(_mform("c2", remote_port="8189", **c), "second ComfyUI", "c1")
        self.assertIsNone(self.row("c2", "comfyui"))
        self.assertEqual(self.save(_mform("c2", remote_port="8188", host_managed="vm2", **c))
                         .status_code, 303)

    def test_slug_collision_is_refused(self):
        self.assertEqual(self.save(_mform("My.Svc")).status_code, 303)
        self.refused(_mform("my-svc", remote_port="8001"), "slug", "my-svc")
        self.assertIsNone(self.row("my-svc"))
        # a ComfyUI service names no files by slug — no clash with it
        self.assertEqual(self.save(_mform("my-svc", type="comfyui", remote_port="8188"))
                         .status_code, 303)

    def test_command_fields_validated(self):
        self.refused(_mform(svc_start="  "), "start command")
        self.refused(_mform(svc_health="no slash"), "health path")
        self.assertIsNone(self.row())
        self.save(_mform(svc_health="/health", svc_setup="pip install -y x\n"))
        b = self.row()
        self.assertEqual(b["svc_health"], "/health")
        self.assertEqual(b["svc_setup"], "pip install -y x\n")

    def test_config_backend_cannot_be_attached(self):
        self.live = [{"name": "vllm", "type": "openai", "url": "http://h:1", "enabled": True,
                      "healthy": True, "models": 0, "source": "config"}]
        self.refused(_mform(orig="openai:vllm"), "config-defined")
        self.assertIsNone(self.row())
        # unattached, the same config backend's copy saves as before
        r = self.save(_mform(orig="openai:vllm", host_managed="", url="http://h:1"))
        self.assertEqual(r.status_code, 303)

    def test_refused_form_is_shown_as_typed(self):
        html = self.refused(_mform(remote_port="lots", svc_setup="echo SETUP-TEXT"),
                            'value="lots"', "echo SETUP-TEXT", "vllm serve m --port 8000")
        self.assertRegex(html, r'<option value="vm1" selected>')

    def test_detach_clears_the_derived_url_and_ports(self):
        self.save(_mform())
        lp = self.row()["local_port"]
        derived = f"http://127.0.0.1:{lp}"
        # the readonly field still carries the tunnel URL: it goes with the host
        self.refused(_mform(orig="openai:vllm", host_managed="", url=derived), "url")
        self.assertEqual(self.row()["host"], "vm1")               # nothing written
        r = self.save(_mform(orig="openai:vllm", host_managed="", url="http://10.0.0.5:8000",
                             host="gpu9"))
        self.assertEqual(r.status_code, 303)
        b = self.row()
        self.assertEqual(b["url"], "http://10.0.0.5:8000")
        self.assertEqual(b["host"], "gpu9")
        self.assertNotIn("local_port", b)
        self.assertNotIn("remote_port", b)

    def test_old_thunder_block_dropped_on_save(self):
        store.upsert_backend({"name": "tc", "type": "comfyui", "url": "http://x:1",
                              "thunder": {"gpu_type": "a6000"}})
        r = self.save({"name": "tc", "type": "comfyui", "url": "http://x:1", "orig": "comfyui:tc"})
        self.assertEqual(r.status_code, 303)
        self.assertNotIn("thunder", store.get_backend("tc", "comfyui"))

    def test_plain_backend_unchanged(self):
        r = self.save({"name": "p", "type": "openai", "url": "http://h:1", "host": "gpu1",
                       "remote_port": "8000", "svc_start": "x"})
        self.assertEqual(r.status_code, 303)
        b = self.row("p")
        self.assertEqual((b["url"], b["host"]), ("http://h:1", "gpu1"))
        self.assertNotIn("remote_port", b)
        self.assertNotIn("local_port", b)

    def test_api_key_unaffected(self):
        self.save(_mform(api_key="K-SECRET"))
        self.assertEqual(self.row()["api_key"], "K-SECRET")


class ManagedForm(_Base):
    def setUp(self):
        super().setUp()
        for n in ("vm1", "vm2"):
            store.set_managed_host(n, {"provider": "thunder", "options": {}, "api_key": "tok"})

    def _b(self, **over):
        b = {"name": "vllm", "type": "openai", "url": "http://127.0.0.1:18100", "host": "vm1",
             "local_port": 18100, "remote_port": 8001, "svc_start": "serve",
             "api_key": "SECRET-TOKEN"}
        b.update(over)
        return b

    def test_host_select_lists_every_managed_host(self):
        html = admin._backend_form(self._b(), ["gpu1"])
        m = re.search(r'<select name="host_managed"[^>]*>(.*?)</select>', html, re.S)
        self.assertIsNotNone(m)
        opts = re.findall(r'<option value="([^"]*)"', m.group(1))
        self.assertEqual(opts, ["", "vm1", "vm2"])
        self.assertIn("(none / free text)", m.group(1))
        self.assertRegex(m.group(1), r'<option value="vm1" selected>')
        self.assertIn('name="host"', html)                      # the free text stays
        plain = admin._backend_form({"name": "p", "type": "openai", "url": "http://h:1",
                                     "host": "gpu1"}, ["gpu1"])
        self.assertRegex(plain, r'<option value="" selected>')
        self.assertIn('name="host" value="gpu1"', plain)

    def test_url_readonly_not_disabled_with_a_hint(self):
        html = admin._backend_form(self._b(), [])
        tag = re.search(r'<input[^>]*name="url"[^>]*>', html).group(0)
        self.assertIn("readonly", tag)
        self.assertNotIn("disabled", tag)
        self.assertIn("derived", html)
        plain = admin._backend_form({"name": "c", "type": "comfyui", "url": "http://x:1"}, [])
        self.assertNotIn("readonly", re.search(r'<input[^>]*name="url"[^>]*>', plain).group(0))

    def test_managed_block_shown_only_with_a_managed_host(self):
        block = r'<fieldset[^>]*data-mhost[^>]*>'
        self.assertNotIn("display:none", re.search(block, admin._backend_form(self._b(), []))
                         .group(0))
        plain = admin._backend_form({"name": "p", "type": "openai", "url": "http://h:1"}, [])
        self.assertIn("display:none", re.search(block, plain).group(0))
        for k in ("remote_port", "svc_setup", "svc_start", "svc_health"):
            self.assertIn(f'name="{k}"', plain)                 # rendered, only hidden

    def test_remote_port_prefilled_from_the_profile(self):
        new = admin._backend_form(None, [])
        self.assertIn('name="remote_port" value="8000"', new)
        comfy = admin._backend_form({"name": "c", "type": "comfyui", "url": "http://x:1"}, [])
        self.assertIn('name="remote_port" value="8188"', comfy)
        self.assertIn('name="remote_port" value="8001"', admin._backend_form(self._b(), []))

    def test_command_fields_and_hints(self):
        html = admin._backend_form(self._b(svc_setup="apt-get install -y x",
                                           svc_health="/health"), [])
        self.assertIn("apt-get install -y x</textarea>", html)
        self.assertIn("serve</textarea>", html)
        self.assertIn('name="svc_health" value="/health"', html)
        self.assertIn("no tokens here", html.lower())
        self.assertIn("plain text", html)
        self.assertIn("HF-token setting", html)
        self.assertIn("-y", html)
        self.assertIn("&lt;/dev/null", html)                    # stdin eats the script
        self.assertIn("127.0.0.1", html)                        # where it must listen
        # the command fields sit in an openai-only block
        m = re.search(r'<div data-btype="openai"[^>]*>(?:(?!</fieldset>).)*name="svc_start"',
                      html, re.S)
        self.assertIsNotNone(m)

    def test_token_never_rendered(self):
        self.assertNotIn("SECRET-TOKEN", admin._backend_form(self._b(), []))

    def test_thunder_fields_are_gone(self):
        for t in ("openai", "comfyui", "meshy", "tripo", "anthropic"):
            html = admin._backend_form({"name": "x", "type": t, "url": "http://x:1",
                                        "thunder": {"gpu_type": "h100"}}, [])
            self.assertNotIn('name="thunder_', html, t)
            self.assertNotIn("Thunder Compute", html, t)
        for gone in ("_THUNDER_GPUS", "_THUNDER_TEMPLATES", "_THUNDER_COMMIT_DEFAULT",
                     "_THUNDER_PORT_DEFAULT", "_THUNDER_DIRS", "_thunder_fieldset"):
            self.assertFalse(hasattr(admin, gone), gone)

    def test_gateway_info_carries_no_thunder_block_nor_token(self):
        cfg = {"name": "tc", "type": "comfyui", "url": "http://x:1", "api_key": "TOK-123",
               "thunder": {"gpu_type": "h100", "api_key": "TOK-456"}}
        saved = (main.backends, main.config_backends)
        self.addCleanup(lambda: (setattr(main, "backends", saved[0]),
                                 setattr(main, "config_backends", saved[1])))
        main.backends = main.config_backends = [cfg]
        s = next(b for b in main.gateway_info()["backends"] if b["name"] == "tc")
        self.assertNotIn("thunder", s)
        self.assertNotIn("TOK-", json.dumps(s))

    def test_local_port_assigner_is_bound(self):
        self.assertIs(admin._assign_local_port, main.assign_local_port)


class Panel(_Base):
    def setUp(self):
        super().setUp()
        self.live = [{"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18100",
                      "host": "tc", "enabled": False, "healthy": False, "models": 0,
                      "source": "ui"}]

    def test_panel_buttons_are_post_actions(self):
        for p in ("/ui/hosts/managed/start", "/ui/hosts/managed/stop",
                  "/ui/hosts/managed/restart-service", "/ui/hosts/managed/resetup",
                  "/ui/hosts/managed/forget", "/ui/hosts/managed/delete",
                  "/ui/hosts/managed/save"):
            self.assertIn(p, admin._POST_ACTIONS)
        self.views = {"tc": _view(phase="ready", uuid="u1", unreconciled_uuids=["u-9"],
                                  services=_svcs())}
        html = self.page()
        self.assertNotRegex(html, r'<a[^>]*href="/ui/hosts/managed/')
        for p in ("stop", "forget"):
            self.assertRegex(html, rf'formaction="/ui/hosts/managed/{p}\?host=tc"')
        for p in ("restart-service", "resetup"):
            self.assertRegex(html, rf'formaction="/ui/hosts/managed/{p}\?host=tc&amp;'
                                   r'bid=comfyui%3Atc"')
        self.assertNotIn("/ui/hosts/managed/start", html)     # running: nothing to start
        self.views = {"tc": _view(phase="off")}
        html = self.page()
        self.assertNotRegex(html, r'<a[^>]*href="/ui/hosts/managed/')
        self.assertRegex(html, r'formaction="/ui/hosts/managed/start\?host=tc"')
        self.assertNotIn("/ui/hosts/managed/stop", html)
        # a start in flight is still `off` until the create: Stop must be there to abort it
        self.views = {"tc": _view(phase="off", op="starting")}
        html = self.page()
        self.assertRegex(html, r'formaction="/ui/hosts/managed/stop\?host=tc"')
        self.assertNotIn("/ui/hosts/managed/start", html)
        # failed after the create: the instance bills — Stop stays, Start does not come back
        self.views = {"tc": _view(phase="failed", uuid="u1", failed_phase="bootstrapping")}
        html = self.page()
        self.assertRegex(html, r'formaction="/ui/hosts/managed/stop\?host=tc"')
        self.assertNotIn("/ui/hosts/managed/start", html)

    def test_stop_and_forget_ask_first(self):
        self.views = {"tc": _view(phase="ready", uuid="u1", unreconciled_uuids=["u-9"])}
        html = self.page()
        for p in ("stop", "forget"):
            m = re.search(rf'<button[^>]*formaction="/ui/hosts/managed/{p}\?[^"]*"[^>]*>', html)
            self.assertIsNotNone(m, p)
            self.assertIn("data-confirm=", m.group(0), p)

    def test_start_asks_first_and_names_the_bill(self):
        # a Start rents a GPU by the hour: one stray click must not do that
        self.views = {"tc": _view(phase="off")}
        html = self.page()
        m = re.search(r'<button[^>]*formaction="/ui/hosts/managed/start[^"]*"[^>]*>', html)
        self.assertIsNotNone(m)
        conf = re.search(r'data-confirm="([^"]*)"', m.group(0))
        self.assertIsNotNone(conf, m.group(0))
        self.assertIn("Start the Thunder Compute host tc?", conf.group(1))
        self.assertIn("bills per hour until you stop it", conf.group(1))

    def test_persist_error_is_on_the_card(self):
        self.views = {"tc": _view(phase="ready", uuid="u1",
                                  persist_error="12:00:00 OSError('disk <full>')")}
        html = self.page()
        self.assertIn('data-k="host-tc-persist"', html)
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
        self.assertNotIn("/ui/hosts/managed/forget", self.page())

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
        store.upsert_backend({"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18100",
                              "host": "tc", "local_port": 18100, "remote_port": 8188})
        for qp in ({"edit": "comfyui:tc"}, {"new": "1"}, {"host": "gpu1"},
                   {"mhost": "tc"}, {"mhost_new": "1"}):
            html = self.page(qp)
            self.assertIn("<main>", html, qp)
            self.assertNotIn("<main data-live", html, qp)
        self.assertIn('<main data-live="3">', self.page())

    def test_refused_save_is_never_live(self):
        self.views = {"tc": _view(phase="ready")}
        store.set_managed_host("tc", {"provider": "thunder", "options": {}, "api_key": "t"})
        r = self.save(_mform("x", host_managed="tc", remote_port=""))
        self.assertEqual(r.status_code, 400)
        self.assertNotIn("<main data-live", r.body.decode())

    def test_buttons_follow_the_controller_refusals(self):
        # op in flight: no Start, no Restart (the controller answers "already …")
        self.views = {"tc": _view(phase="failed", op="stopping", uuid="u1", services=_svcs())}
        html = self.page()
        self.assertNotIn("/ui/hosts/managed/start", html)
        self.assertNotIn("/ui/hosts/managed/restart-service", html)
        self.assertNotIn("/ui/hosts/managed/resetup", html)
        self.assertIn("/ui/hosts/managed/stop", html)
        # failed before the create (no instance): Start yes, Restart no (needs the uuid)
        self.views = {"tc": _view(phase="failed", failed_phase="creating", services=_svcs())}
        html = self.page()
        self.assertIn("/ui/hosts/managed/start", html)
        self.assertNotIn("/ui/hosts/managed/restart-service", html)
        # a service is restarted only on a running host (the controller refuses otherwise)
        self.views = {"tc": _view(phase="failed", uuid="u1", services=_svcs())}
        self.assertNotIn("/ui/hosts/managed/restart-service", self.page())
        self.views = {"tc": _view(phase="ready", uuid="u1", services=_svcs())}
        self.assertIn("/ui/hosts/managed/restart-service", self.page())
        # a host that is not driven (unknown provider, broken entry) has nothing to start
        self.views = {"tc": _view(phase="off", host_error="unknown provider 'runpod'",
                                  error="unknown provider 'runpod'")}
        html = self.page()
        self.assertNotIn("/ui/hosts/managed/start", html)
        self.assertIn("unknown provider &#x27;runpod&#x27;", html)

    def test_odd_numbers_in_the_view_do_not_break_the_tab(self):
        self.views = {"tc": _view(phase="draining", waiting_jobs="?", uptime_s="x",
                                  cost_per_h="n/a", bootstrap_unknown={"a.bin": "big"})}
        html = self.page()
        self.assertIn("a.bin", html)
        self.assertIn("0.0 GB", html)

    def test_not_live_when_everything_is_off(self):
        self.views = {"tc": _view(phase="off")}
        self.assertIn("<main>", self.page())

    def test_card_drain_per_service_and_tunnel_error(self):
        # a host drains EVERY attached backend: the card names who it still waits for;
        # a tunnel whose spawn keeps failing says so (Ruling M3) instead of looking like
        # a restart now and then
        self.views = {"tc": _view(phase="draining", op="stopping",
                                  waiting_jobs={"openai:vllm": 1, "comfyui:tc": 2},
                                  tunnel_error="control socket in use, left in place: '/x'")}
        html = self.page()
        drain = re.search(r'<p class="hint" data-k="host-tc-drain">.*?</p>', html).group(0)
        self.assertIn("2 jobs on comfyui:tc", drain)
        self.assertIn("1 job on openai:vllm", drain)
        tun = re.search(r'<p class="bad" data-k="host-tc-tunnel">.*?</p>', html).group(0)
        self.assertIn("Tunnel will not come up: control socket in use", tun)
        self.views = {"tc": _view(phase="ready")}
        self.assertNotIn("host-tc-tunnel", self.page())

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
        self.assertIn('data-k="host-tc"', html)
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
        log = [f"line {i}" for i in range(hostctl._LOG_MAX + 30)]
        self.views = {"tc": _view(log=log)}
        html = self.page()
        self.assertEqual(admin._HOST_LOG_LINES, hostctl._LOG_MAX)
        self.assertIn(f"log (last {hostctl._LOG_MAX} lines)", html)
        self.assertIn("line 30\n", html)
        self.assertNotIn("line 29\n", html)

    def test_card_rows_are_keyed_and_add_no_script(self):
        self.views = {"tc": _view(phase="ready", orphans=[{"uuid": "o1", "status": "RUNNING"}],
                                  bootstrap_unknown={"a.bin": 1})}
        html = self.page()
        m = re.search(r'<main[^>]*>(.*)</main>', html, re.S)
        self.assertNotIn("<script", m.group(1))
        self.assertIn('data-k="host-tc-orphan-o1"', html)

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
        for path, action in (("start", "start"), ("stop", "stop"),
                             ("forget", "forget_unreconciled")):
            r = self.c.post(f"/ui/hosts/managed/{path}?host=a%26b", headers=SAME,
                            follow_redirects=False)
            self.assertEqual(r.status_code, 303, path)
            loc = urlparse(r.headers["location"])
            self.assertEqual(loc.path, "/ui/backends")
            self.assertIn(f"{action} requested", parse_qs(loc.query)["msg"][0])
            self.assertEqual(self.actions[-1], ("a&b", action))

    def test_service_actions_carry_the_backend_id(self):
        # the host alone is not enough: Restart must reach THIS service, not "the" one
        for path, action in (("restart-service", "restart_service"), ("resetup", "resetup")):
            r = self.c.post(f"/ui/hosts/managed/{path}?host=tc&bid=openai%3Av%26x",
                            headers=SAME, follow_redirects=False)
            self.assertEqual(r.status_code, 303, path)
            self.assertIn(f"{action} requested",
                          parse_qs(urlparse(r.headers["location"]).query)["msg"][0])
            self.assertEqual(self.actions[-1], ("tc", action, "openai:v&x"))
        # as form fields too
        r = self.c.post("/ui/hosts/managed/resetup", data={"host": "tc", "bid": "comfyui:tc"},
                        headers=SAME, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.actions[-1], ("tc", "resetup", "comfyui:tc"))

    def test_host_as_form_field(self):
        r = self.c.post("/ui/hosts/managed/start", data={"host": "tc"}, headers=SAME,
                        follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.actions[-1], ("tc", "start"))

    def test_no_host_named_runs_nothing(self):
        r = self.c.post("/ui/hosts/managed/start", headers=SAME, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn("no managed host named",
                      parse_qs(urlparse(r.headers["location"]).query)["msg"][0])
        self.assertEqual(self.actions, [])

    def test_get_runs_nothing(self):
        for p in ("start?host=tc", "stop?host=tc", "restart-service?host=tc&bid=comfyui:tc",
                  "resetup?host=tc&bid=comfyui:tc", "forget?host=tc", "delete?host=tc",
                  "save?host=tc&new=1&provider=thunder"):
            r = self.c.get(f"/ui/hosts/managed/{p}", headers=SAME, follow_redirects=False)
            self.assertEqual(r.status_code, 405, p)
            self.assertIn("/ui/backends", r.text, p)        # the 405 page leads back
        self.assertEqual((self.actions, self.saved_hosts, self.deleted_hosts), ([], [], []))

    def test_old_thunder_routes_are_gone(self):
        paths = {r.path for r in main.app.routes}
        self.assertFalse([p for p in paths if p.startswith("/ui/thunder")])
        self.assertFalse([p for p in admin._POST_ACTIONS if p.startswith("/ui/thunder")])


GiB = 10 ** 9


def _plan(**over) -> dict:
    """A controller `view()["plan"]` (hostctl.Controller._plan_view) with one alias of
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
    """Task 13: the model-sync half of the host card, its actions and the catalog."""

    def setUp(self):
        super().setUp()
        self.live = [{"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18100",
                      "host": "tc", "enabled": True, "healthy": True, "models": 0,
                      "source": "ui"}]
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
            self.assertIn('data-k="host-tc"', self.page())

    def test_sync_button_is_a_post_and_only_while_running(self):
        self.assertIn("/ui/hosts/managed/sync", admin._POST_ACTIONS)
        html = self.page()
        self.assertRegex(html, r'formaction="/ui/hosts/managed/sync\?host=tc"')
        self.views = {"tc": _view(phase="off", plan=_plan())}
        self.assertNotIn("/ui/hosts/managed/sync", self.page())

    def test_delete_unknown_is_post_with_confirm(self):
        self.assertIn("/ui/hosts/managed/delete-unknown", admin._POST_ACTIONS)
        html = self.main_html()
        form = re.search(r'<form[^>]*action="/ui/hosts/managed/delete-unknown"[^>]*>.*?</form>',
                         html, re.S).group(0)
        self.assertIn('method="post"', form)
        self.assertIn('name="host" value="tc"', form)
        self.assertIn('name="path" value="models/checkpoints/stray.ckpt"', form)
        self.assertIn('value="models/u/&lt;odd&gt;&amp;.bin"', form)
        btn = re.search(r"<button[^>]*data-confirm=\"([^\"]*)\"[^>]*>", form)
        self.assertIsNotNone(btn)
        self.assertIn("2 unknown files", btn.group(1))
        self.assertIn("6.0 GB", btn.group(1))
        # nothing unknown → no form; not running → no form (the controller refuses)
        self.views = {"tc": _view(phase="ready", plan=_plan(unknown=[]))}
        self.assertNotIn("/ui/hosts/managed/delete-unknown", self.page())
        self.views = {"tc": _view(phase="off", plan=_plan())}
        self.assertNotIn("/ui/hosts/managed/delete-unknown", self.page())

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
        det = re.search(r'<details[^>]*data-k="hosts-catalog"[^>]*>.*?</details>', html, re.S)
        self.assertIsNotNone(det)
        self.assertIn('action="/ui/hosts/managed/catalog"', det.group(0))
        self.assertIn('name="catalog"', det.group(0))
        self.assertIn("models/c/", det.group(0))


class SyncActions(Actions):
    def setUp(self):
        super().setUp()
        self.views = {"tc": _view(phase="ready", uuid="u1", plan=_plan())}
        self.live = [{"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18100",
                      "host": "tc", "enabled": True, "healthy": True, "models": 0,
                      "source": "ui"}]

    def test_sync_calls_the_controller(self):
        r = self.c.post("/ui/hosts/managed/sync?host=tc", headers=SAME, follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.synced, ["tc"])
        self.assertIn("sync requested", parse_qs(urlparse(r.headers["location"]).query)["msg"][0])

    def test_delete_unknown_posts_every_ticked_path(self):
        r = self.c.post("/ui/hosts/managed/delete-unknown", headers=SAME, follow_redirects=False,
                        data={"host": "tc", "path": ["models/a.bin", "models/b&c.bin"]})
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.deleted, [("tc", ["models/a.bin", "models/b&c.bin"])])
        self.assertIn("deleted 2", parse_qs(urlparse(r.headers["location"]).query)["msg"][0])

    def test_delete_unknown_without_selection_deletes_nothing(self):
        r = self.c.post("/ui/hosts/managed/delete-unknown", headers=SAME, follow_redirects=False,
                        data={"host": "tc"})
        self.assertEqual(r.status_code, 303)
        self.assertEqual(self.deleted, [])

    def test_get_runs_nothing(self):
        for p in ("sync?host=tc", "delete-unknown?host=tc&path=models/a.bin", "catalog"):
            r = self.c.get(f"/ui/hosts/managed/{p}", headers=SAME, follow_redirects=False)
            self.assertEqual(r.status_code, 405, p)
        self.assertEqual((self.synced, self.deleted, self.saved_catalogs), ([], [], []))

    def test_catalog_invalid_json_is_400_and_not_saved(self):
        typed = '[{"match": {"class": "X"}, "paths": ["models/<x>/"]'
        r = self.c.post("/ui/hosts/managed/catalog", headers=SAME, data={"catalog": typed})
        self.assertEqual(r.status_code, 400)
        body = r.text
        self.assertIn("models/&lt;x&gt;/", body)      # the textarea as typed …
        self.assertIn("not valid JSON", body)         # … and why
        self.assertNotIn("<main data-live", body)     # a refused form is never live
        self.assertRegex(body, r'<details[^>]*data-k="hosts-catalog"[^>]*\bopen\b')
        self.assertEqual(self.saved_catalogs, [])
        # valid JSON the validator refuses: every line of its answer, nothing saved
        typed = json.dumps([{"match": {"alias": "a"}, "paths": ["models/"]}, "junk"])
        r = self.c.post("/ui/hosts/managed/catalog", headers=SAME, data={"catalog": typed})
        self.assertEqual(r.status_code, 400)
        self.assertIn("is a whole root", r.text)
        self.assertIn("entry 2: not an object", r.text)
        self.assertEqual(self.saved_catalogs, [])

    def test_catalog_saved_and_rendered(self):
        cat = [{"match": {"class": "Trellis2LoadModel", "value": "org/repo"},
                "paths": ["models/org/repo/"]}]
        r = self.c.post("/ui/hosts/managed/catalog", headers=SAME, follow_redirects=False,
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
        # the card's actions go through main.host_action — the Thunder wrappers are gone
        self.assertIs(admin._host_action, main.host_action)
        self.assertIs(admin._host_names, main.host_names)
        self.assertIs(admin._host_view, main.host_view)
        self.assertIs(admin._save_managed_host, main.save_managed_host)
        self.assertIs(admin._delete_managed_host, main.delete_managed_host)
        self.assertIs(admin._managed_host_delete_refusal, main.managed_host_delete_refusal)
        for gone in ("thunder_action", "thunder_sync_now", "thunder_delete_unknown"):
            self.assertFalse(hasattr(main, gone), gone)
        for gone in ("_thunder_action", "_thunder_sync_now", "_thunder_delete_unknown",
                     "_thunder_names", "_thunder_view", "_thunder_longrun",
                     "_thunder_panel", "_thunder_card"):
            self.assertFalse(hasattr(admin, gone), gone)

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
    """Managed host `name` with the ComfyUI backend `name` (host `name`) as its service
    — the card name is the HOST name since managed hosts replaced the shim."""
    def __init__(self, ready=(), unknown_ok=True, phase="ready", name="tc"):
        self.ready, self.unknown_ok, self.phase = set(ready), unknown_ok, phase
        self.deleted, self.synced = [], 0
        self.name = name
        self.services = [{"name": name, "type": "comfyui", "host": name}]

    def has_service(self, bid):
        return bid == f"comfyui:{self.services[0]['name']}"

    def is_alias_ready(self, bid, alias):
        return self.has_service(bid) and alias in self.ready

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
        saved = dict(main.host_controllers), main.image_models, main.backends
        self.addCleanup(lambda: (main.host_controllers.clear(),
                                 main.host_controllers.update(saved[0]),
                                 setattr(main, "image_models", saved[1]),
                                 setattr(main, "backends", saved[2])))
        main.host_controllers.clear()
        main.image_models = {}
        # the gate note resolves backend → host → controller (R-W7): the live backends
        main.backends = [{"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18100",
                          "host": "tc"},
                         {"name": "k12", "type": "comfyui", "url": "http://127.0.0.1:18101",
                          "host": "k12"}]
        self.ctl = main.host_controllers["tc"] = _FakeCtl(ready={"done"})
        store.upsert("solo", [{"backend": "tc", "workflow_json": {}}])
        store.upsert("mixed", [{"backend": "tc", "workflow_json": {}},
                               {"backend": "k12", "workflow_json": {}}])
        store.upsert("done", [{"backend": "tc", "workflow_json": {}}])

    @staticmethod
    def delete(paths, name="tc"):
        return main.host_action(name, "delete_unknown", paths=paths)

    def test_gated_only_names_aliases_no_other_backend_serves(self):
        rows = main.host_view("tc")["plan"]["aliases"]
        self.assertEqual({a: r["gated_only"] for a, r in rows.items()},
                         {"solo": True, "mixed": False, "done": False})
        # a second managed host's ComfyUI that has not synced it either is no way out
        main.host_controllers["k12"] = _FakeCtl(name="k12")
        self.assertTrue(main.host_view("tc")["plan"]["aliases"]["mixed"]["gated_only"])
        self.assertIsNone(main.host_view("nope"))

    def test_delete_unknown_answers(self):
        self.assertIn("deleted 2 unknown files",
                      asyncio.run(self.delete(["models/a", "models/b"])))
        self.assertEqual(self.ctl.deleted, [["models/a", "models/b"]])
        self.ctl.unknown_ok = False
        self.assertIn("delete refused: not in the unknown list",
                      asyncio.run(self.delete(["models/a"])))
        self.assertIn("nothing deleted", asyncio.run(self.delete([])))
        self.assertIn("unknown managed host",
                      asyncio.run(self.delete(["models/a"], name="x")))

    def test_sync_now_answers(self):
        self.assertIn("sync", asyncio.run(main.host_action("tc", "sync")))
        self.ctl.phase = "off"
        self.assertIn("sync refused: no running instance (off)",
                      asyncio.run(main.host_action("tc", "sync")))


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
        import hostctl
        self.live = [{"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18100",
                      "host": "tc", "enabled": True, "healthy": True, "models": 0,
                      "source": "ui"}]
        self.views = {"tc": _view(phase="ready", uuid="u1")}
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.scans = []

        async def ssh(argv, stdin=None, timeout=60):
            self.scans.append(list(argv))
            return (0, f"192.168.8.24 ssh-ed25519 {_ED_B64}\n".encode(), b"")
        self.lan = hostctl.LanSource(self.d.name, host=lambda: "modelsrc@192.168.8.24",
                                        ssh=ssh)
        with open(self.lan.key_path + ".pub", "w") as f:
            f.write("ssh-ed25519 AAAAgatewaykey ai-hub\n")
        saved = (main.jobs_cfg, main._modelsrc_obj)
        self.addCleanup(lambda: (setattr(main, "jobs_cfg", saved[0]),
                                 setattr(main, "_modelsrc_obj", saved[1])))
        main.jobs_cfg = dict(main.jobs_cfg, store_path=os.path.join(self.d.name, "store.db"))
        main._modelsrc_obj = self.lan
        for k, v in {"_modelsrc_view": main.modelsrc_view,
                     "_modelsrc_scan": main.modelsrc_scan,
                     "_modelsrc_pin": main.modelsrc_pin}.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)
        self.fp = hostctl.host_key_fingerprint(_ED_B64)

    def block(self) -> str:
        m = re.search(r'<div class="tcard" data-k="hosts-modelsrc">.*?</div></div>',
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
        self.assertRegex(b, r'formaction="/ui/hosts/managed/modelsrc-scan"')
        self.assertNotIn("modelsrc-pin", b)               # nothing fetched: nothing to confirm
        self.assertNotRegex(b, r'<a[^>]*href="/ui/hosts/managed/')

    def test_pin_flow_is_post_and_writes_known_hosts(self):
        for p in ("/ui/hosts/managed/modelsrc-scan", "/ui/hosts/managed/modelsrc-pin"):
            self.assertIn(p, admin._POST_ACTIONS)
            r = self.c.get(p + "?fp=" + self.fp, headers=SAME, follow_redirects=False)
            self.assertEqual(r.status_code, 405, p)
        self.assertEqual(self.scans, [])
        msg = self.post("/ui/hosts/managed/modelsrc-scan")
        self.assertIn(self.fp, msg)
        self.assertEqual(self.scans, [["ssh-keyscan", "-t", "ed25519", "--", "192.168.8.24"]])
        self.assertFalse(os.path.exists(self.lan.known_hosts_path))     # not trusted yet
        b = self.block()
        self.assertIn(self.fp, b)
        btn = re.search(r'<button[^>]*formaction="/ui/hosts/managed/modelsrc-pin\?fp=([^"]+)"[^>]*>',
                        b)
        self.assertIsNotNone(btn)
        self.assertIn(f"data-confirm=\"Pin modelsrc@192.168.8.24&#x27;s host key {self.fp}?",
                      btn.group(0))
        # a confirmation of another key pins nothing
        self.assertIn("not pinned", self.post("/ui/hosts/managed/modelsrc-pin", fp="SHA256:other"))
        self.assertFalse(os.path.exists(self.lan.known_hosts_path))
        from urllib.parse import unquote
        msg = self.post("/ui/hosts/managed/modelsrc-pin", fp=unquote(btn.group(1)))
        self.assertIn("pinned", msg)
        with open(self.lan.known_hosts_path) as f:
            self.assertEqual(f.read(), f"192.168.8.24 ssh-ed25519 {_ED_B64}\n")
        self.assertEqual(os.stat(self.lan.known_hosts_path).st_mode & 0o777, 0o600)
        b = self.block()
        self.assertNotIn("not set up", b)
        self.assertIn(f"host key <code>{self.fp}</code> pinned", b)

    def test_no_lan_block_without_thunder_backends(self):
        self.views = {}
        self.assertNotIn("hosts-modelsrc", self.page())


class LanSourceWiring(CatalogWiring):
    def test_bound_and_deps(self):
        import hostctl
        self.assertIs(admin._modelsrc_view, main.modelsrc_view)
        self.assertIs(admin._modelsrc_scan, main.modelsrc_scan)
        self.assertIs(admin._modelsrc_pin, main.modelsrc_pin)
        # the host rule is main's ship-target rule (plain [user@]host characters)
        self.assertEqual(hostctl._SRC_HOST_RE.pattern, main._VOICE_HOST_RE.pattern)
        self.assertEqual(main._modelsrc_host(), "modelsrc@192.168.8.24")
        store.set_settings({"modelsrc_host": "src@10.0.0.2"})
        self.assertEqual(main._modelsrc_host(), "src@10.0.0.2")
        saved = main.jobs_cfg
        self.addCleanup(setattr, main, "jobs_cfg", saved)
        main.jobs_cfg = dict(saved, store_path=os.path.join(self.tmp.name, "store.db"))
        deps = main._host_deps()
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
        r = self.c.post("/ui/hosts/managed/hf-token", data=data, headers=SAME, follow_redirects=False)
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
        self.assertIn("/ui/hosts/managed/hf-token", admin._POST_ACTIONS)
        r = self.c.get("/ui/hosts/managed/hf-token", headers=SAME, follow_redirects=False)
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
        import hostctl
        for bad in ('hf_a"b', "hf_a\\b", "hf_ä", "h" * 513):
            with self.subTest(bad=bad):
                self.assertFalse(hostctl.hf_token_ok(bad))
                self.post(status=400, hf_token=bad)
                self.assertEqual(main._thunder_hf_token(), "hf_Good1")
        self.assertTrue(hostctl.hf_token_ok("hf_Good1"))
        self.assertRegex(r.text, r'<details class="optblock" data-k="hosts-catalog" open>')


class ModelsrcHostField(Actions):
    """Task 16: `modelsrc_host` is set in the console, next to the LAN block. A value
    that is no plain [user@]host is refused with the form as typed (it would reach an ssh
    argv); a changed host drops the old share's listing and says the pin is for the old
    host — otherwise every list fails as "unreachable" and points at the network."""

    def setUp(self):
        super().setUp()
        self.live = [{"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18100",
                      "host": "tc", "enabled": True, "healthy": True, "models": 0,
                      "source": "ui"}]
        self.views = {"tc": _view(phase="ready", uuid="u1")}
        saved = (main.jobs_cfg, main._modelsrc_obj)
        self.addCleanup(lambda: (setattr(main, "jobs_cfg", saved[0]),
                                 setattr(main, "_modelsrc_obj", saved[1])))
        main.jobs_cfg = dict(main.jobs_cfg, store_path=os.path.join(self.tmp.name, "store.db"))
        main._modelsrc_obj = None
        for k, v in {"_modelsrc_view": main.modelsrc_view,
                     "_save_modelsrc_host": main.save_modelsrc_host}.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)

    def post(self, status=303, **data):
        r = self.c.post("/ui/hosts/managed/modelsrc-host", data=data, headers=SAME,
                        follow_redirects=False)
        self.assertEqual(r.status_code, status, r.text[:300])
        return r

    def test_bound_and_post_only(self):
        self.assertIs(admin._save_modelsrc_host, main.save_modelsrc_host)
        self.assertIn("/ui/hosts/managed/modelsrc-host", admin._POST_ACTIONS)
        r = self.c.get("/ui/hosts/managed/modelsrc-host?modelsrc_host=x", headers=SAME,
                       follow_redirects=False)
        self.assertEqual(r.status_code, 405)

    def test_field_shows_the_current_host(self):
        page = self.page()
        self.assertRegex(page, r'<form method="post" action="/ui/hosts/managed/modelsrc-host"')
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
        m = re.search(r'<div class="tcard" data-k="hosts-modelsrc">.*?</div></div>',
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
                     "_host_longrun": longrun, "_host_view": heavy}.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)
        r = asyncio.run(admin.dashboard_page(_Req({})))
        return r.body.decode()

    def test_card_banner(self):
        self.views = {"tc": self.long_view()}
        page = self.page()
        self.assertRegex(page, r'data-k="host-tc-longrun"')
        self.assertIn("Thunder Compute host tc running for 26 h (≈ $13.02)", page)
        self.views = {"tc": _view(phase="ready", uuid="u1", started_at=1.0,
                                  uptime_s=3600, long_running=False)}
        self.assertNotIn("longrun", self.page())

    def test_dashboard_banner_after_24h(self):
        self.views = {"tc": self.long_view(), "b": _view(name="b")}
        html = self.dash()
        m = re.search(r'<p class="bad" data-k="dash-longrun-tc">.*?</p>', html, re.S)
        self.assertIsNotNone(m, html[:2000])
        self.assertIn("Thunder Compute host tc running for 26 h (≈ $13.02)", m.group(0))
        self.assertIn('href="/ui/backends"', m.group(0))
        self.assertNotIn("dash-longrun-b", html)
        self.assertIn('data-live="4"', html)                 # the Dashboard stays live
        # no price list yet: still the banner, no made-up figure
        self.views = {"tc": self.long_view(session_cost=None, cost_per_h=None)}
        self.assertIn("Thunder Compute host tc running for 26 h (cost unknown", self.dash())
        # under 24 h, or nothing running: no banner at all
        self.views = {"tc": _view(phase="ready", started_at=1.0, uptime_s=3600)}
        self.assertNotIn("dash-longrun", self.dash())

    def test_dashboard_banner_reads_no_store(self):
        """main.host_longrun is what the Dashboard polls every 4 s: Controller.view()
        only — host_view's alias-gate note would read the store per not-ready alias."""
        class Ctl:
            def __init__(self, long, name):
                self.long = long
                self.services = [{"name": name, "type": "comfyui"}]

            def has_service(self, bid):
                return bid == f"comfyui:{self.services[0]['name']}"

            def view(self):
                return {"phase": "ready", "provider": "thunder", "long_running": self.long,
                        "uptime_s": 90000, "session_cost": 1.5,
                        "plan": {"aliases": {"a": {"ready": False}, "b": {"ready": False}}}}
        saved = dict(main.host_controllers)
        self.addCleanup(lambda: (main.host_controllers.clear(),
                                 main.host_controllers.update(saved)))
        main.host_controllers.clear()
        main.host_controllers.update({"tc": Ctl(True, "tc"), "k2": Ctl(False, "k2")})

        def boom(*a, **k):
            raise AssertionError("store read on the Dashboard path")
        for fn in ("get", "list_aliases", "get_setting", "get_settings"):
            self.addCleanup(setattr, store, fn, getattr(store, fn))
            setattr(store, fn, boom)
        self.assertIs(admin._host_longrun, main.host_longrun)
        self.assertEqual([n for n, _ in main.host_longrun()], ["tc"])
        html = admin._dash_hosts()
        self.assertIn('data-k="dash-longrun-tc"', html)
        self.assertIn("Thunder Compute host tc running for 25 h (≈ $1.50)", html)
        self.assertNotIn("k2", html)


class OrphanSnapshots(_Base):
    """Task 16: `aihub-` snapshots no current managed host owns bill $/month unseen
    (a deleted host leaves every old one behind) — shown on the Backends tab, never
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
        m = re.search(r'<div data-k="hosts-orphan-snaps">.*?</table></div>', page, re.S)
        self.assertIsNotNone(m)
        self.assertIn("aihub-old-20260926t120000z", m.group(0))
        self.assertIn("$1.00/month", m.group(0))
        self.assertIn('data-k="hosts-osnap-s2"', m.group(0))
        self.assertNotIn("<button", m.group(0))              # display only, no delete
        self.orph = []
        self.assertNotIn("hosts-orphan-snaps", self.page())

    def test_orphan_instances_priced_on_the_card(self):
        self.orph = []
        self.views = {"tc": _view(orphans=[
            {"uuid": "x", "index": "4", "status": "RUNNING", "gpu_type": "a6000",
             "num_gpus": 1, "cost_per_h": 0.42},
            {"uuid": "y", "index": "5", "status": "RUNNING", "gpu_type": "h200",
             "num_gpus": 2, "cost_per_h": None}])}
        page = self.page()
        x = re.search(r'<tr data-k="host-tc-orphan-x">.*?</tr>', page).group(0)
        self.assertIn("$0.42/h", x)
        self.assertIn("a6000 ×1", x)
        y = re.search(r'<tr data-k="host-tc-orphan-y">.*?</tr>', page).group(0)
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
        saved = dict(main.host_controllers)
        self.addCleanup(lambda: (main.host_controllers.clear(),
                                 main.host_controllers.update(saved)))
        main.host_controllers.clear()
        # ownership is by HOST name (R-W4): the snapshots of host `thunder-tc`
        main.host_controllers["thunder-tc"] = Ctl(
            [snap("aihub-thunder-tc-20260926t120000z", "a"),
             snap("aihub-gone-20260926t120000z", "b")], None)
        main.host_controllers["thunder-k2"] = Ctl(
            [snap("aihub-gone-20260926t120000z", "b"),
             snap("aihub-thunder-k2-20260926t120000z", "c")], {"snapshot_gb": 0.001})
        out = main.thunder_orphan_snapshots()
        self.assertEqual([o["id"] for o in out], ["b"])      # once, though both list it
        self.assertAlmostEqual(out[0]["monthly"], 100 * 0.001 * 730)


class Task16Wiring(unittest.TestCase):
    def test_bound(self):
        self.assertIs(admin._thunder_orphan_snapshots, main.thunder_orphan_snapshots)


# ── the managed-host form, its save and delete (Task 7) ──────────────────────────────

def _host_form(name="gpu-a", new=True, **over) -> dict:
    f = {"host": name, "provider": "thunder", "opt__gpu_type": "h100", "opt__num_gpus": "2",
         "opt__vcpus": "16", "opt__bootstrap_template": "", "opt__reserve_gb": "30",
         "opt__comfy_commit": "", "opt__nodes": "registry:y@2\n", "api_key": ""}
    if new:
        f["new"] = "1"
    f.update(over)
    return f


class HostForm(_Base):
    """The "+ Managed host" form: every provider option comes from the provider's own
    OPTION_FIELDS, each field exactly once, the token never rendered."""

    def form(self, qp) -> str:
        html = self.page(qp)
        m = re.search(r'<form action="/ui/hosts/managed/save" method="post"[^>]*>.*?</form>',
                      html, re.S)
        self.assertIsNotNone(m, html[-3000:])
        return m.group(0)

    def names(self, form) -> list:
        return re.findall(r'<(?:input|select|textarea)[^>]*\bname="([^"]+)"', form)

    def test_add_button_in_the_hosts_area(self):
        html = self.page()
        self.assertIn('href="/ui/backends?mhost_new=1"', html)
        self.assertIn("+ Managed host", html)
        # the button is there with no managed host at all (how else to make the first)
        self.views = {}
        self.assertIn('href="/ui/backends?mhost_new=1"', self.page())

    def test_new_form_renders_the_providers_option_fields(self):
        import hostapi
        import thunder
        f = self.form({"mhost_new": "1"})
        names = self.names(f)
        want = (["new", "host", "provider"] + [f"opt__{x['key']}" for x in thunder.OPTION_FIELDS]
                + ["api_key", "api_key_clear"])
        self.assertEqual(sorted(names), sorted(want))             # each field exactly once
        self.assertEqual(len(names), len(set(names)))
        # "Provider": the providers of hostapi.PROVIDERS, by their display NAME — an
        # English console, the label too (Ruling M6)
        self.assertRegex(f, r'<label[^>]*>Provider</label>')
        self.assertNotIn("Steuerung", self.page({"mhost_new": "1"}))
        self.assertNotIn("Steuerung", self.page({}))
        sel = re.search(r'<select name="provider"[^>]*>.*?</select>', f, re.S).group(0)
        for kind, (mod, _api) in hostapi.PROVIDERS.items():
            self.assertIn(f'<option value="{kind}" selected>{mod.NAME}</option>', sel)
        # every select offers exactly the provider's choices
        for fld in thunder.OPTION_FIELDS:
            if fld["type"] != "select":
                continue
            s = re.search(rf'<select name="opt__{fld["key"]}"[^>]*>(.*?)</select>', f, re.S)
            self.assertEqual(re.findall(r'<option value="([^"]*)"', s.group(1)),
                             [str(c) for c in fld["choices"]], fld["key"])
        # the textareas and ints carry the defaults, the nodes the default node list
        self.assertIn(f'name="opt__comfy_commit" value="{SHA}"', f)
        self.assertIn('name="opt__vcpus" value="8"', f)
        self.assertIn("# defaults\nregistry:x@1.0\n</textarea>", f)
        self.assertIn("40-hex", f)                               # the provider's hints

    def test_template_select_offers_auto_first_and_stores_blank(self):
        # Ruling M5: "auto" first, the stored value "" — a fixed comfy-ui default gave
        # every vLLM-only host the template's ComfyUI and its models
        f = self.form({"mhost_new": "1"})
        s = re.search(r'<select name="opt__bootstrap_template"[^>]*>(.*?)</select>', f, re.S)
        opts = re.findall(r'<option value="([^"]*)"( selected)?>([^<]*)</option>', s.group(1))
        self.assertEqual(opts[0], ("", " selected", "auto"))
        self.assertEqual([o[0] for o in opts], ["", "comfy-ui", "base"])

    def test_edit_form_shows_the_stored_options_and_no_rename(self):
        self.views = {"tc": _view(api_key_set=True, options={
            "gpu_type": "h100", "num_gpus": 2, "vcpus": 16, "bootstrap_template": "base",
            "reserve_gb": 30, "comfy_commit": SHA, "nodes": ["# c", "registry:y@2"]})}
        f = self.form({"mhost": "tc"})
        self.assertRegex(f, r'<option value="h100" selected>')
        self.assertRegex(f, r'<option value="base" selected>')
        for v in ('name="opt__num_gpus" value="2"', 'name="opt__vcpus" value="16"',
                  'name="opt__reserve_gb" value="30"'):
            self.assertIn(v, f)
        self.assertIn("# c\nregistry:y@2</textarea>", f)
        self.assertNotIn("# defaults", f)                   # stored [] or list: never refilled
        # no rename (R-W5): the name travels hidden, the provider cannot change
        self.assertIn('<input type="hidden" name="host" value="tc">', f)
        self.assertIn('<input type="hidden" name="provider" value="thunder">', f)
        self.assertNotIn('name="new"', f)
        self.assertNotRegex(f, r'<select name="provider"')
        self.assertIn("set — blank keeps it", f)

    def test_token_never_rendered(self):
        store.set_managed_host("tc", {"provider": "thunder", "options": {},
                                      "api_key": "th-SECRET-TOKEN"})
        self.views = {"tc": _view(api_key_set=True)}
        for qp in ({}, {"mhost": "tc"}, {"mhost_new": "1"}):
            page = self.page(qp)
            self.assertNotIn("th-SECRET-TOKEN", page, qp)
        f = self.form({"mhost": "tc"})
        self.assertRegex(f, r'<input type="password" name="api_key" value=""')

    def test_unknown_host_says_so(self):
        self.assertIn("no managed host named", self.page({"mhost": "nope"}))


class _HostSaveBase(Actions):
    def post(self, form, status=303):
        r = self.c.post("/ui/hosts/managed/save", data=form, headers=SAME,
                        follow_redirects=False)
        self.assertEqual(r.status_code, status, r.text[-1500:])
        return r


class HostSave(_HostSaveBase):
    def test_create_hands_main_the_typed_options_and_token(self):
        r = self.post(_host_form(api_key="th-TOKEN-1"))
        self.assertIn("saved", parse_qs(urlparse(r.headers["location"]).query)["msg"][0])
        name, entry, new = self.saved_hosts[-1]
        self.assertEqual((name, new), ("gpu-a", True))
        self.assertEqual(entry["provider"], "thunder")
        self.assertEqual(entry["api_key"], "th-TOKEN-1")
        self.assertEqual(entry["options"]["gpu_type"], "h100")
        self.assertEqual(entry["options"]["vcpus"], "16")    # main normalizes (options_of)

    def test_blank_token_keeps_the_stored_one_and_the_box_clears_it(self):
        store.set_managed_host("gpu-a", {"provider": "thunder", "options": {},
                                         "api_key": "th-OLD"})
        self.post(_host_form(new=False))
        self.assertEqual(self.saved_hosts[-1][1]["api_key"], "th-OLD")
        self.post(_host_form(new=False, api_key="th-NEW"))
        self.assertEqual(self.saved_hosts[-1][1]["api_key"], "th-NEW")
        self.post(_host_form(new=False, api_key_clear="1"))
        self.assertEqual(self.saved_hosts[-1][1]["api_key"], "")
        # a typed value next to a ticked box: the value wins (the backend-key rule)
        self.post(_host_form(new=False, api_key="th-X", api_key_clear="1"))
        self.assertEqual(self.saved_hosts[-1][1]["api_key"], "th-X")
        self.assertEqual({x[2] for x in self.saved_hosts}, {False})

    def test_blank_token_with_an_unreadable_store_is_refused(self):
        # "blank keeps it" cannot keep a token it could not read: refused, not cleared
        store.set_managed_host("gpu-a", {"provider": "thunder", "options": {},
                                         "api_key": "th-OLD"})
        self.views = {"gpu-a": _view(api_key_set=True)}
        real, calls = store.get_managed_hosts, []

        def flaky():                    # the save's own read fails, the re-render reads
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("db")
            return real()
        with mock.patch.object(store, "get_managed_hosts", side_effect=flaky):
            r = self.post(_host_form(new=False, opt__vcpus="24"), status=400)
            self.assertIn("could not read the stored host — token not changed", r.text)
            self.assertIn('name="opt__vcpus" value="24"', r.text)     # the form as typed
            self.assertEqual(self.saved_hosts, [])
            # a typed token (or the clear box) needs no read: saved
            calls.clear()
            self.post(_host_form(new=False, api_key="th-NEW"))
        self.assertEqual(self.saved_hosts[-1][1]["api_key"], "th-NEW")

    def test_refused_save_is_400_with_the_form_as_typed(self):
        self.save_refusal = "vcpus: 'lots' is not a whole number ≥ 1"
        r = self.post(_host_form(opt__vcpus="lots", opt__bootstrap_template="auto",
                                 api_key="th-TYPED-SECRET"), status=400)
        body = r.text
        self.assertIn("is not a whole number", body)
        self.assertIn('name="opt__vcpus" value="lots"', body)
        self.assertIn('name="host" value="gpu-a"', body)
        self.assertRegex(body, r'<option value="h100" selected>')
        self.assertRegex(body, r'<option value="" selected>auto</option>')
        self.assertNotIn("th-TYPED-SECRET", body)                # never echoed
        self.assertNotIn("<main data-live", body)               # a refused form is static
        self.assertEqual(self.saved_hosts, [])


class HostSaveMain(_HostSaveBase):
    """The same form against main's real save (store + refusal rules)."""

    def setUp(self):
        super().setUp()
        saved = (main.backends, main.hosts_meta, dict(main.managed_hosts))
        self.addCleanup(lambda: (setattr(main, "backends", saved[0]),
                                 setattr(main, "hosts_meta", saved[1]),
                                 main.managed_hosts.clear(),
                                 main.managed_hosts.update(saved[2])))
        applied = []
        self.addCleanup(setattr, main, "apply_managed_hosts", main.apply_managed_hosts)
        main.apply_managed_hosts = lambda: applied.append(1)
        for k, v in {"_save_managed_host": main.save_managed_host,
                     "_delete_managed_host": main.delete_managed_host,
                     "_managed_host_delete_refusal": main.managed_host_delete_refusal}.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)
        main.backends = [{"name": "k12box", "type": "comfyui", "url": "http://k12:8188"}]

    def raw(self):
        import sqlite3
        with sqlite3.connect(store._DB_PATH) as c:
            row = c.execute("SELECT value_json FROM settings WHERE key='managed_hosts'").fetchone()
        return "" if row is None else row[0]

    def test_create_stores_normalized_options_and_an_encrypted_token(self):
        self.post(_host_form(opt__bootstrap_template="auto", api_key="th-SECRET-9"))
        e = store.get_managed_hosts()["gpu-a"]
        self.assertEqual(e["provider"], "thunder")
        self.assertEqual(e["api_key"], "th-SECRET-9")
        self.assertEqual((e["options"]["vcpus"], e["options"]["bootstrap_template"]), (16, ""))
        self.assertEqual(e["options"]["nodes"], ["registry:y@2"])
        self.assertNotIn("th-SECRET-9", self.raw())              # R-W10: encrypted at rest

    def test_collisions_are_refused_as_typed(self):
        # R-W6: an URL hostname without a dot names a host too
        r = self.post(_host_form(name="k12"), status=400)
        self.assertIn("already runs on a host named", r.text)
        self.assertIn('name="host" value="k12"', r.text)
        store.set_host("box7", {"label": "Box 7"})
        r = self.post(_host_form(name="box7"), status=400)
        self.assertIn("already a host in the Hosts list", r.text)
        self.post(_host_form(name="gpu-a"))
        r = self.post(_host_form(name="gpu-a"), status=400)
        self.assertIn("already exists", r.text)
        r = self.post(_host_form(name="Bad_Name"), status=400)
        self.assertIn("a-z, 0-9", r.text)
        r = self.post(_host_form(name="gpu-b", opt__comfy_commit="master"), status=400)
        self.assertIn("40-hex", r.text)
        self.assertIn('name="opt__comfy_commit" value="master"', r.text)
        r = self.post(_host_form(name="gpu-c", provider="runpod"), status=400)
        self.assertIn("unknown provider", r.text)
        self.assertEqual(sorted(store.get_managed_hosts()), ["gpu-a"])

    def test_provider_cannot_change(self):
        self.post(_host_form())
        r = self.post(_host_form(new=False, provider="runpod"), status=400)
        self.assertIn("unknown provider", r.text)
        self.assertEqual(store.get_managed_hosts()["gpu-a"]["provider"], "thunder")

    def test_blank_token_keeps_the_stored_one(self):
        self.post(_host_form(api_key="th-OLD"))
        self.post(_host_form(new=False, opt__vcpus="32"))
        e = store.get_managed_hosts()["gpu-a"]
        self.assertEqual((e["api_key"], e["options"]["vcpus"]), ("th-OLD", 32))
        self.post(_host_form(new=False, api_key_clear="1"))
        self.assertEqual(store.get_managed_hosts()["gpu-a"]["api_key"], "")


class HostDelete(Actions):
    def delete(self, name, status=303):
        r = self.c.post(f"/ui/hosts/managed/delete?host={name}", headers=SAME,
                        follow_redirects=False)
        self.assertEqual(r.status_code, status, r.text[-800:])
        return r

    def test_delete_only_when_main_allows_it(self):
        self.views = {"tc": _view(phase="off")}
        html = self.page()
        m = re.search(r'<button[^>]*formaction="/ui/hosts/managed/delete\?host=tc"[^>]*>', html)
        self.assertIsNotNone(m)
        self.assertIn("data-confirm=", m.group(0))
        # the snapshots it left bill on at the provider until deleted by hand — never
        # "listed as foreign": with the LAST host gone nothing lists them any more
        conf = html_mod.unescape(re.search(r'data-confirm="([^"]*)"', m.group(0)).group(1))
        self.assertEqual(conf, "Delete the managed host tc? Its state goes; its READY "
                               "snapshots bill on at Thunder Compute until deleted by hand.")
        r = self.delete("tc")
        self.assertIn("deleted", parse_qs(urlparse(r.headers["location"]).query)["msg"][0])
        self.assertEqual(self.deleted_hosts, ["tc"])
        # refused (running, a backend still names it …): no button, and a POST is a 400
        # naming the reason — nothing deleted
        self.delete_refusals["tc"] = "tc is ready — stop it first; only an off host can be deleted"
        self.views = {"tc": _view(phase="ready", uuid="u1")}
        self.assertNotIn("/ui/hosts/managed/delete?", self.page())
        r = self.delete("tc", status=400)
        self.assertIn("stop it first", r.text)
        self.assertNotIn("<main data-live", r.text)
        self.assertEqual(self.deleted_hosts, ["tc"])

    def test_off_host_says_why_it_cannot_be_deleted(self):
        self.delete_refusals["tc"] = ("backends still name tc as their host (comfyui:tc) — "
                                      "move or delete them first")
        self.views = {"tc": _view(phase="off")}
        self.assertIn("move or delete them first", self.page())


class HostDeleteMain(HostDelete):
    """main's real rule behind the button (R-W5), incl. the undriven host."""

    def setUp(self):
        super().setUp()
        saved = (main.backends, dict(main.managed_hosts), dict(main.host_controllers))
        self.addCleanup(lambda: (setattr(main, "backends", saved[0]),
                                 main.managed_hosts.clear(), main.managed_hosts.update(saved[1]),
                                 main.host_controllers.clear(),
                                 main.host_controllers.update(saved[2])))
        self.addCleanup(setattr, main, "apply_managed_hosts", main.apply_managed_hosts)
        main.apply_managed_hosts = lambda: None
        main.backends = []
        main.host_controllers.clear()
        for k, v in {"_delete_managed_host": main.delete_managed_host,
                     "_managed_host_delete_refusal": main.managed_host_delete_refusal}.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)

    def test_delete_only_when_main_allows_it(self):
        store.set_managed_host("tc", {"provider": "thunder", "options": {}, "api_key": ""})
        main.backends = [{"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18100",
                          "host": "tc"}]
        self.views = {"tc": _view(phase="off")}
        self.assertNotIn("/ui/hosts/managed/delete?", self.page())
        r = self.delete("tc", status=400)
        self.assertIn("move or delete them first", r.text)
        main.backends = []
        self.assertIn("/ui/hosts/managed/delete?host=tc", self.page())
        self.delete("tc")
        self.assertNotIn("tc", store.get_managed_hosts())

    def test_undriven_host_with_a_live_record_is_kept(self):
        # an unknown provider (never driven): its stored record is the only pointer to
        # an instance that may still bill
        store.set_managed_host("pod", {"provider": "runpod", "options": {}, "api_key": ""})
        store.set_settings({"host_state": {"pod": {"phase": "ready", "uuid": "u-1"}}})
        self.views = {"pod": _view(name="pod", provider="runpod", phase="off",
                                   error="unknown provider 'runpod'")}
        self.assertNotIn("/ui/hosts/managed/delete?", self.page())
        r = self.delete("pod", status=400)
        self.assertIn("not driven", r.text)
        self.assertIn("pod", store.get_managed_hosts())
        store.set_settings({"host_state": {"pod": {"phase": "off"}}})
        self.delete("pod")
        self.assertNotIn("pod", store.get_managed_hosts())

    def test_off_host_says_why_it_cannot_be_deleted(self):
        store.set_managed_host("tc", {"provider": "thunder", "options": {}, "api_key": ""})
        main.backends = [{"name": "tc", "type": "comfyui", "host": "tc"}]
        self.views = {"tc": _view(phase="off")}
        self.assertIn("move or delete them first", self.page())


class ServiceTable(_Base):
    """The card's service table: every attached backend with its ports, status and
    error, Restart / Re-run setup per SERVICE (POST with the backend id)."""

    def row(self, html, bid):
        m = re.search(rf'<tr data-k="host-tc-svc-{re.escape(bid)}">(.*?)</tr>', html, re.S)
        self.assertIsNotNone(m, bid)
        return m.group(1)

    def test_rows_with_ports_status_error_and_buttons(self):
        self.views = {"tc": _view(phase="ready", uuid="u1", services=_svcs(**{
            "openai:vllm": {"name": "vllm", "type": "openai", "local_port": 18101,
                            "remote_port": 8000, "status": "setup failed",
                            "error": "setup rc 1: <pip> died"}}))}
        html = self.page()
        c = self.row(html, "comfyui:tc")
        self.assertIn("8188", c)
        self.assertIn("18100", c)
        self.assertIn('<span class="badge ok">up</span>', c)
        v = self.row(html, "openai:vllm")
        self.assertIn("8000", v)
        self.assertIn("18101", v)
        self.assertIn("setup failed", v)
        self.assertIn("setup rc 1: &lt;pip&gt; died", v)
        self.assertNotIn("<pip>", html)
        for p in ("restart-service", "resetup"):
            self.assertRegex(v, rf'formaction="/ui/hosts/managed/{p}\?host=tc&amp;'
                                r'bid=openai%3Avllm"')
        m = re.search(r'<main[^>]*>(.*)</main>', html, re.S)
        self.assertNotIn("<script", m.group(1))

    def test_listener_warning_and_restart_pending(self):
        # Ruling M6: a service reachable from outside the VM says so on its row; an
        # automatic restart waiting for requests is `restart pending` (warn)
        self.views = {"tc": _view(phase="ready", uuid="u1", services=_svcs(**{
            "openai:vllm": {"name": "vllm", "type": "openai", "local_port": 18101,
                            "remote_port": 8000, "status": "up", "error": "",
                            "warning": "listening on all interfaces (0.0.0.0:8000) — "
                                       "reachable from outside the VM; bind to 127.0.0.1"},
            "openai:emb": {"name": "emb", "type": "openai", "local_port": 18102,
                           "remote_port": 8001, "status": "restart pending",
                           "error": "2 request(s) in flight"}}))}
        html = self.page()
        v = self.row(html, "openai:vllm")
        self.assertIn('<span class="warn" data-k="host-tc-svcwarn-openai:vllm">⚠ listening '
                      "on all interfaces (0.0.0.0:8000) — reachable from outside the VM; "
                      "bind to 127.0.0.1</span>", v)
        self.assertNotIn("svcwarn", self.row(html, "comfyui:tc"))
        self.assertIn('<span class="badge warn">restart pending</span>',
                      self.row(html, "openai:emb"))

    def test_no_buttons_without_a_running_instance(self):
        self.views = {"tc": _view(phase="off", services=_svcs())}
        html = self.page()
        self.row(html, "comfyui:tc")
        self.assertNotIn("restart-service", html)
        self.assertNotIn("/ui/hosts/managed/resetup", html)

    def test_drain_per_service_and_not_attachable(self):
        self.views = {"tc": _view(
            phase="draining", op="stopping", uuid="u1", services=_svcs(),
            waiting_jobs={"comfyui:tc": 3},
            not_attachable=[{"bid": "openai:cfg",
                             "reason": "config-defined backend — create it in the console"}])}
        html = self.page()
        self.assertIn("3 jobs waiting", self.row(html, "comfyui:tc"))
        na = re.search(r'<p[^>]*data-k="host-tc-na-openai:cfg">.*?</p>', html, re.S).group(0)
        self.assertIn("not attachable", na)
        self.assertIn("create it in the console", na)

    def test_host_without_services_says_how_to_attach(self):
        self.views = {"tc": _view()}
        self.assertIn("No backend attached", self.page())


class HostsPanel(_Base):
    def test_every_managed_host_is_listed_also_without_comfyui(self):
        # spec: "_hosts_panel zeigt jeden gesteuerten Host, auch ohne ComfyUI-Dienst"
        self.live = [{"name": "vllm", "type": "openai", "url": "http://127.0.0.1:18101",
                      "host": "vm2", "enabled": True, "healthy": True, "models": 1,
                      "source": "ui"}]
        self.views = {"vm2": _view(name="vm2"), "vm3": _view(name="vm3")}
        html = self.page()
        panel = html[html.index("Hosts · GPU policy"):]
        self.assertIn("vm2", panel)
        self.assertIn("vllm (openai)", panel)
        self.assertIn("vm3", panel)
        self.assertIn("no backend attached", panel)
        self.assertIn('href="/ui/backends?mhost=vm2"', panel)


if __name__ == "__main__":
    unittest.main()
