"""The Server tab: four sub-tabs (Runtime | Restart | API Keys | Models), every secret the
server holds in ONE place, and the gateway-wide model settings (LAN model source, model
catalog) in another.

Why this fails SILENTLY:
- A sub-tab whose form renders on the wrong tab, or a Save that lands on another tab,
  looks like a setting that "did not save": the operator sees the old value on the tab
  in front of them. A pending restart shown only on the Restart tab is never seen by
  the operator who stays on Runtime.
- A secret row that renders its value hands the key to every screenshot; a blank Save
  that CLEARS instead of keeping it locks every client out (master key) or turns every
  provider call into a 401 — and one row's Save that touched another row's key does
  the same to a key nobody edited.
- A master key saved on the new tab that no longer revokes the sessions opened with
  the old one leaves a rotated key's console sessions valid for 12 h.
- A link to "the setting in the Server tab" that opens the DEFAULT sub-tab sends the
  operator looking for a field that is not on the page.
- The LAN model source and the catalog are one per GATEWAY (every managed host of every
  provider reads them): an action that still lands on the Backends tab shows its answer
  where the block no longer is — the operator sees no result and presses again.
- `server_save` turned "1.5" in an int field (or "abc") into "" = unset, i.e. the
  default — a cap the operator typed silently became no cap.
"""
import asyncio
import html
import os
import re
import sys
import tempfile
import threading
import unittest
from urllib.parse import parse_qs, urlparse

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
    import hostapi
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

from fastapi.testclient import TestClient  # noqa: E402

SAME = {"sec-fetch-site": "same-origin"}


class _Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        saved = (store._DB_PATH, store._active, store._MASTER_KEY)
        self.addCleanup(lambda: (setattr(store, "_DB_PATH", saved[0]),
                                 setattr(store, "_active", saved[1]),
                                 setattr(store, "_MASTER_KEY", saved[2])))
        store._MASTER_KEY = os.urandom(32)
        store.init(os.path.join(self.tmp.name, "store.db"))
        saved_auth = (main.api_key, main.users, main._users_by_key)
        main.api_key, main.users, main._users_by_key = "", [], {}
        self.addCleanup(lambda: (setattr(main, "api_key", saved_auth[0]),
                                 setattr(main, "users", saved_auth[1]),
                                 setattr(main, "_users_by_key", saved_auth[2])))
        # the live globals apply_server_settings writes — restored after each test
        names = ("health_check_interval", "max_concurrent_default", "park_timeout_s",
                 "max_parked", "max_queued_gen", "affinity_max_wait_s",
                 "fast_probe_interval_s", "log_per_call", "model_prefix")
        snap = {n: getattr(main, n) for n in names}
        stats_cfg, jobs_cfg = dict(main.stats_cfg), dict(main.jobs_cfg)

        def restore():
            for n, v in snap.items():
                setattr(main, n, v)
            main.stats_cfg.clear()
            main.stats_cfg.update(stats_cfg)
            main.jobs_cfg.clear()
            main.jobs_cfg.update(jobs_cfg)
        self.addCleanup(restore)
        self.applied_hosts = []
        self.addCleanup(setattr, main, "apply_managed_hosts", main.apply_managed_hosts)
        main.apply_managed_hosts = lambda: self.applied_hosts.append(1)
        self.c = TestClient(main.app)

    def login(self):
        """A console session for the CURRENT master key (a set key locks /ui)."""
        self.c.cookies.set(admin._SESSION_COOKIE, admin._make_session(main._MASTER_ADMIN))

    def get(self, url="/ui/server", **kw):
        r = self.c.get(url, headers=SAME, **kw)
        self.assertEqual(r.status_code, 200, r.text[-400:])
        return r.text

    def post(self, url, data, status=303, **kw):
        r = self.c.post(url, data=data, headers=SAME, follow_redirects=False, **kw)
        self.assertEqual(r.status_code, status, r.text[-600:])
        return r

    @staticmethod
    def loc(r):
        u = urlparse(r.headers["location"])
        return u.path, {k: v[0] for k, v in parse_qs(u.query).items()}

    @staticmethod
    def main_of(page: str) -> str:
        start = page.index("<main", page.index("<body"))
        return page[start:page.index("</main>", start)]


# ── sub-tabs ─────────────────────────────────────────────────────────────────────────

class SubTabs(_Fixture):
    def test_registered_in_subtabs_with_runtime_first(self):
        self.assertEqual(admin.SUBTABS["server"],
                         [("runtime", "Runtime"), ("restart", "Restart"),
                          ("keys", "API Keys"), ("models", "Models")])

    def test_default_is_runtime_and_each_sub_shows_only_its_form(self):
        page = self.get()
        self.assertIn('<nav class="subnav">', page)
        self.assertRegex(page, r'class="on" aria-current="page" href="/ui/server\?sub=runtime"')
        m = self.main_of(page)
        self.assertIn('name="_form" value="runtime"', m)
        self.assertNotIn('name="_form" value="restart"', m)
        self.assertNotIn("/ui/server/api-key", m)
        r = self.main_of(self.get("/ui/server?sub=restart"))
        self.assertIn('name="_form" value="restart"', r)
        self.assertNotIn('name="_form" value="runtime"', r)
        self.assertIn("Restart-required", r)
        k = self.main_of(self.get("/ui/server?sub=keys"))
        self.assertIn('action="/ui/server/api-key"', k)
        self.assertNotIn('name="_form"', k)
        # the intro stays on every sub-tab
        for m in (m, r, k):
            self.assertIn("These override <code>config.yaml</code>", m)
        # an unknown sub falls back to the default
        self.assertIn('name="_form" value="runtime"', self.main_of(self.get("/ui/server?sub=x")))

    def test_api_key_field_left_runtime(self):
        m = self.main_of(self.get("/ui/server?sub=runtime"))
        self.assertNotIn('name="api_key"', m)
        self.assertNotIn("API key (client auth)", m)
        # every other runtime row and flag is still there
        for k, *_ in admin._SRV_RUNTIME:
            self.assertIn(f'name="{k}"', m)
        for b in ("log_per_call", "model_prefix", "show_user_keys"):
            self.assertIn(f'name="{b}"', m)

    def test_pending_restart_is_visible_from_runtime_and_keys(self):
        info = main.server_info()
        info["runtime"] = dict(info["runtime"], stats_db_path="/somewhere/else.db")
        self.addCleanup(setattr, admin, "_server_info", admin._server_info)
        admin._server_info = lambda: info
        for sub in ("runtime", "keys", "restart"):
            page = self.get(f"/ui/server?sub={sub}")
            nav = page[page.index('<nav class="subnav">'):]
            nav = nav[:nav.index("</nav>")]
            self.assertIn("↻ restart", nav, sub)
        m = self.main_of(self.get("/ui/server?sub=runtime"))
        self.assertIn("A restart is pending", m)
        # a state, not an error: warn badge, announced, never the red refusal style
        banner = re.search(r"<p [^>]*data-k='server-restart'[^>]*>.*?</p>", m).group(0)
        self.assertIn("role='status'", banner)
        self.assertIn('<span class="badge warn">↻ restart</span>', banner)
        self.assertNotIn("class='bad'", banner)
        admin._server_info = lambda: main.server_info()
        page = self.get("/ui/server?sub=runtime")
        self.assertNotIn("↻ restart", page)
        self.assertNotIn("A restart is pending", page)


class SaveRedirects(_Fixture):
    def test_runtime_save_lands_on_runtime(self):
        r = self.post("/ui/server/save", {"_form": "runtime", "max_parked": "77"})
        self.assertEqual(self.loc(r), ("/ui/server", {"sub": "runtime", "saved": "1"}))
        self.assertEqual(store.get_setting("max_parked"), 77)
        page = self.get("/ui/server?sub=runtime&saved=1")
        self.assertIn("✓ Saved — runtime settings applied live.", page)

    def test_restart_save_lands_on_restart(self):
        r = self.post("/ui/server/save", {"_form": "restart", "stats_retention_days": "3"})
        self.assertEqual(self.loc(r), ("/ui/server", {"sub": "restart", "saved": "restart"}))
        self.assertEqual(store.get_setting("stats_retention_days"), 3)
        page = self.get("/ui/server?sub=restart&saved=restart")
        self.assertIn("✓ Saved — port/stats/jobs changes need a <b>restart</b> to apply.", page)

    def test_save_without_a_valid_form_is_refused_and_stores_nothing(self):
        # read as the Restart form with every field blank, it used to CLEAR every
        # restart-only override
        store.set_settings({"stats_db_path": "/keep/me.db", "max_parked": 50})
        for data in ({"stats_db_path": ""}, {"_form": "bogus", "max_parked": "1"}):
            r = self.post("/ui/server/save", data, status=400)
            self.assertIn("names no settings form", r.text)
            self.assertIn('name="_form" value="runtime"', r.text)
        self.assertEqual(store.get_setting("stats_db_path"), "/keep/me.db")
        self.assertEqual(store.get_setting("max_parked"), 50)

    def test_runtime_save_never_touches_the_master_key(self):
        store.set_settings({"api_key": "keep-me"})
        self.post("/ui/server/save", {"_form": "runtime", "api_key": "smuggled"})
        self.assertEqual(store.get_setting("api_key"), "keep-me")


class NumberValidation(_Fixture):
    def test_decimal_in_int_field_is_400_as_typed_and_nothing_stored(self):
        store.set_settings({"max_parked": 50})
        r = self.post("/ui/server/save", {"_form": "runtime", "max_parked": "1.5",
                                          "health_check_interval": "12",
                                          "scan_ports": "8080, 9000", "log_per_call": "1"},
                      status=400)
        self.assertIn("max parked calls: &#x27;1.5&#x27; is not a whole number", r.text)
        self.assertIn('name="max_parked" value="1.5"', r.text)          # as typed
        self.assertIn('name="health_check_interval" value="12"', r.text)
        self.assertIn('name="scan_ports" value="8080, 9000"', r.text)
        self.assertRegex(r.text, r'name="log_per_call" value="1" checked')
        self.assertIn('name="_form" value="runtime"', r.text)             # its own tab
        self.assertEqual(store.get_setting("max_parked"), 50)              # nothing stored
        self.assertIsNone(store.get_setting("health_check_interval"))
        self.assertIsNone(store.get_setting("scan_ports"))

    def test_garbage_in_float_and_restart_fields(self):
        r = self.post("/ui/server/save", {"_form": "runtime", "affinity_max_wait_s": "abc"},
                      status=400)
        self.assertIn("affinity max wait", r.text)
        self.assertIsNone(store.get_setting("affinity_max_wait_s"))
        r = self.post("/ui/server/save", {"_form": "restart", "stats_retention_days": "-1",
                                          "jobs_db_path": "x.db"}, status=400)
        self.assertIn("retention days", r.text)
        self.assertIn('name="_form" value="restart"', r.text)
        self.assertIn('name="jobs_db_path" value="x.db"', r.text)
        self.assertIsNone(store.get_setting("jobs_db_path"))
        r = self.post("/ui/server/save", {"_form": "restart", "port": "0"}, status=400)
        self.assertIn("port", r.text)

    def test_blank_is_unset_and_unit_fields_keep_decimals(self):
        self.post("/ui/server/save", {"_form": "runtime", "max_concurrent": "",
                                      "affinity_max_wait_s": "2,5"})
        self.assertEqual(store.get_setting("max_concurrent"), "")
        self.assertEqual(store.get_setting("affinity_max_wait_s"), 2.5)
        # TTL is edited in hours, prune in minutes — decimals there are fine (→ seconds)
        self.post("/ui/server/save", {"_form": "restart", "jobs_default_ttl_s": "1.5",
                                      "jobs_prune_interval_s": "0.5"})
        self.assertEqual(store.get_setting("jobs_default_ttl_s"), 5400)
        self.assertEqual(store.get_setting("jobs_prune_interval_s"), 30)
        self.post("/ui/server/save", {"_form": "restart", "jobs_default_ttl_s": "x"},
                  status=400)
        self.assertEqual(store.get_setting("jobs_default_ttl_s"), 5400)


# ── API Keys ─────────────────────────────────────────────────────────────────────────

class ApiKeysTab(_Fixture):
    def keys(self) -> str:
        return self.main_of(self.get("/ui/server?sub=keys"))

    def row(self, page: str, k: str) -> str:
        m = re.search(rf'<form[^>]*data-k="{k}"[^>]*>.*?</form>', page, re.S)
        self.assertIsNotNone(m, k)
        return m.group(0)

    def test_rows_in_order_and_users_line(self):
        page = self.keys()
        order = ["srvkey-master"] + [f"srvkey-provider-{k}" for k in hostapi.PROVIDERS] \
            + ["srvkey-hf"]
        idx = [page.index(f'data-k="{k}"') for k in order]
        self.assertEqual(idx, sorted(idx))
        self.assertIn("Master API key (client auth)", self.row(page, "srvkey-master"))
        self.assertIn("Thunder Compute API token", self.row(page, "srvkey-provider-thunder"))
        self.assertIn("Create it once in the Thunder Compute console (shown only once there); "
                      "every Thunder Compute managed host uses it.",
                      self.row(page, "srvkey-provider-thunder"))
        self.assertIn("Hugging Face token", self.row(page, "srvkey-hf"))
        self.assertIn("Used by the model sync of managed hosts to download gated Hugging Face "
                      "files; sent only to huggingface.co / hf.co.", self.row(page, "srvkey-hf"))
        self.assertIn("User API keys are managed in the <a href=\"/ui/users\">Users</a> tab.",
                      page)
        self.assertGreater(page.index("User API keys are managed"), idx[-1])

    def test_no_row_renders_a_value_and_every_input_is_new_password(self):
        store.set_settings({"api_key": "MASTER-SECRET", "hf_token": "hf_SECRETVAL"})
        store.set_provider_token("thunder", "th-SECRET")
        page = self.get("/ui/server?sub=keys")
        for secret in ("MASTER-SECRET", "hf_SECRETVAL", "th-SECRET"):
            self.assertNotIn(secret, page)
        pw = re.findall(r'<input type="password"[^>]*>', self.main_of(page))
        self.assertEqual(len(pw), 5 + len(hostapi.PROVIDERS))
        for tag in pw:
            self.assertIn('value=""', tag)
            self.assertIn('autocomplete="new-password"', tag)
        # two rows post `api_key`: each input still has its own id, each label its own
        ids = re.findall(r'\sid="([^"]+)"', self.main_of(page))
        self.assertEqual(len(ids), len(set(ids)), ids)
        fors = re.findall(r'<label for="([^"]+)"', self.main_of(page))
        self.assertEqual(len(fors), len(set(fors)), fors)

    def test_set_and_not_set_badges(self):
        page = self.keys()
        for k in ("srvkey-provider-thunder", "srvkey-hf"):
            self.assertIn('<span class="badge warn">not set</span>', self.row(page, k))
        main.api_key = "m"
        self.login()
        store.set_settings({"hf_token": "hf_x"})
        store.set_provider_token("thunder", "th-x")
        page = self.keys()
        for k in ("srvkey-master", "srvkey-provider-thunder", "srvkey-hf"):
            self.assertIn('<span class="badge ok">set</span>', self.row(page, k))

    def test_master_key_blank_keeps_and_saves_only_itself(self):
        store.set_settings({"hf_token": "hf_keep"})
        store.set_provider_token("thunder", "th-keep")
        r = self.post("/ui/server/api-key", {"api_key": "new-master"})
        self.assertEqual(self.loc(r)[0], "/ui/server")
        self.assertEqual(self.loc(r)[1]["sub"], "keys")
        raw = store.get_settings()
        self.assertEqual(store.get_setting("api_key"), "new-master")
        self.assertEqual(main.api_key, "new-master")                      # applied live
        self.assertEqual(store.get_setting("hf_token"), "hf_keep")
        self.assertEqual(store.get_provider_token("thunder"), "th-keep")
        self.assertNotIn("max_parked", raw)                                # nothing else
        # encrypted at rest
        import sqlite3
        with sqlite3.connect(store._DB_PATH) as c:
            v = c.execute("SELECT value_json FROM settings WHERE key='api_key'").fetchone()[0]
        self.assertNotIn("new-master", v)
        # a blank Save keeps it (the console is locked by the key now)
        self.login()
        r = self.post("/ui/server/api-key", {"api_key": ""})
        self.assertIn("unchanged", self.loc(r)[1]["msg"])
        self.assertEqual(store.get_setting("api_key"), "new-master")

    def test_master_key_with_spaces_or_control_characters_is_refused(self):
        main.api_key = ""
        store.set_settings({"api_key": ""})
        for bad in ("two words", "tab\there", "ctl\x01x", "k" * 1025):
            with self.subTest(bad=bad[:20]):
                r = self.post("/ui/server/api-key", {"api_key": bad}, status=400)
                row = self.row(r.text, "srvkey-master")
                self.assertIn("the master API key may hold only printable characters "
                              "without spaces (at most 1024) — not saved", row)
                self.assertNotIn(bad, r.text)
                self.assertEqual(store.get_setting("api_key"), "")
                self.assertEqual(main.api_key, "")
        # surrounding whitespace (a pasted newline) is trimmed, not refused
        self.post("/ui/server/api-key", {"api_key": "  good-key\n"})
        self.assertEqual(store.get_setting("api_key"), "good-key")

    def test_master_key_on_the_keys_tab_revokes_old_sessions(self):
        main.api_key = "old-master"
        store.set_settings({"api_key": "old-master"})
        self.login()
        tok = self.c.cookies.get(admin._SESSION_COOKIE)
        self.get("/ui/server?sub=keys")                                    # session works
        self.post("/ui/server/api-key", {"api_key": "rotated-master"})
        self.assertEqual(main.api_key, "rotated-master")
        import types
        self.assertIsNone(admin._session_user(
            types.SimpleNamespace(cookies={admin._SESSION_COOKIE: tok})))
        r = self.c.get("/ui/server?sub=keys", headers=SAME, follow_redirects=False)
        self.assertNotEqual(r.status_code, 200)                           # back to login

    def test_provider_token_saves_only_itself_and_lands_on_keys(self):
        main.api_key = ""
        store.set_settings({"hf_token": "hf_keep"})
        r = self.post("/ui/server/provider-token", {"provider": "thunder", "api_key": "th-1"})
        path, q = self.loc(r)
        self.assertEqual((path, q["sub"]), ("/ui/server", "keys"))
        self.assertIn("Thunder Compute API token saved", q["msg"])
        self.assertEqual(store.get_provider_token("thunder"), "th-1")
        self.assertEqual(store.get_setting("hf_token"), "hf_keep")
        self.assertIsNone(store.get_setting("api_key"))
        self.assertEqual(self.applied_hosts, [1])
        page = self.get(f"/ui/server?sub=keys&msg={q['msg']}")
        self.assertIn("Thunder Compute API token saved", page)
        self.assertIn("<p class='ok-banner' role='status' data-k='server-msg'>", page)

    def test_provider_token_refusals_are_400_on_the_keys_tab(self):
        import types
        saved = main.host_controllers
        self.addCleanup(setattr, main, "host_controllers", saved)
        main.host_controllers = {"tc": types.SimpleNamespace(
            kind="thunder", op=None, state=types.SimpleNamespace(phase="ready",
                                                                  pending_snapshot=""))}
        store.set_provider_token("thunder", "th-KEEP")
        r = self.post("/ui/server/provider-token",
                      {"provider": "thunder", "api_key": "", "api_key_clear": "1"}, status=400)
        row = self.row(r.text, "srvkey-provider-thunder")
        self.assertIn(html.escape("Thunder Compute hosts are not off (tc) — stop them first, "
                                  "or enter a new token instead of clearing it"), row)
        self.assertIn('action="/ui/server/api-key"', r.text)               # the keys tab
        self.assertNotIn("th-KEEP", r.text)
        self.assertEqual(store.get_provider_token("thunder"), "th-KEEP")
        r = self.post("/ui/server/provider-token", {"provider": "thunder",
                                                    "api_key": "th BAD SECRET"}, status=400)
        self.assertIn("not saved", r.text)
        self.assertNotIn("th BAD SECRET", r.text)
        r = self.post("/ui/server/provider-token", {"provider": "unknown",
                                                    "api_key": "rp-SECRET"}, status=400)
        self.assertIn("unknown provider", r.text)
        self.assertNotIn("rp-SECRET", r.text)
        self.assertIn('data-k="srvkey-master"', r.text)
        self.assertEqual(self.applied_hosts, [])

    def test_hf_token_blank_keeps_box_clears_and_refusal_on_keys_tab(self):
        store.set_settings({"api_key": "m-keep"})
        main.api_key = ""                     # the console stays open for the test
        r = self.post("/ui/server/hf-token", {"hf_token": "hf_One"})
        self.assertEqual(self.loc(r)[1]["sub"], "keys")
        self.assertEqual(store.get_setting("hf_token"), "hf_One")
        self.assertEqual(store.get_setting("api_key"), "m-keep")
        r = self.post("/ui/server/hf-token", {"hf_token": ""})
        self.assertIn("unchanged", self.loc(r)[1]["msg"])
        self.assertEqual(store.get_setting("hf_token"), "hf_One")
        r = self.post("/ui/server/hf-token", {"hf_token": "hf bad\nX: 1"}, status=400)
        row = self.row(r.text, "srvkey-hf")
        self.assertIn("not saved", row)
        self.assertNotIn("hf bad", r.text)
        self.assertEqual(store.get_setting("hf_token"), "hf_One")
        r = self.post("/ui/server/hf-token", {"hf_token": "", "hf_token_clear": "1"})
        self.assertIn("removed", self.loc(r)[1]["msg"])
        self.assertEqual(store.get_setting("hf_token"), "")

    def test_actions_are_post_only(self):
        for p in ("/ui/server/api-key", "/ui/server/provider-token", "/ui/server/hf-token"):
            self.assertIn(p, admin._POST_ACTIONS)
            r = self.c.get(p, headers=SAME, follow_redirects=False)
            self.assertEqual(r.status_code, 405, p)
        # the old Backends-tab routes are gone
        for p in ("/ui/hosts/managed/provider-token", "/ui/hosts/managed/hf-token"):
            self.assertNotIn(p, admin._POST_ACTIONS)


# ── Models ───────────────────────────────────────────────────────────────────────────

class ModelsTab(_Fixture):
    def setUp(self):
        super().setUp()
        # the LAN source keeps its key and pin next to store.db: a temp dir, a fresh one
        saved = main._modelsrc_obj
        self.addCleanup(setattr, main, "_modelsrc_obj", saved)
        main._modelsrc_obj = None
        main.jobs_cfg["store_path"] = os.path.join(self.tmp.name, "store.db")

    def test_models_tab_holds_the_lan_source_and_the_catalog_only(self):
        m = self.main_of(self.get("/ui/server?sub=models"))
        self.assertIn('data-k="hosts-modelsrc"', m)
        self.assertIn('action="/ui/hosts/managed/modelsrc-host"', m)
        self.assertIn('data-k="hosts-catalog"', m)
        self.assertIn('action="/ui/hosts/managed/catalog"', m)
        self.assertNotIn('name="_form"', m)
        self.assertNotIn("/ui/server/api-key", m)
        self.assertIn("These override <code>config.yaml</code>", m)
        for sub in ("runtime", "restart", "keys"):
            other = self.main_of(self.get(f"/ui/server?sub={sub}"))
            self.assertNotIn('data-k="hosts-modelsrc"', other, sub)
            self.assertNotIn('data-k="hosts-catalog"', other, sub)

    def test_models_tab_is_static(self):
        # List now redirects back with the fresh state; nothing on it needs a poller
        page = self.get("/ui/server?sub=models")
        self.assertNotIn("<main data-live", page)

    def test_catalog_refusal_is_400_on_the_models_tab_and_saves_nothing(self):
        # a catalog of its own (not the default): a refused Save that re-wrote the
        # default, or anything else, would change it
        store.set_settings({"modelsync_catalog": [{"match": {"alias": "mine"}, "paths": []}]})
        before = store.get_setting("modelsync_catalog")
        r = self.post("/ui/hosts/managed/catalog", {"catalog": '[{"x": <typed>}'}, status=400)
        self.assertRegex(r.text, r'class="on" aria-current="page" href="/ui/server\?sub=models"')
        self.assertIn("[{&quot;x&quot;: &lt;typed&gt;}", r.text)
        self.assertIn("not valid JSON", r.text)
        self.assertEqual(store.get_setting("modelsync_catalog"), before)

    def test_a_view_never_seeds_the_catalog(self):
        # rendering the tab (editor + overview) reads the catalog without writing it:
        # a GET changes no state — the default is shown, and stored by the first Save
        self.assertIsNone(store.get_setting("modelsync_catalog"))
        m = self.main_of(self.get("/ui/server?sub=models"))
        self.assertIn("Trellis2LoadModel", m)                     # the default, shown
        self.assertIsNone(store.get_setting("modelsync_catalog"))
        r = self.post("/ui/hosts/managed/catalog", {"catalog": "[{", "catalog_hash": ""},
                      status=400)
        self.assertIsNone(store.get_setting("modelsync_catalog"))   # a 400 re-render neither

    def test_modelsrc_host_save_and_refusal(self):
        r = self.post("/ui/hosts/managed/modelsrc-host", {"modelsrc_host": "src@10.0.0.2"})
        path, q = self.loc(r)
        self.assertEqual((path, q.get("sub")), ("/ui/server", "models"))
        self.assertIn("src@10.0.0.2", q["msg"])
        page = self.get(r.headers["location"])
        self.assertIn('name="modelsrc_host" value="src@10.0.0.2"', page)
        r = self.post("/ui/hosts/managed/modelsrc-host", {"modelsrc_host": "-oProxy=x"},
                      status=400)
        self.assertIn('name="modelsrc_host" value="-oProxy=x"', r.text)
        self.assertEqual(store.get_setting("modelsrc_host"), "src@10.0.0.2")


# ── Models → Model sources ───────────────────────────────────────────────────────────
GB = 10 ** 9


def _row(path, size, kind="lan", **kw):
    r = {"path": path, "size": size, "in_share": True, "kind": kind, "url": "",
         "origin": "", "verified": "", "provisional": False, "reason": "",
         "outdated_entry": "", "entry_key": "", "dir_entry": "", "dir_repo": "",
         "aliases": [["img", ""]]}
    r.update(kw)
    return r


def _overview(**over):
    rows = [
        _row("models/checkpoints/small.safetensors", 2 * GB),
        _row("models/checkpoints/big.safetensors", 7 * GB,
             aliases=[["blk", "not in source: <x>"], ["img", ""]]),
        _row("models/vae/v.safetensors", 3 * GB, "url", url="https://mirror.example/v?a=1&b=<2>",
             origin="file", verified="sha256", entry_key="models/vae/v.safetensors"),
        _row("models/vae/s.safetensors", 1 * GB, "url", url="https://mirror.example/s",
             origin="file", verified="size", entry_key="models/vae/s.safetensors"),
        _row("models/loras/old.safetensors", 4 * GB, "outdated", url="https://mirror.example/o",
             origin="file", reason="size differs: share 4, entry 3",
             entry_key="models/loras/old.safetensors"),
        _row("hf-cache/hub/models--o--r/blobs/" + "ab" * 32, 5 * GB, "hf-auto",
             url="https://huggingface.co/o/r/resolve/" + "1" * 40 + "/m.safetensors",
             origin="hf-auto"),
        _row("hf-cache/hub/models--o--f/blobs/" + "cd" * 32, 6 * GB, "hf-auto",
             url="https://huggingface.co/o/f/resolve/" + "1" * 40 + "/f.safetensors",
             origin="hf-auto"),
        _row("models/cfg/a.json", 100), _row("models/cfg/b.json", 200),
        _row("models/one/lone.json", 10),
        _row("models/<odd>&/x.bin", 2 * GB),
        _row("models/<odd>&/y.bin", 1 * GB),
    ]
    ov = {"rows": rows, "listed": True, "backends": ["cx"],
          "fallback": {"hf-cache/hub/models--o--f/blobs/" + "cd" * 32: "hash differs — "
                       "syncing the share's copy"},
          "checks": {}, "hashing": [], "pending": False, "problem": ""}
    ov.update(over)
    return ov


class ModelSources(_Fixture):
    """The overview section: a fill-in worklist whose order, collapse, sums and
    badges ARE the information — and whose actions are POSTs that land on ?sub=models."""

    def setUp(self):
        super().setUp()
        self.ov = _overview()
        self.calls = []

        async def overview():
            self.calls.append(("overview", threading.get_ident()))
            return self.ov

        async def check(path, url):
            self.calls.append(("check", path, url))
            return f"check of {path} queued"

        async def check_dir(d, repo):
            self.calls.append(("check_dir", d, repo))
            return f"check of {d} queued"

        async def remove(key):
            asyncio.get_running_loop()                  # awaited ON the loop
            self.calls.append(("remove", key, threading.get_ident()))
            return f"source of {key} removed"
        for k, v in (("_model_sources", overview), ("_check_source", check),
                     ("_check_dir_source", check_dir), ("_remove_source", remove)):
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)

    def section(self, url="/ui/server?sub=models"):
        m = self.main_of(self.get(url))
        start = m.index('data-k="msrc"')
        return m[m.rindex("<div", 0, start):]

    def order(self, sec):
        return re.findall(r'<tr data-k="msrc-([fs])-([^"]*)"', sec)

    def test_rows_sorted_lan_outdated_failed_public_with_small_collapsed(self):
        sec = self.section()
        got = [(t, html.unescape(p)) for t, p in self.order(sec)]
        self.assertEqual(got, [
            ("f", "models/checkpoints/big.safetensors"),
            ("f", "models/<odd>&/x.bin"),                  # 2 GB each: then by path
            ("f", "models/checkpoints/small.safetensors"),
            ("f", "models/<odd>&/y.bin"),
            ("s", "models/cfg/"),                          # two small ones collapse (300 B)
            ("f", "models/one/lone.json"),                 # a single small file stays
            ("f", "models/loras/old.safetensors"),
            ("f", "hf-cache/hub/models--o--f/blobs/" + "cd" * 32),     # failed
            ("f", "hf-cache/hub/models--o--r/blobs/" + "ab" * 32),
            ("f", "models/vae/v.safetensors"), ("f", "models/vae/s.safetensors")])
        small = re.search(r'<tr data-k="msrc-s-models/cfg/">(.*?)</tr>', sec, re.S).group(1)
        self.assertIn("2 small files, 300 B — LAN", small)
        self.assertIn("models/cfg/a.json", small)

    def test_summary_sums(self):
        sec = self.section()
        self.assertIn("12 files · 9.0 GB public (HF auto 5.0 GB · URL 4.0 GB) · "
                      "18.0 GB LAN only · 4.0 GB outdated", html.unescape(sec))
        self.assertIn("LAN only includes 6.0 GB whose URL failed — given up until Sync "
                      "now or a new Check &amp; save", sec)

    def test_badges_and_notes(self):
        sec = self.section()
        row = lambda p: re.search(rf'<tr data-k="msrc-f-{re.escape(p)}">(.*?)</tr>',  # noqa: E731
                                  sec, re.S).group(1)
        self.assertIn(">LAN only<", row("models/checkpoints/big.safetensors"))
        self.assertIn(">URL ✓<", row("models/vae/v.safetensors"))
        self.assertIn(">URL ✓ size only<", row("models/vae/s.safetensors"))
        self.assertIn(">outdated — re-check<", row("models/loras/old.safetensors"))
        self.assertIn("size differs: share 4, entry 3", row("models/loras/old.safetensors"))
        self.assertIn(">HF auto<", row("hf-cache/hub/models--o--r/blobs/" + "ab" * 32))
        failed = row("hf-cache/hub/models--o--f/blobs/" + "cd" * 32)
        self.assertIn(">URL failed — LAN<", failed)
        # blocked aliases marked, the reason as a hover — escaped
        big = row("models/checkpoints/big.safetensors")
        self.assertIn('title="not in source: &lt;x&gt;">blk (blocked)</span>', big)

    def test_urls_and_paths_escaped_never_links(self):
        sec = self.section()
        self.assertIn('<code class="msrc-url">https://mirror.example/v?a=1&amp;b=&lt;2&gt;</code>',
                      sec)
        self.assertNotIn("<2>", sec)
        self.assertNotIn("<odd>", sec)
        self.assertIn("models/&lt;odd&gt;&amp;/x.bin", sec)
        self.assertNotRegex(sec, r'href="https?://')
        self.assertNotIn("<script", sec)

    def test_actions_per_source(self):
        sec = self.section()
        row = lambda p: re.search(rf'<tr data-k="msrc-f-{re.escape(p)}">(.*?)</tr>',  # noqa: E731
                                  sec, re.S).group(1)
        lan = row("models/checkpoints/big.safetensors")
        self.assertIn('action="/ui/hosts/managed/source-check"', lan)
        self.assertIn('name="path" value="models/checkpoints/big.safetensors"', lan)
        self.assertIn('name="url" value=""', lan)
        # a models/… directory with several LAN-only files: ONE repo form, first row
        self.assertIn('action="/ui/hosts/managed/source-check-dir"', lan)
        self.assertIn('name="dir" value="models/checkpoints/"', lan)
        self.assertNotIn("source-check-dir", row("models/checkpoints/small.safetensors"))
        self.assertEqual(sec.count('name="dir" value="models/checkpoints/"'), 1)
        self.assertIn('name="dir" value="models/&lt;odd&gt;&amp;/"', sec)
        self.assertNotIn("source-check-dir", row("models/one/lone.json"))
        small = re.search(r'<tr data-k="msrc-s-models/cfg/">(.*?)</tr>', sec, re.S).group(1)
        self.assertIn('name="dir" value="models/cfg/"', small)
        # outdated: re-check prefilled with its URL, and removable
        old = row("models/loras/old.safetensors")
        self.assertIn('name="url" value="https://mirror.example/o"', old)
        self.assertIn("/ui/hosts/managed/source-remove?key=models%2Floras%2Fold.safetensors", old)
        # URL ✓: remove (confirmed), no check form; HF auto: nothing stored → no remove
        ok = row("models/vae/v.safetensors")
        self.assertRegex(ok, r'formaction="/ui/hosts/managed/source-remove\?key=models%2Fvae'
                             r'%2Fv\.safetensors"[^>]*data-confirm="Remove the public URL')
        self.assertNotIn("source-check", ok)
        hf = row("hf-cache/hub/models--o--r/blobs/" + "ab" * 32)
        self.assertNotIn("source-remove", hf)
        self.assertNotIn("source-check", hf)
        # a URL that failed this session: a new URL may be entered; an HF-auto one has
        # nothing stored to remove
        failed = row("hf-cache/hub/models--o--f/blobs/" + "cd" * 32)
        self.assertIn("source-check", failed)
        self.assertNotIn("source-remove", failed)

    def test_a_failed_explicit_mirror_can_be_removed(self):
        # review-4 M-7: an entry whose URL failed on an instance may be dropped right here
        self.ov["fallback"]["models/vae/v.safetensors"] = "tc: HTTP 404 from the URL"
        sec = self.section()
        row = re.search(r'<tr data-k="msrc-f-models/vae/v.safetensors">(.*?)</tr>', sec,
                        re.S).group(1)
        self.assertIn(">URL failed — LAN<", row)
        self.assertIn('title="tc: HTTP 404 from the URL"', row)          # names the host
        self.assertIn("/ui/hosts/managed/source-remove?key=models%2Fvae%2Fv.safetensors", row)
        self.assertIn("source-check", row)

    def test_filter_is_a_get_form(self):
        sec = self.section()
        form = re.search(r'<form method="get" action="/ui/server"[^>]*>(.*?)</form>', sec,
                         re.S).group(1)
        self.assertIn('name="sub" value="models"', form)
        for c in ("lan", "outdated", "failed", "url", "hf"):
            self.assertIn(f'name="src" value="{c}" checked', form)
        self.assertIn("none ticked = all", form)
        sec = self.section("/ui/server?sub=models&src=url&src=bogus")
        got = [html.unescape(p) for _t, p in self.order(sec)]
        self.assertEqual(got, ["models/vae/v.safetensors", "models/vae/s.safetensors"])
        self.assertIn('name="src" value="url" checked', sec)
        self.assertNotIn('name="src" value="lan" checked', sec)
        sec = self.section("/ui/server?sub=models&src=lan")
        self.assertIn('data-k="msrc-s-models/cfg/"', sec)
        self.assertNotIn("models/vae/v.safetensors", sec)

    def test_live_only_while_pending(self):
        page = self.get("/ui/server?sub=models")
        self.assertNotIn("<main data-live", page)
        self.ov = _overview(pending=True, hashing=["models/x/a.bin", "models/x/b.bin"],
                            checks={"models/checkpoints/big.safetensors":
                                    {"kind": "file", "url": "https://mirror.example/b",
                                     "state": "hashing", "at": 2},
                                    "models/q/": {"kind": "dir", "repo": "o/r", "rev": "main",
                                                  "state": "refused", "reason": "HTTP 404 <b>",
                                                  "at": 1,
                                                  "left_out": {"m.bin": "size differs"}}})
        page = self.get("/ui/server?sub=models")
        self.assertIn('<main data-live="3"', page)
        m = self.main_of(page)
        self.assertIn("checking: hashing", m)
        self.assertIn("share hashes: hashing <code>models/x/a.bin</code> · 1 more waiting", m)
        self.assertIn('data-k="msrc-chk-models/q/"', m)
        self.assertIn("HTTP 404 &lt;b&gt;", m)
        self.assertIn("left out: <code>m.bin</code> — size differs", m)
        # the overview is built by main (off the loop); the page asked for it once
        self.assertEqual([c[0] for c in self.calls], ["overview", "overview"])

    def test_not_listed_and_unavailable(self):
        self.ov = _overview(rows=[], listed=False, problem="LAN source unreachable")
        sec = self.section()
        self.assertIn("The share is not listed yet (LAN source unreachable)", sec)
        self.assertIn("List now", sec)

        async def broken():
            raise RuntimeError("x")
        admin._model_sources = broken
        m = self.main_of(self.get("/ui/server?sub=models"))
        self.assertIn("The overview is not available: RuntimeError", m)
        self.assertIn('data-k="hosts-catalog"', m)                   # the tab stays

    def test_actions_are_post_only_and_land_on_models(self):
        for p in ("/ui/hosts/managed/source-check", "/ui/hosts/managed/source-check-dir",
                  "/ui/hosts/managed/source-remove"):
            self.assertIn(p, admin._POST_ACTIONS)
            self.assertEqual(self.c.get(p, headers=SAME, follow_redirects=False).status_code,
                             405, p)
        self.assertEqual(self.calls, [])
        r = self.post("/ui/hosts/managed/source-check",
                      {"path": "models/a b.bin", "url": "https://x.example/a?t=1"})
        path, q = self.loc(r)
        self.assertEqual((path, q["sub"], q["msg"]), ("/ui/server", "models",
                                                      "check of models/a b.bin queued"))
        r = self.post("/ui/hosts/managed/source-check-dir", {"dir": "models/d/", "repo": "o/r"})
        self.assertEqual(self.loc(r)[1]["msg"], "check of models/d/ queued")
        r = self.post("/ui/hosts/managed/source-remove?key=models%2Fd%2F", {})
        self.assertEqual(self.loc(r)[1]["msg"], "source of models/d/ removed")
        self.assertEqual(self.calls[:2], [("check", "models/a b.bin", "https://x.example/a?t=1"),
                                          ("check_dir", "models/d/", "o/r")])
        self.assertEqual(self.calls[2][:2], ("remove", "models/d/"))

    def test_action_errors_are_banners_not_500(self):
        async def boom(*a):
            raise RuntimeError("secret detail https://x.example/?token=abc")
        admin._check_source = boom
        r = self.post("/ui/hosts/managed/source-check", {"path": "p", "url": "u"})
        self.assertEqual(self.loc(r)[1]["msg"], "not checked: RuntimeError")


class CatalogStaleGuard(_Fixture):
    """R-3: two writers of `modelsync_catalog` — Check & save and the JSON editor. A
    form opened before a check wrote its entry must not silently overwrite it."""

    def setUp(self):
        super().setUp()
        saved = main._modelsrc_obj
        self.addCleanup(setattr, main, "_modelsrc_obj", saved)
        main._modelsrc_obj = None
        main.jobs_cfg["store_path"] = os.path.join(self.tmp.name, "store.db")
        store.set_settings({"modelsync_catalog": []})

    def form(self, page):
        f = re.search(r'<form method="post" action="/ui/hosts/managed/catalog"([^>]*)>(.*?)'
                      r"</form>", page, re.S)
        self.assertIsNotNone(f)
        h = re.search(r'name="catalog_hash" value="([0-9a-f]*)"', f.group(2))
        self.assertIsNotNone(h)
        return f.group(1), h.group(1), f.group(2)

    def test_editor_form_is_live_skip_and_carries_the_hash_of_what_it_shows(self):
        attrs, h, _body = self.form(self.get("/ui/server?sub=models"))
        self.assertIn("data-live-skip", attrs)
        self.assertEqual(h, main.modelsync_catalog_hash([]))

    def test_stale_save_refused_with_the_text_kept_and_the_current_hash(self):
        _a, old, _b = self.form(self.get("/ui/server?sub=models"))
        theirs = [{"file": "models/vae/a.st", "url": "https://mirror.example/a"}]
        store.set_settings({"modelsync_catalog": theirs})       # Check & save meanwhile
        mine = '[{"file": "models/vae/b.st", "url": "https://mirror.example/<b>"}]'
        r = self.post("/ui/hosts/managed/catalog", {"catalog": mine, "catalog_hash": old},
                      status=400)
        self.assertIn("the catalog changed since this form was opened — your text is kept "
                      "below; merge and save again", r.text)
        self.assertIn(html.escape(mine), r.text)                   # AS TYPED
        _a, new, _b = self.form(r.text)
        self.assertEqual(new, main.modelsync_catalog_hash(theirs))
        self.assertNotIn("<main data-live", r.text)                # a refusal is never live
        self.assertEqual(store.get_setting("modelsync_catalog"), theirs)
        # merged and saved with the hash the refusal handed back
        r = self.post("/ui/hosts/managed/catalog", {"catalog": mine, "catalog_hash": new})
        self.assertEqual(self.loc(r)[1]["sub"], "models")
        self.assertEqual(store.get_setting("modelsync_catalog")[0]["file"], "models/vae/b.st")

    def test_the_stale_refusal_shows_the_stored_catalog_read_only(self):
        _a, old, _b = self.form(self.get("/ui/server?sub=models"))
        theirs = [{"file": "models/vae/<a>.st", "url": "https://mirror.example/a"}]
        store.set_settings({"modelsync_catalog": theirs})
        r = self.post("/ui/hosts/managed/catalog", {"catalog": "[]", "catalog_hash": old},
                      status=400)
        cur = re.search(r'<pre class="tlog" data-k="hosts-catalog-current"[^>]*>(.*?)</pre>',
                        r.text, re.S)
        self.assertIsNotNone(cur)
        self.assertIn("models/vae/&lt;a&gt;.st", cur.group(1))
        self.assertNotIn("<a>.st", r.text)
        # it is not a second editor: outside the form, no field
        f = re.search(r'action="/ui/hosts/managed/catalog".*?</form>', r.text, re.S).group(0)
        self.assertNotIn("hosts-catalog-current", f)
        # a validation refusal shows no such block
        r = self.post("/ui/hosts/managed/catalog",
                      {"catalog": "[{", "catalog_hash": main.modelsync_catalog_hash()},
                      status=400)
        self.assertNotIn("hosts-catalog-current", r.text)

    def test_a_post_without_the_hash_is_stale(self):
        # review-4 M-1: no form of this release lacks the field — a tab opened before
        # the deploy or a script; refused, and handed the current hash to save with
        r = self.post("/ui/hosts/managed/catalog", {"catalog": "[]"}, status=400)
        self.assertIn("the catalog changed since this form was opened", r.text)
        _a, h, _b = self.form(r.text)
        self.assertEqual(h, main.modelsync_catalog_hash())
        self.post("/ui/hosts/managed/catalog", {"catalog": "[]", "catalog_hash": h})
        self.assertEqual(store.get_setting("modelsync_catalog"), [])

    def test_validation_refusal_keeps_the_hash_it_was_opened_with(self):
        _a, old, _b = self.form(self.get("/ui/server?sub=models"))
        store.set_settings({"modelsync_catalog": [{"file": "models/vae/a.st",
                                                   "url": "https://mirror.example/a"}]})
        r = self.post("/ui/hosts/managed/catalog", {"catalog": '[{"bogus": 1}]',
                                                    "catalog_hash": old}, status=400)
        _a, kept, _b = self.form(r.text)
        self.assertEqual(kept, old)     # not refreshed: the next Save must still see the change

    def test_stale_refusal_names_a_running_check(self):
        _a, old, _b = self.form(self.get("/ui/server?sub=models"))
        store.set_settings({"modelsync_catalog": [{"file": "models/vae/a.st",
                                                   "url": "https://mirror.example/a"}]})
        self.addCleanup(setattr, main, "_src_checks", main._src_checks)
        main._src_checks = {"models/vae/a.st": {"kind": "file", "state": "hashing"}}
        r = self.post("/ui/hosts/managed/catalog", {"catalog": "[]", "catalog_hash": old},
                      status=400)
        self.assertIn("a Check &amp; save is still running", r.text)


class ModelSourcesWiring(_Fixture):
    """The real chain: admin → main.model_sources_overview → the plan over the store's
    aliases and a (fake) share listing → the rendered row and its POST action."""

    def setUp(self):
        super().setUp()

        class Lan:
            generation = sha_generation = 1

            def cached(self):
                return {"models/checkpoints/w.safetensors": 3 * GB}

            def sha_files(self):
                return {}

            def hash_queue(self):
                return []

            def problem(self):
                return ""
        for n in ("modelsrc", "backends", "image_models"):
            self.addCleanup(setattr, main, n, getattr(main, n))
        main.modelsrc = lambda: Lan()
        main.backends = [{"name": "cx", "type": "comfyui", "url": "http://127.0.0.1:1"}]
        main.image_models = {}
        main._msrc_memo[:] = [None, None]
        main._msrc_kinds_memo[:] = [None, None]
        self.addCleanup(lambda: (main._msrc_memo.__setitem__(slice(None), [None, None]),
                                 main._msrc_kinds_memo.__setitem__(slice(None), [None, None])))
        store.set_settings({"modelsync_catalog": []})
        store.upsert("img", [{"backend": "cx", "task": "text2img", "workflow_json": {
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "w.safetensors"}}}}])

    def test_a_refused_url_reaches_neither_the_redirect_nor_the_log(self):
        # review-4 I-1: the banner is a ?msg= redirect (history, access log)
        with self.assertLogs("admin", "INFO") as logs:
            r = self.post("/ui/hosts/managed/source-check",
                          {"path": "models/checkpoints/w.safetensors",
                           "url": "http://mirror.example/w?token=SECRETabc"})
        loc = r.headers["location"]
        self.assertNotIn("SECRET", loc)
        self.assertNotIn("SECRET", html.unescape(self.loc(r)[1]["msg"]))
        self.assertIn("must start with https://", self.loc(r)[1]["msg"])
        self.assertFalse([x for x in logs.output if "SECRET" in x], logs.output)
        self.assertTrue([x for x in logs.output if "source check" in x])

    def test_a_needed_lan_file_renders_with_its_check_form(self):
        m = self.main_of(self.get("/ui/server?sub=models"))
        row = re.search(r'<tr data-k="msrc-f-models/checkpoints/w.safetensors">(.*?)</tr>', m,
                        re.S)
        self.assertIsNotNone(row, m[-2000:])
        self.assertIn(">LAN only<", row.group(1))
        self.assertIn('action="/ui/hosts/managed/source-check"', row.group(1))
        self.assertIn("1 file · 0.0 GB public", m)
        self.assertIs(admin._model_sources, main.model_sources_overview)
        self.assertIs(admin._remove_source, main.remove_source)
        # remove_source cancels tasks: the console awaits it on the loop, never in a thread
        with open(admin.__file__) as fh:
            src = fh.read()
        self.assertNotRegex(src, r"to_thread\(\s*_remove_source")
        self.assertIn("await _remove_source(", src)


# ── links into the Server tab ────────────────────────────────────────────────────────

class ServerLinks(unittest.TestCase):
    def setUp(self):
        with open(admin.__file__) as fh:
            self.src = fh.read()

    def test_every_link_names_its_sub_tab(self):
        hrefs = re.findall(r"""href=\\?['"](/ui/server[^'"\\]*)""", self.src)
        self.assertTrue(hrefs)
        for h in hrefs:
            self.assertRegex(h, r"^/ui/server\?sub=(runtime|restart|keys|models)$", h)

    def test_each_setting_link_opens_the_tab_that_holds_it(self):
        # what the text right before a link names decides the tab it must open
        rules = (("scan_cidrs", "runtime"), ("show_user_keys", "runtime"),
                 ("<b>stats</b>", "restart"), ("master API", "keys"),
                 ("LAN model source", "models"))
        seen = {sub: 0 for _, sub in rules}
        for m in re.finditer(r"/ui/server\?sub=(\w+)", self.src):
            before = self.src[max(0, m.start() - 160):m.start()]
            hit = [sub for kw, sub in rules if kw in before]
            if not hit:
                continue
            self.assertEqual(m.group(1), hit[-1], before[-120:])
            seen[hit[-1]] += 1
        # scan ×2 + show_user_keys; stats ×3; the master-key refusals + bootstrap banner;
        # the Backends pointer and the sync cell's "waiting for LAN source" badge
        self.assertEqual(seen, {"runtime": 3, "restart": 3, "keys": 4, "models": 2})


if __name__ == "__main__":
    unittest.main()
