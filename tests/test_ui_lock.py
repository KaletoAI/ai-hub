"""When the /ui console is locked, and what may never unlock it again.

Why this fails SILENTLY: an open console looks exactly like a working one. `ui_locked()`
used to lock only once an ADMIN credential existed (master key or an admin user), so a
gateway whose operator had created only `role: user` accounts had its API closed — every
client needs a key — and its console wide open to anyone on the LAN, including every
user key the editor pre-fills (review 2026-09-23, S3). The same state was one click away
on a properly locked gateway: deleting, demoting or disabling the LAST admin (no master
key) re-opened the console without a word (U17). Nothing errors in either case, the
login screen simply stops appearing.

So: the console locks as soon as ANY user or a master key exists, and the users editor
refuses every change that would leave no admin credential behind — including the first
non-admin user on an open gateway, which would otherwise lock everybody out. A gateway
that is ALREADY in the users-but-no-admin state (an old store.db) stays locked and the
login page names the way back in: `api_key` in config.yaml, which is hot-reloaded.
"""
import asyncio
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


class LockRule(unittest.TestCase):
    def setUp(self):
        self._saved = (main.api_key, main.users, main._users_by_key)

    def tearDown(self):
        main.api_key, main.users, main._users_by_key = self._saved

    def _set(self, users, key=""):
        main.api_key = key
        main.users = users
        main._users_by_key = {u["api_key"]: u for u in users
                              if u.get("api_key") and u.get("enabled", True)}

    def test_only_plain_users_lock_the_console(self):
        self._set([{"name": "bob", "role": "user", "api_key": "bob-key"}])
        self.assertTrue(main.ui_locked())
        self.assertFalse(main.admin_credential_exists())

    def test_bootstrap_stays_open(self):
        self._set([])
        self.assertFalse(main.ui_locked())

    def test_admin_credential_needs_an_enabled_admin_with_a_key(self):
        self._set([{"name": "kai", "role": "admin", "api_key": "k", "enabled": False}])
        self.assertFalse(main.admin_credential_exists())
        self._set([{"name": "kai", "role": "admin", "api_key": ""}])
        self.assertFalse(main.admin_credential_exists())
        self._set([{"name": "kai", "role": "admin", "api_key": "k"}])
        self.assertTrue(main.admin_credential_exists())
        self._set([], key="master")
        self.assertTrue(main.admin_credential_exists())

    def test_refusal_rule(self):
        kai = {"name": "kai", "role": "admin", "api_key": "k"}
        bob = {"name": "bob", "role": "user", "api_key": "b"}
        self._set([kai, bob])
        # removing / demoting / disabling the last admin
        self.assertTrue(main.admin_change_refusal([bob]))
        self.assertTrue(main.admin_change_refusal([]))
        self.assertTrue(main.admin_change_refusal([dict(kai, role="user"), bob]))
        self.assertTrue(main.admin_change_refusal([dict(kai, enabled=False), bob]))
        self.assertIsNone(main.admin_change_refusal([kai, dict(bob, role="admin")]))
        # a master key makes every one of them fine
        self._set([kai, bob], key="master")
        self.assertIsNone(main.admin_change_refusal([bob]))
        # an open gateway: the first user must be an admin
        self._set([])
        self.assertTrue(main.admin_change_refusal([bob]))
        self.assertIsNone(main.admin_change_refusal([kai]))


class UsersEditor(unittest.TestCase):
    """The users editor end to end: store, apply_users, the guard, the login."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        saved = (store._DB_PATH, store._active, store._MASTER_KEY,
                 main.api_key, main.users, main._users_by_key)

        def restore():
            (store._DB_PATH, store._active, store._MASTER_KEY,
             main.api_key, main.users, main._users_by_key) = saved
        self.addCleanup(restore)
        store._MASTER_KEY = os.urandom(32)
        store.init(os.path.join(self.tmp.name, "store.db"))
        main.api_key = ""
        main.rebuild_users()
        self.c = TestClient(main.app)

    def _add(self, name, role, key):
        return self.c.post("/ui/users/save", data={"name": name, "role": role, "api_key": key,
                                                   "enabled": "on"}, follow_redirects=False)

    def _login(self, key):
        r = self.c.post("/ui/login", data={"key": key, "next": "/ui/users"}, follow_redirects=False)
        self.assertEqual(r.status_code, 303, r.text)

    def test_first_user_must_be_an_admin(self):
        r = self._add("bob", "user", "bob-key")
        self.assertEqual(r.status_code, 303)
        self.assertIn("refused=", r.headers["location"])
        self.assertIsNone(store.get_user("bob"))
        self.assertFalse(main.ui_locked())
        page = self.c.get(r.headers["location"]).text
        self.assertIn("admin", page)
        self.assertIn("class='bad'", page)

    def test_last_admin_cannot_be_deleted_demoted_or_disabled(self):
        self.assertIsNotNone(self._add("kai", "admin", "kai-key"))
        self.assertTrue(main.ui_locked())
        self._login("kai-key")
        self._add("bob", "user", "bob-key")
        self.assertIsNotNone(store.get_user("bob"))
        r = self.c.post("/ui/users/delete?name=kai", follow_redirects=False)
        self.assertIn("refused=", r.headers["location"])
        self.assertIsNotNone(store.get_user("kai"))
        r = self.c.post("/ui/users/save", data={"orig": "kai", "name": "kai", "role": "user",
                                                "enabled": "on"}, follow_redirects=False)
        self.assertIn("refused=", r.headers["location"])
        self.assertEqual(store.get_user("kai")["role"], "admin")
        r = self.c.post("/ui/users/save", data={"orig": "kai", "name": "kai", "role": "admin"},
                        follow_redirects=False)                    # enabled unchecked
        self.assertIn("refused=", r.headers["location"])
        self.assertTrue(store.get_user("kai").get("enabled", True))
        # the console is still locked for a stranger
        self.assertTrue(main.ui_locked())
        r = TestClient(main.app).get("/ui/users", follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn("/ui/login", r.headers["location"])
        # a second admin makes deleting the first one fine
        self._add("eve", "admin", "eve-key")
        r = self.c.post("/ui/users/delete?name=bob", follow_redirects=False)
        self.assertNotIn("refused=", r.headers["location"])
        self.assertIsNone(store.get_user("bob"))

    def test_users_without_admin_lock_and_the_login_names_the_way_back(self):
        store.upsert_user({"name": "bob", "role": "user", "api_key": "bob-key", "enabled": True})
        main.rebuild_users()
        r = self.c.get("/ui/users", follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        page = self.c.get("/ui/login").text
        self.assertIn("api_key", page)
        self.assertIn("config.yaml", page)
        # a plain user's key does not open it
        r = self.c.post("/ui/login", data={"key": "bob-key"}, follow_redirects=False)
        self.assertEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main()
