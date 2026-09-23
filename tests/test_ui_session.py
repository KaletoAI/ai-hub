"""The /ui session cookie — the one credential that guards every key in the store.

Why this fails SILENTLY: a forgeable session looks exactly like a working login. The
cookie used to be decrypted with `store.decrypt_secret`, whose legacy-plaintext
passthrough handed an UNENCRYPTED value straight back — so `gw_session={"u":"admin",
"exp":99999999999}`, typed by anyone on the LAN, was an admin session (review
2026-09-23). Nothing errors, the console simply opens. The same goes for a session
that outlives its credential: a deleted or demoted admin, or a rotated master key,
kept full access for the cookie's 12 h. So both are pinned: only an encrypted cookie
counts, and it names the credential it was opened with, re-checked on every request.
"""
import os
import sys
import tempfile
import types
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


def _req(cookie):
    return types.SimpleNamespace(cookies={admin._SESSION_COOKIE: cookie} if cookie else {})


class SessionCookie(unittest.TestCase):
    def setUp(self):
        self._saved = (store._MASTER_KEY, main.api_key, main.users, main._users_by_key,
                       admin._admin_session_tag)
        store._MASTER_KEY = os.urandom(32)
        main.api_key = "master-key"
        main.users = [{"name": "kai", "role": "admin", "api_key": "kai-key"},
                      {"name": "bob", "role": "user", "api_key": "bob-key"}]
        main._users_by_key = {u["api_key"]: u for u in main.users}
        admin._admin_session_tag = main.admin_session_tag

    def tearDown(self):
        (store._MASTER_KEY, main.api_key, main.users, main._users_by_key,
         admin._admin_session_tag) = self._saved

    def test_plaintext_cookie_is_not_a_session(self):
        forged = '{"u":"admin","exp":99999999999}'
        self.assertIsNone(admin._session_user(_req(forged)))
        forged_user = '{"u":"kai","exp":99999999999}'
        self.assertIsNone(admin._session_user(_req(forged_user)))

    def test_encrypted_cookie_without_credential_tag_is_not_a_session(self):
        # Encrypted by someone holding secret.key but not an admin credential, or a
        # cookie from before the tag existed: both must log in again.
        import json, time
        tok = store.encrypt_secret(json.dumps({"u": "admin", "exp": int(time.time()) + 60}))
        self.assertIsNone(admin._session_user(_req(tok)))

    def test_login_session_round_trips(self):
        self.assertEqual(admin._session_user(_req(admin._make_session(main._MASTER_ADMIN))), "admin")
        kai = main.resolve_admin("kai-key")
        self.assertEqual(admin._session_user(_req(admin._make_session(kai))), "kai")

    def test_expired_session_is_rejected(self):
        tok = admin._make_session(main._MASTER_ADMIN, ttl=-1)
        self.assertIsNone(admin._session_user(_req(tok)))

    def test_rotated_master_key_revokes_its_sessions(self):
        tok = admin._make_session(main._MASTER_ADMIN)
        main.api_key = "rotated"
        self.assertIsNone(admin._session_user(_req(tok)))

    def test_removed_master_key_revokes_its_sessions(self):
        tok = admin._make_session(main._MASTER_ADMIN)
        main.api_key = ""
        self.assertIsNone(admin._session_user(_req(tok)))

    def test_demoted_deleted_or_rekeyed_admin_loses_the_session(self):
        kai = main.resolve_admin("kai-key")
        tok = admin._make_session(kai)
        main.users[0]["role"] = "user"
        self.assertIsNone(admin._session_user(_req(tok)))
        main.users[0]["role"] = "admin"
        self.assertEqual(admin._session_user(_req(tok)), "kai")
        main.users[0]["api_key"] = "kai-new-key"
        self.assertIsNone(admin._session_user(_req(tok)))
        main.users[0]["api_key"] = "kai-key"
        main.users[0]["enabled"] = False
        self.assertIsNone(admin._session_user(_req(tok)))
        main.users.pop(0)
        self.assertIsNone(admin._session_user(_req(tok)))

    def test_a_user_named_admin_cannot_ride_the_master_session(self):
        # The master admin is called "admin"; a user of that name must not validate
        # against the master's credential or vice versa.
        main.users.append({"name": "admin", "role": "admin", "api_key": "other-key"})
        user_admin = dict(main.users[-1])
        tok = admin._make_session(user_admin)
        main.api_key = "other-key"          # same key string on the master side
        main.users[-1]["api_key"] = "changed"
        self.assertIsNone(admin._session_user(_req(tok)))

    def test_decrypt_secret_strict_refuses_plaintext(self):
        self.assertEqual(store.decrypt_secret("plain", strict=True), "")
        self.assertEqual(store.decrypt_secret("plain"), "plain")   # stored legacy keys keep working
        self.assertEqual(store.decrypt_secret(store.encrypt_secret("x"), strict=True), "x")


if __name__ == "__main__":
    unittest.main()
