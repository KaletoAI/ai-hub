"""The /ui login form: how often it may be guessed at, and how its cookie travels.

Why this fails SILENTLY: the login is the one unauthenticated write path into the admin
area, and nothing limited it — a script could try keys as fast as the gateway answers,
and every wrong guess was an ordinary 401 nobody looks at (review 2026-09-23, F2). So
failures are counted per client IP: after `_LOGIN_MAX_FAILS` inside `_LOGIN_WINDOW_S`
the form answers 429 with `Retry-After` WITHOUT checking the key, and a correct login
clears that IP's count.

And the cookie: behind a TLS-terminating proxy the session cookie was still issued
without `Secure`, so any plain-http request to the same host (a typed `http://` URL, a
downgrade) carried the admin session in clear text (S18). It is `Secure` now whenever
the request came in over HTTPS — the URL scheme, or `X-Forwarded-Proto: https` from the
proxy — and stays without it on plain-http LAN installs, where a Secure cookie would
simply never come back and the login would loop.
"""
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


class _Base(unittest.TestCase):
    def setUp(self):
        self._saved = (main.api_key, main.users, main._users_by_key, store._MASTER_KEY)
        store._MASTER_KEY = os.urandom(32)
        main.api_key, main.users, main._users_by_key = "master-key", [], {}
        admin._login_fails.clear()
        self.c = TestClient(main.app)

    def tearDown(self):
        main.api_key, main.users, main._users_by_key, store._MASTER_KEY = self._saved
        admin._login_fails.clear()

    def _login(self, key, headers=None, client=None):
        return (client or self.c).post("/ui/login", data={"key": key}, headers=headers or {},
                                       follow_redirects=False)


class LoginRateLimit(_Base):
    def test_too_many_failures_answer_429_even_for_the_right_key(self):
        for _ in range(admin._LOGIN_MAX_FAILS):
            self.assertEqual(self._login("wrong").status_code, 401)
        r = self._login("wrong")
        self.assertEqual(r.status_code, 429)
        self.assertTrue(int(r.headers["retry-after"]) > 0)
        self.assertEqual(self._login("master-key").status_code, 429)

    def test_the_window_expires(self):
        for _ in range(admin._LOGIN_MAX_FAILS):
            self._login("wrong")
        # age every recorded failure past the window
        for ip, stamps in admin._login_fails.items():
            admin._login_fails[ip] = [t - admin._LOGIN_WINDOW_S - 1 for t in stamps]
        self.assertEqual(self._login("master-key").status_code, 303)

    def test_a_good_login_clears_the_count(self):
        for _ in range(admin._LOGIN_MAX_FAILS - 1):
            self._login("wrong")
        self.assertEqual(self._login("master-key").status_code, 303)
        for _ in range(admin._LOGIN_MAX_FAILS - 1):
            self.assertEqual(self._login("wrong").status_code, 401)

    def test_the_table_is_bounded(self):
        now = 1000.0
        for i in range(admin._LOGIN_MAX_IPS + 50):
            admin._login_note_fail(f"10.0.{i // 250}.{i % 250}", now + i)
        self.assertLessEqual(len(admin._login_fails), admin._LOGIN_MAX_IPS)


class SecureCookie(_Base):
    def test_plain_http_cookie_is_not_secure(self):
        r = self._login("master-key")
        self.assertEqual(r.status_code, 303)
        self.assertNotIn("secure", r.headers["set-cookie"].lower())

    def test_https_behind_a_proxy_gets_a_secure_cookie(self):
        r = self._login("master-key", headers={"x-forwarded-proto": "https"})
        self.assertIn("secure", r.headers["set-cookie"].lower())

    def test_https_scheme_gets_a_secure_cookie(self):
        c = TestClient(main.app, base_url="https://testserver")
        r = self._login("master-key", client=c)
        self.assertIn("secure", r.headers["set-cookie"].lower())


if __name__ == "__main__":
    unittest.main()
