"""Cross-site requests to the /ui console.

Why this fails SILENTLY: a forged request looks exactly like the admin's own click.
Some thirty console actions are plain GET links (delete a user or backend, restart a
ComfyUI, cancel a job …) and the session cookie was `samesite=lax`, which browsers
send on a cross-site top-level GET — so any page the admin happened to open could fire
them with `location = '…/ui/users/delete?name=…'`. In bootstrap-open mode there is no
cookie at all and a cross-site form POST worked too, e.g. into the voice-host setting
that ends up in an `ssh` command line (review 2026-09-23). Nothing logs it; the store
just changes. So the guard refuses what the browser marks as coming from another site
(`Sec-Fetch-Site`, with Origin/Referer as the fallback), a foreign GET gets a page with
a same-origin link instead of the action, and the console cannot be framed.
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

HOST = "testserver"


class CrossSite(unittest.TestCase):
    """Bootstrap-open (no key, no users): the mode where no cookie protects anything."""

    def setUp(self):
        self._saved = (main.api_key, main.users, main._users_by_key)
        main.api_key, main.users, main._users_by_key = "", [], {}
        self.c = TestClient(main.app)

    def tearDown(self):
        main.api_key, main.users, main._users_by_key = self._saved

    def test_cross_site_post_is_refused(self):
        for site in ("cross-site", "same-site"):
            r = self.c.post("/ui/login", data={"key": "x"}, follow_redirects=False,
                            headers={"sec-fetch-site": site})
            self.assertEqual(r.status_code, 403, site)

    def test_cross_site_get_does_not_run_the_action_but_offers_a_link(self):
        r = self.c.get("/ui/users/delete?name=a%26b", follow_redirects=False,
                       headers={"sec-fetch-site": "cross-site"})
        self.assertEqual(r.status_code, 403)
        self.assertIn('href="/ui/users/delete?name=a%26b"', r.text)

    def test_own_requests_pass(self):
        for site in ("same-origin", "none"):          # none = typed URL / bookmark
            r = self.c.get("/ui/login", follow_redirects=False, headers={"sec-fetch-site": site})
            self.assertEqual(r.status_code, 303, site)   # open mode: login → /ui
        r = self.c.post("/ui/login", data={"key": "x"}, follow_redirects=False,
                        headers={"sec-fetch-site": "same-origin", "origin": f"http://{HOST}"})
        self.assertNotEqual(r.status_code, 403)

    def test_origin_and_referer_fallback_without_fetch_metadata(self):
        r = self.c.post("/ui/login", data={"key": "x"}, follow_redirects=False,
                        headers={"origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)
        r = self.c.post("/ui/login", data={"key": "x"}, follow_redirects=False,
                        headers={"origin": "null"})
        self.assertEqual(r.status_code, 403)
        r = self.c.get("/ui/users/delete?name=x", follow_redirects=False,
                       headers={"referer": "http://192.168.8.10:8080/page"})
        self.assertEqual(r.status_code, 403)
        r = self.c.post("/ui/login", data={"key": "x"}, follow_redirects=False,
                        headers={"origin": f"http://{HOST}"})
        self.assertNotEqual(r.status_code, 403)
        r = self.c.post("/ui/login", data={"key": "x"}, follow_redirects=False)  # curl: no headers
        self.assertNotEqual(r.status_code, 403)

    def test_api_is_not_affected(self):
        r = self.c.get("/health", headers={"sec-fetch-site": "cross-site"})
        self.assertNotEqual(r.status_code, 403)

    def test_console_cannot_be_framed(self):
        r = self.c.get("/ui/login", follow_redirects=False)
        self.assertEqual(r.headers.get("x-frame-options"), "DENY")
        self.assertIn("frame-ancestors 'none'", r.headers.get("content-security-policy", ""))
        self.assertEqual(r.headers.get("x-content-type-options"), "nosniff")


class SessionCookieFlags(unittest.TestCase):
    def test_session_cookie_is_samesite_strict(self):
        saved = (store._MASTER_KEY, main.api_key)
        store._MASTER_KEY, main.api_key = os.urandom(32), "master-k"
        try:
            c = TestClient(main.app)
            r = c.post("/ui/login", data={"key": "master-k"}, follow_redirects=False)
            self.assertEqual(r.status_code, 303)
            self.assertIn("samesite=strict", r.headers.get("set-cookie", "").lower())
        finally:
            store._MASTER_KEY, main.api_key = saved


if __name__ == "__main__":
    unittest.main()
