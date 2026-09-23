"""What /health tells whom.

Why this fails SILENTLY: /health is the one route outside /v1 and /ui, and it answered
everyone with the full inventory — every backend and its host, every model id, the
alias map, the physical host grouping, cloud credit balances, the fault log (review
2026-09-23, S9) — on a gateway whose API and console are otherwise locked. Nothing about
that looks wrong from the inside: the console never calls it, and it works. It was also
the biggest response the gateway sends unasked, since every backend listed every model
(an OpenRouter backend alone is hundreds of ids, P15).

So: without a credential a LOCKED gateway answers only `status` and two counts. The full
snapshot needs an admin credential (master key or an admin user's key, as Bearer or
x-api-key), a valid /ui session, or bootstrap-open mode (where everything is open
anyway). Model ids are listed only with `?verbose=1`; otherwise each backend carries
`models_count`.
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

BK = {"name": "llm", "url": "http://10.0.0.5:8080", "type": "openai"}


class HealthAccess(unittest.TestCase):
    def setUp(self):
        names = ("api_key", "users", "_users_by_key", "backends", "backend_models",
                 "backend_healthy")
        self._saved = {n: getattr(main, n) for n in names}
        self._mk = store._MASTER_KEY
        store._MASTER_KEY = os.urandom(32)
        main.backends = [dict(BK)]
        bid = main.backend_id(BK)
        main.backend_models = {bid: {"m1", "m2", "m3"}}
        main.backend_healthy = {bid: True}
        self.bid = bid
        self.c = TestClient(main.app)

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(main, n, v)
        store._MASTER_KEY = self._mk

    def _lock(self):
        main.api_key = "master"
        main.users = [{"name": "kai", "role": "admin", "api_key": "kai-key"},
                      {"name": "bob", "role": "user", "api_key": "bob-key"}]
        main._users_by_key = {u["api_key"]: u for u in main.users}

    def test_locked_gateway_answers_strangers_minimally(self):
        self._lock()
        for headers in ({}, {"authorization": "Bearer bob-key"},
                        {"authorization": "Bearer wrong"}):
            r = self.c.get("/health", headers=headers)
            self.assertEqual(r.status_code, 200)
            d = r.json()
            self.assertEqual(d["status"], "ok")
            self.assertEqual(d["backends_total"], 1)
            self.assertEqual(d["backends_healthy"], 1)
            for k in ("backends", "virtual_models", "hosts", "alias_model_conflicts"):
                self.assertNotIn(k, d, headers)
            self.assertNotIn("10.0.0.5", r.text)

    def test_admin_credentials_get_everything(self):
        self._lock()
        for headers in ({"authorization": "Bearer master"},
                        {"authorization": "Bearer kai-key"}, {"x-api-key": "kai-key"}):
            d = self.c.get("/health", headers=headers).json()
            self.assertIn(self.bid, d["backends"], headers)
        tok = admin._make_session(main.resolve_admin("kai-key"))
        d = self.c.get("/health", cookies={admin._SESSION_COOKIE: tok}).json()
        self.assertIn(self.bid, d["backends"])

    def test_bootstrap_open_shows_everything(self):
        main.api_key, main.users, main._users_by_key = "", [], {}
        d = self.c.get("/health").json()
        self.assertIn(self.bid, d["backends"])

    def test_model_ids_only_with_verbose(self):
        main.api_key, main.users, main._users_by_key = "", [], {}
        b = self.c.get("/health").json()["backends"][self.bid]
        self.assertEqual(b["models_count"], 3)
        self.assertNotIn("models", b)
        b = self.c.get("/health?verbose=1").json()["backends"][self.bid]
        self.assertEqual(b["models"], ["m1", "m2", "m3"])

    def test_internal_snapshot_stays_full(self):
        d = asyncio.run(main.health())
        self.assertEqual(d["backends"][self.bid]["models"], ["m1", "m2", "m3"])


if __name__ == "__main__":
    unittest.main()
