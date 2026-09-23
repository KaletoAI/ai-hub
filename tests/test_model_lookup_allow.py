"""`GET /v1/models/{id}` and the caller's allow-list.

Why this fails SILENTLY: `/v1/models` filters the catalog by the caller's allow-list, but
the single-model lookup only checked that SOME valid key was sent — so a user restricted
to one alias could still probe every alias, backend and model id by name and read its
owner backend and context window (review 2026-09-23, S19). Both answers are 200s that
look like normal client traffic. The lookup now applies the same rule the request path
enforces (`_model_allowed`) and answers a model outside the grant exactly like an
unknown one: 404, so the grant does not leak what exists beyond it.
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
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

from fastapi.testclient import TestClient  # noqa: E402

BK = {"name": "llm", "url": "http://10.0.0.5:8080", "type": "openai"}


class ModelLookup(unittest.TestCase):
    def setUp(self):
        names = ("api_key", "users", "_users_by_key", "backends", "backend_models",
                 "backend_healthy", "virtual_models", "_backend_names")
        self._saved = {n: getattr(main, n) for n in names}
        main.backends = [dict(BK)]
        main._backend_names = {"llm"}
        bid = main.backend_id(BK)
        main.backend_models = {bid: {"secret-model", "open-model"}}
        main.backend_healthy = {bid: True}
        main.virtual_models = {"fast": "open-model", "internal": "secret-model"}
        main.api_key = "master"
        main.users = [{"name": "bob", "role": "user", "api_key": "bob-key", "models": ["fast"]}]
        main._users_by_key = {u["api_key"]: u for u in main.users}
        self.c = TestClient(main.app)

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(main, n, v)

    def _get(self, mid, key):
        return self.c.get(f"/v1/models/{mid}", headers={"authorization": f"Bearer {key}"})

    def test_granted_alias_is_found(self):
        self.assertEqual(self._get("fast", "bob-key").status_code, 200)

    def test_everything_outside_the_grant_is_404(self):
        for mid in ("internal", "llm/secret-model", "secret-model"):
            r = self._get(mid, "bob-key")
            self.assertEqual(r.status_code, 404, mid)
            self.assertNotIn("llm", r.json().get("detail", "").replace(mid, ""))

    def test_master_sees_all(self):
        for mid in ("internal", "llm/secret-model", "secret-model"):
            self.assertEqual(self._get(mid, "master").status_code, 200, mid)


if __name__ == "__main__":
    unittest.main()
