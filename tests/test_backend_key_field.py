"""The backend editor's api key field.

Why this fails SILENTLY: the backend form pre-filled a stored cloud key (Together,
OpenRouter, Anthropic, Meshy, Tripo …) as a plain `type=text` value, so every render of
the edit page put the secret into the HTML — the browser's form history, a screenshot,
a shared screen, the page source (review 2026-09-23, S21). And the opposite trap sat
right behind it: a CONFIG-defined backend's key was never in the form at all (the
editor falls back to a summary without it), so the first Save of such a backend wrote
a store copy WITHOUT its key — the backend then failed auth at discovery, nothing else.

So the field is a blank password input that says whether a key is set; blank keeps the
stored key (or the config one, for a backend that is being copied into the store), a
typed value replaces it, and only the explicit "clear" box removes it.
"""
import asyncio
import os
import sys
import tempfile
import unittest
from urllib.parse import urlencode

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


class _Req:
    def __init__(self, form: dict):
        self._body = urlencode(form).encode()

    async def stream(self):
        yield self._body


class KeyField(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        saved = (store._DB_PATH, store._active, store._MASTER_KEY, main.backends,
                 main.config_backends)

        def restore():
            (store._DB_PATH, store._active, store._MASTER_KEY, main.backends,
             main.config_backends) = saved
        self.addCleanup(restore)
        store._MASTER_KEY = os.urandom(32)
        store.init(os.path.join(self.tmp.name, "store.db"))
        for cb in ("_apply_backends",):
            self.addCleanup(setattr, admin, cb, getattr(admin, cb))
            setattr(admin, cb, lambda: None)

    def _save(self, **form):
        base = {"name": "tg", "type": "openai", "url": "https://api.together.xyz"}
        return asyncio.run(admin.backend_save(_Req({**base, **form})))

    def test_stored_key_is_never_rendered(self):
        html = admin._backend_form({"name": "tg", "type": "openai", "url": "u",
                                    "api_key": "tgp_v1_SECRET"}, [])
        self.assertNotIn("tgp_v1_SECRET", html)
        self.assertRegex(html, r'type="password" name="api_key" value=""')
        self.assertIn("set — blank keeps it", html)

    def test_blank_keeps_typed_replaces_clear_removes(self):
        self._save(api_key="first")
        self.assertEqual(store.get_backend("tg", "openai")["api_key"], "first")
        self._save(orig="openai:tg", api_key="")
        self.assertEqual(store.get_backend("tg", "openai")["api_key"], "first")
        self._save(orig="openai:tg", api_key="second")
        self.assertEqual(store.get_backend("tg", "openai")["api_key"], "second")
        self._save(orig="openai:tg", api_key="", api_key_clear="1")
        self.assertNotIn("api_key", store.get_backend("tg", "openai"))

    def test_config_backend_keeps_its_key_on_first_save(self):
        cfg = {"name": "tg", "type": "openai", "url": "https://api.together.xyz",
               "api_key": "from-config"}
        main.config_backends = [cfg]
        main.backends = [cfg]
        summary = next(b for b in main.gateway_info()["backends"] if b["name"] == "tg")
        self.assertNotIn("api_key", summary)                  # the secret stays out of the summary
        self.assertTrue(summary.get("api_key_set"))
        self._save(orig="openai:tg", api_key="")
        self.assertEqual(store.get_backend("tg", "openai")["api_key"], "from-config")


if __name__ == "__main__":
    unittest.main()
