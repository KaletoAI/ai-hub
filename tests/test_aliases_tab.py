"""The Aliases tab — editor and live overview of chat and media aliases in ONE place.

Why this fails SILENTLY: the tab replaced two (Mapping, and the Chat/Media alias
sub-tabs of Input & Routing), so every old URL is somebody's bookmark or a link in a
doc. A redirect that drops the query opens an EMPTY editor instead of the alias it
named; a sub-tab that is still listed but gone renders a blank page; an overview that
is no longer the idle right column is simply never seen again — none of that errors.
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

SNAP = {"aliases": [{"alias": "tool", "routes": [
            {"backend": "llm-a", "model": "qwen", "healthy": True, "enabled": True}]}],
        "models": [], "conflicts": []}


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
        stubs = {"_routing_snapshot": lambda: SNAP,
                 "_config_chat_aliases": lambda: {"tool": "qwen"},
                 "_gateway_info": lambda: {"backends": [{"name": "gpu", "type": "comfyui"}],
                                           "virtual_models": ["tool"]},
                 "_llm_backends": lambda: [{"name": "llm-a", "models": ["qwen"], "enabled": True}],
                 "_gen_backends": lambda: [{"name": "gpu", "type": "comfyui"}]}
        for k, v in stubs.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)
        store.upsert("img1", [{"backend": "gpu", "task": "text2img", "workflow_json": {}}])
        self.c = TestClient(main.app)

    def get(self, url):
        return self.c.get(url, follow_redirects=False)


class Tabs(unittest.TestCase):
    def test_aliases_replaces_mapping_and_the_routing_alias_subtabs(self):
        keys = [k for k, _ in admin.TABS]
        self.assertIn("aliases", keys)
        self.assertNotIn("mapping", keys)
        self.assertEqual([k for k, _ in admin.SUBTABS["aliases"]], ["chat", "media"])
        routing = [k for k, _ in admin.SUBTABS["routing"]]
        self.assertNotIn("chat", routing)
        self.assertNotIn("gen", routing)


class OldUrls(_Fixture):
    def test_mapping_urls_land_on_the_same_view(self):
        for old, new in (("/ui/mapping", "/ui/aliases"),
                         ("/ui/mapping?edit=a%26b", "/ui/aliases?edit=a%26b"),
                         ("/ui/mapping?cedit=tool&saved=1", "/ui/aliases?cedit=tool&saved=1"),
                         ("/ui/mapping?sub=media", "/ui/aliases?sub=media")):
            r = self.get(old)
            self.assertEqual(r.status_code, 307, old)
            self.assertEqual(r.headers["location"], new)

    def test_routing_alias_subtabs_land_on_the_aliases_overview(self):
        r = self.get("/ui/routing?sub=chat")
        self.assertEqual((r.status_code, r.headers["location"]), (307, "/ui/aliases?sub=chat"))
        r = self.get("/ui/routing?sub=gen&backend=gpu")
        self.assertEqual((r.status_code, r.headers["location"]),
                         (307, "/ui/aliases?sub=media&backend=gpu"))

    def test_action_posts_redirect_to_the_new_tab(self):
        r = self.c.post("/ui/mapping/copy?alias=img1", follow_redirects=False,
                        headers={"sec-fetch-site": "same-origin"})
        self.assertTrue(r.headers["location"].startswith("/ui/aliases?edit="), r.headers)


class Overview(_Fixture):
    def test_idle_chat_column_is_the_live_route_overview(self):
        page = self.get("/ui/aliases?sub=chat").text
        self.assertIn("Chat aliases → routes", page)
        self.assertIn('href="/ui/aliases?cedit=tool"', page)     # alias opens its editor
        self.assertIn("qwen", page)

    def test_idle_media_column_is_the_backend_overview_with_its_filter(self):
        page = self.get("/ui/aliases?sub=media").text
        self.assertIn("Media Generation aliases → backends", page)
        self.assertIn('href="/ui/aliases?edit=img1"', page)
        self.assertIn("<form method='get' action='/ui/aliases'", page)
        filtered = self.get("/ui/aliases?sub=media&backend=gpu").text
        self.assertIn("Media aliases on <b>gpu</b>", filtered)

    def test_the_chat_editor_shows_where_the_alias_resolves(self):
        page = self.get("/ui/aliases?cedit=tool").text
        self.assertIn("Live routes", page)
        self.assertIn("llm-a", page)

    def test_the_tab_is_marked_active(self):
        page = self.get("/ui/aliases?sub=chat").text
        self.assertIn('class="on" aria-current="page" href="/ui/aliases"', page)


if __name__ == "__main__":
    unittest.main()
