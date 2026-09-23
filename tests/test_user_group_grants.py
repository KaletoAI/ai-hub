"""The user editor's "all chat / all image / all backend" boxes.

Why this fails SILENTLY: ticking "all chat" used to tick every chat alias that existed
AT THAT MOMENT and store exactly those names. An alias added next week was simply not
in the list — the user got a 403 for it (and never saw it in /v1/models) although the
editor had said "all", and nothing anywhere pointed at the stale snapshot. So a group
box stores a grant TOKEN (`@chat`, `@image`, `@backends`) that is resolved on every
request, and saving it drops the now-redundant member names.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

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
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


class _State(unittest.TestCase):
    def setUp(self):
        saved = (main.virtual_models, main.backends, main.backend_models, main.image_models,
                 main._backend_names)
        self.addCleanup(lambda: (setattr(main, "virtual_models", saved[0]),
                                 setattr(main, "backends", saved[1]),
                                 setattr(main, "backend_models", saved[2]),
                                 setattr(main, "image_models", saved[3]),
                                 setattr(main, "_backend_names", saved[4])))
        main._backend_names = {"llm-a", "llm-b"}
        self.llm = {"name": "llm-a", "type": "openai", "url": "http://a"}
        main.backends = [self.llm]
        main.backend_models = {main.backend_id(self.llm): {"model-a"}}
        main.virtual_models = {"tool": "model-a"}
        main.image_models = {}


class ModelAllowed(_State):
    def test_all_chat_covers_an_alias_added_later(self):
        u = {"models": ["@chat"]}
        self.assertTrue(main._model_allowed(u, "tool"))
        main.virtual_models = {**main.virtual_models, "brand-new": "model-a"}
        self.assertTrue(main._model_allowed(u, "brand-new"))
        self.assertFalse(main._model_allowed(u, "llm-a/model-a"))     # not a backend grant

    def test_all_backends_covers_a_backend_added_later(self):
        u = {"models": ["@backends"]}
        self.assertTrue(main._model_allowed(u, "llm-a/model-a"))
        b2 = {"name": "llm-b", "type": "openai", "url": "http://b"}
        main.backends = [self.llm, b2]
        main.backend_models = {**main.backend_models, main.backend_id(b2): {"model-b"}}
        self.assertTrue(main._model_allowed(u, "llm-b/model-b"))
        self.assertTrue(main._model_allowed(u, "model-b"))

    def test_all_image_covers_a_media_alias_added_later(self):
        u = {"models": ["@image"]}
        with mock.patch.object(main, "_gen_alias_exists", side_effect=lambda a: a == "new-img"):
            self.assertTrue(main._model_allowed(u, "new-img"))
            self.assertFalse(main._model_allowed(u, "tool"))

    def test_explicit_names_still_work(self):
        self.assertTrue(main._model_allowed({"models": ["tool"]}, "tool"))
        self.assertFalse(main._model_allowed({"models": ["tool"]}, "other"))


class Catalog(_State):
    def test_models_listing_follows_the_group_grant(self):
        import asyncio
        main.virtual_models = {"tool": "model-a", "later": "model-a"}
        user = {"name": "u", "models": ["@chat"]}
        with mock.patch.object(main, "authenticate", return_value=user), \
             mock.patch.object(main, "backend_healthy", {main.backend_id(self.llm): True}), \
             mock.patch.object(main.store, "is_active", return_value=False):
            req = mock.MagicMock()
            req.query_params = {}
            ids = [m["id"] for m in asyncio.run(main.list_models(req, None))["data"]]
        self.assertIn("later", ids)
        self.assertNotIn("model-a", ids)


class Editor(unittest.TestCase):
    def _form(self, models):
        with mock.patch.object(admin, "_gateway_info", return_value={
                 "virtual_models": ["tool", "vision"],
                 "backends": [{"name": "llm-a", "type": "openai"}]}), \
             mock.patch.object(admin.store, "is_active", return_value=True), \
             mock.patch.object(admin.store, "list_aliases", return_value={"img1": []}), \
             mock.patch.object(admin, "_show_user_keys", return_value=False):
            return admin._user_form({"name": "u", "role": "user", "models": models})

    def test_the_group_box_submits_the_token(self):
        html = self._form(["@chat"])
        self.assertIn('name="model" value="@chat" checked', html)
        # members render ticked (the grant covers them) without being stored by name
        self.assertIn('value="tool" data-grp="chat" checked', html)
        self.assertIn('name="model" value="@image"', html)
        self.assertNotIn('value="@image" checked', html)

    def test_admin_and_main_agree_on_the_tokens(self):
        self.assertEqual(admin._GRANT_TOKENS, {"chat": main.GRANT_ALL_CHAT,
                                               "image": main.GRANT_ALL_IMAGE,
                                               "backend": main.GRANT_ALL_BACKENDS})

    def test_saving_drops_members_a_token_covers(self):
        got = admin._normalize_grants(["@chat", "tool", "vision", "img1", "llm-a", "@backends"],
                                      {"chat": ["tool", "vision"], "image": ["img1"],
                                       "backend": ["llm-a"]})
        self.assertEqual(got, ["@chat", "img1", "@backends"])


if __name__ == "__main__":
    unittest.main()
