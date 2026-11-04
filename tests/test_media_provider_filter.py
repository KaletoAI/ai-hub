"""The Media Playground's provider filter above the model picker.

Why this fails SILENTLY: a filter that drops the wrong rows still renders a valid
picker — an alias that also runs on the chosen provider simply is not offered, a stale
`provider=` from a bookmark empties the list ("no media models"), and a Generate that
forgets the filter lands the user back on the unfiltered list with another model
pre-selected. Nothing errors; the next generation just runs somewhere else.
"""
from unittest.mock import patch

import httpx

from tests.test_nanogpt_console import _Controls
from tests.test_ui_form_validation import _Fixture
import admin
import store


def _row(backend, model):
    cand = {"backend": backend, "model": model, "task": "text2img",
            "nanogpt": {"endpoint": "images", "options": {}},
            "model_metadata": {"supported_parameters": {}}}
    return {"id": f"{backend}/{model}", "name": model, "backend": backend,
            "endpoint": "images", "candidate": cand}


class Helpers(_Fixture):
    def test_alias_counts_under_every_backend_it_runs_on(self):
        rows = [{"id": "a", "providers": ["nano", "venice"]},
                {"id": "nano/x", "providers": ["nano"]},
                {"id": "venice/y", "providers": ["venice"]}]
        self.assertEqual(admin._media_providers(rows), [("nano", 2), ("venice", 2)])
        kept, applied = admin._media_by_provider(rows, "venice")
        self.assertEqual(([r["id"] for r in kept], applied), (["a", "venice/y"], "venice"))

    def test_unknown_provider_filters_nothing(self):
        rows = [{"id": "nano/x", "providers": ["nano"]}]
        self.assertEqual(admin._media_by_provider(rows, "gone"), (rows, ""))
        self.assertEqual(admin._media_by_provider(rows, ""), (rows, ""))


class Page(_Fixture):
    def setUp(self):
        super().setUp()
        self.rows = [_row("nano", "flux"), _row("venice", "sd35")]
        by_id = {r["id"]: r["candidate"] for r in self.rows}
        for name, fn in (("_generation_models", lambda: self.rows),
                         ("_generation_candidate", lambda model: by_id.get(model))):
            p = patch.object(admin, name, fn, create=True)
            p.start()
            self.addCleanup(p.stop)

    def _models(self, url):
        page = self.c.get(url)
        self.assertEqual(page.status_code, 200)
        c = _Controls(page.text)
        return c, [o for o in c.options if o and "/" in o]

    def test_select_sits_above_the_model_and_filters(self):
        c, models = self._models("/ui/playground?sub=media")
        self.assertIn("provider", c.names)
        self.assertLess(c.names.index("provider"), c.names.index("model"))
        self.assertEqual(sorted(models), ["nano/flux", "venice/sd35"])
        _c, models = self._models("/ui/playground?sub=media&provider=venice")
        self.assertEqual(models, ["venice/sd35"])

    def test_stale_provider_shows_everything(self):
        _c, models = self._models("/ui/playground?sub=media&provider=gone")
        self.assertEqual(sorted(models), ["nano/flux", "venice/sd35"])

    def test_single_provider_renders_no_filter(self):
        self.rows[:] = [self.rows[0]]
        c, _models = self._models("/ui/playground?sub=media")
        self.assertNotIn("provider", c.names)

    def test_generate_keeps_the_filter(self):
        async def api(request, method, path, **kwargs):
            return httpx.Response(202, json={"job_id": "j1"})

        with patch.object(admin, "_self_api", api):
            r = self.c.post("/ui/playground/generate",
                            data={"model": "venice/sd35", "type": "media", "provider": "venice",
                                  "p__prompt": "x"}, follow_redirects=False)
        self.assertEqual(r.status_code, 303, r.text[:500])
        self.assertIn("provider=venice", r.headers["location"])
