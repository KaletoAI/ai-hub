"""The backend form is ONE form in four tabs — and both halves of that fail silently.

A pane switched by `display` keeps every input in the POST; a field that got lost while
the form was re-sorted, or that ended up rendered TWICE, is not an error anywhere:
`backend_save` reads an absent field as "cleared", so the next Save wipes a stored value
(the api key, the ComfyUI output dir, the sampling defaults) and a duplicated one saves
whichever copy the browser sent last. Likewise a pane whose tab button is missing — or
whose button is hidden for the current type — is simply unreachable in the browser, with
nothing in any log to say so.

So this pins, for every backend type and for both Add (`b=None`) and Edit:
  · every form key `backend_save` reads is rendered EXACTLY ONCE (the key list is
    derived from `backend_save`'s own source, so a new field cannot be forgotten here),
  · every field sits in exactly one pane, and every pane has a tab button,
  · the model whitelist/blacklist pair is there for EVERY type — it filters ComfyUI
    checkpoint lists just as it filters an LLM catalog, so it must not have landed in a
    type-hidden block,
  · and a save roundtrip actually stores those two, for all types.

Run: python -m unittest tests.test_backend_form_tabs -v
"""
import ast
import asyncio
import inspect
import os
import re
import sys
import tempfile
import textwrap
import unittest
from html.parser import HTMLParser
from urllib.parse import urlencode

# `import main` reads ./config.yaml at import time — give it a minimal one in a temp cwd
# (same dance as test_cloud_editor.py).
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main            # noqa: F401  (admin needs it importable, not bound)
    import admin
    import adapters
    import store
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


TYPES = ("openai", "comfyui", "meshy", "tripo", "anthropic")

# A backend with EVERY editable key set, so an Edit render has a value to lose.
FULL = {
    "name": "b1", "url": "http://x:8188", "host": "gpu1", "paid": True,
    "max_concurrent": 2, "api_key": "sek", "models_allow": "gpt-*, claude-*",
    "models_deny": "*-embed, *:free", "chat_only": True, "serverless_only": True,
    "local": True, "prompt_cache": True, "model_context": "glm-*=32768",
    "comfy_output_dir": "/o", "comfy_input_dir": "/i", "auto_restart": True,
    "restart_cooldown_s": 600, "stuck_after_s": 90, "self_retries": 1,
    "max_wait": 600, "poll_interval": 1.5, "auth_mode": "api_key",
    "models": ["claude-sonnet-5"], "sampling_defaults": {"temperature": 0.85},
}

# The id-based type blocks that predate `data-btype` — same meaning, different marker.
_ID_TYPES = {"llmopts": "openai", "comfyopts": "comfyui",
             "cloudopts": "cloud", "anthopts": "anthropic"}


class _FormMap(HTMLParser):
    """Where every named control sits: its enclosing pane (`data-btab`) and the
    type-conditional block it is in, if any (`data-btype`, or one of the four historic
    ids). Hand-written because the console renders HTML as strings — there is nothing
    to query — and because "which pane is this field in" is exactly the question a
    re-sorted form can get wrong."""

    VOID = {"input", "br", "hr", "img", "meta", "link", "source", "col", "area", "base"}
    NAMED = {"input", "select", "textarea"}

    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.stack: list = []          # [(tag, pane, btype)]
        self.fields: dict = {}         # name -> [(pane, btype), …]
        self.panes: list = []          # data-btab values, document order
        self.tabs: list = []           # data-tab button values
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        pane = self.stack[-1][1] if self.stack else None
        btype = self.stack[-1][2] if self.stack else None
        if a.get("data-btab"):
            pane = a["data-btab"]
            self.panes.append(pane)
        if a.get("data-btype"):
            btype = a["data-btype"]
        elif a.get("id") in _ID_TYPES:
            btype = _ID_TYPES[a["id"]]
        if a.get("data-tab"):
            self.tabs.append(a["data-tab"])
        if tag in self.NAMED and a.get("name"):
            self.fields.setdefault(a["name"], []).append((pane, btype))
        if tag not in self.VOID:
            self.stack.append((tag, pane, btype))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return


def _form_keys(fn) -> set:
    """The form keys a handler reads, from its own source — so a field added to
    `backend_save` is covered here without anyone remembering to list it.

    Two shapes: a literal (`f.get("host")`, `f["api_key"]`) and a loop over a tuple of
    literals whose loop variable reaches `f.get(...)` — which is how the boolean flags,
    the numeric keys and the cloud_* pairs are read. `smp_*` is deliberately NOT covered
    (it is built from a table in `_parse_sampling_form`) and is asserted separately."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))

    def get_calls(node):
        for n in ast.walk(node):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "get" and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "f" and n.args):
                yield n.args[0]
            if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                    and n.value.id == "f"):
                yield n.slice

    keys = {a.value for a in get_calls(tree)
            if isinstance(a, ast.Constant) and isinstance(a.value, str)}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.For) and isinstance(n.iter, ast.Tuple)):
            continue
        used = {a.id for a in get_calls(n) if isinstance(a, ast.Name)}
        tgt = n.target
        names = [tgt.id] if isinstance(tgt, ast.Name) else [
            e.id for e in getattr(tgt, "elts", []) if isinstance(e, ast.Name)]
        idxs = [i for i, nm in enumerate(names) if nm in used]
        if not idxs:
            continue                       # a loop over keys it only pops, not reads
        for elt in n.iter.elts:
            vals = [elt] if not isinstance(elt, ast.Tuple) else list(elt.elts)
            for i in idxs:
                if i < len(vals) and isinstance(vals[i], ast.Constant):
                    keys.add(vals[i].value)
    return keys


SAVE_KEYS = _form_keys(admin.backend_save)


def _render(btype: str, edit: bool) -> str:
    b = dict(FULL, type=btype, name=btype) if edit else None
    return admin._backend_form(b, ["gpu1", "gpu2"])


class FormKeysAreDerivable(unittest.TestCase):
    """Guards the derivation itself: if `_form_keys` ever stopped finding the loop-driven
    keys it would return a small set and every test below would pass vacuously."""

    def test_extraction_finds_both_shapes(self):
        for k in ("name", "url", "api_key",           # literal f.get / f[...]
                  "chat_only", "auto_restart",        # the boolean-flag loop
                  "stuck_after_s", "max_wait",        # the numeric loop
                  "cloud_max_wait", "cloud_poll_interval",   # the (src, dst, cast) loop
                  "models_allow", "models_deny"):
            self.assertIn(k, SAVE_KEYS, f"{k} not derived from backend_save")
        self.assertNotIn("k", SAVE_KEYS)              # loop vars are not keys
        self.assertGreater(len(SAVE_KEYS), 20)


class EveryFieldRenderedOnce(unittest.TestCase):
    def test_every_saved_key_appears_exactly_once(self):
        for t in TYPES:
            for edit in (True, False):
                html = _render(t, edit)
                fm = _FormMap(html)
                for key in sorted(SAVE_KEYS):
                    if key == "orig" and not edit:
                        # the rename anchor: emitted only when editing, by design
                        self.assertNotIn("orig", fm.fields, f"{t}/add")
                        continue
                    n = len(fm.fields.get(key, []))
                    self.assertEqual(n, 1, f"{t}/{'edit' if edit else 'add'}: "
                                           f"'{key}' rendered {n}× (expected 1)")

    def test_sampling_inputs_appear_once(self):
        # Not derivable from backend_save (built from _SAMPLING_FIELDS), so checked
        # against the renderer that owns them.
        names = re.findall(r'name="(smp_[^"]+)"', admin._sampling_inputs(None))
        self.assertTrue(names)
        for t in TYPES:
            fm = _FormMap(_render(t, True))
            for nm in names:
                self.assertEqual(len(fm.fields.get(nm, [])), 1, f"{t}: {nm}")

    def test_stored_values_survive_the_render(self):
        # The other half of "rendered once": rendered with the STORED value, not blank.
        html = _render("comfyui", True)
        for val in ('value="gpt-*, claude-*"', 'value="*-embed, *:free"',
                    'value="sek"', 'value="/o"', 'value="/i"'):
            self.assertIn(val, html)


class PanesAndTabs(unittest.TestCase):
    def test_every_field_sits_in_exactly_one_pane(self):
        for t in TYPES:
            for edit in (True, False):
                fm = _FormMap(_render(t, edit))
                for name, spots in fm.fields.items():
                    if name == "orig":
                        continue          # a hidden input, outside the tabbed area
                    panes = {p for p, _ in spots}
                    self.assertEqual(len(panes), 1, f"{t}: {name} in panes {panes}")
                    self.assertIsNotNone(panes.pop(), f"{t}: {name} sits in no pane")

    def test_every_pane_has_a_button_and_vice_versa(self):
        for t in TYPES:
            fm = _FormMap(_render(t, True))
            self.assertEqual(sorted(fm.panes), ["behavior", "general", "models", "type"])
            self.assertEqual(sorted(fm.tabs), sorted(fm.panes))
            self.assertEqual(len(fm.panes), len(set(fm.panes)))

    def test_the_type_tab_is_named_after_the_type(self):
        # Hidden for openai (it has no pane content of its own); named for the rest —
        # a button whose label never changes would send the operator to an empty pane.
        want = {"openai": None, "comfyui": "ComfyUI", "meshy": "Cloud task API",
                "tripo": "Cloud task API", "anthropic": "Anthropic"}
        for t, label in want.items():
            html = _render(t, True)
            m = re.search(r'<button[^>]*id="btab-type"([^>]*)>([^<]*)</button>', html)
            self.assertIsNotNone(m, t)
            attrs, text = m.group(1), m.group(2)
            if label is None:
                self.assertIn("display:none", attrs, t)
            else:
                self.assertNotIn("display:none", attrs, t)
                self.assertEqual(text, label, t)

    def test_general_is_the_pane_the_server_renders_open(self):
        # The JS restores the last tab, but a browser with no JS (or a first visit)
        # must land on a pane that is actually visible.
        html = _render("comfyui", True)
        self.assertIn('<div class="bpane" data-btab="general">', html)
        for other in ("models", "behavior", "type"):
            self.assertIn(f'<div class="bpane" data-btab="{other}" style="display:none">', html)

    def test_panes_are_always_present_never_conditional(self):
        # THE invariant: an absent input is "cleared" to backend_save. Whatever the type,
        # all four panes and every saved key are in the DOM — visibility is CSS only.
        for t in TYPES:
            html = _render(t, True)
            self.assertEqual(html.count('class="bpane"'), 4, t)
            for key in ("comfy_output_dir", "auth_mode", "models", "prompt_cache",
                        "cloud_max_wait", "self_retries", "chat_only"):
                self.assertIn(f'name="{key}"', html, f"{t}: {key} missing entirely")


class ModelFilterFields(unittest.TestCase):
    def test_present_for_every_type_and_never_type_hidden(self):
        for t in TYPES:
            for edit in (True, False):
                fm = _FormMap(_render(t, edit))
                for key in ("models_allow", "models_deny"):
                    spots = fm.fields.get(key, [])
                    self.assertEqual(len(spots), 1, f"{t}: {key}")
                    pane, btype = spots[0]
                    self.assertEqual(pane, "models", f"{t}: {key} in pane {pane}")
                    self.assertIsNone(btype, f"{t}: {key} sits in a {btype}-only block")

    def test_the_hint_says_what_an_empty_result_means(self):
        html = _render("comfyui", True)
        self.assertIn("model whitelist / blacklist", html)
        self.assertIn("blacklist", html)
        self.assertIn("healthy with no models", html)   # the failure mode worth naming


class FilterBadge(unittest.TestCase):
    """`models_filtered` is written by main's discovery; the console only reads it. It
    must survive the key being absent (an older gateway, a backend with no filter) and
    must make "matches nothing" impossible to miss — that backend stays healthy and
    routes nothing, which no other badge in the list would show."""

    def test_absent_or_malformed_renders_nothing(self):
        for v in (None, {}, {"kept": None, "total": 3}, "12/340", {"total": 3}):
            self.assertEqual(admin._filter_badge(v), "")

    def test_partial_filter_is_discreet(self):
        html = admin._filter_badge({"kept": 12, "total": 340})
        self.assertIn("filtered 12/340", html)
        self.assertIn("badge muted", html)
        self.assertIn("model whitelist/blacklist active", html)

    def test_empty_result_is_loud(self):
        html = admin._filter_badge({"kept": 0, "total": 340})
        self.assertIn("badge bad", html)
        self.assertIn("filter matches nothing", html)

    def test_unfiltered_is_silent(self):
        self.assertEqual(admin._filter_badge({"kept": 5, "total": 5}), "")


class _Req:
    """Enough of a Request for `_form()`: it reads the raw body and parses it itself."""

    def __init__(self, form: dict):
        self._body = urlencode(form).encode()

    async def body(self):
        return self._body


class SaveRoundtrip(unittest.TestCase):
    """Rendered → posted → stored → rendered again, on a temp store. The two new keys
    are read outside the cloud branch (which strips the ComfyUI-only keys), so they have
    to survive for EVERY type — including meshy/tripo, where a misplaced read would drop
    them silently on the way to the DB."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._db, self._active = store._DB_PATH, store._active
        self.addCleanup(lambda: setattr(store, "_DB_PATH", self._db))
        self.addCleanup(lambda: setattr(store, "_active", self._active))
        store.init(os.path.join(self.tmp.name, "store.db"))
        for cb in ("_apply_backends", "_apply_chat_aliases"):
            self.addCleanup(setattr, admin, cb, getattr(admin, cb))
            setattr(admin, cb, lambda: None)

    def _save(self, form):
        return asyncio.run(admin.backend_save(_Req(form)))

    def test_filters_are_stored_and_cleared_for_every_type(self):
        for t in TYPES:
            self._save({"name": f"b-{t}", "type": t, "url": "http://x:1",
                        "models_allow": " gpt-*, claude-* ", "models_deny": "*-embed"})
            b = store.get_backend(f"b-{t}", t)
            self.assertEqual(b["models_allow"], "gpt-*, claude-*", t)   # stripped
            self.assertEqual(b["models_deny"], "*-embed", t)
            # and the values come back into the form the operator sees next
            html = admin._backend_form(b, [])
            self.assertIn('name="models_allow" value="gpt-*, claude-*"', html)
            self.assertIn('name="models_deny" value="*-embed"', html)
            # blank clears the key entirely — never an empty glob that matches nothing
            self._save({"name": f"b-{t}", "type": t, "url": "http://x:1",
                        "orig": f"{t}:b-{t}", "models_allow": "", "models_deny": "  "})
            b = store.get_backend(f"b-{t}", t)
            self.assertNotIn("models_allow", b, t)
            self.assertNotIn("models_deny", b, t)


class TabScript(unittest.TestCase):
    """The tab switching is client-side, and every way it can break is silent in the
    browser: a script the live morph never inserts, a hook that lets the morph reset the
    open tab on every tick, or a <button> that submits the form because nobody wrote
    type=button."""

    def test_tab_js_is_on_every_page(self):
        # It must already be there before the form can arrive: `adopt()` strips scripts
        # out of everything the morph inserts.
        self.assertIn(admin._TABS_JS, admin._page("T", "<p>x</p>", "backends", refresh=4))
        self.assertIn(admin._TABS_JS, admin._page("T", "<p>x</p>", "backends"))

    def test_tab_js_registers_a_post_morph_hook(self):
        # The Backends page is live while draining or scanning, and the morph rewrites
        # every inline style from the server's HTML — which always renders General.
        self.assertIn("gwLiveHooks.push", admin._TABS_JS)
        self.assertIn("window.gwBackendTab", admin._TABS_JS)

    def test_tab_js_is_es5(self):
        for bad in ("=>", "let ", "const ", "`"):
            self.assertNotIn(bad, admin._TABS_JS, f"{bad!r} is not ES5")

    def test_buttons_never_submit_the_form(self):
        html = _render("comfyui", True)
        for m in re.finditer(r"<button[^>]*class=\"btab[^\"]*\"[^>]*>", html):
            self.assertIn('type="button"', m.group(0))

    def test_type_select_drives_blocks_and_the_type_tab(self):
        js = admin._type_select("comfyui")
        self.assertIn("data-btype", js)          # the blocks that live in shared panes
        self.assertIn("btab-type", js)           # …and the tab button's own label
        self.assertIn("gwBackendTab", js)        # …then re-apply the active tab
        for anchor in ("cloudopts", "comfyopts", "llmopts", "anthopts"):
            self.assertIn(anchor, js)            # the historic id blocks still switch


if __name__ == "__main__":
    unittest.main()
