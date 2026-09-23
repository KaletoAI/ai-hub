"""Console actions are POST-only, their buttons carry names intact, and the editors
keep what was typed.

Why this fails SILENTLY:
- About thirty console actions (delete a backend/user/alias, drain, restart, cancel a
  job, ship a voice …) were plain GET links. Anything that makes a browser navigate
  fires a GET — a link preview, a prefetch, a pasted URL, a crawler — and nothing
  logs that the store just changed (review 2026-09-23). So every handler that writes
  the store or calls a state-changing callback must be POST-only, and every page must
  render its actions as POST buttons. Checked from BOTH sides: an AST walk over the
  handlers (what they touch) and a crawl of the rendered pages (what they link to) —
  a new action added as a GET route or a hand-written `<a href>` fails here, not in a
  mail client.
- Names were put into those URLs with the HTML escape: `a&b` arrived as `a` plus a
  stray `amp;b`, and `+`, `#`, `%` broke the same way — the action then ran on a
  DIFFERENT alias, or on none, and the page just reloaded.
- The Mapping editor threw away unsaved edits: "Update workflow" submits the whole
  editor form but read only the file, dragging a request field navigated away, and
  every action link left the page — no warning, the typed labels simply gone.
"""
import ast
import asyncio
import html.parser
import json
import os
import re
import sys
import tempfile
import unittest
from urllib.parse import parse_qs, urlsplit

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

NASTY = "a&b+c#d%e f"
SAME = {"sec-fetch-site": "same-origin"}


# ── what a handler touches ──────────────────────────────────────────────────────

_STORE_WRITE = re.compile(r"^(upsert|delete|set_|save_|rename_|bootstrap)")
_JOBS_WRITE = re.compile(r"^(create|complete|fail|set_|delete|prune|update|cancel|claim|mark)")
# Bound main.py callbacks that change gateway state (see admin's callback registry).
_MUTATING_CALLBACKS = {"_cancel_generation", "_drain_backend", "_cancel_drain", "_restart_comfy",
                       "_set_backend_enabled", "_voice_lib_save", "_voice_lib_delete",
                       "_voice_lib_ship", "_scan_start", "_apply_backends", "_apply_chat_aliases",
                       "_apply_server_settings", "_apply_users", "_apply_reasoning", "_apply_hosts"}
# Views that may write despite being a GET: none. (The Users page's reverse-DNS names
# used to be persisted from the render; they now stay in memory until the operator
# presses "Save resolved names", a POST.)
_ALLOWED_VIEW_WRITES: set = set()


def _admin_tree():
    with open(admin.__file__) as fh:
        return ast.parse(fh.read())


def _writes(fn) -> set:
    """Direct state changes inside one function body (nested defs included)."""
    out = set()
    for n in ast.walk(fn):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            if f.value.id == "store" and _STORE_WRITE.match(f.attr):
                out.add(f"store.{f.attr}")
            if f.value.id == "jobs" and _JOBS_WRITE.match(f.attr):
                out.add(f"jobs.{f.attr}")
        if isinstance(f, ast.Name) and f.id in _MUTATING_CALLBACKS:
            out.add(f.id)
    return out


def _calls(fn, names) -> set:
    return {n.func.id for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names}


def _routes(tree):
    reg = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "register")
    out = []
    for n in ast.walk(reg):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_api_route":
            methods = next((kw.value for kw in n.keywords if kw.arg == "methods"), None)
            out.append((n.args[0].value, n.args[1].id, [e.value for e in methods.elts]))
    return out


class RouteInventory(unittest.TestCase):
    def setUp(self):
        self.tree = _admin_tree()
        self.funcs = {n.name: n for n in self.tree.body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.routes = _routes(self.tree)

    def _reach(self, name):
        """Every write reachable from `name` through admin's own functions."""
        seen, todo, found = set(), [name], {}
        while todo:
            cur = todo.pop()
            if cur in seen or cur in _ALLOWED_VIEW_WRITES:
                continue
            seen.add(cur)
            fn = self.funcs[cur]
            for w in _writes(fn):
                found.setdefault(w, cur)
            todo += list(_calls(fn, self.funcs))
        return found

    def test_the_inventory_sees_the_known_writers(self):
        # Guard the guard: a walker that finds nothing would pass everything below.
        self.assertIn("store.delete_user", self._reach("users_del"))
        self.assertIn("_drain_backend", self._reach("backend_drain"))
        self.assertIn("store.upsert", self._reach("cand_add"))
        self.assertTrue(len(self.routes) > 50)

    def test_no_get_route_changes_state(self):
        bad = {}
        for path, handler, methods in self.routes:
            if "GET" in methods and (w := self._reach(handler)):
                bad[path] = w
        self.assertEqual(bad, {}, "GET routes that write — make them POST (_POST_ACTIONS)")

    def test_actions_list_matches_the_registered_routes(self):
        by_path = {p: m for p, _h, m in self.routes}
        for p in admin._POST_ACTIONS:
            self.assertEqual(by_path.get(p), ["POST"], p)


class PostOnlyOverHttp(unittest.TestCase):
    def setUp(self):
        self._saved = (main.api_key, main.users, main._users_by_key)
        main.api_key, main.users, main._users_by_key = "", [], {}
        self.c = TestClient(main.app)

    def tearDown(self):
        main.api_key, main.users, main._users_by_key = self._saved

    def test_a_get_to_an_action_runs_nothing(self):
        for p in admin._POST_ACTIONS:
            url = p.replace("{job_id}", "abc") + "?name=x&alias=x&id=openai:x&idx=0"
            r = self.c.get(url, headers=SAME, follow_redirects=False)
            self.assertEqual(r.status_code, 405, url)

    def test_a_typed_action_url_gets_a_console_page_not_bare_json(self):
        # A bookmark or an old script link to a (formerly GET) action answered with
        # Starlette's bare `{"detail":"Method Not Allowed"}` — no console, no way back.
        for p in sorted(admin._POST_ACTIONS) + ["/ui/backends/save", "/ui/users/save"]:
            url = p.replace("{job_id}", "abc")
            r = self.c.get(url + "?name=x", headers=SAME, follow_redirects=False)
            self.assertEqual(r.status_code, 405, url)
            self.assertIn("text/html", r.headers.get("content-type", ""), url)
            self.assertEqual(r.headers.get("allow"), "POST", url)
            self.assertIn("only runs from its button", r.text, url)
            self.assertIn("<nav", r.text, url)                       # the console chrome
        r = self.c.get("/ui/backends/delete", headers=SAME)
        self.assertIn("href='/ui/backends'", r.text)
        r = self.c.get("/ui/job/abc/cancel", headers=SAME)
        self.assertIn("href='/ui/job/abc'", r.text)


# ── what the pages render ───────────────────────────────────────────────────────

class _Tags(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def _tags(markup):
    p = _Tags()
    p.feed(markup)
    return p.tags


def _qs(url):
    return {k: v[-1] for k, v in parse_qs(urlsplit(url).query, keep_blank_values=True).items()}


class Buttons(unittest.TestCase):
    def test_action_href_renders_a_post_button_on_the_action_form(self):
        (tag, a), = _tags(admin._btn("✕", "/ui/users/delete?name=x", confirm="Sure?"))
        self.assertEqual(tag, "button")
        self.assertEqual(a.get("form"), admin._ACT_FORM_ID)
        self.assertEqual(a.get("formaction"), "/ui/users/delete?name=x")
        self.assertEqual(a.get("formmethod"), "post")
        self.assertEqual(a.get("data-confirm"), "Sure?")

    def test_view_href_stays_a_link(self):
        (tag, a), = _tags(admin._btn("✎", "/ui/users?edit=x"))
        self.assertEqual((tag, a.get("href")), ("a", "/ui/users?edit=x"))

    def test_submit_button_keeps_its_confirm(self):
        (tag, a), = _tags(admin._btn("Save", submit=True, confirm="Really?"))
        self.assertEqual((tag, a.get("type"), a.get("data-confirm")), ("button", "submit", "Really?"))

    def test_every_page_has_the_action_form_outside_main(self):
        page = admin._page("T", "<p>x</p>", "dashboard", refresh=4)
        self.assertIn(f'<form id="{admin._ACT_FORM_ID}" method="post"', page)
        self.assertGreater(page.index(f'id="{admin._ACT_FORM_ID}"'), page.index("</main>"))
        self.assertIn("window.gwPost", page)


class _Store(unittest.TestCase):
    """A temp store holding one of everything, named NASTY, and stubs for main's live
    views — enough for every console page to render its actions."""

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
        backends = [{"name": NASTY, "type": "comfyui", "url": "http://127.0.0.1:9", "enabled": True,
                     "healthy": True, "models": 0, "source": "ui"},
                    {"name": "llm", "type": "openai", "url": "http://127.0.0.1:9", "enabled": True,
                     "healthy": True, "models": 1, "source": "ui"}]
        stubs = {"_gateway_info": lambda: {"backends": backends, "virtual_models": []},
                 "_gen_backends": lambda: [b for b in backends if b["type"] == "comfyui"],
                 "_comfy_backends": lambda: [],
                 "_llm_backends": lambda: [{"name": "llm", "models": ["m1"], "enabled": True},
                                           {"name": "llm2", "models": ["m2"], "enabled": True}],
                 "_config_chat_aliases": lambda: {}}
        for k, v in stubs.items():
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, v)
        for k in [n for n in _MUTATING_CALLBACKS if n.startswith("_apply_")]:
            self.addCleanup(setattr, admin, k, getattr(admin, k))
            setattr(admin, k, lambda: None)
        self.wf = {"1": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 7}},
                   "2": {"class_type": "LoadImage", "inputs": {"image": "x.png"}}}
        store.upsert(NASTY, [{"backend": NASTY, "task": "text2img", "workflow_json": self.wf,
                              "mapping": {"steps": {"node": "1", "field": "steps"},
                                          "cfg": {"node": "1", "field": "cfg"}},
                              "fixed": [], "bypass": ["2"]}])
        store.upsert_chat_alias(NASTY, {"llm": "m1"})
        store.upsert_user({"name": NASTY, "role": "user", "api_key": "k-1"})
        store.set_reasoning_rules([{"order": 1, "match": "*", "backends": ["*"],
                                    "adapter": "none", "param": {}, "enabled": True}])
        store.set_voice_entry(NASTY, {"file": "/nonexistent.wav", "ref_text": "hi"})
        store.set_ip_alias("10.0.0.9", "box")
        self.c = TestClient(main.app)

    def get(self, url):
        r = self.c.get(url, headers=SAME, follow_redirects=False)
        self.assertEqual(r.status_code, 200, url)
        return r.text

    def post(self, url, headers=None, **kw):
        return self.c.post(url, headers={**SAME, **(headers or {})}, follow_redirects=False, **kw)


PAGES = ["/ui/backends", f"/ui/backends?edit=comfyui:{NASTY}", "/ui/mapping?sub=chat",
         "/ui/mapping?cedit=" + NASTY.replace("%", "%25").replace("&", "%26").replace("+", "%2B")
         .replace("#", "%23").replace(" ", "%20"),
         "/ui/mapping?sub=media",
         "/ui/mapping?edit=" + NASTY.replace("%", "%25").replace("&", "%26").replace("+", "%2B")
         .replace("#", "%23").replace(" ", "%20"),
         "/ui/reasoning", "/ui/users", "/ui/playground?sub=voice"]


class RenderedPages(_Store):
    def _crawl(self):
        out = {}
        for url in PAGES:
            out[url] = self.get(url)
        return out

    def test_no_page_links_to_an_action(self):
        for url, page in self._crawl().items():
            for tag, a in _tags(page):
                href = a.get("href") or ""
                self.assertFalse(admin._is_post_action(href), f"{url}: <{tag} href={href}>")
                for ev in ("onclick", "onchange"):
                    self.assertNotRegex(a.get(ev) or "", r"location(\.href)?\s*=",
                                        f"{url}: navigating {ev} on <{tag}>")

    def test_pages_do_offer_their_actions_as_post_buttons(self):
        pages = self._crawl()
        want = {"/ui/backends": "/ui/backends/drain", "/ui/users": "/ui/users/delete",
                "/ui/reasoning": "/ui/reasoning/toggle", "/ui/mapping?sub=media": "/ui/mapping/copy",
                "/ui/playground?sub=voice": "/ui/playground/voice-ship"}
        for url, action in want.items():
            fa = [a.get("formaction", "") for t, a in _tags(pages[url]) if t == "button"]
            self.assertTrue(any(x.startswith(action) for x in fa), f"{url}: no {action} button")

    def test_names_survive_the_round_trip_in_every_action_url(self):
        # Every alias/name/backend parameter of every action and edit link decodes back
        # to the exact name — the HTML escape used to split `a&b` in two.
        seen = 0
        for url, page in self._crawl().items():
            for tag, a in _tags(page):
                target = a.get("formaction") or a.get("data-post") or a.get("href") or ""
                if not target.startswith("/ui/"):
                    continue
                for k, v in _qs(target).items():
                    if k in ("alias", "name", "edit", "cedit", "backend") and "a&b" in target + v:
                        self.assertIn(v, (NASTY, "comfyui:" + NASTY), f"{url}: {target}")
                        seen += 1
        self.assertGreater(seen, 8)

    def test_action_redirects_keep_the_name(self):
        r = self.post("/ui/mapping/copy?alias=" + _q(NASTY))
        self.assertEqual(r.status_code, 303)
        self.assertEqual(_qs(r.headers["location"])["edit"], NASTY + "-copy")
        r = self.post(f"/ui/chat/badd?alias={_q(NASTY)}&backend=llm2")
        self.assertEqual(_qs(r.headers["location"])["cedit"], NASTY)
        self.assertIn("llm2", store.get_chat_alias(NASTY))

    def test_job_rows_cancel_by_post(self):
        row = admin._job_row({"id": "abc123", "status": "running", "alias": "x", "created": 0},
                             0, actions=True)
        fa = [a.get("formaction") for t, a in _tags(row) if t == "button"]
        self.assertIn("/ui/job/abc123/cancel", fa)
        self.assertNotIn("/cancel", "".join(a.get("href", "") for _t, a in _tags(row)))


def _q(s):
    from urllib.parse import quote
    return quote(s, safe="")


# ── the Mapping editor keeps what was typed ─────────────────────────────────────

class EditorKeepsEdits(_Store):
    def _editor_form(self, **over):
        f = {"alias": NASTY, "new_alias": NASTY, "task": "text2img",
             "node__steps": "1", "field__steps": "steps", "label__steps": "Steps!",
             "node__cfg": "1", "field__cfg": "cfg", "retries": ""}
        f.update(over)
        return f

    def test_update_workflow_applies_the_editor_fields_too(self):
        new_wf = dict(self.wf)
        new_wf["3"] = {"class_type": "SaveImage", "inputs": {}}
        r = self.post("/ui/mapping/update-workflow", data=self._editor_form(),
                      files={"workflow_file": ("wf.json", json.dumps(new_wf).encode(), "application/json")})
        self.assertEqual(r.status_code, 303)
        c = store.get(NASTY)[0]
        self.assertIn("3", c["workflow_json"])                               # replaced …
        self.assertEqual(c["mapping"]["steps"].get("label"), "Steps!")       # … and the edit kept

    def test_a_refused_workflow_file_keeps_the_edits_and_says_why(self):
        r = self.post("/ui/mapping/update-workflow", data=self._editor_form(),
                      files={"workflow_file": ("wf.json", b"{not json", "application/json")})
        self.assertEqual(r.status_code, 400)
        self.assertIn("invalid workflow JSON", r.text)
        self.assertEqual(store.get(NASTY)[0]["mapping"]["steps"].get("label"), "Steps!")
        self.assertNotIn("3", store.get(NASTY)[0]["workflow_json"])

    def test_save_stores_the_request_fields_in_row_order(self):
        # The drag-reorder saves the form; the row (DOM) order IS the stored order.
        form = [("alias", NASTY), ("new_alias", NASTY),
                ("node__cfg", "1"), ("field__cfg", "cfg"),
                ("node__steps", "1"), ("field__steps", "steps")]
        r = self.post("/ui/mapping/update", content="&".join(f"{k}={_q(v)}" for k, v in form),
                      headers={"content-type": "application/x-www-form-urlencoded"})
        self.assertEqual(r.status_code, 303)
        self.assertEqual(list(store.get(NASTY)[0]["mapping"]), ["cfg", "steps"])

    def test_editors_are_guarded_and_drop_no_longer_navigates(self):
        page = self.get(PAGES[5])
        forms = [a for t, a in _tags(page) if t == "form" and a.get("action") == "/ui/mapping/update"]
        self.assertEqual(len(forms), 1)
        self.assertIn("data-guard", forms[0])
        self.assertIn("beforeunload", admin._CONFIRM_JS)
        self.assertIn("requestSubmit", admin._reorder_js())
        self.assertNotIn("location.href", admin._reorder_js())
        chat = self.get(PAGES[3])
        self.assertTrue(any(t == "form" and a.get("action") == "/ui/chat/save" and "data-guard" in a
                            for t, a in _tags(chat)))

    def test_rename_onto_a_taken_name_is_refused_and_said(self):
        store.upsert("other", [{"backend": NASTY, "workflow_json": {}, "mapping": {}}])
        r = self.post("/ui/mapping/update", data=self._editor_form(new_alias="other"))
        self.assertEqual(r.status_code, 303)
        self.assertEqual(_qs(r.headers["location"]).get("taken"), "other")
        self.assertEqual(store.get("other")[0]["mapping"], {})                 # untouched
        self.assertEqual(store.get(NASTY)[0]["mapping"]["steps"].get("label"), "Steps!")
        page = self.get(r.headers["location"])
        self.assertIn("not renamed", page)


if __name__ == "__main__":
    unittest.main()
