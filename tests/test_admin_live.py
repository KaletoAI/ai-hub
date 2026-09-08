"""The console's live-update contract.

Three failure modes are silent, which is why they are tested here rather than
eyeballed: a `_page()` that stops emitting `data-live` leaves every auto-updating
view frozen with no error anywhere; a syntax error inside one of the JS blobs
embedded as a Python string simply means the script never runs — the page renders
fine and nothing in the log says otherwise; and a live page whose LATER state
introduces a `<script>` loses it to the morph (`adopt()` strips scripts out of
everything it inserts), so the markup that script was to animate arrives inert —
an un-upgraded custom element, an empty viewer div — while `data-live` disappears
in the same response, stopping the poller so it never self-heals.
"""
import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import admin


class PageLiveAttr(unittest.TestCase):
    def test_refresh_emits_data_live_on_main(self):
        html = admin._page("T", "<p>x</p>", "dashboard", refresh=4)
        self.assertIn('<main data-live="4">', html)

    def test_refresh_no_longer_emits_meta_refresh(self):
        html = admin._page("T", "<p>x</p>", "dashboard", refresh=4)
        self.assertNotIn("http-equiv=\"refresh\"", html)

    def test_no_refresh_emits_plain_main(self):
        # Asserted on the tag, not on the whole page: _LIVE_JS mentions "data-live"
        # in its own source, so a substring check over the document would be a lie.
        html = admin._page("T", "<p>x</p>", "dashboard")
        self.assertIn("<main>", html)
        self.assertNotIn("<main data-live", html)

    def test_body_is_inside_main(self):
        html = admin._page("T", "<p>marker</p>", "dashboard", refresh=2)
        m = re.search(r"<main[^>]*>(.*)</main>", html, re.S)
        self.assertIsNotNone(m, "page must have a <main> element")
        self.assertIn("<p>marker</p>", m.group(1))


class EmbeddedScriptsParse(unittest.TestCase):
    """Every plain inline <script> the console emits must be valid ES5.

    `type=`-carrying scripts (model-viewer's module, the three.js importmap) are
    skipped: they are not plain scripts and `node --check` would reject them for
    reasons that say nothing about our code.
    """

    def test_inline_scripts_are_syntactically_valid(self):
        if not shutil.which("node"):
            self.skipTest("node not installed")
        html = admin._page("T", "<p>x</p>", "dashboard", refresh=4)
        blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
        self.assertTrue(blocks, "expected at least one inline script in the page")
        # One scratch dir for the whole loop: the files exist only for `node --check`
        # and are gone when it returns. Written with delete=False before, they piled
        # up in /tmp — a dozen per run, more with every constant added.
        with tempfile.TemporaryDirectory() as tmp:
            for i, src in enumerate(blocks):
                path = os.path.join(tmp, f"inline{i}.js")
                with open(path, "w") as fh:
                    fh.write(src)
                p = subprocess.run(["node", "--check", path],
                                   capture_output=True, text=True)
                self.assertEqual(p.returncode, 0,
                                 f"inline script #{i} is not valid JS:\n{p.stderr}")

    def test_page_level_script_constants_are_valid(self):
        if not shutil.which("node"):
            self.skipTest("node not installed")
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("_SCROLL_JS", "_SORT_JS", "_LIVE_JS", "_FILTER_JS", "_JOB_TICK"):
                blob = getattr(admin, name)
                for i, src in enumerate(re.findall(r"<script>(.*?)</script>", blob, re.S)):
                    path = os.path.join(tmp, f"{name}{i}.js")
                    with open(path, "w") as fh:
                        fh.write(src)
                    p = subprocess.run(["node", "--check", path],
                                       capture_output=True, text=True)
                    self.assertEqual(p.returncode, 0, f"{name} is not valid JS:\n{p.stderr}")

    def test_fbx_viewer_module_is_valid(self):
        # A `type="module"` block, so it is checked as .mjs — `node --check` rejects
        # `import` in a plain .js file for reasons that say nothing about our code.
        # Checked at all because this blob grew a named function and a hook push, and a
        # broken module fails exactly like a missing one: an empty black viewer box.
        if not shutil.which("node"):
            self.skipTest("node not installed")
        blocks = re.findall(r'<script type="module">(.*?)</script>',
                            admin._FBX_VIEWER_JS, re.S)
        self.assertEqual(len(blocks), 1, "expected one module block in _FBX_VIEWER_JS")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fbx.mjs")
            with open(path, "w") as fh:
                fh.write(blocks[0])
            p = subprocess.run(["node", "--check", path], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0,
                             f"_FBX_VIEWER_JS is not valid JS:\n{p.stderr}")


class LiveScriptPresence(unittest.TestCase):
    def test_live_js_is_always_embedded(self):
        # Embedded unconditionally, like _SORT_JS: the script disables itself when
        # <main> carries no data-live, so a static page pays nothing for it.
        # Compared against the constant itself rather than a keyword — _SORT_JS also
        # mentions gwLiveHooks, so a keyword check would pass with _LIVE_JS missing.
        for kwargs in ({"refresh": 4}, {}):
            html = admin._page("T", "<p>x</p>", "dashboard", **kwargs)
            self.assertIn(admin._LIVE_JS, html, f"_LIVE_JS missing for {kwargs}")

    def test_live_js_declares_the_hook_array(self):
        self.assertIn("window.gwLiveHooks", admin._LIVE_JS)

    def test_sort_and_filter_register_post_morph_hooks(self):
        # The server always renders insertion order and the morph reuses nodes, so
        # without these hooks a clicked sort order is undone on every tick and rows
        # morphed in fresh ignore an active filter.
        self.assertIn("gwLiveHooks.push", admin._SORT_JS)
        self.assertIn("gwLiveHooks.push", admin._FILTER_JS)

    def test_filter_no_longer_restores_focus_and_caret(self):
        # Reload compensation only: the morph never replaces a focused input, so the
        # caret needs no rescuing. The filter TEXT persistence stays — it serves real
        # navigation, which the morph does not cover.
        self.assertNotIn("setSelectionRange", admin._FILTER_JS)
        self.assertNotIn("beforeunload", admin._FILTER_JS)
        self.assertNotIn("selectionStart", admin._FILTER_JS)
        self.assertIn("sessionStorage.setItem", admin._FILTER_JS)
        # …and the load-time restore itself must not vanish with them: over-removal is
        # the direction this task is exposed to, and a missing getItem would be silent.
        self.assertIn("sessionStorage.getItem", admin._FILTER_JS)

    def test_scroll_memory_survives_for_real_navigation(self):
        # All three parts stay. <main> is the page's scroll container (body is
        # overflow:hidden), so its position is what F5 or a nav link back to a long
        # list would otherwise lose — the morph covers live updates, not navigation.
        # beforeunload is asserted too: without it the saves never reach the store,
        # and a partial revert that drops only the listener would look green.
        self.assertIn("|main", admin._SCROLL_JS)
        self.assertIn("|master", admin._SCROLL_JS)
        self.assertIn("beforeunload", admin._SCROLL_JS)

    def test_playground_uses_the_shared_live_mechanism(self):
        # The result column had its own poller because a full-page reload would have
        # wiped the form mid-edit. The morph's dirty-input rule covers that now.
        self.assertFalse(hasattr(admin, "_PG_POLL_JS"),
                         "_PG_POLL_JS should be gone — the live morph replaces it")
        self.assertFalse(hasattr(admin, "playground_status"),
                         "the result-fragment route should be gone")

    def test_voice_upload_uses_the_shared_live_mechanism(self):
        # The old poller ended in location.replace(...&vu=done) purely so the library
        # table would show the new entry. The morph updates that table in the same tick.
        self.assertFalse(hasattr(admin, "_VU_POLL_JS"),
                         "_VU_POLL_JS should be gone — the live morph replaces it")
        self.assertFalse(hasattr(admin, "voice_upload_status"),
                         "the upload-progress fragment route should be gone")


# Fixtures for the identity tests below. Built from what the three row templates
# actually read — nothing invented: `_job_row` needs id/status plus the created/
# updated pair `_job_dur_cell` measures, `_call_row` destructures a 15-column stats
# tuple whose first element is the call id, and `_dash_parked` reads a dict with a
# non-empty `parked_calls` list (it returns "" for an empty one, which would make the
# assertion pass for the wrong reason).
_JOB_FIXTURE = {
    "id": "9f3c1ab27de44b0e",
    "status": "done",
    "task": "image",
    "alias": "sdxl",
    "backend": "comfy-a",
    "created": 1_799_999_400,
    "updated": 1_799_999_460,
    "owner": "kai",
    "result_count": 2,
}

# (cid, ts, dur, backend, source, alias, model, endpoint, status, in, out, cost,
#  preview, has_body, reasoning)
_CALL_FIXTURE = (4711, 1_799_999_400, 1234, "local-llama", "10.0.0.5", "chat",
                 "gemma-4", "/v1/chat/completions", 200, 120, 45, 0.0012,
                 "hello", 1, "off:prefill")

_PARKED_FIXTURE = {"parked_calls": [
    {"alias": "chat", "source": "10.0.0.5", "waited_s": 3.2, "remaining_s": 56.8},
]}


class LiveIdentity(unittest.TestCase):
    """Stable identity for everything the morph reconciles.

    Without keys the morph matches positionally, and all three of these fail
    silently — the page still renders correctly, it just rewrites every cell,
    reuses a table node for a different table, or shows a sort UI that ignores
    clicks. Nothing but these assertions would notice.
    """

    def test_job_row_carries_the_job_id_as_key(self):
        row = admin._job_row(_JOB_FIXTURE, 1_800_000_000)
        self.assertIn('data-k="job-', row)
        self.assertIn(_JOB_FIXTURE["id"], row)

    def test_call_row_carries_the_call_id_as_key(self):
        row = admin._call_row(_CALL_FIXTURE, {})
        self.assertIn('data-k="call-', row)

    def test_parked_rows_stay_unkeyed(self):
        # Parked entries have no identity — only alias, source and two values
        # that change every tick. An invented key would be worse than none.
        self.assertNotIn("data-k", admin._dash_parked(_PARKED_FIXTURE))

    def test_keyof_accepts_data_sk_so_sortable_tables_have_identity(self):
        self.assertIn("data-sk", admin._LIVE_JS)

    def test_sort_js_wires_tables_it_has_not_seen(self):
        # A table the morph inserts mid-session gets no click handlers unless the
        # post-morph hook wires it — before this branch a refresh was a reload and
        # re-bound everything, so an unwired table is a regression, not a gap.
        self.assertIn("__gwWired", admin._SORT_JS)


_RUNNING_JOB = {
    "id": "abc123def4567890", "status": "running", "task": "image", "alias": "mesh",
    "backend": "comfy-a", "created": 1_799_999_400, "updated": 1_799_999_460,
    "owner": "kai", "results": [], "meta": {},
}
# The state the running one becomes. Both artifact kinds that need a viewer are here,
# because they fail differently: the GLB needs a DEFINED custom element (model-viewer),
# the FBX needs code to RUN over a div the server renders empty.
_DONE_JOB = dict(_RUNNING_JOB, status="done", results=[
    {"n": 0, "mime": "model/gltf-binary", "kind": "file", "name": "mesh.glb"},
    {"n": 1, "mime": "application/octet-stream", "kind": "file", "name": "rigged.fbx"},
])


class LiveScriptInvariant(unittest.TestCase):
    """A page `_page()` marks live must ALREADY contain every <script> that any later
    state of that page can render.

    `adopt()` strips <script> from everything the morph inserts — deliberately, so a
    re-inserted `_JOB_TICK` cannot double its setInterval — which means the rule cuts
    both ways: a script that first appears in a later response never executes. It fails
    silently and terminally. The Job detail page is the case that has one (it is live at
    2 s while the job runs and grows a 3D preview when it finishes), and `data-live`
    goes away in that same response, so the poller stops and no later tick can repair
    it — only F5 does, which is precisely what nobody does while watching a job.
    """

    def setUp(self):
        self._saved = {
            "jobs.is_active": admin.jobs.is_active, "jobs.get": admin.jobs.get,
            "jobs.neighbors": admin.jobs.neighbors,
            "jobs.result_path": admin.jobs.result_path,
            "store.is_active": admin.store.is_active,
        }
        admin.jobs.is_active = lambda: True
        admin.jobs.neighbors = lambda jid: (None, None)
        admin.jobs.result_path = lambda jid, n: None
        admin.store.is_active = lambda: False   # no alias config → no mapping section

    def tearDown(self):
        admin.jobs.is_active = self._saved["jobs.is_active"]
        admin.jobs.get = self._saved["jobs.get"]
        admin.jobs.neighbors = self._saved["jobs.neighbors"]
        admin.jobs.result_path = self._saved["jobs.result_path"]
        admin.store.is_active = self._saved["store.is_active"]

    def _render(self, job):
        admin.jobs.get = lambda jid: job
        resp = asyncio.run(admin.job_detail_page(job["id"], None))
        return resp.body.decode()

    @staticmethod
    def _script_blocks(html):
        # Whole elements — attributes AND body — not just the opening tag. A bare
        # inline block added to the finished state opens with the very same
        # `<script>` four already-hoisted blocks use, so an opening-tag comparison
        # calls it "already there" and the invariant test goes green on a script the
        # morph will strip. The body is what makes two inline blocks distinguishable.
        return re.findall(r"<script[^>]*>.*?</script>", html, re.S)

    def test_running_job_page_is_live(self):
        # The premise of everything below. If this ever stops holding, the invariant
        # tests would pass for the wrong reason.
        self.assertIn('<main data-live="2">', self._render(_RUNNING_JOB))

    def test_finished_job_page_stops_the_poller(self):
        # The other half of the trap: the response that brings the viewer markup is
        # also the one that stops the poller, so a stripped script is never retried.
        html = self._render(_DONE_JOB)
        self.assertNotIn("<main data-live", html)

    def test_live_page_already_has_every_script_its_later_state_renders(self):
        # THE invariant, checked the general way: no <script> ELEMENT may be new in the
        # finished page. Catches any future viewer/widget added to a job artifact
        # without hoisting it, not just today's two — including a hand-rolled inline
        # block, which an opening-tag comparison could not tell from the inline blocks
        # already on the page.
        live = self._script_blocks(self._render(_RUNNING_JOB))
        done = self._script_blocks(self._render(_DONE_JOB))
        new = [s for s in done if s not in live]
        self.assertEqual(new, [], "scripts that only a finished job renders are dropped "
                                  "by the morph and never run: " + repr(new))

    def test_running_job_page_carries_the_model_viewer_module(self):
        html = self._render(_RUNNING_JOB)
        self.assertIn(f'<script type="module" src="{admin._MODELVIEWER_SRC}"></script>', html)

    def test_running_job_page_carries_the_fbx_viewer(self):
        # Hoisted whole (import map + module), because the module also has to REGISTER
        # its post-morph hook before the first tick.
        self.assertIn(admin._FBX_VIEWER_JS, self._render(_RUNNING_JOB))

    def test_import_map_precedes_any_module_script(self):
        # An import map added after a module script's load was triggered is rejected
        # outright wherever "multiple import maps" is unsupported (Chrome < 133, older
        # Firefox/Safari), and `import ... from 'three'` then fails to resolve — no
        # gwFbxScan, a permanently black FBX box, nothing in any log. An external
        # <script type="module" src> starts loading as soon as it is parsed, so the map
        # has to come first. Invisible in review; only this test guards it.
        for job in (_RUNNING_JOB, _DONE_JOB):
            html = self._render(job)
            imap = html.find('<script type="importmap"')
            self.assertNotEqual(imap, -1, "no import map on the page")
            mods = [m.start() for m in
                    re.finditer(r'<script[^>]*\btype="module"[^>]*\bsrc=', html)]
            self.assertTrue(mods, "no external module script on the page")
            self.assertLess(imap, min(mods),
                            "the import map must precede every external module script")

    def test_finished_job_actually_renders_the_two_viewers(self):
        # Guards the fixture itself: if _media_tag/_job_thumbs stopped emitting a
        # <model-viewer> or a .fbxview the invariant test above would go vacuously green.
        html = self._render(_DONE_JOB)
        self.assertIn("<model-viewer", html)
        self.assertIn('class="fbxview"', html)

    def test_fbx_scan_is_a_named_post_morph_hook(self):
        # Hoisting alone is not enough: the module body runs at load, when a running
        # job's page has no .fbxview at all. The hook is what initialises the one the
        # morph brings in.
        self.assertIn("window.gwFbxScan", admin._FBX_VIEWER_JS)
        self.assertIn("gwLiveHooks.push", admin._FBX_VIEWER_JS)

    def test_fbx_container_is_skipped_by_the_morph(self):
        # The server renders this div EMPTY; data-init and the three.js canvas are
        # client-only, so a later morph would strip both back out.
        self.assertIn("data-live-skip", self._render(_DONE_JOB))

    def test_job_thumbs_emits_no_import_map(self):
        # The gallery IS the subtree the morph inserts, so a <script> in here is dead
        # code by definition — which is how the regression got in. The gallery is not
        # script-free (a GLB cell carries _media_tag's model-viewer module tag, inert
        # because it is byte-identical to the hoisted one); what must never come back
        # is the import map, whose absence up top kills the whole FBX module.
        gal = admin._job_thumbs("abc", "result", _DONE_JOB["results"])
        self.assertNotIn("<script type=\"importmap\"", gal)
        self.assertNotIn("<script type='importmap'", gal)


class PlaygroundScriptInvariant(unittest.TestCase):
    """The Media Playground is the other live page whose column grows a 3D preview."""

    def setUp(self):
        self._get = admin.store.get
        admin.store.get = lambda alias: None    # no store here; the form only needs a shape

    def tearDown(self):
        admin.store.get = self._get

    def test_playground_hoists_the_model_viewer(self):
        body = admin._playground_body([], {"model": ""}, None, "<p>x</p>")
        self.assertIn(f'<script type="module" src="{admin._MODELVIEWER_SRC}"></script>', body)

    def test_playground_never_renders_an_fbx_viewer(self):
        # Why it needs no gwFbxScan: the playground's result column goes through
        # _media_tag, which has no .fbxview branch — an FBX there is a download card.
        # Written as an assertion so the day that changes, this says what to add.
        self.assertNotIn("fbxview", admin._media_tag("/x.fbx", "application/octet-stream", "file"))
        self.assertNotIn("fbxview", admin._playground_body([], {"model": ""}, None, "<p>x</p>"))


class VoiceUploadProgressIsConsumed(unittest.TestCase):
    """The upload verdict must be shown once and then gone.

    `_voice_upload_prog` lives for the whole process and nothing clears it, so a
    branch that renders it whenever it exists makes the LAST upload's checklist the
    permanent content of the Voice tab — for the 303 landings from ship/delete (where
    it can describe a reference that no longer exists), for a fresh tab hours later,
    for the rest of the session. The old `vu=done` reload made the entry moot by
    navigating away from it; the live morph does not, so the branch has to.
    """

    class _Q(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)

    class _Req:
        cookies: dict = {}

        def __init__(self, qp):
            self.query_params = qp

    def setUp(self):
        self._body = admin._voiceplay_body
        # Stubbed: the real one renders the voice-library panel out of the store, which
        # this test has nothing to say about. What it asserts is the branch's own logic.
        admin._voiceplay_body = lambda vals, result, status="": result
        admin._voice_upload_prog.clear()

    def tearDown(self):
        admin._voiceplay_body = self._body
        admin._voice_upload_prog.clear()

    def _render(self):
        req = self._Req(self._Q(sub="voice"))
        return asyncio.run(admin.playground_page(req)).body.decode()

    def test_running_upload_is_live_and_shows_the_checklist(self):
        admin._voice_upload_prog["default"] = {"name": "kai", "steps": [("run", "storing")],
                                               "done": False, "ok": False}
        html = self._render()
        self.assertIn('<main data-live="1">', html)
        self.assertIn("lib:kai", html)

    def test_finished_upload_shows_the_verdict_once_then_the_hint(self):
        admin._voice_upload_prog["default"] = {"name": "kai", "steps": [("ok", "stored")],
                                               "done": True, "ok": True}
        first = self._render()
        self.assertIn("shipped to all hosts", first)
        self.assertNotIn("<main data-live", first)   # nothing left to poll for
        second = self._render()
        self.assertNotIn("shipped to all hosts", second)
        self.assertIn("Synthesize to hear the result here.", second)
        self.assertNotIn("default", admin._voice_upload_prog)
        # …and it stays gone, rather than resurfacing on the visit after that.
        self.assertNotIn("shipped to all hosts", self._render())

    def test_a_new_upload_is_not_consumed_by_the_old_entry(self):
        admin._voice_upload_prog["default"] = {"name": "old", "steps": [], "done": True, "ok": True}
        self._render()                                   # marks the old one seen
        admin._voice_upload_prog["default"] = {"name": "new", "steps": [("run", "storing")],
                                               "done": False, "ok": False}
        html = self._render()
        self.assertIn('<main data-live="1">', html)
        self.assertIn("lib:new", html)


if __name__ == "__main__":
    unittest.main()
