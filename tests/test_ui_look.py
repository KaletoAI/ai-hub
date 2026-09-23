"""How the /ui console reads, sorts and announces itself.

Why this fails SILENTLY: every mechanism here renders a page that looks fine in the one
browser it was written in. A missing viewport tag leaves a phone at a 980 px desktop
layout scaled to a third (nothing errors — it is just unusable); a sort that compares
"1.2 s" with "102 ms" as TEXT puts them in a plausible-looking wrong order; a label
without `for` is a label a screen reader never announces; a live page that lost its
server shows the last good numbers forever, with nothing saying they are old. So the
contract is pinned on the markup and on the JS, not eyeballed.
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import admin  # noqa: E402
import jobs  # noqa: E402


class PhoneLayout(unittest.TestCase):
    def test_page_declares_viewport_and_language(self):
        html = admin._page("T", "<p>x</p>", "dashboard")
        self.assertIn('<meta name="viewport" content="width=device-width,initial-scale=1">', html)
        self.assertIn('<html lang="en">', html)

    def test_narrow_media_query_lets_the_page_scroll(self):
        # body{overflow:hidden} is right on desktop (<main> is the scroll container) and
        # fatal on a phone once the columns stack — the query must undo it.
        m = re.search(r"@media \(max-width:800px\)\{(.*)\}\s*$", admin._CSS.strip(), re.S)
        self.assertIsNotNone(m, "no narrow-screen media query in _CSS")
        q = m.group(1)
        self.assertIn("body{overflow:visible", q)
        self.assertIn(".cols{display:block", q)
        self.assertIn("table{display:block;overflow-x:auto", q)

    def test_desktop_keeps_main_as_the_scroll_container(self):
        # _SCROLL_JS and CLAUDE.md rely on it; the media query must not leak out.
        desktop = admin._CSS.split("@media")[0]
        self.assertIn("overflow:hidden", re.search(r"\nbody\{([^}]*)\}", desktop).group(1))
        self.assertIn("main{flex:1;min-height:0;overflow-y:auto", desktop)

    def test_css_has_no_duplicate_selectors(self):
        desktop = admin._CSS.split("@media")[0]
        desktop = re.sub(r"/\*.*?\*/", "", desktop, flags=re.S)
        sels = [s.strip() for s in re.findall(r"([^{}]+)\{", desktop)]
        dup = sorted({s for s in sels if sels.count(s) > 1 and not s.startswith("@")})
        self.assertEqual(dup, [], f"selectors defined twice (the later silently wins): {dup}")

    def test_palette_lives_on_root(self):
        self.assertIn(":root{--bg:", admin._CSS)


class Contrast(unittest.TestCase):
    @staticmethod
    def _lum(h):
        h = h.lstrip("#")
        c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    def _ratio(self, a, b):
        la, lb = self._lum(a), self._lum(b)
        return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)

    def test_muted_text_meets_wcag_aa(self):
        var = dict(re.findall(r"--([a-z0-9-]+):(#[0-9a-f]{6})", admin._CSS))
        self.assertIn(".muted{color:var(--muted)}", admin._CSS)
        for bg in ("bg", "row", "sel"):
            self.assertGreaterEqual(self._ratio(var["muted"], var[bg]), 4.5,
                                    f".muted on --{bg} is below 4.5:1")

    def test_keyboard_focus_is_visible(self):
        self.assertIn(".btn:focus-visible", admin._CSS)
        self.assertIn("input:focus-visible", admin._CSS)


_CALL = (4711, 1_799_999_400, 1234, "local-llama", "10.0.0.5", "chat", "gemma-4",
         "/v1/chat/completions", 200, 120, 45, 0.0012, "hello", 1, "off:prefill")


def _node_num(cells):
    """Run _SORT_JS's num() in node over fake cells {text, sv} → list of keys."""
    src = re.search(r"(function num\(td\)\{.*?\})function ind", admin._SORT_JS, re.S)
    assert src, "num() not found in _SORT_JS"
    prog = (src.group(1) + ";var cells=" + json.dumps(cells) + ";"
            "console.log(JSON.stringify(cells.map(function(c){return num({textContent:c.text,"
            "getAttribute:function(k){return k==='data-sv'&&c.sv!==undefined?c.sv:null;}});})));")
    p = subprocess.run(["node", "-e", prog], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


class NumericSort(unittest.TestCase):
    """A header click sorted "1.2 s, 10.1 s, 102 ms" — the numbers carried units, so
    num() gave up and compared them as text (review U4, confirmed in the browser)."""

    def setUp(self):
        if not shutil.which("node"):
            self.skipTest("node not installed")

    def test_units_are_understood(self):
        keys = _node_num([{"text": "1.2 s"}, {"text": "102 ms"}, {"text": "3.5 min"},
                          {"text": "5m"}, {"text": "2h"}, {"text": "$0.0012"},
                          {"text": "<$0.01"}])
        self.assertEqual(keys, [1200, 102, 210000, 300000, 7200000, 0.0012, 0.01])

    def test_raw_value_wins_over_the_text(self):
        self.assertEqual(_node_num([{"text": "09-23 12:00:00", "sv": "1799999400"}]),
                         [1799999400])

    def test_text_stays_text(self):
        self.assertEqual(_node_num([{"text": "gemma-4"}, {"text": "—"}]), [None, None])


class SortKeysInMarkup(unittest.TestCase):
    def test_call_row_carries_raw_values(self):
        row = admin._call_row(_CALL, {})
        self.assertIn('data-sv="1799999400"', row)      # time
        self.assertIn('data-sv="1234"', row)            # dur (ms)
        self.assertIn('data-sv="165"', row)             # tokens in+out
        self.assertIn('data-sv="0.0012"', row)          # cost

    def test_job_row_carries_raw_values(self):
        j = {"id": "9f3c1ab27de44b0e", "status": "done", "task": "image", "alias": "a",
             "backend": "b", "created": 1_799_999_400, "updated": 1_799_999_460, "owner": "k",
             "result_count": 2}
        row = admin._job_row(j, 1_800_000_000, time_col=True)
        self.assertIn('data-sv="1799999400"', row)      # time
        self.assertIn('data-sv="600"', row)             # age (s)
        self.assertIn('data-sv="60000"', row)           # dur (ms)


class LiveStatusChip(unittest.TestCase):
    """A live page whose server went away kept showing its last numbers as if they were
    current (review U12) — the poller only backed off, silently. The chip is the one
    place that says "these numbers are old"."""

    def test_header_carries_a_hidden_chip_outside_main(self):
        html = admin._page("T", "<p>x</p>", "dashboard", refresh=4)
        head = html.split("<body>", 1)[1].split("<main", 1)[0]
        self.assertRegex(head, r'<span id="gwlive" class="livechip"[^>]*\bhidden\b')

    def test_poller_drives_all_three_states(self):
        js = admin._LIVE_JS
        self.assertIn("getElementById('gwlive')", js)
        for state in ("'live'", "'stale'", "'offline'"):
            self.assertIn(state, js)

    def test_dashboard_no_longer_hardcodes_its_cadence(self):
        import inspect
        self.assertNotIn("auto-refresh 4s", inspect.getsource(admin.dashboard_page))


class _Q(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)


class _Req:
    def __init__(self, **qp):
        self.query_params = _Q(qp)
        self.cookies = {}


class MediaJobsList(unittest.TestCase):
    """Media Jobs was live only while a job ran (a job started from ANOTHER client never
    appeared until F5), not sortable unlike every other list, and cut at the newest 200
    with no way to the rest (review U13)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        jobs.init(os.path.join(self.tmp.name, "jobs.db"), os.path.join(self.tmp.name, "b"), 3600)
        self.ids = [jobs.create("img2img", "flux", "comfy-a", job_id=f"job{i:02d}")
                    for i in range(5)]
        with jobs._conn() as c:                  # all five in the SAME second: rowid breaks ties
            c.execute("UPDATE jobs SET created = 1799999000, updated = 1799999010, status='done'")
        self._ipa = admin.store.get_ip_aliases
        admin.store.get_ip_aliases = lambda: {}
        self.addCleanup(setattr, admin.store, "get_ip_aliases", self._ipa)

    def test_recent_pages_backwards_without_gaps_or_repeats(self):
        first = [j["id"] for j in jobs.recent(2, media_only=True)]
        second = [j["id"] for j in jobs.recent(2, media_only=True, before=first[-1])]
        third = [j["id"] for j in jobs.recent(2, media_only=True, before=second[-1])]
        self.assertEqual(first + second + third, list(reversed(self.ids)))

    def test_unknown_cursor_is_an_empty_page(self):
        self.assertEqual(jobs.recent(2, media_only=True, before="pruned-long-ago"), [])

    def _body(self, **qp):
        return asyncio.run(admin._jobs_media_body(_Req(**qp)))

    def test_idle_list_is_still_live(self):
        body, refresh = self._body()
        self.assertTrue(refresh and refresh > 0, "an idle Media Jobs list must keep polling")

    def test_empty_list_is_live_and_already_carries_the_list_scripts(self):
        # The EMPTY state returned early without a refresh: the first job an API client
        # started appeared only on F5 — and the morph that brings the first rows strips
        # every <script> it inserts, so the empty page must already hold the list's own.
        import re
        full, _ = self._body()
        with jobs._conn() as c:
            c.execute("DELETE FROM jobs")
        body, refresh = self._body()
        self.assertIn("No generation jobs yet", body)
        self.assertTrue(refresh and refresh > 0, "an empty Media Jobs list must keep polling")
        scripts = lambda h: re.findall(r"<script[^>]*>.*?</script>", h, re.S)
        self.assertEqual([x for x in scripts(full) if x not in scripts(body)], [])

    def test_list_is_sortable_with_a_stable_key(self):
        body, _ = self._body()
        self.assertRegex(body, r"<table class='filterable sortable' data-sk='media-jobs'>")
        self.assertIn("<th>artifacts</th>", body)

    def test_older_link_pages_through(self):
        old = admin._MEDIA_JOBS_PAGE
        admin._MEDIA_JOBS_PAGE = 2
        try:
            body, _ = self._body()
            self.assertIn("href='/ui/jobs?sub=media&amp;before=job03'", body)
            body, _ = self._body(before="job03")
            self.assertIn("job02", body)
            self.assertNotIn("job04", body)
            self.assertIn("href='/ui/jobs?sub=media'", body)      # back to newest
        finally:
            admin._MEDIA_JOBS_PAGE = old


class FieldLabels(unittest.TestCase):
    """`_field` rendered a bare <label> next to its control: clicking the label did
    nothing and a screen reader announced an unnamed text box (review U14). A hint
    appended inside the control became a flex item that squeezed the input (U20)."""

    def test_label_points_at_the_control(self):
        html = admin._field("name", admin._inp("name", "x"))
        self.assertIn('<label for="fld-name">name</label>', html)
        self.assertIn('id="fld-name"', html)

    def test_existing_id_is_reused(self):
        html = admin._field("filter", "<input id='sf' name='q'>")
        self.assertIn('<label for="sf">', html)
        self.assertEqual(html.count("id="), 1)

    def test_checkbox_labels_are_left_alone(self):
        # _checkbox already wraps its input in a <label>; a second one pointing at it
        # would name the control twice.
        html = admin._field("enabled", admin._checkbox("enabled", True, "enabled"))
        self.assertNotIn("for=", html)
        self.assertNotIn('id="fld-', html)

    def test_hidden_inputs_are_not_labelled(self):
        html = admin._field("x", '<input type="hidden" name="h" value="1">'
                            + admin._select("kind", ["a", "b"]))
        self.assertIn('<label for="fld-kind">', html)

    def test_hint_is_its_own_row(self):
        html = admin._field("port", admin._inp("port", "4000"), hint="needs a <b>restart</b>")
        ctrl = html.split('class="control"', 1)[1].split("</div>", 1)[0]
        self.assertNotIn("restart", ctrl)
        self.assertIn('<div class="fhint" id="fld-port-hint">needs a <b>restart</b></div>', html)
        self.assertIn('class="field hashint"', html)
        self.assertIn('aria-describedby="fld-port-hint"', html)

    def test_icon_buttons_are_named(self):
        html = admin._btn("✕", "/x", "danger", sm=True, icon=True, title="Delete")
        self.assertIn('aria-label="Delete"', html)
        self.assertNotIn("aria-label", admin._btn("Save", "/x", title="Save it"))


class Glyphs(unittest.TestCase):
    """⏻ ⏼ ⧉ are in no font a stock Linux desktop ships (DejaVu, Noto Sans; checked
    with fc-list): the Backends and Mapping action buttons rendered as empty boxes
    (review U15). Nothing errors — the button just has no face."""

    TOFU = "\u23fb\u23fc\u23fd\u29c9"      # ⏻ ⏼ ⏽ ⧉

    def test_console_uses_no_tofu_glyphs(self):
        with open(admin.__file__, encoding="utf-8") as fh:
            src = fh.read()
        found = sorted({c for c in src if c in self.TOFU})
        self.assertEqual(found, [], f"glyphs without a common Linux font: {found}")


class KeyboardReorder(unittest.TestCase):
    """Request fields could only be reordered by mouse drag (review U16)."""

    def test_rows_carry_move_buttons(self):
        rows = admin._req_fields_rows("a", {"3": {"class_type": "KSampler", "inputs": {"steps": 20}}},
                                      {"steps": {"node": "3", "field": "steps"},
                                       "cfg": {"node": "3", "field": "cfg"}}, {})
        self.assertIn('data-mv="-1"', rows)
        self.assertIn('data-mv="1"', rows)
        self.assertIn('aria-label="Move steps up"', rows)

    def test_move_reuses_the_drop_path(self):
        js = admin._reorder_js()
        self.assertIn("data-mv", js)
        self.assertIn("dispatchEvent", js)


class Formatters(unittest.TestCase):
    """U19: a sum of $0.004 read "$0.0040" next to "$12.3400"; a call-log time had no
    year, so last December's rows looked like this week's."""

    def test_sums_use_cents(self):
        self.assertEqual(admin._cost_sum(12.34), "$12.34")
        self.assertEqual(admin._cost_sum(0), "$0.00")
        self.assertEqual(admin._cost_sum(0.004), "<$0.01")

    def test_single_calls_stay_precise(self):
        self.assertEqual(admin._cost(0.0012), "$0.0012")
        self.assertEqual(admin._cost(0.25), "$0.250")
        self.assertEqual(admin._cost(3.5), "$3.50")
        self.assertEqual(admin._cost(0), "$0")
        self.assertEqual(admin._cost(0.00001), "<$0.0001")

    def test_timestamp_shows_the_year_only_when_it_differs(self):
        import time
        now = int(time.time())
        self.assertRegex(admin._ts(now), r"^\d\d-\d\d \d\d:\d\d:\d\d$")
        self.assertRegex(admin._ts(now - 400 * 86400), r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$")

    def test_uncapped_in_flight_says_so(self):
        html = admin._dash_backends([{"name": "a", "type": "openai", "enabled": True,
                                      "healthy": True, "inflight": 0}], [])
        self.assertIn("<td>0 / ∞</td>", html)


class ConsoleScripts(unittest.TestCase):
    """U22. The console's JS is "ES5" only as a SYNTAX rule (no arrow functions,
    let/const, template strings, classes) — it freely uses later DOM APIs (fetch,
    URL, Element.closest, Array.from, replaceChildren), so that is the rule to check."""

    CONSTS = ("_SCROLL_JS", "_SORT_JS", "_LIVE_JS", "_FILTER_JS", "_JOB_TICK", "_TABS_JS",
              "_CONFIRM_JS", "_USER_ACC_JS")

    def test_page_scripts_keep_es5_syntax(self):
        for name in self.CONSTS:
            src = "".join(re.findall(r"<script>(.*?)</script>", getattr(admin, name), re.S))
            for bad in ("=>", "let ", "const ", "`", "class "):
                self.assertNotIn(bad, src, f"{name}: {bad!r} is not ES5 syntax")

    def test_job_tick_starts_one_timer_however_often_it_is_embedded(self):
        # The Dashboard and the job lists each append _JOB_TICK; a page carrying it twice
        # ran two setIntervals rewriting the same cells.
        if not shutil.which("node"):
            self.skipTest("node not installed")
        js = re.findall(r"<script>(.*?)</script>", admin._JOB_TICK, re.S)[0]
        prog = ("var n=0,window=this;function setInterval(){n++;}"
                "var document={querySelectorAll:function(){return [];}};"
                + js + ";" + js + ";console.log(n);")
        p = subprocess.run(["node", "-e", prog], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "1")

    def test_select_all_box_follows_single_boxes(self):
        # gwTogAll set every box of a group, but ticking or unticking ONE box never
        # updated the group's "all" box, which then claimed the opposite of the rows.
        src = admin._USER_ACC_JS
        self.assertIn("gwTogAll", src)
        self.assertIn("addEventListener('change'", src)
        self.assertIn("data-grp-all", src)


class FieldRows(unittest.TestCase):
    """Reported 2026-09-23 after the review deploy: the Users API-key input shrank to
    82 px behind its Generate/Copy buttons, every hint row was cut off on the right, and
    the Server tab's scan ports list did not fit its field — each only at a window a bit
    narrower than the one the layout was checked in."""

    def test_an_input_keeps_its_width_and_its_buttons_wrap(self):
        self.assertRegex(admin._CSS, r"\.field>\.control\{[^}]*flex-wrap:wrap")
        self.assertRegex(admin._CSS, r"\.field>\.control>input:not\(\[type=checkbox\]\)[^{]*\{flex:1 1 220px")

    def test_a_hint_is_indented_inside_its_row_not_pushed_past_it(self):
        rule = re.search(r"\.field>\.fhint\{([^}]*)\}", admin._CSS.split("@media")[0]).group(1)
        self.assertNotIn("margin", rule)                  # 100 % basis + margin = overflow
        self.assertIn("padding-left:154px", rule)

    def test_a_narrow_column_stacks_label_over_control(self):
        self.assertIn("container-type:inline-size", admin._CSS)
        self.assertIn("@container (max-width:560px)", admin._CSS)

    def test_list_settings_use_the_whole_column(self):
        row = admin._srv_runtime_row("scan_ports", "text", "scan ports", "n", "8080, 8000")
        self.assertIn('class="control wide"', row)
        self.assertNotIn("wide", admin._srv_runtime_row("max_parked", "int", "max", "n", 1))

    def test_logout_is_a_button_at_the_right_edge(self):
        saved = admin._ui_locked
        admin._ui_locked = lambda: True
        try:
            nav = admin._nav("dashboard")
        finally:
            admin._ui_locked = saved
        self.assertIn('<a class="btn secondary sm" href="/ui/logout">Logout</a>', nav)
        self.assertIn('class="hdr-right"', nav)


if __name__ == "__main__":
    unittest.main()
