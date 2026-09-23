"""How the /ui console reads, sorts and announces itself.

Why this fails SILENTLY: every mechanism here renders a page that looks fine in the one
browser it was written in. A missing viewport tag leaves a phone at a 980 px desktop
layout scaled to a third (nothing errors — it is just unusable); a sort that compares
"1.2 s" with "102 ms" as TEXT puts them in a plausible-looking wrong order; a label
without `for` is a label a screen reader never announces; a live page that lost its
server shows the last good numbers forever, with nothing saying they are old. So the
contract is pinned on the markup and on the JS, not eyeballed.
"""
import os
import re
import sys
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import admin  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
