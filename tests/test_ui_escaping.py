"""Values the console puts into JavaScript, not HTML.

Why this fails SILENTLY: `html.escape` is the right escape for HTML and the wrong one
for JavaScript inside an attribute — the browser decodes `&#x27;` back to `'` BEFORE the
handler is parsed. So `_btn(confirm="Delete voice 'n'?")` rendered an onclick that was a
SyntaxError: the handler was dropped and the link DELETED WITHOUT ASKING (review
2026-09-23, measured in Chromium: `a.onclick === null`). With an attacker-chosen text
the same hole was script injection: an unauthenticated `x-source: ::1%'+alert(1)+'`
(a valid IPv6 scope id) was stored as an IP alias and rendered into the delete button's
confirm. Likewise `json.dumps` does not escape `</script>`, so a model id reported by a
backend could close the playground's script block. Nothing errors in any of these —
the page renders, it just does the wrong thing.
"""
import html.parser
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import admin  # noqa: E402


class _Attrs(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def _tags(markup):
    p = _Attrs()
    p.feed(markup)
    return p.tags


class ConfirmButtons(unittest.TestCase):
    NASTY = ["Delete voice 'n'?", "Delete IP alias ::1%'+alert(document.domain)+'?",
             'He said "x" </script><img src=x onerror=alert(1)>', "back\\slash\nnewline"]

    def test_confirm_text_survives_as_data_not_code(self):
        for text in self.NASTY:
            markup = admin._btn("✕", href="/ui/x?a=1", confirm=text)
            (tag, attrs), = _tags(markup)
            self.assertEqual(tag, "a")
            self.assertNotIn("onclick", attrs, "confirm must not be inline JS")
            self.assertEqual(attrs.get("data-confirm"), text)

    def test_icon_actions_carry_the_confirm_too(self):
        markup = admin._icon_acts(("✕", "/ui/x", "danger", "Delete", "Delete 'x'?"))
        confirms = [a.get("data-confirm") for _, a in _tags(markup) if "data-confirm" in a]
        self.assertEqual(confirms, ["Delete 'x'?"])

    def test_every_page_carries_the_confirm_handler(self):
        for kw in ({}, {"refresh": 5}, {"nologin": True}):
            self.assertIn(admin._CONFIRM_JS, admin._page("t", "<p>x</p>", active="", **kw))

    def test_confirm_handler_is_valid_es5(self):
        if not shutil.which("node"):
            self.skipTest("node not installed")
        src = admin._CONFIRM_JS.split("<script>", 1)[1].rsplit("</script>", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.js")
            with open(path, "w") as fh:
                fh.write(src)
            p = subprocess.run(["node", "--check", path], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)


class ScriptJson(unittest.TestCase):
    def test_script_json_cannot_close_the_block(self):
        out = admin._js_json({"b": ["</script><img src=x onerror=alert(1)>", "a&b"]})
        self.assertNotIn("<", out)
        self.assertNotIn(">", out)
        import json
        self.assertEqual(json.loads(out), {"b": ["</script><img src=x onerror=alert(1)>", "a&b"]})

    def test_reorder_js_carries_no_alias(self):
        # The drop saves the editor form itself — no alias is spliced into the script.
        self.assertNotIn("location.href", admin._reorder_js())


class PlaygroundResult(unittest.TestCase):
    def test_usage_numbers_from_a_backend_are_escaped(self):
        res = {"status": 200, "backend": "b", "model": "m",
               "response": {"choices": [{"message": {"content": "hi"}}],
                            "usage": {"prompt_tokens": "<img src=x onerror=alert(1)>",
                                      "completion_tokens": 3}}}
        self.assertNotIn("<img", admin._chat_result_html(res))


if __name__ == "__main__":
    unittest.main()
