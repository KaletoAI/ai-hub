"""The removed "playground upload" image pin (`__gw_upload__`) and its migration.

Why this fails SILENTLY: the Mapping editor offered a pinned image loader the value
"playground upload (8×8 if empty)", but nothing ever filled the upload it promised —
`NormalizedRequest.upload_image` had no writer, so such a pin ALWAYS ran on the 8×8
placeholder while the editor said it would take the playground's image (review
2026-09-23). The option is gone; a request image belongs in an image SLOT of the
mapping. Stored pins carrying the old value are rewritten to the placeholder at
startup (what they did anyway). Lost, that migration does not error either: the raw
string `__gw_upload__` would go to ComfyUI as a file name and every job on the alias
would fail at /prompt — so the migration, the resolver's legacy fallback and the
editor's option list are pinned here.
"""
import asyncio
import os
import sys
import tempfile
import types
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main  # noqa: F401  (binds admin's callbacks)
    import admin
    import adapters
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

LEGACY = "__gw_upload__"


class Migration(unittest.TestCase):
    def test_legacy_pin_becomes_the_placeholder(self):
        cands = [{"backend": "a", "fixed": [{"node": "5", "field": "image", "value": LEGACY},
                                            {"node": "6", "field": "ckpt", "value": "x.safetensors"}]},
                 {"backend": "b", "fixed": [{"node": "5", "field": "image", "value": LEGACY}]}]
        self.assertEqual(adapters.migrate_upload_pins(cands), 2)
        self.assertEqual(cands[0]["fixed"][0]["value"], adapters.PLACEHOLDER_SENTINEL)
        self.assertEqual(cands[0]["fixed"][1]["value"], "x.safetensors")
        self.assertEqual(cands[1]["fixed"][0]["value"], adapters.PLACEHOLDER_SENTINEL)
        self.assertEqual(adapters.migrate_upload_pins(cands), 0)       # idempotent
        self.assertEqual(adapters.migrate_upload_pins([{"backend": "c"}]), 0)

    def test_request_has_no_single_upload_field(self):
        self.assertFalse(hasattr(adapters.NormalizedRequest(), "upload_image"))
        self.assertFalse(hasattr(adapters, "UPLOAD_SENTINEL"))

    def test_resolver_treats_a_leftover_legacy_value_as_the_placeholder(self):
        fake = types.SimpleNamespace(ctx=types.SimpleNamespace(http_client=lambda: object()))

        async def ph(_c):
            return "gw_placeholder.png"
        fake._upload_placeholder = ph
        fixed = [{"node": "5", "field": "image", "value": LEGACY},
                 {"node": "7", "field": "image", "value": adapters.PLACEHOLDER_SENTINEL},
                 {"node": "6", "field": "ckpt", "value": "x"}]
        out = asyncio.run(adapters.ComfyUIAdapter._resolve_image_sentinels(fake, fixed))
        self.assertEqual([b["value"] for b in out], ["gw_placeholder.png", "gw_placeholder.png", "x"])


class EditorOption(unittest.TestCase):
    def test_pinned_loader_offers_only_the_placeholder(self):
        wf = {"5": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}}}
        html = admin._value_control("fixed__5__image", "5", "image", None, wf, {})
        self.assertNotIn(LEGACY, html)
        self.assertNotIn("playground upload", html)
        self.assertIn(adapters.PLACEHOLDER_SENTINEL, html)
        self.assertIn("ref.png", html)                 # the workflow's own file stays pickable


if __name__ == "__main__":
    unittest.main()
