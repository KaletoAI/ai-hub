"""What /v1/generations makes of a client's reference images.

Why this fails SILENTLY: an `images` value the gateway could not read — a URL that
answers 404, a truncated base64 string — was simply dropped. The job then ran on the
slot's placeholder (or the workflow's baked-in image) and came back `done`: a plausible
picture of the wrong thing, and nothing anywhere saying the input never arrived.

Run: venv/bin/python -m unittest tests.test_gen_inputs -v
"""
import asyncio
import base64
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
    from fastapi import HTTPException
    import main
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class RefImages(unittest.TestCase):

    def _decode(self, imgs):
        return asyncio.run(main._decode_ref_images(imgs))

    def test_a_readable_image_is_kept(self):
        self.assertEqual(self._decode({"input_image": base64.b64encode(PNG).decode()}),
                         {"input_image": PNG})

    def test_an_empty_slot_stays_empty(self):
        self.assertEqual(self._decode({"input_image": "", "back": None}), {})

    def test_broken_base64_is_a_400_naming_the_slot(self):
        with self.assertRaises(HTTPException) as cm:
            self._decode({"input_image_back": "not-base64!"})
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn("input_image_back", cm.exception.detail)

    def test_an_unreachable_url_is_a_400(self):
        async def _nothing(ref):
            return None                       # what _decode_ref_blob answers for a 404 URL
        orig = main._decode_ref_blob
        main._decode_ref_blob = _nothing
        try:
            with self.assertRaises(HTTPException) as cm:
                self._decode({"input_image": "https://example.com/missing.png"})
        finally:
            main._decode_ref_blob = orig
        self.assertIn("input_image", cm.exception.detail)


if __name__ == "__main__":
    unittest.main()
