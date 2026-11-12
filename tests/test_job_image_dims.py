"""The job page captions an output image with its REAL pixel size.

Why this fails silently: the gallery scales every image to a 260 px thumbnail, so a
wrong caption (the thumbnail's size, an input's size, a stale cache entry) still looks
perfectly plausible, and it is exactly the number someone checks to see whether an
upscale or a 1536² request really landed.

Run: venv/bin/python -m unittest tests.test_job_image_dims -v
"""
import io
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import admin  # noqa: E402
import jobs  # noqa: E402
from PIL import Image  # noqa: E402


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 20, 30)).save(buf, "PNG")
    return types.SimpleNamespace(data=buf.getvalue(), mime="image/png", kind="image",
                                 name="out.png")


class ImageDims(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active)
        jobs.init(os.path.join(self._tmp.name, "jobs.db"), os.path.join(self._tmp.name, "blobs"))

    def tearDown(self):
        jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active = self._saved
        self._tmp.cleanup()

    def _job(self, *blobs):
        jid = jobs.create("text2img", "a", "gpu")
        jobs.set_status(jid, "running")
        jobs.complete(jid, list(blobs), {"backend": "gpu"})
        return jid, jobs.get(jid)["results"]

    def test_result_image_shows_its_real_size(self):
        jid, res = self._job(_png(1536, 864))
        html = admin._job_thumbs(jid, "result", res)
        self.assertIn("1536 × 864 px", html)

    def test_each_image_gets_its_own_size(self):
        jid, res = self._job(_png(64, 32), _png(300, 200))
        html = admin._job_thumbs(jid, "result", res)
        self.assertLess(html.index("64 × 32 px"), html.index("300 × 200 px"))

    def test_unreadable_file_gets_no_caption_and_no_error(self):
        blob = types.SimpleNamespace(data=b"not an image", mime="image/png", kind="image",
                                     name="x.png")
        jid, res = self._job(blob)
        html = admin._job_thumbs(jid, "result", res)
        self.assertNotIn(" px</div>", html)
        self.assertIn("<img", html)


if __name__ == "__main__":
    unittest.main()
