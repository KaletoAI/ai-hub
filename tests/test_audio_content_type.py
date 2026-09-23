"""The content type the console serves a synthesized "audio" with.

Why this fails SILENTLY: the voice playground stashes whatever a TTS backend returned
and serves it back under the backend's own Content-Type, from the console's origin; the
call log does the same for stored audio responses. A backend (or anything answering in
its place) that returns `image/svg+xml` or `text/html` therefore gets a document with
script rendered INSIDE the /ui origin — the admin's session — the moment someone opens
the "audio" link, and the playground just shows a player that does not play (review
2026-09-23, S15). So only an `audio/*` type is served as such; anything else goes out as
`application/octet-stream` with `Content-Disposition: attachment`, and every such
response carries `X-Content-Type-Options: nosniff`.
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
    import stats
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


class VoiceAudio(unittest.TestCase):
    def setUp(self):
        self._saved = dict(admin._voice_out)
        self.req = types.SimpleNamespace(cookies={})

    def tearDown(self):
        admin._voice_out.clear()
        admin._voice_out.update(self._saved)

    def _serve(self, mime):
        admin._voice_out["default"] = (b"<svg onload='alert(1)'/>", mime)
        return asyncio.run(admin.voice_audio(self.req))

    def test_audio_is_served_as_audio(self):
        r = self._serve("audio/mpeg")
        self.assertEqual(r.media_type, "audio/mpeg")
        self.assertEqual(r.headers.get("x-content-type-options"), "nosniff")
        self.assertNotIn("attachment", r.headers.get("content-disposition", ""))

    def test_anything_else_is_a_download(self):
        for mime in ("image/svg+xml", "text/html; charset=utf-8", "application/xhtml+xml", ""):
            r = self._serve(mime)
            self.assertEqual(r.media_type, "application/octet-stream", mime)
            self.assertIn("attachment", r.headers.get("content-disposition", ""), mime)
            self.assertEqual(r.headers.get("x-content-type-options"), "nosniff")


class CallAudio(unittest.TestCase):
    def setUp(self):
        self._saved = stats.get_audio
        self.tmp = tempfile.NamedTemporaryFile(delete=False)
        self.tmp.write(b"<html><script>alert(1)</script>")
        self.tmp.close()

    def tearDown(self):
        stats.get_audio = self._saved
        os.unlink(self.tmp.name)

    def test_stored_non_audio_is_a_download(self):
        stats.get_audio = lambda cid: (self.tmp.name, "text/html")
        r = asyncio.run(admin.call_audio(1))
        self.assertEqual(r.media_type, "application/octet-stream")
        self.assertIn("attachment", r.headers.get("content-disposition", ""))

    def test_stored_audio_plays(self):
        stats.get_audio = lambda cid: (self.tmp.name, "audio/wav")
        r = asyncio.run(admin.call_audio(1))
        self.assertEqual(r.media_type, "audio/wav")


if __name__ == "__main__":
    unittest.main()
