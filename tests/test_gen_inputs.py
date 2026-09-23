"""What /v1/generations makes of a client's reference images.

Why this fails SILENTLY: an `images` value the gateway could not read — a URL that
answers 404, a truncated base64 string — was simply dropped. The job then ran on the
slot's placeholder (or the workflow's baked-in image) and came back `done`: a plausible
picture of the wrong thing, and nothing anywhere saying the input never arrived.

The same outcome, a second way: `images` is filtered down to the alias's image slots,
and the slots were read from the STORED `workflow_json` only. A config alias that names
its workflow by FILE (`workflow: <path>`) has none, so it had no slots, and every
reference image was dropped with an INFO line — the adapter itself loads that file and
would have matched them.

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


WF = {"1": {"class_type": "LoadImage", "inputs": {"image": "default.png"}},
      "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}}}
MAPPING = {"input_image": {"node": "1", "field": "image", "label": "front"},
           "prompt": {"node": "2", "field": "text"}}


class PathWorkflowSlots(unittest.TestCase):
    """A `workflow: <path>` alias (no workflow_json) keeps its reference images."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "wf.json")
        with open(self.path, "w") as f:
            import json
            json.dump(WF, f)
        self.cand = {"backend": "b", "model": "m", "workflow": self.path, "mapping": MAPPING}
        names = ("get_gen_routes", "_decode_ref_image", "run_generation",
                 "api_key", "users", "_users_by_key")
        self._saved = {n: getattr(main, n) for n in names}
        main.api_key, main.users, main._users_by_key = "", [], {}
        main.get_gen_routes = lambda alias: [({"name": "b"}, self.cand)]
        self.handed = None

        async def dec(ref):
            return b"img"

        async def run(body, request, upload_images=None, upload_files=None):
            self.handed = upload_images
            raise HTTPException(418, "stop")
        main._decode_ref_image = dec
        main.run_generation = run

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(main, n, v)
        self.tmp.cleanup()

    def test_slots_come_from_the_workflow_file(self):
        self.assertEqual(main._gen_image_slots("a"), ["input_image"])
        self.assertEqual(main._gen_image_slot_names("a"), {"input_image", "front"})

    def test_generations_keeps_the_image(self):
        from fastapi.testclient import TestClient
        r = TestClient(main.app).post("/v1/generations", json={
            "model": "a", "prompt": "p", "images": {"front": "http://x/1.png"}})
        self.assertEqual(r.status_code, 418)
        self.assertEqual(self.handed, {"front": b"img"})

    def test_an_unreadable_workflow_does_not_filter(self):
        self.cand["workflow"] = os.path.join(self.tmp.name, "missing.json")
        from fastapi.testclient import TestClient
        r = TestClient(main.app).post("/v1/generations", json={
            "model": "a", "prompt": "p", "images": {"whatever": "http://x/1.png"}})
        self.assertEqual(r.status_code, 418)
        self.assertEqual(self.handed, {"whatever": b"img"})

    def test_shim_maps_positionally_onto_the_file_workflow(self):
        from fastapi.testclient import TestClient
        r = TestClient(main.app).post("/v1/images/generations", json={
            "model": "a", "prompt": "p", "ref_images": ["u1"]})
        self.assertEqual(r.status_code, 418)
        self.assertEqual(self.handed, {"input_image": b"img"})


if __name__ == "__main__":
    unittest.main()
