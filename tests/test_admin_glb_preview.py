"""GLB results must show exported motion, not replace it with diagnostic idle.

Exercise both result surfaces against real stored GLBs. Guards missing native
playback/clip choice, unsafe clip labels and losing idle inspection for bare rigs.
Run: venv/bin/python -m unittest tests.test_admin_glb_preview -v
"""
import json
import os
import struct
import tempfile
import unittest
from html.parser import HTMLParser
from types import SimpleNamespace

import admin
import jobs


class Elements(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.labels = []
        self._option = False
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == "option":
            self._option = True
            self.labels.append("")

    def handle_endtag(self, tag):
        if tag == "option":
            self._option = False

    def handle_data(self, data):
        if self._option:
            self.labels[-1] += data

    def attrs(self, tag):
        return [attrs for name, attrs in self.tags if name == tag]


def glb(names=(), rig=True):
    # One triangle plus a translation track. Native clips share its two keyframes.
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "buffers": [{"byteLength": 68}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36},
                        {"buffer": 0, "byteOffset": 36, "byteLength": 8},
                        {"buffer": 0, "byteOffset": 44, "byteLength": 24}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3,
                       "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]},
                      {"bufferView": 1, "componentType": 5126, "count": 2,
                       "type": "SCALAR", "min": [0], "max": [1]},
                      {"bufferView": 2, "componentType": 5126, "count": 2,
                       "type": "VEC3"}],
        "animations": [{"name": name, "samplers": [{"input": 1, "output": 2}],
                        "channels": [{"sampler": 0, "target": {"node": 0,
                                                                 "path": "translation"}}]}
                       for name in names],
    }
    if rig:
        doc["skins"] = [{"joints": [0]}]
    header = json.dumps(doc).encode()
    header += b" " * (-len(header) % 4)
    binary = struct.pack("<17f", 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(header) + len(binary))
            + struct.pack("<I4s", len(header), b"JSON") + header
            + struct.pack("<I4s", len(binary), b"BIN\x00") + binary)


class NativeGlbResults(unittest.TestCase):
    def setUp(self):
        self.saved = {key: getattr(jobs, key) for key in
                      ("_DB_PATH", "_BLOB_DIR", "_DEFAULT_TTL", "_active")}
        self.tmp = tempfile.TemporaryDirectory()
        jobs.init(os.path.join(self.tmp.name, "jobs.db"),
                  os.path.join(self.tmp.name, "blobs"))

    def tearDown(self):
        for key, value in self.saved.items():
            setattr(jobs, key, value)
        self.tmp.cleanup()

    def store(self, names=(), rig=True):
        jid = jobs.create("mesh", "animation", "comfy")
        jobs.complete(jid, [SimpleNamespace(data=glb(names, rig),
                                           mime="model/gltf-binary", kind="file",
                                           name="motion.glb")])
        return jid, jobs.get(jid)

    def surfaces(self, jid, job):
        return (admin._job_result_html(jid, job)[0],
                admin._job_thumbs(jid, "result", job["results"]))

    def test_native_motion_plays_original_artifact_on_both_result_surfaces(self):
        jid, job = self.store(["walk"])
        for html in self.surfaces(jid, job):
            with self.subTest(html=html[:80]):
                elements = Elements(html)
                viewer, = elements.attrs("model-viewer")
                self.assertEqual(viewer["src"], f"/ui/playground/result/{jid}/0")
                self.assertIn("autoplay", viewer)
                self.assertNotIn("auto-rotate", viewer)
                self.assertNotEqual(viewer.get("animation-name"), "idle")
                checkbox, = [a for a in elements.attrs("input") if a.get("type") == "checkbox"]
                self.assertIn("checked", checkbox)
                self.assertEqual(elements.attrs("select"), [])

    def test_multiple_native_clip_names_are_safe_and_selectable(self):
        names = ["walk", '\"</option><img src=x onerror=alert(1)> & <run>']
        jid, job = self.store(names)
        for html in self.surfaces(jid, job):
            elements = Elements(html)
            self.assertEqual(elements.labels, names)
            self.assertEqual(len(elements.attrs("select")), 1)
            self.assertEqual([a["value"] for a in elements.attrs("option")], ["0", "1"])
            self.assertEqual(elements.attrs("img"), [])
            self.assertNotIn("anim=idle", elements.attrs("model-viewer")[0]["src"])

    def test_rig_without_native_motion_keeps_inspection_idle_only(self):
        jid, job = self.store()
        inspection, gallery = self.surfaces(jid, job)
        viewer, = Elements(inspection).attrs("model-viewer")
        self.assertEqual(viewer["src"], f"/ui/playground/result/{jid}/0?anim=idle")
        self.assertEqual(viewer["animation-name"], "idle")
        self.assertNotIn("anim=idle", Elements(gallery).attrs("model-viewer")[0]["src"])

    def test_unrigged_static_mesh_does_not_offer_diagnostic_motion(self):
        jid, job = self.store(rig=False)
        for html in self.surfaces(jid, job):
            elements = Elements(html)
            self.assertNotIn("anim=idle", elements.attrs("model-viewer")[0]["src"])
            self.assertEqual(elements.attrs("input"), [])



if __name__ == "__main__":
    unittest.main()
