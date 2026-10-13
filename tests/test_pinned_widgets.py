"""Pins on non-loader nodes must keep discovery choices and JSON scalar types.

A sampler/resolution pin used to silently become free text: discovery skipped
unmapped non-loaders, and the widget accepted only model-file combos. Saving also
stored every pin as a string, making numeric choices look stale on reopening.
These tests exercise parsed controls and saved values, not HTML snapshots.

Run: venv/bin/python -m unittest tests.test_pinned_widgets -v
"""
import asyncio
import copy
from html.parser import HTMLParser
import unittest
from unittest.mock import patch

import httpx
import admin


class Controls(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.controls = {}
        self.current = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("input", "select") and attrs.get("name"):
            self.controls[attrs["name"]] = dict(attrs, tag=tag, options=[])
            self.current = attrs["name"] if tag == "select" else None
        elif tag == "option" and self.current:
            self.controls[self.current]["options"].append(attrs)

    def handle_endtag(self, tag):
        if tag == "select":
            self.current = None


class PinnedWidgets(unittest.TestCase):
    def test_nonloader_choices_are_discovered_for_each_backend_tab(self):
        """Dropping pinned classes from either fetch silently loses enum widgets."""
        wf = {"7": {"class_type": "TrellisPipeline", "inputs": {"sampler": "euler"}}}
        fixed = [{"node": "7", "field": "sampler", "value": "euler"}]
        cands = [{"backend": "gpu-a", "fixed": fixed}, {"backend": "gpu-b", "fixed": fixed}]

        def response(request):
            choices = ["euler", "heun"] if request.url.host == "gpu-a" else ["euler", "ddim"]
            return httpx.Response(200, json={"TrellisPipeline": {"input": {
                "required": {"sampler": [choices]}}}})

        client = httpx.AsyncClient
        async def render():
            oi, borrowed = await admin._editor_object_info(cands, wf, {})
            return await admin._pinned_block("fixture", cands, fixed, wf, oi, borrowed)

        with patch.object(admin, "_OI_CACHE", {}), \
                patch.object(admin, "_runpod_object_info", lambda name: None), \
                patch.object(admin, "_backend_url", lambda name: "http://" + name), \
                patch.object(admin.httpx, "AsyncClient", lambda **kw: client(
                    transport=httpx.MockTransport(response), **kw)):
            controls = Controls(asyncio.run(render())).controls
        self.assertEqual([o["value"] for o in controls["fixed__7__sampler"]["options"]],
                         ["euler", "heun"])
        self.assertEqual([o["value"] for o in controls["ovr__gpu-b__7__sampler"]["options"]],
                         ["euler", "ddim"])

    def test_numeric_choices_zero_and_stale_values_remain_selected(self):
        """Truthiness/type comparisons must not drop zero or replace stale pins."""
        wf = {"7": {"class_type": "TrellisPipeline", "inputs": {
            "resolution": 0, "strength": 0.5, "enabled": False}}}
        oi = {"TrellisPipeline": admin._oi_fields({"required": {
            "resolution": ["COMBO", {"options": [0, 512, 1024]}],
            "strength": ["FLOAT", {"min": 0, "max": 1, "step": 0.01}],
            "steps": ["INT", {"min": 0, "max": 100, "step": 1}]}})}
        for value, choices in ((0, ["0", "512", "1024"]),
                               ("512", ["0", "512", "1024"]),
                               (2048, ["2048", "0", "512", "1024"])):
            with self.subTest(value=value):
                html = admin._value_control("pin", "7", "resolution", value, wf, oi)
                ctl = Controls(html).controls["pin"]
                self.assertEqual(ctl["tag"], "select")
                self.assertEqual([o["value"] for o in ctl["options"]], choices)
                self.assertEqual([o["value"] for o in ctl["options"] if "selected" in o], [str(value)])
                self.assertEqual("(stale)" in html, value == 2048)
        ctl = Controls(admin._value_control("pin", "7", "strength", 0, wf, oi)).controls["pin"]
        self.assertEqual((ctl["type"], ctl["value"], ctl["min"], ctl["max"], ctl["step"]),
                         ("number", "0", "0", "1", "0.01"))
        ctl = Controls(admin._value_control("pin", "7", "steps", 0, wf, oi)).controls["pin"]
        self.assertEqual((ctl["type"], ctl["value"], ctl["min"], ctl["max"], ctl["step"]),
                         ("number", "0", "0", "100", "1"))
        ctl = Controls(admin._value_control("pin", "7", "enabled", False, wf, oi)).controls["pin"]
        self.assertEqual([o["value"] for o in ctl["options"] if "selected" in o], ["false"])

    def test_save_keeps_scalar_types_in_primary_and_override_pins(self):
        """Form strings must not turn zero/False into missing or string-valued pins."""
        wf = {"7": {"class_type": "TrellisPipeline", "inputs": {
            "resolution": 512, "strength": 0.5, "enabled": True, "sampler": "euler"}}}
        cands = [{"backend": b, "workflow_json": copy.deepcopy(wf)} for b in ("gpu-a", "gpu-b")]
        form = {"fixed__7__resolution": "0", "fixed__7__strength": "0.25",
                "fixed__7__enabled": "false", "fixed__7__sampler": "heun",
                "ovr__gpu-b__7__resolution": "1024", "ovr__gpu-b__7__enabled": "true"}
        with patch.object(admin.store, "get", lambda alias: None):
            admin._apply_update_form(cands, form)
        for cand, expected in zip(cands, ((0, 0.25, False, "heun"), (1024, 0.25, True, "heun"))):
            values = [pin["value"] for pin in cand["fixed"]]
            self.assertEqual(values, list(expected))
            self.assertEqual([type(v) for v in values], [int, float, bool, str])
