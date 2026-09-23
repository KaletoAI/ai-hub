"""ComfyUI discovery: /object_info parsed OFF the event loop, with nothing lost.

Why this fails SILENTLY: `/object_info` is several MB of JSON per ComfyUI backend,
fetched every health cycle (30 s) — and every 3 s per DOWN backend while calls wait
(fast_probe_loop). Parsed and walked on the event loop, it stalls every in-flight
request and SSE stream for the length of the parse, which shows up as nothing but
"the gateway feels sluggish". Moving the parse into a thread must not cost what the
same fetch feeds: models, installed LoRAs, the bypass slot-type cache and the
executor watchdog on /queue.

Run: venv/bin/python -m unittest tests.test_comfy_discover -v
"""
import asyncio
import json
import os
import sys
import threading
import unittest

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import adapters  # noqa: E402

_OBJECT_INFO = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["sdxl.safetensors", "flux.gguf"]]}},
                               "output": ["MODEL", "CLIP", "VAE"]},
    "LoraLoader": {"input": {"required": {"model": ["MODEL"], "lora_name": [["None", "style.safetensors"]]}},
                   "output": ["MODEL"]},
    "KSampler": {"input": {"required": {"model": ["MODEL"], "seed": ["INT", {}]}}, "output": ["LATENT"]},
}


def _adapter(queue):
    def handler(request):
        if request.url.path == "/object_info":
            return httpx.Response(200, content=json.dumps(_OBJECT_INFO).encode())
        if request.url.path == "/queue":
            return httpx.Response(200, json=queue)
        return httpx.Response(404)
    ctx = adapters.AdapterContext(auth_headers=lambda b: {}, inflight_inc=lambda b: None,
                                  inflight_dec=lambda b: None, cost_usd=lambda *a: 0.0,
                                  source_of=lambda r: "", record_call=lambda **k: None,
                                  log_enabled=lambda: False)
    a = adapters.ComfyUIAdapter({"name": "gpu", "type": "comfyui", "url": "http://comfy:8188"}, ctx)
    return a, httpx.AsyncClient(transport=httpx.MockTransport(handler))


class Discover(unittest.TestCase):
    def test_object_info_is_parsed_off_the_loop_and_feeds_everything(self):
        seen = []
        orig = adapters._comfy_node_types

        def spy(oi):
            seen.append(threading.current_thread() is threading.main_thread())
            return orig(oi)
        adapters._comfy_node_types = spy
        a, client = _adapter({"queue_running": [], "queue_pending": []})
        try:
            caps = asyncio.run(a.discover(client))
        finally:
            adapters._comfy_node_types = orig
        self.assertEqual(seen, [False])                              # a worker thread did it
        self.assertEqual(caps.models, {"sdxl.safetensors", "flux.gguf"})
        self.assertEqual(caps.loras, {"style.safetensors"})
        self.assertEqual(a._node_types["KSampler"]["out"], ["LATENT"])

    def test_the_executor_watchdog_still_runs(self):
        a, client = _adapter({"queue_running": [], "queue_pending": [[0, "p1", {}, {}, []]]})
        asyncio.run(a.discover(client))                              # baseline
        a._stuck_since -= 1000                                       # …long ago
        with self.assertRaises(adapters.ComfyExecutorStuck):
            asyncio.run(a.discover(client))


if __name__ == "__main__":
    unittest.main()
