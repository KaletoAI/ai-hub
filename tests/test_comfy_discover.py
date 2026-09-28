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
import unittest.mock

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


class NoCredentialsToComfy(unittest.TestCase):
    """A Thunder backend carries the Thunder API TOKEN in its `api_key` (it pays for the
    instance). ComfyUI behind the tunnel needs no credential — and a token sent there
    would sit in any log or proxy on the way. Pinned: no request the ComfyUI adapter
    makes carries `Authorization`/`x-api-key`, even with an `auth_headers` service that
    would hand one out (the OpenAI adapters' path)."""

    SECRET = "thunder-api-token-sekrit"

    def test_no_request_carries_the_backend_key(self):
        seen = []

        def handler(request):
            seen.append((request.method, request.url.path, dict(request.headers)))
            p = request.url.path
            if p == "/object_info":
                return httpx.Response(200, content=json.dumps(_OBJECT_INFO).encode())
            if p == "/queue":
                return httpx.Response(200, json={"queue_running": [[0, "p1", {}, {}]],
                                                 "queue_pending": []})
            if p == "/prompt":
                return httpx.Response(200, json={"prompt_id": "p1", "number": 1})
            if p.startswith("/history/"):
                return httpx.Response(200, json={})
            if p == "/upload/image":
                return httpx.Response(200, json={"name": "m.glb", "subfolder": ""})
            if p == "/view":
                return httpx.Response(200, content=b"x")
            return httpx.Response(200, json={})

        def client():
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))
        shared = client()
        ctx = adapters.AdapterContext(
            auth_headers=lambda b: ({"authorization": f"Bearer {b['api_key']}"}
                                    if b.get("api_key") else {}),
            inflight_inc=lambda b: None, inflight_dec=lambda b: None,
            cost_usd=lambda *a: 0.0, source_of=lambda r: "", record_call=lambda **k: None,
            log_enabled=lambda: False, http_client=lambda: shared)
        a = adapters.ComfyUIAdapter(
            {"name": "tc", "type": "comfyui", "url": "http://127.0.0.1:18188",
             "api_key": self.SECRET, "poll_interval": 0.01,
             "thunder": {"gpu_type": "a6000"}}, ctx)
        req = adapters.NormalizedRequest(
            alias="a", job_id="job1", upload_prefix="gw_job1",
            workflow_json={"9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}}},
            node_mapping={"prompt": {"node": "9", "field": "filename_prefix"}})
        real = httpx.AsyncClient

        async def go():
            async with client() as c:
                await a.discover(c)
                await a._stop_prompt(c, "http://127.0.0.1:18188", "p1")
            await a.fetch_output("x.png")
            await a.fetch_output("x.png", want_bytes=False)
            await a.upload_input(b"glTF", "m.glb")
            t = asyncio.create_task(a.generate(req))
            for _ in range(500):
                await asyncio.sleep(0.01)
                if any(p.startswith("/history/") for _, p, _ in seen):
                    break
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            await shared.aclose()

        with unittest.mock.patch.object(
                adapters.httpx, "AsyncClient",
                lambda *a_, **kw: real(transport=httpx.MockTransport(handler),
                                       **{k: v for k, v in kw.items() if k != "transport"})):
            asyncio.run(go())
        paths = {p for _, p, _ in seen}
        for want in ("/object_info", "/queue", "/interrupt", "/view", "/upload/image",
                     "/prompt"):
            self.assertIn(want, paths)
        self.assertTrue(any(p.startswith("/history/") for p in paths), paths)
        for method, path, headers in seen:
            self.assertNotIn("authorization", headers, (method, path))
            self.assertNotIn("x-api-key", headers, (method, path))
            self.assertNotIn(self.SECRET, json.dumps(headers), (method, path))


if __name__ == "__main__":
    unittest.main()
