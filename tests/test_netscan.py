"""LAN scan for backends (netscan.py + the console around it).

Why this fails SILENTLY: a wrong address range scans nothing (or a whole site), a
misclassified server pre-fills a backend the gateway then cannot discover, and a
"known" match that misses re-offers every backend that is already there. None of
that raises; the panel just shows a plausible list.
"""
import asyncio
import json
import logging
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import netscan  # noqa: E402

# `import main` reads ./config.yaml at import time — give it a minimal one in a temp cwd.
import tempfile  # noqa: E402
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
try:
    import main   # noqa: E402
    import admin  # noqa: E402
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

logging.getLogger("httpx").setLevel(logging.WARNING)

IP_ADDR = """\
1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever preferred_lft forever
2: eth0    inet 192.168.8.244/24 brd 192.168.8.255 scope global eth0\\       valid_lft forever preferred_lft forever
3: enp1s0f1np1    inet 10.20.0.1/30 scope global enp1s0f1np1\\       valid_lft forever preferred_lft forever
4: big    inet 10.0.5.7/16 brd 10.0.255.255 scope global big\\       valid_lft forever preferred_lft forever
5: eth0    inet6 fe80::1/64 scope link\\       valid_lft forever preferred_lft forever
"""


class Addresses(unittest.TestCase):
    def test_own_subnets_from_ip_addr(self):
        self.assertEqual(netscan.parse_ip_addr(IP_ADDR),
                         ["192.168.8.0/24", "10.20.0.0/30", "10.0.5.0/24"])

    def test_loopback_and_ipv6_are_ignored(self):
        self.assertEqual(netscan.parse_ip_addr("1: lo inet 127.0.0.1/8 scope host lo\n"
                                               "2: e inet6 fe80::1/64 scope link\n"), [])

    def test_local_cidrs_uses_injected_runner_and_survives_failure(self):
        seen = []

        def run(argv):
            seen.append(argv)
            return IP_ADDR
        self.assertEqual(netscan.local_cidrs(run)[0], "192.168.8.0/24")
        self.assertEqual(seen[0], ["ip", "-o", "-4", "addr", "show"])

        def boom(argv):
            raise FileNotFoundError("no ip")
        self.assertEqual(netscan.local_cidrs(boom), [])

    def test_parse_ports(self):
        self.assertEqual(netscan.parse_ports("8080, 8000,x, 8080 ,0, 70000, 8188"), [8080, 8000, 8188])
        self.assertEqual(netscan.parse_ports(""), [])

    def test_expand_targets_drops_network_and_broadcast_and_caps(self):
        hosts, truncated = netscan.expand_targets(["192.168.8.0/24"])
        self.assertEqual(len(hosts), 254)
        self.assertEqual(hosts[0], "192.168.8.1")
        self.assertNotIn("192.168.8.0", hosts)
        self.assertNotIn("192.168.8.255", hosts)
        self.assertFalse(truncated)
        hosts, truncated = netscan.expand_targets(["10.0.0.0/16"], cap=100)
        self.assertEqual(len(hosts), 100)
        self.assertTrue(truncated)

    def test_expand_targets_small_prefix_and_single_host(self):
        hosts, _ = netscan.expand_targets(["10.20.0.0/30", "127.0.0.1/32", "junk", "10.20.0.0/30"])
        self.assertEqual(hosts, ["10.20.0.1", "10.20.0.2", "127.0.0.1"])


def _fetcher(routes: dict):
    """fetch(url) → (status, json) from a {path: (status, obj)} table; unknown → (404, None)."""
    async def fetch(url):
        rest = url.split("//", 1)[1]
        path = "/" + rest.split("/", 1)[1] if "/" in rest else "/"
        return routes.get(path, (404, None))
    return fetch


def _fp(routes):
    return asyncio.run(netscan.fingerprint(_fetcher(routes), "http://10.0.0.5:8080"))


class Fingerprint(unittest.TestCase):
    def test_llama_swap(self):
        f = _fp({"/v1/models": (200, {"data": [{"id": "a", "owned_by": "llama-swap"},
                                               {"id": "b", "owned_by": "llama-swap"}]})})
        self.assertEqual((f.type, f.flavor, f.models, f.needs_key), ("openai", "llama-swap", 2, False))
        self.assertEqual(f.url, "http://10.0.0.5:8080")

    def test_llama_cpp_vllm_ollama_flavors(self):
        for owned, flavor in (("llamacpp", "llama.cpp"), ("vllm", "vLLM"), ("library", "ollama"),
                              ("acme", "openai-compatible")):
            f = _fp({"/v1/models": (200, {"data": [{"id": "m", "owned_by": owned}]})})
            self.assertEqual(f.flavor, flavor, owned)

    def test_bare_list_payload_counts(self):
        f = _fp({"/v1/models": (200, [{"id": "m"}, {"id": "n"}])})
        self.assertEqual((f.type, f.models), ("openai", 2))

    def test_needs_key(self):
        for st in (401, 403):
            f = _fp({"/v1/models": (st, {"error": "no"})})
            self.assertEqual((f.type, f.needs_key, f.models), ("openai", True, None))

    def test_comfyui_via_system_stats(self):
        f = _fp({"/system_stats": (200, {"system": {"comfyui_version": "0.3.40"}})})
        self.assertEqual((f.type, f.flavor), ("comfyui", "ComfyUI 0.3.40"))

    def test_ollama_native_api_only(self):
        f = _fp({"/api/tags": (200, {"models": [{"name": "llama3"}]})})
        self.assertEqual((f.type, f.flavor, f.models), ("openai", "ollama", 1))

    def test_plain_http_server_is_not_a_finding(self):
        self.assertIsNone(_fp({"/": (200, {"hello": 1})}))
        self.assertIsNone(_fp({"/v1/models": (200, {"nope": []})}))

    def test_fetch_exception_is_not_a_finding(self):
        async def fetch(url):
            raise ConnectionError("reset")
        self.assertIsNone(asyncio.run(netscan.fingerprint(fetch, "http://10.0.0.5:8080")))


class Known(unittest.TestCase):
    B = [{"name": "dx10-01", "type": "openai", "url": "http://192.168.8.35:8080"},
         {"name": "dx10-02", "type": "comfyui", "url": "https://192.168.8.36:8188/"}]

    def test_same_host_and_port_matches_whatever_scheme_or_slash(self):
        self.assertEqual(netscan.known_backend_for("http://192.168.8.35:8080/", self.B), "dx10-01")
        self.assertEqual(netscan.known_backend_for("http://192.168.8.36:8188", self.B), "dx10-02")

    def test_other_port_or_host_does_not(self):
        self.assertIsNone(netscan.known_backend_for("http://192.168.8.35:8000", self.B))
        self.assertIsNone(netscan.known_backend_for("http://192.168.8.37:8080", self.B))

    def test_default_ports_by_scheme(self):
        self.assertEqual(netscan.host_port("http://a"), ("a", 80))
        self.assertEqual(netscan.host_port("https://a/"), ("a", 443))
        self.assertIsNone(netscan.host_port("nonsense"))


class _Swap(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "glm", "owned_by": "llama-swap"}]}).encode()
            self.send_response(200)
        else:
            body = b"{}"
            self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Scan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _Swap)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_port
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        cls.closed_port = s.getsockname()[1]
        s.close()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_end_to_end_against_a_real_socket(self):
        import httpx

        async def go():
            async with httpx.AsyncClient(timeout=2.0) as c:
                async def fetch(url):
                    r = await c.get(url)
                    try:
                        return r.status_code, r.json()
                    except ValueError:
                        return r.status_code, None
                progress = []
                res = await netscan.scan(["127.0.0.1"], [self.port, self.closed_port], fetch=fetch,
                                         resolve=lambda h: "localhost",
                                         backends=[{"name": "me", "url": f"http://127.0.0.1:{self.port}"}],
                                         on_progress=lambda d, t: progress.append((d, t)))
                return res, progress
        res, progress = asyncio.run(go())
        self.assertFalse(res.running)
        self.assertEqual(len(res.findings), 1)
        f = res.findings[0]
        self.assertEqual((f.host, f.port, f.flavor, f.known_as, f.hostname),
                         ("127.0.0.1", self.port, "llama-swap", "me", "localhost"))
        self.assertEqual(progress[-1], (1, 1))
        self.assertEqual((res.hosts_done, res.hosts_total), (1, 1))
        self.assertGreater(res.finished, 0)

    def test_result_is_filled_in_place_and_errors_are_recorded(self):
        async def fetch(url):
            return 404, None
        res = netscan.ScanResult(cidrs=["x"], ports=[1], hosts_total=1)

        async def resolve(host):
            raise RuntimeError("dns down")
        out = asyncio.run(netscan.scan(["127.0.0.1"], [self.closed_port], fetch=fetch, resolve=resolve,
                                       result=res))
        self.assertIs(out, res)
        self.assertEqual(res.findings, [])
        self.assertIsNone(res.error)           # a closed port and a failing resolver are normal
        self.assertFalse(res.running)


class MainState(unittest.TestCase):
    def test_status_before_any_scan(self):
        main._scan["task"], main._scan["result"] = None, None
        st = main.scan_status()
        self.assertEqual((st["running"], st["findings"], st["hosts_done"], st["hosts_total"]),
                         (False, [], 0, 0))

    def test_status_reflects_a_result_and_flattens_findings(self):
        res = netscan.ScanResult(cidrs=["127.0.0.1/32"], ports=[1], hosts_total=1, hosts_done=1,
                                 finished=1.0)
        res.findings.append(netscan.Finding("127.0.0.1", 1, "http://127.0.0.1:1", "openai",
                                            "llama-swap", 3, known_as="me"))
        main._scan["task"], main._scan["result"] = None, res
        st = main.scan_status()
        self.assertFalse(st["running"])
        self.assertEqual(st["findings"][0]["known_as"], "me")
        self.assertEqual(st["findings"][0]["flavor"], "llama-swap")
        self.assertEqual(st["cidrs"], ["127.0.0.1/32"])
        main._scan["result"] = None

    def test_start_is_a_noop_while_one_runs(self):
        res = netscan.ScanResult(cidrs=[], ports=[], hosts_total=1)   # running: finished == 0

        class _T:
            def done(self):
                return False
        main._scan["task"], main._scan["result"] = _T(), res
        self.assertFalse(main.start_scan())
        main._scan["task"], main._scan["result"] = None, None

    def test_settings_parse(self):
        saved = (main.scan_cidrs, main.scan_ports)
        try:
            main.scan_cidrs, main.scan_ports = [], []
            main._apply_scan_settings({"scan_cidrs": "192.168.8.0/24, 10.20.0.0/30", "scan_ports": "8080 8188"})
            self.assertEqual(main.scan_cidrs, ["192.168.8.0/24", "10.20.0.0/30"])
            self.assertEqual(main.scan_ports, [8080, 8188])
            main._apply_scan_settings({"scan_cidrs": "", "scan_ports": ""})
            self.assertEqual(main.scan_cidrs, [])
            self.assertEqual(main.scan_ports, netscan.DEFAULT_PORTS)
        finally:
            main.scan_cidrs, main.scan_ports = saved


if __name__ == "__main__":
    unittest.main()
