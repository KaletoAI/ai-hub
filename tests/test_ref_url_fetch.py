"""Client-supplied URLs the gateway fetches (reference images, `files`).

Why this fails SILENTLY: a reference image given as a URL is downloaded BY THE GATEWAY
and kept readable at /v1/jobs/<id>/input/<n>. With no target filter that is a
read-anything proxy for every key holder — `http://127.0.0.1:4000/ui/users`, a backend's
admin port, a router page, 169.254.169.254 — and the job just runs (or fails) like any
other; nothing marks the request as unusual (review 2026-09-23, S5). So: every address
the host resolves to must be public (IPv4-mapped IPv6 judged as the IPv4 it carries), the
connection goes to exactly the address that was checked with the original Host header
(no second lookup to rebind), redirects are not followed, the body is counted while it
streams, and `ref_url_allow_cidrs` opens a chosen private range (the LAN NAS). And only
keys that ARE image slots of the alias are fetched and stored at all — the adapter
ignored the others, but they were still downloaded and kept as job inputs.
"""
import asyncio
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
    import main
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import httpx  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class AddressRule(unittest.TestCase):
    def test_blocked(self):
        for a in ("127.0.0.1", "10.1.2.3", "192.168.8.10", "172.16.0.1", "169.254.169.254",
                  "100.64.0.1", "0.0.0.0", "224.0.0.1", "::1", "fe80::1%eth0", "fd00::1",
                  "::ffff:127.0.0.1", "::ffff:192.168.1.1", "not-an-ip"):
            self.assertTrue(main.ref_addr_blocked(a), a)

    def test_public(self):
        for a in ("93.184.216.34", "1.1.1.1", "2606:4700::1111"):
            self.assertFalse(main.ref_addr_blocked(a), a)

    def test_allow_list_opens_a_range(self):
        import ipaddress
        nas = [ipaddress.ip_network("192.168.8.0/24")]
        self.assertFalse(main.ref_addr_blocked("192.168.8.20", nas))
        self.assertTrue(main.ref_addr_blocked("192.168.9.20", nas))
        self.assertFalse(main.ref_addr_blocked("::ffff:192.168.8.20", nas))


class Fetch(unittest.TestCase):
    def setUp(self):
        self._saved = (main.http_client, main._resolve_ref_host, main._REF_FETCH_MAX_BYTES,
                       dict(main.config))
        self.seen = []
        self.dns = {"img.example": ["93.184.216.34"]}

        async def resolve(host, port):
            if host in self.dns:
                return list(self.dns[host])
            return [host.strip("[]")]
        main._resolve_ref_host = resolve
        self.status, self.body, self.headers = 200, b"PNGDATA", {}

        def handler(req):
            self.seen.append(req)
            return httpx.Response(self.status, content=self.body, headers=self.headers)
        main.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def tearDown(self):
        (main.http_client, main._resolve_ref_host, main._REF_FETCH_MAX_BYTES, cfg) = self._saved
        main.config.clear()
        main.config.update(cfg)

    def _get(self, url):
        return asyncio.run(main._decode_ref_blob(url))

    def test_public_host_is_fetched_pinned_to_the_checked_address(self):
        got = self._get("https://img.example/a/b.png")
        self.assertEqual(got, (b"PNGDATA", "png"))
        req = self.seen[0]
        self.assertEqual(req.url.host, "93.184.216.34")
        self.assertEqual(req.headers["host"], "img.example")
        self.assertEqual(req.extensions.get("sni_hostname"), "img.example")

    def test_private_targets_are_refused_before_any_request(self):
        for url in ("http://127.0.0.1:4000/ui/users", "http://169.254.169.254/latest/meta-data",
                    "http://[::1]/x", "http://[::ffff:10.0.0.1]/x"):
            with self.assertRaises(HTTPException) as cm:
                self._get(url)
            self.assertEqual(cm.exception.status_code, 400, url)
        self.assertEqual(self.seen, [])

    def test_every_resolved_address_counts(self):
        self.dns["mixed.example"] = ["93.184.216.34", "192.168.1.1"]
        with self.assertRaises(HTTPException):
            self._get("http://mixed.example/x.png")
        self.assertEqual(self.seen, [])

    def test_allow_cidrs_opt_in(self):
        main.config["ref_url_allow_cidrs"] = ["192.168.8.0/24"]
        self.assertEqual(self._get("http://192.168.8.20/x.png"), (b"PNGDATA", "png"))
        with self.assertRaises(HTTPException):
            self._get("http://192.168.9.20/x.png")

    def test_redirect_is_not_followed(self):
        self.status, self.headers = 302, {"location": "http://127.0.0.1/secret"}
        self.assertIsNone(self._get("http://img.example/x.png"))
        self.assertEqual(len(self.seen), 1)

    def test_body_is_capped_while_streaming(self):
        main._REF_FETCH_MAX_BYTES = 1000
        self.body = b"x" * 5000
        with self.assertRaises(HTTPException) as cm:
            self._get("http://img.example/x.png")
        self.assertEqual(cm.exception.status_code, 413)


class OnlyRealSlots(unittest.TestCase):
    """/v1/generations `images` and the shims' `ref_images`: nothing beyond the slots."""

    def setUp(self):
        names = ("_gen_image_slot_names", "_gen_image_slots", "_decode_ref_image", "run_generation",
                 "api_key", "users", "_users_by_key")
        self._saved = {n: getattr(main, n) for n in names}
        main.api_key, main.users, main._users_by_key = "", [], {}
        self.decoded, self.handed = [], None

        async def dec(ref):
            self.decoded.append(ref)
            return b"img"

        async def run(body, request, upload_images=None, upload_files=None):
            self.handed = upload_images
            raise HTTPException(418, "stop")
        main._decode_ref_image = dec
        main.run_generation = run
        main._gen_image_slot_names = lambda alias: {"image", "input_image"}
        main._gen_image_slots = lambda alias: ["image"]
        self.c = TestClient(main.app)

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(main, n, v)

    def test_unknown_image_keys_are_neither_fetched_nor_kept(self):
        r = self.c.post("/v1/generations", json={
            "model": "a", "prompt": "p",
            "images": {"input_image": "http://x/1.png", "evil": "http://x/2.png"}})
        self.assertEqual(r.status_code, 418)
        self.assertEqual(self.decoded, ["http://x/1.png"])
        self.assertEqual(set(self.handed), {"input_image"})

    def test_ref_images_beyond_the_slot_count_are_not_fetched(self):
        r = self.c.post("/v1/images/generations", json={
            "model": "a", "prompt": "p", "ref_images": ["u1", "u2", "u3"]})
        self.assertEqual(r.status_code, 418)
        self.assertEqual(self.decoded, ["u1"])


if __name__ == "__main__":
    unittest.main()
