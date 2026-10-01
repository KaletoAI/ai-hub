"""Model sources, Stage 3: Check & save — a public URL (or a Hugging Face repo for a
directory) checked against the share before it becomes a catalog source.

Why this file exists (every case fails SILENTLY):
- The HEAD is the gateway reaching out on an operator's word, with the HF token. A
  redirect hop that is not checked like the first is an SSRF with read-back (the size
  and etag land in the panel); a token sent past the first hop hands it to whatever the
  redirect names; and `Accept-Encoding` other than identity makes a small file's
  Content-Length the COMPRESSED size — a "size differs" refusal for a correct URL.
- HF's facts are on HF's 302 (`X-Linked-Size`/`-Etag`, `X-Repo-Commit`), not on the
  CDN's answer: reading the last hop takes the CDN's ETag (no sha256) for a content
  hash, and `X-Xet-Hash` is another hash altogether — either way a correct URL is
  refused as "hash differs", or a wrong one accepted.
- The stored sha256 must be the SHARE's: the instance verifies every download against
  it and falls back to the share's copy on a mismatch. A sha taken from the URL instead
  makes a share copy that differs from the URL's look verified.
- A directory check accepts per file on SIZE: one 404 must not refuse a whole repo dir,
  and a provisional (HF) sha the share later disagrees with must turn THAT file
  outdated — not leave a mismatch nobody sees until the download.
- One check at a time, its hashes queued behind transfer hashes, and one lock for every
  writer of `modelsync_catalog` — an interleaved read-modify-write loses an entry.

Run: /home/dev/projekte/ai-hub/venv/bin/python -m unittest tests.test_model_sources -v
"""
import asyncio
import os
import shutil
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
    import modelsync
    import store
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import httpx  # noqa: E402

TOKEN = "hf_SECRETtoken123"
SHA_A = "a1" * 32
SHA_B = "b2" * 32
SHA_X = "ee" * 32
COMMIT = "0123456789abcdef0123456789abcdef01234567"
HF = "https://huggingface.co"
PUBLIC = {"huggingface.co": ["18.1.1.1"], "hf.co": ["18.1.1.2"], "cdn-lfs.hf.co": ["18.2.2.2"],
          "mirror.example": ["93.184.216.34"], "other.example": ["93.184.216.35"],
          "evil.example": ["93.184.216.36"], "lan.example": ["172.16.5.5"],
          "mixed.example": ["93.184.216.37", "10.0.0.5"],
          "huggingface.co.evil.example": ["93.184.216.38"]}


def resp(status=200, **headers):
    return {"status": status, "headers": {k.replace("_", "-"): str(v)
                                          for k, v in headers.items()}}


class FakeLan:
    """The LanSource surface Check & save uses: a listing, the persistent sha cache, and
    a queued `sha256` (records `background`, optionally held at `gate`)."""

    def __init__(self, index, share_sha=None, cached=None):
        self.index = dict(index)
        self.share = dict(share_sha or {})          # path → what hashing the share answers
        self.cache = {p: list(r) for p, r in (cached or {}).items()}
        self.hashed, self.background = [], []
        self.gate = None
        self.fail = set()
        self.ok = True

    async def refresh(self, force=False):
        pass

    def cached(self):
        return dict(self.index) if self.ok else {}

    def configured(self):
        return self.ok

    def problem(self):
        return "" if self.ok else "not configured"

    def known_sha(self, path, size):
        r = self.cache.get(path)
        return r[1] if r and r[0] == size else None

    def sha_files(self):
        return {p: list(r) for p, r in self.cache.items()}

    async def sha256(self, path, size, background=False):
        self.background.append(background)
        self.calls = getattr(self, "calls", []) + [path]
        if self.gate is not None:
            await self.gate.wait()
        if path in self.fail:
            raise RuntimeError("source sha256 failed (rc 2): no such file")
        self.hashed.append(path)
        h = self.share[path]
        self.cache[path] = [size, h]
        return h


class _Base(unittest.IsolatedAsyncioTestCase):
    NAMES = ("http_client", "_resolve_ref_host", "modelsrc", "_thunder_hf_token",
             "_src_checks", "_src_check_tasks", "_src_confirming", "_src_confirm_tasks",
             "_src_confirm_run")

    def setUp(self):
        self._saved = {n: getattr(main, n) for n in self.NAMES}
        self._saved_store = (store._DB_PATH, store._active, store._MASTER_KEY)
        self.tmp = tempfile.mkdtemp(prefix="model-sources-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        store._MASTER_KEY = os.urandom(32)
        store.init(os.path.join(self.tmp, "store.db"))
        store.set_settings({"modelsync_catalog": []})
        main._src_checks, main._src_check_tasks, main._src_confirming = {}, {}, {}
        main._src_confirm_tasks, main._src_confirm_run = {}, {}
        self.dns = dict(PUBLIC)

        async def resolve(host, port):
            if host in self.dns:
                return list(self.dns[host])
            if host.replace(".", "").isdigit() or ":" in host:
                return [host.strip("[]")]
            raise OSError("no such host")
        main._resolve_ref_host = resolve
        self.routes, self.seen = {}, []

        def handler(req):
            url = f"https://{req.headers['host']}{req.url.raw_path.decode()}"
            self.seen.append((url, req))
            r = self.routes.get(url)
            if r is None:
                return httpx.Response(404, content=b"<html>SECRET BODY</html>")
            if isinstance(r, Exception):
                raise r
            # a HEAD answer carries no body; an error status gets one the refusal
            # must never echo
            body = b"SECRET BODY never echoed" if r["status"] >= 400 else b""
            return httpx.Response(r["status"], headers=r["headers"], content=body)
        main.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        main._thunder_hf_token = lambda: TOKEN
        self.lan = None
        main.modelsrc = lambda: self.lan

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(main, n, v)
        store._DB_PATH, store._active, store._MASTER_KEY = self._saved_store

    def catalog(self):
        return store.get_setting("modelsync_catalog")

    async def run_check(self, coro):
        msg = await coro
        tasks = list(main._src_check_tasks.values())
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
        return msg

    async def settle(self):
        for _ in range(200):
            if (not main._src_confirming and not main._src_check_tasks
                    and not main._src_confirm_tasks):
                return
            await asyncio.sleep(0.01)
        self.fail("background work did not settle")


# ── the HEAD ──────────────────────────────────────────────────────────────────────
class HeadRules(_Base):
    LFS = f"{HF}/org/repo/resolve/main/model.safetensors"
    CDN = "https://cdn-lfs.hf.co/repos/xx/yy?sig=1"

    async def test_first_hop_headers_and_no_further_connection_when_size_known(self):
        self.routes[self.LFS] = resp(302, location=self.CDN, x_linked_size=7700,
                                     x_linked_etag=f'"{SHA_A}"', x_repo_commit=COMMIT,
                                     etag='"deadbeef"')
        h = await main._head_ref_url(self.LFS)
        self.assertEqual((h.error, h.size, h.sha256, h.commit, h.hops),
                         ("", 7700, SHA_A, COMMIT, 1))
        self.assertEqual(len(self.seen), 1)        # the CDN is never contacted
        req = self.seen[0][1]
        self.assertEqual(req.method, "HEAD")
        self.assertEqual(req.url.host, "18.1.1.1")             # pinned to the checked IP
        self.assertEqual(req.headers["host"], "huggingface.co")
        self.assertEqual(req.extensions.get("sni_hostname"), "huggingface.co")
        self.assertEqual(req.headers["accept-encoding"], "identity")
        self.assertEqual(req.headers["authorization"], f"Bearer {TOKEN}")

    async def test_cdn_etag_and_xet_hash_are_ignored(self):
        """No `X-Linked-Size` on the first hop: the size comes from the CDN's
        Content-Length — the CDN's ETag and an `X-Xet-Hash` are no content sha256."""
        self.routes[self.LFS] = resp(302, location=self.CDN, x_xet_hash=SHA_B,
                                     x_repo_commit=COMMIT)
        self.routes[self.CDN] = resp(200, content_length=123456, etag=f'"{SHA_X}"',
                                     x_xet_hash=SHA_B)
        h = await main._head_ref_url(self.LFS)
        self.assertEqual((h.error, h.size, h.sha256, h.commit, h.hops),
                         ("", 123456, None, COMMIT, 2))
        cdn = self.seen[1][1]
        self.assertEqual(cdn.headers["host"], "cdn-lfs.hf.co")
        self.assertEqual(cdn.url.host, "18.2.2.2")
        self.assertEqual(cdn.extensions.get("sni_hostname"), "cdn-lfs.hf.co")
        self.assertEqual(cdn.headers["accept-encoding"], "identity")
        self.assertNotIn("authorization", cdn.headers)      # never past the first hop

    async def test_linked_etag_forms(self):
        self.assertEqual(main._linked_etag(f'W/"{SHA_A}"'), SHA_A)
        self.assertEqual(main._linked_etag(SHA_A), SHA_A)
        for junk in ("ab" * 20, SHA_A.upper(), f'"{SHA_A[:-1]}"', "", None, f"{SHA_A}x"):
            self.assertIsNone(main._linked_etag(junk), junk)

    async def test_at_most_five_redirects_each_checked(self):
        urls = [f"https://mirror.example/{i}" for i in range(7)]
        for i in range(5):
            self.routes[urls[i]] = resp(302, location=urls[i + 1])
        self.routes[urls[5]] = resp(200, content_length=10)
        h = await main._head_ref_url(urls[0])
        self.assertEqual((h.error, h.size, h.hops), ("", 10, 6))
        self.seen.clear()
        self.routes[urls[5]] = resp(302, location=urls[6])
        self.routes[urls[6]] = resp(200, content_length=10)
        h = await main._head_ref_url(urls[0])
        self.assertEqual(h.error, "more than 5 redirects")
        self.assertEqual(len(self.seen), 6)                  # the 7th never connected

    async def test_a_private_hop_is_refused_before_any_connection(self):
        for target in ("https://lan.example/x", "https://127.0.0.1/ui/users",
                       "https://[::ffff:10.0.0.1]/x", "https://mixed.example/x",
                       "https://169.254.169.254/latest"):
            self.seen.clear()
            self.routes["https://mirror.example/a"] = resp(302, location=target)
            h = await main._head_ref_url("https://mirror.example/a")
            self.assertEqual(h.error, "redirect to a private address", target)
            self.assertEqual(len(self.seen), 1, target)
        self.seen.clear()
        h = await main._head_ref_url("https://lan.example/model.safetensors")
        self.assertEqual(h.error, "the URL's host resolves to a private address")
        self.assertEqual(self.seen, [])

    async def test_relative_redirect_is_resolved_and_checked(self):
        self.routes[f"{HF}/old/repo/resolve/main/m.st"] = resp(307, location="/new/repo/resolve/main/m.st")
        self.routes[f"{HF}/new/repo/resolve/main/m.st"] = resp(200, content_length=5)
        h = await main._head_ref_url(f"{HF}/old/repo/resolve/main/m.st")
        self.assertEqual((h.error, h.size), ("", 5))
        self.assertNotIn("authorization", self.seen[1][1].headers)   # first hop only

    async def test_token_only_on_the_first_hop_and_only_to_hf(self):
        self.routes["https://mirror.example/m"] = resp(302, location=f"{HF}/x/y/resolve/main/m")
        self.routes[f"{HF}/x/y/resolve/main/m"] = resp(200, content_length=3)
        h = await main._head_ref_url("https://mirror.example/m")
        self.assertEqual(h.error, "")
        for _, req in self.seen:                # a redirect INTO hf gets no token either
            self.assertNotIn("authorization", req.headers)
        self.seen.clear()
        self.routes[f"{HF}/a/b/resolve/main/m"] = resp(302, location="https://evil.example/steal")
        self.routes["https://evil.example/steal"] = resp(200, content_length=3)
        await main._head_ref_url(f"{HF}/a/b/resolve/main/m")
        self.assertIn("authorization", self.seen[0][1].headers)
        self.assertNotIn("authorization", self.seen[1][1].headers)
        # off when the caller says so, and a token a curl config line cannot carry
        self.seen.clear()
        await main._head_ref_url(f"{HF}/a/b/resolve/main/m", hf_token_ok=False)
        self.assertNotIn("authorization", self.seen[0][1].headers)
        self.seen.clear()
        main._thunder_hf_token = lambda: 'bad"token'
        await main._head_ref_url(f"{HF}/a/b/resolve/main/m")
        self.assertNotIn("authorization", self.seen[0][1].headers)
        main._thunder_hf_token = lambda: TOKEN
        self.seen.clear()
        self.routes["https://hf.co/a/b/resolve/main/m"] = resp(200, content_length=3)
        await main._head_ref_url("https://hf.co/a/b/resolve/main/m")
        self.assertIn("authorization", self.seen[0][1].headers)

    async def test_lookalike_hosts_get_no_token_and_case_does_not_matter(self):
        """Review-3 M-1: only the exact HF host names (any case) get the token."""
        for url, host, want in (
                ("https://huggingface.co.evil.example/a/b/resolve/main/m",
                 "huggingface.co.evil.example", False),
                ("https://evil.example/huggingface.co/a/b/resolve/main/m", "evil.example", False),
                ("https://huggingface.co@evil.example/a/b/resolve/main/m", "evil.example", False),
                ("https://HUGGINGFACE.CO/a/b/resolve/main/m", "HUGGINGFACE.CO", True)):
            self.seen.clear()
            path = url.split(host, 1)[1]
            self.routes[f"https://{host}{path}"] = resp(200, content_length=3)
            h = await main._head_ref_url(url)
            self.assertEqual(h.error, "", url)
            req = self.seen[0][1]
            self.assertEqual(req.headers["host"], host)
            self.assertEqual("authorization" in req.headers, want, url)

    async def test_hf_headers_believed_from_hf_hosts_only(self):
        """Review-3 M-2: a mirror's `X-Linked-*` neither ends the HEAD early nor names a
        content sha — its file is size-only, measured on the final hop."""
        self.routes["https://mirror.example/m"] = resp(
            302, location="https://other.example/m", x_linked_size=999,
            x_linked_etag=SHA_A, x_repo_commit=COMMIT)
        self.routes["https://other.example/m"] = resp(200, content_length=10,
                                                      x_linked_etag=SHA_A)
        h = await main._head_ref_url("https://mirror.example/m")
        self.assertEqual((h.error, h.size, h.sha256, h.commit, h.hops), ("", 10, None, None, 2))

    async def test_401_after_an_hf_internal_redirect_says_why(self):
        """Review-3 M-7: the token rides on hop 1 only — a gated repo behind a rename
        answers 401 on hop 2, and the refusal names the way out (fixed text)."""
        self.routes[f"{HF}/old/r/resolve/main/m"] = resp(307, location="/new/r/resolve/main/m")
        self.routes[f"{HF}/new/r/resolve/main/m"] = resp(401)
        h = await main._head_ref_url(f"{HF}/old/r/resolve/main/m")
        self.assertEqual(h.error, "HTTP 401 after a redirect within Hugging Face — enter the "
                                  "URL the redirect names")
        self.routes["https://mirror.example/x"] = resp(401)
        self.assertEqual((await main._head_ref_url("https://mirror.example/x")).error,
                         "HTTP 401")

    async def test_fixed_refusals_never_echo_a_body(self):
        cases = [
            ("https://mirror.example/missing", None, "HTTP 404"),
            ("https://mirror.example/http", resp(302, location="http://other.example/x"),
             "redirect to a non-https URL"),
            ("https://mirror.example/noloc", resp(302), "redirect without a Location"),
            ("https://nowhere.example/x", None, "the URL's host does not resolve"),
            ("https://mirror.example/nosize", resp(200), "the URL names no size"),
            ("https://mirror.example/zero", resp(200, content_length=0), "the URL names no size"),
            ("https://mirror.example/boom", httpx.ConnectError("refused SECRET"),
             "the URL did not answer (ConnectError)"),
            ("https://mirror.example/500", resp(500), "HTTP 500"),
        ]
        for url, route, want in cases:
            if route is not None:
                self.routes[url] = route
            h = await main._head_ref_url(url)
            self.assertEqual(h.error, want, url)
            self.assertNotIn("SECRET", h.error)
        self.assertEqual((await main._head_ref_url("http://mirror.example/x")).error,
                         "the URL must be https://")


# ── Check & save: one file ────────────────────────────────────────────────────────
class CheckFile(_Base):
    PATH = "models/diffusion_models/m.safetensors"
    URL = f"{HF}/org/repo/resolve/main/m.safetensors"

    def setUp(self):
        super().setUp()
        self.lan = FakeLan({self.PATH: 1000}, share_sha={self.PATH: SHA_A})

    def lfs(self, size=1000, sha=SHA_A):
        hdr = {"x_linked_size": size, "x_repo_commit": COMMIT,
               "location": "https://cdn-lfs.hf.co/x"}
        if sha:
            hdr["x_linked_etag"] = f'"{sha}"'
        self.routes[self.URL] = resp(302, **hdr)

    async def test_lfs_sha_equal_to_the_cached_share_sha(self):
        self.lfs()
        self.lan.cache[self.PATH] = [1000, SHA_A]
        msg = await self.run_check(main.check_source(self.PATH, self.URL))
        self.assertIn("queued", msg)
        self.assertEqual(self.catalog(), [{"file": self.PATH, "url": self.URL, "size": 1000,
                                           "sha256": SHA_A, "verified": "sha256"}])
        self.assertEqual(self.lan.hashed, [])           # the cache answered
        st = main.source_checks()[self.PATH]
        self.assertEqual((st["state"], st["note"]), ("done", "verified by sha256"))
        self.assertEqual(modelsync.validate_catalog(self.catalog()), [])

    async def test_share_hashed_in_the_background_queue_when_not_cached(self):
        self.lfs()
        await self.run_check(main.check_source(self.PATH, self.URL))
        self.assertEqual(self.lan.hashed, [self.PATH])
        self.assertEqual(self.lan.background, [True])   # behind every transfer hash
        self.assertEqual(self.catalog()[0]["sha256"], SHA_A)

    async def test_lfs_sha_differs_is_refused(self):
        self.lfs(sha=SHA_B)
        await self.run_check(main.check_source(self.PATH, self.URL))
        st = main.source_checks()[self.PATH]
        self.assertEqual(st["state"], "refused")
        self.assertIn("hash differs", st["reason"])
        self.assertEqual(self.catalog(), [])

    async def test_no_lfs_sha_is_size_verified_only_and_stores_the_share_sha(self):
        """A non-LFS URL (or a mirror): accepted on its size, labelled so — and the
        stored sha256 is the SHARE's, never a hash the URL's server named."""
        url = "https://mirror.example/m.safetensors"
        self.routes[url] = resp(200, content_length=1000, etag=f'"{SHA_X}"',
                                x_xet_hash=SHA_B)
        await self.run_check(main.check_source(self.PATH, url))
        self.assertEqual(self.catalog(), [{"file": self.PATH, "url": url, "size": 1000,
                                           "sha256": SHA_A, "verified": "size"}])
        self.assertEqual(main.source_checks()[self.PATH]["note"], "size-verified only")
        self.assertNotIn("authorization", self.seen[0][1].headers)   # not an HF host
        info = modelsync.source_kinds(self.catalog(), {self.PATH: 1000})[self.PATH]
        self.assertEqual((info["kind"], info["verified"]), ("url", "size"))

    async def test_size_differs_is_refused_without_hashing(self):
        self.lfs(size=999_000_000)
        self.lan.index[self.PATH] = 7_700_000_000
        await self.run_check(main.check_source(self.PATH, self.URL))
        st = main.source_checks()[self.PATH]
        self.assertEqual(st["state"], "refused")
        self.assertEqual(st["reason"], "size differs: share 7.70 GB (7700000000 bytes), URL "
                                       "999.0 MB (999000000 bytes)")
        self.assertEqual(self.lan.hashed, [])
        self.assertEqual(self.catalog(), [])

    async def test_head_refusals_are_the_check_reason(self):
        await self.run_check(main.check_source(self.PATH, self.URL))      # no route: 404
        st = main.source_checks()[self.PATH]
        self.assertEqual((st["state"], st["reason"]), ("refused", "HTTP 404"))
        self.routes[self.URL] = resp(302, location="https://lan.example/x")
        await self.run_check(main.check_source(self.PATH, self.URL))
        self.assertEqual(main.source_checks()[self.PATH]["reason"],
                         "redirect to a private address")
        self.assertEqual(self.catalog(), [])

    async def test_refusals_before_any_head(self):
        self.assertIn("not checked", await main.check_source(self.PATH, "http://x/y"))
        self.assertIn("not checked", await main.check_source("models/../etc", self.URL))
        self.assertEqual(main.source_checks(), {})
        await self.run_check(main.check_source("models/vae/unlisted.st", self.URL))
        self.assertEqual(main.source_checks()["models/vae/unlisted.st"]["reason"],
                         "the share does not list this file")
        self.lan.ok = False
        await self.run_check(main.check_source(self.PATH, self.URL))
        self.assertIn("press List now", main.source_checks()[self.PATH]["reason"])
        self.assertEqual(self.seen, [])

    async def test_share_hash_failure_is_refused(self):
        self.lfs()
        self.lan.fail.add(self.PATH)
        await self.run_check(main.check_source(self.PATH, self.URL))
        st = main.source_checks()[self.PATH]
        self.assertEqual(st["state"], "refused")
        self.assertIn("share's sha256 could not be computed", st["reason"])
        self.assertNotIn("rc 2", st["reason"])
        self.assertEqual(self.catalog(), [])

    async def test_entry_replaced_in_place_others_untouched(self):
        other = {"match": {"alias": "x"}, "paths": ["models/vae/"]}
        old = {"file": self.PATH, "url": "https://mirror.example/old"}
        tail = {"file": "models/vae/v.st", "url": "https://mirror.example/v"}
        store.set_settings({"modelsync_catalog": [other, old, tail, dict(old)]})
        self.lfs()
        await self.run_check(main.check_source(self.PATH, self.URL))
        cat = self.catalog()
        self.assertEqual(cat[0], other)
        self.assertEqual(cat[1]["url"], self.URL)
        self.assertEqual(cat[2], tail)
        self.assertEqual(len(cat), 3)                  # the duplicate went too


# ── one check at a time ───────────────────────────────────────────────────────────
class OneAtATime(_Base):
    async def test_checks_queue_and_a_duplicate_is_refused(self):
        a, b = "models/vae/a.st", "models/vae/b.st"
        self.lan = FakeLan({a: 10, b: 20}, share_sha={a: SHA_A, b: SHA_B})
        self.lan.gate = asyncio.Event()
        self.routes["https://mirror.example/a"] = resp(200, content_length=10)
        self.routes["https://mirror.example/b"] = resp(200, content_length=20)
        await main.check_source(a, "https://mirror.example/a")
        await main.check_source(b, "https://mirror.example/b")
        for _ in range(30):
            await asyncio.sleep(0)
        st = main.source_checks()
        self.assertEqual((st[a]["state"], st[b]["state"]), ("hashing", "queued"))
        self.assertEqual([u for u, _ in self.seen], ["https://mirror.example/a"])  # b waits
        self.assertTrue(main.source_checks_pending())
        self.assertIn("already queued", await main.check_source(b, "https://mirror.example/b"))
        self.lan.gate.set()
        await asyncio.wait_for(asyncio.gather(*main._src_check_tasks.values()), 5)
        st = main.source_checks()
        self.assertEqual((st[a]["state"], st[b]["state"]), ("done", "done"))
        self.assertFalse(main.source_checks_pending())
        self.assertEqual([e["file"] for e in self.catalog()], [a, b])


# ── Check & save: a directory against a Hugging Face repo ─────────────────────────
class CheckDir(_Base):
    D = "models/microsoft/TRELLIS-x/"
    REPO = "microsoft/TRELLIS-x"

    def url(self, rev, rel):
        return modelsync.hf_resolve_url(self.REPO, rev, rel)

    def setUp(self):
        super().setUp()
        d = self.D
        self.lan = FakeLan(
            {d + "a.safetensors": 100, d + "b.safetensors": 200, d + "config.json": 30,
             d + "extra.json": 40, d + "c.bin": 50, d + "sub/x.safetensors": 60,
             d + "a.safetensors.part": 5, "models/elsewhere.st": 1},
            share_sha={d + "b.safetensors": SHA_B, d + "config.json": "c0" * 32,
                       d + "sub/x.safetensors": SHA_A},
            cached={d + "a.safetensors": [100, SHA_A]})
        # a.safetensors sorts first: answered at `main`, naming the commit
        self.routes[self.url("main", "a.safetensors")] = resp(
            302, x_linked_size=100, x_linked_etag=f'"{SHA_A}"', x_repo_commit=COMMIT,
            location="https://cdn-lfs.hf.co/a")
        self.routes[self.url(COMMIT, "b.safetensors")] = resp(
            302, x_linked_size=200, x_linked_etag=SHA_B, x_repo_commit=COMMIT,
            location="https://cdn-lfs.hf.co/b")
        self.routes[self.url(COMMIT, "c.bin")] = resp(200, content_length=51,
                                                      x_repo_commit=COMMIT)
        self.routes[self.url(COMMIT, "config.json")] = resp(200, content_length=30,
                                                            etag='"' + "1f" * 20 + '"',
                                                            x_repo_commit=COMMIT)
        # extra.json: not in the repo (404); sub/x: provisional, the share differs
        self.routes[self.url(COMMIT, "sub/x.safetensors")] = resp(
            302, x_linked_size=60, x_linked_etag=SHA_X, x_repo_commit=COMMIT,
            location="https://cdn-lfs.hf.co/x")

    async def test_partial_acceptance_commit_once_and_provisional_confirmation(self):
        msg = await self.run_check(main.check_dir_source(self.D.rstrip("/"), self.REPO))
        self.assertIn("queued", msg)
        # every HEAD after the first is made at the resolved commit
        heads = [u for u, _ in self.seen]
        self.assertEqual(heads[0], self.url("main", "a.safetensors"))
        self.assertTrue(all(f"/resolve/{COMMIT}/" in u for u in heads[1:]), heads)
        self.assertEqual(len(heads), 6)                 # one per share file (no .part)
        cat = self.catalog()
        self.assertEqual(len(cat), 1)
        e = cat[0]
        self.assertEqual((e["dir"], e["repo"], e["rev"]), (self.D, self.REPO, COMMIT))
        st = main.source_checks()[self.D]
        self.assertEqual(st["state"], "done")
        self.assertEqual(set(st["left_out"]), {"extra.json", "c.bin"})
        self.assertEqual(st["left_out"]["extra.json"], "HTTP 404")
        self.assertIn("size differs", st["left_out"]["c.bin"])
        await self.settle()
        e = self.catalog()[0]
        self.assertEqual(e["files"], {
            "a.safetensors": [100, SHA_A, False],        # cached share sha: final at once
            "b.safetensors": [200, SHA_B, False],        # provisional → confirmed
            "config.json": [30, "c0" * 32, False],       # non-LFS: the share's sha filled in
            "sub/x.safetensors": [60, SHA_X, True]})     # the share differs: stays provisional
        self.assertTrue(all(self.lan.background))
        self.assertEqual(sorted(self.lan.hashed), [self.D + "b.safetensors",
                                                  self.D + "config.json",
                                                  self.D + "sub/x.safetensors"])
        st = main.source_checks()[self.D]
        self.assertEqual(st["confirming"], 0)
        self.assertIn("differs from the Hugging Face copy", st["outdated"]["sub/x.safetensors"])
        self.assertFalse(main.source_checks_pending())
        self.assertEqual(modelsync.validate_catalog(self.catalog()), [])
        # …which the plan's source rules read as outdated through the share-sha cache
        kinds = modelsync.source_kinds(self.catalog(), self.lan.cached(), self.lan.sha_files())
        self.assertEqual(kinds[self.D + "sub/x.safetensors"]["kind"], "outdated")
        self.assertEqual(kinds[self.D + "b.safetensors"]["kind"], "url")
        self.assertNotIn(self.D + "extra.json", kinds)     # left out → LAN

    async def test_cached_share_sha_differing_leaves_the_file_out(self):
        self.lan.cache[self.D + "b.safetensors"] = [200, SHA_X]
        await self.run_check(main.check_dir_source(self.D, self.REPO))
        st = main.source_checks()[self.D]
        self.assertIn("hash differs", st["left_out"]["b.safetensors"])
        self.assertNotIn("b.safetensors", self.catalog()[0]["files"])

    async def test_commit_from_a_later_file_when_the_first_fails(self):
        del self.routes[self.url("main", "a.safetensors")]            # 404 at the branch
        self.routes[self.url("main", "b.safetensors")] = self.routes[self.url(COMMIT, "b.safetensors")]
        await self.run_check(main.check_dir_source(self.D, self.REPO))
        e = self.catalog()[0]
        self.assertEqual(e["rev"], COMMIT)
        self.assertNotIn("a.safetensors", e["files"])
        self.assertEqual(main.source_checks()[self.D]["left_out"]["a.safetensors"], "HTTP 404")

    async def test_refused_when_no_file_verifies(self):
        self.routes.clear()
        await self.run_check(main.check_dir_source(self.D, self.REPO))
        st = main.source_checks()[self.D]
        self.assertEqual(st["state"], "refused")
        self.assertIn("no file of this directory verified", st["reason"])
        self.assertEqual(self.catalog(), [])

    async def test_refused_when_the_answer_names_no_commit(self):
        self.routes[self.url("main", "a.safetensors")] = resp(200, content_length=100)
        await self.run_check(main.check_dir_source(self.D, self.REPO))
        self.assertIn("named no commit", main.source_checks()[self.D]["reason"])
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(self.catalog(), [])

    async def test_input_refusals(self):
        for d, repo in (("hf-cache/hub/x/", self.REPO), ("models/", self.REPO),
                        (self.D, "not a repo"), (self.D, "a--b/c"), ("models/../x/", self.REPO)):
            self.assertIn("not checked", await main.check_dir_source(d, repo), (d, repo))
        self.assertIn("not checked", await main.check_dir_source(self.D, self.REPO, "a b"))
        self.assertEqual(main.source_checks(), {})

    async def test_a_dir_entry_replaces_the_same_dir_only(self):
        f = {"file": self.D + "a.safetensors", "url": "https://mirror.example/a"}
        old = {"dir": self.D, "repo": "old/repo", "rev": "f" * 40,
               "files": {"a.safetensors": [100, None, False]}}
        store.set_settings({"modelsync_catalog": [old, f]})
        await self.run_check(main.check_dir_source(self.D, self.REPO))
        cat = self.catalog()
        self.assertEqual(cat[0]["repo"], self.REPO)
        self.assertEqual(cat[1], f)                     # a per-file entry stays (it wins)
        await self.settle()

    async def test_remove_stops_the_background_confirmation(self):
        """Review-3 I-1: a removed directory's share hashes must not keep the queue busy
        for hours — nor write anything when one was already running."""
        self.lan.gate = asyncio.Event()
        await self.run_check(main.check_dir_source(self.D, self.REPO))
        await asyncio.sleep(0.01)
        self.assertTrue(main.source_checks_pending())
        self.assertEqual(len(self.lan.calls), 1)        # the first waits at the gate
        t = main._src_confirm_tasks[self.D]
        self.assertIn("removed", await main.remove_source(self.D))
        await asyncio.gather(t, return_exceptions=True)
        self.assertTrue(t.cancelled())
        self.lan.gate.set()
        await asyncio.sleep(0.05)
        self.assertEqual(len(self.lan.calls), 1)        # no further hash after the remove
        self.assertFalse(main.source_checks_pending())
        self.assertEqual(main._src_confirming, {})
        self.assertEqual(self.catalog(), [])

    async def test_remove_cancels_the_confirm_task_on_the_loop(self):
        """Review-3 RR-1: `remove_source` is a coroutine and cancels ON the loop — a
        cancel from a worker thread may be lost and the removed dir keeps hashing."""
        import inspect
        self.assertTrue(inspect.iscoroutinefunction(main.remove_source))
        self.lan.gate = asyncio.Event()
        await self.run_check(main.check_dir_source(self.D, self.REPO))
        await asyncio.sleep(0.01)
        t = main._src_confirm_tasks[self.D]
        self.assertFalse(t.done())
        await main.remove_source(self.D)
        # cancelled by the time the remove returns (it awaited its write meanwhile)
        await asyncio.sleep(0)
        self.assertTrue(t.cancelled())
        self.assertNotIn(self.D, main._src_confirm_tasks)
        self.assertNotIn(self.D, main._src_confirm_run)

    async def test_a_recheck_replaces_the_confirmation_and_stays_pending(self):
        self.lan.gate = asyncio.Event()
        await self.run_check(main.check_dir_source(self.D, self.REPO))
        await asyncio.sleep(0.01)
        first = main._src_confirm_tasks[self.D]
        await self.run_check(main.check_dir_source(self.D, self.REPO))
        await asyncio.gather(first, return_exceptions=True)
        self.assertTrue(first.cancelled())
        second = main._src_confirm_tasks[self.D]
        self.assertIsNot(first, second)
        await asyncio.sleep(0.01)
        # the old run's end popped none of the new run's markers
        self.assertEqual(sorted(main._src_confirming),
                         [self.D + "b.safetensors", self.D + "config.json",
                          self.D + "sub/x.safetensors"])
        self.assertTrue(main.source_checks_pending())
        self.lan.gate.set()
        await self.settle()
        self.assertFalse(main.source_checks_pending())
        self.assertEqual(self.catalog()[0]["files"]["b.safetensors"], [200, SHA_B, False])
        self.assertEqual(main.source_checks()[self.D]["confirming"], 0)

    async def test_a_stale_run_writes_nothing(self):
        """A hash that was already running when its run ended finishes, but its result
        is judged under the catalog lock against the CURRENT run token."""
        entry = {"dir": self.D, "repo": self.REPO, "rev": COMMIT,
                 "files": {"b.safetensors": [200, SHA_B, True]}}
        store.set_settings({"modelsync_catalog": [entry]})
        main._src_confirm_run[self.D] = (self.D, 2)
        self.assertEqual(main._confirm_row(self.D, self.REPO, COMMIT, "b.safetensors", 200,
                                           SHA_B, (self.D, 1)), "gone")
        self.assertEqual(self.catalog(), [entry])
        self.assertEqual(main._confirm_row(self.D, self.REPO, COMMIT, "b.safetensors", 200,
                                           SHA_B, (self.D, 2)), "confirmed")
        self.assertEqual(self.catalog()[0]["files"]["b.safetensors"], [200, SHA_B, False])

    async def test_confirmations_hash_behind_a_check(self):
        """Review-3 M-3: priorities — a Check & save (1) before confirmations (2)."""
        await self.run_check(main.check_dir_source(self.D, self.REPO))
        await self.settle()
        self.assertEqual(set(self.lan.background), {2})
        a = "models/vae/a.st"
        self.lan.index[a], self.lan.share[a] = 10, SHA_A
        self.routes["https://mirror.example/a"] = resp(200, content_length=10)
        await self.run_check(main.check_source(a, "https://mirror.example/a"))
        self.assertEqual(self.lan.background[-1], 1)

    async def test_a_row_changed_meanwhile_is_not_confirmed(self):
        self.lan.gate = asyncio.Event()
        await self.run_check(main.check_dir_source(self.D, self.REPO))
        store.set_settings({"modelsync_catalog": []})   # the operator removed it
        self.lan.gate.set()
        await self.settle()
        self.assertEqual(self.catalog(), [])


# ── remove, and the one lock for every catalog writer ─────────────────────────────
class RemoveAndLock(_Base):
    async def test_remove_file_and_dir_entries(self):
        f = {"file": "models/vae/a.st", "url": "https://mirror.example/a"}
        d = {"dir": "models/x/", "repo": "o/r", "rev": COMMIT,
             "files": {"m.st": [1, None, False]}}
        m = {"match": {"alias": "z"}, "paths": ["models/vae/a.st"]}
        store.set_settings({"modelsync_catalog": [f, d, m]})
        self.assertIn("removed", await main.remove_source("models/vae/a.st"))
        self.assertEqual(self.catalog(), [d, m])        # a match entry naming it stays
        self.assertIn("removed", await main.remove_source("models/x/"))
        self.assertEqual(self.catalog(), [m])
        self.assertIn("no source entry", await main.remove_source("models/x/"))

    async def test_remove_cancels_a_waiting_check(self):
        a = "models/vae/a.st"
        self.lan = FakeLan({a: 10}, share_sha={a: SHA_A})
        self.lan.gate = asyncio.Event()
        self.routes["https://mirror.example/a"] = resp(200, content_length=10)
        await main.check_source(a, "https://mirror.example/a")
        t = main._src_check_tasks[a]
        for _ in range(20):
            await asyncio.sleep(0)
        await main.remove_source(a)
        await asyncio.gather(t, return_exceptions=True)
        self.assertTrue(t.cancelled())
        self.assertNotIn(a, main.source_checks())
        self.assertEqual(self.catalog(), [])

    async def test_every_writer_holds_the_catalog_lock(self):
        seen = []
        real = store.set_settings

        def spy(d):
            if "modelsync_catalog" in d:
                seen.append(main._catalog_lock.locked())
            return real(d)
        store.set_settings = spy
        try:
            a = "models/vae/a.st"
            self.lan = FakeLan({a: 10}, share_sha={a: SHA_A})
            self.routes["https://mirror.example/a"] = resp(200, content_length=10)
            await self.run_check(main.check_source(a, "https://mirror.example/a"))
            await main.remove_source(a)
            self.assertEqual(main.save_modelsync_catalog([]), [])
        finally:
            store.set_settings = real
        self.assertEqual(seen, [True, True, True])

    async def test_a_write_after_a_remove_is_refused(self):
        """Review-3 M-6: the worker thread a cancel cannot stop re-checks, under the
        catalog lock, that its check still exists."""
        e = {"file": "models/vae/a.st", "url": "https://mirror.example/a", "size": 3}
        self.assertEqual(main._put_entry(e, lambda x: x.get("file") == e["file"],
                                         lambda: False), ["removed while it was checked"])
        self.assertEqual(self.catalog(), [])

    async def test_editor_save_refused_when_the_catalog_changed(self):
        rendered = main.modelsync_catalog_hash()
        f = {"file": "models/vae/a.st", "url": "https://mirror.example/a"}
        store.set_settings({"modelsync_catalog": [f]})            # Check & save meanwhile
        mine = [{"file": "models/vae/b.st", "url": "https://mirror.example/b"}]
        self.assertEqual(main.save_modelsync_catalog(mine, expect_hash=rendered),
                         [main.CATALOG_STALE])
        self.assertEqual(self.catalog(), [f])
        self.assertEqual(main.save_modelsync_catalog(mine, expect_hash=main.modelsync_catalog_hash()), [])
        self.assertEqual(self.catalog(), mine)


# ── pure helpers ──────────────────────────────────────────────────────────────────
class PureRules(unittest.TestCase):
    def test_dir_check_row(self):
        row = modelsync.dir_check_row
        self.assertEqual(row(10, 10, SHA_A, SHA_A), ([10, SHA_A, False], ""))
        self.assertEqual(row(10, 10, SHA_A, None), ([10, SHA_A, True], ""))
        self.assertEqual(row(10, 10, None, SHA_B), ([10, SHA_B, False], ""))
        self.assertEqual(row(10, 10, None, None), ([10, None, False], ""))
        self.assertIsNone(row(10, 10, SHA_A, SHA_B)[0])
        self.assertIn("hash differs", row(10, 10, SHA_A, SHA_B)[1])
        self.assertIn("size differs", row(10, 11, SHA_A, SHA_A)[1])
        self.assertIn("size differs", row(10, None, None, None)[1])

    def test_verified_key(self):
        e = {"file": "models/vae/a.st", "url": "https://x.example/a", "size": 3,
             "sha256": SHA_A}
        self.assertEqual(modelsync.validate_catalog([dict(e, verified="sha256")]), [])
        self.assertEqual(modelsync.validate_catalog([dict(e, verified="size")]), [])
        self.assertTrue(modelsync.validate_catalog([dict(e, verified="yes")]))
        self.assertTrue(modelsync.validate_catalog([{"dir": "models/x/", "repo": "o/r",
                                                     "rev": COMMIT, "verified": "size",
                                                     "files": {"m": [1, None, False]}}]))

    def test_dir_source_error(self):
        self.assertEqual(modelsync.dir_source_error("models/a/b/", "o/r"), "")
        for d, r in (("models/a/b", "o/r"), ("hf-cache/hub/a/", "o/r"), ("models/", "o/r"),
                     ("models/a/", "o"), ("models/.x/", "o/r")):
            self.assertTrue(modelsync.dir_source_error(d, r), (d, r))


if __name__ == "__main__":
    unittest.main()
