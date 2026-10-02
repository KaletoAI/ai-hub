"""LoRA trigger words in main: the store table, the Civitai client, the background
worker, the two API routes and the console tab (spec 2026-10-02).

Why this fails SILENTLY: every failure here ends in a plausible answer. A worker that
re-asks every 404 on each pass hammers Civitai until it rate-limits the gateway; one
that persists a 503 or a challenge page as "found, no words" tells clients a LoRA has
no trigger at all; one that forgets the curated list on a refresh silently undoes the
operator's work; one that puts a file name into the request leaks the share's private
names; an API route that awaits a hash hangs a client for the length of a 2 GB read;
a console that renders Civitai's text unescaped runs a stranger's script in the admin
session. Nothing raises in any of these.
"""
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
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
    import admin
    import loratags
    import store
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

SAME = {"sec-fetch-site": "same-origin"}
URL = "https://civitai.com/api/v1/model-versions/by-hash/"


def version(words, mid=11, vid=22):
    return {"id": vid, "modelId": mid, "name": "v1", "baseModel": "Flux.1 D",
            "trainedWords": words, "model": {"name": "Test LoRA"}}


class FakeCivitai:
    """Civitai's by-hash endpoint behind an httpx.MockTransport: `answers` sha →
    (status, body, headers); unknown = 404. `log` is shared with FakeLan (order)."""

    def __init__(self, log=None):
        self.answers, self.calls = {}, []
        self.log = log if log is not None else []

    def handler(self, request):
        self.calls.append(str(request.url))
        sha = request.url.path.rsplit("/", 1)[-1]
        self.log.append(("civitai", sha))
        st, body, hdr = self.answers.get(sha, (404, {"error": "Model not found"}, {}))
        if isinstance(body, (dict, list)):
            return httpx.Response(st, json=body, headers=hdr)
        return httpx.Response(st, text=body, headers=hdr)

    def client(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


class FakeLan:
    """The LanSource surface the worker uses: a listing, the persistent sha cache
    (path → (size, sha)), and a sha256 that records its priority."""

    def __init__(self, files=None, log=None):
        self.files = dict(files or {})        # share path → bytes
        self.links = {}                       # share path → link target text
        self.problem_text = ""
        self.sha = {}                         # path → (size, sha)
        self.hashed = []                      # (path, background)
        self.fail = set()
        self.refreshes = 0
        self.log = log if log is not None else []

    async def refresh(self, force=False):
        self.refreshes += 1

    def problem(self):
        return self.problem_text

    def cached(self):
        out = {p: len(b) for p, b in self.files.items()}
        out.update({p: {"link": t} for p, t in self.links.items()})
        return out

    def sha_files(self):
        return {p: [n, h] for p, (n, h) in self.sha.items()}

    async def sha256(self, path, size, background=False):
        self.hashed.append((path, background))
        self.log.append(("hash", path))
        if path in self.fail:
            raise RuntimeError("source sha256 failed (rc 2): refused")
        h = hashlib.sha256(self.files[path]).hexdigest()
        self.sha[path] = (size, h)
        return h

    def forget_sha(self, path, size):
        self.sha.pop(path, None)


def sha_of(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class _Fixture(unittest.IsolatedAsyncioTestCase):
    """A temp store, fresh worker state, a FakeLan behind `main.modelsrc`, a FakeCivitai
    behind `main._civitai_client`; everything restored afterwards."""

    NAMES = ("lora_meta", "_lm_errors", "_lm_refetch", "_lm_state", "_lm_wake",
             "_civitai_client", "_CIVITAI_MIN_GAP_S", "modelsrc", "_modelsrc_host",
             "backend_loras", "backends", "image_models", "get_gen_routes",
             "api_key", "users", "_users_by_key")

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        saved_store = (store._DB_PATH, store._active, store._MASTER_KEY)
        store._MASTER_KEY = os.urandom(32)
        store.init(os.path.join(self.tmp.name, "store.db"))
        self._saved = {n: getattr(main, n) for n in self.NAMES}

        def restore():
            for n, v in self._saved.items():
                setattr(main, n, v)
            store._DB_PATH, store._active, store._MASTER_KEY = saved_store
            self.tmp.cleanup()
        self.addCleanup(restore)
        self.log = []
        self.lan = FakeLan(log=self.log)
        self.civ = FakeCivitai(log=self.log)
        self.http = self.civ.client()
        main.lora_meta = {}
        main._lm_errors = {}
        main._lm_refetch = set()
        main._lm_state = main._lm_initial_state()
        main._lm_wake = None
        main._civitai_client = self.http
        main._CIVITAI_MIN_GAP_S = 0.0
        main.modelsrc = lambda: self.lan
        main._modelsrc_host = lambda: "src@share"
        main.backend_loras = {}
        main.api_key, main.users, main._users_by_key = "", [], {}

    async def asyncTearDown(self):
        await self.http.aclose()

    async def drain(self, n=40):
        """Run passes until one ends without a hash (= nothing left to hash)."""
        for _ in range(n):
            if await main.lora_meta_pass() is None:
                return
        self.fail("the worker never went idle")


class StoreTable(unittest.TestCase):
    def test_put_get_survive_reinit(self):
        with tempfile.TemporaryDirectory() as d:
            saved = (store._DB_PATH, store._active)
            try:
                store.init(os.path.join(d, "s.db"))
                store.lora_meta_put("a" * 64, {"curated": ["x"]})
                store.lora_meta_put("a" * 64, {"curated": ["y"], "civitai": None})
                store.lora_meta_put("b" * 64, {"curated": None})
                store.init(os.path.join(d, "s.db"))             # a restart
                self.assertEqual(store.lora_meta_all(),
                                 {"a" * 64: {"curated": ["y"], "civitai": None},
                                  "b" * 64: {"curated": None}})
            finally:
                store._DB_PATH, store._active = saved


class CivitaiClient(_Fixture):
    async def test_found(self):
        s = "c" * 64
        self.civ.answers[s] = (200, version([" w1 ", "w1,, w2"]), {})
        kind, rec = await main._civitai_lookup(s)
        self.assertEqual(kind, "found")
        self.assertEqual((rec["model_id"], rec["version_id"], rec["trained_words"]),
                         (11, 22, ["w1", "w1, w2"]))
        self.assertEqual(self.civ.calls, [URL + s])             # only the hash leaves

    async def test_404_429_5xx(self):
        s = "d" * 64
        self.assertEqual(await main._civitai_lookup(s), ("not_found", None))
        self.civ.answers[s] = (429, {}, {"Retry-After": "120"})
        self.assertEqual(await main._civitai_lookup(s), ("busy", 120.0))
        self.civ.answers[s] = (429, {}, {})
        self.assertEqual(await main._civitai_lookup(s), ("busy", None))
        self.civ.answers[s] = (503, "busy", {})
        self.assertEqual(await main._civitai_lookup(s), ("error", "HTTP 503"))
        self.civ.answers[s] = (302, "", {"Location": "https://elsewhere/"})
        self.assertEqual(await main._civitai_lookup(s), ("error", "HTTP 302"))

    async def test_html_200_is_transient(self):
        # Review Focus 5: a challenge page under 200 is no "found, no words" record
        s = "e" * 64
        self.civ.answers[s] = (200, "<html>Just a moment…</html>", {})
        self.assertEqual(await main._civitai_lookup(s), ("error", "unparsable answer"))

    async def test_oversized_answer(self):
        s = "f" * 64
        self.civ.answers[s] = (200, "x" * (main._CIVITAI_MAX_BYTES + 10), {})
        self.assertEqual(await main._civitai_lookup(s), ("error", "answer larger than 1 MB"))

    async def test_invalid_sha_never_sent(self):
        self.assertEqual(await main._civitai_lookup("../etc/passwd"),
                         ("error", "invalid sha256"))
        self.assertEqual(self.civ.calls, [])


if __name__ == "__main__":
    unittest.main()
