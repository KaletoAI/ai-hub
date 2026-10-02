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

    async def test_pathological_answers_never_raise(self):
        s = "a1" * 32
        self.civ.answers[s] = (200, "[" * 100000, {})
        self.assertEqual(await main._civitai_lookup(s), ("error", "unparsable answer"))
        s2 = "b2" * 32
        self.civ.answers[s2] = (429, {}, {"Retry-After": "\u00b2".encode()})
        res = await main._civitai_lookup(s2)
        self.assertIsInstance(res, tuple)
        self.assertEqual(res[0], "busy", res)
        self.assertIsNone(res[1])


A, B = b"lora-a-bytes", b"lora-b-bytes-longer"
PA, PB = "models/loras/a.safetensors", "models/loras/b.safetensors"


class Worker(_Fixture):
    async def test_no_share_does_nothing(self):
        main._modelsrc_host = lambda: ""

        def boom():
            raise AssertionError("LanSource touched without a share")
        main.modelsrc = boom
        self.assertIsNone(await main.lora_meta_pass())
        self.assertEqual(self.civ.calls, [])
        self.assertEqual(main.lora_meta_view()["rows"], [])
        self.assertFalse(main._lm_snapshot()["configured"])

    async def test_share_not_usable_is_pending_not_a_verdict(self):
        self.lan.files = {PA: A}
        self.lan.problem_text = "not listed yet"
        self.assertIsNone(await main.lora_meta_pass())
        self.assertEqual(self.lan.hashed, [])
        self.assertEqual(loratags.lookup("a.safetensors", main._lm_snapshot())["status"],
                         "pending")

    async def test_hash_priority_3_one_at_a_time_then_civitai_before_next_hash(self):
        self.lan.files = {PA: A, PB: B}
        self.civ.answers[sha_of(A)] = (200, version(["wa"]), {})
        pause = await main.lora_meta_pass()
        self.assertGreaterEqual(pause, main._LM_HASH_PAUSE_MIN_S)
        self.assertEqual(self.lan.hashed, [(PA, 3)])            # ONE hash, priority 3
        await self.drain()
        self.assertEqual(self.log, [("hash", PA), ("civitai", sha_of(A)),
                                    ("hash", PB), ("civitai", sha_of(B))])
        self.assertEqual(main.lora_meta[sha_of(A)]["civitai"]["trained_words"], ["wa"])
        self.assertEqual(main.lora_meta[sha_of(B)]["civitai"]["status"], "not_found")
        self.assertEqual(store.lora_meta_all()[sha_of(B)]["civitai"]["status"], "not_found")

    async def test_backend_offered_loras_first(self):
        self.lan.files = {PA: A, PB: B}
        main.backend_loras = {"comfyui:k": {"b.safetensors"}}
        await main.lora_meta_pass()
        self.assertEqual(self.lan.hashed, [(PB, 3)])

    async def test_404_persisted_and_not_asked_again(self):
        self.lan.files = {PA: A}
        await self.drain()
        await self.drain()
        self.assertEqual(self.civ.calls, [URL + sha_of(A)])

    async def test_only_the_hash_leaves(self):
        self.lan.files = {"models/loras/Private-Name.safetensors": A}
        await self.drain()
        self.assertEqual(self.civ.calls, [URL + sha_of(A)])
        self.assertNotIn("Private", " ".join(self.civ.calls))

    async def test_429_pauses_everything_and_persists_nothing(self):
        self.lan.files = {PA: A, PB: B}
        self.lan.sha = {PA: (len(A), sha_of(A)), PB: (len(B), sha_of(B))}
        self.civ.answers[sha_of(A)] = (429, {}, {"Retry-After": "120"})
        self.civ.answers[sha_of(B)] = (429, {}, {"Retry-After": "120"})
        t0 = time.time()
        await main.lora_meta_pass()
        self.assertEqual(len(self.civ.calls), 1)                # the second is not asked
        self.assertGreaterEqual(main._lm_state["pause_until"], t0 + 119)
        await main.lora_meta_pass()
        self.assertEqual(len(self.civ.calls), 1)                # still paused
        self.assertEqual(store.lora_meta_all(), {})
        self.assertIn("429", main.lora_meta_view()["last_error"])

    async def test_5xx_backs_off_per_sha_memory_only(self):
        self.lan.files = {PA: A}
        self.lan.sha = {PA: (len(A), sha_of(A))}
        self.civ.answers[sha_of(A)] = (503, "busy", {})
        await main.lora_meta_pass()
        err = main._lm_errors[sha_of(A)]
        self.assertEqual((err["error"], err["backoff_s"]), ("Civitai: HTTP 503", 300.0))
        self.assertEqual(store.lora_meta_all(), {})
        await main.lora_meta_pass()
        self.assertEqual(len(self.civ.calls), 1)                # not before next_try
        it = loratags.lookup("a.safetensors", main._lm_snapshot())
        self.assertEqual((it["status"], it["error"]), ("pending", "Civitai: HTTP 503"))
        main._lm_errors[sha_of(A)]["next_try"] = 0              # time passed
        await main.lora_meta_pass()
        self.assertEqual(main._lm_errors[sha_of(A)]["backoff_s"], 600.0)   # doubled

    async def test_identical_copies_share_one_request(self):
        # Review Focus 3
        self.lan.files = {PA: A, "models/loras/copy.safetensors": A}
        self.civ.answers[sha_of(A)] = (200, version(["w"]), {})
        await self.drain()
        self.assertEqual(self.civ.calls, [URL + sha_of(A)])
        snap = main._lm_snapshot()
        for n in ("a.safetensors", "copy.safetensors"):
            self.assertEqual(loratags.lookup(n, snap)["trigger_words"], ["w"], n)

    async def test_hash_failure_retried_later(self):
        self.lan.files = {PA: A}
        self.lan.fail = {PA}
        await main.lora_meta_pass()
        self.assertGreaterEqual(main._lm_errors[PA]["next_try"], time.time() + 590)
        await main.lora_meta_pass()
        self.assertEqual(len(self.lan.hashed), 1)

    async def test_rename_keeps_civitai_and_curated(self):
        self.lan.files = {PA: A}
        self.civ.answers[sha_of(A)] = (200, version(["w"]), {})
        await self.drain()
        self.assertEqual(await main.lora_curate(sha_of(A), ["mine"]), "")
        self.lan.files = {"models/loras/sub/renamed.safetensors": A}
        await self.drain()
        self.assertEqual(len(self.civ.calls), 1)                # same sha: not asked again
        it = loratags.lookup("sub/renamed.safetensors", main._lm_snapshot())
        self.assertEqual((it["status"], it["trigger_words"]), ("curated", ["mine"]))

    async def test_size_change_is_new_content(self):
        self.lan.files = {PA: A}
        await self.drain()
        self.lan.files = {PA: A + b"-v2"}
        await self.drain()
        self.assertEqual([p for p, _ in self.lan.hashed], [PA, PA])
        self.assertEqual(self.civ.calls, [URL + sha_of(A), URL + sha_of(A + b"-v2")])

    async def test_refresh_all_keeps_curated_and_reasks_not_found(self):
        self.lan.files = {PA: A}
        await self.drain()                                      # 404
        self.assertEqual(await main.lora_curate(sha_of(A), ["keep"]), "")
        self.civ.answers[sha_of(A)] = (200, version(["now found"]), {})
        self.assertIn("1", await main.lora_refresh_all())
        await self.drain()
        rec = main.lora_meta[sha_of(A)]
        self.assertEqual(rec["curated"], ["keep"])
        self.assertEqual(rec["civitai"]["trained_words"], ["now found"])
        self.assertEqual(store.lora_meta_all()[sha_of(A)]["curated"], ["keep"])

    async def test_refresh_one_rehashes_and_reasks(self):
        self.lan.files = {PA: A}
        await self.drain()
        msg = await main.lora_refresh("a.safetensors")
        self.assertIn("a.safetensors", msg)
        await self.drain()
        self.assertEqual([p for p, _ in self.lan.hashed], [PA, PA])
        self.assertEqual(len(self.civ.calls), 2)
        self.assertIn("not refreshed", await main.lora_refresh("nope.safetensors"))

    async def test_curate_states_and_stale_sha(self):
        self.lan.files = {PA: A}
        await self.drain()
        self.assertEqual(await main.lora_curate(sha_of(A), ["", "  a ", "a"]), "")
        self.assertEqual(main.lora_meta[sha_of(A)]["curated"], ["a"])
        self.assertEqual(await main.lora_curate(sha_of(A), []), "")
        self.assertEqual(main.lora_meta[sha_of(A)]["curated"], [])
        self.assertEqual(await main.lora_curate(sha_of(A), None), "")
        self.assertIsNone(main.lora_meta[sha_of(A)]["curated"])
        self.assertIn("changed", await main.lora_curate("9" * 64, ["x"]))
        self.assertNotIn("9" * 64, main.lora_meta)

    async def test_boot_loads_store_and_starts_pending(self):
        store.lora_meta_put(sha_of(A), {"curated": ["x"]})
        main.lora_meta = {}
        main.lora_meta_boot()
        self.assertEqual(main.lora_meta[sha_of(A)]["curated"], ["x"])
        self.assertEqual(main._lm_state["problem"], "not listed yet")
        self.assertEqual(loratags.lookup("a.safetensors", main._lm_snapshot())["status"],
                         "pending")

    async def test_view_rows_counts_busy(self):
        main.backends = [{"name": "k", "type": "comfyui", "url": "http://k"}]
        main.backend_loras = {main.backend_id(main.backends[0]): {"a.safetensors",
                                                                  "local.safetensors"}}
        self.lan.files = {PA: A, PB: B}
        await main.lora_meta_pass()
        v = main.lora_meta_view()
        self.assertTrue(v["busy"])
        self.assertEqual([r["name"] for r in v["rows"]],
                         ["a.safetensors", "b.safetensors", "local.safetensors"])
        rows = {r["name"]: r for r in v["rows"]}
        self.assertEqual(rows["a.safetensors"]["backends"], ["k"])
        self.assertEqual(rows["b.safetensors"]["backends"], [])          # share only
        self.assertEqual(rows["local.safetensors"]["status"], "not_on_share")
        await self.drain()
        v = main.lora_meta_view()
        self.assertFalse(v["busy"])
        self.assertEqual(v["counts"], {"files": 2, "hashed": 2, "found": 0,
                                       "not_found": 2, "civitai_pending": 0})

    async def test_loop_survives_a_failing_pass(self):
        calls = []

        async def bad():
            calls.append(1)
            raise ValueError("boom")
        saved = main.lora_meta_pass
        main.lora_meta_pass = bad
        self.addCleanup(setattr, main, "lora_meta_pass", saved)
        task = asyncio.ensure_future(main.lora_meta_loop())
        await asyncio.sleep(0.05)
        main._lm_wake_up()
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(calls), 2)
        self.assertIn("pass failed", main._lm_state["last_error"])


WF_PAIRED = {"1": {"class_type": "LoraStack", "_meta": {"title": "input_lora_high"},
                   "inputs": {"lora_01": "None", "strength_01": 1.0}},
             "2": {"class_type": "LoraStack", "_meta": {"title": "input_lora_low"},
                   "inputs": {"lora_01": "None", "strength_01": 1.0}}}
WF_PLAIN = {"1": {"class_type": "LoraStack", "_meta": {"title": "loras"},
                  "inputs": {"lora_01": "None", "strength_01": 1.0}}}
BK = {"name": "k", "type": "comfyui", "url": "http://k:8188"}


class Api(_Fixture):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        main.backends = [dict(BK)]
        self.cand = {"backend": "k", "workflow_json": WF_PAIRED, "mapping": {}}
        main.image_models = {"wan": [self.cand]}
        main.get_gen_routes = lambda alias: [(BK, self.cand)] if alias == "wan" else []
        self.names = {"w-HIGH.safetensors", "w-LOW.safetensors",
                      "dir/my lora #1+%.safetensors", "local.safetensors"}
        main.backend_loras = {main.backend_id(BK): set(self.names)}
        self.lan.files = {"models/loras/w-HIGH.safetensors": b"hi",
                          "models/loras/w-LOW.safetensors": b"lo",
                          "models/loras/dir/my lora #1+%.safetensors": b"odd"}
        self.civ.answers[sha_of(b"hi")] = (200, version(["hiword"]), {})
        self.civ.answers[sha_of(b"lo")] = (200, version(["loword"]), {})
        await self.drain()
        self.c = TestClient(main.app)

    def test_list_keeps_loras_and_adds_items(self):
        r = self.c.get("/v1/generations/wan/loras")
        self.assertEqual(r.status_code, 200, r.text)
        j = r.json()
        self.assertEqual(set(j), {"object", "alias", "loras", "items"})
        self.assertEqual(j["loras"], sorted(self.names))        # unchanged for old clients
        self.assertEqual([i["name"] for i in j["items"]], j["loras"])
        by = {i["name"]: i for i in j["items"]}
        self.assertEqual(by["local.safetensors"]["status"], "not_on_share")
        self.assertEqual(by["w-HIGH.safetensors"]["status"], "civitai")
        self.assertEqual(by["w-HIGH.safetensors"]["pair"],
                         {"name": "w-LOW.safetensors", "status": "civitai"})
        self.assertEqual(by["w-HIGH.safetensors"]["trigger_words"], ["hiword", "loword"])
        self.assertEqual(by["w-HIGH.safetensors"]["civitai"]["url"],
                         "https://civitai.com/models/11?modelVersionId=22")

    def test_no_pair_without_high_and_low_stacks(self):
        self.cand["workflow_json"] = WF_PLAIN
        by = {i["name"]: i for i in self.c.get("/v1/generations/wan/loras").json()["items"]}
        self.assertIsNone(by["w-HIGH.safetensors"]["pair"])
        self.assertEqual(by["w-HIGH.safetensors"]["trigger_words"], ["hiword"])

    def test_single_route(self):
        r = self.c.get("/v1/generations/wan/loras/w-LOW.safetensors")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["trigger_words"], ["loword", "hiword"])
        self.assertEqual(self.c.get("/v1/generations/wan/loras/other.safetensors").status_code,
                         404)
        self.assertEqual(self.c.get("/v1/generations/nope/loras/x.safetensors").status_code,
                         404)

    def test_single_route_special_characters(self):
        # Review Focus 1: space, #, %, + and a subfolder in a LoRA name
        from urllib.parse import quote
        r = self.c.get("/v1/generations/wan/loras/" + quote("dir/my lora #1+%.safetensors"))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["name"], "dir/my lora #1+%.safetensors")
        self.assertEqual(r.json()["sha256"], sha_of(b"odd"))      # resolved on the share

    def test_routes_never_touch_the_share_or_civitai(self):
        def boom():
            raise AssertionError("request path touched the LanSource")
        main.modelsrc = boom
        main._civitai_client = object()                       # any use would raise
        self.assertEqual(self.c.get("/v1/generations/wan/loras").status_code, 200)
        self.assertEqual(self.c.get("/v1/generations/wan/loras/w-HIGH.safetensors")
                         .status_code, 200)

    def test_gate_applies(self):
        main.api_key = "master"
        self.assertEqual(self.c.get("/v1/generations/wan/loras").status_code, 401)
        self.assertEqual(self.c.get("/v1/generations/wan/loras/w-HIGH.safetensors")
                         .status_code, 401)
        ok = self.c.get("/v1/generations/wan/loras",
                        headers={"authorization": "Bearer master"})
        self.assertEqual(ok.status_code, 200)

    def test_schema_points_to_the_items(self):
        j = self.c.get("/v1/generations/wan/schema").json()
        self.assertEqual(j["loras"]["item_url"], "/v1/generations/wan/loras/{name}")
        self.assertIn("never edits the prompt", j["loras"]["trigger_words"])


class Console(_Fixture):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        main.backends = [dict(BK)]
        main.backend_loras = {main.backend_id(BK): {"a.safetensors"}}
        self.lan.files = {PA: A}
        evil = version(["<script>alert(1)</script>", "fine word"])
        evil["model"]["name"] = "<img src=x onerror=alert(2)>"
        self.civ.answers[sha_of(A)] = (200, evil, {})
        await self.drain()
        self.c = TestClient(main.app)

    def page(self, **kw):
        r = self.c.get("/ui/routing?sub=loras", headers=SAME, **kw)
        self.assertEqual(r.status_code, 200, r.text[-400:])
        return r.text

    def post(self, url, data=None, status=303):
        r = self.c.post(url, data=data or {}, headers=SAME, follow_redirects=False)
        self.assertEqual(r.status_code, status, r.text[-600:])
        return r

    def test_row_escapes_civitai_text_and_links_from_ids(self):
        main.lora_meta[sha_of(A)]["civitai"]["url"] = "javascript:alert(3)"   # never used
        h = self.page()
        self.assertIn('data-k="lora-a.safetensors"', h)
        self.assertNotIn("<script>alert(1)", h)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", h)
        self.assertNotIn("<img src=x", h)
        self.assertIn('href="https://civitai.com/models/11?modelVersionId=22"', h)
        self.assertNotIn("javascript:alert(3)", h)
        self.assertIn("Flux.1 D", h)

    def test_curate_save_clear_and_stale(self):
        r = self.post("/ui/loras/curate", {"sha": sha_of(A), "name": "a.safetensors",
                                           "words": "one\ntwo, three\n"})
        self.assertIn("sub=loras", r.headers["location"])
        self.assertEqual(main.lora_meta[sha_of(A)]["curated"], ["one", "two, three"])
        self.post("/ui/loras/curate", {"sha": sha_of(A), "name": "a.safetensors",
                                       "words": "x", "act": "clear"})
        self.assertIsNone(main.lora_meta[sha_of(A)]["curated"])
        r = self.post("/ui/loras/curate", {"sha": "9" * 64, "name": "a.safetensors",
                                           "words": "x"}, status=400)
        self.assertIn("changed", r.text)
        self.assertNotIn("9" * 64, main.lora_meta)

    def test_curate_crlf_and_blank(self):
        # Review Focus 4: CRLF, trailing blanks, whitespace only → [] (deliberately none)
        self.post("/ui/loras/curate", {"sha": sha_of(A), "name": "a.safetensors",
                                       "words": "a\r\n b \r\n\r\n"})
        self.assertEqual(main.lora_meta[sha_of(A)]["curated"], ["a", "b"])
        self.post("/ui/loras/curate", {"sha": sha_of(A), "name": "a.safetensors",
                                       "words": "  \r\n \r\n"})
        self.assertEqual(main.lora_meta[sha_of(A)]["curated"], [])
        self.assertIn("deliberately none", self.page())

    def test_refresh_actions(self):
        r = self.post("/ui/loras/refresh-all")
        self.assertIn("sub=loras", r.headers["location"])
        self.assertIn(sha_of(A), main._lm_refetch)
        self.post("/ui/loras/refresh?name=a.safetensors")
        self.assertNotIn(PA, main._lm_state["shas"])

    def test_actions_are_post_only(self):
        for u in ("/ui/loras/curate", "/ui/loras/refresh", "/ui/loras/refresh-all"):
            self.assertTrue(admin._is_post_action(u), u)
            self.assertEqual(self.c.get(u, headers=SAME).status_code, 405, u)

    def test_live_only_while_busy(self):
        # `data-live` also appears inside _LIVE_JS — check the <main> attribute itself
        self.assertNotIn("<main data-live", self.page())
        self.lan.files[PB] = B                                    # a new LoRA appears
        main._lm_state["share"] = loratags.share_loras(self.lan.cached())
        self.assertIn('<main data-live="5"', self.page())

    def test_no_share_says_where_to_set_it_up(self):
        main._lm_state = main._lm_initial_state()
        h = self.page()
        self.assertIn("/ui/server?sub=models", h)


if __name__ == "__main__":
    unittest.main()
