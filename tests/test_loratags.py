"""loratags — the pure rules behind LoRA trigger words.

Why this fails SILENTLY: every rule here ends in a plausible answer. A cleaner that
splits a tag chain hands the client words Civitai never listed as one trigger; one that
keeps `" ,, "` debris hands it junk tokens. A name mapped to the wrong one of two
same-named share files delivers ANOTHER LoRA's words. A status that says `not_on_share`
while the share is merely not listed yet reads like a verdict. A pair merged on a
coincidental `high`/`low` in a name mixes two unrelated LoRAs' words. None of these
raise; each looks like data.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loratags  # noqa: E402

SHA_A = "a" * 64
SHA_B = "b" * 64


def civ(words, mid=11, vid=22, base="Flux.1 D"):
    return {"status": "found", "model_id": mid, "version_id": vid, "model_name": "M",
            "version_name": "v1", "base_model": base, "trained_words": words,
            "fetched_at": 0.0}


class CleanWords(unittest.TestCase):
    def test_form_is_cleaned_order_kept_never_split(self):
        raw = ["  m4pdr4w ", "a,,  b ,, c", "1girl, solo,  masterpiece", "", "   ",
               "m4pdr4w", ",lead,", "x  y"]
        self.assertEqual(loratags.clean_words(raw),
                         ["m4pdr4w", "a, b, c", "1girl, solo, masterpiece", "lead", "x y"])

    def test_clean_words_odd_entries(self):
        # Review Focus 2: non-strings, null, nested lists, only commas/whitespace
        self.assertEqual(loratags.clean_words([None, 3, ["x"], {"a": 1}, " , ,", "ok"]),
                         ["ok"])
        self.assertEqual(loratags.clean_words(None), [])
        self.assertEqual(loratags.clean_words("not a list"), [])

    def test_a_sentence_stays_one_entry(self):
        s = "Convert the character in the image to a pixel sprite."
        self.assertEqual(loratags.clean_words([s]), [s])


class ParseCivitai(unittest.TestCase):
    def test_full_answer(self):
        body = {"id": 2982108, "modelId": 799901, "name": "Anima v1", "baseModel": "Anima",
                "trainedWords": ["mapcraft", " mapcraft "],
                "model": {"name": "Mapcraft", "type": "LORA"}, "files": []}
        self.assertEqual(loratags.parse_civitai(body, 5.0), {
            "status": "found", "model_id": 799901, "version_id": 2982108,
            "model_name": "Mapcraft", "version_name": "Anima v1", "base_model": "Anima",
            "trained_words": ["mapcraft"], "fetched_at": 5.0})

    def test_caps(self):
        body = {"id": 1, "name": "v" * 900, "baseModel": "b" * 900,
                "model": {"name": "m" * 900},
                "trainedWords": ["w%d" % i for i in range(500)]}
        r = loratags.parse_civitai(body, 1.0)
        self.assertEqual((len(r["model_name"]), len(r["version_name"]),
                          len(r["base_model"])), (300, 300, 300))
        self.assertEqual(len(r["trained_words"]), 200)
        r = loratags.parse_civitai({"id": 1, "trainedWords": ["x" * 900]}, 1.0)
        self.assertEqual(len(r["trained_words"][0]), 300)

    def test_missing_and_odd_fields(self):
        r = loratags.parse_civitai({"id": 7, "modelId": "8", "model": "x",
                                    "baseModel": 3, "trainedWords": None}, 1.0)
        self.assertEqual((r["version_id"], r["model_id"], r["model_name"], r["base_model"],
                          r["trained_words"]), (7, None, None, None, []))

    def test_not_a_version_raises(self):
        for body in ({}, {"error": "Model not found"}, [], "html", {"id": True},
                     {"id": "1", "modelId": "2"}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                loratags.parse_civitai(body, 0.0)

    def test_url_from_ints_only(self):
        self.assertEqual(loratags.civitai_url(civ([], 5, 6)),
                         "https://civitai.com/models/5?modelVersionId=6")
        self.assertEqual(loratags.civitai_url({"model_id": 5}),
                         "https://civitai.com/models/5")
        self.assertIsNone(loratags.civitai_url({"model_id": "javascript:alert(1)"}))
        self.assertIsNone(loratags.civitai_url({"url": "https://evil", "model_id": None}))

    def test_retry_after(self):
        self.assertEqual(loratags.retry_after_s("120"), 120.0)
        self.assertEqual(loratags.retry_after_s(" 99999 "), 3600.0)
        self.assertIsNone(loratags.retry_after_s("Wed, 21 Oct 2026 07:28:00 GMT"))
        self.assertIsNone(loratags.retry_after_s(None))
        self.assertIsNone(loratags.retry_after_s("\u00b2"))

    def test_valid_sha(self):
        self.assertTrue(loratags.valid_sha(SHA_A))
        for bad in ("A" * 64, "a" * 63, "../x", None, "g" * 64):
            self.assertFalse(loratags.valid_sha(bad), bad)


class SharePaths(unittest.TestCase):
    LISTING = {"models/loras/a.safetensors": 10, "models/loras/wan/b-HIGH.safetensors": 20,
               "models/loras/x/dup.safetensors": 1, "models/loras/y/dup.safetensors": 2,
               "models/loras/notes.txt": 3, "models/loras/c.safetensors.part": 4,
               "models/vae/v.safetensors": 5,
               "models/loras/alias.safetensors": {"link": "a.safetensors"},
               "models/loras/dead.safetensors": {"link": "gone.safetensors"}}

    def test_share_loras(self):
        sl = loratags.share_loras(self.LISTING)
        self.assertEqual(sl["models/loras/a.safetensors"], ("models/loras/a.safetensors", 10))
        self.assertEqual(sl["models/loras/alias.safetensors"],
                         ("models/loras/a.safetensors", 10))       # a link → its target
        for out in ("models/loras/notes.txt", "models/loras/c.safetensors.part",
                    "models/vae/v.safetensors", "models/loras/dead.safetensors"):
            self.assertNotIn(out, sl)

    def test_share_path(self):
        sl = loratags.share_loras(self.LISTING)
        self.assertEqual(loratags.share_path("a.safetensors", sl),
                         ("models/loras/a.safetensors", ""))
        self.assertEqual(loratags.share_path("wan/b-HIGH.safetensors", sl),
                         ("models/loras/wan/b-HIGH.safetensors", ""))
        self.assertEqual(loratags.share_path("b-HIGH.safetensors", sl),
                         ("models/loras/wan/b-HIGH.safetensors", ""))   # unique suffix
        self.assertEqual(loratags.share_path("dup.safetensors", sl), (None, "ambiguous"))
        self.assertEqual(loratags.share_path("nope.safetensors", sl), (None, "missing"))
        for bad in ("", None, "/etc/passwd", "../a.safetensors"):
            self.assertEqual(loratags.share_path(bad, sl)[0], None, bad)


class Status(unittest.TestCase):
    def test_table(self):
        P = "models/loras/a.safetensors"
        S = loratags.status_of
        self.assertEqual(S(False, False, None, None, None), "unavailable")
        self.assertEqual(S(True, False, None, None, None), "pending")      # never a verdict
        self.assertEqual(S(True, True, None, None, None), "not_on_share")
        self.assertEqual(S(True, True, P, None, None), "pending")
        self.assertEqual(S(True, True, P, SHA_A, {}), "pending")
        self.assertEqual(S(True, True, P, SHA_A, {"civitai": civ(["w"])}), "civitai")
        self.assertEqual(S(True, True, P, SHA_A, {"civitai": {"status": "not_found"}}),
                         "not_on_civitai")
        self.assertEqual(S(True, True, P, SHA_A, {"curated": [],
                                                  "civitai": civ(["w"])}), "curated")
        self.assertEqual(S(True, True, P, SHA_A, {"curated": None,
                                                  "civitai": civ(["w"])}), "civitai")

    def test_effective_words(self):
        self.assertEqual(loratags.effective_words(None), [])
        self.assertEqual(loratags.effective_words({"civitai": civ(["w"])}), ["w"])
        self.assertEqual(loratags.effective_words({"curated": [], "civitai": civ(["w"])}), [])
        self.assertEqual(loratags.effective_words({"curated": ["c"]}), ["c"])
        self.assertEqual(loratags.effective_words({"civitai": {"status": "not_found"}}), [])


def snap(**kw):
    s = {"configured": True, "problem": "",
         "share": loratags.share_loras({"models/loras/w-HIGH.safetensors": 1,
                                        "models/loras/w-LOW.safetensors": 2,
                                        "models/loras/solo.safetensors": 3}),
         "shas": {"models/loras/w-HIGH.safetensors": SHA_A,
                  "models/loras/w-LOW.safetensors": SHA_B},
         "meta": {SHA_A: {"civitai": civ(["hi", "both"])},
                  SHA_B: {"curated": ["lo", "both"], "civitai": civ(["raw"])}},
         "errors": {}}
    s.update(kw)
    return s


def swap(n):
    return {"w-HIGH.safetensors": "w-LOW.safetensors",
            "w-LOW.safetensors": "w-HIGH.safetensors",
            "solo.safetensors": None}.get(n)


class Items(unittest.TestCase):
    def test_lookup_shapes(self):
        it = loratags.lookup("w-LOW.safetensors", snap())
        self.assertEqual(it["status"], "curated")
        self.assertEqual(it["trigger_words"], ["lo", "both"])
        self.assertEqual(it["curated"], ["lo", "both"])
        self.assertEqual(it["civitai"]["trained_words"], ["raw"])
        self.assertEqual(it["civitai"]["url"], "https://civitai.com/models/11?modelVersionId=22")
        self.assertEqual(it["civitai"]["fetched_at"], "1970-01-01T00:00:00Z")
        self.assertEqual(it["sha256"], SHA_B)
        self.assertIsNone(it["pair"])
        self.assertNotIn("error", it)

    def test_pending_carries_its_error(self):
        s = snap(errors={"models/loras/solo.safetensors": {"error": "hash: refused"}})
        it = loratags.lookup("solo.safetensors", s)
        self.assertEqual((it["status"], it["error"], it["sha256"]),
                         ("pending", "hash: refused", None))

    def test_unavailable_and_not_listed(self):
        self.assertEqual(loratags.lookup("solo.safetensors",
                                         snap(configured=False))["status"], "unavailable")
        self.assertEqual(loratags.lookup("solo.safetensors",
                                         snap(problem="not listed yet"))["status"], "pending")
        self.assertEqual(loratags.lookup("other.safetensors", snap())["status"],
                         "not_on_share")

    def test_pair_merge_only_with_counterpart(self):
        names = ["solo.safetensors", "w-HIGH.safetensors", "w-LOW.safetensors"]
        merged = loratags.items(names, snap(), swap)
        hi = merged[1]
        self.assertEqual(hi["pair"], {"name": "w-LOW.safetensors", "status": "curated"})
        self.assertEqual(hi["trigger_words"], ["hi", "both", "lo"])     # own first, dedup
        self.assertEqual(merged[2]["trigger_words"], ["lo", "both", "hi"])
        self.assertIsNone(merged[0]["pair"])
        plain = loratags.items(names, snap(), None)                       # no paired stacks
        self.assertIsNone(plain[1]["pair"])
        self.assertEqual(plain[1]["trigger_words"], ["hi", "both"])

    def test_counterpart_outside_the_set_is_no_pair(self):
        it = loratags.items(["w-HIGH.safetensors"], snap(), swap)[0]
        self.assertIsNone(it["pair"])


if __name__ == "__main__":
    unittest.main()
