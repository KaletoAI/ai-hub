"""What a REFUSED call may write into the call log.

Why this fails SILENTLY: refusals are logged before — or without — authentication (a
401 is itself a refusal), and the row carries what the caller chose: the `model` field,
the `x-source` header, the path. Nothing bounded any of them, so an anonymous client
could write a 50 MB "model name" into stats.calls per request, or thousands of 401 rows
a minute that push every real call out of the LLM Calls view (review 2026-09-23, S11).
The database just grows and the console gets slower; nothing errors.

So the caller-chosen strings are cut to a fixed length before they are stored, and 401
rows — the only refusal an unauthenticated stranger can produce at will — are recorded
at most `_UNAUTH_LOG_PER_MIN` per minute; the rest are counted and summarised in one
log line.

And the refused body itself, which is never stored, was still serialised WHOLE on the
event loop just to cut the preview's 50 + 50 characters from it — a stall per refused
retry of a multi-MB context that shows as nothing but a sluggish gateway. The preview
must come out identical from the bounded stand-in (`_ends_only`).
"""
import asyncio
import os
import sys
import tempfile
import types
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
    import stats
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

from fastapi import HTTPException  # noqa: E402


def _req(alias=None, source=None, path="/v1/chat/completions"):
    return types.SimpleNamespace(
        state=types.SimpleNamespace(gw_alias=alias),
        headers={"x-source": source} if source else {},
        client=types.SimpleNamespace(host="10.9.8.7"),
        url=types.SimpleNamespace(path=path))


class RejectedLog(unittest.TestCase):
    def setUp(self):
        self._saved = (stats.is_active, stats.record_call)
        self.rows = []

        async def rec(**kw):
            self.rows.append(kw)
        stats.is_active = lambda: True
        stats.record_call = rec
        main._unauth_log.clear()

    def tearDown(self):
        stats.is_active, stats.record_call = self._saved
        main._unauth_log.clear()

    def _run(self, reqs, status=403):
        async def go():
            for r in reqs:
                main._record_rejected(r, HTTPException(status, "no"))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        asyncio.run(go())

    def test_caller_strings_are_cut(self):
        self._run([_req(alias="m" * 100000, source="s" * 5000, path="/v1/" + "p" * 5000)])
        row = self.rows[0]
        self.assertLessEqual(len(row["alias"]), main._LOG_FIELD_MAX)
        self.assertLessEqual(len(row["source"]), main._LOG_FIELD_MAX)
        self.assertLessEqual(len(row["endpoint"]), main._LOG_FIELD_MAX)

    def test_a_non_string_model_is_stored_as_text(self):
        self._run([_req(alias={"x": 1})])
        self.assertIsInstance(self.rows[0]["alias"], str)

    def test_401_rows_are_rate_limited(self):
        self._run([_req(alias="a") for _ in range(main._UNAUTH_LOG_PER_MIN + 40)], status=401)
        self.assertEqual(len(self.rows), main._UNAUTH_LOG_PER_MIN)

    def test_other_refusals_are_not_rate_limited(self):
        self._run([_req(alias="a") for _ in range(main._UNAUTH_LOG_PER_MIN + 40)], status=503)
        self.assertEqual(len(self.rows), main._UNAUTH_LOG_PER_MIN + 40)

    def test_the_preview_does_not_serialise_the_whole_body(self):
        # R12: the request itself is not stored for a refusal — it only feeds the
        # two-ends preview — yet the whole body was json.dumps'ed on the event loop,
        # ~25-30 ms per MB, per refused retry. The text handed over stays small and
        # yields the SAME preview as the full serialisation.
        import json
        big = {"model": "claude-x", "stream": True, "tools": [{"name": f"t{i}"} for i in range(5000)],
               "messages": [{"role": "user", "content": "héllo " + "x" * 3_000_000},
                            {"role": "assistant", "content": [{"type": "text", "text": "y" * 900}]},
                            {"role": "user", "content": "  the  last\nquestion  " + "z" * 2_000_000 + " end"}]}
        r = _req(alias="claude-x")
        r.state.gw_body = big
        self._run([r])
        text = self.rows[0]["request_text"]
        self.assertLess(len(text), 100_000)
        self.assertEqual(stats._preview(text), stats._preview(json.dumps(big, ensure_ascii=False)))
        self.assertFalse(self.rows[0]["store_request"])

    def test_a_small_body_is_passed_as_is(self):
        import json
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        r = _req(alias="m")
        r.state.gw_body = body
        self._run([r])
        self.assertEqual(self.rows[0]["request_text"], json.dumps(body, ensure_ascii=False))

    def test_source_header_is_cut_everywhere(self):
        self.assertLessEqual(len(main._source_of(_req(source="z" * 9999))), main._LOG_FIELD_MAX)


if __name__ == "__main__":
    unittest.main()
