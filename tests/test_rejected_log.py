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

    def test_source_header_is_cut_everywhere(self):
        self.assertLessEqual(len(main._source_of(_req(source="z" * 9999))), main._LOG_FIELD_MAX)


if __name__ == "__main__":
    unittest.main()
