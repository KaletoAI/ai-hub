"""A dropped `retry-after` fails silently: the caller still gets its 429, so
nothing errors — it just retries blind. Measured 2026-09-01 on prod: one Claude
Code request became ~10 upstream calls in 20 s because the gateway kept only its
own `x-gateway-*` headers and Anthropic's `retry-after` never reached the client.
These tests pin the header down at the two seams that rebuild a response."""

import unittest

import adapters


class RateLimitHeaderTests(unittest.TestCase):
    def test_retry_after_kept(self):
        self.assertEqual(adapters._ratelimit_headers({"retry-after": "30"}),
                         {"retry-after": "30"})

    def test_header_name_is_case_insensitive(self):
        """httpx lowercases, but a plain dict from another code path may not."""
        self.assertEqual(adapters._ratelimit_headers({"Retry-After": "30"}),
                         {"retry-after": "30"})

    def test_absent_header_yields_nothing(self):
        self.assertEqual(adapters._ratelimit_headers({"content-type": "application/json"}), {})

    def test_none_is_tolerated(self):
        self.assertEqual(adapters._ratelimit_headers(None), {})

    def test_nothing_else_leaks_through(self):
        """Upstream `content-length` on a re-serialized body is a corrupt reply."""
        got = adapters._ratelimit_headers(
            {"retry-after": "5", "content-length": "912", "set-cookie": "a=b"})
        self.assertEqual(got, {"retry-after": "5"})


class AnthropicErrorShapeTests(unittest.TestCase):
    """`_anthropic_error` rebuilds the body for Claude Code — the retry hint that
    came with the upstream error has to survive that rebuild."""

    def test_retry_after_survives_the_rebuild(self):
        resp = adapters._anthropic_error(
            429, "rate_limit_error", "rate limited",
            headers={"retry-after": "42", "x-gateway-backend": "claude"})
        self.assertEqual(resp.headers.get("retry-after"), "42")
        self.assertEqual(resp.headers.get("x-gateway-backend"), "claude")

    def test_unrelated_upstream_headers_still_dropped(self):
        resp = adapters._anthropic_error(
            429, "rate_limit_error", "rate limited",
            headers={"retry-after": "42", "content-length": "999"})
        self.assertEqual(resp.headers.get("content-length"), str(len(resp.body)))


if __name__ == "__main__":
    unittest.main()
