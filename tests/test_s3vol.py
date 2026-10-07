"""s3vol — RunPod network-volume S3 client.

Why this fails SILENTLY: a SigV4 slip is a 403 the sync shows as "S3 key rejected" — the
operator then re-enters a correct key forever; a ListV2 reader that ignores
`IsTruncated` sees only the first 1000 files and re-uploads (or calls unknown) the rest.

Run: venv/bin/python -m unittest tests.test_s3vol -v
"""
import calendar
import hashlib
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import s3vol  # noqa: E402

T2015 = calendar.timegm((2015, 8, 30, 12, 36, 0))
T2013 = calendar.timegm((2013, 5, 24, 0, 0, 0))
K1 = ("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
K2 = ("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")


def sig(h):
    return h["authorization"].rsplit("Signature=", 1)[1]


class SigV4(unittest.TestCase):
    """Expected values: AWS SigV4 test suite / S3 docs, cross-checked with botocore 2026-10-07."""

    def test_get_vanilla(self):
        """A wrong scope rejects even an empty request with valid credentials."""
        h = s3vol.sign("GET", "example.amazonaws.com", "/", {}, {}, s3vol.EMPTY_SHA256,
                       *K1, "us-east-1", T2015, service="service")
        self.assertEqual(sig(h), "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31")
        self.assertIn("SignedHeaders=host;x-amz-date,", h["authorization"])
        self.assertEqual(h["x-amz-date"], "20150830T123600Z")

    def test_query_order(self):
        """Insertion order must not change the signature sent to S3."""
        h = s3vol.sign("GET", "example.amazonaws.com", "/", {"Param2": "value2", "Param1": "value1"},
                       {}, s3vol.EMPTY_SHA256, *K1, "us-east-1", T2015, service="service")
        self.assertEqual(sig(h), "b97d918cfa904a5beff61c982a1b6f458b799221646efd99d3219ec94cdf2500")

    def test_s3_get_object(self):
        """Dropping supplied headers invalidates ranged S3 reads."""
        h = s3vol.sign("GET", "examplebucket.s3.amazonaws.com", "/test.txt", {},
                       {"Range": "bytes=0-9", "x-amz-content-sha256": s3vol.EMPTY_SHA256},
                       s3vol.EMPTY_SHA256, *K2, "us-east-1", T2013)
        self.assertEqual(sig(h), "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")

    def test_runpod_key_with_space_and_umlaut(self):
        """Encoding model names twice silently signs a different object."""
        body = hashlib.sha256(b"hello").hexdigest()
        h = s3vol.sign("PUT", "s3api-eu-ro-1.runpod.io", "/vol123/models/loras/a bä.safetensors",
                       {}, {"x-amz-content-sha256": body}, body, *K2, "eu-ro-1", T2013)
        self.assertEqual(sig(h), "1cf7be2fd79ce71a437208f41cfce6e098f007fabc74f44da5055c35a5c62529")
        self.assertEqual(s3vol.encode_path("/vol123/models/loras/a bä.safetensors"),
                         "/vol123/models/loras/a%20b%C3%A4.safetensors")

    def test_post_form(self):
        """Form payloads must be hashed and content-type signed or POSTs get 403s."""
        # AWS suite fixture, also shipped in botocore's tests/unit/auth/aws4_testsuite.
        h = s3vol.sign("POST", "example.amazonaws.com", "/", {},
                       {"Content-Type": "application/x-www-form-urlencoded"},
                       hashlib.sha256(b"Param1=value1").hexdigest(),
                       *K1, "us-east-1", T2015, service="service")
        self.assertEqual(sig(h), "ff11897932ad3f4e8b18135d722051e5ac45fc38421b1da7b9d196a0fe09473a")


LIST_P1 = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<Name>vol</Name><Prefix>models/</Prefix><KeyCount>2</KeyCount><IsTruncated>true</IsTruncated>
<NextContinuationToken>tok 1/2</NextContinuationToken>
<Contents><Key>models/a.safetensors</Key><Size>10</Size></Contents>
<Contents><Key>models/b c.safetensors</Key><Size>20</Size></Contents>
</ListBucketResult>"""


class Parsers(unittest.TestCase):
    def test_parse_list(self):
        """Namespaces must not hide keys or the next-page token."""
        keys, tok, trunc = s3vol.parse_list(LIST_P1)
        self.assertEqual(keys, {"models/a.safetensors": 10, "models/b c.safetensors": 20})
        self.assertEqual((tok, trunc), ("tok 1/2", True))

    def test_parse_error(self):
        """An error body must remain distinguishable from listing data."""
        x = b"<Error><Code>SignatureDoesNotMatch</Code><Message>nope</Message></Error>"
        self.assertEqual(s3vol.parse_error(x), ("SignatureDoesNotMatch", "nope"))
        self.assertEqual(s3vol.parse_error(LIST_P1), ("", ""))
        self.assertEqual(s3vol.parse_error(b"not xml"), ("", ""))

    def test_parse_mpu_list_and_upload_id(self):
        """Losing upload ids leaves multipart data orphaned on restart."""
        x = (b'<ListMultipartUploadsResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
             b"<Upload><Key>models/x</Key><UploadId>u1</UploadId></Upload>"
             b"<Upload><Key>models/y</Key><UploadId>u2</UploadId></Upload>"
             b"</ListMultipartUploadsResult>")
        self.assertEqual(s3vol.parse_mpu_list(x), [("models/x", "u1"), ("models/y", "u2")])
        self.assertEqual(s3vol.parse_upload_id(
            b"<InitiateMultipartUploadResult><UploadId>abc</UploadId></InitiateMultipartUploadResult>"),
            "abc")


from unittest.mock import patch
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET

import asyncio
import httpx
import urllib.parse


class FakeS3:
    """Path-style fake: /<bucket>/<key>. Pages ListV2 at `page` keys."""

    def __init__(self, page=1000):
        self.objs, self.mpu, self.page, self.calls = {}, {}, page, []

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.calls.append((req.method, req.url.raw_path.decode(), req.headers.get("authorization", "")))
        assert req.headers["x-amz-content-sha256"] == hashlib.sha256(req.content).hexdigest()
        path = urllib.parse.unquote(req.url.raw_path.decode().split("?", 1)[0])
        _, bucket, *rest = path.split("/", 2)
        key = rest[0] if rest else ""
        q = dict(req.url.params)
        if req.method == "GET" and "list-type" in q:
            keys = sorted(k for k in self.objs if k.startswith(q.get("prefix", "")))
            start = int(q.get("continuation-token", "0") or 0)
            chunk = keys[start:start + self.page]
            more = start + self.page < len(keys)
            body = "<ListBucketResult>" + "".join(
                f"<Contents><Key>{escape(k)}</Key><Size>{len(self.objs[k])}</Size></Contents>" for k in chunk)
            body += f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>"
            if more:
                body += f"<NextContinuationToken>{start + self.page}</NextContinuationToken>"
            return httpx.Response(200, content=(body + "</ListBucketResult>").encode())
        if req.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(self.objs[key]))}) \
                if key in self.objs else httpx.Response(404)
        if req.method == "GET" and "uploads" in q:
            body = "<ListMultipartUploadsResult>" + "".join(
                f"<Upload><Key>{escape(k)}</Key><UploadId>{uid}</UploadId></Upload>"
                for uid, (k, parts) in self.mpu.items() if k.startswith(q.get("prefix", "")))
            return httpx.Response(200, content=(body + "</ListMultipartUploadsResult>").encode())
        if req.method == "GET":
            return httpx.Response(200, content=self.objs[key]) if key in self.objs else httpx.Response(404)
        if req.method == "PUT" and "partNumber" in q:
            self.mpu[q["uploadId"]][1][int(q["partNumber"])] = req.content
            return httpx.Response(200, headers={"ETag": f'"e{q["partNumber"]}"'})
        if req.method == "PUT":
            self.objs[key] = req.content
            return httpx.Response(200)
        if req.method == "POST" and "uploads" in q:
            uid = f"u{len(self.mpu) + 1}"
            self.mpu[uid] = (key, {})
            return httpx.Response(200, content=f"<R><UploadId>{uid}</UploadId></R>".encode())
        if req.method == "POST" and "uploadId" in q:
            k, parts = self.mpu.pop(q["uploadId"])
            self.objs[k] = b"".join(parts[n] for n in sorted(parts))
            return httpx.Response(200, content=b"<CompleteMultipartUploadResult/>")
        if req.method == "DELETE" and "uploadId" in q:
            self.mpu.pop(q["uploadId"], None)
            return httpx.Response(204)
        if req.method == "DELETE":
            self.objs.pop(key, None)
            return httpx.Response(204)
        return httpx.Response(400)


def vol(fake):
    c = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return s3vol.S3Volume(c, "https://s3api-eu-ro-1.runpod.io", "vol1", "user_x", "rps_y", "eu-ro-1")


class HTTPClient(unittest.TestCase):
    def test_list_follows_continuation(self):
        """files past the first page silently vanish otherwise"""
        async def run():
            fake = FakeS3(page=1000)
            fake.objs = {f"models/{n:04}": b"abc" for n in range(2500)}
            v = vol(fake)
            async with v.client:
                self.assertEqual(await v.list_objects("models/"),
                                 {k: 3 for k in fake.objs})
            self.assertEqual(len(fake.calls), 3)
            self.assertIn("continuation-token=1000", fake.calls[1][1])
            self.assertIn("continuation-token=2000", fake.calls[2][1])
        asyncio.run(run())

    def test_put_get_head_delete_round_trip(self):
        """A Unicode or literal percent key must reach the same object on every verb."""
        async def run():
            fake = FakeS3()
            v = vol(fake)
            async with v.client:
                for key in ("models/a bä.safetensors", "models/a%20&x"):
                    self.assertIsNone(await v.get(key))
                    self.assertIsNone(await v.head(key))
                    await v.put(key, b"hello")
                    self.assertEqual(await v.get(key), b"hello")
                    self.assertEqual(await v.head(key), 5)
                    await v.delete(key)
                    self.assertIsNone(await v.get(key))
                    await v.delete(key)
            self.assertIn("/vol1/models/a%20b%C3%A4.safetensors", fake.calls[2][1])
        asyncio.run(run())

    def test_multipart_round_trip(self):
        """Losing a part or upload id silently corrupts the assembled model."""
        async def run():
            fake = FakeS3()
            v = vol(fake)
            async with v.client:
                uid = await v.mpu_create("models/x")
                self.assertEqual(await v.mpu_list("models/"), [("models/x", uid)])
                etags = [await v.mpu_part("models/x", uid, n, data)
                         for n, data in enumerate((b"a", b"bc", b"def"), 1)]
                self.assertEqual(etags, ['"e1"', '"e2"', '"e3"'])
                await v.mpu_complete("models/x", uid, etags)
                self.assertEqual(await v.get("models/x"), b"abcdef")
                self.assertEqual(await v.mpu_list("models/"), [])
                uid = await v.mpu_create("models/y")
                await v.mpu_abort("models/y", uid)
                self.assertEqual(await v.mpu_list("models/"), [])
        asyncio.run(run())

    def test_complete_200_with_error_body_raises(self):
        """S3 may send an error under HTTP 200; accepting it loses the model."""
        async def run():
            def handler(req):
                return httpx.Response(200, content=b"<Error><Code>InternalError</Code><Message>nope</Message></Error>")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                v = s3vol.S3Volume(c, "https://s3api-eu-ro-1.runpod.io", "vol1", "user_x", "rps_y", "eu-ro-1")
                with self.assertRaises(s3vol.S3Error) as caught:
                    await v.mpu_complete("models/x", "u1", ['"e1"'])
                self.assertEqual((caught.exception.status, caught.exception.code), (200, "InternalError"))
        asyncio.run(run())

    def test_auth_error_class(self):
        """Rejected credentials must pause sync rather than enter transient backoff."""
        async def run():
            for status, code in ((401, ""), (403, "SignatureDoesNotMatch"),
                                 (400, "InvalidAccessKeyId"), (400, "AccessDenied")):
                def handler(req):
                    return httpx.Response(status, content=f"<Error><Code>{code}</Code></Error>".encode())
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                    v = s3vol.S3Volume(c, "https://s3api-eu-ro-1.runpod.io", "vol1", "user_x", "rps_y", "eu-ro-1")
                    with self.assertRaises(s3vol.S3AuthError) as caught:
                        await v.get("x")
                    self.assertEqual(caught.exception.status, status)
        asyncio.run(run())

    def test_secret_not_in_urls(self):
        """URLs are logged by transports, so credentials must stay in signed headers."""
        async def run():
            fake = FakeS3()
            v = vol(fake)
            async with v.client:
                await v.put("models/x", b"hello")
                await v.list_objects("models/")
                uid = await v.mpu_create("models/y")
                await v.mpu_part("models/y", uid, 1, b"x")
                await v.mpu_complete("models/y", uid, ['"e1"'])
            for method, url, auth in fake.calls:
                self.assertNotIn("rps_y", url)
                self.assertIn("Credential=user_x/", auth)
                self.assertNotIn("rps_y", auth)
        asyncio.run(run())

    def test_size_limit_before_send(self):
        """RunPod refuses 500 MB, and rejected buffers must never reach the transport."""
        async def run():
            fake = FakeS3()
            v = vol(fake)
            async with v.client:
                for size in (500 * 1000 * 1000, 500 * 1000 * 1000 + 1):
                    with patch("s3vol.len", return_value=size, create=True):
                        with self.assertRaises(ValueError):
                            await v.put("x", b"x")
                        with self.assertRaises(ValueError):
                            await v.mpu_part("x", "u1", 1, b"x")
                self.assertEqual(fake.calls, [])
                with patch("s3vol.len", return_value=500 * 1000 * 1000 - 1, create=True):
                    await v.put("x", b"x")
                    uid = await v.mpu_create("y")
                    self.assertEqual(await v.mpu_part("y", uid, 1, b"x"), '"e1"')
        asyncio.run(run())

    def test_complete_xml_and_timeout(self):
        """Unescaped ETags or a short timeout can leave successful uploads uncommitted."""
        async def run():
            def handler(req):
                root = ET.fromstring(req.content)
                self.assertEqual(root.tag, "CompleteMultipartUpload")
                self.assertEqual([(p.findtext("PartNumber"), p.findtext("ETag")) for p in root],
                                 [("1", '"a&<b>"'), ("2", '"c"')])
                self.assertEqual(dict(req.url.params), {"uploadId": "u /+"})
                self.assertEqual(req.extensions["timeout"]["read"], 900.0)
                self.assertEqual(req.headers["x-amz-content-sha256"], hashlib.sha256(req.content).hexdigest())
                return httpx.Response(200, content=b"<CompleteMultipartUploadResult/>")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                v = s3vol.S3Volume(c, s3vol.endpoint_for("EU-RO-1"), "vol1", "user_x", "rps_y", "eu-ro-1", now=lambda: T2013)
                await v.mpu_complete("models/x", "u /+", ['"a&<b>"', '"c"'])
        asyncio.run(run())

    def test_missing_deletes_and_transport_failure(self):
        """Missing cleanup is done, but transport failures must reach controller backoff."""
        async def run():
            def handler(req):
                self.assertEqual(req.extensions["timeout"]["read"], 60.0)
                if req.method == "GET":
                    raise httpx.ConnectError("offline", request=req)
                return httpx.Response(404)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                v = s3vol.S3Volume(c, "https://s3api-eu-ro-1.runpod.io", "vol1", "user_x", "rps_y", "eu-ro-1")
                await v.delete("x")
                await v.mpu_abort("x", "u1")
                with self.assertRaises(httpx.ConnectError):
                    await v.get("x")
        asyncio.run(run())

    def test_review_regressions(self):
        """A truncated listing without a token looped forever on page one; a stored
        object whose bytes look like an <Error> document was read as an S3 error; an
        initiate without UploadId went on with an empty id."""
        async def run():
            def handler(req):
                q = dict(req.url.params)
                if req.method == "GET" and "list-type" in q:
                    return httpx.Response(200, content=b"<ListBucketResult><IsTruncated>true"
                                                       b"</IsTruncated></ListBucketResult>")
                if req.method == "GET":
                    return httpx.Response(200, content=b"<Error><Code>X</Code></Error>")
                return httpx.Response(200, content=b"<InitiateMultipartUploadResult/>")
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                v = s3vol.S3Volume(c, "https://s3api-eu-ro-1.runpod.io", "vol1", "user_x", "rps_y", "eu-ro-1")
                with self.assertRaises(s3vol.S3Error):
                    await asyncio.wait_for(v.list_objects("models/"), 5)
                self.assertEqual(await v.get("models/err.xml"), b"<Error><Code>X</Code></Error>")
                with self.assertRaises(s3vol.S3Error):
                    await v.mpu_create("models/x")
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
