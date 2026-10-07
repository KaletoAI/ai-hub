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
