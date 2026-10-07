"""RunPod path-style S3 signing and XML readers without a boto3 dependency.

The volume API has no bulk delete or presigned URLs; requests use the same
encoded path and query as the signer so model names survive unchanged.
"""
import hashlib
import hmac
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_UNRESERVED = "-_.~"


def encode_path(path: str) -> str:
    """RFC 3986 per character, `/` kept — S3 signs the path encoded ONCE (no
    normalisation, no double encoding); the request URL uses the same string."""
    return urllib.parse.quote(path or "/", safe="/" + _UNRESERVED)


def encode_query(query: dict) -> str:
    items = sorted((urllib.parse.quote(str(k), safe=_UNRESERVED),
                    urllib.parse.quote(str(v), safe=_UNRESERVED))
                   for k, v in (query or {}).items())
    return "&".join(f"{k}={v}" for k, v in items)


def _hm(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def sign(method: str, host: str, path: str, query: dict, headers: dict,
         payload_sha256: str, access_key: str, secret: str, region: str, now: float,
         service: str = "s3") -> dict:
    amz = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    day = amz[:8]
    hd = {str(k).lower().strip(): " ".join(str(v).split()) for k, v in (headers or {}).items()}
    hd["host"] = host
    hd["x-amz-date"] = amz
    names = sorted(hd)
    signed = ";".join(names)
    creq = "\n".join([method.upper(), encode_path(path), encode_query(query),
                      "".join(f"{n}:{hd[n]}\n" for n in names), signed, payload_sha256])
    scope = f"{day}/{region}/{service}/aws4_request"
    sts = "\n".join(["AWS4-HMAC-SHA256", amz, scope,
                     hashlib.sha256(creq.encode("utf-8")).hexdigest()])
    k = _hm(("AWS4" + secret).encode("utf-8"), day)
    for part in (region, service, "aws4_request"):
        k = _hm(k, part)
    sig = hmac.new(k, sts.encode("utf-8"), hashlib.sha256).hexdigest()
    return {"x-amz-date": amz,
            "authorization": f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
                             f"SignedHeaders={signed}, Signature={sig}"}


def _root(xml: bytes):
    try:
        return ET.fromstring(xml)
    except ET.ParseError:
        return None


def _tag(el) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _kids(el, name):
    return [c for c in el if _tag(c) == name]


def _text(el, name, default=""):
    for c in el:
        if _tag(c) == name:
            return c.text or ""
    return default


def parse_list(xml: bytes) -> tuple[dict, Optional[str], bool]:
    r = _root(xml)
    if r is None:
        return {}, None, False
    entries = {}
    for c in _kids(r, "Contents"):
        key = _text(c, "Key")
        try:
            size = int(_text(c, "Size"))
        except ValueError:
            continue
        if key:
            entries[key] = size
    return entries, _text(r, "NextContinuationToken") or None, _text(r, "IsTruncated").lower() == "true"


def parse_mpu_list(xml: bytes) -> list[tuple[str, str]]:
    r = _root(xml)
    if r is None:
        return []
    return [(_text(c, "Key"), _text(c, "UploadId")) for c in _kids(r, "Upload")]


def parse_error(xml: bytes) -> tuple[str, str]:
    r = _root(xml)
    if r is None or _tag(r) != "Error":
        return "", ""
    return _text(r, "Code"), _text(r, "Message")


def parse_upload_id(xml: bytes) -> str:
    r = _root(xml)
    return _text(r, "UploadId") if r is not None else ""
