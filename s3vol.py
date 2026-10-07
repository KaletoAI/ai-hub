"""RunPod path-style S3 signing and XML readers without a boto3 dependency.

The volume API has no bulk delete or presigned URLs; requests use the same
encoded path and query as the signer so model names survive unchanged.
"""
import asyncio
import hashlib
import hmac
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional
from xml.sax.saxutils import escape

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


class S3Error(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"S3 {status} {code}: {message}"[:500])
        self.status, self.code = status, code


class S3AuthError(S3Error):
    pass


_AUTH_CODES = {"SignatureDoesNotMatch", "InvalidAccessKeyId", "AccessDenied"}


def endpoint_for(dc: str) -> str:
    return f"https://s3api-{str(dc).lower()}.runpod.io"


class S3Volume:
    """Use the caller's client so volume sync shares its connection lifecycle."""

    def __init__(self, client, endpoint, bucket, access_key, secret, region, now=time.time):
        self.client, self.endpoint, self.bucket = client, endpoint.rstrip("/"), bucket
        self._ak, self._sk, self.region, self._now = access_key, secret, region, now
        self.host = urllib.parse.urlsplit(self.endpoint).netloc

    async def _req(self, method, key, query=None, body=b"", headers=None, timeout=60.0,
                   ok=(200,), missing_ok=False):
        path = f"/{self.bucket}/{key}" if key else f"/{self.bucket}"
        h = dict(headers or {})
        digest = (await asyncio.to_thread(hashlib.sha256, body)
                  if len(body) >= 1024 * 1024 else hashlib.sha256(body))
        h["x-amz-content-sha256"] = digest.hexdigest()
        h.update(sign(method, self.host, path, query or {}, h, h["x-amz-content-sha256"],
                      self._ak, self._sk, self.region, self._now()))
        q = encode_query(query or {})
        url = self.endpoint + encode_path(path) + (f"?{q}" if q else "")
        r = await self.client.request(method, url, content=body or None, headers=h, timeout=timeout)
        if missing_ok and r.status_code == 404:
            return None
        # Completion (a POST) can report an S3 error inside an HTTP 200 response; a GET's
        # 200 body is the object itself and is never read as an error document.
        code, msg = (parse_error(r.content)
                     if method != "HEAD" and (r.status_code not in ok or method == "POST")
                     else ("", ""))
        if r.status_code not in ok or code:
            cls = S3AuthError if r.status_code in (401, 403) or code in _AUTH_CODES else S3Error
            raise cls(r.status_code, code or str(r.status_code), msg or r.text[:200])
        return r

    async def list_objects(self, prefix) -> dict[str, int]:
        entries = {}
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        while True:
            r = await self._req("GET", "", query)
            page, token, truncated = parse_list(r.content)
            entries.update(page)
            if not truncated:
                return entries
            if not token:
                # truncated without a token: asking again would loop on page one forever
                raise S3Error(r.status_code, "NoContinuationToken",
                              f"listing {prefix!r} truncated without NextContinuationToken")
            query["continuation-token"] = token

    async def head(self, key) -> Optional[int]:
        r = await self._req("HEAD", key, missing_ok=True)
        return int(r.headers["content-length"]) if r is not None else None

    async def get(self, key) -> Optional[bytes]:
        r = await self._req("GET", key, missing_ok=True)
        return r.content if r is not None else None

    async def put(self, key, data: bytes) -> None:
        if len(data) >= 500 * 1000 * 1000:
            raise ValueError("RunPod S3 requires payloads smaller than 500 MB")
        await self._req("PUT", key, body=data)

    async def delete(self, key) -> None:
        await self._req("DELETE", key, ok=(200, 204), missing_ok=True)

    async def mpu_create(self, key) -> str:
        r = await self._req("POST", key, {"uploads": ""})
        uid = parse_upload_id(r.content)
        if not uid:
            raise S3Error(r.status_code, "NoUploadId", "CreateMultipartUpload answered no UploadId")
        return uid

    async def mpu_part(self, key, upload_id, n, data) -> str:
        if len(data) >= 500 * 1000 * 1000:
            raise ValueError("RunPod S3 requires payloads smaller than 500 MB")
        r = await self._req("PUT", key, {"partNumber": n, "uploadId": upload_id}, body=data)
        return r.headers["etag"]

    async def mpu_complete(self, key, upload_id, etags: list[str]) -> None:
        body = "<CompleteMultipartUpload>" + "".join(
            f"<Part><PartNumber>{n}</PartNumber><ETag>{escape(e)}</ETag></Part>"
            for n, e in enumerate(etags, 1)) + "</CompleteMultipartUpload>"
        await self._req("POST", key, {"uploadId": upload_id}, body=body.encode("utf-8"),
                        timeout=900.0)

    async def mpu_abort(self, key, upload_id) -> None:
        await self._req("DELETE", key, {"uploadId": upload_id}, ok=(200, 204), missing_ok=True)

    async def mpu_list(self, prefix) -> list[tuple[str, str]]:
        r = await self._req("GET", "", {"uploads": "", "prefix": prefix})
        return parse_mpu_list(r.content)
