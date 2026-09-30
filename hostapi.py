"""The HTTP half of a managed host's PROVIDER — the company whose API creates, lists and
deletes the machine a managed host runs on (Thunder Compute today, RunPod later).

A provider is two modules' worth of names: a PURE module (`thunder.py`: the interface
constants `KIND`/`NAME`/`SSH_USER`/`STOP_MODE`, the host form as data in
`OPTION_FIELDS` + `options_of`, and every parser/decision over the API's dicts) and an
API class here that only does HTTP. `PROVIDERS` is the ONE place the console (the host
form's provider select and its fields) and the host controller look a provider up by
kind — a second list somewhere would let the form offer a provider the controller
cannot drive, or the other way round.

`ProviderApi` holds what every such REST client needs, each rule written down once
because each one fails silently or costs money when it drifts:
- `httpx.Timeout(30, connect=10)`; ANY 2xx is success (Thunder's create answers 201,
  its snapshot create 202 — a `== 200` check would call a paid create a failure);
  anything else raises the provider's `Error` carrying the status.
- A transport failure is an `Error` without status, NAMED by its type: `str()` of an
  httpx error can be empty, and "Thunder API: " alone sends nobody anywhere.
- Every error text passes `_redact`: an API that echoes the request (Authorization
  included) must not put the token into the panel log or the fault log.
- `_by_id`: which id form an item path wants is documented contradictorily (index vs
  uuid), so every such call tries the UUID first and the index only on a 404 — a
  reused index can name a stranger's machine, a uuid cannot.
- `_cached`: price lists change rarely; one fetch per hour, and `cached()` hands the
  last body to a synchronous view that must not wait on the network.

`ThunderApi` moved here unchanged from the former thunderctl.py; hostctl.py reaches it
only through `PROVIDERS`. "Ruling N" below refers to the Thunder integration's decision
ledger (`docs/superpowers/plans/2026-09-27-thunder-comfyui-ledger.md`, kept outside the
repository). No `main`/`adapters` imports. Covered by test_hostapi.py.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional
from urllib.parse import quote

import httpx

import thunder

_TIMEOUT = httpx.Timeout(30, connect=10)
_PRICE_TTL_S = 3600             # /v2/pricing and /v2/specs change rarely; one fetch per hour
_ERR_MAX = 300                  # chars of an API error body kept in the message


def _q(s) -> str:
    """One path segment. An id is the provider's, but quoting keeps a `/` or `..` in it
    from addressing a DIFFERENT endpoint (`/snapshots/a/../b`)."""
    return quote(str(s), safe="")


class ProviderError(RuntimeError):
    """A provider API call failed; `status` is the HTTP status, None for transport. The
    base's default — a provider module brings its own (`thunder.ThunderError`)."""

    def __init__(self, msg, status: Optional[int] = None):
        super().__init__(msg)
        self.status = status


class ProviderApi:
    """HTTP, errors, id fallback and the price cache — no endpoint of its own. A
    subclass names its base URL (`API`), its per-item path (`ITEM_PATH`, with `{id}`
    and `{suffix}`) and its error class (`Error(msg, status)`), then adds the endpoint
    methods; parsing stays in the provider's pure module."""

    API = ""
    ITEM_PATH = ""                  # e.g. "/instances/{id}/{suffix}" — the subclass's
    Error = ProviderError

    def __init__(self, client: httpx.AsyncClient, token: str, base: Optional[str] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._client = client
        self._token = token or ""
        self._base = (self.API if base is None else base).rstrip("/")
        self._clock = clock
        self._cache: dict[str, tuple[float, Any]] = {}

    async def _req(self, method: str, path: str, body=None) -> httpx.Response:
        """One request → the response, whatever its status. A transport failure is an
        `Error` without status, named by its type: `str()` of an httpx error can be
        EMPTY, and "Thunder API: " alone sends nobody anywhere."""
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        try:
            return await self._client.request(method, self._base + path, json=body,
                                              headers=headers, timeout=_TIMEOUT)
        except httpx.HTTPError as e:
            raise self.Error(
                f"{method} {path}: {type(e).__name__}: {e}"[:_ERR_MAX], None) from e

    def _redact(self, text: str) -> str:
        """An error body goes into the panel log and the fault log; an API that echoes
        the request (Authorization included) must not put the token there."""
        return text.replace(self._token, "***") if self._token else text

    def _check(self, r: httpx.Response) -> httpx.Response:
        if not 200 <= r.status_code < 300:
            raise self.Error(
                self._redact(r.text or f"HTTP {r.status_code}")[:_ERR_MAX], r.status_code)
        return r

    def _json(self, r: httpx.Response):
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError as e:
            raise self.Error(f"invalid JSON: {self._redact(r.text[:200])}",
                             r.status_code) from e

    async def _call(self, method: str, path: str, body=None):
        return self._json(self._check(await self._req(method, path, body)))

    async def _by_id(self, method: str, item: dict, suffix: str, body=None,
                     gone_ok: bool = False) -> Optional[httpx.Response]:
        """`ITEM_PATH` with the UUID first, the index only on a 404 (ledger Ruling 11;
        Thunder's openapi names the parameter "Instance ID (index)" for modify/ports but
        plain "Instance ID" for delete, so which form each accepts is verified live).
        UUID first because Thunder REUSES small indices: a stale index can name somebody
        else's instance, a uuid never can. `gone_ok`: 404 on every form means the
        instance no longer exists — the goal of a delete, so not an error there. The
        index is still only safe while `item` came from a fresh list matched by uuid
        (Ruling 10) — never pass an index taken from the state alone."""
        ids = ["" if item.get(k) is None else str(item.get(k)) for k in ("uuid", "index")]
        ids = [i for n, i in enumerate(ids) if i and i not in ids[:n]]
        if not ids:
            raise self.Error("instance has neither index nor uuid", None)
        if not self.ITEM_PATH:
            # a provider without an item path would POST to its API root
            raise self.Error(f"{type(self).__name__} has no ITEM_PATH", None)
        r = None
        for ident in ids:
            r = await self._req(method, self.ITEM_PATH.format(id=_q(ident), suffix=suffix),
                                body)
            if r.status_code != 404:
                return self._check(r)
        if gone_ok:
            return None
        return self._check(r)

    # public price lists, cached
    async def _cached(self, key: str, path: str):
        hit = self._cache.get(key)
        now = self._clock()
        if hit is not None and now - hit[0] < _PRICE_TTL_S:
            return hit[1]
        val = await self._call("GET", path)
        self._cache[key] = (now, val)
        return val

    def cached(self, key: str):
        """The last fetched `pricing`/`specs` body (any age) or None — for the sync
        `view()`, which must not wait on the network."""
        hit = self._cache.get(key)
        return hit[1] if hit is not None else None


class ThunderApi(ProviderApi):
    """Thin async client for the Thunder endpoints the controller uses. Parsing lives
    in the pure `thunder` module; this class only does HTTP, errors and the price
    cache."""

    API = thunder.API
    ITEM_PATH = "/instances/{id}/{suffix}"
    Error = thunder.ThunderError

    def __init__(self, client: httpx.AsyncClient, token: str, base: str = thunder.API,
                 clock: Callable[[], float] = time.monotonic):
        super().__init__(client, token, base, clock)

    # instances
    async def list_instances(self) -> list[dict]:
        return thunder.parse_instances(await self._call("GET", "/instances/list"))

    async def create(self, body: dict) -> dict:
        """→ `{"index": str, "uuid": str}`. `identifier` is an int in the API; the
        controller keeps every id as a string."""
        d = await self._call("POST", "/instances/create", body)
        d = d if isinstance(d, dict) else {}
        ident, uuid = d.get("identifier"), d.get("uuid")
        out = {"index": "" if ident is None else str(ident), "uuid": str(uuid or "")}
        if not out["index"] and not out["uuid"]:
            # the instance may exist and bill anyway — the orphan list is where it shows
            # the body goes into the log and the fault log: an echoed token must not
            raise self.Error(
                self._redact(f"create answered without identifier/uuid: {d}")[:_ERR_MAX], None)
        return out

    async def delete(self, item: dict) -> None:
        await self._by_id("POST", item, "delete", gone_ok=True)

    async def modify(self, item: dict, body: dict) -> None:
        await self._by_id("POST", item, "modify", body)

    async def remove_ports(self, item: dict, ports: list[int]) -> None:
        await self._by_id("PATCH", item, "ports", {"remove_ports": [int(p) for p in ports]})

    # snapshots
    async def snapshots(self) -> list[dict]:
        return thunder.parse_snapshots(await self._call("GET", "/snapshots/list"))

    async def create_snapshot(self, item: dict, name: str) -> str:
        """→ the new snapshot's id. `instanceId` is a STRING in the openapi
        (`CreateSnapshotRequest`) — an int would be a 400 on every stop — and carries
        the index, as the instance endpoints' "(index)" parameters do; the uuid only
        when no index is known. To verify live."""
        idx, uuid = item.get("index"), item.get("uuid")
        inst = str(idx) if idx is not None and str(idx) != "" else str(uuid or "")
        if not inst:
            raise self.Error("instance has neither index nor uuid", None)
        d = await self._call("POST", "/snapshots/create", {"instanceId": inst, "name": name})
        sid = str((d or {}).get("id") or "") if isinstance(d, dict) else ""
        if not sid:
            raise self.Error(
                self._redact(f"snapshot create answered without id: {d}")[:_ERR_MAX], None)
        return sid

    async def delete_snapshot(self, sid: str) -> None:
        """404 = already gone → fine (rotation is re-run after a restart)."""
        if not sid:
            raise self.Error("empty snapshot id", None)
        r = await self._req("DELETE", f"/snapshots/{_q(sid)}")
        if r.status_code != 404:
            self._check(r)

    async def pricing(self) -> dict:
        return await self._cached("pricing", "/v2/pricing")

    async def specs(self) -> dict:
        return await self._cached("specs", "/v2/specs")


# kind → (pure module, API class). The key is the module's KIND (pinned by the test).
PROVIDERS: dict[str, tuple[Any, type]] = {thunder.KIND: (thunder, ThunderApi)}


def provider(kind) -> Optional[tuple[Any, type]]:
    """The `(module, api class)` of a provider kind, None for an unknown one — a host
    entry naming a provider this gateway does not know is shown, never driven."""
    return PROVIDERS.get(kind) if isinstance(kind, str) else None
