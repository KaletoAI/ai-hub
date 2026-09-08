"""LAN scan for backends — the pure half (no main/adapters/admin imports, no module
state; every I/O dependency is injected so tests run without a network).

What it answers: which hosts on the gateway's own subnet run something the gateway
can register as a backend — llama-swap / llama.cpp / vLLM / Ollama (`openai`) or
ComfyUI (`comfyui`) — and whether each one is registered already. It never adds
anything: the console offers each finding as a pre-filled form.
"""
import asyncio
import inspect
import ipaddress
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

DEFAULT_PORTS: list[int] = [8080, 8000, 11434, 8188, 1234, 5000]
HOST_CAP = 1024                       # addresses per scan — a /16 must not become a site scan

_IP_LINE = re.compile(r"\binet (\d+\.\d+\.\d+\.\d+)/(\d+)\b")


def parse_ip_addr(text: str) -> list[str]:
    """CIDRs to scan from `ip -o -4 addr show` output: one per non-loopback IPv4
    address. A prefix shorter than /24 becomes the address's /24 (the box's own
    neighbourhood, not the whole site); a longer one (a /30 link) is kept."""
    out: list[str] = []
    for m in _IP_LINE.finditer(text or ""):
        addr, prefix = m.group(1), int(m.group(2))
        if addr.startswith("127."):
            continue
        net = ipaddress.ip_network(f"{addr}/{max(prefix, 24)}", strict=False)
        s = str(net)
        if s not in out:
            out.append(s)
    return out


def local_cidrs(run: Optional[Callable[[list[str]], str]] = None) -> list[str]:
    """This host's subnets via `ip -o -4 addr show` (runner injectable). [] when the
    command is missing or fails — the operator then lists CIDRs in the Server tab."""
    argv = ["ip", "-o", "-4", "addr", "show"]
    try:
        text = run(argv) if run else subprocess.run(argv, capture_output=True, text=True,
                                                   timeout=5).stdout
    except Exception:
        return []
    return parse_ip_addr(text)


def parse_ports(text) -> list[int]:
    """`"8080, 8000"` → [8080, 8000]; junk dropped, 1–65535 only, deduped, order kept."""
    out: list[int] = []
    for tok in re.split(r"[,\s]+", str(text or "")):
        if tok.isdigit() and 0 < int(tok) < 65536 and int(tok) not in out:
            out.append(int(tok))
    return out


def expand_targets(cidrs: list[str], cap: int = HOST_CAP) -> tuple[list[str], bool]:
    """Host addresses of every CIDR (network/broadcast dropped for /24 and shorter —
    `hosts()` does that; /31 and /32 yield their addresses), deduped, cut at `cap`.
    Returns (hosts, truncated). A CIDR that does not parse is skipped."""
    out: list[str] = []
    seen: set[str] = set()
    for c in cidrs:
        try:
            net = ipaddress.ip_network(str(c).strip(), strict=False)
        except ValueError:
            continue
        for h in net.hosts():
            s = str(h)
            if s in seen:
                continue
            if len(out) >= cap:
                return out, True
            seen.add(s)
            out.append(s)
    return out, False


@dataclass
class Finding:
    """One server the scan can offer as a backend."""
    host: str
    port: int
    url: str                     # what the backend form's `url` will be
    type: str                    # "openai" | "comfyui" — the gateway's backend type
    flavor: str                  # human label: llama-swap / llama.cpp / vLLM / ollama / ComfyUI 0.3
    models: Optional[int]        # count from the listing; None when it could not be read
    needs_key: bool = False      # /v1/models answered 401/403
    known_as: Optional[str] = None   # name of the backend already registered at this url
    hostname: Optional[str] = None   # reverse DNS, when the injected resolver knew one


_FLAVORS = {"llama-swap": "llama-swap", "llamacpp": "llama.cpp", "vllm": "vLLM", "library": "ollama"}


async def _get(fetch, url):
    try:
        return await fetch(url)
    except Exception:
        return 0, None


async def fingerprint(fetch, base: str) -> Optional[Finding]:
    """What runs at `base` (scheme://host:port), first hit wins:
    /v1/models with a `data` list (or a bare list) → openai, flavor from owned_by;
    401/403 there → openai that needs a key; /system_stats with comfyui_version →
    comfyui; /api/tags with `models` → Ollama (which serves /v1 too). None when the
    port speaks none of these — an open port is not a backend."""
    hp = host_port(base)
    if hp is None:
        return None
    host, port = hp
    st, body = await _get(fetch, f"{base}/v1/models")
    if st in (401, 403):
        return Finding(host, port, base, "openai", "openai-compatible", None, needs_key=True)
    data = body.get("data") if isinstance(body, dict) else body
    if st == 200 and isinstance(data, list) and all(isinstance(m, dict) for m in data):
        owners = {str(m.get("owned_by", "")).lower() for m in data}
        flavor = next((_FLAVORS[o] for o in _FLAVORS if o in owners), "openai-compatible")
        return Finding(host, port, base, "openai", flavor, len(data))
    st, body = await _get(fetch, f"{base}/system_stats")
    ver = (body.get("system") or {}).get("comfyui_version") if isinstance(body, dict) else None
    if st == 200 and ver:
        return Finding(host, port, base, "comfyui", f"ComfyUI {ver}", None)
    st, body = await _get(fetch, f"{base}/api/tags")
    if st == 200 and isinstance(body, dict) and isinstance(body.get("models"), list):
        return Finding(host, port, base, "openai", "ollama", len(body["models"]))
    return None


def host_port(url: str) -> Optional[tuple[str, int]]:
    """(host, port) of a backend url; the scheme's default port when none is given."""
    m = re.match(r"^(https?)://([^/:]+)(?::(\d+))?/?", str(url or "").strip(), re.I)
    if not m:
        return None
    scheme, host, port = m.group(1).lower(), m.group(2).lower(), m.group(3)
    return host, int(port) if port else (443 if scheme == "https" else 80)


def known_backend_for(url: str, backends: list[dict]) -> Optional[str]:
    """Name of a configured backend at the same host+port (scheme and trailing slash
    ignored) — so the panel says 'registered as …' instead of offering it again."""
    want = host_port(url)
    if want is None:
        return None
    for b in backends or []:
        if host_port(b.get("url", "")) == want:
            return b.get("name")
    return None


@dataclass
class ScanResult:
    cidrs: list[str]
    ports: list[int]
    hosts_total: int
    hosts_done: int = 0
    findings: list = field(default_factory=list)
    truncated: bool = False
    started: float = 0.0
    finished: float = 0.0
    error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self.finished == 0.0 and self.error is None


async def _open(host: str, port: int, timeout: float) -> bool:
    try:
        _r, w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout)
    except Exception:
        return False
    w.close()
    try:
        await w.wait_closed()
    except Exception:
        pass
    return True


async def _maybe_await(fn, *a):
    out = fn(*a)
    return await out if inspect.isawaitable(out) else out


async def scan(hosts: list[str], ports: list[int], *, fetch, resolve=None, backends=None,
               result: Optional[ScanResult] = None, concurrency: int = 256,
               connect_timeout: float = 0.5, on_progress=None) -> ScanResult:
    """Phase 1: TCP-connect every (host, port) — `concurrency` at a time, `connect_timeout`
    each; phase 2: fingerprint() every open port. `result` (when given) is filled IN
    PLACE so a console can show progress while the scan runs. A resolver or progress
    hook that raises is ignored; only a crash of the sweep itself lands in `error`."""
    res = result or ScanResult(cidrs=[], ports=list(ports), hosts_total=len(hosts))
    res.started, res.hosts_total, res.hosts_done, res.findings = time.time(), len(hosts), 0, []
    sem = asyncio.Semaphore(max(1, concurrency))

    async def probe_host(host: str):
        open_ports = []
        for port in ports:
            async with sem:
                if await _open(host, port, connect_timeout):
                    open_ports.append(port)
        found = []
        for port in open_ports:
            f = await fingerprint(fetch, f"http://{host}:{port}")
            if f is not None:
                found.append(f)
        if found and resolve is not None:
            try:
                name = await _maybe_await(resolve, host)
            except Exception:
                name = None
            for f in found:
                f.hostname = name or None
        for f in found:
            f.known_as = known_backend_for(f.url, backends or [])
        res.findings.extend(found)
        res.hosts_done += 1
        if on_progress:
            try:
                on_progress(res.hosts_done, res.hosts_total)
            except Exception:
                pass

    try:
        await asyncio.gather(*(probe_host(h) for h in hosts))
        res.findings.sort(key=lambda f: (tuple(int(x) for x in f.host.split(".")), f.port))
    except Exception as e:                      # the sweep itself broke — say so, stay usable
        res.error = f"{type(e).__name__}: {e}"
    finally:
        res.finished = time.time()
    return res
