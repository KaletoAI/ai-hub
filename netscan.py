"""LAN scan for backends — the pure half (no main/adapters/admin imports, no module
state; every I/O dependency is injected so tests run without a network).

What it answers: which hosts on the gateway's own subnet run something the gateway
can register as a backend — llama-swap / llama.cpp / vLLM / Ollama (`openai`) or
ComfyUI (`comfyui`) — and whether each one is registered already. It never adds
anything: the console offers each finding as a pre-filled form.
"""
import ipaddress
import re
import subprocess
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
