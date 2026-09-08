# LAN scan for backends — design

**Date:** 2026-09-08 · **Status:** approved in chat (Kai, 2026-09-08), spec for review

## Problem

Every backend the gateway knows was typed into the Backends tab by hand: name, type,
URL. A new llama-swap on a Spark, a ComfyUI on a fresh container, an Ollama someone
started on a laptop — none of them shows up anywhere until the operator remembers the
IP and port. The gateway sits on the same LAN and already speaks every one of those
servers' discovery protocols (`/v1/models`, `/object_info`, `/system_stats`); it can
find them itself.

## Decision (what was agreed in chat)

- **Manual trigger only.** A button in the Backends tab starts one scan. No periodic
  scanning, no scan at startup — the LAN sees scan traffic only when someone clicks.
- **Range = the gateway's own subnet(s)**, derived from its IPv4 addresses, overridable
  by an admin CIDR list. Hard cap of 1024 addresses per scan.
- **Ports = a fixed, editable list.** Default `8080, 8000, 11434, 8188, 1234, 5000`.
- **Type detection over HTTP** per open port; a find is offered with an *Add* link
  that opens the existing backend form pre-filled. **Nothing is ever added
  automatically.**
- Approach A of the three discussed: an in-process asyncio scanner in a new pure
  module. `nmap`/`arp-scan` (external binaries, absent on prod) and passive discovery
  (ARP table, mDNS — sees only hosts the box already talked to) were rejected.

## Components

### `netscan.py` — new, pure (no `main`/`adapters` imports, no module-level state)

| Function | Does |
|---|---|
| `parse_ip_addr(text) -> list[str]` | CIDRs from `ip -o -4 addr show` output: one per non-loopback address. A prefix shorter than /24 becomes the address's **/24** (a /16 would blow the cap and scan a whole site); a longer prefix (/30 on the ConnectX link) is kept as is. |
| `local_cidrs(run=subprocess…) -> list[str]` | Runs `ip -o -4 addr show` (injected runner) and parses it; `[]` when the command is missing — the scan then needs the admin list. |
| `parse_ports(text) -> list[int]` | `"8080, 8000"` → `[8080, 8000]`; junk dropped, 1–65535 only, order kept, deduped. |
| `expand_targets(cidrs, cap=1024) -> (hosts, truncated)` | Host addresses of every CIDR (network/broadcast dropped for /24 and shorter), deduped, cut at `cap`; `truncated` says so. |
| `async scan(hosts, ports, *, fetch, resolve=None, concurrency=256, connect_timeout=0.5, on_progress=None) -> ScanResult` | Phase 1: TCP connect to every (host, port) with a semaphore of `concurrency`; phase 2: `fingerprint()` every open port. `on_progress(done, total)` after each host. |
| `async fingerprint(fetch, base) -> Optional[Finding]` | The type test, in this order, first hit wins: `GET /v1/models` → JSON with a `data` list → **openai** (flavor from `owned_by` values: `llama-swap`, `llamacpp` → llama.cpp, `vllm`, `library` → ollama, else `openai-compatible`; `models` = count); 401/403 there → **openai**, `needs_key=True`, models unknown. Else `GET /system_stats` → JSON with `system.comfyui_version` → **comfyui** (flavor `ComfyUI <version>`). Else `GET /api/tags` → JSON with `models` → **openai**, flavor `ollama` (Ollama serves `/v1` OpenAI-compatibly at the same base). Nothing matched → `None` (an open port that is not a backend is not reported). |
| `known_backend_for(url, backends) -> Optional[str]` | The name of a configured backend whose URL has the same host and port (scheme-insensitive, trailing slash ignored), else None. |

`fetch(url) -> (status, body_json_or_None)` is injected (main hands in an httpx one
with a 2 s timeout); `resolve(host) -> hostname|None` is an optional reverse-DNS hook
(main injects `socket.gethostbyaddr` behind `asyncio.to_thread`; failures → None).

Dataclasses: `Finding(host, port, url, type, flavor, models, needs_key, known_as,
hostname)` and `ScanResult(cidrs, ports, hosts_total, hosts_done, findings, truncated,
started, finished, error)`.

### `main.py`

- Settings (store `settings`, Server tab → runtime group): `scan_cidrs` (text, comma
  separated, blank = derive) and `scan_ports` (text, blank = the default list).
  Read in `_apply_server_settings()` into `scan_cidrs: list[str]` / `scan_ports:
  list[int]`.
- State: `_scan = {"task": Optional[asyncio.Task], "result": Optional[ScanResult]}`.
  One scan at a time — a click while one runs is a no-op that shows the running one.
- `start_scan()` → builds hosts (`scan_cidrs or netscan.local_cidrs()`, expanded,
  capped), spawns the task, stores the result object which the task fills in place
  (progress visible while running). `known_as` is resolved against `backends` (config
  + store, the same list the Backends tab renders) at the end of the scan.
- `scan_status() -> dict` for the console: `running`, `hosts_done/total`, `findings`
  as plain dicts, `truncated`, `error`, `cidrs`, `ports`.
- Bound into admin like the other callbacks (`admin.bind(scan_start=…, scan_status=…)`).

### `admin.py`

- Backends tab: **Scan network** button next to *+ New* → `POST /ui/backends/scan` →
  303 back to `/ui/backends`.
- `_scan_panel(status)` under the hosts panel (`data-sk="scan"` so the live morph
  keeps it as one logical table): while running, `n / N hosts` and a hint; when done,
  one row per finding: `host` (hostname when resolved), `port`, type badge + flavor,
  models count or *needs api key*, and either an **Add** link or *registered as
  `name`*. `truncated` shows a warning row naming the cap; `error` a bad row.
- The page renders with `refresh=2` while a scan runs (the same `_page(refresh=…)`
  the draining state uses), so progress and the finished table arrive via the morph.
- **Add** = `/ui/backends?new=1&url=<url>&type=<type>&name=<suggestion>`;
  `backends_page` passes those into `_backend_form(None, hosts, prefill={…})`, which
  fills `name`, `type`, `url` (and ticks `local` for an openai find — a LAN server is
  local by definition; the operator can untick). Name suggestion = hostname without
  domain, else `host-port` (e.g. `192-168-8-36-8080`) — must be unique per type, and a
  clash is what the existing save validation already refuses.
- Server tab: the two settings rows with notes (`scan_cidrs`: "blank = the /24 of every
  IPv4 address of this host; e.g. `192.168.8.0/24, 10.20.0.0/30`; at most 1024 hosts
  per scan", `scan_ports`: "blank = 8080, 8000, 11434, 8188, 1234, 5000").

## Data flow

```
click → POST /ui/backends/scan → main.start_scan()
        ├─ cidrs = settings or netscan.local_cidrs()
        ├─ hosts, truncated = netscan.expand_targets(cidrs)      (cap 1024)
        └─ task: netscan.scan(hosts, ports, fetch=httpx, resolve=gethostbyaddr)
               phase 1  TCP connect (0.5 s, 256 parallel) → open (host, port)
               phase 2  fingerprint(base) per open port → Finding | None
               known_as = netscan.known_backend_for(url, backends)
GET /ui/backends (live, 2 s) → _scan_panel(main.scan_status())
Add → /ui/backends?new=1&url=…&type=…&name=… → _backend_form(prefill) → Save (unchanged)
```

## Error handling

- No CIDR at all (no `ip` binary, no admin list): the panel says so and names the
  Server-tab field; nothing is scanned.
- A CIDR that does not parse is skipped with a note in the panel; the rest scans.
- Connect failures are the normal case and silent. Fingerprint fetch errors (timeouts,
  non-JSON) mean "not a backend" for that port — silent.
- A crash inside the task lands in `result.error`, shown as a bad row; the button
  works again.
- The scan never writes to the store.

## Testing — `tests/test_netscan.py` (silent failures this guards)

- `parse_ip_addr`: loopback excluded; `/16` narrowed to the address's `/24`; `/30`
  kept; IPv6 lines ignored. (A wrong range silently scans nothing or a whole site.)
- `parse_ports`, `expand_targets`: cap honoured with `truncated=True`; network and
  broadcast dropped.
- `fingerprint` with a stubbed `fetch`: llama-swap, vLLM, generic OpenAI, needs-key
  (401), ComfyUI via `/system_stats`, Ollama via `/api/tags`, and a plain HTTP server
  → None. (A misclassified type pre-fills a form that saves a backend the gateway
  then cannot discover.)
- `known_backend_for`: same host+port with different scheme/trailing slash matches;
  different port does not. (Otherwise every scan re-offers what is already there.)
- `scan` end to end against a real local HTTP stub on `127.0.0.1:<port>`: hosts
  `["127.0.0.1"]`, ports `[stub, closed]` → exactly one finding; `on_progress`
  reaches `(1, 1)`.
- `admin._scan_panel`: running state renders progress; done state renders the Add
  link with the encoded prefill query; a known finding renders *registered as* and
  no Add link. `admin._backend_form(prefill)`: the three fields carry the values.

Everything else (the button, the 303, the live refresh) is verified by running the
gateway and clicking, as the console always is.

## Out of scope

- IPv6, other subnets than the gateway's own (unless listed), UDP/mDNS.
- Detecting Meshy/Tripo/Anthropic (cloud, nothing to scan).
- Auto-adding, auto-enabling, or re-scanning on a schedule.
