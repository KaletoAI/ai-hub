"""Backend power control — the PURE half (spec 2026-11-12-backend-power-control, local
only): some backends sit on hardware a person switches off by hand (an eGPU on USB4).
Cutting its power while the OS still holds the device is a surprise removal that can
hang the host's PCI hotplug until a reboot — taking OTHER backends on that host down
with it. A host-side script detaches the device cleanly first; the gateway drains the
backend, runs that script over ssh and says WHEN it is safe to cut power. The switch
itself stays manual.

What lives here (no I/O, imports neither `main` nor `adapters`): the host-script
contract (the ONLY three remote commands the gateway ever sends, their timeouts), the
judgement of one script run (`parse_result` → outcome), the mapping of an outcome onto
the backend's power state per action, the boot reconciliation of a persisted state, the
authorized_keys line the console suggests, and the small validators of the per-backend
fields. main.py runs the flow; admin.py renders it.

The contract (generic, v1 — any device needs a script honouring it):

    status --json          read-only state                       60 s
    off --stop-ct --json   stop the service, detach the device  240 s
    on --start-ct --json   attach the device, start the service 360 s

One JSON object on stdout (the LAST line that parses as one counts), diagnostics on
stderr. Exit 0 = ok | absent | ready, 1 = busy | absent (on: device not on the bus),
2 = hung (hotplug wedged — a host reboot is needed), 3 = call error (a gateway bug),
4 = error | degraded. The ONLY gate for "switch it off now" is
`safe_to_power_off == true`. ssh's own failure (exit 255) is "host unreachable" — an
on-demand host that is off is a state, not an error — unless ssh says the host key or
the key itself was refused, which needs the operator."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

# ── the contract ──────────────────────────────────────────────────────────────
ACTIONS = ("status", "off", "on")
# The remote command line, verbatim. With a forced-command key the host script gets it
# as SSH_ORIGINAL_COMMAND; nothing an admin typed is ever part of it.
COMMANDS = {"status": "status --json", "off": "off --stop-ct --json",
            "on": "on --start-ct --json"}
TIMEOUTS = {"status": 60, "off": 240, "on": 360}
START_TIMEOUT_S = 600          # `on` succeeded → this long for the first healthy discovery
SSH_FAILED = 255               # ssh's own exit status (connect, auth, host key)
TIMED_OUT = 124                # sshrun.run's answer to a timeout

# ── the per-backend power state ───────────────────────────────────────────────
ON = "on"                      # normal operation / nothing known
DRAINING = "draining"          # switch-off clicked: waiting for in-flight + earlier jobs
SWITCHING_OFF = "switching_off"
OFF = "off"                    # the script said safe_to_power_off — cut the power now
BUSY = "busy"                  # the script refused (the device is in use); backend restored
HUNG = "hung"                  # hotplug wedged — do NOT switch off, the host needs a reboot
ERROR = "error"
SWITCHING_ON = "switching_on"
STARTING = "starting"          # `on` ok, backend enabled, waiting for its first healthy poll
UNREACHABLE = "unreachable"    # Check status found the host off (informational)
STATES = (ON, DRAINING, SWITCHING_OFF, OFF, BUSY, HUNG, ERROR, SWITCHING_ON, STARTING,
          UNREACHABLE)
# an action of the flow is running — a second click is refused, and a gateway restart in
# one of these is an interruption (never re-run automatically)
RUNNING = frozenset({DRAINING, SWITCHING_OFF, SWITCHING_ON})
# no job may CLAIM the backend in these (a failover tail computed before the click
# included): its device is being detached, is detached, or is being attached
NO_DISPATCH = frozenset({SWITCHING_OFF, OFF, HUNG, SWITCHING_ON})
# the console's buttons, by state
OFF_FROM = frozenset({ON, BUSY, ERROR, UNREACHABLE})
ON_FROM = frozenset({OFF, HUNG, ERROR, UNREACHABLE})
INTERRUPTED_MSG = "interrupted by gateway restart — run Status"
UNREACHABLE_MSG = "host unreachable"
ABSENT_ON_MSG = "device not found — is it powered on and its cable connected?"


def remote_command(action: str) -> str:
    """The fixed remote command of `action`; ValueError for anything else."""
    if action not in COMMANDS:
        raise ValueError(f"unknown power action {action!r}")
    return COMMANDS[action]


# ── judging one script run ────────────────────────────────────────────────────

def last_json(text: str) -> Optional[dict]:
    """The LAST stdout line that parses as a JSON object (a script may print other lines
    first; the contract's answer is the last one), None when there is none."""
    for ln in reversed((text or "").splitlines()):
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            v = json.loads(ln)
        except ValueError:
            continue
        if isinstance(v, dict):
            return v
    return None


def _txt(b) -> str:
    if isinstance(b, (bytes, bytearray)):
        return bytes(b).decode("utf-8", "replace")
    return str(b or "")


def _tail(text: str, n: int = 2, limit: int = 300) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return " | ".join(lines[-n:])[:limit]


def parse_result(action: str, rc: int, stdout, stderr) -> dict:
    """One script run → `{outcome, rc, result, message, json, safe}`.

    `outcome` is ok | busy | absent | hung | error | unreachable. `safe` is True ONLY
    when the JSON says `safe_to_power_off: true` — nothing else ever earns "switch it
    off now". A JSON `hotplug_hung: true` is `hung` whatever the exit code (the
    conservative reading: never tell anyone to cut power next to a wedged hotplug)."""
    out, err = _txt(stdout), _txt(stderr)
    js = last_json(out)
    res = str(js.get("result") or "") if js else ""
    msg = str(js.get("message") or "").strip() if js else ""
    safe = bool(js) and js.get("safe_to_power_off") is True
    p = {"rc": rc, "result": res, "json": js, "safe": safe, "message": msg[:500]}

    def done(outcome: str, message: str = "") -> dict:
        p["outcome"] = outcome
        if message:
            p["message"] = message[:500]
        elif not p["message"]:
            p["message"] = _tail(err) or (f"exit {rc}" if rc else "")
        p["safe"] = p["safe"] and outcome == "ok"
        return p

    if js is None and rc == SSH_FAILED:
        low = err.lower()
        if ("host key verification failed" in low or "identification has changed" in low
                or "no ed25519 host key is known" in low):
            return done("error", "host key verification failed — the pinned host key does "
                                 "not match; fetch and confirm it again")
        if "permission denied" in low:
            return done("error", "the host refused the key — install the authorized_keys "
                                 "line shown in the backend form")
        return done("unreachable", UNREACHABLE_MSG + (f" ({_tail(err, 1, 160)})" if err.strip() else ""))
    if js is None and rc == TIMED_OUT:
        return done("error", f"no answer within {TIMEOUTS.get(action, '?')} s — the device "
                             "state is unknown; run Check status")
    if rc == 2 or (js is not None and js.get("hotplug_hung") is True) or res == "hung":
        return done("hung", msg or "hotplug hung — the host needs a reboot")
    if rc == 3:
        return done("error", "call error (exit 3) — the host script refused the call; this "
                             "is an AI-Hub bug" + (f": {_tail(err, 1, 200)}" if err.strip() else ""))
    if rc == 0 and js is None:
        return done("error", "the script answered no JSON (exit 0)")
    if rc == 0:
        if action == "off":
            if res in ("ok", "absent") and safe:
                return done("ok")
            return done("error", (msg + " — " if msg else "")
                        + "the script did not report safe_to_power_off")
        if action == "on":
            if res == "ok":
                return done("ok")
            if res == "absent":
                return done("absent", ABSENT_ON_MSG)
            return done("error", msg or f"unexpected result {res!r}")
        return done("ok")                                      # status: ok | absent | ready
    if rc == 1:
        if res == "busy":
            return done("busy", msg or "busy")
        if res == "absent":
            return done("absent", ABSENT_ON_MSG if action == "on" else (msg or "absent"))
        return done("error", msg or f"exit 1 ({res or 'no result'})")
    return done("error", msg or f"exit {rc}" + (f" ({res})" if res else ""))


def off_state(p: dict) -> tuple[str, str]:
    """`off` run → (state, message). Only ok (which already required
    safe_to_power_off) is OFF; an unreachable host after the drain is an ERROR — the
    device state is unknown, so the backend stays disabled."""
    o, m = p.get("outcome"), p.get("message") or ""
    if o == "ok":
        return OFF, m or "safe to switch off now"
    if o == "busy":
        return BUSY, m
    if o == "hung":
        return HUNG, m
    if o == "unreachable":
        return ERROR, m or UNREACHABLE_MSG
    return ERROR, m


def on_state(p: dict) -> tuple[str, str]:
    """`on` run → (state, message): ok → STARTING (the backend is enabled and waits for
    its first healthy discovery); absent → ERROR asking for the power and the cable."""
    o, m = p.get("outcome"), p.get("message") or ""
    if o == "ok":
        return STARTING, m or "device attached — waiting for the backend"
    if o == "absent":
        return ERROR, ABSENT_ON_MSG
    if o == "hung":
        return HUNG, m
    if o == "unreachable":
        return ERROR, m or UNREACHABLE_MSG
    return ERROR, m


def status_state(p: dict, enabled: bool, held: bool = False) -> tuple[str, str]:
    """`status` run → (state, message). Unreachable is informational; a device the
    script calls safe to cut while the backend is disabled reads as OFF; any other
    good answer is ON (normal / nothing to do) — EXCEPT for a backend the power flow
    switched off (`held`, still disabled): a device that is present again (powered on,
    attached by the host) stays OFF without the safe flag. ON there would drop
    "Switched on again" and leave the plain enable, which skips `on --start-ct`."""
    o, m = p.get("outcome"), p.get("message") or ""
    if o == "unreachable":
        return UNREACHABLE, m or UNREACHABLE_MSG
    if o == "hung":
        return HUNG, m
    if o != "ok":
        return ERROR, m
    if p.get("safe") and not enabled:
        return OFF, m or "safe to switch off"
    if held and not enabled:
        return OFF, ((m + " — ") if m else "") + ("not safe to cut the power now; "
                                                  "Switched on again brings the backend back")
    return ON, m


def summary(js: Optional[dict]) -> str:
    """The parsed status JSON in one line for the console (present, driver, users, the
    container's status, safe_to_power_off, hotplug_hung) — "" without JSON."""
    if not isinstance(js, dict):
        return ""
    yn = lambda v: "yes" if v is True else ("no" if v is False else "?")
    dev = js.get("gpu") if isinstance(js.get("gpu"), dict) else (
        js.get("device") if isinstance(js.get("device"), dict) else {})
    parts = []
    if dev:
        parts.append(f"present {yn(dev.get('present'))}")
        parts.append(f"driver {dev.get('driver') or '-'}")
        users = dev.get("users") if isinstance(dev.get("users"), list) else []
        parts.append(f"users {len(users)}" + (f" ({', '.join(str(u) for u in users[:4])})"
                                               if users else ""))
    ct = js.get("ct") if isinstance(js.get("ct"), dict) else None
    if ct:
        parts.append(f"ct {ct.get('id', '?')} {ct.get('status') or '?'}")
    parts.append(f"safe_to_power_off {yn(js.get('safe_to_power_off'))}")
    parts.append(f"hotplug_hung {yn(js.get('hotplug_hung'))}")
    return " · ".join(parts)[:400]


# ── persisted state ───────────────────────────────────────────────────────────

def reconcile_boot(states) -> tuple[dict, bool]:
    """The persisted states as a fresh gateway must read them: an action that was
    RUNNING (draining, switching off/on) when the gateway stopped is an ERROR asking for
    a status check — never re-run automatically, the device may be anywhere between.
    Malformed records are dropped. → (states, changed)."""
    out, changed = {}, False
    if not isinstance(states, dict):
        return {}, bool(states)
    for bid, rec in states.items():
        if not isinstance(rec, dict) or rec.get("state") not in STATES:
            changed = True
            continue
        rec = dict(rec)
        if rec["state"] in RUNNING:
            rec.update(state=ERROR, message=INTERRUPTED_MSG, waiting=0)
            changed = True
        out[str(bid)] = rec
    return out, changed


def action_refusal(state: str, action: str) -> str:
    """Why `action` may not start in `state` ("" = it may)."""
    if state in RUNNING:
        return f"an action is already running ({state.replace('_', ' ')})"
    if action == "off" and state not in OFF_FROM:
        return {OFF: "already switched off",
                HUNG: "the hotplug is hung — do NOT switch off; reboot the host",
                STARTING: "the backend is still starting"}.get(state, f"not in state {state}")
    if action == "on" and state not in ON_FROM:
        return f"not switched off (state {state})"
    return ""


def job_refusal(name: str, state: str, alias: str = "") -> str:
    """The 503 text for a new job of `alias` that has no usable backend while its
    candidate `name` is held by the power flow in `state` ("" = not held). Says the
    ALIAS has no other backend: another candidate may merely be down, and naming only
    the power backend would send the caller to the wrong box."""
    if state in (DRAINING, SWITCHING_OFF):
        what = f"{name} is being switched off"
    elif state in (SWITCHING_ON, STARTING):
        what = f"{name} is being switched on"
    elif state in (OFF, HUNG, ERROR, UNREACHABLE, BUSY):
        what = f"{name} is switched off"
    else:
        return ""
    return (f"'{alias}': {what} and no other backend can take it" if alias else what)


# ── configuration and keys ────────────────────────────────────────────────────

_TARGET_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*@[A-Za-z0-9_][A-Za-z0-9._-]*$")
_SCRIPT_RE = re.compile(r"^/[A-Za-z0-9._+/-]+$")


def target_ok(v) -> bool:
    """`user@host` — plain characters only (it becomes the last ssh argument, after
    `--`), the user part required (the forced-command key sits in ONE user's file)."""
    return isinstance(v, str) and bool(_TARGET_RE.match(v))


def script_ok(v) -> bool:
    """An absolute script path that can stand inside `command="…"` unquoted-safe: no
    quotes, spaces, `..` or shell characters. Display-only — never sent anywhere."""
    return (isinstance(v, str) and bool(_SCRIPT_RE.match(v))
            and ".." not in v.split("/"))


def port_of(v) -> int:
    """The configured ssh port (22 when unset/invalid — Save refuses invalid ones)."""
    if isinstance(v, bool):
        return 22
    try:
        p = int(v)
    except (TypeError, ValueError):
        return 22
    return p if 1 <= p <= 65535 else 22


def known_hosts_name(host: str, port: int) -> str:
    """The name ssh looks a host key up under: `host`, or `[host]:port` off port 22 —
    what ssh-keyscan prints, so the pinned line matches the later connect."""
    return host if port == 22 else f"[{host}]:{port}"


def file_slug(name: str, btype: str) -> str:
    """`<name>-<type>` as a file name under power-keys/ and power-known_hosts/. A name
    with characters a file name should not carry is replaced and suffixed with a short
    hash of the original, so two names never share a key."""
    raw = f"{name}-{btype}"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", raw).lstrip(".") or "backend"
    if safe != raw:
        safe = f"{safe}-{hashlib.sha256(raw.encode()).hexdigest()[:8]}"
    return safe[:120]


def authorized_keys_line(script: str, pubkey: str, backend_name: str) -> str:
    """The line the host's authorized_keys needs: `restrict` (no pty, forwarding, agent,
    X11, user rc) and a forced command, so the key can run that script and nothing else;
    the script then reads the gateway's fixed command from SSH_ORIGINAL_COMMAND. ""
    while the key or the script is missing."""
    parts = str(pubkey or "").split()
    if len(parts) < 2 or not script_ok(script):
        return ""
    comment = "ai-hub-power-" + re.sub(r"[^A-Za-z0-9._-]", "_", str(backend_name or ""))
    return f'restrict,command="{script} --ssh-forced" {parts[0]} {parts[1]} {comment}'
