"""Service profiles of a managed host (spec 2026-09-29 "Dienst-Profile"): what the
gateway runs on the VM for a backend attached to it, and how it starts, probes, restarts
and stops it. Pure — no `main`/`adapters` imports, no I/O: every function returns a
remote command (a string for `sshrun.exec_argv`) or the bytes that go on its stdin, and
`hostctl` runs them.

Two profiles, picked by the backend's TYPE (`profile_for`):

- **ComfyUI** (`comfyui`): set up by the ComfyUI bootstrap (`ops/thunder-bootstrap.sh`,
  run by the controller with its own done-flags — no per-service setup script here), run
  by the `~/start-comfy.sh` loop it writes. The loop takes the service's REMOTE PORT as
  its argument (Ruling M4): a ComfyUI service on another port than 8188 is started, probed
  and restarted on its own port. Probe `GET /object_info` = 200.
- **Command** (`openai`: vLLM, llama-swap, any OpenAI-compatible server): the backend's
  `svc_setup` (optional shell script, run once per hash) and `svc_start` (the start
  command, required). The gateway writes `~/gw-svc-<slug>.sh` — flock lock, `while :`
  restart loop, log `~/gw-svc-<slug>.log`, its OWN Hugging Face cache
  `HF_HOME=~/hf-cache-<slug>` (R-W2: `~/hf-cache` belongs to the model sync, whose prune
  would otherwise see LLM weights as unknown files) — and starts it with `setsid nohup`.
  Probe `GET <svc_health>` (default `/v1/models`) = 200, 401 or 403 (R-W9: a server with
  its own API key answers 401 and is up).

The admin's text (`svc_setup`, `svc_start`) NEVER appears in a command this module
returns: a command line is visible to every process on the VM (`ps`) and is logged by
the gateway. It travels on stdin only — the setup script into `bash -s`, the start
command as a quoted heredoc INSIDE the wrapper script, which is itself streamed into
`cat >`. Both are normalised first (CRLF from a browser textarea → LF): a `\\r` at a line
end is a different command.

Stopping is by the LOCK: the running loop writes its pid there, and because `setsid`
made it a session and process-group leader, `kill -- -<pid>` ends the loop and every
child of it (a `bash -c` start command that forks its server is one) — a pattern match
could hit the ssh shell that runs the stop. Only a HELD lock is trusted: a pid left in
a lock file nobody holds belongs to a process long gone (or to a stranger by now).
Covered by tests/test_services.py (pure + a real-shell run of the wrapper)."""
from __future__ import annotations

import hashlib
import re
from typing import Optional

import sshrun

HEALTH_DEFAULT = "/v1/models"
HEALTH_RE = re.compile(r"^/[A-Za-z0-9._~/?=&-]*$")
_SLUG_MAX = 48

# The ComfyUI the start loop runs, on ANY port (one ComfyUI per VM — spec): the brackets
# keep the pattern from matching the command line of the remote `bash -c` carrying it.
# The ComfyUI bootstrap's `phase stop` uses exactly this pattern (pinned by a test).
COMFY_MAIN_PATTERN = "[m]ain[.]py --listen 127[.]0[.]0[.]1 --port [0-9]"
_COMFY_LOCK = "~/.start-comfy.lock"
_TERM_TICKS = 200                # × 0.1 s between TERM and KILL of a service's group

# Is any non-zombie process left in process group $1? (A killed process lingers as a
# zombie until reaped; `kill -0` would count it.)
_GROUP_ALIVE = ("_gw_grp() { ps -e -o pgid= -o stat= | "
                "awk -v g=\"$1\" '$1==g && $2 !~ /^Z/ {f=1} END {exit !f}'; }")


def slug(name) -> str:
    """The backend name reduced to `[a-z0-9-]` (runs of anything else → one `-`, no
    `-` at either end, at most 48 characters; "svc" when nothing is left). Names the
    service's files on the VM, so it is a safe file name by construction (R-W8: two
    services of one host with the same slug are refused)."""
    s = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")
    return s[:_SLUG_MAX].strip("-") or "svc"


def _text(v) -> str:
    """Admin text as it goes to the VM: CRLF/CR → LF, one trailing newline, "" when
    blank."""
    if not isinstance(v, str):
        return ""
    t = v.replace("\r\n", "\n").replace("\r", "\n")
    return t.rstrip("\n") + "\n" if t.strip() else ""


def _delimiter(text: str) -> str:
    """A heredoc delimiter no line of `text` equals (derived from the text, so the
    wrapper's bytes — and its hash — are stable)."""
    d = "GW_SVC_START_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12].upper()
    lines = set(text.splitlines())
    while d in lines:
        d += "_X"
    return d


def _group_stop(lock: str) -> str:
    """End the process group whose leader's pid the HELD lock `lock` records: TERM,
    up to 20 s for it to go, then KILL (and a moment for the lock to free, so a start
    right after takes it). Nothing when the lock is free — its pid is stale."""
    return (f"{_GROUP_ALIVE}; lk={lock}; "
            'if [ -f "$lk" ] && ! flock -n "$lk" true; then '
            'read -r p _ < "$lk" || true; '
            'case "${p:-}" in ""|*[!0-9]*) ;; *) '
            'kill -TERM -- "-$p" 2>/dev/null; i=0; '
            f'while [ "$i" -lt {_TERM_TICKS} ] && _gw_grp "$p"; do sleep 0.1; i=$((i+1)); done; '
            'if _gw_grp "$p"; then kill -KILL -- "-$p" 2>/dev/null; flock -w 10 "$lk" true; fi;; '
            "esac; fi")


def _remote_port(backend: dict) -> int:
    """The service's loopback port on the VM (ValueError when it is none)."""
    return sshrun._port((backend or {}).get("remote_port"))


def _port_errors(backend: dict) -> list:
    try:
        _remote_port(backend)
    except (TypeError, ValueError):
        return ["remote port must be a number 1–65535"]
    return []


class ComfyProfile:
    """ComfyUI: bootstrapped by the controller (`ops/thunder-bootstrap.sh`), run by the
    `~/start-comfy.sh <port>` loop that bootstrap writes."""
    KIND = "comfyui"
    default_port = 8188

    def probe_path(self, backend: dict) -> str:
        return "/object_info"

    def probe_ok(self, status) -> bool:
        return status == 200

    def log_path(self, backend: dict) -> str:
        return "~/comfy.log"

    def start_cmd(self, backend: dict) -> str:
        """Start the loop on the service's port. flock-guarded, so a start while it runs
        is a no-op; a missing script (a bootstrap that died before writing it) fails in
        seconds with a named reason instead of a 10-minute probe timeout."""
        port = _remote_port(backend)
        return ("test -x ~/start-comfy.sh || { echo 'start-comfy.sh missing' >&2; exit 3; }; "
                f"setsid nohup ~/start-comfy.sh {port} >/dev/null 2>&1 < /dev/null &")

    def stop_cmd(self, backend: dict) -> str:
        """The loop and its ComfyUI (the lock's process group), then any ComfyUI the
        loop pattern still finds (one started outside the loop)."""
        return (": gw-comfy-stop ; " + _group_stop(_COMFY_LOCK)
                + f"; pkill -f '{COMFY_MAIN_PATTERN}'; true")

    def restart_cmd(self, backend: dict) -> str:
        """Stop, then start on the service's port: a loop still running on another port
        (the service's port changed) is ended rather than kept by the lock."""
        start = self.start_cmd(backend)            # validates the port first
        return self.stop_cmd(backend) + "; " + start

    def wrapper_script(self, backend: dict) -> Optional[bytes]:
        return None                                # written by the ComfyUI bootstrap

    def setup_script(self, backend: dict) -> Optional[bytes]:
        return None                                # the ComfyUI bootstrap (controller)

    def setup_hash(self, backend: dict) -> Optional[str]:
        return None

    def validate(self, backend: dict) -> list:
        return _port_errors(backend)


class CommandProfile:
    """An OpenAI-compatible server the admin describes by a setup script and a start
    command (spec "Befehls-Profil")."""
    KIND = "openai"
    default_port = 8000

    @staticmethod
    def _slug(backend: dict) -> str:
        return slug((backend or {}).get("name"))

    def _file(self, backend: dict, ext: str) -> str:
        return "~/" + sshrun.q(sshrun.safe_rel(f"gw-svc-{self._slug(backend)}{ext}"))

    def probe_path(self, backend: dict) -> str:
        p = (backend or {}).get("svc_health")
        return p if isinstance(p, str) and HEALTH_RE.fullmatch(p) else HEALTH_DEFAULT

    def probe_ok(self, status) -> bool:
        return status in (200, 401, 403)

    def log_path(self, backend: dict) -> str:
        return self._file(backend, ".log")

    def setup_log(self, backend: dict) -> str:
        return self._file(backend, ".setup.log")

    def upload_cmd(self, backend: dict) -> str:
        """The wrapper from stdin, replaced by a rename: the running loop keeps reading
        the OLD inode (bash reads a script as it executes it — overwriting it in place
        would feed the running loop the new file's bytes at its old offset)."""
        f, t = self._file(backend, ".sh"), self._file(backend, ".sh.tmp")
        return f"cat > {t} && chmod 700 {t} && mv -f {t} {f}"

    def start_cmd(self, backend: dict) -> str:
        f = self._file(backend, ".sh")
        name = f[2:]
        return (f"test -x {f} || {{ echo '{name} missing' >&2; exit 3; }}; "
                f"setsid nohup {f} >/dev/null 2>&1 < /dev/null &")

    def stop_cmd(self, backend: dict) -> str:
        return ": gw-svc-stop ; " + _group_stop(self._file(backend, ".lock")) + "; true"

    def restart_cmd(self, backend: dict) -> str:
        """Stop and start a FRESH loop — so a rewritten wrapper (a changed start
        command) is what runs afterwards."""
        return self.stop_cmd(backend) + "; " + self.start_cmd(backend)

    def setup_cmd(self, backend: dict) -> str:
        """What `hostctl` wraps in `bash -o pipefail -c` with the setup script on stdin:
        `bash -s` reads it (never a command line), tee keeps the output on the VM for
        the tail after a timeout. The setup gets the service's own HF cache too."""
        sl = self._slug(backend)
        return (f'export HF_HOME="$HOME/hf-cache-{sl}"; mkdir -p "$HF_HOME"; cd ~ && '
                f"bash -s 2>&1 | tee {self.setup_log(backend)}")

    def setup_script(self, backend: dict) -> Optional[bytes]:
        t = _text((backend or {}).get("svc_setup"))
        return t.encode("utf-8") if t else None

    def setup_hash(self, backend: dict) -> Optional[str]:
        """sha256 of the setup script as it is run; None = no setup. A stored hash
        equal to this one means the setup already ran on the instance."""
        s = self.setup_script(backend)
        return hashlib.sha256(s).hexdigest() if s is not None else None

    def wrapper_script(self, backend: dict) -> bytes:
        """`~/gw-svc-<slug>.sh`: one instance per service (the lock; its pid is what the
        stop reads), the start command as a quoted heredoc run by `bash -c`, restarted
        2 s after it exits, stdout+stderr appended to the log. fd 9 (the lock) is closed
        for every child, so a child outliving a killed loop cannot keep the lock."""
        sl = self._slug(backend)
        text = _text((backend or {}).get("svc_start"))
        delim = _delimiter(text)
        return ("#!/usr/bin/env bash\n"
                f"# Written by the AI-Hub gateway for its service {sl} — the next start\n"
                "# overwrites it. One instance (the lock), restarted 2 s after it exits.\n"
                "set -u\n"
                f'lock="$HOME/gw-svc-{sl}.lock"\n'
                f'log="$HOME/gw-svc-{sl}.log"\n'
                'exec 9>>"$lock"\n'
                "flock -n 9 || exit 0\n"
                'echo "$$" >"$lock"\n'
                f'export HF_HOME="$HOME/hf-cache-{sl}"\n'
                'mkdir -p "$HF_HOME" 9>&-\n'
                f"IFS= read -r -d '' GW_START <<'{delim}' || true\n"
                f"{text}{delim}\n"
                'cd "$HOME" || exit 1\n'
                "while :; do\n"
                '  echo "gw: starting $(date -u +%Y-%m-%dT%H:%M:%SZ 9>&-)" >>"$log"\n'
                '  bash -c "$GW_START" 9>&- >>"$log" 2>&1 </dev/null\n'
                '  echo "gw: exited (rc $?) — restarting in 2 s" >>"$log"\n'
                "  sleep 2 9>&-\n"
                "done\n").encode("utf-8")

    def validate(self, backend: dict) -> list:
        """What keeps this service from running — fixed texts, never the admin's text
        (they reach the log and the card)."""
        b = backend or {}
        errs = _port_errors(b)
        start = b.get("svc_start")
        if not _text(start):
            errs.append("start command (svc_start) is required")
        for key, what in (("svc_start", "start command"), ("svc_setup", "setup script")):
            v = b.get(key)
            if isinstance(v, str) and "\x00" in v:
                errs.append(f"{what} contains a NUL byte")
        h = b.get("svc_health")
        if h not in (None, "") and not (isinstance(h, str) and HEALTH_RE.fullmatch(h)):
            errs.append("health path must match ^/[A-Za-z0-9._~/?=&-]*$")
        return errs


COMFY = ComfyProfile()
COMMAND = CommandProfile()


def profile_for(backend) -> Optional[object]:
    """The profile of a backend by its type: ComfyUI, command (OpenAI-compatible; a
    backend without a type is one, like `main.backend_id`'s default), else None — a
    cloud or Anthropic backend has nothing to run on a VM and cannot be attached."""
    if not isinstance(backend, dict):
        return None
    t = backend.get("type", "openai")
    if t is None:
        t = "openai"
    return {"comfyui": COMFY, "openai": COMMAND}.get(t)
