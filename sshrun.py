"""System-`ssh` plumbing for the Thunder Compute backend: argv builders, the remote
path guard, a one-shot async runner, the supervised tunnel and the key generator.

Why system `ssh` and not asyncssh/paramiko: no new dependency, and OpenSSH already
does everything the controller needs (a `-L` forward that dies loudly with
`ExitOnForwardFailure`, keepalives that notice a vanished instance, a known_hosts file
per instance). What this module guards is the part that fails SILENTLY or dangerously:

- **argv shape.** Every argv ends in `-- <host> [cmd]`: a host string that starts with
  `-` (`-oProxyCommand=…`) would otherwise be parsed as an OPTION and run a local
  command. Remote arguments are the caller's to `q()` (= `shlex.quote`), because the
  remote side hands the command string to a shell. Nothing secret ever goes into argv
  (it is world-readable in /proc) — the key is a FILE named by `-i`.
- **remote paths.** `safe_rel` refuses anything that could leave the model tree once
  quoted into a remote command (absolute, `..`, hidden segments such as
  `hf-cache/.token`, control characters, backslashes). Quoting stops the shell;
  it does not stop `cat ../../.ssh/id_ed25519`.
- **the tunnel.** `Supervisor` restarts `ssh -N -L …` with a doubling backoff (reset
  after a stable minute), logs the stderr tail of every exit — a tunnel that died on a
  changed host key and one that died on a refused port look identical otherwise — and
  on `stop()` ends and REAPS its process: an orphaned `ssh -N` keeps the local port
  bound, and the next tunnel then exits at once on `ExitOnForwardFailure`.

The builders, `safe_rel`, `q` and `next_backoff` are pure (no `main`/`adapters`
imports, no I/O at import, no module-level config); `run`, `Supervisor` and `keygen`
spawn processes and are injected into the controller. Covered by test_sshrun.py.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import signal
import time
from typing import Callable, Optional

STABLE_S = 60                   # a tunnel that lived this long resets the backoff
_STDERR_TAIL = 2048             # bytes of a dead tunnel's stderr kept for the log
_TERM_GRACE_S = 5               # terminate → kill after this many seconds


# ── argv builders (pure) ─────────────────────────────────────────────────────

def ssh_base(key: str, known_hosts: str, strict: str = "accept-new",
             port: Optional[int] = None) -> list[str]:
    """The options every ssh call shares. BatchMode: never prompt (a prompt would hang
    a subprocess nobody can answer); IdentitiesOnly: offer ONLY `key`, not whatever an
    agent holds (a server with MaxAuthTries gives up before reaching the right one)."""
    argv = ["ssh", "-i", key,
            "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes",
            "-o", f"UserKnownHostsFile={known_hosts}",
            "-o", f"StrictHostKeyChecking={strict}",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "ConnectTimeout=15"]
    if port:
        argv += ["-p", str(port)]
    return argv


def tunnel_argv(key: str, known_hosts: str, host: str, port: int, lport: int,
                rport: int = 8188) -> list[str]:
    """`ssh -N -L 127.0.0.1:<lport>:127.0.0.1:<rport> -- host`. Bound to loopback on
    BOTH ends: the local side must not expose the instance's ComfyUI to the LAN, and
    ExitOnForwardFailure makes a taken local port an exit (→ restart, logged) instead
    of a connected-looking tunnel that forwards nothing."""
    return ssh_base(key, known_hosts, port=port) + [
        "-N", "-o", "ExitOnForwardFailure=yes",
        "-L", f"127.0.0.1:{lport}:127.0.0.1:{rport}",
        "--", host]


def exec_argv(key: str, known_hosts: str, host: str, port: int,
              remote_cmd: str) -> list[str]:
    """`ssh … -- host <remote_cmd>`. `remote_cmd` is ONE string the remote shell
    parses — every argument inside it must already have gone through `q()`."""
    return ssh_base(key, known_hosts, port=port) + ["--", host, remote_cmd]


def q(s) -> str:
    """Quote one argument for the remote shell."""
    return shlex.quote(str(s))


def safe_rel(rel: str) -> str:
    """A relative path that stays inside the tree it is resolved against, returned
    unchanged; `ValueError` otherwise. One trailing `/` is allowed (a directory entry
    of the model catalog, e.g. a whole Hub repo) — every other empty segment is not.
    A FIRST segment starting with `-` is refused too: quoting does not stop a remote
    `cat`/`rm` from reading it as an option. Only the first segment is guarded
    (`models/-x` passes) — a later segment cannot stand at the start of an argument."""
    if not isinstance(rel, str) or not rel:
        raise ValueError("empty path")
    if "\\" in rel:
        raise ValueError(f"backslash in path: {rel!r}")
    if any(ord(c) < 32 or ord(c) == 127 for c in rel):
        raise ValueError(f"control character in path: {rel!r}")
    if rel.startswith("/"):
        raise ValueError(f"absolute path: {rel!r}")
    segs = rel.split("/")
    if len(segs) > 1 and segs[-1] == "":
        segs = segs[:-1]        # the one allowed trailing "/" (directory)
    for seg in segs:
        if seg == "":
            raise ValueError(f"empty path segment: {rel!r}")
        if seg.startswith("."):
            # ".", "..", and hidden files (.ssh, .token) alike
            raise ValueError(f"dot segment in path: {rel!r}")
    if segs[0].startswith("-"):
        raise ValueError(f"path reads as an option: {rel!r}")
    return rel


def next_backoff(delay: float, ran_s: float, min_backoff: float, max_backoff: float,
                 stable_s: float = STABLE_S) -> tuple[float, float]:
    """(sleep now, delay for next time). A run of `stable_s` or longer resets to
    `min_backoff` — the instance was fine and just dropped once; otherwise the wait
    doubles up to `max_backoff`, so a dead host is not hammered every two seconds."""
    if ran_s >= stable_s:
        delay = min_backoff
    return delay, min(max_backoff, delay * 2)


# ── processes ────────────────────────────────────────────────────────────────

def _signal(p, sig) -> None:
    """Send `sig` to `p` BY PID. Not `p.terminate()`/`p.kill()`: CPython's
    `Popen.send_signal` first `poll()`s, and for a child that has exited but not yet
    been reported by asyncio's child watcher that poll REAPS it — the watcher then
    logs "exit status already read" and reports rc 255 instead of the real code.
    While asyncio has no returncode the pid is still ours (alive or a zombie)."""
    if p.returncode is None:
        try:
            os.kill(p.pid, sig)
        except ProcessLookupError:
            pass


async def _reap(p) -> None:
    """End `p` (SIGTERM, SIGKILL after the grace) and WAIT for it — an unwaited child
    is a zombie, and an ssh left running keeps its forward's local port bound."""
    if p.returncode is None:
        _signal(p, signal.SIGTERM)
        try:
            await asyncio.wait_for(p.wait(), _TERM_GRACE_S)
            return
        except asyncio.TimeoutError:
            _signal(p, signal.SIGKILL)
    await p.wait()


async def run(argv: list[str], stdin: Optional[bytes] = None,
              timeout: float = 60) -> tuple[int, bytes, bytes]:
    """Run `argv` to completion → (rc, stdout, stderr). A timeout KILLS the process
    and answers rc 124 (timeout(1)'s code), so a hung ssh never outlives its caller;
    a cancelled caller kills it too. A missing binary raises (OSError) — that is a
    broken install, not a remote failure."""
    p = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(p.communicate(stdin), timeout)
    except asyncio.TimeoutError:
        _signal(p, signal.SIGKILL)
        await p.wait()
        return 124, b"", b"timeout"
    except BaseException:
        _signal(p, signal.SIGKILL)
        await p.wait()
        raise
    return p.returncode, out, err


class _Run:
    """One start()…stop() cycle of a Supervisor. Per-run, not per-supervisor: a
    start() that arrives while stop() is still waiting for the old loop must not
    reset THAT loop's stop flag or share its process slot — two loops would then
    both respawn, and stop() could only ever reach one of the processes."""
    __slots__ = ("task", "proc", "stopping", "wake", "attempts")

    def __init__(self):
        self.task: Optional[asyncio.Task] = None
        self.proc = None
        self.stopping = False
        self.wake = asyncio.Event()
        self.attempts = 0


class Supervisor:
    """Keeps `argv_fn()` running: respawns it whenever it exits, with a backoff from
    `min_backoff` doubling to `max_backoff` (reset after a run of `STABLE_S`).
    `argv_fn` is called per spawn, so a changed ip/port/known_hosts is picked up on
    the next restart. `log(str)` gets one line per exit with rc, runtime and the
    stderr tail — best effort: a raising `log` never ends supervision. `spawn` is the
    `create_subprocess_exec` seam for tests. At most ONE process is alive at a time,
    also across a stop() → start() overlap (the new loop waits for the old one)."""

    def __init__(self, argv_fn: Callable[[], list[str]], log: Callable[[str], None],
                 spawn=asyncio.create_subprocess_exec,
                 min_backoff: float = 2, max_backoff: float = 60):
        self._argv_fn = argv_fn
        self._log = log
        self._spawn = spawn
        self._min = min_backoff
        self._max = max_backoff
        self._cur: Optional[_Run] = None     # the latest run, kept after it stopped

    @property
    def running(self) -> bool:
        """A tunnel process is alive right now (not: the supervisor is active)."""
        r = self._cur
        return r is not None and r.proc is not None and r.proc.returncode is None

    @property
    def restarts(self) -> int:
        """Spawn attempts after the first one, in the current run."""
        return max(0, self._cur.attempts - 1) if self._cur is not None else 0

    def _say(self, msg: str) -> None:
        try:
            self._log(msg)
        except Exception:
            pass                        # a broken logger must not end supervision

    def _collect(self, t: asyncio.Task) -> None:
        """Retrieve a finished loop's exception — unretrieved, asyncio reports it as
        "Task exception was never retrieved" when the task is collected."""
        if t.done() and not t.cancelled():
            exc = t.exception()
            if exc is not None:
                self._say(f"tunnel supervisor loop crashed: {exc!r}")

    def start(self) -> None:
        old = self._cur
        if old is not None and old.task is not None:
            if not old.task.done() and not old.stopping:
                return                  # already running
            if old.task.done():
                self._collect(old.task)
        prev = old.task if old is not None and old.task is not None \
            and not old.task.done() else None
        r = _Run()
        r.task = asyncio.get_running_loop().create_task(self._loop(r, prev))
        self._cur = r

    async def stop(self) -> None:
        """Stop respawning, end and reap the current process. Idempotent; a second
        stop() while one is in flight waits for the same end.

        The loop is woken and left to exit BY ITSELF rather than cancelled outright:
        a cancel landing inside `create_subprocess_exec` makes asyncio `poll()` the
        fresh child, which reaps it behind the child watcher's back ("exit status
        already read"). Escalation: SIGTERM → SIGKILL after the grace → cancel."""
        r = self._cur
        if r is None or r.task is None:
            return
        r.stopping = True
        r.wake.set()
        t = r.task
        for sig in (signal.SIGTERM, signal.SIGKILL, None):
            if t.done():
                break
            if sig is not None and r.proc is not None:
                _signal(r.proc, sig)
            if sig is None:
                t.cancel()              # last resort: the loop is stuck somewhere
            # asyncio.wait: neither raises the loop's exception nor cancels it on
            # timeout; a cancelled CALLER still gets its CancelledError
            await asyncio.wait([t], timeout=_TERM_GRACE_S)
        if t.done():
            self._collect(t)

    async def _loop(self, r: _Run, prev: Optional[asyncio.Task]) -> None:
        delay = self._min
        try:
            if prev is not None:
                # the previous run is still ending (stop() → start() overlap): its
                # ssh holds the forward's local port until it is reaped
                await asyncio.wait([prev])
            while not r.stopping:
                r.attempts += 1
                ran = 0.0
                try:
                    ran = await self._once(r)
                except Exception as e:  # unexpected (stderr read, wait, a stub)
                    self._say(f"tunnel supervisor error: {e!r}")
                    p = r.proc
                    if p is not None and p.returncode is None:
                        await _reap(p)  # never leave it running under a respawn
                if r.stopping:
                    break
                sleep_s, delay = next_backoff(delay, ran, self._min, self._max)
                try:
                    await asyncio.wait_for(r.wake.wait(), sleep_s)
                except asyncio.TimeoutError:
                    pass
        finally:
            p = r.proc
            if p is not None and p.returncode is None:
                await _reap(p)          # the proc stays referenced: `running` reads its rc

    async def _once(self, r: _Run) -> float:
        """Spawn once and wait for the exit → seconds it ran (0 if it never started)."""
        try:
            p = await self._spawn(*self._argv_fn(),
                                  stdin=asyncio.subprocess.DEVNULL,
                                  stdout=asyncio.subprocess.DEVNULL,
                                  stderr=asyncio.subprocess.PIPE)
        except Exception as e:          # missing ssh binary, argv_fn raising
            self._say(f"tunnel spawn failed: {e}")
            return 0.0
        r.proc = p
        if r.stopping:                  # stop() arrived during the spawn
            return 0.0
        t0 = time.monotonic()
        tail = b""
        if p.stderr is not None:
            # read continuously: a full stderr pipe would stall ssh itself
            while True:
                chunk = await p.stderr.read(4096)
                if not chunk:
                    break
                tail = (tail + chunk)[-_STDERR_TAIL:]
        rc = await p.wait()
        ran = time.monotonic() - t0
        msg = tail.decode("utf-8", "replace").strip()
        self._say(f"tunnel exited rc={rc} after {ran:.1f}s" + (f": {msg}" if msg else ""))
        return ran


async def keygen(path: str) -> str:
    """The public key of the ed25519 key at `path`, generating it (no passphrase —
    the gateway runs unattended) when absent. Idempotent; the private key is 0600."""
    if not os.path.exists(path):
        rc, _, err = await run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "",
                                "-C", "ai-hub", "-f", path], timeout=30)
        if rc != 0:
            raise RuntimeError(f"ssh-keygen failed (rc {rc}): "
                               f"{err.decode('utf-8', 'replace').strip()}")
    os.chmod(path, 0o600)
    pub = path + ".pub"
    if os.path.exists(pub):
        with open(pub, encoding="utf-8") as f:
            return f.read().strip()
    # the .pub was lost: derive it from the private key instead of failing
    rc, out, err = await run(["ssh-keygen", "-y", "-f", path], timeout=30)
    if rc != 0:
        raise RuntimeError(f"ssh-keygen -y failed (rc {rc}): "
                           f"{err.decode('utf-8', 'replace').strip()}")
    return out.decode("utf-8").strip()
