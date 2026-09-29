"""System-`ssh` plumbing for the Thunder Compute backend: argv builders, the remote
path guard, a one-shot async runner, a two-process stream, the supervised tunnel and the
key generator.

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
- **the tunnel.** One ssh ControlMaster (`tunnel_argv`) carries every forward of a host;
  `control` adds/cancels one on the running master, so attaching a service never
  restarts the tunnel under another service's stream. `Supervisor` restarts it with a
  doubling backoff (reset after a stable minute), logs the stderr tail of every exit
  — a tunnel that died on a changed host key and one that died on a refused port look
  identical otherwise — and on `stop()` ends and REAPS its process: an orphaned
  `ssh -N` keeps the local port bound, and the next tunnel then exits at once on
  `ExitOnForwardFailure`.

- **the LAN stream.** `pipe` copies one process's stdout into another's stdin (the
  share's `cat` into the instance's `cat >> .part`) chunk by chunk, and ends BOTH on an
  idle timeout, on a destination that died, and on cancellation — a source left writing
  into a pipe nobody reads hangs forever, and an ssh left behind keeps its session.

The builders, `check_ctl_path`, `safe_rel`, `q` and `next_backoff` are pure (no
`main`/`adapters` imports, no I/O at import, no module-level config); `run`, `control`,
`pipe`, `Supervisor` and `keygen` spawn processes (`prepare_ctl_path` touches the
disk) and are injected into the controller. Covered by test_sshrun.py.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import signal
import stat
import time
from typing import Callable, Optional

STABLE_S = 60                   # a tunnel that lived this long resets the backoff
_STDERR_TAIL = 2048             # bytes of a dead tunnel's stderr kept for the log
_TERM_GRACE_S = 5               # terminate → kill after this many seconds
_PIPE_CHUNK = 1 << 20           # bytes per read of a LAN stream (one chunk in memory)


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


# The control socket is a Unix socket: its path must fit sun_path (104 bytes on
# BSD/macOS, 108 on Linux, incl. the NUL), and the master first binds
# "<path>.<16 random chars>" and links it into place — a path that fits by itself
# still fails at spawn time, as a tunnel that restarts forever. 104 - 17 - 1:
CTL_PATH_MAX = 86
_CONTROL_OPS = ("forward", "cancel")


def check_ctl_path(ctl_path: str) -> str:
    """The ControlMaster socket path, returned unchanged; `ValueError` otherwise. ssh
    tilde- and %-expands `-S` just like `ControlPath`, and `none` switches
    multiplexing OFF — so only an absolute path without `%` and control characters,
    within `CTL_PATH_MAX` BYTES (a non-ASCII name is longer than it looks)."""
    if not isinstance(ctl_path, str) or not ctl_path:
        raise ValueError("empty control socket path")
    if not ctl_path.startswith("/"):
        raise ValueError(f"control socket path must be absolute: {ctl_path!r}")
    if "%" in ctl_path:
        raise ValueError(f"'%' in control socket path (ssh expands it): {ctl_path!r}")
    if any(ord(c) < 32 or ord(c) == 127 for c in ctl_path):
        raise ValueError(f"control character in control socket path: {ctl_path!r}")
    n = len(ctl_path.encode("utf-8"))
    if n > CTL_PATH_MAX:
        raise ValueError(f"control socket path too long ({n} > {CTL_PATH_MAX} bytes — "
                         f"a Unix socket path is limited): {ctl_path!r}")
    return ctl_path


def _port(v) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 65535:
        raise ValueError(f"not a port: {v!r}")
    return v


def _fwd(lport, rport) -> str:
    """One `-L` spec, loopback on BOTH ends: the local side must not expose the
    instance's services to the LAN, the remote side reaches them on the VM only."""
    return f"127.0.0.1:{_port(lport)}:127.0.0.1:{_port(rport)}"


def tunnel_argv(key: str, known_hosts: str, host: str, port: int,
                forwards: list[tuple[int, int]], ctl_path: str) -> list[str]:
    """The ControlMaster tunnel: `ssh -N -M -S <ctl_path> -L … -L … -- host`, one `-L`
    per `(local, remote)` forward. Forwards added or removed later go through the
    master's socket (`control`), so attaching a service never restarts the tunnel and
    never cuts a stream running through another forward (R-W1); a master that died is
    respawned from THIS argv with every current forward.

    ControlPersist=no: the master is the Supervisor's own child and must not outlive
    it as a detached background process holding the local ports. The path goes in via
    `-S` only (checked by `check_ctl_path`). ExitOnForwardFailure makes a taken local
    port an exit (→ restart, logged) instead of a connected-looking tunnel that
    forwards nothing — which is also why one local port for two targets is refused
    here rather than left to take down every forward at once. Exact duplicates are
    dropped (a second identical `-L` would fail as "port in use")."""
    check_ctl_path(ctl_path)
    specs: list[str] = []
    seen: dict[int, int] = {}
    for lport, rport in forwards:
        spec = _fwd(lport, rport)
        if lport in seen:
            if seen[lport] != rport:
                raise ValueError(f"local port {lport} forwarded to both "
                                 f"{seen[lport]} and {rport}")
            continue
        seen[lport] = rport
        specs.append(spec)
    argv = ssh_base(key, known_hosts, port=port) + [
        "-N", "-M", "-S", ctl_path,
        "-o", "ControlPersist=no",
        "-o", "ExitOnForwardFailure=yes"]
    for spec in specs:
        argv += ["-L", spec]
    return argv + ["--", host]


def control_argv(ctl_path: str, host: str, op: str, lport: int, rport: int) -> list[str]:
    """`ssh -S <ctl_path> -O forward|cancel -L … -- host`: add or remove ONE forward on
    a running master. Only these two ops — `exit`/`stop` would end the master (every
    service's tunnel) from a call meant for one service. The host is required by
    ssh's syntax; the master's own connection is what carries the forward."""
    check_ctl_path(ctl_path)
    if op not in _CONTROL_OPS:
        raise ValueError(f"unknown control op {op!r}")
    return ["ssh", "-S", ctl_path, "-O", op, "-L", _fwd(lport, rport), "--", host]


def prepare_ctl_path(ctl_path: str) -> str:
    """Make the socket's directory (0700 — the socket grants the master's session to
    whoever can connect to it) and remove a STALE socket left there. Called right
    before a master spawns, when no master of ours is alive: one that was SIGKILLed
    leaves its socket behind, and the next master then logs "ControlSocket … already
    exists, disabling multiplexing" and runs WITHOUT one — the tunnel looks fine and
    every later `-O forward` fails. Something that is not a socket is refused, never
    deleted. → the path."""
    check_ctl_path(ctl_path)
    d = os.path.dirname(ctl_path)
    os.makedirs(d, mode=0o700, exist_ok=True)
    os.chmod(d, 0o700)
    try:
        st = os.lstat(ctl_path)
    except FileNotFoundError:
        return ctl_path
    if not stat.S_ISSOCK(st.st_mode):
        raise ValueError(f"not a control socket, left in place: {ctl_path!r}")
    os.remove(ctl_path)
    return ctl_path


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


async def control(ctl_path: str, host: str, op: str, lport: int, rport: int,
                  timeout: float = 15) -> tuple[int, str]:
    """Run `control_argv` → (rc, reason). The reason is ssh's own text (stderr, else
    stdout) — "Port forwarding failed" on a taken port is what the service's `down`
    line must say. The argv is validated BEFORE anything spawns."""
    argv = control_argv(ctl_path, host, op, lport, rport)
    rc, out, err = await run(argv, timeout=timeout)
    return rc, (err.strip() or out.strip()).decode("utf-8", "replace")


async def _kill(p) -> None:
    """SIGKILL `p` by pid and reap it (None = never spawned). Its stdin is closed too: a
    child of `p` that inherited it would otherwise wait for input forever. asyncio's
    `wait()` also waits for the process's PIPES to close, which such a child can hold
    open — so the wait is bounded; the process itself is dead and its rc is known."""
    if p is None:
        return
    _signal(p, signal.SIGKILL)
    if p.stdin is not None:
        try:
            p.stdin.close()
        except Exception:
            pass
    try:
        await asyncio.wait_for(p.wait(), _TERM_GRACE_S)
    except asyncio.TimeoutError:
        pass


async def pipe(src_argv: list[str], dst_argv: list[str], on_bytes: Callable[[int], None],
               timeout_idle: float = 120, spawn=None) -> tuple[int, int, str]:
    """Stream `src`'s stdout into `dst`'s stdin in chunks of at most `_PIPE_CHUNK` bytes,
    calling `on_bytes(n)` per chunk — the LAN transfer (source `cat` → instance
    `cat >> .part`), with nothing buffered on the gateway beyond one chunk. → (rc src,
    rc dst, stderr tail of both).

    Three ends, and every one leaves no process behind:
    - `timeout_idle` seconds without a byte moving (a read OR a write stalled): both are
      killed, rc (124, 124) — timeout(1)'s code, as in `run`;
    - the destination dies early (it cannot write the `.part`): the source is killed
      rather than left blocking on a pipe nobody reads, which would hang forever;
    - the caller is cancelled (a stop): both are killed and reaped, then the cancel
      propagates.
    Processes are ended by pid (`_signal`), never `p.kill()` — see `_signal`."""
    spawn = spawn or asyncio.create_subprocess_exec
    tails = [b"", b""]
    src = dst = None
    readers: list = []

    async def drain_err(i, p):
        # read stderr continuously: a full stderr pipe would stall ssh itself
        while True:
            chunk = await p.stderr.read(4096)
            if not chunk:
                return
            tails[i] = (tails[i] + chunk)[-_STDERR_TAIL:]

    def tail(extra: str = "") -> str:
        parts = [f"{who}: {t.decode('utf-8', 'replace').strip()}"
                 for who, t in (("source", tails[0]), ("destination", tails[1])) if t.strip()]
        return " | ".join(parts + ([extra] if extra else []))

    async def settle_readers():
        if readers:
            await asyncio.wait(readers, timeout=_TERM_GRACE_S)
            for r in readers:
                r.cancel()
            await asyncio.gather(*readers, return_exceptions=True)

    try:
        src = await spawn(*src_argv, stdin=asyncio.subprocess.DEVNULL,
                          stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        dst = await spawn(*dst_argv, stdin=asyncio.subprocess.PIPE,
                          stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        readers = [asyncio.ensure_future(drain_err(0, src)),
                   asyncio.ensure_future(drain_err(1, dst))]
        idle = broken = False
        while True:
            try:
                chunk = await asyncio.wait_for(src.stdout.read(_PIPE_CHUNK), timeout_idle)
            except asyncio.TimeoutError:
                idle = True
                break
            if not chunk:
                break
            if dst.stdin.is_closing():
                broken = True
                break
            try:
                dst.stdin.write(chunk)
                await asyncio.wait_for(dst.stdin.drain(), timeout_idle)
            except asyncio.TimeoutError:
                idle = True
                break
            except ConnectionError:         # BrokenPipe / reset: the destination is gone
                broken = True
                break
            try:
                on_bytes(len(chunk))
            except Exception:
                pass                        # a broken counter must not break the stream
        if idle:
            await _kill(src)
            await _kill(dst)
            await settle_readers()
            return 124, 124, tail(f"no data for {timeout_idle:g}s")
        if broken:
            await _kill(src)                # it would block on a pipe nobody reads
        else:
            try:
                dst.stdin.close()
            except Exception:
                pass
        rcs = []
        for p in (src, dst):
            try:
                rcs.append(await asyncio.wait_for(p.wait(), timeout_idle))
            except asyncio.TimeoutError:
                await _kill(p)
                rcs.append(124)
        await settle_readers()
        return rcs[0], rcs[1], tail()
    except BaseException:
        await _kill(src)
        await _kill(dst)
        for r in readers:
            r.cancel()
        raise


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
