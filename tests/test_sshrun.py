"""ssh argv builders, path guard, supervisor (sshrun.py).
run: venv/bin/python -m unittest tests.test_sshrun -v"""
import asyncio
import gc
import os
import socket
import tempfile
import unittest

import sshrun


class Argv(unittest.TestCase):
    def test_tunnel_has_host_after_double_dash_and_forward_failure(self):
        a = sshrun.tunnel_argv("/k", "/kh", "ubuntu@1.2.3.4", 30022, [(18188, 8188)],
                               "/d/ctl/x")
        self.assertEqual(a[-2:], ["--", "ubuntu@1.2.3.4"])
        self.assertIn("ExitOnForwardFailure=yes", a)
        self.assertIn("127.0.0.1:18188:127.0.0.1:8188", a)
        self.assertEqual(a[a.index("-p") + 1], "30022")
        self.assertIn("UserKnownHostsFile=/kh", a)
        self.assertIn("BatchMode=yes", a)

    def test_hostile_host_cannot_become_option(self):
        a = sshrun.exec_argv("/k", "/kh", "-oProxyCommand=evil", 22, "true")
        self.assertLess(a.index("--"), a.index("-oProxyCommand=evil"))

    def test_strict_pin(self):
        a = sshrun.ssh_base("/k", "/kh", strict="yes")
        self.assertIn("StrictHostKeyChecking=yes", a)

    def test_exec_remote_cmd_is_last_and_quoting_helper(self):
        cmd = "cat " + sshrun.q("models/a b;rm -rf ~")
        a = sshrun.exec_argv("/k", "/kh", "ubuntu@1.2.3.4", 22, cmd)
        self.assertEqual(a[-3:], ["--", "ubuntu@1.2.3.4", cmd])
        self.assertEqual(cmd, "cat 'models/a b;rm -rf ~'")

    def test_no_port_no_dash_p(self):
        self.assertNotIn("-p", sshrun.ssh_base("/k", "/kh"))


class ControlMaster(unittest.TestCase):
    """R-W1: one master connection whose forwards change without a restart."""
    H = "ubuntu@1.2.3.4"

    def test_master_shape(self):
        a = sshrun.tunnel_argv("/k", "/kh", self.H, 30022, [(18188, 8188)], "/d/ctl/x")
        base = sshrun.ssh_base("/k", "/kh", port=30022)
        self.assertEqual(a[:len(base)], base)
        self.assertEqual(a[len(base):len(base) + 9],
                         ["-N", "-M", "-S", "/d/ctl/x", "-o", "ControlPersist=no",
                          "-o", "ExitOnForwardFailure=yes", "-L"])
        # the path is given by -S only, never as a ControlPath= option (a %-pattern
        # would be expanded by ssh), and the master dies with its supervisor
        self.assertFalse(any(x.startswith("ControlPath") for x in a))
        self.assertNotIn("ControlPersist=yes", a)
        self.assertEqual(a[-2:], ["--", self.H])

    def test_every_forward_once_before_the_host(self):
        fw = [(18188, 8188), (18000, 8000), (18188, 8188), (18001, 8001)]
        a = sshrun.tunnel_argv("/k", "/kh", self.H, 22, fw, "/d/ctl/x")
        ls = [a[i + 1] for i, x in enumerate(a) if x == "-L"]
        self.assertEqual(ls, ["127.0.0.1:18188:127.0.0.1:8188",
                              "127.0.0.1:18000:127.0.0.1:8000",
                              "127.0.0.1:18001:127.0.0.1:8001"])     # dup dropped
        self.assertLess(max(i for i, x in enumerate(a) if x == "-L"), a.index("--"))

    def test_one_local_port_for_two_targets_refused(self):
        # ExitOnForwardFailure would end the whole master — every service's tunnel
        with self.assertRaises(ValueError):
            sshrun.tunnel_argv("/k", "/kh", self.H, 22, [(18188, 8188), (18188, 8000)],
                               "/d/ctl/x")

    def test_bad_ports_refused(self):
        for fw in ([(0, 8188)], [(18188, 70000)], [("x", 1)], [(True, 8188)]):
            with self.subTest(fw=fw), self.assertRaises(ValueError):
                sshrun.tunnel_argv("/k", "/kh", self.H, 22, fw, "/d/ctl/x")
        with self.assertRaises(ValueError):
            sshrun.control_argv("/d/ctl/x", self.H, "forward", 18188, 0)

    def test_no_forwards_is_a_bare_master(self):
        a = sshrun.tunnel_argv("/k", "/kh", self.H, 22, [], "/d/ctl/x")
        self.assertNotIn("-L", a)
        self.assertIn("-M", a)

    def test_control_forward_and_cancel(self):
        for op in ("forward", "cancel"):
            with self.subTest(op=op):
                self.assertEqual(
                    sshrun.control_argv("/d/ctl/x", self.H, op, 18000, 8000),
                    ["ssh", "-S", "/d/ctl/x", "-O", op,
                     "-L", "127.0.0.1:18000:127.0.0.1:8000", "--", self.H])

    def test_control_unknown_op_refused(self):
        for op in ("exit", "stop", "check", "forward;rm"):
            with self.subTest(op=op), self.assertRaises(ValueError):
                sshrun.control_argv("/d/ctl/x", self.H, op, 18000, 8000)

    def test_hostile_host_after_double_dash(self):
        a = sshrun.control_argv("/d/ctl/x", "-oProxyCommand=evil", "forward", 1, 2)
        self.assertLess(a.index("--"), a.index("-oProxyCommand=evil"))
        a = sshrun.tunnel_argv("/k", "/kh", "-oProxyCommand=evil", 22, [(1, 2)], "/d/x")
        self.assertLess(a.index("--"), a.index("-oProxyCommand=evil"))

    def test_ctl_path_with_space_is_one_element(self):
        p = "/data dir/thunder-ctl/my host"
        for a in (sshrun.tunnel_argv("/k", "/kh", self.H, 22, [(1, 2)], p),
                  sshrun.control_argv(p, self.H, "cancel", 1, 2)):
            self.assertEqual(a[a.index("-S") + 1], p)

    def test_ctl_path_length_limit(self):
        # sun_path is 104 bytes on BSD/macOS, 108 on Linux, and the master binds
        # "<path>.<16 random chars>" first — a longer path fails only at spawn time
        ok = "/" + "a" * (sshrun.CTL_PATH_MAX - 1)
        sshrun.tunnel_argv("/k", "/kh", self.H, 22, [(1, 2)], ok)
        sshrun.control_argv(ok, self.H, "forward", 1, 2)
        self.assertLessEqual(sshrun.CTL_PATH_MAX + 17 + 1, 104)
        long = ok + "b"
        with self.assertRaisesRegex(ValueError, "too long"):
            sshrun.tunnel_argv("/k", "/kh", self.H, 22, [(1, 2)], long)
        with self.assertRaisesRegex(ValueError, "too long"):
            sshrun.control_argv(long, self.H, "forward", 1, 2)
        # bytes, not characters: a non-ASCII name must not slip past the limit
        with self.assertRaisesRegex(ValueError, "too long"):
            sshrun.check_ctl_path("/" + "\u00e4" * 50)

    def test_ctl_path_shape(self):
        # ssh percent- and tilde-expands -S too; "none" disables multiplexing
        # ssh also expands ${VAR} there, and IGNORES the setting on an undefined one
        for bad in ("", "rel/x", "~/x", "/d/%h", "/d/${HOME}", "/d/$X", "none",
                    "/d/a\nb", "/d/\x7f"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sshrun.check_ctl_path(bad)
        self.assertEqual(sshrun.check_ctl_path("/d/x y"), "/d/x y")


class Control(unittest.IsolatedAsyncioTestCase):
    async def _with_run(self, result):
        calls = []

        async def fake_run(argv, stdin=None, timeout=60):
            calls.append((argv, stdin, timeout))
            return result
        orig, sshrun.run = sshrun.run, fake_run
        try:
            out = await sshrun.control("/d/ctl/x", "ubuntu@h", "forward", 18000, 8000)
        finally:
            sshrun.run = orig
        return out, calls

    async def test_control_runs_the_argv_with_timeout(self):
        out, calls = await self._with_run((0, b"", b""))
        self.assertEqual(out, (0, ""))
        argv, stdin, timeout = calls[0]
        self.assertEqual(argv, sshrun.control_argv("/d/ctl/x", "ubuntu@h", "forward",
                                                   18000, 8000))
        self.assertIsNone(stdin)
        self.assertEqual(timeout, 15)

    async def test_control_failure_carries_the_reason(self):
        out, _ = await self._with_run(
            (255, b"", b"mux_client_forward: forwarding request failed: Port forwarding failed\n"))
        self.assertEqual(out[0], 255)
        self.assertIn("Port forwarding failed", out[1])

    async def test_control_refuses_before_spawning(self):
        with self.assertRaises(ValueError):
            await sshrun.control("/d/%h", "ubuntu@h", "forward", 1, 2)


class PrepareCtl(unittest.TestCase):
    def test_creates_0700_dir_and_clears_stale_socket(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ctl", "h")
            self.assertEqual(sshrun.prepare_ctl_path(p), p)
            self.assertEqual(os.stat(os.path.dirname(p)).st_mode & 0o777, 0o700)
            # a socket left by a SIGKILLed master (bound, nobody listening): ssh would
            # say "already exists, disabling multiplexing" and run on WITHOUT one
            s = socket.socket(socket.AF_UNIX)
            s.bind(p)
            s.close()
            os.chmod(os.path.dirname(p), 0o755)
            sshrun.prepare_ctl_path(p)
            self.assertFalse(os.path.exists(p))
            self.assertEqual(os.stat(os.path.dirname(p)).st_mode & 0o777, 0o700)

    def test_keeps_a_socket_a_master_listens_on(self):
        # "stale" is measured, not assumed: a live master's socket is never removed
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ctl", "h")
            sshrun.prepare_ctl_path(p)
            srv = socket.socket(socket.AF_UNIX)
            try:
                srv.bind(p)
                srv.listen(1)
                with self.assertRaisesRegex(ValueError, "in use"):
                    sshrun.prepare_ctl_path(p)
                self.assertTrue(os.path.exists(p))
            finally:
                srv.close()
            sshrun.prepare_ctl_path(p)          # closed now → stale → removed
            self.assertFalse(os.path.exists(p))

    def test_refuses_a_non_socket_in_the_way(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ctl", "h")
            os.makedirs(os.path.dirname(p))
            with open(p, "w") as f:
                f.write("x")
            with self.assertRaises(ValueError):
                sshrun.prepare_ctl_path(p)
            self.assertTrue(os.path.exists(p))


class SafeRel(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(sshrun.safe_rel("models/vae/x.safetensors"), "models/vae/x.safetensors")

    def test_trailing_slash_directory_kept(self):
        # a directory entry of the model catalog (a whole HF repo) ends in "/"
        self.assertEqual(sshrun.safe_rel("models/microsoft/TRELLIS.2-4B/"),
                         "models/microsoft/TRELLIS.2-4B/")

    def test_rejects(self):
        for bad in ("", "/etc/passwd", "models/../../x", "hf-cache/.token", "a\x00b",
                    "a\nb", "a\\b", "models/./x", ".ssh/id"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sshrun.safe_rel(bad)

    def test_rejects_empty_segments(self):
        for bad in ("a//b", "/", "a//", "//a"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sshrun.safe_rel(bad)

    def test_rejects_del_and_leading_dash(self):
        # quoting stops the remote shell, not `cat -rf` reading an option
        for bad in ("a/\x7fb", "-rf", "--x/y"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sshrun.safe_rel(bad)
        self.assertEqual(sshrun.safe_rel("models/-x"), "models/-x")


class Backoff(unittest.TestCase):
    def test_doubles_to_max_and_resets_after_stable_run(self):
        d, seq = 2, []
        for _ in range(7):
            s, d = sshrun.next_backoff(d, 1.0, 2, 60)
            seq.append(s)
        self.assertEqual(seq, [2, 4, 8, 16, 32, 60, 60])
        s, d = sshrun.next_backoff(60, sshrun.STABLE_S, 2, 60)
        self.assertEqual((s, d), (2, 4))


class Run(unittest.IsolatedAsyncioTestCase):
    async def test_run_and_timeout(self):
        rc, out, _ = await sshrun.run(["sh", "-c", "cat"], stdin=b"hi")
        self.assertEqual((rc, out), (0, b"hi"))
        rc, _, _ = await sshrun.run(["sleep", "5"], timeout=0.2)
        self.assertEqual(rc, 124)

    async def test_supervisor_restarts_with_backoff(self):
        starts = []

        def argv():
            starts.append(1)
            return ["sh", "-c", "exit 1"]

        sup = sshrun.Supervisor(argv, log=lambda m: None, min_backoff=0.01, max_backoff=0.02)
        sup.start()
        await asyncio.sleep(0.3)
        await sup.stop()
        self.assertGreaterEqual(len(starts), 3)
        self.assertFalse(sup.running)

    async def test_supervisor_logs_stderr_tail_and_counts_restarts(self):
        logs = []
        sup = sshrun.Supervisor(lambda: ["sh", "-c", "echo boom >&2; exit 3"],
                                log=logs.append, min_backoff=0.01, max_backoff=0.02)
        sup.start()
        await asyncio.sleep(0.3)
        await sup.stop()
        self.assertGreaterEqual(sup.restarts, 2)
        self.assertTrue(any(m.startswith("tunnel exited rc=3 ") and m.endswith(": boom")
                            for m in logs), logs)

    async def test_supervisor_stop_ends_a_live_process(self):
        # a healthy tunnel never exits by itself — stop() must end it, leave no zombie
        pids = []

        async def spawn(*argv, **kw):
            p = await asyncio.create_subprocess_exec(*argv, **kw)
            pids.append(p)
            return p

        sup = sshrun.Supervisor(lambda: ["sleep", "30"], log=lambda m: None, spawn=spawn)
        sup.start()
        for _ in range(100):
            if sup.running:
                break
            await asyncio.sleep(0.01)
        self.assertTrue(sup.running)
        await sup.stop()
        self.assertFalse(sup.running)
        self.assertEqual(len(pids), 1)
        self.assertIsNotNone(pids[0].returncode)     # reaped, not a zombie
        with self.assertRaises(ProcessLookupError):
            os.kill(pids[0].pid, 0)
        await sup.stop()                             # idempotent

    async def test_supervisor_stop_before_start_is_harmless(self):
        sup = sshrun.Supervisor(lambda: ["true"], log=lambda m: None)
        await sup.stop()
        self.assertFalse(sup.running)

    async def test_supervisor_spawn_failure_backs_off_not_dies(self):
        n = []

        def argv():
            n.append(1)
            return ["/nonexistent/ssh-binary"]

        logs = []
        sup = sshrun.Supervisor(argv, log=logs.append, min_backoff=0.01, max_backoff=0.02)
        sup.start()
        await asyncio.sleep(0.2)
        await sup.stop()
        self.assertGreaterEqual(len(n), 2)
        self.assertTrue(logs)

    async def test_start_during_stop_runs_one_loop_and_orphans_nothing(self):
        # review 2026-09-27: stop() dropped its task before the old loop ended, so a
        # start() in that window reset the old loop's stop flag — two loops, and a
        # `sleep 30` that survived the final stop()
        for mid_flight in (False, True):
            with self.subTest(mid_flight=mid_flight):
                procs, alive_at_spawn = [], []

                async def spawn(*argv, **kw):
                    alive_at_spawn.append(sum(1 for p in procs if p.returncode is None))
                    p = await asyncio.create_subprocess_exec(*argv, **kw)
                    procs.append(p)
                    return p

                sup = sshrun.Supervisor(lambda: ["sleep", "30"], log=lambda m: None,
                                        spawn=spawn, min_backoff=0.01, max_backoff=0.02)
                sup.start()
                await self._until(lambda: sup.running)
                t = asyncio.ensure_future(sup.stop())
                if mid_flight:
                    await asyncio.sleep(0)          # stop() is now awaiting the loop
                sup.start()
                await t
                if mid_flight:
                    await self._until(lambda: sup.running)   # the new loop took over
                await sup.stop()
                self.assertFalse(sup.running)
                self.assertEqual(len(procs), 2 if mid_flight else 1)
                self.assertEqual(max(alive_at_spawn), 0)    # never two at once
                for p in procs:
                    self.assertIsNotNone(p.returncode)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(p.pid, 0)

    async def test_raising_log_does_not_end_supervision(self):
        n = []

        def argv():
            n.append(1)
            return ["sh", "-c", "exit 1"]

        def log(m):
            raise RuntimeError("logger down")

        sup = sshrun.Supervisor(argv, log=log, min_backoff=0.01, max_backoff=0.02)
        sup.start()
        await asyncio.sleep(0.3)
        await sup.stop()
        self.assertGreaterEqual(len(n), 3)
        self.assertFalse(sup.running)

    async def test_start_after_crashed_loop_retrieves_its_exception(self):
        class Boom(BaseException):      # escapes every `except Exception`
            pass

        ctxs, logs = [], []
        asyncio.get_running_loop().set_exception_handler(lambda l, c: ctxs.append(c))
        calls = []

        def argv():
            calls.append(1)
            if len(calls) == 1:
                raise Boom()
            return ["sleep", "30"]

        sup = sshrun.Supervisor(argv, log=logs.append, min_backoff=0.01, max_backoff=0.02)
        sup.start()
        await asyncio.sleep(0.05)       # the first loop is dead by now
        sup.start()
        await self._until(lambda: sup.running)
        await sup.stop()
        gc.collect()
        await asyncio.sleep(0)
        self.assertEqual(ctxs, [])      # no "Task exception was never retrieved"
        self.assertTrue(any("Boom" in m for m in logs), logs)

    async def _until(self, cond, timeout=2.0):
        for _ in range(int(timeout / 0.01)):
            if cond():
                return
            await asyncio.sleep(0.01)
        self.fail("condition not reached")

    # ── pipe (the LAN stream: source `cat` → instance `cat >> .part`) ──────────

    def _spawner(self, procs):
        async def spawn(*argv, **kw):
            p = await asyncio.create_subprocess_exec(*argv, **kw)
            procs.append(p)
            return p
        return spawn

    def _assert_reaped(self, procs):
        self.assertEqual(len(procs), 2)
        for p in procs:
            self.assertIsNotNone(p.returncode)
            with self.assertRaises(ProcessLookupError):
                os.kill(p.pid, 0)

    async def test_pipe_copies_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "out")
            seen = []
            rc_s, rc_d, tail = await sshrun.pipe(["sh", "-c", "printf abc"],
                                                 ["sh", "-c", f"cat >> {f}"], seen.append)
            self.assertEqual((rc_s, rc_d, tail), (0, 0, ""))
            with open(f, "rb") as fh:
                self.assertEqual(fh.read(), b"abc")
            self.assertEqual(sum(seen), 3)

    async def test_pipe_big_stream_in_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "out")
            seen = []
            rc_s, rc_d, _ = await sshrun.pipe(
                ["sh", "-c", "head -c 3000000 /dev/zero"], ["sh", "-c", f"cat > {f}"],
                seen.append)
            self.assertEqual((rc_s, rc_d), (0, 0))
            self.assertEqual(os.path.getsize(f), 3_000_000)
            self.assertEqual(sum(seen), 3_000_000)
            self.assertTrue(all(n <= 1 << 20 for n in seen))

    async def test_pipe_idle_timeout_kills_both(self):
        procs = []
        rc_s, rc_d, tail = await sshrun.pipe(["sleep", "30"], ["sh", "-c", "cat > /dev/null"],
                                             lambda n: None, timeout_idle=0.2,
                                             spawn=self._spawner(procs))
        self.assertEqual((rc_s, rc_d), (124, 124))
        self.assertIn("no data", tail)
        self._assert_reaped(procs)

    async def test_pipe_cancel_leaves_no_orphans(self):
        procs, seen = [], []
        t = asyncio.ensure_future(sshrun.pipe(
            ["sh", "-c", "printf a; exec sleep 30"], ["sh", "-c", "exec cat > /dev/null"],
            seen.append, spawn=self._spawner(procs)))
        await self._until(lambda: seen)
        t.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t
        self._assert_reaped(procs)

    async def test_pipe_destination_dying_ends_the_source(self):
        # the instance side failed (mkdir, a full disk): the source must not be left
        # writing into a pipe nobody reads
        procs = []
        rc_s, rc_d, tail = await asyncio.wait_for(sshrun.pipe(
            ["yes"], ["sh", "-c", "echo nope >&2; exit 3"], lambda n: None,
            spawn=self._spawner(procs)), 10)
        self.assertEqual(rc_d, 3)
        self.assertNotEqual(rc_s, 0)
        self.assertIn("nope", tail)
        self._assert_reaped(procs)

    async def test_pipe_source_failure_and_stderr_tail(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "out")
            rc_s, rc_d, tail = await sshrun.pipe(["sh", "-c", "echo boom >&2; exit 2"],
                                                 ["sh", "-c", f"cat >> {f}"], lambda n: None)
        self.assertEqual((rc_s, rc_d), (2, 0))
        self.assertIn("source: boom", tail)

    async def test_keygen(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "k.key")
            pub = await sshrun.keygen(p)
            self.assertTrue(pub.startswith("ssh-ed25519 "))
            self.assertEqual(os.stat(p).st_mode & 0o777, 0o600)
            self.assertEqual(await sshrun.keygen(p), pub)      # idempotent
            os.remove(p + ".pub")                              # lost .pub: derived
            self.assertEqual((await sshrun.keygen(p)).split()[:2], pub.split()[:2])


if __name__ == "__main__":
    unittest.main()
