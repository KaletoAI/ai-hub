"""ssh argv builders, path guard, supervisor (sshrun.py).
run: venv/bin/python -m unittest tests.test_sshrun -v"""
import asyncio
import gc
import os
import tempfile
import unittest

import sshrun


class Argv(unittest.TestCase):
    def test_tunnel_has_host_after_double_dash_and_forward_failure(self):
        a = sshrun.tunnel_argv("/k", "/kh", "ubuntu@1.2.3.4", 30022, 18188)
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
