"""Service profiles of a managed host (services.py): what the gateway runs on the VM
for a ComfyUI backend and for a command backend (OpenAI-compatible: vLLM, llama-swap …),
and how. The pure half is checked as data; the command wrapper, which only ever fails
on a live instance otherwise, runs for real in `bash -c` (what sshd runs) against a
temp HOME: the lock that makes a second start a no-op, the loop that restarts a killed
service, and the stop that ends the loop AND its child.
run: venv/bin/python -m unittest tests.test_services -v"""
import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import unittest

import services
import sshrun

SECRET_START = "vllm serve org/model --port 8000 --api-key SECRET-START-TEXT"
SECRET_SETUP = "pip install vllm  # SECRET-SETUP-TEXT\n"


def _cmd(name="Qwen_vLLM 1", **kw):
    b = {"name": name, "type": "openai", "local_port": 18200, "remote_port": 8000,
         "svc_setup": SECRET_SETUP, "svc_start": SECRET_START, "svc_health": "/v1/models"}
    b.update(kw)
    return b


def _comfy(**kw):
    b = {"name": "gpu", "type": "comfyui", "local_port": 18188, "remote_port": 8188}
    b.update(kw)
    return b


class Profiles(unittest.TestCase):
    def test_profile_for(self):
        self.assertIsInstance(services.profile_for(_comfy()), services.ComfyProfile)
        self.assertIsInstance(services.profile_for(_cmd()), services.CommandProfile)
        # a backend without a type is an OpenAI one (main.backend_id's default)
        self.assertIsInstance(services.profile_for({"name": "x"}), services.CommandProfile)
        for t in ("meshy", "tripo", "anthropic"):         # nothing to run on a VM
            self.assertIsNone(services.profile_for({"name": "x", "type": t}), t)
        self.assertIsNone(services.profile_for("not a dict"))

    def test_default_ports(self):
        self.assertEqual(services.COMFY.default_port, 8188)
        self.assertEqual(services.COMMAND.default_port, 8000)

    def test_slug(self):
        self.assertEqual(services.slug("Qwen_vLLM 1"), "qwen-vllm-1")
        self.assertEqual(services.slug("--a..b--"), "a-b")
        self.assertEqual(services.slug("ÄÖÜ"), "svc")               # nothing left
        self.assertEqual(services.slug("a" * 100), "a" * 48)
        for n in ("x/../y", "-rf", ".ssh", "a b;rm -rf ~"):
            s = services.slug(n)
            self.assertRegex(s, r"^[a-z0-9][a-z0-9-]*$")
            sshrun.safe_rel(f"gw-svc-{s}.sh")                       # a safe file name

    def test_probe_rules(self):
        # R-W9: a service with its own API key still listens (401/403 = up)
        c = services.COMMAND
        self.assertEqual([s for s in (200, 401, 403, 404, 500, 0) if c.probe_ok(s)],
                         [200, 401, 403])
        self.assertEqual([s for s in (200, 401, 403, 0) if services.COMFY.probe_ok(s)], [200])
        self.assertEqual(c.probe_path(_cmd(svc_health="/health")), "/health")
        self.assertEqual(c.probe_path(_cmd(svc_health="")), "/v1/models")
        self.assertEqual(services.COMFY.probe_path(_comfy()), "/object_info")

    def test_comfy_commands_carry_the_service_port(self):
        # Ruling M4 (d): the start script takes the port; the restart ends the loop of
        # whatever port it runs on and starts one on the service's port
        c = services.COMFY
        start = c.start_cmd(_comfy(remote_port=8190))
        self.assertIn("setsid nohup ~/start-comfy.sh 8190 >/dev/null 2>&1 < /dev/null &",
                      start)
        self.assertIn("start-comfy.sh missing", start)
        r = c.restart_cmd(_comfy(remote_port=8190))
        self.assertTrue(r.endswith("; " + start))
        self.assertIn(c.stop_cmd(_comfy()), r)
        self.assertIn(services.COMFY_MAIN_PATTERN, c.stop_cmd(_comfy()))
        with self.assertRaises(ValueError):
            c.start_cmd(_comfy(remote_port=0))
        with self.assertRaises(ValueError):
            c.start_cmd(_comfy(remote_port="8188; rm -rf ~"))
        self.assertIsNone(c.wrapper_script(_comfy()))
        self.assertIsNone(c.setup_script(_comfy()))
        self.assertIsNone(c.setup_hash(_comfy()))

    def test_bootstrap_stop_pattern_is_the_profiles(self):
        # the ComfyUI bootstrap's `phase stop` and the profile's stop kill the same
        # ComfyUI — any port, since the loop runs on the service's
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "ops", "thunder-bootstrap.sh"), encoding="utf-8") as f:
            script = f.read()
        self.assertIn(f"pkill -f '{services.COMFY_MAIN_PATTERN}'", script)
        self.assertNotIn("--port 8188", script)

    def test_command_paths_and_commands(self):
        c, b = services.COMMAND, _cmd()
        self.assertEqual(c.start_cmd(b),
                         "test -x ~/gw-svc-qwen-vllm-1.sh || { echo 'gw-svc-qwen-vllm-1.sh "
                         "missing' >&2; exit 3; }; setsid nohup ~/gw-svc-qwen-vllm-1.sh "
                         ">/dev/null 2>&1 < /dev/null &")
        self.assertEqual(c.upload_cmd(b), "cat > ~/gw-svc-qwen-vllm-1.sh.tmp && chmod 700 "
                         "~/gw-svc-qwen-vllm-1.sh.tmp && mv -f ~/gw-svc-qwen-vllm-1.sh.tmp "
                         "~/gw-svc-qwen-vllm-1.sh")
        self.assertIn("~/gw-svc-qwen-vllm-1.lock", c.stop_cmd(b))
        self.assertTrue(c.restart_cmd(b).endswith("; " + c.start_cmd(b)))
        self.assertIn(c.stop_cmd(b), c.restart_cmd(b))
        self.assertEqual(c.log_path(b), "~/gw-svc-qwen-vllm-1.log")
        # the setup: stdin into bash -s, tee'd for a later tail, its own HF cache
        setup = c.setup_cmd(b)
        self.assertIn("bash -s 2>&1 | tee ~/gw-svc-qwen-vllm-1.setup.log", setup)
        self.assertIn('HF_HOME="$HOME/hf-cache-qwen-vllm-1"', setup)
        self.assertEqual(c.setup_log(b), "~/gw-svc-qwen-vllm-1.setup.log")

    def test_admin_text_never_in_a_command(self):
        # it travels on stdin only (the wrapper, the setup script) — a command line is
        # visible to every process on the VM and lands in logs
        c, b = services.COMMAND, _cmd()
        for cmd in (c.start_cmd(b), c.restart_cmd(b), c.stop_cmd(b), c.setup_cmd(b),
                    c.upload_cmd(b), c.probe_path(b)):
            self.assertNotIn("SECRET", cmd)
        for e in c.validate(_cmd(svc_start="", svc_health="/bad path SECRET")):
            self.assertNotIn("SECRET", e)

    def test_wrapper_content(self):
        w = services.COMMAND.wrapper_script(_cmd()).decode()
        self.assertTrue(w.startswith("#!/usr/bin/env bash\n"))
        self.assertIn('exec 9>>"$lock"', w)
        self.assertIn("flock -n 9 || exit 0", w)
        self.assertIn('lock="$HOME/gw-svc-qwen-vllm-1.lock"', w)
        self.assertIn('log="$HOME/gw-svc-qwen-vllm-1.log"', w)
        self.assertIn("while :; do", w)
        self.assertIn('export HF_HOME="$HOME/hf-cache-qwen-vllm-1"', w)   # R-W2
        self.assertNotIn('HF_HOME="$HOME/hf-cache"', w)
        self.assertIn(SECRET_START, w)                        # the text block, verbatim
        self.assertIn("9>&-", w)
        subprocess.run(["bash", "-n"], input=w.encode(), check=True)
        # a text that contains the delimiter line cannot end the heredoc early
        d = services._delimiter(SECRET_START + "\n")
        tricky = services.COMMAND.wrapper_script(_cmd(svc_start=f"echo a\n{d}\necho b")).decode()
        self.assertEqual(tricky.count(f"\n{d}\n"), 1)
        # CRLF from a browser textarea never reaches the shell
        crlf = services.COMMAND.wrapper_script(_cmd(svc_start="a\r\nb\r\n")).decode()
        self.assertNotIn("\r", crlf)

    def test_setup_hash(self):
        c = services.COMMAND
        h = c.setup_hash(_cmd())
        self.assertEqual(h, hashlib.sha256(SECRET_SETUP.encode()).hexdigest())
        self.assertEqual(c.setup_hash(_cmd()), h)                     # stable
        self.assertEqual(c.setup_hash(_cmd(svc_setup=SECRET_SETUP.replace("\n", "\r\n"))), h)
        self.assertEqual(c.setup_hash(_cmd(svc_setup=SECRET_SETUP.rstrip("\n"))), h)
        self.assertNotEqual(c.setup_hash(_cmd(svc_setup="pip install vllm==0.9\n")), h)
        for empty in ("", "  \n", None):
            self.assertIsNone(c.setup_hash(_cmd(svc_setup=empty)))
            self.assertIsNone(c.setup_script(_cmd(svc_setup=empty)))
        self.assertEqual(c.setup_script(_cmd()), SECRET_SETUP.encode())

    def test_validate(self):
        c = services.COMMAND
        self.assertEqual(c.validate(_cmd()), [])
        self.assertEqual(c.validate(_cmd(svc_health=None)), [])       # the default
        self.assertTrue(c.validate(_cmd(svc_start="  ")))
        self.assertTrue(c.validate(_cmd(svc_start=None)))
        for bad in ("v1/models", "/a b", "/x;rm", "/x'y", "/$(id)", "/a\nb"):
            self.assertTrue(c.validate(_cmd(svc_health=bad)), bad)
        for ok in ("/health", "/v1/models?x=1&y=2", "/a-b_c.d~e/"):
            self.assertEqual(c.validate(_cmd(svc_health=ok)), [], ok)
        self.assertTrue(c.validate(_cmd(svc_start="a\x00b")))
        self.assertTrue(c.validate(_cmd(remote_port=0)))
        self.assertEqual(services.COMFY.validate(_comfy()), [])
        self.assertTrue(services.COMFY.validate(_comfy(remote_port=None)))


def _alive(pid):
    """Alive and not a zombie (killed processes linger as zombies until reaped)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


class RealShell(unittest.TestCase):
    """The command wrapper as it runs on the VM: `cat >` + `setsid nohup`, the lock,
    the loop, the stop by the lock's pid (the whole process group)."""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="svc-vm-")
        self.addCleanup(shutil.rmtree, self.home, ignore_errors=True)
        self.env = dict(os.environ, HOME=self.home)
        # the stub service: records its pid and HF_HOME, then waits
        self.b = _cmd(name="stub", svc_start='echo "$$ $HF_HOME" >>"$HOME/starts"\n'
                                              'exec sleep 300')
        self.addCleanup(self._kill_all)

    def _kill_all(self):
        self.sh(services.COMMAND.stop_cmd(self.b))

    def sh(self, cmd, stdin=b""):
        r = subprocess.run(["bash", "-c", cmd], input=stdin, capture_output=True,
                           env=self.env, timeout=60)
        return r.returncode, r.stdout.decode(), r.stderr.decode()

    def starts(self):
        try:
            with open(os.path.join(self.home, "starts")) as f:
                return [ln.split() for ln in f.read().splitlines()]
        except FileNotFoundError:
            return []

    def wait(self, pred, secs=10.0):
        end = time.monotonic() + secs
        while not pred():
            if time.monotonic() > end:
                return False
            time.sleep(0.05)
        return True

    def test_wrapper_lock_loop_and_stop(self):
        c = services.COMMAND
        rc, _, err = self.sh(c.upload_cmd(self.b), c.wrapper_script(self.b))
        self.assertEqual(rc, 0, err)
        wrapper = os.path.join(self.home, "gw-svc-stub.sh")
        self.assertTrue(os.access(wrapper, os.X_OK))
        self.assertFalse(os.path.exists(wrapper + ".tmp"))
        # start: the service runs with its own HF cache
        rc, _, err = self.sh(c.start_cmd(self.b))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.wait(lambda: len(self.starts()) == 1), err)
        pid, hf = self.starts()[0]
        self.assertEqual(hf, os.path.join(self.home, "hf-cache-stub"))
        self.assertTrue(os.path.isdir(hf))
        with open(os.path.join(self.home, "gw-svc-stub.lock")) as f:
            wpid = int(f.read().split()[0])
        self.assertTrue(_alive(wpid))
        self.assertEqual(os.getpgid(int(pid)), wpid)          # one process group
        # a second start is a no-op: the lock is held
        self.sh(c.start_cmd(self.b))
        time.sleep(1.0)
        self.assertEqual(len(self.starts()), 1)
        # the service dies → the loop starts it again (2 s later)
        os.kill(int(pid), 9)
        self.assertTrue(self.wait(lambda: len(self.starts()) == 2), self.starts())
        pid2 = self.starts()[1][0]
        self.assertNotEqual(pid2, pid)
        # stop: loop AND child end, the lock is free again
        rc, _, err = self.sh(c.stop_cmd(self.b))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.wait(lambda: not _alive(wpid) and not _alive(int(pid2))))
        rc, _, _ = self.sh(f"flock -n {shlex.quote(self.home)}/gw-svc-stub.lock true")
        self.assertEqual(rc, 0)
        time.sleep(2.5)                                       # nothing comes back
        self.assertEqual(len(self.starts()), 2)
        # a stop with nothing running (or a stale pid in the lock) touches nothing
        rc, _, err = self.sh(c.stop_cmd(self.b))
        self.assertEqual(rc, 0, err)

    def test_restart_replaces_a_changed_wrapper(self):
        c = services.COMMAND
        self.sh(c.upload_cmd(self.b), c.wrapper_script(self.b))
        self.sh(c.start_cmd(self.b))
        self.assertTrue(self.wait(lambda: len(self.starts()) == 1))
        old = int(self.starts()[0][0])
        b2 = dict(self.b, svc_start='echo "$$ v2" >>"$HOME/starts"\nexec sleep 300')
        # the upload replaces the file the running wrapper still reads (mv, new inode)
        rc, _, err = self.sh(c.upload_cmd(b2), c.wrapper_script(b2))
        self.assertEqual(rc, 0, err)
        rc, _, err = self.sh(c.restart_cmd(b2))
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.wait(lambda: len(self.starts()) == 2), self.starts())
        self.assertEqual(self.starts()[1][1], "v2")
        self.assertTrue(self.wait(lambda: not _alive(old)))

    def test_setup_cmd_runs_the_script_from_stdin(self):
        c = services.COMMAND
        b = _cmd(name="stub", svc_setup='echo "setup in $PWD with $HF_HOME"\nexit 7\n')
        rc, out, _ = self.sh(f"bash -o pipefail -c {shlex.quote(c.setup_cmd(b))}",
                             c.setup_script(b))
        self.assertEqual(rc, 7)                               # the script's own status
        want = f"setup in {self.home} with {self.home}/hf-cache-stub"
        self.assertIn(want, out)
        with open(os.path.join(self.home, "gw-svc-stub.setup.log")) as f:
            self.assertIn(want, f.read())


if __name__ == "__main__":
    unittest.main()


class ExposureCheck(unittest.TestCase):
    """Ruling M6: `ss -ltnH` on the VM → the listeners on a service's port that are NOT
    loopback. A miss is silent — an unauthenticated vLLM on 0.0.0.0 looks exactly like
    one on 127.0.0.1 through the tunnel."""

    SS = ("LISTEN 0      4096         0.0.0.0:8000       0.0.0.0:*\n"
          "LISTEN 0      4096       127.0.0.1:8188       0.0.0.0:*    users:((\"python\",pid=9,fd=3))\n"
          "LISTEN 0      128             [::]:8000          [::]:*\n"
          "LISTEN 0      128    127.0.0.53%lo:53         0.0.0.0:*\n"
          "LISTEN 0      128            [::1]:8001          [::]:*\n"
          "LISTEN 0      128        127.0.0.1:8001       0.0.0.0:*\n"
          "LISTEN 0      128         10.1.2.3:8002       0.0.0.0:*\n"
          "LISTEN 0      128                *:8003             *:*\n"
          "LISTEN 0      128   [::ffff:127.0.0.1]:8004        [::]:*\n"
          "LISTEN 0      128   [fe80::1%eth0]:8005          [::]:*\n"
          "LISTEN 0      128            [::]:18000          [::]:*\n")

    def test_command_has_no_admin_text(self):
        self.assertEqual(services.LISTEN_CMD, "ss -ltnH")

    def test_wildcards_are_exposed(self):
        self.assertEqual(services.exposed_listeners(self.SS, 8000),
                         ["0.0.0.0:8000", "[::]:8000"])
        self.assertEqual(services.exposed_listeners(self.SS, 8003), ["*:8003"])

    def test_loopback_is_not(self):
        for port in (8188, 8001, 8004, 53):
            with self.subTest(port=port):
                self.assertEqual(services.exposed_listeners(self.SS, port), [])

    def test_a_specific_address_is_exposed(self):
        self.assertEqual(services.exposed_listeners(self.SS, 8002), ["10.1.2.3:8002"])
        self.assertEqual(services.exposed_listeners(self.SS, 8005), ["[fe80::1%eth0]:8005"])

    def test_port_must_match_exactly(self):
        self.assertEqual(services.exposed_listeners(self.SS, 800), [])
        self.assertEqual(services.exposed_listeners(self.SS, 18000), ["[::]:18000"])
        self.assertEqual(services.exposed_listeners(self.SS, 1800), [])

    def test_unreadable_input(self):
        hdr = "State Recv-Q Send-Q Local Address:Port Peer Address:Port Process\n"
        self.assertEqual(services.exposed_listeners(hdr + self.SS, 8002), ["10.1.2.3:8002"])
        self.assertEqual(services.exposed_listeners(self.SS.encode(), 8002), ["10.1.2.3:8002"])
        for bad in ("", None, "garbage\n", "LISTEN 0 1 nocolon x\n"):
            self.assertEqual(services.exposed_listeners(bad, 8000), [])
        self.assertEqual(services.exposed_listeners(self.SS, None), [])
        dup = "LISTEN 0 1 0.0.0.0:8000 0.0.0.0:*\n" * 2
        self.assertEqual(services.exposed_listeners(dup, 8000), ["0.0.0.0:8000"])

    def test_warning_text(self):
        self.assertEqual(services.exposure_warning([]), "")
        self.assertEqual(services.exposure_warning(["0.0.0.0:8000"]),
                         "listening on all interfaces (0.0.0.0:8000) — reachable from "
                         "outside the VM; bind to 127.0.0.1")
        self.assertIn("a non-loopback address (10.1.2.3:8002)",
                      services.exposure_warning(["10.1.2.3:8002"]))
