"""Backend power control (power.py + main's flow): switch a backend's device off and on
safely through a host script over ssh.

Why this file exists (it fails SILENTLY, and the failure is hardware): the console's
"Safe to switch off now" is the operator's ONLY signal to cut power to an eGPU. Cut it
while the OS still holds the device and the host's PCI hotplug hangs until a reboot —
taking the OTHER backends on that host down too (measured 2026-11-12: Xid 79 + 154,
the card never came back without a K12 reboot). Every case below ends in a state that
looks plausible on its own: a parser that reads a garbage line or ssh's own failure as
success shows the green banner; a drain that drops the backend for jobs already waiting
fails them ("no healthy backend") instead of letting them finish; a flow that disables
before the wait strands running work; a `busy` that leaves the backend disabled takes a
working GPU out of rotation without a word; a restart that re-runs `off` detaches a
device nobody asked about; a freshly enabled backend with a stale "healthy" gets work
before its service is up. No test ever reaches a host: ssh is replaced by a fake.

Run: venv/bin/python -m unittest tests.test_power_control -v
"""
import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    import admin
    import power
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

_ED_B64 = "AAAAC3NzaC1lZDI1NTE5AAAAIHm4E0tb6VPU5qn5zKm6c1tJ4HQ1Pdu6Wf7k6o7V3tJ2"
_OFF_OK = {"action": "off", "result": "ok", "exit_code": 0, "message": "detached",
           "gpu": {"present": False, "driver": None, "users": []},
           "hotplug_hung": False, "ct": {"id": 462, "status": "stopped"},
           "safe_to_power_off": True}


def _js(**kw) -> bytes:
    return (json.dumps(kw) + "\n").encode()


class ParseResult(unittest.TestCase):
    """power.parse_result — the ONE judge of a script run. Pins: only
    `safe_to_power_off == true` (on a successful `off`) ever earns "safe"; ssh's own
    exit 255 is "unreachable", not an error and never a success; garbage stdout is an
    error, never ok; the LAST JSON line counts; `hotplug_hung` wins over any exit code."""

    def test_off_ok_and_safe_is_the_only_safe_answer(self):
        p = power.parse_result("off", 0, json.dumps(_OFF_OK).encode(), b"")
        self.assertEqual((p["outcome"], p["safe"]), ("ok", True))
        self.assertEqual(power.off_state(p)[0], power.OFF)
        p = power.parse_result("off", 0, _js(**{**_OFF_OK, "result": "absent"}), b"")
        self.assertEqual((p["outcome"], p["safe"]), ("ok", True))

    def test_off_ok_without_safe_flag_is_an_error(self):
        for flag in (False, None, "true", 1):
            p = power.parse_result("off", 0, _js(**{**_OFF_OK, "safe_to_power_off": flag}), b"")
            self.assertEqual((p["outcome"], p["safe"]), ("error", False), flag)
            self.assertEqual(power.off_state(p)[0], power.ERROR)

    def test_each_exit_code(self):
        cases = [
            ("off", 1, {"result": "busy", "message": "queue not empty"}, "busy"),
            ("on", 1, {"result": "absent"}, "absent"),
            ("off", 2, {"result": "hung"}, "hung"),
            ("off", 3, None, "error"),
            ("off", 4, {"result": "error", "message": "boom"}, "error"),
            ("on", 4, {"result": "degraded", "message": "driver half up"}, "error"),
            ("status", 0, {"result": "ready", "safe_to_power_off": False}, "ok"),
            ("status", 0, {"result": "absent", "safe_to_power_off": True}, "ok"),
            ("on", 0, {"result": "ok"}, "ok"),
            ("off", 7, {"result": "weird"}, "error"),
        ]
        for action, rc, js, want in cases:
            out = _js(**js) if js else b""
            p = power.parse_result(action, rc, out, b"usage: egpu ...")
            self.assertEqual(p["outcome"], want, (action, rc, js))
            if want != "ok" or action != "status":
                self.assertFalse(p["safe"], (action, rc, js))
        p = power.parse_result("off", 3, b"", b"bad call")
        self.assertIn("AI-Hub bug", p["message"])
        p = power.parse_result("on", 1, _js(result="absent"), b"")
        self.assertEqual(power.on_state(p), (power.ERROR, power.ABSENT_ON_MSG))
        p = power.parse_result("off", 1, _js(result="busy", message="queue not empty"), b"")
        self.assertEqual(power.off_state(p), (power.BUSY, "queue not empty"))

    def test_hotplug_hung_wins_over_exit_zero(self):
        p = power.parse_result("off", 0, _js(**{**_OFF_OK, "hotplug_hung": True}), b"")
        self.assertEqual((p["outcome"], p["safe"]), ("hung", False))
        self.assertEqual(power.off_state(p)[0], power.HUNG)

    def test_garbage_stdout_is_never_ok(self):
        for out in (b"", b"hello\n", b"{broken json\n", b"[1, 2]\n", b"\xff\xfe"):
            p = power.parse_result("off", 0, out, b"")
            self.assertEqual((p["outcome"], p["safe"]), ("error", False), out)

    def test_the_last_json_line_counts(self):
        out = (json.dumps({**_OFF_OK, "safe_to_power_off": False}) + "\nnoise\n"
               + json.dumps(_OFF_OK) + "\n").encode()
        self.assertTrue(power.parse_result("off", 0, out, b"")["safe"])
        out = (json.dumps(_OFF_OK) + "\n" + json.dumps({**_OFF_OK, "safe_to_power_off": False})
               + "\n").encode()
        self.assertFalse(power.parse_result("off", 0, out, b"")["safe"])

    def test_ssh_255_is_unreachable_unless_auth_or_host_key(self):
        p = power.parse_result("status", 255, b"", b"ssh: connect to host k12 port 22: "
                                                   b"No route to host\n")
        self.assertEqual(p["outcome"], "unreachable")
        self.assertEqual(power.status_state(p, False)[0], power.UNREACHABLE)
        self.assertEqual(power.off_state(p)[0], power.ERROR)        # after a drain: unknown
        p = power.parse_result("off", 255, b"", b"Host key verification failed.\n")
        self.assertEqual(p["outcome"], "error")
        self.assertIn("host key", p["message"])
        p = power.parse_result("off", 255, b"", b"root@k12: Permission denied (publickey).\n")
        self.assertEqual(p["outcome"], "error")
        self.assertIn("authorized_keys", p["message"])

    def test_timeout_is_an_error_not_unreachable(self):
        p = power.parse_result("off", 124, b"", b"timeout")
        self.assertEqual((p["outcome"], p["safe"]), ("error", False))

    def test_authorized_keys_line(self):
        line = power.authorized_keys_line("/usr/local/sbin/egpu-4070",
                                          f"ssh-ed25519 {_ED_B64} ai-hub", "rtx 4070")
        self.assertEqual(line, 'restrict,command="/usr/local/sbin/egpu-4070 --ssh-forced" '
                               f"ssh-ed25519 {_ED_B64} ai-hub-power-rtx_4070")
        self.assertEqual(power.authorized_keys_line('/x" y', f"ssh-ed25519 {_ED_B64}", "b"), "")
        self.assertFalse(power.script_ok("/usr/bin/../x"))
        self.assertFalse(power.target_ok("k12"))                   # the user is required
        self.assertFalse(power.target_ok("-oProxyCommand=x@k12"))


class _Env(unittest.TestCase):
    """A ComfyUI backend `egpu` with power control, its key and a pinned host key in a
    temp datadir, ssh replaced by `self.ssh` (records argv, answers `self.answer`)."""

    BID = "comfyui:egpu"

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.saved = {k: getattr(main, k) for k in (
            "backends", "_gen_backends", "image_models", "_power_run", "set_backend_enabled",
            "_POWER_POLL_S", "_POWER_START_POLL_S", "_note_fault")}
        self.saved_store = main.jobs_cfg.get("store_path")
        main.jobs_cfg["store_path"] = os.path.join(self.d, "store.db")
        self.b = {"name": "egpu", "type": "comfyui", "url": "http://127.0.0.1:1",
                  "enabled": True, "power_ssh": "root@k12",
                  "power_script": "/usr/local/sbin/egpu-4070"}
        main.backends = [self.b]
        main._gen_backends = [self.b]
        main.image_models = {"pa": [{"backend": "egpu", "workflow": "w.json"}]}
        main.backend_healthy[self.BID] = True
        main.backend_inflight.pop(self.BID, None)
        main._power_states.clear()
        main._power_draining.clear()
        main._power_tasks.clear()
        main._gen_waiting.clear()
        main._POWER_POLL_S = 0.01
        main._POWER_START_POLL_S = 0.01
        key, kh = main._power_paths(self.b)
        os.makedirs(os.path.dirname(key))
        os.makedirs(os.path.dirname(kh))
        open(key, "w").close()
        with open(kh, "w") as f:
            f.write(f"k12 ssh-ed25519 {_ED_B64}\n")
        self.calls = []
        self.events = []
        self.answer = (0, json.dumps(_OFF_OK).encode(), b"")

        async def ssh(argv, stdin=None, timeout=60):
            self.calls.append((argv, timeout))
            self.events.append(("ssh", argv[-1]))
            return self.answer
        main._power_run = ssh

        def enable(bid, on):
            self.events.append(("enabled", on))
            self.b["enabled"] = bool(on)
            main._gen_backends = [x for x in main.backends if x.get("enabled", True)]
            return True
        main.set_backend_enabled = enable
        self.no_store = mock.patch.object(main.store, "is_active", return_value=False)
        self.no_store.start()

    def tearDown(self):
        self.no_store.stop()
        for k, v in self.saved.items():
            setattr(main, k, v)
        if self.saved_store is None:
            main.jobs_cfg.pop("store_path", None)
        else:
            main.jobs_cfg["store_path"] = self.saved_store
        for d in (main._power_states, main._power_draining, main._power_tasks,
                  main._power_watchers, main._power_scanned):
            d.clear()
        main._gen_waiting.clear()
        main.backend_healthy.pop(self.BID, None)
        main.backend_inflight.pop(self.BID, None)
        main.backend_error.pop(self.BID, None)

    async def settle(self, cond, timeout=3.0):
        end = time.monotonic() + timeout
        while not cond():
            if time.monotonic() > end:
                self.fail("condition not reached")
            await asyncio.sleep(0.01)


class SshArgv(_Env):
    """The remote command is EXACTLY the contract's fixed tokens, the argv ends
    `-- user@host <cmd>`, the host key is checked strictly against the per-backend pin,
    and nothing the admin typed (the script path) reaches argv — the host's forced
    command is all that stands between this key and a root shell."""

    def test_argv_per_action(self):
        key, kh = main._power_paths(self.b)
        for action in power.ACTIONS:
            self.calls.clear()
            asyncio.run(main._power_ssh(self.b, action))
            argv, timeout = self.calls[0]
            self.assertEqual(argv[-3:], ["--", "root@k12", power.COMMANDS[action]])
            self.assertEqual(timeout, power.TIMEOUTS[action])
            self.assertEqual(argv[:3], ["ssh", "-F", "/dev/null"])
            self.assertIn("StrictHostKeyChecking=yes", argv)
            self.assertIn(f"UserKnownHostsFile={kh}", argv)
            self.assertEqual(argv[argv.index("-i") + 1], key)
            self.assertNotIn("egpu-4070", " ".join(argv))
        self.assertEqual(power.COMMANDS, {"status": "status --json",
                                          "off": "off --stop-ct --json",
                                          "on": "on --start-ct --json"})

    def test_no_pin_or_a_pin_for_another_host_refuses(self):
        _key, kh = main._power_paths(self.b)
        with open(kh, "w") as f:
            f.write(f"[k12]:2222 ssh-ed25519 {_ED_B64}\n")
        msg = asyncio.run(main.power_action(self.BID, "status"))
        self.assertIn("refused", msg)
        self.assertIn("pinned for [k12]:2222", msg)
        os.remove(kh)
        self.assertIn("confirm the host key first",
                      asyncio.run(main.power_action(self.BID, "off")))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.b["enabled"], True)


class PreClickJobsKeepTheBackend(_Env):
    """"Abwarten": a generation job admitted BEFORE the switch-off keeps the backend as a
    candidate (and is counted as waiting); a job admitted after it is refused at once
    with the backend named — not parked until the park timeout, and the pre-click job is
    not failed as "no healthy backend"."""

    def test_routes_by_admission_time(self):
        before = time.monotonic()
        main._power_draining[self.BID] = time.monotonic() + 0.001
        _ready, allc = main._gen_routes("pa", admitted=before)
        self.assertEqual([backend for backend, _c in allc], [self.b])
        gated: list = []
        ready, allc = main._gen_routes("pa", gated)
        self.assertEqual((ready, allc), ([], []))
        self.assertEqual(gated, [("egpu", "egpu is being switched off")])
        with self.assertRaises(main.HTTPException) as cm:
            asyncio.run(main._gen_pick("pa", "", {}))
        self.assertEqual((cm.exception.status_code, cm.exception.detail),
                         (503, "egpu is being switched off"))

    def test_waiting_count_and_designation(self):
        t0 = time.monotonic()
        old = {"job_id": "j1", "alias": "pa", "admitted": t0 - 5, "eligible": None,
               "force": ""}
        new = {"job_id": "j2", "alias": "pa", "admitted": t0 + 5, "eligible": None,
               "force": ""}
        main._gen_waiting.extend([old, new])
        main._power_draining[self.BID] = t0
        self.assertEqual(main.power_waiting_jobs(self.BID), 1)
        self.assertTrue(main._entry_can_use(old, self.b))
        self.assertFalse(main._entry_can_use(new, self.b))

    def test_switched_off_backend_names_itself(self):
        self.b["enabled"] = False
        main._gen_backends = []
        main._power_states[self.BID] = {"state": power.OFF}
        gated: list = []
        main._gen_routes("pa", gated)
        self.assertEqual(gated, [("egpu", "egpu is switched off")])


class OffFlow(_Env):
    """The switch-off disables the backend and runs `off` only AFTER in-flight work AND
    the pre-click jobs are done; `busy` gives the backend back its pre-click `enabled`;
    a second action while one runs is refused."""

    def test_disables_only_after_the_wait(self):
        async def go():
            main.backend_inflight[self.BID] = 1
            t0 = time.monotonic()
            main._gen_waiting.append({"job_id": "j1", "alias": "pa", "admitted": t0 - 1,
                                      "eligible": None, "force": ""})
            msg = await main.power_action(self.BID, "off")
            self.assertIn("switching off", msg)
            await self.settle(lambda: main._power_states[self.BID].get("waiting") == 2)
            await asyncio.sleep(0.05)
            self.assertEqual(self.events, [])               # nothing disabled, no ssh yet
            self.assertEqual(main._power_states[self.BID]["state"], power.DRAINING)
            self.assertIn("refused", await main.power_action(self.BID, "status"))
            main.backend_inflight[self.BID] = 0
            await asyncio.sleep(0.05)
            self.assertEqual(self.events, [])               # the queued job still waits
            main._gen_waiting.clear()
            await self.settle(lambda: main._power_states[self.BID]["state"] == power.OFF)
        asyncio.run(go())
        self.assertEqual(self.events, [("enabled", False), ("ssh", power.COMMANDS["off"])])
        self.assertNotIn(self.BID, main._power_draining)
        self.assertTrue(main._power_states[self.BID]["safe"])
        self.assertFalse(self.b["enabled"])

    def test_busy_restores_the_pre_click_state(self):
        self.answer = (1, _js(result="busy", message="queue not empty",
                              ct={"id": 462, "status": "running"}), b"")

        async def go():
            await main.power_action(self.BID, "off")
            await self.settle(lambda: main._power_states[self.BID]["state"] == power.BUSY)
        asyncio.run(go())
        self.assertEqual(self.events, [("enabled", False), ("ssh", power.COMMANDS["off"]),
                                       ("enabled", True)])
        self.assertTrue(self.b["enabled"])
        # a backend that was disabled at the click stays disabled
        self.events.clear()
        main._power_states.clear()
        self.b["enabled"] = False

        async def go2():
            await main.power_action(self.BID, "off")
            await self.settle(lambda: main._power_states[self.BID]["state"] == power.BUSY)
        asyncio.run(go2())
        self.assertEqual(self.events, [("ssh", power.COMMANDS["off"])])
        self.assertFalse(self.b["enabled"])

    def test_hung_and_unreachable_keep_it_disabled(self):
        for answer, state in (((2, _js(result="hung", hotplug_hung=True), b""), power.HUNG),
                              ((255, b"", b"No route to host"), power.ERROR)):
            self.answer = answer
            self.events.clear()
            main._power_states.clear()
            self.b["enabled"] = True

            async def go():
                await main.power_action(self.BID, "off")
                await self.settle(lambda: main._power_states[self.BID]["state"] == state)
            asyncio.run(go())
            self.assertFalse(self.b["enabled"], state)
            self.assertFalse(main._power_states[self.BID]["safe"])

    def test_cancel_restores_routing(self):
        async def go():
            main.backend_inflight[self.BID] = 1
            await main.power_action(self.BID, "off")
            self.assertIn(self.BID, main._power_draining)
            self.assertIn("cancelled", await main.power_action(self.BID, "cancel"))
            await asyncio.sleep(0.05)
        asyncio.run(go())
        self.assertNotIn(self.BID, main._power_draining)
        self.assertEqual(main._power_states[self.BID]["state"], power.ON)
        self.assertEqual(self.events, [])


class OnFlowAndStarting(_Env):
    """After `on` ok the backend is marked unhealthy BEFORE it is enabled — a stale
    "healthy" from before the switch-off must not route work to a service still
    booting — and while `starting` its failing polls write no fault (F4)."""

    def test_no_work_before_first_healthy_discovery(self):
        self.b["enabled"] = False
        main._gen_backends = []
        main._power_states[self.BID] = {"state": power.OFF}
        self.answer = (0, _js(action="on", result="ok"), b"")

        async def go():
            await main.power_action(self.BID, "on")
            await self.settle(lambda: main._power_states[self.BID]["state"] == power.STARTING)
            self.assertTrue(self.b["enabled"])
            self.assertEqual(main._gen_routes("pa"), ([], []))     # enabled, not routable
            main.backend_healthy[self.BID] = True                 # the first good poll
            await self.settle(lambda: main._power_states[self.BID]["state"] == power.ON)
        asyncio.run(go())

    def test_starting_writes_no_fault(self):
        noted = []
        main._note_fault = lambda *a, **kw: noted.append(a)

        class Down:
            async def discover(self, client):
                raise RuntimeError("connection refused while booting")
        old = main.backend_adapters.get(self.BID)
        main.backend_adapters[self.BID] = Down()
        try:
            main._power_states[self.BID] = {"state": power.STARTING}
            asyncio.run(main.refresh_backend(self.b, None))
            self.assertEqual(noted, [])
            main.backend_error.pop(self.BID, None)
            main.backend_healthy[self.BID] = True
            main._power_states[self.BID] = {"state": power.ON}
            asyncio.run(main.refresh_backend(self.b, None))
            self.assertEqual(len(noted), 1)                    # outside `starting` it is one
        finally:
            if old is None:
                main.backend_adapters.pop(self.BID, None)
            else:
                main.backend_adapters[self.BID] = old


class RestartReconciliation(_Env):
    """A gateway restart in the middle of an action never re-runs it (the device may be
    anywhere in between): draining/switching_off/switching_on become `error` asking for a
    status check; a finished state (`off`) survives as it was."""

    def test_boot_marks_interrupted_actions(self):
        saved = {"comfyui:a": {"state": power.DRAINING}, "comfyui:b": {"state": power.SWITCHING_OFF},
                 "comfyui:c": {"state": power.SWITCHING_ON}, "comfyui:d": {"state": power.OFF},
                 "comfyui:e": {"state": "bogus"}}
        written = []
        with mock.patch.object(main.store, "is_active", return_value=True), \
                mock.patch.object(main.store, "get_setting", return_value=saved), \
                mock.patch.object(main.store, "set_settings", side_effect=written.append):
            main.power_boot()
        st = main._power_states
        for bid in ("comfyui:a", "comfyui:b", "comfyui:c"):
            self.assertEqual((st[bid]["state"], st[bid]["message"]),
                             (power.ERROR, power.INTERRUPTED_MSG))
        self.assertEqual(st["comfyui:d"]["state"], power.OFF)
        self.assertNotIn("comfyui:e", st)
        self.assertEqual(written[-1]["power_state"]["comfyui:b"]["state"], power.ERROR)
        self.assertEqual(self.calls, [])                        # nothing re-run


class ConsoleRow(_Env):
    """The row shows the big banners exactly in their states and offers its buttons as
    POSTs; a refusal reason (no pin) replaces the buttons instead of a dead click."""

    def _row(self, rec):
        main._power_states[self.BID] = rec
        info = main.power_info(self.b)
        return admin._power_line({"name": "egpu", "type": "comfyui", **info})

    def test_banners_and_buttons(self):
        off = self._row({"state": power.OFF, "message": "detached"})
        self.assertIn("Safe to switch off now", off)
        self.assertIn('formaction="/ui/backends/power?id=comfyui%3Aegpu&amp;do=on"', off)
        self.assertNotIn("do=off", off)
        hung = self._row({"state": power.HUNG})
        self.assertIn("Do NOT switch off", hung)
        self.assertNotIn("Safe to switch off", hung)
        self.assertNotIn("do=off", hung)
        on = self._row({"state": power.ON})
        self.assertNotIn("Safe to switch off", on)
        self.assertIn("do=off", on)
        self.assertIn("do=status", on)
        main._power_draining[self.BID] = time.monotonic()
        dr = self._row({"state": power.DRAINING, "waiting": 3, "inflight": 1, "jobs": 2})
        self.assertIn("waiting for 3 jobs", dr)
        self.assertIn("do=cancel", dr)
        self.assertNotIn("do=off", dr)
        _key, kh = main._power_paths(self.b)
        os.remove(kh)
        nopin = self._row({"state": power.ON})
        self.assertIn("confirm the host key first", nopin)
        self.assertNotIn("do=off", nopin)
        self.assertTrue(admin._is_post_action("/ui/backends/power"))


if __name__ == "__main__":
    unittest.main()
