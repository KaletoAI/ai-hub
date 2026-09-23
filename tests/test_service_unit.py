"""The systemd unit's sandbox — hardened, but not against its own features.

Why this fails SILENTLY: `ai-hub.service` runs as root (the voice-reference ship uses
root's ssh key, /opt/ai-hub is root-owned), so the unit carries a sandbox that works for
root (review 2026-09-23, S22). The trap is the next well-meant tightening: ProtectHome
hides /root/.ssh, and every voice ship then fails as a mere "scp failed" line in the
Voice tab; ProtectSystem=strict makes the DBs, jobs/ and voiceref/ read-only unless each
is listed; dropping AF_NETLINK breaks `ip addr` behind Scan network, which then reports
"no address range". The service starts fine in all three cases. deploy.sh installs a
changed unit on the next deploy, so this pins both halves: the hardening is there, and
the settings that would break a feature are not.
"""
import os
import unittest

_UNIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ai-hub.service")


def _service_section() -> dict:
    out, sect = {}, None
    with open(_UNIT) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                sect = line
            elif sect == "[Service]" and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


class ServiceUnit(unittest.TestCase):
    def test_hardening_is_present(self):
        s = _service_section()
        for k, v in (("NoNewPrivileges", "yes"), ("PrivateTmp", "yes"), ("ProtectSystem", "full"),
                     ("RestrictSUIDSGID", "yes"), ("ProtectKernelModules", "yes"),
                     ("ProtectControlGroups", "yes")):
            self.assertEqual(s.get(k), v, k)

    def test_nothing_that_breaks_a_feature(self):
        s = _service_section()
        self.assertNotIn("ProtectHome", s)                       # ~/.ssh for the voice ship
        self.assertNotEqual(s.get("ProtectSystem"), "strict")    # DBs / jobs / voiceref writes
        self.assertIn("AF_NETLINK", s.get("RestrictAddressFamilies", "AF_NETLINK"))   # ip addr
        self.assertNotIn("PrivateNetwork", s)
        self.assertEqual(s.get("WorkingDirectory"), "/opt/ai-hub")


if __name__ == "__main__":
    unittest.main()
