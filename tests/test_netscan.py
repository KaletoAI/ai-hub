"""LAN scan for backends (netscan.py + the console around it).

Why this fails SILENTLY: a wrong address range scans nothing (or a whole site), a
misclassified server pre-fills a backend the gateway then cannot discover, and a
"known" match that misses re-offers every backend that is already there. None of
that raises; the panel just shows a plausible list.
"""
import asyncio
import json
import logging
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import netscan  # noqa: E402

logging.getLogger("httpx").setLevel(logging.WARNING)

IP_ADDR = """\
1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever preferred_lft forever
2: eth0    inet 192.168.8.244/24 brd 192.168.8.255 scope global eth0\\       valid_lft forever preferred_lft forever
3: enp1s0f1np1    inet 10.20.0.1/30 scope global enp1s0f1np1\\       valid_lft forever preferred_lft forever
4: big    inet 10.0.5.7/16 brd 10.0.255.255 scope global big\\       valid_lft forever preferred_lft forever
5: eth0    inet6 fe80::1/64 scope link\\       valid_lft forever preferred_lft forever
"""


class Addresses(unittest.TestCase):
    def test_own_subnets_from_ip_addr(self):
        self.assertEqual(netscan.parse_ip_addr(IP_ADDR),
                         ["192.168.8.0/24", "10.20.0.0/30", "10.0.5.0/24"])

    def test_loopback_and_ipv6_are_ignored(self):
        self.assertEqual(netscan.parse_ip_addr("1: lo inet 127.0.0.1/8 scope host lo\n"
                                               "2: e inet6 fe80::1/64 scope link\n"), [])

    def test_local_cidrs_uses_injected_runner_and_survives_failure(self):
        seen = []

        def run(argv):
            seen.append(argv)
            return IP_ADDR
        self.assertEqual(netscan.local_cidrs(run)[0], "192.168.8.0/24")
        self.assertEqual(seen[0], ["ip", "-o", "-4", "addr", "show"])

        def boom(argv):
            raise FileNotFoundError("no ip")
        self.assertEqual(netscan.local_cidrs(boom), [])

    def test_parse_ports(self):
        self.assertEqual(netscan.parse_ports("8080, 8000,x, 8080 ,0, 70000, 8188"), [8080, 8000, 8188])
        self.assertEqual(netscan.parse_ports(""), [])

    def test_expand_targets_drops_network_and_broadcast_and_caps(self):
        hosts, truncated = netscan.expand_targets(["192.168.8.0/24"])
        self.assertEqual(len(hosts), 254)
        self.assertEqual(hosts[0], "192.168.8.1")
        self.assertNotIn("192.168.8.0", hosts)
        self.assertNotIn("192.168.8.255", hosts)
        self.assertFalse(truncated)
        hosts, truncated = netscan.expand_targets(["10.0.0.0/16"], cap=100)
        self.assertEqual(len(hosts), 100)
        self.assertTrue(truncated)

    def test_expand_targets_small_prefix_and_single_host(self):
        hosts, _ = netscan.expand_targets(["10.20.0.0/30", "127.0.0.1/32", "junk", "10.20.0.0/30"])
        self.assertEqual(hosts, ["10.20.0.1", "10.20.0.2", "127.0.0.1"])


if __name__ == "__main__":
    unittest.main()
