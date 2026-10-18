"""Generation must honor the Server park time, not silently expire after 600 s."""
import asyncio
import types
import unittest
from unittest.mock import patch

from tests.test_run_job_failover import main


class GenerationParkTime(unittest.TestCase):
    def run_wait(self, limit, release_after):
        clock = [0.0]
        candidate = ({"name": "gpu"}, {})
        failures = []
        ran = []

        def routes(alias):
            return ([candidate] if clock[0] >= release_after else [], [candidate])

        async def sleep(seconds):
            clock[0] += 601.0

        async def run_job(*args):
            ran.append(clock[0])
            return True

        with patch.object(main, "park_timeout_s", limit), \
             patch.object(main, "time", types.SimpleNamespace(monotonic=lambda: clock[0])), \
             patch.object(main, "_gen_routes", routes), \
             patch.object(main, "_gen_wait_ping"), \
             patch.object(main, "_designated_gen_index", return_value=0), \
             patch.object(main, "_run_job", run_job), \
             patch.object(main.asyncio, "sleep", sleep), \
             patch.object(main.jobs, "fail", side_effect=lambda jid, msg: failures.append(msg)):
            asyncio.run(main._run_gen_parked("probe", "image", None, None))
        return ran, failures

    def test_job_can_start_after_old_600_second_limit(self):
        ran, failures = self.run_wait(3600, 1202)
        self.assertEqual(ran, [1202])
        self.assertEqual(failures, [])

    def test_job_expires_at_configured_limit(self):
        ran, failures = self.run_wait(3600, 10000)
        self.assertEqual(ran, [])
        self.assertEqual(failures, ["park timeout: backend busy for >3600s"])

    def test_zero_disables_waiting(self):
        ran, failures = self.run_wait(0, 10000)
        self.assertEqual(ran, [])
        self.assertEqual(failures, ["park timeout: backend busy for >0s"])
