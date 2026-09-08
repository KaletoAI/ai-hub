"""Unit tests for the execution-fault quarantine — run:
   venv/bin/python -m unittest tests.test_gen_quarantine

Why this file exists (the mechanism fails SILENTLY): a generation backend that
ANSWERS but cannot EXECUTE is invisible to every existing signal. Discovery only
calls /object_info, so `backend_healthy` stays True; the executor watchdog only sees
a stuck queue, and this backend's queue drains fine — it just turns every prompt into
an error in three seconds. Measured 2026-09-03 on comfyui-strix: a torch/ROCm update
broke `flux_time_shift`, so EVERY Flux/Krea2 model failed to load while /health kept
reporting the backend as healthy, and four consecutive user retries all landed on it
because a candidate that never SUCCEEDS never gets a gen_speed measurement and so
keeps the unmeasured "probe-once" head start forever.

The two halves pinned here are the ones that would fail without a trace:
  * a PROVEN fault (this backend failed where another succeeded) must end the probe
    and, on repetition, take the candidate out of rotation;
  * an UNPROVEN one (every candidate failed the same way = the request is at fault)
    must never quarantine anything, or one bad workflow disables a whole alias.
"""
import unittest

import scheduler


class TestExecFaultQuarantine(unittest.TestCase):

    def setUp(self):
        self.state = {}

    def test_single_fault_does_not_quarantine_but_ends_the_probe(self):
        scheduler.exec_fault_note(self.state, "a|bk", now=100.0)
        self.assertFalse(scheduler.exec_quarantined(self.state, "a|bk", 100.0))
        # ... but the candidate has had its probe: it must lose the unmeasured head
        # start, else every retry keeps picking it first (the measured 2026-09-03 loop)
        self.assertTrue(scheduler.exec_probed(self.state, "a|bk"))

    def test_second_fault_quarantines_for_the_window(self):
        scheduler.exec_fault_note(self.state, "a|bk", now=100.0)
        scheduler.exec_fault_note(self.state, "a|bk", now=200.0, quarantine_s=900)
        self.assertTrue(scheduler.exec_quarantined(self.state, "a|bk", 200.0))
        self.assertTrue(scheduler.exec_quarantined(self.state, "a|bk", 1099.0))
        self.assertFalse(scheduler.exec_quarantined(self.state, "a|bk", 1101.0))

    def test_threshold_is_configurable(self):
        scheduler.exec_fault_note(self.state, "a|bk", now=1.0, threshold=1)
        self.assertTrue(scheduler.exec_quarantined(self.state, "a|bk", 1.0))

    def test_success_clears_the_record(self):
        scheduler.exec_fault_note(self.state, "a|bk", now=100.0)
        scheduler.exec_fault_note(self.state, "a|bk", now=101.0)
        scheduler.exec_fault_clear(self.state, "a|bk")
        self.assertFalse(scheduler.exec_quarantined(self.state, "a|bk", 102.0))
        self.assertFalse(scheduler.exec_probed(self.state, "a|bk"))

    def test_quarantine_is_per_alias_and_backend(self):
        """The key is alias|backend on purpose: a bad model name quarantining a whole
        backend for every alias it serves would be worse than the fault it guards."""
        scheduler.exec_fault_note(self.state, "a|bk", now=100.0)
        scheduler.exec_fault_note(self.state, "a|bk", now=101.0)
        self.assertTrue(scheduler.exec_quarantined(self.state, "a|bk", 102.0))
        self.assertFalse(scheduler.exec_quarantined(self.state, "other|bk", 102.0))
        self.assertFalse(scheduler.exec_quarantined(self.state, "a|other", 102.0))

    def test_error_text_and_count_are_kept_for_the_console(self):
        scheduler.exec_fault_note(self.state, "a|bk", now=100.0, error="first")
        e = scheduler.exec_fault_note(self.state, "a|bk", now=101.0, error="boom")
        self.assertEqual(e["error"], "boom")
        self.assertEqual(e["fails"], 2)

    def test_expired_quarantine_keeps_the_probe_spent(self):
        """After the window the candidate is eligible again — but it must NOT get the
        unmeasured head start back, or it goes straight to the front of the queue."""
        scheduler.exec_fault_note(self.state, "a|bk", now=100.0)
        scheduler.exec_fault_note(self.state, "a|bk", now=100.0, quarantine_s=10)
        self.assertFalse(scheduler.exec_quarantined(self.state, "a|bk", 200.0))
        self.assertTrue(scheduler.exec_probed(self.state, "a|bk"))


class TestSplitQuarantined(unittest.TestCase):
    C = [({"name": "good"}, "x"), ({"name": "bad"}, "x")]

    def _key(self, bx):
        return f"a|{bx[0]['name']}"

    def _hold(self, st, name, now=1.0):
        scheduler.exec_fault_note(st, f"a|{name}", now=now)
        scheduler.exec_fault_note(st, f"a|{name}", now=now)

    def test_quarantined_candidate_is_held_back(self):
        st = {}
        self._hold(st, "bad")
        usable, held = scheduler.split_quarantined(list(self.C), self._key, st, 3.0)
        self.assertEqual([b["name"] for b, _ in usable], ["good"])
        self.assertEqual([b["name"] for b, _ in held], ["bad"])

    def test_all_quarantined_means_none_is_held(self):
        """A blocked alias is worse than a slow one: when every candidate is
        quarantined the quarantine is ignored entirely, so the job still gets a try."""
        st = {}
        self._hold(st, "good")
        self._hold(st, "bad")
        usable, held = scheduler.split_quarantined(list(self.C), self._key, st, 3.0)
        self.assertEqual([b["name"] for b, _ in usable], ["good", "bad"])
        self.assertEqual(held, [])

    def test_no_faults_leaves_the_order_untouched(self):
        usable, held = scheduler.split_quarantined(list(self.C), self._key, {}, 3.0)
        self.assertEqual([b["name"] for b, _ in usable], ["good", "bad"])
        self.assertEqual(held, [])

    def test_relative_order_of_the_survivors_is_kept(self):
        cands = [({"name": n}, "x") for n in ("a", "bad", "b", "c")]
        st = {}
        self._hold(st, "bad")
        usable, held = scheduler.split_quarantined(cands, self._key, st, 3.0)
        self.assertEqual([b["name"] for b, _ in usable], ["a", "b", "c"])
        self.assertEqual([b["name"] for b, _ in held], ["bad"])


if __name__ == "__main__":
    unittest.main()
