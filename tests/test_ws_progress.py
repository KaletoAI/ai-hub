"""ComfyUI live progress: folding /ws messages into a job's progress view.

Why this file exists (it fails SILENTLY, and one way of failing is worse than no
feature at all): ComfyUI 0.30 BROADCASTS progress for every prompt on the box to every
listener — measured 2026-09-03, a listener that submitted nothing received another
job's step counter. Without the prompt_id gate a job would display a stranger's
progress, which is not a missing feature but a lying one. The ETA is the same kind of
trap: it looks plausible whatever it says. Timing the steps from job start instead of
from the first step charged the model load to step one and predicted 250 s for a job
that finished in 15 (measured 2026-09-04, before the fix pinned here).

Message shapes below are verbatim captures from ComfyUI 0.30.2.

Run: venv/bin/python -m unittest tests.test_ws_progress -v
"""
import os
import sys
import tempfile
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (tests/ is one level down)
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    from adapters import AdapterContext, ComfyUIAdapter
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

PID = "d8cb1fa6-a189-4b91-b2b9-9fb9234c6c7c"


def _adapter():
    ctx = AdapterContext(
        auth_headers=lambda b: {}, inflight_inc=lambda bid: None,
        inflight_dec=lambda bid: None, cost_usd=lambda *a: 0.0,
        source_of=lambda r: "t", record_call=lambda *a, **k: None,
        log_enabled=lambda: False)
    return ComfyUIAdapter({"name": "gpu", "type": "comfyui",
                           "url": "http://x:8188"}, ctx)


def _progress(value, mx=35, node="34", pid=PID):
    return {"type": "progress",
            "data": {"value": value, "max": mx, "node": node, "prompt_id": pid}}


class ProgressGate(unittest.TestCase):

    def setUp(self):
        self.ad = _adapter()
        self.state = {"prompt_id": PID}

    def test_another_jobs_progress_is_ignored(self):
        """0.30 broadcasts to everyone — showing a stranger's steps would be worse than
        showing none."""
        out = self.ad._progress_apply(self.state, _progress(9, pid="someone-else"), 100.0)
        self.assertIsNone(out)
        self.assertNotIn("value", self.state)

    def test_unknown_message_types_are_ignored(self):
        for msg in ({"type": "status", "data": {"prompt_id": PID}},
                    {"type": "executing", "data": {"prompt_id": PID, "node": "5"}},
                    {"type": "executed", "data": {"prompt_id": PID}}):
            self.assertIsNone(self.ad._progress_apply(self.state, msg, 1.0))

    def test_malformed_progress_is_ignored(self):
        for data in ({"value": None, "max": 35}, {"value": 3, "max": 0},
                     {"value": "x", "max": 35}, {}):
            msg = {"type": "progress", "data": {**data, "prompt_id": PID}}
            self.assertIsNone(self.ad._progress_apply(self.state, msg, 1.0))

    def test_step_and_fraction_are_reported(self):
        v = self.ad._progress_apply(self.state, _progress(25), 100.0)
        self.assertEqual((v["step"], v["steps"]), (25, 35))
        self.assertAlmostEqual(v["fraction"], 0.714, places=3)
        self.assertEqual(v["basis"], "live")
        self.assertEqual(v["node"], "34")


class ProgressEta(unittest.TestCase):

    def setUp(self):
        self.ad = _adapter()
        self.state = {"prompt_id": PID}

    def test_no_eta_before_a_second_step_has_been_seen(self):
        """One step gives no rate. A guess here is what produced the 250 s lie."""
        v = self.ad._progress_apply(self.state, _progress(1), 100.0)
        self.assertIsNone(v.get("eta_s"))

    def test_eta_uses_the_measured_seconds_per_step(self):
        self.ad._progress_apply(self.state, _progress(1), 100.0)   # first step at t=100
        v = self.ad._progress_apply(self.state, _progress(11), 200.0)  # 10 steps / 100 s
        self.assertEqual(v["eta_s"], 240)                          # 24 left * 10 s

    def test_model_load_time_is_not_charged_to_the_first_step(self):
        """The regression this pins: submit at t=0, first step only at t=100 (loading).
        Timing from the submit would call that 100 s/step and predict ~57 min."""
        self.ad._progress_apply(self.state, _progress(1), 100.0)
        v = self.ad._progress_apply(self.state, _progress(3), 101.0)   # 2 steps / 1 s
        self.assertEqual(v["eta_s"], 16)                               # 32 left * 0.5 s

    def test_a_new_node_restarts_the_measurement(self):
        """A graph can hold several samplers, and the second one's pace is its own."""
        self.ad._progress_apply(self.state, _progress(1, node="34"), 100.0)
        self.ad._progress_apply(self.state, _progress(21, node="34"), 200.0)   # 5 s/step
        self.ad._progress_apply(self.state, _progress(1, mx=10, node="99"), 300.0)
        v = self.ad._progress_apply(self.state, _progress(3, mx=10, node="99"), 302.0)
        self.assertEqual(v["eta_s"], 7)          # 1 s/step on node 99, 7 steps left
        self.assertEqual(v["node"], "99")

    def test_a_counter_going_backwards_restarts_the_measurement(self):
        self.ad._progress_apply(self.state, _progress(30), 100.0)
        self.ad._progress_apply(self.state, _progress(2), 200.0)      # a second pass
        v = self.ad._progress_apply(self.state, _progress(4), 202.0)  # 2 steps / 2 s
        self.assertEqual(v["eta_s"], 31)                              # 31 left * 1 s


class ProgressState(unittest.TestCase):

    def setUp(self):
        self.ad = _adapter()
        self.state = {"prompt_id": PID}

    def _msg(self, nodes, pid=PID):
        return {"type": "progress_state", "data": {"prompt_id": pid, "nodes": nodes}}

    def test_finished_nodes_are_counted(self):
        v = self.ad._progress_apply(self.state, self._msg({
            "22": {"value": 1.0, "max": 1.0, "state": "finished"},
            "34": {"value": 4.0, "max": 35.0, "state": "running"},
        }), 1.0)
        self.assertEqual((v["nodes_done"], v["nodes_total"]), (1, 2))

    def test_empty_node_set_is_ignored(self):
        self.assertIsNone(self.ad._progress_apply(self.state, self._msg({}), 1.0))

    def test_node_state_does_not_erase_the_step_counter(self):
        """Both message types arrive interleaved; the step count must survive."""
        self.ad._progress_apply(self.state, _progress(25), 100.0)
        v = self.ad._progress_apply(self.state, self._msg(
            {"34": {"value": 25.0, "max": 35.0, "state": "running"}}), 101.0)
        self.assertEqual(v["step"], 25)

    def test_another_jobs_node_state_is_ignored(self):
        self.assertIsNone(self.ad._progress_apply(
            self.state, self._msg({"1": {"state": "finished"}}, pid="other"), 1.0))


if __name__ == "__main__":
    unittest.main()
