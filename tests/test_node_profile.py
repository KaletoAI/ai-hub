"""Whole-workflow progress from recorded node timings (nodeprofile + the ComfyUI feed).

Why this file exists (it fails SILENTLY — every wrong answer here is a plausible-
looking bar): the live feed only knows the step counter of the node that is sampling,
so a mesh workflow read "35/35, 100 %" while decode/remesh/texture still had minutes to
go. Each run now records seconds per node (`meta.node_times`) and the next run of the
alias reads its live state against them. The traps pinned here:
  - a node ComfyUI serves from its cache must leave the plan, or a warm rerun stalls
    at the share the loaders used to take;
  - a profile from ANOTHER backend gives shares only — its seconds are another GPU's,
    so it must not produce an ETA before this run has measured its own pace;
  - the overall ETA replaces the node's (a sampler's "8 s left" is not the job's);
  - time spent in an expanded node's internal children belongs to the workflow node
    (`display_node`), or the record names nodes the workflow does not have;
  - the bar never goes backwards, and moves on a clock while a node sends nothing;
  - a finished node keeps its step scale, a multi-phase counter is not the node's
    progress, a lazy node's first visit is not "done", a dropped feed records nothing,
    and messages that beat the submit's answer are replayed through the gate;
  - only DONE jobs on the right alias(+backend) feed a profile.

Run: venv/bin/python -m unittest tests.test_node_profile -v
"""
import os
import sys
import tempfile
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
import nodeprofile  # noqa: E402
from tests.test_ws_progress import PID, _adapter, _progress  # noqa: E402


def _ex(node, pid=PID, display=None):
    d = {"node": node, "prompt_id": pid}
    if display:
        d["display_node"] = display
    return {"type": "executing", "data": d}


def _cached(nodes, pid=PID):
    return {"type": "execution_cached", "data": {"nodes": nodes, "prompt_id": pid}}


def _rec(*nodes):
    return {"v": 1, "nodes": [{"id": n, "title": n, "s": s, **({"steps": st} if st else {})}
                              for n, s, st in nodes], "cached": [], "total_s": 0}


PROFILE = nodeprofile.build([_rec(("4", 10.0, None), ("34", 20.0, 20), ("8", 10.0, None),
                                  ("50", 60.0, None))])          # total 100 s


class Record(unittest.TestCase):

    def test_times_follow_the_executing_sequence(self):
        run = {"submit_at": 0.0}
        for kind, data, t in (("execution_start", {}, 2.0),
                              ("executing", {"node": "4"}, 2.0),
                              ("executing", {"node": "34"}, 7.0),
                              ("executing", {"node": None}, 27.0)):
            nodeprofile.fold(run, kind, data, t)
        rec = nodeprofile.record(run, {"4": "Load", "34": "KSampler"}, 30.0)
        self.assertEqual([(n["id"], n["title"], n["s"]) for n in rec["nodes"]],
                         [("4", "Load", 5.0), ("34", "KSampler", 20.0)])
        self.assertEqual((rec["total_s"], rec["wait_s"]), (25.0, 2.0))

    def test_expanded_children_are_booked_on_the_display_node(self):
        run = {}
        nodeprofile.fold(run, "executing", {"node": "50.0.1", "display_node": "50"}, 0.0)
        nodeprofile.fold(run, "executing", {"node": "50.0.2", "display_node": "50"}, 3.0)
        nodeprofile.fold(run, "execution_success", {}, 9.0)
        rec = nodeprofile.record(run, {}, 9.0)
        self.assertEqual([(n["id"], n["s"]) for n in rec["nodes"]], [("50", 9.0)])

    def test_last_node_is_closed_when_the_poll_wins_the_race(self):
        """/history can report the prompt done before the socket's end marker arrives."""
        run = {}
        nodeprofile.fold(run, "executing", {"node": "9"}, 1.0)
        rec = nodeprofile.record(run, {}, 4.0)
        self.assertEqual(rec["nodes"][0]["s"], 3.0)

    def test_nothing_measured_records_nothing(self):
        self.assertIsNone(nodeprofile.record({}, {}, 1.0))


class Build(unittest.TestCase):

    def test_median_per_node_over_runs_it_executed_in(self):
        p = nodeprofile.build([_rec(("4", 30.0, None), ("34", 10.0, 20)),
                               _rec(("34", 14.0, 20)),               # loader cached here
                               _rec(("4", 32.0, None), ("34", 12.0, 20))])
        self.assertEqual(p["nodes"]["4"]["s"], 31.0)                  # cold runs only
        self.assertEqual(p["nodes"]["34"]["s"], 12.0)
        self.assertEqual((p["runs"], p["foreign"]), (3, False))

    def test_unknown_record_versions_are_skipped(self):
        self.assertIsNone(nodeprofile.build([{"v": 99, "nodes": [{"id": "1", "s": 5}]}]))


class Overall(unittest.TestCase):
    PLAN = ["4", "34", "8", "50", "99"]          # 99: not in the profile → weight 0

    def test_finished_nodes_and_running_share(self):
        run = {"done": {"4": 10.0, "34": 20.0}, "cur": "8", "cur_at": 100.0}
        ov = nodeprofile.overall(PROFILE, run, self.PLAN, 105.0)
        self.assertAlmostEqual(ov["fraction"], 0.35)        # (10+20+10*0.5)/100
        self.assertEqual(ov["eta_s"], 65)                   # 5 left on 8 + 60 on 50

    def test_cached_nodes_leave_the_plan(self):
        """Warm rerun: the loader is served from cache — 4/4 done must not read 90 %."""
        run = {"cached": {"4"}, "done": {"34": 20.0, "8": 10.0}, "cur": "50", "cur_at": 0.0}
        ov = nodeprofile.overall(PROFILE, run, self.PLAN, 0.0)
        self.assertAlmostEqual(ov["fraction"], 30 / 90, places=3)

    def test_nodes_missing_from_this_workflow_leave_the_plan(self):
        run = {"done": {"4": 10.0}, "cur": "34", "cur_at": 0.0}
        ov = nodeprofile.overall(PROFILE, run, ["4", "34"], 0.0)   # 8 and 50 were removed
        self.assertAlmostEqual(ov["fraction"], 10 / 30, places=3)

    def test_sampler_counter_drives_the_running_node_and_its_eta(self):
        run = {"done": {"4": 10.0}, "cur": "34", "cur_at": 0.0}
        ov = nodeprofile.overall(PROFILE, run, self.PLAN, 50.0, step=("34", 10, 20, 8))
        self.assertAlmostEqual(ov["fraction"], 0.2)          # (10 + 20*0.5)/100
        self.assertEqual(ov["eta_s"], 8 + 70)                # node's own eta + 8 & 50

    def test_more_steps_than_recorded_lengthen_the_node(self):
        run = {"done": {"4": 10.0}, "cur": "34", "cur_at": 0.0}
        ov = nodeprofile.overall(PROFILE, run, self.PLAN, 0.0, step=("34", 0, 40, None))
        self.assertEqual(ov["expected_s"], 120.0)            # 20 s at 20 steps → 40 s

    def test_this_runs_pace_scales_the_rest(self):
        run = {"done": {"4": 20.0, "34": 40.0}, "cur": "8", "cur_at": 0.0}  # twice as slow
        ov = nodeprofile.overall(PROFILE, run, self.PLAN, 0.0)
        self.assertEqual(ov["eta_s"], 20 + 120)

    def test_a_node_past_its_expected_time_never_reads_done(self):
        run = {"done": {"4": 10.0, "34": 20.0, "8": 10.0}, "cur": "50", "cur_at": 0.0}
        ov = nodeprofile.overall(PROFILE, run, self.PLAN, 600.0)
        self.assertLess(ov["fraction"], 0.99)
        self.assertEqual(ov["eta_s"], 0)

    def test_foreign_profile_gives_no_eta_without_evidence(self):
        prof = dict(PROFILE, foreign=True)
        ov = nodeprofile.overall(prof, {"cur": "4", "cur_at": 0.0}, self.PLAN, 3.0)
        self.assertIsNotNone(ov["fraction"])
        self.assertNotIn("eta_s", ov)
        run = {"done": {"4": 5.0, "34": 10.0}, "cur": "8", "cur_at": 0.0}   # twice as fast
        self.assertEqual(nodeprofile.overall(prof, run, self.PLAN, 0.0)["eta_s"], 35)

    def test_profile_covering_no_planned_node_gives_nothing(self):
        self.assertIsNone(nodeprofile.overall(PROFILE, {}, ["77"], 0.0))


class ReviewFindings(unittest.TestCase):
    """Cases a review found (2026-11-12): each read plausibly and was wrong."""

    def test_finished_node_keeps_its_step_scale(self):
        """A 40-step sampler over a 20-step profile, once finished, must not read as
        'twice as slow' and double the remaining ETA."""
        prof = nodeprofile.build([_rec(("3", 20.0, 20), ("8", 20.0, None))])
        run = {"done": {"3": 40.0}, "steps": {"3": 40}, "cur": "8", "cur_at": 0.0}
        ov = nodeprofile.overall(prof, run, ["3", "8"], 0.0)
        self.assertEqual(ov["eta_s"], 20)
        self.assertAlmostEqual(ov["fraction"], 40 / 60, places=3)

    def test_multi_phase_counter_is_not_the_nodes_progress(self):
        """A node that counts several phases: phase 1 at 50/50 is not the node done,
        and its last phase's max is no step count to scale by."""
        run = {}
        nodeprofile.fold(run, "executing", {"node": "7"}, 0.0)
        for v in range(1, 51):
            nodeprofile.note_steps(run, "7", v, 50)
        for v in range(1, 1001, 100):
            nodeprofile.note_steps(run, "7", v, 1000)
        rec = nodeprofile.record(run, {}, 100.0)
        self.assertEqual(rec["nodes"][0].get("phases"), 2)
        self.assertNotIn("steps", rec["nodes"][0])
        prof = nodeprofile.build([rec])
        run = {"cur": "7", "cur_at": 0.0}
        nodeprofile.note_steps(run, "7", 50, 50)
        ov = nodeprofile.overall(prof, run, ["7"], 10.0, step=("7", 50, 50, 0))
        self.assertAlmostEqual(ov["fraction"], 0.1)      # elapsed/expected, not 50/50
        self.assertEqual(ov["eta_s"], 90)

    def test_lazy_first_visit_is_not_done(self):
        """Lazy node A: `executing` A (0.1 s), its input B, then A again."""
        prof = nodeprofile.build([_rec(("A", 30.0, None), ("B", 5.0, None), ("C", 30.0, None))])
        run = {}
        nodeprofile.fold(run, "executing", {"node": "A"}, 0.0)
        nodeprofile.fold(run, "executing", {"node": "B"}, 0.1)
        ov = nodeprofile.overall(prof, run, ["A", "B", "C"], 2.6)
        self.assertLess(ov["fraction"], 0.1)              # not A's 30 s share
        self.assertEqual(ov["eta_s"], 2 + 60)             # pace untouched by A's 0.1 s
        nodeprofile.fold(run, "executing", {"node": "A"}, 5.1)
        nodeprofile.fold(run, "executing", {"node": "C"}, 35.0)
        self.assertGreater(run["done"]["A"], 29.0)        # both visits booked on A

    def test_broken_feed_records_nothing(self):
        run = {"broken": True}
        nodeprofile.fold(run, "executing", {"node": "4"}, 0.0)
        self.assertIsNone(nodeprofile.record(run, {}, 9.0))


class AdapterFeed(unittest.TestCase):
    """The feed folding in ComfyUIAdapter: profile view, prompt gate, monotonic bar."""

    def setUp(self):
        self.ad = _adapter()
        self.state = {"prompt_id": PID, "run": {}, "plan": ["4", "34", "8", "50"],
                      "titles": {"34": "KSampler"}, "profile": PROFILE}

    def test_overall_replaces_the_node_eta(self):
        self.ad._progress_apply(self.state, _ex("4"), 0.0)
        self.ad._progress_apply(self.state, _ex("34"), 10.0)
        self.ad._progress_apply(self.state, _progress(1, mx=20), 11.0)
        v = self.ad._progress_apply(self.state, _progress(11, mx=20), 21.0)   # 1 s/step
        self.assertEqual(v["basis"], "profile")
        self.assertEqual((v["step"], v["steps"], v["node_eta_s"]), (11, 20, 9))
        self.assertEqual(v["eta_s"], 9 + 70)
        self.assertEqual(v["node_title"], "KSampler")
        self.assertAlmostEqual(v["fraction"], 0.21)                   # (10 + 20*0.55)/100

    def test_another_prompts_executing_is_ignored(self):
        self.assertIsNone(self.ad._progress_apply(self.state, _ex("4", pid="other"), 0.0))
        self.assertNotIn("cur", self.state["run"])

    def test_cached_message_takes_nodes_out(self):
        self.ad._progress_apply(self.state, _cached(["4"]), 0.0)
        v = self.ad._progress_apply(self.state, _ex("34"), 0.0)
        self.assertEqual(v["expected_s"], 90.0)

    def test_bar_moves_on_the_clock_and_never_backwards(self):
        for n, t in (("4", 0.0), ("34", 10.0), ("8", 30.0), ("50", 40.0)):
            self.ad._progress_apply(self.state, _ex(n), t)
        a = self.ad._progress_tick(self.state, 70.0)["fraction"]
        b = self.ad._progress_tick(self.state, 85.0)["fraction"]
        self.assertGreater(b, a)
        self.state["profile"] = dict(PROFILE, nodes={**PROFILE["nodes"],
                                                    "50": {"s": 600.0}})  # would drop
        self.assertGreaterEqual(self.ad._progress_tick(self.state, 86.0)["fraction"], b)

    def test_messages_before_the_prompt_id_are_replayed_through_the_gate(self):
        """The socket connects before the submit; the loader's `executing` can arrive
        before /prompt answers — buffered, then replayed with its own timestamp."""
        self.state["prompt_id"] = None
        self.state["early"] = [(0.0, _ex("4")), (0.0, _ex("1", pid="stranger")),
                               (10.0, _ex("34"))]
        self.assertIsNone(self.ad._drain_early(self.state))    # id still unknown
        self.state["prompt_id"] = PID
        v = self.ad._drain_early(self.state)
        self.assertEqual(self.state["run"]["done"], {"4": 10.0})
        self.assertEqual(v["node"], "34")
        self.assertNotIn("early", self.state)

    def test_quiet_seconds_report_the_last_real_message(self):
        """The clock tick re-emits every second — a hung node must still look stale."""
        self.ad._progress_apply(self.state, _ex("50"), 40.0)
        self.assertEqual(self.ad._progress_tick(self.state, 100.0)["quiet_s"], 60)

    def test_no_profile_keeps_the_node_feed(self):
        self.state["profile"] = None
        self.assertIsNone(self.ad._progress_apply(self.state, _ex("34"), 0.0))
        self.assertIsNone(self.ad._progress_tick(self.state, 5.0))
        v = self.ad._progress_apply(self.state, _progress(5, mx=20), 6.0)
        self.assertEqual(v["basis"], "live")


class Records(unittest.TestCase):
    """jobs.node_time_records: only done jobs of the alias (+backend) feed a profile."""

    def setUp(self):
        import jobs
        self.jobs = jobs
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = (jobs._DB_PATH, jobs._BLOB_DIR, jobs._DEFAULT_TTL, jobs._active)
        jobs.init(os.path.join(self._tmp.name, "jobs.db"), os.path.join(self._tmp.name, "blobs"))

    def tearDown(self):
        j = self.jobs
        j._DB_PATH, j._BLOB_DIR, j._DEFAULT_TTL, j._active = self._saved
        self._tmp.cleanup()

    def _job(self, alias, backend, s, done=True):
        jid = self.jobs.create("text2img", alias, backend)
        self.jobs.set_status(jid, "running")
        if done:
            self.jobs.complete(jid, [], {"backend": backend, "node_times": _rec(("1", s, None))})
        else:
            self.jobs.merge_meta(jid, {"node_times": _rec(("1", s, None))})
            self.jobs.fail(jid, "x")

    def test_done_jobs_of_alias_and_backend_only(self):
        self._job("a", "gpu1", 10.0)
        self._job("a", "gpu2", 20.0)
        self._job("b", "gpu1", 30.0)
        self._job("a", "gpu1", 40.0, done=False)
        got = [r["nodes"][0]["s"] for r in self.jobs.node_time_records("a", "gpu1")]
        self.assertEqual(got, [10.0])
        got = sorted(r["nodes"][0]["s"] for r in self.jobs.node_time_records("a"))
        self.assertEqual(got, [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
