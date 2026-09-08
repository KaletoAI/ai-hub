"""Unit tests for scheduler.py — run: venv/bin/python test_scheduler.py"""
import unittest

import scheduler


def _e(name, age, key):
    return {"name": name, "enqueued_at": 1000.0 - age, "key": key}


class TestEma(unittest.TestCase):
    def test_first_sample_is_taken_verbatim(self):
        self.assertEqual(scheduler.ema(None, 12.0), 12.0)

    def test_ema_blends_with_alpha(self):
        self.assertAlmostEqual(scheduler.ema(10.0, 20.0, alpha=0.3), 13.0)


class TestOrderReady(unittest.TestCase):
    B = [({"name": "slow", "paid": False}, "m"),
         ({"name": "fast", "paid": False}, "m"),
         ({"name": "cloud", "paid": True}, "m"),
         ({"name": "new", "paid": False}, "m")]

    def _speed(self, b, x):
        return {"slow": 5.0, "fast": 50.0, "cloud": 500.0,
                "new": float("inf")}[b["name"]]

    def test_unpaid_beats_paid_and_speed_orders_within_tier(self):
        got = scheduler.order_ready(list(self.B), self._speed, lambda b: b["paid"])
        self.assertEqual([b["name"] for b, _ in got],
                         ["new", "fast", "slow", "cloud"])

    def test_stable_for_equal_keys(self):
        pair = [({"name": "a", "paid": False}, 1), ({"name": "b", "paid": False}, 2)]
        got = scheduler.order_ready(pair, lambda b, x: 1.0, lambda b: False)
        self.assertEqual([b["name"] for b, _ in got], ["a", "b"])


class TestDesignatedTaker(unittest.TestCase):
    def _pick(self, pool, last_key, now=1000.0, max_wait=120.0,
              unservable=()):
        return scheduler.designated_taker(
            pool,
            can_serve=lambda e: e["name"] not in unservable,
            type_key=lambda e: e["key"],
            last_key=last_key, now=now, max_wait_s=max_wait)

    def test_empty_pool_returns_none(self):
        self.assertIsNone(self._pick([], "x"))

    def test_oldest_wins_without_affinity(self):
        pool = [_e("old", 50, "a"), _e("young", 10, "b")]
        self.assertEqual(self._pick(pool, last_key=None)["name"], "old")

    def test_same_type_beats_older_other_type(self):
        pool = [_e("old-a", 50, "a"), _e("young-b", 10, "b")]
        self.assertEqual(self._pick(pool, last_key="b")["name"], "young-b")

    def test_overdue_beats_affinity(self):
        pool = [_e("overdue-a", 200, "a"), _e("young-b", 10, "b")]
        self.assertEqual(self._pick(pool, last_key="b")["name"], "overdue-a")

    def test_oldest_overdue_wins_among_overdue(self):
        pool = [_e("older", 300, "a"), _e("newer", 200, "b")]
        self.assertEqual(self._pick(pool, last_key="b")["name"], "older")

    def test_unservable_entries_are_skipped(self):
        pool = [_e("cant", 300, "a"), _e("can", 10, "b")]
        self.assertEqual(self._pick(pool, last_key=None,
                                    unservable={"cant"})["name"], "can")

    def test_all_unservable_returns_none(self):
        pool = [_e("cant", 300, "a")]
        self.assertIsNone(self._pick(pool, last_key="a", unservable={"cant"}))


class TestFreeVramBeforeJob(unittest.TestCase):
    """The decision fails SILENTLY in both directions: freeing too eagerly only
    costs a model reload nobody attributes to it, and not freeing at all surfaces
    as a CUDA OOM inside a node — never as a gateway error."""

    def test_same_key_keeps_the_cache(self):
        self.assertFalse(scheduler.free_vram_before_job("mesh-mia", "mesh-mia", 0))

    def test_changed_key_frees(self):
        self.assertTrue(scheduler.free_vram_before_job("Trellis2", "mesh-mia", 0))

    def test_unknown_last_key_frees(self):
        # a gateway restart empties backend_last_key, never ComfyUI's VRAM
        self.assertTrue(scheduler.free_vram_before_job(None, "mesh-mia", 0))

    def test_never_while_another_job_runs_there(self):
        self.assertFalse(scheduler.free_vram_before_job(None, "mesh-mia", 1))

    def test_host_policy_off_wins(self):
        self.assertFalse(scheduler.free_vram_before_job(None, "x", 0, enabled=False))


class TestModelSetKey(unittest.TestCase):
    """The VRAM key: a wrong one fails silently in both directions — a reload on every
    job (looks like "generation got slower") or a free that never comes (the OOM the
    free-before-job mechanism exists for)."""

    LOAD = {"class_type": "UNETLoader",
            "inputs": {"unet_name": "a.safetensors", "weight_dtype": "default"}}

    def test_the_same_weights_under_different_ids_and_layout_share_a_key(self):
        a = {"1": self.LOAD, "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "steps": 20}}}
        b = {"7": dict(self.LOAD), "9": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "steps": 40}}}
        self.assertEqual(scheduler.model_set_key(a), scheduler.model_set_key(b))

    def test_a_different_weight_file_is_a_different_key(self):
        a = {"1": self.LOAD}
        b = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "b.safetensors"}}}
        self.assertNotEqual(scheduler.model_set_key(a), scheduler.model_set_key(b))

    def test_how_it_loads_does_not_count(self):
        """trellis2 high vs low: same modelname, different low_vram/keep_models_loaded."""
        a = {"1": {"class_type": "Trellis2LoadModel",
                   "inputs": {"modelname": "microsoft/TRELLIS.2-4B", "low_vram": False, "device": "cuda"}}}
        b = {"1": {"class_type": "Trellis2LoadModel",
                   "inputs": {"modelname": "microsoft/TRELLIS.2-4B", "low_vram": True, "device": "cuda"}}}
        self.assertEqual(scheduler.model_set_key(a), scheduler.model_set_key(b))

    def test_input_loaders_and_links_are_ignored(self):
        a = {"1": self.LOAD, "2": {"class_type": "LoadImage", "inputs": {"image": "gw_job1_x.png"}}}
        b = {"1": self.LOAD, "2": {"class_type": "LoadImage", "inputs": {"image": "gw_job2_x.png"}}}
        self.assertEqual(scheduler.model_set_key(a), scheduler.model_set_key(b))

    def test_a_lora_slot_counts(self):
        a = {"1": self.LOAD, "2": {"class_type": "Lora Loader Stack (rgthree)",
                                   "inputs": {"lora_01": "None", "strength_01": 1}}}
        b = {"1": self.LOAD, "2": {"class_type": "Lora Loader Stack (rgthree)",
                                   "inputs": {"lora_01": "style.safetensors", "strength_01": 1}}}
        self.assertNotEqual(scheduler.model_set_key(a), scheduler.model_set_key(b))

    def test_a_bypassed_loader_loads_nothing(self):
        a = {"1": self.LOAD, "2": {"class_type": "LoaderGGUF", "inputs": {"gguf_name": "x.gguf"}}}
        b = {"1": self.LOAD}
        self.assertEqual(scheduler.model_set_key(a, skip_ids=["2"]), scheduler.model_set_key(b))

    def test_no_loader_means_no_key(self):
        self.assertIsNone(scheduler.model_set_key({"1": {"class_type": "KSampler", "inputs": {}}}))
        self.assertIsNone(scheduler.model_set_key({}))

    def test_the_sample_trellis2_high_and_low_share_a_key(self):
        import json, os
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_comfyui_workflows")
        def load(name):
            with open(os.path.join(d, name)) as f:
                return json.load(f)
        hi, lo, hy = (load(f"img2mesh-{n}_api.json") for n in ("trellis2_high", "trellis2_low", "hunyuan3d"))
        self.assertEqual(scheduler.model_set_key(hi), scheduler.model_set_key(lo))
        self.assertNotEqual(scheduler.model_set_key(hi), scheduler.model_set_key(hy))


class TestHostFlags(unittest.TestCase):
    def test_defaults_in_one_place(self):
        self.assertTrue(scheduler.host_flag_default("comfy_free_before_job", shared=False))
        self.assertFalse(scheduler.host_flag_default("comfy_free_after_job", shared=False))
        self.assertTrue(scheduler.host_flag_default("comfy_free_after_job", shared=True))
        self.assertFalse(scheduler.host_flag_default("llm_unload_before_media", shared=True))

    def test_a_stored_value_wins_over_the_default(self):
        self.assertFalse(scheduler.host_flag({"comfy_free_before_job": False}, "comfy_free_before_job", True))
        self.assertTrue(scheduler.host_flag({}, "comfy_free_before_job", False))
        self.assertTrue(scheduler.host_flag(None, "avoid_llm_during_media", False))


if __name__ == "__main__":
    unittest.main()
