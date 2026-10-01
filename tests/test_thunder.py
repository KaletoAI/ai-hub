"""Pure Thunder-Compute helpers (thunder.py).
run: venv/bin/python -m unittest tests.test_thunder -v"""
import unittest

import thunder

PRICING = {"a6000_x1": 0.35, "l40_x1": 0.79, "a100xl_x1": 1.09, "additional_vcpus": 0.04,
           "disk_gb": 0.0003, "snapshot_gb": 0.00006849}
SPECS = {"specs": {"a6000_x1": {"vcpuOptions": [6, 8], "storageGB": {"min": 100, "max": 500}},
                   "l40_x1": {"vcpuOptions": [6, 8, 12], "storageGB": {"min": 100, "max": 500}}}}
GB = 1024 ** 3


class Instances(unittest.TestCase):
    def test_map_shape(self):
        items = thunder.parse_instances({"0": {"uuid": "u1", "status": "running", "ip": "1.2.3.4",
                                               "port": 30022, "httpPorts": [8188],
                                               "cpuCores": "8", "numGpus": "1"}})
        self.assertEqual(items[0]["index"], "0")
        self.assertEqual(items[0]["status"], "RUNNING")
        self.assertEqual(items[0]["port"], 30022)
        self.assertEqual(items[0]["http_ports"], [8188])
        # what prices a foreign instance (OpenAPI InstanceListItem: counts are strings)
        self.assertEqual((items[0]["gpu_type"], items[0]["num_gpus"], items[0]["cpu_cores"]),
                         ("", 1, 8))
        it = thunder.parse_instances([{"id": "1", "gpuType": "a6000", "numGpus": "2",
                                       "cpuCores": "12", "storage": 300}])[0]
        self.assertEqual((it["gpu_type"], it["num_gpus"], it["cpu_cores"], it["storage"]),
                         ("a6000", 2, 12, 300))

    def test_list_shape_and_missing_fields(self):
        items = thunder.parse_instances([{"id": "3", "uuid": "u3"}])
        self.assertEqual(items[0]["index"], "3")
        self.assertEqual(items[0]["status"], "")
        self.assertIsNone(items[0]["port"])
        self.assertEqual(items[0]["http_ports"], [])

    def test_find_prefers_uuid(self):
        items = thunder.parse_instances({"0": {"uuid": "a"}, "1": {"uuid": "b"}})
        self.assertEqual(thunder.find_instance(items, "b", "0")["index"], "1")
        self.assertEqual(thunder.find_instance(items, "zzz", "0")["uuid"], "a")
        self.assertIsNone(thunder.find_instance(items, "zzz", None))

    def test_unknown_status_is_not_running(self):
        self.assertFalse(thunder.is_running("PROVISIONING"))
        self.assertFalse(thunder.is_gone_status("PROVISIONING"))
        self.assertTrue(thunder.is_running("running"))


class Snapshots(unittest.TestCase):
    def snaps(self):
        return thunder.parse_snapshots([
            {"id": "s1", "name": "aihub-thunder-20260901t000000z", "status": "READY", "minimumDiskSizeGb": 120, "createdAt": 100},
            {"id": "s2", "name": "aihub-thunder-20260902t000000z", "status": "READY", "minimumDiskSizeGb": 130, "createdAt": 200},
            {"id": "s3", "name": "aihub-thunder-20260903t000000z", "status": "FAILED", "minimumDiskSizeGb": 0, "createdAt": 300},
            {"id": "x1", "name": "aihub-other-20260901t000000z", "status": "READY", "minimumDiskSizeGb": 50, "createdAt": 50},
            {"id": "x2", "name": "mine-manual", "status": "READY", "minimumDiskSizeGb": 50, "createdAt": 400},
        ])

    def test_name(self):
        self.assertEqual(thunder.snapshot_name("Thunder_A6000", 0), "aihub-thunder-a6000-19700101t000000z")

    def test_newest_ready_ignores_failed_and_foreign(self):
        self.assertEqual(thunder.newest_ready(self.snaps(), "thunder")["id"], "s2")

    def test_rotation_keeps_newest_ready_deletes_older_and_failed(self):
        self.assertEqual(sorted(thunder.rotation(self.snaps(), "thunder")), ["s1", "s3"])

    def test_rotation_never_deletes_last_ready(self):
        only = [s for s in self.snaps() if s["id"] in ("s1", "s3")]
        self.assertEqual(thunder.rotation(only, "thunder"), ["s3"])

    def test_rotation_sorts_by_created_at_not_name(self):
        s = thunder.parse_snapshots([
            {"id": "old", "name": "aihub-t-20990101t000000z", "status": "READY", "createdAt": 1},
            {"id": "new", "name": "aihub-t-20000101t000000z", "status": "READY", "createdAt": 2}])
        self.assertEqual(thunder.rotation(s, "t"), ["old"])

    def test_creating_snapshot_blocks_nothing_but_is_not_deleted(self):
        s = thunder.parse_snapshots([
            {"id": "a", "name": "aihub-t-20260901t000000z", "status": "READY", "createdAt": 1},
            {"id": "b", "name": "aihub-t-20260902t000000z", "status": "CREATING", "createdAt": 2}])
        self.assertEqual(thunder.rotation(s, "t"), [])

    def test_prefix_of_another_backend_is_not_ours(self):
        # `aihub-thunder-` is also the start of backend `thunder-a6000`'s snapshots:
        # rotating `thunder` must never delete (or restore from) those.
        s = thunder.parse_snapshots([
            {"id": "mine", "name": "aihub-thunder-20260901t000000z", "status": "READY", "createdAt": 1},
            {"id": "other", "name": "aihub-thunder-a6000-20260902t000000z", "status": "READY", "createdAt": 2},
            {"id": "otherf", "name": "aihub-thunder-a6000-20260903t000000z", "status": "FAILED", "createdAt": 3}])
        self.assertEqual(thunder.rotation(s, "thunder"), [])
        self.assertEqual(thunder.newest_ready(s, "thunder")["id"], "mine")
        self.assertEqual(thunder.newest_ready(s, "thunder-a6000")["id"], "other")


    def test_hand_made_snapshot_with_our_prefix_is_not_ours(self):
        s = thunder.parse_snapshots([
            {"id": "manual", "name": "aihub-thunder-manual", "status": "READY", "createdAt": 1},
            {"id": "upper", "name": "aihub-thunder-20260901T000000Z", "status": "FAILED", "createdAt": 2},
            {"id": "mine", "name": "aihub-thunder-20260902t000000z", "status": "READY", "createdAt": 3}])
        self.assertEqual(thunder.rotation(s, "thunder"), [])
        self.assertEqual(thunder.newest_ready(s, "thunder")["id"], "mine")
        only_manual = [x for x in s if x["id"] == "manual"]
        self.assertIsNone(thunder.newest_ready(only_manual, "thunder"))

    def test_snapshot_without_id_is_never_deleted(self):
        s = thunder.parse_snapshots([
            {"name": "aihub-t-20260901t000000z", "status": "FAILED", "createdAt": 1},
            {"id": "", "name": "aihub-t-20260902t000000z", "status": "READY", "createdAt": 2},
            {"id": "b", "name": "aihub-t-20260903t000000z", "status": "READY", "createdAt": 3}])
        self.assertEqual(thunder.rotation(s, "t"), [])
        self.assertNotIn("", thunder.rotation(s, "t"))


class Disk(unittest.TestCase):
    def test_includes_models_base_reserve_rounded(self):
        self.assertEqual(thunder.choose_disk_gb(155 * GB, 40 * GB, 20, 0, 100, 500), 220)

    def test_floor_100_per_gpu_and_snapshot_min(self):
        self.assertEqual(thunder.choose_disk_gb(1 * GB, 1 * GB, 1, 0, 100, 500), 100)
        self.assertEqual(thunder.choose_disk_gb(1 * GB, 1 * GB, 1, 260, 100, 500), 260)

    def test_too_big_raises(self):
        with self.assertRaises(thunder.DiskTooSmall):
            thunder.choose_disk_gb(600 * GB, 0, 0, 0, 100, 500)

    def test_rounding_never_exceeds_a_max_the_need_fits(self):
        self.assertEqual(thunder.choose_disk_gb(497 * GB, 0, 0, 0, 100, 499), 499)


class Cost(unittest.TestCase):
    def test_hourly(self):
        spec = thunder.spec_for(SPECS, "a6000", 1)
        self.assertAlmostEqual(thunder.hourly_cost(PRICING, "a6000", 1, 8, 200, spec),
                               0.35 + 2 * 0.04 + 100 * 0.0003)

    def test_included_disk_is_not_billed(self):
        # Thunder includes 100 GB per GPU: only what lies beyond is billed per GB·h.
        spec = thunder.spec_for(SPECS, "a6000", 1)
        self.assertAlmostEqual(thunder.hourly_cost(PRICING, "a6000", 1, 6, 100, spec), 0.35)
        pricing = dict(PRICING, a6000_x2=0.70)
        self.assertAlmostEqual(thunder.hourly_cost(pricing, "a6000", 2, 6, 200, None), 0.70)
        self.assertAlmostEqual(thunder.hourly_cost(pricing, "a6000", 2, 6, 250, None),
                               0.70 + 50 * 0.0003)

    def test_unknown_gpu_is_none(self):
        self.assertIsNone(thunder.hourly_cost(PRICING, "b200", 1, 8, 100, None))

    def test_snapshot_monthly(self):
        self.assertAlmostEqual(thunder.snapshot_monthly(PRICING, 100), 100 * 0.00006849 * 730)


class ForeignSnapshots(unittest.TestCase):
    """A snapshot named like ours (`aihub-`) that no current Thunder backend owns keeps
    billing $/month unseen — a renamed backend leaves every old one behind, and rotation
    only ever looks at the CURRENT name's snapshots."""

    def _s(self, name, sid, gb=120, status="READY"):
        return {"id": sid, "name": name, "status": status, "min_disk_gb": gb, "created_at": 1}

    def test_orphan_snapshot_warning(self):
        snaps = [self._s("aihub-thunder-20260926t120000z", "s1"),             # owned
                 self._s("aihub-thunder-a6000-20260926t120000z", "s2", 200),  # renamed away
                 self._s("aihub-thunder-manual", "s3", 0),                    # hand-made
                 self._s("my-own-snapshot", "s4")]                            # not ours
        out = thunder.foreign_snapshots(snaps, ["thunder"], PRICING)
        self.assertEqual([o["id"] for o in out], ["s2", "s3"])
        self.assertAlmostEqual(out[0]["monthly"], thunder.snapshot_monthly(PRICING, 200))
        self.assertEqual(out[0]["gb"], 200)
        self.assertIsNone(out[1]["monthly"])            # no size → no made-up price
        # a backend of that name owns it again
        self.assertEqual([o["id"] for o in thunder.foreign_snapshots(
            snaps, ["thunder", "thunder-a6000"], PRICING)], ["s3"])
        # no price list: sizes still shown, no $/month
        self.assertIsNone(thunder.foreign_snapshots(snaps, [], None)[0]["monthly"])
        self.assertEqual(thunder.foreign_snapshots(None, ["x"], PRICING), [])


class Body(unittest.TestCase):
    def test_create_body(self):
        b = thunder.create_body({"gpu_type": "a6000", "num_gpus": 1, "vcpus": 8}, "comfy-ui", 220, "ssh-ed25519 AAA")
        self.assertEqual(b, {"cpu_cores": 8, "disk_size_gb": 220, "gpu_type": "a6000", "num_gpus": 1,
                             "template": "comfy-ui", "public_key": "ssh-ed25519 AAA"})

    def test_body_never_guesses_a_blank_vcpus(self):
        # "" = included is resolved at start (resolve_options); a body built from the
        # blank would have to invent a number — it refuses instead
        with self.assertRaises(ValueError):
            thunder.create_body({"gpu_type": "l40", "num_gpus": 1, "vcpus": ""}, "base", 100, "k")


# the three shapes /v2/specs has been seen in: wrapped map, bare map, string counts
L40 = {"specs": {"l40_x1": {"vcpuOptions": [12, 6, 24], "storageGB": {"min": 100, "max": 500}}}}
L40_BARE = {"l40_x1": {"vcpuOptions": ["24", "6", "12"]}}


class IncludedVcpus(unittest.TestCase):
    """`vcpus` blank = the GPU configuration's INCLUDED count (the smallest option):
    Thunder bills every vCPU above it, so a fixed default of 8 paid 2 extra on an l40
    (operator 2026-10-01: $0.87/h instead of $0.79/h)."""

    def test_field_default_is_included(self):
        f = {x["key"]: x for x in thunder.OPTION_FIELDS}["vcpus"]
        self.assertEqual(f["default"], thunder.VCPUS_INCLUDED)
        self.assertEqual(thunder.VCPUS_INCLUDED, "")
        self.assertEqual(f["aliases"], {"included": ""})
        self.assertIn("included", f["hint"])
        self.assertIn("billed extra", f["hint"])
        opts, errs, _ = thunder.options_of({})
        self.assertEqual((opts["vcpus"], errs), ("", []))
        # the form may send the word; what is stored is the blank
        opts, errs, typed = thunder.options_of({"opt__vcpus": "included"})
        self.assertEqual((opts["vcpus"], errs, typed["vcpus"]), ("", [], "included"))
        # a typed count is still validated as before
        self.assertEqual(thunder.options_of({"opt__vcpus": "12"})[0]["vcpus"], 12)
        self.assertTrue(thunder.options_of({"opt__vcpus": "0"})[1])
        self.assertTrue(thunder.options_of({"opt__vcpus": "6.5"})[1])

    def test_included_over_every_specs_shape(self):
        self.assertEqual(thunder.included_vcpus(L40, "l40", 1), 6)
        self.assertEqual(thunder.included_vcpus(L40_BARE, "l40", 1), 6)
        self.assertEqual(thunder.included_vcpus(SPECS, "a6000", 1), 6)
        self.assertEqual(thunder.vcpu_options(L40_BARE, "l40", 1), [6, 12, 24])
        # missing configuration, no options, junk options, no specs at all → None
        self.assertIsNone(thunder.included_vcpus(L40, "h100", 1))
        self.assertIsNone(thunder.included_vcpus(L40, "l40", 2))
        self.assertIsNone(thunder.included_vcpus({"specs": {"l40_x1": {}}}, "l40", 1))
        self.assertIsNone(thunder.included_vcpus({"specs": {"l40_x1": {"vcpuOptions": ["x"]}}},
                                                 "l40", 1))
        self.assertIsNone(thunder.included_vcpus({"specs": {"l40_x1": {"vcpuOptions": [0]}}},
                                                 "l40", 1))
        for bad in (None, [], "x", {}):
            self.assertIsNone(thunder.included_vcpus(bad, "l40", 1))
            self.assertEqual(thunder.vcpu_options(bad, "l40", 1), [])

    def test_save_refusal(self):
        opts = {"gpu_type": "l40", "num_gpus": 1}
        self.assertEqual(thunder.options_refusal(dict(opts, vcpus=8), L40),
                         "l40 ×1 offers vCPUs 6, 12, 24")
        self.assertIsNone(thunder.options_refusal(dict(opts, vcpus=12), L40))
        self.assertIsNone(thunder.options_refusal(dict(opts, vcpus=""), L40))
        # unknown specs (or a configuration they do not list) cannot judge: the start does
        self.assertIsNone(thunder.options_refusal(dict(opts, vcpus=8), None))
        self.assertIsNone(thunder.options_refusal(dict(opts, vcpus=8, gpu_type="h100"), L40))

    def test_resolve_at_start(self):
        opts = {"gpu_type": "l40", "num_gpus": 1, "vcpus": ""}
        res, why = thunder.resolve_options(opts, L40)
        self.assertEqual((res["vcpus"], why), (6, None))
        self.assertEqual(opts["vcpus"], "")                 # a copy, never the stored dict
        self.assertEqual(thunder.create_body(res, "base", 100, "k")["cpu_cores"], 6)
        # never guess: unreadable specs, or a configuration they do not name
        for specs in (None, {}, {"specs": {"a6000_x1": {"vcpuOptions": [6]}}}):
            res, why = thunder.resolve_options(opts, specs)
            self.assertIsNone(res)
            self.assertEqual(why, "cannot read Thunder's vCPU options for l40 ×1 — set vcpus "
                                  "explicitly or try again")
        # a typed count: refused when the specs know it is not offered, else as is
        res, why = thunder.resolve_options(dict(opts, vcpus=8), L40)
        self.assertEqual((res, why), (None, "l40 ×1 offers vCPUs 6, 12, 24"))
        self.assertEqual(thunder.resolve_options(dict(opts, vcpus=12), L40)[0]["vcpus"], 12)
        self.assertEqual(thunder.resolve_options(dict(opts, vcpus="12"), None)[0]["vcpus"], 12)

    def test_effective_and_label(self):
        o = {"gpu_type": "l40", "num_gpus": 1, "vcpus": ""}
        self.assertEqual(thunder.effective_vcpus(o, L40), 6)
        self.assertIsNone(thunder.effective_vcpus(o, None))
        self.assertEqual(thunder.effective_vcpus(dict(o, vcpus=12), None), 12)
        self.assertEqual(thunder.blank_label("vcpus", o, L40), "included (6 for l40 ×1)")
        self.assertEqual(thunder.blank_label("vcpus", o, None), "included")
        self.assertEqual(thunder.blank_label("reserve_gb", o, L40), "")

    def test_cost_of_included_is_the_rate(self):
        # the resolved count bills nothing extra; an unresolved blank neither
        self.assertAlmostEqual(thunder.hourly_cost(PRICING, "l40", 1, 6, 100,
                                                   thunder.spec_for(L40, "l40", 1)), 0.79)
        self.assertAlmostEqual(thunder.hourly_cost(PRICING, "l40", 1, None, 100,
                                                   thunder.spec_for(L40, "l40", 1)), 0.79)
        self.assertAlmostEqual(thunder.hourly_cost(PRICING, "l40", 1, 8, 100,
                                                   thunder.spec_for(L40, "l40", 1)), 0.87)


if __name__ == "__main__":
    unittest.main()
