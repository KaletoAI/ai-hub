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
            {"id": "s1", "name": "aihub-thunder-20260901T000000Z", "status": "READY", "minimumDiskSizeGb": 120, "createdAt": 100},
            {"id": "s2", "name": "aihub-thunder-20260902T000000Z", "status": "READY", "minimumDiskSizeGb": 130, "createdAt": 200},
            {"id": "s3", "name": "aihub-thunder-20260903T000000Z", "status": "FAILED", "minimumDiskSizeGb": 0, "createdAt": 300},
            {"id": "x1", "name": "aihub-other-20260901T000000Z", "status": "READY", "minimumDiskSizeGb": 50, "createdAt": 50},
            {"id": "x2", "name": "mine-manual", "status": "READY", "minimumDiskSizeGb": 50, "createdAt": 400},
        ])

    def test_name(self):
        self.assertEqual(thunder.snapshot_name("Thunder_A6000", 0), "aihub-thunder-a6000-19700101T000000Z")

    def test_newest_ready_ignores_failed_and_foreign(self):
        self.assertEqual(thunder.newest_ready(self.snaps(), "thunder")["id"], "s2")

    def test_rotation_keeps_newest_ready_deletes_older_and_failed(self):
        self.assertEqual(sorted(thunder.rotation(self.snaps(), "thunder")), ["s1", "s3"])

    def test_rotation_never_deletes_last_ready(self):
        only = [s for s in self.snaps() if s["id"] in ("s1", "s3")]
        self.assertEqual(thunder.rotation(only, "thunder"), ["s3"])

    def test_rotation_sorts_by_created_at_not_name(self):
        s = thunder.parse_snapshots([
            {"id": "old", "name": "aihub-t-20990101T000000Z", "status": "READY", "createdAt": 1},
            {"id": "new", "name": "aihub-t-20000101T000000Z", "status": "READY", "createdAt": 2}])
        self.assertEqual(thunder.rotation(s, "t"), ["old"])

    def test_creating_snapshot_blocks_nothing_but_is_not_deleted(self):
        s = thunder.parse_snapshots([
            {"id": "a", "name": "aihub-t-1", "status": "READY", "createdAt": 1},
            {"id": "b", "name": "aihub-t-2", "status": "CREATING", "createdAt": 2}])
        self.assertEqual(thunder.rotation(s, "t"), [])

    def test_prefix_of_another_backend_is_not_ours(self):
        # `aihub-thunder-` is also the start of backend `thunder-a6000`'s snapshots:
        # rotating `thunder` must never delete (or restore from) those.
        s = thunder.parse_snapshots([
            {"id": "mine", "name": "aihub-thunder-20260901T000000Z", "status": "READY", "createdAt": 1},
            {"id": "other", "name": "aihub-thunder-a6000-20260902T000000Z", "status": "READY", "createdAt": 2},
            {"id": "otherf", "name": "aihub-thunder-a6000-20260903T000000Z", "status": "FAILED", "createdAt": 3}])
        self.assertEqual(thunder.rotation(s, "thunder"), [])
        self.assertEqual(thunder.newest_ready(s, "thunder")["id"], "mine")
        self.assertEqual(thunder.newest_ready(s, "thunder-a6000")["id"], "other")


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
                               0.35 + 2 * 0.04 + 200 * 0.0003)

    def test_unknown_gpu_is_none(self):
        self.assertIsNone(thunder.hourly_cost(PRICING, "b200", 1, 8, 100, None))

    def test_snapshot_monthly(self):
        self.assertAlmostEqual(thunder.snapshot_monthly(PRICING, 100), 100 * 0.00006849 * 730)


class Body(unittest.TestCase):
    def test_create_body(self):
        b = thunder.create_body({"gpu_type": "a6000", "num_gpus": 1, "vcpus": 8}, "comfy-ui", 220, "ssh-ed25519 AAA")
        self.assertEqual(b, {"cpu_cores": 8, "disk_size_gb": 220, "gpu_type": "a6000", "num_gpus": 1,
                             "template": "comfy-ui", "public_key": "ssh-ed25519 AAA"})


if __name__ == "__main__":
    unittest.main()
