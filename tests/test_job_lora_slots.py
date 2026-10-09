"""The job view says WHERE each client LoRA landed.

A client numbers its LoRAs from 1 (`lora_01`, `strength_01`); `_apply_lora_cascade`
drops each onto the next FREE stack slot, so on an alias that pins `lora_01` (a turbo
LoRA) the client's `lora_01` really runs in slot `lora_02`. The job page used to list
the raw client params next to the pinned values — `lora_01 = <client LoRA>` beside
`lora_01 = <turbo>` — which reads as "slot 1 used twice, slot 2 empty" although the
submitted workflow was right (job 47c15611cfc8, 2026-10-09). The landing slot is in
`meta.loras` (`node.field=value`); the page must show it on the param row.
"""
import asyncio
import unittest

import admin


_JOB = {
    "id": "47c15611cfc844639e19214654d20888", "status": "done", "task": "img2img",
    "alias": "Qwen2.1-Turbo", "backend": "k12-gpu", "created": 1_791_582_828,
    "updated": 1_791_582_861, "owner": "kai", "results": [],
    "meta": {
        "inputs": {"prompt": "p", "params": {"lora_01": "Qwen2.1-thighgap.safetensors",
                                             "strength_01": 0.7, "steps": 6}},
        "fixed": ["130.lora_01", "149.steps"],
        "loras": ["130.lora_02=Qwen2.1-thighgap.safetensors"],
    },
}


class JobLoraLanding(unittest.TestCase):
    def setUp(self):
        self._saved = (admin.jobs.is_active, admin.jobs.get, admin.jobs.neighbors,
                       admin.jobs.result_path, admin.store.is_active)
        admin.jobs.is_active = lambda: True
        admin.jobs.neighbors = lambda jid: (None, None)
        admin.jobs.result_path = lambda jid, n: None
        admin.store.is_active = lambda: False

    def tearDown(self):
        (admin.jobs.is_active, admin.jobs.get, admin.jobs.neighbors,
         admin.jobs.result_path, admin.store.is_active) = self._saved

    def _render(self, job):
        admin.jobs.get = lambda jid: job
        return asyncio.run(admin.job_detail_page(job["id"], None)).body.decode()

    def test_client_lora_row_names_its_landing_slot(self):
        html = self._render(_JOB)
        row = html[html.index("<code>lora_01</code>"):]
        row = row[:row.index("</tr>")]
        self.assertIn("lora_02", row)
        self.assertIn("130", row)

    def test_client_strength_row_follows_its_lora(self):
        html = self._render(_JOB)
        row = html[html.index("<code>strength_01</code>"):]
        row = row[:row.index("</tr>")]
        self.assertIn("strength_02", row)

    def test_same_lora_twice_lands_on_two_slots(self):
        job = dict(_JOB, meta=dict(_JOB["meta"],
                                   inputs={"params": {"lora_01": "a.safetensors",
                                                      "lora_02": "a.safetensors"}},
                                   loras=["130.lora_02=a.safetensors",
                                          "130.lora_03=a.safetensors"]))
        self.assertEqual(admin._lora_landing(job["meta"]["inputs"]["params"],
                                             job["meta"]["loras"]),
                         {"lora_01": "130.lora_02", "lora_02": "130.lora_03"})

    def test_unplaced_lora_gets_no_landing(self):
        # No free slot left → the cascade dropped it; never claim a slot it did not get.
        self.assertEqual(admin._lora_landing({"lora_01": "x.safetensors"}, []), {})


if __name__ == "__main__":
    unittest.main()
