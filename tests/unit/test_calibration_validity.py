"""Regression coverage for non-blocking dominance and strict calibration validity."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.files import write_jsonl
from src.queries.registry import QUERY_CLASSES
from src.simulation import controller
from src.simulation.calibration import validate_calibration


class CalibrationValidityTests(unittest.TestCase):
    def test_payload_dominance_passes_but_incomplete_design_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialized = root / "materialized"
            dataset = materialized / "fixture"
            dataset.mkdir(parents=True)
            (dataset / "tokens.bin").write_bytes(struct.pack("<Q", 1) * 20)
            queries = [
                {
                    "query_id": query_class, "query_class": query_class,
                    "dataset_id": "fixture", "graph_version": 1, "replica_id": 0,
                    "worker_token_files": ["tokens.bin"],
                }
                for query_class in QUERY_CLASSES
            ]
            write_jsonl(dataset / "queries.jsonl", queries)
            gate = validate_calibration(
                materialized, result_root=root, expected_queries_per_dataset=4
            )
            self.assertEqual(gate["status"], "PASS")
            self.assertFalse(gate["all_three_plans_win"])
            self.assertTrue(gate["warnings"])
            self.assertFalse((root / "calibration/gates.json").exists())
            self.assertTrue((root / "calibration/payload_gates.json").is_file())
            self.assertEqual(validate_calibration(materialized, result_root=root)["status"], "FAIL")
            write_jsonl(dataset / "queries.jsonl", queries[:-1])
            self.assertEqual(
                validate_calibration(
                    materialized, result_root=root, expected_queries_per_dataset=4
                )["status"], "FAIL"
            )

    @staticmethod
    def wire_rows(plan, *, tie=False):
        return [
            {
                "query_id": query_class, "query_class": query_class,
                "dataset_id": "fixture", "replica_id": 0, "graph_version": 1,
                "selected_rows": 100, "distinct_groups": 10, "mean_label_bytes": 2,
                "predicted_plan_wire_bytes": [1000, 500, 100], "partition_skew_cv": 0,
                "graph_edge_count": 200, "workers": 4,
                "plan": plan,
                "total_wire_bytes": 100 if tie else [1000, 500, 100][plan],
                "latency_simtime_ms": 1, "expected_result_sha256": "same-hash",
            }
            for query_class in QUERY_CLASSES
        ]

    def test_wire_calibration_dominance_and_ties_pass(self):
        for tie in (False, True):
            with self.subTest(tie=tie), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with (
                    patch.object(controller, "RESULT_ROOT", root),
                    patch.object(controller, "DATASET_SCALES", ("fixture",)),
                    patch.object(controller, "_run_case_batch"),
                    patch.object(controller, "_validate_case", side_effect=[
                        self.wire_rows(plan, tie=tie) for plan in (0, 1, 2)
                    ]),
                ):
                    samples = controller._forced_calibration()
                gate = json.loads((root / "calibration/gates.json").read_text())
                self.assertEqual(len(samples), 4)
                self.assertEqual(gate["status"], "PASS")
                self.assertTrue(gate["correctness_validated"])
                self.assertFalse(gate["all_three_plans_win"])
                self.assertEqual(gate["metric_scope"], "INET_MAC_wire_bytes")

    def test_wire_calibration_rejects_missing_branch_and_changed_context(self):
        for failure in ("missing", "graph_version", "plan"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                branches = [self.wire_rows(plan) for plan in (0, 1, 2)]
                if failure == "missing":
                    branches[2].pop()
                elif failure == "graph_version":
                    branches[2][0]["graph_version"] = 2
                else:
                    branches[2][0]["plan"] = 0
                with (
                    patch.object(controller, "RESULT_ROOT", Path(directory)),
                    patch.object(controller, "DATASET_SCALES", ("fixture",)),
                    patch.object(controller, "_run_case_batch"),
                    patch.object(controller, "_validate_case", side_effect=branches),
                ):
                    with self.assertRaises(controller.StudyBlocked):
                        controller._forced_calibration()

    def test_duplicate_query_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "cache/materialized/calibration/fixture/queries.jsonl"
            write_jsonl(contract, [{"query_id": "duplicate"}] * 2)
            with patch.object(controller, "RESULT_ROOT", root):
                with self.assertRaisesRegex(controller.StudyBlocked, "duplicate query contract"):
                    controller._contracts("calibration", "fixture")


if __name__ == "__main__":
    unittest.main()
