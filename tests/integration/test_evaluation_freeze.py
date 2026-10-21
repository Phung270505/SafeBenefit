"""Atomic freezing and validation of the outcome-blind evaluation workload."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.seed import MASTER_SEED
from src.datasets.workload import (
    freeze_evaluation_workload,
    validate_frozen_evaluation_workload,
)
from src.queries.registry import PLAN_NAMES, QUERY_CLASSES


class EvaluationFreezeTests(unittest.TestCase):
    @staticmethod
    def _rows(dataset: str, **_: object) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index in range(192):
            rows.append(
                {
                    "query_id": f"{dataset}-evaluation-{index}",
                    "dataset_id": dataset,
                    "query_class": QUERY_CLASSES[index % len(QUERY_CLASSES)],
                    "schedule_id": index % 8,
                    "schedule_profile": f"schedule-{index % 8}",
                    "replica_id": index % 2,
                    "graph_version_bucket": ("low", "medium", "high")[index % 3],
                    "graph_version": index + 1,
                    "parameters": {"sample": index},
                    "valid_plans": list(PLAN_NAMES),
                    "master_seed": MASTER_SEED,
                }
            )
        return rows

    def test_freeze_is_atomic_project_local_and_self_validating(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "output/current"
            evaluation = output / "cache/workloads/evaluation"
            evaluation.mkdir(parents=True)
            (evaluation / "stale.jsonl").write_text("partial", encoding="utf-8")
            calibration = root / "data/workloads/calibration"
            calibration.mkdir(parents=True)
            gate = output / "calibration/gates.json"
            gate.parent.mkdir(parents=True)
            gate.write_text(
                json.dumps({
                    "schema_version": 2, "status": "PASS", "all_three_plans_win": False,
                    "metric_scope": "INET_MAC_wire_bytes", "correctness_validated": True,
                    "errors": [], "warnings": ["one plan dominates"],
                }), encoding="utf-8"
            )
            manifest = output / "cache/workloads/evaluation_manifest.json"
            datasets = ("d0", "d1", "d2", "d3", "d4")
            patches = (
                patch("src.datasets.workload.OUTPUT_ROOT", output),
                patch("src.datasets.workload.EVALUATION_ROOT", evaluation),
                patch("src.datasets.workload.EVALUATION_MANIFEST", manifest),
                patch("src.datasets.workload.CALIBRATION_ROOT", calibration),
                patch("src.datasets.workload.DATASET_SCALES", datasets),
                patch("src.datasets.workload.FROZEN_IMPLEMENTATION_PATHS", ()),
                patch(
                    "src.datasets.workload.dataset_readiness",
                    return_value={"status": "PASS", "errors": []},
                ),
                patch("src.datasets.workload._rows_for_dataset", side_effect=self._rows),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                valid_gate = gate.read_text(encoding="utf-8")
                for invalid_gate in (
                    {"status": "PASS", "all_three_plans_win": True},
                    json.loads(valid_gate) | {"metric_scope": "exact_application_payload_before_INET_wire_validation"},
                    json.loads(valid_gate) | {"correctness_validated": False},
                    json.loads(valid_gate) | {"status": "FAIL"},
                    json.loads(valid_gate) | {"errors": ["oracle mismatch"]},
                ):
                    gate.write_text(json.dumps(invalid_gate), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "calibration gates"):
                        freeze_evaluation_workload()
                gate.write_text(valid_gate, encoding="utf-8")
                frozen = freeze_evaluation_workload()
                self.assertEqual(frozen["query_count"], 960)
                self.assertEqual(validate_frozen_evaluation_workload(), frozen)
                self.assertFalse((evaluation.parent / "evaluation.staging").exists())
                self.assertFalse((evaluation / "stale.jsonl").exists())
                self.assertTrue(
                    all(key.startswith("cache/workloads/evaluation/") for key in frozen["files"])
                )

                first = evaluation / "d0.jsonl"
                first.write_text(first.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_frozen_evaluation_workload()


if __name__ == "__main__":
    unittest.main()
