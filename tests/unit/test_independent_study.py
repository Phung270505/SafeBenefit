"""Independent datasets, fixed reference time and strict per-scale exports."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis.statistics import analyze_results
from src.common.config import METHOD_IDS, load_study_config
from src.common.design import DATASET_ALIASES, DATASET_IDS
from src.common.paths import PROJECT_ROOT
from src.methods.linucb_training import train_linucb
from src.methods.training import train_calibrated_methods
from src.reporting.dataset_results import PERFORMANCE_FIELDS, PHASES, export_dataset_results
from src.simulation.reference import OPERATION_TYPES, OperationCounts, cpu_seconds
from src.simulation.reference_model import freeze_reference_model


def study_rows(dataset: str = DATASET_IDS[0]) -> list[dict]:
    rows = []
    classes = load_study_config()["query_classes"]
    for method in METHOD_IDS:
        for index in range(192):
            row = {
                "dataset_id": dataset,
                "method": method,
                "method_id": method,
                "query_id": f"{dataset}-q{index}",
                "query_class": classes[index // 48],
                "graph_version": (index // 16) % 3 + 1,
                "graph_version_bucket": ("low", "medium", "high")[(index // 16) % 3],
                "schedule_id": (index // 2) % 8,
                "replica_id": index % 2,
                "initial_plan": 0,
                "plan_id": 0,
                "status": "PASS",
                "outcome": "SUCCESS",
                "result_sha256": "a" * 64,
                "expected_result_sha256": "a" * 64,
                "payload_bytes": 60,
                "control_bytes": 40,
                "retry_bytes": 0,
                "switch_bytes": 0,
                "total_wire_bytes": 100,
                "policy_fallback": False,
                "operation_counts_by_phase": {
                    phase: dict.fromkeys(OPERATION_TYPES, 0) for phase in PHASES
                },
                "normalized_regret": 0,
                "oracle_match": True,
                "unnecessary_switch": False,
                "missed_beneficial_switch": False,
                "wrong_plan": False,
                "decision_latency_ns": 100,
                "reference_model_id": load_study_config()["reference_model"]["model_id"],
            }
            row.update(dict.fromkeys(PERFORMANCE_FIELDS, 0))
            row["cpu_ms"] = 2.0
            row["simulation_time_ms"] = 8.0
            rows.append(row)
    return rows


class IndependentStudyTests(unittest.TestCase):
    def test_child_scopes_are_isolated_by_process(self):
        for dataset, alias in DATASET_ALIASES.items():
            env = dict(os.environ, SAFEBENEFIT_DATASET_ID=dataset)
            process = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import json;from src.common.paths import OUTPUT_ROOT,WORKLOAD_ROOT;"
                    "from src.common.config import load_study_config;"
                    "c=load_study_config();print(json.dumps([str(OUTPUT_ROOT),str(WORKLOAD_ROOT),c['dataset_scales'],c['evaluation_design']]))",
                ],
                cwd=PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            root, workloads, datasets, design = json.loads(process.stdout)
            self.assertEqual(Path(root).name, alias)
            self.assertTrue(Path(workloads).is_relative_to(Path(root)))
            self.assertEqual(datasets, [dataset])
            self.assertEqual(design["expected_method_query_runs"], 768)

    def test_fixed_time_does_not_use_host_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch("time.process_time_ns", side_effect=AssertionError("host clock forbidden")),
                patch("time.perf_counter_ns", side_effect=AssertionError("host clock forbidden")),
            ):
                result = freeze_reference_model(Path(folder), Path(folder) / "reference.json")
            self.assertFalse(result["host_clock_used"])
            counts = OperationCounts()
            counts.add("tuple_visit", 3)
            counts.add("arithmetic", 2)
            units = counts.work_units(result["operation_weights"])
            self.assertEqual(units, 5)
            self.assertAlmostEqual(
                cpu_seconds(units, result["codec_cpu_units_per_second"]), 0.0000005
            )

    def test_cross_dataset_training_is_rejected(self):
        mixed = [{"dataset_id": DATASET_IDS[0]}, {"dataset_id": DATASET_IDS[1]}]
        with tempfile.TemporaryDirectory() as folder:
            for train in (train_linucb, train_calibrated_methods):
                with self.assertRaisesRegex(ValueError, "one dataset"):
                    train(mixed, Path(folder))

    def test_export_contains_four_methods_for_one_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            summary = export_dataset_results(study_rows(), root)
            self.assertEqual(len(summary), 4)
            self.assertEqual(summary[0]["total_wire_bytes"], 19200)
            self.assertEqual(summary[0]["mean_cpu_ms"], 2)
            with (root / "method_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)
            with self.assertRaises(ValueError):
                export_dataset_results(study_rows() + study_rows(DATASET_IDS[1]), root)
            with self.assertRaises(ValueError):
                export_dataset_results(study_rows()[:-1], root)

    def test_bootstrap_is_per_dataset_not_pooled(self):
        analysis = analyze_results(study_rows(), resamples=20)
        self.assertEqual(analysis["paired_parameter_clusters"], 96)
        self.assertEqual(analysis["comparisons"]["STATIC"]["conclusion"], "PRACTICALLY_EQUIVALENT")
        self.assertIsNone(analysis["lowest_cost_method"])
        self.assertEqual(set(analysis["lowest_cost_methods"]), set(METHOD_IDS))
        self.assertEqual(analysis["action_disagreement_queries"], 0)
        with self.assertRaisesRegex(ValueError, "separately"):
            analyze_results(study_rows() + study_rows(DATASET_IDS[1]), resamples=20)

    def test_fresh_handles_frozen_readonly_config_without_touching_data(self):
        from src.common.cleanup import reset_current_output

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = root / "output/current"
            (current / "config").mkdir(parents=True)
            artifact = current / "config/study.json"
            artifact.write_text("{}")
            artifact.chmod(0o444)
            data = root / "data"
            data.mkdir()
            (data / "keep.txt").write_text("keep")
            reset_current_output(root, current)
            self.assertTrue((data / "keep.txt").is_file())
            self.assertFalse(artifact.exists())
