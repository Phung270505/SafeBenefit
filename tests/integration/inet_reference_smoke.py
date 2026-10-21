"""Explicit INET integration check on synthetic fixtures, NEVER research results.

Run after build.sh with SAFEBENEFIT_OUTPUT_ROOT pointing at output/four_method_verification.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from src.common.config import load_study_config
from src.common.design import DATASET_IDS
from src.common.files import sha256_file, stable_json, write_json
from src.common.paths import PROJECT_ROOT
from src.datasets.replay import write_materialization
from src.methods.linucb_training import train_linucb
from src.methods.threshold_calibration import select_switch_threshold
from src.methods.training import train_calibrated_methods
from src.queries.registry import QUERY_CLASSES
from src.simulation import controller
from src.simulation.reference_model import freeze_reference_model

DATASET = DATASET_IDS[0]


def prepare(root: Path, split: str) -> None:
    folder = root / "cache/materialized" / split / DATASET
    contracts = []
    tsv = []
    for index, query_class in enumerate(QUERY_CLASSES):
        for replica in (0, 1):
            offset = 0 if split == "calibration" else 3
            values = list(range(20 + offset))
            if replica:
                values = (
                    [7 + offset] * 10000
                    if query_class == "NEIGHBOR_TYPE_AGG"
                    else list(range(800 + offset))
                )
                if query_class == "TEMPORAL_TYPE_AGG" and split == "evaluation":
                    values = list(range(160000))  # compressed UDP fragmentation regression
            query = f"fixture-{split}-{index}-{replica}"
            tokens = [values[part::8] for part in range(8)]
            canonical = sorted(Counter(values).items())
            contract = write_materialization(
                folder,
                {
                    "query_id": query,
                    "dataset_id": DATASET,
                    "scale_id": DATASET,
                    "query_class": query_class,
                    "replica_id": replica,
                    "schedule_id": 0,
                    "schedule_profile": "fixture",
                    "graph_version_bucket": "low",
                    "graph_version": 1,
                    "reducer": "COUNT" if query_class.endswith("_AGG") else "DISTINCT",
                    "worker_tokens": tokens,
                    "expected_result": canonical,
                    "expected_result_sha256": controller._result_hash(dict(canonical)),
                    "graph_vertex_count": 200000,
                    "graph_edge_count": 300000,
                    "scan_work_by_partition": [2] * 8,
                    "scanned_rows_by_partition": [1] * 8,
                    "scanned_rows": 8,
                },
            )
            contracts.append(contract)
            tsv.append(
                f"{query}\t{query_class}\t0\t0\t0\t1\t1\t1\t200000\t300000\t8\t2,2,2,2,2,2,2,2\t1,1,1,1,1,1,1,1\n"
            )
    (folder / "queries.jsonl").write_text(
        "".join(stable_json(row) + "\n" for row in contracts), encoding="utf-8"
    )
    (folder / "queries.tsv").write_text("".join(tsv), encoding="utf-8")


def forced(root: Path, split: str) -> dict:
    specs = [
        controller.CaseSpec(DATASET, split, "STATIC", str(plan), f"fixture-{split}-p{plan}")
        for plan in (0, 1, 2)
    ]
    controller._run_case_batch(f"fixture-{split}", specs)
    by_query = {}
    for spec in specs:
        for row in controller._validate_case(
            root / "simulation" / spec.run_id, split, DATASET, "STATIC"
        ):
            by_query.setdefault(row["query_id"], {})[int(spec.initial)] = row
    return by_query


def main() -> None:
    root = (PROJECT_ROOT / "output/four_method_verification").resolve()
    # Invalidate only markers in this test namespace; never reuse stale native test cases.
    for marker in (root / "simulation").glob("*/.complete"):
        marker.unlink()
    prepare(root, "calibration")
    prepare(root, "evaluation")
    freeze_reference_model(root, root / "config/runtime_calibration.json")
    with (
        patch.object(controller, "RESULT_ROOT", root),
        patch.object(controller, "DATASET_SCALES", (DATASET,)),
    ):
        calibration = forced(root, "calibration")
        samples = []
        for query, plans in calibration.items():
            row = plans[0]
            samples.append(
                {
                    "query_id": query,
                    "dataset_id": DATASET,
                    "query_class": row["query_class"],
                    "replica_id": row["replica_id"],
                    "graph_version": 1,
                    "graph_scale_edges": row["graph_edge_count"],
                    "worker_count": 4,
                    "selected_rows": row["selected_rows"],
                    "distinct_groups": row["distinct_groups"],
                    "mean_label_bytes": row["mean_label_bytes"],
                    "predicted_bytes": row["predicted_plan_wire_bytes"],
                    "partition_skew_cv": row["partition_skew_cv"],
                    "retry_count": 0,
                    "switch_bytes": 512,
                    "costs": {
                        str(plan): values["total_wire_bytes"] for plan, values in plans.items()
                    },
                }
            )
        policies = root / "policies"
        learned = train_linucb(samples, policies)
        baseline = train_calibrated_methods(samples, policies)
        assert learned["initial_plans"] == baseline["initial_plans"]
        controller._write_initial_plans(learned["initial_plans"])
        checksums = {path.name: sha256_file(path) for path in policies.glob("*.tsv")}
        evaluation = forced(root, "evaluation")
        costs = {
            query: {plan: row["total_wire_bytes"] for plan, row in plans.items()}
            for query, plans in evaluation.items()
        }
        rows = controller._run_evaluation(costs)
        assert len(rows) == 8 * len(controller.METHODS)
        for row in rows:
            if row["method"] == "SWITCH_COST_THRESHOLD":
                expected = select_switch_threshold(
                    int(row["initial_plan"]),
                    {int(plan): float(value) for plan, value in row["predicted_plan_wire_bytes"].items()},
                    float(row["estimated_switch_wire_bytes"]), baseline["switch_threshold"],
                )
                assert int(row["plan_id"]) == expected, "Python/C++ switch-threshold parity"
        for row in rows:
            units = sum(
                sum(counts.values()) for counts in row["operation_counts_by_phase"].values()
            )
            assert math.isclose(row["cpu_ms"], units / 1e7 * 1000, rel_tol=1e-5, abs_tol=1e-6), row[
                "query_id"
            ]
        assert checksums == {path.name: sha256_file(path) for path in policies.glob("*.tsv")}
        suite = copy.deepcopy(load_study_config())
        suite["scalability_evaluation"]["distinct_queries"] = 4
        with patch.object(controller, "load_study_config", return_value=suite):
            scalability = controller._run_scalability()
        assert len(scalability) == 4 * 5 * 3 * len(controller.METHODS)
        stress = controller._run_fault_stress()
        assert stress["status"] == "PASS"
    baseline = root / "persistent_spool_parity_baseline.json"
    if baseline.is_file():
        before = {
            (row["query_id"], row["method"]): row
            for row in json.loads(baseline.read_text(encoding="utf-8"))
        }
        for row in rows:
            original = before[(row["query_id"], row["method"])]
            for field in ("total_wire_bytes", "result_sha256"):
                assert row[field] == original[field], field
    write_json(
        root / "integration_check.json",
        {
            "status": "PASS",
            "scope": "synthetic integration fixture only, not scientific evaluation",
            "main_executions": len(rows),
            "methods": len(controller.METHODS),
            "forced_cases": 6,
            "scalability_executions": len(scalability),
            "scalability_scope": "fixture subset only",
            "oracle_matches": len(rows),
            "reference_cpu_equation": "PASS",
            "stress_scenarios": len(stress["network_faults"]),
            "policy_frozen": True,
            "persistent_spool_wire_result_parity": "PASS" if baseline.is_file() else "NOT_RUN",
            "cpu_model_cross_revision": "not compared; executed scalar counters refined in reference-2",
            "build_receipt_sha256": sha256_file(root / "cache/build/build.sha256"),
        },
    )
    print(json.dumps({"status": "PASS", "scope": "fixture integration", "rows": len(rows)}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
