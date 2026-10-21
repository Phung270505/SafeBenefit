"""Generate calibration data and freeze the outcome-blind evaluation workload."""

from __future__ import annotations

import json
import shutil
from typing import Any

from src.common.design import CALIBRATION_PER_DATASET, CLUSTERS_PER_DATASET, EVALUATION_PER_DATASET
from src.common.files import sha256_file, stable_json, write_json
from src.common.paths import OUTPUT_ROOT, PROJECT_ROOT, WORKLOAD_ROOT
from src.common.seed import MASTER_SEED, STREAM_IDS
from src.datasets.workload_core import (
    DATASET_SCALES,
    _rows_for_dataset,
    _write_rows,
    dataset_readiness,
)
from src.queries.registry import PLAN_NAMES, QUERY_CLASSES

CALIBRATION_ROOT = WORKLOAD_ROOT / "calibration"
EVALUATION_ROOT = OUTPUT_ROOT / "cache" / "workloads" / "evaluation"
EVALUATION_MANIFEST = OUTPUT_ROOT / "cache" / "workloads" / "evaluation_manifest.json"
STUDY_ID = "SAFEBENEFIT-DGQ"

FROZEN_IMPLEMENTATION_PATHS = (
    "run.py",
    "config/study.json",
    "simulator/omnetpp/tools/build.sh",
    "simulator/omnetpp/tools/run_batch.sh",
    "simulator/omnetpp/tools/run_batch_inner.sh",
    "simulator/omnetpp/tools/run_case.sh",
    "simulator/omnetpp/graph_query/GraphQuery.ned",
    "simulator/omnetpp/graph_query/GraphQueryApp.cc",
    "simulator/omnetpp/graph_query/ProjectionOperators.h",
    "simulator/omnetpp/graph_query/OperationCounters.h",
    "simulator/omnetpp/graph_query/SpillStore.h",
    "simulator/omnetpp/graph_query/omnetpp.ini",
    "src/common/config.py",
    "src/common/paths.py",
    "src/analysis/statistics.py",
    "src/analysis/oracle.py",
    "src/analysis/cost_model.py",
    "src/datasets/citation.py",
    "src/datasets/export.py",
    "src/datasets/graph.py",
    "src/datasets/ldbc_sf10.py",
    "src/datasets/ldbc.py",
    "src/datasets/replay.py",
    "src/datasets/workload.py",
    "src/methods/threshold_calibration.py",
    "src/methods/switch_cost_threshold.py",
    "src/methods/linucb.py",
    "src/plans/cost.py",
    "src/plans/switch_cost.py",
    "src/plans/encoding.py",
    "src/methods/decision.py",
    "src/queries/registry.py",
    "src/reporting/report.py",
    "src/reporting/dataset_results.py",
    "src/datasets/snapshot_statistics.py",
    "src/methods/linucb_training.py",
    "src/methods/training.py",
    "src/analysis/ablation.py",
    "src/common/seed.py",
    "src/common/partitioning.py",
    "src/simulation/calibration.py",
    "src/datasets/discrimination.py",
    "src/datasets/integrity.py",
    "src/simulation/reference_model.py",
    "src/simulation/stress.py",
    "src/simulation/wire_metrics.py",
    "src/simulation/controller.py",
    "src/experiment/runner.py",
    "src/common/design.py",
    "src/simulation/reference.py",
    "src/experiment/coordinator.py",
)


def build_calibration_workload() -> dict[str, Any]:
    """Build calibration parameters without reading evaluation outcomes."""
    readiness = dataset_readiness(DATASET_SCALES)
    if readiness["status"] != "PASS":
        raise FileNotFoundError("; ".join(readiness["errors"]))
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_SCALES:
        dataset_rows = _rows_for_dataset(
            dataset,
            split="calibration",
            schedules=4,
            component="calibration_parameters",
            sample_offset=0,
        )
        _write_rows(CALIBRATION_ROOT / f"{dataset}.jsonl", dataset_rows)
        rows.extend(dataset_rows)
    manifest = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "split": "calibration",
        "frozen": False,
        "master_seed": MASTER_SEED,
        "rng_stream_id": STREAM_IDS["calibration_parameters"],
        "query_count": len(rows),
        "expected_query_count": len(DATASET_SCALES) * CALIBRATION_PER_DATASET,
        "dataset_scales": list(DATASET_SCALES),
    }
    if len(rows) != len(DATASET_SCALES) * CALIBRATION_PER_DATASET:
        raise AssertionError(f"calibration design count mismatch: {len(rows)}")
    write_json(WORKLOAD_ROOT / "manifest.json", manifest)
    return manifest


def freeze_evaluation_workload() -> dict[str, Any]:
    """Freeze a new evaluation workload once, only after calibration gates pass."""
    if EVALUATION_MANIFEST.exists():
        raise FileExistsError("evaluation workload already exists; refusing regeneration")
    readiness = dataset_readiness(DATASET_SCALES)
    if readiness["status"] != "PASS":
        raise FileNotFoundError("; ".join(readiness["errors"]))
    gate_path = OUTPUT_ROOT / "calibration" / "gates.json"
    if not gate_path.is_file():
        raise FileNotFoundError("calibration gates have not been produced")
    gates = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        gates.get("schema_version") != 2
        or gates.get("status") != "PASS"
        or gates.get("metric_scope") != "INET_MAC_wire_bytes"
        or gates.get("correctness_validated") is not True
        or gates.get("errors")
    ):
        raise ValueError("calibration gates do not permit evaluation freezing")

    calibration_keys: set[tuple[str, str, str]] = set()
    for path in CALIBRATION_ROOT.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            calibration_keys.add(
                (row["dataset_id"], row["query_class"], stable_json(row["parameters"]))
            )

    workload_parent = EVALUATION_ROOT.parent.resolve()
    staging_root = EVALUATION_ROOT.with_name("evaluation.staging")
    for generated in (staging_root, EVALUATION_ROOT):
        resolved = generated.resolve()
        if resolved.parent != workload_parent:
            raise ValueError(f"unsafe generated workload path: {resolved}")
        if generated.exists():
            shutil.rmtree(generated)
    staging_root.mkdir(parents=True)

    for dataset in DATASET_SCALES:
        excluded_parameters = {
            (query_class, parameters)
            for dataset_id, query_class, parameters in calibration_keys
            if dataset_id == dataset
        }
        rows = _rows_for_dataset(
            dataset,
            split="evaluation",
            schedules=8,
            component="evaluation_parameters",
            sample_offset=16,
            excluded_parameters=excluded_parameters,
        )
        leaked = [
            row["query_id"]
            for row in rows
            if (row["dataset_id"], row["query_class"], stable_json(row["parameters"]))
            in calibration_keys
        ]
        if leaked:
            raise ValueError(f"calibration/evaluation parameter leakage: {leaked[:3]}")
        staging_path = staging_root / f"{dataset}.jsonl"
        _write_rows(staging_path, rows)

    return _commit_evaluation_staging(staging_root, gate_path)


def _commit_evaluation_staging(staging_root: Any, gate_path: Any) -> dict[str, Any]:
    """Validate and atomically promote a complete outcome-blind staging tree."""
    staging_root = staging_root.resolve()
    gate_path = gate_path.resolve()
    expected_names = {f"{dataset}.jsonl" for dataset in DATASET_SCALES}
    actual_names = {path.name for path in staging_root.glob("*.jsonl")}
    if actual_names != expected_names:
        raise ValueError("evaluation staging file registry mismatch")

    files: dict[str, Any] = {}
    query_count = 0
    for dataset in DATASET_SCALES:
        staging_path = staging_root / f"{dataset}.jsonl"
        rows = [line for line in staging_path.read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != EVALUATION_PER_DATASET:
            raise ValueError(f"evaluation staging query count mismatch: {dataset}")
        final_path = EVALUATION_ROOT / staging_path.name
        files[final_path.relative_to(OUTPUT_ROOT).as_posix()] = {
            "sha256": sha256_file(staging_path),
            "queries": len(rows),
        }
        query_count += len(rows)

    implementation = {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in FROZEN_IMPLEMENTATION_PATHS
    }
    manifest = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "FROZEN_BEFORE_OUTCOME_OBSERVATION",
        "master_seed": MASTER_SEED,
        "rng_stream_id": STREAM_IDS["evaluation_parameters"],
        "query_count": query_count,
        "cluster_count": query_count // 2,
        "files": files,
        "calibration_gates_sha256": sha256_file(gate_path),
        "preregistered_implementation_sha256": implementation,
    }
    if query_count != len(DATASET_SCALES) * EVALUATION_PER_DATASET or manifest["cluster_count"] != len(DATASET_SCALES) * CLUSTERS_PER_DATASET:
        raise AssertionError("evaluation design count mismatch")
    if EVALUATION_ROOT.exists():
        raise FileExistsError("evaluation target already exists before atomic commit")
    staging_root.replace(EVALUATION_ROOT)
    write_json(EVALUATION_MANIFEST, manifest)
    try:
        return validate_frozen_evaluation_workload()
    except Exception:
        EVALUATION_MANIFEST.unlink(missing_ok=True)
        EVALUATION_ROOT.replace(staging_root)
        raise


def validate_frozen_evaluation_workload() -> dict[str, Any]:
    """Verify every frozen evaluation file before materialization or resume."""
    if not EVALUATION_MANIFEST.is_file():
        raise FileNotFoundError("frozen evaluation manifest is missing")
    manifest = json.loads(EVALUATION_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("study_id") != STUDY_ID
        or manifest.get("status") != "FROZEN_BEFORE_OUTCOME_OBSERVATION"
        or manifest.get("master_seed") != MASTER_SEED
        or manifest.get("query_count") != len(DATASET_SCALES) * EVALUATION_PER_DATASET
        or manifest.get("cluster_count") != len(DATASET_SCALES) * CLUSTERS_PER_DATASET
    ):
        raise ValueError("frozen evaluation manifest contract mismatch")

    expected_files = {
        (EVALUATION_ROOT / f"{dataset}.jsonl")
        .relative_to(OUTPUT_ROOT)
        .as_posix(): dataset
        for dataset in DATASET_SCALES
    }
    registered_files = manifest.get("files")
    if not isinstance(registered_files, dict) or set(registered_files) != set(expected_files):
        raise ValueError("frozen evaluation file registry mismatch")

    calibration_keys: set[tuple[str, str, str]] = set()
    for calibration_path in CALIBRATION_ROOT.glob("*.jsonl"):
        for line in calibration_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            calibration_keys.add(
                (row["dataset_id"], row["query_class"], stable_json(row["parameters"]))
            )

    query_ids: set[str] = set()
    parameter_keys: set[tuple[str, str, str]] = set()
    observed_count = 0
    for relative, dataset in expected_files.items():
        path = OUTPUT_ROOT / relative
        entry = registered_files[relative]
        if not path.is_file() or entry.get("sha256") != sha256_file(path):
            raise ValueError(f"frozen evaluation checksum mismatch: {dataset}")
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(rows) != EVALUATION_PER_DATASET or entry.get("queries") != len(rows):
            raise ValueError(f"frozen evaluation query count mismatch: {dataset}")
        for row in rows:
            query_id = str(row.get("query_id", ""))
            if (
                not query_id
                or query_id in query_ids
                or row.get("dataset_id") != dataset
                or row.get("query_class") not in QUERY_CLASSES
                or row.get("graph_version_bucket") not in {"low", "medium", "high"}
                or int(row.get("schedule_id", -1)) not in range(8)
                or int(row.get("replica_id", -1)) not in range(2)
                or int(row.get("graph_version", 0)) <= 0
                or row.get("valid_plans") != list(PLAN_NAMES)
                or row.get("master_seed") != MASTER_SEED
            ):
                raise ValueError(f"invalid frozen evaluation query: {query_id or dataset}")
            key = (dataset, row["query_class"], stable_json(row["parameters"]))
            if key in calibration_keys:
                raise ValueError(f"calibration/evaluation leakage: {query_id}")
            if key in parameter_keys:
                raise ValueError(f"duplicate evaluation parameters: {query_id}")
            query_ids.add(query_id)
            parameter_keys.add(key)
        observed_count += len(rows)
    if observed_count != manifest["query_count"]:
        raise ValueError("frozen evaluation total query count mismatch")
    return manifest
