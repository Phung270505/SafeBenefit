"""Strict, per-dataset cost tables; never pool scales or model artifacts."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from src.common.config import METHOD_IDS, load_study_config
from src.common.design import DATASET_IDS, STUDY_SCHEMA_VERSION
from src.common.files import write_json
from src.simulation.reference import OPERATION_TYPES

BYTE_FIELDS = ("payload_bytes", "control_bytes", "retry_bytes", "switch_bytes", "total_wire_bytes")
CONTEXT_FIELDS = {
    "graph_vertex_count",
    "graph_edge_count",
    "workers",
    "network_hops",
    "network_intermediate_nodes",
    "peak_group_buffer_estimated_bytes",
}
PHASES = ("scan", "encode", "decode", "merge", "decision", "protocol")
PERFORMANCE_FIELDS = (
    "cpu_ms",
    "simulation_time_ms",
    "scan_cpu_service_worker_sum_ms",
    "scan_service_critical_path_ms",
    "worker_suffix_cpu_service_sum_ms",
    "worker_suffix_cpu_critical_path_ms",
    "coordinator_merge_cpu_service_ms",
    "decision_cpu_service_ms",
    "decode_cpu_service_ms",
    "protocol_cpu_service_ms",
    "scan_io_service_worker_sum_ms",
    "scan_queue_wait_worker_sum_ms",
    "modeled_io_read_bytes",
    "modeled_io_write_bytes",
    "modeled_spill_bytes",
    "coordinator_queue_wait_ms",
    "grouping_io_read_bytes",
    "grouping_io_write_bytes",
    "peak_group_buffer_estimated_bytes",
    "intermediate_rows",
    "scanned_edges",
    "graph_vertex_count",
    "graph_edge_count",
    "workers",
    "network_hops",
    "network_intermediate_nodes",
    "regret_bytes",
)


def validate_dataset_rows(rows: list[dict[str, Any]], *, expected_per_method: int = 192) -> str:
    """Reject incomplete, noncanonical, duplicated and unpaired result matrices."""
    datasets = {row["dataset_id"] for row in rows}
    if len(datasets) != 1 or not datasets.issubset(DATASET_IDS):
        raise ValueError("result tables require one registered LDBC dataset")
    pairs: set[tuple[str, str]] = set()
    signatures = {}
    observed = {method: set() for method in METHOD_IDS}
    for row in rows:
        method, query = row["method_id"], row["query_id"]
        if method not in observed or (method, query) in pairs:
            raise ValueError("duplicate or unregistered result")
        if row.get("status") != "PASS" or row.get("outcome") != "SUCCESS":
            raise ValueError("unsuccessful main query cannot be ranked")
        if not row.get("result_sha256") or row["result_sha256"] != row["expected_result_sha256"]:
            raise ValueError("result checksum mismatch")
        pairs.add((method, query))
        observed[method].add(query)
        signature = tuple(
            row[field]
            for field in (
                "graph_version",
                "query_class",
                "schedule_id",
                "replica_id",
                "initial_plan",
                "expected_result_sha256",
            )
        )
        if query in signatures and signatures[query] != signature:
            raise ValueError("methods received different inputs")
        signatures[query] = signature
        for field in (*BYTE_FIELDS, *PERFORMANCE_FIELDS):
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {field}")
        if sum(row[field] for field in BYTE_FIELDS[:-1]) != row["total_wire_bytes"]:
            raise ValueError("nonexclusive wire-byte attribution")
        if row.get("reference_model_id") != load_study_config()["reference_model"]["model_id"]:
            raise ValueError("reference-model revision mismatch")
        counts = row["operation_counts_by_phase"]
        if set(counts) != set(PHASES):
            raise ValueError("operation phase registry mismatch")
        for values in counts.values():
            if set(values) != set(OPERATION_TYPES) or any(
                not isinstance(n, int) or n < 0 for n in values.values()
            ):
                raise ValueError("invalid operation counts")
    baseline = observed["STATIC"]
    if len(baseline) != expected_per_method or any(ids != baseline for ids in observed.values()):
        raise ValueError("incomplete paired dataset evaluation")
    return next(iter(datasets))


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_dataset_results(rows: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Export all numeric dimensions, preserving separate units and definitions."""
    dataset = validate_dataset_rows(rows)
    summaries = []
    flat_rows = []
    for row in rows:
        flat = {
            key: value for key, value in row.items() if isinstance(value, (int, float, str, bool))
        }
        for phase, counts in row["operation_counts_by_phase"].items():
            flat.update({f"ops_{phase}_{operation}": count for operation, count in counts.items()})
        flat_rows.append(flat)
    baseline = sum(row["total_wire_bytes"] for row in rows if row["method_id"] == "STATIC")
    for method in METHOD_IDS:
        selected = [row for row in rows if row["method_id"] == method]
        summary = {"dataset_id": dataset, "method_id": method, "queries": len(selected)}
        for field in (*BYTE_FIELDS, *PERFORMANCE_FIELDS):
            values = [float(row[field]) for row in selected]
            if field not in CONTEXT_FIELDS:
                summary[f"sum_{field}"] = sum(row[field] for row in selected)
            else:
                summary[f"min_{field}"] = min(values)
                summary[f"max_{field}"] = max(values)
            summary[f"mean_{field}"] = mean(values)
            summary[f"p95_{field}"] = float(np.quantile(values, 0.95))
        summary["total_wire_bytes"] = sum(row["total_wire_bytes"] for row in selected)
        summary["saving_vs_static_fraction"] = (
            (baseline - summary["total_wire_bytes"]) / baseline if baseline else 0.0
        )
        summary["switches"] = sum(row["initial_plan"] != row["plan_id"] for row in selected)
        summary["fallbacks"] = sum(bool(row["policy_fallback"]) for row in selected)
        summary["retry_queries"] = sum(row["retry_bytes"] > 0 for row in selected)
        summary["correct_queries"] = len(selected)
        for phase in PHASES:
            summary[f"sum_work_units_{phase}"] = sum(
                sum(row["operation_counts_by_phase"][phase].values()) for row in selected
            )
        summaries.append(summary)
    _csv(root / "per_query.csv", flat_rows)
    _csv(root / "method_summary.csv", summaries)
    write_json(
        root / "results.json",
        {
            "schema_version": STUDY_SCHEMA_VERSION,
            "dataset_id": dataset,
            "status": "MAIN_COMPLETE",
            "ranking_metric": "total_wire_bytes",
            "methods": summaries,
            "time_scope": "fixed logical reference model, not measured host time",
            "counter_scope": "logical actions, not physical CPU instructions",
            "memory_scope": "estimated logical buffer bytes, not measured process RSS",
        },
        allow_nan=False,
    )
    return summaries
