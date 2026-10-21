"""Validate calibration coverage and describe exact payload costs without outcome gates."""

from __future__ import annotations

import struct
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.design import CALIBRATION_PER_DATASET
from src.common.files import read_jsonl, write_json, write_jsonl
from src.common.paths import OUTPUT_ROOT
from src.datasets.discrimination import summarize_plan_costs
from src.plans.encoding import encode_labels
from src.queries.registry import PLAN_NAMES, QUERY_CLASSES

TOKEN = struct.Struct("<Q")
RESULT_ROOT = OUTPUT_ROOT


def _tokens(path: Path) -> list[int]:
    payload = path.read_bytes()
    if len(payload) % TOKEN.size:
        raise ValueError(f"truncated token file: {path}")
    return [item[0] for item in struct.iter_unpack(TOKEN.format, payload)]


def exact_plan_payloads(query: dict[str, Any], query_root: Path, workers: int = 4) -> list[int]:
    """Measure codecs after mapping eight stable partitions to the requested workers."""
    worker_tokens: list[list[int]] = [[] for _ in range(workers)]
    files = query["worker_token_files"]
    for partition, relative in enumerate(files):
        worker_tokens[partition % workers].extend(_tokens(query_root / relative))
    return [
        sum(len(encode_labels(tokens, plan)) for tokens in worker_tokens)
        for plan in PLAN_NAMES
    ]


def validate_calibration(
    materialized_root: Path | None = None,
    *,
    result_root: Path = RESULT_ROOT,
    study_id: str = "SAFEBENEFIT-DGQ",
    expected_queries_per_dataset: int = CALIBRATION_PER_DATASET,
) -> dict[str, Any]:
    """Check complete calibration inputs; dominance/ties never prevent evaluation."""
    if expected_queries_per_dataset <= 0 or expected_queries_per_dataset % len(QUERY_CLASSES):
        raise ValueError("calibration size must be positive and balanced across query classes")
    root = materialized_root or result_root / "cache" / "materialized" / "calibration"
    traces: list[dict[str, Any]] = []
    errors: list[str] = []
    dataset_counts: Counter[str] = Counter()
    class_counts: Counter[tuple[str, str]] = Counter()
    for dataset_root in sorted(path for path in root.iterdir() if path.is_dir()):
        contract_path = dataset_root / "queries.jsonl"
        if not contract_path.exists():
            raise FileNotFoundError(f"missing calibration contract: {contract_path}")
        dataset_counts[dataset_root.name] = 0
        for query in read_jsonl(contract_path):
            if query["dataset_id"] != dataset_root.name:
                raise ValueError(f"calibration dataset mismatch: {contract_path}")
            dataset_counts[dataset_root.name] += 1
            class_counts[dataset_root.name, query["query_class"]] += 1
            costs = exact_plan_payloads(query, dataset_root, workers=4)
            minimum = min(costs)
            unique_winner = costs.count(minimum) == 1
            winner = costs.index(minimum) if unique_winner else None
            traces.append(
                {
                    "query_id": query["query_id"],
                    "dataset_id": query["dataset_id"],
                    "query_class": query["query_class"],
                    "graph_version": query["graph_version"],
                    "replica_id": query["replica_id"],
                    "payload_bytes_by_plan": costs,
                    "unique_winner": winner,
                }
            )
    diagnostics = summarize_plan_costs(
        row | {"costs": {str(plan): cost for plan, cost in enumerate(row["payload_bytes_by_plan"])}}
        for row in traces
    )
    errors.extend(diagnostics["errors"])
    for dataset, count in dataset_counts.items():
        if count != expected_queries_per_dataset:
            errors.append(f"{dataset}: expected {expected_queries_per_dataset} calibration queries, got {count}")
        for query_class in QUERY_CLASSES:
            expected = expected_queries_per_dataset // len(QUERY_CLASSES)
            if class_counts[dataset, query_class] != expected:
                errors.append(f"{dataset}/{query_class}: expected {expected} calibration queries")
    gate = {
        **diagnostics,
        "study_id": study_id,
        "status": "PASS" if not errors else "FAIL",
        "metric_scope": "exact_application_payload_before_INET_wire_validation",
        "errors": errors,
    }
    trace_path = result_root / "calibration" / "forced_plan_payloads.jsonl"
    write_jsonl(trace_path, traces)
    # Keep payload diagnostics separate from the later correctness/MAC-wire gate.
    write_json(result_root / "calibration" / "payload_gates.json", gate)
    return gate
