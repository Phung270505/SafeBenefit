"""Load and validate the single authoritative study configuration."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

from src.common.design import (
    ACTIVE_DATASETS,
    CALIBRATION_PER_DATASET,
    CHECKPOINTS,
    CLUSTERS_PER_DATASET,
    DATASET_IDS,
    EVALUATION_PER_DATASET,
    STUDY_SCHEMA_VERSION,
)
from src.common.paths import CONFIG_ROOT
from src.simulation.reference import OPERATION_TYPES

METHOD_IDS = (
    "STATIC",
    "SWITCH_COST_THRESHOLD",
    "LINUCB",
    "SAFE_BENEFIT",
)
PLAN_IDS = ("0", "1", "2")


@lru_cache(maxsize=1)
def load_study_config() -> dict[str, Any]:
    """Return the validated study configuration."""
    path = CONFIG_ROOT / "study.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("study_id") != "SAFEBENEFIT-DGQ":
        raise ValueError("study_id must be SAFEBENEFIT-DGQ")
    if config.get("schema_version") != STUDY_SCHEMA_VERSION:
        raise ValueError("four-method LDBC studies require schema_version 5")
    if config.get("master_seed") != 42:
        raise ValueError("master_seed must be 42")
    if tuple(config.get("active_methods", ())) != METHOD_IDS:
        raise ValueError("method registry differs from the registered method order")
    if (
        config.get("focal_method") != "SAFE_BENEFIT"
        or tuple(config.get("comparators", ())) != tuple(method for method in METHOD_IDS if method != "SAFE_BENEFIT")
        or config.get("statistics", {}).get("holm_comparisons") != 3
    ):
        raise ValueError("exactly three baselines and three Holm comparisons are required")
    if config.get("experiment_contract", {}).get("require_all_plan_winners_in_calibration") is not False:
        raise ValueError("plan dominance must remain a diagnostic, not a benchmark gate")
    if tuple(config.get("plans", {}).keys()) != PLAN_IDS:
        raise ValueError("exactly physical plans 0, 1 and 2 are required")
    if tuple(config.get("dataset_scales", ())) != DATASET_IDS:
        raise ValueError("only the three registered LDBC scales are supported")
    if config.get("graph_checkpoints") != CHECKPOINTS:
        raise ValueError("graph checkpoint registry mismatch")
    design = config.get("evaluation_design", {})
    if design.get("schedules") != 8 or design.get("replicas") != 2:
        raise ValueError("evaluation requires eight schedules and two replicas")
    if config.get("calibration_design") != {
        "schedules": 4,
        "replicas": 2,
        "expected_queries": CALIBRATION_PER_DATASET * len(DATASET_IDS),
    }:
        raise ValueError("calibration design registry mismatch")
    if config.get("per_dataset_design") != {
        "calibration_queries": CALIBRATION_PER_DATASET,
        "evaluation_queries": EVALUATION_PER_DATASET,
        "clusters": CLUSTERS_PER_DATASET,
        "main_executions": EVALUATION_PER_DATASET * len(METHOD_IDS),
        "forced_plan_executions": EVALUATION_PER_DATASET * len(PLAN_IDS),
        "scalability_queries": 60,
    }:
        raise ValueError("per-dataset design registry mismatch")
    if (
        design.get("expected_queries") != EVALUATION_PER_DATASET * len(DATASET_IDS)
        or design.get("expected_method_query_runs")
        != EVALUATION_PER_DATASET * len(DATASET_IDS) * len(METHOD_IDS)
        or design.get("expected_forced_plan_runs")
        != EVALUATION_PER_DATASET * len(DATASET_IDS) * len(PLAN_IDS)
    ):
        raise ValueError("evaluation cardinalities differ from the registered design")
    threshold = config.get("switch_threshold_training", {})
    if (
        threshold.get("minimum_threshold") != 0.0
        or threshold.get("maximum_threshold") != 0.5
        or threshold.get("step") != 0.01
    ):
        raise ValueError("switch-threshold grid must be 0..0.50 in 0.01 steps")
    switch_model = config.get("switch_cost_model", {})
    if (
        float(switch_model.get("intercept_bytes", -1)) < 0
        or float(switch_model.get("per_worker_bytes", -1)) <= 0
    ):
        raise ValueError("invalid switching-cost model")
    scalability = config.get("scalability_evaluation", {})
    if (
        tuple(scalability.get("worker_counts", ())) != (1, 2, 4, 8, 16)
        or tuple(scalability.get("schedule_ids", ())) != (0, 1, 2, 3, 4)
        or scalability.get("distinct_queries") != 60 * len(DATASET_IDS)
    ):
        raise ValueError("scalability design differs from the registered design")
    limits = config.get("resource_limits", {})
    if limits != {
        "io_buffer_bytes": 67108864,
        "sqlite_cache_bytes": 268435456,
        "adjacency_cache_bytes": 67108864,
        "grouping_memory_bytes": 67108864,
    }:
        raise ValueError("reference schema v5 requires the registered bounded resource budgets")
    model = config.get("reference_model", {})
    if any(
        not math.isfinite(model.get(field, 0)) or model.get(field, 0) <= 0
        for field in ("cpu_work_units_per_second", "io_bytes_per_second")
    ):
        raise ValueError("positive fixed reference capacities are required")
    if set(model.get("operation_weights", {})) != set(OPERATION_TYPES) or any(
        value != 1 for value in model["operation_weights"].values()
    ):
        raise ValueError("reference model v1 requires unit logical-operation weights")
    # Each worker process receives exactly one dataset and a private artifact root.
    count = len(ACTIVE_DATASETS)
    config["dataset_scales"] = list(ACTIVE_DATASETS)
    config["evaluation_design"].update(
        expected_queries=EVALUATION_PER_DATASET * count,
        expected_clusters=CLUSTERS_PER_DATASET * count,
        expected_method_query_runs=EVALUATION_PER_DATASET * count * len(METHOD_IDS),
        expected_forced_plan_runs=EVALUATION_PER_DATASET * count * len(PLAN_IDS),
    )
    config["calibration_design"]["expected_queries"] = CALIBRATION_PER_DATASET * count
    config["scalability_evaluation"]["distinct_queries"] = 60 * count
    return config


def require_method(method_id: str) -> str:
    """Reject methods not registered for the current study."""
    if method_id not in METHOD_IDS:
        raise ValueError(f"unregistered method: {method_id}")
    return method_id
