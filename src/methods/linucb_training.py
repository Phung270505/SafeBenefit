"""Calibration-only training for the regularized LinUCB transfer policy."""

from __future__ import annotations

import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from src.common.design import STUDY_SCHEMA_VERSION
from src.common.files import sha256_file, write_json
from src.common.seed import component_seed
from src.methods.linucb import (
    FeatureScaler,
    LinearPolicy,
    export_policy,
    fit_policy,
    model_features,
    select_action,
)
from src.plans.cost import (
    PlanStatistics,
    estimate_plan_wire_bytes,
    estimated_compression_ratio,
)
from src.queries.registry import QUERY_CLASS_IDS, canonical_query_class

BYTES_REFERENCE = 1_000_000.0


def effective_cost(sample: dict[str, Any], action: int, initial_plan: int) -> int:
    """Return measured branch bytes plus the registered plan-switch cost."""
    if action not in (0, 1, 2) or initial_plan not in (0, 1, 2):
        raise ValueError("plans must be 0, 1 or 2")
    measured = int(sample["costs"][str(action)])
    switch = int(sample["switch_bytes"]) if action != initial_plan else 0
    return measured + switch


def choose_initial_plans(samples: list[dict[str, Any]]) -> dict[str, int]:
    """Choose each initial plan using calibration-train rows only."""
    totals: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        key = f"{sample['dataset_id']}:{sample['query_class']}"
        for action in (0, 1, 2):
            totals[key][action].append(int(sample["costs"][str(action)]))
    return {
        key: min(
            (0, 1, 2),
            key=lambda action: (median(costs[action]), action),
        )
        for key, costs in totals.items()
    }


def benefit_fallback_action(
    sample: dict[str, Any],
    initial_plan: int,
    *,
    workers: int = 4,
    safety_margin: float = 0.10,
    uncertainty_z: float = 1.96,
    prediction_relative_std: float = 0.05,
) -> int:
    """Mirror the C++ SafeBenefit fallback using committed safe-point statistics."""
    statistics = PlanStatistics(
        float(sample["selected_rows"]),
        float(sample["distinct_groups"]),
        float(sample["mean_label_bytes"]),
        int(sample.get("worker_count", workers)),
    )
    estimated = (
        {int(plan): float(value) for plan, value in sample["predicted_bytes"].items()}
        if "predicted_bytes" in sample
        else estimate_plan_wire_bytes(statistics)
    )
    best = min(estimated, key=lambda action: (estimated[action], action))
    saving = estimated[initial_plan] - estimated[best]
    uncertainty = uncertainty_z * math.hypot(
        estimated[initial_plan] * prediction_relative_std,
        estimated[best] * prediction_relative_std,
    )
    required = (
        float(sample["switch_bytes"])
        + uncertainty
        + safety_margin * estimated[initial_plan]
    )
    return initial_plan if saving <= required else best


def context_vector(sample: dict[str, Any], initial_plan: int) -> np.ndarray:
    """Build the registered LinUCB predictor context."""
    statistics = PlanStatistics(
        float(sample["selected_rows"]),
        float(sample["distinct_groups"]),
        float(sample["mean_label_bytes"]),
        int(sample.get("worker_count", 4)),
    )
    query_class = canonical_query_class(str(sample["query_class"]))
    return model_features(
        statistics.selected_rows,
        statistics.distinct_groups,
        statistics.mean_label_bytes,
        initial_plan,
        compression_ratio=estimated_compression_ratio(statistics),
        query_class_id=QUERY_CLASS_IDS[query_class],
        graph_scale_edges=float(sample.get("graph_scale_edges", 0)),
        graph_version=float(sample.get("graph_version", 0)),
        worker_count=statistics.worker_count,
        retry_count=int(sample.get("retry_count", 0)),
        switch_bytes=float(sample.get("switch_bytes", 0)),
    )


def validate_policy(
    policy: LinearPolicy,
    rows: list[dict[str, Any]],
    scaler: FeatureScaler,
    initial_plans: dict[str, int],
    ood_threshold: float,
) -> tuple[int, list[dict[str, Any]], int]:
    """Score a policy only on calibration-validation rows."""
    total = 0
    decisions: list[dict[str, Any]] = []
    fallbacks = 0
    for sample in rows:
        key = f"{sample['dataset_id']}:{sample['query_class']}"
        initial = initial_plans[key]
        context = context_vector(sample, initial)
        if scaler.mahalanobis_distance(context) > ood_threshold:
            action = benefit_fallback_action(sample, initial)
            fallback = True
            fallbacks += 1
        else:
            action = select_action(policy, scaler.transform(context))
            fallback = False
        cost = effective_cost(sample, action, initial)
        total += cost
        decisions.append(
            {
                "query_id": sample["query_id"],
                "plan": action,
                "wire_bytes": cost,
                "fallback": fallback,
            }
        )
    return total, decisions, fallbacks


def train_linucb(
    samples: list[dict[str, Any]],
    output: Path,
    *,
    ridge_candidates: tuple[float, ...] = (0.1, 0.01, 0.001),
    ucb_alpha: float = 0.1,
    ood_multiplier: float = 1.5,
    minimum_ood_threshold: float = 4.0,
    seed_component: str = "policy_bootstrap",
) -> dict[str, Any]:
    """Fit candidates on train rows and select exclusively by validation bytes."""
    if samples and len({sample["dataset_id"] for sample in samples}) != 1:
        raise ValueError("LinUCB training must contain exactly one dataset")
    if not samples or not ridge_candidates:
        raise ValueError("invalid registered LinUCB training input")
    if any(set(sample["costs"]) != {"0", "1", "2"} for sample in samples):
        raise ValueError("branch-complete calibration samples are required")
    if any(not math.isfinite(value) or value <= 0 for value in ridge_candidates):
        raise ValueError("ridge candidates must be positive and finite")
    if not math.isfinite(ucb_alpha) or ucb_alpha < 0:
        raise ValueError("UCB alpha must be non-negative and finite")
    if not math.isfinite(ood_multiplier) or ood_multiplier <= 0:
        raise ValueError("OOD multiplier must be positive and finite")
    if not math.isfinite(minimum_ood_threshold) or minimum_ood_threshold <= 0:
        raise ValueError("minimum OOD threshold must be positive and finite")
    split_indices = [
        int(sample.get("replica_id", sample.get("local_index", -1)))
        for sample in samples
    ]
    if any(index not in (0, 1) for index in split_indices):
        raise ValueError("every calibration sample must belong to split 0 or 1")
    train_rows = [sample for sample, index in zip(samples, split_indices, strict=True) if index == 0]
    validation_rows = [
        sample for sample, index in zip(samples, split_indices, strict=True) if index == 1
    ]
    if len(train_rows) + len(validation_rows) != len(samples):
        raise AssertionError("calibration split silently dropped samples")
    if not train_rows or not validation_rows:
        raise ValueError("calibration local_index 0/1 split is incomplete")

    initial_plans = choose_initial_plans(train_rows)
    contexts = [
        context_vector(
            sample,
            initial_plans[f"{sample['dataset_id']}:{sample['query_class']}"],
        )
        for sample in train_rows
    ]
    scaler = FeatureScaler.fit(contexts)
    train_distances = [scaler.mahalanobis_distance(context) for context in contexts]
    ood_threshold = max(
        minimum_ood_threshold,
        float(np.quantile(train_distances, 0.99)) * ood_multiplier,
    )

    output.mkdir(parents=True, exist_ok=True)
    candidate_reports: list[dict[str, Any]] = []
    candidate_paths: list[Path] = []
    training_seeds = tuple(
        component_seed(seed_component, f"{next(iter({row['dataset_id'] for row in samples}))}:replica-{replica}")
        for replica in range(3)
    )
    index = 0
    for training_seed in training_seeds:
        generator = np.random.default_rng(training_seed)
        sampled_indices = generator.integers(0, len(train_rows), size=len(train_rows))
        sampled_rows = [train_rows[int(row)] for row in sampled_indices]
        sampled_contexts = [
            context_vector(
                sample,
                initial_plans[f"{sample['dataset_id']}:{sample['query_class']}"],
            )
            for sample in sampled_rows
        ]
        rewards: dict[int, list[float]] = {0: [], 1: [], 2: []}
        for action in (0, 1, 2):
            for sample in sampled_rows:
                key = f"{sample['dataset_id']}:{sample['query_class']}"
                reward = -effective_cost(sample, action, initial_plans[key]) / BYTES_REFERENCE
                rewards[action].append(reward)
        for ridge in ridge_candidates:
            policy = fit_policy(
                sampled_contexts,
                rewards,
                scaler,
                ridge=ridge,
                alpha=ucb_alpha,
            )
            validation_total, decisions, fallbacks = validate_policy(
                policy, validation_rows, scaler, initial_plans, ood_threshold
            )
            path = output / f"candidate-{index}.tsv"
            checksum = export_policy(
                policy, path, scaler, ridge=ridge, ood_threshold=ood_threshold
            )
            candidate_paths.append(path)
            candidate_reports.append(
                {
                    "candidate": index,
                    "training_seed": training_seed,
                    "ridge": ridge,
                    "validation_wire_bytes": validation_total,
                    "validation_fallbacks": fallbacks,
                    "actions": decisions,
                    "policy_sha256": checksum,
                }
            )
            index += 1

    selected = min(
        candidate_reports,
        key=lambda row: (row["validation_wire_bytes"], row["candidate"]),
    )
    candidate_policy = output / "candidate_policy.tsv"
    deployment_policy = output / "policy.tsv"
    shutil.copyfile(candidate_paths[int(selected["candidate"])], candidate_policy)
    shutil.copyfile(candidate_policy, deployment_policy)

    validation_static = 0
    validation_benefit = 0
    validation_oracle = 0
    for sample in validation_rows:
        key = f"{sample['dataset_id']}:{sample['query_class']}"
        initial = initial_plans[key]
        validation_static += effective_cost(sample, initial, initial)
        validation_benefit += effective_cost(
            sample,
            benefit_fallback_action(sample, initial),
            initial,
        )
        validation_oracle += min(
            effective_cost(sample, action, initial) for action in (0, 1, 2)
        )
    gate_passed = int(selected["validation_wire_bytes"]) <= validation_benefit

    report = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "method": "LINUCB",
        "algorithm": "three-action disjoint LinUCB with Mahalanobis OOD and hard action mask",
        "decision_process": "one_step",
        "master_seed": 42,
        "ridge_candidates": list(ridge_candidates),
        "ucb_alpha": ucb_alpha,
        "selected_candidate": int(selected["candidate"]),
        "selected_training_seed": int(selected["training_seed"]),
        "selected_ridge": float(selected["ridge"]),
        "selection_rule": "minimum validation bytes; registered order breaks ties",
        "initial_plans": initial_plans,
        "train_queries": len(train_rows),
        "train_branch_observations": len(train_rows) * 3,
        "training_seeds": list(training_seeds),
        "validation_queries": len(validation_rows),
        "validation_wire_bytes": {
            "candidate_rl": int(selected["validation_wire_bytes"]),
            "deployed_rl": int(selected["validation_wire_bytes"]),
            "STATIC": validation_static,
            "SAFE_BENEFIT": validation_benefit,
            "branch_oracle_lower_bound": validation_oracle,
        },
        "ood_threshold": ood_threshold,
        "validation_noninferiority_gate_passed": gate_passed,
        "deployment_mode": "learned_policy_with_per_query_OOD_fallback",
        "candidate_reports": candidate_reports,
        "candidate_policy_sha256": sha256_file(candidate_policy),
        "deployment_policy_sha256": sha256_file(deployment_policy),
        "frozen_policy_sha256": sha256_file(deployment_policy),
        "scientific_evaluation_policy": "candidate_policy.tsv",
        "deployment_policy": "policy.tsv",
        "evaluation_updates": False,
        "online_updates_during_evaluation": False,
        "exploration_during_evaluation": 0,
        "status": "PASS",
    }
    write_json(output / "training.json", report)
    return report
