"""Preregistered paired-cluster analysis for the SafeBenefit study."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from src.common.config import METHOD_IDS, load_study_config
from src.common.seed import component_seed
from src.datasets.discrimination import count_action_disagreements

METHODS = METHOD_IDS
COMPARATORS = tuple(method for method in METHODS if method != "SAFE_BENEFIT")


def _cluster_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["dataset_id"]),
        str(row["query_class"]),
        str(row["graph_version_bucket"]),
        int(row["schedule_id"]),
    )


def _cluster_method_totals(
    rows: list[dict[str, Any]],
) -> dict[tuple[Any, ...], dict[str, float]]:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row.get("status") != "PASS":
            raise ValueError(f"failed main query cannot be ranked: {row.get('query_id')}")
        grouped[_cluster_key(row)][str(row["method"])].append(
            float(row["total_wire_bytes"])
        )
    output: dict[tuple[Any, ...], dict[str, float]] = {}
    for cluster, methods in grouped.items():
        if set(methods) != set(METHODS) or any(
            len(values) != 2 for values in methods.values()
        ):
            raise ValueError(f"incomplete paired cluster: {cluster}")
        output[cluster] = {
            method: float(np.mean(values)) for method, values in methods.items()
        }
    return output


def _bootstrap_fraction(
    clusters: dict[tuple[Any, ...], dict[str, float]],
    comparator: str,
    resamples: int,
) -> np.ndarray:
    keys = sorted(clusters)
    random = np.random.default_rng(
        component_seed("bootstrap_statistics", f"paired:{keys[0][0]}:{comparator}")
    )
    benefit = np.asarray([clusters[key]["SAFE_BENEFIT"] for key in keys])
    baseline = np.asarray([clusters[key][comparator] for key in keys])
    sampled = random.integers(
        0,
        len(keys),
        size=(resamples, len(keys)),
        dtype=np.int32,
    )
    benefit_totals = np.sum(benefit[sampled], axis=1)
    baseline_totals = np.sum(baseline[sampled], axis=1)
    return (benefit_totals - baseline_totals) / np.maximum(baseline_totals, 1.0)


def _randomization_pvalue(
    differences: list[float], scope: str, repetitions: int
) -> float:
    observed = abs(float(np.mean(differences)))
    random = np.random.default_rng(
        component_seed("bootstrap_statistics", f"randomization:{scope}")
    )
    extreme = 0
    values = np.asarray(differences)
    batch_size = 1_000
    for offset in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - offset)
        signs = random.integers(
            0, 2, size=(size, len(values)), dtype=np.int8
        ).astype(np.float64)
        signs = signs * 2.0 - 1.0
        statistics = np.abs(np.mean(signs * values, axis=1))
        extreme += int(np.count_nonzero(statistics >= observed))
    return (extreme + 1) / (repetitions + 1)


def _holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def analyze_results(
    rows: list[dict[str, Any]], *, resamples: int = 10_000, rope: float = 0.0025
) -> dict[str, Any]:
    """Compare SafeBenefit with all comparators using the registered dataset-specific paired clusters."""
    if resamples <= 0 or not 0 <= rope < 1:
        raise ValueError("invalid statistical configuration")
    clusters = _cluster_method_totals(rows)
    datasets = {row["dataset_id"] for row in rows}
    if len(datasets) != 1:
        raise ValueError("statistics must be evaluated separately for each dataset")
    expected = load_study_config()["per_dataset_design"]["clusters"]
    if len(clusters) != expected:
        raise ValueError(f"expected {expected} paired clusters, observed {len(clusters)}")
    comparisons: dict[str, dict[str, Any]] = {}
    pvalues: dict[str, float] = {}
    for comparator in COMPARATORS:
        fractions = _bootstrap_fraction(clusters, comparator, resamples)
        lower, upper = (float(np.quantile(fractions, q)) for q in (0.025, 0.975))
        keys = sorted(clusters)
        differences = [
            clusters[key]["SAFE_BENEFIT"] - clusters[key][comparator] for key in keys
        ]
        pvalues[comparator] = _randomization_pvalue(
            differences, f"{keys[0][0]}:{comparator}", resamples
        )
        wins = sum(
            difference < -rope * clusters[key][comparator]
            for key, difference in zip(keys, differences, strict=True)
        )
        losses = sum(
            difference > rope * clusters[key][comparator]
            for key, difference in zip(keys, differences, strict=True)
        )
        if upper < -rope:
            conclusion = "SAFEBENEFIT_WIN"
        elif lower > rope:
            conclusion = "SAFEBENEFIT_LOSS"
        elif lower >= -rope and upper <= rope:
            conclusion = "PRACTICALLY_EQUIVALENT"
        else:
            conclusion = "INCONCLUSIVE"
        benefit_total = sum(value["SAFE_BENEFIT"] for value in clusters.values())
        baseline_total = sum(value[comparator] for value in clusters.values())
        comparisons[comparator] = {
            "saving_fraction": (baseline_total - benefit_total) / baseline_total,
            "difference_fraction_ci95": [lower, upper],
            "win_tie_loss_clusters": [wins, len(keys) - wins - losses, losses],
            "raw_randomization_p": pvalues[comparator],
            "conclusion": conclusion,
        }
    adjusted = _holm_adjust(pvalues)
    for comparator in COMPARATORS:
        comparisons[comparator]["holm_adjusted_p"] = adjusted[comparator]
    totals = {
        method: int(
            sum(float(row["total_wire_bytes"]) for row in rows if row["method"] == method)
        )
        for method in METHODS
    }
    regret = {
        method: {
            "total_bytes": int(
                sum(float(row["regret_bytes"]) for row in rows if row["method"] == method)
            ),
            "mean_bytes": float(
                np.mean(
                    [
                        float(row["regret_bytes"])
                        for row in rows
                        if row["method"] == method
                    ]
                )
            ),
            "zero_regret_queries": sum(
                int(row["regret_bytes"]) == 0
                for row in rows
                if row["method"] == method
            ),
            "mean_normalized_regret": float(
                np.mean(
                    [
                        float(row["normalized_regret"])
                        for row in rows
                        if row["method"] == method
                    ]
                )
            ),
            "oracle_matches": sum(
                bool(row["oracle_match"])
                for row in rows
                if row["method"] == method
            ),
            "unnecessary_switches": sum(
                bool(row["unnecessary_switch"])
                for row in rows
                if row["method"] == method
            ),
            "missed_beneficial_switches": sum(
                bool(row["missed_beneficial_switch"])
                for row in rows
                if row["method"] == method
            ),
            "wrong_plan_queries": sum(
                bool(row["wrong_plan"])
                for row in rows
                if row["method"] == method
            ),
        }
        for method in METHODS
    }
    latency = {}
    for method in METHODS:
        values = [
            float(row["decision_latency_ns"]) / 1000.0
            for row in rows
            if row["method"] == method
        ]
        latency[method] = {
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "maximum": max(values),
        }
    return {
        "schema_version": 2,
        "study_id": "SAFEBENEFIT-DGQ",
        "paired_parameter_clusters": len(clusters),
        "dataset_id": next(iter(datasets)),
        "cluster_scope": "paired parameter groups, not independent graph histories",
        "replicas_per_cluster": 2,
        "bootstrap_resamples": resamples,
        "bootstrap_rng": "NumPy PCG64 derived from master seed 42",
        "practical_equivalence_fraction": rope,
        "total_wire_bytes": totals,
        "forced_plan_regret": regret,
        "action_disagreement_queries": count_action_disagreements(rows),
        "action_diversity_policy": "diagnostic_only; identical decisions are valid",
        "lowest_cost_methods": [method for method in METHODS if totals[method] == min(totals.values())],
        "lowest_cost_method": (
            next(method for method in METHODS if totals[method] == min(totals.values()))
            if sum(total == min(totals.values()) for total in totals.values()) == 1 else None
        ),
        "decision_latency_us": latency,
        "decision_time_scope": "calculated reference-model CPU service, not host wall time",
        "comparisons": comparisons,
        "all_values_finite": all(
            math.isfinite(value)
            for comparison in comparisons.values()
            for value in comparison["difference_fraction_ci95"]
        ),
    }
