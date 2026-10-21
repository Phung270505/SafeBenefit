"""Diagnostics for plan-cost and switching-cost estimates."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping

from src.analysis.oracle import choose_adaptive_oracle


def _skew_bucket(cv: float) -> str:
    if cv < 0.10:
        return "BALANCED"
    if cv < 0.50:
        return "MODERATE"
    if cv < 1.00:
        return "HIGH"
    return "EXTREME"


def analyze_cost_model(main_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure prediction error, wrong decisions and switching-cost fidelity."""
    rows = list(main_rows)
    static_rows = [row for row in rows if row.get("method") == "STATIC"]
    if not static_rows:
        raise ValueError("STATIC rows are required for cost-model analysis")

    errors_by_plan: dict[int, list[float]] = defaultdict(list)
    errors_by_skew: dict[str, list[float]] = defaultdict(list)
    predicted_wrong = 0
    predicted_regret = 0.0
    for row in static_rows:
        predicted = {
            int(plan): float(value)
            for plan, value in row["predicted_plan_wire_bytes"].items()
        }
        actual = {
            int(plan): float(value)
            for plan, value in row["forced_plan_wire_bytes"].items()
        }
        skew = _skew_bucket(float(row["partition_skew_cv"]))
        for plan in sorted(actual):
            ape = abs(predicted[plan] - actual[plan]) / max(actual[plan], 1.0)
            errors_by_plan[plan].append(ape)
            errors_by_skew[skew].append(ape)
        current = int(row["initial_plan"])
        switch = float(row["estimated_switch_wire_bytes"])
        selected = min(predicted, key=lambda plan: (predicted[plan], plan))
        oracle = choose_adaptive_oracle(actual, current, switch)
        predicted_wrong += selected != oracle.plan_id
        selected_actual = actual[selected] + (switch if selected != current else 0.0)
        predicted_regret += selected_actual - oracle.effective_wire_bytes

    switched = [
        row for row in rows if int(row["plan_id"]) != int(row["initial_plan"])
    ]
    switch_absolute_errors = [
        abs(float(row["switch_bytes"]) - float(row["estimated_switch_wire_bytes"]))
        for row in switched
    ]
    switch_percentage_errors = [
        error / max(float(row["switch_bytes"]), 1.0)
        for row, error in zip(switched, switch_absolute_errors, strict=True)
    ]
    return {
        "schema_version": 1,
        "queries": len(static_rows),
        "plan_prediction": {
            str(plan): {
                "mean_absolute_percentage_error": mean(values),
                "observations": len(values),
            }
            for plan, values in sorted(errors_by_plan.items())
        },
        "skew_prediction": {
            bucket: {
                "mean_absolute_percentage_error": mean(values),
                "observations": len(values),
            }
            for bucket, values in sorted(errors_by_skew.items())
        },
        "predicted_best_wrong_plan_queries": predicted_wrong,
        "predicted_best_regret_bytes": int(round(predicted_regret)),
        "switch_cost_validation": {
            "switches": len(switched),
            "mean_absolute_error_bytes": mean(switch_absolute_errors)
            if switch_absolute_errors
            else 0.0,
            "mean_absolute_percentage_error": mean(switch_percentage_errors)
            if switch_percentage_errors
            else 0.0,
        },
    }
