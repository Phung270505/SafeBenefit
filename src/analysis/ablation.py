"""Outcome-blind SafeBenefit decisions with post-hoc forced-plan scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

from src.plans.cost import PlanStatistics, estimate_plan_wire_bytes

ABLATION_VARIANTS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6")


def select_ablation_plan(
    variant: str,
    current_plan: int,
    predicted_bytes: Mapping[int, float],
    *,
    switch_bytes: float,
    safety_margin: float,
    uncertainty_z: float,
    prediction_relative_std: float,
) -> int:
    """Select a plan under one preregistered ablation decision rule."""
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"unknown ablation variant: {variant}")
    costs = {int(plan): float(value) for plan, value in predicted_bytes.items()}
    if (
        current_plan not in costs
        or any(not math.isfinite(value) or value < 0 for value in costs.values())
        or not math.isfinite(switch_bytes)
        or switch_bytes < 0
        or not 0 <= safety_margin <= 1
        or uncertainty_z < 0
        or prediction_relative_std < 0
    ):
        raise ValueError("invalid SafeBenefit ablation input")
    candidate = min(costs, key=lambda plan: (costs[plan], plan))
    if candidate == current_plan:
        return current_plan
    gross = costs[current_plan] - costs[candidate]
    margin = safety_margin * max(costs[current_plan], 1.0)
    uncertainty = uncertainty_z * math.hypot(
        costs[current_plan] * prediction_relative_std,
        costs[candidate] * prediction_relative_std,
    )
    required = {
        "A0": -math.inf,
        "A1": 0.0,
        "A2": switch_bytes,
        "A3": margin,
        "A4": switch_bytes + margin,
        "A5": switch_bytes + uncertainty,
        "A6": switch_bytes + margin + uncertainty,
    }[variant]
    return candidate if gross > required else current_plan


def analyze_safebenefit_ablation(
    main_rows: Iterable[Mapping[str, Any]],
    *,
    safety_margin: float = 0.10,
    uncertainty_z: float = 1.96,
    prediction_relative_std: float = 0.05,
) -> dict[str, Any]:
    """Score A0-A6 from STATIC contexts and forced-plan outcomes."""
    static_rows = [row for row in main_rows if row.get("method") == "STATIC"]
    if not static_rows:
        raise ValueError("STATIC rows are required for ablation analysis")
    totals: dict[str, float] = defaultdict(float)
    counters = {
        variant: defaultdict(int) for variant in ABLATION_VARIANTS
    }
    for row in static_rows:
        current = int(row["initial_plan"])
        forced = {
            int(plan): float(value)
            for plan, value in row["forced_plan_wire_bytes"].items()
        }
        switch = float(row["estimated_switch_wire_bytes"])
        predicted = (
            {
                int(plan): float(value)
                for plan, value in row["predicted_plan_wire_bytes"].items()
            }
            if "predicted_plan_wire_bytes" in row
            else estimate_plan_wire_bytes(
                PlanStatistics(
                    float(row["selected_rows"]),
                    float(row["distinct_groups"]),
                    float(row["mean_label_bytes"]),
                    int(row["workers"]),
                )
            )
        )
        oracle = min(
            forced,
            key=lambda plan: (forced[plan] + (switch if plan != current else 0.0), plan),
        )
        oracle_cost = forced[oracle] + (switch if oracle != current else 0.0)
        optimal = {
            plan for plan in forced
            if forced[plan] + (switch if plan != current else 0.0) == oracle_cost
        }
        for variant in ABLATION_VARIANTS:
            selected = select_ablation_plan(
                variant,
                current,
                predicted,
                switch_bytes=switch,
                safety_margin=safety_margin,
                uncertainty_z=uncertainty_z,
                prediction_relative_std=prediction_relative_std,
            )
            actual = forced[selected] + (switch if selected != current else 0.0)
            totals[variant] += actual
            counters[variant]["queries"] += 1
            counters[variant]["switches"] += selected != current
            counters[variant]["oracle_matches"] += selected in optimal
            counters[variant]["unnecessary_switches"] += selected != current and current in optimal
            counters[variant]["missed_beneficial_switches"] += selected == current and current not in optimal
            counters[variant]["wrong_plan"] += selected not in optimal
            counters[variant]["regret_bytes"] += int(round(actual - oracle_cost))
    return {
        "schema_version": 1,
        "source": "post-hoc forced-plan scoring; decisions use safe-point statistics only",
        "variants": {
            variant: {
                "total_wire_bytes": int(round(totals[variant])),
                **dict(counters[variant]),
                "oracle_match_rate": counters[variant]["oracle_matches"]
                / counters[variant]["queries"],
            }
            for variant in ABLATION_VARIANTS
        },
    }


def analyze_parameter_sensitivity(
    main_rows: Iterable[Mapping[str, Any]],
    *,
    safety_margins: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20),
    switch_estimates: tuple[float, ...] = (0.0, 256.0, 512.0, 768.0, 1024.0, 2048.0),
    relative_stds: tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.10),
    z_scores: tuple[float, ...] = (0.0, 1.0, 1.645, 1.96, 2.576),
) -> dict[str, Any]:
    """Evaluate the preregistered A6 parameter grid without retuning the method."""
    static_rows = [row for row in main_rows if row.get("method") == "STATIC"]
    if not static_rows:
        raise ValueError("STATIC rows are required for sensitivity analysis")
    results: list[dict[str, Any]] = []
    for margin in safety_margins:
        for switch in switch_estimates:
            for relative_std in relative_stds:
                for z_score in z_scores:
                    total = 0.0
                    regret = 0.0
                    switches = 0
                    for row in static_rows:
                        current = int(row["initial_plan"])
                        forced = {
                            int(plan): float(value)
                            for plan, value in row["forced_plan_wire_bytes"].items()
                        }
                        predicted = {
                            int(plan): float(value)
                            for plan, value in row["predicted_plan_wire_bytes"].items()
                        }
                        selected = select_ablation_plan(
                            "A6",
                            current,
                            predicted,
                            switch_bytes=switch,
                            safety_margin=margin,
                            uncertainty_z=z_score,
                            prediction_relative_std=relative_std,
                        )
                        actual = forced[selected] + (switch if selected != current else 0.0)
                        oracle = min(
                            forced,
                            key=lambda plan: (
                                forced[plan] + (switch if plan != current else 0.0),
                                plan,
                            ),
                        )
                        oracle_cost = forced[oracle] + (switch if oracle != current else 0.0)
                        total += actual
                        regret += actual - oracle_cost
                        switches += selected != current
                    results.append(
                        {
                            "safety_margin": margin,
                            "switch_estimate_bytes": switch,
                            "prediction_relative_std": relative_std,
                            "uncertainty_z": z_score,
                            "total_wire_bytes": int(round(total)),
                            "regret_bytes": int(round(regret)),
                            "switches": switches,
                        }
                    )
    return {
        "schema_version": 1,
        "interpretation": "post-hoc sensitivity only; not used to tune evaluation policy",
        "configurations": results,
    }
