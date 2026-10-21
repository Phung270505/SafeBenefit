"""Adaptive forced-plan oracle used only for post-hoc evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AdaptiveOracleDecision:
    """Best plan within the measured candidate set after switching overhead."""

    plan_id: int
    base_plan_wire_bytes: float
    effective_wire_bytes: float
    optimal_plan_ids: tuple[int, ...] = ()


def choose_adaptive_oracle(
    forced_plan_wire_bytes: Mapping[int, float],
    current_plan: int,
    switch_bytes: float,
) -> AdaptiveOracleDecision:
    """Minimize measured branch bytes plus switch cost from the current plan."""
    costs = {int(plan): float(value) for plan, value in forced_plan_wire_bytes.items()}
    if (
        current_plan not in costs
        or len(costs) < 2
        or any(not math.isfinite(value) or value < 0 for value in costs.values())
        or not math.isfinite(switch_bytes)
        or switch_bytes < 0
    ):
        raise ValueError("invalid adaptive-oracle input")
    effective = {
        plan: value + (switch_bytes if plan != current_plan else 0.0)
        for plan, value in costs.items()
    }
    plan = min(effective, key=lambda candidate: (effective[candidate], candidate))
    tied = tuple(sorted(candidate for candidate in effective if effective[candidate] == effective[plan]))
    return AdaptiveOracleDecision(plan, costs[plan], effective[plan], tied)


def oracle_diagnostics(
    selected_plan: int,
    current_plan: int,
    actual_wire_bytes: float,
    oracle: AdaptiveOracleDecision,
) -> dict[str, float | bool]:
    """Return regret and switch-error diagnostics for one deployable method."""
    if not math.isfinite(actual_wire_bytes) or actual_wire_bytes < oracle.effective_wire_bytes:
        raise ValueError("actual cost cannot be below the candidate-set oracle")
    regret = actual_wire_bytes - oracle.effective_wire_bytes
    optimal = oracle.optimal_plan_ids or (oracle.plan_id,)
    return {
        "regret_bytes": regret,
        "normalized_regret": regret / max(oracle.effective_wire_bytes, 1.0),
        "oracle_match": selected_plan in optimal and regret == 0,
        "unnecessary_switch": selected_plan != current_plan
        and current_plan in optimal,
        "missed_beneficial_switch": selected_plan == current_plan
        and current_plan not in optimal,
        "wrong_plan": selected_plan not in optimal,
    }
