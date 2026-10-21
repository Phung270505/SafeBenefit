"""SafeBenefit: benefit-aware transfer-plan adaptation with hard guards."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from src.methods.base import PlanDecision, PlanSelector, validate_candidates


class SafeBenefitSelector(PlanSelector):
    method_id = "SAFE_BENEFIT"

    def __init__(self, safety_margin: float = 0.10, uncertainty_z: float = 1.96) -> None:
        if (
            not math.isfinite(safety_margin)
            or not 0 <= safety_margin <= 1
            or not math.isfinite(uncertainty_z)
            or uncertainty_z < 0
        ):
            raise ValueError("invalid SafeBenefit decision parameters")
        self.safety_margin = float(safety_margin)
        self.uncertainty_z = float(uncertainty_z)

    def fit(self, calibration_rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
        return {
            "method_id": self.method_id,
            "safety_margin": self.safety_margin,
            "uncertainty_z": self.uncertainty_z,
        }

    def select_plan(
        self, context: Mapping[str, Any], valid_plans: Sequence[int]
    ) -> PlanDecision:
        plans = validate_candidates(context, valid_plans)
        current = int(context["current_plan"])
        if not bool(context.get("safe_point", True)):
            return PlanDecision(self.method_id, current, current, "not_safe_point")
        if int(context.get("switch_budget", 1)) <= 0:
            return PlanDecision(self.method_id, current, current, "switch_budget_exhausted")
        if float(context.get("progress", 0.0)) >= 0.90:
            return PlanDecision(self.method_id, current, current, "near_completion")
        state_version = context.get("graph_version")
        statistics_version = context.get("statistics_graph_version", state_version)
        if state_version is not None and statistics_version != state_version:
            return PlanDecision(self.method_id, current, current, "stale_statistics")
        costs = {plan: float(context["predicted_bytes"][plan]) for plan in plans}
        candidate = min(plans, key=lambda plan: (costs[plan], plan))
        uncertainty = context.get("prediction_std", {})
        penalty = self.uncertainty_z * math.hypot(
            float(uncertainty.get(current, 0.0)), float(uncertainty.get(candidate, 0.0))
        )
        switch_bytes = float(context.get("switch_bytes", 0.0)) if candidate != current else 0.0
        net = costs[current] - costs[candidate] - switch_bytes - penalty
        selected = candidate if net > self.safety_margin * max(costs[current], 1.0) else current
        reason = "positive_lower_bound" if selected != current else "insufficient_net_benefit"
        return PlanDecision(self.method_id, selected, current, reason)
