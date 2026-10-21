"""Calibration-selected static physical plan."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from src.methods.base import PlanDecision, PlanSelector, validate_candidates


class StaticSelector(PlanSelector):
    """Choose one dataset/query-class plan during calibration and never switch."""

    method_id = "STATIC"

    def __init__(self) -> None:
        self.initial_plans: dict[str, int] = {}

    def fit(self, calibration_rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
        totals: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for row in calibration_rows:
            key = f"{row['dataset_id']}:{row['query_class']}"
            for plan, cost in row["costs"].items():
                totals[key][int(plan)] += float(cost)
        self.initial_plans = {
            key: min(costs, key=lambda plan: (costs[plan], plan))
            for key, costs in totals.items()
        }
        return {"method_id": self.method_id, "initial_plans": dict(self.initial_plans)}

    def select_plan(
        self, context: Mapping[str, Any], valid_plans: Sequence[int]
    ) -> PlanDecision:
        plans = validate_candidates(context, valid_plans)
        key = f"{context['dataset_id']}:{context['query_class']}"
        plan = self.initial_plans.get(key, int(context["current_plan"]))
        if plan not in plans:
            plan = int(context["current_plan"])
        return PlanDecision(self.method_id, plan, int(context["current_plan"]), "static_locked")
