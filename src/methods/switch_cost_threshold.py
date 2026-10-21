"""Strong threshold baseline that explicitly charges plan-switch bytes."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from src.methods.base import PlanDecision, PlanSelector, validate_candidates
from src.methods.threshold_calibration import SwitchThresholdModel, select_switch_threshold


class SwitchCostThresholdSelector(PlanSelector):
    """Switch when gross saving exceeds switch cost plus a relative threshold."""

    method_id = "SWITCH_COST_THRESHOLD"

    def __init__(self, threshold: float | None = None) -> None:
        if threshold is not None and (
            not math.isfinite(threshold) or not 0 <= threshold <= 1
        ):
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = None if threshold is None else float(threshold)

    def fit(self, calibration_rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
        if self.threshold is None:
            rows = list(calibration_rows)
            if any(int(row.get("replica_id", 0)) not in (0, 1) for row in rows):
                raise ValueError("calibration replica must be train 0 or validation 1")
            cases = [
                dict(row) if "actual_bytes" in row else dict(row, actual_bytes=row["costs"])
                for row in rows if int(row.get("replica_id", 0)) == 0
            ]
            model = SwitchThresholdModel.fit(cases)
            self.threshold = model.threshold
        return {"method_id": self.method_id, "threshold": self.threshold}

    def select_plan(
        self, context: Mapping[str, Any], valid_plans: Sequence[int]
    ) -> PlanDecision:
        plans = validate_candidates(context, valid_plans)
        if self.threshold is None:
            raise RuntimeError(
                "SWITCH_COST_THRESHOLD must be fitted before evaluation"
            )
        current = int(context["current_plan"])
        costs = {plan: float(context["predicted_bytes"][plan]) for plan in plans}
        selected = select_switch_threshold(
            current, costs, float(context.get("switch_bytes", 0.0)), self.threshold
        )
        return PlanDecision(
            self.method_id,
            selected,
            current,
            "switch_cost_threshold_met"
            if selected != current
            else "switch_cost_threshold_not_met",
        )
