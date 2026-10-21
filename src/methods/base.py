"""Common plan-selection contract used by every registered method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PlanDecision:
    """One deterministic choice made at a committed safe point."""

    method_id: str
    plan_id: int
    current_plan: int
    reason: str


class PlanSelector(ABC):
    """Stable public interface for calibration and frozen evaluation."""

    method_id: str

    @abstractmethod
    def fit(self, calibration_rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Fit using calibration rows only and return a serializable artifact."""

    @abstractmethod
    def select_plan(
        self, context: Mapping[str, Any], valid_plans: Sequence[int]
    ) -> PlanDecision:
        """Select one feasible plan without observing its future outcome."""


def validate_candidates(context: Mapping[str, Any], valid_plans: Sequence[int]) -> tuple[int, ...]:
    """Validate the shared decision context and return normalized candidates."""
    plans = tuple(int(plan) for plan in valid_plans)
    if not plans or len(set(plans)) != len(plans) or any(plan not in (0, 1, 2) for plan in plans):
        raise ValueError("valid_plans must be a unique non-empty subset of 0, 1 and 2")
    current = int(context["current_plan"])
    if current not in plans:
        raise ValueError("current_plan must be feasible")
    predicted = context.get("predicted_bytes")
    if not isinstance(predicted, Mapping) or any(plan not in predicted for plan in plans):
        raise ValueError("predicted_bytes must contain every feasible plan")
    return plans
