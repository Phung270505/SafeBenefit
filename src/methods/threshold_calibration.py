"""Fit the switch-aware threshold using the same rule deployed in evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_THRESHOLD_GRID = tuple(index / 100.0 for index in range(51))


def select_switch_threshold(
    current: int, predicted_bytes: Mapping[int, float], switch_bytes: float, threshold: float
) -> int:
    """Switch only when gross saving strictly exceeds switch cost plus margin."""
    costs = {int(plan): float(value) for plan, value in predicted_bytes.items()}
    if (
        current not in costs or not costs or any(plan not in (0, 1, 2) for plan in costs)
        or any(not math.isfinite(value) or value < 0 for value in costs.values())
        or not math.isfinite(switch_bytes) or switch_bytes < 0
        or not math.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        raise ValueError("invalid switch-threshold decision input")
    candidate = min(costs, key=lambda plan: (costs[plan], plan))
    gross = costs[current] - costs[candidate]
    required = switch_bytes + threshold * max(costs[current], 1.0)
    return candidate if candidate != current and gross > required else current


@dataclass(frozen=True)
class SwitchThresholdModel:
    """Frozen per-scale threshold; no validation/evaluation updates."""

    threshold: float
    calibration_observations: int
    calibration_total_bytes: float
    threshold_totals: tuple[tuple[float, float], ...]

    @classmethod
    def fit(
        cls, cases: Iterable[Mapping[str, Any]], *,
        thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    ) -> SwitchThresholdModel:
        """Minimize train actual plan bytes plus switching bytes; ties prefer lower t."""
        rows = list(cases)
        grid = sorted(float(value) for value in thresholds)
        if not rows or not grid or len(set(grid)) != len(grid) or any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in grid
        ):
            raise ValueError("invalid switch-threshold calibration input")
        normalized = []
        for row in rows:
            predicted = {int(plan): float(value) for plan, value in row["predicted_bytes"].items()}
            actual = {int(plan): float(value) for plan, value in row["actual_bytes"].items()}
            if set(predicted) != set(actual) or any(
                not math.isfinite(value) or value < 0 for value in actual.values()
            ):
                raise ValueError("invalid switch-threshold actual costs")
            normalized.append((int(row["current_plan"]), predicted, actual, float(row["switch_bytes"])))
        totals = []
        for threshold in grid:
            total = 0.0
            for current, predicted, actual, switch in normalized:
                selected = select_switch_threshold(current, predicted, switch, threshold)
                total += actual[selected] + (switch if selected != current else 0.0)
            totals.append((threshold, total))
        threshold, total = min(totals, key=lambda item: (item[1], item[0]))
        return cls(threshold, len(rows), total, tuple(totals))

    def select(self, current: int, predicted_bytes: Mapping[int, float], switch_bytes: float) -> int:
        """Use only safe-point predictions and the frozen threshold."""
        return select_switch_threshold(current, predicted_bytes, switch_bytes, self.threshold)
