"""Shared transfer decision primitive for all registered methods."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Mapping

from src.common.config import require_method


@dataclass(frozen=True)
class TransferCost:
    """Measured or predicted bytes attributable to one remaining physical plan."""

    data_payload_bytes: float
    control_bytes: float = 0
    retry_bytes: float = 0
    switch_bytes: float = 0

    @property
    def total_bytes(self) -> float:
        values = (self.data_payload_bytes, self.control_bytes, self.retry_bytes, self.switch_bytes)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("transfer-cost components must be finite and nonnegative")
        return sum(values)


@dataclass(frozen=True)
class TransferDecision:
    method: str
    current_plan: str
    selected_plan: str
    action: str
    reason: str
    current_bytes: float
    selected_bytes: float
    switch_bytes: float
    predicted_saving_bytes: float
    uncertainty_penalty_bytes: float = 0
    lower_confidence_benefit_bytes: float = 0


@dataclass(frozen=True)
class TransferState:
    """Decision state available at a committed safe point; no future outcome."""

    selected_rows: int
    distinct_groups: int
    distinct_ratio: float
    mean_label_bytes: float
    current_plan: int
    graph_version: int
    retry_count: int
    switch_bytes: float
    compression_ratio: float = 1.0
    query_class_id: int = 0
    graph_scale_edges: int = 0
    worker_count: int = 4

    def vector(self) -> tuple[float, ...]:
        values = (
            float(self.selected_rows),
            float(self.distinct_groups),
            self.distinct_ratio,
            self.mean_label_bytes,
            float(self.current_plan),
            float(self.graph_version),
            float(self.retry_count),
            self.switch_bytes,
            self.compression_ratio,
            float(self.query_class_id),
            float(self.graph_scale_edges),
            float(self.worker_count),
        )
        expected_ratio = self.distinct_groups / self.selected_rows if self.selected_rows else 0.0
        if (
            any(not math.isfinite(value) or value < 0 for value in values)
            or self.distinct_groups > self.selected_rows
            or self.current_plan not in (0, 1, 2)
            or not 0 <= self.query_class_id < 4
            or self.worker_count not in (1, 2, 4, 8, 16)
            or not math.isclose(self.distinct_ratio, expected_ratio, abs_tol=1e-12)
        ):
            raise ValueError("invalid compact transfer state")
        return values


def choose_transfer_plan(
    method: str,
    current_plan: str,
    candidates: Mapping[str, TransferCost],
    *,
    reduction_threshold: float = 0.05,
    safety_margin: float = 0.10,
    safe_point: bool = True,
    switch_budget: int = 1,
    rl_action: Callable[[list[str]], str] | None = None,
    uncertainty_by_plan: Mapping[str, float] | None = None,
    uncertainty_z: float = 1.96,
    progress: float = 0.0,
    near_completion_guard: float = 0.90,
    state_graph_version: int | None = None,
    statistics_graph_version: int | None = None,
) -> TransferDecision:
    """Choose by transfer bytes; correctness and safe-point checks are hard guards."""
    method = require_method(method)
    if (
        current_plan not in candidates
        or not 0 <= reduction_threshold <= 1
        or not 0 <= safety_margin <= 1
        or not math.isfinite(uncertainty_z)
        or uncertainty_z < 0
        or not 0 <= progress <= 1
        or not 0 <= near_completion_guard <= 1
    ):
        raise ValueError("invalid transfer decision input")
    uncertainty = dict(uncertainty_by_plan or {plan: 0.0 for plan in candidates})
    if set(uncertainty) != set(candidates) or any(
        not math.isfinite(value) or value < 0 for value in uncertainty.values()
    ):
        raise ValueError("invalid transfer-cost uncertainty")
    totals = {plan: cost.total_bytes for plan, cost in candidates.items()}
    transfer_only = {
        plan: total - candidates[plan].switch_bytes for plan, total in totals.items()
    }
    current = transfer_only[current_plan]
    best = min(transfer_only, key=lambda plan: (transfer_only[plan], plan))
    selected, reason = current_plan, "static_locked" if method == "STATIC" else "current_is_best"
    if method != "STATIC" and (best != current_plan or method == "LINUCB"):
        stale = (
            state_graph_version is not None
            and statistics_graph_version is not None
            and state_graph_version != statistics_graph_version
        )
        if not safe_point or switch_budget <= 0:
            reason = "safety_guard"
        elif progress >= near_completion_guard:
            reason = "near_completion_guard"
        elif stale:
            reason = "stale_statistics_guard"
        else:
            gross = current - transfer_only[best]
            switch = candidates[best].switch_bytes
            if method == "SWITCH_COST_THRESHOLD":
                required = switch + reduction_threshold * max(current, 1)
                if gross > required:
                    selected, reason = best, "switch_cost_threshold"
                else:
                    reason = "below_switch_cost_threshold"
            elif method == "SAFE_BENEFIT":
                uncertainty_penalty = uncertainty_z * math.hypot(
                    uncertainty[current_plan], uncertainty[best]
                )
                net_benefit = gross - switch - uncertainty_penalty
                if net_benefit > safety_margin * max(current, 1):
                    selected, reason = best, "positive_net_transfer_saving"
                else:
                    reason = "insufficient_net_transfer_saving"
            elif method == "LINUCB":
                if rl_action is None:
                    fallback = choose_transfer_plan("SAFE_BENEFIT", current_plan, candidates,
                                                    reduction_threshold=reduction_threshold,
                                                    safety_margin=safety_margin, safe_point=safe_point,
                                                    switch_budget=switch_budget)
                    return replace(
                        fallback,
                        method="LINUCB",
                        reason="rl_unavailable_fallback_" + fallback.reason,
                    )
                proposed = rl_action(sorted(candidates))
                if proposed not in candidates:
                    raise ValueError("RL selected an unknown or masked plan")
                selected, reason = proposed, "rl_policy"
            else:
                raise ValueError(f"unsupported method: {method}")
    switch = candidates[selected].switch_bytes if selected != current_plan else 0
    selected_total = transfer_only[selected] + switch
    uncertainty_penalty = 0.0
    if selected != current_plan:
        uncertainty_penalty = uncertainty_z * math.hypot(
            uncertainty[current_plan], uncertainty[selected]
        )
    return TransferDecision(
        method,
        current_plan,
        selected,
        "KEEP" if selected == current_plan else "SWITCH",
        reason,
        current,
        selected_total,
        switch,
        current - selected_total,
        uncertainty_penalty,
        current - selected_total - uncertainty_penalty,
    )
