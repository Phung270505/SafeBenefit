"""Deterministic regularized linear contextual-bandit policy."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.methods.base import PlanDecision, PlanSelector, validate_candidates

STATE_FEATURE_NAMES = (
    "selected_rows",
    "distinct_groups",
    "distinct_ratio",
    "mean_label_bytes",
    "current_plan",
    "graph_version",
    "retry_count",
    "switch_bytes",
    "estimated_compression_ratio",
    "query_class_id",
    "graph_scale_edges",
    "worker_count",
)

# Graph version, retry count and switch bytes remain available to the runtime
# guards. The predictor uses a compact projection to control variance while
# retaining every registered safe-point signal.
MODEL_FEATURE_NAMES = (
    "log_selected_rows",
    "log_distinct_groups",
    "distinct_ratio",
    "mean_label_bytes",
    "estimated_compression_ratio",
    "current_plan",
    "log_rows_minus_log_groups",
    "distinct_ratio_x_mean_label_bytes",
    "query_neighbor_type_agg",
    "query_temporal_type_agg",
    "query_two_hop_reachability",
    "query_two_hop_join",
    "log_graph_scale_edges",
    "log_graph_version",
    "log_worker_count",
    "log_retry_count",
    "log_switch_bytes",
)


def model_features(
    selected_rows: float,
    distinct_groups: float,
    mean_label_bytes: float,
    current_plan: int,
    *,
    compression_ratio: float = 1.0,
    query_class_id: int = 0,
    graph_scale_edges: float = 0,
    graph_version: float = 0,
    worker_count: int = 4,
    retry_count: int = 0,
    switch_bytes: float = 0,
) -> np.ndarray:
    """Create the feature projection from safe-point observations only."""
    values = (
        selected_rows,
        distinct_groups,
        mean_label_bytes,
        compression_ratio,
        graph_scale_edges,
        graph_version,
        float(worker_count),
        float(retry_count),
        switch_bytes,
    )
    if (
        any(not math.isfinite(value) or value < 0 for value in values)
        or current_plan not in (0, 1, 2)
        or query_class_id not in (0, 1, 2, 3)
        or worker_count not in (1, 2, 4, 8)
    ):
        raise ValueError("invalid contextual-bandit state")
    log_rows = math.log1p(selected_rows)
    log_groups = math.log1p(distinct_groups)
    ratio = distinct_groups / selected_rows if selected_rows else 0.0
    one_hot = [float(query_class_id == index) for index in range(4)]
    return np.asarray(
        [
            log_rows,
            log_groups,
            ratio,
            mean_label_bytes,
            compression_ratio,
            float(current_plan),
            log_rows - log_groups,
            ratio * mean_label_bytes,
            *one_hot,
            math.log1p(graph_scale_edges),
            math.log1p(graph_version),
            math.log1p(worker_count),
            math.log1p(retry_count),
            math.log1p(switch_bytes),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class FeatureScaler:
    """Scaler and regularized covariance model fitted on calibration-train only."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    inverse_covariance: tuple[tuple[float, ...], ...]

    @classmethod
    def fit(cls, rows: Iterable[Sequence[float]]) -> "FeatureScaler":
        matrix = np.asarray(list(rows), dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[0] < 2
            or matrix.shape[1] != len(MODEL_FEATURE_NAMES)
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("invalid LinUCB calibration contexts")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-12] = 1.0
        standardized = (matrix - mean) / scale
        covariance = np.cov(standardized, rowvar=False, ddof=1)
        covariance = np.atleast_2d(covariance)
        covariance += np.eye(covariance.shape[0], dtype=np.float64) * 1e-6
        inverse = np.linalg.pinv(covariance)
        return cls(
            tuple(mean.tolist()),
            tuple(scale.tolist()),
            tuple(tuple(row.tolist()) for row in inverse),
        )

    def transform(self, row: Sequence[float]) -> np.ndarray:
        vector = np.asarray(row, dtype=np.float64)
        if vector.shape != (len(MODEL_FEATURE_NAMES),) or not np.isfinite(vector).all():
            raise ValueError("invalid LinUCB context")
        return (vector - np.asarray(self.mean)) / np.asarray(self.scale)

    def max_abs_distance(self, row: Sequence[float]) -> float:
        """Retained for diagnostics; deployment uses Mahalanobis distance."""
        return float(np.max(np.abs(self.transform(row))))

    def mahalanobis_distance(self, row: Sequence[float]) -> float:
        """Return covariance-aware distance in the standardized feature space."""
        transformed = self.transform(row)
        inverse = np.asarray(self.inverse_covariance, dtype=np.float64)
        if inverse.shape != (len(MODEL_FEATURE_NAMES), len(MODEL_FEATURE_NAMES)):
            raise ValueError("invalid OOD covariance model")
        squared = float(transformed @ inverse @ transformed)
        return math.sqrt(max(0.0, squared))


@dataclass(frozen=True)
class LinearPolicy:
    """Disjoint LinUCB reward models, including intercept and uncertainty."""

    theta_by_action: tuple[tuple[float, ...], ...]
    precision_by_action: tuple[tuple[float, ...], ...] = ()
    alpha: float = 0.0

    @property
    def action_count(self) -> int:
        return len(self.theta_by_action)

    def scores(self, scaled_context: Sequence[float]) -> tuple[float, ...]:
        context = np.r_[1.0, np.asarray(scaled_context, dtype=np.float64)]
        expected = len(MODEL_FEATURE_NAMES) + 1
        if (
            context.shape != (expected,)
            or not self.theta_by_action
            or any(len(theta) != expected for theta in self.theta_by_action)
            or not np.isfinite(context).all()
        ):
            raise ValueError("invalid frozen LinUCB inference input")
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("invalid LinUCB exploration coefficient")
        scores: list[float] = []
        for action, theta in enumerate(self.theta_by_action):
            score = float(context @ np.asarray(theta))
            if self.precision_by_action:
                if len(self.precision_by_action) != self.action_count:
                    raise ValueError("invalid LinUCB precision matrix count")
                precision = np.asarray(self.precision_by_action[action], dtype=np.float64)
                if precision.shape != (expected * expected,):
                    raise ValueError("invalid flattened LinUCB precision matrix")
                matrix = precision.reshape(expected, expected)
                score += self.alpha * math.sqrt(max(0.0, float(context @ matrix @ context)))
            scores.append(score)
        return tuple(scores)


def fit_policy(
    contexts: Sequence[Sequence[float]],
    rewards: dict[int, Sequence[float]],
    scaler: FeatureScaler,
    *,
    ridge: float,
    alpha: float = 0.1,
) -> LinearPolicy:
    """Replay disjoint LinUCB updates over branch-complete calibration traces.

    Only the reward of the action selected at each replay step updates that
    action. The first observations force each action once, preventing a
    deterministic tie from starving an arm. Unselected branch measurements
    remain available for validation/oracle analysis but are not leaked into
    this bandit update.
    """
    actions = sorted(rewards)
    if (
        not math.isfinite(ridge)
        or ridge <= 0
        or not math.isfinite(alpha)
        or alpha < 0
        or len(actions) < 2
        or actions != list(range(len(actions)))
    ):
        raise ValueError("invalid LinUCB training configuration")
    if not contexts:
        raise ValueError("LinUCB needs at least one context")
    scaled = np.stack([scaler.transform(row) for row in contexts])
    design = np.column_stack([np.ones(len(scaled)), scaled])
    targets: dict[int, np.ndarray] = {}
    for action in actions:
        target = np.asarray(rewards[action], dtype=np.float64)
        if target.shape != (len(design),) or not np.isfinite(target).all():
            raise ValueError("invalid LinUCB rewards")
        targets[action] = target

    width = design.shape[1]
    covariance = [np.eye(width, dtype=np.float64) * ridge for _ in actions]
    reward_sum = [np.zeros(width, dtype=np.float64) for _ in actions]
    for index, context in enumerate(design):
        inverse = [np.linalg.pinv(matrix) for matrix in covariance]
        theta = [inverse[action] @ reward_sum[action] for action in actions]
        if index < len(actions):
            selected = index
        else:
            scores = [
                float(theta[action] @ context)
                + alpha
                * math.sqrt(max(0.0, float(context @ inverse[action] @ context)))
                for action in actions
            ]
            selected = max(actions, key=lambda action: (scores[action], -action))
        covariance[selected] += np.outer(context, context)
        reward_sum[selected] += targets[selected][index] * context

    precision = [np.linalg.pinv(matrix) for matrix in covariance]
    coefficients = [precision[action] @ reward_sum[action] for action in actions]
    return LinearPolicy(
        tuple(tuple(row.tolist()) for row in coefficients),
        tuple(tuple(matrix.reshape(-1).tolist()) for matrix in precision),
        alpha,
    )


def select_action(
    policy: LinearPolicy,
    scaled_context: Sequence[float],
    action_mask: Sequence[bool] | None = None,
) -> int:
    """Run deterministic inference with a hard feasibility mask."""
    mask = tuple(action_mask) if action_mask is not None else (True,) * policy.action_count
    if len(mask) != policy.action_count or not any(mask):
        raise ValueError("invalid action mask")
    scores = policy.scores(scaled_context)
    valid = [action for action in range(policy.action_count) if mask[action]]
    return max(valid, key=lambda action: (scores[action], -action))


def export_policy(
    policy: LinearPolicy,
    path: Path,
    scaler: FeatureScaler,
    *,
    ridge: float,
    ood_threshold: float,
) -> str:
    """Export a small text policy that C++ can read without Python or Torch."""
    expected = len(MODEL_FEATURE_NAMES) + 1
    if (
        not math.isfinite(ood_threshold)
        or ood_threshold <= 0
        or not math.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("invalid OOD threshold")
    if (
        policy.action_count != 3
        or len(policy.precision_by_action) != policy.action_count
        or any(len(theta) != expected for theta in policy.theta_by_action)
        or any(
            len(precision) != expected * expected
            for precision in policy.precision_by_action
        )
        or not math.isfinite(policy.alpha)
        or policy.alpha < 0
        or any(not math.isfinite(value) for theta in policy.theta_by_action for value in theta)
        or any(
            not math.isfinite(value)
            for precision in policy.precision_by_action
            for value in precision
        )
        or any(not math.isfinite(value) for value in scaler.mean)
        or any(not math.isfinite(value) or value <= 0 for value in scaler.scale)
        or any(
            not math.isfinite(value)
            for row in scaler.inverse_covariance
            for value in row
        )
    ):
        raise ValueError("incomplete frozen LinUCB policy")

    def values(row: Sequence[float]) -> str:
        return "\t".join(format(float(value), ".17g") for value in row)

    lines = [
        "schema_version\t5",
        "method\tLINUCB",
        "decision_process\tone_step_regularized_contextual_bandit",
        "state_features\t" + "\t".join(STATE_FEATURE_NAMES),
        "model_features\t" + "\t".join(MODEL_FEATURE_NAMES),
        f"input_size\t{len(MODEL_FEATURE_NAMES)}",
        f"ridge\t{ridge:.17g}",
        "mean\t" + values(scaler.mean),
        "scale\t" + values(scaler.scale),
        "ood_inverse_covariance\t"
        + values(value for row in scaler.inverse_covariance for value in row),
        f"action_count\t{policy.action_count}",
        *[
            f"theta_{action}\t" + values(theta)
            for action, theta in enumerate(policy.theta_by_action)
        ],
        *[
            f"precision_{action}\t" + values(precision)
            for action, precision in enumerate(policy.precision_by_action)
        ],
        f"ucb_alpha\t{policy.alpha:.17g}",
        f"ood_threshold\t{ood_threshold:.17g}",
        "exploration_evaluation\t0",
        "online_updates_evaluation\t0",
        "fallback\tSAFE_BENEFIT",
        "force_fallback\t0",
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def enforce_fallback(path: Path) -> str:
    """Mark a frozen candidate as rejected by the deployment safety gate."""
    payload = path.read_text(encoding="utf-8")
    marker = "force_fallback\t0\n"
    if marker not in payload:
        raise ValueError("policy is missing the fallback-gate field")
    encoded = payload.replace(marker, "force_fallback\t1\n", 1).encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


class LinUCBSelector(PlanSelector):
    """Frozen deterministic three-action contextual-bandit selector."""

    method_id = "LINUCB"

    def __init__(self, ridge: float = 0.1, alpha: float = 0.1) -> None:
        self.ridge = ridge
        self.alpha = alpha
        self.scaler: FeatureScaler | None = None
        self.policy: LinearPolicy | None = None

    def fit(self, calibration_rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
        rows = list(calibration_rows)
        contexts = [row["context_vector"] for row in rows]
        rewards = {
            action: [-float(row["costs"][str(action)]) for row in rows]
            for action in range(3)
        }
        self.scaler = FeatureScaler.fit(contexts)
        self.policy = fit_policy(
            contexts, rewards, self.scaler, ridge=self.ridge, alpha=self.alpha
        )
        return {"method_id": self.method_id, "ridge": self.ridge, "alpha": self.alpha}

    def select_plan(
        self, context: Mapping[str, Any], valid_plans: Sequence[int]
    ) -> PlanDecision:
        plans = validate_candidates(context, valid_plans)
        current = int(context["current_plan"])
        if self.scaler is None or self.policy is None:
            raise RuntimeError("LINUCB must be fitted before evaluation")
        mask = tuple(action in plans for action in range(3))
        vector = context["context_vector"]
        if self.scaler.mahalanobis_distance(vector) > float(context.get("ood_threshold", 4.0)):
            return PlanDecision(self.method_id, current, current, "ood_fallback_safe_benefit")
        selected = select_action(self.policy, self.scaler.transform(vector), mask)
        return PlanDecision(self.method_id, selected, current, "frozen_policy")
