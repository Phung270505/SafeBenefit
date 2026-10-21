"""Freeze the switch-aware threshold on calibration-train, separately per scale."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.config import load_study_config
from src.common.design import STUDY_SCHEMA_VERSION
from src.common.files import sha256_file, write_json
from src.methods.linucb_training import choose_initial_plans, effective_cost
from src.methods.threshold_calibration import SwitchThresholdModel
from src.plans.cost import PlanStatistics, estimate_plan_wire_bytes
from src.plans.switch_cost import SwitchCostModel


def _predicted_bytes(sample: dict[str, Any]) -> dict[int, float]:
    if "predicted_bytes" in sample:
        return {
            int(plan): float(value)
            for plan, value in sample["predicted_bytes"].items()
        }
    return estimate_plan_wire_bytes(
        PlanStatistics(
            float(sample["selected_rows"]),
            float(sample["distinct_groups"]),
            float(sample["mean_label_bytes"]),
            int(sample.get("worker_count", 4)),
        )
    )


def train_calibrated_methods(
    samples: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    """Fit the deployed switch-aware rule on train; validation is reporting-only."""
    if samples and len({sample["dataset_id"] for sample in samples}) != 1:
        raise ValueError("baseline training must contain exactly one dataset")
    if not samples or any(
        set(sample.get("costs", {})) != {"0", "1", "2"} for sample in samples
    ):
        raise ValueError("branch-complete calibration samples are required")
    split = [int(sample.get("replica_id", -1)) for sample in samples]
    if any(index not in (0, 1) for index in split):
        raise ValueError("calibration sample must belong to split 0 or 1")
    train = sorted(
        (sample for sample, index in zip(samples, split, strict=True) if index == 0),
        key=lambda sample: (int(sample["graph_version"]), str(sample["query_id"])),
    )
    validation = sorted(
        (sample for sample, index in zip(samples, split, strict=True) if index == 1),
        key=lambda sample: (int(sample["graph_version"]), str(sample["query_id"])),
    )
    if not train or not validation:
        raise ValueError("calibration split is incomplete")

    initial_plans = choose_initial_plans(train)
    threshold_cases: list[dict[str, Any]] = []
    for sample in train:
        key = f"{sample['dataset_id']}:{sample['query_class']}"
        current = initial_plans[key]
        threshold_cases.append(
            {
                "current_plan": current,
                "predicted_bytes": _predicted_bytes(sample),
                "actual_bytes": {
                    action: float(sample["costs"][str(action)])
                    for action in (0, 1, 2)
                },
                "switch_bytes": float(sample["switch_bytes"]),
            }
        )

    model = SwitchThresholdModel.fit(threshold_cases)
    validation_total = 0
    validation_actions: list[dict[str, Any]] = []
    for sample in validation:
        key = f"{sample['dataset_id']}:{sample['query_class']}"
        initial = initial_plans[key]
        action = model.select(initial, _predicted_bytes(sample), float(sample["switch_bytes"]))
        validation_total += effective_cost(sample, action, initial)
        validation_actions.append({
            "query_id": sample["query_id"],
            "initial_plan": initial,
            "switch_cost_threshold_plan": action,
        })

    output.mkdir(parents=True, exist_ok=True)
    threshold_path = output / "switch_threshold.tsv"
    threshold_path.write_text(
        "schema_version\t1\n"
        "method\tSWITCH_COST_THRESHOLD\n"
        f"threshold\t{model.threshold:.17g}\n"
        f"calibration_observations\t{model.calibration_observations}\n"
        f"calibration_total_bytes\t{model.calibration_total_bytes:.17g}\n"
        "updates_evaluation\t0\n",
        encoding="utf-8",
        newline="\n",
    )
    switch_model = SwitchCostModel.from_config(
        load_study_config()["switch_cost_model"]
    )
    switch_model_path = output / "switch_cost_model.tsv"
    switch_model.export(switch_model_path)

    report = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": "SAFEBENEFIT-DGQ",
        "status": "PASS",
        "train_queries": len(train),
        "validation_queries": len(validation),
        "initial_plans": initial_plans,
        "switch_threshold": model.threshold,
        "threshold_candidates": [value for value, _ in model.threshold_totals],
        "threshold_train_total_bytes": {
            format(value, ".2f"): total for value, total in model.threshold_totals
        },
        "validation_wire_bytes": {"SWITCH_COST_THRESHOLD": validation_total},
        "threshold_validation_action_source": "safe_point_cost_model_only",
        "switch_threshold_sha256": sha256_file(threshold_path),
        "switch_cost_model_sha256": sha256_file(switch_model_path),
        "validation_actions": validation_actions,
        "evaluation_updates": False,
    }
    write_json(output / "baseline_training.json", report)
    return report
