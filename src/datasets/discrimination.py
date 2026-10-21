"""Describe plan dominance without selecting benchmarks by their outcomes."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from src.queries.registry import PLAN_NAMES, QUERY_CLASSES


def summarize_plan_costs(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Validate coverage and cost vectors; plan diversity is diagnostic, never a gate.

    Missing classes, duplicate query IDs and malformed costs are genuine errors.
    A single winning plan or ties are valid outcomes, not reasons to retune data.
    """
    winners: dict[str, Counter[int]] = defaultdict(Counter)
    counts: Counter[str] = Counter()
    ties: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    errors: list[str] = []
    for row in samples:
        key = (str(row["dataset_id"]), str(row["query_id"]))
        if key in seen:
            errors.append(f"duplicate calibration query: {key}")
            continue
        seen.add(key)
        query_class = str(row["query_class"])
        costs = row["costs"]
        if query_class not in QUERY_CLASSES:
            errors.append(f"unknown query class: {query_class}")
            continue
        if set(costs) != {str(plan) for plan in PLAN_NAMES} or any(
            type(value) is not int or value < 0 for value in costs.values()
        ):
            errors.append(f"invalid three-plan cost vector: {key}")
            continue
        counts[query_class] += 1
        best = [int(plan) for plan, cost in costs.items() if cost == min(costs.values())]
        if len(best) == 1:
            winners[query_class][best[0]] += 1
        else:
            ties[query_class] += 1
    for query_class in QUERY_CLASSES:
        if not counts[query_class]:
            errors.append(f"missing calibration query class: {query_class}")
    global_winners = set().union(*(set(win_counts) for win_counts in winners.values()))
    warnings = [
        f"{query_class}: fewer than two unique winners; valid benchmark outcome"
        for query_class in QUERY_CLASSES
        if counts[query_class] and len(winners[query_class]) < 2
    ]
    if global_winners != set(PLAN_NAMES):
        warnings.append("not all plans uniquely win; no workload filtering or retuning required")
    return {
        "schema_version": 2,
        "status": "FAIL" if errors else "PASS",
        "rank_reversal_policy": "diagnostic_only",
        "trace_count": sum(counts.values()),
        "query_counts_by_class": dict(counts),
        "winners_by_query_class": {
            query_class: sorted(winners[query_class]) for query_class in QUERY_CLASSES
        },
        "unique_win_counts_by_query_class": {
            query_class: {str(plan): winners[query_class][plan] for plan in PLAN_NAMES}
            for query_class in QUERY_CLASSES
        },
        "tie_counts_by_query_class": {
            query_class: ties[query_class] for query_class in QUERY_CLASSES
        },
        "all_three_plans_win": global_winners == set(PLAN_NAMES),
        "warnings": warnings,
        "errors": errors,
    }


def count_action_disagreements(rows: list[dict[str, Any]]) -> int:
    """Count differing stable actions; zero is valid, not an evaluation failure."""
    by_query: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        if row.get("scenario", "Stable") == "Stable":
            plan = int(row["plan_id"] if "plan_id" in row else row["plan"])
            if plan not in PLAN_NAMES:
                raise ValueError(f"unknown physical plan: {plan}")
            by_query[(str(row["dataset_id"]), str(row["query_id"]))].add(plan)
    return sum(len(plans) > 1 for plans in by_query.values())
