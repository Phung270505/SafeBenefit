"""Query and physical-plan registry shared by all components."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

QUERY_CLASSES = (
    "NEIGHBOR_TYPE_AGG",
    "TEMPORAL_TYPE_AGG",
    "TWO_HOP_REACHABILITY",
    "TWO_HOP_JOIN",
)

QUERY_CLASS_IDS = {name: index for index, name in enumerate(QUERY_CLASSES)}

# Old query aliases are accepted only while reading historical inputs. New artifacts
# always write the descriptive names above.
QUERY_ALIASES = {
    "CQ7": "NEIGHBOR_TYPE_AGG",
    "CQ9": "TEMPORAL_TYPE_AGG",
    "CQ2": "TWO_HOP_REACHABILITY",
    "CQ8": "TWO_HOP_JOIN",
}

PLAN_NAMES = {
    0: "RAW_ROWS",
    1: "LOCAL_REDUCE",
    2: "COMPRESSED_REDUCE",
}
PLAN_IDS = {name: plan_id for plan_id, name in PLAN_NAMES.items()}


def canonical_query_class(value: str) -> str:
    """Return the descriptive query class or reject an unknown class."""
    canonical = QUERY_ALIASES.get(value, value)
    if canonical not in QUERY_CLASS_IDS:
        raise ValueError(f"unknown query class: {value}")
    return canonical


@dataclass(frozen=True)
class QueryRecord:
    """Input contract for one paired query in the study."""

    query_id: str
    dataset_id: str
    scale_id: str
    query_class: str
    schedule_id: int
    replica_id: int
    graph_version: int
    parameters: Mapping[str, Any]
    valid_plans: tuple[int, ...]
    expected_result_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QueryRecord":
        """Parse and validate a JSON-compatible query record."""
        record = cls(
            query_id=str(value["query_id"]),
            dataset_id=str(value["dataset_id"]),
            scale_id=str(value["scale_id"]),
            query_class=canonical_query_class(str(value["query_class"])),
            schedule_id=int(value["schedule_id"]),
            replica_id=int(value["replica_id"]),
            graph_version=int(value["graph_version"]),
            parameters=dict(value["parameters"]),
            valid_plans=tuple(int(plan) for plan in value["valid_plans"]),
            expected_result_sha256=str(value["expected_result_sha256"]),
        )
        validate_query_record(record)
        return record


def validate_query_record(record: QueryRecord) -> None:
    """Enforce the frozen query schema and three-plan contract."""
    if not record.query_id or not record.dataset_id or not record.scale_id:
        raise ValueError("query identifiers must be non-empty")
    canonical_query_class(record.query_class)
    if record.schedule_id < 0 or record.replica_id not in (0, 1):
        raise ValueError("invalid schedule or replica")
    if record.graph_version < 0 or not math.isfinite(float(record.graph_version)):
        raise ValueError("invalid graph version")
    if tuple(sorted(set(record.valid_plans))) != tuple(PLAN_NAMES):
        raise ValueError("queries must expose plans 0, 1 and 2 exactly once")
    digest = record.expected_result_sha256
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise ValueError("expected-result hash must be SHA-256")
