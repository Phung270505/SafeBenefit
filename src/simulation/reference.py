"""Fixed, machine-independent logical-operation reference model.

An operation is a declared logical action, not a physical CPU instruction.
Byte encode/decode counters count bytes processed once by the corresponding codec.
Map updates count logical keyed updates, independent of the library container.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

OPERATION_TYPES = (
    "edge_visit",
    "predicate",
    "aggregate_update",
    "join_match",
    "tuple_visit",
    "encode_byte",
    "decode_byte",
    "merge_update",
    "arithmetic",
    "comparison",
    "special_math",
    "control_message",
)

OPERATION_DEFINITIONS = {
    "edge_visit": "logical edge/adjacency entry consumed, not SQLite pages visited",
    "predicate": "one logical selection check per consumed query entry",
    "aggregate_update": "one keyed count update; container-internal instructions excluded",
    "join_match": "one accepted join pair",
    "tuple_visit": "one tuple consumed by a declared kernel",
    "encode_byte": "one byte emitted by the selected result codec, including its header",
    "decode_byte": "one unique payload byte consumed by the receiver codec",
    "merge_update": "one decoded keyed count merged into the result",
    "arithmetic": "one counted scalar arithmetic operation in optimizer/codec kernels",
    "comparison": "one counted numeric comparison in optimizer kernels",
    "special_math": "one counted logarithm/sqrt/ceil/floor/hypot evaluation",
    "control_message": "one queued protocol send attempt, including DATA, duplicate and retry",
}
# Oracle construction, checksums, Python preprocessing, simulator engine bookkeeping,
# and counter instrumentation itself are outside the declared query kernel model.


@dataclass
class OperationCounts:
    """Nonnegative exact counters for logical actions executed in one phase."""

    values: Counter[str] = field(default_factory=Counter)

    def add(self, operation: str, amount: int = 1) -> None:
        if operation not in OPERATION_TYPES or not isinstance(amount, int) or amount < 0:
            raise ValueError("invalid logical operation count")
        self.values[operation] += amount

    def work_units(self, weights: Mapping[str, float]) -> float:
        if set(weights) != set(OPERATION_TYPES):
            raise ValueError("operation-weight registry mismatch")
        return sum(count * weights[name] for name, count in self.values.items())


def cpu_seconds(work_units: float, rate: float) -> float:
    """Convert logical work to modeled service time, never to measured host time."""
    if not math.isfinite(work_units) or work_units < 0 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("invalid reference work/capacity")
    return work_units / rate
