"""Calibration-facing byte estimators for the three transfer encodings."""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_NETWORK_HEADER_BYTES = 42.0  # Ethernet + IPv4 + UDP; the simulator uses UDP.
DEFAULT_MTU_BYTES = 1500.0
DEFAULT_APPLICATION_CHUNK_BYTES = 900.0


@dataclass(frozen=True)
class PlanStatistics:
    """Cheap statistics committed before a transfer plan is selected."""

    selected_rows: float
    distinct_groups: float
    mean_label_bytes: float
    worker_count: int

    def validate(self) -> None:
        values = (self.selected_rows, self.distinct_groups, self.mean_label_bytes)
        if (
            any(not math.isfinite(value) or value < 0 for value in values)
            or self.distinct_groups > self.selected_rows
            or self.worker_count not in (1, 2, 4, 8, 16)
        ):
            raise ValueError("invalid safe-point plan statistics")


def _framed_bytes(
    payload_bytes: float,
    workers: int,
    *,
    application_chunk_bytes: float,
    mtu_bytes: float,
    network_header_bytes: float,
) -> float:
    if (
        application_chunk_bytes <= 0
        or mtu_bytes <= network_header_bytes
        or network_header_bytes < 0
    ):
        raise ValueError("invalid network framing configuration")
    payload_per_worker = payload_bytes / workers
    fragment_payload = min(application_chunk_bytes, mtu_bytes - network_header_bytes)
    fragments_per_worker = max(1, math.ceil(payload_per_worker / fragment_payload))
    return payload_bytes + workers * fragments_per_worker * network_header_bytes


def estimate_plan_wire_bytes(
    statistics: PlanStatistics,
    *,
    network_header_bytes: float = DEFAULT_NETWORK_HEADER_BYTES,
    mtu_bytes: float = DEFAULT_MTU_BYTES,
    application_chunk_bytes: float = DEFAULT_APPLICATION_CHUNK_BYTES,
) -> dict[int, float]:
    """Estimate plan-dependent UDP/IP/Ethernet bytes without building candidate payloads."""
    statistics.validate()
    rows = statistics.selected_rows
    distinct = statistics.distinct_groups
    label_digits = max(1.0, statistics.mean_label_bytes)
    mean_count = rows / distinct if distinct else 1.0
    count_digits = math.floor(math.log10(max(1.0, mean_count))) + 1.0
    per_worker_envelope = statistics.worker_count * 32.0

    # The compressed plan carries a 256-byte dictionary descriptor plus a
    # four-byte entry count per worker. Its
    # label and count estimates use unsigned varint widths, not a fake fixed
    # network header. Exact bytes are measured only after the selected codec runs.
    label_varint = max(1.0, math.ceil(label_digits * math.log2(10.0) / 7.0))
    count_varint = max(1.0, math.ceil(math.log2(max(2.0, mean_count + 1.0)) / 7.0))
    payloads = {
        0: rows * (label_digits + 1.0) + per_worker_envelope,
        1: distinct * (label_digits + count_digits + 2.0) + per_worker_envelope,
        2: distinct * (label_varint + count_varint) + statistics.worker_count * 292.0,
    }
    return {
        plan: _framed_bytes(
            payload,
            statistics.worker_count,
            application_chunk_bytes=(
                application_chunk_bytes
                if plan in (0, 1)
                else mtu_bytes - network_header_bytes
            ),
            mtu_bytes=mtu_bytes,
            network_header_bytes=network_header_bytes,
        )
        for plan, payload in payloads.items()
    }


def estimate_payload_bytes(
    statistics: PlanStatistics,
    **network: float,
) -> dict[int, float]:
    """Backward-compatible name; values include plan-dependent wire framing."""
    return estimate_plan_wire_bytes(statistics, **network)


def estimated_compression_ratio(statistics: PlanStatistics) -> float:
    """Return compressed/local predicted payload ratio for the policy state."""
    estimates = estimate_plan_wire_bytes(statistics)
    return estimates[2] / max(estimates[1], 1.0)
