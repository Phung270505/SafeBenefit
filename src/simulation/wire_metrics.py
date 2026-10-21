"""Read host-transmitted Ethernet bytes from OMNeT++ scalar output."""

from __future__ import annotations

import re
from pathlib import Path

WIRE_PATTERN = re.compile(
    r"^scalar GraphQuery\.(coordinator|worker\[\d+\])\.eth\[0\]\.mac "
    r"txPk:sum\(packetBytes\) ([0-9]+(?:\.[0-9]+)?)$"
)


def host_wire_bytes(path: Path, *, expected_workers: int) -> int:
    """Sum bytes injected onto coordinator/worker Ethernet links exactly once."""
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = WIRE_PATTERN.match(line)
        if match:
            host, value = match.groups()
            if host in values:
                raise ValueError(f"duplicate wire-byte scalar for {host}")
            values[host] = float(value)
    expected = {"coordinator", *(f"worker[{index}]" for index in range(expected_workers))}
    if set(values) != expected:
        raise ValueError(f"wire-byte hosts differ: found={sorted(values)}, expected={sorted(expected)}")
    total = sum(values.values())
    if total < 0 or not total.is_integer():
        raise ValueError("wire bytes must be a nonnegative integer")
    return int(total)
