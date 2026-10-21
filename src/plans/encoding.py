"""Lossless physical result encodings used by SafeBenefit."""

from __future__ import annotations

import struct
from collections import Counter
from collections.abc import Iterable

from src.queries.registry import PLAN_NAMES

COMPRESSED_MAGIC = b"SBC1"
COMPRESSED_HEADER_SIZE = 256


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint values must be nonnegative")
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _decode_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(payload):
            raise ValueError("truncated varint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError("varint exceeds uint64")


def encode_labels(labels: Iterable[int], plan: int) -> bytes:
    """Encode raw rows, local counts, or delta/varint-compressed counts."""
    values = [int(label) for label in labels]
    if any(label < 0 or label > 0xFFFFFFFFFFFFFFFF for label in values):
        raise ValueError("labels must fit unsigned 64-bit integers")
    if plan == 0:
        return "".join(f"{label} " for label in values).encode("ascii")
    if plan == 1:
        return "".join(
            f"{label}:{count} " for label, count in sorted(Counter(values).items())
        ).encode("ascii")
    if plan == 2:
        counts = sorted(Counter(values).items())
        # The fixed header represents the real per-partition dictionary descriptor,
        # codec metadata and future-compatible reserved bytes carried on the wire.
        # It deliberately makes compression unattractive for tiny outputs.
        output = bytearray(COMPRESSED_HEADER_SIZE)
        output[:4] = COMPRESSED_MAGIC
        output.extend(struct.pack("<I", len(counts)))
        previous = 0
        for label, count in counts:
            output.extend(_encode_varint(label - previous))
            output.extend(_encode_varint(count))
            previous = label
        return bytes(output)
    raise ValueError(f"unknown physical plan: {plan}")


def decode_labels(payload: bytes, plan: int) -> Counter[int]:
    """Decode any registered representation into the canonical multiset."""
    if plan not in PLAN_NAMES:
        raise ValueError(f"unknown physical plan: {plan}")
    result: Counter[int] = Counter()
    if plan == 2:
        if len(payload) < COMPRESSED_HEADER_SIZE + 4 or payload[:4] != COMPRESSED_MAGIC:
            raise ValueError("invalid compressed-reduce header")
        if any(payload[4:COMPRESSED_HEADER_SIZE]):
            raise ValueError("invalid compressed-reduce reserved header")
        entries = struct.unpack(
            "<I", payload[COMPRESSED_HEADER_SIZE : COMPRESSED_HEADER_SIZE + 4]
        )[0]
        offset = COMPRESSED_HEADER_SIZE + 4
        previous = 0
        for _ in range(entries):
            delta, offset = _decode_varint(payload, offset)
            count, offset = _decode_varint(payload, offset)
            if count == 0:
                raise ValueError("compressed count must be positive")
            label = previous + delta
            if label > 0xFFFFFFFFFFFFFFFF:
                raise ValueError("decoded label overflows uint64")
            result[label] += count
            previous = label
        if offset != len(payload):
            raise ValueError("trailing bytes in compressed payload")
        return result
    for token in payload.decode("ascii").split():
        if plan == 0:
            result[int(token)] += 1
        else:
            label, separator, count = token.partition(":")
            if not separator:
                raise ValueError("aggregate tuple is missing its count")
            result[int(label)] += int(count)
    return result


def winning_plans(samples: Iterable[Iterable[int]]) -> set[int]:
    """Return plans that uniquely minimize encoded bytes for at least one sample."""
    winners = set()
    for labels in samples:
        values = list(labels)
        costs = {plan: len(encode_labels(values, plan)) for plan in PLAN_NAMES}
        minimum = min(costs.values())
        if sum(cost == minimum for cost in costs.values()) == 1:
            winners.add(min(costs, key=lambda candidate: (costs[candidate], candidate)))
    return winners


def has_rank_reversal(samples: Iterable[Iterable[int]], *, require_all: bool = False) -> bool:
    """Check that multiple plans, or all plans, win on observed calibration samples."""
    winners = winning_plans(samples)
    return winners == set(PLAN_NAMES) if require_all else len(winners) >= 2
