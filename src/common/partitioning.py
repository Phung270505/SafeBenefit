"""Shared frozen ownership rule used by adapters, queries and distributed storage."""

import hashlib


def stable_owner(vertex_id: str, partitions: int) -> int:
    """Map an ID to an owner using SHA-256's first 16 hex digits."""
    if partitions <= 0:
        raise ValueError("partition count must be positive")
    return int(hashlib.sha256(vertex_id.encode("utf-8")).hexdigest()[:16], 16) % partitions
