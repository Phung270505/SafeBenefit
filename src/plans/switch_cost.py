"""Deterministic switching-overhead model shared by training and simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SwitchCostModel:
    """Model PREPARE/ACK wire overhead as an intercept plus per-worker bytes."""

    intercept_bytes: float
    per_worker_bytes: float
    source: str

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SwitchCostModel":
        model = cls(
            intercept_bytes=float(config.get("intercept_bytes", 0.0)),
            per_worker_bytes=float(config["per_worker_bytes"]),
            source=str(config.get("source", "unspecified")),
        )
        model.validate()
        return model

    def validate(self) -> None:
        if (
            not self.source
            or not math.isfinite(self.intercept_bytes)
            or not math.isfinite(self.per_worker_bytes)
            or self.intercept_bytes < 0
            or self.per_worker_bytes < 0
        ):
            raise ValueError("invalid switching-cost model")

    def estimate(self, worker_count: int, *, retry_rounds: int = 0) -> float:
        """Estimate wire bytes for one PREPARE/ACK barrier and its retries."""
        self.validate()
        if worker_count not in (1, 2, 4, 8, 16) or retry_rounds < 0:
            raise ValueError("invalid switching-cost context")
        one_round = self.intercept_bytes + self.per_worker_bytes * worker_count
        return one_round * (retry_rounds + 1)

    def export(self, path: Path) -> None:
        """Freeze the transparent model as a human-readable TSV artifact."""
        self.validate()
        payload = (
            "schema_version\t1\n"
            "model\tSWITCH_COST\n"
            f"source\t{self.source}\n"
            f"intercept_bytes\t{self.intercept_bytes:.17g}\n"
            f"per_worker_bytes\t{self.per_worker_bytes:.17g}\n"
            "formula\tintercept_bytes+per_worker_bytes*worker_count\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8", newline="\n")
