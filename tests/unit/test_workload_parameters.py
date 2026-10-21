"""Regression tests for disjoint deterministic workload parameters."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.files import stable_json
from src.datasets.workload_core import _rows_for_dataset


class _FakeGraph:
    MEMORY_EDGE_LIMIT = 2_000_000

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def close(self) -> None:
        pass


class WorkloadParameterTests(unittest.TestCase):
    def test_excluded_parameter_advances_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            data_root = Path(folder)
            root = data_root / "processed" / "sample"
            (root / "metadata").mkdir(parents=True)
            (root / "metadata" / "manifest.json").write_text(
                json.dumps(
                    {
                        "snapshot": {"edges": 1},
                        "updates": {"calibration": 10, "evaluation": 10},
                    }
                ),
                encoding="utf-8",
            )
            ranked = {
                bucket: [f"{bucket}-{index}" for index in range(256)]
                for bucket in ("low", "medium", "high")
            }

            def parameters(
                _query_class: str,
                _bucket: str,
                sample_index: int,
                *_: object,
            ) -> dict[str, int]:
                return {"sample": sample_index}

            excluded = {("NEIGHBOR_TYPE_AGG", stable_json({"sample": 16}))}
            with (
                patch("src.datasets.workload_core.DATA_ROOT", data_root),
                patch("src.datasets.workload_core.Graph", _FakeGraph),
                patch("src.datasets.workload_core._ranked_vertices", return_value=ranked),
                patch(
                    "src.datasets.workload_core._split_temporal_bounds",
                    return_value={
                        "low": (1, 10),
                        "medium": (1, 20),
                        "high": (1, 30),
                    },
                ),
                patch("src.datasets.workload_core._query_parameters", side_effect=parameters),
            ):
                first = _rows_for_dataset(
                    "sample",
                    split="evaluation",
                    schedules=1,
                    component="evaluation_parameters",
                    sample_offset=16,
                    excluded_parameters=excluded,
                )
                second = _rows_for_dataset(
                    "sample",
                    split="evaluation",
                    schedules=1,
                    component="evaluation_parameters",
                    sample_offset=16,
                    excluded_parameters=excluded,
                )

            self.assertEqual(first, second)
            keys = {(row["query_class"], stable_json(row["parameters"])) for row in first}
            self.assertTrue(keys.isdisjoint(excluded))
            self.assertEqual(len(keys), len(first))


if __name__ == "__main__":
    unittest.main()
