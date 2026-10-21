"""Crash-resume safety for large disk-backed graph exports."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.datasets.export import _load_resumable_export
from src.datasets.replay import (
    SqliteVersionedGraph,
    write_materialization,
)


class ExportResumeTests(unittest.TestCase):
    @staticmethod
    def _dataset(root: Path) -> Path:
        dataset = root / "dataset"
        (dataset / "snapshot").mkdir(parents=True)
        with (dataset / "snapshot/vertices.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["vertex_id"])
            writer.writeheader()
            writer.writerows([{"vertex_id": "a"}, {"vertex_id": "b"}])
        with (dataset / "snapshot/edges.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["edge_id", "src", "dst", "edge_type", "event_time_ms"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "edge_id": "e0",
                    "src": "a",
                    "dst": "b",
                    "edge_type": "X",
                    "event_time_ms": 0,
                }
            )
        return dataset

    @staticmethod
    def _query(query_id: str, version: int) -> dict[str, object]:
        return {
            "query_id": query_id,
            "query_class": "NEIGHBOR_TYPE_AGG",
            "graph_version": version,
            "parameters": {"seed": "a"},
            "valid_plans": ["RAW_ROWS", "LOCAL_REDUCE", "COMPRESSED_REDUCE"],
        }

    def test_resume_accepts_verified_older_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / "replay.sqlite"
            output = root / "materialized"
            graph = SqliteVersionedGraph(self._dataset(root), database)
            low = self._query("low", 0)
            materialized = graph.materialize(low, partitions=2)
            write_materialization(
                output,
                {
                    **low,
                    **materialized,
                    "graph_state_sha256": graph.state_checksum(),
                    "partition_count": 2,
                },
            )
            graph.apply_event(
                {
                    "seq": 0,
                    "changes": [
                        {
                            "action": "ADD_EDGE",
                            "record": {
                                "edge_id": "e1",
                                "src": "b",
                                "dst": "a",
                                "edge_type": "Y",
                                "event_time_ms": 1,
                            },
                        }
                    ],
                }
            )
            graph.close()

            checkpoint = _load_resumable_export(
                database, output, [low, self._query("high", 1)]
            )
            self.assertIsNotNone(checkpoint)
            assert checkpoint is not None
            self.assertEqual(checkpoint[0], 1)
            self.assertEqual(set(checkpoint[1]), {"low"})

    def test_resume_rejects_missing_query_below_committed_version(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / "replay.sqlite"
            graph = SqliteVersionedGraph(self._dataset(root), database)
            graph.apply_event({"seq": 0, "changes": []})
            graph.close()
            checkpoint = _load_resumable_export(
                database,
                root / "materialized",
                [self._query("missing-low", 0), self._query("high", 1)],
            )
            self.assertIsNone(checkpoint)


if __name__ == "__main__":
    unittest.main()
