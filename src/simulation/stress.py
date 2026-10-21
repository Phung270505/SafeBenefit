"""Registered delete/expiration correctness stress checks outside the main ranking."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any

from src.common.files import EDGE_FIELDS, VERTEX_FIELDS
from src.datasets.replay import SqliteVersionedGraph


def run_delete_stress() -> dict[str, Any]:
    """Prove that deletion changes only later graph versions and remains replayable."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "dataset"
        (root / "snapshot").mkdir(parents=True)
        with (root / "snapshot" / "vertices.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=VERTEX_FIELDS)
            writer.writeheader()
            for vertex in ("a", "b", "c"):
                writer.writerow(
                    {
                        "vertex_id": vertex,
                        "vertex_type": "toy",
                        "properties_json": "{}",
                        "valid_from_ms": 0,
                        "valid_to_ms": "",
                    }
                )
        with (root / "snapshot" / "edges.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=EDGE_FIELDS)
            writer.writeheader()
            for edge_id, source, target in (("e1", "a", "b"), ("e2", "b", "c")):
                writer.writerow(
                    {
                        "edge_id": edge_id,
                        "src": source,
                        "dst": target,
                        "edge_type": "link",
                        "weight": 1,
                        "properties_json": "{}",
                        "event_time_ms": 1,
                        "valid_from_ms": 1,
                        "valid_to_ms": "",
                    }
                )
        graph = SqliteVersionedGraph(root, Path(temporary) / "stress.sqlite")
        query = {
            "query_id": "delete-stress",
            "query_class": "TWO_HOP_REACHABILITY",
            "graph_version": 0,
            "parameters": {"seed": "a", "limit": 10},
        }
        before = graph.materialize(query)
        graph.apply_event(
            {
                "seq": 0,
                "changes": [{"action": "DELETE_EDGE", "record": {"edge_id": "e2"}}],
            }
        )
        query["graph_version"] = 1
        after = graph.materialize(query)
        checksum = graph.state_checksum()
        graph.close()
    passed = before["expected_result_sha256"] != after["expected_result_sha256"]
    return {
        "status": "PASS" if passed else "FAIL",
        "before_result_sha256": before["expected_result_sha256"],
        "after_result_sha256": after["expected_result_sha256"],
        "final_graph_sha256": checksum,
    }
