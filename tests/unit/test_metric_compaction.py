"""Raw result maps remain auditable without inflating derived metric records."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.simulation.controller import StudyBlocked, _result_hash, _validate_case


class MetricCompactionTest(unittest.TestCase):
    def test_result_is_hashed_then_removed_from_derived_row(self) -> None:
        result = {"1": 2, "3": 1}
        contract = {
            "query_id": "q1",
            "query_class": "NEIGHBOR_TYPE_AGG",
            "expected_result_sha256": _result_hash(result),
            "graph_version": 42,
            "scale_id": "fixture",
            "schedule_id": 0,
            "schedule_profile": "stable",
            "replica_id": 0,
            "graph_version_bucket": "low",
        }
        raw = {
            "query_class": "NEIGHBOR_TYPE_AGG",
            "query_id": "q1", "method": "STATIC", "result": result,
            "outcome": "SUCCESS", "graph_version": 42, "workers": 4,
            "total_modeled_cpu_service_ms": 1.0, "latency_simtime_ms": 2.0,
            "selected_rows": 3, "plan": 0, "decision_latency_ns": 10,
            "payload_bytes": 10, "control_bytes": 5, "retry_bytes": 0,
            "switch_wire_bytes": 0, "total_wire_bytes": 15,
        }
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "queries.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")
            with (
                patch("src.simulation.controller._contracts", return_value={"q1": contract}),
                patch("src.simulation.controller.host_wire_bytes", return_value=15),
            ):
                rows = _validate_case(folder, "evaluation", "fixture", "STATIC")
                self.assertEqual(len(rows), 1)
                self.assertNotIn("result", rows[0])
                self.assertEqual(rows[0]["result_sha256"], contract["expected_result_sha256"])
                self.assertEqual(rows[0]["total_wire_bytes"], 15)

                (folder / "queries.jsonl").write_text(
                    json.dumps(raw) + "\n" + json.dumps(raw) + "\n", encoding="utf-8"
                )
                with self.assertRaises(StudyBlocked):
                    _validate_case(folder, "evaluation", "fixture", "STATIC")


if __name__ == "__main__":
    unittest.main()
