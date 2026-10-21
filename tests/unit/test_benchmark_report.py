"""Reports preserve dominance warnings and report tied methods without forcing a winner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis.statistics import analyze_results
from src.common.files import write_json, write_jsonl
from src.datasets.discrimination import summarize_plan_costs
from src.queries.registry import QUERY_CLASSES
from src.reporting import report
from tests.unit.test_independent_study import study_rows


class BenchmarkReportTests(unittest.TestCase):
    def test_report_and_audit_include_both_diagnostics_and_valid_ties(self):
        rows = study_rows()
        for row in rows:
            row["payload_size_bucket"] = "LT_10KB"
        analysis = analyze_results(rows, resamples=20)
        diagnostics = summarize_plan_costs([
            {"dataset_id": rows[0]["dataset_id"], "query_id": query_class,
             "query_class": query_class, "costs": {"0": 100, "1": 100, "2": 100}}
            for query_class in QUERY_CLASSES
        ])
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            root = project / "output/current/ldbc_sf01"
            path = root / "report/final_results.md"
            audit = root / "run_manifest_audit.json"
            write_jsonl(root / "metrics/per_query.jsonl", rows)
            for filename, scope in (
                ("payload_gates.json", "exact_application_payload_before_INET_wire_validation"),
                ("gates.json", "INET_MAC_wire_bytes"),
            ):
                write_json(root / "calibration" / filename, diagnostics | {"metric_scope": scope})
            with (
                patch.object(report, "PROJECT_ROOT", project),
                patch.object(report, "OUTPUT_ROOT", root),
                patch.object(report, "REPORT_PATH", path),
                patch.object(report, "TABLE_PATH", root / "metrics/method_summary.csv"),
                patch.object(report, "AUDIT_PATH", audit),
            ):
                report.generate_report({"status": "PASS"}, analysis)
            text = path.read_text(encoding="utf-8")
            self.assertIn("INET_MAC_wire_bytes", text)
            self.assertIn("exact_application_payload", text)
            self.assertIn("zero is a valid outcome", text)
            self.assertIn("PRACTICALLY_EQUIVALENT", text)
            for method in analysis["lowest_cost_methods"]:
                self.assertIn(method, text)
            self.assertEqual(
                len([key for key in json.loads(audit.read_text())["artifacts"] if "gates.json" in key]),
                2,
            )


if __name__ == "__main__":
    unittest.main()
