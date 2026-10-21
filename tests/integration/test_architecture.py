"""Regression checks for responsibility boundaries and central configuration."""

import ast
import unittest

from src.common.partitioning import stable_owner
from src.common.paths import OUTPUT_ROOT, PROJECT_ROOT, WORKLOAD_ROOT, project_path
from src.methods.linucb import model_features


class ProjectLayoutTests(unittest.TestCase):
    def test_relative_output_is_project_relative(self):
        self.assertEqual(
            project_path("output/current/example"), PROJECT_ROOT / "output/current/example"
        )
        self.assertEqual(project_path(PROJECT_ROOT / "data"), PROJECT_ROOT / "data")
        self.assertEqual(WORKLOAD_ROOT, OUTPUT_ROOT / "cache/workloads")

    def test_configuration_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            model_features(-1, 0, 1, 0)
        with self.assertRaises(ValueError):
            model_features(1, 1, 1, 3)
        with self.assertRaises(ValueError):
            stable_owner("vertex", 0)

    def test_library_does_not_import_cli(self):
        for path in (PROJECT_ROOT / "src").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("scripts."), str(path))

    def test_simulator_environment_does_not_require_training_framework(self):
        source = (PROJECT_ROOT / "simulator/omnetpp/graph_query/GraphQueryApp.cc").read_text()
        self.assertNotIn("torch", source.lower())

    def test_simulator_exports_independent_performance_dimensions(self):
        source = (PROJECT_ROOT / "simulator/omnetpp/graph_query/GraphQueryApp.cc").read_text()
        required_metrics = {
            "total_wire_bytes",
            "scan_cpu_service_worker_sum_ms",
            "scan_service_critical_path_ms",
            "scanned_edges",
        }
        for metric in required_metrics:
            self.assertIn(f'\\"{metric}\\"', source)


if __name__ == "__main__":
    unittest.main()
