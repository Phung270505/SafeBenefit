"""The two workload streams reuse one exact snapshot degree scan."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.datasets.snapshot_statistics import sample_degree_buckets


class SnapshotStatisticsTests(unittest.TestCase):
    def test_calibration_and_evaluation_share_a_single_source_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "snapshot").mkdir()
            (root / "metadata").mkdir()
            (root / "metadata/manifest.json").write_text("{}")
            source = root / "snapshot/edges.csv"
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("edge_id", "src", "dst", "edge_type"))
                for index in range(1000):
                    writer.writerow((str(index), f"v{index}", f"v{(index + 1) % 1000}", "X"))
            original = Path.open
            reads = []

            def tracked(path, *args, **kwargs):
                if path == source:
                    reads.append(path)
                return original(path, *args, **kwargs)

            with (
                patch("src.datasets.snapshot_statistics.OUTPUT_ROOT", root / "output"),
                patch.object(Path, "open", tracked),
            ):
                calibration = sample_degree_buckets(root, "fixture", "calibration_parameters", 768)
                evaluation = sample_degree_buckets(root, "fixture", "evaluation_parameters", 768)
                self.assertEqual(len(reads), 1)
                self.assertEqual(sum(map(len, calibration.values())), 768)
                self.assertEqual(sum(map(len, evaluation.values())), 768)
                self.assertNotEqual(calibration, evaluation)
                self.assertEqual(
                    calibration,
                    sample_degree_buckets(root, "fixture", "calibration_parameters", 768),
                )
