"""A failed dataset must not erase the other dataset's completed report."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.design import DATASET_ALIASES
from src.common.files import write_json
from src.experiment import coordinator


class CoordinatorTests(unittest.TestCase):
    def test_failure_is_local_and_does_not_stop_later_datasets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            visited = []

            class Process:
                pid = 12345
                stdout = []

                def __init__(self, _command, **kwargs):
                    dataset = kwargs["env"]["SAFEBENEFIT_DATASET_ID"]
                    visited.append(dataset)
                    self.code = 1 if len(visited) == 2 else 0
                    write_json(
                        root / DATASET_ALIASES[dataset] / "run_manifest.json",
                        {"status": "PASS" if self.code == 0 else "FAIL"},
                    )

                def wait(self):
                    return self.code

            with (
                patch.object(coordinator, "STUDY_OUTPUT_ROOT", root),
                patch.object(coordinator, "_fingerprint", return_value="frozen"),
                patch.object(coordinator.subprocess, "Popen", Process),
            ):
                self.assertEqual(coordinator._run_all(), 2)
            manifest = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(visited, list(DATASET_ALIASES))
            self.assertEqual(
                list(row["status"] for row in manifest["datasets"].values()),
                ["PASS", "FAIL", "PASS"],
            )
            self.assertNotIn("lowest_cost_method", manifest)

    def test_incompatible_output_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "run_manifest.json", {"schema_version": 1})
            with (
                patch.object(coordinator, "STUDY_OUTPUT_ROOT", root),
                patch.object(coordinator, "_fingerprint", return_value="new"),
            ):
                with self.assertRaisesRegex(RuntimeError, "incompatible"):
                    coordinator._run_all()
            self.assertTrue((root / "run_manifest.json").is_file())
