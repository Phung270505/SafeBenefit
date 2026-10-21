"""Strict schema/dataset resume, without legacy migrations or implicit cleanup."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.design import STUDY_SCHEMA_VERSION
from src.common.files import sha256_file
from src.experiment.runner import StudyBlocked, _load_manifest, _verify_resume_artifacts


class RunnerResumeTests(unittest.TestCase):
    def test_legacy_or_changed_dataset_is_rejected_without_deletion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run_manifest.json"
            for manifest in (
                {"schema_version": 1},
                {"schema_version": 4, "dataset_id": "fixture", "identity": {"code": "same"}},
                {"schema_version": STUDY_SCHEMA_VERSION, "dataset_id": "other", "identity": {"code": "same"}},
            ):
                path.write_text(json.dumps(manifest))
                with (
                    patch("src.experiment.runner.MANIFEST_PATH", path),
                    patch("src.experiment.runner.DATASET_SCALES", ("fixture",)),
                ):
                    with self.assertRaises(StudyBlocked):
                        _load_manifest({"code": "same"})
                self.assertTrue(path.is_file())

    def test_completed_stage_requires_valid_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / "artifact.json"
            with patch.dict(
                "src.experiment.runner.STAGE_ARTIFACTS", {"example": (artifact,)}, clear=True
            ):
                with self.assertRaises(StudyBlocked):
                    _verify_resume_artifacts({"completed_stages": ["example"]})
                artifact.write_text("{}")
                manifest = {
                    "completed_stages": ["example"],
                    "artifact_receipts": {
                        "example": {str(artifact.resolve()): sha256_file(artifact)},
                    },
                }
                _verify_resume_artifacts(manifest)
                artifact.write_text("modified")
                with self.assertRaises(StudyBlocked):
                    _verify_resume_artifacts(manifest)

    def test_incomplete_run_cannot_claim_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": STUDY_SCHEMA_VERSION,
                        "dataset_id": "fixture",
                        "identity": {"code": "same"},
                        "status": "PASS",
                        "completed_stages": [],
                    }
                )
            )
            with (
                patch("src.experiment.runner.MANIFEST_PATH", path),
                patch("src.experiment.runner.DATASET_SCALES", ("fixture",)),
            ):
                with self.assertRaisesRegex(StudyBlocked, "incomplete"):
                    _load_manifest({"code": "same"})
