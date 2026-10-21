"""Checks for the resumable, single-environment simulation batch path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.files import sha256_file
from src.common.paths import PROJECT_ROOT
from src.simulation.controller import (
    CaseSpec,
    StudyBlocked,
    _case_has_output,
    _run_case_batch,
    _verify_frozen_policy,
)


class SimulationBatchTests(unittest.TestCase):
    def test_case_output_requires_artifacts_and_full_query_count(self) -> None:
        spec = CaseSpec("citation_hepth", "evaluation", "STATIC", "0", "case-a")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            input_root = output / "cache" / "materialized" / "evaluation" / spec.dataset
            input_root.mkdir(parents=True)
            (input_root / "queries.tsv").write_text("q0\n", encoding="utf-8")
            case = output / "simulation" / spec.run_id
            case.mkdir(parents=True)
            (case / "queries.jsonl").write_text("{}\n", encoding="utf-8")
            with patch("src.simulation.controller.RESULT_ROOT", output):
                self.assertFalse(_case_has_output(spec))
                (case / "metrics.sca").write_text("scalar x y 1\n", encoding="utf-8")
                self.assertFalse(_case_has_output(spec))
                (case / ".complete").write_text("PASS\n", encoding="utf-8")
                self.assertTrue(_case_has_output(spec))

    def test_batch_wrapper_reuses_ready_environment(self) -> None:
        outer = (PROJECT_ROOT / "simulator/omnetpp/tools/run_batch.sh").read_text()
        inner = (PROJECT_ROOT / "simulator/omnetpp/tools/run_batch_inner.sh").read_text()
        self.assertIn('"$OPP_ENV" run', outer)
        self.assertIn("SAFEBENEFIT_ENV_READY=1", inner)
        self.assertIn("run_case.sh", inner)
        self.assertIn("18", inner)
        self.assertIn("%$'\\r'", inner)
        case = (PROJECT_ROOT / "simulator/omnetpp/tools/run_case.sh").read_text()
        self.assertIn(".complete", case)

    def test_batch_manifest_uses_unix_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "config"
            config.mkdir()
            (config / "runtime_calibration.json").write_text(
                '{"scan_cpu_units_per_second":1,"codec_cpu_units_per_second":1,'
                '"io_bytes_per_second":1}',
                encoding="utf-8",
            )
            spec = CaseSpec(
                "citation_hepth", "evaluation", "STATIC", "0", "line-ending-case"
            )
            with (
                patch("src.simulation.controller.RESULT_ROOT", root),
                patch("src.simulation.controller._wsl_script", return_value=["true"]),
                patch("src.simulation.controller._run", return_value=0),
            ):
                _run_case_batch("line-endings", [spec])
            payload = (root / "cache/batches/line-endings.tsv").read_bytes()
            self.assertNotIn(b"\r", payload)
            self.assertTrue(payload.endswith(b"\n"))

    def test_evaluation_and_deployment_policy_are_both_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            policies = root / "policies"
            policies.mkdir()
            deployment = policies / "policy.tsv"
            evaluation = policies / "candidate_policy.tsv"
            deployment.write_text("same-policy\n", encoding="utf-8")
            evaluation.write_text("same-policy\n", encoding="utf-8")
            digest = sha256_file(deployment)
            (policies / "training.json").write_text(
                "{"
                f'"deployment_policy_sha256":"{digest}",'
                f'"candidate_policy_sha256":"{digest}"'
                "}",
                encoding="utf-8",
            )
            with patch("src.simulation.controller.RESULT_ROOT", root):
                self.assertEqual(_verify_frozen_policy(), digest)
                evaluation.write_text("tampered\n", encoding="utf-8")
                with self.assertRaises(StudyBlocked):
                    _verify_frozen_policy()


if __name__ == "__main__":
    unittest.main()
