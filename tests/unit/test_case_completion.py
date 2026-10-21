"""A successful simulator exit is not enough if a query batch was truncated."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.simulation import controller


class CaseCompletionTest(unittest.TestCase):
    def test_truncated_batch_cannot_be_resumed_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = controller.CaseSpec("fixture", "evaluation", "STATIC", "0", "case")
            inputs = root / "cache" / "materialized" / "evaluation" / "fixture"
            outputs = root / "simulation" / "case"
            inputs.mkdir(parents=True)
            outputs.mkdir(parents=True)
            (inputs / "queries.tsv").write_text("q0\nq1\nq2\n", encoding="utf-8")
            (outputs / "queries.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            (outputs / "metrics.sca").write_text("scalar test 1\n", encoding="utf-8")
            (outputs / ".complete").write_text("PASS\n", encoding="utf-8")
            with patch.object(controller, "RESULT_ROOT", root):
                self.assertFalse(controller._case_has_output(spec))
                (outputs / "queries.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")
                self.assertTrue(controller._case_has_output(spec))


if __name__ == "__main__":
    unittest.main()
