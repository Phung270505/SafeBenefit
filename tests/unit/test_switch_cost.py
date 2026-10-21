import tempfile
import unittest
from pathlib import Path

from src.plans.switch_cost import SwitchCostModel


class SwitchCostModelTests(unittest.TestCase):
    def test_estimate_scales_with_workers_and_retries(self) -> None:
        model = SwitchCostModel(0, 128, "test calibration")
        self.assertEqual(model.estimate(1), 128)
        self.assertEqual(model.estimate(4), 512)
        self.assertEqual(model.estimate(16, retry_rounds=1), 4096)

    def test_export_is_deterministic_and_invalid_context_is_rejected(self) -> None:
        model = SwitchCostModel(0, 128, "test calibration")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "switch.tsv"
            model.export(path)
            first = path.read_bytes()
            model.export(path)
            self.assertEqual(path.read_bytes(), first)
        with self.assertRaises(ValueError):
            model.estimate(3)


if __name__ == "__main__":
    unittest.main()
