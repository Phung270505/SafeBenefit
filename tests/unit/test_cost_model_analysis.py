import unittest

from src.analysis.cost_model import analyze_cost_model


class CostModelAnalysisTests(unittest.TestCase):
    def test_prediction_and_switch_errors_are_reported_separately(self) -> None:
        row = {
            "method": "STATIC",
            "plan_id": 0,
            "initial_plan": 0,
            "partition_skew_cv": 0.25,
            "estimated_switch_wire_bytes": 100,
            "switch_bytes": 0,
            "predicted_plan_wire_bytes": {"0": 1000, "1": 800, "2": 900},
            "forced_plan_wire_bytes": {"0": 1000, "1": 850, "2": 950},
        }
        switched = dict(row, method="SAFE_BENEFIT", plan_id=1, switch_bytes=100)
        report = analyze_cost_model([row, switched])
        self.assertEqual(report["queries"], 1)
        self.assertIn("MODERATE", report["skew_prediction"])
        self.assertEqual(
            report["switch_cost_validation"]["mean_absolute_error_bytes"], 0
        )


if __name__ == "__main__":
    unittest.main()
