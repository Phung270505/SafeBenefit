import unittest

from src.analysis.ablation import (
    ABLATION_VARIANTS,
    analyze_safebenefit_ablation,
    select_ablation_plan,
)


class SafeBenefitAblationTests(unittest.TestCase):
    def test_equal_cost_plan_is_optimal_in_ablation_too(self) -> None:
        analysis = analyze_safebenefit_ablation([{
            "method": "STATIC", "initial_plan": 2,
            "forced_plan_wire_bytes": {"0": 100, "1": 100, "2": 100},
            "estimated_switch_wire_bytes": 0,
            "predicted_plan_wire_bytes": {"0": 100, "1": 100, "2": 100},
        }])
        for values in analysis["variants"].values():
            self.assertEqual(values["regret_bytes"], 0)
            self.assertEqual(values["oracle_match_rate"], 1)
            self.assertEqual(values["wrong_plan"], 0)
            self.assertEqual(values["missed_beneficial_switches"], 0)

    def test_ablation_components_are_monotonically_more_conservative(self) -> None:
        costs = {0: 1_000.0, 1: 850.0, 2: 950.0}
        actions = {
            variant: select_ablation_plan(
                variant,
                0,
                costs,
                switch_bytes=100,
                safety_margin=0.10,
                uncertainty_z=1.96,
                prediction_relative_std=0.05,
            )
            for variant in ABLATION_VARIANTS
        }
        self.assertEqual(actions["A0"], 1)
        self.assertEqual(actions["A2"], 1)
        self.assertEqual(actions["A4"], 0)
        self.assertEqual(actions["A6"], 0)

    def test_unknown_variant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_ablation_plan(
                "AX",
                0,
                {0: 10.0, 1: 5.0},
                switch_bytes=1,
                safety_margin=0,
                uncertainty_z=0,
                prediction_relative_std=0,
            )


if __name__ == "__main__":
    unittest.main()
