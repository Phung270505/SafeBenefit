import unittest

from src.analysis.oracle import choose_adaptive_oracle, oracle_diagnostics


class AdaptiveOracleTests(unittest.TestCase):
    def test_tied_optimal_plan_is_not_a_wrong_decision(self) -> None:
        oracle = choose_adaptive_oracle({0: 1000, 1: 900, 2: 900}, 0, 100)
        self.assertEqual(oracle.optimal_plan_ids, (0, 1, 2))
        for selected in (0, 1, 2):
            diagnostic = oracle_diagnostics(selected, 0, 1000, oracle)
            self.assertEqual(diagnostic["regret_bytes"], 0)
            self.assertTrue(diagnostic["oracle_match"])
            self.assertFalse(diagnostic["wrong_plan"])
            self.assertFalse(diagnostic["missed_beneficial_switch"])

    def test_switching_cost_can_change_the_oracle_plan(self) -> None:
        no_switch_cost = choose_adaptive_oracle({0: 1_000, 1: 900, 2: 950}, 0, 0)
        switch_aware = choose_adaptive_oracle({0: 1_000, 1: 900, 2: 950}, 0, 150)
        self.assertEqual(no_switch_cost.plan_id, 1)
        self.assertEqual(switch_aware.plan_id, 0)

    def test_diagnostics_distinguish_unnecessary_and_missed_switches(self) -> None:
        keep_oracle = choose_adaptive_oracle({0: 1_000, 1: 950}, 0, 100)
        unnecessary = oracle_diagnostics(1, 0, 1_100, keep_oracle)
        self.assertTrue(unnecessary["unnecessary_switch"])

        switch_oracle = choose_adaptive_oracle({0: 1_000, 1: 700}, 0, 100)
        missed = oracle_diagnostics(0, 0, 1_000, switch_oracle)
        self.assertTrue(missed["missed_beneficial_switch"])
        self.assertEqual(missed["regret_bytes"], 200)


if __name__ == "__main__":
    unittest.main()
