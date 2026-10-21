import unittest

from src.methods.decision import TransferCost, TransferState, choose_transfer_plan


class TransferOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.costs = {
            "raw": TransferCost(1000, control_bytes=50),
            "partial": TransferCost(500, control_bytes=50, switch_bytes=100),
        }

    def test_static_never_switches(self):
        decision = choose_transfer_plan("STATIC", "raw", self.costs)
        self.assertEqual((decision.action, decision.selected_plan), ("KEEP", "raw"))

    def test_threshold_and_benefit_count_switch_bytes_once(self):
        threshold = choose_transfer_plan("SWITCH_COST_THRESHOLD", "raw", self.costs)
        benefit = choose_transfer_plan("SAFE_BENEFIT", "raw", self.costs)
        self.assertEqual((threshold.selected_plan, benefit.selected_plan), ("partial", "partial"))
        self.assertEqual(benefit.predicted_saving_bytes, 400)
        self.assertEqual(benefit.selected_bytes, 650)

    def test_threshold_and_benefit_are_not_the_same_algorithm(self):
        marginal = {
            "raw": TransferCost(1_000),
            "partial": TransferCost(850, switch_bytes=50),
        }
        threshold = choose_transfer_plan("SWITCH_COST_THRESHOLD", "raw", marginal)
        benefit = choose_transfer_plan("SAFE_BENEFIT", "raw", marginal)
        self.assertEqual(threshold.action, "SWITCH")
        self.assertEqual(benefit.action, "KEEP")

    def test_switch_cost_threshold_accounts_for_overhead(self):
        expensive = {"raw": TransferCost(1000), "partial": TransferCost(900, switch_bytes=75)}
        cheap = {"raw": TransferCost(1000), "partial": TransferCost(900, switch_bytes=0)}
        self.assertEqual(choose_transfer_plan("SWITCH_COST_THRESHOLD", "raw", expensive).action, "KEEP")
        self.assertEqual(choose_transfer_plan("SWITCH_COST_THRESHOLD", "raw", cheap).action, "SWITCH")

    def test_guards_and_rl_fallback(self):
        self.assertEqual(choose_transfer_plan("SAFE_BENEFIT", "raw", self.costs, safe_point=False).action, "KEEP")
        fallback = choose_transfer_plan("LINUCB", "raw", self.costs)
        self.assertEqual((fallback.method, fallback.selected_plan), ("LINUCB", "partial"))
        with self.assertRaises(ValueError):
            choose_transfer_plan("LINUCB", "raw", self.costs, rl_action=lambda _: "invalid")

    def test_rl_policy_can_override_cost_model_even_when_current_is_estimated_best(self):
        costs = {
            "raw": TransferCost(100),
            "partial": TransferCost(200, switch_bytes=10),
        }
        decision = choose_transfer_plan(
            "LINUCB", "raw", costs, rl_action=lambda _: "partial"
        )
        self.assertEqual((decision.action, decision.selected_plan), ("SWITCH", "partial"))

    def test_reward_is_transfer_primary(self):
        small = TransferCost(100, control_bytes=20, retry_bytes=30, switch_bytes=40)
        large = TransferCost(200)
        self.assertLess(small.total_bytes, large.total_bytes)
        self.assertEqual(small.total_bytes, 190)

    def test_invalid_cost_rejected(self):
        with self.assertRaises(ValueError):
            TransferCost(float("nan")).total_bytes

    def test_state_has_only_observable_transfer_features(self):
        state = TransferState(1000, 500, .5, 4, 0, 100, 1, 40)
        self.assertEqual(len(state.vector()), 12)

    def test_uncertainty_and_staleness_can_prevent_switch(self):
        uncertain = choose_transfer_plan(
            "SAFE_BENEFIT",
            "raw",
            self.costs,
            uncertainty_by_plan={"raw": 200, "partial": 200},
        )
        self.assertEqual(uncertain.action, "KEEP")
        stale = choose_transfer_plan(
            "SAFE_BENEFIT",
            "raw",
            self.costs,
            state_graph_version=10,
            statistics_graph_version=9,
        )
        self.assertEqual(stale.reason, "stale_statistics_guard")
        with self.assertRaises(ValueError):
            TransferState(1, 1, .5, 4, 0, 100, 1, 40).vector()
