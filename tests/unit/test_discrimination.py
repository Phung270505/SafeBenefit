"""Valid benchmark outcomes include dominance, ties and identical decisions."""

import unittest

from src.datasets.discrimination import (
    count_action_disagreements,
    summarize_plan_costs,
)
from src.queries.registry import QUERY_CLASSES


class WorkloadDiscriminationTests(unittest.TestCase):
    @staticmethod
    def samples(costs=None):
        return [
            {"query_id": query_class, "dataset_id": "a", "query_class": query_class,
             "costs": costs or {"0": 1000, "1": 2000, "2": 100}}
            for query_class in QUERY_CLASSES
        ]

    def test_single_plan_dominance_passes_with_warning(self):
        gate = summarize_plan_costs(self.samples())
        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["errors"], [])
        self.assertTrue(gate["warnings"])
        self.assertFalse(gate["all_three_plans_win"])
        self.assertTrue(all(value == [2] for value in gate["winners_by_query_class"].values()))

    def test_all_tied_is_valid_and_not_assigned_to_plan_zero(self):
        gate = summarize_plan_costs(self.samples({"0": 0, "1": 0, "2": 0}))
        self.assertEqual(gate["status"], "PASS")
        self.assertTrue(all(value == 1 for value in gate["tie_counts_by_query_class"].values()))
        self.assertTrue(all(value == [] for value in gate["winners_by_query_class"].values()))

    def test_empty_missing_duplicate_and_bad_costs_still_fail(self):
        samples = self.samples()
        for rows in (
            [], samples[:-1], samples + samples[:1],
            self.samples({"0": 1, "1": 2}),
            self.samples({"0": -1, "1": 2, "2": 3}),
            self.samples({"0": 1.5, "1": 2, "2": 3}),
        ):
            with self.subTest(rows=rows):
                self.assertEqual(summarize_plan_costs(rows)["status"], "FAIL")

    def test_evaluation_allows_identical_actions(self):
        identical = [
            {"scenario": "Stable", "dataset_id": "a", "query_id": "q", "plan": 1},
            {"scenario": "Stable", "dataset_id": "a", "query_id": "q", "plan": 1},
        ]
        self.assertEqual(count_action_disagreements(identical), 0)
        self.assertEqual(count_action_disagreements(identical + [identical[0] | {"plan": 0}]), 1)


if __name__ == "__main__":
    unittest.main()
