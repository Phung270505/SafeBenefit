from __future__ import annotations

import unittest

from src.methods.registry import METHOD_ORDER, create_selector


def calibration_rows() -> list[dict]:
    rows = []
    for index in range(6):
        rows.append({
            "query_id": f"q{index}", "dataset_id": "fixture", "query_class": "NEIGHBOR_TYPE_AGG",
            "replica_id": index % 2, "current_plan": 0,
            "predicted_bytes": {0: 120.0, 1: 90.0, 2: 100.0}, "switch_bytes": 2.0,
            "costs": {"0": 120 + index, "1": 90 + index * 2, "2": 100 - index},
            "context_vector": [float(index + column + 1) for column in range(17)],
        })
    return rows


class AllMethodsSmokeTests(unittest.TestCase):
    def test_all_registered_methods_fit_and_select_on_same_context(self) -> None:
        rows = calibration_rows()
        context = {
            "dataset_id": "fixture", "query_class": "NEIGHBOR_TYPE_AGG", "current_plan": 0,
            "predicted_bytes": {0: 120.0, 1: 90.0, 2: 100.0},
            "prediction_std": {0: 1.0, 1: 1.0, 2: 1.0}, "switch_bytes": 2.0,
            "safe_point": True, "switch_budget": 1, "progress": 0.25,
            "context_vector": rows[0]["context_vector"], "ood_threshold": 1e12,
        }
        decisions = {}
        for method in METHOD_ORDER:
            selector = create_selector(method)
            selector.fit(rows)
            decision = selector.select_plan(context, (0, 1, 2))
            self.assertIn(decision.plan_id, (0, 1, 2))
            decisions[method] = decision
        self.assertEqual(tuple(decisions), METHOD_ORDER)


if __name__ == "__main__":
    unittest.main()
