import unittest

from src.plans.cost import PlanStatistics, estimate_plan_wire_bytes


class CostModelTests(unittest.TestCase):
    def test_udp_framing_and_fragmentation_are_explicit(self) -> None:
        statistics = PlanStatistics(
            selected_rows=10_000,
            distinct_groups=250,
            mean_label_bytes=8,
            worker_count=4,
        )
        payload_only = estimate_plan_wire_bytes(statistics, network_header_bytes=0)
        framed = estimate_plan_wire_bytes(statistics)
        self.assertTrue(all(framed[plan] > payload_only[plan] for plan in (0, 1, 2)))
        self.assertGreater(
            framed[0] - payload_only[0],
            framed[1] - payload_only[1],
        )

    def test_invalid_network_contract_is_rejected(self) -> None:
        statistics = PlanStatistics(10, 2, 3, 4)
        with self.assertRaises(ValueError):
            estimate_plan_wire_bytes(statistics, network_header_bytes=-1)
        with self.assertRaises(ValueError):
            estimate_plan_wire_bytes(statistics, mtu_bytes=42, network_header_bytes=42)


if __name__ == "__main__":
    unittest.main()
