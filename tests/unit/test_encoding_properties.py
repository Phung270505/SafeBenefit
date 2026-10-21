"""Seeded property tests for every registered transfer encoding."""

import random
import unittest
from collections import Counter

from src.plans.encoding import decode_labels, encode_labels
from src.queries.registry import PLAN_NAMES


class EncodingPropertyTests(unittest.TestCase):
    def test_round_trip_for_broad_generated_domain(self):
        """Exercise empty, repeated, sparse and uint64-boundary label multisets."""
        generator = random.Random(42)
        cases: list[list[int]] = [
            [],
            [0],
            [0xFFFFFFFFFFFFFFFF],
            [7] * 2_000,
            list(range(2_000)),
        ]
        boundary = (0, 1, 127, 128, 255, 256, 2**32 - 1, 2**63, 2**64 - 1)
        for _ in range(500):
            size = generator.randrange(0, 300)
            vocabulary = [generator.choice(boundary)]
            vocabulary.extend(generator.getrandbits(64) for _ in range(generator.randrange(1, 40)))
            cases.append([generator.choice(vocabulary) for _ in range(size)])

        for case_number, labels in enumerate(cases):
            expected = Counter(labels)
            for plan in PLAN_NAMES:
                with self.subTest(case=case_number, plan=plan):
                    payload = encode_labels(labels, plan)
                    self.assertEqual(decode_labels(payload, plan), expected)

    def test_encoder_rejects_values_outside_unsigned_64_bit_domain(self):
        for invalid in (-1, 2**64):
            for plan in PLAN_NAMES:
                with self.subTest(value=invalid, plan=plan), self.assertRaises(ValueError):
                    encode_labels([invalid], plan)


if __name__ == "__main__":
    unittest.main()
