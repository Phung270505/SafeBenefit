"""Both encodings must preserve results and create a real byte trade-off."""

import unittest
from collections import Counter

from src.plans.encoding import (
    decode_labels,
    encode_labels,
    has_rank_reversal,
    winning_plans,
)


class EncodingTests(unittest.TestCase):
    def test_both_plans_decode_to_same_multiset(self):
        labels = [10, 20, 10, 30, 10]
        expected = Counter(labels)
        for plan in (0, 1, 2):
            self.assertEqual(decode_labels(encode_labels(labels, plan), plan), expected)

    def test_rank_reversal_is_real_not_injected_cost(self):
        unique = [10000, 20000]
        repeated = [1] * 20
        self.assertLess(len(encode_labels(unique, 0)), len(encode_labels(unique, 1)))
        self.assertLess(len(encode_labels(repeated, 1)), len(encode_labels(repeated, 0)))
        many_large_groups = [10_000_000 + index for index in range(200)]
        samples = [unique, repeated, many_large_groups]
        self.assertTrue(has_rank_reversal(samples, require_all=True))
        self.assertEqual(winning_plans(samples), {0, 1, 2})

    def test_unknown_plan_rejected(self):
        with self.assertRaises(ValueError):
            encode_labels([1], 3)
        for payload in (b"", b"1 2 3 "):
            with self.assertRaises(ValueError):
                decode_labels(payload, 3)

    def test_compressed_codec_rejects_corruption(self):
        payload = encode_labels([1, 1, 1, 1000], 2)
        with self.assertRaises(ValueError):
            decode_labels(payload + b"x", 2)


if __name__ == "__main__":
    unittest.main()
