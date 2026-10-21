"""Binary compressed payloads remain lossless across UDP application fragments."""
import unittest
from collections import Counter

from src.plans.encoding import decode_labels, encode_labels


class CompressedFragmentTests(unittest.TestCase):
    def test_large_binary_stream_reassembles_without_newline_truncation(self):
        labels = list(range(50000))
        encoded = encode_labels(labels, 2)
        self.assertGreater(len(encoded), 65507)
        fragments = [encoded[start:start + 60000] for start in range(0, len(encoded), 60000)]
        self.assertTrue(all(len(payload) <= 60000 for payload in fragments))
        self.assertEqual(decode_labels(b"".join(fragments), 2), Counter(labels))
