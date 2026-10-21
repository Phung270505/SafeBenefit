"""Canonical equality cannot substitute for byte/order integrity of input artifacts."""

import hashlib
import os
import struct
import tempfile
import unittest
from pathlib import Path

from src.common.files import stable_json
from src.datasets.export import _validate_query_contract
from src.datasets.replay import write_materialization


class TokenIntegrityTests(unittest.TestCase):
    def _artifact(self, root):
        source = {
            "query_id": "q",
            "query_class": "NEIGHBOR_TYPE_AGG",
            "graph_version": 1,
            "parameters": {"seed": "a"},
            "valid_plans": [0, 1, 2],
        }
        result = [[1, 1], [2, 1], [3, 1]]
        contract = write_materialization(
            root,
            dict(
                source,
                reducer="COUNT",
                worker_tokens=[[1, 2, 3]],
                expected_result=result,
                expected_result_sha256=hashlib.sha256(stable_json(result).encode()).hexdigest(),
            ),
        )
        _validate_query_contract(root, source, contract)
        return source, contract

    def test_reordered_tokens_are_rejected_even_with_the_same_multiset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, contract = self._artifact(root)
            path = root / contract["worker_token_files"][0]
            timestamp = path.stat().st_mtime_ns
            path.write_bytes(struct.pack("<3Q", 3, 2, 1))
            os.utime(path, ns=(timestamp, timestamp + 10_000_000))
            with self.assertRaisesRegex(ValueError, "byte/order"):
                _validate_query_contract(root, source, contract)

    def test_cached_validation_does_not_hide_modified_oracle_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, contract = self._artifact(root)
            (root / contract["oracle_rows_file"]).write_text("corrupted")
            with self.assertRaisesRegex(ValueError, "oracle"):
                _validate_query_contract(root, source, contract)
