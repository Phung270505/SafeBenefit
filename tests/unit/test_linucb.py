"""Determinism, masking, export and validation tests for LINUCB."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.methods.linucb import (
    FeatureScaler,
    LinearPolicy,
    enforce_fallback,
    fit_policy,
    model_features,
    select_action,
)
from src.methods.linucb_training import train_linucb


class LinUcbTests(unittest.TestCase):
    def setUp(self):
        self.contexts = [
            model_features(10, 10, 8, 0),
            model_features(1000, 4, 1, 0),
            model_features(20, 20, 8, 1),
            model_features(2000, 5, 1, 2),
        ]
        self.scaler = FeatureScaler.fit(self.contexts)

    def test_regularized_fit_and_action_mask(self):
        policy = fit_policy(
            self.contexts,
            {
                0: [-0.1, -0.8, -0.2, -1.0],
                1: [-0.3, -0.2, -0.4, -0.1],
                2: [-0.4, -0.1, -0.5, -0.05],
            },
            self.scaler,
            ridge=0.1,
        )
        state = self.scaler.transform(self.contexts[0])
        self.assertEqual(select_action(policy, state, (True, False, False)), 0)
        self.assertEqual(select_action(policy, state, (False, True, False)), 1)
        self.assertEqual(select_action(policy, state, (False, False, True)), 2)
        with self.assertRaises(ValueError):
            select_action(policy, state, (False, False, False))
        self.assertNotEqual(
            policy.precision_by_action[0],
            policy.precision_by_action[1],
            "sequential LinUCB must maintain action-specific uncertainty",
        )

    def test_scaler_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            self.scaler.transform([np.nan] * 17)

    def test_scaler_requires_two_rows_and_computes_mahalanobis_distance(self):
        with self.assertRaises(ValueError):
            FeatureScaler.fit([self.contexts[0]])
        self.assertTrue(np.isfinite(self.scaler.mahalanobis_distance(self.contexts[0])))
        shifted = self.contexts[0].copy()
        shifted[0] += 100
        self.assertGreater(
            self.scaler.mahalanobis_distance(shifted),
            self.scaler.mahalanobis_distance(self.contexts[0]),
        )

    def test_fallback_flag_changes_checksum(self):
        width = len(self.contexts[0]) + 1
        precision = tuple(np.eye(width).reshape(-1).tolist())
        policy = LinearPolicy(
            tuple((0.0,) * width for _ in range(3)),
            tuple(precision for _ in range(3)),
            0.1,
        )
        from src.methods.linucb import export_policy

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.tsv"
            before = export_policy(policy, path, self.scaler, ridge=0.1, ood_threshold=4)
            after = enforce_fallback(path)
            self.assertNotEqual(before, after)
            self.assertIn("force_fallback\t1\n", path.read_text(encoding="utf-8"))

    def test_training_is_repeatable_and_candidate_passes_toy_gate(self):
        samples = []
        for local_index in (0, 1):
            for offset in (0, 1):
                samples.append(
                    {
                        "query_id": f"toy-{local_index}-{offset}",
                        "dataset_id": "toy",
                        "query_class": "NEIGHBOR_TYPE_AGG",
                        "local_index": local_index,
                        "selected_rows": 10 + local_index + offset * 100,
                        "distinct_groups": 2 + local_index,
                        "mean_label_bytes": 3,
                        "graph_version": 10 + offset,
                        "retry_count": 0,
                        "switch_bytes": 10,
                        "costs": {
                            "0": 100 + offset * 200,
                            "1": 200 - offset * 50,
                            "2": 180 - offset * 60,
                        },
                    }
                )
        with tempfile.TemporaryDirectory() as folder:
            left = Path(folder) / "left"
            right = Path(folder) / "right"
            first = train_linucb(samples, left)
            second = train_linucb(samples, right)
            self.assertEqual(first["selected_ridge"], second["selected_ridge"])
            self.assertEqual(
                (left / "candidate_policy.tsv").read_bytes(),
                (right / "candidate_policy.tsv").read_bytes(),
            )
            self.assertEqual(
                first["candidate_policy_sha256"], second["candidate_policy_sha256"]
            )

    def test_training_rejects_sample_without_registered_split(self):
        sample = {
            "query_id": "missing-split",
            "dataset_id": "toy",
            "query_class": "NEIGHBOR_TYPE_AGG",
            "selected_rows": 10,
            "distinct_groups": 2,
            "mean_label_bytes": 3,
            "graph_version": 1,
            "retry_count": 0,
            "switch_bytes": 10,
            "costs": {"0": 100, "1": 90, "2": 110},
        }
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            train_linucb([sample], Path(folder))


if __name__ == "__main__":
    unittest.main()
