"""Switch-threshold training uses calibration only and its deployed decision rule."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.methods.switch_cost_threshold import SwitchCostThresholdSelector
from src.methods.threshold_calibration import SwitchThresholdModel
from src.methods.training import train_calibrated_methods
from src.simulation import controller


class SwitchThresholdTrainingTests(unittest.TestCase):
    def test_selector_fits_train_only_and_preserves_frozen_threshold(self):
        train = {
            "replica_id": 0, "current_plan": 0, "predicted_bytes": {0: 1000, 1: 900},
            "costs": {"0": 1000, "1": 800}, "switch_bytes": 10,
        }
        selector = SwitchCostThresholdSelector()
        artifact = selector.fit([train, train | {"replica_id": 1, "costs": {"0": 1, "1": 999999}}])
        self.assertEqual(artifact["threshold"], 0)
        decision = selector.select_plan({
            "current_plan": 0, "predicted_bytes": {0: 1000, 1: 900}, "switch_bytes": 10,
        }, (0, 1))
        self.assertEqual(decision.plan_id, 1)
        self.assertEqual(selector.threshold, artifact["threshold"])

    def test_threshold_fit_uses_deployed_switch_aware_rule(self):
        cases = [
            {"current_plan": 0, "predicted_bytes": {0: 1000, 1: 900},
             "actual_bytes": {0: 1000, 1: 1200}, "switch_bytes": 80},
            {"current_plan": 0, "predicted_bytes": {0: 1000, 1: 600},
             "actual_bytes": {0: 1000, 1: 500}, "switch_bytes": 80},
        ]
        model = SwitchThresholdModel.fit(cases, thresholds=(0, 0.01, 0.02, 0.03, 0.5))
        self.assertAlmostEqual(model.threshold, 0.02)
        self.assertEqual(model.calibration_total_bytes, 1580)
        self.assertEqual(model.select(0, {0: 1000, 1: 900}, 80), 0)
        self.assertEqual(model.select(0, {0: 1000, 1: 850}, 80), 1)

    def test_invalid_training_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            SwitchThresholdModel.fit([])
        with self.assertRaises(ValueError):
            SwitchThresholdModel.fit([{
                "current_plan": 0, "predicted_bytes": {0: 100, 1: 50},
                "actual_bytes": {0: 100, 1: 50}, "switch_bytes": -1,
            }])

    def test_training_exports_only_the_frozen_switch_baseline(self):
        samples = []
        for replica in (0, 1):
            for offset in range(3):
                samples.append(
                    {
                        "query_id": f"q-{replica}-{offset}",
                        "dataset_id": "toy",
                        "query_class": "NEIGHBOR_TYPE_AGG",
                        "replica_id": replica,
                        "selected_rows": 10 + offset * 100,
                        "distinct_groups": 2 + offset,
                        "mean_label_bytes": 3,
                        "graph_version": offset + 1,
                        "graph_scale_edges": 1000,
                        "worker_count": 4,
                        "retry_count": 0,
                        "switch_bytes": 10,
                        "costs": {
                            "0": 100 + offset * 100,
                            "1": 150 - offset * 20,
                            "2": 180 - offset * 40,
                        },
                    }
                )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "policies"
            report = train_calibrated_methods(samples, output)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue((output / "switch_threshold.tsv").exists())
            self.assertTrue((output / "switch_cost_model.tsv").exists())
            self.assertIn("method\tSWITCH_COST_THRESHOLD", (output / "switch_threshold.tsv").read_text())
            self.assertEqual(len(report["switch_threshold_sha256"]), 64)
            self.assertEqual(len(report["switch_cost_model_sha256"]), 64)
            self.assertEqual(
                report["threshold_validation_action_source"],
                "safe_point_cost_model_only",
            )
            self.assertEqual(set(report["validation_wire_bytes"]), {"SWITCH_COST_THRESHOLD"})
            self.assertEqual(set(path.name for path in output.iterdir()), {
                "switch_threshold.tsv", "switch_cost_model.tsv", "baseline_training.json"
            })
            changed_validation = [
                sample | {"costs": {key: value + 100000 for key, value in sample["costs"].items()}}
                if sample["replica_id"] == 1 else dict(sample)
                for sample in samples
            ]
            repeated = train_calibrated_methods(changed_validation, Path(folder) / "other")
            self.assertEqual(repeated["switch_threshold_sha256"], report["switch_threshold_sha256"])
            self.assertEqual(repeated["initial_plans"], report["initial_plans"])
            self.assertNotEqual(repeated["validation_wire_bytes"], report["validation_wire_bytes"])
            with (
                patch.object(controller, "RESULT_ROOT", output.parent),
                patch.object(controller, "_verify_frozen_policy", return_value="linucb-hash"),
            ):
                checksums = controller._verify_frozen_method_artifacts()
                self.assertEqual(set(checksums), {"linucb", "switch_threshold", "switch_cost_model"})
                (output / "switch_threshold.tsv").write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(controller.StudyBlocked, "checksum mismatch"):
                    controller._verify_frozen_method_artifacts()


if __name__ == "__main__":
    unittest.main()
