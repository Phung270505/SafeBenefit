from __future__ import annotations

import unittest

from src.common.paths import PROJECT_ROOT


class ProjectLayoutTests(unittest.TestCase):
    def test_public_layout_and_single_entry_point(self) -> None:
        for relative in (
            "run.py", "README.md", "pyproject.toml", "environment", "config/study.json",
            "data/raw", "data/processed", "data/fixtures", "src/common", "src/datasets",
            "src/queries", "src/plans", "src/methods", "src/simulation", "src/experiment",
            "src/analysis", "src/reporting", "simulator/omnetpp", "tests/unit",
            "tests/integration", "tests/smoke",
            "output/current",
        ):
            self.assertTrue((PROJECT_ROOT / relative).exists(), relative)

    def test_legacy_top_level_directories_are_absent(self) -> None:
        for name in ("result", "report", "scripts", "schemas"):
            self.assertFalse((PROJECT_ROOT / name).exists(), name)

    def test_only_ldbc_scales_are_registered(self) -> None:
        from src.common.config import load_study_config
        from src.common.design import ACTIVE_DATASETS
        self.assertEqual(tuple(load_study_config()["dataset_scales"]), ACTIVE_DATASETS)
        self.assertTrue(all(dataset.startswith("ldbc_snb_") for dataset in ACTIVE_DATASETS))


if __name__ == "__main__":
    unittest.main()
