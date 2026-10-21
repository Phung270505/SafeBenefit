"""LDBC experiment registry and process-isolated dataset scope."""

from __future__ import annotations

import os

from src.queries.registry import QUERY_CLASSES

STUDY_SCHEMA_VERSION = 5

DATASET_ALIASES = {
    "ldbc_snb_i1_v100_sf01": "ldbc_sf01",
    "ldbc_snb_i1_v100_sf1": "ldbc_sf1",
    "ldbc_snb_i1_v100_sf10": "ldbc_sf10",
}
DATASET_IDS = tuple(DATASET_ALIASES)
ACTIVE_DATASET = os.environ.get("SAFEBENEFIT_DATASET_ID")
if ACTIVE_DATASET is not None and ACTIVE_DATASET not in DATASET_ALIASES:
    raise ValueError("SAFEBENEFIT_DATASET_ID must be a registered LDBC dataset")
ACTIVE_DATASETS = (ACTIVE_DATASET,) if ACTIVE_DATASET else DATASET_IDS
CHECKPOINTS = {"low": 0.25, "medium": 0.50, "high": 1.0}
REPLICAS = 2
CALIBRATION_PER_DATASET = len(QUERY_CLASSES) * len(CHECKPOINTS) * 4 * REPLICAS
EVALUATION_PER_DATASET = len(QUERY_CLASSES) * len(CHECKPOINTS) * 8 * REPLICAS
CLUSTERS_PER_DATASET = EVALUATION_PER_DATASET // REPLICAS
