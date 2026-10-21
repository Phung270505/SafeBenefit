"""Write the fixed reference model; no host timing or evaluation outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common.config import load_study_config
from src.common.files import write_json
from src.simulation.reference import OPERATION_DEFINITIONS, OPERATION_TYPES


def freeze_reference_model(
    materialized_root: Path, output: Path, *, study_id: str = "SAFEBENEFIT-DGQ"
) -> dict[str, Any]:
    """Freeze configured capacities without a benchmark or evaluation outcome."""
    config = load_study_config()
    model = config["reference_model"]
    if set(model["operation_weights"]) != set(OPERATION_TYPES):
        raise ValueError("incomplete reference operation registry")
    result = {
        "schema_version": 2,
        "study_id": study_id,
        "status": "PASS",
        "source_split": "fixed_reference_not_host_calibration",
        "model_id": model["model_id"],
        "operation_definitions": OPERATION_DEFINITIONS,
        "resource_limits": config["resource_limits"],
        "operation_weights": model["operation_weights"],
        "scan_cpu_units_per_second": model["cpu_work_units_per_second"],
        "codec_cpu_units_per_second": model["cpu_work_units_per_second"],
        "io_bytes_per_second": model["io_bytes_per_second"],
        "evaluation_outcomes_used": False,
        "host_clock_used": False,
        "scope": "abstract logical work; not CPU instructions or measured hardware",
    }
    write_json(output, result, allow_nan=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("materialized_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    freeze_reference_model(args.materialized_root, args.output)
