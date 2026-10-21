"""Streaming integrity audit for the five registered dataset scales."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import orjson

from src.common.files import sha256_file
from src.common.paths import DATA_ROOT
from src.datasets.workload import DATASET_SCALES


def _validate_event_stream(
    path: Path,
    *,
    expected_count: int,
    sequence_offset: int,
    previous_time: int | None,
) -> tuple[int, int | None]:
    count = 0
    last_time = previous_time
    with path.open("rb") as handle:
        for count, line in enumerate(handle, start=1):
            event = orjson.loads(line)
            expected_sequence = sequence_offset + count - 1
            if int(event["seq"]) != expected_sequence:
                raise ValueError(
                    f"{path}: sequence {event['seq']} differs from {expected_sequence}"
                )
            event_time = int(event.get("event_time_ms") or 0)
            if last_time is not None and event_time < last_time:
                raise ValueError(f"{path}: event time moves backwards at sequence {event['seq']}")
            last_time = event_time
    if count != expected_count:
        raise ValueError(f"{path}: {count} events differs from manifest {expected_count}")
    return count, last_time


def validate_dataset_scale(
    dataset: str, *, allowed_scales: tuple[str, ...] = DATASET_SCALES
) -> dict[str, Any]:
    """Verify registered checksums and chronological update separation for one scale."""
    if dataset not in allowed_scales:
        raise ValueError(f"unknown registered dataset scale: {dataset}")
    root = DATA_ROOT / "processed" / dataset
    manifest_path = root / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for relative, expected in manifest.get("artifact_sha256", {}).items():
        path = root / relative
        if not path.exists():
            errors.append(f"missing artifact: {relative}")
        elif sha256_file(path).upper() != str(expected).upper():
            errors.append(f"checksum mismatch: {relative}")
    try:
        calibration_count = int(manifest["updates"]["calibration"])
        evaluation_count = int(manifest["updates"]["evaluation"])
        _, last_time = _validate_event_stream(
            root / "updates" / "calibration" / "events.jsonl",
            expected_count=calibration_count,
            sequence_offset=0,
            previous_time=None,
        )
        _validate_event_stream(
            root / "updates" / "evaluation" / "events.jsonl",
            expected_count=evaluation_count,
            sequence_offset=calibration_count,
            previous_time=last_time,
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))
    return {
        "dataset_id": dataset,
        "status": "PASS" if not errors else "FAIL",
        "snapshot_vertices": int(manifest["snapshot"]["vertices"]),
        "snapshot_edges": int(manifest["snapshot"]["edges"]),
        "errors": errors,
    }


def validate_dataset_suite(
    dataset_scales: tuple[str, ...] = DATASET_SCALES,
    *,
    study_id: str = "SAFEBENEFIT-DGQ",
) -> dict[str, Any]:
    """Run the integrity gate for every registered dataset scale."""
    results = [
        validate_dataset_scale(dataset, allowed_scales=dataset_scales)
        for dataset in dataset_scales
    ]
    return {
        "study_id": study_id,
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "datasets": results,
    }
