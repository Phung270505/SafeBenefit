"""Adapter for a chronologically ordered citation edge list.

The input is a CSV file with ``src,dst,event_time_ms`` columns. This adapter is
appropriate for citation datasets only when a real publication/application
timestamp is available; row order must never be used as a fake timestamp.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

from src.common.files import (
    EDGE_FIELDS,
    VERTEX_FIELDS,
    checksum_manifest,
    dict_writer,
    ensure_dirs,
    sha256_file,
    stable_id,
    stable_json,
    typed_id,
    write_json,
)


def _vertex(identifier: str, timestamp: int) -> dict[str, Any]:
    return {
        "vertex_id": typed_id("publication", identifier),
        "vertex_type": "Publication",
        "properties_json": "{}",
        "valid_from_ms": timestamp,
        "valid_to_ms": "",
    }


def _edge(source: str, target: str, timestamp: int) -> dict[str, Any]:
    src = typed_id("publication", source)
    dst = typed_id("publication", target)
    return {
        "edge_id": stable_id("citation", src, dst, timestamp),
        "src": src,
        "dst": dst,
        "edge_type": "CITES",
        "weight": 1,
        "properties_json": "{}",
        "event_time_ms": timestamp,
        "valid_from_ms": timestamp,
        "valid_to_ms": "",
    }


def _count_and_validate(path: Path) -> int:
    count = 0
    previous: int | None = None
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["src", "dst", "event_time_ms"]:
            raise ValueError("citation input must have src,dst,event_time_ms columns")
        for count, row in enumerate(reader, start=1):
            timestamp = int(row["event_time_ms"])
            if not row["src"] or not row["dst"] or timestamp < 0:
                raise ValueError(f"invalid citation row {count}")
            if previous is not None and timestamp < previous:
                raise ValueError(f"citation timestamps move backwards at row {count}")
            previous = timestamp
    if count < 4:
        raise ValueError("citation dataset needs at least four chronological edges")
    return count


def adapt_temporal_citations(
    edges_csv: Path,
    output: Path,
    *,
    dataset_id: str,
    source_archive: Path | Sequence[Path],
    source_url: str,
    snapshot_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
) -> dict[str, Any]:
    """Create snapshot plus disjoint chronological calibration/evaluation streams."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite processed dataset: {output}")
    source_archives = (
        (source_archive,)
        if isinstance(source_archive, Path)
        else tuple(source_archive)
    )
    if not source_archives or any(not path.is_file() for path in source_archives):
        raise FileNotFoundError("one or more citation source files are missing")
    if not 0 < snapshot_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("invalid temporal split fractions")
    if snapshot_fraction + calibration_fraction >= 1:
        raise ValueError("evaluation split must be non-empty")
    total_edges = _count_and_validate(edges_csv)
    snapshot_limit = int(total_edges * snapshot_fraction)
    calibration_limit = snapshot_limit + int(total_edges * calibration_fraction)
    if snapshot_limit == 0 or calibration_limit == snapshot_limit:
        raise ValueError("temporal splits are empty")

    paths = ensure_dirs(output)
    vertices_seen: set[str] = set()
    snapshot_vertices: dict[str, int] = {}
    snapshot_edges = 0
    calibration_events = 0
    evaluation_events = 0
    sequence = 0
    vertex_handle, vertex_writer = dict_writer(paths["snapshot"] / "vertices.csv", VERTEX_FIELDS)
    edge_handle, edge_writer = dict_writer(paths["snapshot"] / "edges.csv", EDGE_FIELDS)
    calibration_path = paths["calibration"] / "events.jsonl"
    evaluation_path = paths["evaluation"] / "events.jsonl"
    try:
        with edges_csv.open(encoding="utf-8", newline="") as source_handle:
            reader = csv.DictReader(source_handle)
            with calibration_path.open("w", encoding="utf-8", newline="\n") as cal_handle:
                with evaluation_path.open("w", encoding="utf-8", newline="\n") as eval_handle:
                    for index, row in enumerate(reader):
                        timestamp = int(row["event_time_ms"])
                        source, target = row["src"], row["dst"]
                        if index < snapshot_limit:
                            for identifier in (source, target):
                                snapshot_vertices.setdefault(identifier, timestamp)
                                vertices_seen.add(identifier)
                            edge_writer.writerow(_edge(source, target, timestamp))
                            snapshot_edges += 1
                            continue
                        changes: list[dict[str, Any]] = []
                        for identifier in (source, target):
                            if identifier not in vertices_seen:
                                changes.append(
                                    {"action": "ADD_VERTEX", "record": _vertex(identifier, timestamp)}
                                )
                                vertices_seen.add(identifier)
                        changes.append(
                            {"action": "ADD_EDGE", "record": _edge(source, target, timestamp)}
                        )
                        event = {
                            "seq": sequence,
                            "event_time_ms": timestamp,
                            "op_code": 1,
                            "op_name": "ADD_CITATION",
                            "source_stream": "citation_edges",
                            "source_row": index + 2,
                            "changes": changes,
                        }
                        destination = cal_handle if index < calibration_limit else eval_handle
                        destination.write(stable_json(event) + "\n")
                        if index < calibration_limit:
                            calibration_events += 1
                        else:
                            evaluation_events += 1
                        sequence += 1
        for identifier, timestamp in sorted(snapshot_vertices.items()):
            vertex_writer.writerow(_vertex(identifier, timestamp))
    finally:
        vertex_handle.close()
        edge_handle.close()

    write_json(
        paths["metadata"] / "query_mapping.json",
        {
            "NEIGHBOR_TYPE_AGG": "citation degree around one publication",
            "TEMPORAL_TYPE_AGG": "citations created in a real time window",
            "TWO_HOP_REACHABILITY": "publications reachable through two citations",
            "TWO_HOP_JOIN": "join citation edges through an intermediate publication",
        },
    )
    artifacts = [
        output / "snapshot" / "vertices.csv",
        output / "snapshot" / "edges.csv",
        calibration_path,
        evaluation_path,
        output / "metadata" / "query_mapping.json",
    ]
    manifest = {
        "contract_version": "dataset-adapter-v1",
        "dataset_id": dataset_id,
        "source_format": "chronological citation edge list",
        "source_url": source_url,
        "source_files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_archives
        ],
        "snapshot": {
            "vertices": len(snapshot_vertices),
            "edges": snapshot_edges,
            "vertex_types": {"Publication": len(snapshot_vertices)},
            "edge_types": {"CITES": snapshot_edges},
        },
        "updates": {
            "total": calibration_events + evaluation_events,
            "calibration": calibration_events,
            "evaluation": evaluation_events,
            "split_rule": "chronological_edges_70_15_15",
        },
        "query_mapping": "metadata/query_mapping.json",
        "artifact_sha256": checksum_manifest(artifacts, output),
    }
    if len(source_archives) == 1:
        manifest["source_archive_bytes"] = source_archives[0].stat().st_size
        manifest["source_archive_sha256"] = sha256_file(source_archives[0])
    write_json(paths["metadata"] / "manifest.json", manifest)
    return manifest
