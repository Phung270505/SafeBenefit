"""Deterministic parameter generation shared by calibration and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import orjson

from src.common.design import ACTIVE_DATASETS, CHECKPOINTS
from src.common.files import stable_json, write_jsonl
from src.common.paths import DATA_ROOT, OUTPUT_ROOT, PROJECT_ROOT
from src.common.seed import MASTER_SEED, STREAM_IDS, component_seed, rng_for
from src.datasets.graph import Graph
from src.queries.registry import PLAN_NAMES, QUERY_CLASSES

DATASET_SCALES = ACTIVE_DATASETS
SCHEDULE_PROFILES = (
    "original_schedule_00",
    "original_schedule_01",
    "original_schedule_02",
    "original_schedule_03",
    "original_schedule_04",
    "original_schedule_05",
    "original_schedule_06",
    "original_schedule_07",
)
def dataset_readiness(
    dataset_scales: tuple[str, ...] = DATASET_SCALES,
) -> dict[str, Any]:
    """Report missing or malformed full dataset-scale artifacts without downloading."""
    if not dataset_scales or len(set(dataset_scales)) != len(dataset_scales):
        raise ValueError("dataset readiness requires unique dataset IDs")
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    for dataset in dataset_scales:
        root = DATA_ROOT / "processed" / dataset
        manifest_path = root / "metadata" / "manifest.json"
        required = (
            root / "snapshot" / "vertices.csv",
            root / "snapshot" / "edges.csv",
            root / "updates" / "calibration" / "events.jsonl",
            root / "updates" / "evaluation" / "events.jsonl",
            manifest_path,
        )
        missing = [path.relative_to(PROJECT_ROOT).as_posix() for path in required if not path.exists()]
        if missing:
            errors.append(f"{dataset}: missing {', '.join(missing)}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != dataset:
            errors.append(f"{dataset}: manifest dataset_id mismatch")
            continue
        summaries.append(
            {
                "dataset_scale": dataset,
                "snapshot_vertices": int(manifest["snapshot"]["vertices"]),
                "snapshot_edges": int(manifest["snapshot"]["edges"]),
                "calibration_updates": int(manifest["updates"]["calibration"]),
                "evaluation_updates": int(manifest["updates"]["evaluation"]),
            }
        )
    return {"status": "PASS" if not errors else "BLOCKED", "datasets": summaries, "errors": errors}


def _ranked_vertices(
    graph: Graph, *, component: str, dataset: str, reservoir_size: int = 128
) -> dict[str, list[str]]:
    count = graph.connected_vertex_count()
    if count < 96:
        raise ValueError("dataset has too few connected vertices for disjoint parameters")
    third = count // 3
    reservoirs = {bucket: [] for bucket in CHECKPOINTS}
    seen = {bucket: 0 for bucket in CHECKPOINTS}
    random = {
        bucket: rng_for(component, f"{dataset}:degree-reservoir:{bucket}")
        for bucket in CHECKPOINTS
    }
    for index, (_, vertex) in enumerate(graph.degree_rows()):
        bucket = "low" if index < third else "medium" if index < 2 * third else "high"
        seen[bucket] += 1
        values = reservoirs[bucket]
        if len(values) < reservoir_size:
            values.append(vertex)
        else:
            replacement = random[bucket].randrange(seen[bucket])
            if replacement < reservoir_size:
                values[replacement] = vertex
    if any(len(values) < 32 for values in reservoirs.values()):
        raise ValueError("degree bucket has too few sampled vertices")
    return reservoirs


def _sampled_ranked_vertices(
    root: Path, *, component: str, dataset: str, sample_size: int = 768,
) -> dict[str, list[str]]:
    """Reuse a bounded-memory exact degree artifact for both named streams."""
    from src.datasets.snapshot_statistics import sample_degree_buckets
    return sample_degree_buckets(root, dataset, component, sample_size)


def _query_parameters(
    query_class: str,
    bucket: str,
    sample_index: int,
    vertex_pools: dict[tuple[str, str], list[str]],
    temporal_bounds: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    if query_class == "TEMPORAL_TYPE_AGG":
        start, end = temporal_bounds[bucket]
        span = max(1, end - start)
        # Anchor close to the update checkpoint so the temporal query observes
        # the dynamic part of the graph rather than only the old snapshot.
        center = 0.82
        # Each named stream produces a unique, outcome-blind time window.
        jitter = (sample_index - 15.5) * 0.002
        # Cover genuinely sparse through broad temporal selectivity. The two
        # narrow windows are needed to test the fixed overhead of local reduce.
        width_fraction = (0.000001, 0.0001, 0.01, 0.04)[sample_index % 4]
        anchor = start + int(span * min(0.95, max(0.05, center + jitter)))
        width = max(1, int(span * width_fraction))
        return {
            "start_ms": max(start, anchor - width // 2),
            "end_ms": min(end, anchor + width // 2),
        }

    vertices = vertex_pools[(query_class, bucket)]
    if sample_index >= len(vertices):
        raise ValueError(f"insufficient disjoint vertices for {query_class}:{bucket}")
    seed = vertices[sample_index]
    if query_class == "NEIGHBOR_TYPE_AGG":
        return {"seed": seed}
    if query_class == "TWO_HOP_REACHABILITY":
        limits = (128, 500, 5_000, 50_000, 500_000)
        return {"seed": seed, "hops": 2, "limit": limits[sample_index % len(limits)]}
    if query_class == "TWO_HOP_JOIN":
        limits = (128, 500, 5_000, 50_000, 500_000)
        return {"seed": seed, "limit": limits[sample_index % len(limits)]}
    raise ValueError(f"unknown query class: {query_class}")


def _split_temporal_bounds(root: Path, split: str, update_count: int) -> dict[str, tuple[int, int]]:
    """Read only event timestamps needed to place windows before each graph checkpoint."""
    event_path = root / "updates" / split / "events.jsonl"
    targets = {
        bucket: max(1, int(update_count * fraction))
        for bucket, fraction in CHECKPOINTS.items()
    }
    first_time: int | None = None
    end_times: dict[str, int] = {}
    target_to_buckets: dict[int, list[str]] = {}
    for bucket, target in targets.items():
        target_to_buckets.setdefault(target, []).append(bucket)
    for local_index, line in enumerate(event_path.open("rb"), start=1):
        event_time = int(orjson.loads(line)["event_time_ms"])
        if first_time is None:
            first_time = event_time
        for bucket in target_to_buckets.get(local_index, []):
            end_times[bucket] = event_time
        if len(end_times) == len(CHECKPOINTS):
            break
    if first_time is None or set(end_times) != set(CHECKPOINTS):
        raise ValueError(f"cannot derive temporal checkpoint bounds from {event_path}")
    return {bucket: (first_time, end_times[bucket]) for bucket in CHECKPOINTS}


def _rows_for_dataset(
    dataset: str,
    *,
    split: str,
    schedules: int,
    component: str,
    sample_offset: int,
    excluded_parameters: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    root = DATA_ROOT / "processed" / dataset
    manifest = json.loads((root / "metadata" / "manifest.json").read_text(encoding="utf-8"))
    graph: Graph | None = None
    rows: list[dict[str, Any]] = []
    # A different RNG stream does not mathematically guarantee disjoint
    # parameters. Keep an explicit outcome-blind exclusion set and advance to
    # the next deterministic candidate whenever a collision occurs.
    used_parameters = set(excluded_parameters or ())
    try:
        if (int(manifest["snapshot"].get("vertices", 0)) >= 768
                or int(manifest["snapshot"]["edges"]) > Graph.MEMORY_EDGE_LIMIT):
            ranked = _sampled_ranked_vertices(
                root,
                component=component,
                dataset=dataset,
            )
        else:
            graph = Graph(
                root,
                cache_path=(
                    OUTPUT_ROOT
                    / "cache"
                    / "workload_graph"
                    / f"{dataset}.sqlite"
                ),
            )
            ranked = _ranked_vertices(graph, component=component, dataset=dataset)
        vertex_pools: dict[tuple[str, str], list[str]] = {}
        for query_class in QUERY_CLASSES:
            if query_class == "TEMPORAL_TYPE_AGG":
                continue
            for bucket, vertices in ranked.items():
                pool = list(vertices)
                rng_for(component, f"{dataset}:{query_class}:{bucket}").shuffle(pool)
                vertex_pools[(query_class, bucket)] = pool
        local_updates = int(manifest["updates"][split])
        temporal_bounds = _split_temporal_bounds(root, split, local_updates)
        version_offset = 0 if split == "calibration" else int(manifest["updates"]["calibration"])
        for query_class in QUERY_CLASSES:
            for bucket, fraction in CHECKPOINTS.items():
                for schedule_id in range(schedules):
                    for replica_id in range(2):
                        local_index = schedule_id * 2 + replica_id
                        candidate_index = sample_offset + local_index
                        while True:
                            parameters = _query_parameters(
                                query_class,
                                bucket,
                                candidate_index,
                                vertex_pools,
                                temporal_bounds,
                            )
                            parameter_key = (query_class, stable_json(parameters))
                            if parameter_key not in used_parameters:
                                break
                            candidate_index += 1
                            if candidate_index - (sample_offset + local_index) > 128:
                                raise ValueError(
                                    f"cannot find a disjoint parameter for "
                                    f"{dataset}:{query_class}:{bucket}"
                                )
                        used_parameters.add(parameter_key)
                        rows.append(
                            {
                                "query_id": (
                                    f"{dataset}-{split}-{query_class}-{bucket}-"
                                    f"s{schedule_id}-r{replica_id}"
                                ),
                                "dataset_id": dataset,
                                "scale_id": dataset,
                                "query_class": query_class,
                                "schedule_id": schedule_id,
                                "schedule_profile": SCHEDULE_PROFILES[schedule_id],
                                "replica_id": replica_id,
                                "graph_version_bucket": bucket,
                                "graph_version": version_offset + max(1, int(local_updates * fraction)),
                                "parameters": parameters,
                                "valid_plans": list(PLAN_NAMES),
                                "master_seed": MASTER_SEED,
                                "rng_stream_id": STREAM_IDS[component],
                                "sampling_seed": component_seed(
                                    component, f"{dataset}:{query_class}:{bucket}"
                                ),
                            }
                        )
    finally:
        if graph is not None:
            graph.close()
    return rows


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)


