"""Export exact safe-point streams consumed by the INET simulator."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import struct
import time
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

from src.common.files import sha256_file, stable_json, write_json
from src.common.paths import DATA_ROOT, OUTPUT_ROOT, PROJECT_ROOT, WORKLOAD_ROOT
from src.datasets.replay import (
    SqliteVersionedGraph,
    iter_update_events,
    write_materialization,
)
from src.datasets.workload import DATASET_SCALES, EVALUATION_ROOT

RESULT_ROOT = OUTPUT_ROOT
TOKEN = struct.Struct("<Q")
LOGGER = logging.getLogger("safebenefit.materialization")
GRAPH_CHECKSUM_ALGORITHM = "sqlite_file_sha256_wal_checkpoint_v2"
WINDOWS_REPLAY_MOVE_ATTEMPTS = 8
_VALIDATED_TOKENS: dict[str, tuple[Any, ...]] = {}


def _progress(message: str) -> None:
    """Expose long-running dataset work in both the console and run log."""
    line = f"[materialize] {message}"
    print(line, flush=True)
    LOGGER.info(line)


def _read_queries(split: str, dataset: str) -> list[dict[str, Any]]:
    root = WORKLOAD_ROOT / "calibration" if split == "calibration" else EVALUATION_ROOT
    path = root / f"{dataset}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing {split} workload: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_existing_export(
    output_root: Path, query_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Verify a completed immutable export before a resumed one-button run reuses it."""
    manifest_path = output_root / "manifest.json"
    contract_file = output_root / "queries.jsonl"
    query_tsv = output_root / "queries.tsv"
    if not all(path.exists() for path in (manifest_path, contract_file, query_tsv)):
        raise FileExistsError("mutable replay database exists without a complete export")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("query_count") != len(query_rows):
        raise ValueError("existing materialized query count changed")
    if manifest.get("queries_jsonl_sha256") != sha256_file(contract_file):
        raise ValueError("existing materialized contract checksum mismatch")
    if manifest.get("queries_tsv_sha256") != sha256_file(query_tsv):
        raise ValueError("existing materialized query TSV checksum mismatch")
    requested = {row["query_id"]: row for row in query_rows}
    contracts = [
        json.loads(line) for line in contract_file.read_text(encoding="utf-8").splitlines() if line
    ]
    if set(requested) != {row["query_id"] for row in contracts}:
        raise ValueError("existing materialized query identities changed")
    for contract in contracts:
        _validate_query_contract(output_root, requested[contract["query_id"]], contract)
    return manifest


def _validate_query_contract(
    output_root: Path, source: dict[str, Any], contract: dict[str, Any]
) -> None:
    """Validate one crash-resumable query artifact against its frozen request."""
    if contract.get("query_id") != source["query_id"]:
        raise ValueError("existing materialized query identity changed")
    for field in ("query_class", "graph_version", "parameters", "valid_plans"):
        if contract.get(field) != source[field]:
            raise ValueError(f"existing materialized {field} changed")
    paths = [
        (output_root / relative).resolve() for relative in contract.get("worker_token_files", [])
    ]
    if any(not path.is_relative_to(output_root.resolve()) or not path.is_file() for path in paths):
        raise ValueError("materialized token path escapes artifact root")
    oracle = None
    checked_paths = list(paths)
    if contract.get("artifact_schema_version") == 2:
        oracle = (output_root / contract["oracle_rows_file"]).resolve()
        if not oracle.is_relative_to(output_root.resolve()) or not oracle.is_file():
            raise ValueError("invalid oracle artifact path")
        checked_paths.append(oracle)
    signature = (
        contract.get("expected_result_sha256"),
        contract.get("oracle_rows_sha256"),
        stable_json(contract.get("token_sha256", {})),
        *((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in checked_paths),
    )
    cache_key = str(output_root / source["query_id"])
    if _VALIDATED_TOKENS.get(cache_key) == signature:
        return
    canonical: Counter[int] = Counter()
    for relative in contract.get("worker_token_files", []):
        token_path = (output_root / relative).resolve()
        if not token_path.is_relative_to(output_root.resolve()) or not token_path.is_file():
            raise ValueError("materialized token path is missing or leaves its output root")
        digest_bytes = hashlib.sha256()
        with token_path.open("rb") as handle:
            while payload := handle.read(8 * 1024 * 1024):
                digest_bytes.update(payload)
                if len(payload) % TOKEN.size:
                    raise ValueError("truncated materialized token file")
                for (token,) in struct.iter_unpack(TOKEN.format, payload):
                    if contract["reducer"] == "COUNT":
                        canonical[token] += 1
                    else:
                        canonical[token] = 1
        if contract.get("artifact_schema_version") == 2:
            if contract.get("token_sha256", {}).get(relative) != digest_bytes.hexdigest().upper():
                raise ValueError("materialized token byte/order checksum mismatch")
    if contract.get("artifact_schema_version") == 2:
        oracle_path = (output_root / contract["oracle_rows_file"]).resolve()
        if (
            not oracle_path.is_relative_to(output_root.resolve())
            or sha256_file(oracle_path) != contract["oracle_rows_sha256"]
        ):
            raise ValueError("materialized oracle artifact checksum mismatch")
    result = sorted(canonical.items())
    digest = hashlib.sha256(stable_json(result).encode("utf-8")).hexdigest()
    if digest != contract.get("expected_result_sha256"):
        raise ValueError(f"existing materialized oracle mismatch: {contract['query_id']}")
    _VALIDATED_TOKENS[cache_key] = signature


def _load_resumable_export(
    database: Path,
    output_root: Path,
    query_rows: list[dict[str, Any]],
) -> tuple[int, dict[str, dict[str, Any]], dict[str, str]] | None:
    """Return a verified crash checkpoint, or ``None`` when rebuilding is safer.

    SQLite commits only at registered graph checkpoints. Therefore a database
    with the registered schema can be reused when every query below its stored
    version already has a complete, independently verified contract. Missing
    queries at the current version are regenerated deterministically. Full
    integrity is established by the sequential database checksum and query
    oracle, avoiding an additional multi-hour SQLite ``quick_check`` on SF10.
    """
    try:
        connection = sqlite3.connect(database)
        try:
            required_tables = {"vertices", "edges"}
            observed_tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if not required_tables.issubset(observed_tables):
                return None
            current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            has_vertex = connection.execute("SELECT 1 FROM vertices LIMIT 1").fetchone()
        finally:
            connection.close()
        if not has_vertex:
            return None

        contracts: dict[str, dict[str, Any]] = {}
        checksums: dict[str, str] = {}
        for source in query_rows:
            query_root = output_root / source["query_id"]
            contract_path = query_root / "query.json"
            target = int(source["graph_version"])
            if not contract_path.exists():
                if target < current_version:
                    return None
                continue
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            _validate_query_contract(output_root, source, contract)
            if target > current_version:
                return None
            version_key = str(target)
            state_digest = contract.get("graph_state_sha256")
            if not isinstance(state_digest, str) or len(state_digest) != 64:
                return None
            if version_key in checksums and checksums[version_key] != state_digest:
                return None
            checksums[version_key] = state_digest
            contracts[source["query_id"]] = contract
        return current_version, contracts, checksums
    except (json.JSONDecodeError, OSError, sqlite3.DatabaseError, ValueError, KeyError):
        return None


def _remove_incomplete_export(
    database: Path, output_root: Path, *, result_root: Path = RESULT_ROOT
) -> None:
    """Remove only an unfinished project-local replay/export."""
    allowed = result_root.resolve()
    for target in (database, Path(f"{database}-wal"), Path(f"{database}-shm"), output_root):
        resolved = target.resolve()
        if not resolved.is_relative_to(allowed):
            raise ValueError(f"refusing cleanup outside output/current: {resolved}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _move_replay_database(source: Path, target: Path) -> None:
    """Move a closed replay cache, tolerating short-lived Windows file locks.

    Windows refuses ``os.replace`` while any SQLite connection still owns the
    file. The caller must close its probe connection first. Antivirus/indexing
    can retain a transient handle for a few milliseconds, so retry only the
    two lock-related WinError values; every other filesystem failure remains
    immediate and visible.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(WINDOWS_REPLAY_MOVE_ATTEMPTS):
        try:
            source.replace(target)
            return
        except PermissionError as error:
            if (
                getattr(error, "winerror", None) not in {5, 32}
                or attempt + 1 == WINDOWS_REPLAY_MOVE_ATTEMPTS
            ):
                raise
            time.sleep(min(0.5, 0.05 * (2**attempt)))
    raise AssertionError("unreachable replay move retry state")


def export_dataset(
    split: str,
    dataset: str,
    *,
    partitions: int = 8,
    dataset_scales: tuple[str, ...] = DATASET_SCALES,
    result_root: Path = RESULT_ROOT,
) -> dict[str, Any]:
    """Replay one dataset monotonically and export every query at its exact graph version."""
    if split not in {"calibration", "evaluation"}:
        raise ValueError("split must be calibration or evaluation")
    if dataset not in dataset_scales:
        raise ValueError(f"unknown dataset scale: {dataset}")
    query_rows = _read_queries(split, dataset)
    query_ids = [row["query_id"] for row in query_rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("duplicate query_id")

    dataset_root = DATA_ROOT / "processed" / dataset
    output_root = result_root / "cache" / "materialized" / split / dataset
    database = result_root / "cache" / "replay" / split / f"{dataset}.sqlite"
    resume: tuple[int, dict[str, dict[str, Any]], dict[str, str]] | None = None
    if database.exists():
        complete = all(
            (output_root / name).exists()
            for name in ("manifest.json", "queries.jsonl", "queries.tsv")
        )
        if complete:
            return _validate_existing_export(output_root, query_rows)
        resume = _load_resumable_export(database, output_root, query_rows)
        if resume is None:
            _remove_incomplete_export(database, output_root, result_root=result_root)
    seeded_from_calibration = False
    if split == "evaluation":
        calibration_database = (
            result_root / "cache" / "replay" / "calibration" / f"{dataset}.sqlite"
        )
        if not database.exists() and calibration_database.exists():
            dataset_manifest = json.loads(
                (dataset_root / "metadata" / "manifest.json").read_text(encoding="utf-8")
            )
            expected_version = int(dataset_manifest["updates"]["calibration"])
            # A sqlite3 connection's context manager commits/rolls back but
            # does not close the connection. ``closing`` is required before a
            # subsequent rename on Windows.
            with closing(sqlite3.connect(calibration_database)) as source:
                observed_version = int(source.execute("PRAGMA user_version").fetchone()[0])
            if observed_version != expected_version:
                raise ValueError("calibration replay database is not at its frozen final version")
            database.parent.mkdir(parents=True, exist_ok=True)
            # Calibration execution and model fitting are complete before the
            # evaluation export starts. Move the exact replay state instead
            # of duplicating a potentially tens-of-gigabytes SF10 database.
            # Materialized calibration query streams remain immutable and the
            # replay database is a regenerable cache, not a reported result.
            _move_replay_database(calibration_database, database)
            seeded_from_calibration = True
    _progress(f"{split}/{dataset}: open replay database")
    initialization_state: dict[str, int] = {}

    def initialization_progress(phase: str, completed: int, total: int) -> None:
        percent = 100 if total == 0 else int(100 * completed / total)
        bucket = min(100, (percent // 5) * 5)
        if initialization_state.get(phase) == bucket:
            return
        initialization_state[phase] = bucket
        _progress(f"{split}/{dataset}: snapshot {phase} {bucket}% ({completed:,}/{total:,})")

    graph = SqliteVersionedGraph(
        dataset_root,
        database,
        initialization_progress=initialization_progress,
    )
    if resume is not None and graph.version != resume[0]:
        graph.close()
        raise ValueError("replay database version changed after resume validation")
    if split == "evaluation" and not seeded_from_calibration:
        dataset_manifest = json.loads(
            (dataset_root / "metadata" / "manifest.json").read_text(encoding="utf-8")
        )
        calibration_version = int(dataset_manifest["updates"]["calibration"])
        seeded_from_calibration = graph.version >= calibration_version
    events = iter_update_events(
        dataset_root,
        include_evaluation=split == "evaluation",
        start_at_evaluation=seeded_from_calibration,
    )
    # Skip events already committed before an interrupted run.  The predicate
    # deliberately reads the live graph version: after every applied event the
    # next sequential event remains the first one allowed through.
    events = (event for event in events if int(event["seq"]) >= graph.version)
    if graph.version:
        _progress(f"{split}/{dataset}: resume from graph_version={graph.version:,}")
    resumed_contracts = {} if resume is None else resume[1]
    contracts: list[dict[str, Any]] = list(resumed_contracts.values())
    version_checksums: dict[str, str] = {} if resume is None else dict(resume[2])
    try:
        pending = [
            query
            for query in sorted(
                query_rows, key=lambda row: (int(row["graph_version"]), row["query_id"])
            )
            if query["query_id"] not in resumed_contracts
        ]
        completed_queries = len(resumed_contracts)
        for query in pending:
            target = int(query["graph_version"])
            if target < graph.version:
                raise ValueError(
                    "cannot resume because a query below the committed graph version is missing"
                )
            if target > graph.version:
                start_version = graph.version
                _progress(f"{split}/{dataset}: replay {start_version:,} -> {target:,}")

                last_report = -1

                def replay_progress(current: int, final: int) -> None:
                    nonlocal last_report
                    percent = (
                        100
                        if final == start_version
                        else int(100 * (current - start_version) / (final - start_version))
                    )
                    bucket = min(100, (percent // 10) * 10)
                    if bucket != last_report:
                        last_report = bucket
                        _progress(
                            f"{split}/{dataset}: replay {bucket}% (graph_version={current:,})"
                        )

                graph.advance(events, target, progress=replay_progress)
            version_key = str(target)
            if version_key not in version_checksums:
                _progress(
                    f"{split}/{dataset}: checksum graph_version={target:,} "
                    f"using sequential block I/O"
                )
                last_checksum_report = -1

                def checksum_progress(completed: int, total: int) -> None:
                    nonlocal last_checksum_report
                    percent = 100 if total == 0 else int(100 * completed / total)
                    bucket = min(100, (percent // 10) * 10)
                    if bucket != last_checksum_report:
                        last_checksum_report = bucket
                        _progress(
                            f"{split}/{dataset}: checksum {bucket}% "
                            f"({completed / (1024**3):.1f}/{total / (1024**3):.1f} GiB)"
                        )

                version_checksums[version_key] = graph.state_checksum(progress=checksum_progress)
            query_root = output_root / query["query_id"]
            if query_root.exists():
                resolved = query_root.resolve()
                if not resolved.is_relative_to(output_root.resolve()):
                    raise ValueError("refusing to replace partial query outside output root")
                shutil.rmtree(query_root)
            materialized = graph.materialize(query, partitions=partitions)
            contract = write_materialization(
                output_root,
                {
                    **query,
                    **materialized,
                    "graph_state_sha256": version_checksums[version_key],
                    "partition_count": partitions,
                },
            )
            contracts.append(contract)
            completed_queries += 1
            if completed_queries % 8 == 0 or completed_queries == len(query_rows):
                _progress(
                    f"{split}/{dataset}: materialized {completed_queries}/{len(query_rows)} queries"
                )
    finally:
        graph.close()

    ordered = sorted(contracts, key=lambda row: query_ids.index(row["query_id"]))
    contract_file = output_root / "queries.jsonl"
    contract_file.parent.mkdir(parents=True, exist_ok=True)
    contract_file.write_text("".join(stable_json(row) + "\n" for row in ordered), encoding="utf-8")
    # Numeric compatibility fields are inert for materialized queries, but remain
    # explicit to keep the C++ parser small and deterministic.
    query_tsv = output_root / "queries.tsv"
    query_tsv.write_text(
        "".join(
            f"{row['query_id']}\t{row['query_class']}\t0\t0\t0\t"
            f"{row['graph_version']}\t1\t1\t{row['graph_vertex_count']}\t"
            f"{row['graph_edge_count']}\t{row['scanned_rows']}\t"
            f"{','.join(map(str, row['scan_work_by_partition']))}\t"
            f"{','.join(map(str, row['scanned_rows_by_partition']))}\n"
            for row in ordered
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "study_id": "SAFEBENEFIT-DGQ",
        "split": split,
        "dataset_id": dataset,
        "query_count": len(ordered),
        "partition_count": partitions,
        "seeded_from_calibration_replay": seeded_from_calibration,
        "graph_checksum_algorithm": GRAPH_CHECKSUM_ALGORITHM,
        "graph_version_sha256": version_checksums,
        "queries_jsonl_sha256": sha256_file(contract_file),
        "queries_tsv_sha256": sha256_file(query_tsv),
        "implementation_sha256": {
            "src/datasets/export.py": sha256_file(PROJECT_ROOT / "src/datasets/export.py"),
            "src/datasets/replay.py": sha256_file(PROJECT_ROOT / "src/datasets/replay.py"),
        },
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def export_split(
    split: str,
    *,
    partitions: int = 8,
    dataset_scales: tuple[str, ...] = DATASET_SCALES,
    result_root: Path = RESULT_ROOT,
) -> dict[str, Any]:
    """Export all registered dataset scales for one split."""
    manifests: list[dict[str, Any]] = []
    for index, dataset in enumerate(dataset_scales, start=1):
        _progress(f"{split}: dataset {index}/{len(dataset_scales)} {dataset}")
        manifest = export_dataset(
            split,
            dataset,
            partitions=partitions,
            dataset_scales=dataset_scales,
            result_root=result_root,
        )
        manifests.append(manifest)
        _progress(f"{split}/{dataset}: complete ({manifest['query_count']} queries)")
    summary = {
        "schema_version": 1,
        "study_id": "SAFEBENEFIT-DGQ",
        "split": split,
        "status": "PASS",
        "query_count": sum(item["query_count"] for item in manifests),
        "datasets": manifests,
    }
    write_json(result_root / "cache" / "materialized" / split / "manifest.json", summary)
    return summary
