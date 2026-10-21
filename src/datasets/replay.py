"""Exact versioned query oracle and safe-point materialization."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import struct
import sys
import time
from array import array
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import orjson

from src.common.files import sha256_file, stable_json, write_json
from src.common.partitioning import stable_owner
from src.simulation.reference import OperationCounts

TOKEN = struct.Struct("<Q")


def token_id(*parts: Any) -> int:
    """Map a canonical tuple to a portable uint64 transfer token."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


class SqliteVersionedGraph:
    """Disk-backed exact graph state advanced monotonically by source events."""

    def __init__(
        self,
        dataset_root: Path,
        database: Path,
        *,
        initialization_progress: Callable[[str, int, int], None] | None = None,
    ):
        self.dataset_root = dataset_root
        self.database = database
        self.initialization_progress = initialization_progress
        database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database)
        self.version = 0
        self._vertex_count = 0
        self._edge_count = 0
        self._metadata_dirty = False
        self._adjacency_cache: OrderedDict[str, list[str]] = OrderedDict()
        self._adjacency_bytes = 0
        self._initialize()

    def _initialize(self) -> None:
        connection = self.connection
        connection.executescript(
            """
            PRAGMA foreign_keys=ON;
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-262144;
            PRAGMA temp_store=FILE;
            CREATE TABLE IF NOT EXISTS vertices(id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS edges(
                id TEXT PRIMARY KEY, src TEXT NOT NULL, dst TEXT NOT NULL,
                kind TEXT NOT NULL, event_time INTEGER NOT NULL,
                FOREIGN KEY(src) REFERENCES vertices(id),
                FOREIGN KEY(dst) REFERENCES vertices(id)
            );
            CREATE TABLE IF NOT EXISTS graph_meta(
                key TEXT PRIMARY KEY, value INTEGER NOT NULL
            );
            """
        )
        manifest_path = self.dataset_root / "metadata" / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        )
        expected_vertices = int(manifest.get("snapshot", {}).get("vertices", 0))
        expected_edges = int(manifest.get("snapshot", {}).get("edges", 0))
        stored = int(connection.execute("PRAGMA user_version").fetchone()[0])
        metadata = dict(connection.execute("SELECT key,value FROM graph_meta"))
        snapshot_complete = int(metadata.get("snapshot_complete", 0)) == 1
        legacy_complete = (
            "snapshot_complete" not in metadata
            and {"vertex_count", "edge_count"}.issubset(metadata)
            and (
                stored > 0
                or (
                    int(metadata["vertex_count"]) == expected_vertices
                    and int(metadata["edge_count"]) == expected_edges
                )
            )
        )
        if snapshot_complete or legacy_complete:
            self.version = stored
            self._vertex_count = int(metadata["vertex_count"])
            self._edge_count = int(metadata["edge_count"])
            if legacy_complete:
                connection.execute(
                    "INSERT OR REPLACE INTO graph_meta(key,value) VALUES ('snapshot_complete',1)"
                )
                connection.commit()
            return

        loaded_vertices = int(metadata.get("snapshot_vertices_loaded", 0))
        loaded_edges = int(metadata.get("snapshot_edges_loaded", 0))
        if "snapshot_vertices_loaded" not in metadata:
            loaded_vertices = int(connection.execute("SELECT COUNT(*) FROM vertices").fetchone()[0])
        if "snapshot_edges_loaded" not in metadata:
            loaded_edges = int(connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        self._checkpoint_snapshot(loaded_vertices, loaded_edges, complete=False)
        # Dataset integrity was already verified before materialization. Disabling
        # per-row FK probes during the immutable base import removes hundreds of
        # millions of redundant index lookups; updates re-enable FK enforcement.
        connection.execute("PRAGMA foreign_keys=OFF")
        with (self.dataset_root / "snapshot" / "vertices.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            batch_vertices: list[tuple[str]] = []
            for index, row in enumerate(csv.DictReader(handle), start=1):
                if index <= loaded_vertices:
                    continue
                batch_vertices.append((row["vertex_id"],))
                if len(batch_vertices) == 250_000:
                    connection.executemany("INSERT INTO vertices(id) VALUES (?)", batch_vertices)
                    loaded_vertices = index
                    batch_vertices.clear()
                    self._checkpoint_snapshot(loaded_vertices, loaded_edges, complete=False)
                    self._report_initialization("vertices", loaded_vertices, expected_vertices)
            if batch_vertices:
                connection.executemany("INSERT INTO vertices(id) VALUES (?)", batch_vertices)
                loaded_vertices += len(batch_vertices)
                self._checkpoint_snapshot(loaded_vertices, loaded_edges, complete=False)
                self._report_initialization("vertices", loaded_vertices, expected_vertices)
        with (self.dataset_root / "snapshot" / "edges.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            batch: list[tuple[str, str, str, str, int]] = []
            for index, row in enumerate(csv.DictReader(handle), start=1):
                if index <= loaded_edges:
                    continue
                batch.append(
                    (
                        row["edge_id"],
                        row["src"],
                        row["dst"],
                        row["edge_type"],
                        int(row["event_time_ms"] or 0),
                    )
                )
                if len(batch) == 250_000:
                    connection.executemany("INSERT INTO edges VALUES (?,?,?,?,?)", batch)
                    loaded_edges = index
                    batch.clear()
                    self._checkpoint_snapshot(loaded_vertices, loaded_edges, complete=False)
                    self._report_initialization("edges", loaded_edges, expected_edges)
            if batch:
                connection.executemany("INSERT INTO edges VALUES (?,?,?,?,?)", batch)
                loaded_edges += len(batch)
                self._checkpoint_snapshot(loaded_vertices, loaded_edges, complete=False)
                self._report_initialization("edges", loaded_edges, expected_edges)
        # Build secondary indexes once after the bulk load. Creating them before
        # 17–20 million inserts multiplies the full-study preparation time.
        if expected_vertices and loaded_vertices != expected_vertices:
            raise ValueError("snapshot vertex count differs from registered manifest")
        if expected_edges and loaded_edges != expected_edges:
            raise ValueError("snapshot edge count differs from registered manifest")
        indexes = (
            ("edges_src", "CREATE INDEX IF NOT EXISTS edges_src ON edges(src)"),
            ("edges_dst", "CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst)"),
            ("edges_time", "CREATE INDEX IF NOT EXISTS edges_time ON edges(event_time)"),
        )
        for index_number, (name, statement) in enumerate(indexes, start=1):
            self._report_initialization(f"index:{name}", index_number - 1, len(indexes))
            connection.execute(statement)
            connection.commit()
            self._report_initialization(f"index:{name}", index_number, len(indexes))
        self._vertex_count = loaded_vertices
        self._edge_count = loaded_edges
        self._metadata_dirty = True
        self._persist_metadata()
        self._checkpoint_snapshot(loaded_vertices, loaded_edges, complete=True)
        connection.execute("PRAGMA foreign_keys=ON")

    def _report_initialization(self, phase: str, completed: int, total: int) -> None:
        if self.initialization_progress is not None:
            self.initialization_progress(phase, completed, total)

    def _checkpoint_snapshot(
        self, loaded_vertices: int, loaded_edges: int, *, complete: bool
    ) -> None:
        """Make bulk snapshot loading resumable after interruption."""
        self.connection.executemany(
            "INSERT OR REPLACE INTO graph_meta(key,value) VALUES (?,?)",
            (
                ("snapshot_vertices_loaded", loaded_vertices),
                ("snapshot_edges_loaded", loaded_edges),
                ("snapshot_complete", int(complete)),
            ),
        )
        self.connection.commit()

    def _persist_metadata(self) -> None:
        """Persist O(1) graph-size metadata only when the state changed."""
        if not self._metadata_dirty:
            return
        self.connection.executemany(
            "INSERT OR REPLACE INTO graph_meta(key,value) VALUES (?,?)",
            (
                ("vertex_count", self._vertex_count),
                ("edge_count", self._edge_count),
            ),
        )
        self._metadata_dirty = False

    def _commit_version(self) -> None:
        """Commit the current replay position once per durable batch."""
        self.connection.execute(f"PRAGMA user_version={self.version}")
        self._persist_metadata()
        self.connection.commit()

    def apply_event(self, event: dict[str, Any], *, commit: bool = True) -> None:
        """Apply one update atomically; batched replay may defer the commit."""
        sequence = int(event["seq"])
        if sequence != self.version:
            raise ValueError(f"event sequence {sequence} does not follow version {self.version}")
        for change in event.get("changes", []):
            action = change["action"]
            row = change["record"]
            if action == "ADD_VERTEX":
                self.connection.execute("INSERT INTO vertices(id) VALUES (?)", (row["vertex_id"],))
                self._vertex_count += 1
                self._metadata_dirty = True
            elif action == "UPDATE_VERTEX":
                if not self.connection.execute(
                    "SELECT 1 FROM vertices WHERE id=?", (row["vertex_id"],)
                ).fetchone():
                    raise ValueError("cannot update a missing vertex")
            elif action == "ADD_EDGE":
                try:
                    self.connection.execute(
                        "INSERT INTO edges VALUES (?,?,?,?,?)",
                        (
                            row["edge_id"],
                            row["src"],
                            row["dst"],
                            row["edge_type"],
                            int(row.get("event_time_ms") or 0),
                        ),
                    )
                    self._edge_count += 1
                    self._metadata_dirty = True
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        "cannot add duplicate edge or edge with a missing endpoint"
                    ) from error
            elif action == "DELETE_EDGE":
                cursor = self.connection.execute("DELETE FROM edges WHERE id=?", (row["edge_id"],))
                if cursor.rowcount != 1:
                    raise ValueError("cannot delete a missing edge")
                self._edge_count -= 1
                self._metadata_dirty = True
            else:
                raise ValueError(f"unsupported graph action: {action}")
        self._adjacency_cache.clear()
        self._adjacency_bytes = 0
        self.version += 1
        if commit:
            self._commit_version()

    def advance(
        self,
        events: Iterable[dict[str, Any]],
        target_version: int,
        *,
        progress: Callable[[int, int], None] | None = None,
        progress_every: int = 250_000,
    ) -> None:
        """Advance to an exact graph version without skipping or replaying events."""
        if target_version < self.version or progress_every <= 0:
            raise ValueError("graph version cannot move backwards")
        stream = iter(events)
        next_progress = ((self.version // progress_every) + 1) * progress_every
        last_committed = self.version
        while self.version < target_version:
            try:
                event = next(stream)
            except StopIteration as error:
                raise ValueError(
                    f"event stream ended at {self.version}, expected {target_version}"
                ) from error
            self.apply_event(event, commit=False)
            if self.version >= next_progress or self.version == target_version:
                # This is both a progress checkpoint and a crash-safe resume
                # boundary.  Updating PRAGMA user_version for every event was
                # millions of redundant SQLite header writes on SF10.
                self._commit_version()
                last_committed = self.version
                if progress is not None:
                    progress(self.version, target_version)
                next_progress = ((self.version // progress_every) + 1) * progress_every
        if self.version != target_version:
            raise ValueError(f"event stream ended at {self.version}, expected {target_version}")
        if last_committed != self.version:
            self._commit_version()

    def state_checksum(
        self,
        *,
        progress: Callable[[int, int], None] | None = None,
        chunk_size: int = 16 * 1024 * 1024,
    ) -> str:
        """Hash the committed SQLite artifact with sequential block I/O.

        The previous implementation sorted and converted every graph row in
        Python at every checkpoint. SF10 contains more than 200 million rows,
        turning one checksum into hours of random I/O. The database is the exact
        committed graph state (including its indexes and graph version), so a
        sequential SHA-256 provides a stronger artifact-integrity check without
        changing query semantics.
        """
        if chunk_size <= 0:
            raise ValueError("checksum chunk_size must be positive")
        self._persist_metadata()
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        digest = hashlib.sha256()
        total = self.database.stat().st_size
        completed = 0
        with self.database.open("rb", buffering=chunk_size) as handle:
            while block := handle.read(chunk_size):
                digest.update(block)
                completed += len(block)
                if progress is not None:
                    progress(completed, total)
        return digest.hexdigest().upper()

    def graph_size(self) -> tuple[int, int]:
        return self._vertex_count, self._edge_count

    def _neighbors(self, vertex: str) -> list[str]:
        if vertex in self._adjacency_cache:
            self._adjacency_cache.move_to_end(vertex)
            return self._adjacency_cache[vertex]
        values = [
            row[0]
            for row in self.connection.execute(
                "SELECT neighbor FROM (SELECT dst neighbor FROM edges WHERE src=? "
                "UNION SELECT src neighbor FROM edges WHERE dst=?) ORDER BY neighbor",
                (vertex, vertex),
            )
        ]
        size = sum(len(value.encode("utf-8")) + 64 for value in values)
        if size <= 64 * 1024 * 1024:
            while self._adjacency_cache and self._adjacency_bytes + size > 64 * 1024 * 1024:
                _, evicted = self._adjacency_cache.popitem(last=False)
                self._adjacency_bytes -= sum(len(value.encode("utf-8")) + 64 for value in evicted)
            self._adjacency_cache[vertex] = values
            self._adjacency_bytes += size
        return values

    def _two_hop_reachable(
        self,
        seed: str,
        limit: int,
        visit: Callable[[str, int], None],
    ) -> tuple[int, list[str]]:
        """Bulk indexed expansion; temporary dedup tables bound Python memory."""
        connection = self.connection
        for statement in (
            "CREATE TEMP TABLE IF NOT EXISTS query_frontier(id TEXT PRIMARY KEY) WITHOUT ROWID",
            "CREATE TEMP TABLE IF NOT EXISTS query_pairs(middle TEXT, neighbor TEXT, "
            "PRIMARY KEY(middle,neighbor)) WITHOUT ROWID",
            "DELETE FROM query_frontier",
            "DELETE FROM query_pairs",
        ):
            connection.execute(statement)
        connection.execute(
            "INSERT OR IGNORE INTO query_frontier SELECT dst FROM edges WHERE src=? AND dst<>?",
            (seed, seed),
        )
        connection.execute(
            "INSERT OR IGNORE INTO query_frontier SELECT src FROM edges WHERE dst=? AND src<>?",
            (seed, seed),
        )
        first_count = int(connection.execute("SELECT COUNT(*) FROM query_frontier").fetchone()[0])
        first_count += int(
            connection.execute(
                "SELECT EXISTS(SELECT 1 FROM edges WHERE src=? AND dst=?)",
                (seed, seed),
            ).fetchone()[0]
        )
        visit(seed, first_count)
        connection.execute(
            "INSERT OR IGNORE INTO query_pairs SELECT f.id,e.dst "
            "FROM query_frontier f CROSS JOIN edges e WHERE e.src=f.id"
        )
        connection.execute(
            "INSERT OR IGNORE INTO query_pairs SELECT f.id,e.src "
            "FROM query_frontier f CROSS JOIN edges e WHERE e.dst=f.id"
        )
        total = first_count
        for middle, count in connection.execute(
            "SELECT middle,COUNT(*) FROM query_pairs GROUP BY middle"
        ):
            total += int(count)
            visit(middle, int(count))
        vertices = [
            row[0]
            for row in connection.execute(
                "SELECT id FROM query_frontier UNION SELECT neighbor FROM query_pairs "
                "WHERE neighbor<>? ORDER BY 1 LIMIT ?",
                (seed, limit),
            )
        ]
        # Do not commit here: callers may be evaluating an uncommitted replay batch.
        return total, vertices

    def materialize(self, query: dict[str, Any], partitions: int = 8) -> dict[str, Any]:
        """Produce exact local safe-point token streams and an independent oracle."""
        if int(query["graph_version"]) != self.version:
            raise ValueError("query requested the wrong graph version")
        kind = query["query_class"]
        parameters = query["parameters"]
        local: list[list[int]] = [[] for _ in range(partitions)]
        reducer = "COUNT" if kind.endswith("_AGG") else "DISTINCT"
        scanned_rows = 0
        counts = [OperationCounts() for _ in range(partitions)]
        scanned_by_partition = [0] * partitions

        def visit(source: str, amount: int = 1) -> None:
            owner = stable_owner(source, partitions)
            scanned_by_partition[owner] += amount
            counts[owner].add("edge_visit", amount)
            counts[owner].add("predicate", amount)

        wall_started = time.perf_counter_ns()
        cpu_started = time.process_time_ns()
        edge_type_tokens: dict[str, int] = {}

        def edge_type_token(edge_kind: str) -> int:
            try:
                return edge_type_tokens[edge_kind]
            except KeyError:
                token = token_id("edge_type", edge_kind)
                edge_type_tokens[edge_kind] = token
                return token

        if kind == "NEIGHBOR_TYPE_AGG":
            seed = parameters["seed"]
            for src, edge_kind in self.connection.execute(
                "SELECT src,kind FROM edges WHERE src=? OR dst=?", (seed, seed)
            ):
                scanned_rows += 1
                visit(src)
                local[stable_owner(src, partitions)].append(edge_type_token(edge_kind))
        elif kind == "TEMPORAL_TYPE_AGG":
            for src, edge_kind in self.connection.execute(
                "SELECT src,kind FROM edges WHERE event_time BETWEEN ? AND ?",
                (int(parameters["start_ms"]), int(parameters["end_ms"])),
            ):
                scanned_rows += 1
                visit(src)
                local[stable_owner(src, partitions)].append(edge_type_token(edge_kind))
        elif kind == "TWO_HOP_REACHABILITY":
            seed = parameters["seed"]
            hops = int(parameters.get("hops", 2))
            if hops < 0:
                raise ValueError("hops must be nonnegative")
            limit = int(parameters["limit"])
            if limit < 0:
                raise ValueError("query limit must be nonnegative")
            if hops == 2:
                scanned_rows, vertices = self._two_hop_reachable(seed, limit, visit)
            else:
                reached = {seed}
                frontier = {seed}
                for _ in range(hops):
                    next_frontier: set[str] = set()
                    for vertex in sorted(frontier):
                        neighbors = self._neighbors(vertex)
                        scanned_rows += len(neighbors)
                        visit(vertex, len(neighbors))
                        next_frontier.update(neighbors)
                    frontier = next_frontier - reached
                    reached.update(frontier)
                    if not frontier:
                        break
                reached.discard(seed)
                vertices = sorted(reached)[:limit]
            for vertex in vertices:
                token = token_id("vertex", vertex)
                local[stable_owner(vertex, partitions)].append(token)
        elif kind == "TWO_HOP_JOIN":
            seed = parameters["seed"]
            remaining = int(parameters["limit"])
            first_hop = self._neighbors(seed)
            scanned_rows += len(first_hop)
            visit(seed, len(first_hop))
            for middle in first_hop:
                second_hop = self._neighbors(middle)
                scanned_rows += len(second_hop)
                visit(middle, len(second_hop))
                for end in second_hop:
                    if end == seed:
                        continue
                    counts[stable_owner(middle, partitions)].add("join_match")
                    token = token_id("join", middle, end)
                    local[stable_owner(middle, partitions)].append(token)
                    remaining -= 1
                    if remaining == 0:
                        break
                if remaining == 0:
                    break
        else:
            raise ValueError(f"unknown materialized query class: {kind}")

        scan_cpu_ms = (time.process_time_ns() - cpu_started) / 1_000_000
        scan_wall_time_ms = (time.perf_counter_ns() - wall_started) / 1_000_000

        canonical: Counter[int] = Counter()
        for values in local:
            if reducer == "COUNT":
                canonical.update(values)
            else:
                canonical.update({token: 1 for token in values})
        if reducer == "DISTINCT":
            canonical = Counter({token: 1 for token in canonical})
        result = sorted(canonical.items())
        digest = hashlib.sha256(stable_json(result).encode("utf-8")).hexdigest()
        vertices, edges = self.graph_size()
        return {
            "query_id": query["query_id"],
            "reducer": reducer,
            "worker_tokens": local,
            "expected_result": result,
            "expected_result_sha256": digest,
            "graph_vertex_count": vertices,
            "graph_edge_count": edges,
            "intermediate_rows": sum(map(len, local)),
            "scanned_rows": scanned_rows,
            "scanned_rows_by_partition": scanned_by_partition,
            "scan_operation_counts_by_partition": [dict(counter.values) for counter in counts],
            "scan_work_by_partition": [sum(counter.values.values()) for counter in counts],
            "diagnostic_host_scan_cpu_ms": scan_cpu_ms,
            "diagnostic_host_scan_wall_ms": scan_wall_time_ms,
        }

    def close(self) -> None:
        self.connection.rollback()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()


def write_materialization(root: Path, materialized: dict[str, Any]) -> dict[str, Any]:
    """Write partitioned tokens and return the final query contract."""
    query_root = root / materialized["query_id"]
    files: list[str] = []
    for partition, tokens in enumerate(materialized["worker_tokens"]):
        path = query_root / f"part-{partition}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb", buffering=64 * 1024 * 1024) as handle:
            for start in range(0, len(tokens), 1024 * 1024):
                packed = array("Q", tokens[start : start + 1024 * 1024])
                if packed.itemsize != TOKEN.size:
                    raise RuntimeError("platform unsigned-long-long width is not 64 bits")
                if sys.byteorder != "little":
                    packed.byteswap()
                packed.tofile(handle)
        files.append(path.relative_to(root).as_posix())
    record = {
        key: value
        for key, value in materialized.items()
        if key not in {"worker_tokens", "expected_result"}
    }
    oracle_path = query_root / "oracle.jsonl"
    with oracle_path.open("w", encoding="utf-8") as handle:
        for pair in materialized.get("expected_result", []):
            handle.write(stable_json(pair) + "\n")
    record["artifact_schema_version"] = 2
    record["oracle_rows_file"] = oracle_path.relative_to(root).as_posix()
    record["oracle_rows_sha256"] = sha256_file(oracle_path)
    record["token_sha256"] = {relative: sha256_file(root / relative) for relative in files}
    record["oracle_result_count"] = len(materialized.get("expected_result", []))
    record["worker_token_files"] = files
    write_json(query_root / "query.json", record)
    return record


def iter_update_events(
    dataset_root: Path,
    include_evaluation: bool,
    *,
    start_at_evaluation: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield registered updates in chronological sequence without buffering the stream."""
    if start_at_evaluation and not include_evaluation:
        raise ValueError("cannot start at evaluation when evaluation is excluded")
    splits = (
        ("evaluation",)
        if start_at_evaluation
        else (("calibration", "evaluation") if include_evaluation else ("calibration",))
    )
    for split in splits:
        path = dataset_root / "updates" / split / "events.jsonl"
        with path.open("rb") as handle:
            for line in handle:
                if line.strip():
                    yield orjson.loads(line)
