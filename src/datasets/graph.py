from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.partitioning import stable_owner


class Graph:
    """Read-only snapshot access, using RAM or a persistent SQLite query index."""

    MEMORY_EDGE_LIMIT = 2_000_000

    def __init__(self, root: Path, *, cache_path: Path | None = None):
        self.vertices: dict[str, dict[str, str]] = {}
        self.out: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.inc: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.edges: list[dict[str, str]] = []
        self._degrees: Counter[str] = Counter()
        self._time_min = 0
        self._time_max = 0
        self._conn: sqlite3.Connection | None = None
        self._edge_count_cache: dict[tuple[int, int, bool], Any] = {}
        metadata = json.loads((root / "metadata" / "manifest.json").read_text(encoding="utf-8"))
        self.edge_count = int(metadata["snapshot"]["edges"])
        if self.edge_count > self.MEMORY_EDGE_LIMIT:
            self._open_disk_index(root, metadata, cache_path=cache_path)
            return
        with (root / "snapshot" / "vertices.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self.vertices[row["vertex_id"]] = row
        with (root / "snapshot" / "edges.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self.edges.append(row)
                self.out[row["src"]].append(row)
                self.inc[row["dst"]].append(row)
                self._degrees[row["src"]] += 1
                self._degrees[row["dst"]] += 1
                timestamp = int(row["event_time_ms"] or 0)
                if timestamp > 0:
                    self._time_min = (
                        timestamp if self._time_min == 0 else min(self._time_min, timestamp)
                    )
                    self._time_max = max(self._time_max, timestamp)
        for adjacency in (self.out, self.inc):
            for values in adjacency.values():
                values.sort(key=lambda edge: edge["edge_id"])

    def _open_disk_index(
        self, root: Path, metadata: dict[str, Any], *, cache_path: Path | None
    ) -> None:
        db_path = cache_path or root / "metadata" / "workload_graph.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        source_signature = metadata.get("artifact_sha256", {}).get("snapshot/edges.csv", "")
        rebuild = True
        if db_path.exists():
            try:
                probe = sqlite3.connect(db_path)
                stored = probe.execute(
                    "SELECT value FROM metadata WHERE key='source_signature'"
                ).fetchone()
                rebuild = stored is None or stored[0] != source_signature
                probe.close()
            except sqlite3.Error:
                rebuild = True
        if rebuild:
            if db_path.exists():
                db_path.unlink()
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
            conn.execute(
                "CREATE TABLE edges(src TEXT, dst TEXT, edge_type TEXT, event_time_ms INTEGER, src_part4 INTEGER)"
            )
            batch = []
            with (root / "snapshot" / "edges.csv").open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    source, target = row["src"], row["dst"]
                    timestamp = int(row["event_time_ms"] or 0)
                    batch.append(
                        (source, target, row["edge_type"], timestamp, stable_owner(source, 4))
                    )
                    if timestamp > 0:
                        self._time_min = (
                            timestamp if self._time_min == 0 else min(self._time_min, timestamp)
                        )
                        self._time_max = max(self._time_max, timestamp)
                    if len(batch) >= 100_000:
                        conn.executemany("INSERT INTO edges VALUES (?,?,?,?,?)", batch)
                        batch.clear()
            if batch:
                conn.executemany("INSERT INTO edges VALUES (?,?,?,?,?)", batch)
            conn.execute("CREATE INDEX edges_src_idx ON edges(src)")
            conn.execute("CREATE INDEX edges_dst_idx ON edges(dst)")
            conn.execute("CREATE INDEX edges_time_idx ON edges(event_time_ms,edge_type)")
            # Aggregate degree on disk.  Keeping one Python Counter entry for
            # every connected SF10 vertex can consume several gigabytes and
            # makes workload generation fail even though SQLite can stream the
            # same aggregation using bounded memory.
            conn.execute(
                "CREATE TABLE degrees AS "
                "SELECT vertex_id,SUM(partial_degree) AS degree FROM ("
                "SELECT src AS vertex_id,COUNT(*) AS partial_degree FROM edges GROUP BY src "
                "UNION ALL "
                "SELECT dst AS vertex_id,COUNT(*) AS partial_degree FROM edges GROUP BY dst"
                ") GROUP BY vertex_id"
            )
            conn.execute("CREATE UNIQUE INDEX degrees_vertex_idx ON degrees(vertex_id)")
            conn.execute("CREATE INDEX degrees_rank_idx ON degrees(degree,vertex_id)")
            conn.execute("INSERT INTO metadata VALUES ('source_signature',?)", (source_signature,))
            conn.execute("INSERT INTO metadata VALUES ('time_min',?)", (str(self._time_min),))
            conn.execute("INSERT INTO metadata VALUES ('time_max',?)", (str(self._time_max),))
            conn.commit()
            conn.close()
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS degrees_rank_idx ON degrees(degree,vertex_id)"
        )
        self._time_min = int(
            self._conn.execute("SELECT value FROM metadata WHERE key='time_min'").fetchone()[0]
        )
        self._time_max = int(
            self._conn.execute("SELECT value FROM metadata WHERE key='time_max'").fetchone()[0]
        )
        # Older caches used SHA[:8] in src_part4. Join the canonical ownership
        # map instead; do not destructively rebuild a 20M-edge cache.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS canonical_owners_v2(vertex_id TEXT PRIMARY KEY, partition_id INTEGER)"
        )
        owner_count = self._conn.execute("SELECT COUNT(*) FROM canonical_owners_v2").fetchone()[0]
        degree_count = self._conn.execute("SELECT COUNT(*) FROM degrees").fetchone()[0]
        if owner_count != degree_count:
            self._conn.executemany(
                "INSERT OR REPLACE INTO canonical_owners_v2 VALUES (?,?)",
                (
                    (vertex, stable_owner(vertex, 4))
                    for (vertex,) in self._conn.execute("SELECT vertex_id FROM degrees")
                ),
            )
            self._conn.commit()

    def connected_vertices(self) -> tuple[str, ...]:
        """Return only vertices that can participate in structural queries."""
        if self._conn is None:
            return tuple(self._degrees)
        return tuple(row[0] for row in self._conn.execute("SELECT vertex_id FROM degrees"))

    def connected_vertex_count(self) -> int:
        """Return the number of vertices with nonzero degree."""
        if self._conn is None:
            return len(self._degrees)
        return int(self._conn.execute("SELECT COUNT(*) FROM degrees").fetchone()[0])

    def degree_rows(self):
        """Iterate ``(degree, vertex_id)`` in stable degree order without loading all rows."""
        if self._conn is None:
            yield from sorted((degree, vertex) for vertex, degree in self._degrees.items())
            return
        yield from self._conn.execute(
            "SELECT degree,vertex_id FROM degrees ORDER BY degree,vertex_id"
        )

    def neighbors(self, vertex: str) -> set[str]:
        if self._conn is not None:
            outgoing = {
                row[0] for row in self._conn.execute("SELECT dst FROM edges WHERE src=?", (vertex,))
            }
            incoming = {
                row[0] for row in self._conn.execute("SELECT src FROM edges WHERE dst=?", (vertex,))
            }
            return outgoing | incoming
        return {e["dst"] for e in self.out.get(vertex, [])} | {
            e["src"] for e in self.inc.get(vertex, [])
        }

    def degree(self, vertex: str) -> int:
        if self._conn is None:
            return self._degrees[vertex]
        row = self._conn.execute(
            "SELECT degree FROM degrees WHERE vertex_id=?", (vertex,)
        ).fetchone()
        return int(row[0]) if row else 0

    def out_edges(self, vertex: str) -> list[dict[str, str]]:
        if self._conn is None:
            return self.out.get(vertex, [])
        return [
            {"edge_type": row[0]}
            for row in self._conn.execute("SELECT edge_type FROM edges WHERE src=?", (vertex,))
        ]

    def inc_edges(self, vertex: str) -> list[dict[str, str]]:
        if self._conn is None:
            return self.inc.get(vertex, [])
        return [
            {"edge_type": row[0]}
            for row in self._conn.execute("SELECT edge_type FROM edges WHERE dst=?", (vertex,))
        ]

    def time_range(self) -> tuple[int, int]:
        return self._time_min, self._time_max

    def edge_type_counts_between(self, start: int, end: int, partitioned: bool = False):
        cache_key = (start, end, partitioned)
        if cache_key in self._edge_count_cache:
            cached = self._edge_count_cache[cache_key]
            if partitioned:
                return [Counter(partial) for partial in cached]
            return Counter(cached)
        if self._conn is None:
            selected = [
                edge for edge in self.edges if start <= int(edge["event_time_ms"] or 0) <= end
            ]
            if not partitioned:
                result = Counter(edge["edge_type"] for edge in selected)
                self._edge_count_cache[cache_key] = Counter(result)
                return result
            partials = [Counter(), Counter(), Counter(), Counter()]
            for edge in selected:
                partials[stable_owner(edge["src"], 4)][edge["edge_type"]] += 1
            self._edge_count_cache[cache_key] = [Counter(partial) for partial in partials]
            return partials
        if not partitioned:
            result = Counter(
                dict(
                    self._conn.execute(
                        "SELECT edge_type,COUNT(*) FROM edges WHERE event_time_ms>=? AND event_time_ms<=? GROUP BY edge_type",
                        (start, end),
                    )
                )
            )
            self._edge_count_cache[cache_key] = Counter(result)
            return result
        partials = [Counter(), Counter(), Counter(), Counter()]
        for part, kind, count in self._conn.execute(
            "SELECT o.partition_id,e.edge_type,COUNT(*) FROM edges e JOIN canonical_owners_v2 o ON o.vertex_id=e.src "
            "WHERE e.event_time_ms>=? AND e.event_time_ms<=? GROUP BY o.partition_id,e.edge_type",
            (start, end),
        ):
            partials[int(part)][kind] = int(count)
        self._edge_count_cache[cache_key] = [Counter(partial) for partial in partials]
        return partials

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
