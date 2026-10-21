"""One streaming snapshot pass shared by calibration/evaluation degree sampling."""

from __future__ import annotations

import hashlib
import heapq
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from src.common.files import sha256_file
from src.common.paths import OUTPUT_ROOT
from src.common.seed import component_seed


def sample_degree_buckets(
    root: Path, dataset: str, component: str, size: int
) -> dict[str, list[str]]:
    """Sample exact degrees over connected vertices without a second graph copy."""
    if size < 384 or size % 3:
        raise ValueError("degree sample size must be divisible by three and >=384")
    cache = OUTPUT_ROOT / "cache/snapshot_statistics" / f"{dataset}.sqlite"
    cache.parent.mkdir(parents=True, exist_ok=True)
    signature = sha256_file(root / "metadata/manifest.json")
    with closing(sqlite3.connect(cache)) as connection:
        connection.executescript(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-262144;"
            "CREATE TABLE IF NOT EXISTS degree(id TEXT PRIMARY KEY, n INTEGER NOT NULL);"
            "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);"
        )
        stored = connection.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
        if stored != (signature,):
            connection.execute("DELETE FROM degree")
            connection.execute("DELETE FROM metadata")
            batch: Counter[str] = Counter()
            rows = 0

            def flush() -> None:
                connection.executemany(
                    "INSERT INTO degree VALUES(?,?) ON CONFLICT(id) DO UPDATE SET n=n+excluded.n",
                    batch.items(),
                )
                batch.clear()

            with (root / "snapshot/edges.csv").open("rb", buffering=64 * 1024 * 1024) as handle:
                if not handle.readline().startswith(b"edge_id,src,dst,"):
                    raise ValueError("unexpected snapshot edge header")
                for line in handle:
                    parts = line.split(b",", 3)
                    if len(parts) != 4:
                        raise ValueError("malformed snapshot edge row")
                    batch[parts[1].decode("utf-8")] += 1
                    batch[parts[2].decode("utf-8")] += 1
                    rows += 1
                    if rows % 2_000_000 == 0:
                        print(
                            f"[snapshot-statistics] {dataset}: {rows:,} edges in the single source pass",
                            flush=True,
                        )
                    if len(batch) >= 20_000:
                        flush()
            if batch:
                flush()
            connection.execute("INSERT INTO metadata VALUES('snapshot_passes','1')")
            connection.execute("INSERT INTO metadata VALUES('snapshot_rows',?)", (str(rows),))
            connection.execute("INSERT INTO metadata VALUES('source',?)", (signature,))
            connection.commit()
        cache_key = f"sample:{component}:{size}"
        stored = connection.execute(
            "SELECT value FROM metadata WHERE key=?", (cache_key,)
        ).fetchone()
        if stored:
            return json.loads(stored[0])
        # Both streams are scored during the same connected-vertex pass.
        components = ("calibration_parameters", "evaluation_parameters")
        seeds = {
            name: component_seed(name, f"{dataset}:bottom-k-degree-sample") for name in components
        }
        heaps: dict[str, list[tuple[int, str, int]]] = {name: [] for name in components}
        for vertex, degree in connection.execute("SELECT id,n FROM degree"):
            for name, seed in seeds.items():
                score = int.from_bytes(
                    hashlib.sha256(f"{seed}\0{vertex}".encode()).digest()[:16], "big"
                )
                item = (-score, vertex, degree)
                if len(heaps[name]) < size:
                    heapq.heappush(heaps[name], item)
                elif score < -heaps[name][0][0]:
                    heapq.heapreplace(heaps[name], item)
        for name, heap in heaps.items():
            if len(heap) != size:
                raise ValueError("insufficient connected vertices for disjoint workload sampling")
            ranked = sorted((degree, vertex) for _, vertex, degree in heap)
            third = size // 3
            buckets = {
                "low": [v for _, v in ranked[:third]],
                "medium": [v for _, v in ranked[third : 2 * third]],
                "high": [v for _, v in ranked[2 * third :]],
            }
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES(?,?)",
                (f"sample:{name}:{size}", json.dumps(buckets)),
            )
        connection.commit()
        stored = connection.execute(
            "SELECT value FROM metadata WHERE key=?", (cache_key,)
        ).fetchone()
        if stored is None:
            raise ValueError("unsupported parameter sampling stream")
        return json.loads(stored[0])
