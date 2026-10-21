"""Bulk two-hop SQL preserves BFS semantics and logical work on duplicate/self edges."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.common.partitioning import stable_owner
from src.datasets.replay import SqliteVersionedGraph, token_id


class BulkReachabilityTests(unittest.TestCase):
    def test_bulk_expansion_matches_reference_for_all_seeds_and_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "snapshot").mkdir()
            vertices = [f"v{index}" for index in range(20)]
            edges = [(vertices[index % 20], vertices[(index * 7 + 3) % 20]) for index in range(100)]
            edges += [("v0", "v0"), ("v0", "v0"), ("v0", "v1"), ("v0", "v1")]
            with (root / "snapshot/vertices.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["vertex_id"])
                writer.writerows([[vertex] for vertex in vertices])
            with (root / "snapshot/edges.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["edge_id", "src", "dst", "edge_type", "event_time_ms"])
                writer.writerows(
                    [index, source, target, "X", index]
                    for index, (source, target) in enumerate(edges)
                )
            neighbors = {vertex: set() for vertex in vertices}
            for source, target in edges:
                neighbors[source].add(target)
                neighbors[target].add(source)
            graph = SqliteVersionedGraph(root, root / "graph.sqlite")
            before = graph.state_checksum()
            try:
                for seed in vertices:
                    for limit in (0, 1, 5, 100):
                        first = neighbors[seed] - {seed}
                        expected = set(first)
                        for middle in first:
                            expected.update(neighbors[middle])
                        expected.discard(seed)
                        result = graph.materialize(
                            {
                                "query_id": "q",
                                "query_class": "TWO_HOP_REACHABILITY",
                                "graph_version": 0,
                                "parameters": {"seed": seed, "limit": limit, "hops": 2},
                            },
                            partitions=4,
                        )
                        expected_tokens = sorted(
                            token_id("vertex", vertex) for vertex in sorted(expected)[:limit]
                        )
                        actual_tokens = sorted(
                            token for partition in result["worker_tokens"] for token in partition
                        )
                        self.assertEqual(actual_tokens, expected_tokens)
                        expected_scans = len(neighbors[seed]) + sum(
                            len(neighbors[middle]) for middle in first
                        )
                        self.assertEqual(result["scanned_rows"], expected_scans)
                        counts = [0] * 4
                        counts[stable_owner(seed, 4)] += len(neighbors[seed])
                        for middle in first:
                            counts[stable_owner(middle, 4)] += len(neighbors[middle])
                        self.assertEqual(result["scanned_rows_by_partition"], counts)
                self.assertEqual(graph.state_checksum(), before)
            finally:
                graph.close()
