"""Exact version replay and four query classes on a toy dynamic graph."""

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.common.files import sha256_file
from src.common.partitioning import stable_owner
from src.datasets.replay import SqliteVersionedGraph, token_id


class MaterializationTests(unittest.TestCase):
    def _dataset(self, root: Path, *, self_loop: bool = False) -> Path:
        dataset = root / "dataset"
        for folder in ("snapshot", "metadata"):
            (dataset / folder).mkdir(parents=True)
        with (dataset / "snapshot" / "vertices.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["vertex_id"])
            writer.writeheader()
            writer.writerows({"vertex_id": value} for value in ("a", "b", "c", "d"))
        with (dataset / "snapshot" / "edges.csv").open("w", newline="") as handle:
            fields = ["edge_id", "src", "dst", "edge_type", "event_time_ms"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            rows = [
                    {"edge_id": "e1", "src": "a", "dst": "b", "edge_type": "X", "event_time_ms": 10},
                    {"edge_id": "e2", "src": "b", "dst": "c", "edge_type": "X", "event_time_ms": 20},
                    {"edge_id": "e3", "src": "b", "dst": "d", "edge_type": "Y", "event_time_ms": 30},
                ]
            if self_loop:
                rows.append(
                    {"edge_id": "e4", "src": "b", "dst": "b", "edge_type": "S", "event_time_ms": 40}
                )
            writer.writerows(rows)
        (dataset / "metadata" / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset_id": "toy",
                    "snapshot": {"vertices": 4, "edges": len(rows)},
                }
            )
        )
        return dataset

    def test_queries_and_add_delete_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            graph = SqliteVersionedGraph(self._dataset(root), root / "graph.sqlite")
            base = {"query_id": "q", "graph_version": 0}
            queries = [
                {**base, "query_class": "NEIGHBOR_TYPE_AGG", "parameters": {"seed": "b"}},
                {**base, "query_class": "TEMPORAL_TYPE_AGG", "parameters": {"start_ms": 0, "end_ms": 25}},
                {**base, "query_class": "TWO_HOP_REACHABILITY", "parameters": {"seed": "a", "limit": 50}},
                {**base, "query_class": "TWO_HOP_JOIN", "parameters": {"seed": "a", "limit": 50}},
            ]
            for query in queries:
                materialized = graph.materialize(query)
                self.assertTrue(materialized["expected_result"])
                self.assertEqual(len(materialized["expected_result_sha256"]), 64)
                if query["query_class"] == "TWO_HOP_REACHABILITY":
                    for vertex in ("b", "c", "d"):
                        owner = stable_owner(vertex, 8)
                        self.assertIn(token_id("vertex", vertex), materialized["worker_tokens"][owner])
            graph.apply_event(
                {
                    "seq": 0,
                    "changes": [
                        {
                            "action": "ADD_EDGE",
                            "record": {"edge_id": "e4", "src": "c", "dst": "d", "edge_type": "Z", "event_time_ms": 40},
                        }
                    ],
                }
            )
            graph.apply_event(
                {"seq": 1, "changes": [{"action": "DELETE_EDGE", "record": {"edge_id": "e4"}}]}
            )
            self.assertEqual(graph.graph_size(), (4, 3))
            graph.close()

    def test_checksum_is_sequential_stable_and_graph_size_is_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / "graph.sqlite"
            dataset = self._dataset(root)
            graph = SqliteVersionedGraph(dataset, database)
            progress: list[tuple[int, int]] = []
            initial = graph.state_checksum(
                progress=lambda completed, total: progress.append((completed, total)),
                chunk_size=17,
            )
            self.assertEqual(initial, sha256_file(database))
            self.assertEqual(initial, graph.state_checksum(chunk_size=19))
            self.assertTrue(progress)
            self.assertEqual(progress[-1][0], progress[-1][1])
            self.assertEqual(graph.graph_size(), (4, 3))
            graph.apply_event(
                {
                    "seq": 0,
                    "changes": [
                        {
                            "action": "ADD_EDGE",
                            "record": {
                                "edge_id": "e-new",
                                "src": "a",
                                "dst": "d",
                                "edge_type": "N",
                                "event_time_ms": 60,
                            },
                        }
                    ],
                }
            )
            changed = graph.state_checksum(chunk_size=23)
            self.assertNotEqual(initial, changed)
            self.assertEqual(graph.graph_size(), (4, 4))
            graph.close()

            reopened = SqliteVersionedGraph(dataset, database)
            self.assertEqual(reopened.version, 1)
            self.assertEqual(reopened.graph_size(), (4, 4))
            reopened.close()

    def test_partial_snapshot_resumes_from_durable_row_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = self._dataset(root)
            database = root / "graph.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                """
                CREATE TABLE vertices(id TEXT PRIMARY KEY);
                CREATE TABLE edges(
                    id TEXT PRIMARY KEY, src TEXT NOT NULL, dst TEXT NOT NULL,
                    kind TEXT NOT NULL, event_time INTEGER NOT NULL,
                    FOREIGN KEY(src) REFERENCES vertices(id),
                    FOREIGN KEY(dst) REFERENCES vertices(id)
                );
                CREATE TABLE graph_meta(key TEXT PRIMARY KEY, value INTEGER NOT NULL);
                INSERT INTO vertices VALUES ('a');
                INSERT INTO vertices VALUES ('b');
                INSERT INTO graph_meta VALUES ('snapshot_vertices_loaded', 2);
                INSERT INTO graph_meta VALUES ('snapshot_edges_loaded', 0);
                INSERT INTO graph_meta VALUES ('snapshot_complete', 0);
                """
            )
            connection.commit()
            connection.close()

            progress: list[tuple[str, int, int]] = []
            graph = SqliteVersionedGraph(
                dataset,
                database,
                initialization_progress=lambda phase, done, total: progress.append(
                    (phase, done, total)
                ),
            )
            self.assertEqual(graph.graph_size(), (4, 3))
            self.assertTrue(progress)
            graph.close()

            observer = sqlite3.connect(database)
            try:
                metadata = dict(observer.execute("SELECT key,value FROM graph_meta"))
                self.assertEqual(metadata["snapshot_complete"], 1)
                self.assertEqual(metadata["snapshot_vertices_loaded"], 4)
                self.assertEqual(metadata["snapshot_edges_loaded"], 3)
            finally:
                observer.close()

    def test_replay_progress_reports_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            graph = SqliteVersionedGraph(self._dataset(root), root / "graph.sqlite")
            reports: list[tuple[int, int]] = []
            graph.advance(
                iter(
                    [
                        {"seq": 0, "changes": []},
                        {"seq": 1, "changes": []},
                        {"seq": 2, "changes": []},
                    ]
                ),
                3,
                progress=lambda current, target: reports.append((current, target)),
                progress_every=2,
            )
            self.assertEqual(reports, [(2, 3), (3, 3)])
            graph.close()

    def test_generic_hops_commit_and_rollback_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            database = root / "graph.sqlite"
            graph = SqliteVersionedGraph(self._dataset(root), database)
            try:
                base = {
                    "query_id": "reach",
                    "query_class": "TWO_HOP_REACHABILITY",
                    "graph_version": 0,
                }
                one_hop = graph.materialize(
                    {**base, "parameters": {"seed": "a", "hops": 1, "limit": 50}}
                )
                three_hops = graph.materialize(
                    {**base, "parameters": {"seed": "a", "hops": 3, "limit": 50}}
                )
                self.assertEqual(one_hop["intermediate_rows"], 1)
                self.assertEqual(three_hops["intermediate_rows"], 3)
                with self.assertRaises(ValueError):
                    graph.advance(iter(()), -1)
                graph.apply_event(
                    {
                        "seq": 0,
                        "changes": [
                            {
                                "action": "ADD_EDGE",
                                "record": {
                                    "edge_id": "e5",
                                    "src": "c",
                                    "dst": "d",
                                    "edge_type": "Z",
                                    "event_time_ms": 50,
                                },
                            }
                        ],
                    }
                )
                with self.assertRaises(ValueError):
                    graph.apply_event(
                        {
                            "seq": 1,
                            "changes": [
                                {
                                    "action": "ADD_EDGE",
                                    "record": {
                                        "edge_id": "missing-endpoint",
                                        "src": "c",
                                        "dst": "not-present",
                                        "edge_type": "Z",
                                        "event_time_ms": 51,
                                    },
                                }
                            ],
                        }
                    )
                observer = sqlite3.connect(database)
                try:
                    self.assertEqual(observer.execute("PRAGMA user_version").fetchone()[0], 1)
                    self.assertEqual(
                        observer.execute("SELECT COUNT(*) FROM edges WHERE id='e5'").fetchone()[0],
                        1,
                    )
                finally:
                    observer.close()
            finally:
                graph.close()

    def test_two_hop_join_self_loop_matches_declared_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            graph = SqliteVersionedGraph(
                self._dataset(root, self_loop=True), root / "graph.sqlite"
            )
            result = graph.materialize(
                {
                    "query_id": "join-self-loop",
                    "query_class": "TWO_HOP_JOIN",
                    "graph_version": 0,
                    "parameters": {"seed": "a", "limit": 50},
                }
            )
            owner = stable_owner("b", 8)
            self.assertIn(
                token_id("join", "b", "b"),
                result["worker_tokens"][owner],
            )
            graph.close()


if __name__ == "__main__":
    unittest.main()
