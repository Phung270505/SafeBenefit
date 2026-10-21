"""Integrity tests for the temporal citation dataset adapter."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.datasets.citation import adapt_temporal_citations


class CitationAdapterTests(unittest.TestCase):
    def _source(self, root: Path, timestamps: list[int]) -> tuple[Path, Path]:
        edges = root / "edges.csv"
        with edges.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("src", "dst", "event_time_ms"))
            for index, timestamp in enumerate(timestamps):
                writer.writerow((f"p{index}", f"p{index + 1}", timestamp))
        archive = root / "source.dat"
        archive.write_bytes(b"real-source-placeholder-for-checksum")
        return edges, archive

    def test_chronological_split_and_new_vertices_are_replayable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            edges, archive = self._source(root, list(range(10)))
            output = root / "processed"
            manifest = adapt_temporal_citations(
                edges,
                output,
                dataset_id="toy_citations",
                source_archive=archive,
                source_url="https://example.invalid/toy",
            )
            self.assertEqual(manifest["snapshot"]["edges"], 7)
            self.assertEqual(manifest["updates"]["calibration"], 1)
            self.assertEqual(manifest["updates"]["evaluation"], 2)
            events = []
            for split in ("calibration", "evaluation"):
                path = output / "updates" / split / "events.jsonl"
                events.extend(json.loads(line) for line in path.read_text().splitlines())
            self.assertEqual([event["seq"] for event in events], [0, 1, 2])
            self.assertEqual([event["event_time_ms"] for event in events], [7, 8, 9])
            self.assertTrue(
                any(change["action"] == "ADD_VERTEX" for change in events[0]["changes"])
            )

    def test_timestamp_rollback_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            edges, archive = self._source(root, [0, 2, 1, 3])
            with self.assertRaises(ValueError):
                adapt_temporal_citations(
                    edges,
                    root / "processed",
                    dataset_id="toy_citations",
                    source_archive=archive,
                    source_url="https://example.invalid/toy",
                )

if __name__ == "__main__":
    unittest.main()
