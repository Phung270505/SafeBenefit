"""Fail-closed checks for INET build identity and calibration feature encoding."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.simulation.build_identity import check_build


class InetValidationTests(unittest.TestCase):
    def test_duplicate_checksum_entries_do_not_prove_a_build(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "build.sha256"
            path.write_text(("0" * 64 + "  simulator/omnetpp/graph_query/GraphQueryApp.cc\n") * 5)
            with self.assertRaises(ValueError):
                check_build(path)

    def test_unscoped_manifest_rejected_before_file_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "build.sha256"
            path.write_text(("0" * 64 + "  ../../outside\n") * 5)
            with patch("src.simulation.build_identity.sha256_file") as checksum:
                with self.assertRaises(ValueError):
                    check_build(path)
                checksum.assert_not_called()

    def test_simulator_sources_do_not_depend_on_training_framework(self):
        source = Path("simulator/omnetpp/graph_query/GraphQueryApp.cc").read_text()
        self.assertNotIn("torch", source.lower())
        self.assertNotIn("python", source.lower())
