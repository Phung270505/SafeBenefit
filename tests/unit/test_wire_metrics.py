"""Wire-byte accounting uses host MAC transmit scalars, not header guesses."""

import tempfile
import unittest
from pathlib import Path

from src.simulation.wire_metrics import host_wire_bytes


class WireMetricsTests(unittest.TestCase):
    def test_exact_host_sum(self):
        content = "\n".join(
            [
                "scalar GraphQuery.coordinator.eth[0].mac txPk:sum(packetBytes) 100",
                "scalar GraphQuery.worker[0].eth[0].mac txPk:sum(packetBytes) 200",
                "scalar GraphQuery.worker[1].eth[0].mac txPk:sum(packetBytes) 300",
                "scalar GraphQuery.switch.eth[0].mac txPk:sum(packetBytes) 999",
            ]
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.sca"
            path.write_text(content)
            self.assertEqual(host_wire_bytes(path, expected_workers=2), 600)

    def test_missing_host_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.sca"
            path.write_text("scalar GraphQuery.coordinator.eth[0].mac txPk:sum(packetBytes) 1\n")
            with self.assertRaises(ValueError):
                host_wire_bytes(path, expected_workers=1)


if __name__ == "__main__":
    unittest.main()
