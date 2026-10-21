"""Windows-safe transfer of replay databases between study splits."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from src.datasets.export import _move_replay_database


class ReplayMoveTests(unittest.TestCase):
    def test_closed_sqlite_database_moves_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "calibration.sqlite"
            target = Path(folder) / "evaluation" / "graph.sqlite"
            with closing(sqlite3.connect(source)) as connection:
                connection.execute("CREATE TABLE sample(value INTEGER)")
                connection.commit()
            _move_replay_database(source, target)
            self.assertFalse(source.exists())
            self.assertTrue(target.is_file())
            with closing(sqlite3.connect(target)) as connection:
                self.assertIsNotNone(
                    connection.execute(
                        "SELECT name FROM sqlite_master WHERE name='sample'"
                    ).fetchone()
                )

    def test_transient_windows_sharing_violation_is_retried(self) -> None:
        locked = PermissionError(13, "locked")
        locked.winerror = 32
        with (
            patch.object(Path, "replace", side_effect=[locked, None]) as replace,
            patch("src.datasets.export.time.sleep") as sleep,
        ):
            _move_replay_database(Path("source.sqlite"), Path("target.sqlite"))
        self.assertEqual(replace.call_count, 2)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
