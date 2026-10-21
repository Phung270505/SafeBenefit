"""The one-click runner may only reset its own generated output."""

import tempfile
import unittest
from pathlib import Path

from src.common.cleanup import checked_descendant, reset_current_output


class SafeCleanupTests(unittest.TestCase):
    def test_only_output_current_can_be_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "output" / "current").mkdir(parents=True)
            (root / "output" / "current" / "old.txt").write_text("old")
            current = reset_current_output(root, root / "output" / "current")
            self.assertEqual(list(current.iterdir()), [])
            for target in (root, root / "data", root.parent / "outside"):
                with self.assertRaises(ValueError):
                    checked_descendant(target, root)


if __name__ == "__main__":
    unittest.main()
