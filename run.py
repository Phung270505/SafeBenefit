"""The only public entry point for the SafeBenefit study."""

from __future__ import annotations

import sys

from src.experiment.runner import main

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    raise SystemExit(main(sys.argv[1:]))


