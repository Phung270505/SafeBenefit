"""Strict LDBC SNB Interactive v1 SF10 adapter entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.datasets.ldbc import adapt_ldbc_v1

DATASET_ID = "ldbc_snb_i1_v100_sf10"
SOURCE_URL = (
    "https://datasets.ldbcouncil.org/snb-interactive-v1/"
    "social_network-sf10-CsvBasic-LongDateFormatter.tar.zst"
)


def adapt_ldbc_sf10(source: Path, output: Path, *, source_archive: Path) -> dict[str, Any]:
    """Adapt an already downloaded official SF10 archive without fetching data."""
    required = (
        source / "static",
        source / "dynamic",
        source / "updateStream_0_0_forum.csv",
        source / "updateStream_0_0_person.csv",
        source_archive,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("LDBC SF10 source is incomplete: " + ", ".join(missing))
    return adapt_ldbc_v1(
        source,
        output,
        dataset_id=DATASET_ID,
        source_archive=source_archive,
        source_url=SOURCE_URL,
    )
