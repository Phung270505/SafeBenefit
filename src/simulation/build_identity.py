"""Fail-closed validation for the INET GraphQueryApp build receipt."""

from __future__ import annotations

from pathlib import Path

from src.common.files import sha256_file
from src.common.paths import PROJECT_ROOT

BUILD_FILES = (
    "GraphQueryApp.cc",
    "ProjectionOperators.h",
    "OperationCounters.h",
    "SpillStore.h",
    "GraphQuery.ned",
    "omnetpp.ini",
    "libgraph_query.so",
)


def check_build(path: Path) -> None:
    """Reject missing, duplicate, escaped, stale, or malformed checksum entries."""
    entries = path.read_text(encoding="utf-8").splitlines()
    base = PROJECT_ROOT / "simulator" / "omnetpp" / "graph_query"
    required = {f"simulator/omnetpp/graph_query/{name}" for name in BUILD_FILES}
    parsed = [entry.split(maxsplit=1) for entry in entries]
    if any(len(entry) != 2 for entry in parsed):
        raise ValueError("malformed build identity")
    names = [entry[1].replace("\\", "/") for entry in parsed]
    if len(entries) != len(BUILD_FILES) or set(names) != required:
        raise ValueError("incomplete or duplicate build identity")
    for (expected, _), relative in zip(parsed, names, strict=True):
        target = (PROJECT_ROOT / relative).resolve()
        if not target.is_relative_to(base.resolve()):
            raise ValueError("unscoped build checksum")
        if sha256_file(target).lower() != expected.lower():
            raise ValueError(f"stale build/source: {relative}")
