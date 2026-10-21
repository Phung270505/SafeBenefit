from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

VERTEX_FIELDS = ["vertex_id", "vertex_type", "properties_json", "valid_from_ms", "valid_to_ms"]
EDGE_FIELDS = [
    "edge_id",
    "src",
    "dst",
    "edge_type",
    "weight",
    "properties_json",
    "event_time_ms",
    "valid_from_ms",
    "valid_to_ms",
]


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """MD5 is used only to match checksums published by dataset providers."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_id(*parts: Any) -> str:
    return sha256_text("\x1f".join(str(part) for part in parts))[:24]


def parse_time_ms(value: str | int | None) -> int:
    if value is None or value == "":
        return 0
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        number = int(text)
        return number if abs(number) >= 10_000_000_000 else number * 1000
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def typed_id(type_name: str, raw_id: str | int) -> str:
    return f"{type_name.lower()}:{str(raw_id).strip()}"


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "snapshot": root / "snapshot",
        "calibration": root / "updates" / "calibration",
        "evaluation": root / "updates" / "evaluation",
        "metadata": root / "metadata",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, value: Any, *, allow_nan: bool = True) -> None:
    """Write an indented UTF-8 document; strict callers may reject nonfinite values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dict_writer(path: Path, fields: list[str]):
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    return handle, writer


def checksum_manifest(paths: Iterable[Path], relative_to: Path) -> dict[str, str]:
    return {
        path.relative_to(relative_to).as_posix(): sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
    }

def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Stream records without building a second whole-artifact string in RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")
