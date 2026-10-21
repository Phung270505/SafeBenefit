"""Adapter for official LDBC SNB Interactive v1 CsvBasic datasets."""

from __future__ import annotations

import csv
import heapq
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from src.common.files import (
    EDGE_FIELDS,
    VERTEX_FIELDS,
    checksum_manifest,
    dict_writer,
    ensure_dirs,
    sha256_file,
    stable_id,
    stable_json,
    typed_id,
    write_json,
)

TYPE_NAMES = {
    "comment": "Comment",
    "forum": "Forum",
    "organisation": "Organisation",
    "person": "Person",
    "place": "Place",
    "post": "Post",
    "tag": "Tag",
    "tagclass": "TagClass",
}
OPERATION_NAMES = {
    1: "ADD_PERSON",
    2: "ADD_POST_LIKE",
    3: "ADD_COMMENT_LIKE",
    4: "ADD_FORUM",
    5: "ADD_FORUM_MEMBERSHIP",
    6: "ADD_POST",
    7: "ADD_COMMENT",
    8: "ADD_FRIENDSHIP",
}


def _vertex(kind: str, identifier: str, created: int, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "vertex_id": typed_id(kind, identifier),
        "vertex_type": TYPE_NAMES[kind],
        "properties_json": stable_json(properties),
        "valid_from_ms": created,
        "valid_to_ms": "",
    }


def _edge(
    relation: str,
    source_kind: str,
    source: str,
    target_kind: str,
    target: str,
    created: int,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    src = typed_id(source_kind, source)
    dst = typed_id(target_kind, target)
    return {
        "edge_id": stable_id("ldbc-v1", relation, src, dst, created, stable_json(properties or {})),
        "src": src,
        "dst": dst,
        "edge_type": relation.upper(),
        "weight": 1,
        "properties_json": stable_json(properties or {}),
        "event_time_ms": created,
        "valid_from_ms": created,
        "valid_to_ms": "",
    }


def _items(value: str) -> list[str]:
    return [item for item in value.split(";") if item]


def _operation(fields: list[str]) -> list[dict[str, Any]]:
    event_time, operation = int(fields[0]), int(fields[2])
    values = fields[3:]
    changes: list[dict[str, Any]] = []

    def add_vertex(kind: str, identifier: str, created: int, properties: dict[str, Any]) -> None:
        changes.append({"action": "ADD_VERTEX", "record": _vertex(kind, identifier, created, properties)})

    def add_edge(
        relation: str,
        source_kind: str,
        source: str,
        target_kind: str,
        target: str,
        created: int = event_time,
        properties: dict[str, Any] | None = None,
    ) -> None:
        changes.append(
            {
                "action": "ADD_EDGE",
                "record": _edge(
                    relation, source_kind, source, target_kind, target, created, properties
                ),
            }
        )

    if operation == 1:
        person, first, last, gender, birthday, created, ip, browser, city = values[:9]
        languages, emails, tags, studies, works = values[9:14]
        created_ms = int(created)
        add_vertex(
            "person",
            person,
            created_ms,
            {
                "firstName": first,
                "lastName": last,
                "gender": gender,
                "birthday": birthday,
                "locationIP": ip,
                "browserUsed": browser,
                "languages": _items(languages),
                "emails": _items(emails),
            },
        )
        add_edge("ISLOCATEDIN", "person", person, "place", city, created_ms)
        for tag in _items(tags):
            add_edge("HASINTEREST", "person", person, "tag", tag, created_ms)
        for item in _items(studies):
            organisation, year = item.split(",", 1)
            add_edge("STUDYAT", "person", person, "organisation", organisation, created_ms, {"classYear": year})
        for item in _items(works):
            organisation, year = item.split(",", 1)
            add_edge("WORKAT", "person", person, "organisation", organisation, created_ms, {"workFrom": year})
    elif operation in (2, 3):
        person, message, created = values[:3]
        add_edge("LIKES", "person", person, "post" if operation == 2 else "comment", message, int(created))
    elif operation == 4:
        forum, title, created, moderator, tags = values[:5]
        created_ms = int(created)
        add_vertex("forum", forum, created_ms, {"title": title})
        add_edge("HASMODERATOR", "forum", forum, "person", moderator, created_ms)
        for tag in _items(tags):
            add_edge("HASTAG", "forum", forum, "tag", tag, created_ms)
    elif operation == 5:
        forum, person, joined = values[:3]
        add_edge("HASMEMBER", "forum", forum, "person", person, int(joined), {"joinDate": joined})
    elif operation == 6:
        post, image, created, ip, browser, language, content, length, author, forum, country, tags = values[:12]
        created_ms = int(created)
        add_vertex(
            "post",
            post,
            created_ms,
            {"imageFile": image, "locationIP": ip, "browserUsed": browser, "language": language, "content": content, "length": length},
        )
        add_edge("HASCREATOR", "post", post, "person", author, created_ms)
        add_edge("CONTAINEROF", "forum", forum, "post", post, created_ms)
        add_edge("ISLOCATEDIN", "post", post, "place", country, created_ms)
        for tag in _items(tags):
            add_edge("HASTAG", "post", post, "tag", tag, created_ms)
    elif operation == 7:
        comment, created, ip, browser, content, length, author, country, reply_post, reply_comment, tags = values[:11]
        created_ms = int(created)
        add_vertex(
            "comment",
            comment,
            created_ms,
            {"locationIP": ip, "browserUsed": browser, "content": content, "length": length},
        )
        add_edge("HASCREATOR", "comment", comment, "person", author, created_ms)
        add_edge("ISLOCATEDIN", "comment", comment, "place", country, created_ms)
        if reply_post != "-1":
            add_edge("REPLYOF", "comment", comment, "post", reply_post, created_ms)
        elif reply_comment != "-1":
            add_edge("REPLYOF", "comment", comment, "comment", reply_comment, created_ms)
        else:
            raise ValueError("comment update has no reply target")
        for tag in _items(tags):
            add_edge("HASTAG", "comment", comment, "tag", tag, created_ms)
    elif operation == 8:
        first, second, created = values[:3]
        add_edge("KNOWS", "person", first, "person", second, int(created))
    else:
        raise ValueError(f"unsupported LDBC v1 operation: {operation}")
    return changes


def _update_rows(path: Path, stream: str) -> Iterator[tuple[int, str, int, list[str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        for row_number, fields in enumerate(csv.reader(handle, delimiter="|"), start=1):
            if fields:
                yield int(fields[0]), stream, row_number, fields


def _source_events(source: Path) -> Iterator[dict[str, Any]]:
    streams = [
        _update_rows(source / "updateStream_0_0_forum.csv", "forum"),
        _update_rows(source / "updateStream_0_0_person.csv", "person"),
    ]
    for _, stream, row_number, fields in heapq.merge(*streams):
        yield {
            "event_time_ms": int(fields[0]),
            "dependency_time_ms": int(fields[1]),
            "op_code": int(fields[2]),
            "op_name": OPERATION_NAMES[int(fields[2])],
            "source_stream": stream,
            "source_row": row_number,
            "changes": _operation(fields),
        }


def adapt_ldbc_v1(
    source: Path,
    output: Path,
    *,
    dataset_id: str,
    source_archive: Path,
    source_url: str = (
        "https://datasets.ldbcouncil.org/snb-interactive-v1/"
        "social_network-sf1-CsvBasic-LongDateFormatter.tar.zst"
    ),
) -> dict[str, Any]:
    """Convert official CsvBasic snapshot/update files into the project contract."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite processed dataset: {output}")
    paths = ensure_dirs(output)
    vertex_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    vertex_handle, vertex_writer = dict_writer(paths["snapshot"] / "vertices.csv", VERTEX_FIELDS)
    edge_handle, edge_writer = dict_writer(paths["snapshot"] / "edges.csv", EDGE_FIELDS)
    try:
        for group in ("static", "dynamic"):
            for path in sorted((source / group).glob("*.csv")):
                stem = path.stem.rsplit("_", 2)[0]
                parts = stem.split("_")
                if len(parts) == 1 and parts[0] in TYPE_NAMES:
                    kind = parts[0]
                    with path.open(encoding="utf-8", newline="") as handle:
                        for row in csv.DictReader(handle, delimiter="|"):
                            created = int(row.get("creationDate") or 0)
                            properties = {key: value for key, value in row.items() if key != "id"}
                            vertex_writer.writerow(_vertex(kind, row["id"], created, properties))
                            vertex_counts[TYPE_NAMES[kind]] += 1
                elif len(parts) == 3 and parts[0] in TYPE_NAMES and parts[2] in TYPE_NAMES:
                    source_kind, relation, target_kind = parts
                    with path.open(encoding="utf-8", newline="") as handle:
                        reader = csv.reader(handle, delimiter="|")
                        headers = next(reader)
                        for values in reader:
                            properties = dict(zip(headers[2:], values[2:], strict=True))
                            created = int(properties.get("creationDate") or properties.get("joinDate") or 0)
                            edge_writer.writerow(
                                _edge(relation, source_kind, values[0], target_kind, values[1], created, properties)
                            )
                            edge_counts[relation.upper()] += 1
    finally:
        vertex_handle.close()
        edge_handle.close()

    total_events = 0
    for filename in ("updateStream_0_0_forum.csv", "updateStream_0_0_person.csv"):
        with (source / filename).open(encoding="utf-8") as handle:
            total_events += sum(1 for line in handle if line.strip())
    split_at = total_events // 2
    calibration_path = paths["calibration"] / "events.jsonl"
    evaluation_path = paths["evaluation"] / "events.jsonl"
    with calibration_path.open("w", encoding="utf-8", newline="\n") as calibration_handle:
        with evaluation_path.open("w", encoding="utf-8", newline="\n") as evaluation_handle:
            for sequence, event in enumerate(_source_events(source)):
                event["seq"] = sequence
                destination = calibration_handle if sequence < split_at else evaluation_handle
                destination.write(stable_json(event) + "\n")
    write_json(
        paths["metadata"] / "query_mapping.json",
        {
            "NEIGHBOR_TYPE_AGG": "incident edge type count",
            "TEMPORAL_TYPE_AGG": "time-window edge type count",
            "TWO_HOP_REACHABILITY": "undirected two-hop distinct reachability",
            "TWO_HOP_JOIN": "two-hop edge join through a middle vertex",
        },
    )
    artifact_paths = [
        output / "snapshot" / "vertices.csv",
        output / "snapshot" / "edges.csv",
        output / "updates" / "calibration" / "events.jsonl",
        output / "updates" / "evaluation" / "events.jsonl",
        output / "metadata" / "query_mapping.json",
    ]
    manifest = {
        "contract_version": "dataset-adapter-v1",
        "dataset_id": dataset_id,
        "source_format": "LDBC SNB Interactive v1 Datagen 1.0.0 CsvBasic-LongDateFormatter",
        "source_url": source_url,
        "source_archive_bytes": source_archive.stat().st_size,
        "source_archive_sha256": sha256_file(source_archive),
        "snapshot": {
            "vertices": sum(vertex_counts.values()),
            "edges": sum(edge_counts.values()),
            "vertex_types": dict(vertex_counts),
            "edge_types": dict(edge_counts),
        },
        "updates": {
            "total": total_events,
            "calibration": split_at,
            "evaluation": total_events - split_at,
            "split_rule": "chronological_event_order_50_50",
        },
        "query_mapping": "metadata/query_mapping.json",
        "artifact_sha256": checksum_manifest(artifact_paths, output),
    }
    write_json(paths["metadata"] / "manifest.json", manifest)
    return manifest
