"""Generate the final simulation-only report from machine-readable artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.files import read_jsonl, sha256_file, write_json
from src.common.paths import OUTPUT_ROOT, PROJECT_ROOT, REPORT_ROOT
from src.methods.registry import METHOD_ORDER
from src.reporting.dataset_results import export_dataset_results

REPORT_PATH = REPORT_ROOT / "final_results.md"
TABLE_PATH = OUTPUT_ROOT / "metrics" / "method_summary.csv"
AUDIT_PATH = OUTPUT_ROOT / "run_manifest_audit.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return list(read_jsonl(path))


def _write_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return export_dataset_results(rows, OUTPUT_ROOT / "metrics")


def _breakdown(rows: list[dict[str, Any]], dimension: str) -> list[str]:
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        method = str(row.get("method_id", row.get("method")))
        grouped[(str(row[dimension]), method)] += int(row["total_wire_bytes"])
    lines = [f"### Theo `{dimension}`", "", "| Nhóm | Phương pháp | Wire bytes |", "|---|---|---:|"]
    lines.extend(
        f"| {group} | {method} | {value:,} |" for (group, method), value in sorted(grouped.items())
    )
    return [*lines, ""]


def generate_report(manifest: dict[str, Any], analysis: dict[str, Any] | None = None) -> None:
    """Write a fail-closed report; rank methods only after a complete PASS run."""
    rows = _read_jsonl(OUTPUT_ROOT / "metrics" / "per_query.jsonl")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SafeBenefit — Kết quả theo từng scale LDBC",
        "",
        "- Study ID: `SAFEBENEFIT-DGQ`",
        f"- Trạng thái: **{manifest.get('status', 'UNKNOWN')}**",
        "- Master seed: `42`",
        "- Metric xếp hạng duy nhất: `total_wire_bytes`.",
        "- Correctness là ràng buộc cứng, không phải accuracy để tối ưu.",
        "- Toàn bộ số liệu là kết quả mô phỏng OMNeT++/INET; không phải phép đo cloud/hardware thật.",
        "- CPU và thời gian tính từ thao tác logic và năng lực tham chiếu cố định; không đo tốc độ máy.",
        "",
    ]
    for diagnostic_file in ("payload_gates.json", "gates.json"):
        path = OUTPUT_ROOT / "calibration" / diagnostic_file
        if path.is_file():
            diagnostic = json.loads(path.read_text(encoding="utf-8"))
            lines.extend([
                f"## Calibration diagnostics — `{diagnostic.get('metric_scope', 'UNKNOWN')}`",
                "",
                "Plan dominance and ties are valid outcomes. Rank reversal is diagnostic only; the workload is not filtered to force a method or plan to win.",
                "",
                "| Query class | Queries | Unique wins RAW / LOCAL / COMPRESSED | Ties |",
                "|---|---:|---|---:|",
            ])
            for query_class, counts in diagnostic.get("unique_win_counts_by_query_class", {}).items():
                wins = " / ".join(str(counts[str(plan)]) for plan in (0, 1, 2))
                lines.append(
                    f"| {query_class} | {diagnostic['query_counts_by_class'][query_class]} | {wins} | {diagnostic['tie_counts_by_query_class'][query_class]} |"
                )
            lines.extend(f"- Diagnostic: {warning}" for warning in diagnostic.get("warnings", []))
            lines.append("")
    if manifest.get("status") == "PASS" and analysis is not None:
        summaries = _write_summary(rows)
        lines.extend(
            [
                f"- Dataset: **{summaries[0]['dataset_id']}**; calibration/policy/thống kê độc lập.",
                f"- Queries with differing method actions: {analysis.get('action_disagreement_queries', 'not recorded')}; zero is a valid outcome, not a failed benchmark.",
                "",
            ]
        )
        lines.extend(
            [
                "## Kết quả chính",
                "",
                "Các phương pháp có tổng byte thấp nhất trên scale này: **"
                + ", ".join(
                    row["method_id"]
                    for row in summaries
                    if row["total_wire_bytes"]
                    == min(item["total_wire_bytes"] for item in summaries)
                )
                + "**.",
                "",
                "| Phương pháp | Tổng wire bytes | Regret bytes |",
                "|---|---:|---:|",
            ]
        )
        for method in METHOD_ORDER:
            lines.append(
                f"| {method} | {analysis['total_wire_bytes'][method]:,} | {analysis['forced_plan_regret'][method]['total_bytes']:,} |"
            )
        lines.extend(
            [
                "",
                "## Chi phí đa chiều",
                "",
                "| Phương pháp | CPU tính TB (ms) | Hoàn thành mô phỏng TB (ms) | Work units chọn plan | Work units encode | Switch | Fallback |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summaries:
            lines.append(
                f"| {row['method_id']} | {row['mean_cpu_ms']:.6f} | {row['mean_simulation_time_ms']:.6f} | {row['sum_work_units_decision']:,} | {row['sum_work_units_encode']:,} | {row['switches']} | {row['fallbacks']} |"
            )
        lines.extend(
            [
                "",
                "Chi tiết thao tác theo pha, byte I/O, queue, buffer/spill, graph version, vertex/edge, worker/hop nằm trong metrics/per_query.csv và metrics/method_summary.csv. Không gộp chi phí khác đơn vị thành composite score.",
                "",
                "Thao tác là logical actions, không phải instruction CPU vật lý. Buffer là kích thước ước lượng, không phải peak RSS. Không báo chi phí tiền/điện năng vì chưa có mô hình.",
                "Schedule là các nhóm tham số trên cùng event stream; bootstrap theo nhóm tham số không chứng minh các graph history độc lập.",
                "",
            ]
        )
        lines.extend(["", "## So sánh SafeBenefit", ""])
        for method, comparison in analysis["comparisons"].items():
            lower, upper = comparison["difference_fraction_ci95"]
            lines.append(
                f"- So với `{method}`: saving={comparison['saving_fraction']:.3%}; CI95=[{lower:.3%}, {upper:.3%}]; Holm p={comparison['holm_adjusted_p']:.4g}; `{comparison['conclusion']}`."
            )
        lines.append("")
        ablation = analysis.get("safebenefit_ablation", {}).get("variants", {})
        if ablation:
            lines.extend(
                [
                    "## SafeBenefit ablation",
                    "",
                    "| Biến thể | Wire bytes | Switch | Regret | Oracle match |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for variant, values in ablation.items():
                lines.append(
                    f"| {variant} | {values['total_wire_bytes']:,} | "
                    f"{values['switches']:,} | {values['regret_bytes']:,} | "
                    f"{values['oracle_match_rate']:.2%} |"
                )
            lines.append("")
        for dimension in (
            "dataset_id",
            "query_class",
            "graph_version_bucket",
            "payload_size_bucket",
        ):
            lines.extend(_breakdown(rows, dimension))
    else:
        lines.extend(
            [
                "## Chưa có bảng xếp hạng",
                "",
                "Lần chạy chưa vượt qua toàn bộ gate nên báo cáo không kết luận phương pháp thắng.",
                "",
            ]
        )
        blockers = [*manifest.get("blockers", [])]
        if manifest.get("error"):
            blockers.append(manifest["error"])
        lines.extend(f"- {item}" for item in dict.fromkeys(map(str, blockers)))
        lines.append("")
    lines.extend(
        [
            "## Phạm vi",
            "",
            "Native/cloud validation nằm ngoài phạm vi phiên bản này. CPU, thời gian, số worker, hop mạng, kích thước đồ thị và intermediate rows được báo cáo riêng; không gộp thành composite score.",
            "",
            "Báo cáo được sinh trực tiếp từ `output/current`; không nhập kết quả thủ công.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    candidates = (
        PROJECT_ROOT / "config" / "study.json",
        OUTPUT_ROOT / "cache/workloads/manifest.json",
        OUTPUT_ROOT / "policies" / "policy.tsv",
        OUTPUT_ROOT / "policies" / "switch_threshold.tsv",
        OUTPUT_ROOT / "policies" / "switch_cost_model.tsv",
        OUTPUT_ROOT / "metrics" / "per_query.jsonl",
        OUTPUT_ROOT / "metrics" / "statistics.json",
        REPORT_PATH,
        TABLE_PATH,
        OUTPUT_ROOT / "metrics" / "safebenefit_ablation.json",
        OUTPUT_ROOT / "metrics" / "safebenefit_parameter_sensitivity.json",
        OUTPUT_ROOT / "metrics" / "cost_model_diagnostics.json",
        OUTPUT_ROOT / "metrics" / "per_query.csv",
        OUTPUT_ROOT / "metrics" / "results.json",
        OUTPUT_ROOT / "calibration" / "payload_gates.json",
        OUTPUT_ROOT / "calibration" / "gates.json",
    )
    write_json(
        AUDIT_PATH,
        {
            "schema_version": 1,
            "study_id": "SAFEBENEFIT-DGQ",
            "study_status": manifest.get("status"),
            "completed_stages": manifest.get("completed_stages", []),
            "artifacts": {
                path.relative_to(PROJECT_ROOT).as_posix(): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in candidates
                if path.is_file()
            },
        },
    )
