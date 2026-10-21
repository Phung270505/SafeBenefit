"""OMNeT++/INET case execution and correctness validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.analysis.oracle import choose_adaptive_oracle, oracle_diagnostics
from src.common.config import load_study_config
from src.common.files import read_jsonl, sha256_file, stable_json, write_json, write_jsonl
from src.common.paths import OUTPUT_ROOT, PROJECT_ROOT
from src.datasets.discrimination import summarize_plan_costs
from src.datasets.workload import DATASET_SCALES
from src.methods.registry import METHOD_ORDER
from src.plans.switch_cost import SwitchCostModel
from src.simulation.stress import run_delete_stress
from src.simulation.wire_metrics import host_wire_bytes

RESULT_ROOT = OUTPUT_ROOT
METHODS = METHOD_ORDER
LOGGER = logging.getLogger("safebenefit.simulation")


def _payload_size_bucket(payload_bytes: int) -> str:
    """Return the preregistered actual-payload operating region."""
    if payload_bytes < 10_000:
        return "LT_10KB"
    if payload_bytes < 100_000:
        return "10KB_TO_100KB"
    if payload_bytes < 1_000_000:
        return "100KB_TO_1MB"
    if payload_bytes < 10_000_000:
        return "1MB_TO_10MB"
    return "GE_10MB"


@dataclass(frozen=True)
class CaseSpec:
    """One deterministic OMNeT++ execution inside a reusable environment."""

    dataset: str
    split: str
    method: str
    initial: str
    run_id: str
    scenario: str = "Stable"
    workers: int = 4
    datarate: str = "1Gbps"
    delay: str = "50us"
    query_filename: str = "queries.tsv"
    retry_interval_seconds: float | None = None


class StudyBlocked(RuntimeError):
    """A fail-closed simulation or correctness gate was not satisfied."""


def _run(
    command: list[str], log_name: str, *, expected_codes: set[int] | None = None
) -> int:
    log_path = RESULT_ROOT / "logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("command=%s", command)
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:  # pragma: no cover - defensive subprocess contract
            raise StudyBlocked(f"could not capture command output; see {log_path}")
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            if line.startswith("[batch]"):
                print(line.rstrip(), flush=True)
        return_code = process.wait()
    accepted = expected_codes or {0}
    if return_code not in accepted:
        raise StudyBlocked(f"command failed ({return_code}); see {log_path}")
    return return_code


def _wsl_script(script: Path, *arguments: str) -> list[str]:
    if os.name != "nt":
        return ["bash", str(script), *arguments]
    converted = subprocess.run(
        ["wsl.exe", "-d", "OMNeT-opp_env", "--exec", "wslpath", "-u", str(script)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    ).stdout.strip()
    result_root = subprocess.run(
        ["wsl.exe", "-d", "OMNeT-opp_env", "--exec", "wslpath", "-u", str(RESULT_ROOT)],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    ).stdout.strip()
    return [
        "wsl.exe",
        "-d",
        "OMNeT-opp_env",
        "--exec",
        "/usr/bin/env",
        f"SAFEBENEFIT_OUTPUT_ROOT={result_root}",
        "/bin/bash",
        "--noprofile",
        "--norc",
        converted,
        *arguments,
    ]


def _environment_preflight() -> dict[str, Any]:
    if sys.version_info < (3, 12):
        raise StudyBlocked("Python 3.12 or newer is required")
    suite = load_study_config()
    required_packages = {
        "numpy": "2.3.1",
        "orjson": "3.10.18",
        "matplotlib": "3.10.7",
        "python-docx": "1.2.0",
        "ruff": "0.12.5",
    }
    installed_packages: dict[str, str] = {}
    for package, expected in required_packages.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise StudyBlocked(f"required Python package is missing: {package}=={expected}") from error
        if actual != expected:
            raise StudyBlocked(
                f"Python package version mismatch: {package}=={actual}, expected {expected}"
            )
        installed_packages[package] = actual
    if os.name == "nt":
        probe = subprocess.run(
            [
                "wsl.exe",
                "-d",
                "OMNeT-opp_env",
                "--exec",
                "bash",
                "-lc",
                "test -x /home/opp_env/.venv/bin/opp_env && "
                "test -x /home/opp_env/default_workspace/omnetpp-6.4.0/bin/opp_run && "
                "test -d /home/opp_env/default_workspace/inet-4.7.0/src",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if probe.returncode:
            raise StudyBlocked("WSL OMNeT-opp_env, OMNeT++ 6.4.0 or INET 4.7.0 is unavailable")
    return {
        "python": sys.version.split()[0],
        "study_id": suite["study_id"],
        "master_seed": suite["master_seed"],
        "simulation_only": True,
        "python_packages": installed_packages,
    }


def _case_arguments(spec: CaseSpec) -> list[str]:
    """Render the single source of truth for run_case.sh's 18 arguments."""
    suite = load_study_config()
    decision = suite["decision"]
    switch_model = SwitchCostModel.from_config(suite["switch_cost_model"])
    reliability = suite["reliability"]
    runtime_path = RESULT_ROOT / "config" / "runtime_calibration.json"
    if not runtime_path.exists():
        raise StudyBlocked("runtime service-rate calibration is missing")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    return [
        spec.dataset,
        spec.split,
        spec.method,
        spec.scenario,
        spec.initial,
        spec.run_id,
        str(decision["benefit_safety_margin"]),
        str(switch_model.estimate(spec.workers)),
        str(
            reliability["retry_interval_seconds"]
            if spec.retry_interval_seconds is None
            else spec.retry_interval_seconds
        ),
        str(spec.workers),
        spec.datarate,
        spec.delay,
        spec.query_filename,
        str(decision["uncertainty_z"]),
        str(decision["prediction_relative_std"]),
        str(runtime["scan_cpu_units_per_second"]),
        str(runtime["codec_cpu_units_per_second"]),
        str(runtime["io_bytes_per_second"]),
    ]


def _run_case(
    dataset: str,
    split: str,
    method: str,
    initial: str,
    run_id: str,
    *,
    scenario: str = "Stable",
    workers: int = 4,
    datarate: str = "1Gbps",
    delay: str = "50us",
    query_filename: str = "queries.tsv",
    retry_interval_seconds: float | None = None,
) -> Path:
    spec = CaseSpec(
        dataset=dataset,
        split=split,
        method=method,
        initial=initial,
        run_id=run_id,
        scenario=scenario,
        workers=workers,
        datarate=datarate,
        delay=delay,
        query_filename=query_filename,
        retry_interval_seconds=retry_interval_seconds,
    )
    command = _wsl_script(
        PROJECT_ROOT / "simulator" / "omnetpp" / "tools" / "run_case.sh",
        *_case_arguments(spec),
    )
    _run(command, f"{run_id}.log")
    return RESULT_ROOT / "simulation" / run_id


def _case_has_output(spec: CaseSpec) -> bool:
    """Reject success markers from globally truncated OMNeT++ runs."""
    folder = RESULT_ROOT / "simulation" / spec.run_id
    files_complete = all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            folder / "queries.jsonl",
            folder / "metrics.sca",
            folder / ".complete",
        )
    )
    if not files_complete:
        return False
    input_file = (
        RESULT_ROOT / "cache" / "materialized" / spec.split / spec.dataset
        / spec.query_filename
    )
    if not input_file.is_file():
        return False
    with input_file.open("r", encoding="utf-8") as source:
        expected = sum(1 for line in source if line.strip())
    with (folder / "queries.jsonl").open("r", encoding="utf-8") as actual:
        observed = sum(1 for line in actual if line.strip())
    return expected > 0 and observed == expected


def _run_case_batch(batch_id: str, specs: list[CaseSpec]) -> list[Path]:
    """Run pending cases sequentially inside one OMNeT++/INET environment."""
    if not batch_id.replace("-", "").replace("_", "").isalnum():
        raise StudyBlocked(f"invalid simulation batch id: {batch_id}")
    pending = [spec for spec in specs if not _case_has_output(spec)]
    if not pending:
        print(f"[batch] RESUME {batch_id}: all {len(specs)} cases already exist", flush=True)
        return [RESULT_ROOT / "simulation" / spec.run_id for spec in specs]

    batch_root = RESULT_ROOT / "cache" / "batches"
    batch_root.mkdir(parents=True, exist_ok=True)
    manifest = batch_root / f"{batch_id}.tsv"
    rendered: list[str] = []
    for spec in pending:
        arguments = _case_arguments(spec)
        if len(arguments) != 18 or any("\t" in value or "\n" in value for value in arguments):
            raise StudyBlocked(f"unsafe batch argument for {spec.run_id}")
        rendered.append("\t".join(arguments))
    manifest.write_text(
        "\n".join(rendered) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"[batch] QUEUED {batch_id}: {len(pending)} pending, "
        f"{len(specs) - len(pending)} resumed",
        flush=True,
    )
    command = _wsl_script(
        PROJECT_ROOT / "simulator" / "omnetpp" / "tools" / "run_batch.sh",
        batch_id,
    )
    _run(command, f"batch-{batch_id}.log")
    return [RESULT_ROOT / "simulation" / spec.run_id for spec in specs]


def _result_hash(result: dict[str, Any]) -> str:
    canonical = sorted([int(label), int(count)] for label, count in result.items())
    return hashlib.sha256(stable_json(canonical).encode("utf-8")).hexdigest()


def _contracts(split: str, dataset: str) -> dict[str, dict[str, Any]]:
    path = RESULT_ROOT / "cache" / "materialized" / split / dataset / "queries.jsonl"
    contracts: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row["query_id"] in contracts:
            raise StudyBlocked(f"duplicate query contract: {row['query_id']}")
        contracts[row["query_id"]] = row
    return contracts


def _validate_case(
    folder: Path,
    split: str,
    dataset: str,
    method: str,
    *,
    query_ids: set[str] | None = None,
    allow_cancel: bool = False,
) -> list[dict[str, Any]]:
    contracts = _contracts(split, dataset)
    if query_ids is not None:
        contracts = {key: value for key, value in contracts.items() if key in query_ids}
    path = folder / "queries.jsonl"
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["query_id"] in seen_ids:
                raise StudyBlocked(f"duplicate query result: {row['query_id']}")
            seen_ids.add(row["query_id"])
            contract = contracts.get(row["query_id"])
            if (
                contract is None or row["method"] != method
                or row["query_class"] != contract["query_class"]
            ):
                raise StudyBlocked(f"run identity mismatch: {row.get('query_id')}")
            result_hash = _result_hash(row["result"])
            cancelled = row["outcome"] == "CANCELLED" and allow_cancel
            status = (
                "PASS"
                if (cancelled or (
                    row["outcome"] == "SUCCESS"
                    and result_hash == contract["expected_result_sha256"]
                ))
                and int(row["graph_version"]) == int(contract["graph_version"])
                else "FAIL"
            )
            row.update(
                {
                    "run_id": folder.name,
                    "dataset_id": dataset,
                    "method_id": method,
                    "scale_id": contract["scale_id"],
                    "schedule_id": contract["schedule_id"],
                    "schedule_profile": contract["schedule_profile"],
                    "replica_id": contract["replica_id"],
                    "graph_version_bucket": contract["graph_version_bucket"],
                    "result_sha256": result_hash,
                    "expected_result_sha256": contract["expected_result_sha256"],
                    "status": status,
                    "cpu_ms": row["total_modeled_cpu_service_ms"],
                    "simulation_time_ms": row["latency_simtime_ms"],
                    "intermediate_rows": row["selected_rows"],
                    "plan_id": row["plan"],
                    "payload_size_bucket": _payload_size_bucket(
                        int(row["payload_bytes"])
                    ),
                }
            )
            if int(row.get("decision_latency_ns", -1)) < 0:
                raise StudyBlocked(f"invalid decision latency: {row['query_id']}")
            if status != "PASS":
                raise StudyBlocked(f"oracle mismatch: {row['query_id']}")
            components = sum(
                int(row[field])
                for field in ("payload_bytes", "control_bytes", "retry_bytes", "switch_wire_bytes")
            )
            if components != int(row["total_wire_bytes"]):
                raise StudyBlocked(f"wire-byte decomposition mismatch: {row['query_id']}")
            row["switch_bytes"] = row.pop("switch_wire_bytes")
            # Retain canonical hashes in metrics, not the potentially huge result
            # map. Raw per-query simulation records remain unchanged for audit.
            row.pop("result")
            rows.append(row)
    if not rows or len(rows) != len(contracts):
        raise StudyBlocked(f"query count mismatch in {path}")
    scalar_total = host_wire_bytes(folder / "metrics.sca", expected_workers=int(rows[0]["workers"]))
    if scalar_total != sum(int(row["total_wire_bytes"]) for row in rows):
        raise StudyBlocked(f"MAC/per-query wire mismatch: {folder}")
    return rows


def _forced_calibration() -> list[dict[str, Any]]:
    by_query: dict[str, dict[int, dict[str, Any]]] = {}
    specs = [
        CaseSpec(
            dataset=dataset,
            split="calibration",
            method="STATIC",
            initial=str(plan),
            run_id=f"cal-{dataset}-p{plan}",
        )
        for dataset in DATASET_SCALES
        for plan in (0, 1, 2)
    ]
    _run_case_batch("forced-calibration", specs)
    for spec in specs:
        folder = RESULT_ROOT / "simulation" / spec.run_id
        for row in _validate_case(folder, spec.split, spec.dataset, spec.method):
            if int(row["plan"]) != int(spec.initial):
                raise StudyBlocked(f"forced calibration executed wrong plan: {row['query_id']}")
            by_query.setdefault(row["query_id"], {})[int(spec.initial)] = row
    samples: list[dict[str, Any]] = []
    switch_model = SwitchCostModel.from_config(
        load_study_config()["switch_cost_model"]
    )
    for query_id, plans in sorted(by_query.items()):
        if set(plans) != {0, 1, 2}:
            raise StudyBlocked(f"incomplete calibration branches: {query_id}")
        reference = plans[0]
        for field in (
            "query_class",
            "replica_id",
            "expected_result_sha256",
            "selected_rows",
            "distinct_groups",
            "mean_label_bytes",
            "graph_version",
            "predicted_plan_wire_bytes",
            "partition_skew_cv",
        ):
            if any(row[field] != reference[field] for row in plans.values()):
                raise StudyBlocked(f"calibration context differs by plan: {query_id}:{field}")
        costs = {str(plan): int(row["total_wire_bytes"]) for plan, row in plans.items()}
        samples.append(
            {
                "query_id": query_id,
                "dataset_id": reference["dataset_id"],
                "query_class": reference["query_class"],
                "replica_id": reference["replica_id"],
                "graph_version": reference["graph_version"],
                "graph_scale_edges": reference["graph_edge_count"],
                "worker_count": reference["workers"],
                "selected_rows": reference["selected_rows"],
                "distinct_groups": reference["distinct_groups"],
                "mean_label_bytes": reference["mean_label_bytes"],
                "predicted_bytes": reference["predicted_plan_wire_bytes"],
                "partition_skew_cv": reference["partition_skew_cv"],
                "retry_count": 0,
                "switch_bytes": int(switch_model.estimate(reference["workers"])),
                "costs": costs,
                "durations_ms": {
                    str(plan): row["latency_simtime_ms"] for plan, row in plans.items()
                },
            }
        )
    gates = {
        **summarize_plan_costs(samples),
        "metric_scope": "INET_MAC_wire_bytes",
        "correctness_validated": True,
    }
    write_json(RESULT_ROOT / "calibration" / "gates.json", gates)
    trace = RESULT_ROOT / "calibration" / "forced_plan_wire.jsonl"
    write_jsonl(trace, samples)
    if gates["errors"]:
        raise StudyBlocked("; ".join(gates["errors"]))
    for warning in gates["warnings"]:
        LOGGER.warning("calibration diagnostic (non-blocking): %s", warning)
    return samples


def _verify_frozen_policy() -> str:
    """Fail before or after evaluation if the frozen policy changed."""
    training_path = RESULT_ROOT / "policies" / "training.json"
    deployment_path = RESULT_ROOT / "policies" / "policy.tsv"
    evaluation_path = RESULT_ROOT / "policies" / "candidate_policy.tsv"
    if not all(
        path.exists() for path in (training_path, deployment_path, evaluation_path)
    ):
        raise StudyBlocked("frozen LinUCB policy artifact is missing")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    expected_deployment = str(training.get("deployment_policy_sha256", ""))
    expected_evaluation = str(training.get("candidate_policy_sha256", ""))
    actual_deployment = sha256_file(deployment_path)
    actual_evaluation = sha256_file(evaluation_path)
    if (
        not expected_deployment
        or not expected_evaluation
        or actual_deployment != expected_deployment
        or actual_evaluation != expected_evaluation
        or actual_deployment != actual_evaluation
    ):
        raise StudyBlocked("frozen policy checksum mismatch")
    return actual_evaluation


def _verify_frozen_method_artifacts() -> dict[str, str]:
    """Verify every learned/calibrated artifact consumed by evaluation."""
    policy_sha = _verify_frozen_policy()
    training_path = RESULT_ROOT / "policies" / "baseline_training.json"
    threshold_path = RESULT_ROOT / "policies" / "switch_threshold.tsv"
    switch_model_path = RESULT_ROOT / "policies" / "switch_cost_model.tsv"
    if not all(
        path.is_file()
        for path in (training_path, threshold_path, switch_model_path)
    ):
        raise StudyBlocked("frozen calibrated method artifact is missing")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    actual_threshold = sha256_file(threshold_path)
    actual_switch_model = sha256_file(switch_model_path)
    if (
        training.get("status") != "PASS"
        or str(training.get("switch_threshold_sha256", "")).upper() != actual_threshold
        or str(training.get("switch_cost_model_sha256", "")).upper()
        != actual_switch_model
    ):
        raise StudyBlocked("frozen calibrated method checksum mismatch")
    return {
        "linucb": policy_sha,
        "switch_threshold": actual_threshold,
        "switch_cost_model": actual_switch_model,
    }


def _write_initial_plans(initial: dict[str, int]) -> None:
    lines: list[str] = []
    for dataset in DATASET_SCALES:
        for query in _contracts("evaluation", dataset).values():
            key = f"{dataset}:{query['query_class']}"
            lines.append(f"{query['query_id']} {initial[key]}\n")
    path = RESULT_ROOT / "policies" / "initial_plans.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(sorted(lines)), encoding="utf-8")


def _forced_evaluation_costs() -> dict[str, dict[int, int]]:
    """Measure the three forced plans to provide a post-freeze regret lower bound."""
    _verify_frozen_method_artifacts()
    costs: dict[str, dict[int, int]] = {}
    forced_rows: list[dict[str, Any]] = []
    specs = [
        CaseSpec(
            dataset=dataset,
            split="evaluation",
            method="STATIC",
            initial=str(plan),
            run_id=f"oracle-{dataset}-p{plan}",
        )
        for dataset in DATASET_SCALES
        for plan in (0, 1, 2)
    ]
    _run_case_batch("forced-evaluation", specs)
    for spec in specs:
        plan = int(spec.initial)
        folder = RESULT_ROOT / "simulation" / spec.run_id
        selected = _validate_case(folder, spec.split, spec.dataset, spec.method)
        for row in selected:
            row["forced_plan_id"] = plan
            costs.setdefault(row["query_id"], {})[plan] = int(row["total_wire_bytes"])
            forced_rows.append(row)
    expected_queries = sum(
        len(_contracts("evaluation", dataset)) for dataset in DATASET_SCALES
    )
    if len(costs) != expected_queries or any(
        set(values) != {0, 1, 2} for values in costs.values()
    ):
        raise StudyBlocked("forced evaluation plan matrix is incomplete")
    expected_rows = int(
        load_study_config()["evaluation_design"]["expected_forced_plan_runs"]
    )
    if len(forced_rows) != expected_rows:
        raise StudyBlocked(
            f"forced evaluation row count {len(forced_rows)} differs from {expected_rows}"
        )
    path = RESULT_ROOT / "metrics" / "forced_evaluation_plans.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, forced_rows)
    return costs


def _run_evaluation(
    forced_costs: dict[str, dict[int, int]] | None = None,
) -> list[dict[str, Any]]:
    frozen_checksums = _verify_frozen_method_artifacts()
    forced_costs = forced_costs or _forced_evaluation_costs()
    rows: list[dict[str, Any]] = []
    specs = [
        CaseSpec(
            dataset=dataset,
            split="evaluation",
            method=method,
            initial="registered",
            run_id=f"main-{dataset}-{method.lower()}",
        )
        for dataset in DATASET_SCALES
        for method in METHODS
    ]
    _run_case_batch("main-evaluation", specs)
    for spec in specs:
        folder = RESULT_ROOT / "simulation" / spec.run_id
        selected = _validate_case(folder, spec.split, spec.dataset, spec.method)
        for row in selected:
            costs = forced_costs[row["query_id"]]
            initial_plan = int(row["initial_plan"])
            switch_cost = int(round(float(row["estimated_switch_wire_bytes"])))
            oracle = choose_adaptive_oracle(costs, initial_plan, switch_cost)
            oracle_plan = oracle.plan_id
            row["forced_plan_wire_bytes"] = {
                str(plan): value for plan, value in sorted(costs.items())
            }
            row["oracle_plan_id"] = oracle_plan
            row["oracle_optimal_plan_ids"] = list(oracle.optimal_plan_ids)
            row["oracle_base_plan_wire_bytes"] = oracle.base_plan_wire_bytes
            row["oracle_wire_bytes"] = oracle.effective_wire_bytes
            selected_plan = int(row["plan_id"])
            row.update(
                oracle_diagnostics(
                    selected_plan,
                    initial_plan,
                    int(row["total_wire_bytes"]),
                    oracle,
                )
            )
            if row["regret_bytes"] < 0:
                raise StudyBlocked(f"negative forced-plan regret: {row['query_id']}")
        rows.extend(selected)
    path = RESULT_ROOT / "metrics" / "per_query.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, rows)
    expected_queries = sum(
        len(_contracts("evaluation", dataset)) for dataset in DATASET_SCALES
    )
    expected_runs = len(METHODS) * expected_queries
    if len(rows) != expected_runs:
        raise StudyBlocked(
            f"main result count {len(rows)} differs from {expected_runs}"
        )
    if _verify_frozen_method_artifacts() != frozen_checksums:
        raise StudyBlocked("method artifact changed during main evaluation")
    return rows


def _write_query_subset(dataset: str, name: str, query_ids: set[str]) -> str:
    root = RESULT_ROOT / "cache" / "materialized" / "evaluation" / dataset
    source = root / "queries.tsv"
    lines = [
        line
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.split("\t", 1)[0] in query_ids
    ]
    if len(lines) != len(query_ids):
        raise StudyBlocked(f"query subset {name} is incomplete for {dataset}")
    filename = f"queries_{name}.tsv"
    (root / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return filename


def _subset_ids(dataset: str, schedules: set[int], replica: int = 0) -> set[str]:
    return {
        query_id
        for query_id, query in _contracts("evaluation", dataset).items()
        if int(query["schedule_id"]) in schedules and int(query["replica_id"]) == replica
    }


def _run_scalability() -> list[dict[str, Any]]:
    frozen_checksums = _verify_frozen_method_artifacts()
    suite = load_study_config()["scalability_evaluation"]
    rows: list[dict[str, Any]] = []
    schedule_ids = {int(value) for value in suite["schedule_ids"]}
    for dataset in DATASET_SCALES:
        query_ids = _subset_ids(dataset, schedule_ids)
        query_file = _write_query_subset(dataset, "scalability", query_ids)
        specs = [
            CaseSpec(
                dataset=dataset,
                split="evaluation",
                method=method,
                initial="registered",
                run_id=f"scale-{dataset}-{method.lower()}-w{workers}-{profile}",
                workers=int(workers),
                datarate=network["datarate"],
                delay=network["delay"],
                query_filename=query_file,
            )
            for workers in suite["worker_counts"]
            for profile, network in suite["network_profiles"].items()
            for method in METHODS
        ]
        _run_case_batch(f"scalability-{dataset}", specs)
        for spec in specs:
            folder = RESULT_ROOT / "simulation" / spec.run_id
            selected = _validate_case(
                folder,
                spec.split,
                spec.dataset,
                spec.method,
                query_ids=query_ids,
            )
            profile = spec.run_id.rsplit("-", 1)[-1]
            for row in selected:
                row["network_profile"] = profile
            rows.extend(selected)
    path = RESULT_ROOT / "metrics" / "scalability_per_query.jsonl"
    write_jsonl(path, rows)
    expected_rows = sum(
        len(_subset_ids(dataset, schedule_ids))
        * len(suite["worker_counts"])
        * len(suite["network_profiles"])
        * len(METHODS)
        for dataset in DATASET_SCALES
    )
    if len(rows) != expected_rows:
        raise StudyBlocked(
            f"scalability row count {len(rows)} differs from {expected_rows}"
        )
    distinct_queries = len(
        {str(row["query_id"]) for row in rows}
    )
    if distinct_queries != int(suite["distinct_queries"]):
        raise StudyBlocked(
            f"scalability distinct query count {distinct_queries} differs from "
            f"{suite['distinct_queries']}"
        )
    if _verify_frozen_method_artifacts() != frozen_checksums:
        raise StudyBlocked("method artifact changed during scalability evaluation")
    return rows


def _fault_evidence(folder: Path, scenario: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require evidence that the configured fault actually occurred."""
    evidence: dict[str, Any] = {}
    for line in (folder / "metrics.sca").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0] == "scalar" and fields[2] in {
            "injectedLostMessage", "duplicateReplyGenerated", "prepareRejected", "cancelAcknowledged",
        }:
            evidence[fields[2]] = evidence.get(fields[2], 0) + float(fields[3])
    evidence["retry_bytes"] = sum(row["retry_bytes"] for row in rows)
    evidence["cancelled_queries"] = sum(row["outcome"] == "CANCELLED" for row in rows)
    if scenario.startswith("Lost") and (evidence.get("injectedLostMessage", 0) < 1 or evidence["retry_bytes"] <= 0):
        raise StudyBlocked(f"fault did not occur/recover: {scenario}")
    if scenario == "DuplicateReplies" and evidence.get("duplicateReplyGenerated", 0) < 1:
        raise StudyBlocked("duplicate fault did not occur")
    if scenario == "PrepareFailure" and evidence.get("prepareRejected", 0) < 1:
        raise StudyBlocked("prepare rejection did not occur")
    if scenario == "CancelQuery" and (evidence.get("cancelAcknowledged", 0) < 1 or evidence["cancelled_queries"] < 1):
        raise StudyBlocked("cancellation did not occur")
    return evidence


def _run_fault_stress() -> dict[str, Any]:
    scenarios = (
        "LostReady",
        "LostAck",
        "LostData",
        "LostScan",
        "LostExecute",
        "DuplicateReplies",
        "PrepareFailure",
        "CancelQuery",
        "PermanentFailure",
    )
    records: list[dict[str, Any]] = []
    for dataset in DATASET_SCALES:
        query_ids = _subset_ids(dataset, {0})
        query_file = _write_query_subset(dataset, "stress", query_ids)
        successful_specs = [
            CaseSpec(
                dataset=dataset,
                split="evaluation",
                method="SAFE_BENEFIT",
                initial="registered",
                run_id=f"stress-{dataset}-{scenario.lower()}",
                scenario=scenario,
                query_filename=query_file,
                retry_interval_seconds=1.0,
            )
            for scenario in scenarios
            if scenario != "PermanentFailure"
        ]
        _run_case_batch(f"stress-{dataset}", successful_specs)
        for spec in successful_specs:
            folder = RESULT_ROOT / "simulation" / spec.run_id
            validated = _validate_case(
                folder,
                spec.split,
                spec.dataset,
                spec.method,
                query_ids=query_ids,
                allow_cancel=spec.scenario == "CancelQuery",
            )
            records.append(
                {
                    "dataset_id": dataset,
                    "scenario": spec.scenario,
                    "status": "PASS",
                    "queries": len(validated),
                    "cancelled": sum(
                        row["outcome"] == "CANCELLED" for row in validated
                    ),
                    "event_evidence": _fault_evidence(folder, spec.scenario, validated),
                }
            )

        for scenario in ("PermanentFailure",):
            run_id = f"stress-{dataset}-{scenario.lower()}"
            command_folder = RESULT_ROOT / "simulation" / run_id
            spec = CaseSpec(
                dataset=dataset,
                split="evaluation",
                method="SAFE_BENEFIT",
                initial="registered",
                run_id=run_id,
                scenario=scenario,
                query_filename=query_file,
                retry_interval_seconds=1.0,
            )
            command = _wsl_script(
                PROJECT_ROOT / "simulator" / "omnetpp" / "tools" / "run_case.sh",
                *_case_arguments(spec),
            )
            code = _run(command, f"{run_id}.log", expected_codes=set(range(1, 256)))
            failure_log = (RESULT_ROOT / "logs" / f"{run_id}.log").read_text(
                encoding="utf-8", errors="replace"
            )
            if "QUERY_RETRY_EXHAUSTED: no successful result" not in failure_log:
                raise StudyBlocked(
                    f"{run_id} failed for an unexpected reason; see its log"
                )
            records.append(
                {
                    "dataset_id": dataset,
                    "scenario": scenario,
                    "status": "PASS",
                    "expected_failure_exit_code": code,
                    "folder": command_folder.relative_to(PROJECT_ROOT).as_posix(),
                }
            )
    delete = run_delete_stress()
    if delete["status"] != "PASS":
        raise StudyBlocked("delete stress failed")
    expected_records = len(DATASET_SCALES) * len(scenarios)
    if len(records) != expected_records:
        raise StudyBlocked(
            f"stress record count {len(records)} differs from {expected_records}"
        )
    report = {"status": "PASS", "network_faults": records, "delete_stress": delete}
    write_json(RESULT_ROOT / "metrics" / "stress_suite.json", report)
    return report


