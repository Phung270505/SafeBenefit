"""Resumable, fail-closed one-button workflow for SafeBenefit."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.analysis.ablation import (
    analyze_parameter_sensitivity,
    analyze_safebenefit_ablation,
)
from src.analysis.cost_model import analyze_cost_model
from src.analysis.statistics import analyze_results
from src.common.config import load_study_config
from src.common.design import CALIBRATION_PER_DATASET, EVALUATION_PER_DATASET, STUDY_SCHEMA_VERSION
from src.common.files import read_jsonl, sha256_file, write_json
from src.common.paths import OUTPUT_ROOT, PROJECT_ROOT, WORKLOAD_ROOT
from src.datasets.export import export_split
from src.datasets.integrity import validate_dataset_suite
from src.datasets.workload import (
    CALIBRATION_ROOT,
    DATASET_SCALES,
    EVALUATION_MANIFEST,
    EVALUATION_ROOT,
    build_calibration_workload,
    dataset_readiness,
    freeze_evaluation_workload,
    validate_frozen_evaluation_workload,
)
from src.methods.linucb_training import train_linucb
from src.methods.registry import METHOD_ORDER
from src.methods.training import train_calibrated_methods
from src.reporting.report import generate_report
from src.simulation import controller
from src.simulation.calibration import validate_calibration
from src.simulation.reference_model import freeze_reference_model

STUDY_ID = "SAFEBENEFIT-DGQ"
MANIFEST_PATH = OUTPUT_ROOT / "run_manifest.json"
STAGE_COUNT = 16
LOGGER = logging.getLogger("safebenefit.study")


class StudyBlocked(RuntimeError):
    """A scientific, environment or artifact gate failed closed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_output() -> None:
    for relative in (
        "config",
        "cache",
        "calibration",
        "policies",
        "simulation",
        "metrics",
        "figures",
        "report",
        "logs",
    ):
        (OUTPUT_ROOT / relative).mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "config" / "study.json", OUTPUT_ROOT / "config" / "study.json")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(OUTPUT_ROOT / "logs" / "run.log", encoding="utf-8")],
        force=True,
    )


def _run_quality_gates() -> None:
    """Run and persist the exact quality gates used by the one-button workflow."""
    _run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        ],
        "tests.log",
    )
    _run_command(
        [sys.executable, "-m", "ruff", "check", "--no-cache", "src", "tests", "run.py"],
        "ruff.log",
    )
    _run_command(
        [
            sys.executable,
            "-X",
            f"pycache_prefix={OUTPUT_ROOT / 'cache' / 'pycache'}",
            "-m",
            "compileall",
            "-q",
            "src",
            "simulator",
            "run.py",
        ],
        "compileall.log",
    )


def _identity() -> dict[str, str]:
    """Hash every frozen input that may change a resumed scientific run."""
    code_files = sorted(
        [
            PROJECT_ROOT / "run.py",
            PROJECT_ROOT / "pyproject.toml",
            *PROJECT_ROOT.joinpath("src").rglob("*.py"),
            *PROJECT_ROOT.joinpath("tests").rglob("*.py"),
            *PROJECT_ROOT.joinpath("tests", "cpp").rglob("*.cc"),
            *PROJECT_ROOT.joinpath("simulator", "omnetpp").rglob("*.cc"),
            *PROJECT_ROOT.joinpath("simulator", "omnetpp").rglob("*.h"),
            *PROJECT_ROOT.joinpath("simulator", "omnetpp").rglob("*.ned"),
            *PROJECT_ROOT.joinpath("simulator", "omnetpp").rglob("*.ini"),
            *PROJECT_ROOT.joinpath("simulator", "omnetpp").rglob("*.sh"),
        ]
    )
    code_payload = "\n".join(
        f"{path.relative_to(PROJECT_ROOT).as_posix()}:{sha256_file(path)}"
        for path in code_files
        if path.is_file()
    )
    workload_files = sorted(CALIBRATION_ROOT.glob("*.jsonl"))
    workload_payload = "\n".join(f"{path.name}:{sha256_file(path)}" for path in workload_files)
    dataset_files = [PROJECT_ROOT / "data" / "manifests" / "datasets.json"]
    dataset_files.extend(
        PROJECT_ROOT / "data" / "processed" / dataset / "metadata" / "manifest.json"
        for dataset in DATASET_SCALES
    )
    dataset_payload = "\n".join(
        f"{path.relative_to(PROJECT_ROOT).as_posix()}:{sha256_file(path)}"
        for path in sorted(dataset_files)
        if path.is_file()
    )
    for dataset in DATASET_SCALES:
        root = PROJECT_ROOT / "data/processed" / dataset
        for relative in (
            "snapshot/vertices.csv",
            "snapshot/edges.csv",
            "updates/calibration/events.jsonl",
            "updates/evaluation/events.jsonl",
        ):
            path = root / relative
            if path.is_file():
                stat = path.stat()
                dataset_payload += f"\n{dataset}/{relative}:{stat.st_size}:{stat.st_mtime_ns}"

    environment_files = sorted((PROJECT_ROOT / "environment").glob("*"))
    environment_payload = "\n".join(
        f"{path.relative_to(PROJECT_ROOT).as_posix()}:{sha256_file(path)}"
        for path in environment_files
        if path.is_file()
    )

    return {
        "config_sha256": sha256_file(PROJECT_ROOT / "config" / "study.json"),
        "protocol_sha256": sha256_file(PROJECT_ROOT / "config" / "study.json"),
        "code_sha256": hashlib.sha256(code_payload.encode("utf-8")).hexdigest(),
        "calibration_workload_sha256": hashlib.sha256(workload_payload.encode("utf-8")).hexdigest(),
        "dataset_manifests_sha256": hashlib.sha256(dataset_payload.encode("utf-8")).hexdigest(),
        "environment_sha256": hashlib.sha256(environment_payload.encode("utf-8")).hexdigest(),
    }


STAGE_ARTIFACTS: dict[str, tuple[Path, ...]] = {
    "tests": (
        OUTPUT_ROOT / "logs" / "tests.log",
        OUTPUT_ROOT / "logs" / "ruff.log",
        OUTPUT_ROOT / "logs" / "compileall.log",
    ),
    "dataset_integrity": (OUTPUT_ROOT / "config" / "dataset_integrity.json",),
    "calibration_workload": (WORKLOAD_ROOT / "manifest.json",),
    "calibration_materialization": (
        OUTPUT_ROOT / "cache" / "materialized" / "calibration" / "manifest.json",
        OUTPUT_ROOT / "calibration" / "payload_gates.json",
        OUTPUT_ROOT / "calibration" / "forced_plan_payloads.jsonl",
    ),
    "simulator_build": (
        OUTPUT_ROOT / "config" / "runtime_calibration.json",
        OUTPUT_ROOT / "cache" / "build" / "build.sha256",
    ),
    "forced_calibration": (
        OUTPUT_ROOT / "calibration" / "forced_plan_wire.jsonl",
        OUTPUT_ROOT / "calibration" / "gates.json",
    ),
    "method_training": (
        OUTPUT_ROOT / "policies" / "training.json",
        OUTPUT_ROOT / "policies" / "baseline_training.json",
        OUTPUT_ROOT / "policies" / "candidate_policy.tsv",
        OUTPUT_ROOT / "policies" / "policy.tsv",
        OUTPUT_ROOT / "policies" / "switch_threshold.tsv",
        OUTPUT_ROOT / "policies" / "switch_cost_model.tsv",
    ),
    "freeze_evaluation": (
        EVALUATION_MANIFEST,
        *(EVALUATION_ROOT / f"{dataset}.jsonl" for dataset in DATASET_SCALES),
    ),
    "evaluation_materialization": (
        OUTPUT_ROOT / "cache" / "materialized" / "evaluation" / "manifest.json",
        OUTPUT_ROOT / "policies" / "initial_plans.tsv",
        *(
            path
            for dataset in DATASET_SCALES
            for path in (
                OUTPUT_ROOT / "cache" / "materialized" / "evaluation" / dataset / "manifest.json",
                OUTPUT_ROOT / "cache" / "materialized" / "evaluation" / dataset / "queries.jsonl",
                OUTPUT_ROOT / "cache" / "materialized" / "evaluation" / dataset / "queries.tsv",
            )
        ),
    ),
    "forced_evaluation": (OUTPUT_ROOT / "metrics" / "forced_evaluation_plans.jsonl",),
    "main_evaluation": (OUTPUT_ROOT / "metrics" / "per_query.jsonl",),
    "scalability": (OUTPUT_ROOT / "metrics" / "scalability_per_query.jsonl",),
    "stress": (OUTPUT_ROOT / "metrics" / "stress_suite.json",),
    "analysis_report": (
        OUTPUT_ROOT / "metrics" / "statistics.json",
        OUTPUT_ROOT / "metrics" / "safebenefit_ablation.json",
        OUTPUT_ROOT / "metrics" / "safebenefit_parameter_sensitivity.json",
        OUTPUT_ROOT / "metrics" / "cost_model_diagnostics.json",
        OUTPUT_ROOT / "report" / "final_results.md",
        OUTPUT_ROOT / "metrics" / "per_query.csv",
        OUTPUT_ROOT / "metrics" / "method_summary.csv",
        OUTPUT_ROOT / "metrics" / "results.json",
    ),
}


def _artifact_key(path: Path) -> str:
    resolved = path.resolve()
    return (
        resolved.relative_to(PROJECT_ROOT).as_posix()
        if resolved.is_relative_to(PROJECT_ROOT)
        else str(resolved)
    )


def _verify_resume_artifacts(manifest: dict[str, Any]) -> None:
    """Fail closed when a completed stage output is missing or has changed."""
    for stage in manifest.get("completed_stages", []):
        artifacts = STAGE_ARTIFACTS.get(stage, ())
        missing = [path for path in artifacts if not path.is_file()]
        if missing:
            names = ", ".join(_artifact_key(path) for path in missing)
            raise StudyBlocked(f"resume artifacts missing for {stage}: {names}; use --fresh")
        if not artifacts:
            continue
        expected = manifest.get("artifact_receipts", {}).get(stage)
        if not isinstance(expected, dict):
            raise StudyBlocked(f"resume checksum receipt missing for {stage}; use --fresh")
        actual = {_artifact_key(path): sha256_file(path) for path in artifacts}
        if actual != expected:
            raise StudyBlocked(f"resume artifact checksum mismatch for {stage}; use --fresh")


def _save(manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = _now()
    write_json(MANIFEST_PATH, manifest, allow_nan=False)


def _stage(index: int, message: str) -> None:
    line = f"[{index}/{STAGE_COUNT}] {message}"
    print(line, flush=True)
    LOGGER.info(line)


def _complete(manifest: dict[str, Any], stage: str) -> None:
    artifacts = STAGE_ARTIFACTS.get(stage, ())
    if artifacts:
        missing = [path for path in artifacts if not path.is_file()]
        if missing:
            raise StudyBlocked(
                f"stage {stage} did not create required artifacts: "
                + ", ".join(_artifact_key(path) for path in missing)
            )
        manifest.setdefault("artifact_receipts", {})[stage] = {
            _artifact_key(path): sha256_file(path) for path in artifacts
        }
    if stage not in manifest["completed_stages"]:
        manifest["completed_stages"].append(stage)
    _save(manifest)


def _run_command(command: list[str], log_name: str) -> None:
    path = OUTPUT_ROOT / "logs" / log_name
    with path.open("w", encoding="utf-8", errors="replace") as handle:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise StudyBlocked(f"command failed ({result.returncode}); see {path}")


def _load_manifest(identity: dict[str, str]) -> dict[str, Any] | None:
    """Accept only this dataset's exact schema/config/code/data and verified artifacts."""
    if not MANIFEST_PATH.is_file():
        if any(
            path.is_file()
            for directory in ("policies", "simulation", "metrics", "cache/materialized")
            for path in (OUTPUT_ROOT / directory).rglob("*")
        ):
            raise StudyBlocked("orphan dataset artifacts without a manifest; use --fresh")
        return None
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyBlocked("unreadable dataset manifest; use --fresh") from error
    if (
        manifest.get("schema_version") != STUDY_SCHEMA_VERSION
        or len(DATASET_SCALES) != 1
        or manifest.get("dataset_id") != DATASET_SCALES[0]
        or manifest.get("identity") != identity
    ):
        raise StudyBlocked("dataset output is incompatible; use --fresh")
    if "freeze_evaluation" in manifest.get("completed_stages", []):
        if manifest.get("frozen_identity") != identity:
            raise StudyBlocked("frozen dataset identity mismatch")
        validate_frozen_evaluation_workload()
    if manifest.get("status") == "PASS" and manifest.get("completed_stages") != [
        "environment",
        "tests",
        "dataset_integrity",
        "calibration_workload",
        "calibration_materialization",
        "simulator_build",
        "forced_calibration",
        "method_training",
        "freeze_evaluation",
        "evaluation_materialization",
        "forced_evaluation",
        "main_evaluation",
        "scalability",
        "stress",
        "correctness",
        "analysis_report",
    ]:
        raise StudyBlocked("incomplete run cannot be PASS")
    _verify_resume_artifacts(manifest)
    return manifest


def _calibration_manifest() -> dict[str, Any]:
    manifest_path = WORKLOAD_ROOT / "manifest.json"
    if not manifest_path.is_file() or len(list(CALIBRATION_ROOT.glob("*.jsonl"))) != len(
        DATASET_SCALES
    ):
        return build_calibration_workload()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total = sum(
        1
        for path in CALIBRATION_ROOT.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )
    if (
        manifest.get("study_id") != STUDY_ID
        or total != len(DATASET_SCALES) * CALIBRATION_PER_DATASET
    ):
        raise StudyBlocked("registered calibration workload is incompatible")
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def _run_stage(
    manifest: dict[str, Any],
    index: int,
    name: str,
    message: str,
    action: Callable[[], Any],
    *,
    always_verify: bool = False,
) -> Any:
    _stage(index, message)
    if name in manifest["completed_stages"] and not always_verify:
        LOGGER.info("resume: stage %s already complete", name)
        return None
    value = action()
    if isinstance(value, dict) and value.get("status") in {"FAIL", "BLOCKED"}:
        errors = value.get("errors") or [f"{name} returned {value['status']}"]
        raise StudyBlocked("; ".join(map(str, errors)))
    _complete(manifest, name)
    return value


def preflight() -> dict[str, Any]:
    """Run read-only environment and dataset readiness checks."""
    config = load_study_config()
    errors: list[str] = []
    try:
        environment = controller._environment_preflight()
    except controller.StudyBlocked as error:
        environment = {"status": "BLOCKED", "error": str(error)}
        errors.append(str(error))
    datasets = dataset_readiness(DATASET_SCALES)
    errors.extend(datasets["errors"])
    if tuple(config["active_methods"]) != METHOD_ORDER:
        errors.append("method registry mismatch")
    return {
        "status": "PASS" if not errors else "BLOCKED",
        "environment": environment,
        "datasets": datasets,
        "errors": errors,
    }


def run_full() -> int:
    """Execute or safely resume one independent dataset study."""
    if len(DATASET_SCALES) != 1:
        raise StudyBlocked("dataset worker requires exactly one dataset scope")
    _configure_output()
    identity = _identity()
    manifest = _load_manifest(identity)
    if manifest is not None and manifest.get("status") == "PASS":
        return 0
    manifest = manifest or {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "dataset_id": DATASET_SCALES[0],
        "status": "RUNNING",
        "started_at_utc": _now(),
        "master_seed": 42,
        "execution_mode": "full_simulation",
        "identity": identity,
        "completed_stages": [],
    }
    _save(manifest)
    current_stage = "environment"
    analysis: dict[str, Any] | None = None
    try:
        manifest["preflight"] = _run_stage(
            manifest,
            1,
            "environment",
            "Kiểm tra Python, WSL, OMNeT++ và INET",
            preflight,
            always_verify=True,
        )
        if manifest["preflight"]["status"] != "PASS":
            raise StudyBlocked("; ".join(manifest["preflight"]["errors"]))

        current_stage = "tests"
        _run_stage(
            manifest,
            2,
            current_stage,
            "Chạy test, Ruff và compileall",
            _run_quality_gates,
        )

        current_stage = "dataset_integrity"

        def data_gate() -> dict[str, Any]:
            result = validate_dataset_suite(DATASET_SCALES, study_id=STUDY_ID)
            if result["status"] != "PASS":
                raise StudyBlocked("dataset integrity gate failed")
            write_json(OUTPUT_ROOT / "config" / "dataset_integrity.json", result)
            return result

        manifest["dataset_integrity"] = _run_stage(
            manifest,
            3,
            current_stage,
            "Kiểm tra checksum, event order, temporal split và replay",
            data_gate,
        )

        current_stage = "calibration_workload"
        manifest["calibration_workload"] = _run_stage(
            manifest,
            4,
            current_stage,
            f"Kiểm tra hoặc sinh {len(DATASET_SCALES) * CALIBRATION_PER_DATASET} calibration query với seed 42",
            _calibration_manifest,
        )
        generated = _identity()
        if any(generated[k] != identity[k] for k in identity if k != "calibration_workload_sha256"):
            raise StudyBlocked("non-workload input changed during generation")
        identity = generated
        manifest["identity"] = identity
        _save(manifest)

        current_stage = "calibration_materialization"

        def materialize_calibration() -> dict[str, Any]:
            result = export_split("calibration")
            gate = validate_calibration(
                OUTPUT_ROOT / "cache" / "materialized" / "calibration",
                result_root=OUTPUT_ROOT,
                study_id=STUDY_ID,
            )
            if gate["status"] != "PASS":
                raise StudyBlocked("invalid calibration payload inputs: " + "; ".join(gate["errors"]))
            for warning in gate["warnings"]:
                LOGGER.warning("calibration diagnostic (non-blocking): %s", warning)
            return result

        manifest["calibration_materialization"] = _run_stage(
            manifest,
            5,
            current_stage,
            "Materialize các graph version low, medium, high",
            materialize_calibration,
        )

        current_stage = "simulator_build"

        def build() -> dict[str, Any]:
            runtime = freeze_reference_model(
                OUTPUT_ROOT / "cache" / "materialized" / "calibration",
                OUTPUT_ROOT / "config" / "runtime_calibration.json",
                study_id=STUDY_ID,
            )
            controller._run(
                controller._wsl_script(
                    PROJECT_ROOT / "simulator" / "omnetpp" / "tools" / "build.sh"
                ),
                "build.log",
            )
            return runtime

        manifest["runtime_calibration"] = _run_stage(
            manifest,
            6,
            current_stage,
            "Freeze mô hình CPU/I/O tham chiếu và build GraphQueryApp",
            build,
        )

        current_stage = "forced_calibration"
        samples = _run_stage(
            manifest,
            7,
            current_stage,
            "Chạy cưỡng bức ba physical plan trên calibration",
            controller._forced_calibration,
        )
        if samples is None:
            samples = _read_jsonl(OUTPUT_ROOT / "calibration" / "forced_plan_wire.jsonl")

        current_stage = "method_training"

        def train() -> dict[str, Any]:
            linucb = train_linucb(
                samples, OUTPUT_ROOT / "policies", seed_component="policy_bootstrap"
            )
            calibrated = train_calibrated_methods(samples, OUTPUT_ROOT / "policies")
            if linucb["initial_plans"] != calibrated["initial_plans"]:
                raise StudyBlocked("methods received different initial plans")
            return {"linucb": linucb, "calibrated": calibrated}

        trained = _run_stage(
            manifest,
            8,
            current_stage,
            "Fit và freeze SWITCH_COST_THRESHOLD, switching-cost model và LINUCB",
            train,
        )
        if trained is None:
            training = json.loads(
                (OUTPUT_ROOT / "policies" / "training.json").read_text(encoding="utf-8")
            )
            baseline = json.loads(
                (OUTPUT_ROOT / "policies" / "baseline_training.json").read_text(encoding="utf-8")
            )
            trained = {"linucb": training, "calibrated": baseline}
        controller._verify_frozen_method_artifacts()

        current_stage = "freeze_evaluation"

        def freeze() -> dict[str, Any]:
            if EVALUATION_MANIFEST.exists():
                return validate_frozen_evaluation_workload()
            return freeze_evaluation_workload()

        manifest["evaluation_workload"] = _run_stage(
            manifest,
            9,
            current_stage,
            f"Freeze {len(DATASET_SCALES) * EVALUATION_PER_DATASET} evaluation query và checksum",
            freeze,
        )
        validate_frozen_evaluation_workload()
        manifest.setdefault("frozen_identity", identity)
        if manifest["frozen_identity"] != _identity():
            raise StudyBlocked("code/config/calibration checksum changed after evaluation freeze")
        _save(manifest)

        current_stage = "evaluation_materialization"

        def materialize_evaluation() -> dict[str, Any]:
            result = export_split("evaluation")
            controller._write_initial_plans(trained["linucb"]["initial_plans"])
            return result

        evaluation_materialization = _run_stage(
            manifest,
            10,
            current_stage,
            "Replay và materialize evaluation đúng graph_version",
            materialize_evaluation,
        )
        if evaluation_materialization is None:
            export_split("evaluation")
            controller._verify_frozen_method_artifacts()

        current_stage = "forced_evaluation"
        forced = _run_stage(
            manifest,
            11,
            current_stage,
            f"Chạy {len(DATASET_SCALES) * 576} forced-plan execution để tính oracle regret",
            controller._forced_evaluation_costs,
        )
        if manifest["frozen_identity"] != _identity():
            raise StudyBlocked("frozen implementation changed before main evaluation")
        if forced is None:
            forced_rows = _read_jsonl(OUTPUT_ROOT / "metrics" / "forced_evaluation_plans.jsonl")
            forced = {}
            for row in forced_rows:
                forced.setdefault(row["query_id"], {})[int(row["forced_plan_id"])] = int(
                    row["total_wire_bytes"]
                )

        current_stage = "main_evaluation"
        rows = _run_stage(
            manifest,
            12,
            current_stage,
            "Chạy lần lượt bảy phương pháp trên scale LDBC hiện hành",
            lambda: controller._run_evaluation(forced),
        )
        if rows is None:
            rows = _read_jsonl(OUTPUT_ROOT / "metrics" / "per_query.jsonl")

        current_stage = "scalability"
        scalability = _run_stage(
            manifest,
            13,
            current_stage,
            "Chạy scalability: worker count và network profile",
            controller._run_scalability,
        )
        if scalability is not None:
            manifest["scalability_query_runs"] = len(scalability)

        current_stage = "stress"
        manifest["stress_suite"] = _run_stage(
            manifest,
            14,
            current_stage,
            "Chạy retry, duplicate, cancel và permanent-failure stress suite",
            controller._run_fault_stress,
        )
        if manifest["frozen_identity"] != _identity():
            raise StudyBlocked("frozen implementation changed during evaluation")

        current_stage = "correctness"

        def correctness() -> dict[str, Any]:
            expected = load_study_config()["evaluation_design"]["expected_method_query_runs"]
            failed = [row["query_id"] for row in rows if row.get("status") != "PASS"]
            if len(rows) != expected or failed:
                raise StudyBlocked(
                    f"correctness/count gate failed: rows={len(rows)}, failures={len(failed)}"
                )
            return {"status": "PASS", "rows": len(rows), "oracle_matches": len(rows)}

        manifest["correctness"] = _run_stage(
            manifest,
            15,
            current_stage,
            "Xác minh 100% successful query khớp oracle",
            correctness,
            always_verify=True,
        )

        current_stage = "analysis_report"
        _stage(16, "Bootstrap, Holm, practical equivalence và sinh báo cáo")
        analysis = analyze_results(rows)
        ablation = analyze_safebenefit_ablation(rows)
        write_json(OUTPUT_ROOT / "metrics" / "safebenefit_ablation.json", ablation)
        sensitivity = analyze_parameter_sensitivity(rows)
        write_json(
            OUTPUT_ROOT / "metrics" / "safebenefit_parameter_sensitivity.json",
            sensitivity,
        )
        analysis["safebenefit_ablation"] = ablation
        cost_diagnostics = analyze_cost_model(rows)
        write_json(
            OUTPUT_ROOT / "metrics" / "cost_model_diagnostics.json",
            cost_diagnostics,
        )
        analysis["cost_model_diagnostics"] = cost_diagnostics
        analysis["parameter_sensitivity"] = {
            "configurations": len(sensitivity["configurations"]),
            "interpretation": sensitivity["interpretation"],
        }
        write_json(OUTPUT_ROOT / "metrics" / "statistics.json", analysis)
        manifest["status"] = "PASS"
        generate_report(manifest, analysis)
        _complete(manifest, current_stage)
        return 0
    except Exception as error:
        manifest["status"] = (
            "BLOCKED" if isinstance(error, (StudyBlocked, controller.StudyBlocked)) else "FAIL"
        )
        manifest["failed_stage"] = current_stage
        manifest["error"] = str(error)
        manifest["traceback"] = traceback.format_exc()
        _save(manifest)
        generate_report(manifest, analysis)
        LOGGER.exception("workflow stopped at %s", current_stage)
        print(f"Pipeline dừng tại {current_stage}: {error}", file=sys.stderr)
        return 2 if manifest["status"] == "BLOCKED" else 1


def main(arguments: list[str] | None = None) -> int:
    from src.common.design import ACTIVE_DATASET

    if ACTIVE_DATASET is None:
        from src.experiment.coordinator import main as coordinate

        return coordinate(arguments)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight", action="store_true", help="only validate environment and data"
    )
    parser.add_argument("--fresh", action="store_true", help="safely replace output/current")
    options = parser.parse_args(arguments)
    if options.fresh:
        raise StudyBlocked("--fresh is only supported by the parent coordinator")
    if options.preflight:
        result = preflight()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "PASS" else 2
    try:
        return run_full()
    except StudyBlocked as error:
        print(f"Pipeline blocked before execution: {error}", file=sys.stderr)
        return 2
