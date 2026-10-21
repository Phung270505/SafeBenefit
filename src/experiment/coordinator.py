"""Coordinate independent LDBC studies without pooling models or statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from src.common.cleanup import reset_current_output
from src.common.design import DATASET_ALIASES, STUDY_SCHEMA_VERSION
from src.common.files import sha256_file, write_json
from src.common.paths import PROJECT_ROOT, STUDY_OUTPUT_ROOT


@contextmanager
def study_lock() -> Iterator[None]:
    """Use an OS-held lock; process exit releases it even after interruption."""
    path = PROJECT_ROOT / "output" / ".study.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("another SafeBenefit run holds the study lock") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _fingerprint() -> str:
    paths = [
        PROJECT_ROOT / "config/study.json",
        PROJECT_ROOT / "run.py",
        PROJECT_ROOT / "pyproject.toml",
    ]
    paths.extend((PROJECT_ROOT / "src").rglob("*.py"))
    paths.extend((PROJECT_ROOT / "tests").rglob("*.py"))
    paths.extend((PROJECT_ROOT / "tests/cpp").rglob("*.cc"))
    paths.extend(
        path
        for path in (PROJECT_ROOT / "simulator/omnetpp").rglob("*")
        if path.suffix in {".cc", ".h", ".ned", ".ini", ".sh"}
    )
    payload = "\n".join(
        f"{path.relative_to(PROJECT_ROOT)}:{sha256_file(path)}" for path in sorted(paths)
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _active_child_running() -> bool:
    path = STUDY_OUTPUT_ROOT / "run_manifest.json"
    if not path.is_file():
        return False
    pid = json.loads(path.read_text(encoding="utf-8")).get("active_child_pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        # Query status only; never signal/terminate a Windows process.
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 5:
                raise RuntimeError("cannot verify child process: access denied")
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
                raise RuntimeError("cannot verify active child process")
            return code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _run_all() -> int:
    root = STUDY_OUTPUT_ROOT
    manifest_path = root / "run_manifest.json"
    identity = _fingerprint()
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("schema_version") != STUDY_SCHEMA_VERSION or previous.get("code_config_sha256") != identity:
            raise RuntimeError(
                "output/current is incompatible with the independent LDBC design; use --fresh"
            )
    elif root.exists() and any(path.name != ".gitkeep" for path in root.iterdir()):
        raise RuntimeError("output/current contains unregistered artifacts; use --fresh")
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    manifest = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": "SAFEBENEFIT-DGQ",
        "status": "RUNNING",
        "code_config_sha256": identity,
        "master_seed": 42,
        "scope": "three independent dataset studies; no pooled ranking",
        "datasets": {},
    }
    write_json(manifest_path, manifest)
    for dataset, alias in DATASET_ALIASES.items():
        print(
            f"[dataset] {alias}: calibration -> train -> evaluation -> scalability -> stress -> report",
            flush=True,
        )
        environment = dict(
            os.environ,
            SAFEBENEFIT_DATASET_ID=dataset,
            PYTHONHASHSEED="42",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
        )
        log = root / "logs" / f"{alias}.log"
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                [sys.executable, "run.py"],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            manifest["active_child_pid"] = process.pid
            write_json(manifest_path, manifest)
            try:
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    print(f"[{alias}] {line.rstrip()}", flush=True)
                code = process.wait()
            except BaseException:
                # Do not terminate user processes; retain evidence and release parent lock.
                manifest["status"] = "INTERRUPTED"
                write_json(manifest_path, manifest)
                raise
        manifest.pop("active_child_pid", None)
        if _fingerprint() != identity:
            manifest["status"] = "FAIL"
            write_json(manifest_path, manifest)
            raise RuntimeError("code/config changed during execution")
        child_manifest = root / alias / "run_manifest.json"
        child = (
            json.loads(child_manifest.read_text(encoding="utf-8"))
            if child_manifest.is_file()
            else {}
        )
        manifest["datasets"][dataset] = {
            "status": "PASS" if code == 0 and child.get("status") == "PASS" else "FAIL",
            "exit_code": code,
            "manifest": f"{alias}/run_manifest.json",
            "report": f"{alias}/report/final_results.md",
            "log": log.relative_to(root).as_posix(),
        }
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
    manifest["status"] = (
        "PASS" if all(row["status"] == "PASS" for row in manifest["datasets"].values()) else "FAIL"
    )
    write_json(manifest_path, manifest)
    return 0 if manifest["status"] == "PASS" else 2


def main(arguments: list[str] | None = None) -> int:
    """Run the entire study; never silently delete incompatible old output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(arguments)
    if args.preflight:
        from src.experiment.runner import preflight

        result = preflight()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "PASS" else 2
    try:
        with study_lock():
            if _active_child_running():
                raise RuntimeError(
                    "a dataset worker from an interrupted run is still active; stop it before rerunning"
                )
            if args.fresh:
                reset_current_output(PROJECT_ROOT, STUDY_OUTPUT_ROOT)
            return _run_all()
    except (RuntimeError, OSError, ValueError) as error:
        print(f"Pipeline blocked: {error}", file=sys.stderr)
        return 2
