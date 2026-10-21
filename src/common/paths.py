"""Canonical, project-root-relative paths for the SafeBenefit study."""

from pathlib import Path

from src.common.design import ACTIVE_DATASET, DATASET_ALIASES

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    """Resolve an explicit absolute path or a project-relative path."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


DATA_ROOT = PROJECT_ROOT / "data"
STUDY_OUTPUT_ROOT = PROJECT_ROOT / "output" / "current"
OUTPUT_ROOT = (
    (STUDY_OUTPUT_ROOT / DATASET_ALIASES[ACTIVE_DATASET]) if ACTIVE_DATASET else STUDY_OUTPUT_ROOT
)
WORKLOAD_ROOT = OUTPUT_ROOT / "cache" / "workloads"
SIMULATOR_ROOT = PROJECT_ROOT / "simulator" / "omnetpp"
RESULT_ROOT = OUTPUT_ROOT
EXPERIMENT_ROOT = OUTPUT_ROOT
ARTIFACT_ROOT = OUTPUT_ROOT / "cache"
SIMULATION_RESULT_ROOT = OUTPUT_ROOT / "simulation"
REPORT_ROOT = OUTPUT_ROOT / "report"
CONFIG_ROOT = PROJECT_ROOT / "config"
