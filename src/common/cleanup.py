"""Narrow cleanup helpers for generated study output."""

from __future__ import annotations

import shutil
import stat
from pathlib import Path

PROTECTED_NAMES = {"data", "src", "simulator", "tests", "docs", ".git"}


def checked_descendant(path: Path, root: Path) -> Path:
    """Resolve a deletion target and prove that it is below the project root."""
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=False)
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"cleanup target escapes project root: {resolved}")
    relative = resolved.relative_to(resolved_root)
    if relative.parts[0] in PROTECTED_NAMES:
        raise ValueError(f"cleanup target is protected: {resolved}")
    return resolved


def reset_current_output(project_root: Path, current: Path) -> Path:
    """Delete and recreate only project-local output/current."""
    expected = (project_root / "output" / "current").resolve(strict=False)
    resolved = current.resolve(strict=False)
    if resolved != expected:
        raise ValueError(f"only output/current may be reset, got: {resolved}")
    checked_descendant(resolved, project_root)
    if resolved.exists():
        def retry_readonly(function, path, error):
            target = Path(path).resolve()
            if not target.is_relative_to(resolved):
                raise ValueError("cleanup retry escapes output root")
            if not isinstance(error, PermissionError):
                raise error
            target.chmod(target.stat().st_mode | stat.S_IWRITE)
            function(path)
        shutil.rmtree(resolved, onexc=retry_readonly)
    resolved.mkdir(parents=True)
    return resolved
