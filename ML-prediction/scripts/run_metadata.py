"""
Shared experiment metadata helpers for ML-prediction scripts.
Writes JSON with timestamp, seed, paths, and protocol fields for reproducibility.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def try_git_revision(repo_root: Path | None = None) -> str:
    """Return short git SHA if available, else empty string."""
    cwd = repo_root if repo_root is not None else Path.cwd()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ""


def write_experiment_json(
    path: Path,
    payload: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> None:
    """
    Merge standard fields (iso timestamp UTC, git sha) and write UTF-8 JSON.
    """
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": try_git_revision(repo_root),
        **payload,
    }
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
