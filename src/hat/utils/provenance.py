"""
Result provenance and traceability utilities.

This module provides tools for adding provenance metadata to experiment results,
ensuring traceability back to the code, parameters, and environment that produced them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_info() -> Dict[str, str]:
    """Get current git commit hash and status."""
    info = {
        "commit": "unknown",
        "branch": "unknown",
        "dirty": False,
    }
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()[:12]

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            info["dirty"] = bool(result.stdout.strip())

    except Exception:
        pass

    return info


def get_provenance(
    script_path: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate provenance metadata for an experiment result.

    Args:
        script_path: Path to the script that generated this result.
                    If None, uses sys.argv[0].
        args: Dictionary of arguments/parameters used.
        extra_metadata: Additional metadata to include.

    Returns:
        Dictionary with provenance information.

    Example:
        >>> provenance = get_provenance(
        ...     args={"model": "apertus-8b", "layers": [0, 1, 2]},
        ...     extra_metadata={"concept_pack": "sumo-wordnet-v4"}
        ... )
        >>> results = {"accuracy": 0.95, "_provenance": provenance}
    """
    if script_path is None:
        script_path = sys.argv[0]

    git_info = get_git_info()

    provenance = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "script": str(Path(script_path).name),
        "script_path": str(script_path),
        "command_line": " ".join(sys.argv),
        "git": git_info,
        "python_version": sys.version.split()[0],
        "cwd": os.getcwd(),
    }

    if args:
        provenance["args"] = args

    if extra_metadata:
        provenance["metadata"] = extra_metadata

    return provenance


def save_results_with_provenance(
    results: Dict[str, Any],
    output_path: Path,
    args: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    indent: int = 2,
) -> Path:
    """
    Save results to JSON with provenance metadata included.

    Args:
        results: The experiment results to save.
        output_path: Path to save the JSON file.
        args: Dictionary of arguments/parameters used.
        extra_metadata: Additional metadata to include.
        indent: JSON indent level.

    Returns:
        Path to the saved file.

    Example:
        >>> results = {"accuracy": 0.95, "f1": 0.92}
        >>> save_results_with_provenance(
        ...     results,
        ...     Path("results/experiment_1.json"),
        ...     args={"model": "apertus-8b", "epochs": 10}
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add provenance to results
    results_with_provenance = dict(results)
    results_with_provenance["_provenance"] = get_provenance(
        args=args,
        extra_metadata=extra_metadata,
    )

    with open(output_path, "w") as f:
        json.dump(results_with_provenance, f, indent=indent, default=str)

    return output_path


def create_run_directory(
    base_dir: Path,
    experiment_name: str,
    timestamp_format: str = "%Y%m%d_%H%M%S",
) -> Path:
    """
    Create a timestamped run directory for experiment outputs.

    Args:
        base_dir: Base directory (e.g., results/my_experiment)
        experiment_name: Name prefix for the run directory
        timestamp_format: strftime format for timestamp

    Returns:
        Path to the created directory.

    Example:
        >>> run_dir = create_run_directory(
        ...     Path("results/steering_eval"),
        ...     "phase_6"
        ... )
        >>> # Returns: results/steering_eval/phase_6_20251208_143052
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime(timestamp_format)
    run_dir = base_dir / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_manifest(
    run_dir: Path,
    args: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a manifest file to a run directory with full provenance.

    This should be called at the start of an experiment to record
    the full configuration before any results are generated.

    Args:
        run_dir: Directory for this run
        args: Dictionary of arguments/parameters used
        extra_metadata: Additional metadata to include

    Returns:
        Path to the manifest file.
    """
    manifest_path = run_dir / "manifest.json"

    manifest = {
        "provenance": get_provenance(args=args, extra_metadata=extra_metadata),
        "status": "running",
        "started_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest_path


def update_run_manifest(
    run_dir: Path,
    status: str = "completed",
    summary: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Update the manifest file with completion status and summary.

    Args:
        run_dir: Directory for this run
        status: Final status (completed, failed, etc.)
        summary: Summary statistics or key results

    Returns:
        Path to the updated manifest file.
    """
    manifest_path = run_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"provenance": get_provenance()}

    manifest["status"] = status
    manifest["completed_at"] = datetime.utcnow().isoformat() + "Z"

    if "started_at" in manifest:
        started = datetime.fromisoformat(manifest["started_at"].replace("Z", ""))
        completed = datetime.fromisoformat(manifest["completed_at"].replace("Z", ""))
        manifest["duration_seconds"] = (completed - started).total_seconds()

    if summary:
        manifest["summary"] = summary

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest_path


__all__ = [
    "get_git_info",
    "get_provenance",
    "save_results_with_provenance",
    "create_run_directory",
    "write_run_manifest",
    "update_run_manifest",
]
