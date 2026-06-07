"""DVC data versioning utilities for reproducible dataset management."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def _run_dvc(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a DVC CLI command.

    Args:
        args: DVC command arguments.
        cwd: Working directory.

    Returns:
        Completed subprocess result.

    Raises:
        subprocess.CalledProcessError: When DVC command fails.
    """
    cmd = ["dvc", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd)


def _run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git CLI command.

    Args:
        args: Git command arguments.
        cwd: Working directory.

    Returns:
        Completed subprocess result.
    """
    cmd = ["git", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=cwd)


def init_dvc_repo(
    data_dir: Path,
    remote_name: str = "localstore",
    remote_url: str | None = None,
) -> None:
    """Initialize DVC in the repository and configure a local remote.

    Idempotent: safe to call when DVC is already initialized.

    Args:
        data_dir: Directory for dataset storage.
        remote_name: DVC remote name.
        remote_url: Optional remote URL (defaults to ``data_dir/.dvc-storage``).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    dvc_dir = Path(".dvc")
    if not dvc_dir.exists():
        _run_dvc(["init", "--no-scm"])
        logger.info("dvc_initialized")

    storage = remote_url or str(data_dir / ".dvc-storage")
    Path(storage).mkdir(parents=True, exist_ok=True)

    remotes = _run_dvc(["remote", "list"]).stdout
    if remote_name not in remotes:
        _run_dvc(["remote", "add", "-d", remote_name, storage])
        logger.info("dvc_remote_configured", remote=remote_name, url=storage)
    else:
        logger.debug("dvc_remote_exists", remote=remote_name)


def add_dataset_version(data_path: Path, tag: str) -> str:
    """Track a dataset with DVC and create a git tag.

    Idempotent: re-adds existing paths safely.

    Args:
        data_path: Path to dataset file or directory.
        tag: Git tag name for this dataset version.

    Returns:
        DVC file hash for reproducibility logging.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    dvc_file = Path(f"{data_path}.dvc")
    if dvc_file.exists():
        _run_dvc(["add", "-f", str(data_path)])
    else:
        _run_dvc(["add", str(data_path)])

    dataset_hash = get_dataset_hash(data_path)
    _run_git(["tag", "-f", tag])
    logger.info("dataset_version_added", path=str(data_path), tag=tag, hash=dataset_hash)
    return dataset_hash


def get_dataset_hash(data_path: Path) -> str:
    """Return the DVC md5 hash for a tracked dataset path.

    Falls back to a content hash when DVC metadata is unavailable.

    Args:
        data_path: Path to dataset file or directory.

    Returns:
        Hex hash string for reproducibility logging.
    """
    dvc_file = Path(f"{data_path}.dvc")
    if dvc_file.exists():
        result = _run_dvc(["status", str(dvc_file), "--cloud"])
        for line in result.stdout.splitlines():
            if "md5:" in line:
                return line.split("md5:")[-1].strip()

        for line in dvc_file.read_text().splitlines():
            if "md5:" in line:
                return line.split("md5:")[-1].strip()

    if data_path.is_file():
        digest = hashlib.md5(data_path.read_bytes(), usedforsecurity=False).hexdigest()
        return digest

    hasher = hashlib.md5(usedforsecurity=False)
    for file in sorted(data_path.rglob("*")):
        if file.is_file():
            hasher.update(file.read_bytes())
    return hasher.hexdigest()
