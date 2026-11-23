"""Tests for the upload step of .github/workflows/package.yml, run with a fake rattler-build."""

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
PACKAGE_WORKFLOW: Path = REPO_ROOT / ".github" / "workflows" / "package.yml"

RunUpload: TypeAlias = Callable[[], subprocess.CompletedProcess[str]]

# Records the package path (last argument) of each `rattler-build upload` call.
FAKE_RATTLER_BUILD: str = """#!/usr/bin/env bash
printf '%s\\n' "${@: -1}" >> "$UPLOAD_LOG"
"""


@pytest.fixture
def run_upload(tmp_path: Path) -> RunUpload:
    """Return a callable that runs the workflow's upload script in tmp_path."""
    script: str = subprocess.run(
        ["yq", "-r", '.jobs.upload.steps[] | select(.name == "Upload packages to prefix.dev") | .run', str(PACKAGE_WORKFLOW)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert script.strip(), "upload step not found in package.yml"
    bin_dir: Path = tmp_path / "bin"
    bin_dir.mkdir()
    fake: Path = bin_dir / "rattler-build"
    fake.write_text(FAKE_RATTLER_BUILD)
    fake.chmod(0o755)
    env: dict[str, str] = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "UPLOAD_LOG": str(tmp_path / "uploads.txt"),
    }

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(["bash", "-e", "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)

    return run


def test_uploads_flat_and_nested_artifacts(tmp_path: Path, run_upload: RunUpload) -> None:
    (tmp_path / "output" / "noarch").mkdir(parents=True)
    (tmp_path / "output" / "flat.conda").touch()
    (tmp_path / "output" / "noarch" / "nested.conda").touch()
    assert run_upload().returncode == 0
    uploads: list[str] = sorted((tmp_path / "uploads.txt").read_text().splitlines())
    assert uploads == ["output/flat.conda", "output/noarch/nested.conda"]


def test_empty_artifact_download_fails(tmp_path: Path, run_upload: RunUpload) -> None:
    (tmp_path / "output").mkdir()
    assert run_upload().returncode != 0
