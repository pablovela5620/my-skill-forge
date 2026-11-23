"""Tests for .github/scripts/update.sh, run against fake gh and rattler-build commands."""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
UPDATE_SH: Path = REPO_ROOT / ".github" / "scripts" / "update.sh"

# Answers `gh api <endpoint> ...` from a JSON map; an unknown endpoint is a 404.
FAKE_GH: str = """#!/usr/bin/env python3
import json, os, sys
responses = json.loads(os.environ["FAKE_GH_RESPONSES"])
if sys.argv[2] not in responses:
    sys.exit(1)
print(responses[sys.argv[2]])
"""

# Stands in for `rattler-build bump-recipe` finding release 2.0.0.
FAKE_RATTLER_BUILD: str = """#!/usr/bin/env bash
yq -i '.context.version = "2.0.0"' "${@: -1}"
"""

GIT_RECIPE: str = """\
context:
  skill: example
  upstream_path: skills/example

package:
  name: agent-skill-${{ skill }}
  version: "0.0.4"

source:
  git: https://github.com/owner/repo
  rev: old-commit
"""

RELEASE_RECIPE: str = """\
context:
  skill: example
  version: "1.0.0"

package:
  name: agent-skill-${{ skill }}
  version: ${{ version }}
"""


def run_update(tmp_path: Path, recipe: str, strategy: str, gh_responses: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Run update.sh in a scratch repo; return the new recipe text and the GITHUB_OUTPUT values."""
    bin_dir: Path = tmp_path / "bin"
    bin_dir.mkdir()
    for name, script in {"gh": FAKE_GH, "rattler-build": FAKE_RATTLER_BUILD}.items():
        fake: Path = bin_dir / name
        fake.write_text(script)
        fake.chmod(0o755)
    recipe_file: Path = tmp_path / "recipes" / "example" / "recipe.yaml"
    recipe_file.parent.mkdir(parents=True)
    recipe_file.write_text(recipe)
    output_file: Path = tmp_path / "github_output"
    env: dict[str, str] = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GH_RESPONSES": json.dumps(gh_responses),
        "GITHUB_OUTPUT": str(output_file),
    }
    subprocess.run(["bash", str(UPDATE_SH), "example", strategy], cwd=tmp_path, env=env, check=True)
    outputs: dict[str, str] = dict(line.split("=", 1) for line in output_file.read_text().splitlines())
    return recipe_file.read_text(), outputs


def test_git_main_follows_newest_commit_touching_upstream_path(tmp_path: Path) -> None:
    recipe, outputs = run_update(
        tmp_path,
        GIT_RECIPE,
        "git-main",
        {
            "repos/owner/repo/commits?path=skills/example&per_page=10": "path-commit",
            "repos/owner/repo/contents/skills/example?ref=path-commit": "",
            "repos/owner/repo/commits/main": "unrelated-head",
        },
    )
    assert outputs == {"old-version": "old-commit", "new-version": "path-commit"}
    assert GIT_RECIPE.replace("old-commit", "path-commit").replace('"0.0.4"', '"0.0.5"') == recipe


def test_git_main_skips_a_commit_that_removed_the_path(tmp_path: Path) -> None:
    _, outputs = run_update(
        tmp_path,
        GIT_RECIPE,
        "git-main",
        {
            "repos/owner/repo/commits?path=skills/example&per_page=10": "removal-commit",
            "repos/owner/repo/commits/removal-commit": "before-removal",
            "repos/owner/repo/contents/skills/example?ref=before-removal": "",
        },
    )
    assert outputs["new-version"] == "before-removal"


def test_git_main_leaves_an_up_to_date_recipe_untouched(tmp_path: Path) -> None:
    recipe, outputs = run_update(
        tmp_path,
        GIT_RECIPE,
        "git-main",
        {
            "repos/owner/repo/commits?path=skills/example&per_page=10": "old-commit",
            "repos/owner/repo/contents/skills/example?ref=old-commit": "",
        },
    )
    assert outputs == {"old-version": "old-commit", "new-version": "old-commit"}
    assert recipe == GIT_RECIPE


def test_github_latest_release_reports_the_version_bump_recipe_wrote(tmp_path: Path) -> None:
    _, outputs = run_update(tmp_path, RELEASE_RECIPE, "github-latest-release", {})
    assert outputs == {"old-version": "1.0.0", "new-version": "2.0.0"}


def test_unknown_strategy_fails(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        run_update(tmp_path, GIT_RECIPE, "yolo", {})
