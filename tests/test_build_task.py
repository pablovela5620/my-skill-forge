"""The build task finds the recipe of every skill that Autobump bumps."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT: Path = Path(__file__).resolve().parents[1]
MATRIX_SKILLS: list[str] = re.findall(r"^\s+- skill: (\S+)", (ROOT / ".github/workflows/autobump.yml").read_text(encoding="utf-8"), re.M)


def build_recipe_pattern() -> str:
    """Return the `--recipe` argument of the `build` task, with `{{ skill }}` unexpanded."""
    command: str = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))["tasks"]["build"]["cmd"]
    match: re.Match[str] | None = re.search(r"--recipe (\S*\{\{ skill \}\}|\S+)", command)
    assert match is not None, command
    return match.group(1)


def test_autobump_matrix_is_not_empty() -> None:
    """The matrix parse found the skills."""
    assert len(MATRIX_SKILLS) > 10


@pytest.mark.parametrize("skill", MATRIX_SKILLS)
def test_build_task_resolves_one_recipe(skill: str) -> None:
    """`pixi run build <skill>` points at exactly one recipe directory."""
    pattern: str = build_recipe_pattern().replace("{{ skill }}", skill)
    matches: list[Path] = [path for path in ROOT.glob(pattern) if (path / "recipe.yaml").is_file()]
    assert len(matches) == 1, f"{pattern}: {matches}"
