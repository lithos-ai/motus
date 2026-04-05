import os
from collections.abc import Iterator
from pathlib import Path

import pathspec

# Patterns excluded from deploy bundles that aren't always in .gitignore.
# Hidden files (.env, .git/, etc.) are already skipped separately.
DEFAULT_ADDITIONAL_IGNORES = [
    "__pycache__/",
    "*.pyc",
    "*venv*/",
    "*.egg-info/",
    "dist/",
    "build/",
    "htmlcov/",
]


def walk(
    project_path: Path,
    additional_ignores: list[str] = DEFAULT_ADDITIONAL_IGNORES,
) -> Iterator[Path]:
    """Walk a project directory and yield file paths not excluded by ignore rules.

    Applies a three-layer ignore strategy: dotfiles are always skipped,
    then ``additional_ignores`` patterns are checked, and finally any
    ``.gitignore`` files found in the tree are respected.

    Args:
        project_path: Root directory of the project to walk.
        additional_ignores: Extra glob patterns to exclude beyond
            ``.gitignore``. Defaults to ``DEFAULT_ADDITIONAL_IGNORES``.

    Yields:
        Paths relative to *project_path* for every non-ignored file.
    """
    root_extra = pathspec.PathSpec.from_lines("gitwildmatch", additional_ignores)
    gitignore_specs: dict[Path, pathspec.PathSpec] = {}
    project_path = project_path.resolve()

    # Parse root .gitignore if it exists
    root_gitignore = project_path / ".gitignore"
    if root_gitignore.is_file():
        gitignore_specs[project_path] = pathspec.PathSpec.from_lines(
            "gitwildmatch", root_gitignore.read_text().splitlines()
        )

    # Three-layer ignore: dotfiles first, then additional_ignores, then per-directory .gitignore specs.
    def _is_ignored(abs_path: Path, is_dir: bool) -> bool:
        if abs_path.name.startswith("."):
            return True
        rel_path = str(abs_path.relative_to(project_path))
        check_path = rel_path + "/" if is_dir else rel_path
        if root_extra.match_file(check_path):
            return True
        # Check each .gitignore relative to its own directory so nested rules scope correctly.
        for spec_dir, spec in gitignore_specs.items():
            try:
                spec_rel = str(abs_path.relative_to(spec_dir))
            except ValueError:
                continue
            spec_check = spec_rel + "/" if is_dir else spec_rel
            if spec.match_file(spec_check):
                return True
        return False

    for dirpath, dirnames, filenames in os.walk(project_path, topdown=True):
        dirpath = Path(dirpath)

        # Parse .gitignore in this directory (if not already parsed as root)
        gitignore_file = dirpath / ".gitignore"
        if gitignore_file.is_file() and dirpath not in gitignore_specs:
            gitignore_specs[dirpath] = pathspec.PathSpec.from_lines(
                "gitwildmatch", gitignore_file.read_text().splitlines()
            )

        # Prune ignored directories in-place
        dirnames[:] = [d for d in dirnames if not _is_ignored(dirpath / d, True)]

        # Yield non-ignored files
        for filename in filenames:
            abs_file = dirpath / filename
            if not _is_ignored(abs_file, False):
                yield abs_file.relative_to(project_path)
