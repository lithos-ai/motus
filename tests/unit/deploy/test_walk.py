"""Tests for motus.deploy.walk."""

from pathlib import Path

import pytest

from motus.deploy.walk import walk


@pytest.fixture
def project(tmp_path):
    """Create a minimal project tree for walk tests."""
    (tmp_path / "app.py").write_text("print('hello')")
    (tmp_path / "lib" / "utils.py").parent.mkdir()
    (tmp_path / "lib" / "utils.py").write_text("# utils")
    return tmp_path


def _walked(project_path, **kwargs):
    return sorted(walk(project_path, **kwargs))


class TestBasicWalk:
    def test_yields_regular_files(self, project):
        paths = _walked(project)
        assert Path("app.py") in paths
        assert Path("lib/utils.py") in paths

    def test_skips_dotfiles(self, project):
        (project / ".env").write_text("SECRET=x")
        (project / ".hidden_dir").mkdir()
        (project / ".hidden_dir" / "secret.py").write_text("")
        paths = _walked(project)
        assert Path(".env") not in paths
        assert Path(".hidden_dir/secret.py") not in paths

    def test_skips_pycache(self, project):
        cache_dir = project / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "app.cpython-312.pyc").write_text("")
        paths = _walked(project)
        assert not any("__pycache__" in str(p) for p in paths)

    def test_skips_pyc_files(self, project):
        (project / "app.pyc").write_text("")
        paths = _walked(project)
        assert Path("app.pyc") not in paths


class TestGitignore:
    def test_respects_root_gitignore(self, project):
        (project / ".gitignore").write_text("*.log\nbuild/\n")
        (project / "debug.log").write_text("log")
        (project / "build").mkdir()
        (project / "build" / "out.bin").write_text("")
        paths = _walked(project)
        assert Path("debug.log") not in paths
        assert not any("build" in str(p) for p in paths)

    def test_respects_nested_gitignore(self, project):
        sub = project / "sub"
        sub.mkdir()
        (sub / ".gitignore").write_text("local_only.txt\n")
        (sub / "local_only.txt").write_text("")
        (sub / "keep.txt").write_text("")
        paths = _walked(project)
        assert Path("sub/local_only.txt") not in paths
        assert Path("sub/keep.txt") in paths


class TestAdditionalIgnores:
    def test_default_ignores_venv(self, project):
        venv = project / ".venv"
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text("")
        paths = _walked(project)
        # .venv is a dotfile so skipped regardless, but test myvenv too
        venv2 = project / "myvenv"
        venv2.mkdir()
        (venv2 / "pyvenv.cfg").write_text("")
        paths = _walked(project)
        assert not any("myvenv" in str(p) for p in paths)

    def test_default_ignores_egg_info(self, project):
        egg = project / "pkg.egg-info"
        egg.mkdir()
        (egg / "PKG-INFO").write_text("")
        paths = _walked(project)
        assert not any("egg-info" in str(p) for p in paths)

    def test_custom_additional_ignores(self, project):
        (project / "notes.txt").write_text("notes")
        paths = _walked(project, additional_ignores=["*.txt"])
        assert Path("notes.txt") not in paths
        assert Path("app.py") in paths

    def test_empty_additional_ignores(self, project):
        (project / "dist").mkdir()
        (project / "dist" / "bundle.js").write_text("")
        paths = _walked(project, additional_ignores=[])
        # dist/ is in DEFAULT_ADDITIONAL_IGNORES but not in our empty list
        assert Path("dist/bundle.js") in paths
