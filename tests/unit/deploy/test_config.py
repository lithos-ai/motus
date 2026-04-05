"""Tests for motus.config.Config."""

import tomllib
from pathlib import Path

import pytest

from motus.config import Config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Return a Config rooted in a temp directory with no motus.toml."""
    monkeypatch.chdir(tmp_path)
    return Config()


@pytest.fixture
def cfg_with_file(tmp_path, monkeypatch):
    """Return a Config with a pre-existing motus.toml."""
    (tmp_path / "motus.toml").write_text('project_id = "my-project"\n')
    monkeypatch.chdir(tmp_path)
    return Config()


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


class TestRead:
    def test_empty_when_no_file(self, cfg):
        assert dict(cfg) == {}

    def test_reads_existing_file(self, cfg_with_file):
        assert cfg_with_file["project_id"] == "my-project"

    def test_get_missing_key_returns_default(self, cfg_with_file):
        assert cfg_with_file.get("missing") is None
        assert cfg_with_file.get("missing", "fallback") == "fallback"


class TestSetItem:
    def test_setitem_creates_file(self, cfg):
        cfg["project_id"] = "new-project"
        assert cfg._path.exists()
        assert _read_toml(cfg._path)["project_id"] == "new-project"

    def test_setitem_flushes_to_disk(self, cfg_with_file):
        cfg_with_file["build_id"] = "b_123"
        on_disk = _read_toml(cfg_with_file._path)
        assert on_disk["project_id"] == "my-project"
        assert on_disk["build_id"] == "b_123"


class TestDelItem:
    def test_delitem_removes_key_and_flushes(self, cfg_with_file):
        cfg_with_file["build_id"] = "b_123"
        del cfg_with_file["build_id"]
        assert "build_id" not in cfg_with_file
        assert "build_id" not in _read_toml(cfg_with_file._path)

    def test_delitem_missing_key_raises(self, cfg_with_file):
        with pytest.raises(KeyError):
            del cfg_with_file["nonexistent"]


class TestUpdate:
    def test_batch_update_flushes_once(self, cfg_with_file):
        cfg_with_file.update(build_id="b_456", import_path="app:server")
        on_disk = _read_toml(cfg_with_file._path)
        assert on_disk["build_id"] == "b_456"
        assert on_disk["import_path"] == "app:server"
        assert on_disk["project_id"] == "my-project"

    def test_update_with_dict(self, cfg_with_file):
        cfg_with_file.update({"build_id": "b_789"})
        assert _read_toml(cfg_with_file._path)["build_id"] == "b_789"

    def test_empty_update_does_not_flush(self, cfg, tmp_path):
        cfg.update()
        assert not (tmp_path / "motus.toml").exists()


class TestFindWalksUp:
    def test_finds_toml_in_parent(self, tmp_path, monkeypatch):
        (tmp_path / "motus.toml").write_text('project_id = "root"\n')
        child = tmp_path / "a" / "b"
        child.mkdir(parents=True)
        monkeypatch.chdir(child)
        cfg = Config()
        assert cfg["project_id"] == "root"
        assert cfg._path == tmp_path / "motus.toml"
