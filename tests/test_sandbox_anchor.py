"""A4: a relative fs_root anchors to the config file's directory, not CWD."""

import os
from pathlib import Path

from conftest import write_yaml

from config import load_system_config

SYSTEM_YAML = """
providers:
  local:
    endpoint: http://localhost:1
sandbox:
  fs_root: ws
"""


def test_relative_fs_root_is_cwd_independent(tmp_path, monkeypatch):
    write_yaml(tmp_path, "anchored.yaml", SYSTEM_YAML)
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()

    monkeypatch.chdir(tmp_path)
    root_from_here = load_system_config("anchored", base_dir=tmp_path).sandbox.fs_root

    monkeypatch.chdir(other_cwd)
    root_from_there = load_system_config("anchored", base_dir=tmp_path).sandbox.fs_root

    assert root_from_here == root_from_there
    assert Path(root_from_here).is_absolute()
    # Anchored to the config file's directory, not either CWD.
    assert Path(root_from_here) == Path(os.path.realpath(tmp_path / "ws"))


def test_relative_fs_root_with_parent_traversal_anchors(tmp_path):
    """The dev.yaml pattern: ../../workspace relative to the config dir."""
    config_dir = tmp_path / "profiles" / "system"
    config_dir.mkdir(parents=True)
    write_yaml(config_dir, "deep.yaml", SYSTEM_YAML.replace("fs_root: ws", "fs_root: ../../workspace"))
    system = load_system_config("deep", base_dir=config_dir)
    assert Path(system.sandbox.fs_root) == Path(os.path.realpath(tmp_path / "workspace"))


def test_absolute_fs_root_untouched(tmp_path):
    absolute = str(tmp_path / "already-absolute")
    # YAML single-quoted: backslashes are literal, no escaping needed.
    write_yaml(tmp_path, "abs.yaml", SYSTEM_YAML.replace("fs_root: ws", f"fs_root: '{absolute}'"))
    system = load_system_config("abs", base_dir=tmp_path)
    assert system.sandbox.fs_root == absolute


def test_null_fs_root_stays_null(tmp_path):
    write_yaml(tmp_path, "nullroot.yaml", SYSTEM_YAML.replace("  fs_root: ws\n", "  fs_root: null\n"))
    system = load_system_config("nullroot", base_dir=tmp_path)
    assert system.sandbox.fs_root is None


def test_shipped_dev_config_yields_repo_workspace(monkeypatch, tmp_path):
    """The real dev.yaml resolves to <repo>/workspace from any CWD."""
    from conftest import REPO_ROOT

    monkeypatch.chdir(tmp_path)  # somewhere unrelated
    system = load_system_config("dev")
    assert Path(system.sandbox.fs_root) == Path(os.path.realpath(REPO_ROOT / "workspace"))
