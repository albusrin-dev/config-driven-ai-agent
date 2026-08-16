"""A1 / TOCTOU: one blessed resolution, threaded from plan_effects through
the gate into execute; use-time re-verification fails closed."""

import os
import shutil
import subprocess

import pytest

from conftest import make_agent, make_system

from config.models import Autonomy
from core.base import ToolContext
from core.effects import FileMode, FilesystemEffect
from core.enforce import Executed, enforce_and_run
from tools.builtins.files import DeleteFileTool, ReadFileTool, WriteFileTool

ALL_TOOLS = ["read_file", "write_file", "delete_file"]


@pytest.fixture
def sandbox_root(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    return root


def _ctx(sandbox_root):
    return ToolContext(
        agent=make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.AUTONOMOUS_BOUNDED),
        system=make_system(fs_root=sandbox_root),
    )


def _can_symlink(tmp_path) -> bool:
    target = tmp_path / "symlink-probe-target"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "symlink-probe-link"
    try:
        os.symlink(target, link)
    except OSError:
        return False
    return True


def test_execute_uses_blessed_path_not_raw_param(tmp_path, sandbox_root):
    """Unit-level proof: given an effect whose path differs from what the
    raw param would re-derive to, execute acts on the effect's path."""
    inside = sandbox_root / "inside.txt"
    inside.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    tool = ReadFileTool()
    # Raw param points OUTSIDE the sandbox; the blessed effect points inside.
    params = tool.input_schema.model_validate({"path": str(outside)})
    blessed = [FilesystemEffect(path=str(inside), mode=FileMode.READ)]
    result = tool.execute(params, _ctx(sandbox_root), blessed)
    assert result.ok
    assert result.output == "inside"  # a raw-param re-derive would say "outside"


def test_blessed_path_threads_through_enforce(sandbox_root):
    """Integration: relative param -> plan_effects resolves once against
    fs_root -> gate confines -> execute writes exactly there."""
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    system = make_system(fs_root=sandbox_root)
    outcome = enforce_and_run(
        WriteFileTool(), {"path": "sub/../note.txt", "content": "hi"}, agent, system
    )
    assert isinstance(outcome, Executed) and outcome.result.ok
    assert (sandbox_root / "note.txt").read_text(encoding="utf-8") == "hi"


def test_symlink_swap_after_check_fails_closed(tmp_path, sandbox_root):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation not permitted on this system")
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")

    tool = WriteFileTool()
    blessed_target = sandbox_root / "w.txt"
    params = tool.input_schema.model_validate(
        {"path": str(blessed_target), "content": "overwrite"}
    )
    ctx = _ctx(sandbox_root)
    effects = tool.plan_effects(params, ctx)  # blessed while w.txt is clean

    # Attacker swaps the blessed path for a symlink escaping the sandbox
    # between check and use.
    os.symlink(victim, blessed_target)

    result = tool.execute(params, ctx, effects)
    assert not result.ok
    assert "PathResolutionChangedError" in result.error
    assert victim.read_text(encoding="utf-8") == "precious"  # untouched


def test_delete_refuses_swapped_symlink(tmp_path, sandbox_root):
    if not _can_symlink(tmp_path):
        pytest.skip("symlink creation not permitted on this system")
    victim = tmp_path / "victim.txt"
    victim.write_text("precious", encoding="utf-8")

    tool = DeleteFileTool()
    doomed = sandbox_root / "doomed.txt"
    doomed.write_text("bye", encoding="utf-8")
    params = tool.input_schema.model_validate({"path": str(doomed)})
    ctx = _ctx(sandbox_root)
    effects = tool.plan_effects(params, ctx)

    doomed.unlink()
    os.symlink(victim, doomed)

    result = tool.execute(params, ctx, effects)
    assert not result.ok
    assert victim.exists()


def _make_junction(link, target) -> bool:
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
    )
    return result.returncode == 0


@pytest.mark.skipif(os.name != "nt", reason="junction variant is Windows-specific")
def test_directory_junction_swap_fails_closed(tmp_path, sandbox_root):
    """Windows variant of the symlink swap (junctions need no privilege):
    a directory component of the blessed path is swapped for a junction
    escaping the sandbox between check and use."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    victim = outside / "w.txt"
    victim.write_text("precious", encoding="utf-8")

    sub = sandbox_root / "sub"
    sub.mkdir()
    tool = WriteFileTool()
    params = tool.input_schema.model_validate(
        {"path": str(sub / "w.txt"), "content": "attack"}
    )
    ctx = _ctx(sandbox_root)
    effects = tool.plan_effects(params, ctx)  # blessed while 'sub' is a real dir

    shutil.rmtree(sub)
    if not _make_junction(sub, outside):
        pytest.skip("cannot create a directory junction on this system")

    result = tool.execute(params, ctx, effects)
    assert not result.ok
    assert "PathResolutionChangedError" in result.error
    assert victim.read_text(encoding="utf-8") == "precious"  # untouched


def test_ordinary_ops_still_work(sandbox_root):
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    system = make_system(fs_root=sandbox_root)
    target = sandbox_root / "cycle.txt"

    out = enforce_and_run(WriteFileTool(), {"path": str(target), "content": "v1"}, agent, system)
    assert isinstance(out, Executed) and out.result.ok

    out = enforce_and_run(ReadFileTool(), {"path": str(target)}, agent, system)
    assert isinstance(out, Executed) and out.result.output == "v1"

    out = enforce_and_run(
        DeleteFileTool(), {"path": str(target)}, agent, system,
        approver=lambda d: True,
    )
    assert isinstance(out, Executed) and out.result.ok
    assert not target.exists()
