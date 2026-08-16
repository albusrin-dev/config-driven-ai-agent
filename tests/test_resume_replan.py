"""A3: resume never trusts anything blessed at suspend time — the pending
action re-runs the full pipeline (fresh plan_effects -> gate -> enforce)
against current world state."""

import os
import shutil
import subprocess

import pytest

from conftest import make_agent, make_system

from config.models import Autonomy
from core.loop import Completed, Suspended, resume, run_turn
from core.session import PendingAction, ToolCallRecord, new_session
from testing.fake_llm import FakeLLM, call, text_response, tool_response
from tools.registry import ToolRegistry
from tools.builtins.files import ReadFileTool, WriteFileTool


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(ReadFileTool())
    r.register(WriteFileTool())
    return r


def test_pending_action_stores_nothing_blessed():
    """Structural proof: a suspended action records only the raw call and a
    human-readable reason — no effect, no resolved path, nothing that COULD
    be reused stale at resume time."""
    assert set(PendingAction.model_fields) == {"call", "reason", "remaining"}
    assert set(ToolCallRecord.model_fields) == {"id", "name", "params"}


def _suspend_write_under_sub(registry, sandbox_root):
    """Suspend a supervised write to sandbox/sub/w.txt while 'sub' is a
    clean real directory."""
    sub = sandbox_root / "sub"
    sub.mkdir()
    agent = make_agent(allowlist=["write_file"], autonomy=Autonomy.SUPERVISED)
    system = make_system(fs_root=sandbox_root)
    llm = FakeLLM([
        tool_response(call("write_file",
                           {"path": str(sub / "w.txt"), "content": "attack"})),
        text_response("Understood — the write was refused."),
    ])
    session = new_session("tester")
    result = run_turn(session, "write it", llm, registry, agent, system)
    assert isinstance(result, Suspended)
    return session, llm, agent, system, sub


def _assert_resume_denied_on_new_state(session, llm, agent, system, registry, victim_dir):
    result = resume(session, True, llm, registry, agent, system)
    assert isinstance(result, Completed)  # denial fed back, model concluded
    refusal = [m for m in session.conversation if m["role"] == "tool_result"][0]
    assert not refusal["ok"]
    assert "outside sandbox root" in refusal["content"]
    assert session.budget.tool_calls_made == 0  # nothing executed
    assert list(victim_dir.iterdir()) == []  # nothing written outside


@pytest.mark.skipif(os.name != "nt", reason="junction variant is Windows-specific")
def test_resume_replans_against_junction_swap(tmp_path, registry):
    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    session, llm, agent, system, sub = _suspend_write_under_sub(registry, sandbox_root)

    # During the human-time gap: the directory component is swapped for a
    # junction escaping the sandbox. The path now RESOLVES differently.
    shutil.rmtree(sub)
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(sub), str(outside)],
        capture_output=True,
    ).returncode == 0
    if not made:
        pytest.skip("cannot create a directory junction on this system")

    _assert_resume_denied_on_new_state(session, llm, agent, system, registry, outside)


def test_resume_replans_against_symlink_swap(tmp_path, registry):
    """POSIX variant (runs wherever symlinks are permitted, e.g. Linux)."""
    probe_target = tmp_path / "probe-t"
    probe_target.mkdir()
    try:
        os.symlink(probe_target, tmp_path / "probe-l")
    except OSError:
        pytest.skip("symlink creation not permitted on this system")

    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    session, llm, agent, system, sub = _suspend_write_under_sub(registry, sandbox_root)

    sub.rmdir()
    os.symlink(outside, sub)

    _assert_resume_denied_on_new_state(session, llm, agent, system, registry, outside)


def test_resume_still_executes_when_state_is_unchanged(tmp_path, registry):
    """Sanity: re-planning does not break the honest case."""
    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir()
    agent = make_agent(allowlist=["write_file"], autonomy=Autonomy.SUPERVISED)
    system = make_system(fs_root=sandbox_root)
    target = sandbox_root / "fine.txt"
    llm = FakeLLM([
        tool_response(call("write_file", {"path": str(target), "content": "ok"})),
        text_response("Done."),
    ])
    session = new_session("tester")
    assert isinstance(run_turn(session, "write", llm, registry, agent, system), Suspended)
    result = resume(session, True, llm, registry, agent, system)
    assert isinstance(result, Completed)
    assert target.read_text(encoding="utf-8") == "ok"
