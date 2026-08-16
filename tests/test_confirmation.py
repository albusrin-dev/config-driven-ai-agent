"""Durable Rule 9: interactive approve/deny, headless suspend, resume."""

import pytest

from conftest import make_agent, make_system

from config.models import Autonomy
from core.loop import Completed, Suspended, resume, run_turn
from core.session import new_session
from testing.fake_llm import FakeLLM, call, text_response, tool_response
from tools.registry import ToolRegistry
from tools.builtins.files import DeleteFileTool, ReadFileTool, WriteFileTool

ALL_TOOLS = ["read_file", "write_file", "delete_file"]


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(ReadFileTool())
    r.register(WriteFileTool())
    r.register(DeleteFileTool())
    return r


@pytest.fixture
def sandbox_root(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    return root


def _supervised():
    # write_file requires confirmation at 'supervised'
    return make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.SUPERVISED)


def test_interactive_approve_executes(registry, sandbox_root):
    target = sandbox_root / "approved.txt"
    llm = FakeLLM([
        tool_response(call("write_file", {"path": str(target), "content": "yes"})),
        text_response("Written."),
    ])
    session = new_session("tester")
    result = run_turn(session, "write it", llm, registry, _supervised(),
                      make_system(fs_root=sandbox_root),
                      approver=lambda decision: True)
    assert isinstance(result, Completed)
    assert target.read_text(encoding="utf-8") == "yes"


def test_interactive_deny_becomes_denial_and_run_continues(registry, sandbox_root):
    """An explicit human 'no' is a denial fed to the model — NOT a suspension."""
    target = sandbox_root / "refused.txt"
    llm = FakeLLM([
        tool_response(call("write_file", {"path": str(target), "content": "x"})),
        text_response("Understood, I won't write it."),
    ])
    session = new_session("tester")
    result = run_turn(session, "write it", llm, registry, _supervised(),
                      make_system(fs_root=sandbox_root),
                      approver=lambda decision: False)
    assert isinstance(result, Completed)
    assert result.text == "Understood, I won't write it."
    assert not target.exists()
    assert session.status == "done"
    refusal = [m for m in session.conversation if m["role"] == "tool_result"][0]
    assert "declined" in refusal["content"]


def test_headless_suspends_without_executing(registry, sandbox_root):
    target = sandbox_root / "pending.txt"
    llm = FakeLLM([
        tool_response(call("write_file", {"path": str(target), "content": "x"})),
    ])
    session = new_session("tester")
    result = run_turn(session, "write it", llm, registry, _supervised(),
                      make_system(fs_root=sandbox_root))  # no approver
    assert isinstance(result, Suspended)
    assert session.status == "awaiting_approval"
    assert not target.exists()
    assert session.pending_action is not None
    assert session.pending_action.call.name == "write_file"
    assert "confirmation" in session.pending_action.reason


def test_resume_approved_executes_then_continues(registry, sandbox_root):
    target = sandbox_root / "resumed.txt"
    llm = FakeLLM([
        tool_response(call("write_file", {"path": str(target), "content": "later"})),
        text_response("Done after approval."),
    ])
    session = new_session("tester")
    agent, system = _supervised(), make_system(fs_root=sandbox_root)
    assert isinstance(run_turn(session, "write it", llm, registry, agent, system), Suspended)

    result = resume(session, True, llm, registry, agent, system)
    assert isinstance(result, Completed)
    assert result.text == "Done after approval."
    assert target.read_text(encoding="utf-8") == "later"
    assert session.budget.tool_calls_made == 1
    assert session.pending_action is None
    assert session.status == "done"


def test_resume_declined_continues_without_executing(registry, sandbox_root):
    target = sandbox_root / "never.txt"
    llm = FakeLLM([
        tool_response(call("write_file", {"path": str(target), "content": "x"})),
        text_response("Okay, skipping the write."),
    ])
    session = new_session("tester")
    agent, system = _supervised(), make_system(fs_root=sandbox_root)
    run_turn(session, "write it", llm, registry, agent, system)

    result = resume(session, False, llm, registry, agent, system)
    assert isinstance(result, Completed)
    assert not target.exists()
    assert session.budget.tool_calls_made == 0
    refusal = [m for m in session.conversation if m["role"] == "tool_result"][0]
    assert "declined" in refusal["content"]


def test_mid_batch_suspension_carries_remaining_calls(registry, sandbox_root):
    """First call in the batch executes, second suspends, third is carried
    in pending.remaining and runs after approval — each exactly once."""
    f1 = sandbox_root / "one.txt"
    f1.write_text("first", encoding="utf-8")
    f2 = sandbox_root / "two.txt"
    f3 = sandbox_root / "three.txt"
    f3.write_text("third", encoding="utf-8")

    llm = FakeLLM([
        tool_response(
            call("read_file", {"path": str(f1)}, id="c1"),
            call("write_file", {"path": str(f2), "content": "second"}, id="c2"),
            call("read_file", {"path": str(f3)}, id="c3"),
        ),
        text_response("All three handled."),
    ])
    session = new_session("tester")
    agent, system = _supervised(), make_system(fs_root=sandbox_root)

    result = run_turn(session, "do three things", llm, registry, agent, system)
    assert isinstance(result, Suspended)
    assert session.pending_action.call.id == "c2"
    assert [r.id for r in session.pending_action.remaining] == ["c3"]
    # c1 executed before the suspension; c3 has not run yet.
    done_ids = [m["tool_call_id"] for m in session.conversation
                if m["role"] == "tool_result"]
    assert done_ids == ["c1"]

    result = resume(session, True, llm, registry, agent, system)
    assert isinstance(result, Completed)
    assert f2.read_text(encoding="utf-8") == "second"
    done_ids = [m["tool_call_id"] for m in session.conversation
                if m["role"] == "tool_result"]
    assert done_ids == ["c1", "c2", "c3"]
    assert session.budget.tool_calls_made == 3


def test_resume_on_non_suspended_session_errors(registry, sandbox_root):
    from core.loop import Errored

    session = new_session("tester")
    result = resume(session, True, FakeLLM([]), registry, _supervised(),
                    make_system(fs_root=sandbox_root))
    assert isinstance(result, Errored)
