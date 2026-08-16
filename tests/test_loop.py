"""FakeLLM-driven loop tests: tool use, completion, self-correction, and
the loop's strict coupling to the enforcement chokepoint."""

import inspect

import pytest

from conftest import make_agent, make_system

import core.loop
from config.models import Autonomy
from core.loop import Completed, Errored, run_turn
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


def _agent(allowlist=ALL_TOOLS, autonomy=Autonomy.AUTONOMOUS_BOUNDED):
    return make_agent(allowlist=allowlist, autonomy=autonomy)


def test_multi_step_run_completes(registry, sandbox_root):
    """Model reads a file, writes a derived file, then answers."""
    source = sandbox_root / "source.txt"
    source.write_text("payload", encoding="utf-8")
    dest = sandbox_root / "copy.txt"

    llm = FakeLLM([
        tool_response(call("read_file", {"path": str(source)}), text="reading"),
        tool_response(call("write_file", {"path": str(dest), "content": "payload"})),
        text_response("Copied source.txt to copy.txt."),
    ])
    session = new_session("tester")
    result = run_turn(session, "copy source to copy", llm, registry,
                      _agent(), make_system(fs_root=sandbox_root))

    assert isinstance(result, Completed)
    assert result.text == "Copied source.txt to copy.txt."
    assert session.status == "done"
    assert dest.read_text(encoding="utf-8") == "payload"
    assert session.budget.tool_calls_made == 2
    # The read result was fed back to the model verbatim.
    second_call_messages = llm.calls[1]["messages"]
    tool_results = [m for m in second_call_messages if m["role"] == "tool_result"]
    assert tool_results and tool_results[-1]["content"] == "payload"


def test_error_result_fed_back_and_model_recovers(registry, sandbox_root):
    llm = FakeLLM([
        tool_response(call("read_file", {"path": str(sandbox_root / "missing.txt")})),
        text_response("That file does not exist."),
    ])
    session = new_session("tester")
    result = run_turn(session, "read missing", llm, registry,
                      _agent(), make_system(fs_root=sandbox_root))

    assert isinstance(result, Completed)
    errors = [m for m in session.conversation
              if m["role"] == "tool_result" and not m["ok"]]
    assert len(errors) == 1
    assert "read_file failed" in errors[0]["content"]


def test_unknown_tool_call_is_refused_not_crashed(registry, sandbox_root):
    llm = FakeLLM([
        tool_response(call("teleport", {"to": "the moon"})),
        text_response("understood, no teleporting"),
    ])
    session = new_session("tester")
    result = run_turn(session, "teleport", llm, registry,
                      _agent(), make_system(fs_root=sandbox_root))

    assert isinstance(result, Completed)
    refusals = [m for m in session.conversation
                if m["role"] == "tool_result" and "refused" in m["content"]]
    assert len(refusals) == 1
    assert "teleport" in refusals[0]["content"]


def test_denied_tool_surfaced_to_model_and_loop_adapts(registry, sandbox_root):
    """Registered but not allowlisted: the gate denies, the model is told,
    the run continues, and nothing was executed."""
    target = sandbox_root / "never.txt"
    llm = FakeLLM([
        tool_response(call("write_file", {"path": str(target), "content": "x"})),
        text_response("I cannot write files."),
    ])
    session = new_session("tester")
    result = run_turn(session, "write something", llm, registry,
                      _agent(allowlist=["read_file"]), make_system(fs_root=sandbox_root))

    assert isinstance(result, Completed)
    assert not target.exists()
    refusal = [m for m in session.conversation if m["role"] == "tool_result"][0]
    assert not refusal["ok"] and "allowlist" in refusal["content"]


def test_loop_never_calls_execute_directly():
    """Rule 8 at the source level: the loop's only execution path is
    enforce_and_run."""
    source = inspect.getsource(core.loop)
    assert ".execute(" not in source
    assert "enforce_and_run" in source


def test_model_sees_only_allowlisted_schemas(registry, sandbox_root):
    llm = FakeLLM([text_response("hello")])
    session = new_session("tester")
    run_turn(session, "hi", llm, registry,
             _agent(allowlist=["read_file"]), make_system(fs_root=sandbox_root))
    schema_names = [s["name"] for s in llm.calls[0]["tool_schemas"]]
    assert schema_names == ["read_file"]
    assert "properties" in llm.calls[0]["tool_schemas"][0]["input_schema"]


def test_run_turn_on_awaiting_session_is_an_error(registry, sandbox_root):
    llm = FakeLLM([tool_response(call("write_file", {"path": "f.txt", "content": "x"}))])
    session = new_session("tester")
    agent = _agent(autonomy=Autonomy.SUPERVISED)
    system = make_system(fs_root=sandbox_root)
    run_turn(session, "write", llm, registry, agent, system)  # suspends
    assert session.status == "awaiting_approval"
    result = run_turn(session, "another message", llm, registry, agent, system)
    assert isinstance(result, Errored)
    assert "resume" in result.reason
