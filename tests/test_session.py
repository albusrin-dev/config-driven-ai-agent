"""Session serialization: JSON round-trip, resume from a restored session,
and no secret anywhere in the dump."""

import pytest

from conftest import FAKE_KEY, make_agent, make_system

from config.models import Autonomy
from core.loop import Completed, Suspended, resume, run_turn
from core.session import Session, new_session
from testing.fake_llm import FakeLLM, call, text_response, tool_response
from tools.registry import ToolRegistry
from tools.builtins.files import DeleteFileTool, ReadFileTool, WriteFileTool


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


def test_suspended_session_round_trips_and_resumes(registry, sandbox_root, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)  # present in env throughout
    target = sandbox_root / "restored.txt"
    agent = make_agent(allowlist=["write_file"], autonomy=Autonomy.SUPERVISED)
    system = make_system(fs_root=sandbox_root)

    llm = FakeLLM([
        tool_response(call("write_file", {"path": str(target), "content": "persisted"})),
        text_response("Wrote it after the restart."),
    ])
    session = new_session("tester")
    result = run_turn(session, "write it", llm, registry, agent, system)
    assert isinstance(result, Suspended)

    # Serialize (as a persistence layer would), then restore into a fresh object.
    dump = session.model_dump_json()
    assert FAKE_KEY not in dump  # no secret in the session dump, ever
    restored = Session.model_validate_json(dump)
    assert restored == session
    assert restored.status == "awaiting_approval"
    assert restored.pending_action.call.name == "write_file"

    result = resume(restored, True, llm, registry, agent, system)
    assert isinstance(result, Completed)
    assert target.read_text(encoding="utf-8") == "persisted"
    assert restored.status == "done"


def test_completed_session_round_trips(registry, sandbox_root):
    llm = FakeLLM([text_response("hi there")])
    session = new_session("tester")
    run_turn(session, "hello", llm, registry,
             make_agent(allowlist=[]), make_system(fs_root=sandbox_root))
    restored = Session.model_validate_json(session.model_dump_json())
    assert restored == session
    assert restored.status == "done"
    assert restored.budget.tokens_used == session.budget.tokens_used


def test_new_session_defaults():
    session = new_session("scribe")
    assert session.agent_name == "scribe"
    assert session.status == "idle"
    assert session.conversation == []
    assert session.budget.tokens_used == 0
    assert session.pending_action is None
