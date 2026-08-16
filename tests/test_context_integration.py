"""Loop + memory integration: a run long enough to hit the Phase 2 token
wall under 'none' completes under 'window'; stored history stays complete
and untouched (Rule 11); windowing costs nothing and is deterministic."""

import copy

import pytest

from conftest import make_agent, make_system

from config.models import Autonomy
from core.llm import LLMResponse, Usage
from core.loop import BudgetExceeded, Completed, Errored, run_turn
from core.session import new_session
from memory.null import NullMemory
from memory.window import WindowMemory
from testing.fake_llm import FakeLLM, call, text_response, tool_response
from tools.registry import ToolRegistry
from tools.builtins.files import ReadFileTool

N_READS = 8


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(ReadFileTool())
    return r


@pytest.fixture
def sandbox_root(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    # A chunky file so each tool result meaningfully grows the conversation.
    (root / "data.txt").write_text("lorem ipsum " * 60, encoding="utf-8")
    return root


class SizedFakeLLM(FakeLLM):
    """FakeLLM whose input-token usage tracks what it was actually SENT —
    which is what makes the token wall real and windowing measurable."""

    def complete(self, messages, tool_schemas):
        scripted = super().complete(messages, tool_schemas)
        return LLMResponse(
            text=scripted.text,
            tool_calls=scripted.tool_calls,
            usage=Usage(
                input_tokens=self.count_tokens(messages, tool_schemas),
                output_tokens=5,
            ),
            stop_reason=scripted.stop_reason,
        )


def _script(sandbox_root):
    return [
        tool_response(call("read_file", {"path": str(sandbox_root / "data.txt")},
                           id=f"c{i}"))
        for i in range(N_READS)
    ] + [text_response("All reads done.")]


def _run(strategy_memory, registry, sandbox_root, token_cap):
    llm = SizedFakeLLM(_script(sandbox_root))
    session = new_session("tester")
    agent = make_agent(allowlist=["read_file"], autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    agent.memory.budget_tokens = 400  # conversation-context budget
    system = make_system(fs_root=sandbox_root,
                         limits={"max_tokens_per_session": token_cap})
    result = run_turn(session, "read data.txt eight times", llm, registry,
                      agent, system, memory=strategy_memory)
    return result, session, llm


# The cap that separates the two strategies: generous enough for the
# windowed run (bounded per-call input), too small for the ever-growing
# raw buffer. Verified tight by both assertions below.
TOKEN_CAP = 6_000


def test_window_completes_where_none_hits_the_wall(registry, sandbox_root):
    result_w, session_w, _ = _run(WindowMemory(), registry, sandbox_root, TOKEN_CAP)
    assert isinstance(result_w, Completed), getattr(result_w, "reason", result_w)
    assert session_w.budget.tool_calls_made == N_READS

    result_n, session_n, _ = _run(NullMemory(), registry, sandbox_root, TOKEN_CAP)
    assert isinstance(result_n, BudgetExceeded) and result_n.which == "tokens"
    assert session_n.status == "budget_exceeded"
    # The raw buffer burned the cap before the task could finish: the final
    # answer never arrived (the windowed run delivered it under the same cap).
    assert all(m.get("content") != "All reads done."
               for m in session_n.conversation)


def test_stored_history_is_complete_and_marker_free(registry, sandbox_root):
    result, session, llm = _run(WindowMemory(), registry, sandbox_root, TOKEN_CAP)
    assert isinstance(result, Completed)
    # Full history: 1 user + (N_READS tool_use + N_READS results) + final answer.
    assert len(session.conversation) == 1 + 2 * N_READS + 1
    assert all("omitted" not in str(m.get("content")) for m in session.conversation)
    # ... while the model was actually SENT compacted context with a marker.
    late_call = llm.calls[-1]["messages"]
    assert any("omitted" in str(m.get("content")) for m in late_call)
    assert len(late_call) < len(session.conversation)
    # The head survived compaction on every single call.
    assert all(c["messages"][0]["content"] == "read data.txt eight times"
               or c["messages"][0]["role"] == "user"
               for c in llm.calls)
    assert llm.calls[-1]["messages"][0]["content"] == "read data.txt eight times"


def test_windowing_is_free_and_deterministic(registry, sandbox_root):
    """No LLM call, no budget cost: token usage comes only from the LLM's
    reported Usage; assembling twice yields identical output."""
    result, session, llm = _run(WindowMemory(), registry, sandbox_root, TOKEN_CAP)
    assert isinstance(result, Completed)
    # Exactly one LLM call per iteration — assembly added none.
    assert len(llm.calls) == N_READS + 1
    # Usage-only accounting: recompute each call's usage from what was sent
    # (input = counted size, output = 5); the session total must match.
    expected = sum(llm.count_tokens(c["messages"], c["tool_schemas"]) + 5
                   for c in llm.calls)
    assert session.budget.tokens_used == expected

    snapshot = copy.deepcopy(session.conversation)
    window = WindowMemory()
    counter = lambda ms: llm.count_tokens(ms, [])  # noqa: E731
    once = window.assemble_context(session.conversation, 400, counter)
    twice = window.assemble_context(session.conversation, 400, counter)
    assert once == twice
    assert session.conversation == snapshot


def test_single_message_too_large_is_clean_errored(registry, tmp_path):
    sandbox_root = tmp_path / "sb"
    sandbox_root.mkdir()
    (sandbox_root / "data.txt").write_text("x" * 8000, encoding="utf-8")
    llm = SizedFakeLLM([
        tool_response(call("read_file", {"path": str(sandbox_root / "data.txt")})),
        text_response("never reached"),
    ])
    session = new_session("tester")
    agent = make_agent(allowlist=["read_file"], autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    agent.memory.budget_tokens = 50  # far too small for the giant tool result
    system = make_system(fs_root=sandbox_root)
    result = run_turn(session, "read it", llm, registry, agent, system,
                      memory=WindowMemory())
    assert isinstance(result, Errored)
    assert "context assembly failed" in result.reason
    assert session.status == "error"


def test_secret_free_session_dump_after_windowed_run(registry, sandbox_root, monkeypatch):
    from conftest import FAKE_KEY

    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    result, session, _ = _run(WindowMemory(), registry, sandbox_root, TOKEN_CAP)
    assert FAKE_KEY not in session.model_dump_json()
