"""Durable Rule 10: every bound stops the run cleanly; the ceiling is hard."""

import logging

import pytest

from conftest import make_agent, make_system

from config.models import Autonomy
from core.loop import ITERATION_CEILING, BudgetExceeded, Completed, run_turn
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


def _agent():
    return make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.AUTONOMOUS_BOUNDED)


def test_tool_call_cap_stops_mid_batch(registry, sandbox_root):
    """Cap 2, model asks for 3 in one batch: exactly 2 execute."""
    system = make_system(fs_root=sandbox_root,
                         limits={"max_tool_calls_per_session": 2})
    llm = FakeLLM([
        tool_response(
            call("write_file", {"path": str(sandbox_root / "1.txt"), "content": "1"}),
            call("write_file", {"path": str(sandbox_root / "2.txt"), "content": "2"}),
            call("write_file", {"path": str(sandbox_root / "3.txt"), "content": "3"}),
        ),
    ])
    session = new_session("tester")
    result = run_turn(session, "write three files", llm, registry, _agent(), system)

    assert isinstance(result, BudgetExceeded) and result.which == "tool_calls"
    assert session.status == "budget_exceeded"
    assert (sandbox_root / "1.txt").exists()
    assert (sandbox_root / "2.txt").exists()
    assert not (sandbox_root / "3.txt").exists()  # the cap held
    assert session.budget.tool_calls_made == 2


def test_token_cap_stops_next_iteration(registry, sandbox_root):
    system = make_system(fs_root=sandbox_root,
                         limits={"max_tokens_per_session": 30})
    source = sandbox_root / "f.txt"
    source.write_text("x", encoding="utf-8")
    llm = FakeLLM(
        [tool_response(call("read_file", {"path": str(source)}),
                       input_tokens=20, output_tokens=10)],
        repeat_last=True,
    )
    session = new_session("tester")
    result = run_turn(session, "loop forever", llm, registry, _agent(), system)

    assert isinstance(result, BudgetExceeded) and result.which == "tokens"
    assert session.budget.tokens_used == 30
    assert len(llm.calls) == 1  # the cap stopped the second LLM call


def test_cost_cap_with_pricing_configured(registry, sandbox_root):
    # 10 input tokens at $1M per Mtok = $10 > the $5 default cap.
    system = make_system(
        fs_root=sandbox_root,
        pricing={"input_usd_per_mtok": 1_000_000.0, "output_usd_per_mtok": 0.0},
    )
    source = sandbox_root / "f.txt"
    source.write_text("x", encoding="utf-8")
    llm = FakeLLM(
        [tool_response(call("read_file", {"path": str(source)}), input_tokens=10)],
        repeat_last=True,
    )
    session = new_session("tester")
    result = run_turn(session, "expensive", llm, registry, _agent(), system)

    assert isinstance(result, BudgetExceeded) and result.which == "cost"
    assert session.budget.cost_used == pytest.approx(10.0)


def test_cost_cap_inactive_without_pricing(registry, sandbox_root, caplog):
    """No pricing: the cost cap never fires (even at a cap of $0) and its
    inactivity is logged — the cost is never fabricated."""
    system = make_system(fs_root=sandbox_root,
                         limits={"max_cost_per_session_usd": 0.0})
    llm = FakeLLM([text_response("cheap and cheerful")])
    session = new_session("tester")
    with caplog.at_level(logging.INFO, logger="agent.loop"):
        result = run_turn(session, "hello", llm, registry, _agent(), system)

    assert isinstance(result, Completed)
    assert session.budget.cost_used == 0.0
    assert any("cost cap inactive" in r.message for r in caplog.records)


def test_iteration_ceiling_stops_infinite_scripted_loop(registry, sandbox_root):
    """A model that never stops calling tools is cut off by the hard
    ceiling, regardless of generous config limits."""
    system = make_system(
        fs_root=sandbox_root,
        limits={
            "max_tool_calls_per_session": 10_000,
            "max_tokens_per_session": 10_000_000,
        },
    )
    source = sandbox_root / "f.txt"
    source.write_text("x", encoding="utf-8")
    llm = FakeLLM(
        [tool_response(call("read_file", {"path": str(source)}))],
        repeat_last=True,
    )
    session = new_session("tester")
    result = run_turn(session, "never stop", llm, registry, _agent(), system)

    assert isinstance(result, BudgetExceeded) and result.which == "iteration_ceiling"
    assert session.budget.iterations == ITERATION_CEILING
    assert len(llm.calls) == ITERATION_CEILING
