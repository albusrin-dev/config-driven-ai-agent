"""Window strategy unit tests: budget math, pair integrity, head safety,
terminal too-small case, null passthrough, and no mutation (Rule 11)."""

import copy

import pytest

from core.memory import ContextBudgetError
from memory import adapter_for
from memory.null import NullMemory
from memory.window import WindowMemory

from config.models import MemoryStrategy

# Deterministic counter: 10 "tokens" per message — budget math in tens.
count = lambda messages: 10 * len(messages)  # noqa: E731


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text, "tool_calls": []}


def tool_pair(call_id, n_results=1):
    """An assistant tool_use plus its result(s) — an atomic block."""
    msgs = [{"role": "assistant", "content": None,
             "tool_calls": [{"id": call_id, "name": "read_file", "params": {}}]}]
    for i in range(n_results):
        msgs.append({"role": "tool_result", "tool_call_id": call_id,
                     "content": f"result {i}", "ok": True})
    return msgs


def long_conversation():
    """user task, then 4 tool pairs (2 msgs each), then an assistant answer:
    10 messages = 100 'tokens'."""
    convo = [user("the initial task")]
    for i in range(4):
        convo += tool_pair(f"c{i}")
    convo.append(assistant("intermediate answer"))
    return convo


window = WindowMemory()


def test_fits_returns_unchanged():
    convo = long_conversation()
    assert window.assemble_context(convo, 100, count) == convo


def test_over_budget_head_marker_tail_within_budget():
    convo = long_conversation()  # 10 msgs = 100
    result = window.assemble_context(convo, 70, count)
    assert count(result) <= 70
    assert result[0] == convo[0]                       # head: initial task
    assert "omitted" in result[1]["content"]           # marker second
    assert result[2:] == convo[-len(result) + 2:]      # tail is the most recent, verbatim
    # Marker names the number of omitted messages.
    omitted = len(convo) - 1 - (len(result) - 2)
    assert f"[{omitted} earlier messages omitted" in result[1]["content"]


def test_head_is_always_present():
    convo = [{"role": "system", "content": "you are scribe"}] + long_conversation()
    result = window.assemble_context(convo, 60, count)
    assert result[0]["role"] == "system"
    assert result[1] == convo[1]  # the initial user task
    assert "omitted" in result[2]["content"]


def test_tool_pairs_never_split():
    convo = [user("task")]
    for i in range(5):
        convo += tool_pair(f"c{i}", n_results=2)  # 3-message blocks
    # Budgets that would land mid-block if cutting were per-message.
    for budget in (40, 50, 60, 70, 80, 90, 100, 110):
        try:
            result = window.assemble_context(convo, budget, count)
        except ContextBudgetError:
            continue
        sent_use_ids = {c["id"] for m in result
                        if m["role"] == "assistant" for c in (m.get("tool_calls") or [])}
        sent_result_ids = {m["tool_call_id"] for m in result
                           if m["role"] == "tool_result"}
        assert sent_use_ids == sent_result_ids, (
            f"budget {budget}: orphaned tool ids "
            f"{sent_use_ids ^ sent_result_ids}"
        )


def test_cut_is_at_block_boundary():
    convo = [user("task")] + tool_pair("a") + tool_pair("b") + tool_pair("c")
    result = window.assemble_context(convo, 50, count)  # head+marker+one pair
    # First message after the marker must start a block, never continue one.
    after_marker = result[2]
    assert after_marker["role"] != "tool_result"


def test_budget_too_small_is_terminal():
    convo = [user("task")] + tool_pair("a", n_results=2)  # head 10 + block 30
    with pytest.raises(ContextBudgetError) as exc_info:
        # Conversation (40) exceeds 39; head+marker+block = 50 can't fit either.
        window.assemble_context(convo, 39, count)
    assert "budget_tokens=39" in str(exc_info.value)


def test_input_never_mutated():
    convo = long_conversation()
    snapshot = copy.deepcopy(convo)
    window.assemble_context(convo, 60, count)
    assert convo == snapshot


def test_deterministic():
    convo = long_conversation()
    assert (window.assemble_context(convo, 60, count)
            == window.assemble_context(convo, 60, count))


def test_null_adapter_passthrough():
    convo = long_conversation()
    result = NullMemory().assemble_context(convo, 1, count)  # budget ignored
    assert result == convo
    assert result is not convo  # a copy, not an alias into stored history


def test_factory_selects_by_strategy():
    assert isinstance(adapter_for(MemoryStrategy.NONE), NullMemory)
    assert isinstance(adapter_for(MemoryStrategy.WINDOW), WindowMemory)
