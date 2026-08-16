"""A2: the context budget is derived, and count_tokens is local + conservative."""

import urllib.request

import pytest

from conftest import make_agent, make_system

from config.secrets import Secrets
from core.loop import OUTPUT_RESERVE_TOKENS, Completed, effective_context_budget, run_turn
from core.memory import ContextBudgetError
from core.session import new_session
from llm.anthropic import AnthropicAdapter
from memory.window import WindowMemory
from testing.fake_llm import FakeLLM, text_response
from tools.registry import ToolRegistry
from tools.builtins.files import ReadFileTool


def _agent(budget_tokens=None, context_window=None):
    agent = make_agent(allowlist=["read_file"])
    agent.memory.budget_tokens = budget_tokens
    if context_window is not None:
        agent.llm.context_window = context_window
    return agent


# --- Derivation ------------------------------------------------------------

def test_budget_is_derived_from_window_minus_reserves():
    llm = FakeLLM([])
    agent = _agent(context_window=50_000)
    session = new_session("tester", system_prompt="S" * 900)
    schemas = [{"name": "read_file", "description": "d", "input_schema": {}}]

    overhead = llm.count_tokens(
        [{"role": "system", "content": session.system_prompt}], schemas
    )
    expected = 50_000 - OUTPUT_RESERVE_TOKENS - overhead
    assert effective_context_budget(session, agent, llm, schemas) == expected
    assert 0 < expected < 50_000  # reserves genuinely subtracted


def test_default_config_derives_not_8000():
    """The old magic constant is gone: an unconfigured agent derives a
    budget from the (default 200k) model window."""
    agent = _agent()
    assert agent.memory.budget_tokens is None
    session = new_session("tester", system_prompt="be brief")
    budget = effective_context_budget(session, agent, FakeLLM([]), [])
    assert budget > 100_000  # window-derived, nothing like a magic 8000


def test_explicit_override_wins():
    agent = _agent(budget_tokens=400)
    session = new_session("tester", system_prompt="S" * 5000)
    assert effective_context_budget(session, agent, FakeLLM([]), []) == 400


def test_window_too_small_for_overhead_is_terminal():
    agent = _agent(context_window=OUTPUT_RESERVE_TOKENS + 1)
    session = new_session("tester", system_prompt="a long identity " * 50)
    with pytest.raises(ContextBudgetError) as exc_info:
        effective_context_budget(session, agent, FakeLLM([]), [])
    assert "no room for conversation context" in str(exc_info.value)


def test_loop_runs_on_derived_budget(tmp_path):
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    agent = _agent()  # budget_tokens=None -> derived
    system = make_system(fs_root=tmp_path)
    llm = FakeLLM([text_response("fine")])
    session = new_session("tester", system_prompt="be helpful")
    result = run_turn(session, "hi", llm, registry, agent, system,
                      memory=WindowMemory())
    assert isinstance(result, Completed)


# --- count_tokens: local and conservative ----------------------------------

@pytest.fixture
def adapter():
    return AnthropicAdapter(
        provider_name="anthropic",
        model="claude-fable-5",
        secrets=Secrets({"anthropic": "TEST_KEY_VAR"}),
    )


def test_count_tokens_makes_no_network_call(adapter, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("count_tokens must never touch the network (Rule 10)")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    messages = [{"role": "user", "content": "hello there " * 200}]
    schemas = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    assert adapter.count_tokens(messages, schemas) > 0


# (text, floor on the true Claude token count) — floors are deliberately
# generous over-statements of the real counts, so the assertion
# "estimate >= floor" proves over-estimation with margin.
CONSERVATIVE_SAMPLES = [
    ("Hello, world!", 6),
    ("The quick brown fox jumps over the lazy dog.", 12),
    # JSON-ish content tokenizes densely (~3 chars/token).
    ('{"path": "workspace/notes.txt", "content": "milk\\neggs\\ncoffee"}', 25),
    # Long English prose: ~4 chars/token true, chars/3 estimated.
    ("A configuration-driven agent framework separates engine from identity. " * 10,
     10 * 14),
]


@pytest.mark.parametrize("text,true_count_floor", CONSERVATIVE_SAMPLES)
def test_count_tokens_over_estimates(adapter, text, true_count_floor):
    estimate = adapter.count_tokens([{"role": "user", "content": text}], [])
    assert estimate >= true_count_floor


def test_count_tokens_monotone_in_content(adapter):
    short = adapter.count_tokens([{"role": "user", "content": "a"}], [])
    long = adapter.count_tokens([{"role": "user", "content": "a" * 10_000}], [])
    assert long > short
