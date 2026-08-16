"""The bounded reactive loop (Durable Rules 9 & 10).

observe -> LLM (allowlisted tool schemas) -> tool calls routed through
``enforce_and_run`` -> results fed back -> repeat until a final answer, a
budget cap, or the hard iteration ceiling.

Invariants:
- Every tool execution goes through ``enforce_and_run``; this module never
  touches ``tool.execute`` (Rule 8) — tested by source inspection.
- Confirmation means a human decides (Rule 9): an explicit human "no"
  becomes a denial fed back to the model; NO human available means the run
  SUSPENDS with a recorded pending action. Never auto-approve, never
  auto-deny.
- Every run is bounded (Rule 10): budgets from system limits plus the hard
  ``ITERATION_CEILING`` below, which is a constant on purpose — it is not
  configurable away.

Self-correction falls out for free: a tool error result is fed back like
any other, so the model can retry or change course under the same bounds —
no separate retry machinery exists.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .base import ToolContext
from .enforce import Denied, Executed, Pending, enforce_and_run
from .errors import ToolParamError, UnknownToolError
from .llm import LLMPort, ToolCall, estimate_cost
from .memory import ContextBudgetError, MemoryPort
from .session import PendingAction, Session, ToolCallRecord

if TYPE_CHECKING:
    from config.models import AgentConfig, SystemConfig

    from .gate import GateDecision

logger = logging.getLogger("agent.loop")

# Hard per-run ceiling on loop iterations. A constant, not config (Rule 10).
ITERATION_CEILING = 25

# Tokens held back from the model's context window for its own output when
# DERIVING the context budget (A2): budget = context_window − this reserve
# − measured fixed overhead (system prompt + tool schemas).
OUTPUT_RESERVE_TOKENS = 4_096

Approver = Callable[["GateDecision"], bool]

# URLs the USER typed. Trailing punctuation is stripped so "see https://x/y."
# yields the URL without the sentence's full stop.
_URL_RE = re.compile(r"https?://[^\s<>\"'`\\]+", re.IGNORECASE)


def harvest_urls(text: str) -> list[str]:
    """URLs appearing in a user message (Rule 13 provenance: 'user').

    Only ever applied to the USER'S OWN message — never to tool results,
    whose content is untrusted (Rule 12) and must not be able to grant
    provenance to an exfiltration target.
    """
    return [match.group(0).rstrip(".,;:!?)]}'\"") for match in _URL_RE.finditer(text or "")]


# --------------------------------------------------------------------------
# Turn results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Completed:
    text: str


@dataclass(frozen=True)
class Suspended:
    pending: PendingAction


@dataclass(frozen=True)
class BudgetExceeded:
    which: str  # "tokens" | "cost" | "tool_calls" | "iteration_ceiling"


@dataclass(frozen=True)
class Errored:
    reason: str


TurnResult = Completed | Suspended | BudgetExceeded | Errored


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def tool_schemas_for(registry, agent_config: AgentConfig) -> list[dict]:
    """Neutral schemas for exactly the agent's allowlisted tools — the model
    is never shown a tool it cannot use."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema.model_json_schema(),
        }
        for tool in registry.toolset_for(agent_config)
    ]


class _RecordingApprover:
    """Wraps the caller's approver so the loop can tell an explicit human
    "no" (-> denial, Rule 9) apart from "no human available" (-> suspend)."""

    def __init__(self, inner: Approver) -> None:
        self._inner = inner
        self.consulted = False
        self.answer: bool | None = None

    def __call__(self, decision) -> bool:
        self.consulted = True
        self.answer = bool(self._inner(decision))
        return self.answer


def _append_tool_result(session: Session, call_id: str, content: str, ok: bool) -> None:
    session.conversation.append(
        {"role": "tool_result", "tool_call_id": call_id, "content": content, "ok": ok}
    )


def _budget_exhausted(session: Session, system_config, cost_active: bool) -> str | None:
    limits = system_config.limits
    budget = session.budget
    if budget.tokens_used >= limits.max_tokens_per_session:
        return "tokens"
    if cost_active and budget.cost_used >= limits.max_cost_per_session_usd:
        return "cost"
    if budget.tool_calls_made >= limits.max_tool_calls_per_session:
        return "tool_calls"
    return None


# --------------------------------------------------------------------------
# Tool-call processing
# --------------------------------------------------------------------------

def _process_calls(
    session: Session,
    calls: list[ToolCall],
    registry,
    agent_config: AgentConfig,
    system_config: SystemConfig,
    approver: Approver | None,
) -> TurnResult | None:
    """Run a batch of model-requested tool calls through the chokepoint.

    Returns a terminal TurnResult (suspend / budget), or None to continue
    the loop.
    """
    queue = list(calls)
    while queue:
        call = queue.pop(0)

        # Cap checked before EACH execution so the cap can't be overshot
        # by a multi-call batch.
        if session.budget.tool_calls_made >= system_config.limits.max_tool_calls_per_session:
            session.status = "budget_exceeded"
            return BudgetExceeded("tool_calls")

        try:
            tool = registry.get(call.name)
        except UnknownToolError as e:
            # Fail-closed: the model was only shown allowlisted schemas, so
            # this is the model misbehaving — refuse and tell it.
            _append_tool_result(session, call.id, f"action refused: {e}", ok=False)
            continue

        # Built per call so a search earlier in the same batch grants
        # provenance to its results for a fetch later in the batch.
        context = ToolContext(
            agent=agent_config,
            system=system_config,
            user_urls=frozenset(session.user_urls),
            search_urls=frozenset(session.search_urls),
        )
        recording = _RecordingApprover(approver) if approver is not None else None
        try:
            outcome = enforce_and_run(
                tool, call.params, agent_config, system_config,
                context=context, approver=recording,
            )
        except ToolParamError as e:
            _append_tool_result(session, call.id, f"invalid parameters: {e}", ok=False)
            continue

        if isinstance(outcome, Denied):
            _append_tool_result(
                session, call.id, f"action refused: {outcome.reason}", ok=False
            )
        elif isinstance(outcome, Pending):
            if recording is not None and recording.consulted and recording.answer is False:
                # An explicit human "no" is a denial (Rule 9); continue.
                _append_tool_result(
                    session,
                    call.id,
                    f"action refused: the user declined confirmation ({outcome.reason})",
                    ok=False,
                )
            else:
                # No human available: suspend, never auto-approve/deny (Rule 9).
                session.pending_action = PendingAction(
                    call=ToolCallRecord(id=call.id, name=call.name, params=call.params),
                    reason=outcome.reason,
                    remaining=[
                        ToolCallRecord(id=c.id, name=c.name, params=c.params)
                        for c in queue
                    ],
                )
                session.status = "awaiting_approval"
                return Suspended(session.pending_action)
        else:  # Executed
            session.budget.tool_calls_made += 1
            result = outcome.result
            # URLs from a trusted STRUCTURED source (search results) gain
            # 'search' provenance. Tools returning fetched content never
            # populate this, so hostile page text cannot whitelist itself.
            session.search_urls.update(result.discovered_urls)
            content = str(result.output) if result.ok else f"tool error: {result.error}"
            _append_tool_result(session, call.id, content, ok=result.ok)
    return None


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def effective_context_budget(
    session: Session,
    agent_config: AgentConfig,
    llm: LLMPort,
    tool_schemas: list[dict],
) -> int:
    """The conversation-context budget for this run (A2).

    An explicit ``memory.budget_tokens`` wins; otherwise the budget is
    DERIVED from the model's context window minus an output reserve minus
    the measured fixed overhead (system prompt + tool schemas) — no magic
    constant that is only correct by luck."""
    if agent_config.memory.budget_tokens is not None:
        return agent_config.memory.budget_tokens
    overhead_messages = (
        [{"role": "system", "content": session.system_prompt}]
        if session.system_prompt
        else []
    )
    overhead = llm.count_tokens(overhead_messages, tool_schemas)
    derived = agent_config.llm.context_window - OUTPUT_RESERVE_TOKENS - overhead
    if derived <= 0:
        raise ContextBudgetError(
            f"model context window ({agent_config.llm.context_window} tokens) "
            f"cannot hold the output reserve ({OUTPUT_RESERVE_TOKENS}) plus "
            f"fixed overhead ({overhead}); no room for conversation context"
        )
    return derived


def _assemble_context(
    session: Session,
    memory: MemoryPort | None,
    llm: LLMPort,
    agent_config: AgentConfig,
    tool_schemas: list[dict],
) -> list[dict]:
    """What gets SENT this call. The session keeps full history regardless
    (Rule 11); ``memory=None`` sends the raw buffer (= strategy 'none'), so
    the loop never depends on a concrete adapter. The assembled system
    prompt is prepended AFTER windowing — it lives outside the stored
    conversation, so it survives compaction by construction and stored
    history stays pure."""
    if memory is None:
        messages = list(session.conversation)
    else:
        messages = memory.assemble_context(
            session.conversation,
            effective_context_budget(session, agent_config, llm, tool_schemas),
            lambda msgs: llm.count_tokens(msgs, []),
        )
    if session.system_prompt:
        messages = [{"role": "system", "content": session.system_prompt}] + messages
    return messages


def _drive(
    session: Session,
    llm: LLMPort,
    registry,
    agent_config: AgentConfig,
    system_config: SystemConfig,
    approver: Approver | None,
    memory: MemoryPort | None = None,
) -> TurnResult:
    pricing = system_config.providers[agent_config.llm.provider].pricing
    if pricing is None:
        logger.info(
            "cost cap inactive: no pricing configured for provider '%s' "
            "(token and tool-call caps still apply)",
            agent_config.llm.provider,
        )
    while True:
        which = _budget_exhausted(session, system_config, cost_active=pricing is not None)
        if which is not None:
            session.status = "budget_exceeded"
            return BudgetExceeded(which)
        if session.budget.iterations >= ITERATION_CEILING:
            session.status = "budget_exceeded"
            return BudgetExceeded("iteration_ceiling")
        session.budget.iterations += 1

        schemas = tool_schemas_for(registry, agent_config)
        try:
            messages = _assemble_context(session, memory, llm, agent_config, schemas)
        except ContextBudgetError as e:
            session.status = "error"
            return Errored(f"context assembly failed: {e}")

        try:
            response = llm.complete(messages, schemas)
        except Exception as e:
            session.status = "error"
            return Errored(f"LLM call failed: {type(e).__name__}: {e}")

        session.budget.tokens_used += (
            response.usage.input_tokens + response.usage.output_tokens
        )
        if pricing is not None:
            session.budget.cost_used += estimate_cost(response.usage, pricing)

        session.conversation.append(
            {
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "params": c.params}
                    for c in response.tool_calls
                ],
            }
        )

        if not response.tool_calls:
            session.status = "done"
            return Completed(response.text or "")

        result = _process_calls(
            session, list(response.tool_calls), registry, agent_config,
            system_config, approver,
        )
        if result is not None:
            return result


def run_turn(
    session: Session,
    user_message: str,
    llm: LLMPort,
    registry,
    agent_config: AgentConfig,
    system_config: SystemConfig,
    approver: Approver | None = None,
    memory: MemoryPort | None = None,
) -> TurnResult:
    if session.status == "awaiting_approval":
        return Errored(
            "session is awaiting approval of a pending action; "
            "call resume() instead of run_turn()"
        )
    # URLs the user typed are provenance-approved for fetching (Rule 13).
    session.user_urls.update(harvest_urls(user_message))
    session.conversation.append({"role": "user", "content": user_message})
    session.status = "running"
    session.budget.iterations = 0  # the ceiling is per run; resume continues the same run
    return _drive(session, llm, registry, agent_config, system_config, approver, memory)


def resume(
    session: Session,
    approved: bool,
    llm: LLMPort,
    registry,
    agent_config: AgentConfig,
    system_config: SystemConfig,
    approver: Approver | None = None,
    memory: MemoryPort | None = None,
) -> TurnResult:
    """Continue a suspended run after a human decision on the pending action."""
    if session.status != "awaiting_approval" or session.pending_action is None:
        return Errored("nothing to resume: session is not awaiting approval")

    pending = session.pending_action
    session.pending_action = None  # cleared first: the action can never run twice
    session.status = "running"

    call = ToolCall(id=pending.call.id, name=pending.call.name, params=pending.call.params)
    remaining = [ToolCall(id=r.id, name=r.name, params=r.params) for r in pending.remaining]

    if approved:
        # Re-evaluate through the gate with a one-shot approval for exactly
        # this action: if the world changed since suspension the gate can
        # still deny it.
        result = _process_calls(
            session, [call], registry, agent_config, system_config,
            approver=lambda decision: True,
        )
        if result is not None:
            return result
    else:
        _append_tool_result(
            session, call.id,
            "action refused: the user declined confirmation", ok=False,
        )

    result = _process_calls(
        session, remaining, registry, agent_config, system_config, approver
    )
    if result is not None:
        return result
    return _drive(session, llm, registry, agent_config, system_config, approver, memory)
