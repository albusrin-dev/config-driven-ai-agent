"""Session / runtime state: conversation buffer, budget counters, status,
pending action. Fully JSON-serializable (Pydantic) and secret-free by
construction — nothing in here ever touches ``Secrets``.

The conversation buffer is deliberately naive: no summarization or context-
window management, bounded only by the token cap and the iteration ceiling.
Real context management arrives with the memory phase.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCallRecord(_Strict):
    id: str
    name: str
    params: dict[str, Any]


class PendingAction(_Strict):
    """A gate-required confirmation awaiting a human decision, plus the
    rest of the same tool-call batch so resume can finish the round."""

    call: ToolCallRecord
    reason: str
    remaining: list[ToolCallRecord] = Field(default_factory=list)


class BudgetState(_Strict):
    tokens_used: int = 0
    cost_used: float = 0.0
    tool_calls_made: int = 0
    iterations: int = 0


Status = Literal[
    "idle", "running", "awaiting_approval", "done", "error", "budget_exceeded"
]


class Session(_Strict):
    session_id: str
    agent_name: str
    # Assembled once at session start (identity does not change mid-session)
    # and prepended to every LLM call by the loop. Built from profile/user
    # data only, so serializing it keeps the dump secret-free. It is NOT
    # part of ``conversation`` — stored history stays pure (Rule 11).
    system_prompt: str = ""
    conversation: list[dict[str, Any]] = Field(default_factory=list)
    budget: BudgetState = Field(default_factory=BudgetState)
    status: Status = "idle"
    pending_action: PendingAction | None = None


def new_session(agent_name: str, system_prompt: str = "") -> Session:
    return Session(
        session_id=uuid.uuid4().hex,
        agent_name=agent_name,
        system_prompt=system_prompt,
    )
