"""The memory port: the core contract for context assembly (Rule 11).

The session stores the COMPLETE conversation — source of truth, persisted,
audited. The memory layer only decides what subset/summary is SENT to the
model on a given call; it never mutates, summarizes-in-place, or drops
from stored history.

The port is deliberately one method (Rule 7): it is shaped by what context
management needs today. A future persistent-recall adapter implements this
same method (injecting retrieved context into what it returns); its
retrieval-specific surface is added in its own phase, on contact.

``count_tokens`` is injected (from the LLM adapter) so this layer never
depends on a concrete LLM. ``budget_tokens`` covers the conversation
portion only; the caller accounts for fixed per-call overhead (system
prompt, tool schemas) when choosing it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

# Counts the token size of a message list (conversation portion only).
TokenCounter = Callable[[list[dict]], int]


class ContextBudgetError(Exception):
    """The context cannot be assembled within the budget (budget too small
    or a single message too large). Terminal for the turn — never loop
    trying to shrink further."""


class MemoryPort(ABC):
    @abstractmethod
    def assemble_context(
        self,
        conversation: list[dict],
        budget_tokens: int,
        count_tokens: TokenCounter,
    ) -> list[dict]:
        """Return the messages to send for this call, within
        ``budget_tokens``. Must be deterministic, must not mutate
        ``conversation``, and must perform no LLM call."""
