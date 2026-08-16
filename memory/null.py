"""Passthrough adapter: send the raw buffer — exact Phase 2 behavior."""

from __future__ import annotations

from core.memory import MemoryPort, TokenCounter


class NullMemory(MemoryPort):
    def assemble_context(
        self,
        conversation: list[dict],
        budget_tokens: int,
        count_tokens: TokenCounter,
    ) -> list[dict]:
        # A fresh list (never the caller's own) so downstream code can't
        # accidentally alias-and-mutate stored history (Rule 11).
        return [dict(message) for message in conversation]
