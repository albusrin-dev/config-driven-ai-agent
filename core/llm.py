"""The LLM port: the core contract the loop depends on.

Adapters live in ``llm/`` and implement ``LLMPort``; nothing
provider-specific leaks through this module. The loop is injected with an
``LLMPort`` and never imports a concrete adapter.

Neutral message shapes (plain JSON-safe dicts, stored in the session):
  {"role": "user", "content": <text>}
  {"role": "assistant", "content": <text|None>,
   "tool_calls": [{"id", "name", "params"}]}
  {"role": "tool_result", "tool_call_id": <id>, "content": <text>,
   "ok": <bool>}

Neutral tool schema shape:
  {"name": <tool name>, "description": <text>,
   "input_schema": <JSON schema dict>}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.models import PricingConfig


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cost_estimate: float | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str | None
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    stop_reason: str


class LLMPort(ABC):
    @abstractmethod
    def complete(
        self, messages: list[dict], tool_schemas: list[dict]
    ) -> LLMResponse:
        """One model call over the conversation with the given tools."""

    @abstractmethod
    def count_tokens(self, messages: list[dict], tool_schemas: list[dict]) -> int:
        """Estimate the token size of a prospective request."""


def estimate_cost(usage: Usage, pricing: PricingConfig) -> float:
    """USD cost of one call under the provider's configured pricing.

    Lives here (not in ``llm/``) because the loop — core — is the consumer;
    core never imports the adapter package.
    """
    return (
        usage.input_tokens / 1_000_000 * pricing.input_usd_per_mtok
        + usage.output_tokens / 1_000_000 * pricing.output_usd_per_mtok
    )
