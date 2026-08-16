"""Tool registry: name -> tool instance, and allowlist resolution.

The registry gives the allowlist its cross-check (deferred from Phase 0
because no registry existed): an allowlist entry naming an unregistered
tool raises clearly instead of failing silently at call time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.errors import UnknownToolError

if TYPE_CHECKING:
    from config.models import AgentConfig
    from core.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.destructive and not tool.mutating:
            raise ValueError(
                f"tool '{tool.name}': destructive=True requires mutating=True"
            )
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise UnknownToolError([name], sorted(self._tools))
        return self._tools[name]

    def toolset_for(self, agent_config: AgentConfig) -> list[Tool]:
        """The agent's tools, exactly its allowlist — the single source of
        truth for capability. Raises if any entry is unregistered."""
        missing = [n for n in agent_config.tools.allowlist if n not in self._tools]
        if missing:
            raise UnknownToolError(missing, sorted(self._tools))
        return [self._tools[n] for n in agent_config.tools.allowlist]
