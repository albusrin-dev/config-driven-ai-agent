"""Typed errors for the tools/gate layer."""

from __future__ import annotations


class ToolParamError(Exception):
    """Tool parameters failed validation against the tool's input schema."""

    def __init__(self, tool_name: str, problems: list[str]) -> None:
        self.tool_name = tool_name
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"Invalid parameters for tool '{tool_name}':\n{lines}")


class UnknownToolError(Exception):
    """An allowlist entry (or lookup) names a tool that is not registered."""

    def __init__(self, missing: list[str], registered: list[str]) -> None:
        self.missing = missing
        self.registered = registered
        super().__init__(
            f"Unknown tool(s) {missing}: not registered in the tool registry. "
            f"Registered tools: {registered or '(none)'}"
        )
