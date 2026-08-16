"""Tool contracts: the abstract Tool interface, ToolContext, ToolResult.

These live in ``core`` (not ``tools``) so that dependencies point strictly
inward: ``core.gate`` and ``core.enforce`` need ``ToolContext``/``Tool``
without importing the ``tools`` package. ``tools/__init__.py`` re-exports
them for convenience.

Tools contain ZERO permission logic (Durable Rule 8): ``execute`` assumes
the call is already authorised, and every execution goes through
``core.enforce.enforce_and_run``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from .effects import Effect

if TYPE_CHECKING:
    from pydantic import BaseModel

    from config.models import AgentConfig, SystemConfig


@dataclass(frozen=True)
class ToolContext:
    """Passed to effect-planning and execution.

    Minimal by design: the active configs only. No session, no secrets.
    """

    agent: AgentConfig
    system: SystemConfig


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None


class Tool(ABC):
    """Abstract tool.

    ``name`` is an opaque identifier — never branched on. ``destructive``
    implies ``mutating`` (checked at class-definition time). ``plan_effects``
    must declare, from params alone and BEFORE execution, every resource the
    call will touch — that is what keeps the gate tool-agnostic.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]
    mutating: ClassVar[bool] = False
    destructive: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "destructive", False) and not getattr(cls, "mutating", False):
            raise TypeError(
                f"tool class '{cls.__name__}': destructive=True requires "
                f"mutating=True (destructive implies mutating)"
            )

    @abstractmethod
    def plan_effects(self, params: BaseModel, context: ToolContext) -> list[Effect]:
        """Declare the effects this specific call will have. Pure; no IO
        beyond path resolution; must not perform the action."""

    @abstractmethod
    def execute(self, params: BaseModel, context: ToolContext) -> ToolResult:
        """Do the work. Assumes already authorised — no permission checks
        here, ever."""
