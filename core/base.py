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

    Minimal by design: the active configs, plus the URL provenance index the
    gate needs for Rule 13. Still no session object and no secrets.

    ``user_urls`` came from the user's own message; ``search_urls`` came from
    a search provider's STRUCTURED results. Nothing else may enter these
    sets — URLs appearing inside fetched page text are hostile-by-default
    and must never grant themselves provenance.
    """

    agent: AgentConfig
    system: SystemConfig
    user_urls: frozenset[str] = frozenset()
    search_urls: frozenset[str] = frozenset()

    @property
    def known_urls(self) -> frozenset[str]:
        return self.user_urls | self.search_urls


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None
    # URLs a tool obtained from a TRUSTED STRUCTURED source (a search
    # provider's result list). The loop adds these to the session's
    # search-provenance set. A tool that returns fetched CONTENT must never
    # populate this — that would let an injected page whitelist its own
    # exfiltration target.
    discovered_urls: tuple[str, ...] = ()


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
        beyond path resolution; must not perform the action. Path
        resolution happens HERE, once — the resolved path rides in the
        effect and is the only one execution may touch."""

    @abstractmethod
    def execute(
        self, params: BaseModel, context: ToolContext, effects: list[Effect]
    ) -> ToolResult:
        """Do the work. Assumes already authorised — no permission checks
        here, ever. ``effects`` are the gate-blessed effect objects from
        plan_effects: operate on the paths they carry and never re-derive
        a path from the raw params (A1 / TOCTOU)."""
