"""The single enforcement chokepoint (Durable Rule 8).

``enforce_and_run`` is the ONLY path that executes a tool: validate params,
consult the policy gate, emit an audit record, then execute / refuse /
return-pending. No other code path may call ``tool.execute``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from pydantic import ValidationError

from . import audit
from .base import ToolContext, ToolResult
from .errors import ToolParamError
from .gate import Decision, GateDecision, PolicyGate

if TYPE_CHECKING:
    from pydantic import BaseModel

    from config.models import AgentConfig, SystemConfig

    from .base import Tool

# Called only for REQUIRE_CONFIRMATION; True means "approved, execute".
# Exists purely so the approved path is testable — no approval UI/flow yet.
Approver = Callable[[GateDecision], bool]


@dataclass(frozen=True)
class Denied:
    reason: str


@dataclass(frozen=True)
class Pending:
    """Confirmation required and not granted; nothing was executed."""

    reason: str


@dataclass(frozen=True)
class Executed:
    result: ToolResult


Outcome = Denied | Pending | Executed


def _validate_params(tool: Tool, params: dict[str, Any] | BaseModel) -> BaseModel:
    if isinstance(params, tool.input_schema):
        return params
    try:
        return tool.input_schema.model_validate(params)
    except ValidationError as e:
        problems = [
            f"{'.'.join(str(p) for p in err['loc']) or '(top level)'}: {err['msg']}"
            for err in e.errors()
        ]
        raise ToolParamError(tool.name, problems) from None


def enforce_and_run(
    tool: Tool,
    params: dict[str, Any] | BaseModel,
    agent_config: AgentConfig,
    system_config: SystemConfig,
    context: ToolContext | None = None,
    approver: Approver | None = None,
) -> Outcome:
    # 1. Validate params before anything runs.
    validated = _validate_params(tool, params)
    if context is None:
        context = ToolContext(agent=agent_config, system=system_config)

    # 2. The gate is the only place a permission decision is made.
    decision = PolicyGate().evaluate(tool, validated, agent_config, system_config)

    # 3. Audit every decision (redacted).
    audit.emit(
        audit.make_record(
            agent_name=agent_config.name,
            tool_name=tool.name,
            decision=decision.decision.value,
            reason=decision.reason,
            effects_summary=decision.effects_summary,
            params=validated,
        )
    )

    # 4-6. Act on the decision.
    if decision.decision is Decision.DENY:
        return Denied(decision.reason)
    if decision.decision is Decision.REQUIRE_CONFIRMATION:
        if approver is None or not approver(decision):
            return Pending(decision.reason)
    return Executed(tool.execute(validated, context))
