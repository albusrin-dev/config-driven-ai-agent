"""The policy gate: the ONLY component that decides allow / deny / confirm.

``PolicyGate.evaluate`` is pure — no side effects, no execution, no
logging — so the full decision matrix is trivially unit-testable. It works
off the abstract Tool interface and never imports a concrete tool.

Decision algorithm (ordered; first terminating rule wins):
  1. Allowlist: tool not in the agent's allowlist -> DENY.
  2. Caps/effects: every planned effect checked against system caps.
     Filesystem effects must resolve inside sandbox.fs_root and outside all
     denied_paths; fs_root unset -> DENY (fail-closed). Unrecognised effect
     type -> DENY (fail-closed extensibility contract).
  3. Confirmation, precedence: destructive floor > per-tool override >
     autonomy default.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from config.models import Autonomy, Confirm, SandboxConfig

from .base import ToolContext
from .effects import Effect, FilesystemEffect, describe
from .paths import is_confined, resolve_real

if TYPE_CHECKING:
    from pydantic import BaseModel

    from config.models import AgentConfig, SystemConfig

    from .base import Tool


class Decision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_CONFIRMATION = "REQUIRE_CONFIRMATION"


@dataclass(frozen=True)
class GateDecision:
    decision: Decision
    reason: str
    effects_summary: tuple[str, ...] = ()


def _check_filesystem_effect(effect: FilesystemEffect, sandbox: SandboxConfig) -> str | None:
    """Return a denial reason, or None if the effect is within caps."""
    if sandbox.fs_root is None:
        return (
            "sandbox.fs_root is not set; all filesystem effects are denied "
            "(fail-closed)"
        )
    root = resolve_real(sandbox.fs_root)
    path = resolve_real(effect.path)
    if not path.is_relative_to(root):
        return f"path '{effect.path}' resolves outside sandbox root '{root}'"
    for denied in sandbox.denied_paths:
        # Absolute denied paths stand alone; relative ones are under fs_root.
        d = Path(denied)
        denied_path = resolve_real(d if d.is_absolute() else root / d)
        if path.is_relative_to(denied_path):
            return f"path '{effect.path}' is under denied path '{denied}'"
    return None


class PolicyGate:
    """Stateless, pure decision function."""

    def evaluate(
        self,
        tool: Tool,
        params: BaseModel,
        agent_config: AgentConfig,
        system_config: SystemConfig,
    ) -> GateDecision:
        # 1. Allowlist — the single source of truth for capability.
        if tool.name not in agent_config.tools.allowlist:
            return GateDecision(
                Decision.DENY,
                f"tool '{tool.name}' is not in the agent's allowlist",
            )

        # 2. Caps / effects.
        context = ToolContext(agent=agent_config, system=system_config)
        effects: list[Effect] = tool.plan_effects(params, context)
        summary = tuple(describe(e) for e in effects)
        for effect in effects:
            if isinstance(effect, FilesystemEffect):
                problem = _check_filesystem_effect(effect, system_config.sandbox)
                if problem is not None:
                    return GateDecision(Decision.DENY, problem, summary)
            else:
                return GateDecision(
                    Decision.DENY,
                    f"unrecognised effect type '{type(effect).__name__}' — "
                    f"denied (fail-closed)",
                    summary,
                )

        # 3a. Destructive floor — nothing can lower this.
        if tool.destructive:
            return GateDecision(
                Decision.REQUIRE_CONFIRMATION,
                f"tool '{tool.name}' is destructive: confirmation is always "
                f"required (floor; overrides and autonomy cannot lower it)",
                summary,
            )

        # 3b. Per-tool override.
        override = agent_config.tools.overrides.get(tool.name)
        if override is not None and override.confirm is Confirm.ALWAYS:
            return GateDecision(
                Decision.REQUIRE_CONFIRMATION,
                f"per-tool override confirm=always for '{tool.name}'",
                summary,
            )
        if override is not None and override.confirm is Confirm.NEVER:
            return GateDecision(
                Decision.ALLOW,
                f"per-tool override confirm=never for '{tool.name}'",
                summary,
            )

        # 3c. Autonomy table (base need).
        autonomy = agent_config.autonomy
        if not tool.mutating:
            if autonomy is Autonomy.ASSISTED:
                return GateDecision(
                    Decision.REQUIRE_CONFIRMATION,
                    "autonomy 'assisted' requires confirmation for every "
                    "action, including reads",
                    summary,
                )
            return GateDecision(
                Decision.ALLOW,
                f"read action auto-allowed at autonomy '{autonomy.value}'",
                summary,
            )
        if autonomy is Autonomy.AUTONOMOUS_BOUNDED:
            return GateDecision(
                Decision.ALLOW,
                "non-destructive mutation auto-allowed at autonomy "
                "'autonomous_bounded'",
                summary,
            )
        return GateDecision(
            Decision.REQUIRE_CONFIRMATION,
            f"mutating action requires confirmation at autonomy "
            f"'{autonomy.value}'",
            summary,
        )
