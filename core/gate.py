"""The policy gate: the ONLY component that decides allow / deny / confirm.

``PolicyGate.evaluate`` is pure — no side effects, no execution, no
logging — so the full decision matrix is trivially unit-testable. It works
off the abstract Tool interface and never imports a concrete tool.

Decision algorithm (ordered; first terminating rule wins):
  1. Allowlist: tool not in the agent's allowlist -> DENY.
  2. Caps/effects: every planned effect checked against system caps.
     Filesystem effects must resolve inside sandbox.fs_root and outside all
     denied_paths; fs_root unset -> DENY (fail-closed). Network effects are
     checked per Rule 13 (internal-target floor, provenance, egress).
     Unrecognised effect type -> DENY (fail-closed extensibility contract).
  3. Confirmation, precedence: destructive floor > provenance floor >
     per-tool override > autonomy default.

"Pure" here means no mutation and no execution — the gate does perform
read-only RESOLUTION as part of evaluating an effect (realpath for
filesystem containment since Phase 1, DNS for network targets now). That is
the authority-side check: the gate never takes a tool's word for where an
effect will land.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from config.models import Autonomy, Confirm, SandboxConfig

from .base import ToolContext
from .effects import Effect, FilesystemEffect, NetworkEffect, NetworkProvenance, describe
from .netguard import check_public_target, domain_allowed, normalize_url, same_origin
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
    # The blessed effect objects, computed ONCE in evaluate(). The
    # enforcement wrapper threads these into execute so check time and use
    # time share a single path resolution (A1 / TOCTOU).
    effects: tuple[Effect, ...] = ()


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


def _verified_provenance(
    effect: NetworkEffect, context: ToolContext
) -> NetworkProvenance:
    """Re-derive provenance from session state — never trust the label.

    The tool computes a claim; the gate is the authority. Anything it cannot
    verify against the URLs the user actually pasted or a search provider
    actually returned is MODEL, which means the confirmation floor.
    """
    target = normalize_url(effect.url)
    if any(normalize_url(u) == target for u in context.user_urls):
        return NetworkProvenance.USER
    if any(normalize_url(u) == target for u in context.search_urls):
        return NetworkProvenance.SEARCH
    return NetworkProvenance.MODEL


def _check_network_effect(
    effect: NetworkEffect,
    system_config: SystemConfig,
    context: ToolContext,
) -> tuple[str | None, str | None]:
    """Rule 13 evaluation. Returns (deny_reason, confirmation_reason)."""
    search = system_config.search

    # The search provider's own endpoint: the ONE address allowed to be
    # internal (a self-hosted SearXNG typically is), and only for the
    # search tool. The claim is verified against config, not believed.
    if effect.provenance is NetworkProvenance.PROVIDER:
        if search is None:
            return (
                "no search provider is configured (system.search); "
                "search is denied (fail-closed)",
                None,
            )
        if not effect.url or not same_origin(effect.url, search.endpoint):
            return (
                f"effect claims the search provider but targets "
                f"'{effect.domain or effect.url}', which is not the configured "
                f"search endpoint",
                None,
            )
        return (None, None)

    # 1. Internal-target floor (SSRF) — always, independent of provenance,
    # autonomy and allowlist. A fetch may never reach the owner's machine,
    # LAN, or cloud metadata, whoever asked for it.
    problem = check_public_target(effect.url)
    if problem is not None:
        return (
            f"internal-target floor: {problem}; web_fetch may only reach "
            f"public http(s) addresses (Rule 13)",
            None,
        )

    # 3. Egress. Empty allowlist = no domain restriction for these
    # read-only, provenance-gated fetches (see CLAUDE.md: the deliberate
    # refinement); non-empty = strict mode.
    allowlist = system_config.sandbox.egress_allowlist
    if allowlist:
        search_domain = search.domain() if search is not None else ""
        if not domain_allowed(effect.domain, allowlist) and effect.domain != search_domain:
            return (
                f"egress: domain '{effect.domain}' is not in "
                f"sandbox.egress_allowlist {sorted(allowlist)} (strict mode)",
                None,
            )

    # 2. Provenance floor — a confirmation, not a denial: the human sees the
    # full URL and decides (Rule 9 applied to Rule 13).
    provenance = _verified_provenance(effect, context)
    if provenance is NetworkProvenance.MODEL:
        return (
            None,
            f"provenance floor: this URL was composed by the model, not "
            f"provided by you and not returned by a search — confirmation is "
            f"always required, at every autonomy level. Full URL: {effect.url}",
        )
    return (None, None)


class PolicyGate:
    """Stateless decision function (no mutation, no execution)."""

    def evaluate(
        self,
        tool: Tool,
        params: BaseModel,
        agent_config: AgentConfig,
        system_config: SystemConfig,
        context: ToolContext | None = None,
    ) -> GateDecision:
        # 1. Allowlist — the single source of truth for capability.
        if tool.name not in agent_config.tools.allowlist:
            return GateDecision(
                Decision.DENY,
                f"tool '{tool.name}' is not in the agent's allowlist",
            )

        # 2. Caps / effects. plan_effects runs ONCE; the resulting blessed
        # effect objects ride along in the decision so execution uses the
        # exact paths that were confined here (A1 / TOCTOU).
        if context is None:
            context = ToolContext(agent=agent_config, system=system_config)
        blessed = tuple(tool.plan_effects(params, context))
        summary = tuple(describe(e) for e in blessed)
        provenance_confirmation: str | None = None
        for effect in blessed:
            if isinstance(effect, FilesystemEffect):
                problem = _check_filesystem_effect(effect, system_config.sandbox)
                if problem is not None:
                    return GateDecision(Decision.DENY, problem, summary, blessed)
            elif isinstance(effect, NetworkEffect):
                problem, confirmation = _check_network_effect(
                    effect, system_config, context
                )
                if problem is not None:
                    return GateDecision(Decision.DENY, problem, summary, blessed)
                if confirmation is not None:
                    provenance_confirmation = confirmation
            else:
                return GateDecision(
                    Decision.DENY,
                    f"unrecognised effect type '{type(effect).__name__}' — "
                    f"denied (fail-closed)",
                    summary,
                    blessed,
                )

        # 3a. Destructive floor — nothing can lower this.
        if tool.destructive:
            return GateDecision(
                Decision.REQUIRE_CONFIRMATION,
                f"tool '{tool.name}' is destructive: confirmation is always "
                f"required (floor; overrides and autonomy cannot lower it)",
                summary,
                blessed,
            )

        # 3a-bis. Provenance floor (Rule 13) — like the destructive floor,
        # no override and no autonomy level can lower it.
        if provenance_confirmation is not None:
            return GateDecision(
                Decision.REQUIRE_CONFIRMATION,
                provenance_confirmation,
                summary,
                blessed,
            )

        # 3b. Per-tool override.
        override = agent_config.tools.overrides.get(tool.name)
        if override is not None and override.confirm is Confirm.ALWAYS:
            return GateDecision(
                Decision.REQUIRE_CONFIRMATION,
                f"per-tool override confirm=always for '{tool.name}'",
                summary,
                blessed,
            )
        if override is not None and override.confirm is Confirm.NEVER:
            return GateDecision(
                Decision.ALLOW,
                f"per-tool override confirm=never for '{tool.name}'",
                summary,
                blessed,
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
                    blessed,
                )
            return GateDecision(
                Decision.ALLOW,
                f"read action auto-allowed at autonomy '{autonomy.value}'",
                summary,
                blessed,
            )
        if autonomy is Autonomy.AUTONOMOUS_BOUNDED:
            return GateDecision(
                Decision.ALLOW,
                "non-destructive mutation auto-allowed at autonomy "
                "'autonomous_bounded'",
                summary,
                blessed,
            )
        return GateDecision(
            Decision.REQUIRE_CONFIRMATION,
            f"mutating action requires confirmation at autonomy "
            f"'{autonomy.value}'",
            summary,
            blessed,
        )
