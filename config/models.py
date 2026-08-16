"""Pydantic v2 schemas for agent, user, and system configuration.

All models are strict (``extra="forbid"``): unknown fields are an error, not
a warning (Durable Rule 5). Nothing here reads the environment or touches
disk — parsing/IO lives in the loader, secret resolution in ``secrets.py``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base with fail-closed settings shared by every config model."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class Autonomy(str, Enum):
    ASSISTED = "assisted"
    SUPERVISED = "supervised"
    AUTONOMOUS_BOUNDED = "autonomous_bounded"


# Single source of truth for autonomy ordering; used by the loader's
# ceiling check (agent autonomy must not exceed system max_autonomy).
AUTONOMY_RANK: dict[Autonomy, int] = {
    Autonomy.ASSISTED: 0,
    Autonomy.SUPERVISED: 1,
    Autonomy.AUTONOMOUS_BOUNDED: 2,
}


class Env(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class Confirm(str, Enum):
    ALWAYS = "always"
    NEVER = "never"
    DEFAULT = "default"


class MemoryStrategy(str, Enum):
    NONE = "none"      # raw buffer, exact pre-Phase-3 behavior
    WINDOW = "window"  # head + marker + recent tail compaction


# --------------------------------------------------------------------------
# AgentConfig
# --------------------------------------------------------------------------

class PersonaConfig(StrictModel):
    mission: str = Field(min_length=1)
    style: str = "neutral"
    # Optional richer identity, all consumed by the prompt assembler now.
    tone: str = ""
    communication_style: str = ""


class RoleConfig(StrictModel):
    """Optional role block: what the agent is FOR, fed to the assembler."""

    title: str = ""
    responsibilities: list[str] = Field(default_factory=list)


# Per-model context window, config-tunable. The conservative default suits
# current Claude models; the derived context budget (core.loop) subtracts
# an output reserve and measured overhead from this.
DEFAULT_CONTEXT_WINDOW_TOKENS = 200_000


class LLMConfig(StrictModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    context_window: int = Field(default=DEFAULT_CONTEXT_WINDOW_TOKENS, gt=0)


class ToolOverride(StrictModel):
    confirm: Confirm


class ToolsConfig(StrictModel):
    # Empty allowlist = agent can do nothing (safe default, Durable Rule 3).
    allowlist: list[str] = Field(default_factory=list)
    overrides: dict[str, ToolOverride] = Field(default_factory=dict)

    @field_validator("allowlist")
    @classmethod
    def _allowlist_entries(cls, v: list[str]) -> list[str]:
        for entry in v:
            if not entry or not entry.strip():
                raise ValueError("allowlist entries must be non-empty strings")
        if len(set(v)) != len(v):
            dupes = sorted({e for e in v if v.count(e) > 1})
            raise ValueError(f"allowlist entries must be unique; duplicates: {dupes}")
        return v


class MemoryConfig(StrictModel):
    """Context assembly for what is SENT to the model. The session always
    stores full history (Durable Rule 11).

    ``budget_tokens`` is an explicit override; when None (the default) the
    budget is DERIVED per run: llm.context_window minus an output reserve
    minus the measured fixed overhead (system prompt + tool schemas)."""

    strategy: MemoryStrategy = MemoryStrategy.WINDOW
    budget_tokens: int | None = Field(default=None, gt=0)


class AgentConfig(StrictModel):
    # Opaque identifier — never branched on (Durable Rule 2).
    name: str = Field(min_length=1)
    version: int
    description: str = ""
    persona: PersonaConfig
    role: RoleConfig = Field(default_factory=RoleConfig)
    llm: LLMConfig
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    autonomy: Autonomy = Autonomy.ASSISTED
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


# --------------------------------------------------------------------------
# UserConfig
# --------------------------------------------------------------------------

class UserContext(StrictModel):
    """Optional free-text context about the user, fed to the assembler."""

    about: str = ""


class UserConfig(StrictModel):
    id: str = Field(min_length=1)
    display_name: str = ""
    timezone: str = "UTC"
    context: UserContext = Field(default_factory=UserContext)
    # Free-form data woven into the prompt by the assembler. No logic.
    preferences: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# SystemConfig
# --------------------------------------------------------------------------

class PricingConfig(StrictModel):
    """Optional provider pricing, consumed by the cost budget. If absent,
    the cost cap is inactive (never fabricated) — token/tool-call caps
    still apply."""

    input_usd_per_mtok: float = Field(ge=0)
    output_usd_per_mtok: float = Field(ge=0)


class ProviderConfig(StrictModel):
    endpoint: str | None = None
    # NAME of the env var holding the key — never the key itself
    # (Durable Rule 4). May be omitted for a keyless local provider.
    api_key_env: str | None = None
    pricing: PricingConfig | None = None


class SearchProviderConfig(StrictModel):
    """The one configured search provider (Rule 7: a port arrives with the
    second provider, not before).

    Default is SearXNG — self-hosted, open-source, no API key. A keyed
    provider names its env var in ``api_key_env``; the value is resolved on
    demand at request time via ``Secrets``, exactly like the LLM key, and is
    never stored, never in the model's context, never in a dump.
    """

    provider_name: str = "searxng"
    endpoint: str = Field(min_length=1)
    api_key_env: str | None = None
    # SearXNG returns ~20 results/page and has no API-side cap, so the tool
    # slices to a sane count itself.
    max_results: int = Field(default=8, gt=0)

    def domain(self) -> str:
        from core.netguard import url_domain

        return url_domain(self.endpoint)


class LimitsConfig(StrictModel):
    max_autonomy: Autonomy = Autonomy.SUPERVISED
    max_tokens_per_session: int = Field(default=100_000, gt=0)
    max_cost_per_session_usd: float = Field(default=5.0, ge=0)
    max_tool_calls_per_session: int = Field(default=50, gt=0)


class SandboxConfig(StrictModel):
    fs_root: str | None = None
    denied_paths: list[str] = Field(default_factory=list)
    command_deny_patterns: list[str] = Field(default_factory=list)
    # Phase 6 refinement (deliberate, see CLAUDE.md): for READ-ONLY,
    # provenance-gated fetches, empty = no additional domain restriction;
    # non-empty = strict mode (fetches must also be to an allowlisted
    # domain). The fail-closed empty-means-deny rule is retained exactly
    # where the threat is — any future DATA-CARRYING egress.
    egress_allowlist: list[str] = Field(default_factory=list)


class SystemConfig(StrictModel):
    env: Env = Env.DEV
    providers: dict[str, ProviderConfig]
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    search: SearchProviderConfig | None = None
