"""AgentService: the orchestration both clients share.

Extracted on contact (Rule 7), not in anticipation: the CLI was the only
client until the web UI arrived, and duplicating session lifecycle inside a
server is exactly the drift this prevents. It is a consolidation — session
creation, a turn, the suspend/approve/resume cycle — and nothing more.

It holds NO policy. Every action still runs through ``enforce_and_run`` via
the loop; a client that relays an approval is supplying the human's answer
to the gate's question, never bypassing the gate. A client with no approver
gets the headless behaviour built in Phase 2: the run SUSPENDS with a
recorded pending action and resumes only when a human decides (Rule 9) —
which is precisely what the web UI's Approve/Deny buttons drive.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from config.errors import ConfigError
from config.loader import AGENTS_DIR, LoadedConfig, load_agent, load_system_config
from config.models import SystemConfig
from core.activity import ActivitySink
from core.errors import UnknownToolError
from core.identity import build_system_prompt
from core.loop import (
    Approver,
    BudgetExceeded,
    Completed,
    Errored,
    Suspended,
    resume,
    run_turn,
)
from core.paths import PathEscapeError, confine
from core.session import Session, new_session

LLMFactory = Callable[[LoadedConfig], Any]

# Long values (a file's contents on a write) are summarized for display —
# the human approving an action needs to see WHAT and WHERE, not scroll a
# payload. Mirrors the audit layer's discipline.
_PARAM_PREVIEW_CHARS = 300


class SessionNotFoundError(KeyError):
    """No session with that id (or it was never started)."""


@dataclass(frozen=True)
class PendingView:
    """A pending action shaped for a human to read and decide on.

    Built from the stored raw params — the same thing the gate will
    re-evaluate on resume. Nothing pre-resolved is stored or shown as
    authoritative, because a blessed path/URL from suspension time must
    never be trusted after the approval gap (A3).
    """

    tool: str
    reason: str
    params: dict[str, Any]


@dataclass(frozen=True)
class TurnUpdate:
    status: str  # completed | awaiting_approval | budget_exceeded | error
    text: str | None = None
    pending: PendingView | None = None
    detail: str | None = None
    budget: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Entry:
    session: Session
    loaded: LoadedConfig
    llm: Any
    memory: Any


def _default_llm_factory(loaded: LoadedConfig):
    from llm.anthropic import AnthropicAdapter

    provider = loaded.system.providers[loaded.agent.llm.provider]
    return AnthropicAdapter(
        provider_name=loaded.agent.llm.provider,
        model=loaded.agent.llm.model,
        secrets=loaded.secrets,          # references only; resolved per request
        temperature=loaded.agent.llm.temperature,
        endpoint=provider.endpoint,
    )


def _summarize_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str) and len(value) > _PARAM_PREVIEW_CHARS:
            out[key] = value[:_PARAM_PREVIEW_CHARS] + f"… (+{len(value) - _PARAM_PREVIEW_CHARS} chars)"
        else:
            out[key] = value
    return out


class AgentService:
    def __init__(
        self,
        system: SystemConfig | None = None,
        registry=None,
        llm_factory: LLMFactory | None = None,
        memory_factory: Callable[[Any], Any] | None = None,
        env: str | None = None,
    ) -> None:
        if registry is None:
            from tools.defaults import build_registry

            registry = build_registry()
        if memory_factory is None:
            from memory import adapter_for

            memory_factory = adapter_for
        self.system = system if system is not None else load_system_config(env)
        self.registry = registry
        self._llm_factory = llm_factory or _default_llm_factory
        self._memory_factory = memory_factory
        self._sessions: dict[str, _Entry] = {}

    # -- agents -------------------------------------------------------------

    def list_agents(self) -> list[dict[str, Any]]:
        """Profiles that can actually run right now.

        A profile whose allowlist names an unregistered tool is left out —
        the same registry cross-check that would refuse it at run time, so
        the picker never offers an agent that cannot start.
        """
        agents = []
        for path in sorted(AGENTS_DIR.glob("*.yaml")):
            try:
                loaded = load_agent(path.stem, self.system)
                self.registry.toolset_for(loaded.agent)
            except (ConfigError, UnknownToolError):
                continue
            agents.append(
                {
                    "name": loaded.agent.name,
                    "description": loaded.agent.description,
                    "autonomy": loaded.agent.autonomy.value,
                    "tools": list(loaded.agent.tools.allowlist),
                }
            )
        return agents

    # -- session lifecycle --------------------------------------------------

    def start_session(self, agent_name: str) -> str:
        loaded = load_agent(agent_name, self.system)
        self.registry.toolset_for(loaded.agent)  # fail loudly now, not mid-turn
        session = new_session(
            loaded.agent.name,
            system_prompt=build_system_prompt(loaded.agent, loaded.user),
        )
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _Entry(
            session=session,
            loaded=loaded,
            llm=self._llm_factory(loaded),
            memory=self._memory_factory(loaded.agent.memory.strategy),
        )
        return session_id

    def _entry(self, session_id: str) -> _Entry:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFoundError(session_id) from None

    def session(self, session_id: str) -> Session:
        return self._entry(session_id).session

    def agent_name(self, session_id: str) -> str:
        return self._entry(session_id).loaded.agent.name

    def sandbox_root(self, session_id: str) -> Path:
        """The session's sandbox. A config with no sandbox has nowhere to
        put a file — that is a refusal, not a default (fail-closed)."""
        entry = self._entry(session_id)
        fs_root = entry.loaded.system.sandbox.fs_root
        if fs_root is None:
            raise PathEscapeError("(file)", "(no sandbox configured)")
        root = Path(fs_root)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def confine_to_sandbox(self, session_id: str, name: str) -> Path:
        """Resolve a file name inside the session's sandbox, or raise."""
        root = self.sandbox_root(session_id)
        return confine(root / name, root)

    # -- turns --------------------------------------------------------------

    def send(
        self,
        session_id: str,
        message: str,
        on_activity: ActivitySink | None = None,
        approver: Approver | None = None,
    ) -> TurnUpdate:
        entry = self._entry(session_id)
        result = run_turn(
            entry.session, message, entry.llm, self.registry,
            entry.loaded.agent, entry.loaded.system,
            approver=approver, memory=entry.memory, on_activity=on_activity,
        )
        return self._to_update(entry, result)

    def resolve_pending(
        self,
        session_id: str,
        approved: bool,
        on_activity: ActivitySink | None = None,
        approver: Approver | None = None,
    ) -> TurnUpdate:
        """Deliver the human's decision to a suspended run (Rule 9).

        On approval the action goes back through the FULL pipeline — fresh
        effect planning, fresh gate evaluation — so a decision made minutes
        ago cannot carry a stale blessing into a changed world (A3).
        """
        entry = self._entry(session_id)
        result = resume(
            entry.session, approved, entry.llm, self.registry,
            entry.loaded.agent, entry.loaded.system,
            approver=approver, memory=entry.memory, on_activity=on_activity,
        )
        return self._to_update(entry, result)

    def pending_view(self, session_id: str) -> PendingView | None:
        """The outstanding decision, if any. Lives on the session, not on a
        socket, so a client that reconnects still finds it waiting."""
        return self._pending_view(self._entry(session_id))

    def status(self, session_id: str) -> dict[str, Any]:
        entry = self._entry(session_id)
        pending = self._pending_view(entry)
        return {
            "session_id": session_id,
            "agent": entry.loaded.agent.name,
            "autonomy": entry.loaded.agent.autonomy.value,
            "state": entry.session.status,
            "budget": self._budget(entry),
            "pending": None if pending is None else {
                "tool": pending.tool,
                "reason": pending.reason,
                "params": pending.params,
            },
        }

    # -- shaping ------------------------------------------------------------

    @staticmethod
    def _pending_view(entry: _Entry) -> PendingView | None:
        pending = entry.session.pending_action
        if pending is None:
            return None
        return PendingView(
            tool=pending.call.name,
            reason=pending.reason,
            params=_summarize_params(pending.call.params),
        )

    @staticmethod
    def _budget(entry: _Entry) -> dict[str, Any]:
        budget = entry.session.budget
        return {
            "tokens": budget.tokens_used,
            "tool_calls": budget.tool_calls_made,
            "cost_usd": round(budget.cost_used, 4),
        }

    def _to_update(self, entry: _Entry, result) -> TurnUpdate:
        budget = self._budget(entry)
        if isinstance(result, Completed):
            return TurnUpdate("completed", text=result.text, budget=budget)
        if isinstance(result, Suspended):
            return TurnUpdate("awaiting_approval",
                              pending=self._pending_view(entry), budget=budget)
        if isinstance(result, BudgetExceeded):
            return TurnUpdate(
                "budget_exceeded",
                detail=f"this session used up its '{result.which}' budget",
                budget=budget,
            )
        if isinstance(result, Errored):
            return TurnUpdate("error", detail=result.reason, budget=budget)
        raise AssertionError(f"unknown turn result: {type(result).__name__}")
