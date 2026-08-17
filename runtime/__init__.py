"""Runtime: the composition layer that assembles engine parts into sessions.

Sits ABOVE core and below the clients. Wiring config + registry + LLM
adapter + memory adapter is a composition-root concern, so it lives here
rather than in ``core`` — core still imports nothing outward (no tools, no
llm, no memory), exactly as it has since Phase 1.

    ui / server  ->  runtime  ->  core  ->  config.models
                          \\-> tools / llm / memory
"""

from .service import AgentService, PendingView, SessionNotFoundError, TurnUpdate

__all__ = ["AgentService", "PendingView", "SessionNotFoundError", "TurnUpdate"]
