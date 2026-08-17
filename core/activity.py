"""Step-level activity events, for clients that want to show progress.

Purely additive and purely observational: a sink can never change what the
agent does. It cannot raise into the run (every call is wrapped) and it
must not block — the web server's sink only puts an item on a queue.

Not token streaming: one event per model turn and per tool call, which is
enough for an honest "here is what it is doing right now" indicator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger("agent.activity")


@dataclass(frozen=True)
class ActivityEvent:
    kind: str                    # "thinking" | "tool_start" | "tool_end" | "awaiting"
    tool: str | None = None
    detail: str | None = None    # short, human-readable; never file contents
    ok: bool | None = None


ActivitySink = Callable[[ActivityEvent], None]


def emit(sink: ActivitySink | None, event: ActivityEvent) -> None:
    """Deliver an event, swallowing any client-side failure.

    A broken indicator must never break a run — the loop's guarantees are
    about the agent, not about whoever is watching it.
    """
    if sink is None:
        return
    try:
        sink(event)
    except Exception as e:  # noqa: BLE001 — deliberately non-fatal
        logger.debug("activity sink raised %s: %s", type(e).__name__, e)
