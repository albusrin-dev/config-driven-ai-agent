"""Audit records for gate decisions — a logging helper, not a subsystem.

One record per gate decision, emitted through the standard logging
facility. Summaries are redacted: sensitive-looking keys are dropped,
content-like keys and long strings are replaced with length markers.
Results are never audited, so file contents read by a tool can't leak here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger("agent.audit")

# Keys whose values are secret-shaped: never logged at all.
# NOTE (recorded trigger): this redaction is NAME-based, which is safe only
# while no tool takes a credential as a parameter — today the LLM API key
# flows through Secrets.resolve_secret, never through tool params. The
# moment any tool accepts a token/credential param, flip this to an
# allowlist / structured summary instead of name matching.
_SENSITIVE_KEY_PARTS = ("secret", "token", "password", "credential", "api_key", "apikey")
# Keys that carry payloads (file contents etc.): logged as length only.
_PAYLOAD_KEYS = ("content", "data", "body", "text")
_MAX_VALUE_LEN = 80


@dataclass(frozen=True)
class AuditRecord:
    timestamp: str
    agent_name: str
    tool_name: str
    decision: str
    reason: str
    effects_summary: tuple[str, ...]
    params_summary: dict[str, Any]
    # Outcome of an EXECUTED action: "ok" or "error" (None when nothing
    # executed — deny/pending). ``error`` carries the error class/message
    # only, never result contents.
    outcome: str | None = None
    error: str | None = None


def summarize_params(params: BaseModel) -> dict[str, Any]:
    """Redacted view of tool params, safe to log."""
    out: dict[str, Any] = {}
    for key, value in params.model_dump(mode="json").items():
        lowered = key.lower()
        if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
            out[key] = "<redacted>"
        elif isinstance(value, str) and (
            lowered in _PAYLOAD_KEYS or len(value) > _MAX_VALUE_LEN
        ):
            out[key] = f"<str len={len(value)}>"
        else:
            out[key] = value
    return out


def make_record(
    *,
    agent_name: str,
    tool_name: str,
    decision: str,
    reason: str,
    effects_summary: tuple[str, ...],
    params: BaseModel,
    outcome: str | None = None,
    error: str | None = None,
) -> AuditRecord:
    return AuditRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        agent_name=agent_name,
        tool_name=tool_name,
        decision=decision,
        reason=reason,
        effects_summary=effects_summary,
        params_summary=summarize_params(params),
        outcome=outcome,
        error=error,
    )


def emit(record: AuditRecord) -> None:
    logger.info(json.dumps(asdict(record)))
