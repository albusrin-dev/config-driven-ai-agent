"""Effect vocabulary the policy gate reasons over.

Phase 1 shipped the filesystem effect; Phase 6 adds the network effect
alongside its gate rule, exactly as planned. The gate accepts new effect
types structurally (anything subclassing ``Effect``) but DENIES any type it
does not recognise — extensibility stays fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Effect:
    """Base class for all effects. Marker only; carries no behavior."""


class FileMode(str, Enum):
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class FilesystemEffect(Effect):
    """This call will touch ``path`` (absolute) in the given mode."""

    path: str
    mode: FileMode


class NetworkProvenance(str, Enum):
    """Where the URL came from — the basis of Rule 13's provenance floor."""

    USER = "user"          # pasted in the user's own message
    SEARCH = "search"      # returned by a search provider's structured results
    MODEL = "model"        # composed by the model: confirmation floor
    PROVIDER = "provider"  # the configured search endpoint itself


@dataclass(frozen=True)
class NetworkEffect(Effect):
    """This call will reach ``url`` over the network.

    ``provenance`` is the TOOL'S CLAIM and is never trusted on its own: the
    gate re-derives provenance from the session's known URLs and downgrades
    anything it cannot verify to MODEL (fail-closed).
    """

    url: str
    domain: str
    provenance: NetworkProvenance


def describe(effect: Effect) -> str:
    """Short, non-sensitive one-line summary of an effect (for audit).

    Network effects summarize to provenance + domain rather than the full
    URL: a model-composed URL's query string is exactly where exfiltrated
    data would ride, so it does not belong in a routine log line. The full
    URL still reaches the human in the confirmation reason, where it is
    needed to make the decision.
    """
    if isinstance(effect, FilesystemEffect):
        return f"filesystem:{effect.mode.value}:{effect.path}"
    if isinstance(effect, NetworkEffect):
        return f"network:{effect.provenance.value}:{effect.domain}"
    return f"unknown:{type(effect).__name__}"
