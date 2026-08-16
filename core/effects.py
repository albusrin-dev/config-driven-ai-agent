"""Effect vocabulary the policy gate reasons over.

Phase 1 ships the filesystem effect only. The gate accepts new effect
types structurally (anything subclassing ``Effect``) but DENIES any type it
does not recognise — extensibility is fail-closed. A future network effect
would be added here alongside its gate rule, in the phase that enforces it.
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


def describe(effect: Effect) -> str:
    """Short, non-sensitive one-line summary of an effect (for audit)."""
    if isinstance(effect, FilesystemEffect):
        return f"filesystem:{effect.mode.value}:{effect.path}"
    return f"unknown:{type(effect).__name__}"
