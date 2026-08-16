"""Shared path-confinement helper.

Single source of truth for "does this path stay inside that directory?" —
used by the config loader (profile name -> path, A1) and by the policy
gate's filesystem-effect checks. Resolves symlinks and ``..`` via
``os.path.realpath`` before checking containment.
"""

from __future__ import annotations

import os
from pathlib import Path


class PathEscapeError(Exception):
    """A path resolved outside its allowed base directory."""

    def __init__(self, path: str | os.PathLike, base: str | os.PathLike) -> None:
        super().__init__(
            f"path '{path}' resolves outside of the allowed base directory '{base}'"
        )
        self.path = str(path)
        self.base = str(base)


def resolve_real(path: str | os.PathLike) -> Path:
    """Real absolute path: symlinks and '..' resolved. Works for paths that
    don't exist yet (e.g. a file about to be written)."""
    return Path(os.path.realpath(path))


def is_confined(path: str | os.PathLike, base: str | os.PathLike) -> bool:
    """True iff ``path`` (fully resolved) is ``base`` or inside it."""
    return resolve_real(path).is_relative_to(resolve_real(base))


def confine(path: str | os.PathLike, base: str | os.PathLike) -> Path:
    """Resolve ``path`` and assert it stays within ``base``.

    Returns the resolved path; raises ``PathEscapeError`` if it escapes.
    """
    resolved = resolve_real(path)
    if not resolved.is_relative_to(resolve_real(base)):
        raise PathEscapeError(path, base)
    return resolved
