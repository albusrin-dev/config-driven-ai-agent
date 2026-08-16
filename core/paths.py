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


# --------------------------------------------------------------------------
# Use-time guards (A1 / TOCTOU): the gate confines a blessed path at
# decision time; these helpers re-verify at the point of use so a symlink
# swapped in between check and use fails closed instead of escaping.
# --------------------------------------------------------------------------

class PathResolutionChangedError(Exception):
    """A blessed path's resolution changed between authorization and use
    (e.g. a component swapped for a symlink) — fail closed."""

    def __init__(self, path: str | os.PathLike, resolved: Path) -> None:
        super().__init__(
            f"path '{path}' no longer resolves to itself (now '{resolved}'): "
            f"resolution changed after authorization; refusing to touch it"
        )
        self.path = str(path)
        self.resolved = str(resolved)


# O_NOFOLLOW exists on POSIX; on Windows the realpath re-check below is the
# guard (symlink creation is privileged there anyway). O_BINARY keeps the
# raw fd untranslated on Windows; the text layer is added by fdopen.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_BINARY = getattr(os, "O_BINARY", 0)


def _reverify(blessed: str | os.PathLike) -> None:
    """The blessed path was fully resolved at decision time; if resolving it
    again yields anything else, something changed underneath us."""
    resolved = resolve_real(blessed)
    if resolved != Path(blessed):
        raise PathResolutionChangedError(blessed, resolved)


def open_no_follow_read(blessed: str | os.PathLike):
    """Open a gate-blessed path for text reading, refusing symlinks."""
    _reverify(blessed)
    fd = os.open(str(blessed), os.O_RDONLY | _O_NOFOLLOW | _O_BINARY)
    return os.fdopen(fd, "r", encoding="utf-8")


def open_no_follow_read_bytes(blessed: str | os.PathLike):
    """Binary twin of ``open_no_follow_read`` (same re-verify + no-follow
    guard) for tools that parse binary formats (PDF, docx)."""
    _reverify(blessed)
    fd = os.open(str(blessed), os.O_RDONLY | _O_NOFOLLOW | _O_BINARY)
    return os.fdopen(fd, "rb")


def open_no_follow_write(blessed: str | os.PathLike):
    """Open a gate-blessed path for text writing (create/truncate),
    refusing symlinks."""
    _reverify(blessed)
    fd = os.open(
        str(blessed),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _O_NOFOLLOW | _O_BINARY,
    )
    return os.fdopen(fd, "w", encoding="utf-8")


def delete_no_follow(blessed: str | os.PathLike) -> None:
    """Unlink a gate-blessed path after re-verifying its resolution."""
    _reverify(blessed)
    os.unlink(blessed)
