"""The single truncate-with-marker helper (Part A2).

Document extraction (Phase 5) and web fetching (Phase 6) both need the same
size guard, so it lives here once rather than diverging in two tools. The
marker tells the model plainly that it is seeing a prefix, so it can say so
instead of assuming it read the whole thing.
"""

from __future__ import annotations

from typing import Iterable


def truncation_marker(what: str, shown: int, detail: str = "") -> str:
    """'[<what> truncated: showing first N characters<detail>]', newline-led."""
    return f"\n[{what} truncated: showing first {shown} characters{detail}]"


def accumulate_capped(pieces: Iterable[str], cap: int) -> tuple[str, bool, int]:
    """Join text pieces until ``cap`` chars, stopping DURING iteration.

    Bounded by construction: a caller streaming pages/paragraphs/chunks never
    materialises more than the cap. Returns (text, truncated, consumed).
    """
    parts: list[str] = []
    total = 0
    consumed = 0
    truncated = False
    for piece in pieces:
        consumed += 1
        if not piece:
            continue
        room = cap - total
        if room <= 0:
            truncated = True
            consumed -= 1
            break
        if len(piece) > room:
            parts.append(piece[:room])
            total += room
            truncated = True
            break
        parts.append(piece)
        total += len(piece)
    return "\n".join(parts), truncated, consumed


def truncate_with_marker(text: str, cap: int, what: str, detail: str = "") -> str:
    """Cap an already-materialised string, appending the marker if cut."""
    if len(text) <= cap:
        return text
    return text[:cap] + truncation_marker(what, cap, detail)
