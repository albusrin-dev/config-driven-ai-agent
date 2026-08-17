"""Upload handling: a sandbox write, treated as one.

Dropping a file in the browser writes into the agent's sandbox, so it gets
the same confinement as any other write — sanitize the name, resolve the
destination, and confine it with ``core.paths`` before a single byte is
written. Size and type are capped to what the document tools can actually
read; anything else is refused rather than parked in the sandbox.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.paths import PathEscapeError, confine

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Only what read_file / read_pdf / read_docx can actually open.
ALLOWED_SUFFIXES = (".txt", ".md", ".pdf", ".docx")

# Characters no file name may contain on either platform, plus control
# chars. Ordinary punctuation (parentheses, commas, apostrophes) is left
# alone — people name files "Report (final).docx", and the security
# boundary is the confinement check below, not the character set.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class UploadRejected(Exception):
    """The upload was refused. The message is shown to the user verbatim,
    so it says what to do differently."""


def sanitize_filename(raw: str) -> str:
    """Reduce a client-supplied name to a bare, safe file name.

    Path separators and traversal are stripped, not escaped: a name is a
    name, never a location. The destination is confined again afterwards
    (belt and braces).
    """
    name = (raw or "").strip().replace("\\", "/").split("/")[-1]
    name = _ILLEGAL.sub("_", name)
    name = name.strip().lstrip(".")  # no leading dots: no ".." and no hidden files
    if not name or name in (".", ".."):
        raise UploadRejected("That file needs a name we can use — rename it and try again.")
    return name[:120]


def check_suffix(name: str) -> None:
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadRejected(
            f"'{suffix or 'that type'}' isn't readable by this agent. "
            f"Upload one of: {', '.join(ALLOWED_SUFFIXES)}."
        )


def check_size(size: int) -> None:
    if size > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"That file is {size // 1024 // 1024} MB. The limit is "
            f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB."
        )


def store_upload(raw_name: str, data: bytes, sandbox_root: Path) -> Path:
    """Validate and write into the sandbox. Returns the stored path."""
    name = sanitize_filename(raw_name)
    check_suffix(name)
    check_size(len(data))
    try:
        destination = confine(sandbox_root / name, sandbox_root)
    except PathEscapeError:
        raise UploadRejected(
            "That file name would land outside the agent's folder."
        ) from None
    destination.write_bytes(data)
    return destination
