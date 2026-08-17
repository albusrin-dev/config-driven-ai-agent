"""Uploads are sandbox writes: confined, size-capped, type-checked."""

import pytest

from server.uploads import (
    ALLOWED_SUFFIXES,
    MAX_UPLOAD_BYTES,
    UploadRejected,
    sanitize_filename,
    store_upload,
)


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (tmp_path / "outside.txt").write_text("do not touch", encoding="utf-8")
    return root


# --- names ----------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("notes.txt", "notes.txt"),
    ("My Report (final).docx", "My Report (final).docx"),
    ("../../escape.pdf", "escape.pdf"),
    ("..\\..\\escape.pdf", "escape.pdf"),
    ("/etc/passwd.txt", "passwd.txt"),
    ("C:\\Windows\\system.md", "system.md"),
    ("....//sneaky.txt", "sneaky.txt"),
])
def test_sanitize_reduces_to_a_bare_name(raw, expected):
    name = sanitize_filename(raw)
    assert name == expected
    assert "/" not in name and "\\" not in name and not name.startswith(".")


def test_nameless_upload_is_refused():
    with pytest.raises(UploadRejected):
        sanitize_filename("   ")
    with pytest.raises(UploadRejected):
        sanitize_filename("..")


# --- the write ------------------------------------------------------------

def test_upload_lands_in_the_sandbox(sandbox):
    stored = store_upload("notes.txt", b"hello", sandbox)
    assert stored.parent == sandbox.resolve()
    assert stored.read_bytes() == b"hello"


@pytest.mark.parametrize("suffix", ALLOWED_SUFFIXES)
def test_readable_types_are_accepted(sandbox, suffix):
    stored = store_upload(f"doc{suffix}", b"data", sandbox)
    assert stored.exists()


def test_traversal_name_cannot_escape(sandbox):
    """The decisive check: the file lands inside, and the outside file is
    untouched."""
    outside = sandbox.parent / "outside.txt"
    stored = store_upload("../../outside.txt", b"OVERWRITTEN", sandbox)
    assert stored.parent == sandbox.resolve()
    assert outside.read_text(encoding="utf-8") == "do not touch"


def test_absolute_path_name_cannot_escape(sandbox):
    stored = store_upload(str(sandbox.parent / "outside.txt"), b"x", sandbox)
    assert stored.parent == sandbox.resolve()


def test_disallowed_type_is_refused(sandbox):
    for name in ("payload.exe", "script.sh", "archive.zip", "noextension"):
        with pytest.raises(UploadRejected) as exc_info:
            store_upload(name, b"x", sandbox)
        assert "Upload one of" in str(exc_info.value)
    assert list(sandbox.iterdir()) == []


def test_oversize_is_refused(sandbox):
    with pytest.raises(UploadRejected) as exc_info:
        store_upload("big.pdf", b"x" * (MAX_UPLOAD_BYTES + 1), sandbox)
    assert "limit is" in str(exc_info.value)
    assert list(sandbox.iterdir()) == []


def test_at_the_size_limit_is_accepted(sandbox):
    stored = store_upload("edge.txt", b"x" * MAX_UPLOAD_BYTES, sandbox)
    assert stored.stat().st_size == MAX_UPLOAD_BYTES
