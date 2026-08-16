"""Standalone Linux verification of the POSIX no-follow / re-verify guards.

Stdlib-only (no venv needed): run `python3 scripts/verify_posix_nofollow.py`
on any Linux host with Python 3.10+. Exercises core/paths.py — the module
both the loader and the filesystem tools rely on for sandbox containment —
against post-authorization symlink swaps, which cannot be tested on a
Windows account without symlink privilege.

Exit code 0 = all guards hold; 1 = a guard failed (a real fail-closed bug).
The pytest suite covers the same cases (tests/test_path_threading.py,
tests/test_resume_replan.py) when symlinks are permitted.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import paths  # noqa: E402

results = []


def check(name, fn):
    try:
        fn()
        results.append(True)
        print(f"PASS {name}")
    except Exception as e:  # noqa: BLE001 — report and continue
        results.append(False)
        print(f"FAIL {name}: {type(e).__name__}: {e}")


TMP = Path(tempfile.mkdtemp())


def fresh():
    d = Path(tempfile.mkdtemp(dir=TMP))
    sandbox = d / "sandbox"
    sandbox.mkdir()
    victim = d / "victim.txt"
    victim.write_text("precious")
    return sandbox, victim


def test_o_nofollow_is_real():
    assert getattr(os, "O_NOFOLLOW", 0) != 0, "O_NOFOLLOW missing on POSIX"


def test_write_swap_fails_closed():
    sandbox, victim = fresh()
    blessed = paths.resolve_real(sandbox / "w.txt")
    os.symlink(victim, sandbox / "w.txt")  # post-check swap
    try:
        with paths.open_no_follow_write(blessed) as f:
            f.write("attack")
        raise AssertionError("open_no_follow_write followed a swapped symlink")
    except (paths.PathResolutionChangedError, OSError):
        pass
    assert victim.read_text() == "precious", "victim file was modified!"


def test_read_swap_fails_closed():
    sandbox, victim = fresh()
    secret = victim.parent / "secret.txt"
    secret.write_text("SECRET")
    target = sandbox / "r.txt"
    target.write_text("public")
    blessed = paths.resolve_real(target)
    target.unlink()
    os.symlink(secret, target)
    try:
        with paths.open_no_follow_read(blessed) as f:
            content = f.read()
        raise AssertionError(f"read followed a swapped symlink: {content!r}")
    except (paths.PathResolutionChangedError, OSError):
        pass


def test_delete_swap_fails_closed():
    sandbox, victim = fresh()
    doomed = sandbox / "doomed.txt"
    doomed.write_text("bye")
    blessed = paths.resolve_real(doomed)
    doomed.unlink()
    os.symlink(victim, doomed)
    try:
        paths.delete_no_follow(blessed)
        raise AssertionError("delete_no_follow acted on a swapped symlink")
    except (paths.PathResolutionChangedError, OSError):
        pass
    assert victim.exists(), "victim file was deleted!"


def test_parent_dir_symlink_swap_fails_closed():
    sandbox, victim = fresh()
    sub = sandbox / "sub"
    sub.mkdir()
    blessed = paths.resolve_real(sub / "w.txt")
    sub.rmdir()
    os.symlink(victim.parent, sub)  # directory component swapped
    try:
        with paths.open_no_follow_write(blessed) as f:
            f.write("attack")
        raise AssertionError("write escaped through a swapped directory symlink")
    except (paths.PathResolutionChangedError, OSError):
        pass
    assert victim.read_text() == "precious"


def test_raw_o_nofollow_refuses_symlink():
    sandbox, victim = fresh()
    link = sandbox / "link.txt"
    os.symlink(victim, link)
    try:
        os.open(str(link), os.O_RDONLY | os.O_NOFOLLOW)
        raise AssertionError("os.open followed a symlink despite O_NOFOLLOW")
    except OSError:
        pass


def test_ordinary_ops_work():
    sandbox, _ = fresh()
    target = paths.resolve_real(sandbox / "normal.txt")
    with paths.open_no_follow_write(target) as f:
        f.write("hello")
    with paths.open_no_follow_read(target) as f:
        assert f.read() == "hello"
    paths.delete_no_follow(target)
    assert not Path(target).exists()


def test_confinement_catches_symlink_escape():
    sandbox, victim = fresh()
    link = sandbox / "esc"
    os.symlink(victim.parent, link)
    assert not paths.is_confined(link / "victim.txt", sandbox)


if __name__ == "__main__":
    if os.name == "nt":
        print("This verification is for POSIX; run it on Linux.")
        sys.exit(2)
    tests = sorted(
        (name, fn) for name, fn in globals().items() if name.startswith("test_")
    )
    for name, fn in tests:
        check(name, fn)
    passed = sum(results)
    print(f"\n{passed} passed, {len(results) - passed} failed")
    sys.exit(0 if all(results) else 1)
