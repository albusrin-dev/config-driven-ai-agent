"""Guards for two front-end failure modes that are invisible server-side.

Both of these shipped once and looked like application bugs:

1. A `display:` declaration in the `.dropveil` class rule outranks the
   browser's `[hidden] { display: none }`, so the overlay rendered on every
   page load at `inset: 0` and swallowed every click.
2. Browsers kept serving the old `app.js`/`style.css` from cache, so the
   fix appeared not to work "even on refresh".
"""

import os
import re

import pytest

from server.app import STATIC_DIR, render_console_page

CSS = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def css_block(selector: str) -> str:
    """The declarations of the first rule with exactly this selector."""
    match = re.search(
        r"(?:^|\})\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", CSS, re.MULTILINE
    )
    assert match, f"no CSS rule found for {selector!r}"
    return match.group(1)


# --- the overlay may never cover the page at rest -------------------------

def test_veil_markup_starts_hidden():
    assert re.search(r'<div class="dropveil" id="dropveil" hidden>', HTML)


def test_veil_base_rule_is_display_none():
    """The resting state must be stated in the class rule itself. A bare
    `display: grid` here beats [hidden] and pins the overlay over the UI."""
    assert re.search(r"display:\s*none", css_block(".dropveil"))
    assert not re.search(r"display:\s*grid", css_block(".dropveil"))


def test_visible_state_is_scoped_to_not_hidden():
    assert re.search(r"\.dropveil:not\(\[hidden\]\)\s*\{[^}]*display:\s*grid",
                     CSS)


def test_veil_cannot_intercept_clicks():
    """Second line of defence: an indicator that is somehow left on screen
    still must not swallow a click."""
    assert re.search(r"pointer-events:\s*none", css_block(".dropveil"))


# --- the drag lifecycle must not strand the overlay -----------------------

APP_JS = (STATIC_DIR / "app.js").read_text(encoding="utf-8")


@pytest.mark.parametrize("ending", ["drop", "dragend", "blur"])
def test_every_drag_ending_clears_the_overlay(ending):
    """dragleave does not fire when a drag is cancelled or ends off-window,
    so those endings are handled explicitly."""
    assert re.search(rf'addEventListener\("{ending}"', APP_JS)


def test_enter_and_leave_apply_the_same_file_test():
    """If only one side filters non-file drags, the depth counter drifts
    and the overlay is left up."""
    assert APP_JS.count("draggingFiles(event)") >= 3  # enter, over, leave


# --- an edit must reach the browser on a plain refresh --------------------

def test_console_page_versions_its_assets():
    page = render_console_page()
    assert re.search(r'href="/style\.css\?v=\d+"', page)
    assert re.search(r'src="/app\.js\?v=\d+"', page)


def test_version_changes_when_an_asset_changes():
    before = render_console_page()
    script = STATIC_DIR / "app.js"
    original = script.stat()
    try:
        # touch the asset forward in time, as an edit would
        os.utime(script, (original.st_atime, original.st_mtime + 60))
        after = render_console_page()
    finally:
        os.utime(script, (original.st_atime, original.st_mtime))
    assert before != after, "asset version did not follow the file's mtime"
