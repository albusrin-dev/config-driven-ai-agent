"""The egress model: empty = open reads (provenance is the boundary),
non-empty = strict mode. Recorded in CLAUDE.md as a deliberate refinement."""

import io

import pytest

from conftest import make_agent, make_system

from config.models import Autonomy
from core.base import ToolContext
from core.gate import Decision, PolicyGate
from tools.builtins.web import WebFetchTool, WebSearchTool

import core.netguard as netguard
import web.http as web_http

SEARCH = {"provider_name": "searxng", "endpoint": "http://searx.internal:8080"}
APPROVED = "https://example.com/paper"
OTHER = "https://elsewhere.test/page"

gate = PolicyGate()


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(netguard, "resolve_ips", lambda host: ("93.184.216.34",))


def _decide(url, egress=(), user_urls=(APPROVED, OTHER), search=SEARCH):
    system = make_system(egress_allowlist=egress, search=search)
    context = ToolContext(
        agent=make_agent(allowlist=["web_fetch"]),
        system=system,
        user_urls=frozenset(user_urls),
    )
    agent = make_agent(allowlist=["web_fetch"], autonomy=Autonomy.SUPERVISED)
    params = WebFetchTool().input_schema.model_validate({"url": url})
    return gate.evaluate(WebFetchTool(), params, agent, system, context)


# --- Empty allowlist: open reads, gated by provenance ----------------------

def test_empty_allowlist_allows_any_provenance_approved_domain():
    assert _decide(APPROVED).decision is Decision.ALLOW
    assert _decide(OTHER).decision is Decision.ALLOW


def test_empty_allowlist_still_requires_provenance():
    """Open egress is not open season: an unapproved URL still confirms."""
    d = _decide("https://random.test/x", user_urls=(APPROVED,))
    assert d.decision is Decision.REQUIRE_CONFIRMATION


# --- Non-empty allowlist: strict mode --------------------------------------

def test_strict_mode_allows_allowlisted_domain():
    assert _decide(APPROVED, egress=["example.com"]).decision is Decision.ALLOW


def test_strict_mode_denies_non_allowlisted_domain_even_when_user_provided():
    d = _decide(OTHER, egress=["example.com"])
    assert d.decision is Decision.DENY
    assert "egress" in d.reason
    assert "elsewhere.test" in d.reason


def test_strict_mode_allows_subdomain_of_allowlisted_host():
    d = _decide("https://docs.example.com/guide", egress=["example.com"],
                user_urls=("https://docs.example.com/guide",))
    assert d.decision is Decision.ALLOW


def test_strict_mode_does_not_match_lookalike_suffix():
    """'notexample.com' must not pass an 'example.com' allowlist."""
    url = "https://notexample.com/x"
    d = _decide(url, egress=["example.com"], user_urls=(url,))
    assert d.decision is Decision.DENY


def test_search_domain_always_allowed_in_strict_mode():
    tool = WebSearchTool()
    system = make_system(egress_allowlist=["example.com"], search=SEARCH)
    context = ToolContext(agent=make_agent(allowlist=["web_search"]), system=system)
    agent = make_agent(allowlist=["web_search"], autonomy=Autonomy.SUPERVISED)
    params = tool.input_schema.model_validate({"query": "q"})
    assert gate.evaluate(tool, params, agent, system, context).decision is Decision.ALLOW


# --- Redirect re-check at use time -----------------------------------------

def test_redirect_to_non_allowlisted_domain_is_blocked(monkeypatch):
    from web.fetcher import FetchBlocked, fetch_text

    served = []

    def fake_open(url, timeout, headers=None):
        served.append(url)
        if url.startswith("https://example.com"):
            return web_http.RawResponse(
                status=302, headers={"location": "https://elsewhere.test/landing"},
                body=io.BytesIO(b""),
            )
        return web_http.RawResponse(  # pragma: no cover — must not be reached
            status=200, headers={"content-type": "text/html"},
            body=io.BytesIO(b"<html>off-allowlist</html>"),
        )

    monkeypatch.setattr(web_http, "_open", fake_open)

    with pytest.raises(FetchBlocked) as exc_info:
        fetch_text(APPROVED, egress_allowlist=("example.com",))
    assert "egress" in str(exc_info.value)
    assert served == [APPROVED]  # the off-allowlist hop never ran


def test_redirect_within_allowlist_is_followed(monkeypatch):
    from web.fetcher import fetch_text

    def fake_open(url, timeout, headers=None):
        if url.endswith("/start"):
            return web_http.RawResponse(
                status=301, headers={"location": "https://docs.example.com/final"},
                body=io.BytesIO(b""),
            )
        return web_http.RawResponse(
            status=200, headers={"content-type": "text/html"},
            body=io.BytesIO(b"<html><body>arrived</body></html>"),
        )

    monkeypatch.setattr(web_http, "_open", fake_open)
    final_url, text = fetch_text("https://example.com/start",
                                 egress_allowlist=("example.com",))
    assert final_url == "https://docs.example.com/final"
    assert "arrived" in text
