"""Rule 13's internal-target floor: web_fetch can never reach inside."""

import io

import pytest

from conftest import make_agent, make_system

from config.models import Autonomy
from core.base import ToolContext
from core.gate import Decision, PolicyGate
from core.netguard import check_public_target
from tools.builtins.web import WebFetchTool, WebSearchTool

import core.netguard as netguard
import web.http as web_http

SEARCH = {"provider_name": "searxng", "endpoint": "http://localhost:8080"}
gate = PolicyGate()

INTERNAL_URLS = [
    "http://localhost/admin",
    "http://127.0.0.1:8080/",
    "https://10.0.0.5/secrets",
    "http://192.168.1.1/router",
    "http://172.16.4.4/",
    "http://169.254.169.254/latest/meta-data/",   # cloud metadata
    "http://[::1]:9000/",                          # IPv6 loopback
    "http://[::ffff:127.0.0.1]/",                  # IPv4-mapped loopback
    "http://0.0.0.0/",
]

NON_HTTP_URLS = [
    "file:///etc/passwd",
    "ftp://example.com/x",
    "gopher://example.com/",
    "data:text/html,<h1>hi</h1>",
]


def _context(system=None, user_urls=(), search_urls=()):
    return ToolContext(
        agent=make_agent(allowlist=["web_fetch", "web_search"]),
        system=system or make_system(search=SEARCH),
        user_urls=frozenset(user_urls),
        search_urls=frozenset(search_urls),
    )


def _decide(url, context, autonomy=Autonomy.AUTONOMOUS_BOUNDED):
    tool = WebFetchTool()
    agent = make_agent(allowlist=["web_fetch"], autonomy=autonomy)
    params = tool.input_schema.model_validate({"url": url})
    return gate.evaluate(tool, params, agent, context.system, context)


# --- The guard itself ------------------------------------------------------

@pytest.mark.parametrize("url", INTERNAL_URLS)
def test_check_public_target_rejects_internal(url, monkeypatch):
    monkeypatch.setattr(netguard, "resolve_ips",
                        lambda host: ("127.0.0.1",) if host == "localhost" else (host,))
    assert check_public_target(url) is not None


@pytest.mark.parametrize("url", NON_HTTP_URLS)
def test_check_public_target_rejects_non_http_scheme(url):
    problem = check_public_target(url)
    assert problem is not None
    assert "scheme" in problem


def test_public_target_passes(monkeypatch):
    monkeypatch.setattr(netguard, "resolve_ips", lambda host: ("93.184.216.34",))
    assert check_public_target("https://example.com/a") is None


def test_dns_failure_fails_closed(monkeypatch):
    def boom(host):
        raise OSError("nxdomain")

    monkeypatch.setattr(netguard, "resolve_ips", boom)
    problem = check_public_target("https://nope.example/")
    assert problem is not None and "fail-closed" in problem


def test_public_name_resolving_to_internal_ip_is_rejected(monkeypatch):
    """DNS rebinding shape: the NAME looks public, the ADDRESS is not."""
    monkeypatch.setattr(netguard, "resolve_ips", lambda host: ("127.0.0.1",))
    problem = check_public_target("https://totally-public.example/")
    assert problem is not None and "loopback" in problem


def test_any_internal_address_in_the_set_rejects(monkeypatch):
    """A host resolving to one public and one internal address is refused."""
    monkeypatch.setattr(netguard, "resolve_ips",
                        lambda host: ("93.184.216.34", "10.1.2.3"))
    assert check_public_target("https://mixed.example/") is not None


# --- The gate floor --------------------------------------------------------

@pytest.mark.parametrize("url", INTERNAL_URLS + NON_HTTP_URLS)
def test_gate_denies_internal_targets_even_when_user_provided(url, monkeypatch):
    """The floor is independent of provenance: even a URL the user pasted
    and maximum autonomy cannot reach inside."""
    monkeypatch.setattr(netguard, "resolve_ips",
                        lambda host: ("127.0.0.1",) if host == "localhost" else (host,))
    d = _decide(url, _context(user_urls=[url]))
    assert d.decision is Decision.DENY
    assert "internal-target floor" in d.reason


def test_gate_denies_internal_target_even_with_confirm_never(monkeypatch):
    monkeypatch.setattr(netguard, "resolve_ips", lambda host: ("127.0.0.1",))
    agent = make_agent(
        allowlist=["web_fetch"],
        overrides={"web_fetch": {"confirm": "never"}},
        autonomy=Autonomy.AUTONOMOUS_BOUNDED,
    )
    context = _context(user_urls=["http://127.0.0.1/"])
    params = WebFetchTool().input_schema.model_validate({"url": "http://127.0.0.1/"})
    d = gate.evaluate(WebFetchTool(), params, agent, context.system, context)
    assert d.decision is Decision.DENY


def test_search_may_reach_the_internal_search_endpoint(monkeypatch):
    """The one exemption: web_search to the CONFIGURED endpoint, which is
    typically a self-hosted SearXNG on localhost."""
    monkeypatch.setattr(netguard, "resolve_ips", lambda host: ("127.0.0.1",))
    tool = WebSearchTool()
    agent = make_agent(allowlist=["web_search"], autonomy=Autonomy.SUPERVISED)
    context = _context()
    params = tool.input_schema.model_validate({"query": "hello"})
    d = gate.evaluate(tool, params, agent, context.system, context)
    assert d.decision is Decision.ALLOW


def test_fetch_may_not_reach_the_search_endpoint(monkeypatch):
    """The exemption is for web_search only — never for web_fetch."""
    monkeypatch.setattr(netguard, "resolve_ips", lambda host: ("127.0.0.1",))
    d = _decide("http://localhost:8080/admin", _context(user_urls=["http://localhost:8080/admin"]))
    assert d.decision is Decision.DENY
    assert "internal-target floor" in d.reason


def test_search_denied_when_no_provider_configured():
    tool = WebSearchTool()
    agent = make_agent(allowlist=["web_search"], autonomy=Autonomy.SUPERVISED)
    context = _context(system=make_system(search=None))
    params = tool.input_schema.model_validate({"query": "x"})
    d = gate.evaluate(tool, params, agent, context.system, context)
    assert d.decision is Decision.DENY
    assert "no search provider" in d.reason


def test_provider_claim_verified_against_config(monkeypatch):
    """An effect claiming PROVIDER but pointing elsewhere is denied — the
    exemption cannot be borrowed to reach another internal host."""
    from core.effects import NetworkEffect, NetworkProvenance

    class ImpostorTool(WebSearchTool):
        def plan_effects(self, params, context):
            return [NetworkEffect(url="http://192.168.0.9/admin",
                                  domain="192.168.0.9",
                                  provenance=NetworkProvenance.PROVIDER)]

    agent = make_agent(allowlist=["web_search"], autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    context = _context()
    params = ImpostorTool().input_schema.model_validate({"query": "x"})
    d = gate.evaluate(ImpostorTool(), params, agent, context.system, context)
    assert d.decision is Decision.DENY
    assert "not the configured search endpoint" in d.reason


# --- Redirects: blocked on the final hop -----------------------------------

def test_redirect_to_internal_address_is_blocked_at_use_time(monkeypatch):
    """A public URL that 302s to 127.0.0.1 must not be followed."""
    from web.fetcher import FetchBlocked, fetch_text

    def resolve(host):
        return ("127.0.0.1",) if host in ("127.0.0.1", "localhost") else ("93.184.216.34",)

    monkeypatch.setattr(netguard, "resolve_ips", resolve)

    served = []

    def fake_open(url, timeout, headers=None):
        served.append(url)
        if url.startswith("https://public.example"):
            return web_http.RawResponse(
                status=302, headers={"location": "http://127.0.0.1/admin"}, body=io.BytesIO(b"")
            )
        return web_http.RawResponse(  # pragma: no cover — must never be reached
            status=200, headers={"content-type": "text/html"},
            body=io.BytesIO(b"<html>internal admin page</html>"),
        )

    monkeypatch.setattr(web_http, "_open", fake_open)

    with pytest.raises(FetchBlocked) as exc_info:
        fetch_text("https://public.example/start")
    assert "loopback" in str(exc_info.value)
    assert served == ["https://public.example/start"]  # the internal hop never ran
