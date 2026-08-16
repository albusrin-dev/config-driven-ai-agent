"""Tool mechanics: SearXNG JSON parsing, HTML->text, caps, timeouts, errors.
All against the mocked HTTP seam — no running instance, no key, no network."""

import io
import json

import pytest

from conftest import make_agent, make_system

from config.models import Autonomy, SearchProviderConfig
from core.base import ToolContext
from core.enforce import Executed, enforce_and_run
from core.text import accumulate_capped, truncate_with_marker
from tools.builtins.web import WebFetchTool, WebSearchTool
from web.fetcher import MAX_PAGE_CHARS, FetchFailed, fetch_text
from web.html_text import html_to_text
from web.search import SearchError, SearxngClient

import core.netguard as netguard
import web.http as web_http

SEARCH = {"provider_name": "searxng", "endpoint": "http://localhost:8080"}
PAGE_URL = "https://example.com/article"


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(
        netguard, "resolve_ips",
        lambda host: ("127.0.0.1",) if host in ("localhost", "127.0.0.1")
        else ("93.184.216.34",),
    )


def _serve(monkeypatch, body: bytes, status=200, content_type="text/html", capture=None):
    def fake_open(url, timeout, headers=None):
        if capture is not None:
            capture.append({"url": url, "timeout": timeout, "headers": headers})
        return web_http.RawResponse(
            status=status, headers={"content-type": content_type}, body=io.BytesIO(body)
        )

    monkeypatch.setattr(web_http, "_open", fake_open)


def _context(system=None, user_urls=(PAGE_URL,)):
    return ToolContext(
        agent=make_agent(allowlist=["web_search", "web_fetch"]),
        system=system or make_system(search=SEARCH),
        user_urls=frozenset(user_urls),
    )


def _run(tool, params, context, autonomy=Autonomy.SUPERVISED):
    agent = make_agent(allowlist=["web_search", "web_fetch"], autonomy=autonomy)
    return enforce_and_run(tool, params, agent, context.system, context=context)


# --- SearXNG JSON ----------------------------------------------------------

SEARXNG_PAYLOAD = {
    "results": [
        {"title": f"Result {i}", "url": f"https://site{i}.test/a", "content": f"snippet {i}"}
        for i in range(20)  # SearXNG returns ~20/page; the tool slices
    ]
}


def test_search_parses_results_and_slices(monkeypatch):
    _serve(monkeypatch, json.dumps(SEARXNG_PAYLOAD).encode(), content_type="application/json")
    config = SearchProviderConfig(endpoint="http://localhost:8080", max_results=5)
    results = SearxngClient(config).search("quantum")
    assert len(results) == 5
    assert results[0].title == "Result 0"
    assert results[0].url == "https://site0.test/a"
    assert results[0].snippet == "snippet 0"


def test_search_query_url_requests_json_format():
    config = SearchProviderConfig(endpoint="http://localhost:8080/")
    url = SearxngClient(config).query_url("a b")
    assert url.startswith("http://localhost:8080/search?")
    assert "format=json" in url and "q=a+b" in url


def test_search_tool_reports_results_and_discovers_urls(monkeypatch):
    _serve(monkeypatch, json.dumps(SEARXNG_PAYLOAD).encode(), content_type="application/json")
    outcome = _run(WebSearchTool(), {"query": "quantum"}, _context())
    assert isinstance(outcome, Executed) and outcome.result.ok
    assert "https://site0.test/a" in outcome.result.output
    assert outcome.result.discovered_urls[0] == "https://site0.test/a"
    assert len(outcome.result.discovered_urls) == 8  # config default max_results


def test_search_requires_no_api_key(monkeypatch):
    """SearXNG path sends no Authorization header at all."""
    seen = []
    _serve(monkeypatch, b'{"results": []}', content_type="application/json", capture=seen)
    SearxngClient(SearchProviderConfig(endpoint="http://localhost:8080")).search("x")
    assert seen[0]["headers"] is None


def test_keyed_provider_resolves_key_on_demand(monkeypatch):
    seen = []
    _serve(monkeypatch, b'{"results": []}', content_type="application/json", capture=seen)
    monkeypatch.setenv("TEST_SEARCH_KEY", "sk-search-123")
    config = SearchProviderConfig(provider_name="keyed", endpoint="https://api.search.test",
                                  api_key_env="TEST_SEARCH_KEY")
    client = SearxngClient(config)
    client.search("x")
    assert seen[0]["headers"]["Authorization"] == "Bearer sk-search-123"
    # The value is never retained on the client.
    assert "sk-search-123" not in json.dumps({k: repr(v) for k, v in vars(client).items()})


def test_search_non_json_response_is_a_clean_error(monkeypatch):
    _serve(monkeypatch, b"<html>not json</html>")
    with pytest.raises(SearchError) as exc_info:
        SearxngClient(SearchProviderConfig(endpoint="http://localhost:8080")).search("x")
    assert "JSON format" in str(exc_info.value)


def test_search_http_error_is_a_clean_tool_error(monkeypatch):
    _serve(monkeypatch, b"nope", status=503, content_type="text/plain")
    outcome = _run(WebSearchTool(), {"query": "x"}, _context())
    assert isinstance(outcome, Executed) and not outcome.result.ok
    assert "web_search failed" in outcome.result.error


# --- HTML -> text ----------------------------------------------------------

def test_html_to_text_strips_markup_and_scripts():
    html = """
    <html><head><title>T</title><style>body{color:red}</style></head>
    <body><h1>Heading</h1><p>First &amp; second.</p>
    <script>alert('x'); fetch('https://evil.test')</script>
    <p>Third</p></body></html>
    """
    text = html_to_text(html)
    assert "Heading" in text and "First & second." in text and "Third" in text
    assert "alert" not in text and "color:red" not in text
    assert "<" not in text and ">" not in text


def test_fetch_returns_text_not_html(monkeypatch):
    _serve(monkeypatch, b"<html><body><p>Readable body</p></body></html>")
    outcome = _run(WebFetchTool(), {"url": PAGE_URL}, _context())
    assert isinstance(outcome, Executed) and outcome.result.ok
    assert "Readable body" in outcome.result.output
    assert "<p>" not in outcome.result.output


# --- Size caps -------------------------------------------------------------

def test_oversized_page_truncated_with_marker(monkeypatch):
    body = b"<html><body>" + (b"word " * 20_000) + b"</body></html>"
    _serve(monkeypatch, body)
    _, text = fetch_text(PAGE_URL)
    assert "[web page truncated: showing first" in text
    assert len(text) < MAX_PAGE_CHARS + 200


def test_byte_ceiling_stops_reading_early(monkeypatch):
    """The cap applies DURING download: a huge body is not read whole."""
    read_calls = []

    class CountingBody:
        def __init__(self):
            self.remaining = 50_000_000

        def read(self, n):
            read_calls.append(n)
            chunk = b"x" * min(n, self.remaining)
            self.remaining -= len(chunk)
            return chunk

    monkeypatch.setattr(
        web_http, "_open",
        lambda url, timeout, headers=None: web_http.RawResponse(
            status=200, headers={"content-type": "text/plain"}, body=CountingBody()
        ),
    )
    _, text = fetch_text(PAGE_URL, max_bytes=100_000)
    assert sum(read_calls) <= 100_000 + 65_536
    assert len(text) <= MAX_PAGE_CHARS + 200


def test_shared_truncation_helper_is_used_by_both():
    """Part A2: one helper, used by documents and web alike."""
    import tools.builtins.documents as documents
    import web.fetcher as fetcher

    assert documents._accumulate is accumulate_capped
    assert fetcher.truncate_with_marker is truncate_with_marker
    text, truncated, consumed = accumulate_capped(["abc", "def"], 4)
    assert (text, truncated, consumed) == ("abc\nd", True, 2)


# --- Wall-clock: timeout + bounded retry -----------------------------------

def test_timeout_exhausts_bounded_retries_then_fails_cleanly(monkeypatch):
    attempts = []

    def always_timeout(url, timeout, headers=None):
        attempts.append(url)
        raise TimeoutError("simulated stalled socket")

    monkeypatch.setattr(web_http, "_open", always_timeout)
    monkeypatch.setattr(web_http.time, "sleep", lambda s: None)
    with pytest.raises(FetchFailed) as exc_info:
        fetch_text(PAGE_URL)
    assert len(attempts) == 1 + web_http.MAX_RETRIES
    assert "TimeoutError" in str(exc_info.value)


def test_hanging_fetch_becomes_a_clean_tool_error(monkeypatch):
    monkeypatch.setattr(
        web_http, "_open",
        lambda url, timeout, headers=None: (_ for _ in ()).throw(TimeoutError("stalled")),
    )
    monkeypatch.setattr(web_http.time, "sleep", lambda s: None)
    outcome = _run(WebFetchTool(), {"url": PAGE_URL}, _context())
    assert isinstance(outcome, Executed) and not outcome.result.ok
    assert "web_fetch failed" in outcome.result.error


def test_transient_failure_recovers_within_bound(monkeypatch):
    attempts = []

    def flaky(url, timeout, headers=None):
        attempts.append(url)
        if len(attempts) < 2:
            raise ConnectionResetError("reset")
        return web_http.RawResponse(
            status=200, headers={"content-type": "text/html"},
            body=io.BytesIO(b"<html><body>recovered</body></html>"),
        )

    monkeypatch.setattr(web_http, "_open", flaky)
    monkeypatch.setattr(web_http.time, "sleep", lambda s: None)
    _, text = fetch_text(PAGE_URL)
    assert "recovered" in text and len(attempts) == 2


def test_explicit_timeout_is_passed_to_the_transport(monkeypatch):
    seen = []
    _serve(monkeypatch, b"<html>ok</html>", capture=seen)
    fetch_text(PAGE_URL, timeout=7.5)
    assert seen[0]["timeout"] == 7.5


# --- Error handling --------------------------------------------------------

def test_http_404_is_a_clean_error(monkeypatch):
    _serve(monkeypatch, b"missing", status=404, content_type="text/plain")
    outcome = _run(WebFetchTool(), {"url": PAGE_URL}, _context())
    assert isinstance(outcome, Executed) and not outcome.result.ok
    assert "HTTP 404" in outcome.result.error


def test_redirect_loop_is_bounded(monkeypatch):
    monkeypatch.setattr(
        web_http, "_open",
        lambda url, timeout, headers=None: web_http.RawResponse(
            status=302, headers={"location": "https://example.com/next"}, body=io.BytesIO(b"")
        ),
    )
    with pytest.raises(FetchFailed) as exc_info:
        fetch_text(PAGE_URL)
    assert "too many redirects" in str(exc_info.value)


def test_final_url_reported_when_redirected(monkeypatch):
    def fake_open(url, timeout, headers=None):
        if url.endswith("/article"):
            return web_http.RawResponse(
                status=301, headers={"location": "https://example.com/final"},
                body=io.BytesIO(b""),
            )
        return web_http.RawResponse(
            status=200, headers={"content-type": "text/html"},
            body=io.BytesIO(b"<html><body>done</body></html>"),
        )

    monkeypatch.setattr(web_http, "_open", fake_open)
    outcome = _run(WebFetchTool(), {"url": PAGE_URL}, _context())
    assert outcome.result.ok
    assert "[fetched https://example.com/final]" in outcome.result.output


def test_web_tools_are_read_only():
    assert not WebSearchTool.mutating and not WebSearchTool.destructive
    assert not WebFetchTool.mutating and not WebFetchTool.destructive
