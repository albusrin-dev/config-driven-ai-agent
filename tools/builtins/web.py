"""Built-in web tools: web_search, web_fetch — read-only.

Both declare a NetworkEffect and reach the network only through the gate.
Everything they return is untrusted data (Rule 12): a page is written by a
stranger, and any instruction inside it is text to reason about, never a
command to obey.

``web_fetch`` executes against the URL blessed in the gate-evaluated effect
— the same check-time/use-time discipline the filesystem tools use for
paths — and the fetcher re-checks the target on every hop.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from core.base import Tool, ToolContext, ToolResult
from core.effects import Effect, NetworkEffect, NetworkProvenance
from core.netguard import normalize_url, url_domain
from web.fetcher import FetchBlocked, FetchFailed, fetch_text
from web.search import SearchError, SearxngClient


class _SearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)


class _FetchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=1)


def _blessed_url(effects: list[Effect]) -> str:
    for effect in effects:
        if isinstance(effect, NetworkEffect):
            return effect.url
    raise ValueError(
        "execute requires the gate-evaluated NetworkEffect; "
        "it must never run outside enforce_and_run"
    )


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for a query and return a list of results "
        "(title, url, snippet). Read-only."
    )
    input_schema = _SearchParams
    mutating = False
    destructive = False

    def plan_effects(self, params: _SearchParams, context: ToolContext) -> list[Effect]:
        search = context.system.search
        if search is None:
            # No provider configured: declare an empty provider effect and
            # let the gate deny it (fail-closed, one decision point).
            return [NetworkEffect(url="", domain="",
                                  provenance=NetworkProvenance.PROVIDER)]
        url = SearxngClient(search).query_url(params.query)
        return [NetworkEffect(url=url, domain=url_domain(url),
                              provenance=NetworkProvenance.PROVIDER)]

    def execute(
        self, params: _SearchParams, context: ToolContext, effects: list[Effect]
    ) -> ToolResult:
        search = context.system.search
        if search is None:  # pragma: no cover — gate denies first
            return ToolResult(ok=False, error="web_search failed: no search provider configured")
        try:
            results = SearxngClient(search).search(params.query)
        except SearchError as e:
            return ToolResult(ok=False, error=f"web_search failed: {e}")
        if not results:
            return ToolResult(ok=True, output="no results", discovered_urls=())
        lines = [
            f"{i}. {r.title}\n   {r.url}\n   {r.snippet}"
            for i, r in enumerate(results, start=1)
        ]
        return ToolResult(
            ok=True,
            output="\n".join(lines),
            # Structured provider results — the ONLY channel that grants
            # 'search' provenance for a later fetch (Rule 13).
            discovered_urls=tuple(r.url for r in results),
        )


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch a public http(s) web page and return its readable text. "
        "Read-only. A URL you were not given by the user and did not get "
        "from a search result requires the user's confirmation."
    )
    input_schema = _FetchParams
    mutating = False
    destructive = False

    def plan_effects(self, params: _FetchParams, context: ToolContext) -> list[Effect]:
        url = normalize_url(params.url)
        # A CLAIM only: the gate re-derives provenance from session state and
        # downgrades anything it cannot verify (see core/gate.py).
        if any(normalize_url(u) == url for u in context.user_urls):
            provenance = NetworkProvenance.USER
        elif any(normalize_url(u) == url for u in context.search_urls):
            provenance = NetworkProvenance.SEARCH
        else:
            provenance = NetworkProvenance.MODEL
        return [NetworkEffect(url=url, domain=url_domain(url), provenance=provenance)]

    def execute(
        self, params: _FetchParams, context: ToolContext, effects: list[Effect]
    ) -> ToolResult:
        url = _blessed_url(effects)
        search = context.system.search
        try:
            final_url, text = fetch_text(
                url,
                egress_allowlist=tuple(context.system.sandbox.egress_allowlist),
                search_domain=search.domain() if search is not None else "",
            )
        except FetchBlocked as e:
            return ToolResult(ok=False, error=f"web_fetch refused: {e}")
        except FetchFailed as e:
            return ToolResult(ok=False, error=f"web_fetch failed: {e}")
        header = f"[fetched {final_url}]\n" if final_url != url else ""
        return ToolResult(ok=True, output=header + text)
