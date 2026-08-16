"""Search client — SearXNG by default.

SearXNG is self-hosted, open-source and needs no API key, so the whole
phase builds and tests with no key and no running instance (the HTTP seam
in ``web.http`` is what tests replace). A keyed provider is a config swap:
``api_key_env`` names the env var and the value is resolved on demand
through ``Secrets`` at request time — never stored on this object, never in
the model's context, never in a dump.

One provider, no port (Rule 7): the abstraction arrives with the second
provider, not in anticipation of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from .http import DEFAULT_TIMEOUT_SECONDS, HttpError, request

if TYPE_CHECKING:
    from config.models import SearchProviderConfig


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchError(Exception):
    """The search could not be completed."""


class SearxngClient:
    def __init__(self, config: SearchProviderConfig,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._config = config
        self._timeout = timeout

    def query_url(self, query: str) -> str:
        """The URL this search will hit — also what the tool declares as its
        network effect, so the gate checks the same target that is used."""
        base = self._config.endpoint.rstrip("/")
        return f"{base}/search?" + urlencode({"q": query, "format": "json"})

    def _api_key(self) -> str | None:
        """Resolved at request time, never retained (SearXNG: none needed)."""
        if self._config.api_key_env is None:
            return None
        from config.secrets import Secrets

        return Secrets(
            {self._config.provider_name: self._config.api_key_env}
        ).resolve_secret(self._config.provider_name)

    def search(self, query: str) -> list[SearchResult]:
        url = self.query_url(query)
        # Resolved per request and dropped with this frame — never stored on
        # the client, never in the model's context. SearXNG returns None.
        key = self._api_key()
        headers = {"Authorization": f"Bearer {key}"} if key else None
        try:
            response = request(url, timeout=self._timeout, headers=headers)
        except HttpError as e:
            raise SearchError(str(e)) from None
        if response.status >= 400:
            raise SearchError(
                f"search provider '{self._config.provider_name}' returned "
                f"HTTP {response.status}"
            )
        try:
            payload = json.loads(response.body.read().decode("utf-8", errors="replace"))
        except (ValueError, AttributeError) as e:
            raise SearchError(
                f"search provider returned a response that is not JSON "
                f"({type(e).__name__}); is the instance's JSON format enabled?"
            ) from None
        return self._map(payload)

    def _map(self, payload: dict) -> list[SearchResult]:
        results = []
        for item in (payload.get("results") or [])[: self._config.max_results]:
            url = (item.get("url") or "").strip()
            if not url:
                continue
            results.append(
                SearchResult(
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=(item.get("content") or "").strip(),
                )
            )
        return results
