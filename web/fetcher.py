"""Bounded, policy-re-checking page fetch.

Every hop — the first request and each redirect — re-runs the internal
target check and the egress check BEFORE connecting. A provenance-approved
URL that redirects to 127.0.0.1 or to a non-allowlisted domain is stopped on
that hop, not after the fact.

Provenance is deliberately NOT re-required per hop: a redirect target is
chosen by the server, not by the model, so it cannot carry model-composed
data outward. The provenance floor exists to stop the model from composing
an exfiltrating URL; the redirect risks are SSRF and egress-policy escape,
and those are exactly what is re-checked here.

Size is capped DURING download (chunked read to a byte ceiling), so an
enormous page never lands in memory, let alone in the conversation.
"""

from __future__ import annotations

from urllib.parse import urljoin

from core.netguard import check_public_target, domain_allowed, url_domain
from core.text import truncate_with_marker

from .html_text import html_to_text
from .http import DEFAULT_TIMEOUT_SECONDS, REDIRECT_STATUSES, HttpError, request

MAX_REDIRECTS = 4
MAX_FETCH_BYTES = 2_000_000   # hard ceiling on what is read off the wire
MAX_PAGE_CHARS = 20_000       # what may enter the conversation
_CHUNK = 65_536


class FetchBlocked(Exception):
    """A hop violated policy (internal target or egress) — refused."""


class FetchFailed(Exception):
    """The fetch could not complete (HTTP error, too many redirects, ...)."""


def _egress_problem(url: str, egress_allowlist, search_domain: str) -> str | None:
    if not egress_allowlist:
        return None
    domain = url_domain(url)
    if domain == search_domain or domain_allowed(domain, egress_allowlist):
        return None
    return (
        f"egress: '{domain}' is not in the egress allowlist "
        f"{sorted(egress_allowlist)} (strict mode)"
    )


def _read_capped(body, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < max_bytes:
        chunk = body.read(min(_CHUNK, max_bytes - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def fetch_text(
    url: str,
    *,
    egress_allowlist=(),
    search_domain: str = "",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = MAX_FETCH_BYTES,
    max_chars: int = MAX_PAGE_CHARS,
) -> tuple[str, str]:
    """Fetch ``url`` and return (final_url, readable_text)."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        # Use-time re-check, every hop, before any connection is made.
        problem = check_public_target(current)
        if problem is not None:
            raise FetchBlocked(f"refused to fetch '{current}': {problem}")
        problem = _egress_problem(current, egress_allowlist, search_domain)
        if problem is not None:
            raise FetchBlocked(f"refused to fetch '{current}': {problem}")

        try:
            response = request(current, timeout=timeout)
        except HttpError as e:
            raise FetchFailed(str(e)) from None

        if response.status in REDIRECT_STATUSES:
            location = response.header("location")
            if not location:
                raise FetchFailed(f"HTTP {response.status} redirect without a Location header")
            current = urljoin(current, location)
            continue

        if response.status >= 400:
            raise FetchFailed(f"HTTP {response.status} fetching '{current}'")

        raw = _read_capped(response.body, max_bytes)
        content_type = response.header("content-type").lower()
        decoded = raw.decode("utf-8", errors="replace")
        text = html_to_text(decoded) if (
            "html" in content_type or not content_type
        ) else decoded
        return current, truncate_with_marker(text, max_chars, "web page")

    raise FetchFailed(f"too many redirects (limit {MAX_REDIRECTS}) starting from '{url}'")
