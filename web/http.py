"""The one HTTP seam, bounded per Rule 10.

Every outbound request in the project goes through ``request`` here: an
explicit timeout, at most ``1 + MAX_RETRIES`` attempts for transient
failures, and redirects NEVER followed automatically (the fetcher follows
them by hand so each hop can be re-checked). Tests monkeypatch ``_open``.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

USER_AGENT = "config-driven-ai-agent/0.1 (+read-only research agent)"
DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_RETRIES = 2  # constant on purpose — Rule 10 is not configurable away
REDIRECT_STATUSES = (301, 302, 303, 307, 308)


class HttpError(Exception):
    """The request failed (after exhausting the bounded retries, if retryable)."""


@dataclass
class RawResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: object = None  # anything with .read(n) -> bytes

    def header(self, name: str, default: str = "") -> str:
        return self.headers.get(name.lower(), default)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface 3xx to the caller instead of following it silently."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def _is_retryable(error: OSError) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or error.code >= 500
    return True  # timeouts, connection resets, DNS hiccups


def _open(url: str, timeout: float, headers: dict[str, str] | None = None) -> RawResponse:
    """One HTTP GET. The seam contract tests replace."""
    request_obj = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,text/plain;q=0.9",
            **(headers or {}),
        },
        method="GET",
    )
    try:
        response = _opener.open(request_obj, timeout=timeout)
    except urllib.error.HTTPError as e:
        # 3xx arrives here because redirects are disabled; so do 4xx/5xx.
        return RawResponse(
            status=e.code,
            headers={k.lower(): v for k, v in (e.headers or {}).items()},
            body=e,
        )
    return RawResponse(
        status=getattr(response, "status", 200) or 200,
        headers={k.lower(): v for k, v in response.headers.items()},
        body=response,
    )


def request(url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS,
            headers: dict[str, str] | None = None,
            retry_backoff_seconds: float = 0.5) -> RawResponse:
    """GET with bounded retries. Exhausted retries raise — never hang.

    ``headers`` may carry a caller-resolved credential (a keyed search
    provider's key); it is passed straight to the request and never stored.
    """
    last_error: OSError | None = None
    for attempt in range(1 + MAX_RETRIES):
        if attempt:
            time.sleep(retry_backoff_seconds * attempt)
        try:
            return _open(url, timeout, headers)
        except OSError as e:
            if not _is_retryable(e):
                raise HttpError(f"request to {url} failed: {type(e).__name__}: {e}") from None
            last_error = e
    raise HttpError(
        f"request to {url} failed after {1 + MAX_RETRIES} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )
