"""Anthropic adapter for the core LLMPort.

All provider-specific mapping (message block shapes, tool_use parsing,
headers) is sealed in this module — nothing Anthropic-flavored escapes
through the port. The API key is resolved via ``Secrets.resolve_secret``
AT REQUEST TIME and never stored on the adapter (A2's on-demand
resolution; this is its first real consumer). Uses stdlib urllib — no new
dependencies.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from core.llm import LLMPort, LLMResponse, ToolCall, Usage

if TYPE_CHECKING:
    from config.secrets import Secrets

DEFAULT_ENDPOINT = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

# Bounded retry for transient failures (amended Rule 10: bounded wall-clock).
# A constant on purpose — not configurable away. Total attempts = 1 + MAX_RETRIES.
MAX_RETRIES = 2


class LLMRequestError(Exception):
    """The request failed after exhausting the bounded retries."""


def _is_retryable(error: OSError) -> bool:
    """Transient failures only: timeouts, connection errors, throttling,
    server errors. Client errors (4xx other than 429) fail immediately —
    retrying a bad request just burns the bound."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or error.code >= 500
    return True  # TimeoutError, URLError, ConnectionError, ... — all OSError


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Map neutral message dicts (see core.llm) to Anthropic content blocks.

    Consecutive same-role messages are merged, because tool results map to
    user-role blocks and the API wants alternating roles.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "user":
            mapped = {"role": "user",
                      "content": [{"type": "text", "text": message["content"]}]}
        elif role == "assistant":
            content: list[dict[str, Any]] = []
            if message.get("content"):
                content.append({"type": "text", "text": message["content"]})
            for c in message.get("tool_calls", []):
                content.append({"type": "tool_use", "id": c["id"],
                                "name": c["name"], "input": c["params"]})
            mapped = {"role": "assistant", "content": content}
        elif role == "tool_result":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": message["content"],
            }
            if not message.get("ok", True):
                block["is_error"] = True
            mapped = {"role": "user", "content": [block]}
        else:
            raise ValueError(f"unknown message role: {role!r}")

        if out and out[-1]["role"] == mapped["role"]:
            out[-1]["content"].extend(mapped["content"])
        else:
            out.append(mapped)
    return out


class AnthropicAdapter(LLMPort):
    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        secrets: Secrets,
        temperature: float = 0.4,
        system_prompt: str | None = None,
        endpoint: str | None = None,
        max_output_tokens: int = 1024,
        timeout_seconds: float = 120.0,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self._provider_name = provider_name
        self._model = model
        self._secrets = secrets  # holds env-var NAMES only, never values
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._endpoint = endpoint or DEFAULT_ENDPOINT
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds
        self._retry_backoff_seconds = retry_backoff_seconds

    def complete(self, messages: list[dict], tool_schemas: list[dict]) -> LLMResponse:
        # Resolved per request, used for this call, never retained.
        api_key = self._secrets.resolve_secret(self._provider_name)

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_output_tokens,
            "temperature": self._temperature,
            "messages": _to_anthropic_messages(messages),
        }
        if self._system_prompt:
            payload["system"] = self._system_prompt
        if tool_schemas:
            payload["tools"] = [
                {
                    "name": s["name"],
                    "description": s["description"],
                    "input_schema": s["input_schema"],
                }
                for s in tool_schemas
            ]

        data = self._post_with_retries(payload, api_key)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(id=block["id"], name=block["name"],
                             params=block.get("input") or {})
                )
        usage = Usage(
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            output_tokens=data.get("usage", {}).get("output_tokens", 0),
        )
        return LLMResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tuple(tool_calls),
            usage=usage,
            stop_reason=data.get("stop_reason") or "",
        )

    def count_tokens(self, messages: list[dict], tool_schemas: list[dict]) -> int:
        # Local heuristic (~4 chars/token); the provider's count endpoint is
        # a later nicety and nothing currently depends on precision here.
        blob = json.dumps(messages) + json.dumps(tool_schemas)
        return max(1, len(blob) // 4)

    def _post_with_retries(self, payload: dict, api_key: str) -> dict:
        """At most 1 + MAX_RETRIES attempts, backoff between them. Exhausted
        retries raise — the loop turns that into a clean Errored, never a
        hang (the per-request timeout below bounds each attempt)."""
        last_error: OSError | None = None
        for attempt in range(1 + MAX_RETRIES):
            if attempt:
                time.sleep(self._retry_backoff_seconds * attempt)
            try:
                return self._post(payload, api_key)
            except OSError as e:
                if not _is_retryable(e):
                    raise
                last_error = e
        raise LLMRequestError(
            f"request failed after {1 + MAX_RETRIES} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        )

    def _post(self, payload: dict, api_key: str) -> dict:
        """One HTTP POST. Isolated so contract tests mock exactly this seam."""
        request = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": API_VERSION,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
            return json.load(response)
