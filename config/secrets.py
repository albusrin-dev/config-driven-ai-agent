"""In-memory secrets, resolved from the environment at load time.

``Secrets`` is deliberately NOT a Pydantic model and is never a field on any
config model, so no config dump/serialization can ever contain a secret
value (Durable Rule 4). Its repr is redacted.
"""

from __future__ import annotations

import os

from .errors import MissingSecretError
from .models import AgentConfig, SystemConfig


class Secrets:
    """Provider name -> resolved API key. In-memory only; never serialized."""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self._values: dict[str, str] = dict(values or {})

    def get(self, provider: str) -> str:
        return self._values[provider]

    def __contains__(self, provider: str) -> bool:
        return provider in self._values

    def providers(self) -> list[str]:
        return sorted(self._values)

    def __repr__(self) -> str:  # never leak values through logging/repr
        return f"Secrets(providers={self.providers()}, values=<redacted>)"

    __str__ = __repr__


def resolve_secrets(agent: AgentConfig, system: SystemConfig) -> Secrets:
    """Resolve the agent's provider key from the environment.

    The provider's ``api_key_env`` names the env var; the value is read here
    and lives only in the returned ``Secrets`` object. A provider without
    ``api_key_env`` (keyless local provider) resolves to no entry.

    Raises ``MissingSecretError`` if a required env var is unset or empty.
    """
    provider_name = agent.llm.provider
    provider = system.providers[provider_name]  # existence cross-validated by loader

    values: dict[str, str] = {}
    if provider.api_key_env is not None:
        value = os.environ.get(provider.api_key_env)
        if not value:
            raise MissingSecretError(provider_name, provider.api_key_env)
        values[provider_name] = value
    return Secrets(values)
