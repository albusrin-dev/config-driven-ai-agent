"""Secret REFERENCES, resolved from the environment on demand.

The long-lived config bundle never holds a secret value: ``Secrets`` stores
only provider -> env-var NAME. Presence is validated at load time (early,
clear ``MissingSecretError``), and ``resolve_secret(provider)`` reads the
env var at the point of use (the LLM adapter will call it in a later
phase). The guarantee is structural — the value isn't in the object — so no
``asdict``/``model_dump``/``json.dumps`` of the bundle can ever carry a
key. The redacted-style ``__repr__`` remains as defence in depth.
"""

from __future__ import annotations

import os

from .errors import MissingSecretError
from .models import AgentConfig, SystemConfig


class Secrets:
    """Provider name -> env var NAME. Values are never stored."""

    def __init__(self, references: dict[str, str] | None = None) -> None:
        self._references: dict[str, str] = dict(references or {})

    def references(self) -> dict[str, str]:
        """Copy of the provider -> env-var-name mapping (names only)."""
        return dict(self._references)

    def resolve_secret(self, provider: str) -> str:
        """Read the provider's key from the environment NOW.

        Raises ``KeyError`` for a provider with no reference, and
        ``MissingSecretError`` if the env var is unset/empty at call time.
        """
        env_var = self._references[provider]
        value = os.environ.get(env_var)
        if not value:
            raise MissingSecretError(provider, env_var)
        return value

    def __contains__(self, provider: str) -> bool:
        return provider in self._references

    def providers(self) -> list[str]:
        return sorted(self._references)

    def __repr__(self) -> str:  # names only — there are no values to leak
        return f"Secrets(references={self._references!r}, values=<resolved on demand>)"

    __str__ = __repr__


def resolve_secrets(agent: AgentConfig, system: SystemConfig) -> Secrets:
    """Validate presence of the agent's provider key and build references.

    Fails at load with ``MissingSecretError`` if a required env var is
    unset/empty, but stores only the env var NAME. A provider without
    ``api_key_env`` (keyless local provider) yields no reference.
    """
    provider_name = agent.llm.provider
    provider = system.providers[provider_name]  # existence cross-validated by loader

    references: dict[str, str] = {}
    if provider.api_key_env is not None:
        if not os.environ.get(provider.api_key_env):
            raise MissingSecretError(provider_name, provider.api_key_env)
        references[provider_name] = provider.api_key_env

    # A keyed search provider (Phase 6) follows the same rule: presence
    # validated at load, value read on demand at request time. SearXNG needs
    # no key, so the default path adds nothing.
    search = system.search
    if search is not None and search.api_key_env is not None:
        if not os.environ.get(search.api_key_env):
            raise MissingSecretError(search.provider_name, search.api_key_env)
        references[search.provider_name] = search.api_key_env
    return Secrets(references)
