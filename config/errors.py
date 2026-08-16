"""Typed exceptions for configuration loading.

Callers see these instead of raw Pydantic/YAML errors, so every failure
names the file, the field path, the problem, and (where known) the allowed
values.
"""

from __future__ import annotations


class ConfigError(Exception):
    """Base class for all configuration errors."""


class ProfileNotFoundError(ConfigError):
    """A profile file could not be found on disk."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Config file not found: {path}")
        self.path = path


class UnsafeProfileNameError(ConfigError):
    """A profile/environment name is path-like or otherwise unsafe.

    Raised BEFORE any filesystem access. Names must match
    ``^[a-z0-9][a-z0-9_-]*$`` — lowercase letters, digits, hyphen,
    underscore; no separators, no traversal, no absolute paths.
    """

    def __init__(self, name: object) -> None:
        super().__init__(
            f"Unsafe profile/environment name {name!r}: names must match "
            f"^[a-z0-9][a-z0-9_-]*$ (lowercase letters, digits, '-', '_'). "
            f"Path separators, '..', and absolute paths are rejected."
        )
        self.name = name


class ConfigValidationError(ConfigError):
    """A config file failed schema or cross-validation.

    ``problems`` is a list of human-readable strings, each naming the field
    path, what was wrong, and what was allowed.
    """

    def __init__(self, source: str, problems: list[str]) -> None:
        self.source = source
        self.problems = problems
        lines = "\n".join(f"  - {p}" for p in problems)
        super().__init__(f"Invalid config in {source}:\n{lines}")


class MissingSecretError(ConfigError):
    """A required secret environment variable is not set."""

    def __init__(self, provider: str, env_var: str) -> None:
        super().__init__(
            f"Missing secret for provider '{provider}': environment variable "
            f"'{env_var}' is not set. Set it in your environment (see "
            f".env.example); secret values are never stored in config files."
        )
        self.provider = provider
        self.env_var = env_var
