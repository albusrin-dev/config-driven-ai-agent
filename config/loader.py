"""Load, validate, and cross-validate config profiles.

Profiles are loaded by NAME only. A name is an opaque identifier (never
branched on, Durable Rule 2) constrained to a safe pattern and resolved to
``{base_dir}/{name}.yaml``; the resolved path is then confined to the base
directory via the shared helper in ``core.paths`` (defence in depth against
traversal). YAML is parsed with ``yaml.safe_load`` only; nothing from a
config file is ever executed (Durable Rule 1).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from core.paths import PathEscapeError, confine, resolve_real

from .errors import ConfigValidationError, ProfileNotFoundError, UnsafeProfileNameError
from .models import (
    AUTONOMY_RANK,
    AgentConfig,
    SystemConfig,
    UserConfig,
)
from .secrets import Secrets, resolve_secrets

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
AGENTS_DIR = PROFILES_DIR / "agents"
SYSTEM_DIR = PROFILES_DIR / "system"
USERS_DIR = PROFILES_DIR / "users"

# Safe profile/environment names: no separators, no dots, no traversal.
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class LoadedConfig:
    """Validated config bundle. Secrets are a separate object holding only
    env-var references — never values — and are never serialized."""

    agent: AgentConfig
    system: SystemConfig
    secrets: Secrets = field(repr=False)
    user: UserConfig | None = None


# --------------------------------------------------------------------------
# Name -> path resolution (A1: safe pattern + confinement)
# --------------------------------------------------------------------------

def _resolve_profile_path(name: str, base_dir: Path) -> Path:
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        raise UnsafeProfileNameError(name)
    try:
        return confine(Path(base_dir) / f"{name}.yaml", base_dir)
    except PathEscapeError:
        raise UnsafeProfileNameError(name) from None


# --------------------------------------------------------------------------
# YAML parsing + schema validation
# --------------------------------------------------------------------------

def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ProfileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)  # data only; never executes config content
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigValidationError(
            str(path), [f"top level: expected a mapping, got {type(data).__name__}"]
        )
    return data


def _format_pydantic_error(err: dict) -> str:
    loc = ".".join(str(p) for p in err["loc"]) or "(top level)"
    msg = err["msg"]
    err_type = err.get("type", "")
    ctx = err.get("ctx") or {}
    if err_type == "enum" and "expected" in ctx:
        msg = f"invalid value; allowed values: {ctx['expected']}"
    elif err_type == "extra_forbidden":
        msg = "unknown field (extra fields are forbidden — check for typos)"
    elif err_type == "missing":
        msg = "required field is missing"
    got = err.get("input")
    if err_type not in ("missing",) and got is not None and not isinstance(got, dict):
        msg += f" (got: {got!r})"
    return f"{loc}: {msg}"


def _validate(model_cls: type[BaseModel], data: dict, source: str) -> BaseModel:
    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        problems = [_format_pydantic_error(err) for err in e.errors()]
        raise ConfigValidationError(source, problems) from None


# --------------------------------------------------------------------------
# Cross-validation (agent vs system)
# --------------------------------------------------------------------------

def _cross_validate(agent: AgentConfig, system: SystemConfig, source: str) -> None:
    problems: list[str] = []

    if agent.llm.provider not in system.providers:
        allowed = sorted(system.providers)
        problems.append(
            f"llm.provider: '{agent.llm.provider}' is not defined in "
            f"system providers; allowed values: {allowed}"
        )

    unknown_overrides = sorted(set(agent.tools.overrides) - set(agent.tools.allowlist))
    if unknown_overrides:
        problems.append(
            f"tools.overrides: keys must be a subset of tools.allowlist; "
            f"not in allowlist: {unknown_overrides}"
        )

    if AUTONOMY_RANK[agent.autonomy] > AUTONOMY_RANK[system.limits.max_autonomy]:
        problems.append(
            f"autonomy: '{agent.autonomy.value}' exceeds the system cap "
            f"limits.max_autonomy='{system.limits.max_autonomy.value}'"
        )

    if problems:
        raise ConfigValidationError(source, problems)


# --------------------------------------------------------------------------
# Public loaders (name-based only)
# --------------------------------------------------------------------------

def load_agent(
    name: str,
    system_config: SystemConfig,
    user_config: UserConfig | None = None,
    base_dir: Path | str | None = None,
) -> LoadedConfig:
    path = _resolve_profile_path(name, Path(base_dir) if base_dir else AGENTS_DIR)
    data = _read_yaml(path)
    agent = _validate(AgentConfig, data, str(path))
    _cross_validate(agent, system_config, str(path))
    secrets = resolve_secrets(agent, system_config)
    return LoadedConfig(agent=agent, system=system_config, secrets=secrets, user=user_config)


def load_system_config(
    env: str | None = None,
    base_dir: Path | str | None = None,
) -> SystemConfig:
    """Load the system config for an environment name.

    ``env`` defaults to ``APP_ENV`` (default ``dev``); either way the name
    is validated against the safe pattern before any filesystem access.
    """
    if env is None:
        env = os.environ.get("APP_ENV", "dev")
    path = _resolve_profile_path(env, Path(base_dir) if base_dir else SYSTEM_DIR)
    data = _read_yaml(path)
    system = _validate(SystemConfig, data, str(path))
    # A relative sandbox root is anchored to the config file's directory —
    # a stable base — never the process CWD, so the sandbox cannot move
    # depending on where the CLI is launched (A4).
    if system.sandbox.fs_root is not None:
        root = Path(system.sandbox.fs_root)
        if not root.is_absolute():
            system.sandbox.fs_root = str(resolve_real(path.parent / root))
    return system


def load_user_config(
    name: str,
    base_dir: Path | str | None = None,
) -> UserConfig:
    path = _resolve_profile_path(name, Path(base_dir) if base_dir else USERS_DIR)
    data = _read_yaml(path)
    return _validate(UserConfig, data, str(path))
