"""Secrets: reference-only at rest, resolved from env on demand, never serialized."""

import json

import pytest

from conftest import FAKE_KEY

from config import MissingSecretError, load_agent


def test_secret_resolves_on_demand(system_config, anthropic_key):
    loaded = load_agent("jarvis", system_config)
    assert loaded.secrets.resolve_secret("anthropic") == FAKE_KEY


def test_missing_secret_env_var_fails_clearly_at_load(system_config, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingSecretError) as exc_info:
        load_agent("jarvis", system_config)
    msg = str(exc_info.value)
    assert "anthropic" in msg
    assert "ANTHROPIC_API_KEY" in msg


def test_empty_secret_env_var_fails(system_config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(MissingSecretError):
        load_agent("jarvis", system_config)


def test_secret_never_in_config_dumps(system_config, anthropic_key):
    """No serialization of any part of the bundle can contain the value."""
    loaded = load_agent("jarvis", system_config)
    dumps = [
        json.dumps(loaded.agent.model_dump(mode="json")),
        loaded.agent.model_dump_json(),
        json.dumps(loaded.system.model_dump(mode="json")),
        loaded.system.model_dump_json(),
        json.dumps(loaded.secrets.references()),
        repr(loaded),
        str(loaded),
        repr(loaded.secrets),
    ]
    for dump in dumps:
        assert FAKE_KEY not in dump


def test_secrets_repr_holds_no_value(system_config, anthropic_key):
    loaded = load_agent("jarvis", system_config)
    assert FAKE_KEY not in repr(loaded.secrets)
    assert "ANTHROPIC_API_KEY" in repr(loaded.secrets)  # the NAME is fine


def test_secrets_not_a_field_on_config_models(system_config, anthropic_key):
    loaded = load_agent("jarvis", system_config)
    assert "secrets" not in loaded.agent.model_fields_set
    assert "secrets" not in type(loaded.agent).model_fields
    assert "secrets" not in type(loaded.system).model_fields


def test_system_config_stores_env_var_name_not_value(system_config, anthropic_key):
    provider = system_config.providers["anthropic"]
    assert provider.api_key_env == "ANTHROPIC_API_KEY"
    dump = system_config.model_dump_json()
    assert "ANTHROPIC_API_KEY" in dump  # the NAME is fine
    assert FAKE_KEY not in dump  # the VALUE never appears
