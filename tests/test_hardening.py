"""Part A hardening: A1 path-traversal safety, A2 secrets are references only."""

import json

import pytest

from conftest import FAKE_KEY

from config import (
    MissingSecretError,
    UnsafeProfileNameError,
    load_agent,
    load_system_config,
    load_user_config,
)

# ---------------------------------------------------------------------------
# A1 — name -> path resolution rejects anything path-like
# ---------------------------------------------------------------------------

MALICIOUS_NAMES = [
    "../users/owner",          # relative traversal
    "/etc/passwd",             # absolute path (posix)
    "C:\\Windows\\system32",   # absolute path (windows)
    "..%2F..%2Fsecrets",       # encoded traversal
    "agents/jarvis",           # name with slash
    "..\\..\\owner",           # backslash traversal
    "..",                      # bare traversal
    "jarvis.yaml",             # dots not allowed — no extension smuggling
    "",                        # empty
]


@pytest.mark.parametrize("bad_name", MALICIOUS_NAMES)
def test_malicious_agent_names_rejected(system_config, bad_name):
    with pytest.raises(UnsafeProfileNameError) as exc_info:
        load_agent(bad_name, system_config)
    msg = str(exc_info.value)
    assert "Unsafe profile" in msg
    assert "^[a-z0-9][a-z0-9_-]*$" in msg  # tells the caller what IS allowed


@pytest.mark.parametrize("bad_name", ["../agents/jarvis", "/etc/passwd"])
def test_malicious_user_names_rejected(bad_name):
    with pytest.raises(UnsafeProfileNameError):
        load_user_config(bad_name)


def test_malicious_app_env_rejected(monkeypatch):
    monkeypatch.setenv("APP_ENV", "../agents/jarvis")
    with pytest.raises(UnsafeProfileNameError):
        load_system_config()


def test_rejected_before_filesystem_access(system_config, tmp_path):
    """The error fires even when the traversal target actually exists."""
    (tmp_path / "victim.yaml").write_text("name: victim\n", encoding="utf-8")
    with pytest.raises(UnsafeProfileNameError):
        load_agent(f"../{tmp_path.name}/victim", system_config, base_dir=tmp_path / "sub")


def test_ordinary_names_still_load(system_config, anthropic_key):
    for name in ("jarvis", "researcher"):
        assert load_agent(name, system_config).agent.name == name


# ---------------------------------------------------------------------------
# A2 — the bundle never holds a secret value (structural guarantee)
# ---------------------------------------------------------------------------

def test_structural_dump_of_bundle_has_no_secret(system_config, anthropic_key):
    loaded = load_agent("jarvis", system_config)
    blob = json.dumps(
        {
            "agent": loaded.agent.model_dump(mode="json"),
            "system": loaded.system.model_dump(mode="json"),
            "user": None,
            "secrets_references": loaded.secrets.references(),
            "secrets_internals": {k: repr(v) for k, v in vars(loaded.secrets).items()},
        }
    )
    assert FAKE_KEY not in blob
    assert "ANTHROPIC_API_KEY" in blob  # only the reference is stored


def test_no_value_retained_after_env_removed(system_config, monkeypatch):
    """Proof the value was never stored: unset the var after load, and it
    is gone — resolution reads the environment, not a cached copy."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    loaded = load_agent("jarvis", system_config)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(MissingSecretError):
        loaded.secrets.resolve_secret("anthropic")


def test_resolve_reads_current_env_value(system_config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "first-value")
    loaded = load_agent("jarvis", system_config)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "rotated-value")
    assert loaded.secrets.resolve_secret("anthropic") == "rotated-value"


def test_presence_still_validated_at_load(system_config, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingSecretError) as exc_info:
        load_agent("jarvis", system_config)
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)
