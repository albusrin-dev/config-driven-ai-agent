"""Happy-path loading: profiles load, defaults apply, loader is name-agnostic."""

from conftest import REPO_ROOT, write_yaml

from config import (
    Autonomy,
    Confirm,
    Env,
    LoadedConfig,
    load_agent,
    load_system_config,
    load_user_config,
)


def test_load_jarvis_by_name(system_config, anthropic_key):
    loaded = load_agent("jarvis", system_config)
    assert isinstance(loaded, LoadedConfig)
    assert loaded.agent.name == "jarvis"
    assert loaded.agent.llm.provider == "anthropic"
    assert loaded.agent.autonomy is Autonomy.SUPERVISED
    assert loaded.agent.tools.overrides["email.draft"].confirm is Confirm.ALWAYS
    assert loaded.system is system_config


def test_loader_is_name_agnostic(system_config, anthropic_key):
    """Two different profiles load through the identical code path."""
    for name in ("jarvis", "researcher"):
        loaded = load_agent(name, system_config)
        assert loaded.agent.name == name


def test_load_with_custom_base_dir(system_config, anthropic_key, tmp_path):
    write_yaml(
        tmp_path,
        "custom.yaml",
        """
name: custom
version: 1
persona:
  mission: prove base_dir works
llm:
  provider: anthropic
  model: claude-fable-5
""",
    )
    loaded = load_agent("custom", system_config, base_dir=tmp_path)
    assert loaded.agent.name == "custom"


def test_defaults_apply(system_config, anthropic_key, tmp_path):
    write_yaml(
        tmp_path,
        "minimal.yaml",
        """
name: minimal
version: 1
persona:
  mission: do the minimum
llm:
  provider: anthropic
  model: claude-fable-5
""",
    )
    loaded = load_agent("minimal", system_config, base_dir=tmp_path)
    a = loaded.agent
    assert a.description == ""
    assert a.persona.style == "neutral"
    assert a.llm.temperature == 0.4
    assert a.tools.allowlist == []  # empty = can do nothing (safe default)
    assert a.tools.overrides == {}
    assert a.autonomy is Autonomy.ASSISTED  # safe default


def test_system_defaults(tmp_path):
    write_yaml(
        tmp_path,
        "sysmin.yaml",
        """
providers:
  local:
    endpoint: http://localhost:11434
""",
    )
    sys_cfg = load_system_config("sysmin", base_dir=tmp_path)
    assert sys_cfg.env is Env.DEV
    assert sys_cfg.limits.max_autonomy is Autonomy.SUPERVISED
    assert sys_cfg.limits.max_tokens_per_session == 100_000
    assert sys_cfg.limits.max_cost_per_session_usd == 5.0
    assert sys_cfg.limits.max_tool_calls_per_session == 50
    assert sys_cfg.sandbox.fs_root is None
    assert sys_cfg.sandbox.egress_allowlist == []  # fail-closed: deny-all


def test_app_env_selects_system_config(monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    sys_cfg = load_system_config()
    assert sys_cfg.env is Env.DEV


def test_load_user_config():
    user = load_user_config("owner")
    assert user.id == "owner"
    assert user.timezone == "UTC"
    assert user.preferences["units"] == "metric"


def test_keyless_local_provider_needs_no_secret(system_config, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    write_yaml(
        tmp_path,
        "local-agent.yaml",
        """
name: local-agent
version: 1
persona:
  mission: run locally
llm:
  provider: local
  model: llama3
""",
    )
    loaded = load_agent("local-agent", system_config, base_dir=tmp_path)
    assert "local" not in loaded.secrets
