"""Every required failure mode fails loudly, naming file + field + problem."""

import pytest

from conftest import write_yaml

from config import (
    ConfigValidationError,
    ProfileNotFoundError,
    load_agent,
    load_system_config,
)

VALID_AGENT = """
name: test-agent
version: 1
persona:
  mission: test things
llm:
  provider: anthropic
  model: claude-fable-5
"""


def _load_broken(tmp_path, system_config, yaml_text, name="broken"):
    write_yaml(tmp_path, f"{name}.yaml", yaml_text)
    with pytest.raises(ConfigValidationError) as exc_info:
        load_agent(name, system_config, base_dir=tmp_path)
    err = exc_info.value
    assert f"{name}.yaml" in str(err)  # message names the file
    return err


def test_unknown_extra_field(tmp_path, system_config):
    # 'memory' became a real field in Phase 3, so probe with a name that
    # stays unknown; capability booleans are still rejected (Durable Rule 3).
    err = _load_broken(
        tmp_path, system_config, VALID_AGENT + "\nfilesystem: true\n"
    )
    assert "filesystem" in str(err)
    assert "unknown field" in str(err)


def test_unknown_nested_field_typo(tmp_path, system_config):
    err = _load_broken(
        tmp_path,
        system_config,
        VALID_AGENT.replace("model: claude-fable-5", "model: claude-fable-5\n  temprature: 0.5"),
    )
    assert "llm.temprature" in str(err)


def test_missing_required_field(tmp_path, system_config):
    yaml_text = VALID_AGENT.replace("version: 1\n", "")
    err = _load_broken(tmp_path, system_config, yaml_text)
    assert "version" in str(err)
    assert "required field is missing" in str(err)


def test_missing_persona_mission(tmp_path, system_config):
    yaml_text = VALID_AGENT.replace("  mission: test things\n", "  style: neutral\n")
    err = _load_broken(tmp_path, system_config, yaml_text)
    assert "persona.mission" in str(err)


def test_invalid_autonomy_enum(tmp_path, system_config):
    err = _load_broken(
        tmp_path, system_config, VALID_AGENT + "\nautonomy: full_send\n"
    )
    msg = str(err)
    assert "autonomy" in msg
    assert "allowed values" in msg
    assert "assisted" in msg and "supervised" in msg


def test_invalid_confirm_enum(tmp_path, system_config):
    yaml_text = VALID_AGENT + """
tools:
  allowlist: [web.search]
  overrides:
    web.search:
      confirm: sometimes
"""
    err = _load_broken(tmp_path, system_config, yaml_text)
    msg = str(err)
    assert "confirm" in msg
    assert "allowed values" in msg


def test_invalid_env_enum(tmp_path):
    write_yaml(
        tmp_path,
        "sysbad.yaml",
        """
env: production
providers:
  local: {}
""",
    )
    with pytest.raises(ConfigValidationError) as exc_info:
        load_system_config("sysbad", base_dir=tmp_path)
    msg = str(exc_info.value)
    assert "env" in msg
    assert "allowed values" in msg


def test_temperature_out_of_range(tmp_path, system_config):
    yaml_text = VALID_AGENT.replace(
        "model: claude-fable-5", "model: claude-fable-5\n  temperature: 3.5"
    )
    err = _load_broken(tmp_path, system_config, yaml_text)
    msg = str(err)
    assert "llm.temperature" in msg
    assert "2" in msg  # names the bound


def test_override_not_in_allowlist(tmp_path, system_config):
    yaml_text = VALID_AGENT + """
tools:
  allowlist: [web.search]
  overrides:
    email.send:
      confirm: always
"""
    err = _load_broken(tmp_path, system_config, yaml_text)
    msg = str(err)
    assert "tools.overrides" in msg
    assert "email.send" in msg
    assert "allowlist" in msg


def test_unknown_provider(tmp_path, system_config):
    yaml_text = VALID_AGENT.replace("provider: anthropic", "provider: openai")
    err = _load_broken(tmp_path, system_config, yaml_text)
    msg = str(err)
    assert "llm.provider" in msg
    assert "openai" in msg
    assert "anthropic" in msg  # lists what IS allowed


def test_autonomy_exceeds_system_cap(tmp_path, system_config, anthropic_key):
    # dev system caps max_autonomy at 'supervised'
    yaml_text = VALID_AGENT + "\nautonomy: autonomous_bounded\n"
    err = _load_broken(tmp_path, system_config, yaml_text)
    msg = str(err)
    assert "autonomous_bounded" in msg
    assert "max_autonomy" in msg
    assert "supervised" in msg


def test_duplicate_allowlist_entries(tmp_path, system_config):
    yaml_text = VALID_AGENT + """
tools:
  allowlist: [web.search, web.search]
"""
    err = _load_broken(tmp_path, system_config, yaml_text)
    assert "unique" in str(err)


def test_missing_profile_file(system_config):
    with pytest.raises(ProfileNotFoundError) as exc_info:
        load_agent("does-not-exist", system_config)
    assert "does-not-exist" in str(exc_info.value)


def test_multiple_problems_reported_together(tmp_path, system_config):
    yaml_text = """
name: bad
version: 1
persona:
  mission: x
llm:
  provider: anthropic
  model: m
  temperature: 9.0
autonomy: warp_speed
bogus_field: true
"""
    err = _load_broken(tmp_path, system_config, yaml_text)
    msg = str(err)
    assert "temperature" in msg
    assert "autonomy" in msg
    assert "bogus_field" in msg
