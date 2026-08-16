import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SystemConfig, load_system_config  # noqa: E402
from config.models import AgentConfig, Autonomy  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

FAKE_KEY = "sk-test-fake-key-12345"


@pytest.fixture
def system_config() -> SystemConfig:
    return load_system_config("dev")


@pytest.fixture
def anthropic_key(monkeypatch) -> str:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    return FAKE_KEY


def write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# Factories for gate/enforce tests: build configs directly (no YAML/loader).

def make_agent(
    allowlist=(),
    overrides=None,
    autonomy: Autonomy = Autonomy.SUPERVISED,
    name: str = "tester",
) -> AgentConfig:
    return AgentConfig(
        name=name,
        version=1,
        persona={"mission": "exercise the gate"},
        llm={"provider": "local", "model": "test-model"},
        tools={"allowlist": list(allowlist), "overrides": overrides or {}},
        autonomy=autonomy,
    )


def make_system(
    fs_root=None,
    denied_paths=(),
    max_autonomy: Autonomy = Autonomy.AUTONOMOUS_BOUNDED,
    limits: dict | None = None,
    pricing: dict | None = None,
    egress_allowlist=(),
    search: dict | None = None,
) -> SystemConfig:
    provider: dict = {"endpoint": "http://localhost:1"}
    if pricing is not None:
        provider["pricing"] = pricing
    return SystemConfig(
        providers={"local": provider},
        limits={"max_autonomy": max_autonomy, **(limits or {})},
        sandbox={
            "fs_root": str(fs_root) if fs_root is not None else None,
            "denied_paths": [str(d) for d in denied_paths],
            "egress_allowlist": list(egress_allowlist),
        },
        search=search,
    )
