import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SystemConfig, load_system_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_SYSTEM = REPO_ROOT / "profiles" / "system" / "dev.yaml"

FAKE_KEY = "sk-test-fake-key-12345"


@pytest.fixture
def system_config() -> SystemConfig:
    return load_system_config(DEV_SYSTEM)


@pytest.fixture
def anthropic_key(monkeypatch) -> str:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    return FAKE_KEY


def write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p
