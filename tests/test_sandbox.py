"""Sandbox enforcement of filesystem effects + fail-closed extensibility."""

import pytest

from pydantic import BaseModel, ConfigDict

from conftest import make_agent, make_system

from config.models import Autonomy
from core.base import Tool, ToolContext, ToolResult
from core.effects import Effect
from core.gate import Decision, PolicyGate
from tools.builtins.files import ReadFileTool, WriteFileTool

gate = PolicyGate()


def _agent():
    return make_agent(
        allowlist=["read_file", "write_file", "strange_tool"],
        autonomy=Autonomy.AUTONOMOUS_BOUNDED,
    )


def _eval(tool, params, system):
    validated = tool.input_schema.model_validate(params)
    return gate.evaluate(tool, validated, _agent(), system)


@pytest.fixture
def sandbox_root(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    return root


def test_read_outside_fs_root_denied(tmp_path, sandbox_root):
    outside = tmp_path / "outside.txt"
    outside.write_text("top secret", encoding="utf-8")
    d = _eval(ReadFileTool(), {"path": str(outside)}, make_system(fs_root=sandbox_root))
    assert d.decision is Decision.DENY
    assert "outside sandbox root" in d.reason


def test_write_outside_fs_root_denied(tmp_path, sandbox_root):
    d = _eval(
        WriteFileTool(),
        {"path": str(tmp_path / "evil.txt"), "content": "x"},
        make_system(fs_root=sandbox_root),
    )
    assert d.decision is Decision.DENY


def test_traversal_escaping_sandbox_denied(sandbox_root):
    d = _eval(
        ReadFileTool(),
        {"path": str(sandbox_root / ".." / ".." / "victim.txt")},
        make_system(fs_root=sandbox_root),
    )
    assert d.decision is Decision.DENY
    assert "outside sandbox root" in d.reason


def test_relative_traversal_param_denied(sandbox_root):
    # relative param resolved against fs_root, then '..' escapes it
    d = _eval(
        ReadFileTool(),
        {"path": "../victim.txt"},
        make_system(fs_root=sandbox_root),
    )
    assert d.decision is Decision.DENY


def test_path_under_denied_paths_denied(sandbox_root):
    (sandbox_root / "private").mkdir()
    system = make_system(fs_root=sandbox_root, denied_paths=["private"])
    d = _eval(ReadFileTool(), {"path": str(sandbox_root / "private" / "f.txt")}, system)
    assert d.decision is Decision.DENY
    assert "denied path" in d.reason


def test_absolute_denied_path_denied(sandbox_root):
    private = sandbox_root / "vault"
    system = make_system(fs_root=sandbox_root, denied_paths=[str(private)])
    d = _eval(ReadFileTool(), {"path": str(private / "key.txt")}, system)
    assert d.decision is Decision.DENY


def test_sibling_of_denied_path_allowed(sandbox_root):
    system = make_system(fs_root=sandbox_root, denied_paths=["private"])
    d = _eval(ReadFileTool(), {"path": str(sandbox_root / "public.txt")}, system)
    assert d.decision is Decision.ALLOW


def test_fs_root_unset_denies_all_filesystem_effects(sandbox_root):
    system = make_system(fs_root=None)
    d = _eval(ReadFileTool(), {"path": str(sandbox_root / "f.txt")}, system)
    assert d.decision is Decision.DENY
    assert "fs_root is not set" in d.reason
    assert "fail-closed" in d.reason


# --- Extensibility contract: unknown effect types are denied ---------------

class _NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TeleportEffect(Effect):
    """An effect type the Phase 1 gate has never heard of."""


class StrangeTool(Tool):
    name = "strange_tool"
    description = "declares an effect the gate does not recognise"
    input_schema = _NoParams
    mutating = False
    destructive = False

    def plan_effects(self, params, context):
        return [TeleportEffect()]

    def execute(self, params, context):  # pragma: no cover — must never run
        return ToolResult(ok=True, output="teleported")


def test_unrecognised_effect_type_denied(sandbox_root):
    d = _eval(StrangeTool(), {}, make_system(fs_root=sandbox_root))
    assert d.decision is Decision.DENY
    assert "TeleportEffect" in d.reason
    assert "fail-closed" in d.reason


def test_destructive_without_mutating_is_a_type_error():
    with pytest.raises(TypeError, match="destructive implies mutating"):
        class BadTool(Tool):
            name = "bad"
            description = "misclassified"
            input_schema = _NoParams
            mutating = False
            destructive = True

            def plan_effects(self, params, context):
                return []

            def execute(self, params, context):
                return ToolResult(ok=True)
