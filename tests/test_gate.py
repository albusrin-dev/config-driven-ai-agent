"""The full gate decision matrix: autonomy table, overrides, destructive floor."""

import pytest

from conftest import make_agent, make_system

from config.models import Autonomy
from core.gate import Decision, PolicyGate
from tools.builtins.files import DeleteFileTool, ReadFileTool, WriteFileTool

gate = PolicyGate()

ALL_TOOLS = ["read_file", "write_file", "delete_file"]


def _evaluate(tool, tmp_path, agent):
    system = make_system(fs_root=tmp_path)
    params = {"path": str(tmp_path / "f.txt")}
    if isinstance(tool, WriteFileTool):
        params["content"] = "x"
    validated = tool.input_schema.model_validate(params)
    return gate.evaluate(tool, validated, agent, system)


# --- 1. Allowlist ---------------------------------------------------------

def test_tool_not_in_allowlist_is_denied(tmp_path):
    agent = make_agent(allowlist=[], autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    d = _evaluate(ReadFileTool(), tmp_path, agent)
    assert d.decision is Decision.DENY
    assert "allowlist" in d.reason


# --- 3. Autonomy table (the matrix) ---------------------------------------

MATRIX = [
    (ReadFileTool, Autonomy.ASSISTED, Decision.REQUIRE_CONFIRMATION),
    (ReadFileTool, Autonomy.SUPERVISED, Decision.ALLOW),
    (ReadFileTool, Autonomy.AUTONOMOUS_BOUNDED, Decision.ALLOW),
    (WriteFileTool, Autonomy.ASSISTED, Decision.REQUIRE_CONFIRMATION),
    (WriteFileTool, Autonomy.SUPERVISED, Decision.REQUIRE_CONFIRMATION),
    (WriteFileTool, Autonomy.AUTONOMOUS_BOUNDED, Decision.ALLOW),
    (DeleteFileTool, Autonomy.ASSISTED, Decision.REQUIRE_CONFIRMATION),
    (DeleteFileTool, Autonomy.SUPERVISED, Decision.REQUIRE_CONFIRMATION),
    (DeleteFileTool, Autonomy.AUTONOMOUS_BOUNDED, Decision.REQUIRE_CONFIRMATION),
]


@pytest.mark.parametrize(
    "tool_cls,autonomy,expected",
    MATRIX,
    ids=[f"{t.name}-{a.value}" for t, a, _ in MATRIX],
)
def test_decision_matrix(tmp_path, tool_cls, autonomy, expected):
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=autonomy)
    d = _evaluate(tool_cls(), tmp_path, agent)
    assert d.decision is expected, d.reason


# --- 3b. Per-tool overrides ------------------------------------------------

def test_confirm_always_forces_confirmation_on_read_at_autonomous(tmp_path):
    agent = make_agent(
        allowlist=ALL_TOOLS,
        overrides={"read_file": {"confirm": "always"}},
        autonomy=Autonomy.AUTONOMOUS_BOUNDED,
    )
    d = _evaluate(ReadFileTool(), tmp_path, agent)
    assert d.decision is Decision.REQUIRE_CONFIRMATION
    assert "confirm=always" in d.reason


def test_confirm_never_allows_nondestructive_mutation(tmp_path):
    agent = make_agent(
        allowlist=ALL_TOOLS,
        overrides={"write_file": {"confirm": "never"}},
        autonomy=Autonomy.ASSISTED,
    )
    d = _evaluate(WriteFileTool(), tmp_path, agent)
    assert d.decision is Decision.ALLOW
    assert "confirm=never" in d.reason


def test_confirm_never_is_ignored_for_destructive_tool(tmp_path):
    """The destructive floor: no override can lower it."""
    agent = make_agent(
        allowlist=ALL_TOOLS,
        overrides={"delete_file": {"confirm": "never"}},
        autonomy=Autonomy.AUTONOMOUS_BOUNDED,
    )
    d = _evaluate(DeleteFileTool(), tmp_path, agent)
    assert d.decision is Decision.REQUIRE_CONFIRMATION
    assert "floor" in d.reason


def test_confirm_default_falls_through_to_autonomy(tmp_path):
    agent = make_agent(
        allowlist=ALL_TOOLS,
        overrides={"read_file": {"confirm": "default"}},
        autonomy=Autonomy.SUPERVISED,
    )
    d = _evaluate(ReadFileTool(), tmp_path, agent)
    assert d.decision is Decision.ALLOW


# --- Gate purity -----------------------------------------------------------

def test_gate_reports_effects_and_reason(tmp_path):
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.SUPERVISED)
    d = _evaluate(ReadFileTool(), tmp_path, agent)
    assert d.reason
    assert len(d.effects_summary) == 1
    assert d.effects_summary[0].startswith("filesystem:read:")


def test_gate_does_not_execute(tmp_path):
    """Evaluating a write leaves the filesystem untouched (pure gate)."""
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    target = tmp_path / "f.txt"
    d = _evaluate(WriteFileTool(), tmp_path, agent)
    assert d.decision is Decision.ALLOW
    assert not target.exists()
