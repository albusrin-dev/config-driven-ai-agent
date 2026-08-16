"""The enforcement chokepoint: deny/pending never execute; allow does.
Plus param validation and audit records."""

import json
import logging

import pytest

from conftest import make_agent, make_system

from config.models import Autonomy
from core.enforce import Denied, Executed, Pending, enforce_and_run
from core.errors import ToolParamError
from tools.builtins.files import DeleteFileTool, ReadFileTool, WriteFileTool

ALL_TOOLS = ["read_file", "write_file", "delete_file"]


@pytest.fixture
def sandbox_root(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    return root


@pytest.fixture
def system(sandbox_root):
    return make_system(fs_root=sandbox_root)


# --- DENY does not execute -------------------------------------------------

def test_deny_does_not_execute(system, sandbox_root):
    agent = make_agent(allowlist=[], autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    target = sandbox_root / "f.txt"
    outcome = enforce_and_run(
        WriteFileTool(), {"path": str(target), "content": "x"}, agent, system
    )
    assert isinstance(outcome, Denied)
    assert "allowlist" in outcome.reason
    assert not target.exists()  # the side effect did not happen


def test_sandbox_deny_does_not_execute(system, tmp_path):
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    outside = tmp_path / "outside.txt"
    outcome = enforce_and_run(
        WriteFileTool(), {"path": str(outside), "content": "x"}, agent, system
    )
    assert isinstance(outcome, Denied)
    assert not outside.exists()


# --- REQUIRE_CONFIRMATION without approval does not execute ----------------

def test_pending_without_approver_does_not_execute(system, sandbox_root):
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.SUPERVISED)
    target = sandbox_root / "f.txt"
    outcome = enforce_and_run(
        WriteFileTool(), {"path": str(target), "content": "x"}, agent, system
    )
    assert isinstance(outcome, Pending)
    assert not target.exists()


def test_pending_when_approver_declines(system, sandbox_root):
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    target = sandbox_root / "keep-me.txt"
    target.write_text("precious", encoding="utf-8")
    outcome = enforce_and_run(
        DeleteFileTool(), {"path": str(target)}, agent, system,
        approver=lambda decision: False,
    )
    assert isinstance(outcome, Pending)
    assert target.exists()  # the file was NOT deleted


def test_approved_confirmation_executes(system, sandbox_root):
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    target = sandbox_root / "doomed.txt"
    target.write_text("bye", encoding="utf-8")
    seen = []
    outcome = enforce_and_run(
        DeleteFileTool(), {"path": str(target)}, agent, system,
        approver=lambda decision: seen.append(decision) or True,
    )
    assert isinstance(outcome, Executed)
    assert outcome.result.ok
    assert not target.exists()
    assert len(seen) == 1  # approver consulted exactly once, with the decision


# --- ALLOW executes --------------------------------------------------------

def test_allow_executes_read(system, sandbox_root):
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.SUPERVISED)
    target = sandbox_root / "hello.txt"
    target.write_text("hello world", encoding="utf-8")
    outcome = enforce_and_run(ReadFileTool(), {"path": str(target)}, agent, system)
    assert isinstance(outcome, Executed)
    assert outcome.result.ok
    assert outcome.result.output == "hello world"


def test_allow_executes_write_at_autonomous(system, sandbox_root):
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    target = sandbox_root / "new.txt"
    outcome = enforce_and_run(
        WriteFileTool(), {"path": str(target), "content": "written"}, agent, system
    )
    assert isinstance(outcome, Executed)
    assert target.read_text(encoding="utf-8") == "written"


# --- Param validation ------------------------------------------------------

def test_bad_params_raise_clear_error(system):
    agent = make_agent(allowlist=ALL_TOOLS)
    with pytest.raises(ToolParamError) as exc_info:
        enforce_and_run(WriteFileTool(), {"path": "f.txt"}, agent, system)  # no content
    msg = str(exc_info.value)
    assert "write_file" in msg
    assert "content" in msg


def test_unknown_param_rejected(system):
    agent = make_agent(allowlist=ALL_TOOLS)
    with pytest.raises(ToolParamError) as exc_info:
        enforce_and_run(
            ReadFileTool(), {"path": "f.txt", "mode": "sneaky"}, agent, system
        )
    assert "mode" in str(exc_info.value)


# --- Audit -----------------------------------------------------------------

def _audit_records(caplog):
    return [json.loads(r.message) for r in caplog.records if r.name == "agent.audit"]


def test_audit_record_emitted_per_decision(system, sandbox_root, caplog):
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.SUPERVISED)
    target = sandbox_root / "audited.txt"
    target.write_text("data", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger="agent.audit"):
        enforce_and_run(ReadFileTool(), {"path": str(target)}, agent, system)
        enforce_and_run(
            WriteFileTool(), {"path": str(target), "content": "x"}, agent, system
        )
    records = _audit_records(caplog)
    assert len(records) == 2
    allow, pending = records
    assert allow["tool_name"] == "read_file"
    assert allow["decision"] == "ALLOW"
    assert allow["agent_name"] == "tester"
    assert allow["reason"]
    assert allow["effects_summary"][0].startswith("filesystem:read:")
    assert pending["decision"] == "REQUIRE_CONFIRMATION"


def test_audit_never_contains_file_contents(system, sandbox_root, caplog):
    agent = make_agent(allowlist=ALL_TOOLS, autonomy=Autonomy.AUTONOMOUS_BOUNDED)
    secret_content = "TOP-SECRET-CONTENT " * 50
    with caplog.at_level(logging.INFO, logger="agent.audit"):
        enforce_and_run(
            WriteFileTool(),
            {"path": str(sandbox_root / "s.txt"), "content": secret_content},
            agent, system,
        )
    [record] = _audit_records(caplog)
    blob = json.dumps(record)
    assert "TOP-SECRET-CONTENT" not in blob
    assert record["params_summary"]["content"] == f"<str len={len(secret_content)}>"


def test_audit_redacts_sensitive_keys():
    from pydantic import BaseModel

    from core.audit import summarize_params

    class P(BaseModel):
        path: str
        api_key: str

    summary = summarize_params(P(path="f.txt", api_key="sk-live-999"))
    assert summary["api_key"] == "<redacted>"
    assert summary["path"] == "f.txt"
