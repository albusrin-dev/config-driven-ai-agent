"""The CLI still behaves as it did, now on the shared AgentService."""

import pytest

from conftest import FAKE_KEY

from config import load_system_config
from runtime import AgentService
from testing.fake_llm import FakeLLM, call, text_response, tool_response
from tools.defaults import build_registry
from ui.cli import run_cli


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def drive(sandbox, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)

    def run(script, inputs):
        system = load_system_config("dev")
        system.sandbox.fs_root = str(sandbox)
        service = AgentService(
            system=system,
            registry=build_registry(),
            llm_factory=lambda loaded: FakeLLM(script),
        )
        typed = iter(inputs)
        printed: list[str] = []
        run_cli(service, "scribe",
                input_fn=lambda prompt: next(typed),
                print_fn=lambda line: printed.append(str(line)))
        return printed

    return run


def test_cli_prints_the_reply(drive):
    printed = drive([text_response("Hello from the terminal.")], ["hi", "exit"])
    assert any("Hello from the terminal." in line for line in printed)
    assert any("ready (autonomy: supervised)" in line for line in printed)


def test_cli_reports_the_session_budget_on_exit(drive):
    printed = drive([text_response("done")], ["hi", "exit"])
    assert any(line.startswith("[session ") and "tokens=" in line for line in printed)


def test_cli_inline_approval_executes(drive, sandbox):
    script = [
        tool_response(call("write_file",
                           {"path": str(sandbox / "note.txt"), "content": "typed"})),
        text_response("Saved."),
    ]
    printed = drive(script, ["save a note", "y", "exit"])
    assert (sandbox / "note.txt").read_text(encoding="utf-8") == "typed"
    assert any("[confirm]" in line for line in printed)
    assert any("effect: filesystem:write:" in line for line in printed)


def test_cli_inline_refusal_does_not_execute(drive, sandbox):
    script = [
        tool_response(call("write_file",
                           {"path": str(sandbox / "note.txt"), "content": "typed"})),
        text_response("Understood — I left it alone."),
    ]
    drive(script, ["save a note", "n", "exit"])
    assert not (sandbox / "note.txt").exists()


def test_cli_shows_step_activity(drive, sandbox):
    (sandbox / "in.txt").write_text("data", encoding="utf-8")
    script = [
        tool_response(call("read_file", {"path": str(sandbox / "in.txt")})),
        text_response("read it"),
    ]
    printed = drive(script, ["read in.txt", "exit"])
    assert any(line.strip() == "… read_file" for line in printed)


def test_blank_line_quits(drive):
    printed = drive([text_response("unused")], [""])
    assert any(line.startswith("[session ") for line in printed)
