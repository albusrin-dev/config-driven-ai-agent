"""AgentService: session lifecycle, suspend/approve/resume, activity events."""

import json

import pytest

from conftest import FAKE_KEY

from config import load_system_config
from core.paths import PathEscapeError
from runtime import AgentService, SessionNotFoundError
from testing.fake_llm import FakeLLM, call, text_response, tool_response
from tools.defaults import build_registry


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def make_service(sandbox, monkeypatch):
    """Real profiles and a real registry; scripted model, tmp sandbox."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)

    def factory(script=None):
        system = load_system_config("dev")
        system.sandbox.fs_root = str(sandbox)
        llm = FakeLLM(script or [text_response("ready")])
        service = AgentService(
            system=system,
            registry=build_registry(),
            llm_factory=lambda loaded: llm,
        )
        service.test_llm = llm
        return service

    return factory


# --- agents ---------------------------------------------------------------

def test_list_agents_only_runnable(make_service):
    names = [a["name"] for a in make_service().list_agents()]
    assert "scribe" in names
    assert "researcher" in names
    # jarvis allowlists tools that have no implementation: excluded, exactly
    # as the registry cross-check would refuse it at run time.
    assert "jarvis" not in names


def test_listed_agents_carry_what_the_picker_needs(make_service):
    [scribe] = [a for a in make_service().list_agents() if a["name"] == "scribe"]
    assert scribe["autonomy"] == "supervised"
    assert "read_file" in scribe["tools"]


def test_start_unrunnable_agent_fails_loudly(make_service):
    with pytest.raises(Exception) as exc_info:
        make_service().start_session("jarvis")
    assert "not registered" in str(exc_info.value)


# --- lifecycle ------------------------------------------------------------

def test_start_session_reports_status(make_service):
    service = make_service()
    session_id = service.start_session("scribe")
    status = service.status(session_id)
    assert status["agent"] == "scribe"
    assert status["autonomy"] == "supervised"
    assert status["pending"] is None
    assert status["budget"] == {"tokens": 0, "tool_calls": 0, "cost_usd": 0.0}


def test_unknown_session_raises(make_service):
    with pytest.raises(SessionNotFoundError):
        make_service().status("nope")


def test_send_returns_the_reply(make_service):
    service = make_service([text_response("Hello — what shall I do?")])
    session_id = service.start_session("scribe")
    update = service.send(session_id, "hi")
    assert update.status == "completed"
    assert update.text == "Hello — what shall I do?"
    assert update.budget["tokens"] > 0


def test_identity_is_assembled_for_the_session(make_service):
    service = make_service()
    session_id = service.start_session("scribe")
    prompt = service.session(session_id).system_prompt
    assert "Mission" in prompt and "untrusted data" in prompt


# --- the confirmation cycle (the web UI's Approve/Deny) -------------------

def _write_script(sandbox):
    return [
        tool_response(call("write_file",
                           {"path": str(sandbox / "note.txt"), "content": "hello"})),
        text_response("Saved note.txt."),
    ]


def test_pending_action_is_presented_for_a_decision(make_service, sandbox):
    service = make_service(_write_script(sandbox))
    session_id = service.start_session("scribe")
    update = service.send(session_id, "save a note")

    assert update.status == "awaiting_approval"
    assert update.pending.tool == "write_file"
    assert update.pending.reason  # says why a human is needed
    # The details a person needs in order to judge it.
    assert update.pending.params["path"] == str(sandbox / "note.txt")
    assert update.pending.params["content"] == "hello"
    assert not (sandbox / "note.txt").exists()  # nothing ran


def test_pending_survives_a_client_disconnect(make_service, sandbox):
    """It lives on the session, not on a socket: a reconnecting client can
    still find the decision waiting."""
    service = make_service(_write_script(sandbox))
    session_id = service.start_session("scribe")
    service.send(session_id, "save a note")
    view = service.pending_view(session_id)
    assert view is not None and view.tool == "write_file"
    assert service.status(session_id)["pending"]["tool"] == "write_file"


def test_approve_executes(make_service, sandbox):
    service = make_service(_write_script(sandbox))
    session_id = service.start_session("scribe")
    service.send(session_id, "save a note")
    update = service.resolve_pending(session_id, True)
    assert update.status == "completed"
    assert (sandbox / "note.txt").read_text(encoding="utf-8") == "hello"
    assert update.budget["tool_calls"] == 1


def test_deny_does_not_execute(make_service, sandbox):
    service = make_service(_write_script(sandbox))
    session_id = service.start_session("scribe")
    service.send(session_id, "save a note")
    update = service.resolve_pending(session_id, False)
    assert update.status == "completed"
    assert not (sandbox / "note.txt").exists()
    assert update.budget["tool_calls"] == 0


def test_long_content_is_previewed_not_dumped(make_service, sandbox):
    big = "x" * 5000
    service = make_service([
        tool_response(call("write_file",
                           {"path": str(sandbox / "big.txt"), "content": big})),
        text_response("done"),
    ])
    session_id = service.start_session("scribe")
    update = service.send(session_id, "write a lot")
    shown = update.pending.params["content"]
    assert len(shown) < 400 and "+4700 chars" in shown


# --- activity -------------------------------------------------------------

def test_activity_events_describe_the_steps(make_service, sandbox):
    service = make_service(_write_script(sandbox))
    session_id = service.start_session("scribe")
    seen = []
    service.send(session_id, "save a note", on_activity=seen.append)

    kinds = [e.kind for e in seen]
    assert kinds[0] == "thinking"
    assert "tool_start" in kinds and "awaiting" in kinds
    assert any(e.tool == "write_file" for e in seen)


def test_activity_records_completion_of_a_tool(make_service, sandbox):
    (sandbox / "in.txt").write_text("data", encoding="utf-8")
    service = make_service([
        tool_response(call("read_file", {"path": str(sandbox / "in.txt")})),
        text_response("read it"),
    ])
    session_id = service.start_session("scribe")
    seen = []
    service.send(session_id, "read in.txt", on_activity=seen.append)
    ends = [e for e in seen if e.kind == "tool_end"]
    assert ends and ends[0].ok is True


def test_a_broken_activity_sink_cannot_break_the_run(make_service):
    service = make_service([text_response("still fine")])
    session_id = service.start_session("scribe")

    def explode(event):
        raise RuntimeError("the indicator is broken")

    update = service.send(session_id, "hi", on_activity=explode)
    assert update.status == "completed"
    assert update.text == "still fine"


# --- files ----------------------------------------------------------------

def test_sandbox_confinement_for_file_names(make_service, sandbox):
    service = make_service()
    session_id = service.start_session("scribe")
    assert service.confine_to_sandbox(session_id, "ok.txt").parent == sandbox.resolve()
    with pytest.raises(PathEscapeError):
        service.confine_to_sandbox(session_id, "../escape.txt")


# --- secrets --------------------------------------------------------------

def test_no_secret_in_anything_the_service_hands_out(make_service, sandbox):
    service = make_service(_write_script(sandbox))
    session_id = service.start_session("scribe")
    update = service.send(session_id, "save a note")
    blob = json.dumps({
        "status": service.status(session_id),
        "pending": {"tool": update.pending.tool, "reason": update.pending.reason,
                    "params": update.pending.params},
        "session": json.loads(service.session(session_id).model_dump_json()),
    })
    assert FAKE_KEY not in blob
