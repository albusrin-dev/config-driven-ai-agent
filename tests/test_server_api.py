"""The web console's API: chat, decisions, files, loopback, secrets."""

import io
import json

import pytest
from fastapi.testclient import TestClient

from conftest import FAKE_KEY

from config import load_system_config
from runtime import AgentService
from server.app import BIND_HOST, create_app
from testing.fake_llm import FakeLLM, call, text_response, tool_response
from tools.defaults import build_registry


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def make_client(sandbox, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)

    def factory(script=None):
        system = load_system_config("dev")
        system.sandbox.fs_root = str(sandbox)
        service = AgentService(
            system=system,
            registry=build_registry(),
            llm_factory=lambda loaded: FakeLLM(script or [text_response("ready")]),
        )
        return TestClient(create_app(service))

    return factory


def _start(client, agent="scribe"):
    response = client.post("/api/session", json={"agent": agent})
    assert response.status_code == 200
    return response.json()["session_id"]


# --- binding --------------------------------------------------------------

def test_binds_loopback_only():
    """The console is reachable from this machine and nowhere else."""
    assert BIND_HOST == "127.0.0.1"
    assert BIND_HOST != "0.0.0.0"


def test_host_is_not_configurable():
    """No config field can move it onto a network — exposing it is a
    hosting decision, not a setting."""
    from config.models import ServerConfig

    assert "host" not in ServerConfig.model_fields
    assert set(ServerConfig.model_fields) == {"port"}


# --- agents & sessions ----------------------------------------------------

def test_agents_endpoint_lists_only_runnable(make_client):
    client = make_client()
    names = [a["name"] for a in client.get("/api/agents").json()["agents"]]
    assert "scribe" in names and "researcher" in names
    assert "jarvis" not in names


def test_start_session_returns_status(make_client):
    client = make_client()
    body = client.post("/api/session", json={"agent": "scribe"}).json()
    assert body["agent"] == "scribe"
    assert body["state"] == "idle"
    assert body["pending"] is None


def test_start_unknown_agent_is_a_clear_400(make_client):
    client = make_client()
    response = client.post("/api/session", json={"agent": "nope"})
    assert response.status_code == 400


def test_unknown_session_is_404(make_client):
    assert make_client().get("/api/session/nosuch").status_code == 404


# --- chat over HTTP and the socket ---------------------------------------

def test_message_returns_the_reply(make_client):
    client = make_client([text_response("Hello from scribe.")])
    session_id = _start(client)
    body = client.post(f"/api/session/{session_id}/message",
                       json={"text": "hi"}).json()
    assert body["status"] == "completed"
    assert body["text"] == "Hello from scribe."


def test_researcher_also_answers(make_client):
    client = make_client([text_response("Researching that now.")])
    session_id = _start(client, "researcher")
    body = client.post(f"/api/session/{session_id}/message",
                       json={"text": "find something"}).json()
    assert body["text"] == "Researching that now."


def test_empty_message_is_refused(make_client):
    client = make_client()
    session_id = _start(client)
    assert client.post(f"/api/session/{session_id}/message",
                       json={"text": "   "}).status_code == 400


def test_websocket_streams_activity_then_the_update(make_client, sandbox):
    (sandbox / "in.txt").write_text("data", encoding="utf-8")
    client = make_client([
        tool_response(call("read_file", {"path": str(sandbox / "in.txt")})),
        text_response("I read it."),
    ])
    session_id = _start(client)
    with client.websocket_connect(f"/api/session/{session_id}/stream") as ws:
        ws.send_json({"type": "message", "text": "read in.txt"})
        frames = []
        while True:
            frame = ws.receive_json()
            frames.append(frame)
            if frame["type"] == "update":
                break

    kinds = [f.get("kind") for f in frames if f["type"] == "activity"]
    assert "thinking" in kinds
    assert "tool_start" in kinds
    assert any(f.get("tool") == "read_file" for f in frames if f["type"] == "activity")
    assert frames[-1]["status"] == "completed" and frames[-1]["text"] == "I read it."


# --- the Approve / Deny cycle --------------------------------------------

def _write_script(sandbox):
    return [
        tool_response(call("write_file",
                           {"path": str(sandbox / "note.txt"), "content": "hello"})),
        text_response("Saved it."),
    ]


def test_pending_action_is_returned_with_full_details(make_client, sandbox):
    client = make_client(_write_script(sandbox))
    session_id = _start(client)
    body = client.post(f"/api/session/{session_id}/message",
                       json={"text": "save a note"}).json()
    assert body["status"] == "awaiting_approval"
    assert body["pending"]["tool"] == "write_file"
    assert body["pending"]["params"]["path"] == str(sandbox / "note.txt")
    assert body["pending"]["reason"]
    assert not (sandbox / "note.txt").exists()


def test_approve_endpoint_executes(make_client, sandbox):
    client = make_client(_write_script(sandbox))
    session_id = _start(client)
    client.post(f"/api/session/{session_id}/message", json={"text": "save a note"})
    body = client.post(f"/api/session/{session_id}/approve").json()
    assert body["status"] == "completed"
    assert (sandbox / "note.txt").read_text(encoding="utf-8") == "hello"


def test_deny_endpoint_does_not_execute(make_client, sandbox):
    client = make_client(_write_script(sandbox))
    session_id = _start(client)
    client.post(f"/api/session/{session_id}/message", json={"text": "save a note"})
    body = client.post(f"/api/session/{session_id}/deny").json()
    assert body["status"] == "completed"
    assert not (sandbox / "note.txt").exists()


def test_decision_over_the_socket(make_client, sandbox):
    client = make_client(_write_script(sandbox))
    session_id = _start(client)
    with client.websocket_connect(f"/api/session/{session_id}/stream") as ws:
        ws.send_json({"type": "message", "text": "save a note"})
        while ws.receive_json()["type"] != "update":
            pass
        ws.send_json({"type": "decision", "approved": True})
        while True:
            frame = ws.receive_json()
            if frame["type"] == "update":
                break
    assert frame["status"] == "completed"
    assert (sandbox / "note.txt").read_text(encoding="utf-8") == "hello"


def test_pending_survives_socket_close(make_client, sandbox):
    """Closing the tab must not lose a decision that is waiting."""
    client = make_client(_write_script(sandbox))
    session_id = _start(client)
    with client.websocket_connect(f"/api/session/{session_id}/stream") as ws:
        ws.send_json({"type": "message", "text": "save a note"})
        while ws.receive_json()["type"] != "update":
            pass
    # socket closed; reconnect and the pending action is still there
    status = client.get(f"/api/session/{session_id}").json()
    assert status["pending"]["tool"] == "write_file"
    assert client.post(f"/api/session/{session_id}/approve").json()["status"] == "completed"


# --- the gate is not bypassable from the browser -------------------------

def test_the_web_layer_cannot_execute_a_denied_action(make_client, tmp_path, sandbox):
    """Approving something the GATE denies changes nothing: approval answers
    a confirmation question; it does not overrule a denial."""
    outside = tmp_path / "outside.txt"
    client = make_client([
        tool_response(call("write_file", {"path": str(outside), "content": "escape"})),
        text_response("I could not write there."),
    ])
    session_id = _start(client)
    body = client.post(f"/api/session/{session_id}/message",
                       json={"text": "write outside"}).json()
    # Denied outright — no pending action ever offered to the human.
    assert body["status"] == "completed"
    assert body["pending"] is None
    assert not outside.exists()
    assert client.post(f"/api/session/{session_id}/approve").json()["status"] == "error"


def test_no_secret_reaches_the_browser(make_client, sandbox):
    client = make_client(_write_script(sandbox))
    session_id = _start(client)
    bodies = [
        client.get("/api/agents").text,
        client.get(f"/api/session/{session_id}").text,
        client.post(f"/api/session/{session_id}/message", json={"text": "save"}).text,
        client.post(f"/api/session/{session_id}/approve").text,
    ]
    for body in bodies:
        assert FAKE_KEY not in body
        assert "ANTHROPIC_API_KEY" not in body


# --- files ----------------------------------------------------------------

def test_upload_then_the_agent_reads_it(make_client, sandbox):
    client = make_client([
        tool_response(call("read_file", {"path": str(sandbox / "brief.txt")})),
        text_response("The brief says: ship it."),
    ])
    session_id = _start(client)
    response = client.post(
        f"/api/session/{session_id}/upload",
        files={"file": ("brief.txt", io.BytesIO(b"ship it"), "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "brief.txt"
    assert (sandbox / "brief.txt").read_bytes() == b"ship it"

    body = client.post(f"/api/session/{session_id}/message",
                       json={"text": "read brief.txt"}).json()
    assert body["status"] == "completed"
    assert "ship it" in body["text"]


def test_upload_traversal_is_refused_by_the_endpoint(make_client, sandbox, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("do not touch", encoding="utf-8")
    client = make_client()
    session_id = _start(client)
    response = client.post(
        f"/api/session/{session_id}/upload",
        files={"file": ("../../outside.txt", io.BytesIO(b"x"), "text/plain")},
    )
    assert response.status_code == 200  # sanitized into the sandbox...
    assert outside.read_text(encoding="utf-8") == "do not touch"  # ...never outside
    assert (sandbox / "outside.txt").exists()


def test_upload_bad_type_and_oversize_are_refused(make_client, sandbox):
    from server.uploads import MAX_UPLOAD_BYTES

    client = make_client()
    session_id = _start(client)
    bad = client.post(f"/api/session/{session_id}/upload",
                      files={"file": ("payload.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
    assert bad.status_code == 400 and "Upload one of" in bad.json()["detail"]

    big = client.post(
        f"/api/session/{session_id}/upload",
        files={"file": ("big.pdf", io.BytesIO(b"x" * (MAX_UPLOAD_BYTES + 1)), "application/pdf")},
    )
    assert big.status_code == 400 and "limit is" in big.json()["detail"]
    assert list(sandbox.iterdir()) == []


def test_download_is_confined(make_client, sandbox, tmp_path):
    (sandbox / "report.md").write_text("# done", encoding="utf-8")
    (tmp_path / "secret.md").write_text("private", encoding="utf-8")
    client = make_client()
    session_id = _start(client)

    ok = client.get(f"/api/session/{session_id}/files/report.md")
    assert ok.status_code == 200 and ok.text == "# done"

    escaped = client.get(f"/api/session/{session_id}/files/..%2Fsecret.md")
    assert escaped.status_code in (400, 404)
    assert "private" not in escaped.text


# --- the page itself ------------------------------------------------------

def test_console_page_is_served_same_origin(make_client):
    client = make_client()
    page = client.get("/")
    assert page.status_code == 200
    assert "Agent console" in page.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200
