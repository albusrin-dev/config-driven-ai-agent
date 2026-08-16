"""A2: LLM requests are wall-clock bounded — capped retries, clean failure."""

import urllib.error

import pytest

from conftest import make_agent, make_system

from config.secrets import Secrets
from core.loop import Errored, run_turn
from core.session import new_session
from llm.anthropic import MAX_RETRIES, AnthropicAdapter, LLMRequestError
from tools.registry import ToolRegistry

KEY_VAR = "TEST_ANTHROPIC_KEY"

OK_RESPONSE = {
    "content": [{"type": "text", "text": "fine"}],
    "usage": {"input_tokens": 1, "output_tokens": 1},
    "stop_reason": "end_turn",
}


def _adapter():
    return AnthropicAdapter(
        provider_name="anthropic",
        model="claude-fable-5",
        secrets=Secrets({"anthropic": KEY_VAR}),
        retry_backoff_seconds=0.0,  # keep tests fast; bound is on attempts
    )


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv(KEY_VAR, "k")


def test_timeout_exhausts_bounded_retries_then_raises(monkeypatch):
    attempts = []

    def timing_out(self, payload, api_key):
        attempts.append(1)
        raise TimeoutError("simulated stalled socket (urlopen timeout fired)")

    monkeypatch.setattr(AnthropicAdapter, "_post", timing_out)
    adapter = _adapter()
    with pytest.raises(LLMRequestError) as exc_info:
        adapter.complete([{"role": "user", "content": "hi"}], [])
    assert len(attempts) == 1 + MAX_RETRIES  # bounded, exactly
    assert "TimeoutError" in str(exc_info.value)


def test_loop_turns_exhausted_retries_into_errored(monkeypatch):
    monkeypatch.setattr(
        AnthropicAdapter, "_post",
        lambda self, payload, api_key: (_ for _ in ()).throw(TimeoutError("stalled")),
    )
    session = new_session("tester")
    result = run_turn(session, "hi", _adapter(), ToolRegistry(),
                      make_agent(allowlist=[]), make_system())
    assert isinstance(result, Errored)
    assert "LLM call failed" in result.reason
    assert session.status == "error"


def test_transient_failure_recovers_within_bound(monkeypatch):
    attempts = []

    def flaky(self, payload, api_key):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.URLError("connection reset")
        return OK_RESPONSE

    monkeypatch.setattr(AnthropicAdapter, "_post", flaky)
    response = _adapter().complete([{"role": "user", "content": "hi"}], [])
    assert response.text == "fine"
    assert len(attempts) == 3  # failed twice, succeeded on the last allowed attempt


def test_client_error_fails_fast_no_retry(monkeypatch):
    attempts = []

    def bad_request(self, payload, api_key):
        attempts.append(1)
        raise urllib.error.HTTPError("http://x", 400, "bad request", None, None)

    monkeypatch.setattr(AnthropicAdapter, "_post", bad_request)
    with pytest.raises(urllib.error.HTTPError):
        _adapter().complete([{"role": "user", "content": "hi"}], [])
    assert len(attempts) == 1  # retrying a bad request just burns the bound


def test_server_error_is_retried(monkeypatch):
    attempts = []

    def flaky_server(self, payload, api_key):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.HTTPError("http://x", 529, "overloaded", None, None)
        return OK_RESPONSE

    monkeypatch.setattr(AnthropicAdapter, "_post", flaky_server)
    response = _adapter().complete([{"role": "user", "content": "hi"}], [])
    assert response.text == "fine"
    assert len(attempts) == 2


def test_normal_request_single_attempt(monkeypatch):
    attempts = []

    def ok(self, payload, api_key):
        attempts.append(1)
        return OK_RESPONSE

    monkeypatch.setattr(AnthropicAdapter, "_post", ok)
    assert _adapter().complete([{"role": "user", "content": "hi"}], []).text == "fine"
    assert len(attempts) == 1
