"""AnthropicAdapter contract tests against a mocked HTTP seam (_post)."""

import json

import pytest

from config.errors import MissingSecretError
from config.secrets import Secrets
from llm.anthropic import AnthropicAdapter, _to_anthropic_messages
from tools.builtins.files import ReadFileTool

KEY_VAR = "TEST_ANTHROPIC_KEY"

CANNED_RESPONSE = {
    "content": [
        {"type": "text", "text": "I'll read that file."},
        {"type": "tool_use", "id": "tu_1", "name": "read_file",
         "input": {"path": "notes.txt"}},
    ],
    "usage": {"input_tokens": 42, "output_tokens": 7},
    "stop_reason": "tool_use",
}


@pytest.fixture
def adapter():
    return AnthropicAdapter(
        provider_name="anthropic",
        model="claude-fable-5",
        secrets=Secrets({"anthropic": KEY_VAR}),
        temperature=0.3,
        system_prompt="Be a scribe.",
    )


@pytest.fixture
def captured(adapter, monkeypatch):
    box = {}

    def fake_post(self, payload, api_key):
        box["payload"] = payload
        box["api_key"] = api_key
        return CANNED_RESPONSE

    monkeypatch.setattr(AnthropicAdapter, "_post", fake_post)
    return box


def test_secret_resolved_at_request_time_never_stored(adapter, captured, monkeypatch):
    monkeypatch.setenv(KEY_VAR, "key-one")
    adapter.complete([{"role": "user", "content": "hi"}], [])
    assert captured["api_key"] == "key-one"

    monkeypatch.setenv(KEY_VAR, "key-two")  # rotated between requests
    adapter.complete([{"role": "user", "content": "hi"}], [])
    assert captured["api_key"] == "key-two"

    # The adapter never retains a key value.
    internals = json.dumps({k: repr(v) for k, v in vars(adapter).items()})
    assert "key-one" not in internals and "key-two" not in internals


def test_missing_secret_fails_at_request_time(adapter, captured, monkeypatch):
    monkeypatch.delenv(KEY_VAR, raising=False)
    with pytest.raises(MissingSecretError):
        adapter.complete([{"role": "user", "content": "hi"}], [])


def test_tool_schema_mapping(adapter, captured, monkeypatch):
    monkeypatch.setenv(KEY_VAR, "k")
    tool = ReadFileTool()
    schemas = [{
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema.model_json_schema(),
    }]
    adapter.complete([{"role": "user", "content": "read it"}], schemas)

    payload = captured["payload"]
    assert payload["model"] == "claude-fable-5"
    assert payload["temperature"] == 0.3
    assert payload["system"] == "Be a scribe."
    [mapped] = payload["tools"]
    assert mapped["name"] == "read_file"
    assert "path" in mapped["input_schema"]["properties"]


def test_system_role_message_hoisted_into_system_field(adapter, captured, monkeypatch):
    """The loop's assembled identity prompt arrives as a system-role
    message; the API has no such role in `messages`, so the adapter hoists
    it into the top-level `system` field (after any constructor prompt)."""
    monkeypatch.setenv(KEY_VAR, "k")
    adapter.complete(
        [
            {"role": "system", "content": "## Mission\nCurate the archive."},
            {"role": "user", "content": "hi"},
        ],
        [],
    )
    payload = captured["payload"]
    assert payload["system"] == "Be a scribe.\n\n## Mission\nCurate the archive."
    # No system-role message may leak into the messages array.
    assert all(m["role"] in ("user", "assistant") for m in payload["messages"])
    assert payload["messages"][0]["content"][0]["text"] == "hi"


def test_message_mapping_and_same_role_merge():
    neutral = [
        {"role": "user", "content": "copy the file"},
        {"role": "assistant", "content": "on it", "tool_calls": [
            {"id": "a", "name": "read_file", "params": {"path": "f.txt"}},
            {"id": "b", "name": "read_file", "params": {"path": "g.txt"}},
        ]},
        {"role": "tool_result", "tool_call_id": "a", "content": "alpha", "ok": True},
        {"role": "tool_result", "tool_call_id": "b", "content": "boom", "ok": False},
    ]
    mapped = _to_anthropic_messages(neutral)

    assert [m["role"] for m in mapped] == ["user", "assistant", "user"]
    assistant_blocks = mapped[1]["content"]
    assert assistant_blocks[0] == {"type": "text", "text": "on it"}
    assert assistant_blocks[1]["type"] == "tool_use"
    assert assistant_blocks[1]["input"] == {"path": "f.txt"}
    # Two consecutive tool results merged into ONE user message.
    result_blocks = mapped[2]["content"]
    assert len(result_blocks) == 2
    assert result_blocks[0] == {"type": "tool_result", "tool_use_id": "a",
                                "content": "alpha"}
    assert result_blocks[1]["is_error"] is True


def test_response_parsing(adapter, captured, monkeypatch):
    monkeypatch.setenv(KEY_VAR, "k")
    response = adapter.complete([{"role": "user", "content": "go"}], [])

    assert response.text == "I'll read that file."
    [tc] = response.tool_calls
    assert (tc.id, tc.name, tc.params) == ("tu_1", "read_file", {"path": "notes.txt"})
    assert response.usage.input_tokens == 42
    assert response.usage.output_tokens == 7
    assert response.stop_reason == "tool_use"


def test_count_tokens_is_positive(adapter):
    assert adapter.count_tokens([{"role": "user", "content": "hello"}], []) >= 1
