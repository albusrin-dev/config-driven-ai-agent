"""ToolRegistry: registration, lookup, allowlist resolution."""

import pytest

from conftest import make_agent

from core.errors import UnknownToolError
from tools.registry import ToolRegistry
from tools.builtins.files import DeleteFileTool, ReadFileTool, WriteFileTool


@pytest.fixture
def registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(ReadFileTool())
    r.register(WriteFileTool())
    r.register(DeleteFileTool())
    return r


def test_get_registered_tool(registry):
    assert registry.get("read_file").name == "read_file"


def test_get_unknown_tool_raises(registry):
    with pytest.raises(UnknownToolError) as exc_info:
        registry.get("teleport")
    msg = str(exc_info.value)
    assert "teleport" in msg
    assert "read_file" in msg  # names what IS registered


def test_duplicate_registration_rejected(registry):
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ReadFileTool())


def test_toolset_for_filters_by_allowlist(registry):
    agent = make_agent(allowlist=["read_file", "write_file"])
    names = [t.name for t in registry.toolset_for(agent)]
    assert names == ["read_file", "write_file"]  # delete_file excluded


def test_empty_allowlist_yields_no_tools(registry):
    assert registry.toolset_for(make_agent(allowlist=[])) == []


def test_allowlist_naming_unregistered_tool_raises(registry):
    agent = make_agent(allowlist=["read_file", "summon_demon"])
    with pytest.raises(UnknownToolError) as exc_info:
        registry.toolset_for(agent)
    msg = str(exc_info.value)
    assert "summon_demon" in msg
    assert "not registered" in msg
