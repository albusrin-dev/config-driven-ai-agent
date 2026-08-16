"""Built-in filesystem tools: read_file, write_file, delete_file.

Each declares a FilesystemEffect from its params in ``plan_effects``,
resolving relative paths against ``sandbox.fs_root``. ``execute`` contains
zero permission logic — sandbox containment is the gate's job, and the
only path to ``execute`` is ``core.enforce.enforce_and_run``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from core.base import Tool, ToolContext, ToolResult
from core.effects import Effect, FileMode, FilesystemEffect
from core.paths import resolve_real


def _resolve_path(path: str, context: ToolContext) -> Path:
    """Resolve a path param: relative paths are under sandbox.fs_root.

    Resolution only — containment is checked by the gate, not here.
    """
    p = Path(path)
    if not p.is_absolute() and context.system.sandbox.fs_root is not None:
        p = Path(context.system.sandbox.fs_root) / p
    return resolve_real(p)


class _PathParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)


class _WriteParams(_PathParams):
    content: str


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a UTF-8 text file inside the sandbox."
    input_schema = _PathParams
    mutating = False
    destructive = False

    def plan_effects(self, params: _PathParams, context: ToolContext) -> list[Effect]:
        return [FilesystemEffect(path=str(_resolve_path(params.path, context)), mode=FileMode.READ)]

    def execute(self, params: _PathParams, context: ToolContext) -> ToolResult:
        target = _resolve_path(params.path, context)
        try:
            return ToolResult(ok=True, output=target.read_text(encoding="utf-8"))
        except OSError as e:
            return ToolResult(ok=False, error=f"read_file failed: {e}")


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write a UTF-8 text file inside the sandbox (creates or overwrites)."
    input_schema = _WriteParams
    mutating = True
    destructive = False

    def plan_effects(self, params: _WriteParams, context: ToolContext) -> list[Effect]:
        return [FilesystemEffect(path=str(_resolve_path(params.path, context)), mode=FileMode.WRITE)]

    def execute(self, params: _WriteParams, context: ToolContext) -> ToolResult:
        target = _resolve_path(params.path, context)
        try:
            target.write_text(params.content, encoding="utf-8")
            return ToolResult(ok=True, output=f"wrote {len(params.content)} chars to {target}")
        except OSError as e:
            return ToolResult(ok=False, error=f"write_file failed: {e}")


class DeleteFileTool(Tool):
    name = "delete_file"
    description = "Delete a file inside the sandbox."
    input_schema = _PathParams
    mutating = True
    destructive = True

    def plan_effects(self, params: _PathParams, context: ToolContext) -> list[Effect]:
        return [FilesystemEffect(path=str(_resolve_path(params.path, context)), mode=FileMode.WRITE)]

    def execute(self, params: _PathParams, context: ToolContext) -> ToolResult:
        target = _resolve_path(params.path, context)
        try:
            target.unlink()
            return ToolResult(ok=True, output=f"deleted {target}")
        except OSError as e:
            return ToolResult(ok=False, error=f"delete_file failed: {e}")
