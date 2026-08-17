"""Local web server: a second client of AgentService.

It owns no policy and calls no tool. A turn runs through the same loop and
the same gate the CLI uses; the Approve/Deny buttons are the human's answer
to a gate question, delivered through ``resolve_pending`` -> ``resume``.
The browser never gets an approver of its own, so a suspended action stays
suspended until a person decides — the Phase 2 headless path, with buttons.

BINDING: loopback only, as a module constant. There is deliberately no
config field for the host, so no configuration mistake can expose this to a
network. There is no authentication either — the loopback boundary IS the
control, which is exactly why hosting is a separate track with its own
decisions (accounts, isolation, and who pays for model usage).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from core.activity import ActivityEvent
from core.paths import PathEscapeError
from runtime import AgentService, SessionNotFoundError, TurnUpdate

from .uploads import UploadRejected, store_upload

logger = logging.getLogger("agent.server")

# Not configurable, by design (see module docstring).
BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

STATIC_DIR = Path(__file__).resolve().parent / "static"


class _RevalidatingStatic(StaticFiles):
    """Serve the console's assets with `Cache-Control: no-cache`.

    Not "don't cache" — "check with me first". The browser keeps the file
    and revalidates, so it stays cheap, but an edited stylesheet or script
    always takes effect on the next refresh.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _asset_version() -> str:
    """Newest mtime across the front-end assets, as a cache key."""
    stamps = [
        int(path.stat().st_mtime)
        for path in (STATIC_DIR / "app.js", STATIC_DIR / "style.css")
        if path.is_file()
    ]
    return str(max(stamps)) if stamps else "0"


def render_console_page() -> str:
    """index.html with version-stamped asset links.

    Headers cannot dislodge a copy a browser has already cached, so the URL
    itself changes when a file changes. This is what makes an edit show up
    on a plain refresh instead of stranding someone on an old page — the
    failure mode that hid the drop-veil bug behind "it does it even after
    reloading".
    """
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    version = _asset_version()
    return (html
            .replace('href="/style.css"', f'href="/style.css?v={version}"')
            .replace('src="/app.js"', f'src="/app.js?v={version}"'))


class StartSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent: str = Field(min_length=1)


def _update_payload(update: TurnUpdate) -> dict[str, Any]:
    return {
        "status": update.status,
        "text": update.text,
        "detail": update.detail,
        "budget": update.budget,
        "pending": None if update.pending is None else {
            "tool": update.pending.tool,
            "reason": update.pending.reason,
            "params": update.pending.params,
        },
    }


def create_app(service: AgentService | None = None) -> FastAPI:
    app = FastAPI(title="Agent console", docs_url=None, redoc_url=None)
    app.state.service = service if service is not None else AgentService()

    def get_service() -> AgentService:
        return app.state.service

    # -- agents & sessions --------------------------------------------------

    @app.get("/api/agents")
    def list_agents() -> dict[str, Any]:
        return {"agents": get_service().list_agents()}

    @app.post("/api/session")
    def start_session(request: StartSessionRequest) -> dict[str, Any]:
        try:
            session_id = get_service().start_session(request.agent)
        except Exception as e:  # unknown/unrunnable profile
            raise HTTPException(status_code=400, detail=str(e)) from None
        return get_service().status(session_id)

    @app.get("/api/session/{session_id}")
    def session_status(session_id: str) -> dict[str, Any]:
        try:
            return get_service().status(session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="No such session.") from None

    # -- turns over HTTP (the WebSocket is the live path) --------------------

    @app.post("/api/session/{session_id}/message")
    async def send_message(session_id: str, body: dict) -> dict[str, Any]:
        text = (body or {}).get("text", "")
        if not str(text).strip():
            raise HTTPException(status_code=400, detail="Type a message first.")
        try:
            update = await asyncio.to_thread(get_service().send, session_id, str(text))
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="No such session.") from None
        return _update_payload(update)

    async def _decide(session_id: str, approved: bool) -> dict[str, Any]:
        try:
            update = await asyncio.to_thread(
                get_service().resolve_pending, session_id, approved
            )
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="No such session.") from None
        return _update_payload(update)

    @app.post("/api/session/{session_id}/approve")
    async def approve(session_id: str) -> dict[str, Any]:
        return await _decide(session_id, True)

    @app.post("/api/session/{session_id}/deny")
    async def deny(session_id: str) -> dict[str, Any]:
        return await _decide(session_id, False)

    # -- files --------------------------------------------------------------

    @app.post("/api/session/{session_id}/upload")
    async def upload(session_id: str, file: UploadFile) -> dict[str, Any]:
        try:
            root = get_service().sandbox_root(session_id)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="No such session.") from None
        except PathEscapeError:
            raise HTTPException(
                status_code=400,
                detail="This agent has no folder to store files in.",
            ) from None
        data = await file.read()
        try:
            stored = store_upload(file.filename or "", data, root)
        except UploadRejected as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        return {"name": stored.name, "bytes": len(data)}

    @app.get("/api/session/{session_id}/files/{name}")
    def download(session_id: str, name: str) -> FileResponse:
        try:
            path = get_service().confine_to_sandbox(session_id, name)
        except SessionNotFoundError:
            raise HTTPException(status_code=404, detail="No such session.") from None
        except PathEscapeError:
            raise HTTPException(status_code=400, detail="That file is out of bounds.") from None
        if not path.is_file():
            raise HTTPException(status_code=404, detail="No such file.")
        return FileResponse(path, filename=path.name)

    # -- the live channel ---------------------------------------------------

    @app.websocket("/api/session/{session_id}/stream")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        service = get_service()
        try:
            service.status(session_id)
        except SessionNotFoundError:
            await websocket.send_json({"type": "error", "message": "No such session."})
            await websocket.close()
            return

        event_loop = asyncio.get_running_loop()

        async def run(work, *args) -> None:
            """Run a turn off the event loop, streaming activity as it goes."""
            queue: asyncio.Queue = asyncio.Queue()

            def sink(event: ActivityEvent) -> None:
                # Called from the worker thread: hand off, never block.
                event_loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "activity", "kind": event.kind,
                     "tool": event.tool, "detail": event.detail},
                )

            task = asyncio.create_task(asyncio.to_thread(work, *args, sink))
            while True:
                drain = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait(
                    {drain, task}, return_when=asyncio.FIRST_COMPLETED
                )
                if drain in done:
                    await websocket.send_json(drain.result())
                    continue
                drain.cancel()
                break
            while not queue.empty():
                await websocket.send_json(queue.get_nowait())
            update = task.result()
            await websocket.send_json({"type": "update", **_update_payload(update)})

        try:
            while True:
                message = await websocket.receive_json()
                kind = message.get("type")
                if kind == "message":
                    text = str(message.get("text", "")).strip()
                    if not text:
                        continue
                    await run(
                        lambda sid, txt, sink: service.send(sid, txt, on_activity=sink),
                        session_id, text,
                    )
                elif kind == "decision":
                    approved = bool(message.get("approved"))
                    await run(
                        lambda sid, ok, sink: service.resolve_pending(
                            sid, ok, on_activity=sink),
                        session_id, approved,
                    )
                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Unknown message type: {kind!r}"}
                    )
        except WebSocketDisconnect:
            # The session (and any pending decision) lives in the service,
            # so a dropped socket loses nothing — reconnect and it is there.
            logger.info("client disconnected from session %s", session_id)
        except Exception as e:  # noqa: BLE001 — report, never crash the socket
            logger.exception("stream failed")
            try:
                await websocket.send_json(
                    {"type": "error", "message": f"{type(e).__name__}: {e}"}
                )
            except Exception:
                pass

    @app.exception_handler(SessionNotFoundError)
    def _session_missing(_request, _exc) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "No such session."})

    if STATIC_DIR.is_dir():
        # Registered before the mount so it wins for the page itself; the
        # mount still serves the assets it points at.
        @app.get("/", include_in_schema=False)
        def console_page() -> HTMLResponse:
            return HTMLResponse(
                render_console_page(), headers={"Cache-Control": "no-store"}
            )

        app.mount("/", _RevalidatingStatic(directory=STATIC_DIR, html=True), name="static")
    return app


def run(port: int = DEFAULT_PORT) -> None:  # pragma: no cover — process entry
    import uvicorn

    print(f"Agent console: http://{BIND_HOST}:{port}")
    uvicorn.run(create_app(), host=BIND_HOST, port=port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run the local agent console.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    run(parser.parse_args().port)
