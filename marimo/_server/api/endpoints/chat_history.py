# Copyright 2026 Marimo. All rights reserved.
"""Endpoints persisting the chat panel's history next to the notebook.

The chat panel POSTs after every turn so the popover's "previous chats"
list survives browser refreshes, sandbox restarts, and machine moves —
chats follow the notebook, not the browser's localStorage. Storage layout
is owned by ``marimo._server.chat_history_store``.

URL prefix: ``/api/chat_history`` (see router.py for the mount point).
Frontend client: ``frontend/src/core/ai/chat-persistence.ts``.

DATAGEN-FORK note: this whole module is additive (no upstream marimo
file depends on it). Removing the include_router call in router.py is
a clean revert to upstream-vanilla behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from starlette.authentication import requires
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, Response

from marimo import _loggers
from marimo._server.api.deps import AppState
from marimo._server.chat_history_store import (
    delete_chat,
    load_chat,
    load_index,
    save_chat,
    set_active_chat_id,
)
from marimo._server.router import APIRouter

if TYPE_CHECKING:
    from starlette.requests import Request

LOGGER = _loggers.marimo_logger()

router = APIRouter()


def _notebook_path(state: AppState) -> str | None:
    """Resolve the absolute path of the current session's notebook,
    or ``None`` for unsaved buffers."""
    session = state.require_current_session()
    return session.app_file_manager.path


@router.get("/chats")
@requires("edit")
async def list_chats(*, request: Request) -> JSONResponse:
    """Return the chat index — the cheap subset the popover needs.

    Shape:
        {"activeChatId": str | null,
         "chats": [{id, title, createdAt, updatedAt, agentSessionId,
                    messageCount}, ...]}

    Returns an empty index for unsaved notebooks. Auto-migrates a
    legacy ``chats.json`` if one is present.
    """
    state = AppState(request)
    state.require_current_session()
    return JSONResponse(load_index(_notebook_path(state)))


@router.get("/chats/{chat_id}")
@requires("edit")
async def get_chat(*, request: Request) -> Response:
    """Return one chat's full body (messages, attachments, etc.).

    404 when the chat doesn't exist. Notebook path is taken from the
    server's session, never from the client.
    """
    state = AppState(request)
    state.require_current_session()
    chat_id = request.path_params["chat_id"]
    try:
        chat = load_chat(_notebook_path(state), chat_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if chat is None:
        raise HTTPException(status_code=404, detail="chat not found")
    return JSONResponse(chat)


@router.post("/chats/{chat_id}")
@requires("edit")
async def upsert_chat(*, request: Request) -> Response:
    """Atomically write one chat blob.

    Body must be a JSON object with at least an ``id`` field that
    matches the URL's ``chat_id``. Drops the write for unsaved
    notebooks (client keeps a localStorage copy).
    """
    state = AppState(request)
    state.require_current_session()
    chat_id = request.path_params["chat_id"]
    body: Any
    try:
        body = await request.json()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {e}")
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="chat must be a JSON object"
        )
    if body.get("id") != chat_id:
        raise HTTPException(
            status_code=400,
            detail="body 'id' must match URL chat_id",
        )
    try:
        saved_to = save_chat(_notebook_path(state), body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if saved_to is None:
        return Response(status_code=204)
    return JSONResponse({"path": str(saved_to)})


@router.delete("/chats/{chat_id}")
@requires("edit")
async def remove_chat(request: Request) -> Response:
    # NB: marimo's `APIRouter.delete` registers the handler directly,
    # without the keyword-binding wrapper that `APIRouter.post` uses
    # (router.py:46 vs router.py:114). starlette therefore calls this
    # endpoint with `request` as a positional argument, so the
    # signature must be positional-friendly. Sister handlers in this
    # file that use `*, request:` are fine because they're all POST.
    state = AppState(request)
    state.require_current_session()
    chat_id = request.path_params["chat_id"]
    try:
        delete_chat(_notebook_path(state), chat_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(status_code=204)


@router.post("/chats:active")
@requires("edit")
async def set_active(*, request: Request) -> Response:
    """Persist the popover's active selection without touching any chat
    body. Body: ``{"activeChatId": str | null}``."""
    state = AppState(request)
    state.require_current_session()
    try:
        body = await request.json()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {e}")
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="body must be a JSON object"
        )
    active = body.get("activeChatId")
    if active is not None and not isinstance(active, str):
        raise HTTPException(
            status_code=400, detail="activeChatId must be a string or null"
        )
    set_active_chat_id(_notebook_path(state), active)
    return Response(status_code=204)
