# Copyright 2026 DataGen. All rights reserved.
"""Pluggable chat-history backends for the AI panel popover.

DATAGEN-FORK addition. The existing ``chat_history_store`` writes JSON
blobs to ``<notebook_dir>/.session/`` on the notebook's filesystem.
That's fine for OSS marimo but doesn't fit DataGen's deployment, where
chat threads must survive sandbox recreates and be visible to multiple
client paths (popover, "+ new chat" button, future cross-channel UI).

This module introduces an abstract provider so the endpoint layer
(``chat_history.py``) can call ``provider.load_index(notebook_path)``
without caring whether the data lives on disk or behind an HTTP API.
The provider is selected once at import time via the
``MARIMO_CHAT_HISTORY_PROVIDER`` env var:

  * ``file`` (default) - the original on-disk store. Behavior unchanged
    from upstream-vanilla marimo + the prior fork commits.

  * ``http`` - call out to a DataGen Wasp endpoint
    (``MARIMO_CHAT_HISTORY_URL``) authenticated via
    ``MARIMO_CHAT_HISTORY_AUTH_HEADER`` (sent as ``x-api-key``). The
    Wasp endpoint surfaces NOTEBOOK-channel ``ConversationSession``
    rows from postgres.

Scope of the v1 HTTP provider:
  * ``load_index`` is fully wired - the popover lists postgres-backed
    sessions filtered by notebook file. This is the main UX win.
  * ``load_chat``, ``save_chat``, ``delete_chat``, ``set_active_chat_id``
    are stubbed with TODO comments. Calling them today returns sensible
    "empty / no-op" responses so the panel doesn't crash. Full
    transcript translation (anthropic JSONL events -> marimo chat
    messages) is tracked separately - the SDK transcript bytes live in
    DataGen's ``NotebookAgentChatMessage`` deltas and need a careful
    translator. Until that lands, users see thread titles in the
    popover but clicking a past row shows an empty transcript.

The endpoint surface (``/api/chat_history/chats``,
``/api/chat_history/chats/{chat_id}``, etc.) is unchanged - only the
backing store flips. Reverting to the file store at any time is just an
env-var toggle.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from marimo import _loggers
from marimo._server import chat_history_store as file_store

LOGGER = _loggers.marimo_logger()

# Env var names (also set by DataGen's notebook sandbox boot script).
ENV_PROVIDER = "MARIMO_CHAT_HISTORY_PROVIDER"
ENV_URL = "MARIMO_CHAT_HISTORY_URL"
ENV_AUTH = "MARIMO_CHAT_HISTORY_AUTH_HEADER"


class ChatHistoryProvider(ABC):
    """Interface the chat_history endpoints call through.

    Methods mirror the existing ``chat_history_store`` function set so
    the endpoint layer's logic stays identical regardless of provider.
    Each method takes ``notebook_path`` as its first arg because that's
    how scoping works in the file store (relative to the notebook on
    disk); the HTTP provider passes the same path on to Wasp, which
    normalizes it server-side.
    """

    @abstractmethod
    def load_index(self, notebook_path: str | None) -> dict[str, Any]:
        ...

    @abstractmethod
    def load_chat(
        self, notebook_path: str | None, chat_id: str
    ) -> dict[str, Any] | None:
        ...

    @abstractmethod
    def save_chat(
        self, notebook_path: str | None, chat: dict[str, Any]
    ) -> Path | str | None:
        ...

    @abstractmethod
    def delete_chat(self, notebook_path: str | None, chat_id: str) -> bool:
        ...

    @abstractmethod
    def set_active_chat_id(
        self, notebook_path: str | None, active_chat_id: str | None
    ) -> Any:
        ...


class FileChatHistoryProvider(ChatHistoryProvider):
    """Delegates to the original on-disk ``chat_history_store``.

    Default selection. Keeps OSS marimo deployments untouched.
    """

    def load_index(self, notebook_path: str | None) -> dict[str, Any]:
        return file_store.load_index(notebook_path)

    def load_chat(
        self, notebook_path: str | None, chat_id: str
    ) -> dict[str, Any] | None:
        return file_store.load_chat(notebook_path, chat_id)

    def save_chat(
        self, notebook_path: str | None, chat: dict[str, Any]
    ) -> Path | None:
        return file_store.save_chat(notebook_path, chat)

    def delete_chat(self, notebook_path: str | None, chat_id: str) -> bool:
        return file_store.delete_chat(notebook_path, chat_id)

    def set_active_chat_id(
        self, notebook_path: str | None, active_chat_id: str | None
    ) -> Path | None:
        return file_store.set_active_chat_id(notebook_path, active_chat_id)


class HttpChatHistoryProvider(ChatHistoryProvider):
    """Talks to DataGen's Wasp ``/api/notebook/conversation-sessions``
    surface.

    Auth: ``MARIMO_CHAT_HISTORY_AUTH_HEADER`` is sent as ``x-api-key``.
    That value is the per-MarimoSession ``notebookApiKey`` injected at
    boot time - same credential marimo.toml's ``[ai.anthropic]`` block
    already uses, so there's nothing new for operators to provision.

    All calls degrade to "empty" rather than raising. The chat popover
    can render an empty index; crashing on an upstream blip would
    completely break the panel. Errors are logged at WARNING for ops
    visibility.
    """

    def __init__(self, base_url: str, auth_header: str | None) -> None:
        # Strip trailing slash so ``f"{base}/sessions"`` doesn't double.
        self._base_url = base_url.rstrip("/")
        self._auth_header = auth_header
        # Conservative timeout. The popover blocks the panel UI on this
        # call, but Wasp is on the same host (or one network hop) and
        # the underlying query is a single indexed SELECT.
        self._timeout = 10.0

    def _build_request(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
    ) -> urllib.request.Request:
        headers: dict[str, str] = {"accept": "application/json"}
        if self._auth_header:
            headers["x-api-key"] = self._auth_header
        data: bytes | None = None
        if body is not None:
            headers["content-type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        return urllib.request.Request(
            url, data=data, headers=headers, method=method
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        if params:
            # Drop None-valued params so absent notebook_path doesn't
            # become "notebookPath=None" on the wire.
            filtered = {k: v for k, v in params.items() if v is not None}
            if filtered:
                url = f"{url}?{urllib.parse.urlencode(filtered)}"
        req = self._build_request(method, url, body)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status == 204:
                    return None
                raw = resp.read()
                if not raw:
                    return None
                return json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            LOGGER.warning(
                "HttpChatHistoryProvider: %s %s failed: %s", method, url, exc
            )
            return None

    @staticmethod
    def _wasp_to_marimo_entry(row: dict[str, Any]) -> dict[str, Any]:
        """Translate a ``ConversationSessionIndex`` (Wasp wire shape) to
        the marimo popover's chat-index entry shape.

        Wasp shape: id, marimoChatId, title, agentId, agentName,
        conversationKey, claudeSessionId, notebookPath, rootExecutionId,
        lastTurnAt, turnCount.

        Marimo shape: id, title, createdAt, updatedAt, agentSessionId,
        messageCount.

        Mapping choices:
          * ``id`` = marimoChatId when present (so marimo sees its own
            client-minted chat_id round-trip). Falls back to
            ``ConversationSession.id`` (UUID) for legacy rows that
            predate the chat_id wiring (slot="0" in conversationKey).
          * ``title`` = the row's stored title (set by marimo's
            save_chat); falls back to agentName when null so the
            popover always shows something meaningful.
          * ``createdAt`` / ``updatedAt`` both come from ``lastTurnAt``.
          * ``agentSessionId`` maps to ``claudeSessionId`` - the SDK
            resume target the chat panel uses to continue this thread.
          * ``messageCount`` is ``turnCount``.
        """
        agent_name = row.get("agentName") or ""
        stored_title = row.get("title")
        title = stored_title or agent_name or "(deleted agent)"
        last_turn = row.get("lastTurnAt") or ""
        # Marimo's frontend expects unix-ms timestamps, but the Wasp
        # endpoint returns ISO 8601 strings. Convert when possible;
        # fall back to 0 so the popover's sort doesn't NaN-explode.
        ts_ms = 0
        if last_turn:
            try:
                from datetime import datetime

                ts_ms = int(
                    datetime.fromisoformat(
                        last_turn.replace("Z", "+00:00")
                    ).timestamp()
                    * 1000
                )
            except (ValueError, TypeError):
                ts_ms = 0
        external_id = row.get("marimoChatId") or row.get("id")
        return {
            "id": external_id,
            "title": title,
            "createdAt": ts_ms,
            "updatedAt": ts_ms,
            "agentSessionId": row.get("claudeSessionId"),
            "messageCount": row.get("turnCount") or 0,
        }

    @staticmethod
    def _events_to_ui_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate Claude SDK JSONL events to marimo's ``UIMessage[]``
        (Vercel AI SDK shape).

        SDK events to keep:
          * ``{"type":"user","message":{role:"user",content:[...]}}`` →
            one UIMessage with role="user" and parts mirrored from content.
          * ``{"type":"assistant","message":{role:"assistant",content:[...],
            id:"msg_..."}}`` → one UIMessage with role="assistant" and
            parts including text + tool-call entries.
          * ``{"type":"tool_result","tool_use_id":"tu_X","content":...}``
            → attach the result onto the matching tool-call part of the
            most recent assistant message (by tool_use id), upgrading
            its state to ``"output-available"``.

        Events to skip (marimo-fork wrappers + SDK internals not user-
        visible): ``ai-title``, ``queue-operation``, ``last-prompt``,
        ``attachment``, ``system`` / ``system/init``, any event without
        a ``message`` field that we can flatten.

        Content-item mapping:
          * ``{"type":"text","text":"..."}``  → ``{"type":"text","text":...}``
          * ``{"type":"tool_use","id":...,"name":...,"input":...}`` →
            ``{"type":"tool-<name>","toolCallId":...,
              "state":"input-available","input":...}``
            (state upgrades to ``"output-available"`` if a tool_result
            event lands for this id)
          * other content shapes are best-effort emitted as text using
            their JSON representation, so the chat panel renders SOMETHING
            instead of dropping the part silently.

        The output is a list of UIMessage dicts ready to drop into a
        marimo Chat blob's ``messages`` array.
        """
        ui_messages: list[dict[str, Any]] = []
        # Index from tool_use id → (message index in ui_messages, part
        # index inside that message's parts). Used to upgrade a tool-call
        # part's state when a matching tool_result event arrives.
        tool_call_index: dict[str, tuple[int, int]] = {}

        for ev in events:
            t = ev.get("type")
            if t == "user":
                msg = ev.get("message") or {}
                if msg.get("role") != "user":
                    continue
                parts = []
                for c in msg.get("content") or []:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append({"type": "text", "text": c.get("text", "")})
                    elif isinstance(c, str):
                        parts.append({"type": "text", "text": c})
                if not parts:
                    continue
                uuid = ev.get("uuid") or f"u_{len(ui_messages)}"
                ui_messages.append(
                    {"id": uuid, "role": "user", "parts": parts}
                )
            elif t == "assistant":
                msg = ev.get("message") or {}
                if msg.get("role") != "assistant":
                    continue
                parts: list[dict[str, Any]] = []
                for c in msg.get("content") or []:
                    if not isinstance(c, dict):
                        continue
                    ctype = c.get("type")
                    if ctype == "text":
                        parts.append(
                            {"type": "text", "text": c.get("text", "")}
                        )
                    elif ctype == "tool_use":
                        tool_name = c.get("name", "tool")
                        tool_id = c.get("id") or f"tu_{len(parts)}"
                        parts.append(
                            {
                                "type": f"tool-{tool_name}",
                                "toolCallId": tool_id,
                                "state": "input-available",
                                "input": c.get("input"),
                            }
                        )
                        # Record for later result attachment.
                        tool_call_index[tool_id] = (
                            len(ui_messages),
                            len(parts) - 1,
                        )
                    elif ctype == "thinking":
                        # Map Claude's thinking blocks to reasoning parts
                        # so they render with appropriate UX.
                        parts.append(
                            {"type": "reasoning", "text": c.get("thinking", "")}
                        )
                if not parts:
                    continue
                msg_id = msg.get("id") or ev.get("uuid") or f"a_{len(ui_messages)}"
                ui_messages.append(
                    {"id": msg_id, "role": "assistant", "parts": parts}
                )
            elif t == "tool_result":
                tool_use_id = ev.get("tool_use_id") or (
                    ev.get("message", {}) or {}
                ).get("tool_use_id")
                if not tool_use_id or tool_use_id not in tool_call_index:
                    continue
                msg_idx, part_idx = tool_call_index[tool_use_id]
                part = ui_messages[msg_idx]["parts"][part_idx]
                # Tool result `content` can be a string or a list of
                # content blocks. Flatten to a string for the part's
                # output; the chat panel can render either.
                raw = ev.get("content") or (ev.get("message", {}) or {}).get(
                    "content"
                )
                if isinstance(raw, list):
                    out_text = "\n".join(
                        c.get("text", "")
                        for c in raw
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                else:
                    out_text = str(raw) if raw is not None else ""
                part["state"] = "output-available"
                part["output"] = out_text
            # Everything else (ai-title, queue-operation, last-prompt,
            # attachment, system, ...) is filtered out: not user-visible
            # content.

        return ui_messages

    def load_index(self, notebook_path: str | None) -> dict[str, Any]:
        # Wasp accepts the absolute path and strips NOTEBOOKS_DIR
        # server-side via ``normalizeNotebookPath``. Sending it raw
        # keeps the provider stateless about the sandbox layout.
        result = self._request_json(
            "GET",
            "",
            params={"notebookPath": notebook_path},
        )
        if not isinstance(result, dict) or not isinstance(
            result.get("sessions"), list
        ):
            return {"activeChatId": None, "chats": []}
        entries = [
            self._wasp_to_marimo_entry(row) for row in result["sessions"]
        ]
        return {"activeChatId": None, "chats": entries}

    def load_chat(
        self, notebook_path: str | None, chat_id: str
    ) -> dict[str, Any] | None:
        """Fetch a single ConversationSession's transcript and translate
        its SDK JSONL events into marimo's UIMessage[] shape so the chat
        panel can repaint when the user clicks a past popover row.

        ``chat_id`` is whatever the popover surfaced as the row's id.
        That's normally marimo's external ``marimoChatId``; for legacy
        rows (slot=0) it's a ConversationSession UUID. The endpoint
        accepts either.
        """
        result = self._request_json("GET", f"/{urllib.parse.quote(chat_id)}")
        if not isinstance(result, dict):
            return None
        session = result.get("session") or {}
        events = result.get("events") or []
        if not isinstance(events, list):
            events = []
        agent_name = session.get("agentName") or ""
        stored_title = session.get("title")
        title = stored_title or agent_name or ""
        last_turn = session.get("lastTurnAt") or ""
        ts_ms = 0
        if last_turn:
            try:
                from datetime import datetime

                ts_ms = int(
                    datetime.fromisoformat(
                        last_turn.replace("Z", "+00:00")
                    ).timestamp()
                    * 1000
                )
            except (ValueError, TypeError):
                ts_ms = 0
        external_id = session.get("marimoChatId") or session.get("id") or chat_id
        return {
            "id": external_id,
            "title": title,
            "createdAt": ts_ms,
            "updatedAt": ts_ms,
            "agentSessionId": session.get("claudeSessionId"),
            "messages": self._events_to_ui_messages(events),
        }

    def save_chat(
        self, notebook_path: str | None, chat: dict[str, Any]
    ) -> str | None:
        """Persist marimo's chat blob: refresh the popover row's title.

        Most of the work the file-backed provider does (storing full
        message bytes) is already done by the proxy turn path: when the
        AI panel POSTs ``/v1/messages`` with ``X-Marimo-Chat-Id``, the
        proxy upserts the ConversationSession + appends the JSONL
        delta. By the time ``save_chat`` is invoked, postgres already
        has the SDK transcript.

        What save_chat brings to the table: the popover ROW TITLE.
        Marimo's frontend derives a title from the first user message
        (or sometimes from an AI summary) and includes it in the Chat
        blob. We POST it as a title-only update so the popover renders
        the right label.

        The POST is a no-op when ``chat["id"]`` doesn't reference an
        existing row (e.g. the user clicked "+" but hasn't sent a
        message yet — the proxy hasn't created the row). The endpoint
        returns 400; we swallow the error and return None.
        """
        chat_id = chat.get("id")
        if not isinstance(chat_id, str) or not chat_id:
            return None
        title = chat.get("title")
        if not isinstance(title, str):
            return None
        title_stripped = title.strip()
        if not title_stripped:
            return None
        # marimoChatId-only update: server resolves the existing row by
        # (workspace, user, marimoChatId) and refreshes title. No
        # agentId required (already set on the existing row).
        self._request_json(
            "POST",
            "",
            body={"marimoChatId": chat_id, "title": title_stripped},
        )
        return None

    def delete_chat(self, notebook_path: str | None, chat_id: str) -> bool:
        # The Wasp DELETE endpoint returns 204 on success and 404 on
        # unknown id. ``_request_json`` returns ``None`` either way; we
        # report True optimistically so the popover refreshes.
        self._request_json("DELETE", f"/{urllib.parse.quote(chat_id)}")
        return True

    def set_active_chat_id(
        self, notebook_path: str | None, active_chat_id: str | None
    ) -> None:
        # We don't currently persist "active" selection server-side.
        # The popover treats the active id as client-side state; this
        # is just an acknowledgement.
        return None


def _build_provider() -> ChatHistoryProvider:
    """Read the env vars once at import time and instantiate the
    provider. Errors here log + fall back to the file provider, since
    the chat panel breaking the whole notebook editor would be a much
    worse failure mode than missing chat history.
    """
    choice = os.environ.get(ENV_PROVIDER, "file").strip().lower()
    if choice == "http":
        url = os.environ.get(ENV_URL, "").strip()
        if not url:
            LOGGER.warning(
                "MARIMO_CHAT_HISTORY_PROVIDER=http but "
                "MARIMO_CHAT_HISTORY_URL is unset; falling back to file store"
            )
            return FileChatHistoryProvider()
        auth = os.environ.get(ENV_AUTH, "").strip() or None
        if not auth:
            LOGGER.info(
                "HttpChatHistoryProvider: no auth header set "
                "(MARIMO_CHAT_HISTORY_AUTH_HEADER unset); Wasp will likely "
                "reject every request"
            )
        return HttpChatHistoryProvider(base_url=url, auth_header=auth)
    if choice not in ("file", ""):
        LOGGER.warning(
            "MARIMO_CHAT_HISTORY_PROVIDER=%r unrecognized; using file store",
            choice,
        )
    return FileChatHistoryProvider()


_provider: ChatHistoryProvider | None = None


def get_provider() -> ChatHistoryProvider:
    """Return the process-wide provider instance, building it lazily on
    first call so import-time env var changes (e.g. test fixtures) are
    picked up before the first endpoint runs."""
    global _provider
    if _provider is None:
        _provider = _build_provider()
    return _provider
