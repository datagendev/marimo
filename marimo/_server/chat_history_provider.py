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

        Wasp shape: id, agentId, agentName, conversationKey,
        claudeSessionId, notebookPath, rootExecutionId, lastTurnAt,
        turnCount.

        Marimo shape: id, title, createdAt, updatedAt, agentSessionId,
        messageCount.

        Mapping choices:
          * ``id`` carried through verbatim - ConversationSession.id is
            stable, opaque, and globally unique.
          * ``title`` derived from agent name; if the agent was deleted
            we fall back to a short stub. Marimo's popover shows this
            as the row label.
          * ``createdAt`` / ``updatedAt`` both come from the Wasp row's
            ``lastTurnAt``. We don't ship a separate createdAt today
            (the popover only sorts on updatedAt). The frontend
            tolerates equality.
          * ``agentSessionId`` maps to ``claudeSessionId`` - they're
            both "the SDK resume target" by another name.
          * ``messageCount`` is ``turnCount``.
        """
        agent_name = row.get("agentName") or ""
        title = agent_name if agent_name else "(deleted agent)"
        last_turn = row.get("lastTurnAt") or ""
        return {
            "id": row.get("id"),
            "title": title,
            "createdAt": last_turn,
            "updatedAt": last_turn,
            "agentSessionId": row.get("claudeSessionId"),
            "messageCount": row.get("turnCount") or 0,
        }

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
        """Fetch a single ConversationSession's transcript.

        TODO(notebook-conversations): translate the Wasp
        ``events: SDKEvent[]`` array (anthropic JSONL parsed) into
        marimo's per-chat ``messages: [{role, content, ...}]`` shape.
        Until that translator lands, return a minimal envelope so the
        chat panel doesn't crash when a user clicks a past row - they
        see the title + empty body. New turns still work because they
        go through the proxy, which writes deltas directly into
        ``NotebookAgentChatMessage`` and shows them live via streaming.
        """
        result = self._request_json("GET", f"/{urllib.parse.quote(chat_id)}")
        if not isinstance(result, dict):
            return None
        session = result.get("session") or {}
        last_turn = session.get("lastTurnAt") or ""
        agent_session_id = session.get("claudeSessionId")
        return {
            "id": session.get("id") or chat_id,
            "title": session.get("agentName") or "",
            "createdAt": last_turn,
            "updatedAt": last_turn,
            "agentSessionId": agent_session_id,
            # TODO: replace with parseDeltasToEvents-equivalent that
            # flattens anthropic-shape events into marimo's
            # role/content message form.
            "messages": [],
        }

    def save_chat(
        self, notebook_path: str | None, chat: dict[str, Any]
    ) -> str | None:
        """No-op upsert.

        Marimo POSTs the whole chat blob after every turn so the
        on-disk store can re-serialize messages. In DataGen's model the
        proxy's ``runChatTurn`` already persisted JSONL deltas before
        the panel got its assistant response, so there's nothing new
        for the popover to write. Returning ``None`` (which the
        endpoint translates to a 204) keeps the panel happy.

        We do NOT create new ConversationSession rows here. The "+" new-
        chat path runs through ``POST /api/notebook/conversation-sessions``
        directly; this method is only called for save-on-turn-complete.
        """
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
