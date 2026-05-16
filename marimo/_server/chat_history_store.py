"""Project-local persistence for the chat panel.

Layout (next to the notebook):

    <notebook_dir>/.session/
      index.json                 # cheap to load: titles, timestamps, active id
      <chat_id>/
        chat.json                # full chat: messages + agent_session_id

The index lets the popover render instantly without parsing every chat
body. Individual chats are fetched on demand when the user opens one.

The endpoint layer treats individual chat blobs as opaque — the only
fields we look at are ``id``, ``title``, ``updatedAt``, and
``agentSessionId`` (plus ``messages`` length, for the index's
``messageCount`` hint). The schema lives in the frontend.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from marimo import _loggers

LOGGER = _loggers.marimo_logger()

CHAT_STORE_DIRNAME = ".session"
INDEX_FILENAME = "index.json"
CHAT_FILENAME = "chat.json"
LEGACY_FILENAME = "chats.json"

# Cell-id-style ids are short alphanumerics; ban path separators and
# leading dots so a malicious id can't escape ``.session/``. Anything
# weird gets rejected up-front rather than written under a surprising
# filesystem location.
_BANNED_PATH_PARTS = {"", ".", ".."}


def _validate_chat_id(chat_id: str) -> None:
    if not isinstance(chat_id, str) or chat_id in _BANNED_PATH_PARTS:
        raise ValueError(f"invalid chat_id: {chat_id!r}")
    if "/" in chat_id or "\\" in chat_id or "\0" in chat_id:
        raise ValueError(f"invalid chat_id: {chat_id!r}")
    if chat_id.startswith("."):
        raise ValueError(f"invalid chat_id: {chat_id!r}")


def store_dir(notebook_path: str | None) -> Path | None:
    """Return ``<notebook_dir>/.session/`` for the given notebook, or
    ``None`` if the notebook is unsaved."""
    if not notebook_path:
        return None
    nb = Path(notebook_path).expanduser().resolve()
    return nb.parent / CHAT_STORE_DIRNAME


def index_path(notebook_path: str | None) -> Path | None:
    d = store_dir(notebook_path)
    return d / INDEX_FILENAME if d else None


def chat_path(notebook_path: str | None, chat_id: str) -> Path | None:
    _validate_chat_id(chat_id)
    d = store_dir(notebook_path)
    return d / chat_id / CHAT_FILENAME if d else None


def _atomic_write_json(target: Path, data: Any) -> None:
    """Write JSON via temp file + ``os.replace`` so a crash mid-write
    can't leave a half-written file in the user's project."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp-",
        suffix=".json",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _read_json(p: Path) -> Any:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        LOGGER.warning("chat_storage: failed to read %s: %s", p, e)
        return None


def _index_entry_from_chat(chat: dict[str, Any]) -> dict[str, Any]:
    """Pick the fields the popover needs out of a full chat blob."""
    msgs = chat.get("messages") or []
    return {
        "id": chat.get("id"),
        "title": chat.get("title", ""),
        "createdAt": chat.get("createdAt"),
        "updatedAt": chat.get("updatedAt"),
        "agentSessionId": chat.get("agentSessionId"),
        "messageCount": len(msgs) if isinstance(msgs, list) else 0,
    }


def _migrate_legacy(notebook_path: str) -> None:
    """One-shot migrator: split a legacy single-file ``chats.json`` into
    the per-chat layout. Run lazily on read.

    Old shape (frontend's localStorage wire format):
        {"chats": [[id, chat], ...], "activeChatId": str | null}

    Skipped silently if the legacy file is missing, malformed, or the
    new layout already exists alongside it.
    """
    d = store_dir(notebook_path)
    if d is None:
        return
    legacy = d / LEGACY_FILENAME
    idx = d / INDEX_FILENAME
    if not legacy.exists() or idx.exists():
        return
    raw = _read_json(legacy)
    if not isinstance(raw, dict):
        return
    entries = raw.get("chats") or []
    if not isinstance(entries, list):
        return
    LOGGER.info(
        "chat_storage: migrating %d chats from legacy chats.json layout",
        len(entries),
    )
    index: list[dict[str, Any]] = []
    for entry in entries:
        # Each entry is [id, chat] from Map.entries() serialization.
        if not (
            isinstance(entry, list) and len(entry) == 2 and isinstance(entry[1], dict)
        ):
            continue
        chat_id, chat = entry
        try:
            _validate_chat_id(chat_id)
        except ValueError:
            LOGGER.warning(
                "chat_storage: skipping legacy chat with invalid id %r", chat_id
            )
            continue
        target = d / chat_id / CHAT_FILENAME
        try:
            _atomic_write_json(target, chat)
        except OSError as e:
            LOGGER.warning(
                "chat_storage: failed to migrate chat %s: %s", chat_id, e
            )
            continue
        index.append(_index_entry_from_chat(chat))
    _atomic_write_json(
        idx,
        {"activeChatId": raw.get("activeChatId"), "chats": index},
    )
    # Park the legacy file alongside the new layout so the user can
    # confirm the migration before we ever delete it. Easier than a
    # backup elsewhere; harmless if it sits on disk forever.
    try:
        legacy.rename(legacy.with_suffix(".json.migrated"))
    except OSError:
        pass


def load_index(notebook_path: str | None) -> dict[str, Any]:
    """Return ``{"activeChatId": str|null, "chats": [entry, ...]}``.

    Returns an empty index when the notebook is unsaved or no chats
    have been written yet. Auto-migrates a legacy
    ``chats.json`` if one is sitting next to the new layout.
    """
    if notebook_path:
        _migrate_legacy(notebook_path)
    p = index_path(notebook_path)
    if p is None or not p.exists():
        return {"activeChatId": None, "chats": []}
    raw = _read_json(p)
    if not isinstance(raw, dict):
        return {"activeChatId": None, "chats": []}
    return {
        "activeChatId": raw.get("activeChatId"),
        "chats": raw.get("chats") or [],
    }


def load_chat(notebook_path: str | None, chat_id: str) -> dict[str, Any] | None:
    """Return the full chat blob for ``chat_id`` or ``None`` if missing."""
    p = chat_path(notebook_path, chat_id)
    if p is None or not p.exists():
        return None
    raw = _read_json(p)
    return raw if isinstance(raw, dict) else None


def save_chat(
    notebook_path: str | None,
    chat: dict[str, Any],
) -> Path | None:
    """Atomically write a single chat and refresh its index entry.

    Returns the path written, or ``None`` for unsaved notebooks. Raises
    ``ValueError`` if the chat blob has no usable id.
    """
    chat_id = chat.get("id")
    if not isinstance(chat_id, str):
        raise ValueError("chat must have a string 'id' field")
    _validate_chat_id(chat_id)

    target = chat_path(notebook_path, chat_id)
    if target is None:
        return None
    _atomic_write_json(target, chat)

    # Refresh the index so the popover reflects the new title / time
    # without having to re-scan every chat dir.
    idx_path = index_path(notebook_path)
    assert idx_path is not None
    current = _read_json(idx_path) if idx_path.exists() else None
    if not isinstance(current, dict):
        current = {"activeChatId": None, "chats": []}
    chats: list[dict[str, Any]] = current.get("chats") or []
    new_entry = _index_entry_from_chat(chat)
    replaced = False
    for i, existing in enumerate(chats):
        if isinstance(existing, dict) and existing.get("id") == chat_id:
            chats[i] = new_entry
            replaced = True
            break
    if not replaced:
        chats.append(new_entry)
    _atomic_write_json(
        idx_path,
        {"activeChatId": current.get("activeChatId"), "chats": chats},
    )
    return target


def delete_chat(notebook_path: str | None, chat_id: str) -> bool:
    """Remove a chat from disk. Returns True iff something was removed."""
    _validate_chat_id(chat_id)
    d = store_dir(notebook_path)
    if d is None:
        return False
    chat_dir = d / chat_id
    removed = False
    if chat_dir.exists():
        shutil.rmtree(chat_dir, ignore_errors=True)
        removed = True

    # Drop the entry from the index too so the popover stays accurate.
    idx_path = d / INDEX_FILENAME
    if idx_path.exists():
        current = _read_json(idx_path)
        if isinstance(current, dict):
            chats = [
                c
                for c in (current.get("chats") or [])
                if isinstance(c, dict) and c.get("id") != chat_id
            ]
            active = current.get("activeChatId")
            if active == chat_id:
                active = None
            _atomic_write_json(
                idx_path, {"activeChatId": active, "chats": chats}
            )
    return removed


def set_active_chat_id(
    notebook_path: str | None, active_chat_id: str | None
) -> Path | None:
    """Persist the popover's active selection without rewriting any chat."""
    p = index_path(notebook_path)
    if p is None:
        return None
    current = _read_json(p) if p.exists() else None
    if not isinstance(current, dict):
        current = {"activeChatId": None, "chats": []}
    current["activeChatId"] = active_chat_id
    _atomic_write_json(p, current)
    return p
