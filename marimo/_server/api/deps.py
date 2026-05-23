# Copyright 2026 Marimo. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from marimo import _loggers as loggers
from marimo._cli.tips import CliTip
from marimo._config.manager import MarimoConfigManager, ScriptConfigManager
from marimo._server.config import StarletteServerState
from marimo._server.session_manager import SessionManager
from marimo._server.tokens import SkewProtectionToken
from marimo._session.model import SessionMode
from marimo._types.ids import SessionId

if TYPE_CHECKING:
    from starlette.applications import Starlette
    from starlette.datastructures import State
    from starlette.requests import Request
    from starlette.websockets import WebSocket
    from uvicorn import Server

    from marimo._session import Session

LOGGER = loggers.marimo_logger()


class AppStateBase:
    """The app state."""

    @staticmethod
    def from_request(request: Request | WebSocket) -> AppState:
        """Get the app state with a request."""
        return AppState(request)

    @staticmethod
    def from_app(asgi: Starlette) -> AppStateBase:
        """Get the app state with an ASGIApp app."""
        return AppStateBase(cast(Any, asgi).state)

    def __init__(self, state: State) -> None:
        """Initialize the app state."""
        self.state = cast(StarletteServerState, state)

    @property
    def session_manager(self) -> SessionManager:
        return self.state.session_manager

    @property
    def mode(self) -> SessionMode:
        return self.session_manager.mode

    @property
    def quiet(self) -> bool:
        return self.state.quiet

    @property
    def host(self) -> str:
        return self.state.host

    @property
    def port(self) -> int:
        return self.state.port

    @property
    def maybe_port(self) -> int | None:
        return getattr(self.state, "port", None)

    @property
    def base_url(self) -> str:
        return self.state.base_url

    @property
    def server(self) -> Server:
        return self.state.server

    @property
    def config_manager(self) -> MarimoConfigManager:
        return self.state.config_manager

    @property
    def headless(self) -> bool:
        return self.state.headless

    @property
    def skew_protection(self) -> bool:
        return self.state.skew_protection

    @property
    def skew_protection_token(self) -> SkewProtectionToken:
        return self.session_manager.skew_protection_token

    @property
    def remote_url(self) -> str | None:
        if hasattr(self.state, "remote_url"):
            return self.state.remote_url
        return None

    @property
    def mcp_server_enabled(self) -> bool:
        return self.state.mcp_server_enabled

    @property
    def asset_url(self) -> str | None:
        if hasattr(self.state, "asset_url"):
            return self.state.asset_url
        return None

    @property
    def enable_auth(self) -> bool:
        if hasattr(self.state, "enable_auth"):
            return self.state.enable_auth
        return True

    @property
    def startup_tip(self) -> CliTip | None:
        startup_tip = getattr(self.state, "startup_tip", None)
        return cast(CliTip | None, startup_tip)

    @property
    def html_head(self) -> str | None:
        if hasattr(self.state, "html_head"):
            return cast(str | None, self.state.html_head)
        return None


class AppState(AppStateBase):
    """The app state with a request."""

    def __init__(self, request: Request | WebSocket) -> None:
        """Initialize the app state with a request."""
        super().__init__(request.app.state)
        self.request = request

    def get_current_session_id(self) -> SessionId | None:
        """Get the current session."""
        session_id = self.request.headers.get("Marimo-Session-Id")
        return SessionId(session_id) if session_id is not None else None

    def require_current_session_id(self) -> SessionId:
        """Get the current session or raise an error."""
        session_id = self.get_current_session_id()
        if session_id is None:
            raise ValueError("Missing Marimo-Session-Id header")
        return session_id

    def get_current_session(self) -> Session | None:
        """Get the current session."""
        session_id = self.get_current_session_id()
        if session_id is None:
            return None
        return self.session_manager.get_session(session_id)

    def require_current_session(self) -> Session:
        """Get the current session or raise an error.

        DATAGEN-FORK: when the inbound ``Marimo-Session-Id`` header is
        valid in shape (``/^s_[\\da-z]{6}$/``) but not registered in the
        session manager, lazy-register it against the kernel's unique
        file_key instead of raising. This handles the realistic case
        where the iframe loaded WITHOUT a ``?session_id=`` query param —
        e.g. Wasp's server-side ``POST /api/sessions/open`` call to the
        Modal tunnel URL failed or timed out, the row's
        ``marimoSessionId`` is null, the iframe URL omits the param, and
        the frontend's ``generateSessionId()`` mints a fresh cuid2 id
        (``s_<6 chars from [a-z0-9]>``) that the server never saw.

        Without this fallback, every AI chat call (``ai.py`` line 216,
        325, 399, 544; ``deps.py`` is on the require_current_session
        path) raises ``Invalid session id: s_xxxxxx`` and the panel is
        permanently broken until the user reboots the kernel. With this
        fallback, the kernel adopts the frontend's id on first reference
        and the rest of the request proceeds normally — same outcome as
        if ``/api/sessions/open`` had been called with that id, just
        deferred to the moment of first use.

        Safe-by-construction: only applies when ``file_router`` reports
        exactly one file_key. The DataGen sandbox always boots marimo
        against a single ``--port 2718 <notebookPath>`` argument, so
        this is the normal case for us; the upstream multi-file home
        page is intentionally NOT served here (it goes through
        ``/api/sessions/open`` or the WS handshake) so this lazy path
        never fires there.
        """
        session_id = self.require_current_session_id()
        session = self.session_manager.get_session(session_id)
        if session is None:
            # Attempt lazy-register before logging + raising. Local
            # imports avoid any circular dep with sessions_open.py.
            session = self._lazy_register_session(session_id)
            if session is not None:
                return session
            LOGGER.warning(
                "Valid sessions ids: %s",
                list(self.session_manager.sessions.keys()),
            )
            LOGGER.warning(
                "Valid consumers ids: %s",
                [
                    list(session.consumers.values())
                    for session in self.session_manager.sessions.values()
                    if session.consumers
                ],
            )
            raise ValueError(f"Invalid session id: {session_id}")
        return session

    def _lazy_register_session(
        self, session_id: SessionId
    ) -> Session | None:
        """DATAGEN-FORK: register a session under a frontend-minted id.

        Returns the newly-created Session on success, or None when the
        kernel doesn't have a unique file_key to bind against (in which
        case the caller falls through to raise as before).
        """
        from marimo._server.api.endpoints.sessions_open import (
            _HeadlessSessionConsumer,
        )

        sm = self.session_manager

        # Resolve file_key with three fallbacks, in order of trust:
        #   (1) router.get_unique_file_key() — works when marimo was
        #       launched against a single file path. Returns None for
        #       LazyListOfFilesAppFileRouter (directory mode), which is
        #       what DataGen actually uses (`marimo edit <dir>`).
        #   (2) request's ?file= query param — every iframe request and
        #       chat-history fetch carries this; it's the same value the
        #       WebSocket handshake uses for FILE_QUERY_PARAM_KEY.
        #   (3) any existing session's bound file — unambiguous only when
        #       there's exactly one open file, which is our deployment
        #       invariant (one notebook per sandbox). Lets a fresh probe
        #       latch onto whatever the browser-side session already
        #       registered.
        file_key: str | None = None
        try:
            file_key = sm.file_router.get_unique_file_key()
        except Exception:
            LOGGER.exception(
                "[deps] lazy-register: file_router.get_unique_file_key threw"
            )

        if file_key is None:
            try:
                file_key = self.request.query_params.get("file")
            except Exception:
                LOGGER.exception(
                    "[deps] lazy-register: reading ?file= query param threw"
                )

        if file_key is None:
            existing_paths = {
                getattr(getattr(s, "app_file_manager", None), "path", None)
                for s in sm.sessions.values()
            }
            existing_paths.discard(None)
            if len(existing_paths) == 1:
                file_key = next(iter(existing_paths))

        if not file_key:
            LOGGER.warning(
                "[deps] lazy-register: no file_key resolvable for session %s "
                "(router=%s, sessions=%d) — falling back to error",
                session_id,
                type(sm.file_router).__name__,
                len(sm.sessions),
            )
            return None

        # DEDUP: if a Session already serves this file_key, register the
        # incoming id as an ALIAS that points at the existing Session
        # instead of creating a parallel one. This preserves the
        # upstream "one Session per file" invariant that two other
        # mechanisms rely on:
        #
        #   - ws_endpoint._can_connect uses
        #     `manager.any_clients_connected(file_key)` to enforce
        #     "only one frontend per file". Parallel sessions all
        #     attach consumers for the same file_key, which makes
        #     every new tab see MARIMO_ALREADY_CONNECTED forever.
        #
        #   - SkewProtectionToken does leader election per-Session.
        #     Aliased ids share one Session and therefore one token,
        #     so the "Take over session" button actually takes over
        #     the only seat at the table.
        #
        # Aliasing = same Session object stored under multiple keys in
        # sm.sessions. `get_session(id)` continues to return the same
        # Session for any of those ids. No new kernel is spawned. No
        # consumer is duplicated.
        for existing_sid, existing_session in list(sm.sessions.items()):
            existing_fm = getattr(existing_session, "app_file_manager", None)
            existing_path = getattr(existing_fm, "path", None)
            if existing_path == file_key and existing_sid != session_id:
                LOGGER.info(
                    "[deps] lazy-register: aliasing %s -> existing session %s "
                    "for file_key=%s (preserves one-Session-per-file invariant)",
                    session_id,
                    existing_sid,
                    file_key,
                )
                sm.sessions[session_id] = existing_session
                return existing_session

        LOGGER.info(
            "[deps] lazy-registering session %s for file_key=%s "
            "(no prior session — creating fresh)",
            session_id,
            file_key,
        )
        consumer = _HeadlessSessionConsumer(consumer_id=str(session_id))
        try:
            return sm.create_session(
                session_id=session_id,
                session_consumer=consumer,
                query_params={},
                file_key=file_key,
                auto_instantiate=True,
            )
        except Exception:
            LOGGER.exception(
                "[deps] lazy-register: create_session failed for %s",
                session_id,
            )
            return None

    def require_query_params(self, param: str) -> str:
        """Get a query parameter or raise an error."""
        value = self.request.query_params[param]
        if not value:
            raise ValueError(f"Missing query parameter: {param}")
        return value

    def query_params(self, param: str) -> str | None:
        """Get a query parameter."""
        if param not in self.request.query_params:
            return None
        return self.request.query_params[param]

    # Config manager for the marimo file that we are running.
    # This could have custom config in the script metadata.
    @property
    def app_config_manager(self) -> MarimoConfigManager:
        session = self.require_current_session()
        return session.config_manager

    # We may have not created a session yet, but we know the file where we will
    # create one.
    # Use this file to override the config manager.
    def config_manager_at_file(self, path: str) -> MarimoConfigManager:
        return super().config_manager.with_overrides(
            ScriptConfigManager(path).get_config()
        )
