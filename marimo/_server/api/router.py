# Copyright 2026 Marimo. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING

from marimo._server.api.endpoints.ai import router as ai_router
from marimo._server.api.endpoints.assets import router as assets_router
from marimo._server.api.endpoints.cache import router as cache_router
from marimo._server.api.endpoints.config import router as config_router
from marimo._server.api.endpoints.datasources import (
    router as datasources_router,
)
from marimo._server.api.endpoints.document import router as document_router
from marimo._server.api.endpoints.documentation import (
    router as documentation_router,
)
from marimo._server.api.endpoints.editing import router as editing_router
from marimo._server.api.endpoints.execution import router as execution_router
from marimo._server.api.endpoints.export import router as export_router
from marimo._server.api.endpoints.file_explorer import (
    router as file_explorer_router,
)
from marimo._server.api.endpoints.files import router as files_router
from marimo._server.api.endpoints.health import router as health_router
from marimo._server.api.endpoints.home import router as home_router
from marimo._server.api.endpoints.login import router as login_router
from marimo._server.api.endpoints.lsp import router as lsp_router
from marimo._server.api.endpoints.packages import router as packages_router
from marimo._server.api.endpoints.secrets import router as secrets_router
from marimo._server.api.endpoints.sql import router as sql_router
from marimo._server.api.endpoints.storage import router as storage_router
from marimo._server.api.endpoints.terminal import router as terminal_router
from marimo._server.api.endpoints.ws_endpoint import router as ws_router
from marimo._server.router import APIRouter

# --- DATAGEN-FORK: chat-history persistence routes (datagendev/marimo) ---
# Self-contained additive endpoint mounted last to minimize upstream merge
# conflicts. Removing this import + the include_router call below is a
# clean revert to upstream-vanilla router behavior.
from marimo._server.api.endpoints.chat_history import (
    router as chat_history_router,
)

# --- DATAGEN-FORK: headless session-open route (datagendev/marimo) ---
# Lets HTTP-only drivers (marimo-pair, datagen-marimo) bootstrap a
# kernel session for a file path without a WebSocket consumer. Mounted
# at /api/sessions/open. Removing this import + the include_router
# call below is a clean revert.
from marimo._server.api.endpoints.sessions_open import (
    router as sessions_open_router,
)

if TYPE_CHECKING:
    from starlette.routing import BaseRoute


# Define the app routes
def build_routes(base_url: str = "") -> list[BaseRoute]:
    app_router = APIRouter(prefix=base_url)
    app_router.include_router(
        execution_router, prefix="/api/kernel", name="execution"
    )
    app_router.include_router(
        config_router, prefix="/api/kernel", name="config"
    )
    app_router.include_router(
        editing_router, prefix="/api/kernel", name="editing"
    )
    app_router.include_router(files_router, prefix="/api/kernel", name="files")
    app_router.include_router(
        file_explorer_router, prefix="/api/files", name="file_explorer"
    )
    app_router.include_router(
        secrets_router, prefix="/api/secrets", name="secrets"
    )
    app_router.include_router(cache_router, prefix="/api/cache", name="cache")
    app_router.include_router(
        documentation_router, prefix="/api/documentation", name="documentation"
    )
    app_router.include_router(
        document_router, prefix="/api/document", name="document"
    )
    app_router.include_router(
        datasources_router, prefix="/api/datasources", name="datasources"
    )
    app_router.include_router(sql_router, prefix="/api/sql", name="sql")
    app_router.include_router(
        storage_router, prefix="/api/storage", name="storage"
    )
    app_router.include_router(ai_router, prefix="/api/ai", name="ai")
    app_router.include_router(home_router, prefix="/api/home", name="home")
    app_router.include_router(login_router, prefix="/auth", name="auth")
    app_router.include_router(
        export_router, prefix="/api/export", name="export"
    )
    app_router.include_router(
        terminal_router, prefix="/terminal", name="terminal"
    )
    app_router.include_router(
        packages_router, prefix="/api/packages", name="packages"
    )
    app_router.include_router(lsp_router, prefix="/api/lsp", name="lsp")

    # --- DATAGEN-FORK: chat-history persistence ---
    # MUST be registered BEFORE health_router, ws_router, and
    # assets_router — all three of those are mounted with empty
    # prefix (""), which Starlette treats as a wildcard prefix that
    # matches every path. Once Starlette resolves a Mount, it does NOT
    # fall through to the next Mount even if the inner router 404s;
    # mount-match wins, inner-router 404s, request fails.
    #
    # Every other /api/* router in this file is registered ABOVE the
    # empty-prefix routers for the same reason — we slot in here at
    # the end of the "real prefix" block.
    app_router.include_router(
        chat_history_router,
        prefix="/api/chat_history",
        name="chat_history",
    )

    # --- DATAGEN-FORK: headless session-open ---
    # Same ordering rule as chat_history above.
    app_router.include_router(
        sessions_open_router,
        prefix="/api/sessions",
        name="sessions_open",
    )

    app_router.include_router(health_router, name="health")
    app_router.include_router(ws_router, name="ws")

    # assets is the catch-all SPA static handler — keep last so any
    # request that didn't match a real API route falls through to the
    # frontend bundle.
    app_router.include_router(assets_router, name="assets")

    return app_router.routes
