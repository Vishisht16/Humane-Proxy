"""SQLite-backed escalation logging and per-session rate limiting.

This module now delegates to the swappable storage backend via
:func:`humane_proxy.storage.factory.get_store`.  The public API is
preserved for backward compatibility with existing code that imports
``init_db``, ``log_escalation``, ``check_rate_limit``, and ``_get_db_path``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("humane_proxy.escalation")

# ---------------------------------------------------------------------------
# Legacy DB path — still needed by admin.py for direct SQLite queries.
# ---------------------------------------------------------------------------
_DEFAULT_DB_PATH: str = str(Path(__file__).resolve().parent / "escalations.db")


def _get_db_path() -> str:
    """Return the DB path, checking env var at runtime (not import time)."""
    return os.getenv("HUMANE_PROXY_DB_PATH", _DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# Public API — delegates to the storage factory
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the backend storage (tables, indexes, etc.).

    Safe to call multiple times.
    """
    from humane_proxy.storage.factory import get_store
    store = get_store()
    store.init()
    logger.info("Escalation storage initialised (backend: %s)", type(store).__name__)


def check_rate_limit(session_id: str) -> bool:
    """Return ``True`` if the session is **within** its allowed quota."""
    from humane_proxy.storage.factory import get_store
    return get_store().check_rate_limit(session_id)


def log_escalation(
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
    message_hash: str | None = None,
    stage_reached: int = 1,
    reasoning: str | None = None,
) -> None:
    """Persist an escalation event to the configured backend."""
    from humane_proxy.storage.factory import get_store
    get_store().log(
        session_id=session_id,
        category=category,
        risk_score=risk_score,
        triggers=triggers,
        message_hash=message_hash,
        stage_reached=stage_reached,
        reasoning=reasoning,
    )
