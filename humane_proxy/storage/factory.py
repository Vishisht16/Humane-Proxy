"""Storage backend factory — returns a cached singleton store instance."""

from __future__ import annotations

import logging
import threading
from typing import Any

from humane_proxy.storage.base import EscalationStore

logger = logging.getLogger("humane_proxy.storage.factory")

_store: EscalationStore | None = None
_lock = threading.Lock()


def get_store(config: dict | None = None) -> EscalationStore:
    """Return an :class:`EscalationStore` instance based on configuration.

    The store is created once and cached for the lifetime of the process.
    Passing *config* is only necessary on the first call; subsequent calls
    ignore it and return the cached instance.

    Parameters
    ----------
    config:
        Full application config dict.  If ``None``, loads via
        :func:`humane_proxy.config.get_config`.
    """
    global _store
    if _store is not None:
        return _store

    with _lock:
        if _store is not None:
            return _store

        if config is None:
            from humane_proxy.config import get_config
            config = get_config()

        _store = _create_store(config)
        return _store


def _create_store(config: dict) -> EscalationStore:
    """Instantiate the correct backend based on ``storage.backend``."""
    backend = config.get("storage", {}).get("backend", "sqlite")
    esc_cfg = config.get("escalation", {})
    rate_max = esc_cfg.get("rate_limit_max", 3)
    rate_hours = esc_cfg.get("rate_limit_window_hours", 1)

    if backend == "redis":
        from humane_proxy.storage.redis import RedisStore
        store = RedisStore(config, rate_limit_max=rate_max, rate_limit_window_hours=rate_hours)
        logger.info("Using Redis storage backend")

    elif backend == "postgres":
        from humane_proxy.storage.postgres import PostgresStore
        store = PostgresStore(config, rate_limit_max=rate_max, rate_limit_window_hours=rate_hours)
        logger.info("Using PostgreSQL storage backend")

    else:
        from humane_proxy.storage.sqlite import SQLiteStore
        store = SQLiteStore(config, rate_limit_max=rate_max, rate_limit_window_hours=rate_hours)
        logger.info("Using SQLite storage backend (default)")

    return store


def reset_store() -> None:
    """Reset the cached store (for testing)."""
    global _store
    with _lock:
        _store = None
