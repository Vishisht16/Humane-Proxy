"""SQLite-backed escalation logging and per-session rate limiting."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("humane_proxy.escalation")

# ---------------------------------------------------------------------------
# Database location — overridable via environment variable.
# ---------------------------------------------------------------------------
_DEFAULT_DB_PATH: str = str(Path(__file__).resolve().parent / "escalations.db")


def _get_db_path() -> str:
    """Return the DB path, checking env var at runtime (not import time)."""
    return os.getenv("HUMANE_PROXY_DB_PATH", _DEFAULT_DB_PATH)


# Rate-limit: read from config, with sensible fallbacks.
def _get_rate_limit_max() -> int:
    from humane_proxy.config import get_config
    return get_config().get("escalation", {}).get("rate_limit_max", 3)


def _get_rate_limit_window() -> timedelta:
    from humane_proxy.config import get_config
    hours = get_config().get("escalation", {}).get("rate_limit_window_hours", 1)
    return timedelta(hours=hours)


def _get_conn() -> sqlite3.Connection:
    """Return a connection with WAL journal mode for better concurrency."""
    conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Schema bootstrap + migration
# ---------------------------------------------------------------------------

def _migrate_add_column(conn: sqlite3.Connection, name: str, definition: str) -> None:
    """Safely add a column if it doesn't exist (no-op on duplicate)."""
    try:
        conn.execute(f"ALTER TABLE escalations ADD COLUMN {name} {definition}")
    except sqlite3.OperationalError:
        pass  # Column already exists.


def init_db() -> None:
    """Create the ``escalations`` table if it does not already exist.

    Safe to call multiple times (uses ``IF NOT EXISTS``).
    Timestamps are stored as REAL (Unix epoch) for fast, unambiguous
    comparisons instead of ISO-8601 TEXT.
    """
    conn = _get_conn()
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS escalations (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id     TEXT    NOT NULL,
                    category       TEXT    NOT NULL DEFAULT 'unknown',
                    risk_score     REAL    NOT NULL,
                    triggers       TEXT    NOT NULL,
                    timestamp      REAL    NOT NULL,
                    message_hash   TEXT,
                    stage_reached  INTEGER DEFAULT 1,
                    reasoning      TEXT
                )
                """
            )
            # Index for the rate-limit lookup (session + time range).
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_esc_session_ts
                ON escalations (session_id, timestamp)
                """
            )

            # Migration: add new columns if upgrading from Phase 1.
            _migrate_add_column(conn, "message_hash", "TEXT")
            _migrate_add_column(conn, "stage_reached", "INTEGER DEFAULT 1")
            _migrate_add_column(conn, "reasoning", "TEXT")
    finally:
        conn.close()

    logger.info("Escalation DB initialised at %s", _get_db_path())


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def check_rate_limit(session_id: str) -> bool:
    """Return ``True`` if the session is **within** its allowed quota.

    The session may have at most ``rate_limit_max`` recorded escalations
    inside the last ``rate_limit_window``.  If the quota is exhausted,
    this returns ``False`` (i.e. "rate-limited — do NOT escalate again").
    """
    cutoff = (datetime.now(timezone.utc) - _get_rate_limit_window()).timestamp()

    conn = _get_conn()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM escalations
            WHERE session_id = ? AND timestamp >= ?
            """,
            (session_id, cutoff),
        ).fetchone()
    finally:
        conn.close()

    count: int = row[0] if row else 0
    return count < _get_rate_limit_max()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_escalation(
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
    message_hash: str | None = None,
    stage_reached: int = 1,
    reasoning: str | None = None,
) -> None:
    """Persist an escalation event to SQLite.

    ``triggers`` is stored as a JSON-encoded string so it round-trips
    cleanly without schema gymnastics.
    """
    triggers = triggers or []

    conn = _get_conn()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO escalations
                    (session_id, category, risk_score, triggers, timestamp,
                     message_hash, stage_reached, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    category,
                    risk_score,
                    json.dumps(triggers),
                    datetime.now(timezone.utc).timestamp(),
                    message_hash,
                    stage_reached,
                    reasoning,
                ),
            )
    finally:
        conn.close()
