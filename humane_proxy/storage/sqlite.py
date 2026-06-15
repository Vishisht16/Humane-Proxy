"""SQLite storage backend — the zero-config default."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from humane_proxy.storage.base import EscalationStore

logger = logging.getLogger("humane_proxy.storage.sqlite")

_LEGACY_DB_PATH: str = str(Path(__file__).resolve().parent.parent / "escalation" / "escalations.db")

# ── statically generated SQL templates ──────────────────────────────
# All possible WHERE clause patterns are generated once at import time
# so CodeQL sees only hardcoded strings, never user-controlled values.

_WHERE_CLAUSES: dict[tuple[bool, bool, bool, bool], str] = {}
for cat in (False, True):
    for sid in (False, True):
        for df in (False, True):
            for dt in (False, True):
                parts = []
                if cat:
                    parts.append("category = ?")
                if sid:
                    parts.append("session_id = ?")
                if df:
                    parts.append("timestamp >= ?")
                if dt:
                    parts.append("timestamp <= ?")
                _WHERE_CLAUSES[(cat, sid, df, dt)] = (
                    "WHERE " + " AND ".join(parts) if parts else ""
                )

_SORT_SPECS = [
    ("timestamp",    "asc"),
    ("timestamp",    "desc"),
    ("risk_score",   "asc"),
    ("risk_score",   "desc"),
    ("category",     "asc"),
    ("category",     "desc"),
    ("session_id",   "asc"),
    ("session_id",   "desc"),
    ("stage_reached","asc"),
    ("stage_reached","desc"),
]

_QUERY_TEMPLATES: dict[tuple[tuple[bool, bool, bool, bool], str, str], str] = {}
for wk, ws in _WHERE_CLAUSES.items():
    for sc, sd in _SORT_SPECS:
        _QUERY_TEMPLATES[(wk, sc, sd)] = (
            f"SELECT * FROM escalations {ws}"
            f" ORDER BY {sc} {sd.upper()}"
            f" LIMIT ? OFFSET ?"
        )

_COUNT_TEMPLATES: dict[tuple[bool, bool, bool, bool], str] = {}
for wk, ws in _WHERE_CLAUSES.items():
    if ws:
        _COUNT_TEMPLATES[wk] = f"SELECT COUNT(*) FROM escalations {ws}"
    else:
        _COUNT_TEMPLATES[wk] = "SELECT COUNT(*) FROM escalations"


class SQLiteStore(EscalationStore):
    """SQLite-backed escalation storage.

    Parameters
    ----------
    config:
        The ``storage.sqlite`` config block (or full config).
    rate_limit_max:
        Max escalations per session per window.
    rate_limit_window_hours:
        Window duration in hours.
    """

    def __init__(
        self,
        config: dict,
        rate_limit_max: int = 3,
        rate_limit_window_hours: int = 1,
    ) -> None:
        sqlite_cfg = config.get("storage", {}).get("sqlite", {})
        env_path = os.getenv("HUMANE_PROXY_DB_PATH")
        if env_path:
            self._db_path = env_path
        elif sqlite_cfg.get("path"):
            self._db_path = sqlite_cfg["path"]
        else:
            self._db_path = _LEGACY_DB_PATH
        esc_cfg = config.get("escalation", {})
        self._rate_limit_max = esc_cfg.get("rate_limit_max", rate_limit_max)
        self._rate_limit_window = timedelta(
            hours=esc_cfg.get("rate_limit_window_hours", rate_limit_window_hours)
        )

    def _conn(self) -> sqlite3.Connection:
        """Open and return a new SQLite connection."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init(self) -> None:
        """Create the escalations table and indexes if they do not exist."""
        conn = self._conn()
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
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_esc_session_ts
                    ON escalations (session_id, timestamp)
                    """
                )
                for col, defn in [
                    ("message_hash", "TEXT"),
                    ("stage_reached", "INTEGER DEFAULT 1"),
                    ("reasoning", "TEXT"),
                ]:
                    try:
                        conn.execute(f"ALTER TABLE escalations ADD COLUMN {col} {defn}")
                    except sqlite3.OperationalError:
                        pass
        finally:
            conn.close()
        logger.info("SQLite store initialised at %s", self._db_path)

    def log(
        self,
        session_id: str,
        category: str,
        risk_score: float,
        triggers: list[str],
        message_hash: str | None = None,
        stage_reached: int = 1,
        reasoning: str | None = None,
    ) -> None:
        """Insert one escalation record."""
        conn = self._conn()
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
                        json.dumps(triggers or []),
                        datetime.now(timezone.utc).timestamp(),
                        message_hash,
                        stage_reached,
                        reasoning,
                    ),
                )
        finally:
            conn.close()

    def query(
        self,
        *,
        category: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        date_from: float | None = None,
        date_to: float | None = None,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
    ) -> list[dict[str, Any]]:
        """Return escalation records matching the filters."""
        params = self._build_params(category, session_id, date_from, date_to)
        where_key = (category is not None, session_id is not None, date_from is not None, date_to is not None)
        allowed_sort = {
            "timestamp": "timestamp",
            "risk_score": "risk_score",
            "category": "category",
            "session_id": "session_id",
            "stage_reached": "stage_reached",
        }
        sort_col = allowed_sort.get(sort_by, "timestamp")
        sort_dir = "asc" if sort_order.lower() == "asc" else "desc"
        sql = _QUERY_TEMPLATES[(where_key, sort_col, sort_dir)]
        conn = self._conn()
        try:
            rows = conn.execute(sql, params + [limit, offset]).fetchall()
        finally:
            conn.close()
        return [self._row_to_dict(r) for r in rows]

    def count(
        self,
        *,
        category: str | None = None,
        session_id: str | None = None,
        date_from: float | None = None,
        date_to: float | None = None,
    ) -> int:
        """Return the number of matching records."""
        params = self._build_params(category, session_id, date_from, date_to)
        where_key = (category is not None, session_id is not None, date_from is not None, date_to is not None)
        conn = self._conn()
        try:
            row = conn.execute(
                _COUNT_TEMPLATES[where_key], params
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else 0

    def get_by_id(self, escalation_id: int) -> dict[str, Any] | None:
        """Return a single escalation record by primary key, or None."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM escalations WHERE id = ?", (escalation_id,)
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_dict(row) if row else None

    def delete_session(self, session_id: str) -> int:
        """Delete all records for a session and return the count deleted."""
        conn = self._conn()
        try:
            with conn:
                return conn.execute(
                    "DELETE FROM escalations WHERE session_id = ?", (session_id,)
                ).rowcount
        finally:
            conn.close()

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics including advanced breakdowns."""
        cutoff_24h = datetime.now(timezone.utc).timestamp() - 86400
        conn = self._conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
            by_category = dict(conn.execute(
                "SELECT category, COUNT(*) FROM escalations GROUP BY category"
            ).fetchall())
            avg_score = conn.execute(
                "SELECT AVG(risk_score) FROM escalations"
            ).fetchone()[0]
            by_day = dict(conn.execute(
                """SELECT date(timestamp, 'unixepoch') as day, COUNT(*)
                   FROM escalations GROUP BY day ORDER BY day DESC LIMIT 30"""
            ).fetchall())
            top_sessions = [
                {"session_id": s, "count": c, "avg_score": round(a or 0, 3)}
                for s, c, a in conn.execute(
                    """SELECT session_id, COUNT(*) as cnt, AVG(risk_score) as avg_score
                       FROM escalations GROUP BY session_id
                       ORDER BY cnt DESC LIMIT 10"""
                ).fetchall()
            ]
            by_stage = dict(conn.execute(
                "SELECT stage_reached, COUNT(*) FROM escalations GROUP BY stage_reached"
            ).fetchall())
            hourly = dict(conn.execute(
                """SELECT strftime('%H', timestamp, 'unixepoch') as hour, COUNT(*)
                   FROM escalations WHERE timestamp >= ?
                   GROUP BY hour ORDER BY hour""",
                (cutoff_24h,),
            ).fetchall())
        finally:
            conn.close()
        return {
            "total_escalations": total,
            "by_category": by_category,
            "average_risk_score": round(avg_score or 0.0, 3),
            "by_day": by_day,
            "top_sessions": top_sessions,
            "by_stage": by_stage,
            "hourly_last_24h": hourly,
            "limited_stats": False,
        }

    def check_rate_limit(self, session_id: str) -> bool:
        """Return True if the session is within the rate limit."""
        cutoff = (datetime.now(timezone.utc) - self._rate_limit_window).timestamp()
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM escalations WHERE session_id = ? AND timestamp >= ?",
                (session_id, cutoff),
            ).fetchone()
        finally:
            conn.close()
        count = row[0] if row else 0
        return count < self._rate_limit_max

    # --- internal helpers ---

    @staticmethod
    def _build_params(
        category: str | None,
        session_id: str | None,
        date_from: float | None = None,
        date_to: float | None = None,
    ) -> list[Any]:
        """Build params list from filter arguments."""
        params: list[Any] = []
        if category:
            params.append(category)
        if session_id:
            params.append(session_id)
        if date_from is not None:
            params.append(date_from)
        if date_to is not None:
            params.append(date_to)
        return params

    _COLS = ["id", "session_id", "category", "risk_score", "triggers",
             "timestamp", "message_hash", "stage_reached", "reasoning"]

    @classmethod
    def _row_to_dict(cls, row: tuple) -> dict[str, Any]:
        """Convert a raw SQLite row tuple to a dict with parsed triggers."""
        rec: dict[str, Any] = dict(zip(cls._COLS, row))
        try:
            rec["triggers"] = json.loads(rec["triggers"])
        except Exception:
            pass
        return rec