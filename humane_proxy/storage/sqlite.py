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
        self._rate_limit_max = rate_limit_max
        self._rate_limit_window = timedelta(hours=rate_limit_window_hours)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init(self) -> None:
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
                # Migration: add columns if upgrading from older versions.
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
    ) -> list[dict[str, Any]]:
        clauses, params = self._build_where(category, session_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        conn = self._conn()
        try:
            rows = conn.execute(
                f"SELECT * FROM escalations {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_dict(r) for r in rows]

    def count(
        self,
        *,
        category: str | None = None,
        session_id: str | None = None,
    ) -> int:
        clauses, params = self._build_where(category, session_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        conn = self._conn()
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM escalations {where}", params
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else 0

    def get_by_id(self, escalation_id: int) -> dict[str, Any] | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM escalations WHERE id = ?", (escalation_id,)
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_dict(row) if row else None

    def delete_session(self, session_id: str) -> int:
        conn = self._conn()
        try:
            with conn:
                return conn.execute(
                    "DELETE FROM escalations WHERE session_id = ?", (session_id,)
                ).rowcount
        finally:
            conn.close()

    def stats(self) -> dict[str, Any]:
        conn = self._conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
            by_category = dict(conn.execute(
                "SELECT category, COUNT(*) FROM escalations GROUP BY category"
            ).fetchall())
            avg_score = conn.execute(
                "SELECT AVG(risk_score) FROM escalations"
            ).fetchone()[0]
        finally:
            conn.close()
        return {
            "total_escalations": total,
            "by_category": by_category,
            "average_risk_score": round(avg_score or 0.0, 3),
        }

    def check_rate_limit(self, session_id: str) -> bool:
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
    def _build_where(
        category: str | None, session_id: str | None,
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if category:
            clauses.append("category = ?")
            params.append(category)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        return clauses, params

    _COLS = ["id", "session_id", "category", "risk_score", "triggers",
             "timestamp", "message_hash", "stage_reached", "reasoning"]

    @classmethod
    def _row_to_dict(cls, row: tuple) -> dict[str, Any]:
        rec: dict[str, Any] = dict(zip(cls._COLS, row))
        try:
            rec["triggers"] = json.loads(rec["triggers"])
        except Exception:
            pass
        return rec
