"""PostgreSQL storage backend for HumaneProxy escalation data.

Requires: pip install humane-proxy[postgres]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from humane_proxy.storage.base import EscalationStore

logger = logging.getLogger("humane_proxy.storage.postgres")

try:
    import psycopg
    from psycopg.rows import dict_row
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False
    psycopg = None  # type: ignore[assignment]


class PostgresStore(EscalationStore):
    """PostgreSQL-backed escalation storage.

    Uses ``psycopg`` (v3) with synchronous connections.

    Parameters
    ----------
    config:
        Full application config dict.
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
        if not _PG_AVAILABLE:
            raise RuntimeError(
                "PostgreSQL storage requires the 'psycopg' package. "
                "Install with: pip install humane-proxy[postgres]"
            )
        pg_cfg = config.get("storage", {}).get("postgres", {})
        self._dsn = pg_cfg.get("dsn", "postgresql://localhost:5432/humane_proxy")
        self._rate_limit_max = rate_limit_max
        self._rate_limit_window = timedelta(hours=rate_limit_window_hours)

    def _conn(self):
        return psycopg.connect(self._dsn, row_factory=dict_row)

    def init(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS escalations (
                    id             SERIAL PRIMARY KEY,
                    session_id     TEXT    NOT NULL,
                    category       TEXT    NOT NULL DEFAULT 'unknown',
                    risk_score     DOUBLE PRECISION NOT NULL,
                    triggers       TEXT    NOT NULL,
                    timestamp      DOUBLE PRECISION NOT NULL,
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
            conn.commit()
        logger.info("PostgreSQL store initialised: %s", self._dsn.split("@")[-1] if "@" in self._dsn else "(local)")

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
        ts = datetime.now(timezone.utc).timestamp()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO escalations
                    (session_id, category, risk_score, triggers, timestamp,
                     message_hash, stage_reached, reasoning)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    category,
                    risk_score,
                    json.dumps(triggers or []),
                    ts,
                    message_hash,
                    stage_reached,
                    reasoning,
                ),
            )
            conn.commit()

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
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM escalations {where} ORDER BY timestamp DESC LIMIT %s OFFSET %s",
                params + [limit, offset],
            ).fetchall()
        return [self._parse(r) for r in rows]

    def count(
        self,
        *,
        category: str | None = None,
        session_id: str | None = None,
    ) -> int:
        clauses, params = self._build_where(category, session_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM escalations {where}", params
            ).fetchone()
        return row["cnt"] if row else 0

    def get_by_id(self, escalation_id: int) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM escalations WHERE id = %s", (escalation_id,)
            ).fetchone()
        return self._parse(row) if row else None

    def delete_session(self, session_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM escalations WHERE session_id = %s", (session_id,)
            )
            conn.commit()
            return cur.rowcount

    def stats(self) -> dict[str, Any]:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM escalations").fetchone()["cnt"]
            by_category = {
                r["category"]: r["cnt"]
                for r in conn.execute(
                    "SELECT category, COUNT(*) as cnt FROM escalations GROUP BY category"
                ).fetchall()
            }
            avg = conn.execute(
                "SELECT AVG(risk_score) as avg FROM escalations"
            ).fetchone()["avg"]
        return {
            "total_escalations": total,
            "by_category": by_category,
            "average_risk_score": round(avg or 0.0, 3),
        }

    def check_rate_limit(self, session_id: str) -> bool:
        cutoff = (datetime.now(timezone.utc) - self._rate_limit_window).timestamp()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM escalations WHERE session_id = %s AND timestamp >= %s",
                (session_id, cutoff),
            ).fetchone()
        return (row["cnt"] if row else 0) < self._rate_limit_max

    @staticmethod
    def _build_where(
        category: str | None, session_id: str | None,
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if category:
            clauses.append("category = %s")
            params.append(category)
        if session_id:
            clauses.append("session_id = %s")
            params.append(session_id)
        return clauses, params

    @staticmethod
    def _parse(row: dict[str, Any]) -> dict[str, Any]:
        rec = dict(row)
        try:
            rec["triggers"] = json.loads(rec["triggers"])
        except Exception:
            pass
        return rec
