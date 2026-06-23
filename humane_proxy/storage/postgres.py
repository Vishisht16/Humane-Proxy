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
                    parts.append("category = %s")
                if sid:
                    parts.append("session_id = %s")
                if df:
                    parts.append("timestamp >= %s")
                if dt:
                    parts.append("timestamp <= %s")
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
            f" LIMIT %s OFFSET %s"
        )

_COUNT_TEMPLATES: dict[tuple[bool, bool, bool, bool], str] = {}
for wk, ws in _WHERE_CLAUSES.items():
    _COUNT_TEMPLATES[wk] = (
        f"SELECT COUNT(*) as cnt FROM escalations {ws}" if ws
        else "SELECT COUNT(*) as cnt FROM escalations"
    )


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
        with self._conn() as conn:
            rows = conn.execute(sql, params + [limit, offset]).fetchall()
        return [self._parse(r) for r in rows]

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
        with self._conn() as conn:
            row = conn.execute(
                _COUNT_TEMPLATES[where_key], params
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
        """Return aggregate statistics including advanced breakdowns."""
        cutoff_24h = datetime.now(timezone.utc).timestamp() - 86400
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM escalations"
            ).fetchone()["cnt"]
            by_category = {
                r["category"]: r["cnt"]
                for r in conn.execute(
                    "SELECT category, COUNT(*) as cnt FROM escalations GROUP BY category"
                ).fetchall()
            }
            avg = conn.execute(
                "SELECT AVG(risk_score) as avg FROM escalations"
            ).fetchone()["avg"]
            by_day = {
                r["day"]: r["cnt"]
                for r in conn.execute(
                    """SELECT TO_CHAR(TO_TIMESTAMP(timestamp), 'YYYY-MM-DD') as day,
                              COUNT(*) as cnt
                       FROM escalations GROUP BY day ORDER BY day DESC LIMIT 30"""
                ).fetchall()
            }
            top_sessions = [
                {"session_id": r["session_id"], "count": r["cnt"],
                 "avg_score": round(r["avg_score"] or 0, 3)}
                for r in conn.execute(
                    """SELECT session_id, COUNT(*) as cnt, AVG(risk_score) as avg_score
                       FROM escalations GROUP BY session_id
                       ORDER BY cnt DESC LIMIT 10"""
                ).fetchall()
            ]
            by_stage = {
                r["stage_reached"]: r["cnt"]
                for r in conn.execute(
                    "SELECT stage_reached, COUNT(*) as cnt FROM escalations GROUP BY stage_reached"
                ).fetchall()
            }
            hourly = {
                r["hour"]: r["cnt"]
                for r in conn.execute(
                    """SELECT TO_CHAR(TO_TIMESTAMP(timestamp), 'HH24') as hour,
                              COUNT(*) as cnt
                       FROM escalations WHERE timestamp >= %s
                       GROUP BY hour ORDER BY hour""",
                    (cutoff_24h,),
                ).fetchall()
            }
        return {
            "total_escalations": total,
            "by_category": by_category,
            "average_risk_score": round(avg or 0.0, 3),
            "by_day": by_day,
            "top_sessions": top_sessions,
            "by_stage": by_stage,
            "hourly_last_24h": hourly,
            "limited_stats": False,
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
    def _build_params(
        category: str | None,
        session_id: str | None,
        date_from: float | None = None,
        date_to: float | None = None,
    ) -> list[Any]:
        """Build params list from filters."""
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

    @staticmethod
    def _parse(row: dict[str, Any]) -> dict[str, Any]:
        rec = dict(row)
        try:
            rec["triggers"] = json.loads(rec["triggers"])
        except Exception:
            pass
        return rec