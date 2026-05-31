"""HumaneProxy REST Admin API.

Mounted at ``/admin`` on the main FastAPI app.

Authentication: Bearer token from ``HUMANE_PROXY_ADMIN_KEY`` env var.
If not set, admin API is disabled (all requests → 403).

Endpoints:
  GET  /admin/health               — health check (no auth)
  GET  /admin/config               — active configuration (sanitised)
  GET  /admin/escalations          — paginated, filterable list
  GET  /admin/escalations/export   — CSV export
  GET  /admin/escalations/{id}     — single record
  GET  /admin/sessions/{id}/risk   — per-session trajectory
  GET  /admin/stats                — aggregate counts
  DELETE /admin/sessions/{id}      — delete session data (privacy)
"""

from __future__ import annotations

import csv
import hmac
import io
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from humane_proxy.escalation.local_db import _get_db_path
from humane_proxy.storage.factory import get_store

logger = logging.getLogger("humane_proxy.api.admin")

router = APIRouter(prefix="/admin", tags=["admin"])

_security = HTTPBearer(auto_error=False)

_start_time = time.monotonic()


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def _require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> str:
    """Validate admin Bearer token."""
    admin_key = os.environ.get("HUMANE_PROXY_ADMIN_KEY", "")
    if not admin_key:
        raise HTTPException(
            status_code=403,
            detail=(
                "Admin API is disabled. Set HUMANE_PROXY_ADMIN_KEY "
                "environment variable to enable it."
            ),
        )
    if credentials is None or not hmac.compare_digest(credentials.credentials, admin_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLS = ["id", "session_id", "category", "risk_score", "triggers",
         "timestamp", "message_hash", "stage_reached", "reasoning"]


def _row_to_dict(row: tuple) -> dict[str, Any]:
    rec: dict[str, Any] = dict(zip(_COLS, row))
    try:
        rec["triggers"] = json.loads(rec["triggers"])
    except Exception:
        pass
    return rec


def _run_sql_query(store: Any, query_str: str, params: list) -> list[dict]:
    from humane_proxy.storage.sqlite import SQLiteStore
    from humane_proxy.storage.postgres import PostgresStore
    
    if isinstance(store, SQLiteStore):
        import sqlite3
        conn = sqlite3.connect(store._db_path, check_same_thread=False)
        try:
            rows = conn.execute(query_str, params).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()
            
    elif isinstance(store, PostgresStore):
        import psycopg
        from psycopg.rows import dict_row
        pg_query = query_str.replace("?", "%s")
        with psycopg.connect(store._dsn, row_factory=dict_row) as conn:
            rows = conn.execute(pg_query, params).fetchall()
            return [store._parse(r) for r in rows]
            
    return []


def _run_sql_count(store: Any, query_str: str, params: list) -> int:
    from humane_proxy.storage.sqlite import SQLiteStore
    from humane_proxy.storage.postgres import PostgresStore
    
    if isinstance(store, SQLiteStore):
        import sqlite3
        conn = sqlite3.connect(store._db_path, check_same_thread=False)
        try:
            row = conn.execute(query_str, params).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
            
    elif isinstance(store, PostgresStore):
        import psycopg
        from psycopg.rows import dict_row
        pg_query = query_str.replace("?", "%s")
        with psycopg.connect(store._dsn, row_factory=dict_row) as conn:
            row = conn.execute(pg_query, params).fetchone()
            if row:
                return list(row.values())[0]
            return 0
            
    return 0


def _get_enhanced_sql_stats(store: Any) -> dict[str, Any]:
    from humane_proxy.storage.sqlite import SQLiteStore
    from humane_proxy.storage.postgres import PostgresStore
    
    if isinstance(store, SQLiteStore):
        import sqlite3
        conn = sqlite3.connect(store._db_path, check_same_thread=False)
        try:
            by_day = conn.execute(
                """SELECT date(timestamp, 'unixepoch') as day, COUNT(*)
                   FROM escalations GROUP BY day ORDER BY day DESC LIMIT 30"""
            ).fetchall()
            top_sessions = conn.execute(
                """SELECT session_id, COUNT(*) as cnt, AVG(risk_score) as avg_score
                   FROM escalations GROUP BY session_id ORDER BY cnt DESC LIMIT 10"""
            ).fetchall()
            by_stage = conn.execute(
                "SELECT stage_reached, COUNT(*) FROM escalations GROUP BY stage_reached"
            ).fetchall()
            cutoff_24h = datetime.now(timezone.utc).timestamp() - 86400
            hourly = conn.execute(
                """SELECT strftime('%H', timestamp, 'unixepoch') as hour, COUNT(*)
                   FROM escalations WHERE timestamp >= ?
                   GROUP BY hour ORDER BY hour""",
                (cutoff_24h,),
            ).fetchall()
        finally:
            conn.close()
        return {
            "by_day": dict(by_day),
            "top_sessions": [
                {"session_id": s, "count": c, "avg_score": round(a or 0, 3)}
                for s, c, a in top_sessions
            ],
            "by_stage": dict(by_stage),
            "hourly_last_24h": dict(hourly),
        }
        
    elif isinstance(store, PostgresStore):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError:
            return {}
        
        with psycopg.connect(store._dsn, row_factory=dict_row) as conn:
            by_day = conn.execute(
                """SELECT TO_CHAR(TO_TIMESTAMP(timestamp), 'YYYY-MM-DD') as day, COUNT(*) as cnt
                   FROM escalations GROUP BY day ORDER BY day DESC LIMIT 30"""
            ).fetchall()
            top_sessions = conn.execute(
                """SELECT session_id, COUNT(*) as cnt, AVG(risk_score) as avg_score
                   FROM escalations GROUP BY session_id ORDER BY cnt DESC LIMIT 10"""
            ).fetchall()
            by_stage = conn.execute(
                "SELECT stage_reached, COUNT(*) as cnt FROM escalations GROUP BY stage_reached"
            ).fetchall()
            cutoff_24h = datetime.now(timezone.utc).timestamp() - 86400
            hourly = conn.execute(
                """SELECT TO_CHAR(TO_TIMESTAMP(timestamp), 'HH24') as hour, COUNT(*) as cnt
                   FROM escalations WHERE timestamp >= %s
                   GROUP BY hour ORDER BY hour""",
                (cutoff_24h,),
            ).fetchall()
            
        return {
            "by_day": {r["day"]: r["cnt"] for r in by_day},
            "top_sessions": [
                {"session_id": r["session_id"], "count": r["cnt"], "avg_score": round(r["avg_score"] or 0, 3)}
                for r in top_sessions
            ],
            "by_stage": {r["stage_reached"]: r["cnt"] for r in by_stage},
            "hourly_last_24h": {r["hour"]: r["cnt"] for r in hourly},
        }
        
    return {}


# ---------------------------------------------------------------------------
# Health check (no auth required)
# ---------------------------------------------------------------------------

@router.get("/health")
def health_check() -> dict:
    """Health check — uptime, version, active stages, backend."""
    from humane_proxy import __version__
    from humane_proxy.config import get_config

    config = get_config()
    return {
        "status": "healthy",
        "version": __version__,
        "uptime_seconds": round(time.monotonic() - _start_time, 1),
        "enabled_stages": config.get("pipeline", {}).get("enabled_stages", [1]),
        "storage_backend": config.get("storage", {}).get("backend", "sqlite"),
    }


# ---------------------------------------------------------------------------
# Config view (secrets redacted)
# ---------------------------------------------------------------------------

@router.get("/config")
def get_active_config(_: str = Depends(_require_admin)) -> dict:
    """Return the active merged config with secrets redacted."""
    import copy
    from humane_proxy.config import get_config

    config = copy.deepcopy(get_config())

    # Redact sensitive values.
    _REDACT_PATHS = [
        ("admin", "api_key"),
        ("escalation", "webhooks", "slack_url"),
        ("escalation", "webhooks", "discord_url"),
        ("escalation", "webhooks", "pagerduty_routing_key"),
        ("escalation", "webhooks", "teams_url"),
        ("storage", "redis", "url"),
        ("storage", "postgres", "dsn"),
    ]
    for path in _REDACT_PATHS:
        node = config
        for part in path[:-1]:
            node = node.get(part, {})
            if not isinstance(node, dict):
                break
        else:
            key = path[-1]
            if key in node and node[key]:
                node[key] = "***REDACTED***"

    # Redact email password.
    email_cfg = config.get("escalation", {}).get("webhooks", {}).get("email", {})
    if isinstance(email_cfg, dict) and email_cfg.get("password"):
        email_cfg["password"] = "***REDACTED***"

    return config


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/escalations")
def list_escalations(
    category: str | None = Query(None, description="Filter by category"),
    session_id: str | None = Query(None, description="Filter by session ID"),
    date_from: str | None = Query(None, description="Start date (ISO format, e.g. 2026-01-01)"),
    date_to: str | None = Query(None, description="End date (ISO format, e.g. 2026-12-31)"),
    sort_by: str = Query("timestamp", description="Sort field: timestamp, risk_score, category"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: str = Depends(_require_admin),
) -> dict:
    """List escalation records, filterable and paginated."""
    from humane_proxy.storage.factory import get_store
    from humane_proxy.storage.redis import RedisStore

    store = get_store()

    if isinstance(store, RedisStore):
        raw_items = store.query(category=category, session_id=session_id, limit=1000)
        filtered = []
        for item in raw_items:
            ts = item.get("timestamp", 0.0)
            if date_from:
                try:
                    dt = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
                    if ts < dt.timestamp():
                        continue
                except ValueError:
                    raise HTTPException(400, f"Invalid date_from format: {date_from}")
            if date_to:
                try:
                    dt = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc)
                    if ts > dt.timestamp():
                        continue
                except ValueError:
                    raise HTTPException(400, f"Invalid date_to format: {date_to}")
            filtered.append(item)

        allowed_sort = {"timestamp", "risk_score", "category", "session_id", "stage_reached"}
        sort_col = sort_by if sort_by in allowed_sort else "timestamp"
        reverse = sort_order.lower() == "desc"
        filtered.sort(key=lambda x: x.get(sort_col, 0) if x.get(sort_col) is not None else 0, reverse=reverse)

        total = len(filtered)
        items = filtered[offset : offset + limit]

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": items,
        }

    clauses: list[str] = []
    params: list[Any] = []

    if category:
        clauses.append("category = ?")
        params.append(category)
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    if date_from:
        try:
            dt = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
            clauses.append("timestamp >= ?")
            params.append(dt.timestamp())
        except ValueError:
            raise HTTPException(400, f"Invalid date_from format: {date_from}")
    if date_to:
        try:
            dt = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc)
            clauses.append("timestamp <= ?")
            params.append(dt.timestamp())
        except ValueError:
            raise HTTPException(400, f"Invalid date_to format: {date_to}")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    allowed_sort = {"timestamp", "risk_score", "category", "session_id", "stage_reached"}
    sort_col = sort_by if sort_by in allowed_sort else "timestamp"
    sort_dir = "ASC" if sort_order.lower() == "asc" else "DESC"

    items = _run_sql_query(
        store,
        f"SELECT * FROM escalations {where} ORDER BY {sort_col} {sort_dir} LIMIT ? OFFSET ?",
        params + [limit, offset],
    )
    total = _run_sql_count(
        store,
        f"SELECT COUNT(*) FROM escalations {where}",
        params,
    )

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
    }


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

@router.get("/escalations/export")
def export_escalations(
    category: str | None = Query(None),
    session_id: str | None = Query(None),
    _: str = Depends(_require_admin),
) -> StreamingResponse:
    """Export all matching escalations as CSV."""
    from humane_proxy.storage.factory import get_store
    from humane_proxy.storage.redis import RedisStore

    store = get_store()

    if isinstance(store, RedisStore):
        raw_items = store.query(category=category, session_id=session_id, limit=1000)
        rows = []
        for item in raw_items:
            row = (
                item.get("id"),
                item.get("session_id"),
                item.get("category"),
                item.get("risk_score"),
                json.dumps(item.get("triggers") or []),
                item.get("timestamp"),
                item.get("message_hash"),
                item.get("stage_reached"),
                item.get("reasoning"),
            )
            rows.append(row)
    else:
        clauses: list[str] = []
        params: list[Any] = []

        if category:
            clauses.append("category = ?")
            params.append(category)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        from humane_proxy.storage.sqlite import SQLiteStore
        from humane_proxy.storage.postgres import PostgresStore

        if isinstance(store, SQLiteStore):
            import sqlite3
            conn = sqlite3.connect(store._db_path, check_same_thread=False)
            try:
                rows = conn.execute(
                    f"SELECT * FROM escalations {where} ORDER BY timestamp DESC",
                    params,
                ).fetchall()
            finally:
                conn.close()
        elif isinstance(store, PostgresStore):
            import psycopg
            pg_query = f"SELECT * FROM escalations {where} ORDER BY timestamp DESC".replace("?", "%s")
            with psycopg.connect(store._dsn) as conn:
                rows = conn.execute(pg_query, params).fetchall()
        else:
            rows = []

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(_COLS)
    for row in rows:
        writer.writerow(row)

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=escalations.csv"},
    )


@router.get("/escalations/{escalation_id}")
def get_escalation(
    escalation_id: int,
    _: str = Depends(_require_admin),
) -> dict:
    """Get a single escalation record by ID."""
    from humane_proxy.storage.factory import get_store

    store = get_store()
    row = store.get_by_id(escalation_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found.")
    return row


@router.get("/sessions/{session_id}/risk")
def get_session_risk(
    session_id: str,
    _: str = Depends(_require_admin),
) -> dict:
    """Return escalation history + current trajectory for a session."""
    from humane_proxy.storage.factory import get_store

    store = get_store()
    rows = store.query(session_id=session_id, limit=1000)
    rows.sort(key=lambda x: x.get("timestamp", 0.0))

    from humane_proxy.risk.trajectory import snapshot

    trajectory = snapshot(session_id)

    return {
        "session_id": session_id,
        "escalation_count": len(rows),
        "history": rows,
        "trajectory": {
            "spike_detected": trajectory.spike_detected,
            "trend": trajectory.trend,
            "window_scores": trajectory.window_scores,
            "category_counts": trajectory.category_counts,
            "message_count": trajectory.message_count,
        },
    }


@router.get("/stats")
def get_stats(_: str = Depends(_require_admin)) -> dict:
    """Return aggregate safety statistics with enhanced breakdowns."""
    from humane_proxy.storage.factory import get_store

    store = get_store()
    base_stats = store.stats()

    enhanced = {
        "by_day": {},
        "top_sessions": [],
        "by_stage": {},
        "hourly_last_24h": {},
    }

    sql_stats = _get_enhanced_sql_stats(store)
    enhanced.update(sql_stats)

    return {
        "total_escalations": base_stats.get("total_escalations", 0),
        "by_category": base_stats.get("by_category", {}),
        "by_day": enhanced["by_day"],
        "average_risk_score": base_stats.get("average_risk_score", 0.0),
        "top_sessions": enhanced["top_sessions"],
        "by_stage": enhanced["by_stage"],
        "hourly_last_24h": enhanced["hourly_last_24h"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.delete("/sessions/{session_id}", status_code=204, response_class=Response)
def delete_session_data(
    session_id: str,
    _: str = Depends(_require_admin),
) -> Response:
    """Delete all escalation records for a session (privacy right to erasure)."""
    from humane_proxy.storage.factory import get_store

    store = get_store()
    deleted = store.delete_session(session_id)

    logger.info("Deleted %d records for session %s (admin request)", deleted, session_id)
    return Response(status_code=204)
