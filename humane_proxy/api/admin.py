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
  GET  /admin/analytics/top-triggers — aggregate trigger frequencies
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
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from humane_proxy.escalation.local_db import _get_db_path

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


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(_get_db_path(), check_same_thread=False)


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

    conn = _get_conn()
    try:
        rows = conn.execute(
            f"SELECT * FROM escalations {where} ORDER BY {sort_col} {sort_dir} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        total = conn.execute(
            f"SELECT COUNT(*) FROM escalations {where}", params
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [_row_to_dict(r) for r in rows],
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
    clauses: list[str] = []
    params: list[Any] = []

    if category:
        clauses.append("category = ?")
        params.append(category)
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    conn = _get_conn()
    try:
        rows = conn.execute(
            f"SELECT * FROM escalations {where} ORDER BY timestamp DESC",
            params,
        ).fetchall()
    finally:
        conn.close()

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
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM escalations WHERE id = ?", (escalation_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found.")
    return _row_to_dict(row)


@router.get("/sessions/{session_id}/risk")
def get_session_risk(
    session_id: str,
    _: str = Depends(_require_admin),
) -> dict:
    """Return escalation history + current trajectory for a session."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM escalations WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()

    from humane_proxy.risk.trajectory import snapshot

    trajectory = snapshot(session_id)
    history = [_row_to_dict(r) for r in rows]

    peak_risk_score = None
    peak_risk_timestamp = None
    category_transitions = []
    
    if history:
        # Initialize with the first item to avoid iterating against it
        first_item = history[0]
        peak_risk_score = first_item.get("risk_score")
        peak_risk_timestamp = first_item.get("timestamp")
        last_category = first_item.get("category")
        
        for r in history[1:]:
            score = r.get("risk_score")
            if score is not None:
                if peak_risk_score is None or score > peak_risk_score:
                    peak_risk_score = score
                    peak_risk_timestamp = r.get("timestamp")

            cat = r.get("category")
            if cat and last_category and cat != last_category:
                category_transitions.append({
                    "from": last_category,
                    "to": cat,
                    "timestamp": r.get("timestamp")
                })
                last_category = cat

    return {
        "session_id": session_id,
        "escalation_count": len(rows),
        "peak_risk_score": peak_risk_score,
        "peak_risk_timestamp": peak_risk_timestamp,
        "category_transitions": category_transitions,
        "history": history,
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
    conn = _get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
        by_category_rows = conn.execute(
            "SELECT category, COUNT(*) FROM escalations GROUP BY category"
        ).fetchall()
        by_day = conn.execute(
            """SELECT date(timestamp, 'unixepoch') as day, COUNT(*)
               FROM escalations GROUP BY day ORDER BY day DESC LIMIT 30"""
        ).fetchall()
        avg_score = conn.execute(
            "SELECT AVG(risk_score) FROM escalations"
        ).fetchone()[0]

        # Enhanced: top sessions (most flagged).
        top_sessions = conn.execute(
            """SELECT session_id, COUNT(*) as cnt, AVG(risk_score) as avg_score
               FROM escalations GROUP BY session_id ORDER BY cnt DESC LIMIT 10"""
        ).fetchall()

        # Enhanced: stage distribution.
        by_stage = conn.execute(
            "SELECT stage_reached, COUNT(*) FROM escalations GROUP BY stage_reached"
        ).fetchall()

        # Enhanced: hourly breakdown (last 24h).
        cutoff_24h = (datetime.now(timezone.utc).timestamp()) - 86400
        hourly = conn.execute(
            """SELECT strftime('%H', timestamp, 'unixepoch') as hour, COUNT(*)
               FROM escalations WHERE timestamp >= ?
               GROUP BY hour ORDER BY hour""",
            (cutoff_24h,),
        ).fetchall()
        
        now_dt = datetime.now(timezone.utc)
        current_week_start_dt = now_dt.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=now_dt.weekday())
        
        current_week_start = current_week_start_dt.timestamp()
        previous_week_start = (current_week_start_dt - timedelta(days=7)).timestamp()
        now_ts = now_dt.timestamp()
        
        current_week_count = conn.execute(
            "SELECT COUNT(*) FROM escalations WHERE timestamp >= ? AND timestamp <= ?",
            (current_week_start, now_ts)
        ).fetchone()[0]
        
        previous_week_count = conn.execute(
            "SELECT COUNT(*) FROM escalations WHERE timestamp >= ? AND timestamp < ?",
            (previous_week_start, current_week_start)
        ).fetchone()[0]

    finally:
        conn.close()
        
    by_category_dict = dict(by_category_rows)
    
    # New: category_percentages
    category_percentages = {}
    if total > 0:
        for cat, count in by_category_dict.items():
            category_percentages[cat] = round((count / total) * 100, 1)

    return {
        "total_escalations": total,
        "by_category": by_category_dict,
        "category_percentages": category_percentages,
        "period_comparison": {
            "current_week": current_week_count,
            "previous_week": previous_week_count,
        },
        "by_day": dict(by_day),
        "average_risk_score": round(avg_score or 0.0, 3),
        "top_sessions": [
            {"session_id": s, "count": c, "avg_score": round(a or 0, 3)}
            for s, c, a in top_sessions
        ],
        "by_stage": dict(by_stage),
        "hourly_last_24h": dict(hourly),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Analytics Endpoints
# ---------------------------------------------------------------------------

@router.get("/analytics/top-triggers")
def get_top_triggers(
    limit: int = Query(10, ge=1, le=100),
    category: str | None = Query(None, description="Filter by category"),
    _: str = Depends(_require_admin)
) -> dict:
    """Aggregate all escalation trigger keywords and rank them by frequency."""
    conn = _get_conn()
    clauses: list[str] = []
    params: list[Any] = []

    if category:
        clauses.append("category = ?")
        params.append(category)
        
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    try:
        rows = conn.execute(
            f"SELECT triggers FROM escalations {where}", params
        ).fetchall()
    finally:
        conn.close()

    trigger_counter: Counter[str] = Counter()
    for (triggers_raw,) in rows:
        if not triggers_raw:
            continue
        try:
            parsed = json.loads(triggers_raw)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str):
                        trigger_counter[item.strip().lower()] += 1
        except (json.JSONDecodeError, TypeError):
            continue
            
    top_triggers = [
        {"trigger": trigger, "count": count} 
        for trigger, count in trigger_counter.most_common(limit)
    ]

    return {"top_triggers": top_triggers}


@router.delete("/sessions/{session_id}", status_code=204, response_class=Response)
def delete_session_data(
    session_id: str,
    _: str = Depends(_require_admin),
) -> Response:
    """Delete all escalation records for a session (privacy right to erasure)."""
    conn = _get_conn()
    try:
        with conn:
            deleted = conn.execute(
                "DELETE FROM escalations WHERE session_id = ?", (session_id,)
            ).rowcount
    finally:
        conn.close()

    logger.info("Deleted %d records for session %s (admin request)", deleted, session_id)
    return Response(status_code=204)