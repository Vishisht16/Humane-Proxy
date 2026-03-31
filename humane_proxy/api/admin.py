"""HumaneProxy REST Admin API.

Mounted at ``/admin`` on the main FastAPI app.

Authentication: Bearer token from ``HUMANE_PROXY_ADMIN_KEY`` env var.
If not set, admin API is disabled (all requests → 403).

Endpoints:
  GET  /admin/escalations          — paginated, filterable list
  GET  /admin/escalations/{id}     — single record
  GET  /admin/sessions/{id}/risk   — per-session trajectory
  GET  /admin/stats                — aggregate counts
  DELETE /admin/sessions/{id}      — delete session data (privacy)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from humane_proxy.escalation.local_db import _get_db_path

logger = logging.getLogger("humane_proxy.api.admin")

router = APIRouter(prefix="/admin", tags=["admin"])

_security = HTTPBearer(auto_error=False)


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
    if credentials is None or credentials.credentials != admin_key:
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
# Routes
# ---------------------------------------------------------------------------

@router.get("/escalations")
def list_escalations(
    category: str | None = Query(None, description="Filter by category"),
    session_id: str | None = Query(None, description="Filter by session ID"),
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

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    conn = _get_conn()
    try:
        rows = conn.execute(
            f"SELECT * FROM escalations {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
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

    from humane_proxy.risk.trajectory import analyze

    # Build trajectory by replaying each escalation.
    trajectory = None
    for row in rows:
        rec = _row_to_dict(row)
        trajectory = analyze(
            session_id + "_admin_replay",  # isolated session key
            rec["risk_score"],
            rec.get("category", "safe"),
        )

    return {
        "session_id": session_id,
        "escalation_count": len(rows),
        "history": [_row_to_dict(r) for r in rows],
        "trajectory": (
            {
                "spike_detected": trajectory.spike_detected,
                "trend": trajectory.trend,
                "window_scores": trajectory.window_scores,
                "category_counts": trajectory.category_counts,
            }
            if trajectory
            else None
        ),
    }


@router.get("/stats")
def get_stats(_: str = Depends(_require_admin)) -> dict:
    """Return aggregate safety statistics."""
    conn = _get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
        by_category = conn.execute(
            "SELECT category, COUNT(*) FROM escalations GROUP BY category"
        ).fetchall()
        by_day = conn.execute(
            """SELECT date(timestamp, 'unixepoch') as day, COUNT(*)
               FROM escalations GROUP BY day ORDER BY day DESC LIMIT 30"""
        ).fetchall()
        avg_score = conn.execute(
            "SELECT AVG(risk_score) FROM escalations"
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "total_escalations": total,
        "by_category": dict(by_category),
        "by_day": dict(by_day),
        "average_risk_score": round(avg_score or 0.0, 3),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session_data(
    session_id: str,
    _: str = Depends(_require_admin),
) -> None:
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
