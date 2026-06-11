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
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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

    email_cfg = config.get("escalation", {}).get("webhooks", {}).get("email", {})
    if isinstance(email_cfg, dict) and email_cfg.get("password"):
        email_cfg["password"] = "***REDACTED***"

    return config


# ---------------------------------------------------------------------------
# Escalations list
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
    # Validate date filters early before hitting the store.
    ts_from: float | None = None
    ts_to: float | None = None
    if date_from:
        try:
            ts_from = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            raise HTTPException(400, f"Invalid date_from format: {date_from}")
    if date_to:
        try:
            ts_to = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            raise HTTPException(400, f"Invalid date_to format: {date_to}")

    store = get_store()
    items = store.query(
        category=category,
        session_id=session_id,
        limit=limit,
        offset=offset,
        date_from=ts_from,
        date_to=ts_to,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    total = store.count(
        category=category,
        session_id=session_id,
        date_from=ts_from,
        date_to=ts_to,
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
    store = get_store()
    items = store.query(category=category, session_id=session_id, limit=10_000, offset=0)

    _COLS = ["id", "session_id", "category", "risk_score", "triggers",
             "timestamp", "message_hash", "stage_reached", "reasoning"]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=_COLS, extrasaction="ignore")
    writer.writeheader()
    for item in items:
        writer.writerow(item)

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=escalations.csv"},
    )


# ---------------------------------------------------------------------------
# Single escalation
# ---------------------------------------------------------------------------

@router.get("/escalations/{escalation_id}")
def get_escalation(
    escalation_id: int,
    _: str = Depends(_require_admin),
) -> dict:
    """Get a single escalation record by ID."""
    store = get_store()
    record = store.get_by_id(escalation_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found.")
    return record


# ---------------------------------------------------------------------------
# Session risk
# ---------------------------------------------------------------------------

@router.get("/sessions/{session_id}/risk")
def get_session_risk(
    session_id: str,
    _: str = Depends(_require_admin),
) -> dict:
    """Return escalation history + current trajectory for a session."""
    store = get_store()
    history = store.query(session_id=session_id, limit=500, offset=0)

    from humane_proxy.risk.trajectory import snapshot
    trajectory = snapshot(session_id)

    return {
        "session_id": session_id,
        "escalation_count": len(history),
        "history": history,
        "trajectory": {
            "spike_detected": trajectory.spike_detected,
            "trend": trajectory.trend,
            "window_scores": trajectory.window_scores,
            "category_counts": trajectory.category_counts,
            "message_count": trajectory.message_count,
        },
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@router.get("/stats")
def get_stats(_: str = Depends(_require_admin)) -> dict:
    """Return aggregate safety statistics with enhanced breakdowns."""
    store = get_store()
    data = store.stats()
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    return data


# ---------------------------------------------------------------------------
# Delete session
# ---------------------------------------------------------------------------

@router.delete("/sessions/{session_id}", status_code=204, response_class=Response)
def delete_session_data(
    session_id: str,
    _: str = Depends(_require_admin),
) -> Response:
    """Delete all escalation records for a session (privacy right to erasure)."""
    store = get_store()
    deleted = store.delete_session(session_id)
    logger.info("Deleted %d records for session %s (admin request)", deleted, session_id)
    return Response(status_code=204)