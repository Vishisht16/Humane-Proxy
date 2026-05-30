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
# Helpers
# ---------------------------------------------------------------------------

_COLS = ["id", "session_id", "category", "risk_score", "triggers",
         "timestamp", "message_hash", "stage_reached", "reasoning"]


def _parse_iso_timestamp(value: str | None, param_name: str) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        raise HTTPException(400, f"Invalid {param_name} format: {value}")


def _matches_date_range(
    record: dict[str, Any],
    *,
    start_ts: float | None,
    end_ts: float | None,
) -> bool:
    ts = float(record.get("timestamp", 0))
    if start_ts is not None and ts < start_ts:
        return False
    if end_ts is not None and ts > end_ts:
        return False
    return True


def _get_matching_records(
    *,
    category: str | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    store = get_store()
    total = store.count(category=category, session_id=session_id)
    if total <= 0:
        return []
    return store.query(category=category, session_id=session_id, limit=total, offset=0)


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        import json
        return json.dumps(value)
    return value


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
    allowed_sort = {"timestamp", "risk_score", "category", "session_id", "stage_reached"}
    sort_col = sort_by if sort_by in allowed_sort else "timestamp"
    reverse = sort_order.lower() != "asc"

    start_ts = _parse_iso_timestamp(date_from, "date_from")
    end_ts = _parse_iso_timestamp(date_to, "date_to")
    store = get_store()

    if start_ts is None and end_ts is None and sort_col == "timestamp" and reverse:
        items = store.query(category=category, session_id=session_id, limit=limit, offset=offset)
        total = store.count(category=category, session_id=session_id)
    else:
        records = [
            record
            for record in _get_matching_records(category=category, session_id=session_id)
            if _matches_date_range(record, start_ts=start_ts, end_ts=end_ts)
        ]
        records.sort(key=lambda record: record.get(sort_col) or 0, reverse=reverse)
        total = len(records)
        items = records[offset:offset + limit]

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
    rows = _get_matching_records(category=category, session_id=session_id)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(_COLS)
    for row in rows:
        writer.writerow([_csv_value(row.get(col, "")) for col in _COLS])

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
    row = get_store().get_by_id(escalation_id)

    if row is None:
        raise HTTPException(status_code=404, detail=f"Escalation {escalation_id} not found.")
    return row


@router.get("/sessions/{session_id}/risk")
def get_session_risk(
    session_id: str,
    _: str = Depends(_require_admin),
) -> dict:
    """Return escalation history + current trajectory for a session."""
    rows = _get_matching_records(session_id=session_id)
    rows.sort(key=lambda record: record.get("timestamp") or 0)

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
    stats = get_store().stats()
    stats["generated_at"] = datetime.now(timezone.utc).isoformat()
    return stats


@router.delete("/sessions/{session_id}", status_code=204, response_class=Response)
def delete_session_data(
    session_id: str,
    _: str = Depends(_require_admin),
) -> Response:
    """Delete all escalation records for a session (privacy right to erasure)."""
    deleted = get_store().delete_session(session_id)

    logger.info("Deleted %d records for session %s (admin request)", deleted, session_id)
    return Response(status_code=204)
