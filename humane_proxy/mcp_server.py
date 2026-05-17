"""HumaneProxy MCP Server — expose safety tools via Model Context Protocol.

Run with:  humane-proxy mcp-serve
Or import: from humane_proxy.mcp_server import mcp

Requires: pip install humane-proxy[mcp]
"""

from __future__ import annotations

import ipaddress
import logging
import os

logger = logging.getLogger("humane_proxy.mcp")

MCP_TOKEN_ENV = "HUMANE_PROXY_ADMIN_KEY"
MCP_DEFAULT_HOST = "127.0.0.1"


def _is_public_bind_host(host: str) -> bool:
    """Return whether an HTTP bind host may be reachable beyond localhost."""
    normalized = (host or "").strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]

    if not normalized:
        return True
    if normalized.lower() == "localhost":
        return False

    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return True

    return not address.is_loopback


def _get_mcp_auth_provider():
    """Return a FastMCP Bearer auth provider when HTTP MCP auth is configured."""
    token = os.environ.get(MCP_TOKEN_ENV, "").strip()
    if not token:
        return None

    try:
        from fastmcp.server.auth import BearerTokenAuth  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            f"{MCP_TOKEN_ENV} is set, but this FastMCP version does not expose "
            "server Bearer token auth. Upgrade fastmcp to use HTTP MCP auth."
        ) from exc

    return BearerTokenAuth(token=token)


try:
    from fastmcp import FastMCP  # type: ignore[import]

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    FastMCP = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# MCP app instance
# ---------------------------------------------------------------------------


def _init_telemetry() -> None:
    """Initialize OpenTelemetry for MCP server execution."""
    from humane_proxy.config import get_config
    from humane_proxy.telemetry import setup_telemetry

    config = get_config()
    setup_telemetry(config)

if _MCP_AVAILABLE:
    
    mcp_kwargs = {}
    auth_provider = _get_mcp_auth_provider()
    if auth_provider is not None:
        mcp_kwargs["auth"] = auth_provider
    mcp = FastMCP(
        "humane-proxy",
        **mcp_kwargs,
    )

    @mcp.tool()
    async def check_message_safety(
        message: str,
        session_id: str = "mcp-default",
    ) -> dict:
        """Classify a message for self-harm or criminal intent.

        Parameters
        ----------
        message:
            The user message to classify.
        session_id:
            Optional session identifier for trajectory tracking.

        Returns
        -------
        dict
            ``{"safe": bool, "category": str, "score": float, "triggers": list,
               "stage_reached": int, "should_escalate": bool, ...}``
        """
        from humane_proxy.config import get_config
        from humane_proxy.classifiers.pipeline import SafetyPipeline

        config = get_config()
        pipeline = SafetyPipeline(config)
        result = await pipeline.classify(message, session_id)
        return result.to_dict()

    @mcp.tool()
    async def get_session_risk(session_id: str) -> dict:
        """Return the current risk trajectory for a session.

        Parameters
        ----------
        session_id:
            The session identifier to query.

        Returns
        -------
        dict
            ``{"spike_detected": bool, "trend": str, "window_scores": list,
               "category_counts": dict, "message_count": int}``
        """
        from humane_proxy.risk.trajectory import snapshot, to_dict

        return to_dict(snapshot(session_id))

    @mcp.tool()
    async def list_recent_escalations(
        limit: int = 20,
        category: str | None = None,
    ) -> list[dict]:
        """Return recent escalation events from the audit log.

        Parameters
        ----------
        limit:
            Maximum number of events to return (default 20).
        category:
            Filter by category (``"self_harm"`` or ``"criminal_intent"``).
            Omit for all categories.

        Returns
        -------
        list[dict]
            List of escalation records.
        """
        import json
        import sqlite3
        from humane_proxy.escalation.query import normalize_escalation_query
        from humane_proxy.escalation.local_db import _get_db_path

        limit, category = normalize_escalation_query(limit, category)

        conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
        try:
            if category:
                rows = conn.execute(
                    "SELECT * FROM escalations WHERE category=? ORDER BY timestamp DESC LIMIT ?",
                    (category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM escalations ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        finally:
            conn.close()

        cols = [
            "id",
            "session_id",
            "category",
            "risk_score",
            "triggers",
            "timestamp",
            "message_hash",
            "stage_reached",
            "reasoning",
        ]
        result = []
        for row in rows:
            rec = dict(zip(cols, row))
            try:
                rec["triggers"] = json.loads(rec["triggers"])
            except Exception:
                pass
            result.append(rec)
        return result

else:
    mcp = None  # type: ignore[assignment]

def serve() -> None:
    """Start the MCP server in stdio mode (called by `humane-proxy mcp-serve`)."""
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "MCP server requires fastmcp. Install with: pip install humane-proxy[mcp]"
        )

    _init_telemetry()

    assert mcp is not None
    mcp.run()


def serve_http(host: str = MCP_DEFAULT_HOST, port: int = 3000) -> None:
    """Start the MCP server in Streamable HTTP mode."""

    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "MCP server requires fastmcp. Install with: pip install humane-proxy[mcp]"
        )

    if _is_public_bind_host(host) and not os.environ.get(MCP_TOKEN_ENV, "").strip():
        logger.warning(
            "Starting HTTP MCP on public host %s without %s. "
            "Set a bearer token before exposing this server beyond localhost.",
            host,
            MCP_TOKEN_ENV,
        )

    _init_telemetry()

    assert mcp is not None
    mcp.run(transport="http", host=host, port=port)
