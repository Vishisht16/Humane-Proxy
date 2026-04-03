"""HumaneProxy MCP Server — expose safety tools via Model Context Protocol.

Run with:  humane-proxy mcp-serve
Or import: from humane_proxy.mcp_server import mcp

Requires: pip install humane-proxy[mcp]
"""

from __future__ import annotations

import logging

logger = logging.getLogger("humane_proxy.mcp")

try:
    from fastmcp import FastMCP  # type: ignore[import]
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    FastMCP = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# MCP app instance
# ---------------------------------------------------------------------------

if _MCP_AVAILABLE:
    mcp = FastMCP(
        "humane-proxy"
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
        from humane_proxy.risk.trajectory import analyze

        # Analyze with a neutral message to get current state.
        result = analyze(session_id, 0.0, "safe")
        return {
            "spike_detected": result.spike_detected,
            "trend": result.trend,
            "window_scores": result.window_scores,
            "category_counts": result.category_counts,
            "message_count": result.message_count,
        }

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
        from humane_proxy.escalation.local_db import _get_db_path

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

        cols = ["id", "session_id", "category", "risk_score", "triggers",
                "timestamp", "message_hash", "stage_reached", "reasoning"]
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
    assert mcp is not None
    mcp.run()


def serve_http(host: str = "0.0.0.0", port: int = 3000) -> None:
    """Start the MCP server in Streamable HTTP mode.

    This exposes the MCP tools over HTTP, making the server compatible
    with remote MCP clients and registries like Smithery that require
    a publicly accessible HTTPS endpoint.

    Parameters
    ----------
    host:
        Bind address (default ``"0.0.0.0"``).
    port:
        Bind port (default ``3000``).
    """
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "MCP server requires fastmcp. Install with: pip install humane-proxy[mcp]"
        )
    assert mcp is not None
    mcp.run(transport="http", host=host, port=port)

