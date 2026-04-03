"""LlamaIndex integration for HumaneProxy.

Provides safety tools as native LlamaIndex ``FunctionTool`` instances
for use in any LlamaIndex agent or query pipeline.

Quick start::

    from humane_proxy.integrations.llamaindex import get_safety_tools
    tools = get_safety_tools()
    # → [check_message_safety, get_session_risk, list_recent_escalations]

Requires::

    pip install humane-proxy[llamaindex]
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("humane_proxy.integrations.llamaindex")


def _check_message_safety(message: str, session_id: str = "llamaindex-default") -> dict:
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
        ``{"safe": bool, "category": str, "score": float, "triggers": list, ...}``
    """
    from humane_proxy import HumaneProxy

    proxy = HumaneProxy()
    return proxy.check(message, session_id=session_id)


def _get_session_risk(session_id: str) -> dict:
    """Return the current risk trajectory for a session.

    Parameters
    ----------
    session_id:
        The session identifier to query.

    Returns
    -------
    dict
        ``{"spike_detected": bool, "trend": str, "window_scores": list, ...}``
    """
    from humane_proxy.risk.trajectory import analyze

    result = analyze(session_id, 0.0, "safe")
    return {
        "spike_detected": result.spike_detected,
        "trend": result.trend,
        "window_scores": result.window_scores,
        "category_counts": result.category_counts,
        "message_count": result.message_count,
    }


def _list_recent_escalations(limit: int = 20, category: str | None = None) -> list[dict]:
    """Return recent escalation events from the audit log.

    Parameters
    ----------
    limit:
        Maximum number of events to return.
    category:
        Filter by category (``"self_harm"`` or ``"criminal_intent"``).

    Returns
    -------
    list[dict]
        List of escalation records.
    """
    from humane_proxy.storage.factory import get_store

    store = get_store()
    return store.query(category=category, limit=limit)


def get_safety_tools() -> list:
    """Return HumaneProxy safety tools as LlamaIndex FunctionTool instances.

    Returns
    -------
    list[FunctionTool]
        Three tools: check_message_safety, get_session_risk, list_recent_escalations.

    Raises
    ------
    ImportError
        If ``llama-index-core`` is not installed.
    """
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError:
        raise ImportError(
            "LlamaIndex integration requires llama-index-core.\n"
            "Install with: pip install humane-proxy[llamaindex]"
        )

    tools = [
        FunctionTool.from_defaults(
            fn=_check_message_safety,
            name="check_message_safety",
            description=(
                "Classify a user message for self-harm or criminal intent. "
                "Returns safety verdict, category, score, and triggers."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_get_session_risk,
            name="get_session_risk",
            description=(
                "Get the current risk trajectory for a session including "
                "spike detection, trend, and category distribution."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_list_recent_escalations,
            name="list_recent_escalations",
            description="Query recent escalation events from the safety audit log.",
        ),
    ]

    logger.info("Created %d LlamaIndex safety tools", len(tools))
    return tools
