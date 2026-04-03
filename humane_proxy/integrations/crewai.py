"""CrewAI integration for HumaneProxy.

Provides safety tools as native CrewAI ``BaseTool`` subclasses
with Pydantic input schemas for structured validation.

Quick start::

    from humane_proxy.integrations.crewai import get_safety_tools
    tools = get_safety_tools()
    # Use with: Agent(tools=tools, ...)

Requires::

    pip install humane-proxy[crewai]
"""

from __future__ import annotations

import logging
from typing import Any, Type

logger = logging.getLogger("humane_proxy.integrations.crewai")


def get_safety_tools() -> list:
    """Return HumaneProxy safety tools as CrewAI BaseTool instances.

    Returns
    -------
    list[BaseTool]
        Three tools: CheckMessageSafetyTool, GetSessionRiskTool,
        ListEscalationsTool.

    Raises
    ------
    ImportError
        If ``crewai`` is not installed.
    """
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "CrewAI integration requires crewai.\n"
            "Install with: pip install humane-proxy[crewai]"
        )

    # --- Input schemas ---

    class CheckMessageInput(BaseModel):
        """Input for checking message safety."""
        message: str = Field(..., description="The user message to classify.")
        session_id: str = Field(
            default="crewai-default",
            description="Session identifier for trajectory tracking.",
        )

    class SessionRiskInput(BaseModel):
        """Input for getting session risk."""
        session_id: str = Field(..., description="The session identifier to query.")

    class ListEscalationsInput(BaseModel):
        """Input for listing escalations."""
        limit: int = Field(default=20, description="Maximum number of events to return.")
        category: str = Field(
            default="",
            description="Filter by category: self_harm or criminal_intent. Empty for all.",
        )

    # --- Tool classes ---

    class CheckMessageSafetyTool(BaseTool):
        name: str = "check_message_safety"
        description: str = (
            "Classify a user message for self-harm or criminal intent. "
            "Returns safety verdict, category, score, and triggers."
        )
        args_schema: Type[BaseModel] = CheckMessageInput

        def _run(self, message: str, session_id: str = "crewai-default") -> str:
            from humane_proxy import HumaneProxy
            import json

            proxy = HumaneProxy()
            result = proxy.check(message, session_id=session_id)
            return json.dumps(result, indent=2)

    class GetSessionRiskTool(BaseTool):
        name: str = "get_session_risk"
        description: str = (
            "Get the current risk trajectory for a session including "
            "spike detection, trend, and category distribution."
        )
        args_schema: Type[BaseModel] = SessionRiskInput

        def _run(self, session_id: str) -> str:
            from humane_proxy.risk.trajectory import analyze
            import json

            result = analyze(session_id, 0.0, "safe")
            return json.dumps({
                "spike_detected": result.spike_detected,
                "trend": result.trend,
                "window_scores": result.window_scores,
                "category_counts": result.category_counts,
                "message_count": result.message_count,
            }, indent=2)

    class ListEscalationsTool(BaseTool):
        name: str = "list_recent_escalations"
        description: str = "Query recent escalation events from the safety audit log."
        args_schema: Type[BaseModel] = ListEscalationsInput

        def _run(self, limit: int = 20, category: str = "") -> str:
            from humane_proxy.storage.factory import get_store
            import json

            store = get_store()
            results = store.query(
                category=category if category else None,
                limit=limit,
            )
            return json.dumps(results, indent=2, default=str)

    tools = [
        CheckMessageSafetyTool(),
        GetSessionRiskTool(),
        ListEscalationsTool(),
    ]
    logger.info("Created %d CrewAI safety tools", len(tools))
    return tools
