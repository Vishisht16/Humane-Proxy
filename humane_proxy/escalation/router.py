"""Escalation router — rate-limited, DB-backed, with structured logging and webhooks."""

from __future__ import annotations

import asyncio
import logging

from humane_proxy.config import get_config
from humane_proxy.escalation.local_db import check_rate_limit, log_escalation

logger = logging.getLogger("humane_proxy.escalation")


def escalate(
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> dict:
    """Handle a flagged interaction.

    Flow
    ----
    1. Check the per-session rate limit.
    2. If allowed → log to SQLite, emit a CRITICAL log, fire webhooks.
    3. Return a result dict indicating the outcome.

    Parameters
    ----------
    session_id:
        Identifier for the session / user that was flagged.
    risk_score:
        The final aggregated risk score that caused the flag.
    triggers:
        Human-readable list of triggers that contributed to the score.
    category:
        The detected category (``"self_harm"``, ``"criminal_intent"``).

    Returns
    -------
    dict
        ``{"escalated": True/False, "reason": "...", "category": "..."}``
    """
    triggers = triggers or []

    # --- Rate-limit gate ---
    if not check_rate_limit(session_id):
        logger.warning(
            "[RATE-LIMITED] session=%s  category=%s  risk_score=%.2f — suppressed (quota exhausted)",
            session_id,
            category,
            risk_score,
        )
        return {
            "escalated": False,
            "reason": "rate_limited",
            "session_id": session_id,
            "category": category,
        }

    # --- Persist (with failure protection) ---
    try:
        log_escalation(session_id, risk_score, triggers, category)
    except Exception:
        logger.exception(
            "Failed to write escalation to DB for session=%s", session_id
        )
        return {
            "escalated": True,
            "reason": "logged_with_db_error",
            "session_id": session_id,
            "category": category,
            "risk_score": risk_score,
            "triggers": list(triggers),
        }

    # --- Highly-visible structured log ---
    category_emoji = "🆘" if category == "self_harm" else "⚠️"
    border = "!" * 60
    logger.critical(
        "\n%s\n"
        "!!  %s CRITICAL SAFETY ALERT\n"
        "!!  Category : %s\n"
        "!!  Session  : %s\n"
        "!!  Risk Score: %.2f\n"
        "!!  Triggers : %s\n"
        "%s",
        border, category_emoji, category, session_id, risk_score, triggers, border,
    )

    # --- Fire webhooks (async, non-blocking) ---
    _fire_webhooks(session_id, risk_score, list(triggers), category)

    return {
        "escalated": True,
        "reason": "logged",
        "session_id": session_id,
        "category": category,
        "risk_score": risk_score,
        "triggers": list(triggers),    # defensive copy
    }


def _fire_webhooks(
    session_id: str, risk_score: float, triggers: list[str], category: str
) -> None:
    """Dispatch webhooks in the background — never blocks or crashes."""
    try:
        from humane_proxy.escalation.webhooks import dispatch_webhooks

        config = get_config()
        webhooks = config.get("escalation", {}).get("webhooks", {})

        # Only bother if at least one URL is configured.
        has_any = any(
            webhooks.get(k)
            for k in ("slack_url", "discord_url", "pagerduty_routing_key")
        )
        if not has_any:
            return

        # If there's a running event loop (FastAPI), schedule as a task.
        # Otherwise (CLI / tests), run synchronously.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                dispatch_webhooks(config, session_id, risk_score, triggers, category)
            )
        except RuntimeError:
            # No running loop — run in a fresh one (CLI context).
            asyncio.run(
                dispatch_webhooks(config, session_id, risk_score, triggers, category)
            )
    except Exception:
        logger.exception("Webhook dispatch setup failed")
