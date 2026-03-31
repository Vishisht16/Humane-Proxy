"""Webhook dispatcher for escalation alerts — Slack, Discord, PagerDuty.

All dispatchers are **fire-and-forget**: they never raise exceptions to the
caller, and they never block the request pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import httpx

logger = logging.getLogger("humane_proxy.escalation.webhooks")


async def _post(url: str, payload: dict, *, headers: dict | None = None) -> None:
    """POST JSON to *url*, swallowing all errors."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers or {})
            if resp.status_code >= 400:
                logger.warning(
                    "Webhook %s returned HTTP %d: %s",
                    url[:60], resp.status_code, resp.text[:200],
                )
    except Exception:
        logger.exception("Webhook dispatch to %s failed", url[:60])


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

async def send_slack(
    webhook_url: str,
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> None:
    """Send a Slack Block Kit formatted alert."""
    category_emoji = "🆘" if category == "self_harm" else "⚠️"
    trigger_text = "\n".join(f"• {t}" for t in triggers) or "(none)"
    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{category_emoji} HumaneProxy Alert — {category}", "emoji": True},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Session:*\n`{session_id}`"},
                    {"type": "mrkdwn", "text": f"*Risk Score:*\n`{risk_score:.2f}`"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Triggers:*\n{trigger_text}"},
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"⏱ {datetime.now(timezone.utc).isoformat()}"},
                ],
            },
        ]
    }
    await _post(webhook_url, payload)


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

async def send_discord(
    webhook_url: str,
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> None:
    """Send a Discord embed formatted alert."""
    # Red for self-harm, orange for criminal intent
    color = 15158332 if category == "self_harm" else 16744192
    category_emoji = "🆘" if category == "self_harm" else "⚠️"
    trigger_text = "\n".join(f"• {t}" for t in triggers) or "(none)"
    payload = {
        "embeds": [
            {
                "title": f"{category_emoji} HumaneProxy Alert — {category}",
                "color": color,
                "fields": [
                    {"name": "Session", "value": f"`{session_id}`", "inline": True},
                    {"name": "Risk Score", "value": f"`{risk_score:.2f}`", "inline": True},
                    {"name": "Category", "value": f"`{category}`", "inline": True},
                    {"name": "Triggers", "value": trigger_text, "inline": False},
                ],
                "footer": {"text": datetime.now(timezone.utc).isoformat()},
            }
        ]
    }
    await _post(webhook_url, payload)


# ---------------------------------------------------------------------------
# PagerDuty (Events API v2)
# ---------------------------------------------------------------------------

async def send_pagerduty(
    routing_key: str,
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> None:
    """Send a PagerDuty Events API v2 trigger event."""
    payload = {
        "routing_key": routing_key,
        "event_action": "trigger",
        "payload": {
            "summary": f"HumaneProxy: [{category}] session {session_id} flagged (score={risk_score:.2f})",
            "severity": "critical",
            "source": "humane-proxy",
            "custom_details": {
                "session_id": session_id,
                "category": category,
                "risk_score": risk_score,
                "triggers": triggers,
            },
        },
    }
    await _post(
        "https://events.pagerduty.com/v2/enqueue",
        payload,
        headers={"Content-Type": "application/json"},
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

async def dispatch_webhooks(
    config: dict,
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> None:
    """Fire all configured webhooks.  Called from the escalation router."""
    webhooks = config.get("escalation", {}).get("webhooks", {})

    slack_url = webhooks.get("slack_url", "")
    if slack_url:
        await send_slack(slack_url, session_id, risk_score, triggers, category)

    discord_url = webhooks.get("discord_url", "")
    if discord_url:
        await send_discord(discord_url, session_id, risk_score, triggers, category)

    pd_key = webhooks.get("pagerduty_routing_key", "")
    if pd_key:
        await send_pagerduty(pd_key, session_id, risk_score, triggers, category)
