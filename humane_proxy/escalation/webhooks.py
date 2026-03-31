"""Enhanced webhooks — Slack, Discord, PagerDuty, Microsoft Teams, Email.

All dispatchers are **fire-and-forget**: they never raise exceptions to the
caller, and they never block the request pipeline.
"""

from __future__ import annotations

import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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
                "text": {
                    "type": "plain_text",
                    "text": f"{category_emoji} HumaneProxy Alert — {category}",
                    "emoji": True,
                },
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
# Microsoft Teams (Adaptive Card via Incoming Webhook)
# ---------------------------------------------------------------------------

async def send_teams(
    webhook_url: str,
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> None:
    """Send a Microsoft Teams adaptive card alert."""
    category_emoji = "🆘" if category == "self_harm" else "⚠️"
    trigger_text = "\n\n".join(f"• {t}" for t in triggers) or "(none)"
    color = "FF0000" if category == "self_harm" else "FF8C00"
    payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {
                            "type": "TextBlock",
                            "text": f"{category_emoji} HumaneProxy Alert — {category}",
                            "weight": "Bolder",
                            "size": "Large",
                            "color": "Attention" if category == "self_harm" else "Warning",
                        },
                        {
                            "type": "FactSet",
                            "facts": [
                                {"title": "Session", "value": session_id},
                                {"title": "Risk Score", "value": f"{risk_score:.2f}"},
                                {"title": "Category", "value": category},
                                {"title": "Time", "value": datetime.now(timezone.utc).isoformat()},
                            ],
                        },
                        {
                            "type": "TextBlock",
                            "text": f"**Triggers:**\n\n{trigger_text}",
                            "wrap": True,
                        },
                    ],
                },
            }
        ],
    }
    await _post(webhook_url, payload)


# ---------------------------------------------------------------------------
# Email (via smtplib — stdlib, zero extra deps)
# ---------------------------------------------------------------------------

async def send_email(
    smtp_config: dict,
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
) -> None:
    """Send an email alert using stdlib smtplib (runs in thread pool)."""
    import asyncio

    def _send_sync() -> None:
        host = smtp_config.get("host", "localhost")
        port = smtp_config.get("port", 587)
        user = smtp_config.get("username", "")
        password = smtp_config.get("password", "")
        from_addr = smtp_config.get("from", user)
        to_addrs = smtp_config.get("to", [])

        if not to_addrs:
            return

        category_emoji = "🆘" if category == "self_harm" else "⚠️"
        trigger_list = "\n".join(f"  • {t}" for t in triggers) or "  (none)"
        body = (
            f"{category_emoji} HumaneProxy Safety Alert\n"
            f"{'=' * 50}\n\n"
            f"Category  : {category}\n"
            f"Session   : {session_id}\n"
            f"Risk Score: {risk_score:.2f}\n"
            f"Time      : {datetime.now(timezone.utc).isoformat()}\n\n"
            f"Triggers:\n{trigger_list}\n"
        )

        msg = MIMEMultipart()
        msg["Subject"] = f"[HumaneProxy] {category_emoji} {category} alert — session {session_id}"
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)
        msg.attach(MIMEText(body, "plain", "utf-8"))

        try:
            with smtplib.SMTP(host, port, timeout=10) as smtp:
                if smtp_config.get("use_tls", True):
                    smtp.starttls()
                if user and password:
                    smtp.login(user, password)
                smtp.sendmail(from_addr, to_addrs, msg.as_string())
        except Exception:
            logger.exception("Email alert failed to send")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _send_sync)
    except Exception:
        logger.exception("Email dispatch setup failed")


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

    teams_url = webhooks.get("teams_url", "")
    if teams_url:
        await send_teams(teams_url, session_id, risk_score, triggers, category)

    smtp_cfg = webhooks.get("email", {})
    if smtp_cfg and smtp_cfg.get("to"):
        await send_email(smtp_cfg, session_id, risk_score, triggers, category)
