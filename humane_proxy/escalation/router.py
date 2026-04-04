"""Escalation router — rate-limited, DB-backed, with self-harm care response."""

from __future__ import annotations

import asyncio
import logging

from humane_proxy.config import get_config
from humane_proxy.escalation.local_db import check_rate_limit, log_escalation

logger = logging.getLogger("humane_proxy.escalation")

# ---------------------------------------------------------------------------
# International crisis resource database
# ---------------------------------------------------------------------------

CARE_RESPONSE_BLOCK = """\
It sounds like you may be going through something really difficult right now. \
You are not alone, and there are people who care about you and want to help.

Please reach out to a crisis service near you:

🇺🇸 United States
  • 988 Suicide & Crisis Lifeline: Call or text 988
  • Crisis Text Line: Text HOME to 741741

🇮🇳 India
  • iCall (TISS): 9152987821
  • Vandrevala Foundation: 1860-2662-345 (24/7)
  • NIMHANS: 080-46110007

🇬🇧 United Kingdom
  • Samaritans: 116 123 (free, 24/7)
  • PAPYRUS (youth): 0800 068 4141

🇦🇺 Australia
  • Lifeline: 13 11 14
  • Beyond Blue: 1300 22 4636

🇨🇦 Canada
  • Talk Suicide Canada: 1-833-456-4566
  • Crisis Text Line: Text HOME to 686868

🇩🇪 Germany
  • Telefonseelsorge: 0800 111 0 111 (free, 24/7)

🇫🇷 France
  • Suicide Écoute: 01 45 39 40 00

🇧🇷 Brazil
  • CVV (Centro de Valorização da Vida): 188

🇿🇦 South Africa
  • SADAG: 0800 567 567

🇯🇵 Japan
  • TELL Japan: 03-5774-0992 (English/Japanese)
  • Yorisoi Hotline: 0120-279-338

🇰🇷 South Korea
  • Suicide Prevention Hotline: 1393

🇮🇹 Italy
  • Telefono Amico: 02 2327 2327

🇪🇸 Spain
  • Línea 024: 024 (free, 24/7)

🇲🇽 Mexico
  • Línea de la Vida: 800 911 2000

🇳🇿 New Zealand
  • Need to Talk?: 1737 (Call or text)

🇮🇪 Ireland
  • Samaritans: 116 123

🇵🇭 Philippines
  • Hopeline PH: 02-8804-4673 or 2919 (Toll-free for Globe/TM)

🇳🇬 Nigeria
  • MANI: 0806 210 6340

🇦🇷 Argentina
  • Casistencia al Suicida: 0800 345 1435

🌐 International:
  • IASP Crisis Centres: https://www.iasp.info/resources/Crisis_Centres/
  • Befrienders Worldwide: https://www.befrienders.org

If you are in immediate danger, please call your local emergency number (e.g. 112, 911, 999, 100).\
"""

CARE_CONTEXT_INJECTION = (
    "\n[SYSTEM NOTE: The user's message may indicate emotional distress or self-harm ideation. "
    "Please respond with deep empathy, validate their feelings, gently provide crisis resources "
    "(and international helplines if relevant), and encourage professional support. "
    "Do not be dismissive. Do not give advice that could cause harm.]\n"
)


def _get_care_response() -> str:
    """Return the configured block message, falling back to the built-in one."""
    cfg = get_config()
    custom = (
        cfg.get("safety", {})
        .get("categories", {})
        .get("self_harm", {})
        .get("block_message", "")
    )
    return custom.strip() if custom else CARE_RESPONSE_BLOCK


def _get_response_mode() -> str:
    """Return 'block' (default) or 'forward'."""
    cfg = get_config()
    return (
        cfg.get("safety", {})
        .get("categories", {})
        .get("self_harm", {})
        .get("response_mode", "block")
    )


def get_self_harm_response(original_payload: dict | None = None) -> dict:
    """Build the appropriate response dict for a self-harm detection event.

    Parameters
    ----------
    original_payload:
        The original request JSON (only used in ``\"forward\"`` mode to inject
        care context into the messages array before forwarding to the LLM).

    Returns
    -------
    dict
        ``{"mode": "block"|"forward", "message": str, "payload": dict|None}``
    """
    mode = _get_response_mode()

    if mode == "forward" and original_payload:
        # Inject care context as a system message at the front.
        messages = list(original_payload.get("messages", []))
        messages.insert(0, {"role": "system", "content": CARE_CONTEXT_INJECTION.strip()})
        return {
            "mode": "forward",
            "message": CARE_CONTEXT_INJECTION.strip(),
            "payload": {**original_payload, "messages": messages},
        }

    # Default: block with empathetic care response.
    return {
        "mode": "block",
        "message": _get_care_response(),
        "payload": None,
    }


def escalate(
    session_id: str,
    risk_score: float,
    triggers: list[str],
    category: str = "unknown",
    message_hash: str | None = None,
    stage_reached: int = 1,
    reasoning: str | None = None,
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
    message_hash:
        SHA-256 hash of the original message (if privacy mode is on).
    stage_reached:
        Which pipeline stage (1, 2, or 3) produced the final result.
    reasoning:
        Stage-3 reasoning string (if available).

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
            session_id, category, risk_score,
        )
        return {
            "escalated": False,
            "reason": "rate_limited",
            "session_id": session_id,
            "category": category,
        }

    # --- Persist (with failure protection) ---
    try:
        log_escalation(
            session_id, risk_score, triggers, category,
            message_hash=message_hash,
            stage_reached=stage_reached,
            reasoning=reasoning,
        )
    except Exception:
        logger.exception("Failed to write escalation to DB for session=%s", session_id)
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
        "!!  Category     : %s\n"
        "!!  Session      : %s\n"
        "!!  Risk Score   : %.2f\n"
        "!!  Stage Reached: %d\n"
        "!!  Triggers     : %s\n"
        "%s",
        border, category_emoji, category, session_id,
        risk_score, stage_reached, triggers, border,
    )

    # --- Fire webhooks (async, non-blocking) ---
    _fire_webhooks(session_id, risk_score, list(triggers), category)

    return {
        "escalated": True,
        "reason": "logged",
        "session_id": session_id,
        "category": category,
        "risk_score": risk_score,
        "triggers": list(triggers),
        "stage_reached": stage_reached,
    }


def _fire_webhooks(
    session_id: str, risk_score: float, triggers: list[str], category: str
) -> None:
    """Dispatch webhooks in the background — never blocks or crashes."""
    try:
        from humane_proxy.escalation.webhooks import dispatch_webhooks

        config = get_config()
        webhooks = config.get("escalation", {}).get("webhooks", {})

        has_any = any(
            webhooks.get(k)
            for k in ("slack_url", "discord_url", "pagerduty_routing_key",
                       "teams_url", "email_to")
        )
        if not has_any:
            return

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                dispatch_webhooks(config, session_id, risk_score, triggers, category)
            )
        except RuntimeError:
            asyncio.run(
                dispatch_webhooks(config, session_id, risk_score, triggers, category)
            )
    except Exception:
        logger.exception("Webhook dispatch setup failed")
