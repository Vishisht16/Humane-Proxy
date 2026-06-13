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

# International crisis resource database
# ---------------------------------------------------------------------------
# Single source of truth: per-country resource blocks, assembled into
# CARE_RESPONSE_BLOCK below. Keeping these structured (rather than one opaque
# string) lets an operator optionally surface a specific region's resources
# first via `safety.categories.self_harm.region`, without dropping the others.

_CARE_INTRO = (
    "It sounds like you may be going through something really difficult right now. \\\n"
    "You are not alone, and there are people who care about you and want to help.\n\n"
    "Please reach out to a crisis service near you:"
)

# Ordered: code -> resource block. Order preserved for the default (no-region) message.
_CRISIS_RESOURCES: dict[str, str] = {
    "US": "🇺🇸 United States\n  • 988 Suicide & Crisis Lifeline: Call or text 988\n  • Crisis Text Line: Text HOME to 741741",
    "IN": "🇮🇳 India\n  • iCall (TISS): 9152987821\n  • Vandrevala Foundation: 1860-2662-345 (24/7)\n  • NIMHANS: 080-46110007",
    "GB": "🇬🇧 United Kingdom\n  • Samaritans: 116 123 (free, 24/7)\n  • PAPYRUS (youth): 0800 068 4141",
    "AU": "🇦🇺 Australia\n  • Lifeline: 13 11 14\n  • Beyond Blue: 1300 22 4636",
    "CA": "🇨🇦 Canada\n  • Talk Suicide Canada: 1-833-456-4566\n  • Crisis Text Line: Text HOME to 686868",
    "DE": "🇩🇪 Germany\n  • Telefonseelsorge: 0800 111 0 111 (free, 24/7)",
    "FR": "🇫🇷 France\n  • Suicide Écoute: 01 45 39 40 00\n  • Numéro National Prévention Suicide: 3114",
    "BR": "🇧🇷 Brazil\n  • CVV (Centro de Valorização da Vida): 188",
    "ZA": "🇿🇦 South Africa\n  • SADAG: 0800 567 567",
    "JP": "🇯🇵 Japan\n  • Inochi no Denwa: 0120-783-556",
    "KR": "🇰🇷 South Korea\n  • Crisis Counseling Hotline: 1393",
    "ES": "🇪🇸 Spain\n  • Línea de Atención a conducta suicida: 024",
    "IT": "🇮🇹 Italy\n  • Telefono Amico: 800 274 274",
    "MX": "🇲🇽 Mexico\n  • SAPTEL: 800 290 0024",
    "NZ": "🇳🇿 New Zealand\n  • Need to Talk?: 1737",
}

_CARE_INTERNATIONAL = (
    "🌐 International:\n"
    "  • IASP Crisis Centres: https://www.iasp.info/resources/Crisis_Centres/\n"
    "  • Befrienders Worldwide: https://www.befrienders.org"
)

_CARE_EMERGENCY = (
    "If you are in immediate danger, please call your local emergency number "
    "(e.g. 112, 911, 999, 100)."
)


def _build_care_block(region: str | None = None) -> str:
    """Assemble the crisis-resource block.

    When ``region`` matches a known country code, that country's resources are
    surfaced first; all other resources still follow. When ``region`` is None
    or unknown, the full list is returned in the default order.
    """
    ordered = list(_CRISIS_RESOURCES.values())
    if region:
        block = _CRISIS_RESOURCES.get(region.strip().upper())
        if block:
            ordered = [block] + [b for b in ordered if b != block]
    body = "\n\n".join(ordered)
    return f"{_CARE_INTRO}\n\n{body}\n\n{_CARE_INTERNATIONAL}\n\n{_CARE_EMERGENCY}"


# Default full message (no region). Kept as a module constant for backward
# compatibility — existing callers and tests reference CARE_RESPONSE_BLOCK.
CARE_RESPONSE_BLOCK = _build_care_block()

CARE_CONTEXT_INJECTION = (
    "\n[SYSTEM NOTE: The user's message may indicate emotional distress or self-harm ideation. "
    "Please respond with deep empathy, validate their feelings, gently provide crisis resources "
    "(and international helplines if relevant), and encourage professional support. "
    "Do not be dismissive. Do not give advice that could cause harm.]\n"
)


def _get_care_response() -> str:
    """Return the configured block message, falling back to the built-in one.

    Honors an optional ``safety.categories.self_harm.region`` setting, which
    surfaces that country's resources first. A custom ``block_message`` still
    takes precedence over both.
    """
    cfg = get_config()
    self_harm = cfg.get("safety", {}).get("categories", {}).get("self_harm", {})
    custom = self_harm.get("block_message", "")
    if custom and custom.strip():
        return custom.strip()
    region = self_harm.get("region", "")
    if region:
        return _build_care_block(region)
    return CARE_RESPONSE_BLOCK

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
