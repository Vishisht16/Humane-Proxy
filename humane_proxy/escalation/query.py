"""Shared escalation query validation helpers."""

from __future__ import annotations

ALLOWED_ESCALATION_CATEGORIES = frozenset({"self_harm", "criminal_intent"})
DEFAULT_ESCALATION_LIMIT = 20
MAX_ESCALATION_LIMIT = 100


def normalize_escalation_query(
    limit: int = DEFAULT_ESCALATION_LIMIT,
    category: str | None = None,
) -> tuple[int, str | None]:
    """Clamp escalation query size and validate optional category filters."""
    try:
        normalized_limit = int(limit)
    except (TypeError, ValueError):
        normalized_limit = DEFAULT_ESCALATION_LIMIT

    normalized_limit = max(1, min(normalized_limit, MAX_ESCALATION_LIMIT))
    normalized_category = category.strip() if isinstance(category, str) else None
    if normalized_category == "":
        normalized_category = None

    if normalized_category and normalized_category not in ALLOWED_ESCALATION_CATEGORIES:
        allowed = ", ".join(sorted(ALLOWED_ESCALATION_CATEGORIES))
        raise ValueError(f"category must be one of: {allowed}")

    return normalized_limit, normalized_category
