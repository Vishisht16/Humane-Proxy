"""Tier-1 heuristic classifier — keyword + pattern-based safety check.

Detects self-harm ideation and criminal intent through:
1. Word-boundary keyword matching for known harmful phrases
2. Intent-pattern regex matching for grammatical intent expressions
3. Context reducers to minimize false positives (e.g. "I want to die laughing")

Categories: ``"self_harm"``, ``"criminal_intent"``, ``"safe"``

Scoring is intentionally tuned for **high recall over precision** for
self-harm: a single self-harm keyword match results in a score of 1.0
(critical override).  For criminal intent, scores accumulate from
keyword and pattern matches.
"""

from __future__ import annotations

import re

from humane_proxy import load_config

_CFG: dict = load_config().get("heuristics", {})

# ---------------------------------------------------------------------------
# Self-harm keywords
# ---------------------------------------------------------------------------
_SELF_HARM_KEYWORDS: list[str] = [
    kw.lower() for kw in _CFG.get("self_harm_keywords", [])
]
_SELF_HARM_KEYWORD_SCORE: float = _CFG.get("self_harm_keyword_score", 0.7)

_SELF_HARM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (kw, re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)", re.I))
    for kw in _SELF_HARM_KEYWORDS
]

# ---------------------------------------------------------------------------
# Criminal intent keywords
# ---------------------------------------------------------------------------
_CRIMINAL_KEYWORDS: list[str] = [
    kw.lower() for kw in _CFG.get("criminal_keywords", [])
]
_CRIMINAL_KEYWORD_SCORE: float = _CFG.get("criminal_keyword_score", 0.6)

_CRIMINAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (kw, re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)", re.I))
    for kw in _CRIMINAL_KEYWORDS
]

# ---------------------------------------------------------------------------
# Intent patterns — regex patterns that detect grammatical intent structures
# ---------------------------------------------------------------------------
_INTENT_PATTERN_SCORE: float = _CFG.get("intent_pattern_score", 0.7)

# Each entry: (name, category, compiled_pattern)
_INTENT_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "first_person_self_harm",
        "self_harm",
        re.compile(
            r"i\s+(want|need|plan|am going|have decided|intend)\s+to\s+"
            r"(kill\s+myself|hurt\s+myself|end\s+my\s+life|harm\s+myself)",
            re.I,
        ),
    ),
    (
        "self_annihilation",
        "self_harm",
        re.compile(r"(end|take).{0,20}(my life|myself|it all)", re.I),
    ),
    (
        "method_seeking",
        "self_harm",
        re.compile(
            r"(painless|easy|quick|best).{0,20}(way|method|how).{0,20}"
            r"(die|kill myself|end it|end my life)",
            re.I,
        ),
    ),
    (
        "first_person_harm_others",
        "criminal_intent",
        re.compile(
            r"i\s+(want|need|plan|am going|intend)\s+to\s+"
            r"(kill|hurt|harm|poison|attack|murder|kidnap|assault)\s+(?!myself)",
            re.I,
        ),
    ),
    (
        "weapons_synthesis",
        "criminal_intent",
        re.compile(
            r"how\s+(do\s+i|to|can\s+i).{0,30}"
            r"(make|build|synthesize|create|manufacture).{0,20}"
            r"(bomb|weapon|explosive|poison|drug|meth|ricin|sarin|fentanyl)",
            re.I,
        ),
    ),
    (
        "acquisition_criminal",
        "criminal_intent",
        re.compile(
            r"where\s+(can\s+i|do\s+i|to).{0,20}"
            r"(buy|get|acquire|find|obtain).{0,20}"
            r"(gun|weapon|drugs|fentanyl|illegal)",
            re.I,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Context reducers — phrases that indicate non-harmful context
# ---------------------------------------------------------------------------
_CONTEXT_REDUCERS: list[str] = [
    r.lower() for r in _CFG.get("context_reducers", [])
]

_CONTEXT_REDUCER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(rf"(?<!\w){re.escape(r)}(?!\w)", re.I)
    for r in _CONTEXT_REDUCERS
]


def classify(text: str) -> tuple[str, float, list[str]]:
    """Run heuristic checks on *text* and return ``(category, score, triggers)``.

    Parameters
    ----------
    text:
        The raw user message to evaluate.

    Returns
    -------
    tuple[str, float, list[str]]
        A category string (``"self_harm"``, ``"criminal_intent"``, or
        ``"safe"``), a risk score clamped to ``[0.0, 1.0]``, and a
        **deduplicated** list of human-readable trigger descriptions.
    """
    # Guard: empty / whitespace-only input.
    if not text or not text.strip():
        return "safe", 0.0, []

    # Normalize whitespace: collapse runs of spaces/tabs/newlines into a
    # single space.  Defeats evasion tricks like embedded newlines.
    text = re.sub(r"\s+", " ", text).strip()

    score: float = 0.0
    category: str = "safe"
    seen: set[str] = set()       # O(1) dedup
    triggers: list[str] = []     # ordered output
    trigger_count: int = 0       # count of hard triggers for reducer gating

    # --- Self-harm keyword scan ---
    for keyword, pattern in _SELF_HARM_PATTERNS:
        if pattern.search(text):
            tag = f"self_harm_keyword:{keyword}"
            if tag not in seen:
                seen.add(tag)
                triggers.append(tag)
                score += _SELF_HARM_KEYWORD_SCORE
                trigger_count += 1
                category = "self_harm"
            if score >= 1.0:
                break

    # --- Criminal keyword scan ---
    for keyword, pattern in _CRIMINAL_PATTERNS:
        if pattern.search(text):
            tag = f"criminal_keyword:{keyword}"
            if tag not in seen:
                seen.add(tag)
                triggers.append(tag)
                score += _CRIMINAL_KEYWORD_SCORE
                trigger_count += 1
                if category != "self_harm":
                    category = "criminal_intent"
            if score >= 1.0:
                break

    # --- Intent pattern scan ---
    for name, pat_category, pattern in _INTENT_PATTERNS:
        if pattern.search(text):
            tag = f"intent_pattern:{name}"
            if tag not in seen:
                seen.add(tag)
                triggers.append(tag)
                score += _INTENT_PATTERN_SCORE
                trigger_count += 1
                # Self-harm patterns upgrade the category.
                if pat_category == "self_harm":
                    category = "self_harm"
                elif category == "safe":
                    category = pat_category
            if score >= 1.0:
                break

    # --- Context reducer check ---
    # Only activate when there's a single hard trigger — multiple triggers
    # indicate genuine concern that shouldn't be neutralized by stray context.
    if score > 0.0 and trigger_count == 1:
        for reducer_pattern in _CONTEXT_REDUCER_PATTERNS:
            if reducer_pattern.search(text):
                score *= 0.1
                triggers.append("context_reduced")
                category = "safe"
                break

    # --- Self-harm critical override ---
    # If category is still self_harm after context reduction, force score
    # to 1.0.  Every self-harm signal matters.
    if category == "self_harm":
        score = 1.0

    score = min(score, 1.0)

    return category, score, triggers
