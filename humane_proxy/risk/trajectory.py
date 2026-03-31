"""RiskTracker — trajectory-based spike detection and trend analysis."""

from __future__ import annotations

from collections import deque
from statistics import mean

from humane_proxy import load_config
from humane_proxy.classifiers.models import TrajectoryResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CFG: dict = load_config().get("trajectory", {})
_WINDOW_SIZE: int = _CFG.get("window_size", 5)
_SPIKE_DELTA: float = _CFG.get("spike_delta", 0.35)

# Maximum distinct sessions to track before eviction (memory-leak prevention).
_MAX_SESSIONS: int = 1000

# ---------------------------------------------------------------------------
# In-memory session stores
# ---------------------------------------------------------------------------
session_history: dict[str, deque[float]] = {}
_category_history: dict[str, deque[str]] = {}


def _evict_oldest_sessions() -> None:
    """Pop roughly 10 % of sessions (FIFO order) when we exceed the cap.

    ``dict`` in CPython 3.7+ preserves insertion order, so popping the
    first *n* keys approximates LRU well enough for v0.1.
    """
    evict_count = max(1, len(session_history) // 10)
    for _ in range(evict_count):
        oldest_key = next(iter(session_history))
        del session_history[oldest_key]
        _category_history.pop(oldest_key, None)


def detect_spike(session_id: str, current_score: float) -> bool:
    """Return ``True`` if the current score spikes above the recent average.

    Math
    ----
    ``delta = current_score - mean(last N scores)``

    If ``delta > _SPIKE_DELTA`` (default **0.35**), the interaction is
    considered a behavioural spike.

    Parameters
    ----------
    session_id:
        The session / user identifier.
    current_score:
        The heuristic risk score for the current message.

    Returns
    -------
    bool
        Whether a spike was detected.
    """
    # --- memory-leak guard ---
    if len(session_history) > _MAX_SESSIONS and session_id not in session_history:
        _evict_oldest_sessions()

    # Initialise the deque on first encounter.
    if session_id not in session_history:
        session_history[session_id] = deque(maxlen=_WINDOW_SIZE)

    history = session_history[session_id]

    # Not enough history to calculate a meaningful delta yet.
    if len(history) == 0:
        history.append(current_score)
        return False

    avg = mean(history)
    delta = current_score - avg

    # Always record *after* computing the delta so the current score
    # doesn't influence the baseline it's compared against.
    history.append(current_score)

    return delta > _SPIKE_DELTA


# ---------------------------------------------------------------------------
# Enhanced trajectory analysis (Phase 2)
# ---------------------------------------------------------------------------

def analyze(
    session_id: str,
    score: float,
    category: str = "safe",
) -> TrajectoryResult:
    """Record score + category, run spike detection, and compute trend.

    This is the preferred entry point for the pipeline.  It calls
    :func:`detect_spike` internally, so callers should use **either**
    ``analyze()`` **or** ``detect_spike()`` for a given session — never both.

    Parameters
    ----------
    session_id:
        The session / user identifier.
    score:
        The risk score for the current message.
    category:
        The detected category for the current message.

    Returns
    -------
    TrajectoryResult
        Rich trajectory analysis including spike detection, trend, and
        category distribution.
    """
    # Run spike detection (this also appends the score to session_history).
    spike = detect_spike(session_id, score)

    # Track category history.
    if session_id not in _category_history:
        _category_history[session_id] = deque(maxlen=_WINDOW_SIZE)
    _category_history[session_id].append(category)

    # Get current window.
    history = session_history.get(session_id, deque())
    scores = list(history)

    # Trend detection: compare first half vs second half of the window.
    trend = "stable"
    if len(scores) >= 4:
        mid = len(scores) // 2
        first_half_avg = mean(scores[:mid])
        second_half_avg = mean(scores[mid:])
        trend_delta = second_half_avg - first_half_avg
        if trend_delta > 0.15:
            trend = "escalating"
        elif trend_delta < -0.15:
            trend = "declining"

    # Category distribution.
    cat_counts: dict[str, int] = {}
    for c in _category_history.get(session_id, []):
        cat_counts[c] = cat_counts.get(c, 0) + 1

    return TrajectoryResult(
        spike_detected=spike,
        trend=trend,
        window_scores=scores,
        category_counts=cat_counts,
        message_count=len(scores),
    )
