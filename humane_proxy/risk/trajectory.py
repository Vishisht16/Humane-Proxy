"""RiskTracker — trajectory-based spike detection for session risk scores."""

from __future__ import annotations

from collections import deque
from statistics import mean

from humane_proxy import load_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CFG: dict = load_config().get("trajectory", {})
_WINDOW_SIZE: int = _CFG.get("window_size", 5)
_SPIKE_DELTA: float = _CFG.get("spike_delta", 0.35)

# Maximum distinct sessions to track before eviction (memory-leak prevention).
_MAX_SESSIONS: int = 1000

# ---------------------------------------------------------------------------
# In-memory session score history
# ---------------------------------------------------------------------------
session_history: dict[str, deque[float]] = {}


def _evict_oldest_sessions() -> None:
    """Pop roughly 10 % of sessions (FIFO order) when we exceed the cap.

    ``dict`` in CPython 3.7+ preserves insertion order, so popping the
    first *n* keys approximates LRU well enough for v0.1.
    """
    evict_count = max(1, len(session_history) // 10)
    for _ in range(evict_count):
        oldest_key = next(iter(session_history))
        del session_history[oldest_key]


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
