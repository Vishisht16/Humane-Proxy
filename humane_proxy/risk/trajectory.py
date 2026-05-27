"""RiskTracker — trajectory-based spike detection and trend analysis.

Supports **exponential time-decay** so that stale scores from hours or
days ago naturally fade toward zero, giving returning users a fair
baseline while still catching rapid within-session escalation.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any

from humane_proxy.config import get_config
from humane_proxy.classifiers.models import TrajectoryResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CFG: dict = get_config().get("trajectory", {})
_WINDOW_SIZE: int = _CFG.get("window_size", 5)
_SPIKE_DELTA: float = _CFG.get("spike_delta", 0.35)

# Decay half-life in hours.  After this many hours a historical score
# contributes only 50 % of its original weight to the rolling baseline.
# Set to 0 or negative to disable decay entirely.
_DECAY_HALF_LIFE_HOURS: float = _CFG.get("decay_half_life_hours", 24.0)

# Precompute lambda: λ = ln(2) / half_life.
_DECAY_LAMBDA: float = (
    math.log(2) / (_DECAY_HALF_LIFE_HOURS * 3600)
    if _DECAY_HALF_LIFE_HOURS > 0
    else 0.0
)

# Maximum distinct sessions to track before eviction (memory-leak prevention).
_MAX_SESSIONS: int = 1000

# ---------------------------------------------------------------------------
# In-memory session stores
# ---------------------------------------------------------------------------
# Each entry is (score, timestamp_seconds).
session_history: dict[str, deque[tuple[float, float]]] = {}
_category_history: dict[str, deque[str]] = {}
_redis_trajectory_store = None
_redis_trajectory_store_key: tuple[str, str, str] | None = None


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


# ---------------------------------------------------------------------------
# Decay-weighted mean
# ---------------------------------------------------------------------------

def _weighted_mean(history: deque[tuple[float, float]], now: float) -> float:
    """Compute the exponentially time-decayed weighted mean of *history*.

    Each entry ``(score, ts)`` is weighted by ``e^{-λ(now-ts)}``.
    When decay is disabled (λ = 0), this collapses to a plain mean.

    Returns 0.0 when *history* is empty (should never happen in practice
    because callers gate on ``len(history) == 0`` first).
    """
    if _DECAY_LAMBDA == 0.0:
        # Fast path: decay disabled — plain mean.
        return sum(s for s, _ in history) / len(history) if history else 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for score, ts in history:
        dt = now - ts  # seconds elapsed
        w = math.exp(-_DECAY_LAMBDA * dt)
        weighted_sum += score * w
        total_weight += w

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def _current_backend() -> str:
    return get_config().get("storage", {}).get("backend", "sqlite")


def _get_redis_trajectory_store():
    global _redis_trajectory_store, _redis_trajectory_store_key
    config = get_config()
    redis_cfg = config.get("storage", {}).get("redis", {})
    fingerprint = (
        config.get("storage", {}).get("backend", "sqlite"),
        redis_cfg.get("url", "redis://localhost:6379/0"),
        redis_cfg.get("key_prefix", "humane_proxy:"),
    )
    if _redis_trajectory_store is None or _redis_trajectory_store_key != fingerprint:
        from humane_proxy.risk.redis_trajectory import RedisTrajectoryStore

        _redis_trajectory_store = RedisTrajectoryStore(config, window_size=_WINDOW_SIZE)
        _redis_trajectory_store_key = fingerprint
    return _redis_trajectory_store


def _window_scores(window: list[dict[str, Any]]) -> list[float]:
    return [float(entry["score"]) for entry in window]


def _window_categories(window: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in window:
        category = entry.get("category") or "safe"
        counts[category] = counts.get(category, 0) + 1
    return counts


def _trend_from_scores(scores: list[float]) -> str:
    if len(scores) < 4:
        return "stable"

    mid = len(scores) // 2
    first_half_avg = sum(scores[:mid]) / mid
    second_half_avg = sum(scores[mid:]) / (len(scores) - mid)
    trend_delta = second_half_avg - first_half_avg
    if trend_delta > 0.15:
        return "escalating"
    if trend_delta < -0.15:
        return "declining"
    return "stable"


def _spike_from_scores(scores: list[float], timestamps: list[float]) -> bool:
    if not scores:
        return False
    if len(scores) == 1:
        return False

    history = deque(zip(scores[:-1], timestamps[:-1]))
    avg = _weighted_mean(history, timestamps[-1])
    return scores[-1] - avg > _SPIKE_DELTA


def _append_local(session_id: str, current_score: float, category: str | None = None) -> list[dict[str, Any]]:
    now = time.time()

    if len(session_history) > _MAX_SESSIONS and session_id not in session_history:
        _evict_oldest_sessions()

    if session_id not in session_history:
        session_history[session_id] = deque(maxlen=_WINDOW_SIZE)

    history = session_history[session_id]
    history.append((current_score, now))

    if category is not None:
        if session_id not in _category_history:
            _category_history[session_id] = deque(maxlen=_WINDOW_SIZE)
        _category_history[session_id].append(category)

    window = [
        {"score": score, "timestamp": ts, "category": None}
        for score, ts in history
    ]
    if category is not None and session_id in _category_history:
        categories = list(_category_history[session_id])
        for index, value in enumerate(categories[-len(window):]):
            window[index]["category"] = value
    return window


def _build_result(session_id: str, window: list[dict[str, Any]]) -> tuple[bool, list[float], dict[str, int], str]:
    scores = _window_scores(window)
    timestamps = [float(entry["timestamp"]) for entry in window]
    spike = _spike_from_scores(scores, timestamps)
    return spike, scores, _window_categories(window), _trend_from_scores(scores)


def detect_spike(session_id: str, current_score: float) -> bool:
    """Return ``True`` if the current score spikes above the recent average.

    Math
    ----
    ``delta = current_score - weighted_mean(last N scores)``

    Historical scores are weighted by ``e^{-λ Δt}`` (exponential
    time-decay with a configurable half-life, default 24 h).

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
    if _current_backend() == "redis":
        window = _get_redis_trajectory_store().append(session_id, current_score, None)
        scores = _window_scores(window)
        timestamps = [float(entry["timestamp"]) for entry in window]
        return _spike_from_scores(scores, timestamps)

    window = _append_local(session_id, current_score)
    spike, _, _, _ = _build_result(session_id, window)
    return spike


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
    if _current_backend() == "redis":
        window = _get_redis_trajectory_store().append(session_id, score, category)
    else:
        window = _append_local(session_id, score, category)

    spike, scores, cat_counts, trend = _build_result(session_id, window)

    return TrajectoryResult(
        spike_detected=spike,
        trend=trend,
        window_scores=scores,
        category_counts=cat_counts,
        message_count=len(scores),
    )
