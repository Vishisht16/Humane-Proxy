"""Tests for humane_proxy.risk.trajectory (Phase 2 enhancements + time-decay)."""

import math
import time
from collections import deque
from unittest.mock import patch

from humane_proxy.risk.trajectory import (
    analyze,
    detect_spike,
    session_history,
    _category_history,
    _weighted_mean,
)
from humane_proxy.classifiers.models import TrajectoryResult


class TestSpikeDetection:
    def test_first_message_no_spike(self):
        sid = "traj-first-v3"
        assert detect_spike(sid, 0.0) is False

    def test_stable_low_scores_no_spike(self):
        sid = "traj-stable-v3"
        for _ in range(5):
            assert detect_spike(sid, 0.1) is False

    def test_sudden_spike_detected(self):
        sid = "traj-spike-v3"
        for _ in range(3):
            detect_spike(sid, 0.1)
        assert detect_spike(sid, 0.9) is True

    def test_gradual_increase_no_spike(self):
        sid = "traj-gradual-v3"
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            result = detect_spike(sid, s)
        assert result is False

    def test_spike_after_zeros(self):
        sid = "traj-zero-spike-v3"
        for _ in range(5):
            detect_spike(sid, 0.0)
        assert detect_spike(sid, 0.5) is True


class TestTimeDecay:
    """Test exponential decay mechanics."""

    def test_recent_scores_full_weight(self):
        """Scores from seconds ago should be nearly full weight."""
        now = time.time()
        history = deque([
            (0.5, now - 1),   # 1 second ago
            (0.5, now - 2),   # 2 seconds ago
        ])
        avg = _weighted_mean(history, now)
        # Should be very close to 0.5 (negligible decay within seconds).
        assert abs(avg - 0.5) < 0.01

    def test_old_scores_decay(self):
        """Scores from 48 hours ago should carry ~25 % weight (two half-lives)."""
        now = time.time()
        two_days_ago = now - (48 * 3600)
        history = deque([
            (0.8, two_days_ago),  # old high score
            (0.2, now - 1),       # recent low score
        ])
        avg = _weighted_mean(history, now)
        # Old score decayed by ~75 %, recent at ~100 %.
        # Weighted toward 0.2 rather than the plain mean of 0.5.
        assert avg < 0.40

    def test_decay_disabled_when_zero(self):
        """When λ = 0, _weighted_mean collapses to plain mean."""
        now = time.time()
        long_ago = now - (7 * 24 * 3600)  # a week ago
        history = deque([
            (0.8, long_ago),
            (0.2, now),
        ])
        with patch("humane_proxy.risk.trajectory._DECAY_LAMBDA", 0.0):
            avg = _weighted_mean(history, now)
        assert abs(avg - 0.5) < 0.01  # plain average of 0.8 and 0.2

    def test_spike_after_long_absence(self):
        """Returning after days with a moderate score should spike
        because decayed history → near-zero baseline."""
        sid = "traj-decay-spike"
        now = time.time()
        three_days = 3 * 24 * 3600
        # Manually insert old history with timestamps from 3 days ago.
        session_history[sid] = deque(maxlen=5)
        session_history[sid].append((0.1, now - three_days))
        session_history[sid].append((0.1, now - three_days + 1))
        session_history[sid].append((0.1, now - three_days + 2))

        # A moderate score of 0.5 should spike against the decayed baseline.
        result = detect_spike(sid, 0.5)
        assert result is True


class TestAnalyze:
    def test_returns_trajectory_result(self):
        result = analyze("analyze-test-v3", 0.5, "safe")
        assert isinstance(result, TrajectoryResult)

    def test_first_message_stable(self):
        result = analyze("analyze-first-v3", 0.1, "safe")
        assert result.spike_detected is False
        assert result.trend == "stable"
        assert result.message_count == 1
        assert result.category_counts == {"safe": 1}

    def test_spike_detected(self):
        sid = "analyze-spike-v3"
        for _ in range(3):
            analyze(sid, 0.1, "safe")
        result = analyze(sid, 0.9, "self_harm")
        assert result.spike_detected is True

    def test_category_tracking(self):
        sid = "analyze-cats-v3"
        analyze(sid, 0.0, "safe")
        analyze(sid, 0.0, "safe")
        analyze(sid, 0.8, "self_harm")
        result = analyze(sid, 0.0, "safe")
        assert result.category_counts["safe"] == 3
        assert result.category_counts["self_harm"] == 1

    def test_window_scores_are_raw(self):
        """window_scores should return raw floats, not tuples."""
        sid = "analyze-raw-v3"
        analyze(sid, 0.3, "safe")
        result = analyze(sid, 0.7, "criminal_intent")
        for s in result.window_scores:
            assert isinstance(s, float)


class TestTrendDetection:
    def test_escalating_trend(self):
        sid = "trend-escalate-v3"
        for s in [0.1, 0.1, 0.5, 0.6]:
            result = analyze(sid, s, "safe")
        # first half avg: 0.1, second half avg: 0.55 → delta 0.45 > 0.15
        assert result.trend == "escalating"

    def test_declining_trend(self):
        sid = "trend-decline-v3"
        for s in [0.8, 0.7, 0.2, 0.1]:
            result = analyze(sid, s, "safe")
        # first half avg: 0.75, second half avg: 0.15 → delta -0.6 < -0.15
        assert result.trend == "declining"

    def test_stable_trend(self):
        sid = "trend-stable-v3"
        for s in [0.3, 0.35, 0.3, 0.35]:
            result = analyze(sid, s, "safe")
        assert result.trend == "stable"

    def test_not_enough_for_trend(self):
        sid = "trend-short-v3"
        result = analyze(sid, 0.5, "safe")
        assert result.trend == "stable"


class TestMemoryEviction:
    def test_session_cap(self):
        for i in range(1001):
            session_history[f"evict-test-v3-{i}"] = deque(
                [(0.1, time.time())], maxlen=5
            )

        detect_spike("evict-new-v3", 0.5)
        assert len(session_history) <= 1001

    def test_eviction_preserves_new_sessions(self):
        sid = "evict-survivor-v3"
        detect_spike(sid, 0.3)
        assert sid in session_history
