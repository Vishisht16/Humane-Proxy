"""Tests for humane_proxy.risk.trajectory (Phase 2 enhancements)."""

from humane_proxy.risk.trajectory import (
    analyze,
    detect_spike,
    session_history,
    _category_history,
)
from humane_proxy.classifiers.models import TrajectoryResult


class TestSpikeDetection:
    def test_first_message_no_spike(self):
        sid = "traj-first-v2"
        assert detect_spike(sid, 0.0) is False

    def test_stable_low_scores_no_spike(self):
        sid = "traj-stable-v2"
        for _ in range(5):
            assert detect_spike(sid, 0.1) is False

    def test_sudden_spike_detected(self):
        sid = "traj-spike-v2"
        for _ in range(3):
            detect_spike(sid, 0.1)
        assert detect_spike(sid, 0.9) is True

    def test_gradual_increase_no_spike(self):
        sid = "traj-gradual-v2"
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            result = detect_spike(sid, s)
        assert result is False

    def test_spike_after_zeros(self):
        sid = "traj-zero-spike-v2"
        for _ in range(5):
            detect_spike(sid, 0.0)
        assert detect_spike(sid, 0.5) is True


class TestAnalyze:
    def test_returns_trajectory_result(self):
        result = analyze("analyze-test-1", 0.5, "safe")
        assert isinstance(result, TrajectoryResult)

    def test_first_message_stable(self):
        result = analyze("analyze-first", 0.1, "safe")
        assert result.spike_detected is False
        assert result.trend == "stable"
        assert result.message_count == 1
        assert result.category_counts == {"safe": 1}

    def test_spike_detected(self):
        sid = "analyze-spike"
        for _ in range(3):
            analyze(sid, 0.1, "safe")
        result = analyze(sid, 0.9, "self_harm")
        assert result.spike_detected is True

    def test_category_tracking(self):
        sid = "analyze-cats"
        analyze(sid, 0.0, "safe")
        analyze(sid, 0.0, "safe")
        analyze(sid, 0.8, "self_harm")
        result = analyze(sid, 0.0, "safe")
        assert result.category_counts["safe"] == 3
        assert result.category_counts["self_harm"] == 1


class TestTrendDetection:
    def test_escalating_trend(self):
        sid = "trend-escalate"
        for s in [0.1, 0.1, 0.5, 0.6]:
            result = analyze(sid, s, "safe")
        # first half avg: 0.1, second half avg: 0.55 → delta 0.45 > 0.15
        assert result.trend == "escalating"

    def test_declining_trend(self):
        sid = "trend-decline"
        for s in [0.8, 0.7, 0.2, 0.1]:
            result = analyze(sid, s, "safe")
        # first half avg: 0.75, second half avg: 0.15 → delta -0.6 < -0.15
        assert result.trend == "declining"

    def test_stable_trend(self):
        sid = "trend-stable"
        for s in [0.3, 0.35, 0.3, 0.35]:
            result = analyze(sid, s, "safe")
        assert result.trend == "stable"

    def test_not_enough_for_trend(self):
        sid = "trend-short"
        result = analyze(sid, 0.5, "safe")
        assert result.trend == "stable"


class TestMemoryEviction:
    def test_session_cap(self):
        for i in range(1001):
            session_history[f"evict-test-v2-{i}"] = __import__("collections").deque([0.1], maxlen=5)

        detect_spike("evict-new-v2", 0.5)
        assert len(session_history) <= 1001

    def test_eviction_preserves_new_sessions(self):
        sid = "evict-survivor-v2"
        detect_spike(sid, 0.3)
        assert sid in session_history
