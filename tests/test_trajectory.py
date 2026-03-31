"""Tests for humane_proxy.risk.trajectory."""

from humane_proxy.risk.trajectory import detect_spike, session_history


class TestSpikeDetection:
    def test_first_message_no_spike(self):
        sid = "traj-first"
        assert detect_spike(sid, 0.0) is False

    def test_stable_low_scores_no_spike(self):
        sid = "traj-stable"
        for _ in range(5):
            assert detect_spike(sid, 0.1) is False

    def test_sudden_spike_detected(self):
        sid = "traj-spike"
        for _ in range(3):
            detect_spike(sid, 0.1)
        assert detect_spike(sid, 0.9) is True

    def test_gradual_increase_no_spike(self):
        sid = "traj-gradual"
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            result = detect_spike(sid, s)
        # 0.5 - mean(0.1,0.2,0.3,0.4) = 0.5 - 0.25 = 0.25 < 0.35
        assert result is False

    def test_spike_after_zeros(self):
        sid = "traj-zero-spike"
        for _ in range(5):
            detect_spike(sid, 0.0)
        assert detect_spike(sid, 0.5) is True  # delta = 0.5 > 0.35


class TestMemoryEviction:
    def test_session_cap(self):
        """Ensure sessions are evicted when exceeding 1000."""
        for i in range(1001):
            session_history[f"evict-test-{i}"] = __import__("collections").deque([0.1], maxlen=5)

        # Trigger eviction by adding a new session.
        detect_spike("evict-new", 0.5)

        # Should have evicted ~10% of the oldest, so well under 1001 now.
        assert len(session_history) <= 1001

    def test_eviction_preserves_new_sessions(self):
        """Newest sessions should survive eviction."""
        sid = "evict-survivor"
        detect_spike(sid, 0.3)
        assert sid in session_history
