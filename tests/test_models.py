"""Tests for humane_proxy.classifiers.models."""

from humane_proxy.classifiers.models import (
    ClassificationResult,
    PipelineResult,
    TrajectoryResult,
)


class TestClassificationResult:
    def test_defaults(self):
        r = ClassificationResult()
        assert r.category == "safe"
        assert r.score == 0.0
        assert r.triggers == []
        assert r.stage == 1
        assert r.reasoning is None

    def test_custom_values(self):
        r = ClassificationResult(
            category="self_harm", score=1.0,
            triggers=["t1"], stage=3, reasoning="test"
        )
        assert r.category == "self_harm"
        assert r.score == 1.0
        assert r.triggers == ["t1"]
        assert r.stage == 3
        assert r.reasoning == "test"


class TestTrajectoryResult:
    def test_defaults(self):
        t = TrajectoryResult()
        assert t.spike_detected is False
        assert t.trend == "stable"
        assert t.window_scores == []
        assert t.category_counts == {}
        assert t.message_count == 0


class TestPipelineResult:
    def test_defaults(self):
        p = PipelineResult()
        assert p.should_escalate is False
        assert p.should_block is False
        assert p.message_hash is None

    def test_to_dict_safe(self):
        p = PipelineResult()
        d = p.to_dict()
        assert d["safe"] is True
        assert d["category"] == "safe"
        assert d["score"] == 0.0
        assert d["triggers"] == []
        assert d["stage_reached"] == 1

    def test_to_dict_flagged(self):
        p = PipelineResult(
            classification=ClassificationResult(
                category="self_harm", score=1.0,
                triggers=["t1"], stage=2, reasoning="reason"
            ),
            trajectory=TrajectoryResult(
                spike_detected=True, trend="escalating", message_count=5
            ),
            should_escalate=True,
            should_block=True,
            message_hash="abc123",
        )
        d = p.to_dict()
        assert d["safe"] is False
        assert d["category"] == "self_harm"
        assert d["score"] == 1.0
        assert d["stage_reached"] == 2
        assert d["reasoning"] == "reason"
        assert d["trajectory"]["spike_detected"] is True
        assert d["trajectory"]["trend"] == "escalating"
        assert d["message_hash"] == "abc123"

    def test_to_dict_no_reasoning(self):
        p = PipelineResult()
        d = p.to_dict()
        assert "reasoning" not in d

    def test_to_dict_no_trajectory(self):
        p = PipelineResult()
        d = p.to_dict()
        assert "trajectory" not in d
