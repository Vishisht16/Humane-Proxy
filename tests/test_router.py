"""Tests for humane_proxy.escalation.router."""

from unittest.mock import patch

from humane_proxy.escalation.router import escalate


class TestEscalation:
    def test_basic_escalation(self):
        result = escalate("esc-sess", 0.95, ["trigger1"], "self_harm")
        assert result["escalated"] is True
        assert result["reason"] == "logged"
        assert result["risk_score"] == 0.95
        assert result["triggers"] == ["trigger1"]
        assert result["category"] == "self_harm"

    def test_category_in_result(self):
        result = escalate("cat-sess", 0.9, ["t"], "criminal_intent")
        assert result["category"] == "criminal_intent"

    def test_defensive_triggers_copy(self):
        original = ["mutable"]
        result = escalate("copy-sess", 0.9, original, "self_harm")
        original.append("sneaky")
        assert "sneaky" not in result["triggers"]

    def test_none_triggers_handled(self):
        result = escalate("none-sess", 0.9, None, "self_harm")  # type: ignore[arg-type]
        assert result["escalated"] is True


class TestRateLimiting:
    def test_rate_limit_blocks_after_max(self):
        sid = "ratelimit-sess"
        for _ in range(3):
            escalate(sid, 0.9, ["t"], "self_harm")
        result = escalate(sid, 0.9, ["t"], "self_harm")
        assert result["escalated"] is False
        assert result["reason"] == "rate_limited"


class TestDbFailure:
    def test_db_failure_graceful(self):
        with patch(
            "humane_proxy.escalation.router.log_escalation",
            side_effect=Exception("DB boom"),
        ):
            result = escalate("fail-sess", 0.99, ["boom"], "self_harm")
            assert result["escalated"] is True
            assert result["reason"] == "logged_with_db_error"
