"""Tests for humane_proxy.escalation.local_db."""

from humane_proxy.escalation.local_db import (
    check_rate_limit,
    init_db,
    log_escalation,
)


class TestInitDb:
    def test_idempotent_init(self):
        """Calling init_db() twice should not raise."""
        init_db()
        init_db()


class TestLogEscalation:
    def test_basic_log(self):
        log_escalation("test-sess", 0.9, ["trigger1"], "self_harm")
        assert check_rate_limit("test-sess") is True  # 1/3 used

    def test_none_triggers_handled(self):
        """triggers=None should not crash."""
        log_escalation("test-none", 0.5, None, "criminal_intent")  # type: ignore[arg-type]

    def test_empty_triggers(self):
        log_escalation("test-empty", 0.5, [], "safe")

    def test_category_stored(self):
        """Category parameter should be accepted without error."""
        log_escalation("test-cat", 0.9, ["t"], "self_harm")
        log_escalation("test-cat2", 0.8, ["t"], "criminal_intent")


class TestRateLimit:
    def test_within_limit(self):
        for _ in range(3):
            log_escalation("rate-sess", 0.9, ["t"], "self_harm")
        assert check_rate_limit("rate-sess") is False  # 3/3 exhausted

    def test_new_session_allowed(self):
        assert check_rate_limit("never-seen") is True

    def test_partial_usage(self):
        log_escalation("partial-sess", 0.9, ["t"], "self_harm")
        log_escalation("partial-sess", 0.9, ["t"], "self_harm")
        assert check_rate_limit("partial-sess") is True  # 2/3 used
