
"""
Robust configuration validation tests for SafetyPipeline.
 
This module documents the *current* behaviour of SafetyPipeline when it
receives malformed, incomplete, or otherwise invalid configuration values.
 
Purpose
-------
* Prove which invalid-config scenarios the pipeline already survives (PASS).
* Explicitly mark known crashes as expected failures (xfail) so CI stays
  green while the bugs are tracked and fixed in a dedicated follow-up issue.
* Provide a regression guard so that once those bugs are fixed the xfail
  tests are automatically promoted to normal passes.
 
Findings summary
----------------
* PASS  – missing config sections, invalid stage ordering, duplicate stages,
           negative / oversized thresholds, invalid Stage-3 provider strings,
           invalid trajectory config, empty enabled_stages list.
* XFAIL – ``enabled_stages`` set to a plain string → TypeError at init time.
* XFAIL – ``enabled_stages`` set to None → TypeError at init time.
* XFAIL – threshold value set to a non-numeric string → TypeError at
           classify_sync() time.
 
These three xfail cases should be addressed in a follow-up issue by adding
input validation / type coercion inside ``SafetyPipeline.__init__``.
"""
 
from __future__ import annotations
 
import pytest
 
from humane_proxy.classifiers.pipeline import SafetyPipeline
 
 
# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
 
@pytest.fixture()
def base_config() -> dict:
    """Return a minimal, fully-valid pipeline configuration.
 
    Defined locally so this test file has no dependency on any other test
    module.  All tests that need a config start from a fresh copy of this
    fixture value and mutate only the key they are testing.
    """
    return {
        "safety": {"risk_threshold": 0.7, "spike_boost": 0.25},
        "pipeline": {
            "enabled_stages": [1],
            "stage1_ceiling": 0.3,
            "stage2_ceiling": 0.4,
        },
        "stage2": {"model": "test", "safe_threshold": 0.35},
        "stage3": {"provider": "none"},
        "trajectory": {"window_size": 5, "spike_delta": 0.35},
        "privacy": {"store_message_text": False},
    }
 
 
# ---------------------------------------------------------------------------
# Invalid enabled_stages values
# ---------------------------------------------------------------------------
 
class TestInvalidEnabledStages:
    """Tests for non-list / falsy values supplied as ``enabled_stages``.
 
    Current behaviour: passing a plain string or None raises a TypeError
    during pipeline initialisation because the pipeline does no type-checking
    before using the value in membership tests (``2 in self.enabled_stages``).
    These cases are marked xfail and should be fixed in a follow-up issue by
    adding input validation in ``SafetyPipeline.__init__``.
    """
 
    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason=(
            "BUG: enabled_stages='invalid_string' causes TypeError "
            "('in <string>' requires string as left operand, not int). "
            "Pipeline should validate and fall back to [1]."
        ),
    )
    def test_enabled_stages_not_list(self, base_config):
        """A plain string for enabled_stages currently crashes with TypeError.
 
        Expected (ideal): pipeline falls back to ``[1]`` and returns a result.
        Actual (current): ``TypeError`` at ``__init__`` time.
        """
        base_config["pipeline"]["enabled_stages"] = "invalid_string"
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
 
    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason=(
            "BUG: enabled_stages=None causes TypeError "
            "('NoneType' is not iterable). "
            "Pipeline should validate and fall back to [1]."
        ),
    )
    def test_enabled_stages_none(self, base_config):
        """None for enabled_stages currently crashes with TypeError.
 
        Expected (ideal): pipeline falls back to ``[1]`` and returns a result.
        Actual (current): ``TypeError`` at ``__init__`` time.
        """
        base_config["pipeline"]["enabled_stages"] = None
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
 
    def test_enabled_stages_empty_list_behavior(self, base_config):
        """An empty list for enabled_stages is handled without crashing.
 
        With no stages enabled the pipeline should still return a result
        whose category is one of the known safe fallback values.
        """
        base_config["pipeline"]["enabled_stages"] = []
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("hello world", "test")
        assert result.classification.category in {"safe", "unknown"}
        assert result.should_escalate is False
 
 
# ---------------------------------------------------------------------------
# Invalid threshold values
# ---------------------------------------------------------------------------
 
class TestInvalidThresholdValues:
    """Tests for out-of-range or wrongly-typed threshold configuration values.
 
    Numeric edge cases (negative, very large) are already handled gracefully
    by the pipeline.  A non-numeric string type causes a TypeError at
    classify_sync() time and is marked xfail.
    """
 
    def test_negative_thresholds(self, base_config):
        """Negative threshold values do not crash the pipeline.
 
        With a stage1_ceiling of -5.0 every Stage-1 score exceeds the
        ceiling, so the pipeline never early-exits at Stage 1.  Safe input
        should still be classified as safe.
        """
        base_config["pipeline"]["stage1_ceiling"] = -5.0
        base_config["pipeline"]["stage2_ceiling"] = -1.0
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("normal text", "test")
        assert result.classification.category == "safe"
        assert result.should_escalate is False
 
    def test_threshold_too_large(self, base_config):
        """An oversized threshold (999.0) does not suppress self-harm detection.
 
        Self-harm must be caught regardless of ceiling values because the
        pipeline has a dedicated self-harm early-exit path before any ceiling
        comparison takes place.
        """
        base_config["pipeline"]["stage1_ceiling"] = 999.0
        base_config["pipeline"]["stage2_ceiling"] = 999.0
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("I want to kill myself", "test")
        assert result.classification.category == "self_harm"
        assert result.should_escalate is True
 
    @pytest.mark.xfail(
        strict=True,
        raises=TypeError,
        reason=(
            "BUG: stage1_ceiling='not_a_float' causes TypeError "
            "('<=' not supported between float and str) at classify_sync() time. "
            "Pipeline should coerce or reject non-numeric threshold values."
        ),
    )
    def test_threshold_wrong_type(self, base_config):
        """A string value for stage1_ceiling currently crashes with TypeError.
 
        Expected (ideal): pipeline coerces or rejects the value and returns
        a result.
        Actual (current): ``TypeError`` raised inside ``classify_sync``.
        """
        base_config["pipeline"]["stage1_ceiling"] = "not_a_float"
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
 
 
# ---------------------------------------------------------------------------
# Missing config sections
# ---------------------------------------------------------------------------
 
class TestMissingConfigSections:
    """Tests for configs with entire top-level sections removed.
 
    The pipeline uses ``config.get(section, {})`` throughout, so missing
    sections should fall back to defaults without crashing.
    """
 
    def test_missing_pipeline_section(self, base_config):
        """Pipeline operates with default settings when 'pipeline' key absent."""
        base_config.pop("pipeline", None)
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
        assert result.should_escalate is False
 
    def test_missing_safety_section(self, base_config):
        """Pipeline operates with default thresholds when 'safety' key absent."""
        base_config.pop("safety", None)
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
        assert result.should_escalate is False
 
    def test_missing_stage2_config(self, base_config):
        """Pipeline operates correctly when 'stage2' config block is absent."""
        base_config.pop("stage2", None)
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
        assert result.should_escalate is False
 
 
# ---------------------------------------------------------------------------
# Unsupported Stage-3 provider values
# ---------------------------------------------------------------------------
 
class TestUnsupportedStage3Provider:
    """Tests for unknown or null Stage-3 provider identifiers.
 
    The pipeline logs a warning for unknown providers and sets
    ``self._stage3 = None``, so classification should continue via
    earlier stages only.
    """
 
    def test_invalid_stage3_provider_string(self, base_config):
        """An unrecognised provider string disables Stage 3 without crashing."""
        base_config["stage3"]["provider"] = "random_invalid_provider"
        pipeline = SafetyPipeline(base_config)
        assert pipeline._stage3 is None
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
 
    def test_none_provider_behavior(self, base_config):
        """A None provider value disables Stage 3 without crashing."""
        base_config["stage3"]["provider"] = None
        pipeline = SafetyPipeline(base_config)
        assert pipeline._stage3 is None
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
 
 
# ---------------------------------------------------------------------------
# Invalid trajectory configuration
# ---------------------------------------------------------------------------
 
class TestInvalidTrajectoryConfig:
    """Tests for malformed trajectory configuration values.
 
    Trajectory analysis is performed inside ``_finalize`` via an external
    ``analyze`` function.  Missing or wrongly-typed trajectory config should
    not prevent the pipeline from returning a result.
    """
 
    def test_missing_trajectory_section(self, base_config):
        """Pipeline runs without crashing when 'trajectory' config is absent."""
        base_config.pop("trajectory", None)
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
 
    def test_invalid_trajectory_types(self, base_config):
        """A string value for the entire trajectory config does not crash."""
        base_config["trajectory"] = "invalid_string"
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
 
    def test_invalid_window_size(self, base_config):
        """A negative window_size does not crash the pipeline."""
        base_config["trajectory"]["window_size"] = -10
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
 
 
# ---------------------------------------------------------------------------
# Duplicate and out-of-order stage lists
# ---------------------------------------------------------------------------
 
class TestDuplicateAndCorruptStageConfig:
    """Tests for unusual but list-typed enabled_stages values.
 
    The pipeline iterates over enabled_stages without deduplicating, so
    duplicate entries may cause stages to initialise or run more than once.
    These tests verify no crash occurs.
    """
 
    def test_duplicate_enabled_stages(self, base_config):
        """Duplicate stage numbers in enabled_stages do not crash the pipeline."""
        base_config["pipeline"]["enabled_stages"] = [1, 1, 2, 2]
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
 
    def test_invalid_stage_ordering(self, base_config):
        """Out-of-order stage numbers (e.g. [3, 1, 2]) do not crash the pipeline."""
        base_config["pipeline"]["enabled_stages"] = [3, 1, 2]
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("hello", "test")
        assert result.classification.category == "safe"
 
 
# ---------------------------------------------------------------------------
# Regression: self-harm detection must survive bad config
# ---------------------------------------------------------------------------
 
class TestRegressionSafetyBehavior:
    """Regression tests ensuring critical self-harm detection is never silenced.
 
    Even when the pipeline is given a broken or unusual configuration,
    explicit self-harm messages must still be escalated.  This is the
    most important safety invariant in the system.
    """
 
    def test_pipeline_still_detects_self_harm_under_bad_config(self, base_config):
        """Self-harm detection works even with a negative stage1_ceiling.
 
        A stage1_ceiling of -1.0 means no score can satisfy the early-exit
        condition, but the dedicated self-harm early-exit path (which runs
        before any ceiling comparison) must still catch the message.
        """
        base_config["pipeline"]["enabled_stages"] = [1, 2]
        base_config["pipeline"]["stage1_ceiling"] = -1.0
        pipeline = SafetyPipeline(base_config)
        result = pipeline.classify_sync("I want to kill myself", "test-regression")
        assert result.classification.category == "self_harm"
        assert result.classification.score == 1.0
        assert result.should_escalate is True
        assert result.should_block is True
 