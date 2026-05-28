
"""
Robust configuration validation tests for SafetyPipeline.

These tests ensure:
- pipeline does not crash on malformed configs
- system degrades safely instead of throwing runtime TypeErrors
- self-harm detection always survives bad config
"""

from humane_proxy.classifiers.pipeline import SafetyPipeline
from .test_pipeline import _base_config


class TestInvalidEnabledStages:

    def test_enabled_stages_not_list(self):
        config = _base_config()
        config["pipeline"]["enabled_stages"] = "invalid_string"

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None

    def test_enabled_stages_none(self):
        config = _base_config()
        config["pipeline"]["enabled_stages"] = None

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None

    def test_enabled_stages_empty_list_behavior(self):
        config = _base_config()
        config["pipeline"]["enabled_stages"] = []

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello world", "test")

        assert result is not None
        assert result.classification.category in {"safe", "unknown"}


class TestInvalidThresholdValues:

    def test_negative_thresholds(self):
        config = _base_config()
        config["pipeline"]["stage1_ceiling"] = -5.0
        config["pipeline"]["stage2_ceiling"] = -1.0

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("normal text", "test")

        assert result is not None
        assert result.classification.category in {"safe", "unknown"}

    def test_threshold_too_large(self):
        config = _base_config()
        config["pipeline"]["stage1_ceiling"] = 999.0
        config["pipeline"]["stage2_ceiling"] = 999.0

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync(
            "I want to kill myself",
            "test"
        )

        assert result.classification.category == "self_harm"

    def test_threshold_wrong_type(self):
        config = _base_config()
        config["pipeline"]["stage1_ceiling"] = "not_a_float"

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None


class TestMissingConfigSections:

    def test_missing_pipeline_section(self):
        config = _base_config()
        config.pop("pipeline", None)

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None

    def test_missing_safety_section(self):
        config = _base_config()
        config.pop("safety", None)

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None

    def test_missing_stage2_config(self):
        config = _base_config()
        config.pop("stage2", None)

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None


class TestUnsupportedStage3Provider:

    def test_invalid_stage3_provider_string(self):
        config = _base_config()
        config["stage3"]["provider"] = "random_invalid_provider"

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None

    def test_none_provider_behavior(self):
        config = _base_config()
        config["stage3"]["provider"] = None

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None


class TestInvalidTrajectoryConfig:

    def test_missing_trajectory_section(self):
        config = _base_config()
        config.pop("trajectory", None)

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None

    def test_invalid_trajectory_types(self):
        config = _base_config()
        config["trajectory"] = "invalid_string"

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None

    def test_invalid_window_size(self):
        config = _base_config()
        config["trajectory"]["window_size"] = -10

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None


class TestDuplicateAndCorruptStageConfig:

    def test_duplicate_enabled_stages(self):
        config = _base_config()
        config["pipeline"]["enabled_stages"] = [1, 1, 2, 2]

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None

    def test_invalid_stage_ordering(self):
        config = _base_config()
        config["pipeline"]["enabled_stages"] = [3, 1, 2]

        pipeline = SafetyPipeline(config)
        result = pipeline.classify_sync("hello", "test")

        assert result is not None


class TestRegressionSafetyBehavior:

    def test_pipeline_still_detects_self_harm_under_bad_config(self):
        config = _base_config()
        config["pipeline"]["enabled_stages"] = [1, 2]
        config["pipeline"]["stage1_ceiling"] = -1.0

        pipeline = SafetyPipeline(config)

        result = pipeline.classify_sync(
            "I want to kill myself",
            "test-regression"
        )

        assert result.classification.category == "self_harm"
        assert result.should_escalate is True

