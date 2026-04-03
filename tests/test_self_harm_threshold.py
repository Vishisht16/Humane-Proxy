"""Tests for the new configurable self-harm threshold logic."""

from humane_proxy.classifiers.pipeline import SafetyPipeline
from humane_proxy.classifiers.models import ClassificationResult

def test_self_harm_above_threshold_escalates():
    config = {
        "safety": {
            "categories": {
                "self_harm": {
                    "escalate_threshold": 0.5
                }
            }
        }
    }
    pipeline = SafetyPipeline(config)
    
    # Simulate a Stage 2 result that scored 0.60
    st2_res = ClassificationResult(category="self_harm", score=0.60, stage=2)
    
    # Finalize should escalate and force score to 1.0 (definitive self-harm)
    final = pipeline._finalize(st2_res, "test-session", "test message")
    
    assert final.classification.category == "self_harm"
    assert final.classification.score == 1.0
    assert final.should_escalate is True
    assert final.should_block is True

def test_self_harm_below_threshold_downgrades():
    config = {
        "safety": {
            "categories": {
                "self_harm": {
                    "escalate_threshold": 0.5
                }
            }
        }
    }
    pipeline = SafetyPipeline(config)
    
    # Simulate an ambiguous Stage 2 result scoring 0.40
    st2_res = ClassificationResult(category="self_harm", score=0.40, stage=2)
    
    # Finalize should downgrade it to safe
    final = pipeline._finalize(st2_res, "test-session", "test message")
    
    assert final.classification.category == "safe"
    assert final.classification.score == 0.40
    assert final.should_escalate is False
    assert final.should_block is False
    assert any("self_harm_below_threshold" in t for t in final.classification.triggers)
