"""
tests/test_telemetry.py

Deterministic CI tests for Issue #7 — OTel tracing integration.

All tests use InMemorySpanExporter — no real OTLP backend required.
No timing-based assertions.  Fully deterministic.

Run with:
    pytest tests/test_telemetry.py -v
"""

import asyncio
import hashlib
import pytest

otel = pytest.importorskip(
    "opentelemetry",
    reason="opentelemetry not installed. Run: pip install humane-proxy[telemetry]",
)

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from humane_proxy.telemetry import (
    setup_telemetry,
    setup_telemetry_with_memory_exporter,
    traced_stage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def _make_config(enabled: bool) -> dict:
    return {"telemetry": {"enabled": enabled, "endpoint": "http://localhost:4317"}}


# ---------------------------------------------------------------------------
# Test 1: disabled → zero spans
# ---------------------------------------------------------------------------

def test_disabled_produces_no_spans():
    exporter = setup_telemetry_with_memory_exporter()
    setup_telemetry(_make_config(enabled=False))

    @traced_stage("pipeline.classify")
    def dummy(msg):
        return {"category": "safe", "score": 0.0, "stage_reached": 1}

    dummy("hello")
    assert len(exporter.get_finished_spans()) == 0


# ---------------------------------------------------------------------------
# Test 2: enabled + sync → 1 span with correct name
# ---------------------------------------------------------------------------

def test_enabled_sync_records_span():
    exporter = setup_telemetry_with_memory_exporter()

    @traced_stage("pipeline.classify")
    def classify_sync(msg, session_id=""):
        return {
            "safe": True, "category": "safe", "score": 0.1,
            "stage_reached": 1, "triggers_count": 0,
            "message_hash": _h(msg), "session_id": session_id,
        }

    classify_sync("hello", session_id="s1")
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "pipeline.classify"


# ---------------------------------------------------------------------------
# Test 3: enabled + async → 1 span with correct name
# ---------------------------------------------------------------------------

def test_enabled_async_records_span():
    exporter = setup_telemetry_with_memory_exporter()

    @traced_stage("pipeline.classify")
    async def classify_async(msg, session_id=""):
        return {
            "safe": False, "category": "self_harm", "score": 0.9,
            "stage_reached": 3, "triggers_count": 2,
            "message_hash": _h(msg), "session_id": session_id,
        }

    asyncio.run(classify_async("test", session_id="s2"))
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "pipeline.classify"


# ---------------------------------------------------------------------------
# Test 4: parent-child span hierarchy
# ---------------------------------------------------------------------------

def test_parent_child_hierarchy():
    exporter = setup_telemetry_with_memory_exporter()

    @traced_stage("stage1.heuristics")
    def stage1(msg):
        return {"category": "safe", "score": 0.05, "stage_reached": 1, "triggers_count": 0}

    @traced_stage("pipeline.classify")
    def classify(msg, session_id=""):
        r = stage1(msg)
        return {
            "safe": True, "category": r["category"], "score": r["score"],
            "stage_reached": 1, "triggers_count": 0,
            "message_hash": _h(msg), "session_id": session_id,
        }

    classify("safe message", session_id="s3")
    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    root  = next(s for s in spans if s.name == "pipeline.classify")
    child = next(s for s in spans if s.name == "stage1.heuristics")

    assert child.parent is not None
    assert child.parent.span_id == root.context.span_id


# ---------------------------------------------------------------------------
# Test 5: span attributes — correct values, no raw message text
# ---------------------------------------------------------------------------

def test_span_attributes_privacy_safe():
    exporter = setup_telemetry_with_memory_exporter()
    raw = "I want to end my life"

    @traced_stage("pipeline.classify")
    def classify(msg, session_id=""):
        return {
            "safe": False, "category": "self_harm", "score": 1.0,
            "stage_reached": 1, "triggers_count": 3,
            "message_hash": _h(msg), "session_id": session_id,
        }

    classify(raw, session_id="priv-session")
    attrs = dict(exporter.get_finished_spans()[0].attributes)

    assert attrs.get("humane_proxy.session_id")    == "priv-session"
    assert attrs.get("humane_proxy.category")      == "self_harm"
    assert attrs.get("humane_proxy.final_score")   == 1.0
    assert attrs.get("humane_proxy.stage_reached") == 1
    assert attrs.get("humane_proxy.triggers_count") == 3
    assert attrs.get("humane_proxy.message_hash")  == _h(raw)

    # Raw message must never appear in any attribute value.
    for key, val in attrs.items():
        assert raw not in str(val), f"Raw message leaked into span attribute '{key}'"


# ---------------------------------------------------------------------------
# Test 6: sync and async produce identical span structure
# ---------------------------------------------------------------------------

def test_sync_async_parity():
    shared = {
        "safe": True, "category": "safe", "score": 0.1,
        "stage_reached": 1, "triggers_count": 0,
        "message_hash": _h("hello"), "session_id": "parity",
    }

    @traced_stage("pipeline.classify")
    def cs(msg, session_id=""): return shared

    @traced_stage("pipeline.classify")
    async def ca(msg, session_id=""): return shared

    exporter = setup_telemetry_with_memory_exporter()
    cs("hello", session_id="parity")
    sync_spans = list(exporter.get_finished_spans())
    exporter.clear()

    asyncio.run(ca("hello", session_id="parity"))
    async_spans = list(exporter.get_finished_spans())

    assert len(sync_spans) == len(async_spans) == 1
    assert sync_spans[0].name == async_spans[0].name
    assert dict(sync_spans[0].attributes) == dict(async_spans[0].attributes)


# ---------------------------------------------------------------------------
# Test 7: exception recorded on span and re-raised
# ---------------------------------------------------------------------------

def test_exception_recorded_and_reraised():
    exporter = setup_telemetry_with_memory_exporter()

    @traced_stage("stage3.reasoning_llm")
    async def failing_stage(msg):
        raise ConnectionError("Groq API timeout")

    with pytest.raises(ConnectionError, match="Groq API timeout"):
        asyncio.run(failing_stage("some message"))

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert any(e.name == "exception" for e in spans[0].events)


# ---------------------------------------------------------------------------
# Test 8: toggling telemetry does not change classification output
# ---------------------------------------------------------------------------

def test_toggle_does_not_affect_result():
    expected = {
        "safe": True, "category": "safe", "score": 0.05,
        "stage_reached": 1, "triggers_count": 0,
        "message_hash": _h("safe msg"), "session_id": "toggle",
    }

    @traced_stage("pipeline.classify")
    def classify(msg, session_id=""): return expected

    setup_telemetry_with_memory_exporter()
    r1 = classify("safe msg", session_id="toggle")

    setup_telemetry(_make_config(enabled=False))
    r2 = classify("safe msg", session_id="toggle")

    assert r1 == r2 == expected