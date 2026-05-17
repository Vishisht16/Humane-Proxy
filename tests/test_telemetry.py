"""Telemetry tests for HumaneProxy.

Goals:
- Deterministic span validation
- Enabled vs disabled telemetry
- Sync + async parity
- Span hierarchy validation
- Attribute whitelist enforcement
- No exporter/network dependency
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

opentelemetry = pytest.importorskip("opentelemetry")

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
except ImportError:
    pytest.skip("OpenTelemetry SDK not installed", allow_module_level=True)

from humane_proxy import HumaneProxy
from humane_proxy.classifiers.models import ClassificationResult
from humane_proxy.classifiers.pipeline import SafetyPipeline

ALLOWED_ATTRIBUTES = {
    "session_id",
    "score",
    "final_score",
    "category",
    "stage_reached",
    "triggers_count",
    "message_hash",
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


class DummyStage2:
    """Deterministic fake Stage-2 classifier."""

    def classify(self, text: str) -> ClassificationResult:
        return ClassificationResult(
            category="criminal_intent",
            score=0.82,
            triggers=["embedding_match"],
            stage=2,
        )


class DummyStage3:
    """Deterministic fake Stage-3 classifier."""

    async def classify(
        self,
        text: str,
        previous: ClassificationResult,
    ) -> ClassificationResult:
        return ClassificationResult(
            category="criminal_intent",
            score=0.91,
            triggers=["llm_reasoning"],
            stage=3,
            reasoning="deterministic-test",
        )


def make_config(enabled: bool = True) -> dict[str, Any]:
    return {
        "pipeline": {
            "enabled_stages": [1, 2, 3],
            "stage1_ceiling": 0.3,
            "stage2_ceiling": 0.4,
        },
        "safety": {
            "risk_threshold": 0.7,
            "spike_boost": 0.25,
            "categories": {
                "self_harm": {
                    "escalate_threshold": 0.5,
                }
            },
        },
        "privacy": {
            "store_message_text": False,
        },
        "telemetry": {
            "enabled": enabled,
            "service_name": "humane-proxy-test",
        },
    }


def setup_inmemory_tracing():
    """
    Create a fresh in-memory OpenTelemetry provider/exporter
    for deterministic isolated tests.
    """

    exporter = InMemorySpanExporter()

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "humane-proxy-test",
            }
        )
    )

    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # ---------------------------------------------------------
    # HARD RESET GLOBAL OTEL STATE
    # ---------------------------------------------------------

    import opentelemetry.trace as trace_api
    import opentelemetry.util._once as ot_once

    # reset the "set once" guard
    if hasattr(trace_api, "_TRACER_PROVIDER_SET_ONCE"):
        trace_api._TRACER_PROVIDER_SET_ONCE = ot_once.Once()

    # clear existing provider
    if hasattr(trace_api, "_TRACER_PROVIDER"):
        trace_api._TRACER_PROVIDER = None

    # install fresh provider
    trace.set_tracer_provider(provider)

    return exporter


def assert_allowed_attributes(span):
    """Ensure span attributes follow whitelist policy."""

    invalid = []

    for key in span.attributes.keys():
        normalized = key.split(".")[-1]

        if normalized not in ALLOWED_ATTRIBUTES:
            invalid.append(key)

    assert not invalid, (
        f"Disallowed attributes found on span " f"{span.name}: {invalid}"
    )


def build_pipeline(enabled: bool = True) -> SafetyPipeline:
    pipeline = SafetyPipeline(make_config(enabled=enabled))

    pipeline._stage2 = DummyStage2()
    pipeline._stage3 = DummyStage3()

    return pipeline


# -------------------------------------------------------------------
# Telemetry Disabled
# -------------------------------------------------------------------


def test_telemetry_disabled_sync():
    """Pipeline must work fully when telemetry disabled."""

    pipeline = build_pipeline(enabled=False)

    result = pipeline.classify_sync(
        "hello world",
        "session-1",
    )

    assert result.classification.category == "criminal_intent"
    assert result.classification.stage == 2
    assert result.should_escalate is True


@pytest.mark.asyncio
async def test_telemetry_disabled_async():
    """Async pipeline must work with telemetry disabled."""

    pipeline = build_pipeline(enabled=False)

    result = await pipeline.classify(
        "hello world",
        "session-1",
    )

    assert result.classification.category == "criminal_intent"
    assert result.classification.stage == 3
    assert result.should_escalate is True


# -------------------------------------------------------------------
# Telemetry Enabled
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_pipeline_spans():
    """Validate async span hierarchy."""

    exporter = setup_inmemory_tracing()

    pipeline = build_pipeline(enabled=True)

    await pipeline.classify(
        "hello world",
        "session-async",
    )

    spans = exporter.get_finished_spans()

    names = [s.name for s in spans]

    assert "humane_proxy.pipeline.classify" in names
    assert "humane_proxy.stage1" in names
    assert "humane_proxy.stage2" in names
    assert "humane_proxy.stage3" in names
    assert "humane_proxy.pipeline.finalize" in names

    # ---------------------------------------------------------
    # Validate hierarchy
    # ---------------------------------------------------------

    by_name = {s.name: s for s in spans}

    root = by_name["humane_proxy.pipeline.classify"]

    for child_name in [
        "humane_proxy.stage1",
        "humane_proxy.stage2",
        "humane_proxy.stage3",
        "humane_proxy.pipeline.finalize",
    ]:
        child = by_name[child_name]

        assert child.parent.span_id == root.context.span_id

    # ---------------------------------------------------------
    # Validate allowed attributes only
    # ---------------------------------------------------------

    for span in spans:
        assert_allowed_attributes(span)


def test_sync_pipeline_spans():
    """Validate sync pipeline tracing."""

    exporter = setup_inmemory_tracing()

    pipeline = build_pipeline(enabled=True)

    pipeline.classify_sync(
        "hello sync",
        "session-sync",
    )

    spans = exporter.get_finished_spans()

    names = [s.name for s in spans]

    assert "humane_proxy.pipeline.classify" in names
    assert "humane_proxy.stage1" in names
    assert "humane_proxy.stage2" in names
    assert "humane_proxy.pipeline.finalize" in names

    assert "humane_proxy.stage3" not in names

    by_name = {s.name: s for s in spans}

    root = by_name["humane_proxy.pipeline.classify"]

    for child_name in [
        "humane_proxy.stage1",
        "humane_proxy.stage2",
        "humane_proxy.pipeline.finalize",
    ]:
        child = by_name[child_name]

        assert child.parent.span_id == root.context.span_id

    for span in spans:
        assert_allowed_attributes(span)


def test_proxy_check_span_sync():
    """Validate proxy.check spans are emitted and parented correctly."""

    exporter = setup_inmemory_tracing()
    proxy = object.__new__(HumaneProxy)
    proxy._config = make_config(enabled=True)
    proxy._pipeline = build_pipeline(enabled=True)
    proxy._proxy_tracer = trace.get_tracer("humane_proxy.proxy")

    proxy.check("hello proxy", "proxy-session")

    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]

    assert "humane_proxy.proxy.check" in names
    assert "humane_proxy.pipeline.classify" in names

    proxy_root = next(s for s in spans if s.name == "humane_proxy.proxy.check")
    pipeline_root = next(s for s in spans if s.name == "humane_proxy.pipeline.classify")
    assert pipeline_root.parent.span_id == proxy_root.context.span_id

    for span in spans:
        assert_allowed_attributes(span)


@pytest.mark.asyncio
async def test_proxy_check_async_span():
    """Validate proxy.check_async spans are emitted and parented correctly."""

    exporter = setup_inmemory_tracing()
    proxy = object.__new__(HumaneProxy)
    proxy._config = make_config(enabled=True)
    proxy._pipeline = build_pipeline(enabled=True)
    proxy._proxy_tracer = trace.get_tracer("humane_proxy.proxy")

    await proxy.check_async("hello proxy", "proxy-session")

    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]

    assert "humane_proxy.proxy.check_async" in names
    assert "humane_proxy.pipeline.classify" in names

    proxy_root = next(s for s in spans if s.name == "humane_proxy.proxy.check_async")
    pipeline_root = next(s for s in spans if s.name == "humane_proxy.pipeline.classify")
    assert pipeline_root.parent.span_id == proxy_root.context.span_id

    for span in spans:
        assert_allowed_attributes(span)


# -------------------------------------------------------------------
# Attribute Validation
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_id_is_hashed():
    """Raw session IDs must never appear in telemetry."""

    exporter = setup_inmemory_tracing()

    pipeline = build_pipeline(enabled=True)

    raw_session = "VERY_SECRET_SESSION"

    await pipeline.classify(
        "hello",
        raw_session,
    )

    spans = exporter.get_finished_spans()

    for span in spans:
        values = list(span.attributes.values())

        assert raw_session not in values


@pytest.mark.asyncio
async def test_message_hash_present():
    """message_hash must exist when store_message_text=False."""

    exporter = setup_inmemory_tracing()

    pipeline = build_pipeline(enabled=True)

    await pipeline.classify(
        "hello",
        "session-hash",
    )

    spans = exporter.get_finished_spans()

    finalize_span = next(s for s in spans if s.name == "humane_proxy.pipeline.finalize")

    assert "humane_proxy.message_hash" in finalize_span.attributes


# -------------------------------------------------------------------
# Stage Validation
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stage3_attributes():
    """Stage-3 span must contain required attributes."""

    exporter = setup_inmemory_tracing()

    pipeline = build_pipeline(enabled=True)

    await pipeline.classify(
        "hello",
        "stage3-session",
    )

    spans = exporter.get_finished_spans()

    stage3 = next(s for s in spans if s.name == "humane_proxy.stage3")

    attrs = stage3.attributes

    assert attrs["humane_proxy.category"] == "criminal_intent"
    assert attrs["humane_proxy.final_score"] == 0.91
    assert attrs["humane_proxy.triggers_count"] == 1


def test_stage2_attributes_sync():
    """Stage-2 sync span validation."""

    exporter = setup_inmemory_tracing()

    pipeline = build_pipeline(enabled=True)

    pipeline.classify_sync(
        "sync message",
        "sync-stage2",
    )

    spans = exporter.get_finished_spans()

    stage2 = next(s for s in spans if s.name == "humane_proxy.stage2")

    attrs = stage2.attributes

    assert attrs["humane_proxy.category"] == "criminal_intent"
    assert attrs["humane_proxy.final_score"] == 0.82
    assert attrs["humane_proxy.triggers_count"] == 1


# -------------------------------------------------------------------
# Escalation Validation
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalation_attributes():
    """Final spans should expose escalation classification."""

    exporter = setup_inmemory_tracing()

    pipeline = build_pipeline(enabled=True)

    result = await pipeline.classify(
        "hello escalation",
        "escalation-session",
    )

    assert result.should_escalate is True

    spans = exporter.get_finished_spans()

    root = next(s for s in spans if s.name == "humane_proxy.pipeline.classify")

    attrs = root.attributes

    assert attrs["humane_proxy.category"] == "criminal_intent"
    assert attrs["humane_proxy.stage_reached"] == 3
    assert attrs["humane_proxy.final_score"] == 0.91


# -------------------------------------------------------------------
# Determinism
# -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeatability():
    """Telemetry tests must be deterministic."""

    exporter1 = setup_inmemory_tracing()

    pipeline1 = build_pipeline(enabled=True)

    await pipeline1.classify(
        "repeatability",
        "repeat-session",
    )

    spans1 = exporter1.get_finished_spans()

    exporter2 = setup_inmemory_tracing()

    pipeline2 = build_pipeline(enabled=True)

    await pipeline2.classify(
        "repeatability",
        "repeat-session",
    )

    spans2 = exporter2.get_finished_spans()

    names1 = [s.name for s in spans1]
    names2 = [s.name for s in spans2]

    assert names1 == names2


# -------------------------------------------------------------------
# Exporter Failure Resilience
# -------------------------------------------------------------------


def test_pipeline_survives_without_exporter():
    """Pipeline must function even without exporter."""

    provider = TracerProvider()

    trace.set_tracer_provider(provider)

    pipeline = build_pipeline(enabled=True)

    result = pipeline.classify_sync(
        "hello",
        "resilience-session",
    )

    assert result.classification.category == "criminal_intent"


# -------------------------------------------------------------------
# Event Loop Safety
# -------------------------------------------------------------------


def test_async_pipeline_multiple_runs():
    """Multiple async executions should remain stable."""

    exporter = setup_inmemory_tracing()

    pipeline = build_pipeline(enabled=True)

    async def runner():
        await asyncio.gather(
            pipeline.classify("a", "1"),
            pipeline.classify("b", "2"),
            pipeline.classify("c", "3"),
        )

    asyncio.run(runner())

    spans = exporter.get_finished_spans()

    roots = [s for s in spans if s.name == "humane_proxy.pipeline.classify"]

    assert len(roots) == 3
