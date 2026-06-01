"""
humane_proxy/telemetry.py

Owns ALL OpenTelemetry logic for HumaneProxy.

- setup_telemetry(config) must be called once inside _lifespan.
- @traced_stage(span_name) wraps any sync or async pipeline method with a span.

When telemetry is disabled, a NoOpTracerProvider is registered.
All OTel API calls are zero-overhead no-ops at the library level.

Privacy guarantee: raw message text is NEVER added to any span attribute.
Only hashed identifiers and numeric scores are recorded.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Lazy imports — opentelemetry is an optional dependency.

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

# Module-level tracer.  Always set by setup_telemetry() before first use.
# Tests inject directly via setup_telemetry_with_memory_exporter().
_tracer: Optional[Any] = None


# Public setup API


def setup_telemetry(config: Any) -> None:
    """
    Initialise the global tracer.  Must be called once inside _lifespan.

    Decision tree
    -------------
    1. Check HUMANE_PROXY_TELEMETRY_ENABLED env var (wins over yaml).
    2. Fall back to config.telemetry.enabled (default: False).
    3. disabled  → NoOpTracerProvider, zero overhead.
    4. enabled but opentelemetry not installed → warn, fall back to no-op.
    5. enabled and opentelemetry installed     → OTLP TracerProvider.
    """
    global _tracer

    import os

    env_val = os.environ.get("HUMANE_PROXY_TELEMETRY_ENABLED", "").strip().lower()
    if env_val in ("1", "true", "yes"):
        enabled = True
    elif env_val in ("0", "false", "no"):
        enabled = False
    else:
        try:
            tel_cfg = (
                config.telemetry if hasattr(config, "telemetry")
                else config.get("telemetry", {})
            )
            enabled = (
                tel_cfg.get("enabled", False)
                if isinstance(tel_cfg, dict)
                else getattr(tel_cfg, "enabled", False)
            )
        except Exception:
            enabled = False

    if not enabled:
        _tracer = _make_noop_tracer()
        logger.debug("HumaneProxy telemetry: disabled.")
        return

    if not _OTEL_AVAILABLE:
        logger.warning(
            "HumaneProxy telemetry: enabled in config but 'opentelemetry' is not "
            "installed. Run: pip install humane-proxy[telemetry]. Using no-op tracer."
        )
        _tracer = _make_noop_tracer()
        return

    try:
        tel_cfg = (
            config.telemetry if hasattr(config, "telemetry")
            else config.get("telemetry", {})
        )
        endpoint = (
            tel_cfg.get("endpoint", "http://localhost:4317")
            if isinstance(tel_cfg, dict)
            else getattr(tel_cfg, "endpoint", "http://localhost:4317")
        )
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        # Set global provider for context propagation, then get our tracer.
        trace.set_tracer_provider(provider)
        _tracer = provider.get_tracer("humane_proxy")
        logger.info("HumaneProxy telemetry: enabled → %s", endpoint)
    except Exception as exc:
        logger.warning(
            "HumaneProxy telemetry: OTLP init failed (%s). Using no-op tracer.", exc
        )
        _tracer = _make_noop_tracer()


def setup_telemetry_with_memory_exporter() -> "InMemorySpanExporter":
    """
    TEST HELPER — wire a fresh InMemorySpanExporter and return it.

    Bypasses the OTel global singleton entirely: creates a new TracerProvider,
    injects it as both the global provider (for context propagation) and
    as the module-level _tracer directly.

    Each test should call this at the start to get a clean exporter.

        exporter = setup_telemetry_with_memory_exporter()
        # ... run pipeline code ...
        spans = exporter.get_finished_spans()
    """
    global _tracer

    if not _OTEL_AVAILABLE:
        raise ImportError(
            "opentelemetry is required for tests. "
            "Install with: pip install humane-proxy[telemetry]"
        )

    memory_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    trace.set_tracer_provider(provider)
    # IMPORTANT: get tracer from THIS provider directly, not from global.
    _tracer = provider.get_tracer("humane_proxy")
    return memory_exporter


def get_tracer() -> Any:
    """Return the active tracer.  Always safe to call."""
    global _tracer
    if _tracer is None:
        _tracer = _make_noop_tracer()
    return _tracer



# The decorator


def traced_stage(span_name: str) -> Callable:
    """
    Wrap a pipeline stage method in an OTel span.

    Works transparently on both sync and async functions.

    Span attributes are populated from the function return value (dict).
    Only privacy-safe keys are written — never raw message text.

    Exceptions are recorded on the span and re-raised.
    """
    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    try:
                        result = await fn(*args, **kwargs)
                        _set_safe_attributes(span, result, kwargs)
                        return result
                    except Exception as exc:
                        _record_exception(span, exc)
                        raise
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.start_as_current_span(span_name) as span:
                    try:
                        result = fn(*args, **kwargs)
                        _set_safe_attributes(span, result, kwargs)
                        return result
                    except Exception as exc:
                        _record_exception(span, exc)
                        raise
            return sync_wrapper
    return decorator


# Private helpers

_SAFE_ATTRIBUTES = {
    "session_id":     "humane_proxy.session_id",
    "category":       "humane_proxy.category",
    "score":          "humane_proxy.final_score",
    "stage_reached":  "humane_proxy.stage_reached",
    "triggers_count": "humane_proxy.triggers_count",
    "message_hash":   "humane_proxy.message_hash",
}


def _set_safe_attributes(span: Any, result: Any, call_kwargs: dict) -> None:
    """Populate span with privacy-safe attributes from the result dict."""
    if not _OTEL_AVAILABLE:
        return

    data: dict = result if isinstance(result, dict) else {}

    # Pull session_id from call kwargs if not present in result.
    if "session_id" not in data and "session_id" in call_kwargs:
        data = {**data, "session_id": call_kwargs["session_id"]}

    for result_key, attr_name in _SAFE_ATTRIBUTES.items():
        if result_key in data and data[result_key] is not None:
            val = data[result_key]
            span.set_attribute(
                attr_name,
                val if isinstance(val, (str, bool, int, float)) else str(val),
            )


def _record_exception(span: Any, exc: Exception) -> None:
    """Record exception on span if OTel is available."""
    if not _OTEL_AVAILABLE:
        return
    try:
        from opentelemetry.trace import StatusCode
        span.record_exception(exc)
        span.set_status(StatusCode.ERROR, str(exc))
    except Exception:
        pass  # Never let telemetry break the main flow


def _make_noop_tracer() -> Any:
    """Return a no-op tracer (OTel built-in or pure Python fallback)."""
    if _OTEL_AVAILABLE:
        from opentelemetry.trace import NoOpTracerProvider
        return NoOpTracerProvider().get_tracer("humane_proxy")
    return _PurePythonNoOpTracer()


# Pure Python no-op fallback (used when opentelemetry is not installed)


class _PurePythonNoOpTracer:
    def start_as_current_span(self, name: str, **kwargs):
        return _NoOpSpanCtx()


class _NoOpSpanCtx:
    def __enter__(self):
        return _NoOpSpan()

    def __exit__(self, *args):
        pass


class _NoOpSpan:
    def set_attribute(self, key, value):
        pass

    def record_exception(self, exc):
        pass

    def set_status(self, *args, **kwargs):
        pass