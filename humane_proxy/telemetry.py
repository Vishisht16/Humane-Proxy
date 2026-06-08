# Copyright 2026 Vishisht Mishra (Vishisht16)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenTelemetry tracing support for HumaneProxy (Issue #7).

Owns ALL OTel logic so the rest of the codebase stays clean.

Public API
----------
setup_telemetry(config)
    Call once inside _lifespan after config is loaded.
    Reads ``telemetry.enabled`` (yaml) or ``HUMANE_PROXY_TELEMETRY_ENABLED``
    (env var, wins over yaml).  When disabled, registers a NoOpTracerProvider
    so every OTel call is a zero-overhead no-op — no if/else in hot paths.

@traced_stage(span_name)
    Decorator for any sync or async pipeline method.
    Creates a span, populates privacy-safe attributes from the return dict,
    records exceptions, and re-raises them — never swallows errors.

setup_telemetry_with_memory_exporter()   [TEST HELPER]
    Wires a fresh InMemorySpanExporter and returns it.
    Used in tests to inspect recorded spans without a real OTLP backend.

Privacy guarantee
-----------------
Raw message text is NEVER written to a span attribute.
Only: session_id, category, score, stage_reached, triggers_count, message_hash.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("humane_proxy.telemetry")

# ---------------------------------------------------------------------------
# Optional import — only available when humane-proxy[telemetry] is installed.
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

# Module-level tracer — set by setup_telemetry() before first use.
_tracer: Optional[Any] = None


# ---------------------------------------------------------------------------
# Public: initialise
# ---------------------------------------------------------------------------

def setup_telemetry(config: Any) -> None:
    """Initialise the global tracer.  Call once inside _lifespan.

    Priority: HUMANE_PROXY_TELEMETRY_ENABLED env var > config.telemetry.enabled.
    Default: disabled.
    """
    global _tracer

    import os

    # 1. Env override wins.
    env_val = os.environ.get("HUMANE_PROXY_TELEMETRY_ENABLED", "").strip().lower()
    if env_val in ("1", "true", "yes"):
        enabled = True
    elif env_val in ("0", "false", "no"):
        enabled = False
    else:
        # 2. Read from config dict.
        try:
            tel_cfg = (
                config.get("telemetry", {})
                if isinstance(config, dict)
                else getattr(config, "telemetry", {})
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
            "HumaneProxy telemetry: enabled in config but opentelemetry is not "
            "installed. Run: pip install humane-proxy[telemetry]  Using no-op tracer."
        )
        _tracer = _make_noop_tracer()
        return

    # 3. Build real TracerProvider with OTLP exporter.
    try:
        tel_cfg = (
            config.get("telemetry", {})
            if isinstance(config, dict)
            else getattr(config, "telemetry", {})
        )
        endpoint = (
            tel_cfg.get("endpoint", "http://localhost:4317")
            if isinstance(tel_cfg, dict)
            else getattr(tel_cfg, "endpoint", "http://localhost:4317")
        )
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = provider.get_tracer("humane_proxy")
        logger.info("HumaneProxy telemetry: enabled → %s", endpoint)
    except Exception as exc:
        logger.warning(
            "HumaneProxy telemetry: OTLP init failed (%s). Using no-op tracer.", exc
        )
        _tracer = _make_noop_tracer()


def setup_telemetry_with_memory_exporter() -> "InMemorySpanExporter":
    """TEST HELPER: wire a fresh InMemorySpanExporter and return it.

    Creates a new TracerProvider and injects it directly as the module-level
    tracer so each test gets a clean, isolated exporter.

        exporter = setup_telemetry_with_memory_exporter()
        # run pipeline code
        spans = exporter.get_finished_spans()
    """
    global _tracer

    if not _OTEL_AVAILABLE:
        raise ImportError(
            "opentelemetry is required for tests. "
            "Install: pip install humane-proxy[telemetry]"
        )

    memory_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    trace.set_tracer_provider(provider)
    _tracer = provider.get_tracer("humane_proxy")
    return memory_exporter


def get_tracer() -> Any:
    """Return the active tracer.  Always safe to call."""
    global _tracer
    if _tracer is None:
        _tracer = _make_noop_tracer()
    return _tracer


# ---------------------------------------------------------------------------
# Public: decorator
# ---------------------------------------------------------------------------

def traced_stage(span_name: str) -> Callable:
    """Wrap a sync or async pipeline method in an OTel span.

    Span attributes are populated from the function return value (dict).
    Only privacy-safe keys are written — never raw message text.
    Exceptions are recorded on the span and re-raised.

    Example::

        @traced_stage("stage1.heuristics")
        def run(self, text: str, session_id: str = "") -> dict: ...

        @traced_stage("pipeline.classify")
        async def classify(self, text: str, session_id: str) -> PipelineResult: ...
    """
    def decorator(fn: Callable) -> Callable:
        """Wrap *fn* with a span named *span_name*, handling sync and async."""
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                """Async span wrapper — creates span, sets attributes, records errors."""
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
                """Sync span wrapper — creates span, sets attributes, records errors."""
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# These are the ONLY keys ever written to spans.  Never raw message text.
_SAFE_ATTRIBUTES: dict[str, str] = {
    "session_id":     "humane_proxy.session_id",
    "category":       "humane_proxy.category",
    "score":          "humane_proxy.final_score",
    "stage_reached":  "humane_proxy.stage_reached",
    "triggers_count": "humane_proxy.triggers_count",
    "message_hash":   "humane_proxy.message_hash",
}


def _set_safe_attributes(span: Any, result: Any, call_kwargs: dict) -> None:
    """Write privacy-safe span attributes from the result.

    Supports three result types:
    - Plain dict (stage classifiers return dicts)
    - PipelineResult dataclass (classify / classify_sync return this)
    - ClassificationResult dataclass (stage3 providers return this)
    """
    if not _OTEL_AVAILABLE:
        return

    data: dict = {}

    if isinstance(result, dict):
        # Stage classifiers return plain dicts.
        data = result

    elif hasattr(result, "classification"):
        # PipelineResult — the final output of classify / classify_sync.
        # Fields: classification (ClassificationResult), message_hash, trajectory
        try:
            cr = result.classification          # ClassificationResult
            data = {
                "category":       cr.category,
                "score":          cr.score,
                "stage_reached":  cr.stage,
                "triggers_count": len(cr.triggers),
                "message_hash":   result.message_hash,
            }
        except AttributeError:
            pass

    elif hasattr(result, "category") and hasattr(result, "score"):
        # ClassificationResult — returned by stage3 providers.
        # Fields: category, score, triggers, stage, reasoning
        try:
            data = {
                "category":       result.category,
                "score":          result.score,
                "stage_reached":  result.stage,
                "triggers_count": len(result.triggers),
            }
        except AttributeError:
            pass

    # session_id is a call kwarg, not in the result object.
    if "session_id" not in data and "session_id" in call_kwargs:
        data["session_id"] = call_kwargs["session_id"]

    for result_key, attr_name in _SAFE_ATTRIBUTES.items():
        val = data.get(result_key)
        if val is None:
            continue
        span.set_attribute(
            attr_name,
            val if isinstance(val, (str, bool, int, float)) else str(val),
        )


def _record_exception(span: Any, exc: Exception) -> None:
    """Record exception on span without letting telemetry break the main flow."""
    if not _OTEL_AVAILABLE:
        return
    try:
        from opentelemetry.trace import StatusCode
        span.record_exception(exc)
        span.set_status(StatusCode.ERROR, str(exc))
    except Exception:
        pass


def _make_noop_tracer() -> Any:
    """Return a no-op tracer — OTel built-in or pure Python fallback."""
    if _OTEL_AVAILABLE:
        from opentelemetry.trace import NoOpTracerProvider
        return NoOpTracerProvider().get_tracer("humane_proxy")
    return _PurePythonNoOpTracer()


# ---------------------------------------------------------------------------
# Pure Python no-op — used when opentelemetry is not installed at all.
# ---------------------------------------------------------------------------

class _PurePythonNoOpTracer:
    """Pure Python no-op tracer used when opentelemetry is not installed.

    Implements the minimal subset of the OTel Tracer API used by
    :func:`traced_stage` so the decorator works with zero overhead
    even when the optional dependency is absent.
    """

    def start_as_current_span(self, name: str, **kwargs: Any):
        """Return a no-op context manager that yields a no-op span."""
        return _NoOpCtx()


class _NoOpCtx:
    """Context manager that yields a :class:`_NoOpSpan` and does nothing else."""

    def __enter__(self):
        """Enter the context and return a no-op span."""
        return _NoOpSpan()

    def __exit__(self, *args: Any):
        """Exit the context without any action."""
        pass


class _NoOpSpan:
    """No-op span whose every method is a silent pass.

    Used as the span object when opentelemetry is not installed,
    ensuring that all ``span.set_attribute`` / ``span.record_exception``
    calls in :func:`traced_stage` are safe to make unconditionally.
    """

    def set_attribute(self, key: str, value: Any) -> None:
        """Accept and silently discard a span attribute."""
        pass

    def record_exception(self, exc: Exception) -> None:
        """Accept and silently discard an exception record."""
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        """Accept and silently discard a status update."""
        pass