from __future__ import annotations

import logging
import threading

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace import NoOpTracerProvider

    OTEL_AVAILABLE = True

except ImportError:
    trace = None
    NoOpTracerProvider = None
    OTEL_AVAILABLE = False


logger = logging.getLogger("humane_proxy.telemetry")

_INITIALIZED = False
_lock = threading.Lock()


def setup_telemetry(config: dict) -> None:
    """
    Initialize OpenTelemetry tracing.

    Design goals:
    - Fully opt-in
    - No duplicate initialization
    - NoOp provider when disabled
    - Graceful exporter failures
    """

    global _INITIALIZED

    with _lock:
        if _INITIALIZED:
            return

        if not OTEL_AVAILABLE:
            logger.info("OpenTelemetry dependencies not installed")
            return

        telemetry_cfg = config.get("telemetry", {})

        enabled = telemetry_cfg.get("enabled", False)

        # ---------------------------------------------------------
        # DISABLED MODE
        # ---------------------------------------------------------

        if not enabled:

            if NoOpTracerProvider is not None:
                trace.set_tracer_provider(NoOpTracerProvider())

            _INITIALIZED = True

            logger.info("Telemetry disabled -> using NoOpTracerProvider")

            return

        # ---------------------------------------------------------
        # ENABLED MODE
        # ---------------------------------------------------------

        endpoint = telemetry_cfg.get(
            "otlp_endpoint",
            "http://localhost:4318/v1/traces",
        )

        service_name = telemetry_cfg.get(
            "service_name",
            "humane-proxy",
        )

        resource = Resource.create(
            {
                "service.name": service_name,
            }
        )

        provider = TracerProvider(resource=resource)

        try:
            exporter = OTLPSpanExporter(
                endpoint=endpoint,
            )

            processor = BatchSpanProcessor(exporter)

            provider.add_span_processor(processor)

        except Exception:
            logger.exception("Failed to initialize OTLP exporter")

        trace.set_tracer_provider(provider)
        _INITIALIZED = True

        logger.info(
            "OpenTelemetry initialized -> %s",
            endpoint,
        )


def shutdown_telemetry() -> None:
    if not OTEL_AVAILABLE:
        return

    try:
        provider = trace.get_tracer_provider()

        shutdown = getattr(provider, "shutdown", None)

        if callable(shutdown):
            shutdown()

    except Exception:
        logger.exception("Telemetry shutdown failed")
