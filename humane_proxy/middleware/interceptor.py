"""HumaneProxy interceptor — FastAPI proxy with full safety pipeline + care response."""

from __future__ import annotations

import logging
import os
import uuid
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, nullcontext
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from humane_proxy.escalation.local_db import init_db
from humane_proxy.escalation.router import escalate, get_self_harm_response

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

except ImportError:
    trace = None
    Status = None
    StatusCode = None

from humane_proxy.config import get_config
from humane_proxy.telemetry import setup_telemetry, shutdown_telemetry

def _set_attr(span, key: str, value):
    if span is not None:
        span.set_attribute(key, value)

logger = logging.getLogger("humane_proxy")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

def _get_tracer():
    if trace is None:
        return None

    return trace.get_tracer("humane_proxy.interceptor")

LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_API_URL: str = os.environ.get("LLM_API_URL", "")
HUMANE_PROXY_API_KEY: str = os.environ.get(
    "HUMANE_PROXY_API_KEY",
    ""
)

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from humane_proxy.config import get_config
        from humane_proxy.classifiers.pipeline import SafetyPipeline
        _pipeline = SafetyPipeline(get_config())
    return _pipeline


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    init_db()

    config = get_config()
    setup_telemetry(config)

    if not LLM_API_URL:
        logger.warning(
            "[HumaneProxy] LLM_API_URL is not configured."
        )
    _get_pipeline()
    logger.info("[HumaneProxy] Database initialised. Server is ready.")

    # Mount admin router.
    try:
        from humane_proxy.api.admin import router as admin_router
        app.include_router(admin_router)
        logger.info("[HumaneProxy] Admin API mounted at /admin")
    except Exception:
        logger.warning("[HumaneProxy] Admin API could not be loaded.")

    yield

    shutdown_telemetry()

app = FastAPI(
    title="HumaneProxy",
    version="0.4.0",
    description="Lightweight AI safety middleware that protects humans.",
    lifespan=_lifespan,
)

_REQUEST_COUNT = 0
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    global _REQUEST_COUNT
    _REQUEST_COUNT += 1
    start = time.perf_counter()

    tracer = _get_tracer()

    span_ctx = (
        tracer.start_as_current_span("humane_proxy.http.request")
        if tracer is not None
        else nullcontext()
    )
    
    with span_ctx as span:

        try:
            response = await call_next(request)

            if Status and StatusCode:
                span.set_status(Status(StatusCode.OK))

        except Exception as exc:
            if span is not None:
                span.record_exception(exc)

            if span is not None and Status and StatusCode:
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        str(exc),
                    )
                )

            raise

        elapsed_ms = (time.perf_counter() - start) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = f"{elapsed_ms:.2f}"

    logger.info(
        "[%s] %s %s -> %s (%.2fms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )

    return response

def _resolve_session_id(payload: dict[str, Any], request: Request) -> str:
    return payload.get("session_id") or (
        request.client.host if request.client else "unknown"
    )


def _extract_last_user_message(payload: dict[str, Any]) -> str:
    messages: list[dict[str, str]] = payload.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""

@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "HumaneProxy",
    }

_REQUEST_COUNT = 0

@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "HumaneProxy",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/metrics")
async def metrics():
    return {
        "requests_total": _REQUEST_COUNT,
    }

def _authorize(request: Request) -> bool:
    if not HUMANE_PROXY_API_KEY:
        return True

    auth = request.headers.get("Authorization", "")

    if not auth.startswith("Bearer "):
        return False

    token = auth.replace("Bearer ", "").strip()

    return token == HUMANE_PROXY_API_KEY

@app.post("/chat")
async def chat(request: Request) -> JSONResponse:
    """Intercept a chat request, evaluate safety, then forward or respond."""
    payload: dict[str, Any] = await request.json()

    if not _authorize(request):
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "message": "Unauthorized",
            },
        )
    
    session_id = _resolve_session_id(payload, request)
    span = trace.get_current_span() if trace else None
    if span is not None:
        import hashlib

        safe_session = hashlib.sha256(
            session_id.encode("utf-8")
        ).hexdigest()

        _set_attr(
            span,
            "humane_proxy.session_id",
            safe_session,
        )

    user_message = _extract_last_user_message(payload)

    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "No user message found in payload."},
        )

    pipeline = _get_pipeline()
    result = await pipeline.classify(user_message, session_id)

    _set_attr(
        span,
        "humane_proxy.final_score",
        result.classification.score,
    )

    _set_attr(
        span,
        "humane_proxy.category",
        result.classification.category,
    )

    _set_attr(
        span,
        "humane_proxy.stage_reached",
        result.classification.stage,
    )

    if result.should_escalate:
        cls = result.classification

        esc = escalate(
            session_id,
            cls.score,
            cls.triggers,
            cls.category,
            message_hash=result.message_hash,
            stage_reached=cls.stage,
            reasoning=cls.reasoning,
        )

        # Self-harm: return care response instead of generic flagged message.
        if cls.category == "self_harm":
            care = get_self_harm_response(payload)

            if care["mode"] == "block":
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "care_response",
                        "category": "self_harm",
                        "message": care["message"],
                        "escalation": esc,
                    },
                )
            else:
                # Forward mode: inject care context and pass to LLM.
                payload = care["payload"]

        else:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "flagged",
                    "category": cls.category,
                    "message": "Content flagged for review.",
                    "stage_reached": cls.stage,
                    "escalation": esc,
                },
            )

    # Safe (or forward mode) — forward to upstream LLM.
    if not LLM_API_URL:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "LLM_API_URL is not configured."},
        )

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient() as client:
            llm_response = await client.post(
                LLM_API_URL, 
                headers=headers, json=payload, 
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=30.0,
                    write=10.0,
                    pool=5.0,
                )
            )
        try:
            body = llm_response.json()
        except (ValueError, TypeError):
            body = {
                "status": "error",
                "message": f"Upstream returned non-JSON (HTTP {llm_response.status_code}).",
                "raw": llm_response.text[:500],
            }
        return JSONResponse(status_code=llm_response.status_code, content=body)

    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={
                "status": "error",
                "message": "Upstream LLM timed out.",
            },
        )
    
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": f"Upstream LLM unavailable: {type(exc).__name__}: {exc}",
            },
        )
