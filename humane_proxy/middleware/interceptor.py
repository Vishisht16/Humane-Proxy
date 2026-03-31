"""HumaneProxy interceptor — FastAPI proxy with full safety pipeline."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from humane_proxy.escalation.local_db import init_db
from humane_proxy.escalation.router import escalate

logger = logging.getLogger("humane_proxy")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_API_URL: str = os.environ.get("LLM_API_URL", "")


# ---------------------------------------------------------------------------
# Pipeline singleton (lazily initialised)
# ---------------------------------------------------------------------------

_pipeline = None


def _get_pipeline():
    """Return the singleton SafetyPipeline instance."""
    global _pipeline
    if _pipeline is None:
        from humane_proxy.config import get_config
        from humane_proxy.classifiers.pipeline import SafetyPipeline

        _pipeline = SafetyPipeline(get_config())
    return _pipeline


# ---------------------------------------------------------------------------
# FastAPI lifespan — runs init_db() on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise resources on startup; clean up on shutdown."""
    init_db()
    _get_pipeline()  # Warm up the pipeline (triggers Stage-3 warning if needed).
    logger.info("[HumaneProxy] Database initialised.  Server is ready.")
    yield
    # Shutdown — nothing to tear down in v0.2.


app = FastAPI(
    title="HumaneProxy",
    version="0.2.0",
    description="Lightweight AI safety middleware that protects humans.",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_session_id(payload: dict[str, Any], request: Request) -> str:
    """Derive a session identifier from the request."""
    return payload.get("session_id") or (
        request.client.host if request.client else "unknown"
    )


def _extract_last_user_message(payload: dict[str, Any]) -> str:
    """Pull the last user message from an OpenAI-style messages array."""
    messages: list[dict[str, str]] = payload.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(request: Request) -> JSONResponse:
    """Intercept a chat request, evaluate safety, then forward or flag."""
    payload: dict[str, Any] = await request.json()

    session_id = _resolve_session_id(payload, request)
    user_message = _extract_last_user_message(payload)

    if not user_message:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "No user message found in payload.",
            },
        )

    # --- Full async safety pipeline ---
    pipeline = _get_pipeline()
    result = await pipeline.classify(user_message, session_id)

    # --- Escalation decision ---
    if result.should_escalate:
        esc = escalate(
            session_id,
            result.classification.risk_score if hasattr(result.classification, 'risk_score') else result.classification.score,
            result.classification.triggers,
            result.classification.category,
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "flagged",
                "category": result.classification.category,
                "message": "Content flagged for review.",
                "stage_reached": result.classification.stage,
                "escalation": esc,
            },
        )

    # --- Safe — forward to upstream LLM ---
    if not LLM_API_URL:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "LLM_API_URL is not configured.",
            },
        )

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient() as client:
            llm_response = await client.post(
                LLM_API_URL,
                headers=headers,
                json=payload,
                timeout=30.0,
            )

        # Guard against non-JSON responses (e.g. HTML 502 gateway pages).
        try:
            body = llm_response.json()
        except (ValueError, TypeError):
            body = {
                "status": "error",
                "message": f"Upstream returned non-JSON (HTTP {llm_response.status_code}).",
                "raw": llm_response.text[:500],
            }

        return JSONResponse(
            status_code=llm_response.status_code,
            content=body,
        )

    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": (
                    f"Upstream LLM unavailable: {type(exc).__name__}: {exc}"
                ),
            },
        )
