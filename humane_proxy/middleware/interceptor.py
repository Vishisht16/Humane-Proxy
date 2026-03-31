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

from humane_proxy import load_config
from humane_proxy.classifiers.heuristics import classify
from humane_proxy.escalation.local_db import init_db
from humane_proxy.escalation.router import escalate
from humane_proxy.risk.trajectory import detect_spike

logger = logging.getLogger("humane_proxy")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CFG: dict = load_config()
_RISK_THRESHOLD: float = _CFG.get("safety", {}).get("risk_threshold", 0.7)
_SPIKE_BOOST: float = _CFG.get("safety", {}).get("spike_boost", 0.25)

LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_API_URL: str = os.environ.get("LLM_API_URL", "")


# ---------------------------------------------------------------------------
# FastAPI lifespan — runs init_db() on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise resources on startup; clean up on shutdown."""
    init_db()
    logger.info("[HumaneProxy] Database initialised.  Server is ready.")
    yield
    # Shutdown — nothing to tear down in v0.1.


app = FastAPI(
    title="HumaneProxy",
    version="0.1.0",
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
# Safety pipeline
# ---------------------------------------------------------------------------

def _run_safety_pipeline(
    user_message: str,
    session_id: str,
) -> tuple[str, float, list[str]]:
    """Execute the full safety evaluation pipeline.

    Order
    -----
    1. Heuristic classification (category-aware)
    2. Trajectory spike detection
    3. Score boost if spike detected
    4. Escalation decision (handled by the caller)
    """
    # Step 1 — Heuristic classifier
    category, score, triggers = classify(user_message)
    triggers = triggers or []

    # Step 2 — Trajectory spike detection
    is_spike: bool = detect_spike(session_id, score)

    # Step 3 — Boost score on spike and record the trigger
    if is_spike:
        score = min(score + _SPIKE_BOOST, 1.0)
        triggers.append("trajectory_spike")

    return category, score, triggers


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

    # --- Safety pipeline ---
    category, risk_score, triggers = _run_safety_pipeline(user_message, session_id)

    # --- Step 4: Escalation decision ---
    # Self-harm: always escalate
    if category == "self_harm":
        result = escalate(session_id, risk_score, triggers, category)
        return JSONResponse(
            status_code=200,
            content={
                "status": "flagged",
                "category": "self_harm",
                "message": "Content flagged for review.",
                "escalation": result,
            },
        )

    # Criminal intent: escalate if above threshold
    if category == "criminal_intent" and risk_score >= _RISK_THRESHOLD:
        result = escalate(session_id, risk_score, triggers, category)
        return JSONResponse(
            status_code=200,
            content={
                "status": "flagged",
                "category": "criminal_intent",
                "message": "Content flagged for review.",
                "escalation": result,
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
