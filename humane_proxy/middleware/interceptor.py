"""HumaneProxy interceptor — FastAPI proxy with full safety pipeline + care response."""

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
from humane_proxy.escalation.router import escalate, get_self_harm_response

logger = logging.getLogger("humane_proxy")

LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
LLM_API_URL: str = os.environ.get("LLM_API_URL", "")

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


app = FastAPI(
    title="HumaneProxy",
    version="0.2.3",
    description="Lightweight AI safety middleware that protects humans.",
    lifespan=_lifespan,
)


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


@app.post("/chat")
async def chat(request: Request) -> JSONResponse:
    """Intercept a chat request, evaluate safety, then forward or respond."""
    payload: dict[str, Any] = await request.json()

    session_id = _resolve_session_id(payload, request)
    user_message = _extract_last_user_message(payload)

    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "No user message found in payload."},
        )

    pipeline = _get_pipeline()
    result = await pipeline.classify(user_message, session_id)

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
                LLM_API_URL, headers=headers, json=payload, timeout=30.0
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

    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": f"Upstream LLM unavailable: {type(exc).__name__}: {exc}",
            },
        )
