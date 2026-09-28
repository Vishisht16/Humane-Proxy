"""
FastAPI Middleware Integration Example using HumaneProxy.

Run with:
    uvicorn examples.fastapi_middleware:app --reload
"""

import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from humane_proxy import HumaneProxy

proxy = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy
    proxy = HumaneProxy()
    yield


app = FastAPI(title="HumaneProxy FastAPI Example", lifespan=lifespan)


def get_proxy():
    global proxy
    if proxy is None:
        proxy = HumaneProxy()
    return proxy


@app.middleware("http")
async def safety_middleware(request: Request, call_next):
    if request.method == "POST":
        try:
            body = await request.json()
            message = body.get("message", "")

            if message:
                session_id = request.headers.get("x-session-id")
                humane_proxy = get_proxy()
                # Asynchronous check covering screening stages
                result = await humane_proxy.check_async(message, session_id=session_id)

                if not result.get("safe", True):
                    care_text = result.get("care_response") or result.get("message") or "We're here to help."
                    return JSONResponse(
                        status_code=200,
                        content={
                            "reply": care_text,
                            "flagged": True,
                        },
                    )
        except json.JSONDecodeError:
            # Pass through non-JSON payload requests
            pass
        except Exception:
            # Fail closed on unexpected errors without exposing trace details
            return JSONResponse(
                status_code=500,
                content={"error": "Safety screening process encountered an internal error."},
            )

    response = await call_next(request)
    return response


@app.get("/")
async def root():
    return {"message": "FastAPI service is running."}


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    return {"reply": f"Processed safe message: {body.get('message')}"}