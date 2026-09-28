"""
Flask before_request Hook Integration Example using HumaneProxy.

Run with:
    python examples/flask_integration.py
"""

import os
from flask import Flask, jsonify, request
from humane_proxy import HumaneProxy

app = Flask(__name__)
proxy = None


def get_proxy():
    global proxy
    if proxy is None:
        proxy = HumaneProxy()
    return proxy


@app.before_request
def screen_request():
    # Only screen POST requests with JSON payload on protected routes
    if request.method == "POST" and request.is_json:
        data = request.get_json() or {}
        message = data.get("message", "")

        if message:
            session_id = request.headers.get("x-session-id")
            humane_proxy = get_proxy()
            # Synchronous check (Stages 1 + 2)
            result = humane_proxy.check(message, session_id=session_id)

            if not result.get("safe", True):
                care_text = result.get("care_response") or result.get("message") or "We're here to help."
                return (
                    jsonify(
                        {
                            "reply": care_text,
                            "flagged": True,
                        }
                    ),
                    200,
                )


@app.route("/")
def index():
    return jsonify({"message": "Flask service is running."})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    return jsonify({"reply": f"Processed safe message: {data.get('message')}"})


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
    app.run(port=5000, debug=debug_mode)