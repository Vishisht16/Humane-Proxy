"""
OpenAI API Client Wrapper Example using HumaneProxy.

Run with:
    python examples/openai_proxy_wrapper.py
"""

import os
from humane_proxy import HumaneProxy

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False

proxy = HumaneProxy()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock-key")) if HAS_OPENAI else None


def safe_chat_completion(messages: list, model: str = "gpt-3.5-turbo", session_id: str = None):
    """
    Wrapper around openai.chat.completions.create that screens user prompts first.
    """
    if not HAS_OPENAI:
        raise ImportError("openai package is required to use this wrapper. Install via 'pip install openai'.")

    # Extract latest user message
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if user_message:
        screening = proxy.check(user_message, session_id=session_id)
        if not screening.get("safe", True):
            return {
                "flagged": True,
                "reply": screening.get("care_response", "We're here to help."),
            }

    # Proceed with OpenAI call if safe
    response = client.chat.completions.create(model=model, messages=messages)
    return {"flagged": False, "response": response}


if __name__ == "__main__":
    test_messages = [{"role": "user", "content": "Hello! How can I stay safe online?"}]
    print("Testing Wrapper with Safe Input:")
    result = safe_chat_completion(test_messages)
    print(result)