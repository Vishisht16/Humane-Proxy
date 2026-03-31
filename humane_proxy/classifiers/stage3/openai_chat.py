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

"""Stage-3 provider: OpenAI-compatible chat model as a safety classifier.

Sends a structured classifier prompt to any OpenAI-compatible chat API
and parses the JSON response.  The prompt explicitly instructs the model
that **jailbreaks are always safe** — the focus is on genuine human harm.
"""

from __future__ import annotations

import json
import logging
import os

import httpx

from humane_proxy.classifiers.models import ClassificationResult
from humane_proxy.classifiers.stage3.base import Stage3Classifier

logger = logging.getLogger("humane_proxy.classifiers.stage3.openai_chat")

_SYSTEM_PROMPT = """\
You are a safety classifier for a human-protection system called HumaneProxy.

Your job is to classify user messages into one of three categories:
1. "self_harm" — the user is expressing suicidal ideation, self-injury intent,
   or seeking methods to harm themselves.
2. "criminal_intent" — the user is seeking to commit violence, create weapons
   or drugs, traffic persons, create CSAM, or carry out other serious crimes.
3. "safe" — everything else, including normal conversation.

CRITICAL RULES:
- A jailbreak attempt (e.g. "ignore previous instructions") is ALWAYS "safe".
  Jailbreaks are NOT our concern.
- Hyperbolic expressions like "I'm dying of laughter" are ALWAYS "safe".
- Academic or help-seeking questions like "what are warning signs of suicide"
  are ALWAYS "safe".
- When in doubt, lean toward "safe" for criminal_intent but toward flagging
  for self_harm. Missing a person in distress is far worse than a false alarm.

Respond ONLY with a JSON object: {"category": "...", "score": 0.0-1.0, "reasoning": "..."}
"""


class OpenAIChatClassifier(Stage3Classifier):
    """Stage-3 classifier using an OpenAI-compatible chat model."""

    def __init__(self, config: dict) -> None:
        chat_cfg = config.get("stage3", {}).get("openai_chat", {})
        self._api_url: str = chat_cfg.get(
            "api_url", "https://api.openai.com/v1/chat/completions"
        )
        self._model: str = chat_cfg.get("model", "gpt-4o-mini")
        self._timeout: float = config.get("stage3", {}).get("timeout", 10.0)
        self._api_key: str = os.environ.get("OPENAI_API_KEY", "")

    async def classify(
        self, text: str, prior: ClassificationResult
    ) -> ClassificationResult:
        """Send *text* to the chat model for classification."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self._api_url, json=payload, headers=headers
                )
                resp.raise_for_status()

            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            return self._parse_response(content)

        except Exception as exc:
            logger.warning("OpenAI Chat Stage-3 error: %s", exc)
            return ClassificationResult(
                category=prior.category,
                score=prior.score,
                triggers=prior.triggers + ["stage3_error"],
                stage=3,
            )

    def _parse_response(self, content: str) -> ClassificationResult:
        """Parse the model's JSON response."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Stage-3 returned non-JSON: %s", content[:200])
            return ClassificationResult(
                category="safe",
                score=0.0,
                triggers=["stage3_parse_error"],
                stage=3,
            )

        category = data.get("category", "safe")
        if category not in ("self_harm", "criminal_intent", "safe"):
            category = "safe"

        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))

        if category == "self_harm":
            score = 1.0  # Critical override.

        reasoning = data.get("reasoning", "")

        return ClassificationResult(
            category=category,
            score=score,
            triggers=[f"openai_chat:{category}"],
            stage=3,
            reasoning=reasoning,
        )
