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

"""Stage-3 provider: OpenAI Moderation API.

Uses the **free** ``/v1/moderations`` endpoint (available with any
OpenAI API key).  Maps OpenAI's moderation categories to HumaneProxy
categories.
"""

from __future__ import annotations

import logging
import os

import httpx

from humane_proxy.classifiers.models import ClassificationResult
from humane_proxy.classifiers.stage3.base import Stage3Classifier

logger = logging.getLogger("humane_proxy.classifiers.stage3.openai_mod")


class OpenAIModerationClassifier(Stage3Classifier):
    """Stage-3 classifier using OpenAI's Moderation API."""

    def __init__(self, config: dict) -> None:
        mod_cfg = config.get("stage3", {}).get("openai_moderation", {})
        self._api_url: str = mod_cfg.get(
            "api_url", "https://api.openai.com/v1/moderations"
        )
        self._timeout: float = config.get("stage3", {}).get("timeout", 10.0)
        self._api_key: str = os.environ.get("OPENAI_API_KEY", "")

    async def classify(
        self, text: str, prior: ClassificationResult
    ) -> ClassificationResult:
        """Send *text* to the OpenAI Moderation endpoint."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"input": text}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    self._api_url, json=payload, headers=headers
                )
                resp.raise_for_status()

            body = resp.json()
            return self._parse_response(body)

        except Exception as exc:
            logger.warning("OpenAI Moderation Stage-3 error: %s", exc)
            return ClassificationResult(
                category=prior.category,
                score=prior.score,
                triggers=prior.triggers + ["stage3_error"],
                stage=3,
            )

    def _parse_response(self, body: dict) -> ClassificationResult:
        """Parse the moderation API response.

        Maps OpenAI categories to ours:
        - ``self-harm``, ``self-harm/intent``, ``self-harm/instructions`` → self_harm
        - ``violence``, ``violence/graphic`` → criminal_intent
        - ``sexual/minors`` → criminal_intent
        """
        results = body.get("results", [{}])[0]
        categories: dict = results.get("categories", {})
        scores: dict = results.get("category_scores", {})

        flagged = results.get("flagged", False)
        if not flagged:
            return ClassificationResult(
                category="safe",
                score=0.0,
                triggers=["openai_moderation:safe"],
                stage=3,
                reasoning="OpenAI Moderation: not flagged",
            )

        # Determine our category from OpenAI's categories.
        detected_triggers: list[str] = []
        category = "safe"
        max_score = 0.0

        # Self-harm checks (highest priority).
        for key in ("self-harm", "self-harm/intent", "self-harm/instructions"):
            if categories.get(key, False):
                category = "self_harm"
                score_val = scores.get(key, 0.9)
                max_score = max(max_score, score_val)
                detected_triggers.append(f"openai_mod:{key}")

        # Violence / criminal checks.
        if category == "safe":
            for key in ("violence", "violence/graphic", "sexual/minors"):
                if categories.get(key, False):
                    category = "criminal_intent"
                    score_val = scores.get(key, 0.8)
                    max_score = max(max_score, score_val)
                    detected_triggers.append(f"openai_mod:{key}")

        # Other flags (harassment, etc.) — log but treat as safe for now.
        if category == "safe" and flagged:
            for key, val in categories.items():
                if val:
                    detected_triggers.append(f"openai_mod:{key}")
            # Keep as safe — other moderation flags aren't our domain.

        if category == "self_harm":
            max_score = 1.0  # Critical override.

        return ClassificationResult(
            category=category,
            score=min(1.0, max_score),
            triggers=detected_triggers or ["openai_moderation:flagged"],
            stage=3,
            reasoning=f"OpenAI Moderation flagged: {', '.join(detected_triggers)}",
        )
