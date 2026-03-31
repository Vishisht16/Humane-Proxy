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

"""Stage-3 provider: LlamaGuard via any OpenAI-compatible endpoint.

Uses LlamaGuard-3-8B (or compatible) through Groq, Together, or a
local vLLM server.  Maps LlamaGuard safety categories to HumaneProxy
categories.
"""

from __future__ import annotations

import logging
import os

import httpx

from humane_proxy.classifiers.models import ClassificationResult
from humane_proxy.classifiers.stage3.base import Stage3Classifier

logger = logging.getLogger("humane_proxy.classifiers.stage3.llamaguard")

# LlamaGuard category → HumaneProxy category mapping.
# See: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama-guard-3/
_LLAMAGUARD_MAP: dict[str, str] = {
    "S1": "criminal_intent",   # Violent Crimes
    "S2": "criminal_intent",   # Non-Violent Crimes
    "S3": "criminal_intent",   # Sex-Related Crimes
    "S4": "criminal_intent",   # Child Sexual Exploitation
    "S5": "criminal_intent",   # Defamation
    "S6": "criminal_intent",   # Specialized Advice
    "S7": "safe",              # Privacy (not our domain)
    "S8": "safe",              # Intellectual Property (not our domain)
    "S9": "safe",              # Indiscriminate Weapons → criminal_intent
    "S10": "self_harm",        # Hate → could be self-harm adjacent
    "S11": "self_harm",        # Suicide & Self-Harm
    "S12": "criminal_intent",  # Sexual Content (minors)
    "S13": "safe",             # Elections (not our domain)
    "S14": "safe",             # Code Interpreter Abuse
}


class LlamaGuardClassifier(Stage3Classifier):
    """Stage-3 classifier using LlamaGuard-3 via an OpenAI-compatible API."""

    def __init__(self, config: dict) -> None:
        lg_cfg = config.get("stage3", {}).get("llamaguard", {})
        self._api_url: str = lg_cfg.get(
            "api_url", "https://api.groq.com/openai/v1/chat/completions"
        )
        self._model: str = lg_cfg.get("model", "meta-llama/llama-guard-3-8b")
        self._timeout: float = config.get("stage3", {}).get("timeout", 10.0)

        # API key: try GROQ_API_KEY first, then LLM_API_KEY.
        self._api_key: str = os.environ.get(
            "GROQ_API_KEY", os.environ.get("LLM_API_KEY", "")
        )

    async def classify(
        self, text: str, prior: ClassificationResult
    ) -> ClassificationResult:
        """Send *text* to LlamaGuard and parse the safety verdict."""
        messages = [
            {"role": "user", "content": text},
        ]
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 100,
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
            output = body["choices"][0]["message"]["content"].strip()
            return self._parse_output(output)

        except Exception as exc:
            logger.warning("LlamaGuard Stage-3 error: %s", exc)
            return ClassificationResult(
                category=prior.category,
                score=prior.score,
                triggers=prior.triggers + ["stage3_error"],
                stage=3,
            )

    def _parse_output(self, output: str) -> ClassificationResult:
        """Parse LlamaGuard's text output into a ClassificationResult.

        LlamaGuard outputs either:
        - ``"safe"``
        - ``"unsafe\\nS1,S2"``  (one or more category codes)
        """
        lines = output.strip().split("\n")
        verdict = lines[0].strip().lower()

        if verdict == "safe":
            return ClassificationResult(
                category="safe",
                score=0.0,
                triggers=["llamaguard:safe"],
                stage=3,
                reasoning="LlamaGuard verdict: safe",
            )

        # Parse category codes from the second line.
        categories_found: list[str] = []
        if len(lines) > 1:
            codes = [c.strip() for c in lines[1].split(",")]
            categories_found = [
                _LLAMAGUARD_MAP.get(c, "safe") for c in codes
            ]

        # Priority: self_harm > criminal_intent > safe
        if "self_harm" in categories_found:
            category = "self_harm"
        elif "criminal_intent" in categories_found:
            category = "criminal_intent"
        else:
            category = "safe"

        score = 1.0 if category == "self_harm" else 0.85

        return ClassificationResult(
            category=category,
            score=score,
            triggers=[f"llamaguard:{category}"],
            stage=3,
            reasoning=f"LlamaGuard verdict: {output.strip()}",
        )
