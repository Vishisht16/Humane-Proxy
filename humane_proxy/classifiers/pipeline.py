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

"""3-stage cascade safety pipeline orchestrator.

Stage 1 — Heuristics (always runs, < 1 ms)
Stage 2 — Semantic embeddings (optional, ~100 ms local, behind [ml] flag)
Stage 3 — Reasoning LLM (optional, async, auto-detected from API keys)

Each stage is independently configurable and skippable.  The pipeline
uses a **max-score** combination strategy with **self_harm priority**:
if any stage detects self-harm, that category wins regardless of score.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from humane_proxy.classifiers.models import (
    ClassificationResult,
    PipelineResult,
    TrajectoryResult,
)

logger = logging.getLogger("humane_proxy.pipeline")

_stage3_warning_shown = False


class SafetyPipeline:
    """Orchestrates the 3-stage safety classification cascade.

    Usage::

        pipeline = SafetyPipeline(config)

        # Async (all 3 stages):
        result = await pipeline.classify("I want to end my life", "session-1")

        # Sync (stages 1+2 only):
        result = pipeline.classify_sync("Hello world", "session-1")
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        pipeline_cfg = config.get("pipeline", {})
        safety_cfg = config.get("safety", {})

        self.enabled_stages: list[int] = pipeline_cfg.get("enabled_stages", [1])
        self.stage1_ceiling: float = pipeline_cfg.get("stage1_ceiling", 0.3)
        self.stage2_ceiling: float = pipeline_cfg.get("stage2_ceiling", 0.4)
        self.spike_boost: float = safety_cfg.get("spike_boost", 0.25)
        self.risk_threshold: float = safety_cfg.get("risk_threshold", 0.7)
        self.store_message_text: bool = config.get("privacy", {}).get(
            "store_message_text", False
        )

        # Stage 2: embedding classifier.
        self._stage2: Any = None
        if 2 in self.enabled_stages:
            self._init_stage2()

        # Stage 3: reasoning LLM.
        self._stage3: Any = None
        if 3 in self.enabled_stages:
            self._init_stage3()

        self._show_stage3_warning()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_stage2(self) -> None:
        """Instantiate the embedding classifier (lazy model load)."""
        try:
            from humane_proxy.classifiers.embedding_classifier import (
                EmbeddingClassifier,
            )

            self._stage2 = EmbeddingClassifier(self._config)
        except Exception:
            logger.exception("Failed to initialise Stage-2 embedding classifier")
            self._stage2 = None

    def _init_stage3(self) -> None:
        """Instantiate the Stage-3 provider based on config / API keys."""
        provider = self._config.get("stage3", {}).get("provider", "auto")

        if provider == "none":
            self._stage3 = None
            return

        # Auto-detection.
        if provider == "auto":
            if os.environ.get("OPENAI_API_KEY"):
                provider = "openai_moderation"
            elif os.environ.get("GROQ_API_KEY"):
                provider = "llamaguard"
            else:
                self._stage3 = None
                return

        try:
            if provider == "openai_moderation":
                from humane_proxy.classifiers.stage3.openai_moderation import (
                    OpenAIModerationClassifier,
                )

                self._stage3 = OpenAIModerationClassifier(self._config)
                logger.info("Stage-3 provider: OpenAI Moderation API")

            elif provider == "llamaguard":
                from humane_proxy.classifiers.stage3.llamaguard import (
                    LlamaGuardClassifier,
                )

                self._stage3 = LlamaGuardClassifier(self._config)
                logger.info("Stage-3 provider: LlamaGuard (Groq)")

            elif provider == "openai_chat":
                from humane_proxy.classifiers.stage3.openai_chat import (
                    OpenAIChatClassifier,
                )

                self._stage3 = OpenAIChatClassifier(self._config)
                logger.info("Stage-3 provider: OpenAI Chat")

            else:
                logger.warning("Unknown Stage-3 provider: %s", provider)
                self._stage3 = None

        except Exception:
            logger.exception("Failed to initialise Stage-3 provider: %s", provider)
            self._stage3 = None

    def _show_stage3_warning(self) -> None:
        """Log a clear guidance message when Stage 3 is not active."""
        global _stage3_warning_shown
        if self._stage3 is not None or _stage3_warning_shown:
            return
        _stage3_warning_shown = True
        logger.warning(
            "\n"
            "⚠️  Stage-3 classification is DISABLED (no API key detected).\n"
            "    For stronger protection, set up a Stage-3 provider:\n"
            "\n"
            "    Option A — OpenAI Moderation (free with any OpenAI key):\n"
            "      export OPENAI_API_KEY=sk-...\n"
            "\n"
            "    Option B — LlamaGuard via Groq (free tier):\n"
            "      export GROQ_API_KEY=gsk_...\n"
            "      # Set in humane_proxy.yaml:\n"
            "      # stage3:\n"
            "      #   provider: \"llamaguard\"\n"
            "\n"
            "    See: https://github.com/Vishisht16/Humane-Proxy#stage-3-setup"
        )

    # ------------------------------------------------------------------
    # Classification — async (all 3 stages)
    # ------------------------------------------------------------------

    async def classify(
        self, text: str, session_id: str
    ) -> PipelineResult:
        """Run the full async pipeline (Stages 1 + 2 + 3)."""
        # Stage 1 — Heuristics (always).
        result = self._run_stage1(text)

        # Early exit: clear dangerous (self_harm).
        if result.category == "self_harm":
            return self._finalize(result, session_id, text)

        # Early exit: clear safe.
        if result.score <= self.stage1_ceiling and result.category == "safe":
            return self._finalize(result, session_id, text)

        # Stage 2 — Embeddings (if enabled).
        if 2 in self.enabled_stages and self._stage2 is not None:
            s2 = self._stage2.classify(text)
            result = self._combine(result, s2)

            # Early exit after Stage 2.
            if result.category == "self_harm":
                return self._finalize(result, session_id, text)
            if result.score <= self.stage2_ceiling and result.category == "safe":
                return self._finalize(result, session_id, text)

        # Stage 3 — Reasoning LLM (if enabled).
        if 3 in self.enabled_stages and self._stage3 is not None:
            try:
                s3 = await self._stage3.classify(text, result)
                result = self._combine(result, s3)
            except Exception:
                logger.exception("Stage-3 classification failed")
                result.triggers.append("stage3_error")

        return self._finalize(result, session_id, text)

    # ------------------------------------------------------------------
    # Classification — sync (Stages 1 + 2 only)
    # ------------------------------------------------------------------

    def classify_sync(
        self, text: str, session_id: str
    ) -> PipelineResult:
        """Run the synchronous pipeline (Stages 1 + 2 only — no async)."""
        result = self._run_stage1(text)

        if result.category == "self_harm":
            return self._finalize(result, session_id, text)

        if result.score <= self.stage1_ceiling and result.category == "safe":
            return self._finalize(result, session_id, text)

        if 2 in self.enabled_stages and self._stage2 is not None:
            s2 = self._stage2.classify(text)
            result = self._combine(result, s2)

        return self._finalize(result, session_id, text)

    # ------------------------------------------------------------------
    # Stage 1 wrapper
    # ------------------------------------------------------------------

    @staticmethod
    def _run_stage1(text: str) -> ClassificationResult:
        """Run the heuristic classifier and wrap in a ClassificationResult."""
        from humane_proxy.classifiers.heuristics import classify

        category, score, triggers = classify(text)
        return ClassificationResult(
            category=category,
            score=score,
            triggers=triggers,
            stage=1,
        )

    # ------------------------------------------------------------------
    # Combination
    # ------------------------------------------------------------------

    @staticmethod
    def _combine(
        a: ClassificationResult, b: ClassificationResult
    ) -> ClassificationResult:
        """Combine two stage results.

        Strategy:
        - Self-harm always wins if either result indicates it.
        - Otherwise the higher-scoring category wins.
        - Triggers are merged (deduplicated, order preserved).
        """
        # Merge triggers.
        seen: set[str] = set()
        merged_triggers: list[str] = []
        for t in a.triggers + b.triggers:
            if t not in seen:
                seen.add(t)
                merged_triggers.append(t)

        merged_score = max(a.score, b.score)

        # Category priority: self_harm > criminal_intent > safe.
        if "self_harm" in (a.category, b.category):
            category = "self_harm"
        elif "criminal_intent" in (a.category, b.category):
            category = "criminal_intent"
        else:
            category = "safe"

        # Pick reasoning from the most relevant stage.
        reasoning = b.reasoning or a.reasoning

        return ClassificationResult(
            category=category,
            score=merged_score,
            triggers=merged_triggers,
            stage=max(a.stage, b.stage),
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(
        self,
        result: ClassificationResult,
        session_id: str,
        text: str,
    ) -> PipelineResult:
        """Apply trajectory, self-harm override, and escalation logic."""
        from humane_proxy.risk.trajectory import analyze

        # Trajectory analysis.
        traj = analyze(session_id, result.score, result.category)

        # Spike boost.
        if traj.spike_detected:
            result.score = min(result.score + self.spike_boost, 1.0)
            result.triggers.append("trajectory_spike")

        # Self-harm critical override.
        if result.category == "self_harm":
            result.score = 1.0

        result.score = min(result.score, 1.0)

        # Escalation decision.
        should_escalate = (
            result.category == "self_harm"
            or (
                result.category == "criminal_intent"
                and result.score >= self.risk_threshold
            )
        )
        should_block = should_escalate

        # Privacy: hash the message.
        message_hash: str | None = None
        if not self.store_message_text:
            message_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        return PipelineResult(
            classification=result,
            trajectory=traj,
            should_escalate=should_escalate,
            should_block=should_block,
            message_hash=message_hash,
        )
