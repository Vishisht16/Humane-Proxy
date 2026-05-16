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
from contextlib import nullcontext

# Optional OpenTelemetry support.
try:
    from opentelemetry import trace
    from opentelemetry.trace import (
        Status,
        StatusCode,
    )

except ImportError:
    trace = None
    Status = None
    StatusCode = None

from humane_proxy.classifiers.models import (
    ClassificationResult,
    PipelineResult,
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

        if trace is not None:
            self._tracer = trace.get_tracer("humane_proxy.pipeline")
        else:
            self._tracer = None

        # Stage 2: embedding classifier.
        self._stage2: Any = None
        if 2 in self.enabled_stages:
            self._init_stage2()

        # Stage 3: reasoning LLM.
        self._stage3: Any = None
        if 3 in self.enabled_stages:
            self._init_stage3()

        self._show_stage3_warning()

    def _span(self, name: str):
        """
        Safe span helper.

        Returns:
            Active span context manager when telemetry exists.
            nullcontext() when telemetry disabled.
        """

        if self._tracer is None:
            return nullcontext()

        return self._tracer.start_as_current_span(name)

    def _set_attr(self, span, key: str, value):
        if span is not None:
            span.set_attribute(key, value)

    def _set_status(self, span, status):
        if span is not None:
            span.set_status(status)

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
        """Log a clear guidance message when Stage 3 is not active.

        Only shown when the user has explicitly enabled Stage 3 in their
        ``enabled_stages`` config but no provider could be initialised
        (typically because no API key is set).
        """
        global _stage3_warning_shown
        if 3 not in self.enabled_stages:
            return  # User didn't ask for Stage 3 — no warning needed.
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
            '      #   provider: "llamaguard"\n'
            "\n"
            "    See: https://github.com/Vishisht16/Humane-Proxy#stage-3-setup"
        )

    # ------------------------------------------------------------------
    # Classification — async (all 3 stages)
    # ------------------------------------------------------------------

    async def classify(self, text: str, session_id: str) -> PipelineResult:
        """Run the full async pipeline (Stages 1 + 2 + 3)."""

        with self._span("humane_proxy.pipeline.classify") as span:
            safe_session = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
            self._set_attr(span, "humane_proxy.session_id", safe_session)
            # Stage 1 — Heuristics
            # ------------------------------------------------------------------

            with self._span("humane_proxy.stage1") as stage1_span:
                result = self._run_stage1(text)

                self._set_attr(
                    stage1_span,
                    "humane_proxy.category",
                    result.category,
                )
                self._set_attr(
                    stage1_span,
                    "humane_proxy.final_score",
                    result.score,
                )
                self._set_attr(
                    stage1_span,
                    "humane_proxy.triggers_count",
                    len(result.triggers),
                )

                # Early exit: clear dangerous (self_harm).
                if result.category == "self_harm":
                    final = self._finalize(result, session_id, text)

                    self._set_attr(
                        stage1_span,
                        "humane_proxy.category",
                        final.classification.category,
                    )
                    self._set_attr(
                        stage1_span,
                        "humane_proxy.score",
                        final.classification.score,
                    )
                    self._set_attr(
                        stage1_span,
                        "humane_proxy.triggers_count",
                        len(final.classification.triggers),
                    )
                    self._set_attr(
                        stage1_span,
                        "humane_proxy.stage_reached",
                        final.classification.stage,
                    )

                    if final.message_hash:
                        self._set_attr(
                            stage1_span,
                            "humane_proxy.message_hash",
                            final.message_hash,
                        )

                    return final

            # Early exit: clear safe — but only when no later stages can
            # add value.
            stage2_enabled = 2 in self.enabled_stages and self._stage2 is not None

            if (
                not stage2_enabled
                and result.score <= self.stage1_ceiling
                and result.category == "safe"
            ):

                final = self._finalize(result, session_id, text)

                self._set_attr(
                    stage1_span,
                    "humane_proxy.category",
                    final.classification.category,
                )
                self._set_attr(
                    stage1_span,
                    "humane_proxy.score",
                    final.classification.score,
                )
                self._set_attr(
                    stage1_span,
                    "humane_proxy.triggers_count",
                    len(final.classification.triggers),
                )
                self._set_attr(
                    stage1_span,
                    "humane_proxy.stage_reached",
                    final.classification.stage,
                )

                if final.message_hash:
                    self._set_attr(
                        stage1_span,
                        "humane_proxy.message_hash",
                        final.message_hash,
                    )

                if span is not None and Status and StatusCode:
                    span.set_status(Status(StatusCode.OK))

                return final

            # ------------------------------------------------------------------
            # Stage 2 — Embeddings
            # ------------------------------------------------------------------

            if stage2_enabled:

                with self._span("humane_proxy.stage2") as stage2_span:

                    s2 = self._stage2.classify(text)
                    result = self._combine(result, s2)

                    self._set_attr(
                        stage2_span,
                        "humane_proxy.category",
                        s2.category,
                    )
                    self._set_attr(
                        stage2_span,
                        "humane_proxy.final_score",
                        s2.score,
                    )
                    self._set_attr(
                        stage2_span,
                        "humane_proxy.triggers_count",
                        len(s2.triggers),
                    )

                    # Early exit after Stage 2.
                    if result.category == "self_harm":

                        final = self._finalize(result, session_id, text)

                        self._set_attr(
                            stage2_span,
                            "humane_proxy.category",
                            final.classification.category,
                        )
                        self._set_attr(
                            stage1_span,
                            "humane_proxy.final_score",
                            final.classification.score,
                        )
                        self._set_attr(
                            stage2_span,
                            "humane_proxy.final_score",
                            final.classification.score,
                        )
                        self._set_attr(
                            stage2_span,
                            "humane_proxy.triggers_count",
                            len(final.classification.triggers),
                        )
                        self._set_attr(
                            stage2_span,
                            "humane_proxy.stage_reached",
                            final.classification.stage,
                        )

                        if final.message_hash:
                            self._set_attr(
                                stage2_span,
                                "humane_proxy.message_hash",
                                final.message_hash,
                            )

                        return final

                    if (
                        result.score <= self.stage2_ceiling
                        and result.category == "safe"
                    ):

                        final = self._finalize(result, session_id, text)

                        return final

            # ------------------------------------------------------------------
            # Stage 3 — Reasoning LLM
            # ------------------------------------------------------------------

            if 3 in self.enabled_stages and self._stage3 is not None:

                try:
                    with self._span("humane_proxy.stage3") as stage3_span:

                        s3 = await self._stage3.classify(text, result)
                        result = self._combine(result, s3)

                        self._set_attr(
                            stage3_span,
                            "humane_proxy.category",
                            s3.category,
                        )
                        self._set_attr(
                            stage3_span,
                            "humane_proxy.final_score",
                            s3.score,
                        )
                        self._set_attr(
                            stage3_span,
                            "humane_proxy.triggers_count",
                            len(s3.triggers),
                        )

                except Exception as e:

                    logger.exception("Stage-3 classification failed")

                    if span is not None:
                        span.record_exception(e)

                    if span is not None and Status and StatusCode:
                        span.set_status(Status(StatusCode.ERROR))

                    result.triggers.append("stage3_error")

            # ------------------------------------------------------------------
            # Finalization
            # ------------------------------------------------------------------

            final = self._finalize(result, session_id, text)

            self._set_attr(
                span,
                "humane_proxy.category",
                final.classification.category,
            )

            self._set_attr(
                span,
                "humane_proxy.score",
                final.classification.score,
            )

            self._set_attr(
                span,
                "humane_proxy.final_score",
                final.classification.score,
            )

            self._set_attr(
                span,
                "humane_proxy.triggers_count",
                len(final.classification.triggers),
            )

            self._set_attr(
                span,
                "humane_proxy.stage_reached",
                final.classification.stage,
            )

            if final.message_hash:
                self._set_attr(
                    span,
                    "humane_proxy.message_hash",
                    final.message_hash,
                )

            if span is not None and Status and StatusCode:
                span.set_status(Status(StatusCode.OK))

            return final

    # ------------------------------------------------------------------
    # Classification — sync (Stages 1 + 2 only)
    # ------------------------------------------------------------------

    def classify_sync(
        self,
        text: str,
        session_id: str,
    ) -> PipelineResult:
        """Run synchronous pipeline (Stages 1 + 2 only)."""

        with self._span("humane_proxy.pipeline.classify") as span:

            safe_session = hashlib.sha256(session_id.encode("utf-8")).hexdigest()

            self._set_attr(
                span,
                "humane_proxy.session_id",
                safe_session,
            )

            with self._span("humane_proxy.stage1") as stage1_span:
                result = self._run_stage1(text)

                self._set_attr(
                    stage1_span,
                    "humane_proxy.category",
                    result.category,
                )

                self._set_attr(
                    stage1_span,
                    "humane_proxy.final_score",
                    result.score,
                )

                self._set_attr(
                    stage1_span,
                    "humane_proxy.triggers_count",
                    len(result.triggers),
                )

            if result.category == "self_harm":
                final = self._finalize(
                    result,
                    session_id,
                    text,
                )

                return final

            stage2_enabled = 2 in self.enabled_stages and self._stage2 is not None

            if (
                not stage2_enabled
                and result.score <= self.stage1_ceiling
                and result.category == "safe"
            ):
                return self._finalize(
                    result,
                    session_id,
                    text,
                )

            if stage2_enabled:
                with self._span("humane_proxy.stage2") as stage2_span:

                    s2 = self._stage2.classify(text)

                    self._set_attr(
                        stage2_span,
                        "humane_proxy.category",
                        s2.category,
                    )

                    self._set_attr(
                        stage2_span,
                        "humane_proxy.final_score",
                        s2.score,
                    )

                    self._set_attr(
                        stage2_span,
                        "humane_proxy.triggers_count",
                        len(s2.triggers),
                    )

                    self._set_attr(
                        stage2_span,
                        "humane_proxy.stage_reached",
                        s2.stage,
                    )

                    result = self._combine(result, s2)

            final = self._finalize(result, session_id, text)

            self._set_attr(
                span,
                "humane_proxy.category",
                final.classification.category,
            )

            self._set_attr(
                span,
                "humane_proxy.score",
                final.classification.score,
            )

            self._set_attr(
                span,
                "humane_proxy.final_score",
                final.classification.score,
            )

            self._set_attr(
                span,
                "humane_proxy.triggers_count",
                len(final.classification.triggers),
            )

            self._set_attr(
                span,
                "humane_proxy.stage_reached",
                final.classification.stage,
            )

            if final.message_hash:
                self._set_attr(
                    span,
                    "humane_proxy.message_hash",
                    final.message_hash,
                )

            if span is not None and Status and StatusCode:
                span.set_status(Status(StatusCode.OK))

            return final

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
        """Apply trajectory, self-harm threshold, and escalation logic."""
        from humane_proxy.risk.trajectory import analyze

        with self._span("humane_proxy.pipeline.finalize") as finalize_span:

            # Trajectory analysis.
            traj = analyze(session_id, result.score, result.category)

            # Spike boost.
            if traj.spike_detected:
                result.score = min(result.score + self.spike_boost, 1.0)
                result.triggers.append("trajectory_spike")

            # Self-harm threshold-aware override.
            self_harm_cfg = (
                self._config.get("safety", {})
                .get("categories", {})
                .get("self_harm", {})
            )
            self_harm_threshold = self_harm_cfg.get("escalate_threshold", 0.5)

            if result.category == "self_harm":
                if result.score >= self_harm_threshold:
                    # Definitive self-harm — force to 1.0.
                    result.score = 1.0
                else:
                    # Below threshold — downgrade to safe for transparency.
                    result.triggers.append(
                        f"self_harm_below_threshold:{result.score:.3f}<{self_harm_threshold}"
                    )
                    result.category = "safe"

            result.score = min(result.score, 1.0)

            # Escalation decision.
            should_escalate = result.category == "self_harm" or (
                result.category == "criminal_intent"
                and result.score >= self.risk_threshold
            )
            should_block = should_escalate

            # Privacy: hash the message.
            message_hash: str | None = None
            if not self.store_message_text:
                message_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if message_hash:
                    self._set_attr(
                        finalize_span,
                        "humane_proxy.message_hash",
                        message_hash,
                    )

            if finalize_span is not None and Status and StatusCode:
                self._set_status(finalize_span, Status(StatusCode.OK))

            self._set_attr(
                finalize_span,
                "humane_proxy.category",
                result.category,
            )

            self._set_attr(
                finalize_span,
                "humane_proxy.final_score",
                result.score,
            )

            self._set_attr(
                finalize_span,
                "humane_proxy.triggers_count",
                len(result.triggers),
            )

            self._set_attr(
                finalize_span,
                "humane_proxy.stage_reached",
                result.stage,
            )

            return PipelineResult(
                classification=result,
                trajectory=traj,
                should_escalate=should_escalate,
                should_block=should_block,
                message_hash=message_hash,
            )
