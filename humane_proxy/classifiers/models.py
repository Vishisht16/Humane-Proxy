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

"""Structured data models for the HumaneProxy classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClassificationResult:
    """Output of a single classification stage.

    Flows through the pipeline and accumulates information as each
    stage contributes its analysis.
    """

    category: str = "safe"
    """``"self_harm"`` | ``"criminal_intent"`` | ``"safe"``"""

    score: float = 0.0
    """Risk score in ``[0.0, 1.0]``."""

    triggers: list[str] = field(default_factory=list)
    """Human-readable trigger descriptions (deduplicated)."""

    stage: int = 1
    """Which pipeline stage produced this result (1, 2, or 3)."""

    reasoning: str | None = None
    """Stage-3 reasoning text (if available)."""


@dataclass
class TrajectoryResult:
    """Rich trajectory analysis for a session."""

    spike_detected: bool = False
    """Whether the current score represents a behavioural spike."""

    trend: str = "stable"
    """``"escalating"`` | ``"stable"`` | ``"declining"``"""

    window_scores: list[float] = field(default_factory=list)
    """Recent scores in the trajectory window."""

    category_counts: dict[str, int] = field(default_factory=dict)
    """Distribution of categories in this session: ``{"self_harm": 2, "safe": 8}``"""

    message_count: int = 0
    """Total messages tracked for this session."""


@dataclass
class PipelineResult:
    """Final output of the full safety pipeline.

    Wraps the classification, trajectory, and escalation decision
    into a single structured result.
    """

    classification: ClassificationResult = field(default_factory=ClassificationResult)
    trajectory: TrajectoryResult | None = None
    should_escalate: bool = False
    should_block: bool = False
    message_hash: str | None = None

    def to_dict(self) -> dict:
        """Serialize to a flat dictionary for API responses."""
        result: dict = {
            "safe": not self.should_escalate,
            "category": self.classification.category,
            "score": round(self.classification.score, 4),
            "triggers": self.classification.triggers,
            "stage_reached": self.classification.stage,
        }
        if self.classification.reasoning:
            result["reasoning"] = self.classification.reasoning
        if self.trajectory:
            result["trajectory"] = {
                "spike_detected": self.trajectory.spike_detected,
                "trend": self.trajectory.trend,
                "message_count": self.trajectory.message_count,
            }
        if self.message_hash:
            result["message_hash"] = self.message_hash
        return result
