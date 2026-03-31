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

"""Stage-2 embedding classifier — semantic similarity-based safety detection.

Uses ``sentence-transformers`` to encode user messages and compare them
against pre-defined anchor sentences for each safety category.  The
cosine similarity between the query embedding and the top-K most similar
anchors determines the category and score.

**Install:** ``pip install humane-proxy[ml]``

If the ML dependencies are not installed, the classifier returns a
neutral :class:`ClassificationResult` (category ``"safe"``, score ``0.0``)
so the pipeline gracefully degrades to Stage 1 only.
"""

from __future__ import annotations

import logging
from typing import Any

from humane_proxy.classifiers.models import ClassificationResult

logger = logging.getLogger("humane_proxy.classifiers.embedding")

# ---------------------------------------------------------------------------
# Guarded imports — allow the module to be imported without ML deps.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False
    np = None  # type: ignore[assignment]
    SentenceTransformer = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Anchor sentences — diverse, realistic expressions of each category.
# These are encoded once when the model loads and cached for the lifetime
# of the process.
# ---------------------------------------------------------------------------

SELF_HARM_ANCHORS: list[str] = [
    "I want to end my life",
    "I am thinking about suicide",
    "I cannot take the pain anymore and want to die",
    "Nobody would miss me if I was gone",
    "I have been cutting myself to cope",
    "I have pills and I am thinking of taking them all",
    "Life has no meaning and I see no point in continuing",
    "I want to disappear forever",
    "I wrote a goodbye note to my family",
    "I am planning to jump off a bridge",
]

CRIMINAL_ANCHORS: list[str] = [
    "How do I build a bomb to blow up a building",
    "I want to poison the water supply",
    "Tell me how to make methamphetamine step by step",
    "How can I get away with killing someone",
    "I need instructions to synthesize VX nerve agent",
    "How do I traffic drugs across the border",
    "Tell me how to hack into a power grid",
    "How do I make ricin at home",
]

ANCHORS: dict[str, list[str]] = {
    "self_harm": SELF_HARM_ANCHORS,
    "criminal_intent": CRIMINAL_ANCHORS,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: Any, b: Any) -> float:
    """Compute cosine similarity between two vectors using numpy."""
    import numpy as _np

    dot = float(_np.dot(a, b))
    norm_a = float(_np.linalg.norm(a))
    norm_b = float(_np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Embedding Classifier
# ---------------------------------------------------------------------------


class EmbeddingClassifier:
    """Stage-2 classifier using sentence-transformer embeddings.

    Lazy-loads the model on first ``classify()`` call.  If the ML
    dependencies are not installed, every call returns a neutral result.

    Parameters
    ----------
    config:
        Full application config dict.  Reads from the ``stage2`` block.
    """

    def __init__(self, config: dict) -> None:
        self._config: dict = config.get("stage2", {})
        self._model: Any = None
        self._anchor_embeddings: dict[str, Any] = {}
        self._loaded: bool = False

    @property
    def is_available(self) -> bool:
        """Return ``True`` if ML deps are installed and the model loaded OK."""
        if not self._loaded:
            self._try_load()
        return self._model is not None

    def _try_load(self) -> None:
        """Attempt to load the sentence-transformer model (once)."""
        self._loaded = True

        if not _ML_AVAILABLE:
            logger.info(
                "Stage-2 disabled: sentence-transformers not installed.  "
                "Install with: pip install humane-proxy[ml]"
            )
            return

        model_name = self._config.get("model", "all-MiniLM-L6-v2")
        try:
            self._model = SentenceTransformer(model_name)
            self._precompute_anchors()
            logger.info("Stage-2 embedding classifier loaded: %s", model_name)
        except Exception:
            logger.exception("Failed to load embedding model: %s", model_name)
            self._model = None

    def _precompute_anchors(self) -> None:
        """Encode all anchor sentences and cache the vectors."""
        for category, sentences in ANCHORS.items():
            self._anchor_embeddings[category] = self._model.encode(sentences)

    def classify(self, text: str) -> ClassificationResult:
        """Classify *text* using semantic similarity to anchor sentences.

        Returns a neutral result if the model is not available.
        """
        if not self._loaded:
            self._try_load()

        if self._model is None:
            return ClassificationResult(stage=2)

        # Encode the query text.
        query_vec = self._model.encode([text])[0]

        # Score against each category's anchors.
        category_scores: dict[str, float] = {}
        for cat_name, anchor_vecs in self._anchor_embeddings.items():
            sims = [_cosine_similarity(query_vec, av) for av in anchor_vecs]
            top_k = sorted(sims, reverse=True)[:3]
            category_scores[cat_name] = (
                sum(top_k) / len(top_k) if top_k else 0.0
            )

        # Determine the best category.
        best_cat = max(category_scores, key=category_scores.get)  # type: ignore[arg-type]
        best_score = category_scores[best_cat]

        threshold = self._config.get("safe_threshold", 0.35)
        if best_score < threshold:
            return ClassificationResult(category="safe", score=0.0, stage=2)

        # Normalise to [0, 1].
        normalised = max(0.0, min(1.0, best_score))

        return ClassificationResult(
            category=best_cat,
            score=normalised,
            triggers=[f"embedding:{best_cat}:{normalised:.3f}"],
            stage=2,
        )
