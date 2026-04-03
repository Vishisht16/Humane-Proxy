"""Tests for embedding classifier singleton cache and ambiguity dampening."""

import pytest
from unittest.mock import patch, MagicMock

from humane_proxy.classifiers.embedding_classifier import (
    EmbeddingClassifier,
    _model_cache,
    _load_model_singleton,
)
from humane_proxy.classifiers.models import ClassificationResult


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure the model cache is empty before every test."""
    _model_cache.clear()
    yield
    _model_cache.clear()


def test_singleton_cache():
    """Verify that the model is cached and reused across instances."""
    mock_model = MagicMock()

    with patch("humane_proxy.classifiers.embedding_classifier._ML_AVAILABLE", True), \
         patch("humane_proxy.classifiers.embedding_classifier.SentenceTransformer", return_value=mock_model) as mock_st:

        clf1 = EmbeddingClassifier({"stage2": {"model": "test-singleton"}})
        clf1._try_load()

        # SentenceTransformer should have been called exactly once.
        assert mock_st.call_count == 1
        assert clf1._model is mock_model

        # Second instance with the same model name should reuse the cache.
        clf2 = EmbeddingClassifier({"stage2": {"model": "test-singleton"}})
        clf2._try_load()

        assert mock_st.call_count == 1  # No additional call.
        assert clf2._model is mock_model
        assert clf1._model is clf2._model


def test_singleton_cache_different_models():
    """Different model names should load separately."""
    mock_a = MagicMock()
    mock_b = MagicMock()
    call_count = 0

    def make_model(name):
        nonlocal call_count
        call_count += 1
        return mock_a if "model-a" in name else mock_b

    with patch("humane_proxy.classifiers.embedding_classifier._ML_AVAILABLE", True), \
         patch("humane_proxy.classifiers.embedding_classifier.SentenceTransformer", side_effect=make_model):

        clf_a = EmbeddingClassifier({"stage2": {"model": "model-a"}})
        clf_a._try_load()
        assert clf_a._model is mock_a

        clf_b = EmbeddingClassifier({"stage2": {"model": "model-b"}})
        clf_b._try_load()
        assert clf_b._model is mock_b

        assert call_count == 2


def test_ml_unavailable_returns_neutral():
    """When ML deps are missing, classify returns a safe neutral result."""
    with patch("humane_proxy.classifiers.embedding_classifier._ML_AVAILABLE", False):
        clf = EmbeddingClassifier({})
        result = clf.classify("any text")
        assert result.category == "safe"
        assert result.score == 0.0
        assert result.stage == 2
