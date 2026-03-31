"""Tests for humane_proxy.classifiers.embedding_classifier."""

from unittest.mock import MagicMock, patch

import pytest

from humane_proxy.classifiers.models import ClassificationResult


class TestCosineHelper:
    """Test the _cosine_similarity helper directly."""

    def test_identical_vectors(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.embedding_classifier import _cosine_similarity

        a = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.embedding_classifier import _cosine_similarity

        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.embedding_classifier import _cosine_similarity

        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.embedding_classifier import _cosine_similarity

        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert _cosine_similarity(a, b) == 0.0


class TestEmbeddingClassifierWithMock:
    """Test the classifier with a mocked sentence-transformer model."""

    def _make_classifier(self, config=None):
        from humane_proxy.classifiers.embedding_classifier import EmbeddingClassifier
        return EmbeddingClassifier(config or {})

    def test_neutral_when_ml_unavailable(self):
        with patch("humane_proxy.classifiers.embedding_classifier._ML_AVAILABLE", False):
            classifier = self._make_classifier()
            result = classifier.classify("test text")
            assert isinstance(result, ClassificationResult)
            assert result.category == "safe"
            assert result.score == 0.0
            assert result.stage == 2

    def test_is_available_false_when_no_ml(self):
        with patch("humane_proxy.classifiers.embedding_classifier._ML_AVAILABLE", False):
            classifier = self._make_classifier()
            assert classifier.is_available is False

    def test_classify_with_mocked_model(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.embedding_classifier import EmbeddingClassifier

        mock_model = MagicMock()
        # Make model.encode return vectors that are very similar to self_harm anchors.
        mock_model.encode.return_value = np.array([[0.9, 0.1, 0.0]] * 10)

        with patch("humane_proxy.classifiers.embedding_classifier._ML_AVAILABLE", True):
            with patch("humane_proxy.classifiers.embedding_classifier.SentenceTransformer", return_value=mock_model):
                classifier = EmbeddingClassifier({})
                # Force load.
                classifier._try_load()

                # When cosine sim is high (identical vectors), should flag.
                mock_model.encode.return_value = np.array([[0.9, 0.1, 0.0]])
                result = classifier.classify("I want to end my life")
                assert isinstance(result, ClassificationResult)
                assert result.stage == 2

    def test_classify_returns_safe_below_threshold(self):
        np = pytest.importorskip("numpy")
        from humane_proxy.classifiers.embedding_classifier import EmbeddingClassifier

        mock_model = MagicMock()
        # Anchors: some arbitrary vectors.
        mock_model.encode.side_effect = [
            np.random.rand(10, 3),  # self_harm anchors
            np.random.rand(8, 3),   # criminal anchors
            np.array([[0.0, 0.0, 0.0]]),  # query vector (zero = orthogonal to everything)
        ]

        with patch("humane_proxy.classifiers.embedding_classifier._ML_AVAILABLE", True):
            with patch("humane_proxy.classifiers.embedding_classifier.SentenceTransformer", return_value=mock_model):
                classifier = EmbeddingClassifier({"stage2": {"safe_threshold": 0.99}})
                classifier._try_load()
                result = classifier.classify("hello")
                assert result.category == "safe"
                assert result.score == 0.0
