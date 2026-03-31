"""Tests for Stage-3 providers (llamaguard, openai_moderation, openai_chat).

All tests mock HTTP calls — no real API keys needed.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from humane_proxy.classifiers.models import ClassificationResult


# -----------------------------------------------------------------------
# LlamaGuard
# -----------------------------------------------------------------------

class TestLlamaGuard:
    def _make(self, config=None):
        from humane_proxy.classifiers.stage3.llamaguard import LlamaGuardClassifier
        return LlamaGuardClassifier(config or {})

    @pytest.mark.asyncio
    async def test_safe_verdict(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "safe"}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("hello", ClassificationResult())
            assert result.category == "safe"
            assert result.score == 0.0
            assert result.stage == 3

    @pytest.mark.asyncio
    async def test_unsafe_self_harm(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "unsafe\nS11"}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("I want to die", ClassificationResult())
            assert result.category == "self_harm"
            assert result.score == 1.0
            assert result.stage == 3

    @pytest.mark.asyncio
    async def test_unsafe_criminal(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "unsafe\nS1"}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("bomb instructions", ClassificationResult())
            assert result.category == "criminal_intent"
            assert result.stage == 3

    @pytest.mark.asyncio
    async def test_api_error_graceful(self):
        cls = self._make()
        prior = ClassificationResult(category="safe", score=0.0, triggers=["t1"])
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("timeout")):
            result = await cls.classify("test", prior)
            assert "stage3_error" in result.triggers
            assert result.stage == 3

    @pytest.mark.asyncio
    async def test_multiple_category_codes(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "unsafe\nS1,S11"}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("text", ClassificationResult())
            # S11 = self_harm takes priority over S1 = criminal
            assert result.category == "self_harm"


# -----------------------------------------------------------------------
# OpenAI Moderation
# -----------------------------------------------------------------------

class TestOpenAIModeration:
    def _make(self, config=None):
        from humane_proxy.classifiers.stage3.openai_moderation import OpenAIModerationClassifier
        return OpenAIModerationClassifier(config or {})

    @pytest.mark.asyncio
    async def test_not_flagged(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{"flagged": False, "categories": {}, "category_scores": {}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("hello", ClassificationResult())
            assert result.category == "safe"
            assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_self_harm_flagged(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{
            "flagged": True,
            "categories": {"self-harm": True, "self-harm/intent": True},
            "category_scores": {"self-harm": 0.95, "self-harm/intent": 0.9},
        }]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("test", ClassificationResult())
            assert result.category == "self_harm"
            assert result.score == 1.0  # critical override

    @pytest.mark.asyncio
    async def test_violence_flagged(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{
            "flagged": True,
            "categories": {"violence": True},
            "category_scores": {"violence": 0.8},
        }]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("test", ClassificationResult())
            assert result.category == "criminal_intent"

    @pytest.mark.asyncio
    async def test_harassment_stays_safe(self):
        """Harassment flags are noted but kept as 'safe' — not our domain."""
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [{
            "flagged": True,
            "categories": {"harassment": True},
            "category_scores": {"harassment": 0.7},
        }]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("test", ClassificationResult())
            assert result.category == "safe"

    @pytest.mark.asyncio
    async def test_api_error_graceful(self):
        cls = self._make()
        prior = ClassificationResult(category="safe", score=0.1, triggers=["t1"])
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("nope")):
            result = await cls.classify("test", prior)
            assert "stage3_error" in result.triggers


# -----------------------------------------------------------------------
# OpenAI Chat
# -----------------------------------------------------------------------

class TestOpenAIChat:
    def _make(self, config=None):
        from humane_proxy.classifiers.stage3.openai_chat import OpenAIChatClassifier
        return OpenAIChatClassifier(config or {})

    @pytest.mark.asyncio
    async def test_safe_response(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"category": "safe", "score": 0.0, "reasoning": "Normal chat"}'}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("hello", ClassificationResult())
            assert result.category == "safe"
            assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_self_harm_response(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"category": "self_harm", "score": 0.95, "reasoning": "Suicidal ideation"}'}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("I want to die", ClassificationResult())
            assert result.category == "self_harm"
            assert result.score == 1.0  # critical override

    @pytest.mark.asyncio
    async def test_criminal_response(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"category": "criminal_intent", "score": 0.8, "reasoning": "Violence"}'}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("bomb", ClassificationResult())
            assert result.category == "criminal_intent"
            assert result.score == 0.8

    @pytest.mark.asyncio
    async def test_non_json_response(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "I cannot classify this"}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("test", ClassificationResult())
            assert result.category == "safe"
            assert "stage3_parse_error" in result.triggers

    @pytest.mark.asyncio
    async def test_unknown_category_defaults_safe(self):
        cls = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"category": "unknown_thing", "score": 0.5}'}}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await cls.classify("test", ClassificationResult())
            assert result.category == "safe"

    @pytest.mark.asyncio
    async def test_api_error_graceful(self):
        cls = self._make()
        prior = ClassificationResult(category="safe", triggers=["t1"])
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, side_effect=Exception("boom")):
            result = await cls.classify("test", prior)
            assert "stage3_error" in result.triggers
