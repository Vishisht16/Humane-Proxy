"""Tests for humane_proxy.escalation.webhooks."""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from humane_proxy.escalation.webhooks import (
    dispatch_webhooks,
    send_discord,
    send_pagerduty,
    send_slack,
    _sanitize_webhook_url,
)


@pytest.mark.asyncio
class TestSlack:
    async def test_send_slack_success(self):
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_slack("https://hooks.slack.com/test", "sess-1", 0.9, ["t1"], "self_harm")
            mock.assert_called_once()
            payload = mock.call_args[0][1]
            assert "blocks" in payload

    async def test_send_slack_error_swallowed(self):
        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=Exception("network fail"),
        ):
            await send_slack("https://hooks.slack.com/test", "sess-1", 0.9, ["t1"], "self_harm")

    async def test_slack_includes_category(self):
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_slack("https://hooks.slack.com/test", "sess-1", 0.9, ["t1"], "self_harm")
            payload = mock.call_args[0][1]
            header_text = payload["blocks"][0]["text"]["text"]
            assert "self_harm" in header_text

    async def test_post_redacts_webhook_url_and_body(self, caplog):
        class FakeResponse:
            status_code = 400
            text = "bad request for /services/T000/B000/SECRET?token=abc123"

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=FakeResponse()):
            caplog.set_level(logging.WARNING, logger="humane_proxy.escalation.webhooks")
            from humane_proxy.escalation.webhooks import _post

            await _post("https://hooks.slack.com/services/T000/B000/SECRET?token=abc123", {"x": 1})

        assert "hooks.slack.com" in caplog.text
        assert "/services/T000/B000/SECRET" not in caplog.text
        assert "token=abc123" not in caplog.text
        assert "bad request for" not in caplog.text


class TestSanitizeWebhookUrl:
    def test_sanitize_webhook_url_strips_sensitive_parts(self):
        assert _sanitize_webhook_url("https://hooks.slack.com/services/T000/B000/SECRET?token=abc123") == "https://hooks.slack.com"
        assert _sanitize_webhook_url("not-a-url") == "not-a-url"


@pytest.mark.asyncio
class TestDiscord:
    async def test_send_discord_success(self):
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_discord("https://discord.com/test", "sess-1", 0.9, ["t1"], "criminal_intent")
            mock.assert_called_once()
            payload = mock.call_args[0][1]
            assert "embeds" in payload


@pytest.mark.asyncio
class TestPagerDuty:
    async def test_send_pagerduty_success(self):
        with patch("humane_proxy.escalation.webhooks._post", new_callable=AsyncMock) as mock:
            await send_pagerduty("routing-key", "sess-1", 0.9, ["t1"], "self_harm")
            mock.assert_called_once()
            payload = mock.call_args[0][1]
            assert payload["routing_key"] == "routing-key"
            assert payload["event_action"] == "trigger"
            assert "self_harm" in payload["payload"]["summary"]


@pytest.mark.asyncio
class TestDispatch:
    async def test_dispatch_no_webhooks_configured(self):
        """No crash when all webhook URLs are empty."""
        config = {"escalation": {"webhooks": {"slack_url": "", "discord_url": "", "pagerduty_routing_key": ""}}}
        await dispatch_webhooks(config, "sess-1", 0.9, ["t1"], "self_harm")

    async def test_dispatch_calls_configured(self):
        config = {"escalation": {"webhooks": {
            "slack_url": "https://hooks.slack.com/test",
            "discord_url": "",
            "pagerduty_routing_key": "",
        }}}
        with patch("humane_proxy.escalation.webhooks.send_slack", new_callable=AsyncMock) as mock:
            await dispatch_webhooks(config, "sess-1", 0.9, ["t1"], "self_harm")
            mock.assert_called_once()
