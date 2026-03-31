"""Tests for humane_proxy.escalation.webhooks."""

from unittest.mock import AsyncMock, patch

import pytest

from humane_proxy.escalation.webhooks import (
    dispatch_webhooks,
    send_discord,
    send_pagerduty,
    send_slack,
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
