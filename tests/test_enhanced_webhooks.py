"""Tests for Phase 3 enhanced webhooks (Teams, Email, category formatting)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from humane_proxy.escalation.webhooks import (
    dispatch_webhooks,
    send_teams,
    send_email,
    send_slack,
    send_discord,
)


class TestTeamsWebhook:
    @pytest.mark.asyncio
    async def test_teams_payload_structure(self):
        posted = []

        async def _mock_post(url, payload, **kw):
            posted.append(payload)

        with patch("humane_proxy.escalation.webhooks._post", _mock_post):
            await send_teams(
                "https://teams.webhook/test",
                "sess-1", 0.9, ["keyword:suicide"], "self_harm",
            )

        assert len(posted) == 1
        body = posted[0]
        assert body["type"] == "message"
        card = body["attachments"][0]["content"]
        assert card["type"] == "AdaptiveCard"

    @pytest.mark.asyncio
    async def test_teams_self_harm_uses_attention_color(self):
        posted = []

        async def _mock_post(url, payload, **kw):
            posted.append(payload)

        with patch("humane_proxy.escalation.webhooks._post", _mock_post):
            await send_teams("https://test", "s", 1.0, [], "self_harm")

        card = posted[0]["attachments"][0]["content"]
        title_block = card["body"][0]
        assert title_block["color"] == "Attention"

    @pytest.mark.asyncio
    async def test_teams_criminal_uses_warning_color(self):
        posted = []

        async def _mock_post(url, payload, **kw):
            posted.append(payload)

        with patch("humane_proxy.escalation.webhooks._post", _mock_post):
            await send_teams("https://test", "s", 0.8, [], "criminal_intent")

        card = posted[0]["attachments"][0]["content"]
        title_block = card["body"][0]
        assert title_block["color"] == "Warning"


class TestEmailWebhook:
    @pytest.mark.asyncio
    async def test_email_skipped_if_no_recipients(self):
        """Should not attempt to connect if 'to' list is empty."""
        smtp_config = {"host": "smtp.test", "port": 587, "to": []}
        # No exception should be raised and no SMTP connection made.
        with patch("smtplib.SMTP") as mock_smtp:
            await send_email(smtp_config, "sess", 0.9, [], "self_harm")
        mock_smtp.assert_not_called()

    @pytest.mark.asyncio
    async def test_email_subject_contains_category(self):
        sent = []

        def _fake_smtp(*args, **kwargs):
            class FakeSMTP:
                def __enter__(self): return self
                def __exit__(self, *a): pass
                def starttls(self): pass
                def login(self, *a): pass
                def sendmail(self, from_addr, to_list, content):
                    sent.append(content)
            return FakeSMTP()

        smtp_config = {
            "host": "smtp.test", "port": 587,
            "use_tls": False, "username": "", "password": "",
            "from": "hp@test.com", "to": ["admin@test.com"],
        }
        with patch("smtplib.SMTP", _fake_smtp):
            await send_email(smtp_config, "sess-1", 0.95, ["keyword:suicide"], "self_harm")

        assert len(sent) == 1
        raw = sent[0]
        # The body is base64-encoded UTF-8. "c2VsZl9oYXJt" is b64("self_harm").
        # Also look for "Q2F0ZWdvcnk" which is b64 prefix of "Category".
        # Either proves the email body was constructed correctly.
        import base64, email
        msg = email.message_from_string(raw)
        body = ""
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="ignore")
                    break
        assert "self_harm" in body




class TestDispatchWebhooks:
    @pytest.mark.asyncio
    async def test_dispatch_teams_only(self):
        config = {
            "escalation": {
                "webhooks": {"teams_url": "https://teams.test/"}
            }
        }
        with patch("humane_proxy.escalation.webhooks.send_teams", new_callable=AsyncMock) as mock_teams, \
             patch("humane_proxy.escalation.webhooks.send_slack", new_callable=AsyncMock) as mock_slack:
            await dispatch_webhooks(config, "sess", 0.8, ["t1"], "self_harm")

        mock_teams.assert_awaited_once()
        mock_slack.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_all_channels(self):
        config = {
            "escalation": {
                "webhooks": {
                    "slack_url": "https://slack/",
                    "discord_url": "https://discord/",
                    "pagerduty_routing_key": "key",
                    "teams_url": "https://teams/",
                }
            }
        }
        with patch("humane_proxy.escalation.webhooks.send_slack", new_callable=AsyncMock) as ms, \
             patch("humane_proxy.escalation.webhooks.send_discord", new_callable=AsyncMock) as md, \
             patch("humane_proxy.escalation.webhooks.send_pagerduty", new_callable=AsyncMock) as mp, \
             patch("humane_proxy.escalation.webhooks.send_teams", new_callable=AsyncMock) as mt:
            await dispatch_webhooks(config, "sess", 0.9, [], "criminal_intent")

        ms.assert_awaited_once()
        md.assert_awaited_once()
        mp.assert_awaited_once()
        mt.assert_awaited_once()
