"""Tests for Phase 3 care response system."""

from unittest.mock import patch

import pytest

from humane_proxy.escalation.router import (
    CARE_RESPONSE_BLOCK,
    CARE_CONTEXT_INJECTION,
    get_self_harm_response,
)


class TestCareResponseBlock:
    """Default block mode returns empathetic message with crisis resources."""

    def test_returns_block_mode_by_default(self):
        result = get_self_harm_response()
        assert result["mode"] == "block"

    def test_returns_built_in_message(self):
        result = get_self_harm_response()
        assert result["message"] == CARE_RESPONSE_BLOCK

    def test_payload_is_none_in_block_mode(self):
        result = get_self_harm_response()
        assert result["payload"] is None

    def test_message_contains_us_resource(self):
        assert "988" in CARE_RESPONSE_BLOCK

    def test_message_contains_india_resource(self):
        assert "iCall" in CARE_RESPONSE_BLOCK or "9152987821" in CARE_RESPONSE_BLOCK

    def test_message_contains_uk_resource(self):
        assert "Samaritans" in CARE_RESPONSE_BLOCK

    def test_message_contains_international_resource(self):
        assert "iasp.info" in CARE_RESPONSE_BLOCK or "Befrienders" in CARE_RESPONSE_BLOCK

    def test_message_contains_emergency_guidance(self):
        assert "emergency" in CARE_RESPONSE_BLOCK.lower()

    def test_message_contains_japan_resource(self):
        assert "0120-783-556" in CARE_RESPONSE_BLOCK

    def test_message_contains_south_korea_resource(self):
        assert "1393" in CARE_RESPONSE_BLOCK

    def test_message_contains_spain_resource(self):
        assert "Línea de Atención a conducta suicida: 024" in CARE_RESPONSE_BLOCK

    def test_message_contains_italy_resource(self):
        assert "800 274 274" in CARE_RESPONSE_BLOCK

    def test_message_contains_mexico_resource(self):
        assert "800 290 0024" in CARE_RESPONSE_BLOCK

    def test_message_contains_new_zealand_resource(self):
        assert "1737" in CARE_RESPONSE_BLOCK

    def test_message_contains_france_second_resource(self):
        assert "3114" in CARE_RESPONSE_BLOCK


class TestCareResponseForward:
    """Forward mode injects care context into the LLM payload."""

    def _fwd_config(self):
        return {
            "safety": {
                "categories": {
                    "self_harm": {"response_mode": "forward"}
                }
            }
        }

    def test_forward_mode_returns_forward(self):
        with patch("humane_proxy.escalation.router.get_config", return_value=self._fwd_config()):
            payload = {"messages": [{"role": "user", "content": "I feel hopeless"}]}
            result = get_self_harm_response(payload)
        assert result["mode"] == "forward"

    def test_forward_mode_injects_system_message(self):
        with patch("humane_proxy.escalation.router.get_config", return_value=self._fwd_config()):
            payload = {"messages": [{"role": "user", "content": "I feel hopeless"}]}
            result = get_self_harm_response(payload)
        messages = result["payload"]["messages"]
        assert messages[0]["role"] == "system"
        assert "empathy" in messages[0]["content"].lower() or "distress" in messages[0]["content"].lower()

    def test_forward_mode_preserves_user_message(self):
        with patch("humane_proxy.escalation.router.get_config", return_value=self._fwd_config()):
            payload = {"messages": [{"role": "user", "content": "I feel hopeless"}]}
            result = get_self_harm_response(payload)
        messages = result["payload"]["messages"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert any("hopeless" in m["content"] for m in user_msgs)

    def test_forward_mode_no_payload_falls_back_to_block(self):
        with patch("humane_proxy.escalation.router.get_config", return_value=self._fwd_config()):
            result = get_self_harm_response(None)
        # No payload → can't inject, falls back to block
        assert result["mode"] == "block"


class TestCustomBlockMessage:
    """Operator can override the built-in block message."""

    def _custom_config(self):
        return {
            "safety": {
                "categories": {
                    "self_harm": {
                        "response_mode": "block",
                        "block_message": "Custom care message for tests.",
                    }
                }
            }
        }

    def test_custom_message_overrides_default(self):
        with patch("humane_proxy.escalation.router.get_config", return_value=self._custom_config()):
            result = get_self_harm_response()
        assert result["message"] == "Custom care message for tests."

class TestRegionAwareCareResponse:
    """Optional region surfaces the matching country's resources first."""

    def _region_config(self, region):
        return {
            "safety": {
                "categories": {
                    "self_harm": {
                        "response_mode": "block",
                        "region": region,
                    }
                }
            }
        }

    def test_no_region_returns_full_default_block(self):
        # Sanity: with no region set, the message is the unchanged default.
        result = get_self_harm_response()
        assert result["message"] == CARE_RESPONSE_BLOCK

    def test_region_surfaces_matching_country_first(self):
        with patch("humane_proxy.escalation.router.get_config", return_value=self._region_config("IN")):
            result = get_self_harm_response()
        body = result["message"].split("Please reach out to a crisis service near you:")[1].lstrip()
        # India's block should come first after the intro.
        assert body.startswith("🇮🇳")

    def test_region_still_includes_other_countries(self):
        with patch("humane_proxy.escalation.router.get_config", return_value=self._region_config("IN")):
            result = get_self_harm_response()
        # Surfacing first must not drop the others.
        assert "988" in result["message"]          # US
        assert "Samaritans" in result["message"]    # UK
        assert "iasp.info" in result["message"]      # international fallback

    def test_region_is_case_insensitive(self):
        with patch("humane_proxy.escalation.router.get_config", return_value=self._region_config("in")):
            result = get_self_harm_response()
        body = result["message"].split("Please reach out to a crisis service near you:")[1].lstrip()
        assert body.startswith("🇮🇳")

    def test_unknown_region_falls_back_to_full_default(self):
        with patch("humane_proxy.escalation.router.get_config", return_value=self._region_config("ZZ")):
            result = get_self_harm_response()
        # Unknown code → unchanged default order.
        assert result["message"] == CARE_RESPONSE_BLOCK

    def test_custom_block_message_takes_precedence_over_region(self):
        cfg = {
            "safety": {
                "categories": {
                    "self_harm": {
                        "response_mode": "block",
                        "region": "IN",
                        "block_message": "Custom care message for tests.",
                    }
                }
            }
        }
        with patch("humane_proxy.escalation.router.get_config", return_value=cfg):
            result = get_self_harm_response()
        assert result["message"] == "Custom care message for tests."
        
