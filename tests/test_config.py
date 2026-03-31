"""Tests for humane_proxy.config (layered config loader)."""

import os

import pytest

from humane_proxy.config import _build_config, _deep_merge, reload_config


class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_override(self):
        base = {"safety": {"risk_threshold": 0.7, "spike_boost": 0.25}}
        override = {"safety": {"risk_threshold": 0.5}}
        result = _deep_merge(base, override)
        assert result["safety"]["risk_threshold"] == 0.5
        assert result["safety"]["spike_boost"] == 0.25

    def test_non_mutating(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1  # original unchanged


class TestLoadConfig:
    def test_defaults_loaded(self):
        config = _build_config()
        assert "safety" in config
        assert "heuristics" in config
        assert config["safety"]["risk_threshold"] == 0.7

    def test_user_config_override(self, tmp_path, monkeypatch):
        user_cfg = tmp_path / "humane_proxy.yaml"
        user_cfg.write_text("safety:\n  risk_threshold: 0.5\n")
        monkeypatch.setenv("HUMANE_PROXY_CONFIG", str(user_cfg))

        config = _build_config()
        assert config["safety"]["risk_threshold"] == 0.5
        # Defaults still present for other keys.
        assert config["safety"]["spike_boost"] == 0.25

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("HUMANE_PROXY_PORT", "9999")
        config = _build_config()
        assert config["server"]["port"] == 9999

    def test_reload_picks_up_changes(self, tmp_path, monkeypatch):
        user_cfg = tmp_path / "humane_proxy.yaml"
        user_cfg.write_text("safety:\n  risk_threshold: 0.6\n")
        monkeypatch.setenv("HUMANE_PROXY_CONFIG", str(user_cfg))

        config1 = reload_config()
        assert config1["safety"]["risk_threshold"] == 0.6

        user_cfg.write_text("safety:\n  risk_threshold: 0.3\n")
        config2 = reload_config()
        assert config2["safety"]["risk_threshold"] == 0.3

    def test_self_harm_keywords_in_defaults(self):
        config = _build_config()
        keywords = config.get("heuristics", {}).get("self_harm_keywords", [])
        assert "suicide" in keywords
        assert "kill myself" in keywords

    def test_criminal_keywords_in_defaults(self):
        config = _build_config()
        keywords = config.get("heuristics", {}).get("criminal_keywords", [])
        assert "how to make a bomb" in keywords

    def test_context_reducers_in_defaults(self):
        config = _build_config()
        reducers = config.get("heuristics", {}).get("context_reducers", [])
        assert "laughing" in reducers
        assert "warning signs" in reducers
