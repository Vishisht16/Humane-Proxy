"""Layered configuration loader for HumaneProxy.

Priority (highest → lowest):
1. HUMANE_PROXY_* environment variables (for individual overrides)
2. User's ``humane_proxy.yaml`` in CWD (or path in HUMANE_PROXY_CONFIG)
3. Package-bundled ``config.yaml`` defaults
"""

from __future__ import annotations

import copy
import logging
import os
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("humane_proxy.config")

_PACKAGE_DEFAULTS_PATH: Path = Path(__file__).resolve().parent / "config.yaml"

# Thread-safe cache
_lock = threading.Lock()
_cached_config: dict | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-mutating)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning {} on any error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return {}


def _apply_env_overrides(config: dict) -> dict:
    """Apply HUMANE_PROXY_* environment variables as flat overrides.

    Mapping (env var → config path):
        HUMANE_PROXY_RISK_THRESHOLD  →  safety.risk_threshold
        HUMANE_PROXY_HOST            →  server.host
        HUMANE_PROXY_PORT            →  server.port
        HUMANE_PROXY_SPIKE_BOOST     →  safety.spike_boost
        HUMANE_PROXY_SLACK_URL       →  escalation.webhooks.slack_url
        HUMANE_PROXY_DISCORD_URL     →  escalation.webhooks.discord_url
        HUMANE_PROXY_PAGERDUTY_KEY   →  escalation.webhooks.pagerduty_routing_key
        HUMANE_PROXY_DB_PATH         →  escalation.db_path
    """
    _ENV_MAP: dict[str, tuple[list[str], type]] = {
        "HUMANE_PROXY_RISK_THRESHOLD": (["safety", "risk_threshold"], float),
        "HUMANE_PROXY_SPIKE_BOOST": (["safety", "spike_boost"], float),
        "HUMANE_PROXY_HOST": (["server", "host"], str),
        "HUMANE_PROXY_PORT": (["server", "port"], int),
        "HUMANE_PROXY_RELOAD": (["server", "reload"], bool),
        "HUMANE_PROXY_SLACK_URL": (["escalation", "webhooks", "slack_url"], str),
        "HUMANE_PROXY_DISCORD_URL": (["escalation", "webhooks", "discord_url"], str),
        "HUMANE_PROXY_PAGERDUTY_KEY": (["escalation", "webhooks", "pagerduty_routing_key"], str),
        "HUMANE_PROXY_DB_PATH": (["escalation", "db_path"], str),
        "HUMANE_PROXY_RATE_LIMIT_MAX": (["escalation", "rate_limit_max"], int),
        # Phase 2 additions:
        "HUMANE_PROXY_STAGE3_PROVIDER": (["stage3", "provider"], str),
        "HUMANE_PROXY_STAGE3_TIMEOUT": (["stage3", "timeout"], float),
        "HUMANE_PROXY_STAGE1_CEILING": (["pipeline", "stage1_ceiling"], float),
        "HUMANE_PROXY_STAGE2_CEILING": (["pipeline", "stage2_ceiling"], float),
    }

    for env_key, (path, cast) in _ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        try:
            if cast is bool:
                value: Any = raw.lower() in ("1", "true", "yes")
            else:
                value = cast(raw)
        except (ValueError, TypeError):
            logger.warning("Invalid env var %s=%r, skipping", env_key, raw)
            continue

        # Walk the nested path and set the value.
        node = config
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value

    # Special handling: HUMANE_PROXY_ENABLED_STAGES (comma-separated ints).
    stages_raw = os.environ.get("HUMANE_PROXY_ENABLED_STAGES")
    if stages_raw:
        try:
            stages = [int(s.strip()) for s in stages_raw.split(",") if s.strip()]
            config.setdefault("pipeline", {})["enabled_stages"] = stages
        except ValueError:
            logger.warning(
                "Invalid HUMANE_PROXY_ENABLED_STAGES=%r (expected comma-separated ints), skipping",
                stages_raw,
            )

    return config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_config() -> dict:
    """Return the merged configuration (cached after first call)."""
    global _cached_config
    if _cached_config is not None:
        return _cached_config
    with _lock:
        if _cached_config is not None:  # double-check
            return _cached_config
        _cached_config = _build_config()
        return _cached_config


def reload_config() -> dict:
    """Force a re-read from disk and return the fresh config."""
    global _cached_config
    with _lock:
        _cached_config = _build_config()
        return _cached_config


def _build_config() -> dict:
    """Construct the layered config: defaults → user file → env vars."""
    # 1. Package defaults
    config = _load_yaml(_PACKAGE_DEFAULTS_PATH)

    # 2. User project file
    user_path_str = os.environ.get("HUMANE_PROXY_CONFIG")
    if user_path_str:
        user_path = Path(user_path_str)
    else:
        user_path = Path.cwd() / "humane_proxy.yaml"

    user_config = _load_yaml(user_path)
    if user_config:
        logger.info("Loaded user config from %s", user_path)
        config = _deep_merge(config, user_config)

    # 3. Environment variable overrides
    config = _apply_env_overrides(config)

    return config
