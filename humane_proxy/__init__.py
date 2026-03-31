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

"""HumaneProxy — lightweight AI safety middleware that protects humans."""

from __future__ import annotations

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Legacy API — keep backward compatibility with existing modules that call
# ``from humane_proxy import load_config``.
# ---------------------------------------------------------------------------

from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config() -> dict:
    """Load the package-level config.yaml (legacy, used by existing modules).

    New code should use :func:`humane_proxy.config.get_config` instead.
    """
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Plug-and-play public API
# ---------------------------------------------------------------------------

from humane_proxy.config import get_config as _get_config  # noqa: E402


class HumaneProxy:
    """High-level, plug-and-play interface to the HumaneProxy safety pipeline.

    Usage::

        from humane_proxy import HumaneProxy

        proxy = HumaneProxy()
        result = proxy.check("I want to end my life")
        # {"safe": False, "category": "self_harm", "score": 1.0, "triggers": [...]}

        app = proxy.as_fastapi_app()
        # Returns the configured FastAPI application
    """

    def __init__(self, config_path: str | None = None) -> None:
        import os
        if config_path:
            os.environ["HUMANE_PROXY_CONFIG"] = str(config_path)

        from humane_proxy.config import reload_config
        self._config = reload_config()

        # Ensure DB is initialised.
        from humane_proxy.escalation.local_db import init_db
        init_db()

    @property
    def config(self) -> dict:
        """Return the active merged configuration."""
        return self._config

    def check(self, text: str, session_id: str = "programmatic") -> dict:
        """Run the full safety pipeline on *text* and return a result dict.

        Returns
        -------
        dict
            ``{"safe": bool, "category": str, "score": float, "triggers": list[str]}``
        """
        from humane_proxy.classifiers.heuristics import classify
        from humane_proxy.risk.trajectory import detect_spike

        category, score, triggers = classify(text)
        triggers = triggers or []

        spike_boost = self._config.get("safety", {}).get("spike_boost", 0.25)
        threshold = self._config.get("safety", {}).get("risk_threshold", 0.7)

        is_spike = detect_spike(session_id, score)
        if is_spike:
            score = min(score + spike_boost, 1.0)
            triggers.append("trajectory_spike")

        is_safe = not (
            category == "self_harm"
            or (category == "criminal_intent" and score >= threshold)
        )

        return {
            "safe": is_safe,
            "category": category,
            "score": round(score, 4),
            "triggers": triggers,
        }

    def as_fastapi_app(self):
        """Return the configured FastAPI application instance."""
        from humane_proxy.middleware.interceptor import app
        return app
