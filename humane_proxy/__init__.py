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

__version__ = "0.2.1"

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

        # Synchronous check (Stages 1+2):
        result = proxy.check("I want to end my life")

        # Async check (all 3 stages):
        result = await proxy.check_async("I want to end my life")

        app = proxy.as_fastapi_app()
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

        # Initialise the pipeline.
        from humane_proxy.classifiers.pipeline import SafetyPipeline
        self._pipeline = SafetyPipeline(self._config)

    @property
    def config(self) -> dict:
        """Return the active merged configuration."""
        return self._config

    @property
    def pipeline(self):
        """Return the underlying SafetyPipeline instance."""
        return self._pipeline

    def check(self, text: str, session_id: str = "programmatic") -> dict:
        """Run the synchronous safety pipeline on *text* (Stages 1+2).

        Returns
        -------
        dict
            ``{"safe": bool, "category": str, "score": float, "triggers": list,
               "stage_reached": int, ...}``
        """
        result = self._pipeline.classify_sync(text, session_id)
        return result.to_dict()

    async def check_async(self, text: str, session_id: str = "programmatic") -> dict:
        """Run the full async safety pipeline on *text* (all 3 stages).

        Returns
        -------
        dict
            Same as :meth:`check`, but potentially enriched with Stage-3
            reasoning and higher accuracy.
        """
        result = await self._pipeline.classify(text, session_id)
        return result.to_dict()

    def as_fastapi_app(self):
        """Return the configured FastAPI application instance."""
        from humane_proxy.middleware.interceptor import app
        return app
