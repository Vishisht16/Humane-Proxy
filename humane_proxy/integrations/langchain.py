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

"""LangChain integration for HumaneProxy.

Provides helpers to plug HumaneProxy's MCP safety tools into any
LangChain or LangGraph agent via ``langchain-mcp-adapters``.

Quick start::

    from humane_proxy.integrations.langchain import get_safety_tools
    tools = await get_safety_tools()
    # → [check_message_safety, get_session_risk, list_recent_escalations]

Requires::

    pip install humane-proxy[mcp] langchain-mcp-adapters
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("humane_proxy.integrations.langchain")


def get_langchain_mcp_config() -> dict[str, Any]:
    """Return a ``MultiServerMCPClient``-compatible config dict.

    Use this to connect a LangChain agent to HumaneProxy's MCP tools::

        from langchain_mcp_adapters.client import MultiServerMCPClient
        from humane_proxy.integrations.langchain import get_langchain_mcp_config

        config = get_langchain_mcp_config()
        async with MultiServerMCPClient(config) as client:
            tools = client.get_tools()
            # pass tools to your agent

    Returns
    -------
    dict
        Configuration for ``MultiServerMCPClient`` with HumaneProxy
        connected via stdio transport.
    """
    return {
        "humane-proxy": {
            "command": "humane-proxy",
            "args": ["mcp-serve"],
            "transport": "stdio",
        }
    }


async def get_safety_tools() -> list:
    """Convenience function: connect to HumaneProxy MCP and return LangChain tools.

    This starts the MCP server as a subprocess, connects via stdio, and
    returns a list of LangChain-compatible tools ready to be passed to
    ``create_react_agent()`` or similar.

    Returns
    -------
    list
        List of LangChain ``Tool`` objects:
        ``check_message_safety``, ``get_session_risk``, ``list_recent_escalations``.

    Raises
    ------
    ImportError
        If ``langchain-mcp-adapters`` is not installed.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        raise ImportError(
            "LangChain integration requires langchain-mcp-adapters.\n"
            "Install with: pip install langchain-mcp-adapters"
        )

    config = get_langchain_mcp_config()
    async with MultiServerMCPClient(config) as client:
        tools = client.get_tools()
        logger.info("Loaded %d HumaneProxy safety tools via MCP", len(tools))
        return tools
