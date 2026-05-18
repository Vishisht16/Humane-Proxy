"""Smoke tests for framework integrations."""

import pytest
from unittest.mock import patch, MagicMock

from humane_proxy.risk.trajectory import analyze, session_history, _category_history

def test_llamaindex_tools_creation():
    """Verify LlamaIndex tools can be created when dependency exists."""
    import sys
    mock_li = MagicMock()
    mock_ft = MagicMock()
    mock_li.core.tools.FunctionTool = mock_ft
    sys.modules["llama_index"] = mock_li
    sys.modules["llama_index.core"] = mock_li.core
    sys.modules["llama_index.core.tools"] = mock_li.core.tools
    
    # We patch inside the module directly instead of system-wide
    with patch("humane_proxy.integrations.llamaindex.FunctionTool", mock_ft, create=True):
        try:
            from humane_proxy.integrations.llamaindex import get_safety_tools
            tools = get_safety_tools()
            assert len(tools) == 3
            assert mock_ft.from_defaults.call_count == 3
        finally:
            del sys.modules["llama_index"]
            del sys.modules["llama_index.core"]
            del sys.modules["llama_index.core.tools"]

def test_crewai_tools_creation():
    """Verify CrewAI tools can be created when dependency exists."""
    import sys
    from pydantic import BaseModel
    
    class MockBaseTool:
        pass
        
    mock_crewai = MagicMock()
    mock_crewai.tools.BaseTool = MockBaseTool
    sys.modules["crewai"] = mock_crewai
    sys.modules["crewai.tools"] = mock_crewai.tools
    
    try:
        from humane_proxy.integrations.crewai import get_safety_tools
        tools = get_safety_tools()
        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "check_message_safety" in tool_names
    finally:
        del sys.modules["crewai"]
        del sys.modules["crewai.tools"]

def test_autogen_tools_creation():
    """Verify AutoGen tool registration logic."""
    from humane_proxy.integrations.autogen import get_safety_functions, register_safety_tools
    
    funcs = get_safety_functions()
    assert len(funcs) == 3
    assert hasattr(funcs[0]["function"], "__call__")
    
    # Mock agents and the autogen module import
    import sys
    sys.modules["autogen"] = MagicMock()
    
    try:
        mock_assistant = MagicMock()
        mock_proxy = MagicMock()
        
        # Setup the chained call: register_for_llm(name, desc)(fn)
        mock_decorator = MagicMock()
        mock_assistant.register_for_llm.return_value = mock_decorator
        
        register_safety_tools(mock_assistant, mock_proxy)
        
        assert mock_assistant.register_for_llm.call_count == 3
        assert mock_proxy.register_for_execution.call_count == 3
    finally:
        del sys.modules["autogen"]


def test_autogen_session_risk_is_read_only():
    from humane_proxy.integrations.autogen import get_session_risk

    sid = "autogen-risk-read-only"
    analyze(sid, 0.4, "safe")
    before_count = len(session_history[sid])

    get_session_risk(sid)

    assert len(session_history[sid]) == before_count
    assert len(_category_history[sid]) == before_count


def test_llamaindex_session_risk_is_read_only():
    from humane_proxy.integrations.llamaindex import _get_session_risk

    sid = "llamaindex-risk-read-only"
    analyze(sid, 0.6, "criminal_intent")
    before_count = len(session_history[sid])

    result = _get_session_risk(sid)

    assert result["message_count"] == before_count
    assert len(session_history[sid]) == before_count
    assert len(_category_history[sid]) == before_count


def test_crewai_session_risk_is_read_only():
    import json
    import sys

    class MockBaseTool:
        pass

    mock_crewai = MagicMock()
    mock_crewai.tools.BaseTool = MockBaseTool
    sys.modules["crewai"] = mock_crewai
    sys.modules["crewai.tools"] = mock_crewai.tools

    try:
        from humane_proxy.integrations.crewai import get_safety_tools

        sid = "crewai-risk-read-only"
        analyze(sid, 0.7, "self_harm")
        before_count = len(session_history[sid])

        tools = get_safety_tools()
        risk_tool = next(tool for tool in tools if tool.name == "get_session_risk")
        result = json.loads(risk_tool._run(sid))

        assert result["message_count"] == before_count
        assert len(session_history[sid]) == before_count
        assert len(_category_history[sid]) == before_count
    finally:
        del sys.modules["crewai"]
        del sys.modules["crewai.tools"]
