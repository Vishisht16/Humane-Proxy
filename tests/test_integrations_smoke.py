"""Smoke tests for framework integrations."""

import pytest
from unittest.mock import patch, MagicMock

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
