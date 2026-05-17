"""Security-focused MCP helper tests."""

import sys
import types

import pytest

from humane_proxy.escalation.query import normalize_escalation_query
from humane_proxy.mcp_server import (
    MCP_DEFAULT_HOST,
    MCP_TOKEN_ENV,
    _get_mcp_auth_provider,
    _is_public_bind_host,
    serve_http,
)


def test_http_mcp_defaults_to_localhost():
    assert MCP_DEFAULT_HOST == "127.0.0.1"
    assert serve_http.__defaults__[0] == "127.0.0.1"


def test_mcp_auth_provider_uses_configured_bearer_token(monkeypatch):
    class FakeBearerTokenAuth:
        def __init__(self, token: str):
            self.token = token

    fastmcp_module = types.ModuleType("fastmcp")
    server_module = types.ModuleType("fastmcp.server")
    auth_module = types.ModuleType("fastmcp.server.auth")
    auth_module.BearerTokenAuth = FakeBearerTokenAuth

    monkeypatch.setitem(sys.modules, "fastmcp", fastmcp_module)
    monkeypatch.setitem(sys.modules, "fastmcp.server", server_module)
    monkeypatch.setitem(sys.modules, "fastmcp.server.auth", auth_module)
    monkeypatch.setenv(MCP_TOKEN_ENV, "test-mcp-secret")

    auth = _get_mcp_auth_provider()

    assert isinstance(auth, FakeBearerTokenAuth)
    assert auth.token == "test-mcp-secret"


def test_mcp_auth_provider_is_optional(monkeypatch):
    monkeypatch.delenv(MCP_TOKEN_ENV, raising=False)
    assert _get_mcp_auth_provider() is None


@pytest.mark.parametrize(
    ("raw_limit", "expected"),
    [
        (0, 1),
        (-50, 1),
        (25, 25),
        (500, 100),
        ("not-a-number", 20),
    ],
)
def test_escalation_query_limit_is_clamped(raw_limit, expected):
    limit, category = normalize_escalation_query(raw_limit, "self_harm")

    assert limit == expected
    assert category == "self_harm"


def test_escalation_query_rejects_unknown_categories():
    with pytest.raises(ValueError, match="category must be one of"):
        normalize_escalation_query(20, "all_data")


def test_escalation_query_treats_whitespace_category_as_unfiltered():
    limit, category = normalize_escalation_query(20, "   ")

    assert limit == 20
    assert category is None


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", False),
        ("::1", False),
        ("localhost", False),
        ("0.0.0.0", True),
        ("::", True),
        ("[::]", True),
        ("192.168.1.10", True),
        ("10.0.0.5", True),
        ("mcp.example.com", True),
        ("", True),
    ],
)
def test_public_bind_detection_flags_non_loopback_hosts(host, expected):
    assert _is_public_bind_host(host) is expected
