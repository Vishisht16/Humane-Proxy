"""
Unit and integration tests for code examples in examples/ directory.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import pytest


# --- 1. FastAPI Middleware Tests ---
@pytest.fixture
def fastapi_client():
    from examples.fastapi_middleware import app as fastapi_app
    with TestClient(fastapi_app) as client:
        yield client


def test_fastapi_safe_message(fastapi_client):
    mock_proxy = MagicMock()
    mock_proxy.check_async = AsyncMock(return_value={"safe": True})

    with patch("examples.fastapi_middleware.get_proxy", return_value=mock_proxy):
        response = fastapi_client.post("/chat", json={"message": "Hello!"})
        assert response.status_code == 200
        assert "Processed safe message" in response.json()["reply"]


def test_fastapi_unsafe_message(fastapi_client):
    mock_proxy = MagicMock()
    mock_proxy.check_async = AsyncMock(
        return_value={
            "safe": False,
            "care_response": "Unsafe content detected.",
        }
    )

    with patch("examples.fastapi_middleware.get_proxy", return_value=mock_proxy):
        response = fastapi_client.post("/chat", json={"message": "Unsafe text"})
        assert response.status_code == 200
        assert response.json()["flagged"] is True
        assert response.json()["reply"] == "Unsafe content detected."


# --- 2. Flask Integration Tests ---
@pytest.fixture
def flask_client():
    pytest.importorskip("flask")
    from examples.flask_integration import app as flask_app

    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def test_flask_safe_message(flask_client):
    mock_proxy = MagicMock()
    mock_proxy.check.return_value = {"safe": True}

    with patch("examples.flask_integration.get_proxy", return_value=mock_proxy):
        response = flask_client.post("/chat", json={"message": "Hello!"})
        assert response.status_code == 200
        assert "Processed safe message" in response.get_json()["reply"]


def test_flask_unsafe_message(flask_client):
    mock_proxy = MagicMock()
    mock_proxy.check.return_value = {
        "safe": False,
        "care_response": "Unsafe content detected.",
    }

    with patch("examples.flask_integration.get_proxy", return_value=mock_proxy):
        response = flask_client.post("/chat", json={"message": "Unsafe text"})
        assert response.status_code == 200
        data = response.get_json()
        assert data["flagged"] is True
        assert data["reply"] == "Unsafe content detected."


# --- 3. OpenAI Wrapper Tests ---
def test_openai_wrapper_safe():
    pytest.importorskip("openai")
    from examples.openai_proxy_wrapper import safe_chat_completion

    with patch("examples.openai_proxy_wrapper.proxy.check") as mock_check, patch(
        "examples.openai_proxy_wrapper.client"
    ) as mock_client:
        mock_check.return_value = {"safe": True}
        mock_client.chat.completions.create.return_value = "Mocked Response"

        messages = [{"role": "user", "content": "How's the weather?"}]
        result = safe_chat_completion(messages)

        assert result["flagged"] is False
        assert result["response"] == "Mocked Response"


def test_openai_wrapper_unsafe():
    pytest.importorskip("openai")
    from examples.openai_proxy_wrapper import safe_chat_completion

    with patch("examples.openai_proxy_wrapper.proxy.check") as mock_check:
        mock_check.return_value = {
            "safe": False,
            "care_response": "Cannot fulfill request.",
        }

        messages = [{"role": "user", "content": "Bad query"}]
        result = safe_chat_completion(messages)

        assert result["flagged"] is True
        assert result["reply"] == "Cannot fulfill request."