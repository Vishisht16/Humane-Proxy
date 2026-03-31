# Contributing to HumaneProxy

Thanks for your interest in contributing! HumaneProxy is a community project that aims to protect human lives through AI safety middleware. Every contribution matters.

## Getting Started

### 1. Fork & clone

```bash
git clone https://github.com/<your-username>/Humane-Proxy.git
cd Humane-Proxy
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

### 3. Install in development mode

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode plus `pytest` and `pytest-asyncio`.

### 4. Run the test suite

```bash
pytest tests/ -v
```

All tests should pass before you start making changes.

---

## Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b my-feature
   ```
2. **Make your changes** — keep commits focused and well-described.
3. **Add or update tests** for any new or changed behaviour (see _Writing Tests_ below).
4. **Run the full test suite** and make sure everything passes:
   ```bash
   pytest tests/ -v
   ```
5. **Push and open a PR** against `main`.

---

## Writing Tests

Tests live in the `tests/` directory. We use **pytest** with **pytest-asyncio** for async tests.

### Where to put your test

| You're changing… | Test file |
|---|---|
| Heuristic keyword/regex rules | `test_heuristics.py` |
| Pipeline cascade logic | `test_pipeline.py` |
| Stage-3 providers | `test_stage3.py` |
| Embedding classifier | `test_embedding_classifier.py` |
| Escalation logic / care response | `test_router.py` / `test_care_response.py` |
| Admin API endpoints | `test_admin_api.py` |
| Webhooks | `test_webhooks.py` / `test_enhanced_webhooks.py` |
| Interceptor (FastAPI middleware) | `test_interceptor.py` |
| Trajectory / risk analysis | `test_trajectory.py` |

### Example: adding a heuristic test

```python
# tests/test_heuristics.py

from humane_proxy.classifiers.heuristics import classify

class TestMyNewKeyword:
    def test_detects_new_phrase(self):
        category, score, triggers = classify("some dangerous phrase")
        assert category == "criminal_intent"
        assert score > 0.0
        assert any("keyword" in t for t in triggers)

    def test_safe_variation_not_flagged(self):
        category, score, triggers = classify("safe version of that phrase")
        assert category == "safe"
```

### Example: testing an async Stage-3 provider

```python
# tests/test_stage3.py

from unittest.mock import AsyncMock, patch
import pytest

class TestMyProvider:
    @pytest.mark.asyncio
    async def test_safe_response(self):
        # Mock the HTTP call
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.json.return_value = {"safe": True}
            mock_post.return_value.status_code = 200

            # ... test your provider
```

### Guidelines

- **No real API calls.** All external services (OpenAI, Groq, webhooks) must be mocked.
- **Deterministic.** Tests should produce the same result every time.
- **Fast.** The full suite runs in under 15 seconds — keep it that way.
- **Descriptive names.** `test_self_harm_keyword_detected` > `test_1`.

---

## Project Structure

```
humane_proxy/
├── classifiers/          # Stage 1 (heuristics), Stage 2 (embeddings), Stage 3 (LLM)
│   ├── heuristics.py     # Keyword + regex rules
│   ├── embedding_classifier.py
│   ├── pipeline.py       # 3-stage cascade orchestrator
│   └── stage3/           # LlamaGuard, OpenAI moderation, OpenAI chat
├── escalation/           # DB logging, webhooks, care response
├── middleware/            # FastAPI interceptor
├── api/                  # REST admin API
├── risk/                 # Trajectory analysis
├── mcp_server.py         # MCP server for AI agents
├── cli.py                # CLI commands
├── config.py             # Layered config system
└── config.yaml           # Package defaults
```

---

## Sensitivity Notice

This project deals with **self-harm and criminal intent detection**. When writing tests or adding keywords:

- Use established clinical/research terminology, not sensationalised language.
- Test data should be minimal and clearly marked as test fixtures.
- If you're adding crisis resources, verify the numbers/URLs are current and correct.

---

## Questions?

Open an issue or reach out to [@Vishisht16](https://github.com/Vishisht16). We're happy to help you get started.
