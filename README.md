<p align="center">
  <h1 align="center">🛡️ HumaneProxy</h1>
  <p align="center">
    <strong>Lightweight, plug-and-play AI safety middleware that protects humans.</strong>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-how-it-works">How It Works</a> •
    <a href="#-cli">CLI</a> •
    <a href="#-programmatic-api">Programmatic API</a> •
    <a href="#-configuration">Configuration</a> •
    <a href="#-webhooks">Webhooks</a> •
    <a href="#-roadmap">Roadmap</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="Version">
    <img src="https://img.shields.io/badge/python-≥3.10-green" alt="Python">
    <img src="https://img.shields.io/badge/license-Apache_2.0-orange" alt="License">
    <img src="https://img.shields.io/badge/framework-FastAPI-009688" alt="FastAPI">
  </p>
</p>

---

HumaneProxy is an open-source safety middleware that sits between your application and any upstream LLM API. It intercepts every user message, runs a multi-stage safety pipeline to detect **self-harm ideation** and **criminal intent**, and either **forwards safe requests** or **flags and escalates dangerous ones** — with zero changes to your existing integration.

We protect **humans**, not LLMs. Our focus is on detecting genuine distress and harmful intent so operators can intervene when it matters most.

```bash
pip install humane-proxy
humane-proxy init
humane-proxy start
```

That's it. Your users are now monitored for safety.

---

## 📑 Table of Contents

1.  [Features](#-features)
2.  [Quick Start](#-quick-start)
3.  [CLI Reference](#-cli)
4.  [Programmatic API](#-programmatic-api)
5.  [How It Works](#-how-it-works)
6.  [API Reference](#-api-reference)
7.  [Configuration](#-configuration)
8.  [Webhooks](#-webhooks)
9.  [Project Structure](#-project-structure)
10. [Roadmap](#-roadmap)
11. [Contributing](#-contributing)
12. [License](#-license)

---

## ✨ Features

### v0.1 — What Ships Today

| Feature | Description |
|---------|-------------|
| **🆘 Self-Harm Detection** | Keyword + intent pattern matching for suicidal ideation and self-injury |
| **⚠️ Criminal Intent Detection** | Keyword + regex patterns for violence, weapons, drug synthesis |
| **🎯 Context-Aware False Positive Reduction** | "I want to die laughing" → safe. "Warning signs of suicide" → safe. |
| **📈 Trajectory Analysis** | Rolling-window spike detection across session history |
| **🚨 Category-Aware Escalation** | Self-harm always escalates. Criminal intent escalates above threshold. |
| **🔔 Webhook Alerts** | Fire-and-forget alerts to Slack, Discord, and PagerDuty |
| **⚡ FastAPI Proxy** | Async reverse proxy to any OpenAI-compatible LLM API |
| **🖥️ CLI** | `humane-proxy init`, `start`, `check`, `version` — zero-config workflow |
| **🐍 Programmatic API** | `from humane_proxy import HumaneProxy` — use in any Python project |
| **⚙️ Layered Config** | Package defaults → user YAML → env vars (automatic merge) |
| **💾 SQLite Audit Trail** | WAL-mode, epoch timestamps, category logging |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- An upstream LLM API endpoint (OpenAI, Groq, Anthropic, Ollama, etc.)

### Installation

```bash
pip install humane-proxy
```

Or from source:

```bash
git clone https://github.com/your-org/humane-proxy.git
cd humane-proxy
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"
```

### Initialize Your Project

```bash
humane-proxy init
```

This creates:
- `humane_proxy.yaml` — your project config (edit thresholds, keywords, webhooks)
- `.env.example` — template for `LLM_API_KEY` and `LLM_API_URL`

```bash
# Copy and fill in your secrets:
cp .env.example .env
```

### Start the Server

```bash
humane-proxy start
```

---

## 🖥 CLI

### `humane-proxy init`

Scaffolds `humane_proxy.yaml` and `.env.example` in your current directory.

### `humane-proxy start [OPTIONS]`

Starts the HumaneProxy proxy server.

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | from config | Bind address |
| `--port` / `-p` | from config | Bind port |
| `--reload` / `--no-reload` | from config | Auto-reload on code changes |

### `humane-proxy check TEXT [OPTIONS]`

Quick safety evaluation without starting the server.

```bash
$ humane-proxy check "I want to end my life"

  🆘 FLAGGED — self_harm
  Score   : 1.0
  Category: self_harm
  Triggers: self_harm_keyword:end my life, intent_pattern:self_annihilation

$ humane-proxy check "What is the capital of France?"

  ✅ SAFE
  Score   : 0.0
  Category: safe
  Triggers: (none)

$ humane-proxy check "I want to die laughing at this meme"

  ✅ SAFE
  Score   : 0.07
  Category: safe
  Triggers: self_harm_keyword:want to die, context_reduced
```

| Option | Default | Description |
|--------|---------|-------------|
| `--session` / `-s` | `"cli"` | Session ID for trajectory tracking |

### `humane-proxy version`

```bash
$ humane-proxy version
HumaneProxy v0.1.0
```

---

## 🐍 Programmatic API

Use HumaneProxy directly in your Python code — no server needed:

```python
from humane_proxy import HumaneProxy

proxy = HumaneProxy()

# Quick safety check
result = proxy.check("What is the capital of France?")
print(result)
# {"safe": True, "category": "safe", "score": 0.0, "triggers": []}

result = proxy.check("I want to end my life")
print(result)
# {"safe": False, "category": "self_harm", "score": 1.0, "triggers": ["self_harm_keyword:end my life", ...]}

# Custom config
proxy = HumaneProxy(config_path="my_config.yaml")

# Get the FastAPI app for custom mounting
app = proxy.as_fastapi_app()
```

**Return format:**
```python
{
    "safe": bool,           # True if no escalation needed
    "category": str,        # "self_harm", "criminal_intent", or "safe"
    "score": float,         # 0.0 - 1.0
    "triggers": list        # Human-readable trigger descriptions
}
```

---

## 🔬 How It Works

### The Safety Pipeline

```
  Incoming Message
        │
        ▼
  ┌─────────────────────────┐
  │ 1. WHITESPACE           │   Normalize spacing, newlines, tabs
  │    NORMALIZATION        │
  └──────────┬──────────────┘
             ▼
  ┌─────────────────────────┐
  │ 2. HEURISTIC            │   Self-harm keywords (15) + criminal
  │    CLASSIFICATION       │   keywords (12) + 6 intent patterns
  │                         │   + context reducers → (category,
  │                         │   score, triggers)
  └──────────┬──────────────┘
             ▼
  ┌─────────────────────────┐
  │ 3. TRAJECTORY           │   current_score - mean(last 5) > 0.35?
  │    SPIKE DETECTION      │   → boost score by 0.25
  └──────────┬──────────────┘
             ▼
  ┌─────────────────────────┐
  │ 4. ESCALATION           │   self_harm → ALWAYS escalate + log
  │    DECISION             │   criminal_intent ≥ 0.7 → escalate
  │                         │   safe → forward to LLM
  └─────────────────────────┘
```

### Heuristic Classifier

**15 self-harm keywords** with word-boundary matching — detect expressions of suicidal ideation and self-injury intent.

**12 criminal intent keywords** — detect requests for weapons, drugs, violence, and exploitation.

**6 intent patterns** with human-readable names:

| Name | Category | Catches |
|------|----------|---------|
| `first_person_self_harm` | self_harm | "I want to kill myself", "I plan to hurt myself" |
| `self_annihilation` | self_harm | "end my life", "take it all" |
| `method_seeking` | self_harm | "painless way to die", "easy method to end it" |
| `first_person_harm_others` | criminal_intent | "I want to kill someone", "I plan to attack" |
| `weapons_synthesis` | criminal_intent | "how to make a bomb", "how to synthesize ricin" |
| `acquisition_criminal` | criminal_intent | "where to buy drugs", "where to get a gun" |

**Context Reducers:** 22 phrases that neutralize false positives when a single trigger fires alongside contextual clues:

- **Humor/hyperbole:** "laughing", "of boredom"
- **Fiction/gaming:** "in the game", "in a novel", "my character"
- **Help-seeking:** "warning signs", "prevention", "how to help"

**Scoring:**

| Category | Keyword Score | Pattern Score | Override |
|----------|--------------|---------------|----------|
| Self-harm | 0.7 | 0.7 | Always → 1.0 (critical) |
| Criminal intent | 0.6 | 0.7 | Accumulative |

---

## 📡 API Reference

### POST /chat

Accepts **OpenAI-compatible** payloads:

```json
{
  "model": "llama-3.3-70b-versatile",
  "session_id": "user-42",
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ]
}
```

**Responses:**

| Status | Meaning |
|--------|---------|
| `200` | Safe → LLM response forwarded; or Risky → `{"status": "flagged", "category": "..."}` |
| `400` | No user message in payload |
| `503` | LLM_API_URL not configured or upstream unreachable |

---

## ⚙ Configuration

HumaneProxy uses a **layered config system** (highest priority first):

1. **`HUMANE_PROXY_*` env vars** — override any single value
2. **`humane_proxy.yaml` in CWD** — your project-level config
3. **Package defaults** — bundled sensible defaults

### Environment Variables

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | Bearer token for upstream LLM |
| `LLM_API_URL` | Upstream LLM endpoint URL |
| `HUMANE_PROXY_CONFIG` | Path to custom config YAML |
| `HUMANE_PROXY_PORT` | Override server port |
| `HUMANE_PROXY_RISK_THRESHOLD` | Override risk threshold |
| `HUMANE_PROXY_SLACK_URL` | Slack webhook URL |
| `HUMANE_PROXY_DISCORD_URL` | Discord webhook URL |
| `HUMANE_PROXY_PAGERDUTY_KEY` | PagerDuty routing key |
| `HUMANE_PROXY_DB_PATH` | Override SQLite DB location |
| `HUMANE_PROXY_RATE_LIMIT_MAX` | Override rate limit max |

---

## 🔔 Webhooks

HumaneProxy fires real-time alerts to your team when escalations occur. All webhooks are **fire-and-forget** — they never block or crash the pipeline.

Alerts include the **category** (🆘 self_harm / ⚠️ criminal_intent) for immediate triage.

### Slack

Set `HUMANE_PROXY_SLACK_URL` or configure in YAML.

### Discord

Set `HUMANE_PROXY_DISCORD_URL`. Sends rich embed messages with severity-based colors.

### PagerDuty

Set `HUMANE_PROXY_PAGERDUTY_KEY`. Triggers a critical severity event via Events API v2.

---

## 📂 Project Structure

```
humane-proxy/
├── pyproject.toml                          # Package config + CLI entry point
├── README.md
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
│
├── humane_proxy/                           # Main package
│   ├── __init__.py                         # HumaneProxy class + load_config()
│   ├── config.py                           # Layered config loader
│   ├── cli.py                              # Click-based CLI
│   ├── config.yaml                         # Package defaults
│   │
│   ├── classifiers/
│   │   ├── __init__.py
│   │   └── heuristics.py                   # Self-harm + criminal intent classifier
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   └── trajectory.py                   # Spike detector
│   │
│   ├── escalation/
│   │   ├── __init__.py
│   │   ├── router.py                       # Category-aware escalation
│   │   ├── local_db.py                     # SQLite backend
│   │   └── webhooks.py                     # Slack, Discord, PagerDuty
│   │
│   └── middleware/
│       ├── __init__.py
│       └── interceptor.py                  # FastAPI app + POST /chat
│
└── tests/                                  # Unit tests
    ├── conftest.py
    ├── test_heuristics.py
    ├── test_trajectory.py
    ├── test_local_db.py
    ├── test_router.py
    ├── test_interceptor.py
    ├── test_config.py
    ├── test_cli.py
    └── test_webhooks.py
```

---

## 🗺 Roadmap

### v0.1 (Current) ✅

- [x] Heuristic keyword & intent pattern classifier for self-harm and criminal intent
- [x] Context-aware false positive reduction
- [x] Trajectory-based spike detection
- [x] Category-aware escalation with SQLite logging
- [x] FastAPI reverse proxy with error handling
- [x] Layered YAML + env config system
- [x] CLI: `init`, `start`, `check`, `version`
- [x] Programmatic `HumaneProxy` class
- [x] Webhook alerts (Slack, Discord, PagerDuty)
- [x] pip-installable package

---

### v0.2 — ML Pipeline & Integrations

- [ ] **ML-based Semantic Classifier** — Sentence-transformer embeddings for meaning-level detection
- [ ] **3-Stage Cascade Pipeline** — Heuristics → Embeddings → Reasoning LLM
- [ ] **LlamaGuard / OpenAI Moderation** — Configurable Stage-3 classification
- [ ] **Self-harm Care Response** — Configurable empathetic crisis resource response
- [ ] **MCP Server Integration** — Expose safety tools via Model Context Protocol
- [ ] **REST Admin API** — Query, filter, and paginate escalation history
- [ ] **Enhanced Webhooks** — Email, Microsoft Teams, category-aware formatting
- [ ] **Privacy Controls** — Message hashing, no raw text storage

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/your-org/humane-proxy.git
cd humane-proxy
python -m venv venv
venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 📄 License

HumaneProxy is released under the **Apache License 2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

---

<p align="center">
  <strong>Built for a safer world.</strong>
</p>
