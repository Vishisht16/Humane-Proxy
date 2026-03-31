# 🛡️ HumaneProxy

**Lightweight, plug-and-play AI safety middleware that protects humans.**

HumaneProxy sits between your users and any LLM. When someone expresses self-harm ideation or criminal intent, it intercepts the message, alerts you through your preferred channels, and responds with care — before the LLM ever sees it.

[![PyPI](https://img.shields.io/pypi/v/humane-proxy.svg)](https://pypi.org/project/humane-proxy/)
[![Python](https://img.shields.io/pypi/pyversions/humane-proxy.svg)](https://pypi.org/project/humane-proxy/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/Vishisht16/Humane-Proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/Vishisht16/Humane-Proxy/actions/workflows/ci.yml)

---

## What it does

```
User message → HumaneProxy → (safe?) → Upstream LLM → Response
                    ↓
              (self_harm or criminal_intent?)
                    ↓
              Empathetic care response  +  Operator alert
```

- 🆘 **Self-harm detected** → Blocked with international crisis resources. Operator notified.
- ⚠️ **Criminal intent detected** → Blocked or flagged. Operator notified.
- ✅ **Safe** → Forwarded to your LLM transparently.

Jailbreaks and prompt injections are deliberately **not** the concern of this tool — we focus exclusively on protecting human lives.

---

## Quick Start

```bash
pip install humane-proxy

# Scaffold config in your project directory
humane-proxy init

# Start the proxy (set LLM_API_KEY and LLM_API_URL in .env first)
humane-proxy start
```

### As a Python library

```python
from humane_proxy import HumaneProxy

proxy = HumaneProxy()

# Sync check (Stages 1+2)
result = proxy.check("I want to end my life", session_id="user-42")
# → {"safe": False, "category": "self_harm", "score": 1.0, "triggers": [...]}

# Async check (all 3 stages)
result = await proxy.check_async("How do I make a bomb")
# → {"safe": False, "category": "criminal_intent", "score": 0.9, ...}
```

---

## 3-Stage Cascade Pipeline

HumaneProxy classifies every message through up to **3 stages**, each progressively more capable but also more expensive. Stages exit early when confident.

```
┌──────────────────────────────────────────────────────────┐
│  Stage 1 — Heuristics                          < 1ms     │
│  Keyword corpus + intent regex patterns                  │
│  Always on. Catches clear cases instantly.               │
└──────────────────────────────────────────────────────────┘
             ↓ (ambiguous or medium-score)
┌──────────────────────────────────────────────────────────┐
│  Stage 2 — Semantic Embeddings               ~100ms      │
│  sentence-transformers cosine similarity                 │
│  vs. curated anchor sentences (self-harm + criminal)     │
│  Optional: pip install humane-proxy[ml]                  │
└──────────────────────────────────────────────────────────┘
             ↓ (still ambiguous)
┌──────────────────────────────────────────────────────────┐
│  Stage 3 — Reasoning LLM                     ~1–3s      │
│  LlamaGuard (Groq) or OpenAI Moderation API              │
│  Optional: set OPENAI_API_KEY or GROQ_API_KEY            │
└──────────────────────────────────────────────────────────┘
```

### Configuring the Pipeline

In `humane_proxy.yaml`:

```yaml
pipeline:
  # Which stages to run. [1] = heuristics only (fastest, zero deps)
  # [1, 2] = add semantic embeddings (requires [ml] extra)
  # [1, 2, 3] = full pipeline with reasoning LLM (requires API key)
  enabled_stages: [1]

  # Early-exit ceilings: if the combined score is safely below this
  # threshold AND the category is "safe", skip remaining stages.
  stage1_ceiling: 0.3    # exit after Stage 1 if score ≤ 0.3 and safe
  stage2_ceiling: 0.4    # exit after Stage 2 if score ≤ 0.4 and safe
```

### Stage 2 — Semantic Embeddings

Requires the `[ml]` extra:

```bash
pip install humane-proxy[ml]
```

In `humane_proxy.yaml`:

```yaml
pipeline:
  enabled_stages: [1, 2]

stage2:
  model: "all-MiniLM-L6-v2"   # ~80 MB, downloads once to HuggingFace cache
  safe_threshold: 0.35         # cosine similarity below this → safe
```

The model lazy-loads on first use. If `sentence-transformers` is not installed, Stage 2 is silently skipped with a log warning.

### Stage 3 — Reasoning LLM

Set your API key and optionally configure the provider:

```bash
# Option A — OpenAI Moderation (free with any OpenAI key):
export OPENAI_API_KEY=sk-...

# Option B — LlamaGuard via Groq (free tier, very fast):
export GROQ_API_KEY=gsk_...
```

In `humane_proxy.yaml`:

```yaml
pipeline:
  enabled_stages: [1, 2, 3]

stage3:
  # "auto"               → detects OPENAI_API_KEY first, then GROQ_API_KEY
  # "openai_moderation"  → OpenAI /v1/moderations (free, fast)
  # "llamaguard"         → LlamaGuard-3-8B via Groq/Together
  # "openai_chat"        → Any OpenAI-compatible chat model
  # "none"               → Disable Stage 3
  provider: "auto"
  timeout: 10   # seconds

  openai_moderation:
    api_url: "https://api.openai.com/v1/moderations"

  llamaguard:
    api_url: "https://api.groq.com/openai/v1/chat/completions"
    model: "meta-llama/llama-guard-3-8b"

  openai_chat:
    api_url: "https://api.openai.com/v1/chat/completions"
    model: "gpt-4o-mini"
```

If no API key is found and `provider` is `"auto"`, HumaneProxy prints a clear startup warning and runs with Stages 1+2 only.

---

## Self-Harm Care Response

When self-harm is detected, HumaneProxy can respond in two ways:

### Mode B — Block (default)

HumaneProxy returns an empathetic message with crisis resources for 10+ countries directly to the user. Your LLM is never involved.

```yaml
safety:
  categories:
    self_harm:
      response_mode: "block"     # default

      # Optional: override the built-in message
      block_message: "We're here for you. Please reach out to..."
```

Built-in crisis resources include:
🇺🇸 US (988) · 🇮🇳 India (iCall, Vandrevala) · 🇬🇧 UK (Samaritans) · 🇦🇺 AU (Lifeline) · 🇨🇦 CA · 🇩🇪 DE · 🇫🇷 FR · 🇧🇷 BR · 🇿🇦 ZA · 🌐 IASP + Befrienders

### Mode A — Forward with care context

Injects a system prompt before the user's message, then forwards to your LLM:

```yaml
safety:
  categories:
    self_harm:
      response_mode: "forward"
```

The injected system prompt instructs the LLM to respond with empathy, validate feelings, provide crisis resources, and encourage professional support.

---

## Alert Webhooks

Configure in `humane_proxy.yaml`:

```yaml
escalation:
  rate_limit_max: 3            # max alerts per session per window
  rate_limit_window_hours: 1

  webhooks:
    slack_url: "https://hooks.slack.com/services/..."
    discord_url: "https://discord.com/api/webhooks/..."
    pagerduty_routing_key: "your-routing-key"
    teams_url: "https://outlook.office.com/webhook/..."

    # Email alerts via SMTP (stdlib, no extra deps)
    email:
      host: "smtp.gmail.com"
      port: 587
      use_tls: true
      username: "your@gmail.com"
      password: "app-password"
      from: "humane-proxy@yourorg.com"
      to:
        - "safety-team@yourorg.com"
        - "oncall@yourorg.com"
```

---

## CLI Reference

```bash
# Safety check
humane-proxy check "I want to end my life"
# 🆘 FLAGGED — self_harm
# Score   : 1.0
# Category: self_harm

# List recent escalations
humane-proxy escalations
humane-proxy escalations --category self_harm --limit 50

# Session risk history
humane-proxy session user-42

# Start proxy server
humane-proxy start [--host 0.0.0.0] [--port 8000]

# MCP server (requires [mcp] extra)
humane-proxy mcp-serve
```

---

## REST Admin API

Mounted at `/admin`, secured with `HUMANE_PROXY_ADMIN_KEY` Bearer token:

```bash
export HUMANE_PROXY_ADMIN_KEY=your-secret-key

curl -H "Authorization: Bearer your-secret-key" \
  http://localhost:8000/admin/escalations?category=self_harm&limit=10

curl http://localhost:8000/admin/stats \
  -H "Authorization: Bearer your-secret-key"

# Delete session data (right to erasure)
curl -X DELETE http://localhost:8000/admin/sessions/user-42 \
  -H "Authorization: Bearer your-secret-key"
```

| Endpoint | Description |
|---|---|
| `GET /admin/escalations` | Paginated list, filterable by `category`, `session_id` |
| `GET /admin/escalations/{id}` | Single escalation detail |
| `GET /admin/sessions/{id}/risk` | Session history + trajectory |
| `GET /admin/stats` | Aggregate counts by category and day |
| `DELETE /admin/sessions/{id}` | Delete all session records |

---

## MCP Server (for AI Agents)

```bash
pip install humane-proxy[mcp]
humane-proxy mcp-serve
```

Exposes three tools via Model Context Protocol:

| Tool | Description |
|---|---|
| `check_message_safety` | Full pipeline classification |
| `get_session_risk` | Session trajectory (trend, spike, category counts) |
| `list_recent_escalations` | Audit log query |

Deploy to Smithery using the included `smithery.yaml`.

---

## Configuration Reference

All values can be set in `humane_proxy.yaml` (project root) or via `HUMANE_PROXY_*` environment variables. Environment variables always win.

| YAML key | Env var | Default | Description |
|---|---|---|---|
| `safety.risk_threshold` | `HUMANE_PROXY_RISK_THRESHOLD` | `0.7` | Score threshold for criminal_intent escalation |
| `safety.spike_boost` | — | `0.25` | Score boost on trajectory spike |
| `server.port` | `HUMANE_PROXY_PORT` | `8000` | Proxy port |
| `pipeline.enabled_stages` | `HUMANE_PROXY_ENABLED_STAGES` | `[1]` | Active stages |
| `pipeline.stage1_ceiling` | `HUMANE_PROXY_STAGE1_CEILING` | `0.3` | Early exit after Stage 1 |
| `pipeline.stage2_ceiling` | `HUMANE_PROXY_STAGE2_CEILING` | `0.4` | Early exit after Stage 2 |
| `stage3.provider` | `HUMANE_PROXY_STAGE3_PROVIDER` | `"auto"` | Stage 3 provider |
| `stage3.timeout` | `HUMANE_PROXY_STAGE3_TIMEOUT` | `10` | Stage 3 timeout (s) |
| `privacy.store_message_text` | — | `false` | Store raw text (vs SHA-256 hash) |
| `escalation.rate_limit_max` | — | `3` | Max alerts per session/window |
| `safety.categories.self_harm.response_mode` | — | `"block"` | `"block"` or `"forward"` |

---

## Privacy

By default HumaneProxy **never stores raw message text**. Only a SHA-256 hash is persisted for correlation. The escalation DB stores:

- `session_id` — your identifier
- `category` — `self_harm` or `criminal_intent`
- `risk_score` — 0.0–1.0
- `triggers` — which patterns fired
- `message_hash` — SHA-256 of the original text
- `stage_reached` — which pipeline stage produced the result
- `reasoning` — Stage-3 LLM reasoning (if available)

To enable raw text storage (e.g. for human review):

```yaml
privacy:
  store_message_text: true
```

---

## Installation Extras

| Extra | Command | What it adds |
|---|---|---|
| *(none)* | `pip install humane-proxy` | Stage 1 heuristics + full API + CLI |
| `ml` | `pip install humane-proxy[ml]` | Stage 2 semantic embeddings (`sentence-transformers`) |
| `mcp` | `pip install humane-proxy[mcp]` | MCP server for AI agent integration (`fastmcp`) |
| `all` | `pip install humane-proxy[all]` | Everything above |

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Copyright 2026 Vishisht Mishra (@Vishisht16). Any attribution is appreciated.

See [NOTICE](NOTICE) for full attribution information.

---

Built for a safer world.
