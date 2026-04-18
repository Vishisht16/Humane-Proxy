# 🛡️ HumaneProxy — Launch Guide

## What is HumaneProxy?

HumaneProxy is a lightweight, plug-and-play AI safety middleware that sits between users and any LLM. It intercepts messages expressing **self-harm ideation** or **criminal intent** before the LLM ever sees them — responding with empathetic care resources and alerting the operator through Slack, Discord, PagerDuty, or custom webhooks.

Unlike jailbreak or prompt-injection detectors, HumaneProxy is built exclusively to **protect human lives**. When someone in crisis reaches out to a chatbot, HumaneProxy ensures they receive immediate help instead of a generated response.

---

## How It Works — The 3-Stage Safety Pipeline

HumaneProxy classifies every message through up to **3 stages**, each progressively more capable:

| Stage | Method | Latency | What it catches |
|---|---|---|---|
| **Stage 1** | Heuristic keyword matching + intent regex | < 1ms | Clear cases — explicit self-harm phrases, criminal how-to queries |
| **Stage 2** | Semantic embedding similarity (sentence-transformers) | ~100ms | Subtle, paraphrased, or coded language that evades keyword lists |
| **Stage 3** | Reasoning LLM (LlamaGuard via Groq / OpenAI Moderation) | ~1–3s | Ambiguous cases requiring contextual judgment |

Messages exit the pipeline as soon as a definitive classification is made — most safe messages never leave Stage 1.

---

## Prerequisites

- **Python 3.10+**
- **pip** or **uv** package manager
- (Optional) An OpenAI or Groq API key for Stage 3 reasoning

---

## Quick Setup for MCP Clients

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "humane-proxy": {
      "command": "uvx",
      "args": ["--from", "humane-proxy[mcp]", "humane-proxy", "mcp-serve"],
      "env": {}
    }
  }
}
```

### Cursor / Windsurf

Add to your MCP settings:

```json
{
  "mcpServers": {
    "humane-proxy": {
      "command": "uvx",
      "args": ["--from", "humane-proxy[mcp]", "humane-proxy", "mcp-serve"],
      "env": {}
    }
  }
}
```

> **Note:** No API keys are required for Stage 1 (heuristic) classification. Add `OPENAI_API_KEY` or `GROQ_API_KEY` to the `env` block to enable Stage 3 LLM reasoning.

---

## Available MCP Tools

Once connected, your AI agent has access to these tools:

### `check_message_safety`

Classify a message for self-harm or criminal intent through the full pipeline.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `message` | string | ✅ | The user message to classify |
| `session_id` | string | ❌ | Session identifier for trajectory tracking (default: `"mcp-default"`) |

**Returns:** `{ "safe": bool, "category": str, "score": float, "triggers": list, "stage_reached": int, "should_escalate": bool }`

### `get_session_risk`

Return the current risk trajectory for a session, including spike detection and trend analysis.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | ✅ | The session identifier to query |

**Returns:** `{ "spike_detected": bool, "trend": str, "window_scores": list, "category_counts": dict, "message_count": int }`

### `list_recent_escalations`

Return recent escalation events from the audit log.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `limit` | int | ❌ | Maximum events to return (default: 20) |
| `category` | string | ❌ | Filter by `"self_harm"` or `"criminal_intent"` |

**Returns:** List of escalation records with session IDs, scores, timestamps, and triggers.

---

## Configuration

### Key Environment Variables

| Variable | Description | Required |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key for Stage 3 reasoning | No |
| `GROQ_API_KEY` | Groq API key for LlamaGuard Stage 3 | No |
| `HUMANE_PROXY_ENABLED_STAGES` | Active stages as JSON array (e.g., `"[1, 2, 3]"`) | No (default: `[1]`) |
| `HUMANE_PROXY_RISK_THRESHOLD` | Score threshold for escalation (default: `0.7`) | No |
| `HUMANE_PROXY_DECAY_HALF_LIFE` | Time-decay half-life in hours (default: `24.0`) | No |

### Configuration File

Run `humane-proxy init` to scaffold a `humane_proxy.yaml` in your project directory. This file lets you customise keywords, thresholds, webhook URLs, and pipeline behaviour.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| **Stage 2 not running** | Install the ML extra: `pip install humane-proxy[ml]` |
| **Stage 3 not running** | Set `OPENAI_API_KEY` or `GROQ_API_KEY` and enable Stage 3: `HUMANE_PROXY_ENABLED_STAGES="[1, 2, 3]"` |
| **MCP tools not detected** | Ensure `fastmcp` is installed: `pip install humane-proxy[mcp]` |
| **"Module not found" errors** | Run `pip install humane-proxy[all]` to install all dependencies |

---

## Links

- **GitHub:** [Vishisht16/Humane-Proxy](https://github.com/Vishisht16/Humane-Proxy)
- **PyPI:** [humane-proxy](https://pypi.org/project/humane-proxy/)
- **MCP Marketplace:** [humane-proxy](https://mcp-marketplace.io/server/io-github-vishisht16-humane-proxy)
- **Glama Registry:** [Humane-Proxy](https://glama.ai/mcp/servers/Vishisht16/Humane-Proxy)
