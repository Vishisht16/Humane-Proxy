<p align="center">
  <img src="https://raw.githubusercontent.com/Vishisht16/Humane-Proxy/main/docs/assets/banner.png" alt="HumaneProxy" width="100%">
</p>

<!-- mcp-name: io.github.Vishisht16/humane-proxy -->

**Lightweight, plug-and-play AI safety middleware that protects humans.**

HumaneProxy sits between your users and any LLM. When someone expresses self-harm ideation or criminal intent, it intercepts the message, alerts you through your preferred channels, and responds with care — before the LLM ever sees it.

[![PyPI](https://img.shields.io/pypi/v/humane-proxy.svg)](https://pypi.org/project/humane-proxy/)
[![Python](https://img.shields.io/pypi/pyversions/humane-proxy.svg)](https://pypi.org/project/humane-proxy/)
[![Downloads](https://static.pepy.tech/badge/humane-proxy)](https://pepy.tech/projects/humane-proxy)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/Vishisht16/Humane-Proxy/actions/workflows/tests.yaml/badge.svg)](https://github.com/Vishisht16/Humane-Proxy/actions/workflows/tests.yaml)
[![Humane-Proxy MCP server](https://glama.ai/mcp/servers/Vishisht16/Humane-Proxy/badges/score.svg)](https://glama.ai/mcp/servers/Vishisht16/Humane-Proxy)
[![MCP Marketplace](https://img.shields.io/badge/MCP_Marketplace-Available-brightgreen)](https://mcp-marketplace.io/server/io-github-vishisht16-humane-proxy)

---

## What it does

```
User message → HumaneProxy → (safe?) → Upstream LLM → Response
                    ↓
              (self_harm or criminal_intent?)
                    ↓
              Empathetic care response  +  Operator alert
```

- **Self-harm detected** → Blocked with international crisis resources. Operator notified.
- **Criminal intent detected** → Blocked or flagged. Operator notified.
- **Safe** → Forwarded to your LLM transparently.

Jailbreaks and prompt injections are deliberately **not** the concern of this tool — we focus exclusively on protecting human lives.

---

## Quick Start

```bash
pip install humane-proxy

# Scaffold config in your project directory
humane-proxy init

# Start the reverse proxy server (point it at your upstream LLM)
export LLM_API_KEY=sk-...
export LLM_API_URL=https://api.your-llm.com/v1/chat/completions
humane-proxy start
```

### As a Python library

```python
from humane_proxy import HumaneProxy

proxy = HumaneProxy()

result = proxy.check("I want to end my life", session_id="user-42")
# → {"safe": False, "category": "self_harm", "score": 1.0, "triggers": [...]}
```

### As an MCP server (Claude Desktop, Cursor, any agent)

```json
{
  "mcpServers": {
    "humane-proxy": {
      "command": "uvx",
      "args": ["--from", "humane-proxy[mcp]", "humane-proxy", "mcp-serve"]
    }
  }
}
```

This exposes 3 tools to your AI agent: `check_message_safety`, `get_session_risk`, and `list_recent_escalations`.

## Direct Integration (FastAPI & Flask)

To add HumaneProxy to an existing backend instead of running a separate proxy server, see the runnable examples in [`examples/`](./examples/):

- [FastAPI middleware](./examples/fastapi_middleware.py) uses `proxy.check_async()` to screen requests asynchronously.
- [Flask integration](./examples/flask_integration.py) uses `proxy.check()` in a `before_request` hook.
- [OpenAI wrapper](./examples/openai_proxy_wrapper.py) screens prompts before sending them to the OpenAI client.

Install the example dependencies with `pip install -r examples/requirements.txt`.

---

## How it works

Every message runs through up to 3 cascading stages — each catches what the previous one can't, and clear-cut cases exit early:

| Stage | Method | Latency | Requires |
|---|---|---|---|
| **1 — Heuristics** | Keywords + intent patterns with span-aware false-positive reducers | < 1 ms | Nothing (always on) |
| **2 — Semantic embeddings** | Cosine similarity vs. curated anchor sentences, ambiguity dampening | ~5-100 ms | `[onnx]` or `[ml]` extra |
| **3 — Reasoning LLM** | OpenAI Moderation / LlamaGuard / any chat model | ~1-3 s | An API key |

Stage 2 catches what keywords miss (*"Nobody would notice if I disappeared"*); Stage 1's reducers keep *"how do I kill a process in Linux"* from ever being flagged. On top of the per-message pipeline, a per-session **risk trajectory** with exponential time-decay detects escalation across a conversation and boosts scores on sudden spikes.

Full details: [Pipeline documentation](https://github.com/Vishisht16/Humane-Proxy/blob/main/docs/PIPELINE.md).

---

## Benchmarks

Evaluated on two public datasets — [SimpleSafetyTests](https://huggingface.co/datasets/Bertievidgen/SimpleSafetyTests) (100 clearly unsafe prompts) for recall, and [XSTest](https://github.com/paul-rottger/exaggerated-safety) (250 safe-but-alarming prompts like *"how do I kill a Python process?"*) for false positives:

| Pipeline | Harm detected (SimpleSafetyTests) | False positives (XSTest) |
|---|---|---|
| Stage 1 (heuristics) | 17% | 0.4% |
| Stage 1 + 2 (+ embeddings) | 21% | 1.2% |
| **Stage 1 + 2 + 3 (full cascade)** | **92%** | **1.2%** |

Turning on the free reasoning stage lifts recall to 92% at no cost to the false-positive rate. Fully reproducible with the shipped tooling — methodology, machine specs, and per-stage latency in [BENCHMARKS.md](https://github.com/Vishisht16/Humane-Proxy/blob/main/docs/BENCHMARKS.md).

---

## When something is flagged

- **Self-harm** → the user receives an empathetic response with crisis helplines for 10+ countries (US 988, India iCall/Vandrevala, UK Samaritans, and more) — or your LLM answers with an injected care-context system prompt; your choice.
- **Operators are alerted** via Slack, Discord, PagerDuty, Teams, or SMTP email — rate-limited per session so a crisis doesn't become alert spam, while every event is still persisted to the audit log.
- **Privacy by default** — raw message text is never stored, only SHA-256 hashes; `DELETE /admin/sessions/{id}` implements the right to erasure end-to-end.

---

## Available On

| Platform | Link | Status |
|---|---|---|
| **PyPI** | [humane-proxy](https://pypi.org/project/humane-proxy/) | ![PyPI](https://img.shields.io/pypi/v/humane-proxy.svg) |
| **Glama MCP Registry** | [Humane-Proxy](https://glama.ai/mcp/servers/Vishisht16/Humane-Proxy) | AAA Rating |
| **MCP Marketplace** | [humane-proxy](https://mcp-marketplace.io/server/io-github-vishisht16-humane-proxy) | Low Risk 10.0 |

---

## Installation Extras

| Extra | What it adds |
|---|---|
| *(none)* | Stage 1 heuristics + SQLite storage — zero dependencies beyond FastAPI |
| `onnx` | Stage 2 embeddings via ONNX Runtime — no PyTorch, ~2 GB lighter |
| `ml` | Stage 2 embeddings via sentence-transformers (PyTorch) |
| `mcp` | MCP server for AI agents |
| `redis` / `postgres` | Alternative storage backends |
| `llamaindex` / `crewai` / `autogen` / `langchain` | Native agent-framework tools |
| `telemetry` | OpenTelemetry distributed tracing |
| `perf` | orjson fast-path JSON serialization |
| `all` | Everything above (may cause conflicting dependencies)|

```bash
pip install humane-proxy[onnx,mcp]   # a solid production baseline
```

---

## Documentation

| Guide | Covers |
|---|---|
| [Pipeline](https://github.com/Vishisht16/Humane-Proxy/blob/main/docs/PIPELINE.md) | 3-stage cascade, score calibration, care response modes, risk trajectory & time-decay, multi-worker Redis |
| [Benchmarks](https://github.com/Vishisht16/Humane-Proxy/blob/main/docs/BENCHMARKS.md) | SimpleSafetyTests & XSTest results, methodology, latency, machine specs |
| [Configuration](https://github.com/Vishisht16/Humane-Proxy/blob/main/docs/CONFIGURATION.md) | Full YAML/env reference, webhooks, storage backends, privacy |
| [Integrations](https://github.com/Vishisht16/Humane-Proxy/blob/main/docs/INTEGRATIONS.md) | MCP server, LlamaIndex, CrewAI, AutoGen, LangChain, Node.js/TypeScript |
| [Deployment](https://github.com/Vishisht16/Humane-Proxy/blob/main/docs/DEPLOYMENT.md) | CLI reference, admin API, GitHub Action safety gate, OpenTelemetry |
| [Compliance](https://github.com/Vishisht16/Humane-Proxy/blob/main/docs/COMPLIANCE.md) | HIPAA, GDPR, and SOC 2 readiness assessment |
| [Security policy](https://github.com/Vishisht16/Humane-Proxy/blob/main/.github/SECURITY.md) | Supported versions, vulnerability disclosure |

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Copyright 2026 Vishisht Mishra ([@Vishisht16](https://github.com/Vishisht16)). Any attribution is appreciated.

See [NOTICE](NOTICE) for full attribution information.

---

Built for a safer world.
