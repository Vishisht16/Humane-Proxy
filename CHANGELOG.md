# Changelog

All notable changes to HumaneProxy will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.1] — 2026-03-31

### Added

- **MCP HTTP transport mode:** `humane-proxy mcp-serve --transport http` exposes tools over Streamable HTTP for remote clients and registry listing.
- **Official MCP Registry integration:** `server.json` metadata + `publish-mcp.yml` GitHub Actions workflow for automated publishing via OIDC.
- **LangChain integration:** `humane_proxy.integrations.langchain` module with `get_safety_tools()` and `get_langchain_mcp_config()` helpers. New `[langchain]` install extra.
- **`<!-- mcp-name: humane-proxy -->`** marker in README for MCP Registry discovery.

### Removed

- `smithery.yaml` — Smithery no longer supports GitHub-based stdio imports; HTTP transport is used instead.

### Changed

- `humane-proxy mcp-serve` now accepts `--transport` (`stdio` | `http`), `--host`, and `--port` options.
- `[all]` install extra now includes `[langchain]`.

---

## [0.2.0] — 2026-03-31

### Added

- **3-Stage Cascade Safety Pipeline** — fully configurable per stage:
  - **Stage 1 (Heuristics):** Always-on, sub-millisecond keyword + regex classifier.
  - **Stage 2 (Embeddings):** Semantic similarity using `sentence-transformers` (`pip install humane-proxy[ml]`).
  - **Stage 3 (Reasoning LLM):** Optional; auto-detected from API keys. Supports LlamaGuard (Groq), OpenAI Moderation API, or any OpenAI-compatible chat model.
- **International Self-Harm Care Response System:**
  - **Block mode (default):** Replies with an empathetic message and crisis resources for 10+ countries (US, India, UK, Australia, Canada, Germany, France, Brazil, South Africa + IASP/Befrienders international).
  - **Forward mode:** Injects a care-context system prompt before forwarding to the upstream LLM.
  - Both modes are configurable via `humane_proxy.yaml`.
- **MCP Server Integration:** Expose safety tools via the Model Context Protocol (`pip install humane-proxy[mcp]`). Tools: `check_message_safety`, `get_session_risk`, `list_recent_escalations`. Smithery-compatible.
- **REST Admin API:** Mounted at `/admin`, secured with `HUMANE_PROXY_ADMIN_KEY` Bearer token. Endpoints: list escalations (paginated + filterable), single record, session risk history, aggregate stats, session data deletion (privacy right to erasure).
- **Enhanced CLI commands:**
  - `humane-proxy escalations [--category] [--limit] [--session]` — audit log viewer.
  - `humane-proxy session <id>` — per-session risk history.
  - `humane-proxy mcp-serve` — starts the MCP stdio server.
- **Microsoft Teams webhook** (adaptive card format) alongside existing Slack, Discord, PagerDuty.
- **Email alerts** via SMTP (stdlib `smtplib`, zero extra deps).
- **Privacy controls:** SHA-256 message hashing, `stage_reached` and `reasoning` stored per escalation.
- **Enhanced Risk Trajectory:** Trend detection (escalating / stable / declining), category distribution per session, spike detection.
- **BYOK Stage-3:** Auto-detects `OPENAI_API_KEY` → OpenAI Moderation, `GROQ_API_KEY` → LlamaGuard; prints clear setup guidance if neither is found.
- **PyPI publish workflow** (`.github/workflows/pypi.yml`) via Trusted Publishers (OIDC, no token needed).

### Changed

- Risk threshold default lowered 0.8 → 0.7 (better recall for human safety cases).
- Self-harm keyword score raised 0.5 → 0.7 (these are high-intent by nature).
- Classification pipeline now returns structured `ClassificationResult` with `category`, `score`, `triggers`, `stage`, and `reasoning`.
- `PipelineResult.to_dict()` includes `stage_reached` for explainability.
- Interceptor `/chat` endpoint now uses the full async 3-stage pipeline.
- Escalation DB schema extended: `message_hash`, `stage_reached`, `reasoning` columns with auto-migration.

---

## [0.1.0] — 2026-03-31

### Added

- Heuristic keyword + regex classifier specifically for **self-harm** and **criminal intent** (not jailbreaks).
- Context-aware false positive reduction — e.g. `"I want to die laughing"` → `safe`.
- First-person intent patterns and method-seeking patterns.
- Per-session risk trajectory with spike detection.
- SQLite-backed escalation logging with configurable per-session rate limiting.
- Webhook alerts: Slack (Block Kit), Discord (embeds), PagerDuty (Events API v2).
- FastAPI reverse proxy middleware (`humane-proxy start`) that intercepts, classifies, and optionally forwards.
- CLI: `init`, `start`, `check`, `version`.
- Programmatic `HumaneProxy` class usable as a Python library.
- Layered configuration: package defaults → `humane_proxy.yaml` → `HUMANE_PROXY_*` env vars.
- Apache 2.0 licence. Attribution to Vishisht Mishra (@Vishisht16).
- 94 unit & integration tests.
