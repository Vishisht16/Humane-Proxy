# Changelog

All notable changes to HumaneProxy will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.3] — 2026-04-01

### Fixed

- **Stage 2 never ran:** The pipeline's early-exit logic was too aggressive — when Stage 1 scored a message as safe (score 0.0), it would early-exit before the embedding classifier had a chance to evaluate it. This defeated Stage 2's entire purpose: catching *semantically* dangerous messages that keyword matching misses. Now, when Stage 2 is enabled, all messages that Stage 1 does not flag as `self_harm` proceed to the embedding classifier.
- **Stage 3 warning shown incorrectly:** Users with `enabled_stages: [1]` or `[1, 2]` saw a Stage 3 "DISABLED" warning even though they never configured Stage 3. The warning now only appears when `3` is in `enabled_stages` but no API key / provider is available.
- **`HUMANE_PROXY_ENABLED_STAGES` env var not wired up:** Documented in README but not implemented in the config loader. Now accepts comma-separated ints (e.g. `"1,2"`).
- **FastAPI app version** pinned to `0.2.0` — updated to `0.2.3`.

### Added

- **`glama.json`** metadata for Glama MCP directory listing.
- **Real embedding model tests:** New `TestEmbeddingClassifierReal` test class that exercises the full `all-MiniLM-L6-v2` classify flow (guarded by `pytest.importorskip`, auto-skipped in CI).
- **Pipeline early-exit regression tests:** `TestStage2EarlyExitFix` class verifying that Stage 2 is always invoked when enabled, even for messages heuristics considers safe.
- **Stage 3 warning tests:** Tests that verify the warning is only shown when Stage 3 is in `enabled_stages`.
- **Glama badges** in README for MCP server card and quality score.

### Changed

- README updated to clarify Stage 2 behaviour: when enabled, all messages flow through the embedding classifier. Stage 1 heuristics becomes an early-exit optimisation for clear self-harm only, not a safety determiner.
- README updated to clarify `LLM_API_KEY` / `LLM_API_URL` are only needed for the reverse proxy server, not for the library API or MCP server.

---

## [0.2.2] — 2026-03-31

### Added

- **MCP HTTP transport mode:** `humane-proxy mcp-serve --transport http` exposes tools over Streamable HTTP for remote clients and registry listing.
- **Official MCP Registry integration:** `server.json` metadata + `publish-mcp.yml` GitHub Actions workflow for automated publishing via OIDC.
- **LangChain integration:** `humane_proxy.integrations.langchain` module with `get_safety_tools()` and `get_langchain_mcp_config()` helpers. New `[langchain]` install extra.
- **`<!-- mcp-name -->`** marker in README for MCP Registry discovery.

### Removed

- `smithery.yaml` — Smithery no longer supports GitHub-based stdio imports; HTTP transport is used instead.

### Changed

- `humane-proxy mcp-serve` now accepts `--transport` (`stdio` | `http`), `--host`, and `--port` options.
- `[all]` install extra now includes `[langchain]`.

### Fixed

- `server.json` now uses the correct MCP Registry schema (`static.modelcontextprotocol.io`) and `packages` format.
- `publish-mcp.yml` uses proper `login github-oidc` two-step auth.
- `mcp-name` marker uses full reverse-DNS namespace for PyPI ownership validation.

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
