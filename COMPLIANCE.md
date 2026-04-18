# Compliance & Data Privacy

This document describes HumaneProxy's architecture in relation to major compliance frameworks. It is intended to help engineering teams evaluate HumaneProxy for use in regulated environments.

> **Disclaimer:** This document is informational and does not constitute legal advice. Consult your legal and compliance teams before deploying HumaneProxy in environments subject to HIPAA, GDPR, or other regulatory requirements.

---

## Architecture Overview

HumaneProxy is a **self-hosted, stateless safety middleware**. It runs entirely within your infrastructure:

```
User Message → [Your Application] → [HumaneProxy Pipeline] → [Upstream LLM]
                                          │
                                          ├─ Stage 1: Local heuristics (no network)
                                          ├─ Stage 2: Local embedding model (no network)
                                          └─ Stage 3: Optional external LLM (OpenAI/Groq)
```

- **Stages 1 and 2** process all data locally. No message content leaves your infrastructure.
- **Stage 3** is optional and operator-configured. If enabled, message content is sent to the configured LLM provider (OpenAI or Groq) for reasoning. This is the **only** point where data may leave your infrastructure.

---

## HIPAA (Health Insurance Portability and Accountability Act)

### What HumaneProxy Does

| Requirement | Status | Notes |
|---|---|---|
| **Data at rest** | ✅ No PHI stored by default | Messages are SHA-256 hashed before persistence. Raw text is never stored unless the operator explicitly enables `privacy.store_message_text: true`. |
| **Data in transit** | ✅ Operator-controlled | HumaneProxy runs locally. TLS configuration for upstream connections is the operator's responsibility. |
| **Access controls** | ✅ Admin API secured | The `/admin` API requires a Bearer token (`HUMANE_PROXY_ADMIN_KEY`). |
| **Audit logging** | ✅ Built-in | All escalation events are logged with session ID, category, score, stage, and timestamp. |
| **BAA requirement** | ⚠️ Conditional | No BAA is needed for Stages 1–2 (fully local). If Stage 3 is enabled, a BAA with the LLM provider (e.g., OpenAI) may be required if processing PHI. |

### What the Operator Must Do

- **Do not enable `privacy.store_message_text: true`** if the escalation database is not encrypted and access-controlled.
- **If using Stage 3:** Ensure your LLM provider offers a BAA (OpenAI offers BAAs on enterprise plans). Alternatively, use a self-hosted LLM to keep all processing local.
- **Encrypt the SQLite database at rest** if it resides on shared storage.
- **Restrict access** to the host machine and the `/admin` API.

---

## GDPR (General Data Protection Regulation)

### What HumaneProxy Does

| Requirement | Status | Notes |
|---|---|---|
| **Data minimisation** | ✅ By design | Only SHA-256 hashes, scores, and categories are stored. No PII is persisted by default. |
| **Right to erasure** | ✅ Built-in | `DELETE /admin/sessions/{id}` removes all records for a session. Also available via CLI. |
| **Purpose limitation** | ✅ Clear scope | Data is processed solely for safety classification. No profiling, analytics, or marketing use. |
| **Data portability** | ✅ CSV export | `GET /admin/escalations/export` provides full CSV export of escalation data. |
| **Consent** | ⚠️ Operator-managed | HumaneProxy does not collect consent. The operator's application must obtain appropriate consent for message processing. |
| **Data residency** | ✅ Self-hosted | All data stays within the operator's infrastructure (Stages 1–2). Stage 3 routing is operator-configured. |
| **Cross-border transfers** | ⚠️ Stage 3 only | If Stage 3 routes to a US-based LLM provider, standard contractual clauses or adequacy decisions may apply. |

### What the Operator Must Do

- **Inform users** that messages are being screened for safety purposes (transparency principle).
- **Configure data retention** policies — HumaneProxy does not auto-delete old records. Implement a cron job or use the admin API for periodic cleanup.
- **If using Stage 3 with a non-EU provider:** Ensure appropriate safeguards (SCCs, adequacy decisions) are in place for cross-border data transfers.

---

## SOC 2

HumaneProxy's architecture supports SOC 2 controls:

| Control Area | Support |
|---|---|
| **Security** | Admin API authentication, no default data persistence, SHA-256 hashing |
| **Availability** | Lightweight runtime, no external dependencies for Stages 1–2 |
| **Confidentiality** | No raw message storage by default, local processing |
| **Processing Integrity** | Deterministic pipeline with audit trail |

---

## Summary

| Aspect | HumaneProxy Default | With Stage 3 Enabled |
|---|---|---|
| **Data leaves infrastructure?** | ❌ No | ✅ Yes (to LLM provider) |
| **PII stored?** | ❌ No (SHA-256 only) | ❌ No (SHA-256 only) |
| **BAA required?** | ❌ No | ⚠️ Possibly (depends on data type) |
| **GDPR-compatible?** | ✅ Yes | ✅ Yes (with operator safeguards) |
| **Right to erasure?** | ✅ Built-in API | ✅ Built-in API |
| **Audit trail?** | ✅ SQLite + webhooks | ✅ SQLite + webhooks |

---

## Questions?

If you need help evaluating HumaneProxy for your compliance requirements, open an issue or reach out to [@Vishisht16](https://github.com/Vishisht16).
