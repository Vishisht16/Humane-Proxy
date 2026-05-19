# Security Policy

## Reporting a vulnerability

Please open a private report via GitHub Security Advisories (preferred), or file an issue with minimal details if private reporting is not available.

## Notes for operators

### `session_id` ownership binding

HumaneProxy supports caller-provided `session_id` values for trajectory tracking and escalation auditing. In multi-tenant deployments, a user who can guess another user’s `session_id` could otherwise poison their risk trajectory or generate false escalations.

To mitigate this, HumaneProxy binds each `session_id` to a per-caller **owner token** on first use. Subsequent writes to the same `session_id` must match the original owner token.

For the built-in HTTP proxy (`POST /chat`), the owner token is derived from the client IP address and **hardened** with `HUMANE_PROXY_SESSION_SECRET` when set.

#### Recommended configuration

- Set `HUMANE_PROXY_SESSION_SECRET` to a long random value (and keep it stable across deploys).
- Avoid predictable `session_id` values (usernames, emails, sequential IDs). Prefer random IDs.

