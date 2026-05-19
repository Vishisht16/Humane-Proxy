"""Project-wide exception types."""

from __future__ import annotations


class HumaneProxyError(Exception):
    """Base exception for HumaneProxy."""


class SessionOwnershipError(HumaneProxyError):
    """Raised when a session_id is used by a different caller/owner."""

