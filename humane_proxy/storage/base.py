"""Abstract base class for HumaneProxy escalation storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EscalationStore(ABC):
    """Interface that all storage backends must implement.

    Provides CRUD operations for escalation records and
    per-session rate limiting.
    """

    @abstractmethod
    def init(self) -> None:
        """Initialise the storage backend (create tables, ensure indexes, etc.)."""
        ...

    @abstractmethod
    def log(
        self,
        session_id: str,
        category: str,
        risk_score: float,
        triggers: list[str],
        message_hash: str | None = None,
        stage_reached: int = 1,
        reasoning: str | None = None,
    ) -> None:
        """Persist a single escalation event."""
        ...

    @abstractmethod
    def query(
        self,
        *,
        category: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return escalation records matching the filters."""
        ...

    @abstractmethod
    def count(
        self,
        *,
        category: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Return the number of matching records."""
        ...

    @abstractmethod
    def get_by_id(self, escalation_id: int) -> dict[str, Any] | None:
        """Return a single record by its ID, or ``None``."""
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> int:
        """Delete all records for a session.  Return the deleted count."""
        ...

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics."""
        ...

    @abstractmethod
    def check_rate_limit(self, session_id: str) -> bool:
        """Return ``True`` if the session is within its allowed escalation quota."""
        ...
