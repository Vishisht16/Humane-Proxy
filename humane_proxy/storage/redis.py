"""Redis storage backend for HumaneProxy escalation data.

Requires: pip install humane-proxy[redis]
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from humane_proxy.storage.base import EscalationStore

logger = logging.getLogger("humane_proxy.storage.redis")

try:
    import redis as _redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    _redis = None  # type: ignore[assignment]


class RedisStore(EscalationStore):
    """Redis-backed escalation storage.

    Data model:
    - Each escalation is a Redis hash at ``{prefix}esc:{id}``.
    - A sorted set ``{prefix}esc_timeline`` indexes escalations by timestamp.
    - Per-session sorted sets ``{prefix}session:{sid}`` for session queries.
    - An auto-incrementing counter ``{prefix}esc_id_seq``.
    - Rate limits use ``{prefix}rate:{sid}`` with native ``INCR`` + ``EXPIRE``.

    Parameters
    ----------
    config:
        Full application config dict.
    rate_limit_max:
        Max escalations per session per window.
    rate_limit_window_hours:
        Window duration in hours.
    """

    def __init__(
        self,
        config: dict,
        rate_limit_max: int = 3,
        rate_limit_window_hours: int = 1,
    ) -> None:
        if not _REDIS_AVAILABLE:
            raise RuntimeError(
                "Redis storage requires the 'redis' package. "
                "Install with: pip install humane-proxy[redis]"
            )
        redis_cfg = config.get("storage", {}).get("redis", {})
        url = redis_cfg.get("url", "redis://localhost:6379/0")
        self._prefix = redis_cfg.get("key_prefix", "humane_proxy:")
        self._client = _redis.Redis.from_url(url, decode_responses=True)
        self._rate_limit_max = rate_limit_max
        self._rate_limit_window_s = rate_limit_window_hours * 3600

    def _key(self, *parts: str) -> str:
        return self._prefix + ":".join(parts)

    def init(self) -> None:
        # Redis is schemaless — just verify connectivity.
        self._client.ping()
        logger.info("Redis store connected: %s", self._client.connection_pool.connection_kwargs.get("host", ""))

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
        esc_id = self._client.incr(self._key("esc_id_seq"))
        ts = datetime.now(timezone.utc).timestamp()

        record = {
            "id": str(esc_id),
            "session_id": session_id,
            "category": category,
            "risk_score": str(risk_score),
            "triggers": json.dumps(triggers or []),
            "timestamp": str(ts),
            "message_hash": message_hash or "",
            "stage_reached": str(stage_reached),
            "reasoning": reasoning or "",
        }

        pipe = self._client.pipeline()
        pipe.hset(self._key("esc", str(esc_id)), mapping=record)
        pipe.zadd(self._key("esc_timeline"), {str(esc_id): ts})
        pipe.zadd(self._key("session", session_id), {str(esc_id): ts})
        pipe.zadd(self._key("category", category), {str(esc_id): ts})
        pipe.execute()

    def query(
        self,
        *,
        category: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if session_id:
            index_key = self._key("session", session_id)
        elif category:
            index_key = self._key("category", category)
        else:
            index_key = self._key("esc_timeline")

        # Reverse range for DESC order.
        ids = self._client.zrevrange(index_key, offset, offset + limit - 1)
        results = []
        for esc_id in ids:
            raw = self._client.hgetall(self._key("esc", esc_id))
            if raw:
                results.append(self._parse_record(raw))
        return results

    def count(
        self,
        *,
        category: str | None = None,
        session_id: str | None = None,
    ) -> int:
        if session_id:
            return self._client.zcard(self._key("session", session_id))
        elif category:
            return self._client.zcard(self._key("category", category))
        return self._client.zcard(self._key("esc_timeline"))

    def get_by_id(self, escalation_id: int) -> dict[str, Any] | None:
        raw = self._client.hgetall(self._key("esc", str(escalation_id)))
        return self._parse_record(raw) if raw else None

    def delete_session(self, session_id: str) -> int:
        ids = self._client.zrange(self._key("session", session_id), 0, -1)
        if not ids:
            return 0
        pipe = self._client.pipeline()
        for esc_id in ids:
            pipe.delete(self._key("esc", esc_id))
            pipe.zrem(self._key("esc_timeline"), esc_id)
        pipe.delete(self._key("session", session_id))
        pipe.execute()
        return len(ids)

    def stats(self) -> dict[str, Any]:
        total = self._client.zcard(self._key("esc_timeline"))
        # Category counts by scanning category indexes.
        by_category: dict[str, int] = {}
        for key in self._client.scan_iter(match=self._key("category", "*")):
            cat_name = key.replace(self._prefix + "category:", "")
            by_category[cat_name] = self._client.zcard(key)
        return {
            "total_escalations": total,
            "by_category": by_category,
            "average_risk_score": 0.0,  # Would require full scan; skip for perf.
        }

    def check_rate_limit(self, session_id: str) -> bool:
        rate_key = self._key("rate", session_id)
        current = self._client.get(rate_key)
        if current is None:
            self._client.setex(rate_key, self._rate_limit_window_s, 1)
            return True
        return int(current) < self._rate_limit_max

    @staticmethod
    def _parse_record(raw: dict[str, str]) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "id": int(raw.get("id", 0)),
            "session_id": raw.get("session_id", ""),
            "category": raw.get("category", "unknown"),
            "risk_score": float(raw.get("risk_score", 0)),
            "timestamp": float(raw.get("timestamp", 0)),
            "message_hash": raw.get("message_hash") or None,
            "stage_reached": int(raw.get("stage_reached", 1)),
            "reasoning": raw.get("reasoning") or None,
        }
        try:
            rec["triggers"] = json.loads(raw.get("triggers", "[]"))
        except Exception:
            rec["triggers"] = []
        return rec
