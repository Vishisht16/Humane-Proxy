"""Redis-backed trajectory window for distributed deployments."""

from __future__ import annotations

import time
from typing import Any

try:
    import redis as _redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    _redis = None  # type: ignore[assignment]


class RedisTrajectoryStore:
    def __init__(
        self,
        config: dict,
        window_size: int = 5,
        client: Any | None = None,
    ) -> None:
        if not _REDIS_AVAILABLE and client is None:
            raise RuntimeError(
                "Redis trajectory storage requires the 'redis' package. "
                "Install with: pip install humane-proxy[redis]"
            )

        redis_cfg = config.get("storage", {}).get("redis", {})
        url = redis_cfg.get("url", "redis://localhost:6379/0")
        self._prefix = redis_cfg.get("key_prefix", "humane_proxy:")
        self._window_size = window_size
        self._client = client or _redis.Redis.from_url(url, decode_responses=True)
        self._append_script = self._client.register_script(
            """
            local seq = redis.call('INCR', KEYS[1])
            local payload = ARGV[1] .. '|' .. ARGV[2] .. '|' .. ARGV[3]
            redis.call('ZADD', KEYS[2], ARGV[3], seq)
            redis.call('HSET', KEYS[3], seq, payload)

            local count = redis.call('ZCARD', KEYS[2])
            local excess = count - tonumber(ARGV[4])
            if excess > 0 then
                local removed = redis.call('ZRANGE', KEYS[2], 0, excess - 1)
                if #removed > 0 then
                    redis.call('ZREMRANGEBYRANK', KEYS[2], 0, excess - 1)
                    redis.call('HDEL', KEYS[3], unpack(removed))
                end
            end

            local ids = redis.call('ZRANGE', KEYS[2], 0, -1)
            local response = {}
            for _, id in ipairs(ids) do
                local payload_value = redis.call('HGET', KEYS[3], id)
                table.insert(response, id)
                table.insert(response, payload_value)
            end
            return response
            """
        )

    def _key(self, *parts: str) -> str:
        return self._prefix + ":".join(parts)

    def append(
        self,
        session_id: str,
        score: float,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        raw_window = self._append_script(
            keys=[
                self._key("traj", session_id, "seq"),
                self._key("traj", session_id, "window"),
                self._key("traj", session_id, "payload"),
            ],
            args=[
                str(score),
                category or "",
                str(time.time()),
                str(self._window_size),
            ],
        )
        return self._decode_window(raw_window)

    def snapshot(self, session_id: str) -> list[dict[str, Any]]:
        """Read the current window without mutating it."""
        ids = self._client.zrange(self._key("traj", session_id, "window"), 0, -1)
        if not ids:
            return []

        window: list[dict[str, Any]] = []
        for entry_id in ids:
            payload = self._client.hget(self._key("traj", session_id, "payload"), entry_id)
            if payload is None:
                continue
            window.append(self._parse_payload(entry_id, payload))
        return window

    @staticmethod
    def _parse_payload(entry_id: str, payload: str) -> dict[str, Any]:
        score_str, category, timestamp_str = payload.split("|", 2)
        return {
            "id": entry_id,
            "score": float(score_str),
            "category": category or None,
            "timestamp": float(timestamp_str),
        }

    @classmethod
    def _decode_window(cls, raw_window: list[Any]) -> list[dict[str, Any]]:
        window: list[dict[str, Any]] = []
        for index in range(0, len(raw_window), 2):
            entry_id = str(raw_window[index])
            payload = str(raw_window[index + 1])
            window.append(cls._parse_payload(entry_id, payload))
        return window