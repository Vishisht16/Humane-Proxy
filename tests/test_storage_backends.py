"""Tests for swappable storage backend implementations."""

import pytest
import sqlite3
from unittest.mock import MagicMock, patch

from humane_proxy.storage.factory import _create_store
from humane_proxy.storage.sqlite import SQLiteStore


def test_sqlite_store(tmp_path):
    """Test the real SQLite backend."""
    db_path = tmp_path / "test.db"
    config = {
        "storage": {
            "backend": "sqlite",
            "sqlite": {"path": str(db_path)}
        }
    }
    
    store = _create_store(config)
    assert isinstance(store, SQLiteStore)
    
    store.init()
    
    # Test rate limiting empty
    assert store.check_rate_limit("user-1") is True
    
    # Write a log
    store.log("user-1", "self_harm", 0.95, ["keyword"])
    
    # Query
    results = store.query(session_id="user-1")
    assert len(results) == 1
    assert results[0]["session_id"] == "user-1"
    assert results[0]["category"] == "self_harm"
    assert results[0]["risk_score"] == 0.95
    
    # Count stats
    assert store.count() == 1
    stats = store.stats()
    assert stats["total_escalations"] == 1
    assert stats["by_category"]["self_harm"] == 1
    
    # Delete
    deleted = store.delete_session("user-1")
    assert deleted == 1
    assert store.count() == 0


@patch("humane_proxy.storage.redis._redis", create=True)
@patch("humane_proxy.storage.redis._REDIS_AVAILABLE", True)
def test_redis_store_creation(mock_redis):
    """Verify Redis store is created correctly when configured."""
    config = {
        "storage": {
            "backend": "redis",
            "redis": {"url": "redis://localhost/0"}
        }
    }
    
    from humane_proxy.storage.redis import RedisStore
    store = _create_store(config)
    assert isinstance(store, RedisStore)
    assert mock_redis.Redis.from_url.called


@patch("humane_proxy.storage.postgres.psycopg", create=True)
@patch("humane_proxy.storage.postgres._PG_AVAILABLE", True)
def test_postgres_store_creation(mock_psycopg):
    """Verify Postgres store is created correctly when configured."""
    config = {
        "storage": {
            "backend": "postgres",
            "postgres": {"dsn": "postgres://user@localhost/db"}
        }
    }
    
    from humane_proxy.storage.postgres import PostgresStore
    store = _create_store(config)
    assert isinstance(store, PostgresStore)
