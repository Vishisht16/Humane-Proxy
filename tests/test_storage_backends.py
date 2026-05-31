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


@patch("humane_proxy.storage.redis._redis", create=True)
@patch("humane_proxy.storage.redis._REDIS_AVAILABLE", True)
def test_redis_store_operations(mock_redis):
    """Test Redis store logic (logging, rate limiting, delete session) using mocks."""
    mock_client = MagicMock()
    mock_redis.Redis.from_url.return_value = mock_client
    
    # Setup incr to return 42
    mock_client.incr.return_value = 42
    
    # Setup mock pipeline
    mock_pipe = MagicMock()
    mock_client.pipeline.return_value = mock_pipe
    mock_pipe.execute.return_value = []
    
    config = {
        "storage": {
            "backend": "redis",
            "redis": {"url": "redis://localhost/0"}
        },
        "escalation": {
            "rate_limit_max": 3,
            "rate_limit_window_hours": 1
        }
    }
    
    from humane_proxy.storage.redis import RedisStore
    store = _create_store(config)
    assert isinstance(store, RedisStore)
    
    # 1. Test log
    store.log("session-x", "self_harm", 0.9, ["trigger1"])
    mock_client.incr.assert_called_with("humane_proxy:esc_id_seq")
    mock_pipe.hset.assert_called_once()
    assert mock_pipe.zadd.call_count == 3
    zadd_keys = [args[0] for args, _ in mock_pipe.zadd.call_args_list]
    assert "humane_proxy:esc_timeline" in zadd_keys
    assert "humane_proxy:session:session-x" in zadd_keys
    assert "humane_proxy:category:self_harm" in zadd_keys
    mock_pipe.execute.assert_called_once()
    
    # 2. Test rate limiting: zcount
    mock_client.zcount.return_value = 2
    assert store.check_rate_limit("session-x") is True
    
    mock_client.zcount.return_value = 3
    assert store.check_rate_limit("session-x") is False
    
    # 3. Test delete_session (fetches categories then executes pipeline)
    mock_client.zrange.return_value = ["42"]
    mock_read_pipe = MagicMock()
    mock_client.pipeline.side_effect = [mock_read_pipe, mock_pipe]
    mock_read_pipe.execute.return_value = ["self_harm"]
    mock_pipe.delete.reset_mock()
    mock_pipe.zrem.reset_mock()
    
    deleted_count = store.delete_session("session-x")
    assert deleted_count == 1
    mock_pipe.delete.assert_any_call("humane_proxy:esc:42")
    mock_pipe.zrem.assert_any_call("humane_proxy:esc_timeline", "42")
    mock_pipe.zrem.assert_any_call("humane_proxy:category:self_harm", "42")
    mock_pipe.delete.assert_any_call("humane_proxy:session:session-x")


@patch("humane_proxy.storage.postgres.psycopg", create=True)
@patch("humane_proxy.storage.postgres._PG_AVAILABLE", True)
def test_postgres_store_operations(mock_psycopg):
    """Test Postgres store query, log, count, delete, stats, rate limit using psycopg mocks."""
    mock_conn = MagicMock()
    mock_psycopg.connect.return_value = mock_conn
    mock_conn.__enter__.return_value = mock_conn
    
    mock_cur = MagicMock()
    mock_conn.execute.return_value = mock_cur
    
    config = {
        "storage": {
            "backend": "postgres",
            "postgres": {"dsn": "postgres://user@localhost/db"}
        },
        "escalation": {
            "rate_limit_max": 2,
            "rate_limit_window_hours": 1
        }
    }
    
    from humane_proxy.storage.postgres import PostgresStore
    store = _create_store(config)
    assert isinstance(store, PostgresStore)
    
    # 1. Test log
    store.log("sess-1", "criminal_intent", 0.8, ["trig"])
    mock_conn.execute.assert_called_once()
    mock_conn.commit.assert_called_once()
    
    # Reset mocks
    mock_conn.execute.reset_mock()
    
    # 2. Test rate limit
    mock_cur.fetchone.return_value = {"cnt": 1}
    assert store.check_rate_limit("sess-1") is True
    
    mock_cur.fetchone.return_value = {"cnt": 2}
    assert store.check_rate_limit("sess-1") is False
    
    # 3. Test delete_session
    mock_cur.rowcount = 5
    deleted = store.delete_session("sess-1")
    assert deleted == 5
