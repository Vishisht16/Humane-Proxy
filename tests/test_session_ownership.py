"""Tests for session ownership binding (Issue #24)."""

import pytest

from humane_proxy.errors import SessionOwnershipError
from humane_proxy.storage.factory import _create_store


def test_sqlite_session_owner_binding(tmp_path):
    db_path = tmp_path / "owner.db"
    config = {"storage": {"backend": "sqlite", "sqlite": {"path": str(db_path)}}}
    store = _create_store(config)
    store.init()

    store.assert_session_owner("sid-1", "owner-a")
    store.assert_session_owner("sid-1", "owner-a")  # idempotent

    with pytest.raises(SessionOwnershipError):
        store.assert_session_owner("sid-1", "owner-b")


def test_delete_session_clears_owner(tmp_path):
    db_path = tmp_path / "owner-delete.db"
    config = {"storage": {"backend": "sqlite", "sqlite": {"path": str(db_path)}}}
    store = _create_store(config)
    store.init()

    store.assert_session_owner("sid-2", "owner-a")
    store.log("sid-2", "safe", 0.1, [])
    assert store.delete_session("sid-2") == 1

    # After delete, the session can be re-bound to a new owner.
    store.assert_session_owner("sid-2", "owner-b")

