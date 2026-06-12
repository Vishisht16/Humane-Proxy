"""Tests for REST Admin API (GET/DELETE endpoints, auth, stats)."""

import json
import time

import pytest # pyright: ignore[reportMissingImports]
from fastapi.testclient import TestClient
from fastapi import FastAPI

from humane_proxy.api.admin import router

# Build a minimal test app
_test_app = FastAPI()
_test_app.include_router(router)
client = TestClient(_test_app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _set_admin_key(monkeypatch):
    monkeypatch.setenv("HUMANE_PROXY_ADMIN_KEY", "test-admin-secret")


@pytest.fixture()
def _seeded_db(tmp_path, monkeypatch):
    """Create a temp SQLiteStore with 3 escalation rows and inject via get_store."""
    db_path = tmp_path / "test_admin.db"

    # Build config pointing at temp DB path.
    config = {
        "storage": {"backend": "sqlite", "sqlite": {"path": str(db_path)}},
        "escalation": {"rate_limit_max": 3, "rate_limit_window_hours": 1},
    }

    from humane_proxy.storage.sqlite import SQLiteStore
    store = SQLiteStore(config)
    store.init()

    # Seed 3 rows directly via the store's log() method.
    store.log("sess-1", "self_harm", 1.0,
              ["keyword:kill myself"], message_hash=None, stage_reached=1)
    store.log("sess-2", "criminal_intent", 0.75,
              ["keyword:how to make a bomb"], message_hash=None, stage_reached=1)
    store.log("sess-1", "self_harm", 1.0,
              ["pattern:self_annihilation"], message_hash=None, stage_reached=2,
              reasoning="LLM reasoning")

    # Inject the store so admin.py uses it instead of the real singleton.
    monkeypatch.setattr("humane_proxy.api.admin.get_store", lambda: store)
    return store


class TestAdminAuth:
    def test_no_key_returns_403(self, monkeypatch):
        monkeypatch.delenv("HUMANE_PROXY_ADMIN_KEY", raising=False)
        resp = client.get("/admin/escalations")
        assert resp.status_code == 403

    def test_wrong_key_returns_401(self):
        resp = client.get("/admin/escalations", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_correct_key_passes(self, _seeded_db):
        resp = client.get("/admin/escalations", headers={"Authorization": "Bearer test-admin-secret"})
        assert resp.status_code == 200


class TestListEscalations:
    HEADERS = {"Authorization": "Bearer test-admin-secret"}

    def test_returns_all(self, _seeded_db):
        resp = client.get("/admin/escalations", headers=self.HEADERS)
        data = resp.json()
        assert data["total"] == 3

    def test_filter_by_category(self, _seeded_db):
        resp = client.get("/admin/escalations?category=self_harm", headers=self.HEADERS)
        data = resp.json()
        assert data["total"] == 2
        assert all(r["category"] == "self_harm" for r in data["items"])

    def test_filter_by_session(self, _seeded_db):
        resp = client.get("/admin/escalations?session_id=sess-2", headers=self.HEADERS)
        data = resp.json()
        assert data["total"] == 1

    def test_limit_respected(self, _seeded_db):
        resp = client.get("/admin/escalations?limit=1", headers=self.HEADERS)
        data = resp.json()
        assert len(data["items"]) == 1


class TestGetEscalation:
    HEADERS = {"Authorization": "Bearer test-admin-secret"}

    def test_get_existing(self, _seeded_db):
        resp = client.get("/admin/escalations/1", headers=self.HEADERS)
        assert resp.status_code == 200
        assert resp.json()["id"] == 1

    def test_get_missing_404(self, _seeded_db):
        resp = client.get("/admin/escalations/999", headers=self.HEADERS)
        assert resp.status_code == 404


class TestStats:
    HEADERS = {"Authorization": "Bearer test-admin-secret"}

    def test_stats_returns_total(self, _seeded_db):
        resp = client.get("/admin/stats", headers=self.HEADERS)
        data = resp.json()
        assert data["total_escalations"] == 3

    def test_stats_by_category(self, _seeded_db):
        resp = client.get("/admin/stats", headers=self.HEADERS)
        by_cat = resp.json()["by_category"]
        assert by_cat["self_harm"] == 2
        assert by_cat["criminal_intent"] == 1

    def test_stats_has_advanced_fields(self, _seeded_db):
        resp = client.get("/admin/stats", headers=self.HEADERS)
        data = resp.json()
        assert "by_day" in data
        assert "top_sessions" in data
        assert "by_stage" in data
        assert "hourly_last_24h" in data
        assert "limited_stats" in data
        assert data["limited_stats"] is False


class TestSessionRisk:
    HEADERS = {"Authorization": "Bearer test-admin-secret"}

    def test_session_risk_is_read_only(self, _seeded_db):
        from humane_proxy.risk.trajectory import analyze, session_history, _category_history

        sid = "sess-1"
        analyze(sid, 0.1, "safe")
        analyze(sid, 0.1, "safe")
        analyze(sid, 0.1, "safe")
        analyze(sid, 0.9, "self_harm")
        before_score_count = len(session_history[sid])
        before_category_count = len(_category_history[sid])

        resp = client.get(f"/admin/sessions/{sid}/risk", headers=self.HEADERS)

        assert resp.status_code == 200
        data = resp.json()
        assert data["trajectory"]["message_count"] == before_score_count
        assert data["trajectory"]["spike_detected"] is True
        assert len(session_history[sid]) == before_score_count
        assert len(_category_history[sid]) == before_category_count


class TestDeleteSession:
    HEADERS = {"Authorization": "Bearer test-admin-secret"}

    def test_delete_removes_records(self, _seeded_db):
        resp = client.delete("/admin/sessions/sess-1", headers=self.HEADERS)
        assert resp.status_code == 204

        # Verify deletion via list endpoint.
        list_resp = client.get("/admin/escalations?session_id=sess-1", headers=self.HEADERS)
        assert list_resp.json()["total"] == 0