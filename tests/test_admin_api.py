"""Tests for REST Admin API (GET/DELETE endpoints, auth, stats)."""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
    """Create a temp DB with 3 escalation rows."""
    db_path = tmp_path / "test_admin.db"
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute(
            """CREATE TABLE escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, category TEXT, risk_score REAL,
                triggers TEXT, timestamp REAL,
                message_hash TEXT, stage_reached INTEGER, reasoning TEXT
            )"""
        )
        rows = [
            ("sess-1", "self_harm", 1.0, '["keyword:kill myself"]', time.time(), None, 1, None),
            ("sess-2", "criminal_intent", 0.75, '["keyword:how to make a bomb"]', time.time(), None, 1, None),
            ("sess-1", "self_harm", 1.0, '["pattern:self_annihilation"]', time.time(), None, 2, "LLM reasoning"),
        ]
        conn.executemany(
            "INSERT INTO escalations (session_id, category, risk_score, triggers, timestamp, message_hash, stage_reached, reasoning) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
    monkeypatch.setattr("humane_proxy.api.admin._get_db_path", lambda: str(db_path))
    return db_path



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

        # Verify deletion
        list_resp = client.get("/admin/escalations?session_id=sess-1", headers=self.HEADERS)
        assert list_resp.json()["total"] == 0



# ---------------------------------------------------------------------------
# Enhanced Analytics and Testing Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _blank_db(tmp_path, monkeypatch):
    """Creates a fresh, empty temp DB isolated from original seeded data."""
    db_path = tmp_path / "test_admin_enhanced.db"
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute(
            """CREATE TABLE escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, category TEXT, risk_score REAL,
                triggers TEXT, timestamp REAL,
                message_hash TEXT, stage_reached INTEGER, reasoning TEXT
            )"""
        )
    monkeypatch.setattr("humane_proxy.api.admin._get_db_path", lambda: str(db_path))
    return str(db_path)


def _insert_raw(db_path: str, session_id: str, category: str, risk_score: float, triggers_raw: str | None, timestamp: float):
    """Bypasses JSON dumping to allow intentional malformed data injections."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            "INSERT INTO escalations (session_id, category, risk_score, triggers, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, category, risk_score, triggers_raw, timestamp)
        )
    conn.close()


def _insert_row(db_path: str, session_id: str, category: str, risk_score: float, triggers: list, timestamp: float):
    """Standard cleanly formatted row insertion for analytical tests."""
    raw = json.dumps(triggers) if triggers is not None else None
    _insert_raw(db_path, session_id, category, risk_score, raw, timestamp)



class TestStatsEnhanced:
    """Validates aggregate mathematical behavior in the /admin/stats endpoint."""
    HEADERS = {"Authorization": "Bearer test-admin-secret"}

    def test_category_percentages_math(self, _blank_db):
        now = time.time()
        # Create a 62.5% / 37.5% split (5 vs 3 out of 8 total)
        for _ in range(5):
            _insert_row(_blank_db, "s1", "self_harm", 0.9, [], now)
        for _ in range(3):
            _insert_row(_blank_db, "s2", "criminal_intent", 0.8, [], now)
            
        resp = client.get("/admin/stats", headers=self.HEADERS)
        assert resp.status_code == 200
        
        pcts = resp.json()["category_percentages"]
        assert pcts["self_harm"] == 62.5
        assert pcts["criminal_intent"] == 37.5
        assert sum(pcts.values()) == 100.0

    def test_category_percentages_empty_db_safe(self, _blank_db):
        """Ensures ZeroDivisionError is avoided on fresh deployments."""
        resp = client.get("/admin/stats", headers=self.HEADERS)
        assert resp.status_code == 200
        assert resp.json()["category_percentages"] == {}

    def test_period_comparison_strict_bounds(self, _blank_db):
        """Validates exact UTC week boundary calculations."""
        now_dt = datetime.now(timezone.utc)
        current_week_start_dt = now_dt.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=now_dt.weekday())
        
        cw_start = current_week_start_dt.timestamp()
        pw_start = (current_week_start_dt - timedelta(days=7)).timestamp()
        
        # 2 records firmly inside the current week
        _insert_row(_blank_db, "s1", "c1", 0.5, [], cw_start + 100)
        _insert_row(_blank_db, "s2", "c1", 0.5, [], cw_start + 200)
        
        # 3 records firmly inside the previous week
        _insert_row(_blank_db, "s3", "c2", 0.5, [], pw_start + 100)
        _insert_row(_blank_db, "s4", "c2", 0.5, [], pw_start + 200)
        _insert_row(_blank_db, "s5", "c2", 0.5, [], pw_start + 300)
        
        resp = client.get("/admin/stats", headers=self.HEADERS)
        assert resp.status_code == 200
        
        comp = resp.json()["period_comparison"]
        assert comp["current_week"] == 2
        assert comp["previous_week"] == 3



class TestAnalyticsTopTriggers:
    """Validates the GET /admin/analytics/top-triggers endpoint."""
    HEADERS = {"Authorization": "Bearer test-admin-secret"}

    def test_trigger_aggregation_and_ranking(self, _blank_db):
        now = time.time()
        _insert_row(_blank_db, "s1", "cat1", 0.5, ["bomb", "suicide"], now)
        _insert_row(_blank_db, "s2", "cat1", 0.5, ["bomb"], now)
        _insert_row(_blank_db, "s3", "cat2", 0.5, ["suicide", "bomb"], now)
        
        resp = client.get("/admin/analytics/top-triggers", headers=self.HEADERS)
        assert resp.status_code == 200
        
        triggers = resp.json()["top_triggers"]
        assert len(triggers) == 2
        # 'bomb' occurs 3 times, 'suicide' occurs 2 times. Descending rank expected.
        assert triggers[0] == {"trigger": "bomb", "count": 3}
        assert triggers[1] == {"trigger": "suicide", "count": 2}

    def test_trigger_normalization(self, _blank_db):
        """Ensures varied casings and spacings resolve to a single trigger."""
        now = time.time()
        _insert_row(_blank_db, "s1", "cat1", 0.5, ["BOMB ", " bomb", "Bomb"], now)
        
        resp = client.get("/admin/analytics/top-triggers", headers=self.HEADERS)
        triggers = resp.json()["top_triggers"]
        assert len(triggers) == 1
        assert triggers[0] == {"trigger": "bomb", "count": 3}

    def test_trigger_limit(self, _blank_db):
        now = time.time()
        _insert_row(_blank_db, "s1", "cat1", 0.5, ["a", "b", "c"], now)
        
        resp = client.get("/admin/analytics/top-triggers?limit=2", headers=self.HEADERS)
        triggers = resp.json()["top_triggers"]
        assert len(triggers) == 2

    def test_trigger_category_filter(self, _blank_db):
        now = time.time()
        _insert_row(_blank_db, "s1", "cat1", 0.5, ["bomb"], now)
        _insert_row(_blank_db, "s2", "cat2", 0.5, ["gun"], now)
        
        resp = client.get("/admin/analytics/top-triggers?category=cat2", headers=self.HEADERS)
        triggers = resp.json()["top_triggers"]
        assert len(triggers) == 1
        assert triggers[0]["trigger"] == "gun"

    def test_malformed_json_handling(self, _blank_db):
        """Intentionally corrupting a record to ensure the aggregation doesn't crash."""
        now = time.time()
        _insert_row(_blank_db, "s1", "cat1", 0.5, ["valid"], now)
        _insert_raw(_blank_db, "s2", "cat1", 0.5, "{invalid-json]", now)
        
        resp = client.get("/admin/analytics/top-triggers", headers=self.HEADERS)
        assert resp.status_code == 200
        triggers = resp.json()["top_triggers"]
        # The invalid payload is safely skipped
        assert len(triggers) == 1
        assert triggers[0]["trigger"] == "valid"

    def test_empty_triggers_handling(self, _blank_db):
        """Ensures null and empty array records are safely ignored."""
        now = time.time()
        _insert_row(_blank_db, "s1", "cat1", 0.5, [], now)
        _insert_raw(_blank_db, "s2", "cat1", 0.5, None, now)
        
        resp = client.get("/admin/analytics/top-triggers", headers=self.HEADERS)
        assert resp.status_code == 200
        assert resp.json()["top_triggers"] == []



class TestSessionRiskEnhanced:
    """Validates peak extraction and transition detection in session risk endpoint."""
    HEADERS = {"Authorization": "Bearer test-admin-secret"}

    def test_peak_risk_extraction(self, _blank_db):
        base_ts = 1700000000
        
        # Insert chronologically
        _insert_row(_blank_db, "sess1", "safe", 0.1, [], base_ts)
        _insert_row(_blank_db, "sess1", "self_harm", 0.95, [], base_ts + 10) # PEAK
        _insert_row(_blank_db, "sess1", "self_harm", 0.6, [], base_ts + 20)
        
        resp = client.get("/admin/sessions/sess1/risk", headers=self.HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["peak_risk_score"] == 0.95
        assert data["peak_risk_timestamp"] == base_ts + 10

    def test_category_transitions(self, _blank_db):
        base_ts = 1700000000
        
        # Chronological progression: safe -> cat1 -> cat2. (Repeated safe/cat2 shouldn't trigger transition)
        _insert_row(_blank_db, "sess1", "safe", 0.1, [], base_ts)
        _insert_row(_blank_db, "sess1", "safe", 0.2, [], base_ts + 10) 
        _insert_row(_blank_db, "sess1", "cat1", 0.5, [], base_ts + 20) # Transition 1
        _insert_row(_blank_db, "sess1", "cat2", 0.8, [], base_ts + 30) # Transition 2
        _insert_row(_blank_db, "sess1", "cat2", 0.9, [], base_ts + 40)
        
        resp = client.get("/admin/sessions/sess1/risk", headers=self.HEADERS)
        assert resp.status_code == 200
        transitions = resp.json()["category_transitions"]
        
        assert len(transitions) == 2
        assert transitions[0] == {"from": "safe", "to": "cat1", "timestamp": base_ts + 20}
        assert transitions[1] == {"from": "cat1", "to": "cat2", "timestamp": base_ts + 30}

    def test_empty_session_safe(self, _blank_db):
        """Validates that a non-existent session handles edge cases gracefully."""
        resp = client.get("/admin/sessions/unknown-session/risk", headers=self.HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        
        assert data["peak_risk_score"] is None
        assert data["peak_risk_timestamp"] is None
        assert data["category_transitions"] == []
        assert data["history"] == []