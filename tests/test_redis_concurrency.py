"""Concurrency tests for the Redis backend under threaded contention."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import fakeredis

from humane_proxy.config import reload_config
from humane_proxy.risk import trajectory as trajectory_module
from humane_proxy.risk.redis_trajectory import RedisTrajectoryStore
from humane_proxy.risk.trajectory import analyze
from humane_proxy.storage.redis import RedisStore


def _install_fake_redis_client() -> fakeredis.FakeRedis:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    return fake_client


def test_redis_rate_limit_is_atomic_under_contention(monkeypatch):
    fake_client = _install_fake_redis_client()
    monkeypatch.setattr(
        "humane_proxy.storage.redis._redis.Redis.from_url",
        lambda *args, **kwargs: fake_client,
    )

    store = RedisStore(
        {"storage": {"redis": {"url": "redis://fake/0"}}},
        rate_limit_max=3,
        rate_limit_window_hours=1,
    )

    session_id = "redis-rate-limit-threaded"
    barrier = threading.Barrier(10)
    results: list[bool] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        allowed = store.check_rate_limit(session_id)
        with results_lock:
            results.append(allowed)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker) for _ in range(10)]
        for future in futures:
            future.result()

    assert sum(results) == 3
    assert int(fake_client.get(store._key("rate", session_id))) == 10


def test_redis_log_is_atomic_under_contention(monkeypatch):
    fake_client = _install_fake_redis_client()
    monkeypatch.setattr(
        "humane_proxy.storage.redis._redis.Redis.from_url",
        lambda *args, **kwargs: fake_client,
    )

    store = RedisStore(
        {"storage": {"redis": {"url": "redis://fake/0"}}},
        rate_limit_max=3,
        rate_limit_window_hours=1,
    )

    session_id = "redis-log-threaded"
    barrier = threading.Barrier(12)

    def worker(index: int) -> None:
        barrier.wait()
        store.log(
            session_id=session_id,
            category="self_harm" if index % 2 == 0 else "criminal_intent",
            risk_score=0.1 + index * 0.01,
            triggers=[f"trigger-{index}"],
        )

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(worker, index) for index in range(12)]
        for future in futures:
            future.result()

    records = store.query(session_id=session_id)
    assert len(records) == 12
    assert {record["id"] for record in records} == set(range(1, 13))


def test_redis_trajectory_window_is_atomic_under_contention(monkeypatch):
    fake_client = _install_fake_redis_client()
    store = RedisTrajectoryStore(
        {"storage": {"redis": {"url": "redis://fake/0"}}},
        window_size=5,
        client=fake_client,
    )

    monkeypatch.setenv("HUMANE_PROXY_STORAGE_BACKEND", "redis")
    reload_config()
    monkeypatch.setattr(trajectory_module, "_get_redis_trajectory_store", lambda: store)
    trajectory_module.session_history.clear()
    trajectory_module._category_history.clear()

    session_id = "redis-trajectory-threaded"
    barrier = threading.Barrier(10)

    def worker(index: int) -> None:
        barrier.wait()
        analyze(session_id, 0.05 + index * 0.05, "safe")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, index) for index in range(10)]
        for future in futures:
            future.result()

    snapshot = store.snapshot(session_id)
    assert len(snapshot) == 5
    assert len({entry["score"] for entry in snapshot}) == 5
    assert all(isinstance(entry["score"], float) for entry in snapshot)
