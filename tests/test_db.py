"""Storage: schema, idempotent upserts, cascade, and the derived counts."""

import pytest

from polarswim import db
from polarswim.parse import parse_details


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    yield c
    c.close()


def test_schema_applies_cleanly(conn):
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"workouts", "lengths", "hr_samples", "raw_payloads",
            "sync_runs", "model_params", "predictions"} <= tables


def test_indexes_exist_for_the_access_patterns(conn):
    idx = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name IS NOT NULL")}
    assert {"idx_workouts_start", "idx_workouts_sport", "idx_lengths_duration"} <= idx


def test_upsert_loads_workout_and_children(conn, pool_swim_payload):
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(conn, w, raw=pool_swim_payload)
    s = db.summary(conn)
    assert s["workouts"] == 1
    assert s["pool_swims"] == 1
    assert s["lengths"] == 8
    assert s["hr_samples"] == 10


def test_upsert_is_idempotent(conn, pool_swim_payload):
    """Re-syncing a range must not duplicate anything."""
    (w,) = parse_details(pool_swim_payload)
    for _ in range(3):
        db.upsert_workout(conn, w, raw=pool_swim_payload)
    s = db.summary(conn)
    assert (s["workouts"], s["lengths"], s["hr_samples"]) == (1, 8, 10)


def test_refetch_replaces_children_rather_than_appending(conn, pool_swim_payload):
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(conn, w)
    w.lengths = w.lengths[:3]           # Polar revised the session downward
    db.upsert_workout(conn, w)
    assert db.summary(conn)["lengths"] == 3
    assert conn.execute("SELECT n_lengths FROM workouts").fetchone()[0] == 3


def test_hr_samples_are_timestamped_from_the_interval(conn, pool_swim_payload):
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(conn, w)
    ts = [r[0] for r in conn.execute(
        "SELECT t_s FROM hr_samples ORDER BY t_s")]
    assert ts == [float(i) for i in range(10)]


def test_delete_cascades_to_children(conn, pool_swim_payload):
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(conn, w, raw=pool_swim_payload)
    with conn:
        conn.execute("DELETE FROM workouts WHERE id=?", (w.id,))
    s = db.summary(conn)
    assert (s["lengths"], s["hr_samples"]) == (0, 0)
    assert conn.execute("SELECT COUNT(*) FROM raw_payloads").fetchone()[0] == 0


def test_known_ids_supports_incremental_sync(conn, pool_swim_payload):
    assert db.known_workout_ids(conn) == set()
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(conn, w)
    assert db.known_workout_ids(conn) == {w.id}


def test_raw_payload_round_trips_for_reprocessing(conn, pool_swim_payload):
    import json
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(conn, w, raw=pool_swim_payload)
    stored = conn.execute("SELECT payload FROM raw_payloads").fetchone()[0]
    assert parse_details(json.loads(stored))[0].id == w.id
