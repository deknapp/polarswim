"""Storage: schema, idempotent upserts, cascade, and portability."""

import json

import pytest
import sqlalchemy as sa

from polarswim import db
from polarswim.models import hr_samples, lengths, raw_payloads, workouts
from polarswim.parse import parse_details


@pytest.fixture
def engine():
    return db.connect(":memory:")


def test_schema_applies_cleanly(engine):
    names = set(sa.inspect(engine).get_table_names())
    assert {"workouts", "lengths", "hr_samples", "raw_payloads",
            "sync_runs", "model_params", "predictions"} <= names


def test_indexes_exist_for_the_access_patterns(engine):
    insp = sa.inspect(engine)
    found = {i["name"] for t in ("workouts", "lengths") for i in insp.get_indexes(t)}
    assert {"idx_workouts_start", "idx_workouts_sport", "idx_lengths_duration"} <= found


def test_upsert_loads_workout_and_children(engine, pool_swim_payload):
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(engine, w, raw=pool_swim_payload)
    s = db.summary(engine)
    assert (s["workouts"], s["pool_swims"], s["lengths"], s["hr_samples"]) == (1, 1, 8, 10)


def test_upsert_is_idempotent(engine, pool_swim_payload):
    """Re-syncing an overlapping date range must not duplicate anything."""
    (w,) = parse_details(pool_swim_payload)
    for _ in range(3):
        db.upsert_workout(engine, w, raw=pool_swim_payload)
    s = db.summary(engine)
    assert (s["workouts"], s["lengths"], s["hr_samples"]) == (1, 8, 10)


def test_refetch_replaces_children_rather_than_appending(engine, pool_swim_payload):
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(engine, w)
    w.lengths = w.lengths[:3]              # Polar revised the session downward
    db.upsert_workout(engine, w)
    assert db.summary(engine)["lengths"] == 3
    with engine.connect() as c:
        assert c.execute(sa.select(workouts.c.n_lengths)).scalar() == 3


def test_hr_samples_are_timestamped_from_the_interval(engine, pool_swim_payload):
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(engine, w)
    with engine.connect() as c:
        ts = [r[0] for r in c.execute(
            sa.select(hr_samples.c.t_s).order_by(hr_samples.c.t_s))]
    assert ts == [float(i) for i in range(10)]


def test_delete_cascades_to_children(engine, pool_swim_payload):
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(engine, w, raw=pool_swim_payload)
    with engine.begin() as c:
        c.execute(sa.delete(workouts).where(workouts.c.id == w.id))
    s = db.summary(engine)
    assert (s["lengths"], s["hr_samples"]) == (0, 0)
    with engine.connect() as c:
        assert c.execute(sa.select(sa.func.count()).select_from(raw_payloads)).scalar() == 0


def test_known_ids_supports_incremental_sync(engine, pool_swim_payload):
    assert db.known_workout_ids(engine) == set()
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(engine, w)
    assert db.known_workout_ids(engine) == {w.id}


def test_raw_payload_round_trips_for_reprocessing(engine, pool_swim_payload):
    """Reprocessing after a parser change must never need the network."""
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(engine, w, raw=pool_swim_payload)
    with engine.connect() as c:
        stored = c.execute(sa.select(raw_payloads.c.payload)).scalar()
    assert parse_details(json.loads(stored))[0].id == w.id


def test_model_params_round_trip(engine):
    db.save_model_params(engine, {"_global": {"pace_p50": 26.0, "n_obs": 7615}})
    assert db.load_model_params(engine)["_global"]["pace_p50"] == 26.0


def test_model_params_update_in_place(engine):
    """Learned values are refined as workouts arrive, not appended."""
    db.save_model_params(engine, {"_global": {"pace_p50": 26.0}})
    db.save_model_params(engine, {"_global": {"pace_p50": 25.1}})
    assert db.load_model_params(engine)["_global"]["pace_p50"] == 25.1


def test_predictions_are_replaced_per_workout(engine, pool_swim_payload):
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(engine, w)
    rows = [dict(workout_id=w.id, idx=i, predicted="freestyle", confidence=0.8,
                 method="t", set_id=1, inferred_split=0) for i in range(1, 9)]
    db.save_predictions(engine, rows)
    db.save_predictions(engine, rows[:3])
    assert db.summary(engine)["predictions"] == 3


class TestPortability:
    """The schema is declared once and targets more than SQLite."""

    def test_path_becomes_a_sqlite_url(self):
        assert db.make_url("/tmp/x.db").startswith("sqlite+pysqlite:///")

    def test_a_full_url_is_passed_through(self):
        url = "postgresql+psycopg://user@host/polarswim"
        assert db.make_url(url) == url

    def test_schema_compiles_for_postgres(self):
        """Catches SQLite-only constructs before they reach a real server."""
        from sqlalchemy.dialects import postgresql
        from polarswim.models import ALL_TABLES
        for table in ALL_TABLES:
            ddl = str(sa.schema.CreateTable(table).compile(
                dialect=postgresql.dialect()))
            assert "CREATE TABLE" in ddl


# --- additive migration ----------------------------------------------------
def test_migrate_is_a_no_op_on_a_current_database(engine):
    assert db.migrate(engine) == []


def test_migrate_adds_a_column_the_model_gained(engine):
    """The failure this exists to prevent: create_all leaves an existing table
    untouched, so a column added to models.py never reaches an old database."""
    with engine.begin() as c:
        c.execute(sa.text("ALTER TABLE predictions DROP COLUMN set_id"))
    assert any("set_id" in ddl for ddl in db.migrate(engine))
    cols = {c["name"] for c in sa.inspect(engine).get_columns("predictions")}
    assert "set_id" in cols


def test_migrate_preserves_rows_in_the_altered_table(engine, pool_swim_payload):
    for w in parse_details(pool_swim_payload):
        db.upsert_workout(engine, w)
    with engine.connect() as c:
        wid, idx = c.execute(sa.select(lengths.c.workout_id, lengths.c.idx)).first()
    db.save_predictions(engine, [dict(
        workout_id=wid, idx=idx, predicted="freestyle", confidence=0.8,
        method="t", set_id=1, inferred_split=0)])
    with engine.begin() as c:
        c.execute(sa.text("ALTER TABLE predictions DROP COLUMN confidence"))
    db.migrate(engine)
    with engine.connect() as c:
        row = c.execute(sa.text("SELECT predicted, confidence FROM predictions")).first()
    assert row[0] == "freestyle"
    assert row[1] is None          # old rows never observed the new column


def test_migrate_refuses_to_guess_at_a_new_primary_key(engine, monkeypatch):
    """Anything beyond adding a nullable column is a rebuild, not a migration."""
    from polarswim import models
    extra = sa.Column("scheme", sa.Integer, primary_key=True)
    monkeypatch.setattr(
        db, "ALL_TABLES",
        (sa.Table("model_params", sa.MetaData(),
                  sa.Column("class_name", sa.String(24), primary_key=True),
                  sa.Column("param", sa.String(32), primary_key=True),
                  sa.Column("value", sa.Float, nullable=False),
                  sa.Column("updated_at", sa.String(32), nullable=False),
                  extra),))
    with pytest.raises(db.SchemaDrift, match="reparse"):
        db.migrate(engine)


def test_a_column_the_database_has_but_the_model_dropped_is_left_alone(engine):
    with engine.begin() as c:
        c.execute(sa.text("ALTER TABLE workouts ADD COLUMN legacy_note TEXT"))
    assert db.migrate(engine) == []
    cols = {c["name"] for c in sa.inspect(engine).get_columns("workouts")}
    assert "legacy_note" in cols
