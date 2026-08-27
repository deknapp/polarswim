"""Schema as SQLAlchemy Core tables.

Declaring the schema once in SQLAlchemy rather than in hand-written DDL means the
same definitions target SQLite locally and PostgreSQL on a server by changing only
the connection URL — `polarswim --db postgresql+psycopg://host/polarswim` works
without touching this file. Core (not the ORM) because the workload is bulk
inserts and analytical reads, where an identity map buys nothing.

Design notes:
  * Polar's own training id is the primary key throughout. It is stable and
    globally unique, so re-syncing any date range is naturally idempotent.
  * Raw payloads are retained so parsing and analysis can be re-run over history
    without re-fetching — the credential is short-lived and the API rate limited.
  * `lengths` is the fact table, keyed (workout_id, idx) since a pool length has
    no identifier of its own.
  * Indexes cover the three real access patterns: browse by date, filter to
    swims, and pull one workout's children.
"""

from __future__ import annotations

from sqlalchemy import (
    Column, Float, ForeignKey, ForeignKeyConstraint, Index, Integer,
    MetaData, String, Table, Text,
)

metadata = MetaData()

workouts = Table(
    "workouts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),  # Polar training id
    Column("start_time", String(32), nullable=False),
    Column("start_epoch", Integer, nullable=False),                # derived, for range scans
    Column("stop_time", String(32)),
    Column("sport_parent", String(32)),
    Column("sport_id", Integer),
    Column("duration_s", Float),
    Column("distance_m", Float),
    Column("calories", Integer),
    Column("avg_hr", Integer),
    Column("max_hr", Integer),
    Column("pool_length_m", Float),          # NULL when Polar omitted the pool config
    Column("pool_type", String(16)),
    Column("pool_lengths_reported", Integer),
    Column("n_lengths", Integer, nullable=False, default=0),
    Column("hr_interval_s", Float),
    Column("n_hr_samples", Integer, nullable=False, default=0),
    Column("synced_at", String(32), nullable=False),
    Index("idx_workouts_start", "start_epoch"),
    Index("idx_workouts_sport", "sport_parent", "start_epoch"),
)

lengths = Table(
    "lengths", metadata,
    Column("workout_id", Integer,
           ForeignKey("workouts.id", ondelete="CASCADE"), primary_key=True),
    Column("idx", Integer, primary_key=True, autoincrement=False),
    Column("start_offset_s", Float, nullable=False),
    Column("duration_s", Float, nullable=False),
    Column("polar_style", String(24)),       # Polar's claim; 'OTHER' when it cannot tell
    Column("strokes", Integer),
    Index("idx_lengths_duration", "duration_s"),
)

hr_samples = Table(
    "hr_samples", metadata,
    Column("workout_id", Integer,
           ForeignKey("workouts.id", ondelete="CASCADE"), primary_key=True),
    Column("t_s", Float, primary_key=True),  # seconds from workout start
    Column("hr", Integer, nullable=False),
    sqlite_with_rowid=False,                 # narrow composite-key table; rowid is waste
)

raw_payloads = Table(
    "raw_payloads", metadata,
    Column("workout_id", Integer,
           ForeignKey("workouts.id", ondelete="CASCADE"), primary_key=True),
    Column("fetched_at", String(32), nullable=False),
    Column("payload", Text, nullable=False),
)

sync_runs = Table(
    "sync_runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("started_at", String(32), nullable=False),
    Column("finished_at", String(32)),
    Column("window_start", String(16)),
    Column("window_end", String(16)),
    Column("events_seen", Integer, nullable=False, default=0),
    Column("fetched", Integer, nullable=False, default=0),
    Column("skipped", Integer, nullable=False, default=0),
    Column("errors", Integer, nullable=False, default=0),
    Column("note", Text),
)

# Learned per-class parameters. Keeping the model in the database rather than a
# pickle makes it inspectable, diffable, and versioned alongside the data it was
# estimated from — and lets it be refined incrementally as workouts arrive.
model_params = Table(
    "model_params", metadata,
    Column("class_name", String(24), primary_key=True),
    Column("param", String(32), primary_key=True),
    Column("value", Float, nullable=False),
    Column("updated_at", String(32), nullable=False),
)

# Inference output, deliberately separate from observed data so the classifier can
# be re-run and compared without touching anything Polar reported.
predictions = Table(
    "predictions", metadata,
    Column("workout_id", Integer, primary_key=True),
    Column("idx", Integer, primary_key=True, autoincrement=False),
    Column("predicted", String(24), nullable=False),
    Column("confidence", Float),
    Column("method", String(32)),
    Column("set_id", Integer),               # which set within the workout
    Column("inferred_split", Integer, nullable=False, default=0),  # 1 = boundary we invented
    Column("predicted_at", String(32), nullable=False),
    ForeignKeyConstraint(["workout_id", "idx"], ["lengths.workout_id", "lengths.idx"],
                         ondelete="CASCADE"),
    Index("idx_predictions_class", "predicted"),
)

ALL_TABLES = (workouts, lengths, hr_samples, raw_payloads,
              sync_runs, model_params, predictions)
