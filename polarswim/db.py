"""Storage layer over SQLAlchemy Core.

Every write is an upsert keyed on Polar's own identifiers, so a sync can be re-run
over any date range without creating duplicates and without a "have I seen this"
bookkeeping table.

The engine is created from a URL, so the same code runs against local SQLite or a
PostgreSQL server. SQLite needs two pragmas set per-connection (foreign keys are
off by default there, and WAL lets the web UI read during a sync); both are
applied only when the dialect is SQLite.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.engine import Engine

from .models import (ALL_TABLES, hr_samples, labels, lengths, metadata,
                     model_params, predictions, raw_payloads, sync_runs, workouts)
from .parse import Workout

DEFAULT_DB_PATH = Path.home() / ".polarswim" / "polarswim.db"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def epoch_of(iso: str | None) -> int:
    """Best-effort epoch seconds from Flow's naive local timestamps.

    The original string is stored verbatim; this is derived purely for range
    queries, so an unparseable value degrades to 0 rather than failing the load.
    """
    if not iso:
        return 0
    try:
        return int(dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def make_url(path_or_url: str | Path | None) -> str:
    """Accept a filesystem path or a full SQLAlchemy URL."""
    if path_or_url is None:
        return f"sqlite+pysqlite:///{DEFAULT_DB_PATH}"
    s = str(path_or_url)
    if "://" in s:
        return s
    if s == ":memory:":
        return "sqlite+pysqlite:///:memory:"
    return f"sqlite+pysqlite:///{s}"


def connect(path_or_url: str | Path | None = None) -> Engine:
    """Create the engine and ensure the schema exists."""
    url = make_url(path_or_url)
    if url.startswith("sqlite") and ":memory:" not in url:
        Path(url.split("///", 1)[1]).parent.mkdir(parents=True, exist_ok=True)

    # An in-memory SQLite database lives inside a single connection, so the
    # default pool would hand out a fresh, empty database on every checkout.
    # StaticPool keeps one connection for the engine's lifetime.
    kwargs = {}
    if ":memory:" in url:
        from sqlalchemy.pool import StaticPool
        kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
    engine = sa.create_engine(url, future=True, **kwargs)

    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):    # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")    # off by default in SQLite
            cur.execute("PRAGMA journal_mode=WAL")   # readers don't block on a sync
            cur.close()

    metadata.create_all(engine)
    migrate(engine)
    return engine


class SchemaDrift(RuntimeError):
    """A schema change the additive migration cannot perform."""


def migrate(engine: Engine) -> list[str]:
    """Bring an existing database up to the columns declared in `models`.

    `create_all` creates missing TABLES but never touches a table that already
    exists, so adding a column to `models.py` would otherwise leave every
    already-synced database one column short — a failure that surfaces much later
    as an OperationalError on a column the code is certain exists. This closes
    that gap for the only schema change that happens in practice: a new nullable
    column with a default.

    Anything beyond that — a dropped column, a changed type, a new primary key —
    is deliberately NOT attempted. Guessing at those risks the data. It raises
    `SchemaDrift` instead, and the fix is to delete the database and run
    `polarswim reparse`, which rebuilds everything from the stored raw payloads
    with no network and no credential.

    Returns the DDL it applied, so callers and tests can see what moved.
    """
    inspector = sa.inspect(engine)
    existing_tables = set(inspector.get_table_names())
    applied: list[str] = []

    for table in ALL_TABLES:
        if table.name not in existing_tables:
            continue                     # create_all just made it, in full
        have = {c["name"] for c in inspector.get_columns(table.name)}

        # A column in the database but not in the model is harmless — older data
        # we no longer read. A column in the model but not the database is the
        # case worth fixing.
        for name in (c.name for c in table.columns if c.name not in have):
            column = table.columns[name]
            if column.primary_key:
                raise SchemaDrift(
                    f"{table.name}.{name} is a new primary-key column; "
                    "delete the database and run `polarswim reparse` to rebuild it")
            applied.append(_add_column(engine, table, column))

    return applied


def _add_column(engine: Engine, table: sa.Table, column: sa.Column) -> str:
    """ALTER TABLE ... ADD COLUMN, with a default every backend will accept.

    SQLite refuses a NOT NULL column with no default on a table that already has
    rows — there would be nothing to put in the existing ones. Where the model
    supplies a scalar default we use it; where it does not, the column is added
    nullable, since a NULL in old rows is honest about the fact that the value was
    never observed.
    """
    ddl = f'ALTER TABLE {table.name} ADD COLUMN {column.name} '
    ddl += column.type.compile(engine.dialect)

    default = getattr(column.default, "arg", None)
    if default is not None and not callable(default):
        literal = f"'{default}'" if isinstance(default, str) else default
        ddl += f" DEFAULT {literal}"
        if not column.nullable:
            ddl += " NOT NULL"

    with engine.begin() as c:
        c.execute(sa.text(ddl))
    return ddl


# --- reads -----------------------------------------------------------------
def known_workout_ids(engine: Engine) -> set[int]:
    with engine.connect() as c:
        return {r[0] for r in c.execute(sa.select(workouts.c.id))}


def summary(engine: Engine) -> dict:
    with engine.connect() as c:
        scalar = lambda stmt: c.execute(stmt).scalar() or 0
        return {
            "workouts": scalar(sa.select(sa.func.count()).select_from(workouts)),
            "pool_swims": scalar(sa.select(sa.func.count()).select_from(workouts)
                                 .where(workouts.c.n_lengths > 0)),
            "lengths": scalar(sa.select(sa.func.count()).select_from(lengths)),
            "hr_samples": scalar(sa.select(sa.func.count()).select_from(hr_samples)),
            "predictions": scalar(sa.select(sa.func.count()).select_from(predictions)),
            "earliest": c.execute(sa.select(sa.func.min(workouts.c.start_time))).scalar() or "-",
            "latest": c.execute(sa.select(sa.func.max(workouts.c.start_time))).scalar() or "-",
        }


# --- writes ----------------------------------------------------------------
def upsert_workout(engine: Engine, w: Workout, raw: dict | None = None) -> None:
    """Load one workout and all its children in a single transaction."""
    interval = w.hr_interval_s or 1.0
    row = dict(
        id=w.id, start_time=w.start_time, start_epoch=epoch_of(w.start_time),
        stop_time=w.stop_time, sport_parent=w.sport_parent, sport_id=w.sport_id,
        duration_s=w.duration_s, distance_m=w.distance_m, calories=w.calories,
        avg_hr=w.avg_hr, max_hr=w.max_hr, pool_length_m=w.pool_length_m,
        pool_type=w.pool_type, pool_lengths_reported=w.pool_lengths_reported,
        n_lengths=len(w.lengths), hr_interval_s=w.hr_interval_s,
        n_hr_samples=len(w.hr_values), synced_at=now_iso(),
    )

    with engine.begin() as c:                    # commits, or rolls back entirely
        existing = c.execute(
            sa.select(workouts.c.id).where(workouts.c.id == w.id)).scalar()
        if existing is None:
            c.execute(sa.insert(workouts).values(**row))
        else:
            c.execute(sa.update(workouts).where(workouts.c.id == w.id)
                      .values(**{k: v for k, v in row.items() if k != "id"}))

        # Children are replaced wholesale: a re-fetch is authoritative, so this
        # avoids orphans if Polar revises a session's length count.
        c.execute(sa.delete(lengths).where(lengths.c.workout_id == w.id))
        if w.lengths:
            c.execute(sa.insert(lengths), [
                dict(workout_id=w.id, idx=l.idx, start_offset_s=l.start_offset_s,
                     duration_s=l.duration_s, polar_style=l.polar_style,
                     strokes=l.strokes) for l in w.lengths])

        c.execute(sa.delete(hr_samples).where(hr_samples.c.workout_id == w.id))
        if w.hr_values:
            c.execute(sa.insert(hr_samples), [
                dict(workout_id=w.id, t_s=i * interval, hr=hr)
                for i, hr in enumerate(w.hr_values)])

        if raw is not None:
            c.execute(sa.delete(raw_payloads).where(raw_payloads.c.workout_id == w.id))
            c.execute(sa.insert(raw_payloads).values(
                workout_id=w.id, fetched_at=now_iso(),
                payload=json.dumps(raw, separators=(",", ":"))))


def start_run(engine: Engine, window_start: str, window_end: str) -> int:
    with engine.begin() as c:
        return c.execute(sa.insert(sync_runs).values(
            started_at=now_iso(), window_start=window_start,
            window_end=window_end)).inserted_primary_key[0]


def finish_run(engine: Engine, run_id: int, *, events: int, fetched: int,
               skipped: int, errors: int, note: str = "") -> None:
    with engine.begin() as c:
        c.execute(sa.update(sync_runs).where(sync_runs.c.id == run_id).values(
            finished_at=now_iso(), events_seen=events, fetched=fetched,
            skipped=skipped, errors=errors, note=note))


def save_predictions(engine: Engine, rows: list[dict]) -> int:
    """Replace predictions for every workout represented in `rows`."""
    if not rows:
        return 0
    ids = {r["workout_id"] for r in rows}
    stamp = now_iso()
    with engine.begin() as c:
        c.execute(sa.delete(predictions).where(predictions.c.workout_id.in_(ids)))
        c.execute(sa.insert(predictions),
                  [{**r, "predicted_at": stamp} for r in rows])
    return len(rows)


# --- corrections ------------------------------------------------------------
def save_labels(engine: Engine, rows: list[dict]) -> int:
    """Record the swimmer's corrections, replacing any earlier one per length.

    Replace rather than accumulate: a person correcting the same set twice means
    the second answer, not two conflicting opinions to be averaged.
    """
    if not rows:
        return 0
    stamp = now_iso()
    with engine.begin() as c:
        for r in rows:
            c.execute(sa.delete(labels).where(sa.and_(
                labels.c.workout_id == r["workout_id"], labels.c.idx == r["idx"])))
        c.execute(sa.insert(labels),
                  [{"source": "human", **r, "labelled_at": stamp} for r in rows])
    return len(rows)


def clear_labels(engine: Engine, workout_id: int, set_id: int | None = None) -> int:
    """Withdraw corrections, for a whole workout or one set of it."""
    stmt = sa.delete(labels).where(labels.c.workout_id == workout_id)
    if set_id is not None:
        stmt = stmt.where(labels.c.set_id == set_id)
    with engine.begin() as c:
        return c.execute(stmt).rowcount


def load_labels(engine: Engine, workout_id: int | None = None) -> dict:
    """Corrections as {(workout_id, idx): stroke}."""
    stmt = sa.select(labels.c.workout_id, labels.c.idx, labels.c.stroke)
    if workout_id is not None:
        stmt = stmt.where(labels.c.workout_id == workout_id)
    with engine.connect() as c:
        return {(r.workout_id, r.idx): r.stroke for r in c.execute(stmt)}


def label_counts(engine: Engine) -> dict[str, int]:
    """How many lengths are labelled per stroke — where the training data is thin."""
    stmt = (sa.select(labels.c.stroke, sa.func.count())
            .group_by(labels.c.stroke).order_by(sa.func.count().desc()))
    with engine.connect() as c:
        return {r[0]: r[1] for r in c.execute(stmt)}


def save_model_params(engine: Engine, params: dict[str, dict[str, float]]) -> None:
    """Upsert the learned per-class parameters."""
    stamp = now_iso()
    with engine.begin() as c:
        for cls, kv in params.items():
            for k, v in kv.items():
                hit = c.execute(sa.select(model_params.c.value).where(
                    sa.and_(model_params.c.class_name == cls,
                            model_params.c.param == k))).scalar()
                vals = dict(value=float(v), updated_at=stamp)
                if hit is None:
                    c.execute(sa.insert(model_params).values(
                        class_name=cls, param=k, **vals))
                else:
                    c.execute(sa.update(model_params).where(sa.and_(
                        model_params.c.class_name == cls,
                        model_params.c.param == k)).values(**vals))


def load_model_params(engine: Engine) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with engine.connect() as c:
        for r in c.execute(sa.select(model_params)):
            out.setdefault(r.class_name, {})[r.param] = r.value
    return out
