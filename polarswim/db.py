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

from .models import (ALL_TABLES, hr_samples, lengths, metadata, model_params,
                     predictions, raw_payloads, sync_runs, workouts)
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

    engine = sa.create_engine(url, future=True)

    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):    # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")    # off by default in SQLite
            cur.execute("PRAGMA journal_mode=WAL")   # readers don't block on a sync
            cur.close()

    metadata.create_all(engine)
    return engine


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
