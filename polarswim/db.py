"""SQLite storage: schema management and idempotent loading.

Every write is an upsert keyed on Polar's own identifiers, so a sync can be
re-run over any range without creating duplicates or needing a "have I seen this"
table. `INSERT ... ON CONFLICT DO UPDATE` keeps that in one statement per row
rather than a read-then-write race.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

from .parse import Workout

DEFAULT_DB = Path.home() / ".polarswim" / "polarswim.db"
_SCHEMA = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _epoch(iso: str | None) -> int:
    """Best-effort epoch seconds from Flow's local-time strings.

    Flow emits naive local timestamps ('2026-08-19T17:11:11'). We store the
    original string verbatim and derive this only for range queries, so a missing
    or odd value degrades to 0 rather than failing the load.
    """
    if not iso:
        return 0
    try:
        return int(dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the database with the schema applied."""
    p = Path(path) if path else DEFAULT_DB
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")     # concurrent reads during a sync
    conn.executescript(_SCHEMA.read_text())
    return conn


def known_workout_ids(conn: sqlite3.Connection) -> set[int]:
    """Ids already stored, so a sync can skip re-fetching them."""
    return {r[0] for r in conn.execute("SELECT id FROM workouts")}


def upsert_workout(conn: sqlite3.Connection, w: Workout, raw: dict | None = None) -> None:
    """Load one workout and all its children in a single transaction."""
    interval = w.hr_interval_s or 1.0
    with conn:                                    # commits, or rolls back entirely
        conn.execute(
            """INSERT INTO workouts
                 (id, start_time, start_epoch, stop_time, sport_parent, sport_id,
                  duration_s, distance_m, calories, avg_hr, max_hr, pool_length_m,
                  pool_type, pool_lengths_reported, n_lengths, hr_interval_s,
                  n_hr_samples, synced_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                  start_time=excluded.start_time, start_epoch=excluded.start_epoch,
                  stop_time=excluded.stop_time, sport_parent=excluded.sport_parent,
                  sport_id=excluded.sport_id, duration_s=excluded.duration_s,
                  distance_m=excluded.distance_m, calories=excluded.calories,
                  avg_hr=excluded.avg_hr, max_hr=excluded.max_hr,
                  pool_length_m=excluded.pool_length_m, pool_type=excluded.pool_type,
                  pool_lengths_reported=excluded.pool_lengths_reported,
                  n_lengths=excluded.n_lengths, hr_interval_s=excluded.hr_interval_s,
                  n_hr_samples=excluded.n_hr_samples, synced_at=excluded.synced_at""",
            (w.id, w.start_time, _epoch(w.start_time), w.stop_time, w.sport_parent,
             w.sport_id, w.duration_s, w.distance_m, w.calories, w.avg_hr, w.max_hr,
             w.pool_length_m, w.pool_type, w.pool_lengths_reported, len(w.lengths),
             w.hr_interval_s, len(w.hr_values), _now()),
        )

        # Children are replaced wholesale: a re-fetch is authoritative, and this
        # avoids orphans if Polar revises a session's length count.
        conn.execute("DELETE FROM lengths WHERE workout_id = ?", (w.id,))
        conn.executemany(
            """INSERT INTO lengths
                 (workout_id, idx, start_offset_s, duration_s, polar_style, strokes)
               VALUES (?,?,?,?,?,?)""",
            [(w.id, l.idx, l.start_offset_s, l.duration_s, l.polar_style, l.strokes)
             for l in w.lengths],
        )

        conn.execute("DELETE FROM hr_samples WHERE workout_id = ?", (w.id,))
        conn.executemany(
            "INSERT INTO hr_samples (workout_id, t_s, hr) VALUES (?,?,?)",
            [(w.id, i * interval, hr) for i, hr in enumerate(w.hr_values)],
        )

        if raw is not None:
            conn.execute(
                """INSERT INTO raw_payloads (workout_id, fetched_at, payload)
                   VALUES (?,?,?)
                   ON CONFLICT(workout_id) DO UPDATE SET
                     fetched_at=excluded.fetched_at, payload=excluded.payload""",
                (w.id, _now(), json.dumps(raw, separators=(",", ":"))),
            )


def start_run(conn: sqlite3.Connection, window_start: str, window_end: str) -> int:
    cur = conn.execute(
        "INSERT INTO sync_runs (started_at, window_start, window_end) VALUES (?,?,?)",
        (_now(), window_start, window_end))
    conn.commit()
    return cur.lastrowid


def finish_run(conn: sqlite3.Connection, run_id: int, *, events: int, fetched: int,
               skipped: int, errors: int, note: str = "") -> None:
    conn.execute(
        """UPDATE sync_runs SET finished_at=?, events_seen=?, fetched=?, skipped=?,
             errors=?, note=? WHERE id=?""",
        (_now(), events, fetched, skipped, errors, note, run_id))
    conn.commit()


def summary(conn: sqlite3.Connection) -> dict:
    """Headline counts, used by the CLI and handy in tests."""
    q = lambda sql: conn.execute(sql).fetchone()[0]
    return {
        "workouts": q("SELECT COUNT(*) FROM workouts"),
        "pool_swims": q("SELECT COUNT(*) FROM workouts WHERE n_lengths > 0"),
        "lengths": q("SELECT COUNT(*) FROM lengths"),
        "hr_samples": q("SELECT COUNT(*) FROM hr_samples"),
        "earliest": q("SELECT COALESCE(MIN(start_time),'-') FROM workouts"),
        "latest": q("SELECT COALESCE(MAX(start_time),'-') FROM workouts"),
    }
