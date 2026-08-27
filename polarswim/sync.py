"""Orchestration: discover sessions, fetch what's new, load it.

Kept deliberately thin — discovery, fetch, parse and load each live elsewhere, so
this module is only the loop and its bookkeeping. It is incremental by default:
ids already in the database are skipped without a request, which matters because
the credential is short-lived and the API is rate-limited.

A single session that fails to fetch or parse is recorded and stepped over; one
bad payload should not abandon a multi-year backfill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from sqlalchemy.engine import Engine

from . import db
from .client import FlowClient, FlowError, SessionExpired
from .parse import ParseError, parse_details


@dataclass
class SyncResult:
    events_seen: int = 0
    fetched: int = 0
    skipped: int = 0
    errors: int = 0
    pool_swims: int = 0
    failures: list[tuple[int, str]] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"{self.events_seen} sessions found, {self.fetched} fetched "
                f"({self.pool_swims} pool swims), {self.skipped} already stored, "
                f"{self.errors} failed")


def sync_range(
    engine: Engine,
    client: FlowClient,
    start: date,
    end: date,
    *,
    force: bool = False,
    limit: int | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> SyncResult:
    """Fetch every training session in [start, end] that isn't already stored.

    `force` re-fetches sessions we already have (use after a parser change, though
    reprocessing from `raw_payloads` is usually the better option).
    """
    res = SyncResult()
    run_id = db.start_run(engine, start.isoformat(), end.isoformat())

    progress(f"listing sessions {start} .. {end}")
    try:
        ids = client.exercise_ids(start, end)
    except SessionExpired:
        db.finish_run(engine, run_id, events=0, fetched=0, skipped=0, errors=1,
                      note="session expired during discovery")
        raise
    res.events_seen = len(ids)
    progress(f"{len(ids)} sessions in range")

    known = set() if force else db.known_workout_ids(engine)
    todo = [i for i in ids if i not in known]
    res.skipped = len(ids) - len(todo)
    if limit is not None:
        todo = todo[:limit]

    for n, tid in enumerate(todo, 1):
        try:
            payload = client.analysis_details(tid)
            workouts = parse_details(payload)
        except SessionExpired:
            db.finish_run(engine, run_id, events=res.events_seen, fetched=res.fetched,
                          skipped=res.skipped, errors=res.errors + 1,
                          note=f"session expired after {res.fetched} fetched")
            raise
        except (FlowError, ParseError) as e:
            res.errors += 1
            res.failures.append((tid, str(e)[:200]))
            progress(f"  [{n}/{len(todo)}] {tid}: FAILED {type(e).__name__}")
            continue

        for w in workouts:
            db.upsert_workout(engine, w, raw=payload if w.id == tid else None)
            if w.is_pool_swim:
                res.pool_swims += 1
        res.fetched += 1
        head = workouts[0] if workouts else None
        progress(f"  [{n}/{len(todo)}] {tid}: {head.sport_parent if head else '?'}"
                 f" {len(head.lengths) if head else 0} lengths")

    db.finish_run(engine, run_id, events=res.events_seen, fetched=res.fetched,
                  skipped=res.skipped, errors=res.errors,
                  note=f"{res.pool_swims} pool swims")
    return res
