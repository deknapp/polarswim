"""Command line entry point.

    polarswim sync   [--from YYYY-MM-DD] [--to ...] [--limit N] [--force]
    polarswim status
    polarswim lengths <workout_id>
    polarswim reparse
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from . import db
from .auth import AuthError, assert_valid, load_cookie, session_expiry
from .client import ClientConfig, FlowClient, FlowError, SessionExpired
from .parse import parse_details
from .sync import sync_range


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def cmd_sync(args) -> int:
    cookie = load_cookie(args.cookie)
    assert_valid(cookie)
    exp = session_expiry(cookie)
    if exp:
        mins = int((exp - dt.datetime.now(dt.timezone.utc).timestamp()) / 60)
        print(f"session valid for ~{mins} more minutes")

    conn = db.connect(args.db)
    client = FlowClient(cookie, ClientConfig(min_interval_s=args.interval))
    res = sync_range(conn, client, args.date_from, args.date_to,
                     force=args.force, limit=args.limit, progress=print)
    print(f"\n{res}")
    for tid, err in res.failures[:10]:
        print(f"  failed {tid}: {err}")
    print()
    cmd_status(args, conn)
    return 1 if res.errors and not res.fetched else 0


def cmd_status(args, conn=None) -> int:
    conn = conn or db.connect(args.db)
    s = db.summary(conn)
    print(f"database: {args.db or db.DEFAULT_DB}")
    print(f"  workouts     {s['workouts']}")
    print(f"  pool swims   {s['pool_swims']}")
    print(f"  lengths      {s['lengths']}")
    print(f"  hr samples   {s['hr_samples']}")
    print(f"  range        {s['earliest'][:10]} .. {s['latest'][:10]}")
    rows = conn.execute(
        """SELECT start_time, distance_m, n_lengths, pool_type, pool_length_m
           FROM workouts WHERE n_lengths > 0
           ORDER BY start_epoch DESC LIMIT 10""").fetchall()
    if rows:
        print("\n  most recent pool swims:")
        for r in rows:
            print(f"    {r['start_time'][:16]}  {r['distance_m']:7.0f}m  "
                  f"{r['n_lengths']:3d} lengths  ({r['pool_length_m']}m {r['pool_type']})")
    return 0


def cmd_lengths(args) -> int:
    conn = db.connect(args.db)
    rows = conn.execute(
        """SELECT idx, start_offset_s, duration_s, polar_style, strokes
           FROM lengths WHERE workout_id=? ORDER BY idx""", (args.workout_id,)).fetchall()
    if not rows:
        print(f"no lengths stored for workout {args.workout_id}", file=sys.stderr)
        return 1
    print(f"{'#':>4} {'start':>8} {'dur':>7}  {'polar':<8} strokes")
    for r in rows:
        print(f"{r['idx']:>4} {r['start_offset_s']:>7.1f}s {r['duration_s']:>6.1f}s  "
              f"{str(r['polar_style']):<8} {r['strokes']}")
    return 0


def cmd_reparse(args) -> int:
    """Re-run the parser over stored raw payloads. No network, no credential."""
    conn = db.connect(args.db)
    rows = conn.execute("SELECT workout_id, payload FROM raw_payloads").fetchall()
    n = 0
    for r in rows:
        for w in parse_details(json.loads(r["payload"])):
            db.upsert_workout(conn, w)
            n += 1
    print(f"reparsed {n} workouts from {len(rows)} stored payloads")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="polarswim",
                                description="Pull Polar Flow swim data into SQLite.")
    p.add_argument("--db", default=None, help=f"database path (default {db.DEFAULT_DB})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="fetch sessions from Polar Flow")
    s.add_argument("--from", dest="date_from", type=_date,
                   default=dt.date.today() - dt.timedelta(days=365))
    s.add_argument("--to", dest="date_to", type=_date, default=dt.date.today())
    s.add_argument("--limit", type=int, default=None, help="stop after N new sessions")
    s.add_argument("--force", action="store_true", help="re-fetch sessions already stored")
    s.add_argument("--cookie", default=None, help="path to a cookie/cURL file")
    s.add_argument("--interval", type=float, default=0.4,
                   help="minimum seconds between requests")
    s.set_defaults(func=cmd_sync)

    st = sub.add_parser("status", help="what's in the database")
    st.set_defaults(func=lambda a: cmd_status(a))

    ln = sub.add_parser("lengths", help="print one workout's lengths")
    ln.add_argument("workout_id", type=int)
    ln.set_defaults(func=cmd_lengths)

    rp = sub.add_parser("reparse", help="re-run the parser over stored raw payloads")
    rp.set_defaults(func=cmd_reparse)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except AuthError as e:
        print(f"auth: {e}", file=sys.stderr)
        return 2
    except SessionExpired as e:
        print(f"session expired: {e}", file=sys.stderr)
        return 3
    except FlowError as e:
        print(f"api: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
