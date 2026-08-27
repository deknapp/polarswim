"""Command line entry point.

    polarswim sync [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--limit N] [--force]
    polarswim status
    polarswim analyze [--workout ID]
    polarswim card <date|id|latest>   Strava-pasteable Unicode card
    polarswim review <date|id|latest>   AI review (needs ANTHROPIC_API_KEY)

A workout is named by date (2026-08-19), by Polar's training id, or by `latest`.
Dates are what a person actually remembers, so they are the intended form; the id
is there because it is what the database and the web UI show.
    polarswim report [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--json]
    polarswim serve [--port 8770]   local web UI
    polarswim reparse
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys

import sqlalchemy as sa

from . import ai, analyze, db, render, report
from .auth import AuthError, assert_valid, load_cookie, session_expiry
from .client import ClientConfig, FlowClient, FlowError, SessionExpired
from .models import workouts
from .parse import parse_details
from .sync import sync_range


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def resolve_workout(engine, token: str) -> int:
    """Turn a date, a Polar training id, or `latest` into a workout id.

    Nobody remembers a training id, so a date is the form a person will reach for.
    An ambiguous date lists the candidates rather than silently picking one.
    """
    token = token.strip()

    if token.lower() in ("latest", "last"):
        heads = report.workout_headers(engine)
        if heads.empty:
            raise SystemExit("no pool swims stored — run `polarswim sync` first")
        return int(heads.iloc[-1]["id"])

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        with engine.connect() as c:
            rows = c.execute(
                sa.select(workouts.c.id, workouts.c.start_time, workouts.c.n_lengths)
                .where(workouts.c.start_time.like(f"{token}%"))
                .where(workouts.c.n_lengths > 0)
                .order_by(workouts.c.start_epoch)).all()
        if not rows:
            raise SystemExit(f"no pool swim found on {token}")
        if len(rows) > 1:
            listing = "\n".join(
                f"    {r.id}  {r.start_time[11:16]}  {r.n_lengths} lengths" for r in rows)
            raise SystemExit(
                f"{len(rows)} swims on {token} — name one by id:\n{listing}")
        return int(rows[0].id)

    if token.isdigit():
        with engine.connect() as c:
            hit = c.execute(sa.select(workouts.c.id)
                            .where(workouts.c.id == int(token))).scalar()
        if hit is None:
            raise SystemExit(f"no workout with id {token} — try a date, or `latest`")
        return int(hit)

    raise SystemExit(
        f"could not read {token!r} as a workout. Use a date (2026-08-19), "
        "a Polar training id, or `latest`.")


def _header(engine, workout_id: int) -> dict:
    with engine.connect() as c:
        row = c.execute(sa.select(workouts)
                        .where(workouts.c.id == workout_id)).mappings().first()
    if row is None:
        raise SystemExit(f"workout {workout_id} not found — run `polarswim sync` first")
    return dict(row)


# --- commands --------------------------------------------------------------
def cmd_sync(args) -> int:
    cookie = load_cookie(args.cookie)
    assert_valid(cookie)
    exp = session_expiry(cookie)
    if exp:
        mins = int((exp - dt.datetime.now(dt.timezone.utc).timestamp()) / 60)
        print(f"session valid for ~{mins} more minutes")

    engine = db.connect(args.db)
    client = FlowClient(cookie, ClientConfig(min_interval_s=args.interval))
    res = sync_range(engine, client, args.date_from, args.date_to,
                     force=args.force, limit=args.limit, progress=print)
    print(f"\n{res}")
    for tid, err in res.failures[:10]:
        print(f"  failed {tid}: {err}")
    print()
    return cmd_status(args, engine)


def cmd_status(args, engine=None) -> int:
    engine = engine or db.connect(args.db)
    s = db.summary(engine)
    print(f"database: {db.make_url(args.db)}")
    for k in ("workouts", "pool_swims", "lengths", "hr_samples", "predictions"):
        print(f"  {k:<12} {s[k]:,}")
    print(f"  range        {s['earliest'][:10]} .. {s['latest'][:10]}")
    heads = report.workout_headers(engine)
    if not heads.empty:
        print("\n  most recent pool swims:")
        for r in heads.tail(8).iloc[::-1].itertuples():
            print(f"    {r.id}  {r.start_time[:16]}  {r.distance_m:7.0f}m  "
                  f"{r.n_lengths:3d} lengths")
    return 0


def cmd_analyze(args) -> int:
    engine = db.connect(args.db)
    res = analyze.analyze(engine, workout_id=args.workout)
    if not res.n_lengths:
        print("no lengths to analyze — run `polarswim sync` first", file=sys.stderr)
        return 1
    print(f"analyzed {res.n_lengths:,} lengths")
    print(f"repaired {len(res.repairs)} merged length(s) (missed wall turns)")
    print("\n  inferred stroke mix:")
    total = sum(res.counts().values())
    for k, v in sorted(res.counts().items(), key=lambda kv: -kv[1]):
        print(f"    {k:<14} {v:>6,}  {100*v/total:>5.1f}%  {'▇' * int(50*v/total)}")
    g = res.params.get("_global", {})
    if g:
        print(f"\n  learned reference paces (s/25yd) from {int(g['n_obs']):,} lengths:")
        print(f"    p10 {g['pace_p10']:.1f}   p50 {g['pace_p50']:.1f}   "
              f"p90 {g['pace_p90']:.1f}")
    return 0


def cmd_card(args) -> int:
    engine = db.connect(args.db)
    wid = resolve_workout(engine, args.workout)
    df = report.classified_lengths(engine, wid)
    if df.empty:
        print(f"no lengths for workout {wid}", file=sys.stderr)
        return 1
    print(render.strava_block(df, _header(engine, wid)))
    return 0


def cmd_review(args) -> int:
    engine = db.connect(args.db)
    wid = resolve_workout(engine, args.workout)
    df = report.classified_lengths(engine, wid)
    if df.empty:
        print(f"no lengths for workout {wid}", file=sys.stderr)
        return 1
    res = analyze.analyze(engine, workout_id=wid, persist=False)
    sets = report.sets_for_workout(df, {(r.workout_id, r.idx) for r in res.repairs})
    header = _header(engine, wid)

    if not ai.available():
        print("(no ANTHROPIC_API_KEY found — offline summary)\n")
    out = ai.review(header, sets, res.params)
    print(out.text)
    if out.output_tokens:
        print(f"\n[{out.model}: {out.input_tokens:,} in / {out.output_tokens:,} out]")
    return 0


def cmd_report(args) -> int:
    engine = db.connect(args.db)
    s = report.season_summary(engine, args.date_from, args.date_to)
    rng = (f"{args.date_from or 'start'} .. {args.date_to or 'today'}")
    print(f"\n  polarswim report   {rng}\n")
    print(report.format_season(s))
    if args.json:
        print("\n" + json.dumps(s, indent=1, default=str))
    return 0


def cmd_serve(args) -> int:
    from .web import serve
    serve(db_url=args.db, port=args.port)
    return 0


def cmd_reparse(args) -> int:
    """Re-run the parser over stored raw payloads. No network, no credential."""
    from .models import raw_payloads
    engine = db.connect(args.db)
    with engine.connect() as c:
        rows = c.execute(sa.select(raw_payloads)).mappings().all()
    n = 0
    for r in rows:
        for w in parse_details(json.loads(r["payload"])):
            db.upsert_workout(engine, w)
            n += 1
    print(f"reparsed {n} workouts from {len(rows)} stored payloads")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="polarswim",
        description="Pull Polar Flow swim data into SQLite and infer stroke per length.")
    p.add_argument("--db", default=None,
                   help="database path or SQLAlchemy URL (default ~/.polarswim/polarswim.db)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="fetch sessions from Polar Flow")
    s.add_argument("--from", dest="date_from", type=_date,
                   default=dt.date.today() - dt.timedelta(days=365))
    s.add_argument("--to", dest="date_to", type=_date, default=dt.date.today())
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--force", action="store_true")
    s.add_argument("--cookie", default=None)
    s.add_argument("--interval", type=float, default=0.4)
    s.set_defaults(func=cmd_sync)

    st = sub.add_parser("status", help="what's in the database")
    st.set_defaults(func=lambda a: cmd_status(a))

    an = sub.add_parser("analyze", help="infer stroke per length and persist")
    an.add_argument("--workout", type=int, default=None)
    an.set_defaults(func=cmd_analyze)

    cd = sub.add_parser("card", help="Unicode card to paste into Strava")
    cd.add_argument("workout", metavar="date|id|latest",
                    help="e.g. 2026-08-19, a Polar training id, or 'latest'")
    cd.set_defaults(func=cmd_card)

    rv = sub.add_parser("review", help="AI review of one workout")
    rv.add_argument("workout", metavar="date|id|latest",
                    help="e.g. 2026-08-19, a Polar training id, or 'latest'")
    rv.set_defaults(func=cmd_review)

    rp = sub.add_parser("report", help="summary over a date range")
    rp.add_argument("--from", dest="date_from", type=_date, default=None)
    rp.add_argument("--to", dest="date_to", type=_date, default=None)
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(func=cmd_report)

    sv = sub.add_parser("serve", help="local web UI")
    sv.add_argument("--port", type=int, default=8770)
    sv.set_defaults(func=cmd_serve)

    rs = sub.add_parser("reparse", help="re-run the parser over stored payloads")
    rs.set_defaults(func=cmd_reparse)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except AuthError as e:
        print(f"auth: {e}", file=sys.stderr); return 2
    except SessionExpired as e:
        print(f"session expired: {e}", file=sys.stderr); return 3
    except FlowError as e:
        print(f"api: {e}", file=sys.stderr); return 4
    except ai.AIError as e:
        print(f"ai: {e}", file=sys.stderr); return 5


if __name__ == "__main__":
    raise SystemExit(main())
