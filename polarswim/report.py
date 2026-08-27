"""Reporting over a date range.

pandas does the aggregation here because the shape of the work — group by set, by
workout, by month; resample; pivot stroke mix — is exactly what it is for, and the
resulting code is far shorter and clearer than the equivalent SQL plus manual
reshaping.

`sets_for_workout` is the shared derivation used by the Unicode card, the AI
review, and the web UI, so all three describe a workout identically.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from . import analyze
from .models import workouts


def workout_headers(engine: Engine, start: dt.date | None = None,
                    end: dt.date | None = None,
                    pool_only: bool = True) -> pd.DataFrame:
    stmt = sa.select(workouts)
    if pool_only:
        stmt = stmt.where(workouts.c.n_lengths > 0)
    if start:
        stmt = stmt.where(workouts.c.start_time >= start.isoformat())
    if end:
        stmt = stmt.where(workouts.c.start_time <= end.isoformat() + "T23:59:59")
    with engine.connect() as c:
        return pd.DataFrame(c.execute(
            stmt.order_by(workouts.c.start_epoch)).mappings().all())


def classified_lengths(engine: Engine, workout_id: int | None = None,
                       params: dict | None = None) -> pd.DataFrame:
    """Lengths with sets, features and predictions attached."""
    df = analyze.load_lengths(engine, workout_id)
    if df.empty:
        return df
    df = analyze.assign_sets(df)
    hr = analyze.load_hr(engine, sorted(df["workout_id"].unique().tolist()))
    df = analyze.add_features(df, hr)
    if params is None:
        from . import db
        params = db.load_model_params(engine) or analyze.learn_params(df)
    return analyze.classify(df, params)


def sets_for_workout(df: pd.DataFrame, repairs: set[tuple[int, int]] | None = None
                     ) -> list[dict]:
    """Collapse one workout's lengths into per-set rows."""
    repairs = repairs or set()
    out = []
    for sid, g in df.groupby("set_id"):
        mode = g["predicted"].mode()
        note = "repaired" if any((r.workout_id, r.idx) in repairs
                                 for r in g.itertuples()) else ""
        out.append({
            "set_id": int(sid),
            "n": int(len(g)),
            "stroke": mode.iloc[0] if len(mode) else "undetermined",
            "confidence": float(g["confidence"].mean()),
            "pace_s": float(g["pace_s"].median()),
            "hr_cost": float(g["hr_cost"].mean()) if g["hr_cost"].notna().any() else 0.0,
            "rest_before_s": float(g["rest_before_s"].iloc[0]),
            "note": note,
        })
    return out


def season_summary(engine: Engine, start: dt.date | None = None,
                   end: dt.date | None = None) -> dict:
    """Totals, stroke mix, and a month-by-month trend over a date range."""
    heads = workout_headers(engine, start, end)
    if heads.empty:
        return {"workouts": 0}

    lengths_df = classified_lengths(engine)
    lengths_df = lengths_df[lengths_df["workout_id"].isin(heads["id"])]

    heads = heads.assign(month=pd.to_datetime(heads["start_time"]).dt.to_period("M"))
    monthly = heads.groupby("month").agg(
        swims=("id", "count"),
        yards=("distance_m", lambda s: round(s.sum() / 0.9144)),
        avg_hr=("avg_hr", "mean"),
    ).reset_index()
    # Periods are convenient to group by but are not JSON-serializable, and this
    # result is returned over HTTP as well as printed.
    monthly["month"] = monthly["month"].astype(str)

    mix = (lengths_df["predicted"].value_counts(normalize=True) * 100).round(1)
    pace_trend = (lengths_df.assign(
        month=lengths_df["workout_id"].map(
            heads.set_index("id")["month"].to_dict()))
        .groupby("month")["pace_s"].median().round(1))

    return {
        "workouts": int(len(heads)),
        "yards": int(round(heads["distance_m"].sum() / 0.9144)),
        "hours": round(float(heads["duration_s"].sum()) / 3600, 1),
        "lengths": int(len(lengths_df)),
        "first": str(heads["start_time"].min())[:10],
        "last": str(heads["start_time"].max())[:10],
        "stroke_mix_pct": mix.to_dict(),
        "monthly": [{k: (None if pd.isna(v) else v) for k, v in row.items()}
                    for row in monthly.to_dict("records")],
        "median_pace_by_month": {str(k): v for k, v in pace_trend.items()},
    }


def format_season(summary: dict) -> str:
    """Terminal-friendly rendering of `season_summary`."""
    if not summary.get("workouts"):
        return "no workouts in range"
    L = [
        f"  {summary['workouts']} swims   {summary['yards']:,} yd   "
        f"{summary['hours']} h   {summary['lengths']:,} lengths",
        f"  {summary['first']} .. {summary['last']}",
        "",
        "  stroke mix (inferred)",
    ]
    for k, v in sorted(summary["stroke_mix_pct"].items(), key=lambda kv: -kv[1]):
        L.append(f"    {k:<14} {v:>5.1f}%  {'▇' * int(v / 2)}")
    L += ["", "  by month"]
    for row in summary["monthly"]:
        month = str(row["month"])
        pace = summary["median_pace_by_month"].get(month, "-")
        hr = f"{row['avg_hr']:.0f}" if row.get("avg_hr") is not None else "-"
        L.append(f"    {month}  {row['swims']:>2} swims  {row['yards']:>6,} yd  "
                 f"med {pace}s/25  avg HR {hr}")
    return "\n".join(L)
