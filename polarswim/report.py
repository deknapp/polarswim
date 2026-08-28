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
    # Repairs are applied here too, so the card, the image and the web UI all
    # describe the swim with the same corrected paces the classifier used.
    df = analyze.apply_repairs(df, analyze.detect_repairs(df))
    if params is None:
        from . import db
        params = db.load_model_params(engine) or analyze.learn_params(df)
    df = analyze.classify(df, params)
    df = analyze.label_im(df, analyze.detect_im(df))

    # Order matters and is the whole contract. The fitted model may improve on the
    # rules; the swimmer's corrections outrank everything, including the model
    # that was trained on them. The model is loaded, not fitted — fitting happens
    # once, in `analyze`, over every labelled length there is.
    from . import db, learn
    model = learn.from_params(params)
    if model.is_usable():
        df = learn.apply(df, model)
    return learn.apply_labels(df, db.load_labels(engine, workout_id))


def sets_for_workout(df: pd.DataFrame, repairs: set[tuple[int, int]] | None = None,
                     ref=None) -> list[dict]:
    """Collapse one workout's lengths into per-set rows."""
    repairs = repairs or set()
    out = []
    pool_yd = 25
    if "pool_m" in df.columns and df["pool_m"].notna().any():
        pool_yd = int(round(float(df["pool_m"].iloc[0]) / 0.9144))
    for sid, g in df.groupby("set_id"):
        # A medley set has no single stroke: its lengths are one of each, so the
        # mode is a four-way tie that pandas breaks alphabetically — which is how
        # three 100 IMs came to be reported as 300 yards of backstroke, while the
        # stroke mix beneath it correctly showed 75. Name it for what it is.
        if "im_continuous" in g.columns and g["im_continuous"].all():
            mode = pd.Series(["IM"])
        else:
            mode = g["predicted"].mode()
        # Short-circuit on the empty case: the generator would otherwise read
        # workout_id off every row just to test membership of an empty set, which
        # needlessly requires a column a caller may not have.
        note = "repaired" if repairs and any((r.workout_id, r.idx) in repairs
                                             for r in g.itertuples()) else ""
        reps = g.groupby("rep_id")
        n_reps = reps.ngroups
        out.append({
            "set_id": int(sid),
            "reps": int(n_reps),
            "rep_yards": int(round(len(g) / n_reps)) * pool_yd,
            "rep_seconds": float(reps["duration_s"].sum().median()),
            "n": int(len(g)),
            "stroke": mode.iloc[0] if len(mode) else "undetermined",
            "confidence": float(g["confidence"].mean()),
            "pace_s": float(g["pace_s"].median()),
            # Per 50 for display: a 50 is the unit swimmers actually quote, and
            # it makes a 100 set comparable with a 50 set at a glance.
            "pace_50_s": float(reps["duration_s"].sum().median())
                         / max(1e-9, (int(round(len(g) / n_reps)) * pool_yd) / 50.0),
            "hr_cost": float(g["hr_cost"].mean()) if g["hr_cost"].notna().any() else 0.0,
            # Optional: a caller may hand us a frame assembled without the set
            # features, and a missing rest reads better as zero than as a crash.
            "rest_before_s": (float(g["rest_before_s"].iloc[0])
                              if "rest_before_s" in g.columns else 0.0),
            "note": note,
        })
        if ref is not None:
            row = out[-1]
            mean_hr = ref.hr_max * 0 + _absolute_hr(g)
            row["hr_zone"] = ref.hr_zone(mean_hr)
            fastest = float(reps["duration_s"].sum().min())
            median_rep = float(reps["duration_s"].sum().median())
            row["speed"] = (ref.im_percentile(row["rep_yards"], median_rep)
                            if row["stroke"] == "IM"
                            else ref.speed_percentile(row["rep_yards"], median_rep,
                                                      row["stroke"]))
            row["pr"] = ref.check_pr(row["rep_yards"], row["stroke"], fastest,
                                     int(g["workout_id"].iloc[0]))
            row["best_rep_s"] = fastest
    return out


def _absolute_hr(g: pd.DataFrame) -> float:
    """Mean heart rate over a set, in bpm.

    `hr_cost` is relative to the workout's own baseline, which is right for
    comparing effort within a session but wrong for zoning — a zone is defined
    against absolute maximum heart rate.
    """
    if "hr_abs" in g.columns and g["hr_abs"].notna().any():
        return float(g["hr_abs"].mean())
    return float("nan")


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


def overall_summary(engine: Engine, ref) -> dict:
    """Everything for the summary tab: totals, heart rate, load, and mix."""
    import numpy as np
    from .models import hr_samples as hr_t

    heads = workout_headers(engine)
    lengths_df = classified_lengths(engine)
    if heads.empty:
        return {"workouts": 0}

    with engine.connect() as c:
        hr = np.array([r[0] for r in c.execute(sa.select(hr_t.c.hr))], dtype=float)

    # Time in each zone across the whole database, at one sample per second.
    zone_time = []
    bounds = ref.zone_bounds()
    for i, band in enumerate(bounds):
        lo, hi = band["low"], band["high"]
        # Zones start at resting heart rate, so anything below it belongs to the
        # bottom band rather than to no band at all — otherwise the breakdown
        # quietly loses the warm-up and stops summing to 100%.
        if i == 0:
            lo = 0
        if i == len(bounds) - 1:
            hi = max(hi, int(hr.max()) + 1) if len(hr) else hi
        seconds = int(((hr >= lo) & (hr < hi)).sum()) if len(hr) else 0
        zone_time.append({**band, "seconds": seconds,
                          "pct": round(100 * seconds / max(1, len(hr)), 1)})

    efforts = [ref.effort_score(int(w)) for w in heads["id"]]
    efforts = [e["trimp"] for e in efforts if e]

    mix = (lengths_df["predicted"].value_counts(normalize=True) * 100).round(1)
    dates = pd.to_datetime(heads["start_time"])

    return {
        "workouts": int(len(heads)),
        "yards": int(round(heads["distance_m"].sum() / 0.9144)),
        "hours": round(float(heads["duration_s"].sum()) / 3600, 1),
        "lengths": int(len(lengths_df)),
        "first": str(heads["start_time"].min())[:10],
        "last": str(heads["start_time"].max())[:10],
        "weeks": max(1, int((dates.max() - dates.min()).days / 7)),
        "hr_max": int(ref.hr_max),
        "hr_mean": int(np.nanmean(hr)) if len(hr) else 0,
        "hr_p95": int(np.percentile(hr, 95)) if len(hr) else 0,
        "hr_samples": int(len(hr)),
        "zone_time": zone_time,
        "total_trimp": round(sum(efforts), 1),
        "mean_trimp": round(sum(efforts) / max(1, len(efforts)), 1),
        "stroke_mix_pct": mix.to_dict(),
        "longest_yards": int(round(heads["distance_m"].max() / 0.9144)),
        "implausible_reps": int(getattr(ref, "implausible_reps", 0)),
    }


def personal_bests(engine: Engine, ref) -> list[dict]:
    """Every best the history supports, single-stroke and medley alike.

    Each row carries whether its distance is one the stroke is actually raced at,
    so the UI can lead with the events that mean something without throwing away
    the rest — a fastest 75 is still this swimmer's fastest 75.
    """
    from . import metrics

    out = []
    for (yards, stroke), best in ref.best_rep.items():
        out.append({
            "yards": int(yards),
            "stroke": stroke,
            "seconds": round(best["seconds"], 1),
            "pace_per_50": round(best["seconds"] / (yards / 50.0), 1),
            "date": best["date"],
            "workout_id": int(best["workout_id"]),
            "n_attempts": int(len(ref.rep_times_by_distance_stroke.get(
                (int(yards), stroke), []))),
            "competitive": metrics.is_competitive(yards, stroke),
            "form": None,
            "splits_s": None,
        })

    for yards, best in getattr(ref, "best_im", {}).items():
        out.append({
            "yards": int(yards),
            "stroke": "IM",
            "seconds": round(best["seconds"], 1),
            "pace_per_50": round(best["seconds"] / (yards / 50.0), 1),
            "date": best["date"],
            "workout_id": int(best["workout_id"]),
            "n_attempts": int(best["n_rounds"]),
            "competitive": metrics.is_competitive(yards, "IM"),
            # A broken medley off four walls is quicker than one swum straight,
            # so the two are labelled rather than silently compared.
            "form": "continuous" if best["continuous"] else "broken",
            "splits_s": best["splits_s"],
        })

    out.sort(key=lambda r: (r["stroke"], r["yards"]))
    return out


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
