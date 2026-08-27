"""Infer stroke per length, without labels.

Polar reports `OTHER` for every length on an arm-worn sensor, so there is no
ground truth to train against and no vendor label to fall back on. What we do have
is 7,000+ lengths of timing plus a 1 Hz heart-rate stream, and enough structure to
work with:

  1. Reps and sets.  A gap over `REST_GAP_S` is a rest; below it the swimmer never
     stopped, so those lengths are one continuous **rep** — four unbroken lengths of
     a 25 yd pool is a 100, not four 25s, and reporting it as four 25s misdescribes
     the practice. Consecutive reps of equal distance form a **set** (4x100). The
     rep is the unit a swimmer thinks in; the set is the unit with enough lengths
     to estimate from, so statistics are computed over the set.

  2. Turn-detection defects.  Polar's turn detection misses walls, fusing two
     lengths into one record. A slow length is ambiguous on its own — it could be a
     merged pair or a genuinely slow drill — but not in context: a merge is an
     ISOLATED near-integer multiple of its set's median, while a drill set is
     uniformly slow. Repairing this first matters, because an unrepaired 2x length
     would otherwise be classified as a stroke.

  3. Pace and cost.  Per length we derive normalized pace (seconds per 25 yd, so
     pools are comparable) and heart-rate cost above that workout's own baseline.
     These two axes separate strokes that a single axis cannot: breaststroke is
     slow AND cheap (long glide), while a weak backstroke is slow AND expensive.
     That matters because per-swimmer speed ORDER is not universal — plenty of
     swimmers are slower at backstroke than breaststroke — so nothing here assumes
     a ranking. Clusters are found in the data and identified by their signature.

Identification anchors, in descending confidence:
    freestyle   the dominant fast cluster (the most-swum stroke for most swimmers)
    other       slow, uniform sets with low cost — drill and kick
    butterfly   the highest cost per unit pace, whatever its speed
    breast/back split on cost rather than speed; `undetermined` when they overlap

Everything the model learns is written to `model_params` and reused, so estimates
tighten as more workouts are synced. If ground-truth labels are ever supplied, the
same table is where they would pin the clusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from .models import hr_samples, lengths, workouts

REFERENCE_LENGTH_M = 22.86          # 25 yards; the unit all pace is normalized to
# A gap longer than this is a rest; at or below it the swim is continuous. The
# observed distribution is sharply bimodal — 64.6% of gaps are exactly zero and
# only 0.5% fall between zero and two seconds — so anything from 0.5 s to 5 s
# gives the same answer. Two seconds sits comfortably in that dead zone.
REST_GAP_S = 2.0
HR_LAG_S = 15                       # cardiac response lag when attributing HR
CLASSES = ("freestyle", "backstroke", "breaststroke", "butterfly",
           "other", "undetermined")


@dataclass
class Repair:
    """One turn-detection defect we corrected."""
    workout_id: int
    idx: int
    observed_s: float
    set_median_s: float
    factor: int                     # how many lengths the record actually covers
    kind: str                       # 'merged'


@dataclass
class AnalysisResult:
    predictions: list[dict] = field(default_factory=list)
    params: dict[str, dict[str, float]] = field(default_factory=dict)
    repairs: list[Repair] = field(default_factory=list)
    n_lengths: int = 0

    def counts(self) -> dict[str, int]:
        out = {c: 0 for c in CLASSES}
        for p in self.predictions:
            out[p["predicted"]] = out.get(p["predicted"], 0) + 1
        return {k: v for k, v in out.items() if v}


# --- loading ---------------------------------------------------------------
def load_lengths(engine: Engine, workout_id: int | None = None) -> pd.DataFrame:
    """Lengths joined to their workout, with pool length resolved.

    Polar sometimes omits `poolInfo` entirely. When it does, the pool length is
    still recoverable as distance / length-count, so those workouts stay usable
    instead of being dropped.
    """
    stmt = (sa.select(
                lengths.c.workout_id, lengths.c.idx, lengths.c.start_offset_s,
                lengths.c.duration_s, lengths.c.polar_style,
                workouts.c.start_time, workouts.c.pool_length_m,
                workouts.c.distance_m, workouts.c.n_lengths)
            .select_from(lengths.join(workouts, lengths.c.workout_id == workouts.c.id)))
    if workout_id is not None:
        stmt = stmt.where(lengths.c.workout_id == workout_id)

    with engine.connect() as c:
        df = pd.DataFrame(c.execute(stmt.order_by(
            lengths.c.workout_id, lengths.c.idx)).mappings().all())
    if df.empty:
        return df

    derived = df["distance_m"] / df["n_lengths"].replace(0, np.nan)
    df["pool_m"] = df["pool_length_m"].fillna(derived)
    df = df[df["pool_m"].notna() & (df["pool_m"] > 0)].copy()
    df["pace_s"] = df["duration_s"] * (REFERENCE_LENGTH_M / df["pool_m"])
    return df


def load_hr(engine: Engine, workout_ids: list[int]) -> dict[int, np.ndarray]:
    """HR series per workout, indexed by whole seconds from start."""
    if not workout_ids:
        return {}
    stmt = (sa.select(hr_samples.c.workout_id, hr_samples.c.t_s, hr_samples.c.hr)
            .where(hr_samples.c.workout_id.in_(workout_ids))
            .order_by(hr_samples.c.workout_id, hr_samples.c.t_s))
    with engine.connect() as c:
        df = pd.DataFrame(c.execute(stmt).mappings().all())
    return {} if df.empty else {
        wid: g.sort_values("t_s")["hr"].to_numpy(dtype=float)
        for wid, g in df.groupby("workout_id")
    }


# --- structure -------------------------------------------------------------
def assign_sets(df: pd.DataFrame) -> pd.DataFrame:
    """Group lengths into reps, then reps into sets.

    A rep is an unbroken swim — consecutive lengths with no rest between them. A
    set is a run of consecutive reps covering the same distance, which is how a
    practice is actually written down (4x100, not 16x25).
    """
    df = df.sort_values(["workout_id", "idx"]).copy()
    end = df["start_offset_s"] + df["duration_s"]
    prev_end = end.groupby(df["workout_id"]).shift(1)
    gap = df["start_offset_s"] - prev_end
    df["rest_before_s"] = gap.fillna(0.0).clip(lower=0.0)

    new_rep = (gap > REST_GAP_S) | gap.isna()
    df["rep_id"] = new_rep.groupby(df["workout_id"]).cumsum().astype(int)
    df["rep_lengths"] = df.groupby(["workout_id", "rep_id"])["idx"].transform("size")

    # A set is a run of consecutive reps of equal length. Detect the run breaks on
    # one row per rep, then broadcast the set number back to every length.
    reps = (df.drop_duplicates(["workout_id", "rep_id"])
              .loc[:, ["workout_id", "rep_id", "rep_lengths"]].copy())
    changed = (reps["rep_lengths"] != reps.groupby("workout_id")["rep_lengths"].shift(1))
    reps["set_id"] = changed.groupby(reps["workout_id"]).cumsum().astype(int)
    df = df.merge(reps[["workout_id", "rep_id", "set_id"]],
                  on=["workout_id", "rep_id"], how="left")
    return df.sort_values(["workout_id", "idx"]).reset_index(drop=True)


def detect_merges(df: pd.DataFrame, max_factor: int = 4,
                  tolerance: float = 0.28) -> list[Repair]:
    """Find lengths that are really N lengths fused by a missed wall turn.

    A merged record sits at a near-integer multiple of its set's median AND is an
    outlier within that set. A uniformly slow set is a drill, not a defect — so
    the set's own median, not a global threshold, is the reference. Sets too short
    to have a trustworthy median are left alone.
    """
    repairs: list[Repair] = []
    for (wid, sid), g in df.groupby(["workout_id", "set_id"]):
        if len(g) < 4:
            continue
        med = float(g["pace_s"].median())
        if med <= 0:
            continue
        for row in g.itertuples():
            ratio = row.pace_s / med
            if ratio < 1.6:
                continue                       # not long enough to be two lengths
            factor = int(round(ratio))
            if factor < 2 or factor > max_factor:
                continue
            if abs(ratio - factor) > tolerance:
                continue                       # not near-integer: a slow length
            repairs.append(Repair(wid, row.idx, row.pace_s, med, factor, "merged"))
    return repairs


# --- features --------------------------------------------------------------
def add_features(df: pd.DataFrame, hr: dict[int, np.ndarray]) -> pd.DataFrame:
    """Attach heart-rate cost and set-relative pace to each length.

    HR is expressed above each workout's own 10th percentile so that fitness,
    day-to-day variation, and warm-up drift cancel out. The read window is shifted
    by the cardiac lag, otherwise a length's HR reflects the previous one.
    """
    df = df.copy()
    costs = np.full(len(df), np.nan)
    for i, row in enumerate(df.itertuples()):
        series = hr.get(row.workout_id)
        if series is None or len(series) < 60:
            continue
        base = float(np.percentile(series, 10))
        a = int(row.start_offset_s) + HR_LAG_S
        b = int(row.start_offset_s + row.duration_s) + HR_LAG_S + 5
        seg = series[min(a, len(series) - 1):min(b, len(series))]
        if len(seg) >= 3:
            costs[i] = float(seg.mean()) - base
    df["hr_cost"] = costs

    # Statistics use the set, not the rep: a set pools every rep of the same
    # distance, which is far more lengths to estimate a median and spread from.
    grp = df.groupby(["workout_id", "set_id"])["pace_s"]
    df["set_median_pace_s"] = grp.transform("median")
    df["set_size"] = df.groupby(["workout_id", "set_id"])["idx"].transform("size")
    df["set_cv"] = grp.transform(lambda s: s.std() / s.mean() if s.mean() else 0.0)
    df["pace_rel"] = df["pace_s"] / df["set_median_pace_s"]
    df["rep_duration_s"] = df.groupby(
        ["workout_id", "rep_id"])["duration_s"].transform("sum")
    return df


# --- learning + classification ---------------------------------------------
def learn_params(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Estimate the swimmer's own reference points from their whole history.

    These are population statistics of this one swimmer, not hard-coded constants,
    which is what lets the classifier work for a swimmer whose stroke speeds don't
    follow the usual ordering.
    """
    pace = df["pace_s"].dropna()
    cost = df["hr_cost"].dropna()
    if pace.empty:
        return {}
    fast = float(pace.quantile(0.30))          # the freestyle-dominated mode
    return {
        "_global": {
            "pace_p10": float(pace.quantile(0.10)),
            "pace_p30": fast,
            "pace_p50": float(pace.median()),
            "pace_p70": float(pace.quantile(0.70)),
            "pace_p90": float(pace.quantile(0.90)),
            "cost_p33": float(cost.quantile(0.33)) if len(cost) else 0.0,
            "cost_p67": float(cost.quantile(0.67)) if len(cost) else 0.0,
            "n_obs": float(len(pace)),
        }
    }


def classify(df: pd.DataFrame, params: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Assign a class and a confidence to every length.

    Deliberately transparent rules over the two learned axes rather than an opaque
    clustering: each decision is auditable, and `undetermined` is used honestly
    wherever the evidence genuinely doesn't separate two classes.
    """
    g = params.get("_global", {})
    p30, p50, p70, p90 = (g.get("pace_p30", 24), g.get("pace_p50", 27),
                          g.get("pace_p70", 30), g.get("pace_p90", 35))
    c33, c67 = g.get("cost_p33", 0.0), g.get("cost_p67", 0.0)
    # Freestyle is the majority stroke, so it is the default hypothesis. Another
    # stroke is only called on positive evidence; where the evidence is weak the
    # answer is `undetermined` rather than a coin flip dressed up as a result.

    out, conf = [], []
    for r in df.itertuples():
        pace, cost, cv = r.pace_s, r.hr_cost, r.set_cv

        # A uniformly slow set at low cardiac cost is drill or kick work.
        if r.set_median_pace_s >= p90 and cv < 0.18:
            out.append("other"); conf.append(0.72); continue

        # The dominant fast mode is freestyle.
        if pace <= p30:
            out.append("freestyle"); conf.append(0.80); continue

        if pace <= p70:
            # Fast-to-typical: still freestyle unless unusually expensive, which
            # is butterfly's signature — fast for the effort it costs.
            if not np.isnan(cost) and cost >= c67:
                out.append("butterfly"); conf.append(0.45)
            else:
                out.append("freestyle"); conf.append(0.65)
            continue

        if pace <= p90:
            if np.isnan(cost):
                out.append("undetermined"); conf.append(0.30); continue
            if cost >= c67:
                # Slow and expensive: working hard, not travelling.
                out.append("backstroke"); conf.append(0.45)
            elif cost <= c33:
                # Slow and cheap: the glide phase of breaststroke.
                out.append("breaststroke"); conf.append(0.50)
            else:
                out.append("undetermined"); conf.append(0.33)
            continue

        out.append("other"); conf.append(0.45)

    df = df.copy()
    df["predicted"] = out
    df["confidence"] = conf
    return df


def analyze(engine: Engine, workout_id: int | None = None,
            persist: bool = True) -> AnalysisResult:
    """Run the full analysis and, by default, persist model and predictions."""
    from . import db

    df = load_lengths(engine, workout_id)
    if df.empty:
        return AnalysisResult()

    df = assign_sets(df)
    hr = load_hr(engine, sorted(df["workout_id"].unique().tolist()))
    df = add_features(df, hr)
    repairs = detect_merges(df)

    # Learn from the whole history so a single-workout run still uses good
    # estimates, then classify only what was asked for.
    full = df if workout_id is None else assign_sets(load_lengths(engine))
    if workout_id is not None:
        full = add_features(full, load_hr(engine, sorted(full["workout_id"].unique().tolist())))
    params = learn_params(full)
    df = classify(df, params)

    merged = {(r.workout_id, r.idx) for r in repairs}
    rows = [dict(workout_id=int(r.workout_id), idx=int(r.idx),
                 predicted=r.predicted, confidence=float(r.confidence),
                 method="pace_cost_v1", set_id=int(r.set_id),
                 inferred_split=int((r.workout_id, r.idx) in merged))
            for r in df.itertuples()]

    if persist:
        db.save_model_params(engine, params)
        db.save_predictions(engine, rows)

    return AnalysisResult(predictions=rows, params=params, repairs=repairs,
                          n_lengths=len(df))
