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
    """One turn-detection defect we corrected.

    Both defects are expressed the same way, as how many real pool lengths one
    record covers. A missed wall fuses two lengths into one record (factor 2); a
    spurious wall splits one length across two records (factor 0.5 each). Dividing
    the record's time by its factor recovers the per-length pace in both cases,
    which is why one field describes both defects.
    """
    workout_id: int
    idx: int
    observed_s: float
    set_median_s: float
    factor: float                   # real lengths this record covers: 2, 3, 4, or 0.5
    kind: str                       # 'merged' | 'split'


@dataclass
class AnalysisResult:
    predictions: list[dict] = field(default_factory=list)
    params: dict[str, dict[str, float]] = field(default_factory=dict)
    repairs: list[Repair] = field(default_factory=list)
    im_rounds: list["IMRound"] = field(default_factory=list)
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
            repairs.append(Repair(wid, row.idx, row.pace_s, med,
                                  float(factor), "merged"))
    return repairs


def detect_splits(df: pd.DataFrame, fast_ratio: float = 0.72,
                  pair_tolerance: float = 0.30) -> list[Repair]:
    """Find pairs of records that are really one length cut in two.

    The mirror image of a merge: instead of missing a wall, the sensor invents
    one, and a single length arrives as two impossibly fast records. One fast
    record on its own is just a fast length, so the evidence required is a PAIR —
    adjacent, with no rest between them (a swimmer cannot rest mid-length), each
    implausibly fast against the set's median, and together summing back to about
    one normal length. That conjunction is what a real sprint does not produce.
    """
    repairs: list[Repair] = []
    for (wid, sid), g in df.groupby(["workout_id", "set_id"]):
        if len(g) < 4:
            continue
        med = float(g["pace_s"].median())
        if med <= 0:
            continue
        rows = list(g.sort_values("idx").itertuples())
        i = 0
        while i < len(rows) - 1:
            a, b = rows[i], rows[i + 1]
            pair_ok = (
                b.idx == a.idx + 1                      # adjacent records
                and a.rep_id == b.rep_id                # no rest between them
                and a.pace_s / med < fast_ratio
                and b.pace_s / med < fast_ratio
                and abs((a.pace_s + b.pace_s) / med - 1.0) <= pair_tolerance
            )
            if pair_ok:
                for row in (a, b):
                    repairs.append(Repair(wid, row.idx, row.pace_s, med, 0.5, "split"))
                i += 2                                  # don't reuse b in a new pair
            else:
                i += 1
    return repairs


def detect_repairs(df: pd.DataFrame) -> list[Repair]:
    """Every turn-detection defect we can identify, of either kind."""
    return detect_merges(df) + detect_splits(df)


def apply_repairs(df: pd.DataFrame, repairs: list[Repair]) -> pd.DataFrame:
    """Correct the pace of repaired records, then restate the set statistics.

    Detecting a defect is only half the job: until the correction is applied, a
    merged record still carries a doubled time, still falls past the slow end of
    the distribution, and is still classified on a pace no swimmer swam. So the
    record's pace is divided by the number of lengths it covers, and every
    set-level statistic derived from pace is recomputed from the corrected values.

    `pace_s` afterwards always means "seconds per real pool length". The observed
    figure is preserved as `pace_observed_s` — Polar's record is data, and the
    correction is inference, so the two are never conflated.
    """
    df = df.copy()
    df["pace_observed_s"] = df["pace_s"]
    df["length_factor"] = 1.0
    if repairs:
        factors = {(r.workout_id, r.idx): r.factor for r in repairs}
        key = list(zip(df["workout_id"], df["idx"]))
        df["length_factor"] = [factors.get(k, 1.0) for k in key]
        df["pace_s"] = df["pace_observed_s"] / df["length_factor"]
    return _set_stats(df)


# --- features --------------------------------------------------------------
def add_features(df: pd.DataFrame, hr: dict[int, np.ndarray]) -> pd.DataFrame:
    """Attach heart-rate cost and set-relative pace to each length.

    HR is expressed above each workout's own 10th percentile so that fitness,
    day-to-day variation, and warm-up drift cancel out. The read window is shifted
    by the cardiac lag, otherwise a length's HR reflects the previous one.
    """
    df = df.copy()
    costs = np.full(len(df), np.nan)
    absolute = np.full(len(df), np.nan)
    for i, row in enumerate(df.itertuples()):
        series = hr.get(row.workout_id)
        if series is None or len(series) < 60:
            continue
        base = float(np.percentile(series, 10))
        a = int(row.start_offset_s) + HR_LAG_S
        b = int(row.start_offset_s + row.duration_s) + HR_LAG_S + 5
        seg = series[min(a, len(series) - 1):min(b, len(series))]
        if len(seg) >= 3:
            mean = float(seg.mean())
            costs[i] = mean - base          # relative: effort within this session
            absolute[i] = mean              # absolute: needed for HR zoning
    df["hr_cost"] = costs
    df["hr_abs"] = absolute

    return _set_stats(df)


def _set_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Set-level summaries of pace, rest and size.

    Statistics use the set, not the rep: a set pools every rep of the same
    distance, which is far more lengths to estimate a median and spread from.
    Recomputed after any repair, since a corrected pace changes every one of them.
    """
    df = df.copy()
    grp = df.groupby(["workout_id", "set_id"])["pace_s"]
    df["set_median_pace_s"] = grp.transform("median")
    df["set_size"] = df.groupby(["workout_id", "set_id"])["idx"].transform("size")
    df["set_cv"] = grp.transform(lambda s: s.std() / s.mean() if s.mean() else 0.0)
    df["pace_rel"] = df["pace_s"] / df["set_median_pace_s"]
    df["rep_duration_s"] = df.groupby(
        ["workout_id", "rep_id"])["duration_s"].transform("sum")

    # Rest is a property of the boundary between reps, so it is read once per rep
    # — from the first length of each — and not averaged over the zeros that sit
    # between the lengths inside a rep.
    rep_first = df.drop_duplicates(["workout_id", "rep_id"])
    set_rest = (rep_first.groupby(["workout_id", "set_id"])["rest_before_s"]
                .median().rename("set_rest_s"))
    # Recomputed, not merged alongside: this runs a second time after a repair,
    # and a merge onto an existing column would silently produce _x/_y suffixes.
    df = df.drop(columns=["set_rest_s"], errors="ignore")
    df = df.merge(set_rest, on=["workout_id", "set_id"], how="left")
    df["set_rest_s"] = df["set_rest_s"].fillna(0.0)
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

    # Rest is learned per SET, not per length: a set is one decision the swimmer
    # made about how much recovery the work needed, and repeating it once per
    # length would weight long sets out of proportion.
    sets = (df.drop_duplicates(["workout_id", "set_id"])["set_rest_s"].dropna()
            if "set_rest_s" in df.columns else pd.Series(dtype=float))
    return {
        "_global": {
            "pace_p10": float(pace.quantile(0.10)),
            "pace_p30": fast,
            "pace_p50": float(pace.median()),
            "pace_p70": float(pace.quantile(0.70)),
            "pace_p90": float(pace.quantile(0.90)),
            "cost_p33": float(cost.quantile(0.33)) if len(cost) else 0.0,
            "cost_p67": float(cost.quantile(0.67)) if len(cost) else 0.0,
            "rest_p50": float(sets.quantile(0.50)) if len(sets) else 0.0,
            "rest_p80": float(sets.quantile(0.80)) if len(sets) else 0.0,
            "n_obs": float(len(pace)),
        }
    }


def classify(df: pd.DataFrame, params: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Assign a class and a confidence to every length.

    Deliberately transparent rules over learned axes rather than an opaque
    clustering: each decision is auditable, and `undetermined` is used honestly
    wherever the evidence genuinely doesn't separate two classes.

    Three axes, not two. Pace and heart-rate cost separate most of the field, but
    they leave butterfly and backstroke overlapping — both are slow and expensive.
    **Rest** is what parts them. Rest is a decision the swimmer makes about how
    much recovery the work needs, so a set that bought unusually long rest was
    unusually hard for its distance, and butterfly buys more rest than anything
    else at the same pace. Set size is used differently: it does not name a
    stroke, it says how much to trust the set statistics the other rules rest on.
    """
    g = params.get("_global", {})
    p30, p50, p70, p90 = (g.get("pace_p30", 24), g.get("pace_p50", 27),
                          g.get("pace_p70", 30), g.get("pace_p90", 35))
    c33, c67 = g.get("cost_p33", 0.0), g.get("cost_p67", 0.0)
    r50, r80 = g.get("rest_p50", 0.0), g.get("rest_p80", 0.0)
    have_rest = r80 > 0
    # Freestyle is the majority stroke, so it is the default hypothesis. Another
    # stroke is only called on positive evidence; where the evidence is weak the
    # answer is `undetermined` rather than a coin flip dressed up as a result.

    out, conf = [], []
    for r in df.itertuples():
        pace, cost, cv = r.pace_s, r.hr_cost, r.set_cv
        rest = getattr(r, "set_rest_s", 0.0)
        size = getattr(r, "set_size", 0)
        long_rest = have_rest and rest >= r80
        # A set of one or two lengths has no usable median or spread, so every
        # rule below that reads a set statistic is reading noise. Say so in the
        # confidence rather than pretending the call is as good as any other.
        trust = 1.0 if size >= 4 else 0.8

        def call(name: str, c: float) -> None:
            out.append(name)
            conf.append(round(c * trust, 3))

        # Drill and kick: a set that is uniformly slow, cheap, and taken on short
        # rest. The rest condition matters — a uniformly slow set that bought long
        # rest is a hard stroke set, not drill, and the old rule called it drill.
        if (size >= 4 and r.set_median_pace_s >= p90 and cv < 0.18
                and not long_rest):
            call("other", 0.74); continue

        # The dominant fast mode is freestyle.
        if pace <= p30:
            call("freestyle", 0.80); continue

        if pace <= p70:
            # Fast-to-typical: still freestyle unless it was unusually expensive,
            # which is butterfly's signature — fast for what it costs. Long rest
            # on top of that cost is the confirming evidence.
            if not np.isnan(cost) and cost >= c67:
                call("butterfly", 0.58 if long_rest else 0.45)
            else:
                call("freestyle", 0.65)
            continue

        if pace <= p90:
            if np.isnan(cost):
                # No heart rate. Rest alone cannot name a stroke, but a slow set
                # on short rest is far more likely aerobic work than a race stroke.
                call("other" if have_rest and rest <= r50 else "undetermined", 0.30)
                continue
            if cost >= c67:
                # Slow and expensive: working hard without travelling. Backstroke
                # and butterfly both look like this; the one that bought the most
                # rest is the one that cost the most to swim.
                call("butterfly", 0.42) if long_rest else call("backstroke", 0.48)
            elif cost <= c33:
                # Slow and cheap: the glide phase of breaststroke.
                call("breaststroke", 0.50)
            else:
                call("undetermined", 0.33)
            continue

        # Slower than anything else in the history. Drill, kick, or recovery —
        # unless it bought long rest, in which case it was hard, not easy.
        call("undetermined" if long_rest else "other", 0.45)

    df = df.copy()
    df["predicted"] = out
    df["confidence"] = conf
    return df


# --- individual medley ------------------------------------------------------
# An IM is the one place where stroke order is known rather than inferred: fly,
# back, breast, free, always in that order. That turns it into a STRUCTURAL
# signal, which is far more reliable here than the per-length classifier — we do
# not need to name each stroke, only to recognise the repeating four-part shape.
IM_ORDER = ("butterfly", "backstroke", "breaststroke", "freestyle")
IM_MIN_ROUNDS = 2          # one four-part round alone is not evidence; see below
IM_SEPARATION = 1.5        # positions must differ more than they wobble
IM_MIN_SPREAD = 0.08       # and differ by enough to not be a flat freestyle set
# A medley covers all four strokes equally, which in a 25 yd pool makes exactly
# three distances. Four 75s share the period-4 shape but total 300, which is not
# an event anyone swims, so requiring a real distance rejects a whole class of
# coincidental matches at no cost.
IM_DISTANCES_YD = (100, 200, 400)


@dataclass
class IMRound:
    """One complete fly-back-breast-free cycle."""
    workout_id: int
    set_id: int
    round_no: int
    yards: int
    seconds: float                  # swim time; excludes rest inside a broken round
    continuous: bool                # swum unbroken, or four reps off the wall
    idxs: list[int] = field(default_factory=list)
    splits_s: list[float] = field(default_factory=list)


def _im_signature(values: np.ndarray) -> bool:
    """Does a rounds x 4 matrix of times look like a repeated medley?

    Two things have to hold at once. The four positions must be separated —
    consistently different from each other across rounds, rather than wobbling
    around one number, which is what tells a medley from a set of the same stroke.
    And the fourth position must be the fastest, because freestyle comes last in
    an IM and is the fastest stroke for anyone who would be swimming one.

    Deliberately NOT checked: that the middle two are in any particular order.
    Whether backstroke or breaststroke is the slower of them is a fact about the
    individual swimmer, and assuming it is exactly the assumption this project
    refuses to make elsewhere.
    """
    if values.shape[0] < IM_MIN_ROUNDS:
        return False
    means = values.mean(axis=0)
    if means.min() <= 0 or means.argmin() != 3:
        return False                              # freestyle is not the fastest leg
    if (means.max() - means.min()) / means.mean() < IM_MIN_SPREAD:
        return False                              # four legs too alike to be strokes
    within = float(values.std(axis=0, ddof=0).mean())
    between = float(means.std(ddof=0))
    return between >= IM_SEPARATION * max(within, 1e-6)


def detect_im(df: pd.DataFrame) -> list[IMRound]:
    """Find medley rounds, continuous or broken into four reps.

    Both shapes a swimmer actually writes down are recognised: `4x100 IM`, where
    each rep is an unbroken medley, and `16x25 IM`, where the medley is broken
    across four reps off the wall. In both cases the evidence is the same repeated
    four-part structure, read at the level the repetition happens.

    A SINGLE four-part round is not claimed. Four lengths that happen to descend
    are indistinguishable from one medley on the evidence available, so at least
    two rounds are required and a one-off IM is left unlabelled rather than
    guessed at.
    """
    out: list[IMRound] = []
    if df.empty:
        return out

    for (wid, sid), g in df.groupby(["workout_id", "set_id"]):
        g = g.sort_values("idx")
        pool_yd = float(g["pool_m"].iloc[0]) / 0.9144
        reps = list(g.groupby("rep_id", sort=True))

        # Continuous: every rep is itself a medley, so the four legs are lengths.
        rep_lengths = {len(r) for _, r in reps}
        uniform = len(rep_lengths) == 1
        if uniform and len(reps) >= IM_MIN_ROUNDS:
            n = next(iter(rep_lengths))
            yards = int(round(n * pool_yd))
            if n % 4 == 0 and yards in IM_DISTANCES_YD:
                legs = n // 4          # lengths per stroke, 1 for a 100 in a 25 pool
                matrix = np.array([
                    r["duration_s"].to_numpy().reshape(4, legs).sum(axis=1)
                    for _, r in reps])
                if _im_signature(matrix):
                    for i, (_, r) in enumerate(reps, start=1):
                        out.append(IMRound(
                            workout_id=int(wid), set_id=int(sid), round_no=i,
                            yards=yards,
                            seconds=float(r["duration_s"].sum()), continuous=True,
                            idxs=[int(x) for x in r["idx"]],
                            splits_s=[float(x) for x in matrix[i - 1]]))
                    continue

        # Broken: each rep is one leg, so four consecutive reps make a round.
        if uniform and len(reps) >= 4 * IM_MIN_ROUNDS:
            times = np.array([float(r["duration_s"].sum()) for _, r in reps])
            n_rounds = len(times) // 4
            matrix = times[:n_rounds * 4].reshape(n_rounds, 4)
            rep_yd = int(round(len(reps[0][1]) * pool_yd))
            if rep_yd * 4 in IM_DISTANCES_YD and _im_signature(matrix):
                for i in range(n_rounds):
                    members = reps[i * 4:(i + 1) * 4]
                    out.append(IMRound(
                        workout_id=int(wid), set_id=int(sid), round_no=i + 1,
                        yards=rep_yd * 4, seconds=float(matrix[i].sum()),
                        continuous=False,
                        idxs=[int(x) for _, r in members for x in r["idx"]],
                        splits_s=[float(x) for x in matrix[i]]))
    return out


def label_im(df: pd.DataFrame, rounds: list[IMRound]) -> pd.DataFrame:
    """Overwrite predictions inside a medley round with the known stroke order.

    Where a round is identified the strokes are no longer inferred — the order is
    what makes it a medley — so these labels carry higher confidence than anything
    the pace/cost rules produce.
    """
    df = df.copy()
    df["im_continuous"] = False
    if not rounds:
        return df
    labels: dict[tuple[int, int], str] = {}
    continuous: set[tuple[int, int]] = set()
    for rnd in rounds:
        per_leg = max(1, len(rnd.idxs) // 4)
        for pos, stroke in enumerate(IM_ORDER):
            for idx in rnd.idxs[pos * per_leg:(pos + 1) * per_leg]:
                labels[(rnd.workout_id, idx)] = stroke
        if rnd.continuous:
            continuous.update((rnd.workout_id, i) for i in rnd.idxs)
    key = list(zip(df["workout_id"], df["idx"]))
    hit = [k in labels for k in key]
    df.loc[hit, "predicted"] = [labels[k] for k in key if k in labels]
    df.loc[hit, "confidence"] = 0.85
    # A rep that is a whole medley is not a rep of any one stroke, so it must not
    # compete for the 100 backstroke best just because backstroke was its mode.
    # A leg of a BROKEN medley is a genuine 25 off the wall and stays eligible.
    df["im_continuous"] = [k in continuous for k in key]
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

    # Repair first, then classify. An uncorrected merge carries a doubled time and
    # would be classified on a pace nobody swam, which is the whole reason the
    # correction has to land before the classifier sees the data.
    repairs = detect_repairs(df)
    df = apply_repairs(df, repairs)

    # Learn from the whole history so a single-workout run still uses good
    # estimates, then classify only what was asked for.
    if workout_id is None:
        full = df
    else:
        full = assign_sets(load_lengths(engine))
        full = add_features(
            full, load_hr(engine, sorted(full["workout_id"].unique().tolist())))
        full = apply_repairs(full, detect_repairs(full))
    params = learn_params(full)
    df = classify(df, params)

    # Medley rounds are recognised structurally, so their strokes are known rather
    # than inferred and they overwrite whatever the pace/cost rules guessed.
    im = detect_im(df)
    df = label_im(df, im)

    kinds = {(r.workout_id, r.idx): r.kind for r in repairs}
    rows = [dict(workout_id=int(r.workout_id), idx=int(r.idx),
                 predicted=r.predicted, confidence=float(r.confidence),
                 method="pace_cost_rest_v2", set_id=int(r.set_id),
                 length_factor=float(r.length_factor),
                 repair_kind=kinds.get((r.workout_id, r.idx)),
                 inferred_split=int(kinds.get((r.workout_id, r.idx)) == "merged"))
            for r in df.itertuples()]

    if persist:
        db.save_model_params(engine, params)
        db.save_predictions(engine, rows)

    return AnalysisResult(predictions=rows, params=params, repairs=repairs,
                          im_rounds=im, n_lengths=len(df))
