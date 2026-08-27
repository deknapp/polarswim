"""Swimmer-calibrated metrics: heart-rate zone, relative speed, and personal bests.

Every reference here is derived from this swimmer's own database rather than from a
formula or a population table, which matters more in swimming than in most sports:

  * **Heart rate.** Maximum heart rate in the water runs roughly 10-13 bpm below a
    land-based maximum — the horizontal position, the cooling water, and the smaller
    working muscle mass all suppress it. So an age formula, or a maximum measured
    running, would put every zone boundary too high. The reference used is the
    highest heart rate actually observed while swimming.

  * **Speed.** A rep is ranked against this swimmer's other reps of the **same
    distance** — "this 50 was faster than 62% of your 50s". Ranking within an
    inferred stroke was tried first and rejected as circular: the classifier assigns
    fast lengths to freestyle, so a percentile within freestyle largely re-expresses
    the classifier's own threshold, and it collapsed every slow length to 0%.
    Distance is measured rather than inferred, so it carries no such feedback.

  * **Personal bests.** The fastest recorded time for a given distance and stroke.
    Two caveats are built in. Stroke labels are inferred rather than measured, so a
    best inherits that uncertainty. And the sensor's turn detection occasionally
    splits one length into two, producing an impossibly fast record — an unfiltered
    table showed a 25 yd "best" of 13.6 s against a 26 s median, which is a hardware
    artifact, not a swim. Reps faster than `PLAUSIBLE_FLOOR_RATIO` of the swimmer's
    median pace are therefore excluded from bests and counted separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from .models import hr_samples, workouts

# Five-zone model as a fraction of maximum heart rate. Standard boundaries; what
# makes them swimming-specific is the maximum they are applied to.
ZONE_BANDS = [
    ("Z1", 0.00, 0.60, "recovery",   "#6b7280"),
    ("Z2", 0.60, 0.70, "endurance",  "#4aa3ff"),
    ("Z3", 0.70, 0.80, "tempo",      "#3ddc84"),
    ("Z4", 0.80, 0.90, "threshold",  "#f0a848"),
    ("Z5", 0.90, 1.01, "max",        "#f0645a"),
]

# Colour ramp for relative speed, slowest to fastest.
SPEED_COLORS = [
    (0,  "#6b7280"), (20, "#8b949e"), (40, "#4aa3ff"),
    (60, "#3ddc84"), (80, "#f0a848"), (95, "#f0645a"),
]

MIN_OBSERVATIONS = 30      # below this a percentile is not worth reporting

# A rep faster than this fraction of the swimmer's median pace is treated as a
# turn-detection artifact rather than a swim. Nobody halves their own pace.
PLAUSIBLE_FLOOR_RATIO = 0.65


@dataclass
class SwimmerReference:
    """Everything derived from the swimmer's full history, computed once."""

    hr_max: int
    pace_by_stroke: dict[str, np.ndarray] = field(default_factory=dict)
    rep_times_by_distance: dict[int, np.ndarray] = field(default_factory=dict)
    best_rep: dict[tuple[int, str], dict] = field(default_factory=dict)
    median_pace_s: float = 0.0
    implausible_reps: int = 0      # excluded from bests as sensor artifacts

    # --- heart rate ---------------------------------------------------------
    def zone_bounds(self) -> list[dict]:
        """The zone table, in bpm, for display as a key."""
        out = []
        for name, lo, hi, label, colour in ZONE_BANDS:
            out.append({
                "zone": name, "label": label, "color": colour,
                "low": int(round(lo * self.hr_max)),
                "high": int(round(min(hi, 1.0) * self.hr_max)),
            })
        return out

    def hr_zone(self, bpm: float | None) -> dict | None:
        """Which zone a heart rate falls in."""
        if bpm is None or not np.isfinite(bpm) or bpm <= 0:
            return None
        frac = bpm / self.hr_max
        for name, lo, hi, label, colour in ZONE_BANDS:
            if lo <= frac < hi:
                return {"zone": name, "label": label, "color": colour,
                        "pct_max": round(100 * frac)}
        return {"zone": "Z5", "label": "max", "color": ZONE_BANDS[-1][4],
                "pct_max": round(100 * frac)}

    # --- speed --------------------------------------------------------------
    def speed_percentile(self, yards: int, seconds: float) -> dict | None:
        """How a rep ranks against the swimmer's other reps of the same distance.

        Reported as "faster than N% of your Xs", so higher is better even though
        the underlying time is lower-is-faster. Returns None when that distance has
        too little history to rank against, rather than inventing a number.
        """
        history = self.rep_times_by_distance.get(int(yards))
        if (history is None or len(history) < MIN_OBSERVATIONS
                or not np.isfinite(seconds)):
            return None
        pct = float((history > seconds).mean() * 100.0)    # invert: low time = fast
        colour = SPEED_COLORS[0][1]
        for threshold, c in SPEED_COLORS:
            if pct >= threshold:
                colour = c
        return {"percentile": round(pct), "color": colour,
                "n": int(len(history)), "distance": int(yards)}

    # --- personal bests -----------------------------------------------------
    def check_pr(self, yards: int, stroke: str, seconds: float,
                 workout_id: int) -> bool:
        """True when this workout holds the fastest time at this distance/stroke."""
        best = self.best_rep.get((yards, stroke))
        return bool(best and best["workout_id"] == workout_id
                    and abs(best["seconds"] - seconds) < 0.5)


def build_reference(engine: Engine, lengths_df: pd.DataFrame) -> SwimmerReference:
    """Derive the swimmer's reference points from their whole history.

    `lengths_df` must already carry `predicted`, `rep_id` and `pace_s` — i.e. the
    output of the analysis, for every workout, not just the one being viewed.
    """
    with engine.connect() as c:
        hr_max = c.execute(sa.select(sa.func.max(workouts.c.max_hr))).scalar()
        if not hr_max:
            hr_max = c.execute(sa.select(sa.func.max(hr_samples.c.hr))).scalar()
    hr_max = int(hr_max or 180)

    ref = SwimmerReference(hr_max=hr_max)
    if lengths_df is None or lengths_df.empty:
        return ref

    for stroke, g in lengths_df.groupby("predicted"):
        paces = g["pace_s"].dropna().to_numpy()
        if len(paces):
            ref.pace_by_stroke[str(stroke)] = np.sort(paces)

    ref.median_pace_s = float(lengths_df["pace_s"].median())

    # Fastest rep per (distance, stroke). A rep's stroke is the majority label of
    # its lengths, and its time is the sum of them.
    reps = (lengths_df.groupby(["workout_id", "rep_id"])
            .agg(lengths=("idx", "size"), seconds=("duration_s", "sum"),
                 pool_m=("pool_m", "first"),
                 stroke=("predicted", lambda s: s.mode().iloc[0] if len(s.mode()) else "undetermined"),
                 start=("start_time", "first"))
            .reset_index())
    reps["yards"] = (reps["lengths"] * reps["pool_m"] / 0.9144).round().astype(int)

    # Drop physically impossible reps before ranking. A split length produces a
    # record no swimmer could hold, and it would win every best it touched.
    reps["pace_per_25"] = reps["seconds"] / reps["lengths"] * (
        22.86 / reps["pool_m"])
    floor = ref.median_pace_s * PLAUSIBLE_FLOOR_RATIO
    plausible = reps["pace_per_25"] >= floor
    ref.implausible_reps = int((~plausible).sum())
    reps = reps[plausible]

    for yards, g in reps.groupby("yards"):
        times = np.sort(g["seconds"].to_numpy())
        if len(times):
            ref.rep_times_by_distance[int(yards)] = times

    for (yards, stroke), g in reps.groupby(["yards", "stroke"]):
        row = g.loc[g["seconds"].idxmin()]
        ref.best_rep[(int(yards), str(stroke))] = {
            "seconds": float(row["seconds"]),
            "workout_id": int(row["workout_id"]),
            "date": str(row["start"])[:10],
        }
    return ref
