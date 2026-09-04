"""Swimmer-calibrated metrics: heart-rate zone, relative speed, and personal bests.

Every reference here is derived from this swimmer's own database rather than from a
formula or a population table, which matters more in swimming than in most sports:

  * **Heart rate.** Maximum heart rate in the water runs roughly 10-13 bpm below a
    land-based maximum — the horizontal position, the cooling water, and the smaller
    working muscle mass all suppress it. So an age formula, or a maximum measured
    running, would put every zone boundary too high. The reference used is the
    highest heart rate actually observed while swimming.

  * **Speed.** A rep is ranked against this swimmer's other reps of the same
    **distance and stroke** — "this backstroke 50 was faster than 62% of your
    backstroke 50s" — or it is not ranked at all. Distance alone was tried first
    and is wrong: it ranks a backstroke 100 against a field of mostly freestyle
    100s. It survived for a while as a fallback for thin pairs, which was worse
    than dropping it, because a wrong number is harder to discount than a blank:
    a 30 s backstroke 25 came back at 49% purely by being measured against 401
    freestyle reps, while the same 30 s read 0% wherever real backstroke history
    existed. The fallback is gone.

    What remains is a caveat that cannot be engineered away here, only disclosed.
    The stroke labels are inferred FROM PACE, so each stroke's population is
    partly a pace bin: this swimmer's freestyle 25s run 16.8-28.8 s and their
    breaststroke 25s 29.6-34.4 s, ranges that barely touch. A rep at the slow edge
    of its own label reads near 0% whatever it was, and an identical 30 s 25 is
    0% as freestyle and 84% as breaststroke — not because the swims differ, but
    because the labels do. Reps therefore carry `edge`, true where the rep sits at
    or past the end of the range its own label spans, and the only real fix is
    ground truth: corrections saved through the editor train the model on
    something other than pace and break the circle.

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

# Five-zone model as a fraction of heart-rate RESERVE — the Karvonen method —
# rather than of maximum heart rate.
#
# The percentages below are the conventional five-zone boundaries; what changed
# is the span they divide. Taking them as a fraction of maximum measures from a
# heart rate of zero, which nobody has, so the bottom zone covers everything up
# to 60% of max and is unreachable in the water: it put an easy swim in Z3 and a
# steady aerobic set in Z4, reading a whole zone hot. Measuring from resting
# heart rate to maximum divides the range a swimmer actually occupies, so the
# bottom zone becomes real easy swimming and the top one becomes genuinely hard.
#
# This is also the scheme most training plans and coaches use, which matters for
# a number anyone might compare against their own.
ZONE_BANDS = [
    ("Z1", 0.00, 0.60, "easy",       "#6b7280"),
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

# A distance/stroke pair needs this many reps before a percentile off it means
# anything. Below it the rep is left unranked: there is no distance-only fallback,
# because ranking one stroke against a field of another is a wrong answer rather
# than a rough one.
MIN_OBSERVATIONS_STROKE = 15

# The distances actually raced in a 25 yd pool. A practice throws off bests at
# every distance a set happens to be written at — 75s, 125s, 150s — and burying
# the 100 free among them makes the table unreadable. These are the ones that
# mean something; the rest stay available behind a toggle rather than being
# discarded, since they are still this swimmer's own fastest times.
COMPETITIVE_YARDS = {
    "freestyle":    (25, 50, 100, 200, 500, 1000, 1650),
    "backstroke":   (25, 50, 100, 200),
    "breaststroke": (25, 50, 100, 200),
    "butterfly":    (25, 50, 100, 200),
    "IM":           (100, 200, 400),
}
# 25 is not an event, but it is the pool's own unit and this swimmer's most
# common rep, so leaving it out would empty the table it is meant to clarify.
DEFAULT_COMPETITIVE = (25, 50, 100, 200)


def is_competitive(yards: int, stroke: str) -> bool:
    """Is this a distance the stroke is actually raced at?"""
    return int(yards) in COMPETITIVE_YARDS.get(stroke, DEFAULT_COMPETITIVE)

# Edwards TRIMP zone weights, kept for the time-in-zone breakdown.
ZONE_WEIGHTS = {"Z1": 1, "Z2": 2, "Z3": 3, "Z4": 4, "Z5": 5}

# Banister TRIMP weights each sample by heart-rate reserve, exponentially, so a
# minute near threshold counts for far more than a minute in recovery. Preferred
# over Edwards' linear 1-5 zone weights, and over integrating heart rate directly.
BANISTER_A, BANISTER_B = 0.64, 1.92

# A rep faster than this fraction of the swimmer's median pace is treated as a
# turn-detection artifact rather than a swim. Nobody halves their own pace.
PLAUSIBLE_FLOOR_RATIO = 0.65


@dataclass
class SwimmerReference:
    """Everything derived from the swimmer's full history, computed once."""

    hr_max: int
    pace_by_stroke: dict[str, np.ndarray] = field(default_factory=dict)
    rep_times_by_distance: dict[int, np.ndarray] = field(default_factory=dict)
    rep_times_by_distance_stroke: dict[tuple[int, str], np.ndarray] = field(
        default_factory=dict)
    hr_rest: int = 60
    trimp_by_workout: dict[int, float] = field(default_factory=dict)
    trimp_sorted: np.ndarray = field(default_factory=lambda: np.array([]))
    intensity_by_workout: dict[int, float] = field(default_factory=dict)
    intensity_sorted: np.ndarray = field(default_factory=lambda: np.array([]))
    best_rep: dict[tuple[int, str], dict] = field(default_factory=dict)
    median_pace_s: float = 0.0
    implausible_reps: int = 0      # excluded from bests as sensor artifacts
    # Medley rounds are ranked separately from single-stroke reps: a 100 IM and a
    # 100 free are different events, and pooling them would rank neither.
    im_times_by_distance: dict[int, np.ndarray] = field(default_factory=dict)
    best_im: dict[int, dict] = field(default_factory=dict)

    # --- heart rate ---------------------------------------------------------
    def _bpm_at(self, fraction: float) -> int:
        """The heart rate at a given fraction of reserve, in bpm."""
        span = max(1, self.hr_max - self.hr_rest)
        return int(round(self.hr_rest + fraction * span))

    def zone_bounds(self) -> list[dict]:
        """The zone table, in bpm, for display as a key."""
        return [{
            "zone": name, "label": label, "color": colour,
            "low": self._bpm_at(lo),
            "high": min(self._bpm_at(hi), self.hr_max),
        } for name, lo, hi, label, colour in ZONE_BANDS]

    def hr_zone(self, bpm: float | None) -> dict | None:
        """Which zone a heart rate falls in, by fraction of reserve.

        `pct_max` is still reported as a fraction of maximum, because that is the
        number people quote to each other; the zone it lands in is what changed.
        """
        if bpm is None or not np.isfinite(bpm) or bpm <= 0:
            return None
        span = max(1, self.hr_max - self.hr_rest)
        frac = (bpm - self.hr_rest) / span
        pct_max = round(100 * bpm / self.hr_max)
        for name, lo, hi, label, colour in ZONE_BANDS:
            if lo <= frac < hi:
                return {"zone": name, "label": label, "color": colour,
                        "pct_max": pct_max, "pct_reserve": round(100 * frac)}
        # Below resting, or above maximum: both ends clamp rather than vanish.
        band = ZONE_BANDS[0] if frac < 0 else ZONE_BANDS[-1]
        return {"zone": band[0], "label": band[3], "color": band[4],
                "pct_max": pct_max, "pct_reserve": round(100 * frac)}

    # --- speed --------------------------------------------------------------
    def speed_percentile(self, yards: int, seconds: float,
                         stroke: str | None = None) -> dict | None:
        """How a rep ranks against this swimmer's own reps of the same event.

        Same distance AND same stroke, or nothing. There used to be a fallback to
        distance alone when the pair was thin, and it was worse than useless: at
        25 yd the distance-only field is 401 freestyle reps against 9 backstroke,
        so a 30 s backstroke 25 came back at 49% — a respectable-looking number
        produced entirely by ranking backstroke against freestyle. The same 30 s
        against actual backstroke history is a different question with a different
        answer, and where there is not enough history to ask it, the honest reply
        is no number rather than a flattering one.

        Higher is better, even though the underlying time is lower-is-faster.

        The unavoidable caveat, which the caller must surface: the stroke labels
        are inferred FROM PACE, so each stroke's population is partly a pace bin.
        This swimmer's freestyle 25s top out at 28.8 s not because they never swim
        a slower 25 but because a slower one gets called something else. A rep at
        the slow edge of its own stroke will therefore read near 0% no matter how
        it was swum, and only real corrections can break that circle.
        """
        if not np.isfinite(seconds) or not stroke:
            return None
        history = self.rep_times_by_distance_stroke.get((int(yards), stroke))
        if history is None or len(history) < MIN_OBSERVATIONS_STROKE:
            return None

        pct = float((history > seconds).mean() * 100.0)    # invert: low time = fast
        return {"percentile": round(pct), "color": self._pct_colour(pct),
                "n": int(len(history)), "distance": int(yards), "basis": "stroke",
                "stroke": stroke,
                # True when the rep sits outside the range the label itself
                # spans, which is where the pace-binning above bites hardest.
                "edge": bool(seconds >= history.max() or seconds <= history.min())}

    def im_percentile(self, yards: int, seconds: float) -> dict | None:
        """How a medley round ranks against this swimmer's other medleys.

        A 100 IM must not be ranked against 100 frees. The generic path would do
        exactly that — there is no (100, "IM") entry in the per-stroke history,
        so it falls back to distance alone, where the field is overwhelmingly
        freestyle and every medley lands near the bottom regardless of how well
        it was swum.
        """
        history = self.im_times_by_distance.get(int(yards))
        if history is None or len(history) < 3 or not np.isfinite(seconds):
            return None
        pct = float((history > seconds).mean() * 100.0)
        return {"percentile": round(pct), "color": self._pct_colour(pct),
                "n": int(len(history)), "distance": int(yards), "basis": "medley",
                "stroke": "IM"}

    # --- training load ------------------------------------------------------
    def hr_reserve(self, hr_series: np.ndarray) -> np.ndarray:
        """Fraction of heart-rate reserve used, clipped to a sane range."""
        span = max(1.0, float(self.hr_max - self.hr_rest))
        return np.clip((np.asarray(hr_series, dtype=float) - self.hr_rest) / span,
                       0.0, 1.2)

    def trimp(self, hr_series: np.ndarray, interval_s: float = 1.0) -> float:
        """Banister TRIMP — accumulated load, weighting intensity exponentially.

        Each sample contributes `x · a · e^(b·x)` where x is the fraction of heart
        rate reserve in use. The exponential is what separates this from a plain
        integral of heart rate: a minute at threshold is worth several minutes of
        recovery swimming, which is what makes the number track how a session
        actually felt.
        """
        if hr_series is None or len(hr_series) == 0:
            return 0.0
        x = self.hr_reserve(hr_series)
        return float(np.sum(x * BANISTER_A * np.exp(BANISTER_B * x))
                     * interval_s / 60.0)

    def edwards_trimp(self, hr_series: np.ndarray, interval_s: float = 1.0) -> float:
        """Zone-weighted load, kept for comparison against the Banister figure."""
        if hr_series is None or len(hr_series) == 0:
            return 0.0
        total = 0.0
        for name, lo, hi, _label, _c in ZONE_BANDS:
            in_zone = ((hr_series >= lo * self.hr_max)
                       & (hr_series < hi * self.hr_max)).sum()
            total += (in_zone * interval_s / 60.0) * ZONE_WEIGHTS[name]
        return float(total)

    @staticmethod
    def _pct_colour(pct: float) -> str:
        colour = SPEED_COLORS[0][1]
        for threshold, c in SPEED_COLORS:
            if pct >= threshold:
                colour = c
        return colour

    def effort_score(self, workout_id: int) -> dict | None:
        """Two 0-100 scores, because "hard" is two different questions.

        `load` is accumulated stress, so it grows with duration — a three-hour swim
        outranks a sharp hour, correctly, because it is more total work. `intensity`
        is load per minute, which is duration-independent and answers "how hard was
        this while it lasted". The two rank the database quite differently, and
        reporting only one of them would misrepresent half the sessions.

        Both are percentiles against this swimmer's own history, so they stay
        meaningful as fitness changes.
        """
        load = self.trimp_by_workout.get(int(workout_id))
        if load is None or len(self.trimp_sorted) < 3:
            return None
        load_pct = float((self.trimp_sorted <= load).mean() * 100.0)

        intensity = self.intensity_by_workout.get(int(workout_id))
        out = {"score": round(load_pct), "trimp": round(load, 1),
               "color": self._pct_colour(load_pct), "n": int(len(self.trimp_sorted))}
        if intensity is not None and len(self.intensity_sorted) >= 3:
            i_pct = float((self.intensity_sorted <= intensity).mean() * 100.0)
            out.update(intensity=round(i_pct), trimp_per_min=round(intensity, 2),
                       intensity_color=self._pct_colour(i_pct))
        return out

    # --- personal bests -----------------------------------------------------
    def check_pr(self, yards: int, stroke: str, seconds: float,
                 workout_id: int) -> bool:
        """True when this workout holds the fastest time at this distance/stroke.

        Medleys are looked up in their own table. `best_rep` is built with medley
        reps deliberately excluded — a 100 IM must not win the 100 backstroke just
        because backstroke was its modal length — with the consequence that a
        medley best could never be found here at all, and a personal-best 200 IM
        went unmarked on every card it appeared on.
        """
        best = (self.best_im.get(int(yards)) if stroke == "IM"
                else self.best_rep.get((yards, stroke)))
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
    # A rep's distance is how many real lengths it covers, which is not the same
    # as how many records the sensor wrote for it. A merged record is one row
    # covering two lengths; a split pair is two rows covering one. Counting rows
    # would file those reps under the wrong distance entirely.
    if "length_factor" not in lengths_df.columns:
        lengths_df = lengths_df.assign(length_factor=1.0)
    single_stroke = lengths_df
    if "im_continuous" in lengths_df.columns:
        # Reps that are themselves a medley are ranked as medleys, below.
        medley_reps = (lengths_df.loc[lengths_df["im_continuous"],
                                      ["workout_id", "rep_id"]]
                       .drop_duplicates())
        if len(medley_reps):
            single_stroke = lengths_df.merge(
                medley_reps.assign(_im=True), on=["workout_id", "rep_id"],
                how="left")
            single_stroke = single_stroke[single_stroke["_im"].isna()]
    reps = (single_stroke.groupby(["workout_id", "rep_id"])
            .agg(lengths=("length_factor", "sum"), seconds=("duration_s", "sum"),
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
        times = np.sort(g["seconds"].to_numpy())
        if len(times):
            ref.rep_times_by_distance_stroke[(int(yards), str(stroke))] = times

    for (yards, stroke), g in reps.groupby(["yards", "stroke"]):
        row = g.loc[g["seconds"].idxmin()]
        ref.best_rep[(int(yards), str(stroke))] = {
            "seconds": float(row["seconds"]),
            "workout_id": int(row["workout_id"]),
            "date": str(row["start"])[:10],
        }

    _load_medley_bests(ref, lengths_df)
    _load_training_load(engine, ref)
    return ref


def _load_medley_bests(ref: SwimmerReference, lengths_df: pd.DataFrame) -> None:
    """Rank medley rounds as their own events.

    A 100 IM belongs beside other 100 IMs, not beside 100 frees — pooling them
    would put every medley at the bottom of a freestyle field and tell the swimmer
    nothing. Continuous and broken rounds are ranked together, with each best
    recording which it was, because a broken 100 IM off four walls is genuinely
    quicker than swimming it straight and the table should not hide that.
    """
    from . import analyze

    # Medley detection reads the set structure. A frame assembled without it -
    # the rep-level fixtures in the tests, an older caller - simply has no
    # medleys to report rather than failing.
    if not {"set_id", "idx", "pool_m"} <= set(lengths_df.columns):
        return
    rounds = analyze.detect_im(lengths_df)
    if not rounds:
        return

    by_distance: dict[int, list] = {}
    for rnd in rounds:
        by_distance.setdefault(rnd.yards, []).append(rnd)

    dates = (lengths_df.drop_duplicates("workout_id")
             .set_index("workout_id")["start_time"].to_dict())
    for yards, group in by_distance.items():
        ref.im_times_by_distance[yards] = np.sort(
            np.array([r.seconds for r in group]))
        fastest = min(group, key=lambda r: r.seconds)
        ref.best_im[yards] = {
            "seconds": float(fastest.seconds),
            "workout_id": int(fastest.workout_id),
            "date": str(dates.get(fastest.workout_id, ""))[:10],
            "continuous": bool(fastest.continuous),
            "splits_s": [round(x, 1) for x in fastest.splits_s],
            "n_rounds": len(group),
        }


def _load_training_load(engine: Engine, ref: SwimmerReference) -> None:
    """Compute load and intensity for every workout with a heart-rate series."""
    stmt = (sa.select(hr_samples.c.workout_id, hr_samples.c.hr)
            .order_by(hr_samples.c.workout_id, hr_samples.c.t_s))
    with engine.connect() as c:
        df = pd.DataFrame(c.execute(stmt).mappings().all())
    if df.empty:
        return

    # A true resting heart rate is not in the data, so the lowest sustained rate
    # observed stands in for it. The 1st percentile rather than the minimum, since
    # the minimum is usually a sensor dropout.
    ref.hr_rest = int(np.percentile(df["hr"].to_numpy(dtype=float), 1))

    for wid, g in df.groupby("workout_id"):
        hr = g["hr"].to_numpy(dtype=float)
        if len(hr) < 60:
            continue
        load = ref.trimp(hr)
        ref.trimp_by_workout[int(wid)] = load
        ref.intensity_by_workout[int(wid)] = load / (len(hr) / 60.0)

    ref.trimp_sorted = np.sort(np.array(list(ref.trimp_by_workout.values())))
    ref.intensity_sorted = np.sort(np.array(list(ref.intensity_by_workout.values())))
