"""Recognise medley-order stroke patterns inside a single rep.

`analyze.detect_im` can only see a medley that REPEATS: it needs two or more
identical four-part rounds before it will claim anything. That covers `4x100 IM`
and misses everything else a swimmer actually writes down — a lone 200 IM, a
150 with the free dropped, an IM ladder where every rep is a different length.
On the 2026-09-11 practice the last five reps were

    50 fly · 100 fly/back · 150 fly/back/breast · 200 IM · 150 back/breast/free

and the repeating-round detector saw none of it, because no two reps had the same
shape. The pace/cost classifier then called the 200 IM "butterfly" and the two
150s "backstroke" and "freestyle".

The structure those five reps share is not repetition. It is that each one is a
**contiguous window of the medley order** — fly, back, breast, free — swum at one
distance per stroke. Drop the front and you get `back/breast/free`; drop the back
and you get `fly/back/breast`; take the whole thing and you get an IM; run it
backwards and you get a reverse IM. That is a small, closed grammar, and it is
worth recognising because within it the ORDER does the work that pace cannot:

    This swimmer's backstroke and breaststroke sit at 1.35 and 1.35 times
    freestyle pace — the same number. No threshold on time will ever tell them
    apart. But backstroke always comes BEFORE breaststroke in a medley, so
    inside a recognised pattern their positions name them and no discrimination
    by speed is needed at all.

So the method is:

  1. **Profile.** Learn what each stroke costs this swimmer, as a multiple of
     their freestyle pace, from reps that are unambiguously full medleys.
  2. **Anchor locally.** Day-to-day pace varies enormously here (a 100 IM at
     24 s/length in March, 60 s/length in April), so where a workout contains a
     full IM its own legs become that day's profile. The history-wide ratios are
     the fallback, and they carry lower confidence because they are looser.
  3. **Match.** Score every window of the grammar against the rep's leg times,
     and require the winner to beat both the runner-up AND the "it was all one
     stroke" hypothesis by a clear margin. Anything short of that is left alone
     for the ordinary classifier — a pattern claimed wrongly is worse than no
     pattern, because it renames four lengths at high confidence.

What this deliberately does NOT do is guess at the two-leg patterns whose strokes
are indistinguishable for the swimmer in question. With backstroke and
breaststroke at the same pace, a 100 of `fly/back` and a 100 of `back/breast`
have the same signature, and the margin rule rejects both rather than picking
one. A local anchor, which measures the two separately on the day, is usually
what rescues those.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analyze import IM_ORDER, REFERENCE_LENGTH_M

# A leg is one stroke's share of the rep, and it is swum at a distance a swimmer
# would write down. Allowing arbitrary leg lengths is what lets a 150 be read as
# two 75s — a shape nobody swims — so the legal leg distances are named instead.
LEG_YARDS = (25.0, 50.0, 100.0)
LEG_TOLERANCE_YD = 2.0

MIN_LEGS = 2               # one leg is not a pattern, it is just a rep
MAX_LEGS = 4               # the medley has four strokes
# ...but a rep read ON ITS OWN needs three. A two-leg rep offers two numbers, and
# there are six two-leg windows to spread over them, so one of them fits almost
# any pair: a 50 that went 31.2 then 24.8 is a perfect `back/fly` against this
# swimmer's profile and is far more likely a descending 50 free. Requiring three
# legs to seed a match cut this history from 273 claims to a defensible core,
# almost all of the loss being exactly those two-leg coincidences. Two-leg reps
# are still labelled — but only from inside a block, where the reps around them
# supply the evidence the rep itself lacks.
SEED_MIN_LEGS = 3

# How well the winning window has to fit, as mean relative error against the
# profile. At 0.10 a 30 s leg may sit 3 s off its prediction, which is about the
# spread a real leg shows; loosening it to 0.15 starts admitting fading
# freestyle reps as reverse medleys.
MAX_ERROR = 0.10
# ...and how much better it has to fit than its two rivals. `UNIFORM_MARGIN`
# guards against the commonest false positive by far — a descending freestyle
# swim, whose legs genuinely do get faster — by insisting the pattern explain the
# rep far better than a single stroke does. `RIVAL_MARGIN` is what makes the
# detector decline to choose between two windows that fit equally well.
UNIFORM_MARGIN = 1.8
RIVAL_MARGIN = 1.4
# Reverse medleys are real but rare: 9 of this swimmer's 85 candidate medleys ran
# backwards, against 76 forwards. A rep that starts fast and finishes slow is far
# more often a swimmer fading than a reverse IM, so the backwards windows are
# scored with a handicap and win only when they fit clearly better.
REVERSE_PENALTY = 1.15

# Names a swimmer would use. Everything else is spelled out leg by leg.
_ABBREV = {"butterfly": "fly", "backstroke": "back",
           "breaststroke": "breast", "freestyle": "free"}
_NAMED = {
    IM_ORDER: "IM",
    IM_ORDER[::-1]: "reverse IM",
    IM_ORDER[:3]: "IM no free",
    IM_ORDER[1:]: "IM no fly",
    IM_ORDER[:3][::-1]: "reverse IM no free",
    IM_ORDER[1:][::-1]: "reverse IM no fly",
}


def pattern_name(legs: tuple[str, ...]) -> str:
    """What a swimmer would call this window of the medley order."""
    if legs in _NAMED:
        return _NAMED[legs]
    return "/".join(_ABBREV[s] for s in legs)


def windows(k: int) -> list[tuple[tuple[str, ...], bool]]:
    """Every contiguous run of `k` strokes in medley order, each way round.

    Returned with a flag saying whether the run is backwards, because the
    backwards ones are scored against a prior and the caller needs to know which
    is which.
    """
    if not MIN_LEGS <= k <= MAX_LEGS:
        return []
    out = [(IM_ORDER[i:i + k], False) for i in range(len(IM_ORDER) - k + 1)]
    out += [(legs[::-1], True) for legs, _ in out]
    return out


@dataclass
class Profile:
    """Seconds per reference length for each stroke, and where it came from.

    `source` is not decoration. A profile measured from a full IM in the same
    workout is a measurement of this swimmer on this day; one derived from
    history-wide ratios is an estimate that ignores how the day is going. Both
    are usable and they are not equally good, so the difference is carried
    through to the confidence of everything matched against them.
    """
    pace: dict[str, float]
    source: str                     # 'workout' | 'history'
    free_pace_s: float = 0.0

    def usable(self) -> bool:
        return len(self.pace) == len(IM_ORDER) and all(v > 0 for v in self.pace.values())


@dataclass
class PatternMatch:
    """One rep read as a window of the medley order."""
    workout_id: int
    rep_id: int
    set_id: int
    legs: tuple[str, ...]
    leg_lengths: int                # pool lengths per leg
    leg_pace_s: list[float]         # observed, per reference length
    idxs: list[int]
    score: float                    # mean relative error against the profile
    margin: float                   # runner-up error / winner error
    confidence: float
    source: str
    block: int = 0                  # consecutive matched reps share a block number

    @property
    def name(self) -> str:
        return pattern_name(self.legs)

    @property
    def reverse(self) -> bool:
        """Whether the window runs backwards through the medley order."""
        return (len(self.legs) >= MIN_LEGS
                and IM_ORDER.index(self.legs[1]) < IM_ORDER.index(self.legs[0]))

    @property
    def is_medley(self) -> bool:
        """A full four-stroke medley, either direction.

        Reported separately because a whole medley is not a rep of any one
        stroke — it must not compete for the 100 backstroke best — while a
        partial window is still three strokes' worth of ordinary swimming.
        """
        return len(self.legs) == len(IM_ORDER)

    def label_for(self, idx: int) -> str | None:
        """Which stroke covers record `idx`, or None if this rep does not."""
        if idx not in self.idxs:
            return None
        return self.legs[self.idxs.index(idx) // self.leg_lengths]


# --- profile ---------------------------------------------------------------
# An anchor is a rep that can only be a full medley. The test is deliberately
# not "the last leg is fastest and the legs differ" — that is also what a
# descending freestyle 300 looks like, and 27 of this swimmer's 112 reps passing
# that test were exactly that. What separates a medley is the SIZE of the gap:
# the non-free legs of a real medley run 1.1 to 1.5 times freestyle pace, while
# a descending free swim holds every leg within a few percent of it.
ANCHOR_MIN_RATIO = 1.10        # every non-free leg, against the free leg
ANCHOR_MIN_MEDIAN_RATIO = 1.20  # and the typical one, by more


def find_anchors(df: pd.DataFrame) -> list[dict]:
    """Reps whose leg times can only be a full medley.

    Both directions are looked for, so a workout of reverse IMs can still
    anchor itself. Nothing here consults the classifier — the point of an anchor
    is to be evidence the classifier does not have.
    """
    out: list[dict] = []
    if df.empty:
        return out
    for (wid, rid), g in df.groupby(["workout_id", "rep_id"], sort=True):
        g = g.sort_values("idx")
        legs = _leg_pace(g, len(IM_ORDER))
        if legs is None:
            continue
        for reverse in (False, True):
            v = legs[::-1] if reverse else legs
            free, others = v[-1], v[:-1]
            if free <= 0 or others.min() < free * ANCHOR_MIN_RATIO:
                continue
            if float(np.median(others)) < free * ANCHOR_MIN_MEDIAN_RATIO:
                continue
            order = IM_ORDER[::-1] if reverse else IM_ORDER
            out.append(dict(workout_id=int(wid), rep_id=int(rid), reverse=reverse,
                            pace={s: float(x) for s, x in zip(order, v)}))
            break                   # a rep anchors one way or the other, not both
    return out


def learn_ratios(df: pd.DataFrame) -> dict[str, float]:
    """Each stroke's pace as a multiple of freestyle pace, over the whole history.

    Ratios rather than absolute times, because absolute times do not transfer
    between days: this swimmer's 100 IM legs range from 18 s to 60 s depending on
    the session, while the RELATIVE cost of butterfly against freestyle is a fact
    about the swimmer and holds across all of it.
    """
    anchors = find_anchors(df)
    if not anchors:
        return {}
    rows = []
    for a in anchors:
        free = a["pace"]["freestyle"]
        if free > 0:
            rows.append({s: v / free for s, v in a["pace"].items()})
    if not rows:
        return {}
    med = pd.DataFrame(rows).median()
    return {s: float(med[s]) for s in IM_ORDER}


def profile_for(g: pd.DataFrame, ratios: dict[str, float],
                anchors: list[dict]) -> Profile | None:
    """The best per-stroke pace estimate available for one workout.

    A full medley swum in this very workout beats anything learned from history,
    because it measures all four strokes under the conditions of the day. Where
    there is none, the history-wide ratios are scaled by the workout's own
    freestyle pace — which is estimated as a low percentile of its length times,
    the fastest thing in a practice being freestyle for this swimmer.
    """
    mine = [a for a in anchors if a["workout_id"] == int(g["workout_id"].iloc[0])]
    if mine:
        pace = {s: float(np.median([a["pace"][s] for a in mine])) for s in IM_ORDER}
        return Profile(pace=pace, source="workout", free_pace_s=pace["freestyle"])

    if not ratios:
        return None
    # The 15th percentile, not the minimum: the minimum in this history is
    # routinely a mis-timed record (a missed wall leaves a 15 s "length"), and
    # anchoring a whole workout's profile on a sensor defect would drag every
    # prediction fast.
    free = float(np.nanpercentile(g["pace_s"].to_numpy(dtype=float), 15))
    if not np.isfinite(free) or free <= 0:
        return None
    return Profile(pace={s: free * ratios[s] for s in IM_ORDER},
                   source="history", free_pace_s=free)


# --- matching ---------------------------------------------------------------
def _leg_pace(g: pd.DataFrame, k: int) -> np.ndarray | None:
    """Split one rep into `k` equal legs, as seconds per reference length.

    Returns None when the rep cannot be divided that way, or when the resulting
    leg is not a distance anyone swims. Normalising to the reference length is
    what lets a 25 yd pool and a 25 m pool be matched against one profile.
    """
    n = len(g)
    if k <= 0 or n % k:
        return None
    per_leg = n // k
    pool_m = float(g["pool_m"].iloc[0])
    if pool_m <= 0:
        return None
    leg_yd = per_leg * pool_m / 0.9144
    if not any(abs(leg_yd - y) <= LEG_TOLERANCE_YD for y in LEG_YARDS):
        return None
    norm = REFERENCE_LENGTH_M / pool_m
    secs = g["duration_s"].to_numpy(dtype=float)
    if not np.isfinite(secs).all():
        return None
    return secs.reshape(k, per_leg).sum(axis=1) * norm / per_leg


def _error(legs_pace: np.ndarray, strokes: tuple[str, ...],
           profile: Profile) -> float:
    """Mean relative error of one window against the profile.

    Relative, so a slow stroke is not penalised merely for being slow, and the
    same threshold works for a 25 yd leg and a 100 yd one.
    """
    pred = np.array([profile.pace[s] for s in strokes], dtype=float)
    return float(np.mean(np.abs(legs_pace - pred) / pred))


def match_rep(g: pd.DataFrame, profile: Profile) -> PatternMatch | None:
    """Read one rep as a window of the medley order, or decline to.

    Every legal division of the rep into equal legs is tried, and within each
    division the windows of the grammar compete with the four "it was all one
    stroke" hypotheses. Comparing them at the SAME division matters: a two-leg
    reading has two residuals and a four-leg reading has four, so an error
    averaged over one is not comparable with an error averaged over the other,
    and the uniform hypothesis has to be evaluated leg by leg rather than over
    the rep as a whole to be a fair rival.
    """
    if not profile.usable():
        return None
    g = g.sort_values("idx")
    best: tuple[float, float, tuple[str, ...], int, np.ndarray] | None = None

    for k in range(SEED_MIN_LEGS, MAX_LEGS + 1):
        legs_pace = _leg_pace(g, k)
        if legs_pace is None or legs_pace.min() <= 0:
            continue

        scored = []
        for strokes, reverse in windows(k):
            err = _error(legs_pace, strokes, profile)
            scored.append((err * (REVERSE_PENALTY if reverse else 1.0), strokes))
        if not scored:
            continue
        scored.sort()
        (winner_err, winner), = scored[:1]

        # The rival that matters most: one stroke, held for the whole rep. A
        # descending freestyle swim beats every medley window on this test.
        uniform = min(_error(legs_pace, (s,) * k, profile) for s in IM_ORDER)
        if uniform < winner_err * UNIFORM_MARGIN:
            continue
        runner_up = scored[1][0] if len(scored) > 1 else np.inf
        if runner_up < winner_err * RIVAL_MARGIN:
            continue                        # two windows fit; say nothing
        if winner_err > MAX_ERROR:
            continue

        margin = min(runner_up, uniform) / max(winner_err, 1e-9)
        if best is None or winner_err < best[0]:
            best = (winner_err, margin, winner, len(g) // k, legs_pace)

    if best is None:
        return None
    err, margin, strokes, per_leg, legs_pace = best
    # Fit quality first, then how far clear of the alternatives it landed, then a
    # penalty for a profile borrowed from other days rather than measured today.
    conf = 0.92 - 2.5 * err + min(0.04, 0.01 * (margin - RIVAL_MARGIN))
    if profile.source != "workout":
        conf -= 0.15
    return PatternMatch(
        workout_id=int(g["workout_id"].iloc[0]), rep_id=int(g["rep_id"].iloc[0]),
        set_id=int(g["set_id"].iloc[0]) if "set_id" in g.columns else 0,
        legs=strokes, leg_lengths=per_leg,
        leg_pace_s=[round(float(x), 2) for x in legs_pace],
        idxs=[int(i) for i in g["idx"]], score=round(err, 4),
        margin=round(float(margin), 2), confidence=round(min(0.95, max(0.35, conf)), 3),
        source=profile.source)


# --- blocks -----------------------------------------------------------------
# A rep read on its own is often genuinely ambiguous, and the reference practice
# shows why. Its `100 fly/back` legs came in at 29.2 and 32.0 s per length. Read
# alone, `fly/back` (27.2, 32.8) and a uniform breaststroke 100 (31.2, 31.2) fit
# about equally well, so `match_rep` correctly refuses to choose. What resolves it
# is not a better threshold but the four reps around it: a 50, a 150, a 200 IM and
# another 150, all swum at 50 yd per stroke. Fix the leg distance at 50 and each
# rep's NUMBER of legs is forced by its distance — the 50 has one, the 100 two, the
# 150 three, the 200 four — leaving only the window's starting stroke free. That is
# at most four possibilities per rep, and one shared profile has to explain every
# leg of every rep at once.
#
# Under that constraint the ambiguity disappears. `fly/back` explains the 100 in a
# block whose 200 is a measured IM; uniform breaststroke explains the block not at
# all.
BLOCK_MARGIN = 1.8         # the block must beat "every rep was one stroke" by this
# Ladders and sliding windows move the window one stroke at a time. That is a
# preference, not a rule: the cost is set far too small to override a real
# difference in fit, and only decides between readings that fit equally well.
TRANSITION_COST = 0.004
# A single-leg rep contributes one number, which is thin evidence. One is allowed
# at each END of a block — an IM ladder opens with a 50 fly and closes with a 50
# free — but never in the middle, where it would let a block grow on through a
# whole freestyle set.
EDGE_SINGLE_MAX_ERROR = 0.06


@dataclass
class Block:
    """A run of consecutive reps read together as one medley structure."""
    workout_id: int
    reverse: bool
    leg_lengths: int
    matches: list[PatternMatch] = field(default_factory=list)
    error: float = 0.0              # mean relative error of the joint reading
    null_error: float = 0.0         # ...of "every rep was a single stroke"

    @property
    def margin(self) -> float:
        return self.null_error / max(self.error, 1e-9)


def _window_starts(k: int) -> range:
    return range(0, len(IM_ORDER) - k + 1)


def _fit_block(reps: list[tuple[int, pd.DataFrame]], leg_lengths: int,
               reverse: bool, profile: Profile) -> Block | None:
    """Choose every rep's window at once, by shortest path over window starts.

    Each rep is a column of candidate starting strokes, and the arcs between
    columns cost a little where the window jumps by more than one stroke — so of
    two readings that fit the times equally, the one that reads as a progression
    wins. With at most four candidates per rep this is exact, not a heuristic.
    """
    order = IM_ORDER[::-1] if reverse else IM_ORDER
    cols: list[tuple[int, pd.DataFrame, int, np.ndarray]] = []
    for rid, g in reps:
        if len(g) % leg_lengths:
            return None
        k = len(g) // leg_lengths
        if not 1 <= k <= len(IM_ORDER):
            return None
        legs = _leg_pace(g, k)
        if legs is None or legs.min() <= 0:
            return None
        cols.append((int(rid), g, k, legs))
    if not any(k >= MIN_LEGS for _, _, k, _ in cols):
        return None

    prev_cost: dict[int, float] = {}
    prev_k = 0
    back: list[dict[int, int]] = []
    for i, (_, _, k, legs) in enumerate(cols):
        cost: dict[int, float] = {}
        ptr: dict[int, int] = {}
        for j in _window_starts(k):
            own = _error(legs, tuple(order[j:j + k]), profile)
            if i == 0:
                cost[j], ptr[j] = own, -1
                continue
            best: tuple[float, int] | None = None
            for a, c in prev_cost.items():
                step = max(abs(j - a), abs((j + k) - (a + prev_k)))
                total = c + own + TRANSITION_COST * max(0, step - 1)
                if best is None or total < best[0]:
                    best = (total, a)
            cost[j], ptr[j] = best
        back.append(ptr)
        prev_cost, prev_k = cost, k

    j = min(prev_cost, key=lambda s: prev_cost[s])
    starts = [0] * len(cols)
    for i in range(len(cols) - 1, -1, -1):
        starts[i] = j
        j = back[i][j]

    block = Block(workout_id=int(cols[0][1]["workout_id"].iloc[0]),
                  reverse=reverse, leg_lengths=leg_lengths)
    errors, nulls = [], []
    for (rid, g, k, legs), start in zip(cols, starts):
        strokes = tuple(order[start:start + k])
        err = _error(legs, strokes, profile)
        errors.append(err)
        nulls.append(min(_error(legs, (s,) * k, profile) for s in IM_ORDER))
        block.matches.append(PatternMatch(
            workout_id=block.workout_id, rep_id=rid,
            set_id=int(g["set_id"].iloc[0]) if "set_id" in g.columns else 0,
            legs=strokes, leg_lengths=leg_lengths,
            leg_pace_s=[round(float(x), 2) for x in legs],
            idxs=[int(i) for i in g["idx"]],
            score=round(err, 4), margin=0.0, confidence=0.0,
            source=profile.source))
    block.error = float(np.mean(errors))
    block.null_error = float(np.mean(nulls))
    return block


def _grow(seed: PatternMatch, reps: dict[int, pd.DataFrame],
          profile: Profile) -> Block | None:
    """Extend a confident rep outwards while the neighbours still fit.

    Growth stops at the first rep that cannot be divided into legs of the block's
    distance, or that the block's profile fails to explain. A single-leg rep ends
    the growth in its direction, so a block can pick up the 50 that opens a ladder
    without running on through the freestyle set beyond it.
    """
    L = seed.leg_lengths
    order = sorted(reps)
    if seed.rep_id not in reps:
        return None
    reverse = seed.reverse

    def fits(g: pd.DataFrame) -> bool:
        if len(g) % L:
            return False
        k = len(g) // L
        if not 1 <= k <= len(IM_ORDER):
            return False
        legs = _leg_pace(g, k)
        if legs is None or legs.min() <= 0:
            return False
        ordering = IM_ORDER[::-1] if reverse else IM_ORDER
        best = min(_error(legs, tuple(ordering[j:j + k]), profile)
                   for j in _window_starts(k))
        return best <= (MAX_ERROR if k >= MIN_LEGS else EDGE_SINGLE_MAX_ERROR)

    at = order.index(seed.rep_id)
    lo = hi = at
    for step in (-1, 1):
        i = at
        while True:
            nxt = i + step
            if not 0 <= nxt < len(order) or order[nxt] != order[i] + step:
                break
            g = reps[order[nxt]]
            if not fits(g):
                break
            i = nxt
            if len(g) % L == 0 and len(g) // L < MIN_LEGS:
                break               # one single-leg rep per side, then stop
        lo, hi = (i, hi) if step < 0 else (lo, i)

    if hi - lo < 1:
        return None
    return _fit_block([(r, reps[r]) for r in order[lo:hi + 1]], L, reverse, profile)


def _score_block(block: Block) -> None:
    """Turn a block's joint fit into a per-rep confidence."""
    strength = min(0.06, 0.02 * (block.margin - BLOCK_MARGIN))
    for m in block.matches:
        conf = 0.92 - 2.5 * m.score + strength
        if m.source != "workout":
            conf -= 0.15
        if len(m.legs) < MIN_LEGS:
            # One leg agreeing with the profile is the thinnest evidence here. It
            # is only worth reporting because the reps around it are strong.
            conf -= 0.18
        m.margin = round(block.margin, 2)
        m.confidence = round(min(0.95, max(0.35, conf)), 3)


def detect_patterns(df: pd.DataFrame,
                    ratios: dict[str, float] | None = None,
                    anchors: list[dict] | None = None) -> list[PatternMatch]:
    """Every rep in `df` that reads as a window of the medley order.

    Two passes, and the order is the point. First every rep is read on its own
    under strict margins, which finds the ones that can only be a medley window —
    a 200 IM, a 150 with the free dropped. Those become seeds. Each seed then
    grows outwards into the run of reps around it and the whole run is re-read
    together, which is what recovers the reps that are hopelessly ambiguous alone.

    A block is also the answer to a reporting problem. An IM ladder is one set in
    the swimmer's head and five reps of five different distances to `assign_sets`,
    which groups by distance and so splits it five ways. Every rep of a block
    carries the same `block` number, so the ladder can be shown as the one thing
    it was.
    """
    out: list[PatternMatch] = []
    if df.empty:
        return out
    if anchors is None:
        anchors = find_anchors(df)
    if ratios is None:
        ratios = learn_ratios(df)

    block_no = 0
    for _, w in df.groupby("workout_id", sort=True):
        profile = profile_for(w, ratios, anchors)
        if profile is None:
            continue
        reps = {int(rid): g.sort_values("idx") for rid, g in w.groupby("rep_id")}
        seeds = [m for m in (match_rep(g, profile) for g in reps.values())
                 if m is not None]
        # Best-fitting seed first, so the strongest evidence claims its
        # neighbourhood before a weaker seed can take part of it.
        claimed: set[int] = set()
        for seed in sorted(seeds, key=lambda m: m.score):
            if seed.rep_id in claimed:
                continue
            block_no += 1
            block = _grow(seed, reps, profile)
            if (block is not None and block.error <= MAX_ERROR
                    and block.margin >= BLOCK_MARGIN
                    and not (claimed & {m.rep_id for m in block.matches})):
                _score_block(block)
                for m in block.matches:
                    m.block = block_no
                    claimed.add(m.rep_id)
                out.extend(block.matches)
            else:
                seed.block = block_no
                claimed.add(seed.rep_id)
                out.append(seed)

    out.sort(key=lambda m: (m.workout_id, m.rep_id))
    return out


# --- labelling --------------------------------------------------------------
def label_patterns(df: pd.DataFrame, matches: list[PatternMatch]) -> pd.DataFrame:
    """Write a matched rep's known stroke order over whatever was inferred.

    Inside a recognised pattern the strokes are not guesses — the order is what
    made it recognisable — so these labels replace the pace/cost prediction and
    carry the match's own confidence.

    Four columns come out of this, and they answer four different questions.
    `pattern` is what to CALL the rep (`IM`, `IM no fly`). `pattern_block` is
    which run of reps it belongs to, so a ladder can be shown as the one set it
    was. `mixed_rep` is whether the rep covered more than one stroke, which is
    what disqualifies it from a single-stroke personal best — a 150 of
    back/breast/free must not win the 150 backstroke. And `im_continuous` is the
    narrower question of whether it was a WHOLE medley, which is what qualifies it
    for the medley bests instead.
    """
    df = df.copy()
    for col, fill in (("pattern", None), ("pattern_block", 0),
                      ("mixed_rep", False)):
        if col not in df.columns:
            df[col] = fill
    if "im_continuous" not in df.columns:
        df["im_continuous"] = False
    if not matches:
        return df

    labels: dict[tuple[int, int], tuple[str, float, str, int, bool, bool]] = {}
    for m in matches:
        mixed = len(m.legs) >= MIN_LEGS
        for pos, idx in enumerate(m.idxs):
            labels[(m.workout_id, idx)] = (
                m.legs[pos // m.leg_lengths], m.confidence,
                # A single-leg member of a block is not a pattern, it is a rep
                # whose stroke the block happened to identify. Naming it one
                # would cost it its own personal best: the 50 fly that opens an
                # IM ladder is a genuine 50 fly and belongs in the 50 fly field,
                # while the 150 of back/breast/free beside it is not an event at
                # all. So the useful part — the stroke — is written either way,
                # and only a multi-stroke rep is named and set aside.
                m.name if mixed else None,
                m.block, m.is_medley, mixed)

    key = list(zip(df["workout_id"], df["idx"]))
    hit = [k in labels for k in key]
    if not any(hit):
        return df
    chosen = [labels[k] for k in key if k in labels]
    df.loc[hit, "predicted"] = [c[0] for c in chosen]
    df.loc[hit, "confidence"] = [c[1] for c in chosen]
    df.loc[hit, "pattern"] = [c[2] for c in chosen]
    df.loc[hit, "pattern_block"] = [c[3] for c in chosen]
    df.loc[hit, "im_continuous"] = [c[4] for c in chosen]
    df.loc[hit, "mixed_rep"] = [c[5] for c in chosen]
    return df


def as_params(ratios: dict[str, float]) -> dict[str, dict[str, float]]:
    """The learned ratios, shaped for `model_params` so they persist."""
    return {"_stroke_ratio": dict(ratios)} if ratios else {}


def from_params(params: dict[str, dict[str, float]]) -> dict[str, float]:
    """Read the ratios back, ignoring a stored set that is not complete."""
    r = (params or {}).get("_stroke_ratio") or {}
    return {s: float(r[s]) for s in IM_ORDER} if all(s in r for s in IM_ORDER) else {}
