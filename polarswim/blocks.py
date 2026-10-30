"""Recognise unbroken swims built of freestyle and blocks of one other stroke.

`200 Swim (Odd 50 Back/Even 50 Free)`, `300 Swim (75 Back/150 Free/75 Back)`,
`400 Swim (100 Back/200 Free/100 Back)` — on 2026-10-30 three such swims were
swum unbroken, and the label-free pipeline read every one as freestyle:
`enforce_rep_consistency` makes a rep one stroke, and freestyle was the
majority. `inserts` does not reach them either; it reads freestyle with a few
single-length inserts, capped at 40% of the rep, and these are half back.

The shape is a two-level one. Every leg is a whole 50, 75 or 100, and each
leg runs either at the rep's freestyle or at one clearly slower pace. So the
rep is cut into units of two, three or four lengths, each unit is read at its
median (Polar's misplaced turns move time between neighbouring lengths, and a
median shrugs off one bad length where a mean does not), and the units must
fall into two levels with a clean gap between them.

What it takes, and why:

  * **Six lengths or more** (a 150), at least two units of each level. A 100
    of two 50s is a descending 100 as often as it is back/free.
  * **Two levels, clearly apart.** The slow units' median at least
    `MIN_RATIO` times the fast units', the slowest fast unit at least
    `MIN_GAP` below the fastest slow one, and the two-level fit of the unit
    medians better than one level by `MIN_GAIN` per length. A descend 200
    (30/28/26/24 per 50) has no gap.
  * **A freestyle base.** The fast level must be the day's freestyle pace
    (`BASE_MAX_RATIO`), so a backstroke 200 that slowed is not read as one.
  * **A coach's layout.** Alternating units, the slow stroke at both ends
    (`75 Back/150 Free/75 Back`), or slow first then free. A free swim that
    slows at the end is how a tired swimmer finishes, not a written set, so
    that one shape is refused.

Naming the slow stroke. Fly, back and breast all run near 1.25-1.35x this
swimmer's freestyle, so pace barely decides it; the learned back/breast split
(`analyze.learn_back_breast`) is the best there is, and it is used only when
the slow level sits clearly on one side of it. Block swims in one workout
whose slow levels agree (within `POOL_SPREAD`) are named together, from their
pooled level: a coach writes a set around one stroke, and one 200 swum fresh
is weaker evidence than three swims at once. Otherwise the slow lengths are
`undetermined` — the freestyle legs are still right, and the rep is no longer
called a 300 free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_REP_LENGTHS = 6
UNITS = (2, 3, 4)            # 50s, 75s, 100s in a 25 pool
MIN_RATIO = 1.18             # slow level against fast level
MIN_GAP = 1.10               # fastest slow unit against slowest fast unit
MIN_GAIN = 0.03              # mean |log residual| per length a second level must save
RESIDUAL_CLIP = 0.25         # one misplaced turn costs at most this
BASE_MAX_RATIO = 1.20        # fast level against the day's freestyle reference
MIN_SLOW_SHARE = 0.25
MAX_SLOW_SHARE = 0.75
NAME_MARGIN = 0.03           # how clearly the slow level must clear the split
POOL_SPREAD = 0.15           # slow levels this close in one workout: one stroke
CONFIDENCE = 0.65


@dataclass
class BlockRep:
    workout_id: int
    rep_id: int
    idxs: list[int]
    unit: int                       # lengths per leg
    slow_units: list[int]           # which units are the other stroke
    slow_ratio: float               # slow level / day's freestyle reference
    labels: dict[int, str]          # idx -> stroke


def _cost(logp: np.ndarray, level: float) -> float:
    return float(np.minimum(np.abs(logp - level), RESIDUAL_CLIP).sum())


def _layout_ok(slow: list[bool]) -> bool:
    k = len(slow)
    if all(slow[i] != slow[i + 1] for i in range(k - 1)):
        return True                                       # alternating
    runs = [slow[0]] + [slow[i] for i in range(1, k) if slow[i] != slow[i - 1]]
    if runs == [True, False, True]:
        return True                                       # slow ends, free middle
    return runs == [True, False]                          # slow first, then free


def read_rep(p: np.ndarray, free_ref: float) -> tuple[int, list[int], float] | None:
    """(unit, slow unit positions, slow level) for a rep that fits, else None."""
    n = len(p)
    if n < MIN_REP_LENGTHS or not np.all(np.isfinite(p)) or (p <= 0).any():
        return None
    logp = np.log(p)
    best = None
    for u in UNITS:
        if n % u or n // u < 3:
            continue
        # Scored on unit medians, weighted by unit size: a misplaced turn moves
        # time between two lengths of one unit and should not cost the fit.
        med = np.array([np.median(logp[i:i + u]) for i in range(0, n, u)])
        one = u * _cost(med, float(np.median(med)))
        order = np.argsort(med)
        # Two levels split at the largest gap between sorted unit medians.
        for cut in range(2, len(med) - 1):
            fast, slow = order[:cut], order[cut:]
            if len(slow) < 2:
                continue
            fast_lv, slow_lv = float(np.median(med[fast])), float(np.median(med[slow]))
            if slow_lv - fast_lv < np.log(MIN_RATIO):
                continue
            if med[slow].min() - med[fast].max() < np.log(MIN_GAP):
                continue
            is_slow = [i in set(slow.tolist()) for i in range(len(med))]
            share = sum(is_slow) / len(med)
            if not (MIN_SLOW_SHARE <= share <= MAX_SLOW_SHARE) or not _layout_ok(is_slow):
                continue
            lv = np.where(is_slow, slow_lv, fast_lv)
            two = u * float(np.minimum(np.abs(med - lv), RESIDUAL_CLIP).sum())
            if (one - two) / n < MIN_GAIN:
                continue
            if np.isfinite(free_ref) and free_ref > 0 \
                    and np.exp(fast_lv) / free_ref > BASE_MAX_RATIO:
                continue
            if best is None or two < best[0]:
                best = (two, u, sorted(slow.tolist()), float(np.exp(slow_lv)))
    return None if best is None else best[1:]


def _name(slow_ratio: float, params: dict) -> str:
    bb = params.get("_back_breast") or {}
    split = bb.get("split")
    if not split or abs(slow_ratio - split) < NAME_MARGIN:
        return "undetermined"
    slower_is_back = bool(bb.get("back_slower", 1.0))
    return "backstroke" if (slow_ratio > split) == slower_is_back else "breaststroke"


def detect_blocks(df: pd.DataFrame, params: dict | None = None,
                  skip: set[tuple[int, int]] | None = None) -> list[BlockRep]:
    """Every rep that reads as freestyle and blocks of one slower stroke.

    `skip` holds (workout_id, rep_id) already read by something more specific.
    """
    if df.empty or "rep_id" not in df.columns:
        return []
    params, skip = params or {}, skip or set()
    found: list[BlockRep] = []
    for (wid, rid), g in df.groupby(["workout_id", "rep_id"], sort=False):
        if (int(wid), int(rid)) in skip:
            continue
        g = g.sort_values("idx")
        free_ref = float(g["free_ref_s"].iloc[0]) if "free_ref_s" in g.columns else np.nan
        got = read_rep(g["pace_s"].to_numpy(dtype=float), free_ref)
        if got is None:
            continue
        unit, slow_units, slow_lv = got
        ratio = slow_lv / free_ref if np.isfinite(free_ref) and free_ref > 0 else np.nan
        idxs = [int(i) for i in g["idx"]]
        found.append(BlockRep(int(wid), int(rid), idxs, unit, slow_units,
                              float(ratio), {}))
    by_workout: dict[int, list[BlockRep]] = {}
    for r in found:
        by_workout.setdefault(r.workout_id, []).append(r)
    for reps in by_workout.values():
        ok = [r for r in reps if np.isfinite(r.slow_ratio)]
        ratios = [r.slow_ratio for r in ok]
        pooled = (ok and max(ratios) - min(ratios) <= POOL_SPREAD)
        if pooled:
            w = [len(r.slow_units) * r.unit for r in ok]
            level = float(np.average(ratios, weights=w))
        for r in reps:
            if not np.isfinite(r.slow_ratio):
                stroke = "undetermined"
            else:
                stroke = _name(level if pooled else r.slow_ratio, params)
            r.labels = {i: stroke if q // r.unit in r.slow_units else "freestyle"
                        for q, i in enumerate(r.idxs)}
    return found


def label_blocks(df: pd.DataFrame, found: list[BlockRep]) -> pd.DataFrame:
    """Write the reading over the per-length guesses, and mark the reps mixed,
    which keeps `enforce_rep_consistency` from collapsing them back to free."""
    df = df.copy()
    for col, fill in (("mixed_rep", False), ("pattern", None)):
        if col not in df.columns:
            df[col] = fill
    if not found:
        return df
    labels = {(r.workout_id, i): st for r in found for i, st in r.labels.items()}
    key = list(zip(df["workout_id"], df["idx"]))
    hit = [k in labels for k in key]
    df.loc[hit, "predicted"] = [labels[k] for k in key if k in labels]
    df.loc[hit, "confidence"] = CONFIDENCE
    df.loc[hit, "mixed_rep"] = True
    df.loc[hit, "pattern"] = "free+blocks"
    return df


def without_blocks(matches: list, found: list[BlockRep]) -> list:
    """Medley-pattern matches that touch no block rep (the block reading wins)."""
    taken = {(r.workout_id, i) for r in found for i in r.idxs}
    return [m for m in matches
            if not any((m.workout_id, i) in taken for i in m.idxs)]
