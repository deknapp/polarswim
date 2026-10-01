"""Recognise freestyle swims with single-length stroke inserts in them.

A coach's aerobic IM set is very often written as freestyle with the other
strokes dropped in: `2x (75 FR + 25 Fly + 75 FR + 25 BK)`, or the longer
`50 FR + 25 Fly + 50 FR + 50 BK + 50 FR + 25 BR`. On 2026-09-30 four such 200s
were swum unbroken, and the label-free pipeline got almost all of it wrong: the
medley-pattern matcher read one as a 200 IM in 50s, and `enforce_rep_consistency`
collapsed another to eight lengths of butterfly — because a rep is one stroke,
and the insert lengths had pulled the majority off freestyle.

The shape is easy to see once it is named. Most of the rep runs at one steady
pace, and a few lengths, in the same places in every rep of the set, run a
fifth to a third slower. Those few are the inserts; everything else is the
freestyle the set is built on.

What it takes to call a rep that, and why each condition is there:

  * **A freestyle base.** The rep's quick lengths (the fastest two thirds) must
    run at about the day's freestyle pace. A backstroke 200 with one slow
    length is a backstroke 200, not freestyle with inserts.
  * **Clearly slow inserts.** At least `INSERT_RATIO` times that base, and not
    so slow they are kick (`MAX_RATIO`).
  * **Structure, not noise.** A traffic-slowed length is also a slow length.
    So the slow positions must either RECUR — the same positions slow in at
    least two reps of equal length in the set, read from the set's per-position
    mean — or, for a rep on its own, be evenly spaced with at least two of
    them, the way `75 + 25 + 75 + 25` is.
  * **A minority, and at least two.** Inserts are at most 40% of the rep,
    and a rep needs two or more: one slow length is as likely traffic, and
    reading it gains nothing, since the rest of the rep was freestyle anyway.
    Outside the insert positions the rep must be steady.

Naming the inserts. Pace cannot do it — this swimmer's fly, back and breast
25s sit within a few percent of each other — but coaches write inserts in
medley order, as they write everything else. So a rep with three insert groups
is fly, back, breast; with four it is a medley. With one or two the order alone
does not decide (`fly + back` and `back + breast` are both medley order), and
the inserts are left `undetermined`: the freestyle is the part this recognises
reliably, and claiming a stroke for the rest would be a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

IM_ORDER = ("butterfly", "backstroke", "breaststroke", "freestyle")

MIN_REP_LENGTHS = 6          # 75 + 25 is the shortest unit worth reading
INSERT_RATIO = 1.15          # an insert against the rep's own freestyle
MAX_RATIO = 1.70             # beyond this it is kick, not a stroke
SET_MIN_RATIO = 1.10         # every rep of the set must agree, if loosely
BASE_MAX_RATIO = 1.15        # the rep's base against the day's freestyle pace
MAX_SHARE = 0.4             # the bonus set: 100 of its 250 yards are inserts
CONFIDENCE = 0.70


@dataclass
class InsertRep:
    workout_id: int
    rep_id: int
    idxs: list[int]
    inserts: list[int]              # positions within the rep, 0-based
    labels: dict[int, str]          # idx -> stroke


def _base(p: np.ndarray) -> float:
    """The rep's freestyle: the median of its fastest two thirds."""
    k = max(2, int(round(len(p) * 2 / 3)))
    return float(np.median(np.sort(p)[:k]))


def _groups(positions: list[int]) -> list[list[int]]:
    """Contiguous runs: a `50 BK` insert is two slow lengths, one insert."""
    out: list[list[int]] = []
    for q in positions:
        if out and q == out[-1][-1] + 1:
            out[-1].append(q)
        else:
            out.append([q])
    return out


def _evenly_spaced(positions: list[int], n: int) -> bool:
    groups = _groups(positions)
    if len(groups) < 2:
        return False
    # Repeating units that each close with their insert, as `75 + 25` does:
    # the first insert ends unit one, and every later one a unit further on.
    ends = [g[-1] for g in groups]
    unit = ends[0] + 1
    return unit >= 2 and n % unit == 0 and ends == list(range(unit - 1, n, unit))


def _name(groups: list[list[int]]) -> list[str]:
    if len(groups) == 3:
        return list(IM_ORDER[:3])
    if len(groups) == 4:
        return list(IM_ORDER)
    return ["undetermined"] * len(groups)


def detect_inserts(df: pd.DataFrame) -> list[InsertRep]:
    """Every rep that reads as freestyle with stroke inserts."""
    if df.empty or "rep_id" not in df.columns:
        return []
    found: list[InsertRep] = []
    for (wid, sid), s in df.groupby(["workout_id", "set_id"], sort=False):
        free_ref = float(s["free_ref_s"].iloc[0]) if "free_ref_s" in s.columns else np.nan
        reps = [(rid, g.sort_values("idx")) for rid, g in s.groupby("rep_id", sort=False)]
        reps = [(rid, g) for rid, g in reps if len(g) >= MIN_REP_LENGTHS]
        by_n: dict[int, list[tuple[int, pd.DataFrame]]] = {}
        for rid, g in reps:
            by_n.setdefault(len(g), []).append((rid, g))
        for n, group in by_n.items():
            ratios = []
            for _, g in group:
                p = g["pace_s"].to_numpy(dtype=float)
                ratios.append(p / _base(p))
            ratios = np.array(ratios)
            if len(group) >= 2:
                mean = ratios.mean(axis=0)
                shared = [q for q in range(n) if mean[q] >= INSERT_RATIO
                          and (ratios[:, q] >= SET_MIN_RATIO).all()
                          and mean[q] <= MAX_RATIO]
            else:
                shared = None
            for (rid, g), r in zip(group, ratios):
                p = g["pace_s"].to_numpy(dtype=float)
                if np.isfinite(free_ref) and free_ref > 0 and _base(p) / free_ref > BASE_MAX_RATIO:
                    continue
                if shared is not None:
                    pos = shared
                else:
                    pos = [q for q in range(n) if INSERT_RATIO <= r[q] <= MAX_RATIO]
                    if not _evenly_spaced(pos, n):
                        continue
                if not pos or len(pos) > n * MAX_SHARE:
                    continue
                groups = _groups(pos)
                # One slow length in a rep is as likely traffic as a stroke, and
                # reading it costs nothing the majority rule had not already got
                # right — the rest of the rep was freestyle either way.
                if len(groups) < 2:
                    continue
                # The base must be steady: a slow length outside the insert
                # positions means this rep does not follow the set's shape.
                if any(r[q] >= INSERT_RATIO for q in range(n) if q not in pos):
                    continue
                names = _name(groups)
                idxs = [int(i) for i in g["idx"]]
                labels = {i: "freestyle" for i in idxs}
                for grp, st in zip(groups, names):
                    for q in grp:
                        labels[idxs[q]] = st
                found.append(InsertRep(int(wid), int(rid), idxs, pos, labels))
    return found


def label_inserts(df: pd.DataFrame, found: list[InsertRep]) -> pd.DataFrame:
    """Write the reading over the per-length guesses, and mark the reps mixed.

    `mixed_rep` keeps `enforce_rep_consistency` off them — collapsing one to its
    majority is the error this module exists to stop — and keeps a 200 with
    inserts out of the 200 freestyle bests.
    """
    df = df.copy()
    if "mixed_rep" not in df.columns:
        df["mixed_rep"] = False
    if "pattern" not in df.columns:
        df["pattern"] = None
    if not found:
        return df
    labels = {(r.workout_id, i): st for r in found for i, st in r.labels.items()}
    key = list(zip(df["workout_id"], df["idx"]))
    hit = [k in labels for k in key]
    df.loc[hit, "predicted"] = [labels[k] for k in key if k in labels]
    df.loc[hit, "confidence"] = CONFIDENCE
    df.loc[hit, "mixed_rep"] = True
    df.loc[hit, "pattern"] = "free+inserts"
    return df


def without_inserts(matches: list, found: list[InsertRep]) -> list:
    """Medley-pattern matches that touch no insert rep (the insert reading wins)."""
    taken = {(r.workout_id, i) for r in found for i in r.idxs}
    return [m for m in matches
            if not any((m.workout_id, i) in taken for i in m.idxs)]
