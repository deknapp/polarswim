"""Read sets whose reps alternate shape, where nothing else read them.

`sendoff` reads `odd free / even IM` off the clock. Plenty of such sets are not
swum on a send-off — 2026-11-20's ten 50s rested anywhere from 10 s to 3:30 —
and without the clock nothing looked at them as a set at all. Each rep went to
the per-length classifier alone, and this swimmer's fly is the stroke it cannot
see: a fly 25 runs within a few percent of his freestyle (25.6 s against 24.8),
so every fly length that day read as free.

What gives fly away is not its own pace but the SHAPE of the rep it is in. Read
each rep as two halves:

  * ``F``  — the whole rep at freestyle pace.
  * ``FS`` — a first half at (nearly) freestyle pace and a second clearly
    slower. In medley order the only pair that does this is fly then back:
    fly is his quick stroke, back his slow one. (Free then breast would too,
    but a reverse pair is rare, and the set reading below rules it out.)
  * ``SF`` — slow then free pace: breast then free, the one medley pair that
    finishes on the fast stroke.
  * ``S``  — slow throughout, one non-free stroke.

A single rep's shape proves little. A free 50 that fades from 25.6 to 31.2 is
rare but not impossible, and across the history 109 two- and four-length reps
have an ``FS`` shape, most of them unplanned and many surely freestyle. What a
fade never does is ALTERNATE. So shapes are only read as a set:

  1. **Halves sets** — four or more consecutive reps alternating ``FS`` and
     ``SF``: `100 fly/back, 100 breast/free, ...` (2026-11-20 lengths 64-79).
     Nothing else in the history fires.
  2. **Parity sets** — six or more consecutive reps of one distance where
     every other rep is ``F`` and the others are not: `odd free / even
     stroke`. The slow parity splits again by rep number into two groups
     (coaches alternate the evens too); a group whose reps are mostly ``FS``
     is fly/back. Its partner, uniform and slow, is read as ONE stroke — an
     unbroken swim is one stroke unless its shape says otherwise — and that
     stroke is breast, the one the fly/back group leaves out. Without an
     ``FS`` group the slow reps are only marked not-free where the
     classifier called them free; which stroke they were stays a guess.

A planned workout is never read here; the plan already says what was swum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sendoff import Reading

IM_ORDER = ("butterfly", "backstroke", "breaststroke", "freestyle")

REP_LENGTHS = (2, 4)         # 50s and 100s in a 25 pool: two halves of 25 or 50
# Shape thresholds, as multiples of the day's freestyle reference (its 15th-
# percentile length) and as the ratio of second half to first.
FREE_MEAN = 1.10             # an F rep's mean pace
FREE_HALF = 1.18             # ...and neither half slower than this
FAST_HALF = 1.18             # the quick half of FS (fly runs ~1.05-1.11x free)
FREE_FINISH = 1.15           # the free half of SF
SLOW_MEAN = 1.15             # an S rep, and the slow half of SF
SHIFT = 1.12                 # second half against first, for FS / SF

HALVES_MIN_REPS = 4
PARITY_MIN_REPS = 6
PARITY_AGREE = 0.7           # share of the fast parity that must be F
PARITY_SLOW_MAX_F = 0.2      # ...and of the slow parity that may be
GROUP_AGREE = 2 / 3          # share of a slow sub-group that must be FS
CONFIDENCE = 0.6


def token(g: pd.DataFrame, ref: float) -> str:
    """The shape of one rep: F, FS, SF, S, or X for none of them."""
    p = g["pace_s"].to_numpy(dtype=float)
    if len(p) not in REP_LENGTHS or not np.isfinite(p).all() or (p <= 0).any():
        return "X"
    h = len(p) // 2
    a, b = float(p[:h].mean()), float(p[h:].mean())
    s = b / a
    if p.mean() <= FREE_MEAN * ref and max(a, b) <= FREE_HALF * ref:
        return "F"
    if a <= FAST_HALF * ref and s >= SHIFT:
        return "FS"
    if b <= FREE_FINISH * ref and a >= SLOW_MEAN * ref and s <= 1 / SHIFT:
        return "SF"
    if p.mean() >= SLOW_MEAN * ref and 1 / SHIFT < s < SHIFT:
        return "S"
    return "X"


def _runs(reps: list[pd.DataFrame]) -> list[list[pd.DataFrame | None]]:
    """Maximal runs of reps of one distance.

    One stray rep of another length inside a run keeps its slot as None, so
    the count — and with it the parity — carries on past it. 2026-11-20's
    seventh 50 came out as three records (16.8 40.8 24.8, a wall Polar put in
    the wrong place) between two ordinary 50s.
    """
    out = []
    i = 0
    while i < len(reps):
        n = len(reps[i])
        if n not in REP_LENGTHS:
            i += 1
            continue
        run: list[pd.DataFrame | None] = [reps[i]]
        j = i + 1
        while j < len(reps):
            if len(reps[j]) == n:
                run.append(reps[j])
                j += 1
            elif (run[-1] is not None and j + 1 < len(reps)
                  and len(reps[j + 1]) == n):
                run.append(None)
                j += 1
            else:
                break
        out.append(run)
        i = j
    return out


def _halves(w: int, run, toks) -> list[Reading]:
    out = []
    k = 0
    while k < len(toks):
        if toks[k] not in ("FS", "SF"):
            k += 1
            continue
        e = k + 1
        while e < len(toks) and toks[e] in ("FS", "SF") and toks[e] != toks[e - 1]:
            e += 1
        if e - k >= HALVES_MIN_REPS:
            labels, pattern, idxs, slow = {}, {}, [], set()
            for g, t in zip(run[k:e], toks[k:e]):
                pair = IM_ORDER[:2] if t == "FS" else IM_ORDER[2:]
                ids = [int(i) for i in g["idx"]]
                h = len(ids) // 2
                for q, i in enumerate(ids):
                    labels[i] = pair[q >= h]
                    pattern[i] = "fly/back" if t == "FS" else "breast/free"
                idxs += ids
                if t == "SF":
                    slow |= set(ids)
            out.append(Reading(w, labels, pattern, idxs, slow))
        k = max(e, k + 1)
    return out


def _parity(w: int, run, toks, pred) -> Reading | None:
    if len(run) < PARITY_MIN_REPS:
        return None
    for fast in (0, 1):
        f = [t for t in toks[fast::2] if t is not None]
        s = [t for t in toks[1 - fast::2] if t is not None]
        if not f or not s:
            continue
        if np.mean([t == "F" for t in f]) < PARITY_AGREE:
            continue
        if np.mean([t == "F" for t in s]) > PARITY_SLOW_MAX_F:
            continue
        if np.mean([t != "F" for t in s]) < PARITY_AGREE:
            continue
        slow_slots = [k for k in range(1 - fast, len(run), 2)]
        groups = [[k for q, k in enumerate(slow_slots) if q % 2 == p] for p in (0, 1)]
        fs_share = [np.mean([toks[k] == "FS" for k in g if run[k] is not None] or [0])
                    for g in groups]
        flyback = next((p for p in (0, 1) if fs_share[p] >= GROUP_AGREE
                        and fs_share[1 - p] < GROUP_AGREE), None)

        labels, pattern, idxs, slow = {}, {}, [], set()
        for k, g in enumerate(run):
            if g is None:
                continue
            ids = [int(i) for i in g["idx"]]
            idxs += ids
            if k % 2 == fast:
                if toks[k] == "F":
                    for i in ids:
                        labels[i] = "freestyle"
                continue
            slow |= set(ids)
            if flyback is not None:
                if k in groups[flyback]:
                    h = len(ids) // 2
                    for q, i in enumerate(ids):
                        labels[i] = IM_ORDER[q >= h]
                        pattern[i] = "fly/back"
                elif toks[k] == "S":
                    for i in ids:
                        labels[i] = "breaststroke"
            elif toks[k] in ("S", "SF", "FS"):
                for i in ids:
                    if pred.get(i) == "freestyle":
                        labels[i] = "undetermined"
        return Reading(w, labels, pattern, idxs, slow)
    return None


def read_alternation(df: pd.DataFrame, claimed: set[tuple[int, int]],
                     planned: set[int] | frozenset = frozenset()) -> list[Reading]:
    """Halves and parity sets in reps nothing else has read."""
    out: list[Reading] = []
    if df.empty or "free_ref_s" not in df.columns:
        return out
    for wid, W in df.groupby("workout_id", sort=True):
        w = int(wid)
        if w in planned:
            continue
        ref = float(W["free_ref_s"].iloc[0])
        if not np.isfinite(ref) or ref <= 0:
            continue
        pred = dict(zip(W["idx"].astype(int), W["predicted"])) \
            if "predicted" in W.columns else {}
        reps = [g.sort_values("idx") for _, g in W.groupby("rep_id", sort=True)]
        # A rep something else already read is a wall here: never re-read it,
        # and never let a run reach across it.
        segments: list[list[pd.DataFrame]] = [[]]
        for g in reps:
            if any((w, int(i)) in claimed for i in g["idx"]):
                segments.append([])
            else:
                segments[-1].append(g)
        for seg in segments:
            for run in _runs(seg):
                toks = [None if g is None else token(g, ref) for g in run]
                found = [r for part in _split_none(run, toks)
                         for r in _halves(w, *part)]
                if found:
                    out.extend(found)
                    continue
                r = _parity(w, run, toks, pred)
                if r is not None:
                    out.append(r)
    return out


def _split_none(run, toks):
    """Sub-runs between stray slots, for the halves reading (no gaps there)."""
    cur_r, cur_t = [], []
    for g, t in zip(run, toks):
        if g is None:
            if cur_r:
                yield cur_r, cur_t
            cur_r, cur_t = [], []
        else:
            cur_r.append(g)
            cur_t.append(t)
    if cur_r:
        yield cur_r, cur_t
