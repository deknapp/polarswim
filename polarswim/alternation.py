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

  4. **Stroke cycles** — eight or more 50s in a row that go round the four
     strokes one whole 50 each: `16 x 50 free, breast, back, fly` (2026-11-25
     lengths 37-68). His fly 50 runs at his free pace (27 s against 26), so
     pace splits the cycle only into a fast pair and a slow pair, but a
     medley cycle always puts fly next to free and back next to breast, so
     the set must go fast, fast, slow, slow round and round. Every 50 of the
     window has to land in its pair's band, and a stretch of free with one
     slow 50 never does that. Which way round it goes (fly, back, breast,
     free or free, breast, back, fly) is read from two cues that must agree:
     free is the faster of the fast pair and back the slower of the slow pair.
     If they disagree, only the faster fast position is named (free) and the
     rest are marked undetermined where the classifier said free. Polar
     running two 50s into one record (53-56 that day) doesn't matter: the
     cycle is read in 50s, not in Polar's reps — but at least three in four
     of the 50s must be a rep of their own. Odd free 100s / even 100 IMs in
     25s (2026-05-01) cut into 50s also go fast, fast, slow, slow; there no
     50 is its own rep.

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
STROKE_MIN_REPS = 4
STROKE_MIN_FS = 2
STROKE_MAX_SHIFT = 1.5       # an FS slower than this in its back half is a misplaced wall
STROKE_MIN_LEN = 0.8         # ...as is any length under this x the day's free
CYCLE_MIN_UNITS = 8          # two full rounds of the four strokes
CYCLE_SPLIT = 1.15           # fast pair under this x the day's free, slow pair over it
CYCLE_GAP = 1.05             # slowest fast 50 against quickest slow 50
CYCLE_OWN_REP = 0.75         # share of the 50s that must be a Polar rep of their own
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


def _plausible_fs(g: pd.DataFrame, ref: float) -> bool:
    p = g["pace_s"].to_numpy(dtype=float)
    h = len(p) // 2
    return (p[h:].mean() / p[:h].mean() <= STROKE_MAX_SHIFT
            and p.min() >= STROKE_MIN_LEN * ref)


def _stroke_sets(w: int, run, toks, pred, params) -> list[Reading]:
    """Runs of slow reps holding fly/back pairs; see the module docstring, 3."""
    from .blocks import _name

    out = []
    k = 0
    while k < len(toks):
        if toks[k] in (None, "F"):
            k += 1
            continue
        e = k
        while e < len(toks) and toks[e] not in (None, "F"):
            e += 1
        # A misplaced wall (16.0 then 46.4 s on 2026-07-02) also reads FS:
        # demote it before counting fly/back pairs.
        seg = [t if t != "FS" or _plausible_fs(g, ref) else "X"
               for g, t in zip(run[k:e], toks[k:e])
               for ref in [float(g["free_ref_s"].iloc[0])]]
        if (e - k >= STROKE_MIN_REPS and seg.count("FS") >= STROKE_MIN_FS
                and "S" in seg):
            labels, pattern, idxs = {}, {}, []
            for g, t in zip(run[k:e], seg):
                ids = [int(i) for i in g["idx"]]
                idxs += ids
                if t == "FS":
                    h = len(ids) // 2
                    for q, i in enumerate(ids):
                        labels[i] = IM_ORDER[q >= h]
                        pattern[i] = "fly/back"
                elif t == "S" and all(pred.get(i) in (None, "freestyle")
                                      for i in ids):
                    # Only over a free guess: drill and kick calls stand.
                    ratio = float(g["pace_s"].mean() / g["free_ref_s"].iloc[0])
                    name = _name(ratio, params or {})
                    for i in ids:
                        labels[i] = name
                else:
                    for i in ids:
                        if pred.get(i) == "freestyle":
                            labels[i] = "undetermined"
            out.append(Reading(w, labels, pattern, idxs, set(idxs)))
        k = e
    return out


def _unit_runs(seg: list[pd.DataFrame]) -> list[tuple[list[list[int]], list[bool]]]:
    """Maximal stretches of whole 50s (two lengths each), as lists of idxs.

    A rep of four or six lengths is cut into 50s: Polar sometimes runs two
    50s together. A rep of an odd number of lengths ends the stretch.
    """
    out, cur, own = [], [], []
    for g in seg:
        ids = [int(i) for i in g["idx"]]
        if len(ids) % 2 or len(ids) > 6:
            if cur:
                out.append((cur, own))
            cur, own = [], []
            continue
        cur += [ids[q:q + 2] for q in range(0, len(ids), 2)]
        own += [len(ids) == 2] * (len(ids) // 2)
    if cur:
        out.append((cur, own))
    return out


def _cycles(w: int, units: list[list[int]], own: list[bool], pace: dict[int, float],
            ref: float, pred: dict) -> list[Reading]:
    """Four-stroke cycles of whole 50s; see the module docstring, 4."""
    means = [float(np.mean([pace[i] for i in u])) for u in units]
    if any(not np.isfinite(m) for m in means):
        return []
    fast = [m <= CYCLE_SPLIT * ref for m in means]
    out = []
    k = 0
    while k + CYCLE_MIN_UNITS <= len(units):
        best = None
        for off in range(4):
            # Template fast, fast, slow, slow, rotated to start at `off`.
            want = lambda q: ((q + off) % 4) < 2
            e = k
            while e < len(units) and fast[e] == want(e - k):
                e += 1
            if e - k >= CYCLE_MIN_UNITS and (best is None or e > best[1]):
                best = (off, e)
        if best is None:
            k += 1
            continue
        off, e = best
        pos = [(q - k + off) % 4 for q in range(k, e)]   # 0,1 fast; 2,3 slow
        m = {p: [means[q] for q, pp in zip(range(k, e), pos) if pp == p] for p in range(4)}
        f_max = max(m[0] + m[1])
        s_min = min(m[2] + m[3])
        if (s_min < CYCLE_GAP * f_max or any(len(v) < 2 for v in m.values())
                or np.mean(own[k:e]) < CYCLE_OWN_REP):
            k += 1
            continue
        a, b, c, d = (float(np.median(m[p])) for p in range(4))
        # Forward (fly back breast free): round the cycle it is free, fly,
        # back, breast. Reverse (free breast back fly): fly, free, breast, back.
        free_first = a < b
        back_first = c > d
        if free_first == back_first:
            names = (("freestyle", "butterfly", "backstroke", "breaststroke")
                     if free_first else
                     ("butterfly", "freestyle", "breaststroke", "backstroke"))
        else:
            names = (("freestyle", None, None, None) if a < b
                     else (None, "freestyle", None, None))
        labels, pattern, idxs = {}, {}, []
        for u, p in zip(units[k:e], pos):
            for i in u:
                idxs.append(i)
                if names[p] is not None:
                    labels[i] = names[p]
                elif pred.get(i) == "freestyle":
                    labels[i] = "undetermined"
        out.append(Reading(w, labels, pattern, idxs, set()))
        k = e
    return out


def read_alternation(df: pd.DataFrame, claimed: set[tuple[int, int]],
                     planned: set[int] | frozenset = frozenset(),
                     params: dict | None = None) -> list[Reading]:
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
        pace = dict(zip(W["idx"].astype(int), W["pace_s"].astype(float)))
        for seg in segments:
            # Stroke cycles first: they are the strictest reading, and the
            # stretch they claim is not offered to the others.
            cyc = [r for units, own in _unit_runs(seg)
                   for r in _cycles(w, units, own, pace, ref, pred)]
            # A Polar rep holding two 50s of different strokes is a mixed rep,
            # so the later whole-rep pass doesn't flatten it to one stroke.
            short = {"freestyle": "free", "butterfly": "fly", "backstroke": "back",
                     "breaststroke": "breast"}
            for r in cyc:
                for g in seg:
                    ids = [int(i) for i in g["idx"]]
                    got = [r.labels.get(i) for i in ids]
                    if all(x is not None for x in got) and len(set(got)) > 1:
                        name = "/".join(dict.fromkeys(short.get(x, x) for x in got))
                        for i in ids:
                            r.pattern[i] = name
            out.extend(cyc)
            taken = {i for r in cyc for i in r.idxs}
            seg = [g for g in seg if not any(int(i) in taken for i in g["idx"])]
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
                    continue
                out.extend(r for part in _split_none(run, toks)
                           for r in _stroke_sets(w, *part, pred, params))
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
