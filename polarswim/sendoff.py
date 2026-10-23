"""Read reps off the send-off clock.

A set swum on a send-off — `25 x 50 @ 1:00` — has a clock in it that Polar
does not use. Every rep starts on the minute, so a rep starts wherever a length
starts on the minute, whether or not Polar logged a rest there. On 2026-10-23
it often did not: after eight of the 24 IM and free 50s Polar recorded no
rest at all, because the 8-12 s at the wall had been folded into the last
length. Reps are cut where rests are, so those 50s became 100s,
the detector read one fly/back 50 plus the free 50 after it as a 100 IM, and
the paces it reported for them were swim plus rest.

The clock recovers all of that. What it takes to call a run of lengths a
train, and why each condition is there:

  * **Starts on a grid.** Each rep's first length starts near `t0 + k * P`,
    for P on a coach's five-second grid: up to `TOL_S` early, since Polar's
    times are quantised to 0.8 s and its walls land a second or so off, and up
    to `LATE_S` late when the rep before was still swimming at the send-off —
    it overran, so this one left late (10/23 rep 25, 5.6 s after a 65 s IM 50).
  * **The same distance every time.** Each slot holds the same number of
    lengths, give or take one in a few slots (Polar drops or doubles a wall).
  * **Long enough to be a set.** `MIN_REPS` swum slots: four 50s on the minute
    can line up by accident in a long swim; eight cannot.
  * **Rest, somewhere.** At least a third of the rep starts show a logged
    rest. A continuous swim at a steady pace also has lengths starting on a
    grid; what it never has is a rest before a third of them.
  * **Rest at both kinds of start.** Polar misses rests, but not always at
    the same reps. On 3/6, 100s on about 2:10 whose slow first 50 took half
    the send-off read as 50s on 1:05 — with no rest, ever, before the second
    50 of each. That is a 100, and the coarser grid reads it. On 10/23 Polar
    missed most rests after the IM 50s, but logged four of them.
  * **A skipped rep is a hole.** An empty slot is allowed — the swimmer sat
    one out — but not two in a row, and not many. The hole keeps the count:
    rep 9 is still rep 9, which matters because sets alternate by rep number.
  * **Extra rest is a re-phase.** When the train breaks and a new one starts
    at the same send-off and distance soon after (10/23: thirty seconds extra
    after rep 13), it is the same set. The rep count carries on across it.

Alternation. Coaches write these sets `odd free / even IM` far more often than
not, so a train is read by parity. The evidence per rep is its FIRST length:
it starts on the clock and ends at a wall Polar saw, so it is a clean time,
where the last length may have a rest in it. When one parity runs at the day's
freestyle pace and the other clearly slower, the fast reps are freestyle and
the slow ones are not.

Naming the slow reps is harder, and it is done only when they split again by
rep number into two clear groups (10/23: `25 fly + 25 back` and `25 back + 25
breast`, every other IM). Medley-order pairs that are not free are fly/back
and back/breast, so the faster group starts with fly — the learned ratios must
confirm this swimmer's fly is clearly quicker than his back — and the slower
with back, each carrying on in medley order. Back against breast is the weak
step: his two run within a few percent, so a `fly/back ⇄ breast/free` set
would be misnamed. Without that split the slow reps are left `undetermined`
where the classifier called them free and the rep was itself slow: they are
not free, and which stroke they were is a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

IM_ORDER = ("butterfly", "backstroke", "breaststroke", "freestyle")

SENDOFFS_S = tuple(float(p) for p in range(30, 185, 5))
TOL_S = 4.0                  # early: the clock does not run fast
LATE_S = 8.0                 # late: a slow rep overran the send-off
MIN_REPS = 8
MIN_REST_SHARE = 1 / 3       # of rep starts with a logged rest
MIN_REST_S = 5.0             # a typical seen rep ends at least this far inside P
MAX_ODD_COUNT_SHARE = 0.2    # slots one length off the train's distance
MAX_EMPTY_SHARE = 0.15
REPHASE_SLOTS = 2.5          # a new train this soon after, same P and n, continues
REST_GAP_S = 2.0             # as `analyze.REST_GAP_S`

PARITY_RATIO = 1.12          # slow parity's first lengths against the fast one's
PARITY_AGREE = 0.7           # share of slow reps above the fast parity's median
FREE_MAX_RATIO = 1.25        # fast parity's first lengths against the day's free reference
SUB_RATIO = 1.10             # the slow reps' two groups, median to median
SUB_MIN_REPS = 3             # per group, to name the slow reps' strokes
BACK_MARGIN = 0.05           # fly must beat backstroke, as ratios to free, by this
CONFIDENCE = 0.60


@dataclass
class Train:
    workout_id: int
    sendoff_s: float
    lengths_per_rep: int
    slots: list[list[int] | None]           # idx lists, None = a rep sat out
    # Lengths whose time includes the rest before the next rep, because Polar
    # logged none between them.
    rest_hidden: set[int] = field(default_factory=set)

    @property
    def idxs(self) -> list[int]:
        return [i for s in self.slots if s for i in s]


def _walk(t: np.ndarray, d: np.ndarray, a: int, P: float) -> list[list[int] | None]:
    """Slots from length `a` on, every P seconds, until the grid breaks."""
    n = len(t)
    slots: list[list[int] | None] = []
    j, k = a, 0
    while j < n:
        s = t[a] + k * P
        idx = [i for i in range(j, n) if s - TOL_S <= t[i] < s + P - TOL_S]
        if not idx:
            if t[j] < s - TOL_S or (slots and slots[-1] is None):
                break
            slots.append(None)
        elif t[idx[0]] - s > (LATE_S if _overran(slots, t, d, s) else TOL_S):
            break
        else:
            slots.append(idx)
            j = idx[-1] + 1
        k += 1
    while slots and slots[-1] is None:
        slots.pop()
    return slots


def _overran(slots, t: np.ndarray, d: np.ndarray, s: float) -> bool:
    """Was the last rep still swimming when this one was due to leave?"""
    prev = slots[-1] if slots else None
    return bool(prev) and t[prev[-1]] + d[prev[-1]] > s - TOL_S


def _qualifies(slots: list[list[int] | None], rest: np.ndarray,
               t: np.ndarray, d: np.ndarray, P: float) -> int | None:
    """The train's lengths per rep, or None when it is not a train."""
    filled = [s for s in slots if s]
    if len(filled) < MIN_REPS:
        return None
    if sum(s is None for s in slots) > MAX_EMPTY_SHARE * len(slots):
        return None
    counts = [len(s) for s in filled]
    m = int(np.median(counts))
    off = [c for c in counts if c != m]
    if any(abs(c - m) > 1 for c in off) or len(off) > MAX_ODD_COUNT_SHARE * len(filled):
        return None
    if sum(rest[s[0]] > REST_GAP_S for s in filled[1:]) < MIN_REST_SHARE * len(filled):
        return None
    # A send-off leaves rest. Where Polar saw the rep end, it must have ended
    # well inside the send-off: 25s that take 33 s on :35 (5/17) are slow
    # continuous swimming, not a set on the clock.
    swims = [t[a[-1]] + d[a[-1]] - t[a[0]] for a, b in zip(slots, slots[1:])
             if a and b and rest[b[0]] > REST_GAP_S]
    if not swims or float(np.median(swims)) > P - MIN_REST_S:
        return None
    # Every other rep start, and never a rest at one of them: the halves of
    # longer swims, not a train of short ones.
    seen = [[], []]
    for k, (a, b) in enumerate(zip(slots, slots[1:])):
        if a and b:
            seen[(k + 1) % 2].append(rest[b[0]] > REST_GAP_S)
    if any(len(v) >= 3 and not any(v) for v in seen):
        return None
    return m


def find_trains(df: pd.DataFrame) -> list[Train]:
    """Every send-off train in every workout of `df` (after `assign_sets`)."""
    out: list[Train] = []
    for wid, g in df.groupby("workout_id", sort=False):
        g = g.sort_values("idx")
        t = g["start_offset_s"].to_numpy(dtype=float)
        rest = g["rest_before_s"].to_numpy(dtype=float)
        d = g["duration_s"].to_numpy(dtype=float)
        idx = g["idx"].to_numpy(dtype=int)
        cands = []
        for a in range(len(t)):
            # A set starts after a rest Polar saw. Walking from a wall inside a
            # rep finds a grid of second lengths that, on 10/23, happened to
            # line up again with the reps after the extra rest.
            if a and rest[a] <= REST_GAP_S:
                continue
            for P in SENDOFFS_S:
                slots = _walk(t, d, a, P)
                m = _qualifies(slots, rest, t, d, P)
                if m is not None:
                    covered = sum(len(s) for s in slots if s)
                    cands.append((covered, -sum(s is None for s in slots), a, P, m, slots))
        cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
        used: set[int] = set()
        trains = []
        for _, _, a, P, m, slots in cands:
            pos = {i for s in slots if s for i in s}
            if pos & used:
                continue
            used |= pos
            trains.append((a, P, m, slots))
        trains.sort(key=lambda c: c[0])

        # Extra rest mid-set re-phases the clock; the count carries on.
        joined: list[tuple[float, int, list]] = []
        for a, P, m, slots in trains:
            if joined:
                pP, pm, pslots = joined[-1]
                last = next(s for s in reversed(pslots) if s)
                gap = t[a] - t[last[0]]
                if (pP == P and pm == m and gap < REPHASE_SLOTS * P
                        and a == last[-1] + 1):
                    skipped = [None] * max(0, int(gap // P) - 1)
                    joined[-1] = (P, m, pslots + skipped + slots)
                    continue
            joined.append((P, m, slots))

        for P, m, slots in joined:
            named = [[int(idx[i]) for i in s] if s else None for s in slots]
            hidden = set()
            for s, nxt in zip(slots, slots[1:]):
                if s and nxt and rest[nxt[0]] <= REST_GAP_S:
                    hidden.add(int(idx[s[-1]]))
            out.append(Train(int(wid), P, m, named, hidden))
    return out


def recut(df: pd.DataFrame, trains: list[Train]) -> pd.DataFrame:
    """Each slot its own rep, each train one set; ids renumbered in time order."""
    if not trains:
        return df
    df = df.sort_values(["workout_id", "idx"]).copy()
    if "rest_hidden" not in df.columns:
        df["rest_hidden"] = False
    rep_start: dict[tuple[int, int], int] = {}
    set_start: dict[tuple[int, int], int] = {}
    for tr in trains:
        filled = [s for s in tr.slots if s]
        for s in filled:
            rep_start[(tr.workout_id, s[0])] = 1
        set_start[(tr.workout_id, filled[0][0])] = 1
        after = filled[-1][-1]
        set_start[(tr.workout_id, after + 1)] = 1
        rep_start[(tr.workout_id, after + 1)] = 1
    key = list(zip(df["workout_id"], df["idx"]))
    in_train = {(tr.workout_id, i) for tr in trains for i in tr.idxs}
    hidden = {(tr.workout_id, i) for tr in trains for i in tr.rest_hidden}

    prev_rep = df.groupby("workout_id")["rep_id"].shift(1)
    prev_set = df.groupby("workout_id")["set_id"].shift(1)
    new_rep = []
    new_set = []
    for k, (r, pr, s, ps) in enumerate(zip(df["rep_id"], prev_rep, df["set_id"], prev_set)):
        kk = key[k]
        if pd.isna(pr):
            new_rep.append(True)
            new_set.append(True)
            continue
        if kk in in_train:
            new_rep.append(kk in rep_start)
            new_set.append(kk in set_start)
        else:
            new_rep.append(r != pr or kk in rep_start)
            new_set.append(s != ps or kk in set_start)
    df["rep_id"] = pd.Series(new_rep, index=df.index).groupby(df["workout_id"]).cumsum().astype(int)
    df["set_id"] = pd.Series(new_set, index=df.index).groupby(df["workout_id"]).cumsum().astype(int)
    df["rep_lengths"] = df.groupby(["workout_id", "rep_id"])["idx"].transform("size")
    df.loc[[k in hidden for k in key], "rest_hidden"] = True
    return df.reset_index(drop=True)


@dataclass
class Reading:
    workout_id: int
    labels: dict[int, str]          # idx -> stroke
    pattern: dict[int, str]         # idx -> name of a mixed rep
    idxs: list[int]
    slow: set[int] = field(default_factory=set)    # idxs of the slow parity


def _medley_from(first: str, n: int) -> list[str]:
    i = IM_ORDER.index(first)
    return [IM_ORDER[min(i + q, len(IM_ORDER) - 1)] for q in range(n)]


def read_parity(df: pd.DataFrame, trains: list[Train],
                ratios: dict[str, float] | None = None) -> list[Reading]:
    """Odd/even alternation in each train, named where the evidence allows."""
    from .patterns import pattern_name
    if not trains:
        return []
    dur = {(int(w), int(i)): float(d) for w, i, d in
           zip(df["workout_id"], df["idx"], df["pace_s"])}
    pred = {(int(w), int(i)): p for w, i, p in
            zip(df["workout_id"], df["idx"], df["predicted"])}
    free_ref = {}
    if "free_ref_s" in df.columns:
        free_ref = df.groupby("workout_id")["free_ref_s"].first().to_dict()
    fly_faster = bool(ratios) and (
        ratios.get("backstroke", 0) >= ratios.get("butterfly", np.inf) + BACK_MARGIN)

    out: list[Reading] = []
    for tr in trains:
        w = tr.workout_id
        first = {k: dur[(w, s[0])] for k, s in enumerate(tr.slots) if s}
        par = [[v for k, v in first.items() if k % 2 == p] for p in (0, 1)]
        if min(len(p) for p in par) < MIN_REPS // 2:
            continue
        med = [float(np.median(p)) for p in par]
        fast = int(np.argmin(med))
        slow = 1 - fast
        if med[slow] < PARITY_RATIO * med[fast]:
            continue
        if np.mean([v > med[fast] for v in par[slow]]) < PARITY_AGREE:
            continue
        # The fast parity must be freestyle, not merely the quicker of two
        # strokes: a back/breast set alternating is no "free / not free". First
        # lengths carry the push off the wall, hence the loose bound against
        # the day's freestyle reference (its 15th-percentile length).
        ref = free_ref.get(w)
        if ref is not None and np.isfinite(ref) and med[fast] > FREE_MAX_RATIO * float(ref):
            continue

        labels: dict[int, str] = {}
        pattern: dict[int, str] = {}
        slow_slots = [k for k in first if k % 2 == slow]
        sub = {k: (k // 2) % 2 for k in slow_slots}
        groups = [[first[k] for k in slow_slots if sub[k] == q] for q in (0, 1)]
        named = None
        if (fly_faster and min(len(g) for g in groups) >= SUB_MIN_REPS):
            gm = [float(np.median(g)) for g in groups]
            hi = int(np.argmax(gm))
            if gm[hi] >= SUB_RATIO * gm[1 - hi]:
                named = {hi: "backstroke", 1 - hi: "butterfly"}
        for k, s in enumerate(tr.slots):
            if not s:
                continue
            if k % 2 == fast:
                for i in s:
                    labels[i] = "freestyle"
                continue
            if named is not None:
                strokes = _medley_from(named[sub[k]], len(s))
                for i, st in zip(s, strokes):
                    labels[i] = st
                if len(set(strokes)) > 1:
                    name = pattern_name(tuple(dict.fromkeys(strokes)))
                    for i in s:
                        pattern[i] = name
            elif first[k] >= PARITY_RATIO * med[fast]:
                # Only a rep that was itself slow: a quick one in the slow
                # parity is as likely the freestyle it looks like.
                for i in s:
                    if pred.get((w, i)) == "freestyle":
                        labels[i] = "undetermined"
        slow_idx = {i for k in slow_slots for i in tr.slots[k]}
        out.append(Reading(w, labels, pattern, tr.idxs, slow_idx))
    return out


def label_parity(df: pd.DataFrame, readings: list[Reading]) -> pd.DataFrame:
    """Write the reading over the per-length guesses; mixed reps stay mixed.

    `parity` says which half of an odd/even set a length is in (0 the fast
    half, 1 the slow), so the card can show `13×50 free` and `11×50 IM`
    rather than 24 alternating rows.
    """
    df = df.copy()
    for col, fill in (("pattern", None), ("mixed_rep", False), ("parity", np.nan)):
        if col not in df.columns:
            df[col] = fill
    if not readings:
        return df
    labels = {(r.workout_id, i): st for r in readings for i, st in r.labels.items()}
    names = {(r.workout_id, i): p for r in readings for i, p in r.pattern.items()}
    key = list(zip(df["workout_id"], df["idx"]))
    hit = [k in labels for k in key]
    df.loc[hit, "predicted"] = [labels[k] for k in key if k in labels]
    df.loc[hit, "confidence"] = CONFIDENCE
    slow = {(r.workout_id, i) for r in readings for i in r.slow}
    df.loc[hit, "parity"] = [float(k in slow) for k in key if k in labels]
    named = [k in names for k in key]
    df.loc[named, "pattern"] = [names[k] for k in key if k in names]
    df.loc[named, "mixed_rep"] = True
    return df


def without_trains(matches: list, readings: list[Reading]) -> list:
    """Medley-pattern matches that touch no read train (the clock wins)."""
    taken = {(r.workout_id, i) for r in readings for i in r.idxs}
    return [m for m in matches
            if not any((m.workout_id, i) in taken for i in m.idxs)]
