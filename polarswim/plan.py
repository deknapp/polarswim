"""Read a workout against the plan the swimmer was given.

From lap times and heart rate alone, stroke is barely identifiable — this
swimmer's backstroke and breaststroke both run about 1.33x freestyle pace, and a
tired freestyle 50 is indistinguishable from either. Polar's per-length splits
are the only timing there is, and on a crowded lane they are wrong in ways no
rule can undo on its own: stopping short of the wall, a turn placed late, a
200 broken by traffic.

A written workout changes the problem. `5 x 50 Free @1:10` says how many swims,
how far, which stroke and roughly when each one started, so the question is no
longer "what stroke is this length" but "which planned swim is this length part
of" — an alignment problem, and one the data can answer well, because the
structure (rests, distances, send-offs) carries most of the information that
pace and heart rate cannot.

The alignment is a dynamic program over lengths x planned swims. Each planned
swim claims a contiguous run of Polar lengths, and the path is scored on:

  * count   — claiming more or fewer lengths than the distance needs. Allowed,
              because Polar does miss and invent walls, but it costs.
  * rest    — a swim should begin where the swimmer stopped. Starting one in
              the middle of an unbroken run is expensive; a stop in the middle
              of a swim (traffic, a short wall) is cheap, since it happens.
  * pace    — the swim's pace against what that stroke costs this swimmer, as a
              multiple of the day's own freestyle. Deliberately soft: pace is
              the weak signal here, and it only has to break ties.
  * send-off — swim time plus the rest after it, against the nearest of the
              listed intervals. Loose, because a shared lane is loose.

Lengths that fit no planned swim are left unplanned (warm-up before the plan
started, "random stuff" after it) and go back to the ordinary classifier, as
does any swim written as "choice".

What the alignment is used for:

  1. Stroke.  A planned stroke is the swimmer's own word about the swim, so it
     outranks every inference — and ranks below only a hand correction.
  2. Structure.  Each planned swim becomes one rep and each plan line one set,
     so a 200 broken by traffic still reads as a 200, and the card mirrors the
     written workout.
  3. Splits.  Where Polar's turn points are visibly wrong inside one swim — a
     28.8 s length followed by an 18.4 s one, which is a misplaced wall or a
     short finish rather than a swimmer changing speed by half — the swim's
     time is spread evenly across its lengths. The swim's total is kept; only
     the per-length division, which is the part Polar got wrong, is replaced.
     Where Polar's length COUNT is wrong, every length is rescaled so the swim
     covers the distance the plan says it did.
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_LENGTH_M = 22.86

# Pace as a multiple of the day's freestyle reference (the 15th percentile of the
# day's lengths). Freestyle sits a little above 1.0 because the reference is the
# fast end of the day. The three strokes are overridden by the swimmer's own
# learned medley ratios when those exist.
EXPECTED_RATIO = {
    "freestyle": 1.08, "butterfly": 1.28, "backstroke": 1.34,
    "breaststroke": 1.33, "kick": 1.55, "drill": 1.30, None: 1.12,
}
# How much the ratio may wander, in log units. Kick and "choice" are wide on
# purpose: a fast kick and an easy kick differ enormously, and a choice swim
# could be anything.
PACE_SIGMA = {"kick": 0.22, None: 0.25}
DEFAULT_SIGMA = 0.14
EASY_FACTOR = 1.10               # an EZ swim runs about a tenth slower

# Costs, in units where one pace standard deviation squared costs 0.5.
COST_UNPLANNED = 2.0             # per length left out of the plan, mid-workout
# ...but a swim the swimmer forgot to mention is ONE omission, however long it
# was. At full price per length, a 150 missing from the account cost 12 and the
# aligner would rather carve it up and put the planned 25s at its start.
COST_UNPLANNED_CONT = 0.5
# Before the plan starts and after it ends, unplanned swimming is normal — the
# swimmer arrived late, or added "random stuff" at the end. Charging those
# lengths the full price made the aligner stretch planned swims over them.
COST_UNPLANNED_EDGE = 0.3
# Per planned length the swimmer did not swim. Dearer than a miscounted length:
# a swimmer who says they did the workout is more likely right than the sensor.
COST_SKIP_SWIM = 5.0
COST_COUNT = 2.5                 # per length claimed beyond/short of the distance
# Where a swim of legs has one record per planned length, each length's pace
# against the swim's own freestyle says WHERE the fly/back/breast leg fell —
# `75 FR + 25 Fly + 75 FR + 25 BK` is three quick lengths, a slow one, three
# quick, a slow one. Relative to the swim itself, so the day's speed cancels.
LEG_SIGMA = 0.10
COST_LEG = 1.0                   # weight of the per-length shape term
COST_START_MIDSWIM = 3.0         # a swim beginning where the swimmer did not stop
COST_BROKEN = 0.6                # a stop inside a swim (traffic, short wall)
# ...and a long one is not a traffic stop. Beyond this, each second costs, so a
# 200 is never stretched across a two-minute rest.
BROKEN_FREE_S = 20.0
COST_BROKEN_PER_S = 0.08
COST_SENDOFF = 0.04              # per second outside the send-off tolerance
SENDOFF_TOLERANCE_S = 8.0
# Heart rate, used for the two swims where it is unambiguous: butterfly is the
# most expensive thing in a practice, and an EZ swim is the cheapest. Scored on
# the swim's heart-rate percentile within the day, so fitness and drift cancel.
COST_HR = 4.0
FLY_HR_PCT = 0.55                # fly below this percentile starts to cost
EASY_HR_PCT = 0.50               # an EZ swim above it starts to cost
HR_LAG_S = 15
REST_GAP_S = 2.0                 # same boundary `analyze` uses

# A swim is re-split only when its lengths disagree by more than this ratio. The
# history's 50s split 1.06 : 1 on the median, so this is far outside ordinary
# pacing; 28.8 / 18.4 = 1.57 is what it is there to catch.
RESPLIT_RATIO = 1.25


# --- parsing ----------------------------------------------------------------
IM_ORDER = ("butterfly", "backstroke", "breaststroke", "freestyle")


@dataclass
class Swim:
    """One planned swim: a distance started on its own, of one stroke or of legs."""
    yards: int
    stroke: str | None            # None = choice / unspecified
    line_no: int                  # which plan line it came from; one set per line
    text: str                     # the line, for display
    easy: bool = False
    intervals: list[float] = field(default_factory=list)
    joined: bool = False          # may run straight into the next swim (25 kick + 50 swim)
    last_of_line: bool = False
    # Strokes that change inside the swim, in order: `50 FL/25 BK/25 BR/25 FR`,
    # `25 swim / 25 drill`. Empty when the swim is one stroke (or an even IM).
    legs: list[tuple[int, str | None]] = field(default_factory=list)


_STROKE_WORDS = (
    (r"\bkick", "kick"),
    (r"\bdrill", "drill"),
    (r"\b(?:fly|butterfly)\b", "butterfly"),
    (r"\bback(?:stroke)?\b", "backstroke"),
    (r"\bbreast(?:stroke)?\b", "breaststroke"),
    (r"\b(?:free|freestyle)\b", "freestyle"),
    (r"\bpull\b", "freestyle"),          # a pull set is freestyle unless it says otherwise
    (r"\bIM\b", "IM"),
    # Coaches' shorthand. Case-sensitive, so "fr" inside a word never counts.
    (r"(?-i:\bFL\b)", "butterfly"),
    (r"(?-i:\bBK\b)", "backstroke"),
    (r"(?-i:\bBR\b)", "breaststroke"),
    (r"(?-i:\bFR\b)", "freestyle"),
)


def _stroke_of(text: str, swim_is_free: bool = True) -> str | None:
    """The stroke a fragment names. Kick first: `Kick: Odd Back/Even Breast` is kick.

    A bare `Swim` is freestyle, as a coach means it — set against kick, drill
    and pull, not a choice. Inside an IMO rotation it is whatever the rotation
    says, so the caller turns that off there.
    """
    for pat, name in _STROKE_WORDS:
        if re.search(pat, text, flags=re.I):
            return name
    if swim_is_free and re.search(r"\bswim\b", text, re.I) \
            and not re.search(r"\bchoice\b", text, re.I):
        return "freestyle"
    return None


def _clock(s: str) -> float:
    """`1:10` -> 70, `:55` -> 55."""
    m, _, sec = s.rpartition(":")
    return float(m or 0) * 60 + float(sec)


# Send-offs are written `@1:10`, `on 2:00/2:15`, or just as a column of clocks.
_INTERVALS = re.compile(
    r"(?:(?:@|\bon\b)\s*|(?<=\s))(\d{0,2}:\d{2}(?:\s*/\s*\d{0,2}:\d{2})*)(?!\d)", re.I)
_REPEAT = re.compile(r"^\s*(\d+)\s*[x×]\s*(.*)$", re.I)
# `50`, `50s`, `25's` — the plural is how a coach writes a set of them.
_DIST = re.compile(r"^\s*(\d{2,4})(?:['’]?s(?![a-z]))?\s*(.*)$", re.I)
_THRU = re.compile(r"^\s*(\d+)\s*[x×]\s*thr(?:ough|u)\b", re.I)
_TOTAL = re.compile(r"^\(\s*\d+(?:\s*/\s*\d+)*\s*\)$")
_TRAILING_TOTAL = re.compile(r"\S.*\(\s*\d{3,}(?:\s*/\s*\d+)*\s*\)\s*$")
_HEADING = re.compile(r"^[A-Za-z][^:\d]*:\s*(?=\d)")     # `Cool Down: 1 x 200`
_IMO = re.compile(r"\bIMO\b|\bIM\s+order\b", re.I)


_SHORT_NAME = {"butterfly": "fly", "backstroke": "back",
               "breaststroke": "breast", "freestyle": "free"}


def _legs(desc: str, yards: int, swim_is_free: bool = True) -> list[tuple[int, str | None]]:
    """`50 FL/25 BK/25 BR/25 FR` -> legs, when they add up to the swim.

    Anything that does not sum to the distance is not a leg list — `Kick: Odd
    Back/Even Breast` also has slashes in it.
    """
    desc = desc.lstrip(": ")
    parts = [p for p in re.split(r"\s*/\s*", desc) if p.strip()]
    if len(parts) < 2:
        return []
    out = []
    for p in parts:
        m = _DIST.match(p)
        if not m:
            return []
        out.append((int(m.group(1)), _stroke_of(m.group(2), swim_is_free)))
    return out if sum(y for y, _ in out) == yards else []


def _leg_line(line: str) -> list[tuple[int, str | None]]:
    """A line that is nothing but legs, written under a repeat to spell out each rep."""
    total = 0
    parts = [p for p in re.split(r"\s*/\s*", line) if p.strip()]
    for p in parts:
        m = _DIST.match(p)
        if not m or not m.group(2).strip():
            return []
        total += int(m.group(1))
    return _legs(line, total) if len(parts) >= 2 else []


def _pieces(body: str) -> list[tuple[int, str]]:
    """`(25 Fast Kick + 50 Fast Swim Free)` -> [(25, 'Fast Kick'), (50, '...')]."""
    body = body.strip()
    if body.startswith("("):
        body = body[1:body.index(")")] if ")" in body else body[1:]
    out = []
    for part in body.split("+"):
        m = _DIST.match(part)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


_SWIM_STROKES = set(IM_ORDER)
# `2x (8 x 25) R1-Breast & R2-Fly`: rounds of a repeat, each round its own stroke.
_NESTED = re.compile(r"^\s*(\d+)\s*[x×]\s*\(\s*(\d+)\s*[x×]\s*(\d{2,4})\s*\)\s*(.*)$", re.I)
_ROUND_NAME = re.compile(r"\bR(\d)\s*[-:]\s*([A-Za-z]+)")
# `R1: Odd 1 Pull-2 Kick/Even Breast DPS` — what each rep of round 1 is.
_ROUND_LINE = re.compile(r"^\s*R(\d)\s*:\s*(.*)$")
# `Odd 100 Free/Even 100 IM`, written under the repeat it describes.
_ODD_EVEN_LINE = re.compile(r"^\s*odd\b.*/\s*even\b", re.I)
# `@base + 15"` is a send-off, not a 15-yard swim.
_BASE = re.compile(r"@\s*base\b.*$", re.I)
# `75 IM w/o Free`: a medley of the strokes left, in order. The slash in `w/o`
# must not read as an odd/even pair.
_IM_WITHOUT = re.compile(
    r"\bIM\s+(?:w/o|without|minus|no|less)\s+(fly|butterfly|back(?:stroke)?|"
    r"breast(?:stroke)?|free(?:style)?)\b", re.I)


def _alternation(desc: str, drill: bool = False) -> list[str | None]:
    """`Odd Free/ Even Back` -> ['freestyle', 'backstroke'], else [].

    Only for a pair that names two different things and no distances — a slash
    with yards on both sides is a leg list, and `Kick: Odd Back/Even Breast` is
    a kick set whatever strokes it mentions. Inside a drill set, the half that
    is not a swim stroke (`1 Pull-2 Kick`, `"Stone Skip"`) is the drill.
    """
    parts = [p for p in re.split(r"\s*/\s*", desc) if p.strip()]
    if len(parts) != 2:
        return []
    out = []
    for part in parts:
        part = re.sub(r"^\s*(?:odd|even)\b\s*(?:\d{2,4}\b)?", "", part.strip(), flags=re.I)
        st = _stroke_of(part)
        if drill and st not in _SWIM_STROKES and st != "IM":
            st = "drill"
        out.append(st)
    if None in out or out[0] == out[1]:
        return []
    return out


def _parse_line(line: str, line_no: int) -> list[Swim]:
    line = _BASE.sub("", line)
    intervals = []
    m = _INTERVALS.search(line)
    if m:
        intervals = [_clock(x) for x in re.split(r"\s*/\s*", m.group(1))]
        line_wo = line[:m.start()]
    else:
        line_wo = line
    line_wo = re.sub(r"\(\s*\d+(?:\s*/\s*\d+)*\s*\)\s*$", "", line_wo).strip()
    line_wo = _HEADING.sub("", line_wo)

    rm = _REPEAT.match(line_wo)
    reps, body = (int(rm.group(1)), rm.group(2)) if rm else (1, line_wo)
    # `6 x 50 (25 Fast Free + 25 Fly)`: one 50 in two legs, not two swims.
    dm = re.match(r"^\s*(\d{2,4})\s*(\(.*)$", body)
    if dm and "+" in dm.group(2):
        inner = _pieces(dm.group(2))
        if inner and sum(y for y, _ in inner) == int(dm.group(1)):
            body = dm.group(2)
    pieces = _pieces(body)
    # A compound of nothing but swim strokes — `(75 FR + 25 Fly + 75 FR + 25 BK)`,
    # `(25 Fast Free + 25 Fly)` — is one unbroken swim of legs. One with kick or
    # drill in it (`25 Kick + 50 Swim`) stays separate swims that may run on.
    if len(pieces) >= 2 and all(_stroke_of(d) in _SWIM_STROKES for _, d in pieces):
        legs = [(y, _stroke_of(d)) for y, d in pieces]
        pieces = [(sum(y for y, _ in legs), "/".join(f"{y} {d}" for y, d in pieces))]
    imo = bool(_IMO.search(line_wo))
    # `IMO no fly`: the rotation without the strokes named after "no".
    order = [st for st in IM_ORDER if not re.search(
        r"\bno\s+" + _SHORT_NAME[st] + r"\b", line_wo, re.I)] if imo else []
    drop = _IM_WITHOUT.search(line_wo)
    medley = [st for st in IM_ORDER if st != _stroke_of(drop.group(1))] if drop else []
    swims: list[Swim] = []
    for r in range(reps):
        # IMO: the reps go fly, back, breast, free — 8 x 50 is two of each.
        rot = (order[r * len(order) // reps]
               if imo and order and reps >= len(order) else None)
        for i, (yards, desc) in enumerate(pieces):
            stroke = _stroke_of(desc, swim_is_free=not imo)
            legs = _legs(desc, yards, swim_is_free=not imo)
            if legs:
                legs = [(y, st if st is not None else rot) for y, st in legs]
                strokes = {st for _, st in legs}
                stroke = strokes.pop() if len(strokes) == 1 else None
            elif stroke is None:
                stroke = rot
            if medley and not legs and yards % len(medley) == 0:
                legs = [(yards // len(medley), st) for st in medley]
                stroke = None
            alt = (_alternation(desc) if reps > 1 and not medley
                   and stroke not in ("kick", "drill") else [])
            if alt and not legs:
                stroke = alt[r % 2]
            swims.append(Swim(
                yards=yards, stroke=stroke, line_no=line_no,
                text=line, easy=bool(re.search(r"\b(?:EZ|easy)\b", desc, re.I)),
                intervals=intervals, joined=i < len(pieces) - 1, legs=legs))
    if swims:
        swims[-1].last_of_line = True
    return swims


def parse_plan(text: str) -> list[Swim]:
    """The swims a written workout asks for, in order.

    Understands the way workouts are written on a whiteboard: `5 x 50 Free
    @1:00/1:10/1:15`, `300 Swim`, `4 x (25 Kick + 50 Swim)`, `1 x 200 EZ Choice`,
    `8x50s Drill IMO`, `1x600: 200 Swim/200 Kick/200 Choice`, a `3x125 IM` with
    one `50 FL/25 BK/25 BR/25 FR` line under it per rep, and `2x thru` over a
    block. Headings, set totals like `(900)` and anything else are ignored. A
    line the swimmer did not do is removed by starting it with `#`.
    """
    swims: list[Swim] = []
    block: list[Swim] | None = None       # the swims of an open `Nx thru`
    rounds = 1
    pending: list[Swim] = []              # reps still waiting for their leg line
    last_line: list[Swim] = []            # the previous line's swims, for an Odd/Even line
    rounds_of: dict[int, list[Swim]] = {}  # `R1:` -> that round's reps
    section = ""                          # the heading over the line: `Drill:`

    def close_block():
        nonlocal block
        if block is not None:
            base = list(block)
            for _ in range(rounds - 1):
                swims.extend(Swim(**{**s.__dict__, "legs": list(s.legs),
                                     "intervals": list(s.intervals)}) for s in base)
            block = None

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        t = _THRU.match(line)
        if t:
            close_block()
            block, rounds, pending = [], int(t.group(1)), []
            continue
        if line.endswith(":") and not re.search(r"\d", line):
            section = line
        drill = bool(re.search(r"\bdrill", section, re.I))
        rl = _ROUND_LINE.match(line)
        if rl and int(rl.group(1)) in rounds_of:
            alt = _alternation(rl.group(2), drill=drill)
            for r, sw in enumerate(rounds_of[int(rl.group(1))]):
                if alt:
                    sw.stroke = alt[r % 2]
            continue
        if _ODD_EVEN_LINE.match(line) and last_line:
            alt = _alternation(line, drill=drill)
            for r, sw in enumerate(last_line):
                if alt:
                    sw.stroke = alt[r % 2]
            continue
        nm = _NESTED.match(_BASE.sub("", line))
        if nm:
            k, n, yards, rest = int(nm.group(1)), int(nm.group(2)), int(nm.group(3)), nm.group(4)
            names = {int(i): _stroke_of(w) for i, w in _ROUND_NAME.findall(rest)}
            got, rounds_of = [], {}
            for rnd in range(1, k + 1):
                st = names.get(rnd) or _stroke_of(rest)
                reps_ = [Swim(yards=yards, stroke=st, line_no=line_no, text=line)
                         for _ in range(n)]
                rounds_of[rnd] = reps_
                got.extend(reps_)
            got[-1].last_of_line = True
            swims.extend(got)
            last_line, pending = got, []
            if block is not None:
                block.extend(got)
            continue
        legs = _leg_line(line)
        if legs and pending and sum(y for y, _ in legs) == pending[0].yards:
            s = pending.pop(0)
            s.legs = legs
            strokes = {st for _, st in legs}
            s.stroke = strokes.pop() if len(strokes) == 1 else None
            continue
        if _HEADING.match(line):              # `Cool Down: 1 x 200` is past the block
            close_block()
        if legs:                              # legs on their own: one swim of them
            got = [Swim(yards=sum(y for y, _ in legs), stroke=None, line_no=line_no,
                        text=line, legs=legs, last_of_line=True)]
        else:
            got = _parse_line(line, line_no)
        if not got:
            # A heading or a set total ends a `thru` block; a bullet does not.
            if block is not None and (_TOTAL.match(line) or line.endswith(":")):
                close_block()
            continue
        pending = [s for s in got if s.stroke == "IM" and not s.legs]
        last_line, rounds_of = got, {}
        swims.extend(got)
        if block is not None:
            block.extend(got)
            if _TRAILING_TOTAL.search(line):  # `1x200 IM 3:00 (3000)` ends the block
                close_block()
    close_block()
    return swims


def legs_of(s: Swim) -> list[tuple[int, str | None]]:
    """The swim as legs: its own, an even IM, or one leg of one stroke."""
    if s.legs:
        return s.legs
    if s.stroke == "IM":
        return [(s.yards // 4, st) for st in IM_ORDER]
    return [(s.yards, s.stroke)]


def read_plan_file(path: str | Path) -> str:
    """Plan text from a .txt, or from a PDF via poppler's `pdftotext`."""
    path = Path(path).expanduser()
    if path.suffix.lower() == ".pdf":
        try:
            return subprocess.run(["pdftotext", "-layout", str(path), "-"],
                                  check=True, capture_output=True, text=True).stdout
        except FileNotFoundError as e:
            raise RuntimeError("reading a PDF needs `pdftotext` (brew install poppler); "
                               "or paste the workout into a .txt file") from e
    return path.read_text()


# --- alignment --------------------------------------------------------------
@dataclass
class Segment:
    """A run of Polar lengths and the planned swim it was matched to, if any."""
    swim_no: int | None           # index into the plan, None = unplanned
    idxs: list[int]


def _expected_ratio(swim: Swim, ratios: dict[str, float]) -> tuple[float, float]:
    legs = legs_of(swim)
    base = sum(y * (ratios.get(st) or EXPECTED_RATIO.get(st, 1.12)) for y, st in legs) \
        / sum(y for y, _ in legs)
    if swim.easy:
        base *= EASY_FACTOR
    return base, PACE_SIGMA.get(swim.stroke, DEFAULT_SIGMA)


def _hr_means(start: np.ndarray, end: np.ndarray, hr: np.ndarray | None) -> np.ndarray:
    """Mean heart rate over each length, shifted by the cardiac lag."""
    out = np.full(len(start), np.nan)
    if hr is None or len(hr) < 60:
        return out
    for i, (a, b) in enumerate(zip(start, end)):
        seg = hr[min(int(a) + HR_LAG_S, len(hr) - 1):min(int(b) + HR_LAG_S + 5, len(hr))]
        if len(seg):
            out[i] = float(seg.mean())
    return out


def align(g: pd.DataFrame, swims: list[Swim],
          ratios: dict[str, float] | None = None,
          hr: np.ndarray | None = None) -> list[Segment]:
    """Match one workout's lengths to its plan. `g` is one workout, idx order.

    The DP state also carries whether the previous swim was a joined one (the
    `25 kick` of `25 kick + 50 swim`), because only then may the next swim start
    without a stop in front of it.
    """
    ratios = ratios or {}
    g = g.sort_values("idx")
    idx = g["idx"].to_numpy()
    start = g["start_offset_s"].to_numpy(dtype=float)
    dur = g["duration_s"].to_numpy(dtype=float)
    end = start + dur
    n, m = len(g), len(swims)
    pool_m = float(g["pool_m"].iloc[0])
    pool_yd = pool_m / 0.9144
    norm = REFERENCE_LENGTH_M / pool_m

    gap_before = np.r_[np.inf, start[1:] - end[:-1]]
    rest_start = gap_before > REST_GAP_S           # length i begins after a stop
    rest_after = np.r_[start[1:] - end[:-1], np.inf]
    free_ref = float(np.percentile(dur * norm, 15))
    csum = np.r_[0.0, np.cumsum(dur)]
    stops = np.r_[0, np.cumsum(rest_start[1:].astype(int))]   # stops before i
    hr_len = _hr_means(start, end, hr)
    hr_ok = np.isfinite(hr_len)
    hr_sorted = np.sort(hr_len[hr_ok])

    def hr_pct(i: int, k: int) -> float | None:
        v = hr_len[i:i + k][hr_ok[i:i + k]]
        if not len(v) or not len(hr_sorted):
            return None
        return float(np.searchsorted(hr_sorted, v.mean())) / len(hr_sorted)

    def length_strokes(s: Swim, want: int) -> list[str | None] | None:
        """The planned stroke of each length of a swim of legs, or None."""
        if not s.legs:
            return None
        out: list[str | None] = []
        for y, st in s.legs:
            out.extend([st] * max(1, int(round(y / pool_yd))))
        return out if len(out) == want else None

    def leg_cost(s: Swim, i: int, k: int, want: int) -> float:
        per = length_strokes(s, want)
        if per is None or k != want:
            return 0.0
        exp = np.array([ratios.get(st) or EXPECTED_RATIO.get(st, 1.12) for st in per])
        d = dur[i:i + k]
        # Only a swim that is mostly freestyle has a freestyle pace of its own to
        # measure the other legs against. A floating IM has one free length, and
        # its fly 25s run at about free pace for this swimmer, so there the term
        # was noise — it moved a length between two of the 9/28 floating IMs.
        free = [t for t, st in enumerate(per) if st == "freestyle"]
        if len(free) < max(2, len(per) / 2) or len(free) == len(per):
            return 0.0
        scale = float(np.median(d[free] / exp[free]))
        z = np.log(d / (scale * exp)) / LEG_SIGMA
        return COST_LEG * 0.5 * float(np.minimum(z * z, 16.0).sum()) / len(per) * 2

    def swim_cost(j: int, i: int, k: int) -> float:
        s = swims[j]
        want = max(1, int(round(s.yards / pool_yd)))
        exp, sigma = _expected_ratio(s, ratios)
        # How many real lengths the claimed records cover: a record about twice
        # a length's time is a missed wall, one at a length's time is a length.
        # Short of the distance with no doubled records means the swimmer swam
        # less (or the sensor lost lengths outright) — not that they swam at
        # twice their speed, which is how dividing by the planned count read it.
        one = free_ref * exp / norm
        cover = int(sum(max(1, round(x / one)) for x in dur[i:i + k]))
        cover = min(max(cover, k), want) if k < want else k
        c = COST_COUNT * abs(k - want)
        # stops strictly inside the claimed run, dearer the longer they were
        for t in range(i + 1, i + k):
            if rest_start[t]:
                c += COST_BROKEN + COST_BROKEN_PER_S * max(0.0, gap_before[t] - BROKEN_FREE_S)
        # pace per planned length, so a missed wall does not read as slow swimming
        pace = (csum[i + k] - csum[i]) / cover * norm
        z = math.log(pace / (free_ref * exp)) / sigma
        c += 0.5 * min(z * z, 16.0)
        c += leg_cost(s, i, k, want)
        pct = hr_pct(i, k) if (s.stroke == "butterfly" or s.easy) else None
        if pct is not None:
            if s.stroke == "butterfly":
                c += COST_HR * max(0.0, FLY_HR_PCT - pct)
            if s.easy:
                c += COST_HR * max(0.0, pct - EASY_HR_PCT)
        if s.intervals and not s.last_of_line and np.isfinite(rest_after[i + k - 1]):
            cycle = (end[i + k - 1] - start[i]) + rest_after[i + k - 1]
            off = min(abs(cycle - iv) for iv in s.intervals)
            c += COST_SENDOFF * max(0.0, off - SENDOFF_TOLERANCE_S)
        return c

    INF = float("inf")
    # best[i, j, f]: first i lengths and first j swims accounted for; f = 1 when
    # the last move was a joined swim, so the next may start mid-swim.
    best = np.full((n + 1, m + 1, 2), INF)
    back: dict[tuple[int, int, int], tuple[int, int, int, str]] = {}
    best[0, 0, 0] = 0.0

    def relax(key, v, prev):
        if v < best[key]:
            best[key] = v
            back[key] = prev

    for i in range(n + 1):
        for j in range(m + 1):
            for f in (0, 1):
                b = best[i, j, f]
                if b == INF:
                    continue
                if j < m:                       # the swimmer skipped swim j
                    want = max(1, int(round(swims[j].yards / pool_yd)))
                    relax((i, j + 1, 0), b + COST_SKIP_SWIM * want, (i, j, f, "skip"))
                if i == n:
                    continue
                edge = j == 0 or j == m
                # Unplanned swimming, like a planned swim, begins where the swimmer
                # stopped: cutting a planned swim off mid-run to hand the rest of
                # the run to "unplanned" is a split nobody swam.
                last = back.get((i, j, f), (0, 0, 0, ""))[3]
                cut = (COST_START_MIDSWIM if i and not rest_start[i] and last == "swim"
                       else 0.0)
                if edge:
                    extra = COST_UNPLANNED_EDGE
                elif i and not rest_start[i] and last == "free":
                    extra = COST_UNPLANNED_CONT     # the same unplanned swim, going on
                else:
                    extra = COST_UNPLANNED
                relax((i + 1, j, 0), b + cut + extra, (i, j, f, "free"))
                if j == m:
                    continue
                s_j = swims[j]
                want = max(1, int(round(s_j.yards / pool_yd)))
                spread = 1 if want < 8 else 2
                # As few as half the lengths: a 100 EZ that Polar logged as two
                # lengths (the 2026-09-30 easy swims, both of them) must still be
                # placeable, or the plan slides two lengths to make it fit.
                lo = max(1, min(want - spread, (want + 1) // 2))
                boundary = 0.0 if (rest_start[i] or f) else COST_START_MIDSWIM
                nf = 1 if s_j.joined else 0
                for k in range(lo, want + spread + 1):
                    if i + k > n:
                        break
                    # a joined swim must hand over mid-run, not at a stop
                    if s_j.joined and i + k < n and rest_start[i + k]:
                        pen = COST_BROKEN
                    else:
                        pen = 0.0
                    relax((i + k, j + 1, nf), b + boundary + pen + swim_cost(j, i, k),
                          (i, j, f, "swim"))

    segs: list[Segment] = []
    key = (n, m, int(np.argmin(best[n, m])))
    while key != (0, 0, 0):
        pi, pj, pf, kind = back[key]
        i = key[0]
        if kind == "swim":
            segs.append(Segment(pj, [int(x) for x in idx[pi:i]]))
        elif kind == "free":
            segs.append(Segment(None, [int(idx[pi])]))
        key = (pi, pj, pf)
    segs.reverse()
    # Merge consecutive unplanned lengths into one segment per unbroken run.
    out: list[Segment] = []
    for s in segs:
        if (s.swim_no is None and out and out[-1].swim_no is None):
            out[-1].idxs.extend(s.idxs)
        else:
            out.append(s)
    return out


# --- applying ---------------------------------------------------------------
@dataclass
class PlanReading:
    """What the plan says about each length of one workout."""
    stroke: dict[int, str]            # idx -> stroke, where the plan names one
    factor: dict[int, float]          # idx -> real lengths this record covers
    kind: dict[int, str]              # idx -> 'resplit'
    rep_of: dict[int, int]            # idx -> rep number (one per planned swim)
    set_of: dict[int, int]            # idx -> set number (one per plan line)
    label: dict[int, str]             # idx -> the plan line
    segments: list[Segment]
    # idx -> what to call a swim of several strokes: `IM` for any medley-order
    # swim of all four (a floating IM too), else its strokes, `back/drill`.
    shape: dict[int, str] = field(default_factory=dict)
    # Lengths whose time includes a rest Polar did not see: the plan ends a swim
    # there, nothing joins it to the next, yet Polar logged no gap. On a 25s set
    # on a send-off the swimmer rested 10-20 s at the wall and Polar folded it
    # into the length, so its time is swim + rest and says nothing about pace.
    rest_hidden: set[int] = field(default_factory=set)


def read(g: pd.DataFrame, swims: list[Swim],
         ratios: dict[str, float] | None = None,
         hr: np.ndarray | None = None) -> PlanReading:
    segs = align(g, swims, ratios, hr)
    g = g.set_index("idx")
    pool_yd = float(g["pool_m"].iloc[0]) / 0.9144
    stroke, factor, kind, rep_of, set_of, label = {}, {}, {}, {}, {}, {}
    shape: dict[int, str] = {}

    rep = 0
    set_no = 0
    prev_key = object()
    for seg in segs:
        if seg.swim_no is None:
            # Unplanned: keep the rest structure the data shows.
            prev_rest = None
            for i in seg.idxs:
                if prev_rest is None or g.at[i, "rest_before_s"] > REST_GAP_S:
                    rep += 1
                rep_of[i] = rep
                prev_rest = g.at[i, "rest_before_s"]
                key = ("free", g.at[i, "set_id"])
                if key != prev_key:
                    set_no += 1
                    prev_key = key
                set_of[i] = set_no
            continue

        s = swims[seg.swim_no]
        rep += 1
        key = ("plan", s.line_no)
        if key != prev_key:
            set_no += 1
            prev_key = key
        want = max(1, int(round(s.yards / pool_yd)))
        d = g.loc[seg.idxs, "duration_s"].to_numpy(dtype=float)
        # Even split: each record covers (its time / the swim's mean length time)
        # real lengths, which divides the swim's time evenly and still sums to the
        # distance the plan says was swum.
        per_length = d.sum() / want
        # Re-split only when Polar's count agrees with the plan. When it does not,
        # every record here was a normal single length (a fused one would sit near
        # twice the others and `analyze` repairs it on its own), so the swimmer
        # simply swam a different distance — a 200 cut to 150 in traffic — and
        # rescaling would invent yards nobody swam.
        lopsided = (len(d) == want and len(d) > 1
                    and d.max() / d.min() > RESPLIT_RATIO)
        legs = legs_of(s)
        one_stroke = len(legs) == 1
        # Where each length sits in the swim, in yards: by count when Polar's
        # count agrees with the plan, else by the share of the swim's time.
        if len(d) == want:
            pos = (np.arange(len(d)) + 0.5) / len(d) * s.yards
        else:
            pos = (np.cumsum(d) - d / 2) / d.sum() * s.yards
        bounds = np.cumsum([y for y, _ in legs])
        for i, t, p in zip(seg.idxs, d, pos):
            rep_of[i] = rep
            set_of[i] = set_no
            label[i] = s.text
            leg_stroke = legs[min(int(np.searchsorted(bounds, p, side="right")),
                                  len(legs) - 1)][1]
            if leg_stroke is not None:
                stroke[i] = leg_stroke
            if not one_stroke:
                shape[i] = _shape(legs)
            if lopsided and one_stroke:
                factor[i] = float(t / per_length)
                kind[i] = "resplit"
    hidden = set()
    for a, b in zip(segs, segs[1:]):
        if (a.swim_no is not None and b.swim_no is not None and a.idxs and b.idxs
                and not swims[a.swim_no].joined
                and g.at[b.idxs[0], "rest_before_s"] <= REST_GAP_S):
            hidden.add(a.idxs[-1])
    return PlanReading(stroke, factor, kind, rep_of, set_of, label, segs, shape,
                       hidden)


def _shape(legs: list[tuple[int, str | None]]) -> str:
    """`IM` for fly-back-breast-free in order however the yards fall, else the
    strokes in order: `back/drill`."""
    seq = []
    for _, st in legs:
        if st is not None and (not seq or seq[-1] != st):
            seq.append(st)
    if tuple(seq) == IM_ORDER:
        return "IM"
    return "/".join(_SHORT.get(st, st) for st in seq) or "choice"


def restructure(df: pd.DataFrame, readings: dict[int, PlanReading]) -> pd.DataFrame:
    """Re-cut reps and sets along the plan, for every workout that has one.

    Reps become planned swims and sets become plan lines. Unplanned stretches
    keep the structure the rests imply. Numbers are rewritten for the whole
    workout so rep and set ids stay unique and in time order.
    """
    if not readings:
        return df
    df = df.copy()
    df["rest_hidden"] = False
    for wid, r in readings.items():
        rows = df["workout_id"] == wid
        df.loc[rows, "rep_id"] = [r.rep_of.get(int(i), 0) for i in df.loc[rows, "idx"]]
        df.loc[rows, "set_id"] = [r.set_of.get(int(i), 0) for i in df.loc[rows, "idx"]]
        df.loc[rows, "rest_hidden"] = [int(i) in r.rest_hidden for i in df.loc[rows, "idx"]]
    df["rep_lengths"] = df.groupby(["workout_id", "rep_id"])["idx"].transform("size")
    return df


def repairs(readings: dict[int, PlanReading], df: pd.DataFrame) -> list:
    """The plan's split corrections, as the `Repair` records `analyze` applies."""
    from .analyze import Repair
    out = []
    pace = {(int(w), int(i)): float(p) for w, i, p in
            zip(df["workout_id"], df["idx"], df["pace_s"])}
    for wid, r in readings.items():
        for i, f in r.factor.items():
            out.append(Repair(int(wid), int(i), pace.get((int(wid), int(i)), 0.0),
                              0.0, f, r.kind[i]))
    return out


def labels(readings: dict[int, PlanReading]) -> dict[tuple[int, int], str]:
    return {(int(w), int(i)): s for w, r in readings.items() for i, s in r.stroke.items()}


def load_readings(engine, df: pd.DataFrame,
                  ratios: dict[str, float] | None = None) -> dict[int, PlanReading]:
    """Align every workout in `df` that has a stored plan."""
    from . import db
    from .analyze import load_hr
    plans = db.load_plans(engine, sorted(df["workout_id"].unique().tolist()))
    hr = load_hr(engine, sorted(plans)) if plans else {}
    out = {}
    for wid, text in plans.items():
        swims = parse_plan(text)
        g = df[df["workout_id"] == wid]
        if swims and not g.empty:
            out[wid] = read(g, swims, ratios, hr.get(wid))
    return out


def without_planned(matches: list, readings: dict[int, PlanReading]) -> list:
    """Medley-pattern matches that touch no planned swim.

    A pattern match can re-cut a rep into rungs, and on a planned swim that
    undoes the plan: a 150 free warm-up came out as a 50 and a 100 of fly/back.
    The plan already says what the swim was, so the guess is not wanted there.
    """
    planned = {(int(w), i) for w, r in readings.items()
               for seg in r.segments if seg.swim_no is not None for i in seg.idxs}
    return [m for m in matches
            if not any((int(m.workout_id), int(i)) in planned for i in m.idxs)]


def shapes(readings: dict[int, PlanReading]) -> dict[tuple[int, int], str]:
    return {(int(w), int(i)): s for w, r in readings.items() for i, s in r.shape.items()}


def apply_labels(df: pd.DataFrame, plan_labels: dict[tuple[int, int], str],
                 plan_shapes: dict[tuple[int, int], str] | None = None) -> pd.DataFrame:
    """Stamp the planned stroke over every inference.

    The swimmer's own word about the swim, so it outranks the rules, the model,
    medley detection and rep consistency — and is itself outranked only by a
    hand correction, which is applied after this.
    """
    df = df.copy()
    if "label_source" not in df.columns:
        df["label_source"] = "rules"
    if not plan_labels:
        return df
    key = list(zip(df["workout_id"], df["idx"]))
    hit = [k in plan_labels for k in key]
    if not any(hit):
        return df
    df.loc[hit, "predicted"] = [plan_labels[k] for k in key if k in plan_labels]
    df.loc[hit, "confidence"] = 0.95
    df.loc[hit, "label_source"] = "plan"
    if "pattern" in df.columns:
        df.loc[hit, "pattern"] = None
    if plan_shapes:
        # A swim of several strokes is named for its shape, and one of all four is
        # a medley: kept out of every single-stroke ranking.
        named = [k in plan_shapes for k in key]
        for col, default in (("pattern", None), ("im_continuous", False),
                             ("mixed_rep", False)):
            if col not in df.columns:
                df[col] = default
        df["pattern"] = df["pattern"].astype(object)
        df.loc[named, "pattern"] = [plan_shapes[k] for k in key if k in plan_shapes]
        df.loc[named, "im_continuous"] = [plan_shapes[k] == "IM" for k in key
                                          if k in plan_shapes]
        df.loc[named, "mixed_rep"] = True
    return df


def merge_repairs(generic: list, planned: list) -> list:
    """The plan's re-splits replace any generic repair on the same length."""
    taken = {(r.workout_id, r.idx) for r in planned}
    return [r for r in generic if (r.workout_id, r.idx) not in taken] + planned


_SHORT = {"freestyle": "free", "butterfly": "fly", "backstroke": "back",
          "breaststroke": "breast", "kick": "kick", "drill": "drill", "IM": "IM",
          None: "choice"}


def describe(g: pd.DataFrame, swims: list[Swim], reading: PlanReading) -> str:
    """The alignment as a table a swimmer can check against their memory."""
    g = g.set_index("idx")
    pool_yd = float(g["pool_m"].iloc[0]) / 0.9144
    lines = []
    for seg in reading.segments:
        t = g.loc[seg.idxs, "duration_s"].to_numpy(dtype=float)
        span = f"{seg.idxs[0]:>3}-{seg.idxs[-1]:<3}"
        if seg.swim_no is None:
            name = "(not in the plan)"
        else:
            s = swims[seg.swim_no]
            what = ("/".join(f"{y} {_SHORT.get(st, st)}" for y, st in s.legs)
                    if s.legs else _SHORT.get(s.stroke, s.stroke))
            name = f"{s.yards} {what}{' EZ' if s.easy else ''}"
            want = max(1, int(round(s.yards / pool_yd)))
            if len(seg.idxs) != want:
                name += f"  [Polar: {len(seg.idxs)} lengths]"
        fixed = " re-split" if any(i in reading.factor for i in seg.idxs) else ""
        lines.append(f"{span} {name:<34} {t.sum():6.1f}s  "
                     + " ".join(f"{x:.1f}" for x in t) + fixed)
    return "\n".join(lines)


def training_labels(readings: dict[int, PlanReading]) -> dict[tuple[int, int], str]:
    """Planned strokes fit to train the correction model — or none at all.

    A plan labels whole workouts, and one workout is mostly freestyle: the first
    planned swim fitted a model on 44 free and 8 back lengths and it re-labelled
    2,228 lengths across the history, nearly all toward freestyle, erasing every
    kick and drill set. So planned labels train the model only once they cover
    all four strokes with a usable number of each. Until then they correct only
    the workout they describe.
    """
    from .learn import MIN_LABELS_PER_CLASS
    got = labels(readings)
    counts = pd.Series(list(got.values()), dtype=object).value_counts()
    strokes = ("freestyle", "backstroke", "breaststroke", "butterfly")
    if all(counts.get(s, 0) >= MIN_LABELS_PER_CLASS for s in strokes):
        return got
    return {}
