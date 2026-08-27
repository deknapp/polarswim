"""Unicode workout cards, sized to paste into a Strava description.

Strava descriptions are plain text — no markdown, no colour, and read on a phone —
so the constraints are real: a fixed narrow width, block and box-drawing glyphs
only, and no reliance on alignment surviving a proportional font (Strava renders
monospace-ish, but we keep bars left-anchored so drift doesn't break the shape).

Two views:
    set_card      one line per set: distance, stroke, a pace bar, and mean HR
    length_chart  one column per length, a sparkline of the whole practice
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WIDTH = 34                       # comfortable on a phone in the Strava app
BLOCKS = "▁▂▃▄▅▆▇█"
BAR = "▇"

STROKE_GLYPH = {
    "freestyle": "free", "backstroke": "back", "breaststroke": "brst",
    "butterfly": "fly ", "other": "drll", "undetermined": "  ? ",
}


def _fmt_clock(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def sparkline(values: list[float] | np.ndarray, invert: bool = False) -> str:
    """Map values onto block glyphs. `invert` so faster (lower) reads taller."""
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return ""
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return BLOCKS[len(BLOCKS) // 2] * len(v)
    scaled = (v - lo) / (hi - lo)
    if invert:
        scaled = 1.0 - scaled
    idx = np.clip((scaled * (len(BLOCKS) - 1)).round().astype(int), 0, len(BLOCKS) - 1)
    return "".join(BLOCKS[i] for i in idx)


def _bar(value: float, lo: float, hi: float, width: int) -> str:
    """Horizontal bar; faster pace draws longer, so the eye reads speed."""
    if hi - lo < 1e-9:
        return BAR * (width // 2)
    frac = 1.0 - (value - lo) / (hi - lo)          # invert: low pace = long bar
    return BAR * max(1, int(round(np.clip(frac, 0, 1) * width)))


def set_card(df: pd.DataFrame, header: dict, hr_series=None) -> str:
    """Per-set summary card.

    `df` is one workout's lengths with `set_id`, `pace_s`, `predicted`, and
    (optionally) `hr_cost` already attached.
    """
    lines: list[str] = []
    date = str(header.get("start_time", ""))[:10]
    dist_yd = round((header.get("distance_m") or 0) / 0.9144)
    dur = _fmt_clock(header.get("duration_s") or 0)
    avg_hr = header.get("avg_hr")

    lines.append("┏" + "━" * (WIDTH - 2) + "┓")
    title = f"🏊 {date}  {dist_yd:,} yd  {dur}"
    lines.append("┃" + title.ljust(WIDTH - 2)[:WIDTH - 2] + "┃")
    sub = f"   {len(df)} lengths" + (f"  ·  avg {avg_hr} bpm" if avg_hr else "")
    lines.append("┃" + sub.ljust(WIDTH - 2)[:WIDTH - 2] + "┃")
    lines.append("┡" + "━" * (WIDTH - 2) + "┩")

    paces = df["pace_s"].dropna()
    lo, hi = (float(paces.min()), float(paces.max())) if len(paces) else (0.0, 1.0)

    for sid, g in df.groupby("set_id"):
        n = len(g)
        med = float(g["pace_s"].median())
        stroke = g["predicted"].mode()
        label = STROKE_GLYPH.get(stroke.iloc[0] if len(stroke) else "undetermined", "  ? ")
        bar = _bar(med, lo, hi, 8)
        body = f"{n:>3}×25 {label} {bar:<8} {med:>3.0f}s"
        lines.append("│" + body.ljust(WIDTH - 2)[:WIDTH - 2] + "│")

    lines.append("└" + "─" * (WIDTH - 2) + "┘")
    return "\n".join(lines)


def length_chart(df: pd.DataFrame, per_row: int = 30) -> str:
    """Every length as one glyph — the shape of the whole practice."""
    paces = df.sort_values("idx")["pace_s"].to_numpy()
    rows = []
    for start in range(0, len(paces), per_row):
        chunk = paces[start:start + per_row]
        rows.append(f"{start + 1:>3} {sparkline(chunk, invert=True)}")
    return "\n".join(rows)


def strava_block(df: pd.DataFrame, header: dict) -> str:
    """The full paste-ready block: card, length chart, and a stroke tally."""
    parts = [set_card(df, header), "", "pace by length (taller = faster)",
             length_chart(df)]

    tally = df["predicted"].value_counts()
    if len(tally):
        yards = {k: int(v) * 25 for k, v in tally.items()}
        parts += ["", "  ".join(f"{STROKE_GLYPH.get(k, k).strip()} {v}yd"
                                for k, v in yards.items())]
    parts += ["", "— polarswim"]
    return "\n".join(parts)
