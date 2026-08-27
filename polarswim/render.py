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

# Strava descriptions are plain text — no markdown, no HTML, no ANSI colour. Emoji
# are the only characters that render in colour, so the palette is built from the
# coloured-square set. These are also double-width in most renderers, which is why
# exactly one appears per line: a uniform shift preserves the column alignment.
STROKE_COLOR = {
    "freestyle": "🟦", "backstroke": "🟩", "breaststroke": "🟧",
    "butterfly": "🟪", "other": "⬜", "undetermined": "⬛",
}
MIX_WIDTH = 12          # squares in the stacked bar; 12 keeps it inside a phone


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


def stroke_mix(df: pd.DataFrame) -> list[tuple[str, int, float]]:
    """(stroke, lengths, percent) ordered biggest first."""
    counts = df["predicted"].value_counts()
    total = int(counts.sum()) or 1
    return [(k, int(v), 100.0 * v / total) for k, v in counts.items()]


def mix_bar(df: pd.DataFrame, width: int = MIX_WIDTH) -> str:
    """Stroke mix as one proportional stacked bar of coloured squares.

    A pie cannot be drawn in text, but a stacked bar carries the same proportions
    and survives a plain-text field. Largest share first, and every stroke that
    appears at all gets at least one square so nothing silently vanishes.
    """
    mix = stroke_mix(df)
    if not mix:
        return ""
    counts = {k: max(1, round(pct / 100 * width)) for k, _, pct in mix}
    # Trim or pad the largest share so the bar is exactly `width` squares.
    while sum(counts.values()) != width and counts:
        biggest = max(counts, key=counts.get)
        counts[biggest] += 1 if sum(counts.values()) < width else -1
        if counts[biggest] <= 0:
            del counts[biggest]
    return "".join(STROKE_COLOR.get(k, "⬛") * n for k, n in counts.items())


def mix_legend(df: pd.DataFrame, pool_yd: int = 25) -> list[str]:
    """One line per stroke: colour, name, distance, share."""
    return [f"{STROKE_COLOR.get(k, '⬛')} {STROKE_GLYPH.get(k, k).strip():<5} "
            f"{n * pool_yd:>5} yd  {pct:>4.0f}%"
            for k, n, pct in stroke_mix(df)]


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

    lines.append(f"🏊 {date}   {dist_yd:,} yd   {dur}")
    lines.append(f"   {len(df)} lengths" + (f"   ·   avg {avg_hr} bpm" if avg_hr else ""))
    bar = mix_bar(df)
    if bar:
        lines += ["", bar]
    lines.append("─" * (WIDTH - 2))

    paces = df["pace_s"].dropna()
    lo, hi = (float(paces.min()), float(paces.max())) if len(paces) else (0.0, 1.0)

    for sid, g in df.groupby("set_id"):
        n = len(g)
        med = float(g["pace_s"].median())
        stroke = g["predicted"].mode()
        label = STROKE_GLYPH.get(stroke.iloc[0] if len(stroke) else "undetermined", "  ? ")
        stroke = g["predicted"].mode()
        key = stroke.iloc[0] if len(stroke) else "undetermined"
        bar = _bar(med, lo, hi, 8)
        lines.append(f"{STROKE_COLOR.get(key, '⬛')} {n:>2}×25 {label} "
                     f"{bar:<8} {med:>3.0f}s")

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

    legend = mix_legend(df)
    if legend:
        parts += [""] + legend
    parts += ["", "— polarswim"]
    return "\n".join(parts)
