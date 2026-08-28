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

# A character budget, not a display width: the coloured squares are one character
# each here but render double-width nearly everywhere, so the visual line is a
# little wider than this number. Raised from 34 when each row gained its zone,
# speed and pace fields — the old rows carried a bar and a time and nothing else.
WIDTH = 42
BLOCKS = "▁▂▃▄▅▆▇█"
BAR = "▇"

STROKE_GLYPH = {
    "freestyle": "free", "backstroke": "back", "breaststroke": "brst",
    "butterfly": "fly ", "other": "drll", "undetermined": "  ? ", "IM": "IM  ",
}

# Strava descriptions are plain text — no markdown, no HTML, no ANSI colour. Emoji
# are the only characters that render in colour, so the palette is built from the
# coloured-square set. These are also double-width in most renderers, which is why
# exactly one appears per line: a uniform shift preserves the column alignment.
STROKE_COLOR = {
    "freestyle": "🟦", "backstroke": "🟩", "breaststroke": "🟧",
    "butterfly": "🟪", "other": "⬜", "undetermined": "⬛", "IM": "🟨",
}
MIX_WIDTH = 12          # squares in the stacked bar; 12 keeps it inside a phone

# Zone colours as emoji. Deliberately a different family from the stroke palette
# where possible, and both bars are labelled, so the two are never confused.
# Circles, not squares. The stroke palette is squares, and when both used them a
# blue square meant freestyle in one column and Z2 in another.
ZONE_COLOR = {"Z1": "⚪", "Z2": "🔵", "Z3": "🟢", "Z4": "🟠", "Z5": "🔴"}


def _fmt_clock(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def _fmt_rep(seconds: float) -> str:
    """Rep time: bare seconds under a minute, m:ss above — how a swimmer reads it."""
    return f"{seconds:.0f}s" if seconds < 60 else _fmt_clock(seconds)


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


def _pool_yards(header: dict, df: pd.DataFrame) -> int:
    """Pool length in yards, from the header or recovered from the lengths."""
    metres = header.get("pool_length_m")
    if not metres and "pool_m" in df.columns and df["pool_m"].notna().any():
        metres = float(df["pool_m"].iloc[0])
    return int(round((metres or 22.86) / 0.9144))


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


def mix_legend(df: pd.DataFrame, pool_yd: int = 25) -> list[str]:  # noqa: D401
    """One line per stroke: colour, name, distance, share."""
    return [f"{STROKE_COLOR.get(k, '⬛')} {STROKE_GLYPH.get(k, k).strip():<5} "
            f"{n * pool_yd:>5} yd  {pct:>4.0f}%"
            for k, n, pct in stroke_mix(df)]


def zone_bar(zone_time: list[dict], width: int = MIX_WIDTH) -> str:
    """Time-in-zone as a proportional stacked bar, same idea as the stroke mix."""
    total = sum(z.get("seconds", 0) for z in zone_time)
    if not total:
        return ""
    counts = {}
    for z in zone_time:
        share = z["seconds"] / total
        if share > 0:
            counts[z["zone"]] = max(1, round(share * width))
    while counts and sum(counts.values()) != width:
        biggest = max(counts, key=counts.get)
        counts[biggest] += 1 if sum(counts.values()) < width else -1
        if counts[biggest] <= 0:
            del counts[biggest]
    order = [z["zone"] for z in zone_time]
    return "".join(ZONE_COLOR.get(k, "⬜") * counts[k]
                   for k in order if k in counts)


def set_card(df: pd.DataFrame, header: dict, sets: list[dict] | None = None,
             hr_series=None) -> str:
    """Per-set summary card.

    Rows are built from the same `sets_for_workout` derivation the dashboard
    table uses, so the card cannot disagree with the screen it is meant to
    mirror — which is how three 100 IMs once appeared as 300 yards of backstroke
    in one place and 75 in another.

    The old pace bar is gone. It was drawn relative to the fastest and slowest
    length in that one workout, so it had no scale a reader could hold in their
    head: four blocks meant nothing except "compared to the rest of today". The
    columns that replaced it — effort zone, speed percentile, pace per 50 — each
    mean something on their own.

    Fields are separated by `·` rather than aligned in columns. Strava renders
    descriptions in a proportional font, where space-padded columns drift out of
    line on a phone, and a row that has drifted is unreadable in a way a row of
    labelled fields is not.
    """
    lines: list[str] = []
    date = str(header.get("start_time", ""))[:10]
    dist_yd = round((header.get("distance_m") or 0) / 0.9144)
    dur = _fmt_clock(header.get("duration_s") or 0)
    avg_hr = header.get("avg_hr")

    lines.append(f"🏊 {date}   {dist_yd:,} yd   {dur}")
    second = f"   {len(df)} lengths" + (f" · avg {avg_hr} bpm" if avg_hr else "")
    effort = header.get("effort") or {}
    if effort.get("score") is not None:
        second += f" · load {effort['score']}"
        if effort.get("intensity") is not None:
            second += f"/int {effort['intensity']}"
    lines.append(second)

    if sets is None:
        sets = _sets(df, header)

    # Name the fields. Without this the card is five unlabelled numbers per row,
    # which is the whole reason it was unreadable.
    lines += ["", "set · time · zone · speed · pace/50"]

    for row in sets:
        label = STROKE_GLYPH.get(row["stroke"], row["stroke"][:4]).strip()
        square = STROKE_COLOR.get(row["stroke"], "⬛")
        parts = [f"{square} {row['reps']}×{row['rep_yards']} {label}",
                 _fmt_rep(row["rep_seconds"])]

        zone = row.get("hr_zone")
        parts.append(f"{ZONE_COLOR.get(zone['zone'], '⚪')}{zone['zone']}"
                     if zone else "—")

        speed = row.get("speed")
        # A distance with too few comparable reps cannot be ranked honestly, so
        # it says so rather than inventing a percentile.
        parts.append(f"{speed['percentile']}%" if speed else "—")
        parts.append(f"{row['pace_50_s']:.0f}s")

        line = " · ".join(parts)
        if row.get("pr"):
            line += " ★"
        lines.append(line)

    return "\n".join(lines)


def _sets(df: pd.DataFrame, header: dict) -> list[dict]:
    """Fall back to deriving the set rows when the caller has none to hand."""
    from . import report
    return report.sets_for_workout(df)


def strava_block(df: pd.DataFrame, header: dict,
                 sets: list[dict] | None = None) -> str:
    """The full paste-ready block: the set table, then the two keys it needs.

    The per-length sparkline that used to sit under the table is gone for the
    same reason as the pace bar — it was a shape with no scale. Everything left
    here is either a number or the key to reading one.
    """
    parts = [set_card(df, header, sets)]

    # Stroke colours, doubling as the distance breakdown.
    legend = mix_legend(df)
    if legend:
        parts += [""] + legend

    zone_time = header.get("zone_time") or []
    active = [z for z in zone_time if z.get("seconds")]
    if active:
        parts += ["", "  ".join(
            f"{ZONE_COLOR.get(z['zone'], '⚪')}{z['zone']} {z['pct']:.0f}%"
            for z in active)]
    else:
        parts += ["", "  ".join(f"{c}{z}" for z, c in ZONE_COLOR.items())]
    parts += ["speed = your own percentile at that distance",
              "★ = personal best"]

    parts += ["", "— polarswim"]
    return "\n".join(parts)
