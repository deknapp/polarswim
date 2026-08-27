"""Render a workout summary as a self-contained SVG, for uploading to Strava.

Strava accepts photos on an activity, so the analysis that cannot survive a
plain-text description can go up as an image instead. The graphic is built as SVG
rather than drawn with a plotting library for three reasons: it adds no dependency,
it is deterministic and therefore testable as text, and the browser can rasterise it
to PNG through a canvas without any external script.

Layout is fixed-column rather than measured, because text width cannot be computed
server-side. Every glyph is plain `<text>` — no `foreignObject`, no web fonts —
since both taint a canvas and would block the PNG export.
"""

from __future__ import annotations

import pandas as pd

W = 1080                      # a comfortable width in the Strava feed
PAD = 40
ROW_H = 38
HEAD_H = 210
FOOT_H = 76

BG = "#0f1419"
PANEL = "#171d24"
LINE = "#252d36"
FG = "#e6edf3"
DIM = "#8b949e"
ACCENT = "#4aa3ff"
PR_GOLD = "#f0a848"

STROKE_LABEL = {
    "freestyle": "free", "backstroke": "back", "breaststroke": "breast",
    "butterfly": "fly", "other": "drill/kick", "undetermined": "unknown",
}


def _esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _fmt_time(seconds: float) -> str:
    seconds = float(seconds or 0)
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def _pie(mix: list[dict], cx: float, cy: float, r: float) -> str:
    """Donut of the stroke mix. A single 100% slice needs a circle, not an arc."""
    import math
    if not mix:
        return ""
    parts = []
    if len(mix) == 1:
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{mix[0]["color"]}"/>')
    else:
        a0 = -math.pi / 2
        for m in mix:
            a1 = a0 + 2 * math.pi * m["pct"] / 100.0
            large = 1 if (a1 - a0) > math.pi else 0
            x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
            x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
            parts.append(
                f'<path d="M {cx:.1f} {cy:.1f} L {x0:.1f} {y0:.1f} '
                f'A {r} {r} 0 {large} 1 {x1:.1f} {y1:.1f} Z" '
                f'fill="{m["color"]}" stroke="{BG}" stroke-width="2"/>')
            a0 = a1
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r * 0.55}" fill="{BG}"/>')
    return "".join(parts)


def workout_svg(header: dict, sets: list[dict], mix: list[dict],
                zones: list[dict], hr_max: int, df: pd.DataFrame | None = None) -> str:
    """Full summary graphic: totals, stroke mix, and the set table."""
    height = HEAD_H + max(1, len(sets)) * ROW_H + FOOT_H
    date = str(header.get("start_time", ""))[:10]
    yards = round((header.get("distance_m") or 0) / 0.9144)
    dur = _fmt_time(header.get("duration_s") or 0)
    avg_hr = header.get("avg_hr")

    o: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
        f'viewBox="0 0 {W} {height}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{height}" fill="{BG}"/>',
    ]

    # --- header
    o.append(f'<text x="{PAD}" y="62" fill="{FG}" font-size="34" '
             f'font-weight="700">{_esc(date)}</text>')
    o.append(f'<text x="{PAD}" y="104" fill="{ACCENT}" font-size="26">'
             f'{yards:,} yd · {dur}'
             + (f' · avg {avg_hr} bpm' if avg_hr else '') + '</text>')
    o.append(f'<text x="{PAD}" y="138" fill="{DIM}" font-size="16">'
             f'{sum(s["n"] for s in sets)} lengths · '
             f'{len(sets)} sets · zones from a {hr_max} bpm swim max</text>')

    # --- stroke mix, top right. The donut sits hard against the right edge and the
    # legend to its left, with a gap: an earlier layout let the legend's value
    # column run over the donut and paint it out entirely.
    pie_cx, pie_r = W - 92, 58
    legend_x = W - 400                      # swatch
    legend_gap = pie_cx - pie_r - 24        # nearest the legend may extend
    o.append(_pie(mix, pie_cx, 100, pie_r))
    ly = 46
    for m in mix[:6]:
        o.append(f'<rect x="{legend_x}" y="{ly - 10}" width="12" height="12" '
                 f'rx="2" fill="{m["color"]}"/>')
        o.append(f'<text x="{legend_x + 20}" y="{ly}" fill="{FG}" font-size="15">'
                 f'{_esc(STROKE_LABEL.get(m["stroke"], m["stroke"]))}</text>')
        o.append(f'<text x="{legend_x + 118}" y="{ly}" fill="{DIM}" font-size="15" '
                 f'text-anchor="end">'
                 f'<tspan x="{legend_gap}" text-anchor="end">'
                 f'{m["yards"]} yd · {m["pct"]:.0f}%</tspan></text>')
        ly += 21

    # --- table header
    y = HEAD_H - 22
    cols = [(PAD, "set"), (PAD + 120, "stroke"), (PAD + 280, "time"),
            (PAD + 390, "zone"), (PAD + 530, "speed"), (PAD + 720, "pace/25"),
            (PAD + 900, "rest")]
    for x, label in cols:
        o.append(f'<text x="{x}" y="{y}" fill="{DIM}" font-size="14" '
                 f'letter-spacing="1">{label.upper()}</text>')
    o.append(f'<line x1="{PAD}" y1="{y + 12}" x2="{W - PAD}" y2="{y + 12}" '
             f'stroke="{LINE}" stroke-width="1"/>')

    # --- rows
    paces = [s["pace_s"] for s in sets if s.get("pace_s")]
    slowest = max(paces) if paces else 1.0
    y = HEAD_H + 18
    for i, s in enumerate(sets):
        if i % 2 == 0:
            o.append(f'<rect x="{PAD - 10}" y="{y - 24}" width="{W - 2 * PAD + 20}" '
                     f'height="{ROW_H}" rx="5" fill="{PANEL}"/>')
        o.append(f'<text x="{PAD}" y="{y}" fill="{FG}" font-size="19" '
                 f'font-weight="600">{s["reps"]}×{s["rep_yards"]}</text>')
        o.append(f'<text x="{PAD + 120}" y="{y}" fill="{FG}" font-size="17">'
                 f'{_esc(STROKE_LABEL.get(s["stroke"], s["stroke"]))}</text>')
        if s.get("confidence", 1) < 0.4:
            o.append(f'<text x="{PAD + 232}" y="{y}" fill="{PR_GOLD}" '
                     f'font-size="13">?</text>')
        o.append(f'<text x="{PAD + 280}" y="{y}" fill="{FG}" font-size="17">'
                 f'{_fmt_time(s.get("rep_seconds", 0))}</text>')

        zone = s.get("hr_zone")
        if zone:
            o.append(f'<rect x="{PAD + 390}" y="{y - 15}" width="34" height="20" '
                     f'rx="4" fill="{zone["color"]}"/>')
            o.append(f'<text x="{PAD + 407}" y="{y}" fill="#08131f" font-size="13" '
                     f'font-weight="700" text-anchor="middle">{zone["zone"]}</text>')
            o.append(f'<text x="{PAD + 433}" y="{y}" fill="{DIM}" font-size="14">'
                     f'{zone["pct_max"]}%</text>')

        speed = s.get("speed")
        if speed:
            o.append(f'<rect x="{PAD + 530}" y="{y - 12}" width="110" height="10" '
                     f'rx="5" fill="{LINE}"/>')
            o.append(f'<rect x="{PAD + 530}" y="{y - 12}" '
                     f'width="{110 * speed["percentile"] / 100:.0f}" height="10" '
                     f'rx="5" fill="{speed["color"]}"/>')
            o.append(f'<text x="{PAD + 650}" y="{y}" fill="{DIM}" font-size="14">'
                     f'{speed["percentile"]}%</text>')

        pace = s.get("pace_s") or 0
        bar_w = max(6, 70 * (1 - pace / slowest) + 8)
        o.append(f'<rect x="{PAD + 720}" y="{y - 12}" width="{bar_w:.0f}" height="10" '
                 f'rx="5" fill="{ACCENT}" opacity="0.75"/>')
        o.append(f'<text x="{PAD + 800}" y="{y}" fill="{DIM}" font-size="14">'
                 f'{pace:.0f}s</text>')
        o.append(f'<text x="{PAD + 900}" y="{y}" fill="{DIM}" font-size="15">'
                 f'{s.get("rest_before_s", 0):.0f}s</text>')
        if s.get("pr"):
            o.append(f'<text x="{PAD + 960}" y="{y}" fill="{PR_GOLD}" font-size="16" '
                     f'font-weight="700">★ PR</text>')
        y += ROW_H

    # --- footer: the zone key, and an honest note about the labels
    fy = height - FOOT_H + 26
    x = PAD
    for z in zones:
        o.append(f'<rect x="{x}" y="{fy - 12}" width="26" height="16" rx="3" '
                 f'fill="{z["color"]}"/>')
        o.append(f'<text x="{x + 13}" y="{fy}" fill="#08131f" font-size="11" '
                 f'font-weight="700" text-anchor="middle">{z["zone"]}</text>')
        o.append(f'<text x="{x + 32}" y="{fy}" fill="{DIM}" font-size="13">'
                 f'{z["low"]}-{z["high"]}</text>')
        x += 118
    o.append(f'<text x="{PAD}" y="{fy + 26}" fill="{DIM}" font-size="12">'
             f'stroke inferred from pace and heart rate — not measured by the '
             f'sensor · polarswim</text>')

    o.append("</svg>")
    return "".join(o)
