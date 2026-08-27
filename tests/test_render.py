"""The Strava card: shape, width, and safety on odd input."""

import pandas as pd
import pytest

from polarswim import render


@pytest.fixture
def one_workout():
    """Three unbroken reps of four lengths each — i.e. 3x100 in a 25 yd pool."""
    rows = []
    for i in range(1, 13):
        rows.append(dict(idx=i, rep_id=1 + (i - 1) // 4, set_id=1,
                         rep_lengths=4, duration_s=24.0 + (i % 4) * 3,
                         pace_s=24.0 + (i % 4) * 3, pool_m=22.86,
                         predicted="freestyle" if i % 4 else "breaststroke",
                         confidence=0.7, hr_cost=30.0))
    return pd.DataFrame(rows)


HEADER = {"start_time": "2026-08-19T17:11:11", "distance_m": 1394.46,
          "duration_s": 2822.0, "avg_hr": 126}


SQUARES = set(render.STROKE_COLOR.values())


def _set_rows(card: str) -> list[str]:
    """The per-set lines only — not the header, the mix bar, or the rule."""
    return [l for l in card.splitlines() if l[:1] in SQUARES and "×" in l]


def test_set_rows_share_one_width(one_workout):
    """Ragged set rows are what make a pasted card look broken in Strava."""
    rows = _set_rows(render.set_card(one_workout, HEADER))
    assert rows and len({len(r) for r in rows}) == 1


def test_every_set_row_carries_exactly_one_colour(one_workout):
    """Emoji are double-width, so a uniform one-per-row shift preserves columns."""
    for row in _set_rows(render.set_card(one_workout, HEADER)):
        assert len([c for c in row if c in SQUARES]) == 1


def test_set_row_colour_matches_its_stroke(one_workout):
    card = render.set_card(one_workout, HEADER)
    for row in _set_rows(card):
        label = row.split("×")[1].split(maxsplit=1)[1].split()[0]
        expected = next(k for k, v in render.STROKE_GLYPH.items()
                        if v.strip() == label)
        assert row[0] == render.STROKE_COLOR[expected]


def test_card_fits_a_phone(one_workout):
    for line in render.set_card(one_workout, HEADER).splitlines():
        assert len(line) <= render.WIDTH


def test_card_reports_real_totals(one_workout):
    card = render.set_card(one_workout, HEADER)
    assert "2026-08-19" in card and "1,525 yd" in card and "12 lengths" in card


def test_card_reports_reps_not_raw_lengths(one_workout):
    """Four unbroken lengths of a 25 yd pool is a 100, not four 25s."""
    card = render.set_card(one_workout, HEADER)
    assert "3×100" in card
    assert "12×25" not in card


def test_rep_time_is_the_sum_of_its_lengths(one_workout):
    """A 100 shows its 100 time, not the time of one length inside it."""
    card = render.set_card(one_workout, HEADER)
    expected = one_workout.groupby("rep_id")["duration_s"].sum().median()
    assert render._fmt_rep(expected) in card


def test_pool_length_is_read_from_the_data():
    """A 50 m pool must report 50 m reps, not 25 yd ones."""
    df = pd.DataFrame([dict(idx=i, rep_id=1, set_id=1, rep_lengths=2,
                            duration_s=45.0, pace_s=22.5, pool_m=45.72,
                            predicted="freestyle", confidence=0.7, hr_cost=20.0)
                       for i in (1, 2)])
    card = render.set_card(df, {"start_time": "2026-01-01", "distance_m": 91.44,
                                "duration_s": 90.0, "avg_hr": 120})
    assert "1×100" in card


def test_sparkline_length_matches_input():
    assert len(render.sparkline([1, 2, 3, 4, 5])) == 5


def test_sparkline_inverts_so_faster_reads_taller():
    fast_first = render.sparkline([10.0, 30.0], invert=True)
    assert fast_first[0] > fast_first[1]


def test_sparkline_handles_constant_input():
    assert len(render.sparkline([7.0] * 4)) == 4


def test_sparkline_handles_empty_input():
    assert render.sparkline([]) == ""


def test_mix_bar_is_exactly_the_declared_width(one_workout):
    bar = render.mix_bar(one_workout)
    assert len([c for c in bar if c in set(render.STROKE_COLOR.values())]) \
        == render.MIX_WIDTH


def test_mix_bar_shows_every_stroke_present(one_workout):
    """A stroke that was actually swum must never round away to nothing."""
    bar = render.mix_bar(one_workout)
    for stroke in one_workout["predicted"].unique():
        assert render.STROKE_COLOR[stroke] in bar


def test_mix_bar_is_empty_for_no_data():
    assert render.mix_bar(pd.DataFrame({"predicted": []})) == ""


def test_legend_percentages_sum_to_a_hundred(one_workout):
    total = sum(pct for _, _, pct in render.stroke_mix(one_workout))
    assert total == pytest.approx(100.0)


def test_legend_covers_every_stroke_in_the_bar(one_workout):
    legend = "\n".join(render.mix_legend(one_workout))
    for stroke in one_workout["predicted"].unique():
        assert render.STROKE_COLOR[stroke] in legend


def test_colours_are_distinct_per_stroke():
    """Two strokes sharing a colour would make the bar unreadable."""
    assert len(set(render.STROKE_COLOR.values())) == len(render.STROKE_COLOR)


def test_strava_block_is_plain_text(one_workout):
    """Strava descriptions render no markup — the card must not rely on any."""
    block = render.strava_block(one_workout, HEADER)
    assert "<" not in block and "**" not in block
    assert "polarswim" in block


class TestColourPaletteIsShared:
    """The pasted card and the web dashboard must agree on what colour a stroke is."""

    def test_web_palette_covers_every_stroke(self):
        from polarswim.web import PIE_COLORS
        assert set(PIE_COLORS) == set(render.STROKE_COLOR)

    def test_web_colours_are_distinct(self):
        from polarswim.web import PIE_COLORS
        assert len(set(PIE_COLORS.values())) == len(PIE_COLORS)


def test_mix_handles_a_single_stroke_workout():
    """A 100%-freestyle practice must not break the bar or the pie data."""
    df = pd.DataFrame({"predicted": ["freestyle"] * 8})
    mix = render.stroke_mix(df)
    assert len(mix) == 1 and mix[0][2] == pytest.approx(100.0)
    assert len(render.mix_bar(df)) == render.MIX_WIDTH
