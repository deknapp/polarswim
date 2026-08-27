"""The Strava card: shape, width, and safety on odd input."""

import pandas as pd
import pytest

from polarswim import render


@pytest.fixture
def one_workout():
    rows = []
    for i in range(1, 13):
        rows.append(dict(idx=i, set_id=1 + (i - 1) // 4, pace_s=24.0 + (i % 4) * 3,
                         predicted="freestyle" if i % 4 else "breaststroke",
                         confidence=0.7, hr_cost=30.0))
    return pd.DataFrame(rows)


HEADER = {"start_time": "2026-08-19T17:11:11", "distance_m": 1394.46,
          "duration_s": 2822.0, "avg_hr": 126}


def test_card_lines_share_one_width(one_workout):
    """Ragged edges are what make a pasted card look broken in Strava."""
    lines = render.set_card(one_workout, HEADER).splitlines()
    assert len({len(l) for l in lines}) == 1


def test_card_fits_a_phone(one_workout):
    for line in render.set_card(one_workout, HEADER).splitlines():
        assert len(line) <= render.WIDTH


def test_card_reports_real_totals(one_workout):
    card = render.set_card(one_workout, HEADER)
    assert "2026-08-19" in card and "1,525 yd" in card and "12 lengths" in card


def test_sparkline_length_matches_input():
    assert len(render.sparkline([1, 2, 3, 4, 5])) == 5


def test_sparkline_inverts_so_faster_reads_taller():
    fast_first = render.sparkline([10.0, 30.0], invert=True)
    assert fast_first[0] > fast_first[1]


def test_sparkline_handles_constant_input():
    assert len(render.sparkline([7.0] * 4)) == 4


def test_sparkline_handles_empty_input():
    assert render.sparkline([]) == ""


def test_strava_block_is_plain_text(one_workout):
    """Strava descriptions render no markup — the card must not rely on any."""
    block = render.strava_block(one_workout, HEADER)
    assert "<" not in block and "**" not in block
    assert "polarswim" in block
