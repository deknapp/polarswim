"""Time the plan knows better than Polar: a folded break, and his own time.

2026-11-20's ladder opened 100 free then 200 free with a ~10 s break between,
which Polar never saw: lengths 29 and 30 (here 4 and 5) came out 26.4 and 32.8
against usual lengths of 24.0 and 25.6. He timed the 100 at about 1:32.
"""

import pytest

from polarswim import plan, report
from tests.test_analyze import _featured

DUR = [24.8, 23.2, 24.0, 26.4, 32.8, 24.8, 24.8, 26.4, 24.0, 25.6, 26.4, 27.2]
GAPS = [30.0] + [0.0] * 11


def _reading(text):
    g = _featured(DUR, gaps=GAPS)
    return g, plan.read(g, plan.parse_plan(text))


def test_a_hand_time_is_read_and_not_taken_for_a_send_off():
    (s,) = plan.parse_plan("1 x 100 Free in 1:32")
    assert s.hand_time_s == 92.0 and s.intervals == [] and s.stroke == "freestyle"
    (s,) = plan.parse_plan("1 x 100 Free @ 1:45")
    assert s.hand_time_s is None and s.intervals == [105.0]


def test_a_folded_break_is_measured_not_flagged():
    _, r = _reading("1 x 100 Free\n1 x 200 Free")
    assert r.folded[4] == (pytest.approx(2.4), False)
    assert r.folded[5] == (pytest.approx(7.2), True)
    assert not r.rest_hidden
    # The re-split that spread the break over the 200 as pace is undone.
    assert not any(r.kind.get(i) == "resplit" for i in range(5, 13))


def test_without_usual_lengths_the_length_is_still_flagged():
    g = _featured([24.0, 30.0, 25.0, 25.0], gaps=[30.0, 0, 0, 0])
    r = plan.read(g, plan.parse_plan("1 x 50 Free\n1 x 50 Free"))
    assert r.folded == {} and r.rest_hidden == {2}


def _card(text):
    g, r = _reading(text)
    df = plan.restructure(g.assign(predicted="freestyle"), {1: r})
    return report.correct_wall_time(df).set_index("idx")


def _total(df, idxs):
    return round(float(df.loc[list(idxs), "duration_s"].sum()), 1)


def test_the_card_puts_the_break_back_into_rest():
    df = _card("1 x 100 Free\n1 x 200 Free")
    assert _total(df, range(1, 5)) == 96.0             # 98.4 less 2.4
    assert df.loc[5, "duration_s"] == pytest.approx(25.6)
    assert df.loc[5, "rest_before_s"] == pytest.approx(9.6)
    # Polar's own record is kept for rankings.
    assert df.loc[4, "duration_observed_s"] == 26.4


def test_his_time_replaces_polars_total():
    df = _card("1 x 100 Free in 1:32\n1 x 200 Free")
    assert _total(df, range(1, 5)) == 92.0
    # Every length gives up its share; none is singled out.
    ratios = (df.loc[1:4, "duration_s"] / df.loc[1:4, "duration_observed_s"]).round(3)
    assert ratios.iloc[:3].nunique() == 1
    assert df.loc[5, "rest_before_s"] > 9.6
    assert df.loc[1:4, "plan_timed"].all()
