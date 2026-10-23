"""Reps read off the send-off clock: `25 x 50 @ 1:00`, odd free / even IM."""

import pandas as pd

from polarswim import analyze, sendoff
from tests.test_analyze import _featured

# The 25 x 50 on 1:00 from 2026-10-23, as Polar logged it (lengths 33-78):
# rep 8 sat out, 30 s extra after rep 13, and after the IM reps the rest is
# mostly folded into the last length. Evens alternate fly/back and back/breast.
MAIN = [24.0, 22.4, 25.6, 31.2, 24.8, 25.6, 32.8, 28.8, 27.2, 22.4, 28.8, 33.6,
        25.6, 25.6, 24.0, 22.4, 28.8, 33.6, 24.8, 27.2, 34.4, 28.8, 27.2, 24.0,
        26.4, 28.8, 26.4, 23.2, 32.0, 54.4, 30.4, 31.2, 26.4, 27.2, 34.4, 28.8,
        26.4, 25.6, 34.4, 32.8, 23.2, 24.0, 33.6, 32.0, 23.2, 16.8]
RESTS = [68.8, 0.0, 12.8, 0.0, 4.0, 0.0, 8.8, 0.0, 0.0, 0.0, 8.8, 0.0, 0.0, 0.0,
         67.2, 0.0, 12.8, 0.0, 0.0, 0.0, 5.6, 0.0, 0.0, 0.0, 35.2, 0.0, 4.8, 0.0,
         10.4, 0.0, 33.6, 0.0, 0.0, 0.0, 5.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         9.6, 0.0, 0.0, 0.0]
RATIOS = {"butterfly": 1.24, "backstroke": 1.34, "breaststroke": 1.34, "freestyle": 1.0}


def _main():
    return _featured(MAIN, gaps=RESTS)


def test_the_set_is_one_train_with_a_hole_where_a_rep_was_sat_out():
    trains = sendoff.find_trains(_main())
    assert len(trains) == 1
    tr = trains[0]
    assert (tr.sendoff_s, tr.lengths_per_rep) == (60.0, 2)
    # 25 reps by the clock, re-phased across the extra rest; rep 8 is the hole.
    assert len(tr.slots) == 25
    assert [k for k, s in enumerate(tr.slots) if s is None] == [7, 16]
    assert tr.slots[0] == [1, 2] and tr.slots[-1] == [45, 46]   # rep 25 too


def test_reps_are_cut_where_the_clock_says_not_where_polar_rested():
    df = sendoff.recut(_main(), sendoff.find_trains(_main()))
    assert set(df["rep_lengths"]) <= {1, 2, 3}
    # Lengths 7-10 ran on with no rest logged: two 50s, not a 100 IM.
    assert df.loc[df["idx"] == 7, "rep_id"].item() != df.loc[df["idx"] == 9, "rep_id"].item()
    assert df.loc[df["idx"] == 8, "rest_hidden"].item()
    assert not df.loc[df["idx"] == 2, "rest_hidden"].item()


def test_odd_reps_are_free_and_even_reps_name_their_medley_pair():
    df = _main()
    trains = sendoff.find_trains(df)
    df = sendoff.recut(df, trains)
    (r,) = sendoff.read_parity(df.assign(predicted="freestyle"), trains, RATIOS)
    assert all(r.labels[i] == "freestyle" for i in (1, 2, 5, 6, 9, 10, 15, 16))
    assert [r.labels[i] for i in (3, 4)] == ["butterfly", "backstroke"]
    assert [r.labels[i] for i in (7, 8)] == ["backstroke", "breaststroke"]
    # Rep 12 follows rep 10 across the hole, still back/breast.
    assert [r.labels[i] for i in (21, 22)] == ["backstroke", "breaststroke"]
    out = sendoff.label_parity(df.assign(predicted="freestyle", confidence=0.5), [r])
    assert out.loc[out["idx"] == 3, "pattern"].item() == "fly/back"
    assert out.loc[out["idx"] == 3, "mixed_rep"].item()


def test_without_a_fly_faster_than_back_the_slow_reps_stay_unnamed():
    df = _main()
    trains = sendoff.find_trains(df)
    flat = {s: 1.3 for s in ("butterfly", "backstroke", "breaststroke")}
    (r,) = sendoff.read_parity(df.assign(predicted="freestyle"), trains, flat)
    assert r.labels[1] == "freestyle"
    assert r.labels[7] == "undetermined"
    assert "butterfly" not in r.labels.values()


def test_a_continuous_swim_is_not_a_train():
    assert sendoff.find_trains(_featured([26.0] * 40)) == []


def test_one_stroke_on_a_send_off_has_no_parity_to_read():
    df = _featured([24.0, 24.0] * 12, gaps=[12.0, 0.0] * 12)
    trains = sendoff.find_trains(df)
    assert len(trains) == 1
    assert sendoff.read_parity(df.assign(predicted="freestyle"), trains, RATIOS) == []


def test_halves_of_longer_swims_are_not_a_train():
    """3/6: 100s on ~2:10, a slow 50 taking half the send-off, then a fast 50.
    On a 1:05 grid there is never a rest before the second 50: they are 100s."""
    rep = [32.0, 32.0, 21.6, 26.4]
    durations, gaps = [], []
    for _ in range(8):
        durations += rep
        gaps += [18.0, 0.0, 0.0, 0.0]          # 2:10 start to start
    trains = sendoff.find_trains(_featured(durations, gaps=gaps))
    assert [(t.sendoff_s, t.lengths_per_rep) for t in trains] == [(130.0, 4)]


def test_medley_matches_on_a_read_train_are_dropped():
    class M:
        workout_id, idxs = 1, [7, 8, 9, 10]
    r = sendoff.Reading(1, {7: "backstroke"}, {}, [7, 8, 9, 10])
    assert sendoff.without_trains([M()], [r]) == []


def test_the_card_shows_the_two_halves_as_two_rows():
    from polarswim import report
    df = _main()
    trains = sendoff.find_trains(df)
    df = sendoff.recut(df, trains)
    parity = sendoff.read_parity(df.assign(predicted="freestyle"), trains, RATIOS)
    df = sendoff.label_parity(df.assign(predicted="freestyle", confidence=0.5), parity)
    rows = report.sets_for_workout(df)
    # 12 free, not 13: rep 17 started inside a length where Polar missed the
    # wall, so the clock files it as a hole beside rep 16.
    assert [(r["reps"], r["rep_yards"]) for r in rows] == [(12, 50), (11, 50)]
    assert rows[0]["stroke"] == "freestyle"
    assert rows[1]["pattern"] == "fl/bk ⇄ bk/br"
    # The free row's time is the free 50s that ended in a rest Polar saw.
    assert not rows[0]["rest_hidden"] and rows[0]["rep_seconds"] < 55
