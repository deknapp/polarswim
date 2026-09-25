"""Reading a swim against its written workout."""

import numpy as np
import pandas as pd
import pytest

from polarswim import analyze, db, plan

POOL_M = 22.86

WORKOUT = """\
Descending Intervals            Friday
Warm up:
#300 Swim: (every 3rd 25 "Long Doggy Paddle")
4 x (25 Fast Kick + 50 Fast Swim Free)        (800)
Maintain IMO Pace for 50's:
3 x 50 Fly           @1:00/1:10/1:15
1 x 200 EZ Choice    @4:00                      (900)
"""


def test_parse_reads_repeats_compounds_intervals_and_skips():
    swims = plan.parse_plan(WORKOUT)
    assert [(s.yards, s.stroke) for s in swims[:2]] == [(25, "kick"), (50, "freestyle")]
    assert len(swims) == 4 * 2 + 3 + 1          # the #-line is not swum
    assert swims[0].joined and not swims[1].joined
    fly = swims[8]
    assert (fly.stroke, fly.intervals) == ("butterfly", [60.0, 70.0, 75.0])
    assert swims[-1].stroke is None and swims[-1].easy
    assert swims[-2].last_of_line and not swims[-3].last_of_line


def test_kick_outranks_the_strokes_it_names():
    (s, *_), = [plan.parse_plan("4 x 50 Kick: Odd Back/Even Breast")]
    assert s.stroke == "kick"


def _lengths(reps, rest=15.0, lead=5.0):
    """reps: list of lists of length durations; each inner list is unbroken."""
    rows, t, idx = [], lead, 0
    for rep in reps:
        for d in rep:
            idx += 1
            rows.append(dict(workout_id=1, idx=idx, start_offset_s=t, duration_s=d,
                             pool_m=POOL_M, pace_s=d, set_id=0))
            t += d
        t += rest
    df = pd.DataFrame(rows)
    return analyze.assign_sets(df)


def test_alignment_finds_the_plan_and_leaves_extras_unplanned():
    swims = plan.parse_plan("2 x 50 Free @1:05\n2 x 50 Back @1:15")
    # two free 50s, two slow back 50s, then a stray 25 after the plan ended
    df = _lengths([[23, 22], [23, 23], [31, 30], [31, 31], [26]])
    segs = plan.align(df, swims)
    got = [(s.swim_no, s.idxs) for s in segs]
    assert got == [(0, [1, 2]), (1, [3, 4]), (2, [5, 6]), (3, [7, 8]), (None, [9])]


def test_a_joined_swim_may_run_into_the_next_without_a_stop():
    swims = plan.parse_plan("2 x (25 Kick + 50 Free)")
    df = _lengths([[34, 23, 22], [33, 23, 23]])
    segs = plan.align(df, swims)
    assert [s.idxs for s in segs] == [[1], [2, 3], [4], [5, 6]]


def test_lopsided_split_is_evened_out_keeping_the_swim_time():
    swims = plan.parse_plan("3 x 50 Free")
    df = _lengths([[23, 23], [28.8, 18.4], [23, 22]])
    r = plan.read(df, swims)
    assert set(r.factor) == {3, 4}
    assert r.factor[3] + r.factor[4] == pytest.approx(2.0)
    per_length = (28.8 + 18.4) / 2
    assert 28.8 / r.factor[3] == pytest.approx(per_length)
    assert r.kind[3] == "resplit"


def test_a_short_swim_is_relabelled_but_not_stretched():
    """Six normal lengths for a planned 200 means the swimmer swam 150: the
    plan names the stroke but must not invent the missing yards."""
    swims = plan.parse_plan("1 x 200 Free")
    df = _lengths([[25, 24, 25, 24, 25, 24]])
    r = plan.read(df, swims)
    assert r.factor == {}
    assert set(r.stroke.values()) == {"freestyle"}


def test_plan_labels_outrank_inference_and_lose_to_corrections():
    df = pd.DataFrame(dict(workout_id=[1, 1], idx=[1, 2],
                           predicted=["backstroke", "backstroke"],
                           confidence=[0.3, 0.3]))
    df = plan.apply_labels(df, {(1, 1): "butterfly", (1, 2): "butterfly"})
    assert list(df["predicted"]) == ["butterfly", "butterfly"]
    from polarswim import learn
    df = learn.apply_labels(df, {(1, 2): "freestyle"})
    assert list(df["predicted"]) == ["butterfly", "freestyle"]
    assert list(df["label_source"]) == ["plan", "corrected"]


def test_plan_round_trips_and_drives_the_analysis(tmp_path, pool_swim_payload):
    from polarswim.parse import parse_details
    engine = db.connect(":memory:")
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(engine, w, raw=pool_swim_payload)
    db.save_plan(engine, w.id, "4 x 50 Fly")
    assert db.load_plans(engine) == {w.id: "4 x 50 Fly"}
    analyze.analyze(engine)
    import sqlalchemy as sa
    from polarswim.models import predictions
    with engine.connect() as c:
        got = [r[0] for r in c.execute(sa.select(predictions.c.predicted))]
    assert got.count("butterfly") >= 6          # 8 lengths; the plan claims the 50s
    assert db.clear_plan(engine, w.id) == 1
