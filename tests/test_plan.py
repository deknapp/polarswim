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


def test_an_unbalanced_plan_does_not_train_the_model():
    r = plan.PlanReading(stroke={i: "freestyle" for i in range(1, 45)}
                         | {i: "backstroke" for i in range(45, 53)},
                         factor={}, kind={}, rep_of={}, set_of={}, label={}, segments=[])
    assert plan.training_labels({1: r}) == {}
    r.stroke |= {i: "butterfly" for i in range(53, 61)} | {i: "breaststroke" for i in range(61, 69)}
    assert len(plan.training_labels({1: r})) == 68


# The coach's PDF for 2026-09-28, as `pdftotext -layout` gives it.
TERRIBLE_TUESDAY = """\
Building 200 IM               "Terrible Tuesday"
Warm up
1x600: 200 Swim/200 Kick/200 Choice
                                               (600)
Building to a 200 IM:
8x50s Drill IMO
  ● 1-1-1 or Single-Arm Fly
  ● Catch-Up Free
4x50s 25’sSwim/ 25’s Drill IMO (see above)
4x50s IMO
                                             (1400)
2x thru “floating IMs”:
3x125 IM                on 2:00/2:15/2:30
    50 FL/25 BK/25 BR/25 FR
    25 FL/50 BK/25 BR/25 FR
    25 FL/25 BK/50 BR/25 FR
1x25 Swim EZ                        on :45
                                              (1800)
2 x thru (no rest between rounds):
1x200 Negative Split   2:50/3:00/3:10
1x200 Pull             2:50/3:00/3:10
1x200 IM               3:00/3:10/3:20        (3000)

Cool Down: 1 x 200                           (3200)
"""


def test_parse_reads_a_coach_pdf_layout():
    swims = plan.parse_plan(TERRIBLE_TUESDAY)
    warm = swims[0]
    assert (warm.yards, warm.legs) == (600, [(200, "freestyle"), (200, "kick"), (200, None)])
    drill = swims[1:9]
    assert all((s.yards, s.stroke) == (50, "drill") for s in drill)   # `50s` is 50
    swim_drill = swims[9:13]
    assert [s.legs for s in swim_drill] == [[(25, st), (25, "drill")] for st in plan.IM_ORDER]
    assert [s.stroke for s in swims[13:17]] == list(plan.IM_ORDER)    # IMO rotates
    floating = swims[17:21]
    assert floating[0].legs == [(50, "butterfly"), (25, "backstroke"),
                                (25, "breaststroke"), (25, "freestyle")]
    assert floating[2].legs[2] == (50, "breaststroke")
    assert floating[0].intervals == [120.0, 135.0, 150.0]              # `on 2:00/...`
    assert floating[3].easy and floating[3].intervals == [45.0]
    assert [s.yards for s in swims[21:25]] == [125, 125, 125, 25]      # 2x thru
    assert swims[25].intervals == [170.0, 180.0, 190.0]                # bare clocks
    assert swims[26].stroke == "freestyle"                             # pull
    assert [s.yards for s in swims[25:31]] == [200] * 6                # 2x, not 3x
    assert swims[-1].yards == 200 and swims[-1].line_no == 24          # cool down
    assert len(swims) == 32


def test_legs_name_each_length_of_a_floating_im():
    swims = plan.parse_plan("1x125 IM\n    25 FL/50 BK/25 BR/25 FR")
    df = _lengths([[27, 31, 30, 32, 25]])
    r = plan.read(df, swims)
    assert [r.stroke[i] for i in range(1, 6)] == [
        "butterfly", "backstroke", "backstroke", "breaststroke", "freestyle"]
    assert r.factor == {}                    # a multi-stroke swim is never re-split


def test_a_swim_missing_from_the_plan_is_left_whole():
    """The swimmer forgot a 150 between the 200 and the last two 25s. The
    aligner must not carve the 150 up to put the 25s at its start."""
    swims = plan.parse_plan("1x200 Free\n1x25 Breast\n1x25 Free")
    df = _lengths([[25, 22, 25, 26, 26, 25, 24, 26], [26, 22, 23, 25, 24, 28],
                   [31], [25]], rest=30)
    segs = plan.align(df, swims)
    assert [(s.swim_no, s.idxs) for s in segs] == [
        (0, list(range(1, 9))), (None, list(range(9, 15))), (1, [15]), (2, [16])]


def test_imo_can_leave_a_stroke_out():
    swims = plan.parse_plan("3x50s 25's Swim/25's Drill IMO no fly")
    assert [s.legs[0][1] for s in swims] == ["backstroke", "breaststroke", "freestyle"]


def test_a_floating_im_is_named_im_and_a_swim_drill_by_its_parts():
    swims = plan.parse_plan("1x125 IM\n    25 FL/50 BK/25 BR/25 FR\n"
                            "1x50 25 Back/25 Drill")
    df = _lengths([[27, 31, 30, 32, 25], [33, 36]])
    r = plan.read(df, swims)
    assert {r.shape[i] for i in range(1, 6)} == {"IM"}
    assert r.shape[6] == "back/drill"
    out = plan.apply_labels(df.assign(predicted="freestyle", confidence=0.5),
                            plan.labels({1: r}), plan.shapes({1: r}))
    assert out["im_continuous"].tolist() == [True] * 5 + [False] * 2


def test_swim_is_freestyle_except_in_an_imo_rotation():
    assert plan.parse_plan("1x150 Swim")[0].stroke == "freestyle"
    assert plan.parse_plan("1x200 Choice")[0].stroke is None
    imo = plan.parse_plan("4x50 25 Swim/25 Drill IMO")
    assert imo[0].legs[0] == (25, "butterfly")


def test_medley_patterns_keep_off_planned_swims():
    from polarswim.patterns import PatternMatch
    r = plan.PlanReading(stroke={}, factor={}, kind={}, rep_of={}, set_of={}, label={},
                         segments=[plan.Segment(0, [6, 7, 8]), plan.Segment(None, [9, 10])])
    m = lambda idxs: PatternMatch(1, 1, 1, ("butterfly", "backstroke"), 1, [], idxs,
                                  0.0, 2.0, 0.9, "history")
    kept = plan.without_planned([m([6, 7]), m([9, 10])], {1: r})
    assert [k.idxs for k in kept] == [[9, 10]]


AEROBIC_IM = """\
Warm Up:
6 x 50 "6-3-6" Odd Free/ Even Back (fins?)
6 x 50 (25 Fast Free + 25 Fly DPS)             (900)
Drill:
2x (8 x 25) R1-Breast & R2-Fly
    R1: Odd 1 Pull-2 Kick/Even Breast DPS
    R2: Odd "Stone Skip"/Even Fly DPS          (400)
Aerobic IM:
2x (75 FR + 25 Fly + 75 FR + 25 BK)    R:15"
1 x 100 Ez Choice                              (500)
4 x 100 - Consistent Pacing         @base + 15"
    Odd 100 Free/Even 100 IM                   (400)
"""


def test_parse_reads_odd_even_rounds_and_stroke_inserts():
    swims = plan.parse_plan(AEROBIC_IM)
    assert [s.stroke for s in swims[:6]] == ["freestyle", "backstroke"] * 3
    fifty = swims[6]                                  # one 50 in two legs
    assert (fifty.yards, fifty.legs) == (50, [(25, "freestyle"), (25, "butterfly")])
    drill = [s.stroke for s in swims[12:28]]
    assert drill == ["drill", "breaststroke"] * 4 + ["drill", "butterfly"] * 4
    insert = swims[28]                                # unbroken, not four swims
    assert insert.yards == 200 and [st for _, st in insert.legs] == [
        "freestyle", "butterfly", "freestyle", "backstroke"]
    assert swims[30].easy and swims[30].yards == 100
    hundreds = swims[31:]
    assert [(s.yards, s.stroke) for s in hundreds] == [
        (100, "freestyle"), (100, "IM")] * 2          # `+ 15"` is not a swim


def test_a_swim_polar_logged_short_can_still_be_placed():
    """Both 100 EZs of 2026-09-30 came out as two lengths. Placing one must not
    slide the next 200 two lengths into the middle of an unbroken run."""
    swims = plan.parse_plan("1 x 100 EZ Choice\n"
                            "1 x 200: 75 FR/25 BK/75 FR/25 BR")
    df = _lengths([[28, 26], [23, 21, 25, 30, 24, 25, 26, 33]], rest=50)
    segs = plan.align(df, swims)
    assert [(s.swim_no, s.idxs) for s in segs] == [(0, [1, 2]), (1, list(range(3, 11)))]


def test_inserts_land_where_the_slow_lengths_are():
    swims = plan.parse_plan("2x (75 FR + 25 Fly + 75 FR + 25 BK)")
    df = _lengths([[23, 23, 24, 33, 24, 25, 24, 32], [25, 24, 24, 33, 26, 25, 26, 32]])
    r = plan.read(df, swims)
    assert [r.stroke[i] for i in range(1, 9)] == (
        ["freestyle"] * 3 + ["butterfly"] + ["freestyle"] * 3 + ["backstroke"])
