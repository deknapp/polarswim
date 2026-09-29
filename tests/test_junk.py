"""Lengths that were not swimming: found at the start, or named by the swimmer."""

import numpy as np
import pytest

from polarswim import analyze, db, report
from polarswim.cli import _idx_ranges

from test_analyze import _lengths


def _swim(junk, rest=30.0, reps=15, rep=(26, 27, 26, 28)):
    """Junk records (list of unbroken runs) then `reps` real 100s, and the gaps."""
    durs, gaps = [], []
    for run in junk:
        for k, d in enumerate(run):
            durs.append(d)
            gaps.append(rest if k == 0 else 0.0)
    swim_from = len(durs)
    for _ in range(reps):
        for k, d in enumerate(rep):
            durs.append(d)
            gaps.append(20.0 if k == 0 else 0.0)
    df = analyze.assign_sets(_lengths(durs, gaps=gaps))
    return df, float(df["start_offset_s"].iloc[swim_from])


def _hr(df, low_until, low=88.0, high=130.0):
    end = int(df["start_offset_s"].iloc[-1] + df["duration_s"].iloc[-1]) + 60
    t = np.arange(end)
    return {1: np.where(t < low_until + 40, low, high).astype(float)}


def test_fiddling_at_the_wall_is_found_as_one_block():
    """2026-09-28: 20, 16 | 30 | 16, 15 at 82-99 bpm, then the swim. The 30 s
    record in the middle is not fast, but it is inside the fiddling."""
    df, start = _swim([[20, 16], [30], [16, 15]])
    assert analyze.detect_junk(df, _hr(df, start)) == [(1, i) for i in range(1, 6)]


def test_a_real_warmup_starting_at_a_low_heart_rate_is_kept():
    df, start = _swim([[27, 28], [29]])
    assert analyze.detect_junk(df, _hr(df, start)) == []


def test_fast_lengths_while_working_hard_are_not_junk():
    df, start = _swim([[20, 16], [16, 15]])
    hot = {1: np.full(len(_hr(df, start)[1]), 140.0)}
    assert analyze.detect_junk(df, hot) == []


def test_without_heart_rate_nothing_is_excluded():
    df, _ = _swim([[20, 16]])
    assert analyze.detect_junk(df, {}) == []


def test_length_ranges():
    assert _idx_ranges(["1-5", "9", "11,12"]) == [1, 2, 3, 4, 5, 9, 11, 12]
    with pytest.raises(SystemExit):
        _idx_ranges(["a-b"])


def test_excluded_lengths_leave_every_view_and_the_totals(pool_swim_payload):
    from polarswim.parse import parse_details
    engine = db.connect(":memory:")
    (w,) = parse_details(pool_swim_payload)
    db.upsert_workout(engine, w, raw=pool_swim_payload)
    analyze.analyze(engine)
    before = analyze.load_lengths(engine, w.id)
    head = report.workout_headers(engine).iloc[0].to_dict()

    db.exclude(engine, w.id, [1, 2])
    analyze.analyze(engine)
    after = analyze.load_lengths(engine, w.id)
    assert len(after) == len(before) - 2 and 1 not in set(after["idx"])
    assert len(analyze.load_lengths(engine, w.id, keep_excluded=True)) == len(before)

    trimmed = report.workout_headers(engine).iloc[0].to_dict()
    pool_m = head["distance_m"] / head["n_lengths"]
    assert trimmed["distance_m"] == pytest.approx(head["distance_m"] - 2 * pool_m)
    assert trimmed["duration_s"] < head["duration_s"]      # the clock starts at length 3

    # an automatic pass never removes the swimmer's own exclusions
    db.save_auto_excluded(engine, [w.id], [])
    assert db.load_excluded(engine, [w.id]) == {(w.id, 1): "manual", (w.id, 2): "manual"}
    assert db.clear_excluded(engine, w.id) == 2
