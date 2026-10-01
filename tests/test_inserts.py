"""Freestyle with single-length stroke inserts: `2x (75 FR + 25 Fly + 75 FR + 25 BK)`."""

import pandas as pd

from polarswim import analyze, inserts
from tests.test_analyze import _featured

# Two unbroken 200s from 2026-09-30, the 4th and 8th length of each slow.
REP_A = [23.2, 23.2, 24.8, 35.2, 27.2, 25.6, 27.2, 32.0]
REP_B = [27.2, 25.6, 24.0, 32.8, 28.8, 26.4, 28.0, 32.0]


def _set(*reps, rest=20.0):
    durations, gaps = [], []
    for rep in reps:
        gaps += [rest] + [0.0] * (len(rep) - 1)
        durations += rep
    return _featured(durations, gaps=gaps)


def test_inserts_are_found_where_every_rep_slows():
    df = _set(REP_A, REP_B)
    found = inserts.detect_inserts(df)
    assert [r.inserts for r in found] == [[3, 7], [3, 7]]
    out = inserts.label_inserts(df.assign(predicted="butterfly", confidence=0.5), found)
    first = out[out["idx"] <= 8]["predicted"].tolist()
    assert first == ["freestyle"] * 3 + ["undetermined"] + ["freestyle"] * 3 + ["undetermined"]
    assert out["mixed_rep"].all()


def test_three_inserts_are_named_in_medley_order():
    rep = [24, 24, 33, 24, 24, 32, 32, 24, 24, 33]     # 50 FR/25 Fly/50 FR/50 BK/50 FR/25 BR
    found = inserts.detect_inserts(_set(rep, list(rep)))
    labels = found[0].labels
    assert [labels[i] for i in (3, 6, 7, 10)] == [
        "butterfly", "backstroke", "backstroke", "breaststroke"]


def test_a_lone_rep_needs_evenly_spaced_inserts():
    assert len(inserts.detect_inserts(_set(REP_A))) == 1
    uneven = [24, 24, 24, 24, 32, 24, 24, 32]           # 5th and 8th: not 75+25 units
    assert inserts.detect_inserts(_set(uneven)) == []


def test_one_slow_length_is_traffic_not_an_insert():
    rep = [24, 24, 24, 33, 24, 24, 24, 24]
    assert inserts.detect_inserts(_set(rep, list(rep))) == []


def test_a_slow_base_is_not_freestyle():
    """A backstroke 200 with slow lengths in it is a backstroke 200."""
    free = [24.0] * 16                                  # sets the day's freestyle pace
    back = [32, 32, 32, 42, 32, 32, 32, 42]
    df = _set(free, back, list(back))
    assert all(r.idxs[0] <= 16 for r in inserts.detect_inserts(df))


def test_back_breast_split_is_learned_from_labels_not_assumed():
    df = _featured([24.0] * 10 + [33.0] * 8 + [30.0] * 8)
    labels = {(1, i): "backstroke" for i in range(11, 19)}
    labels |= {(1, i): "breaststroke" for i in range(19, 27)}
    bb = analyze.learn_back_breast(df, labels)
    assert bb["back_slower"] == 1.0
    flipped = {k: ("breaststroke" if v == "backstroke" else "backstroke")
               for k, v in labels.items()}
    assert analyze.learn_back_breast(df, flipped)["back_slower"] == 0.0
    few = dict(list(labels.items())[:5])
    assert analyze.learn_back_breast(df, few) == {}
