"""Unbroken swims of free and whole-leg blocks of one other stroke (10/30)."""

import numpy as np

from polarswim import blocks
from tests.test_inserts import _set

# 2026-10-30, as Polar logged them: misplaced turns (18.4, 47.2) and all.
TWO_HUNDRED = [30.4, 29.6, 26.4, 25.6, 28.0, 32.8, 24.0, 18.4]       # odd 50 back
THREE_HUNDRED = [28.8, 47.2, 31.2, 26.4, 18.4, 31.2,
                 26.4, 24.8, 26.4, 32.8, 34.4, 32.0]                  # 75 bk/150 fr/75 bk
FREE_REF = 24.12


def test_back_free_swims_read_as_blocks_despite_misplaced_turns():
    assert blocks.read_rep(np.array(TWO_HUNDRED), FREE_REF)[:2] == (2, [0, 2])
    assert blocks.read_rep(np.array(THREE_HUNDRED), FREE_REF)[:2] == (3, [0, 3])


def test_free_swims_that_drift_are_not_blocks():
    descend = np.array([30, 30, 28, 28, 26, 26, 24, 24.0])
    tired = np.array([24, 24, 24, 24, 25, 25, 30, 30, 31, 31.0])   # slows at the end
    steady = np.array([25, 26, 25, 27, 26, 25, 26, 25.0])
    for p in (descend, tired, steady):
        assert blocks.read_rep(p, FREE_REF) is None


def test_slow_backstroke_is_not_freestyle_with_blocks():
    back = np.array([33, 33, 32, 34, 40, 40, 33, 33.0])
    assert blocks.read_rep(back, FREE_REF) is None


def test_one_workouts_blocks_are_named_together_and_only_when_clear():
    df = _set(TWO_HUNDRED, THREE_HUNDRED, rest=40.0).assign(free_ref_s=FREE_REF)
    split = {"split": 1.31, "back_slower": 1.0}
    found = blocks.detect_blocks(df, {"_back_breast": split})
    assert len(found) == 2
    # Pooled 1.30 sits on the split: the slow legs stay undetermined, free is free.
    names = {st for r in found for st in r.labels.values()}
    assert names == {"freestyle", "undetermined"}
    found = blocks.detect_blocks(df, {"_back_breast": {**split, "split": 1.20}})
    slow = [r.labels[r.idxs[0]] for r in found]
    assert slow == ["backstroke", "backstroke"]
    out = blocks.label_blocks(df.assign(predicted="freestyle", confidence=0.5), found)
    assert out["mixed_rep"].all() and set(out["pattern"]) == {"free+blocks"}
