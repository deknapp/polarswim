"""Sets read by the shape of alternating reps: 2026-11-20, lengths 52-106."""

from polarswim import alternation
from tests.test_analyze import _featured

# As Polar logged it. Index 1 here is length 52 of the day:
#   1-8 200 free, 9-12 100 free (end of the ladder)
#   13-28 100 fly/back, 100 breast/free, x2;  29-32 100 IM
#   33-54 ten 50s, odd free / even 25 fly + 25 back alternating with 50 breast;
#   45-47 is the seventh 50 as three records; 53-54 an easy 50 free after.
DUR = [26.4, 24.0, 24.8, 24.8, 25.6, 25.6, 25.6, 27.2, 27.2, 23.2, 25.6, 26.4,
       26.4, 28.0, 33.6, 30.4, 31.2, 32.0, 24.0, 28.0, 26.4, 28.8, 34.4, 30.4,
       31.2, 30.4, 25.6, 28.8, 26.4, 29.6, 32.0, 21.6, 24.8, 23.2, 28.8, 31.2,
       25.6, 20.8, 32.8, 30.4, 24.0, 23.2, 25.6, 31.2, 16.8, 40.8, 24.8, 30.4,
       30.4, 26.4, 24.0, 25.6, 32.0, 25.6, 24.8]
REST = [107.2, 0, 0, 0, 0, 0, 0, 0, 65.6, 0, 0, 0, 99.2, 0, 0, 0, 26.4, 0, 0, 0,
        34.4, 0, 0, 0, 59.2, 0, 0, 0, 88.0, 0, 0, 0, 140.8, 0, 21.6, 0, 15.2, 0,
        29.6, 0, 70.4, 0, 90.4, 0, 12.0, 0, 0, 126.4, 0, 9.6, 0, 214.4, 0, 5.6, 0]
IM = (29, 30, 31, 32)


def _read(claimed=()):
    df = _featured(DUR, gaps=REST).assign(predicted="freestyle")
    return df, alternation.read_alternation(df, {(1, i) for i in claimed})


def _labels(readings):
    return {i: s for r in readings for i, s in r.labels.items()}


def test_alternating_halves_read_as_fly_back_and_breast_free():
    _, rs = _read(claimed=IM)
    lab = _labels(rs)
    assert [lab[i] for i in range(13, 17)] == ["butterfly"] * 2 + ["backstroke"] * 2
    assert [lab[i] for i in range(17, 21)] == ["breaststroke"] * 2 + ["freestyle"] * 2
    assert [lab[i] for i in range(21, 29)] == (["butterfly"] * 2 + ["backstroke"] * 2
                                               + ["breaststroke"] * 2 + ["freestyle"] * 2)
    # The free 100 before the set is not part of it.
    assert not any(i in lab for i in range(9, 13))


def test_odd_free_even_fly_back_or_breast_50s():
    _, rs = _read(claimed=IM)
    lab = _labels(rs)
    for i in (33, 34, 37, 38, 41, 42, 50, 51):
        assert lab[i] == "freestyle", i
    # The fly/back group, the first with a free-pace first length (35: 28.8 s) too.
    for a, b in ((35, 36), (43, 44), (52, 53)):
        assert (lab[a], lab[b]) == ("butterfly", "backstroke")
    # Its partner group swam one stroke, the one fly/back leaves out.
    for i in (39, 40, 48, 49):
        assert lab[i] == "breaststroke"
    # The seventh 50, three records with a misplaced wall, is passed over.
    assert not any(i in lab for i in (45, 46, 47))


def test_claimed_reps_are_walls():
    """A rep another reader took is never re-read, and no run crosses it."""
    _, rs = _read(claimed=IM + tuple(range(17, 21)))
    lab = _labels(rs)
    assert not any(i in lab for i in range(13, 29))   # 2 reps each side: too few


def test_a_steady_free_set_is_left_alone():
    dur = [25.6, 26.4] * 10
    rest = [0, 0] + [15.0, 0] * 9
    df = _featured(dur, gaps=rest).assign(predicted="freestyle")
    assert alternation.read_alternation(df, set()) == []


def test_a_fading_free_50_alone_is_not_fly():
    """One FS-shaped rep among free reps is a fade, not a set."""
    dur = [25.6, 26.4] * 4 + [25.6, 31.2] + [25.6, 26.4] * 4
    rest = [0, 0] + [15.0, 0] * 8
    df = _featured(dur, gaps=rest).assign(predicted="freestyle")
    assert alternation.read_alternation(df, set()) == []


def test_planned_workouts_are_skipped():
    df = _featured(DUR, gaps=REST).assign(predicted="freestyle")
    assert alternation.read_alternation(df, set(), planned={1}) == []
