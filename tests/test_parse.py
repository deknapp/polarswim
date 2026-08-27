"""Parsing Flow's payload: durations, per-length records, and non-swim sessions."""

import pytest

from polarswim.parse import ParseError, iso_duration_seconds, parse_details


@pytest.mark.parametrize("text,expected", [
    ("PT29.6S", 29.6),
    ("PT28S", 28.0),
    ("PT1M", 60.0),
    ("PT1M29.6S", 89.6),
    ("PT47M2.875S", 2822.875),
    ("PT3H39M39.197S", 13179.197),
    ("PT1S", 1.0),
    ("P1DT2H", 93600.0),
])
def test_iso_duration_parsing(text, expected):
    assert iso_duration_seconds(text) == pytest.approx(expected)


@pytest.mark.parametrize("empty", [None, ""])
def test_iso_duration_empty_is_none(empty):
    assert iso_duration_seconds(empty) is None


def test_iso_duration_rejects_garbage():
    # Better to fail loudly than silently mis-time a length if Flow changes format.
    with pytest.raises(ParseError):
        iso_duration_seconds("29.6 seconds")


def test_parse_pool_swim(pool_swim_payload):
    (w,) = parse_details(pool_swim_payload)
    assert w.sport_parent == "SWIMMING"
    assert w.is_pool_swim
    assert len(w.lengths) == 8
    assert w.pool_type == "YARDS"
    assert w.pool_length_m == pytest.approx(22.86)
    assert w.hr_interval_s == 1.0
    assert len(w.hr_values) == 10


def test_lengths_are_ordered_and_timed(pool_swim_payload):
    (w,) = parse_details(pool_swim_payload)
    assert [l.idx for l in w.lengths] == list(range(1, 9))
    assert w.lengths[0].duration_s == pytest.approx(29.6)
    assert w.lengths[1].start_offset_s == pytest.approx(89.6)
    # Offsets must increase; a decreasing offset would mean we mis-ordered laps.
    offsets = [l.start_offset_s for l in w.lengths]
    assert offsets == sorted(offsets)


def test_polar_cannot_classify_stroke(pool_swim_payload):
    """The whole reason this project exists: Polar reports OTHER for every length."""
    (w,) = parse_details(pool_swim_payload)
    assert {l.polar_style for l in w.lengths} == {"OTHER"}


def test_non_swim_session_parses_with_no_lengths(run_payload):
    (w,) = parse_details(run_payload)
    assert w.sport_parent == "RUNNING"
    assert not w.is_pool_swim
    assert w.lengths == []


def test_missing_exercises_raises():
    with pytest.raises(ParseError):
        parse_details({"swimDatas": {}})
