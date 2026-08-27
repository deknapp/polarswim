"""Swimmer-calibrated metrics: zones, relative speed, and personal bests."""

import numpy as np
import pandas as pd
import pytest

from polarswim import metrics
from polarswim.metrics import SwimmerReference


@pytest.fixture
def ref():
    r = SwimmerReference(hr_max=172, median_pace_s=26.0)
    r.rep_times_by_distance = {
        50: np.sort(np.array([50.0 + i * 0.1 for i in range(100)])),
        25: np.sort(np.array([24.0 + i * 0.1 for i in range(100)])),
        200: np.array([200.0, 210.0]),          # too few to rank
    }
    r.best_rep = {(50, "freestyle"): {"seconds": 44.0, "workout_id": 7, "date": "2026-05-01"}}
    return r


class TestHeartRateZones:
    """Zones are anchored to the observed swim maximum, not a formula."""

    def test_zone_boundaries_scale_with_the_swimmers_max(self, ref):
        bounds = ref.zone_bounds()
        assert bounds[0]["low"] == 0
        assert bounds[-1]["high"] == 172
        assert [z["zone"] for z in bounds] == ["Z1", "Z2", "Z3", "Z4", "Z5"]

    def test_zones_are_contiguous(self, ref):
        bounds = ref.zone_bounds()
        for lower, upper in zip(bounds, bounds[1:]):
            assert lower["high"] == upper["low"]

    @pytest.mark.parametrize("bpm,zone", [
        (80, "Z1"), (110, "Z2"), (130, "Z3"), (145, "Z4"), (165, "Z5"), (172, "Z5"),
    ])
    def test_heart_rate_maps_to_the_right_zone(self, ref, bpm, zone):
        assert ref.hr_zone(bpm)["zone"] == zone

    def test_above_max_still_returns_the_top_zone(self, ref):
        assert ref.hr_zone(200)["zone"] == "Z5"

    def test_missing_heart_rate_returns_nothing(self, ref):
        assert ref.hr_zone(None) is None
        assert ref.hr_zone(float("nan")) is None

    def test_each_zone_has_its_own_colour(self, ref):
        colours = [z["color"] for z in ref.zone_bounds()]
        assert len(set(colours)) == len(colours)


class TestRelativeSpeed:
    """Ranked against the same distance, because distance is measured and stroke
    is only inferred — ranking within an inferred stroke would be circular."""

    def test_a_fast_rep_ranks_high(self, ref):
        assert ref.speed_percentile(50, 50.0)["percentile"] >= 95

    def test_a_slow_rep_ranks_low(self, ref):
        assert ref.speed_percentile(50, 59.9)["percentile"] <= 5

    def test_the_median_ranks_near_fifty(self, ref):
        assert 45 <= ref.speed_percentile(50, 55.0)["percentile"] <= 55

    def test_thin_history_reports_nothing_rather_than_guessing(self, ref):
        assert ref.speed_percentile(200, 205.0) is None

    def test_unknown_distance_reports_nothing(self, ref):
        assert ref.speed_percentile(500, 400.0) is None

    def test_result_carries_its_sample_size(self, ref):
        assert ref.speed_percentile(50, 55.0)["n"] == 100

    def test_faster_reps_never_rank_below_slower_ones(self, ref):
        a = ref.speed_percentile(50, 51.0)["percentile"]
        b = ref.speed_percentile(50, 57.0)["percentile"]
        assert a > b


class TestPersonalBests:
    def test_the_holding_workout_is_flagged(self, ref):
        assert ref.check_pr(50, "freestyle", 44.0, workout_id=7)

    def test_another_workout_is_not_flagged(self, ref):
        assert not ref.check_pr(50, "freestyle", 44.0, workout_id=8)

    def test_a_slower_time_is_not_a_best(self, ref):
        assert not ref.check_pr(50, "freestyle", 48.0, workout_id=7)

    def test_unseen_distance_or_stroke_is_not_a_best(self, ref):
        assert not ref.check_pr(75, "freestyle", 44.0, workout_id=7)
        assert not ref.check_pr(50, "butterfly", 44.0, workout_id=7)


class TestArtifactFiltering:
    """A split length produces a time no swimmer could hold, and it would
    otherwise win every personal best it touched."""

    def test_implausibly_fast_reps_are_excluded_from_bests(self):
        from polarswim import db
        engine = db.connect(":memory:")
        rows = []
        for wid in (1, 2):
            for i in range(1, 5):
                rows.append(dict(workout_id=wid, rep_id=1, idx=i,
                                 duration_s=26.0, pace_s=26.0, pool_m=22.86,
                                 predicted="freestyle", start_time="2026-01-01"))
        # A third workout with a physically impossible 12s length.
        for i in range(1, 5):
            rows.append(dict(workout_id=3, rep_id=1, idx=i,
                             duration_s=12.0, pace_s=12.0, pool_m=22.86,
                             predicted="freestyle", start_time="2026-01-02"))
        ref = metrics.build_reference(engine, pd.DataFrame(rows))
        assert ref.implausible_reps == 1
        best = ref.best_rep.get((100, "freestyle"))
        assert best and best["workout_id"] != 3

    def test_the_floor_is_relative_to_the_swimmer(self):
        assert metrics.PLAUSIBLE_FLOOR_RATIO < 1.0
