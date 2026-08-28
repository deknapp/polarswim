"""Swimmer-calibrated metrics: zones, relative speed, and personal bests."""

import numpy as np
import pandas as pd
import pytest

from polarswim import metrics
from polarswim.metrics import SwimmerReference


@pytest.fixture
def ref():
    # rest 79 / max 172 is this swimmer's real pair, so the boundaries below are
    # the ones the app actually shows.
    r = SwimmerReference(hr_max=172, median_pace_s=26.0, hr_rest=79)
    r.rep_times_by_distance = {
        50: np.sort(np.array([50.0 + i * 0.1 for i in range(100)])),
        25: np.sort(np.array([24.0 + i * 0.1 for i in range(100)])),
        200: np.array([200.0, 210.0]),          # too few to rank
    }
    r.best_rep = {(50, "freestyle"): {"seconds": 44.0, "workout_id": 7, "date": "2026-05-01"}}
    return r


class TestHeartRateZones:
    """Zones divide the range between resting and maximum, not zero and maximum."""

    def test_zones_span_resting_to_observed_max(self, ref):
        bounds = ref.zone_bounds()
        assert bounds[0]["low"] == 79          # resting, not zero
        assert bounds[-1]["high"] == 172       # the observed swim maximum
        assert [z["zone"] for z in bounds] == ["Z1", "Z2", "Z3", "Z4", "Z5"]

    def test_the_bottom_zone_is_reachable_in_the_water(self, ref):
        """Measured from zero, Z1 ended at 60% of max — a heart rate no swimmer
        holds while swimming, so easy work was landing two zones high."""
        z1 = ref.zone_bounds()[0]
        assert z1["high"] > 0.6 * ref.hr_max

    def test_zones_are_contiguous(self, ref):
        bounds = ref.zone_bounds()
        for lower, upper in zip(bounds, bounds[1:]):
            assert lower["high"] == upper["low"]

    @pytest.mark.parametrize("bpm,zone", [
        (80, "Z1"), (110, "Z1"), (130, "Z1"), (140, "Z2"), (148, "Z3"),
        (158, "Z4"), (165, "Z5"), (172, "Z5"),
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


class TestTrainingLoad:
    """Load and intensity answer different questions and must not be conflated."""

    @pytest.fixture
    def ref(self):
        r = SwimmerReference(hr_max=172, hr_rest=79)
        return r

    def test_harder_work_scores_more_load_than_easier_work(self, ref):
        easy = np.full(600, 100.0)
        hard = np.full(600, 160.0)
        assert ref.trimp(hard) > ref.trimp(easy)

    def test_intensity_is_weighted_exponentially_not_linearly(self, ref):
        """A minute at threshold must be worth well over a minute of recovery —
        that is the whole reason for Banister over a plain heart-rate integral."""
        recovery = ref.trimp(np.full(600, 95.0))
        threshold = ref.trimp(np.full(600, 155.0))
        linear_expectation = recovery * ((155 - 79) / (95 - 79))
        assert threshold > linear_expectation

    def test_longer_work_at_the_same_intensity_scores_more_load(self, ref):
        assert ref.trimp(np.full(1200, 130.0)) > ref.trimp(np.full(600, 130.0))

    def test_resting_heart_rate_contributes_nothing(self, ref):
        assert ref.trimp(np.full(600, 79.0)) == pytest.approx(0.0, abs=1e-6)

    def test_empty_series_is_zero(self, ref):
        assert ref.trimp(np.array([])) == 0.0

    def test_reserve_is_clipped_so_a_spike_cannot_explode_the_score(self, ref):
        assert ref.hr_reserve(np.array([400.0]))[0] <= 1.2
        assert ref.hr_reserve(np.array([10.0]))[0] == 0.0

    def test_a_short_hard_swim_can_outscore_a_long_easy_one(self, ref):
        """The exponential weighting is steep enough that intensity can beat
        duration outright — 30 min at 155 bpm outweighs 2 h at 115 bpm."""
        assert ref.trimp(np.full(1800, 155.0)) > ref.trimp(np.full(7200, 115.0))

    def test_load_and_intensity_rank_differently(self, ref):
        """Given enough duration, load still favours the long swim while intensity
        favours the hard one. If the two agreed, one would be redundant."""
        sessions = {
            1: np.full(18000, 110.0),      # 5 h easy
            2: np.full(1800, 160.0),       # 30 min hard
            3: np.full(3600, 130.0),       # 1 h moderate
        }
        ref.trimp_by_workout = {k: ref.trimp(v) for k, v in sessions.items()}
        ref.intensity_by_workout = {
            k: ref.trimp(v) / (len(v) / 60) for k, v in sessions.items()}
        ref.trimp_sorted = np.sort(np.array(list(ref.trimp_by_workout.values())))
        ref.intensity_sorted = np.sort(
            np.array(list(ref.intensity_by_workout.values())))
        long_easy, short_hard = ref.effort_score(1), ref.effort_score(2)
        assert long_easy["score"] > short_hard["score"]
        assert short_hard["intensity"] > long_easy["intensity"]

    def test_effort_needs_a_populated_database(self, ref):
        assert ref.effort_score(1) is None


class TestStrokeAwareSpeed:
    """Ranking a backstroke 100 against mostly-freestyle 100s buries it."""

    @pytest.fixture
    def ref(self):
        r = SwimmerReference(hr_max=172)
        r.rep_times_by_distance = {100: np.sort(np.array([95.0 + i * 0.1
                                                          for i in range(200)]))}
        r.rep_times_by_distance_stroke = {
            (100, "backstroke"): np.sort(np.array([125.0 + i * 0.2
                                                   for i in range(40)])),
            (100, "butterfly"): np.array([120.0, 122.0]),   # too thin
        }
        return r

    def test_uses_the_stroke_when_it_has_enough_history(self, ref):
        out = ref.speed_percentile(100, 128.0, "backstroke")
        assert out["basis"] == "stroke" and out["stroke"] == "backstroke"

    def test_stroke_ranking_is_fairer_than_distance_ranking(self, ref):
        by_stroke = ref.speed_percentile(100, 128.0, "backstroke")["percentile"]
        by_distance = ref.speed_percentile(100, 128.0)["percentile"]
        assert by_stroke > by_distance

    def test_falls_back_to_distance_when_the_stroke_is_thin(self, ref):
        out = ref.speed_percentile(100, 121.0, "butterfly")
        assert out["basis"] == "distance" and out["stroke"] is None

    def test_reports_nothing_when_neither_has_history(self, ref):
        assert ref.speed_percentile(400, 300.0, "freestyle") is None


class TestMedleyRanking:
    """A 100 IM belongs beside other 100 IMs."""

    def test_a_medley_is_ranked_against_medleys(self):
        r = SwimmerReference(hr_max=172, hr_rest=79)
        r.im_times_by_distance = {100: np.sort(np.array([95.0, 100.0, 105.0, 110.0]))}
        assert r.im_percentile(100, 96.0)["percentile"] == 75
        assert r.im_percentile(100, 96.0)["basis"] == "medley"

    def test_too_few_medleys_ranks_nothing(self):
        r = SwimmerReference(hr_max=172, hr_rest=79)
        r.im_times_by_distance = {100: np.array([95.0, 100.0])}
        assert r.im_percentile(100, 96.0) is None

    def test_a_medley_never_falls_back_to_the_freestyle_field(self):
        """The generic path would rank a 100 IM against 100 frees, where every
        medley lands near the bottom however well it was swum."""
        r = SwimmerReference(hr_max=172, hr_rest=79)
        r.rep_times_by_distance = {100: np.sort(np.arange(60.0, 90.0))}
        assert r.im_percentile(100, 105.0) is None


class TestCompetitiveDistances:
    def test_medley_only_has_its_three_real_distances(self):
        assert metrics.is_competitive(400, "IM")
        assert not metrics.is_competitive(75, "IM")
