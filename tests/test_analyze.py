"""Set segmentation, merged-turn repair, features, and classification."""

import numpy as np
import pandas as pd
import pytest

from polarswim import analyze


def _lengths(durations, pool_m=22.86, gaps=None, workout_id=1):
    """Build a lengths frame with explicit rest gaps between lengths."""
    gaps = gaps or [0.0] * len(durations)
    rows, t = [], 0.0
    for i, (d, g) in enumerate(zip(durations, gaps), start=1):
        t += g
        rows.append(dict(workout_id=workout_id, idx=i, start_offset_s=t,
                         duration_s=d, polar_style="OTHER",
                         start_time="2026-08-19T17:11:11", pool_length_m=pool_m,
                         distance_m=pool_m * len(durations), n_lengths=len(durations),
                         pool_m=pool_m, pace_s=d * (analyze.REFERENCE_LENGTH_M / pool_m)))
        t += d
    return pd.DataFrame(rows)


class TestRepsAndSets:
    """A rep is an unbroken swim; a set is a run of equal-distance reps."""

    def test_unbroken_lengths_are_one_rep(self):
        """Four lengths with no rest is a 100, not four 25s."""
        df = analyze.assign_sets(_lengths([25] * 4, gaps=[0] * 4))
        assert df["rep_id"].nunique() == 1
        assert df["rep_lengths"].unique().tolist() == [4]

    def test_a_rest_starts_a_new_rep(self):
        df = analyze.assign_sets(_lengths([25] * 4, gaps=[0, 0, 30, 0]))
        assert df["rep_id"].tolist() == [1, 1, 2, 2]

    def test_gap_at_the_threshold_is_not_a_rest(self):
        """A turn is not a rest; only a gap strictly over the threshold is."""
        df = analyze.assign_sets(
            _lengths([25] * 4, gaps=[0, analyze.REST_GAP_S, analyze.REST_GAP_S, 0]))
        assert df["rep_id"].nunique() == 1

    def test_equal_reps_group_into_one_set(self):
        """4x100 is one set of four reps, not four separate sets."""
        gaps = []
        for r in range(4):
            gaps += [30 if r else 0] + [0, 0, 0]
        df = analyze.assign_sets(_lengths([25] * 16, gaps=gaps))
        assert df["rep_id"].nunique() == 4
        assert df["set_id"].nunique() == 1

    def test_a_change_of_distance_starts_a_new_set(self):
        """2x50 then 2x100 is two sets."""
        gaps = [0, 0] + [30, 0] + [30, 0, 0, 0] + [30, 0, 0, 0]
        df = analyze.assign_sets(_lengths([25] * 12, gaps=gaps))
        sizes = df.drop_duplicates("rep_id")["rep_lengths"].tolist()
        assert sizes == [2, 2, 4, 4]
        assert df["set_id"].nunique() == 2

    def test_rest_before_is_recorded(self):
        df = analyze.assign_sets(_lengths([25] * 3, gaps=[0, 0, 40]))
        assert df["rest_before_s"].tolist() == pytest.approx([0.0, 0.0, 40.0])


class TestMergeRepair:
    def test_isolated_double_length_is_flagged(self):
        """A 2x outlier inside an otherwise-consistent set is a missed wall turn."""
        df = analyze.assign_sets(_lengths([24, 24, 48, 24, 24, 24]))
        repairs = analyze.detect_merges(df)
        assert [r.idx for r in repairs] == [3]
        assert repairs[0].factor == 2

    def test_uniformly_slow_set_is_a_drill_not_a_defect(self):
        """The whole point of the set-relative rule: slow drills must survive."""
        df = analyze.assign_sets(_lengths([48, 50, 47, 49, 48, 51]))
        assert analyze.detect_merges(df) == []

    def test_quadruple_merge_is_caught(self):
        df = analyze.assign_sets(_lengths([24, 24, 96, 24, 24, 24]))
        repairs = analyze.detect_merges(df)
        assert repairs and repairs[0].factor == 4

    def test_non_integer_outlier_is_not_split(self):
        """1.7x is not a whole number of lengths — more likely a slow length."""
        df = analyze.assign_sets(_lengths([24, 24, 41, 24, 24, 24]))
        assert analyze.detect_merges(df) == []

    def test_short_sets_are_left_alone(self):
        """Too few lengths for the median to mean anything."""
        df = analyze.assign_sets(_lengths([24, 48]))
        assert analyze.detect_merges(df) == []


class TestPaceNormalization:
    def test_pace_is_normalized_across_pool_lengths(self):
        """A 50 m length must not look four times slower than a 25 yd one."""
        short = _lengths([25], pool_m=22.86)["pace_s"].iloc[0]
        long_ = _lengths([50], pool_m=45.72)["pace_s"].iloc[0]
        assert short == pytest.approx(long_, rel=0.01)


class TestClassification:
    @pytest.fixture
    def params(self):
        return {"_global": {"pace_p10": 22, "pace_p30": 24, "pace_p50": 26,
                            "pace_p70": 29, "pace_p90": 34,
                            "cost_p33": 20, "cost_p67": 35, "n_obs": 7000}}

    def _classified(self, durations, costs, params, gaps=None):
        df = analyze.assign_sets(_lengths(durations, gaps=gaps))
        df = analyze.add_features(df, {})
        df["hr_cost"] = costs
        return analyze.classify(df, params)

    def test_fast_lengths_are_freestyle(self, params):
        out = self._classified([22] * 6, [25] * 6, params)
        assert set(out["predicted"]) == {"freestyle"}

    def test_slow_uniform_low_cost_set_is_other(self, params):
        """Drill and kick share one class by design."""
        out = self._classified([36] * 6, [10] * 6, params)
        assert set(out["predicted"]) == {"other"}

    def test_slow_and_cheap_reads_as_breaststroke(self, params):
        out = self._classified([31, 31, 31, 31], [5, 5, 5, 5], params)
        assert set(out["predicted"]) == {"breaststroke"}

    def test_slow_and_expensive_reads_as_backstroke(self, params):
        """Speed order is never assumed — cost is what separates these two."""
        out = self._classified([31, 31, 31, 31], [50, 50, 50, 50], params)
        assert set(out["predicted"]) == {"backstroke"}

    def test_ambiguous_cost_is_undetermined_not_a_guess(self, params):
        out = self._classified([31, 31, 31, 31], [27, 27, 27, 27], params)
        assert set(out["predicted"]) == {"undetermined"}

    def test_missing_hr_does_not_invent_a_stroke(self, params):
        out = self._classified([31] * 4, [np.nan] * 4, params)
        assert set(out["predicted"]) == {"undetermined"}

    def test_every_label_is_a_known_class(self, params):
        out = self._classified([20, 26, 31, 40], [10, 40, 25, 5], params)
        assert set(out["predicted"]) <= set(analyze.CLASSES)

    def test_confidence_is_a_probability(self, params):
        out = self._classified([20, 26, 31, 40], [10, 40, 25, 5], params)
        assert out["confidence"].between(0, 1).all()


class TestLearning:
    def test_params_come_from_the_data_not_constants(self):
        fast = analyze.learn_params(pd.DataFrame(
            {"pace_s": [18.0] * 100, "hr_cost": [20.0] * 100}))
        slow = analyze.learn_params(pd.DataFrame(
            {"pace_s": [40.0] * 100, "hr_cost": [20.0] * 100}))
        assert fast["_global"]["pace_p50"] < slow["_global"]["pace_p50"]

    def test_empty_input_yields_no_params(self):
        assert analyze.learn_params(pd.DataFrame({"pace_s": [], "hr_cost": []})) == {}
