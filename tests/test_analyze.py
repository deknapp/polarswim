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


def _featured(durations, gaps=None, hr=None, workout_id=1):
    """A frame carried all the way to classification-ready features."""
    df = analyze.assign_sets(_lengths(durations, gaps=gaps, workout_id=workout_id))
    series = {workout_id: np.full(3600, hr, dtype=float)} if hr else {}
    return analyze.add_features(df, series)


class TestSplitRepair:
    """The mirror of a merge: one length arriving as two impossibly fast records."""

    def test_a_split_pair_is_detected(self):
        durations = [26] * 8
        durations[3:5] = [12, 14]          # one 26 s length cut in two
        df = _featured(durations)
        splits = analyze.detect_splits(df)
        assert [r.idx for r in splits] == [4, 5]
        assert all(r.factor == 0.5 for r in splits)

    def test_a_single_fast_length_is_not_a_split(self):
        """One fast record is a fast length. Only a PAIR is evidence."""
        durations = [26] * 8
        durations[3] = 14
        assert analyze.detect_splits(_featured(durations)) == []

    def test_a_pair_separated_by_rest_is_not_a_split(self):
        """A swimmer cannot rest in the middle of a length."""
        durations = [26] * 8
        durations[3:5] = [12, 14]
        gaps = [0] * 8
        gaps[4] = 30                        # rest between the two fast records
        assert analyze.detect_splits(_featured(durations, gaps=gaps)) == []

    def test_a_genuinely_fast_set_is_not_split_repair(self):
        """A whole set of sprints has no outlier: everything is fast together."""
        assert analyze.detect_splits(_featured([13] * 8)) == []


class TestRepairsAreApplied:
    """Detecting a defect is only half the job."""

    def test_a_merged_record_is_corrected_to_a_per_length_pace(self):
        durations = [26] * 8
        durations[3] = 52                   # a missed wall fused two lengths
        df = _featured(durations)
        out = analyze.apply_repairs(df, analyze.detect_repairs(df))
        row = out[out["idx"] == 4].iloc[0]
        assert row["length_factor"] == 2
        assert row["pace_observed_s"] == pytest.approx(52, abs=0.5)
        assert row["pace_s"] == pytest.approx(26, abs=0.5)

    def test_a_split_pair_is_corrected_to_a_per_length_pace(self):
        durations = [26] * 8
        durations[3:5] = [13, 13]
        df = _featured(durations)
        out = analyze.apply_repairs(df, analyze.detect_repairs(df))
        pair = out[out["idx"].isin([4, 5])]
        assert (pair["length_factor"] == 0.5).all()
        assert pair["pace_s"].tolist() == pytest.approx([26, 26], abs=0.5)

    def test_an_untouched_length_keeps_its_observed_pace(self):
        df = _featured([26] * 8)
        out = analyze.apply_repairs(df, analyze.detect_repairs(df))
        assert (out["length_factor"] == 1.0).all()
        assert out["pace_s"].tolist() == pytest.approx(out["pace_observed_s"].tolist())

    def test_set_statistics_are_restated_after_a_repair(self):
        """A corrected pace changes the median every later rule reads."""
        durations = [26] * 8
        durations[3] = 52
        df = _featured(durations)
        out = analyze.apply_repairs(df, analyze.detect_repairs(df))
        assert out["set_median_pace_s"].iloc[0] == pytest.approx(26, abs=0.5)
        assert out["set_cv"].iloc[0] < 0.05      # uniform once repaired

    def test_a_merged_length_is_no_longer_misclassified_as_slow_work(self):
        """The reason repair must precede classification: an uncorrected merge
        carries a doubled time and is classified on a pace nobody swam."""
        durations = [26] * 12
        durations[5] = 52
        df = _featured(durations)
        # Learned from a history with real spread. A single uniform set is its own
        # p90, which would make every length in it look like the slow tail.
        params = {"_global": {"pace_p30": 26, "pace_p50": 28, "pace_p70": 31,
                              "pace_p90": 38, "cost_p33": 5, "cost_p67": 20,
                              "rest_p50": 20, "rest_p80": 45}}

        unrepaired = analyze.classify(df, params)
        repaired = analyze.classify(
            analyze.apply_repairs(df, analyze.detect_repairs(df)), params)

        assert unrepaired[unrepaired["idx"] == 6]["predicted"].iloc[0] == "other"
        assert repaired[repaired["idx"] == 6]["predicted"].iloc[0] == "freestyle"


class TestRestAndSetSizeInform_TheClassifier:
    """Rest is the third axis; set size is how much to trust the other two."""

    def test_rest_is_read_once_per_rep_not_per_length(self):
        """The zeros between lengths inside a rep are turns, not rest."""
        df = _featured([26] * 8, gaps=[0, 0, 0, 0, 40, 0, 0, 0])
        assert df["set_rest_s"].iloc[0] == pytest.approx(20.0)   # median of 0 and 40

    def test_long_rest_separates_butterfly_from_backstroke(self):
        """Both are slow and expensive; the one that bought more rest cost more."""
        params = {"_global": {"pace_p30": 24, "pace_p50": 27, "pace_p70": 30,
                              "pace_p90": 40, "cost_p33": 5, "cost_p67": 20,
                              "rest_p50": 20, "rest_p80": 45}}
        short = _featured([34] * 8, gaps=[10] * 8, hr=160)
        long = _featured([34] * 8, gaps=[60] * 8, hr=160)
        short["hr_cost"] = 30.0
        long["hr_cost"] = 30.0
        assert analyze.classify(short, params)["predicted"].iloc[0] == "backstroke"
        assert analyze.classify(long, params)["predicted"].iloc[0] == "butterfly"

    def test_a_slow_set_on_long_rest_is_not_called_drill(self):
        """Drill is slow AND cheap AND taken on short rest. Long rest means work."""
        params = {"_global": {"pace_p30": 24, "pace_p50": 27, "pace_p70": 30,
                              "pace_p90": 33, "cost_p33": 5, "cost_p67": 20,
                              "rest_p50": 20, "rest_p80": 45}}
        drill = _featured([36] * 8, gaps=[8] * 8)
        drill["hr_cost"] = 10.0
        assert analyze.classify(drill, params)["predicted"].iloc[0] == "other"

        hard = _featured([36] * 8, gaps=[70] * 8)
        hard["hr_cost"] = 10.0
        assert analyze.classify(hard, params)["predicted"].iloc[0] != "other"

    def test_a_tiny_set_is_called_with_lower_confidence(self):
        """Two lengths have no usable median; the call says so."""
        params = analyze.learn_params(_featured([26] * 20))
        big = analyze.classify(_featured([26] * 8), params)
        small = analyze.classify(_featured([26] * 2), params)
        assert small["confidence"].iloc[0] < big["confidence"].iloc[0]


class TestMedleyDetection:
    """An IM is the one place stroke order is known, so it is read structurally."""

    @staticmethod
    def _im_set(rounds=4, legs=(31, 28, 30, 24), gap=45):
        """`rounds` continuous medleys, each one rep of four lengths."""
        durations, gaps = [], []
        for _ in range(rounds):
            durations += list(legs)
            gaps += [gap, 0, 0, 0]
        return analyze.add_features(
            analyze.assign_sets(_lengths(durations, gaps=gaps)), {})

    def test_a_repeated_continuous_medley_is_found(self):
        rounds = analyze.detect_im(self._im_set())
        assert len(rounds) == 4
        assert all(r.continuous for r in rounds)
        assert {r.yards for r in rounds} == {100}

    def test_the_round_time_is_the_sum_of_its_four_legs(self):
        r = analyze.detect_im(self._im_set())[0]
        assert r.seconds == pytest.approx(31 + 28 + 30 + 24)
        assert r.splits_s == pytest.approx([31, 28, 30, 24])

    def test_a_uniform_freestyle_set_is_not_a_medley(self):
        """Four equal legs have no medley shape, however many times repeated."""
        assert analyze.detect_im(self._im_set(legs=(26, 26, 26, 26))) == []

    def test_free_must_be_the_fastest_leg(self):
        """Freestyle comes last in an IM and is the fastest stroke for anyone
        swimming one, so a set where it is not is some other kind of work."""
        assert analyze.detect_im(self._im_set(legs=(24, 28, 30, 31))) == []

    def test_the_middle_two_may_come_in_either_order(self):
        """Whether back or breast is slower is a fact about the swimmer, and this
        project does not assume it anywhere else either."""
        back_slower = analyze.detect_im(self._im_set(legs=(31, 33, 28, 24)))
        breast_slower = analyze.detect_im(self._im_set(legs=(31, 28, 33, 24)))
        assert len(back_slower) == 4 and len(breast_slower) == 4

    def test_a_single_round_is_not_claimed(self):
        """Four lengths that descend are indistinguishable from one medley."""
        assert analyze.detect_im(self._im_set(rounds=1)) == []

    def test_a_broken_medley_across_four_reps_is_found(self):
        """16x25 IM: the medley is broken, so each leg is its own rep."""
        durations = [31, 28, 30, 24] * 4
        df = analyze.add_features(
            analyze.assign_sets(_lengths(durations, gaps=[20] * 16)), {})
        rounds = analyze.detect_im(df)
        assert len(rounds) == 4
        assert all(not r.continuous for r in rounds)
        assert {r.yards for r in rounds} == {100}

    def test_a_broken_round_excludes_the_rest_between_its_legs(self):
        durations = [31, 28, 30, 24] * 4
        df = analyze.add_features(
            analyze.assign_sets(_lengths(durations, gaps=[20] * 16)), {})
        assert analyze.detect_im(df)[0].seconds == pytest.approx(113)

    def test_a_longer_medley_uses_more_lengths_per_leg(self):
        """A 200 IM in a 25 yd pool is two lengths of each stroke."""
        legs = (31, 31, 28, 28, 30, 30, 24, 24)
        durations, gaps = [], []
        for _ in range(3):
            durations += list(legs)
            gaps += [45] + [0] * 7
        df = analyze.add_features(
            analyze.assign_sets(_lengths(durations, gaps=gaps)), {})
        rounds = analyze.detect_im(df)
        assert {r.yards for r in rounds} == {200}
        assert rounds[0].splits_s == pytest.approx([62, 56, 60, 48])

    def test_medley_labels_replace_the_inferred_ones(self):
        df = self._im_set()
        params = analyze.learn_params(df)
        labelled = analyze.label_im(analyze.classify(df, params), analyze.detect_im(df))
        first = labelled.sort_values("idx")["predicted"].tolist()[:4]
        assert first == list(analyze.IM_ORDER)
        assert labelled["confidence"].iloc[0] > 0.8

    def test_lengths_outside_a_medley_keep_their_inferred_label(self):
        df = self._im_set()
        params = analyze.learn_params(df)
        classified = analyze.classify(df, params)
        labelled = analyze.label_im(classified, [])
        assert labelled["predicted"].tolist() == classified["predicted"].tolist()

    def test_four_seventy_fives_are_not_a_medley(self):
        """The shape matches but 300 is not a medley distance, so it is rejected."""
        durations = [31, 28, 30, 24] * 4          # three lengths per leg = 75 yd
        blown = [d for d in durations for _ in range(3)]
        gaps = [20 if i % 3 == 0 else 0 for i in range(len(blown))]
        df = analyze.add_features(
            analyze.assign_sets(_lengths(blown, gaps=gaps)), {})
        assert all(r.yards in analyze.IM_DISTANCES_YD for r in analyze.detect_im(df))

    def test_a_set_that_is_not_a_multiple_of_four_is_not_truncated_to_fit(self):
        """A 9x50 is not two medley rounds plus a spare. Dropping the ninth rep to
        make the shape fit is fitting the data to the hypothesis."""
        durations = [31, 28, 30, 24] * 2 + [26]
        df = analyze.add_features(
            analyze.assign_sets(_lengths(durations, gaps=[20] * 9)), {})
        assert analyze.detect_im(df) == []

    def test_legs_that_barely_differ_are_not_called_a_medley(self):
        """Four strokes differ by much more than one stroke varies between rounds."""
        assert analyze.detect_im(self._im_set(legs=(27, 28, 27, 26))) == []


class TestOneStrokePerUnbrokenSwim:
    """A rep has no rest in it, and nobody changes stroke without a wall."""

    @staticmethod
    def _rep(labels, confidences=None, im=False):
        n = len(labels)
        return pd.DataFrame({
            "workout_id": [1] * n, "rep_id": [1] * n, "idx": range(1, n + 1),
            "set_id": [1] * n, "predicted": labels,
            "confidence": confidences or [0.8] * n,
            "im_continuous": [im] * n,
        })

    def test_a_stray_length_takes_the_strokes_of_the_swim_it_is_in(self):
        """One butterfly length inside a continuous 1250 freestyle is not a
        stroke change; it is a length that drifted into another cluster."""
        df = self._rep(["freestyle"] * 47 + ["butterfly"] + ["undetermined"] * 2)
        out = analyze.enforce_rep_consistency(df)
        assert set(out["predicted"]) == {"freestyle"}

    def test_an_already_consistent_rep_is_untouched(self):
        df = self._rep(["freestyle"] * 8)
        out = analyze.enforce_rep_consistency(df)
        assert out["predicted"].tolist() == df["predicted"].tolist()
        assert out["confidence"].tolist() == df["confidence"].tolist()

    def test_a_medley_rep_keeps_its_four_strokes(self):
        """Four strokes in one unbroken swim is exactly what an IM is."""
        df = self._rep(list(analyze.IM_ORDER), im=True)
        out = analyze.enforce_rep_consistency(df)
        assert out["predicted"].tolist() == list(analyze.IM_ORDER)

    def test_a_tie_goes_to_the_label_the_rules_were_surer_of(self):
        df = self._rep(["freestyle", "freestyle", "butterfly", "butterfly"],
                       confidences=[0.4, 0.4, 0.9, 0.9])
        out = analyze.enforce_rep_consistency(df)
        assert set(out["predicted"]) == {"butterfly"}

    def test_confidence_reflects_how_much_of_the_rep_agreed(self):
        """A rep where one length in ten disagreed is a weaker call than one
        where every length agreed."""
        clean = analyze.enforce_rep_consistency(self._rep(["freestyle"] * 10))
        split = analyze.enforce_rep_consistency(
            self._rep(["freestyle"] * 9 + ["butterfly"]))
        assert split["confidence"].iloc[0] < clean["confidence"].iloc[0]

    def test_reps_are_collapsed_independently_of_each_other(self):
        a = self._rep(["freestyle"] * 3 + ["butterfly"])
        b = self._rep(["breaststroke"] * 3 + ["freestyle"])
        b["rep_id"] = 2
        b["idx"] = range(5, 9)
        out = analyze.enforce_rep_consistency(pd.concat([a, b], ignore_index=True))
        assert set(out[out["rep_id"] == 1]["predicted"]) == {"freestyle"}
        assert set(out[out["rep_id"] == 2]["predicted"]) == {"breaststroke"}

    def test_a_lone_medley_is_exempt(self):
        """A lone 100 IM is four lengths, one of each stroke — no label above
        25%. Structural detection needs two rounds, so a medley swum once arrives
        here unmarked, and collapsing it on a plurality erased a real four-stroke
        swim to invent a fake one-stroke one."""
        df = self._rep(list(analyze.IM_ORDER))
        out = analyze.enforce_rep_consistency(df)
        assert out["predicted"].tolist() == list(analyze.IM_ORDER)
        # And marked, so the set table names it IM and the bests rank it as one.
        assert out["im_continuous"].all()

    def test_an_even_split_of_two_strokes_is_not_a_medley(self):
        """The exemption is for all four strokes in equal measure. Two strokes
        half and half is an ordinary noisy rep, and still collapses."""
        df = self._rep(["freestyle", "freestyle", "butterfly", "butterfly"])
        assert analyze.enforce_rep_consistency(df)["predicted"].nunique() == 1

    def test_three_strokes_do_not_make_a_medley(self):
        df = self._rep(["freestyle", "butterfly", "backstroke", "freestyle"])
        assert analyze.enforce_rep_consistency(df)["predicted"].nunique() == 1

    def test_a_clear_majority_still_absorbs_its_strays(self):
        df = self._rep(["freestyle"] * 19 + ["butterfly"] * 3)
        assert set(analyze.enforce_rep_consistency(df)["predicted"]) == {"freestyle"}
