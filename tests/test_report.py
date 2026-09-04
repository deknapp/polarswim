"""Pace and time summaries, and the rules about what may be ranked.

The numbers here are checked against Polar's own summary of the 2026-09-04 swim
— 1:31:01 elapsed, 50:15 swum, 2,600 yd, 1:56 /100 yd average, 0:58 fastest —
because a figure meant to be compared with the watch has to reproduce the watch.
"""

import numpy as np
import pandas as pd
import pytest

from polarswim import analyze, metrics, report


def _lengths(spec, pool_m=22.86, workout_id=1):
    """Build a classified frame from (duration_s, stroke) pairs, one per length."""
    rows = []
    for i, (dur, stroke) in enumerate(spec, start=1):
        rows.append(dict(workout_id=workout_id, idx=i, duration_s=float(dur),
                         pool_m=pool_m, pace_s=float(dur), predicted=stroke,
                         confidence=0.6, length_factor=1.0, rep_id=i, set_id=1,
                         start_offset_s=0.0, rest_before_s=0.0, hr_cost=np.nan,
                         hr_abs=np.nan, start_time="2026-09-04T12:04:48"))
    return pd.DataFrame(rows)


class TestSwimTimeAndPace:
    """Elapsed time is not swim time, and pace is quoted against the second."""

    @pytest.fixture
    def summary(self):
        # 8 x 25 yd at 30 s each: 200 yd swum in 4:00, inside a 10:00 session.
        df = _lengths([(30.0, "freestyle")] * 8)
        return report.pace_summary(df, {"duration_s": 600.0, "distance_m": 182.88})

    def test_swim_time_is_the_sum_of_the_lengths(self, summary):
        assert summary["swim_time_s"] == 240.0

    def test_rest_is_what_the_elapsed_clock_has_left_over(self, summary):
        assert summary["rest_s"] == 360.0
        assert summary["rest_pct"] == 60.0

    def test_pace_is_quoted_against_swim_time_not_elapsed_time(self, summary):
        # 200 yd in 240 s is 2:00 /100. Against the 10:00 elapsed clock it would
        # be 5:00, which is not a pace anyone swam.
        assert summary["avg_pace_100_s"] == 120.0

    def test_an_empty_frame_summarises_to_nothing(self):
        assert report.pace_summary(pd.DataFrame(), {"duration_s": 600.0}) == {}


class TestPaceByStroke:
    """One number for the whole swim averages a drill set into a sprint set."""

    @pytest.fixture
    def summary(self):
        df = _lengths([(25.0, "freestyle")] * 4 + [(35.0, "breaststroke")] * 4
                      + [(60.0, "other")] * 2)
        return report.pace_summary(df, {"duration_s": 900.0, "distance_m": 228.6})

    def test_each_stroke_gets_its_own_pace(self, summary):
        by = {r["stroke"]: r for r in summary["by_stroke"]}
        assert by["freestyle"]["pace_100_s"] == 100.0
        assert by["breaststroke"]["pace_100_s"] == 140.0
        assert by["other"]["pace_100_s"] == 240.0

    def test_rows_run_fastest_first(self, summary):
        paces = [r["pace_100_s"] for r in summary["by_stroke"]]
        assert paces == sorted(paces)

    def test_each_row_carries_its_share_of_the_distance(self, summary):
        by = {r["stroke"]: r for r in summary["by_stroke"]}
        assert by["freestyle"]["yards"] == 100
        assert by["other"]["pct"] == 20.0

    def test_drill_and_unknown_are_marked_as_not_a_stroke(self, summary):
        by = {r["stroke"]: r for r in summary["by_stroke"]}
        assert by["freestyle"]["named"] and by["breaststroke"]["named"]
        assert not by["other"]["named"]


class TestTheConfidentSwim:
    """The question is how the SWIMMING went, so drill and kick come out."""

    @pytest.fixture
    def summary(self):
        # 200 yd of real swimming at 2:00 /100, plus 100 yd of very slow drill.
        df = _lengths([(30.0, "freestyle")] * 8 + [(90.0, "other")] * 2
                      + [(90.0, "undetermined")] * 2)
        return report.pace_summary(df, {"duration_s": 1800.0, "distance_m": 274.32})

    def test_the_confident_pace_excludes_drill_and_unknown(self, summary):
        assert summary["confident"]["pace_100_s"] == 120.0

    def test_the_overall_pace_is_dragged_down_by_the_drill(self, summary):
        """Which is exactly why the confident figure exists."""
        assert summary["avg_pace_100_s"] > summary["confident"]["pace_100_s"]

    def test_it_says_how_much_of_the_swim_it_covers(self, summary):
        assert summary["confident"]["yards"] == 200
        assert summary["confident"]["pct_of_yards"] == pytest.approx(66.7, abs=0.1)

    def test_it_names_the_strokes_it_kept(self, summary):
        assert summary["confident"]["strokes"] == ["freestyle"]

    def test_the_best_length_is_reported_per_hundred(self, summary):
        df = _lengths([(30.0, "freestyle")] * 7 + [(24.0, "freestyle")])
        out = report.pace_summary(df, {"duration_s": 600.0, "distance_m": 182.88})
        assert out["confident"]["best_100_s"] == 96.0

    def test_a_swim_that_was_all_drill_has_no_confident_figure(self):
        df = _lengths([(60.0, "other")] * 4)
        assert report.pace_summary(df, {"duration_s": 600.0})["confident"] is None


class TestRepairsAreNotHiddenInThePace:
    """Polar's distance is data; the repair is inference. Both are reported."""

    def test_the_headline_pace_reconciles_with_the_watch(self):
        df = _lengths([(30.0, "freestyle")] * 8)
        df.loc[0, "length_factor"] = 2.0        # one missed wall
        out = report.pace_summary(df, {"duration_s": 600.0, "distance_m": 182.88})
        assert out["reported_yards"] == 200
        assert out["avg_pace_100_s"] == 120.0   # exactly the watch's own figure

    def test_the_repaired_distance_rides_alongside(self):
        df = _lengths([(30.0, "freestyle")] * 8)
        df.loc[0, "length_factor"] = 2.0
        out = report.pace_summary(df, {"duration_s": 600.0, "distance_m": 182.88})
        assert out["yards"] == 225
        assert out["repaired_pace_100_s"] == pytest.approx(106.7, abs=0.1)

    def test_no_repair_means_no_second_figure_to_confuse_it_with(self):
        df = _lengths([(30.0, "freestyle")] * 8)
        out = report.pace_summary(df, {"duration_s": 600.0, "distance_m": 182.88})
        assert out["repaired_pace_100_s"] is None


class TestWhatMayBeRanked:
    """A drill is slow because it is a drill. There is no speed to report."""

    @pytest.fixture
    def ref(self):
        r = metrics.SwimmerReference(hr_max=172, median_pace_s=26.0)
        for stroke in ("freestyle", "other", "undetermined"):
            r.rep_times_by_distance_stroke[(25, stroke)] = np.sort(
                np.array([30.0 + i * 0.5 for i in range(40)]))
        return r

    def _rows(self, spec, ref):
        return report.sets_for_workout(_lengths(spec), set(), ref)

    def test_drill_sets_carry_no_speed_percentile(self, ref):
        """A 68 s kick 50 came back at the 80th percentile — ranked against other
        kick — and read as having swum a fast 50."""
        rows = self._rows([(40.0, "other")] * 4, ref)
        assert rows and all(r["speed"] is None for r in rows)

    def test_unidentified_sets_carry_no_speed_percentile(self, ref):
        rows = self._rows([(40.0, "undetermined")] * 4, ref)
        assert rows and all(r["speed"] is None for r in rows)

    def test_a_named_stroke_is_still_ranked(self, ref):
        rows = self._rows([(30.0, "freestyle")] * 4, ref)
        assert rows and all(r["speed"] is not None for r in rows)


class TestRepDistanceCountsRealLengths:
    """`metrics` files a repaired rep at its true distance; the set rows must too,
    or the rep is ranked against the wrong distance and can never match its own
    personal best."""

    def test_a_merged_record_counts_as_the_lengths_it_covers(self):
        df = _lengths([(30.0, "freestyle")] * 4)
        df["rep_id"] = 1                      # one unbroken 100
        df.loc[0, "length_factor"] = 2.0      # a missed wall: really a 125
        rows = report.sets_for_workout(df)
        assert rows[0]["rep_yards"] == 125

    def test_an_unrepaired_rep_is_unchanged(self):
        df = _lengths([(30.0, "freestyle")] * 4)
        df["rep_id"] = 1
        assert report.sets_for_workout(df)[0]["rep_yards"] == 100

    def test_a_frame_with_no_repair_column_still_works(self):
        df = _lengths([(30.0, "freestyle")] * 4).drop(columns=["length_factor"])
        df["rep_id"] = 1
        assert report.sets_for_workout(df)[0]["rep_yards"] == 100


class TestMedleyBestsAreReachable:
    """`best_rep` excludes medley reps by design, so looking a medley up there
    always missed and a personal-best 200 IM went unmarked on every card."""

    @pytest.fixture
    def ref(self):
        r = metrics.SwimmerReference(hr_max=172)
        r.best_im = {200: {"seconds": 168.0, "workout_id": 7, "date": "2026-05-01",
                           "continuous": True, "splits_s": [], "n_rounds": 4}}
        return r

    def test_the_holding_workout_is_flagged(self, ref):
        assert ref.check_pr(200, "IM", 168.0, workout_id=7)

    def test_another_workout_is_not(self, ref):
        assert not ref.check_pr(200, "IM", 168.0, workout_id=8)

    def test_a_slower_medley_is_not_a_best(self, ref):
        assert not ref.check_pr(200, "IM", 180.0, workout_id=7)


class TestNamedStrokesAreDefinedOnce:
    def test_the_six_classes_split_into_named_and_not(self):
        assert (set(analyze.NAMED_STROKES) | set(analyze.UNNAMED_STROKES)
                == set(analyze.CLASSES))
        assert not set(analyze.NAMED_STROKES) & set(analyze.UNNAMED_STROKES)


class TestOneDistancePerStroke:
    """The pie legend counted sensor records and the pace table counted repaired
    lengths, so one screen showed freestyle as both 1,450 yd and 1,475 yd."""

    def test_the_mix_and_the_pace_table_agree(self):
        from polarswim.web import create_app
        import pathlib
        sample = str(pathlib.Path(__file__).resolve().parents[1] / "sample" / "sample.db")
        client = create_app(sample).test_client()
        from polarswim import db as _db, report as _report
        wid = int(_report.workout_headers(_db.connect(sample))["id"].iloc[0])
        d = client.get(f"/api/workout/{wid}").get_json()
        by_stroke = {r["stroke"]: r["yards"] for r in d["pace"]["by_stroke"]}
        for slice_ in d["mix"]:
            assert slice_["yards"] == by_stroke[slice_["stroke"]]

    def test_the_shares_still_close_the_circle(self):
        df = _lengths([(30.0, "freestyle")] * 3 + [(40.0, "other")] * 3
                      + [(35.0, "backstroke")])
        out = report.pace_summary(df, {"duration_s": 600.0, "distance_m": 160.02})
        assert sum(r["pct"] for r in out["by_stroke"]) == pytest.approx(100.0)
