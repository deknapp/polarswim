"""Medley-order pattern recognition inside a rep, and across a block of reps."""

import numpy as np
import pandas as pd
import pytest

from polarswim import analyze, patterns

# This swimmer's measured profile from the 2026-09-11 practice, in seconds per
# 25 yd. Backstroke and breaststroke sit within 1.6 s of each other, which is the
# whole reason the ORDER has to do the identifying.
PROFILE = patterns.Profile(
    pace={"butterfly": 27.2, "backstroke": 32.8,
          "breaststroke": 31.2, "freestyle": 22.0},
    source="workout", free_pace_s=22.0)


def _rep(per_length, workout_id=1, rep_id=1, set_id=1, pool_m=22.86, start=0.0):
    """One unbroken rep from a list of per-length durations."""
    rows, t = [], start
    for i, d in enumerate(per_length, start=1):
        rows.append(dict(workout_id=workout_id, idx=i, start_offset_s=t,
                         duration_s=d, rep_id=rep_id, set_id=set_id,
                         pool_m=pool_m,
                         pace_s=d * (analyze.REFERENCE_LENGTH_M / pool_m)))
        t += d
    return pd.DataFrame(rows)


def _legs(strokes, per_leg=1, profile=PROFILE, jitter=0.0):
    """Per-length durations for a rep swum as `strokes`, `per_leg` lengths each."""
    out = []
    for n, s in enumerate(strokes):
        for j in range(per_leg):
            out.append(profile.pace[s] + jitter * (1 if (n + j) % 2 else -1))
    return out


def _workout(reps, workout_id=1):
    """Consecutive reps, numbered and with distinct indexes."""
    frames, idx, t = [], 1, 0.0
    for rid, per_length in enumerate(reps, start=1):
        f = _rep(per_length, workout_id=workout_id, rep_id=rid, set_id=rid, start=t)
        f["idx"] = range(idx, idx + len(f))
        idx += len(f)
        t = float(f["start_offset_s"].iloc[-1] + f["duration_s"].iloc[-1]) + 30.0
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


class TestGrammar:
    """The candidate patterns are the contiguous windows of the medley order."""

    def test_full_medley_both_directions(self):
        assert (analyze.IM_ORDER, False) in patterns.windows(4)
        assert (analyze.IM_ORDER[::-1], True) in patterns.windows(4)
        assert len(patterns.windows(4)) == 2

    def test_three_leg_windows_drop_one_end_stroke(self):
        forward = [legs for legs, rev in patterns.windows(3) if not rev]
        assert forward == [analyze.IM_ORDER[:3], analyze.IM_ORDER[1:]]

    def test_windows_are_contiguous_only(self):
        """fly/breast skips backstroke, so it is not a medley window."""
        assert all(("butterfly", "breaststroke") != legs
                   for legs, _ in patterns.windows(2))

    def test_one_leg_is_not_a_pattern(self):
        assert patterns.windows(1) == []

    @pytest.mark.parametrize("legs,name", [
        (analyze.IM_ORDER, "IM"),
        (analyze.IM_ORDER[::-1], "reverse IM"),
        (analyze.IM_ORDER[:3], "IM no free"),
        (analyze.IM_ORDER[1:], "IM no fly"),
        (("butterfly", "backstroke"), "fly/back"),
    ])
    def test_names_are_what_a_swimmer_would_write(self, legs, name):
        assert patterns.pattern_name(legs) == name


class TestLegSplitting:
    """A rep divides into equal legs, and only at distances people swim."""

    def test_a_200_splits_into_four_50s(self):
        legs = patterns._leg_pace(_rep([25.0] * 8), 4)
        assert legs is not None and len(legs) == 4

    def test_a_150_is_not_read_as_two_75s(self):
        """75 yd per stroke is not a shape anyone swims, so the split is refused."""
        assert patterns._leg_pace(_rep([25.0] * 6), 2) is None
        assert patterns._leg_pace(_rep([25.0] * 6), 3) is not None

    def test_uneven_division_is_refused(self):
        assert patterns._leg_pace(_rep([25.0] * 6), 4) is None

    def test_pace_is_normalised_to_the_reference_length(self):
        """A 50 m pool halves the per-reference-length pace of the same time."""
        short = patterns._leg_pace(_rep([30.0] * 4, pool_m=22.86), 4)
        long = patterns._leg_pace(_rep([30.0] * 4, pool_m=45.72), 4)
        assert long[0] == pytest.approx(short[0] / 2, rel=1e-6)


class TestSingleRepMatching:
    """What one rep can and cannot be claimed on its own evidence."""

    def test_a_200_im_is_recognised_alone(self):
        m = patterns.match_rep(_rep(_legs(analyze.IM_ORDER, per_leg=2)), PROFILE)
        assert m is not None
        assert m.name == "IM"
        assert m.legs == analyze.IM_ORDER
        assert m.leg_lengths == 2

    def test_a_150_with_the_free_dropped_is_recognised(self):
        m = patterns.match_rep(_rep(_legs(analyze.IM_ORDER[:3], per_leg=2)), PROFILE)
        assert m is not None and m.name == "IM no free"

    def test_a_75_with_the_fly_dropped_is_recognised(self):
        """Three 25s of back, breast, free — the shape asked for."""
        m = patterns.match_rep(_rep(_legs(analyze.IM_ORDER[1:])), PROFILE)
        assert m is not None and m.name == "IM no fly"

    def test_a_reverse_medley_is_recognised(self):
        m = patterns.match_rep(
            _rep(_legs(analyze.IM_ORDER[::-1], per_leg=2)), PROFILE)
        assert m is not None and m.name == "reverse IM"

    def test_a_uniform_rep_is_not_a_pattern(self):
        """Four equal 50s is a 200 of one stroke, whatever stroke that was."""
        assert patterns.match_rep(_rep([26.0] * 8), PROFILE) is None

    def test_a_descending_freestyle_swim_is_not_a_medley(self):
        """The commonest false positive: legs that get faster on purpose."""
        assert patterns.match_rep(_rep([25.0, 24.0, 23.0, 22.0]), PROFILE) is None

    def test_a_fading_freestyle_swim_is_not_a_reverse_medley(self):
        assert patterns.match_rep(_rep([22.0, 23.0, 24.0, 25.0]), PROFILE) is None

    def test_two_legs_alone_are_never_claimed(self):
        """A 50 that went 31.2 then 24.8 fits `back/fly` exactly and is far more
        likely a descending 50 free, so two legs need the block to be labelled."""
        m = patterns.match_rep(_rep([32.8, 27.2]), PROFILE)
        assert m is None

    def test_noise_within_a_leg_still_matches(self):
        m = patterns.match_rep(
            _rep(_legs(analyze.IM_ORDER, per_leg=2, jitter=1.2)), PROFILE)
        assert m is not None and m.name == "IM"

    def test_an_unusable_profile_matches_nothing(self):
        bad = patterns.Profile(pace={"freestyle": 22.0}, source="history")
        assert patterns.match_rep(_rep(_legs(analyze.IM_ORDER, per_leg=2)), bad) is None


class TestBlocks:
    """Reps that are ambiguous alone become identifiable together."""

    def _ladder(self):
        """The 2026-09-11 closing set: 50 fly, 100, 150, 200 IM, 150."""
        return _workout([
            _legs(analyze.IM_ORDER[:1], per_leg=2),      # 50 fly
            _legs(analyze.IM_ORDER[:2], per_leg=2),      # 100 fly/back
            _legs(analyze.IM_ORDER[:3], per_leg=2),      # 150 IM no free
            _legs(analyze.IM_ORDER, per_leg=2),          # 200 IM
            _legs(analyze.IM_ORDER[1:], per_leg=2),      # 150 IM no fly
        ])

    def test_an_im_ladder_is_read_end_to_end(self):
        found = patterns.detect_patterns(self._ladder(), ratios={}, anchors=[
            dict(workout_id=1, rep_id=4, reverse=False, pace=dict(PROFILE.pace))])
        assert [m.name for m in found] == [
            "fly", "fly/back", "IM no free", "IM", "IM no fly"]

    def test_the_whole_ladder_is_one_block(self):
        """Five reps of five distances are one set in the swimmer's head."""
        found = patterns.detect_patterns(self._ladder(), ratios={}, anchors=[
            dict(workout_id=1, rep_id=4, reverse=False, pace=dict(PROFILE.pace))])
        assert len({m.block for m in found}) == 1

    def test_the_two_leg_rep_is_carried_by_its_neighbours(self):
        """`fly/back` is unidentifiable alone and identified inside the block."""
        rep = _rep(_legs(analyze.IM_ORDER[:2], per_leg=2))
        assert patterns.match_rep(rep, PROFILE) is None
        found = patterns.detect_patterns(self._ladder(), ratios={}, anchors=[
            dict(workout_id=1, rep_id=4, reverse=False, pace=dict(PROFILE.pace))])
        carried = [m for m in found if m.rep_id == 2][0]
        assert carried.legs == analyze.IM_ORDER[:2]

    def test_a_block_does_not_swallow_a_freestyle_set(self):
        """Six 100s of freestyle next to a 200 IM stay freestyle."""
        df = _workout([_legs(analyze.IM_ORDER, per_leg=2)]
                      + [[22.0] * 4] * 6)
        found = patterns.detect_patterns(df, ratios={}, anchors=[
            dict(workout_id=1, rep_id=1, reverse=False, pace=dict(PROFILE.pace))])
        assert [m.name for m in found] == ["IM"]

    def test_confidence_is_lower_without_a_profile_from_the_day(self):
        df = _workout([_legs(analyze.IM_ORDER, per_leg=2)])
        ratios = {s: PROFILE.pace[s] / PROFILE.pace["freestyle"]
                  for s in analyze.IM_ORDER}
        local = patterns.detect_patterns(df, ratios=ratios, anchors=[
            dict(workout_id=1, rep_id=1, reverse=False, pace=dict(PROFILE.pace))])
        borrowed = patterns.detect_patterns(df, ratios=ratios, anchors=[])
        assert borrowed and local
        assert borrowed[0].confidence < local[0].confidence


class TestProfile:
    """Where the per-stroke reference times come from."""

    def test_an_anchor_needs_a_real_medley_gap(self):
        """A descending 300 freestyle passes 'last leg fastest' and is not an IM."""
        df = _workout([[25.9, 25.9, 24.5, 24.5, 24.8, 24.8, 21.9, 21.9]])
        assert patterns.find_anchors(df) == []

    def test_a_real_medley_anchors(self):
        df = _workout([_legs(analyze.IM_ORDER, per_leg=2)])
        anchors = patterns.find_anchors(df)
        assert len(anchors) == 1
        assert anchors[0]["pace"]["butterfly"] == pytest.approx(27.2, abs=0.1)

    def test_ratios_are_relative_to_freestyle(self):
        df = _workout([_legs(analyze.IM_ORDER, per_leg=2)])
        ratios = patterns.learn_ratios(df)
        assert ratios["freestyle"] == pytest.approx(1.0)
        assert ratios["backstroke"] == pytest.approx(32.8 / 22.0, rel=1e-3)

    def test_ratios_survive_a_round_trip_through_params(self):
        ratios = {s: 1.3 for s in analyze.IM_ORDER}
        assert patterns.from_params(patterns.as_params(ratios)) == ratios

    def test_an_incomplete_stored_ratio_set_is_ignored(self):
        assert patterns.from_params({"_stroke_ratio": {"freestyle": 1.0}}) == {}

    def test_a_workout_anchor_beats_history(self):
        df = _workout([_legs(analyze.IM_ORDER, per_leg=2)])
        p = patterns.profile_for(df, {s: 1.3 for s in analyze.IM_ORDER}, [
            dict(workout_id=1, rep_id=1, reverse=False, pace=dict(PROFILE.pace))])
        assert p.source == "workout"
        assert p.pace["backstroke"] == pytest.approx(32.8)

    def test_history_ratios_scale_to_the_workout(self):
        df = _workout([[40.0] * 4])
        p = patterns.profile_for(df, {"freestyle": 1.0, "butterfly": 1.3,
                                      "backstroke": 1.35, "breaststroke": 1.35}, [])
        assert p.source == "history"
        assert p.pace["butterfly"] == pytest.approx(40.0 * 1.3, rel=1e-3)


class TestLabelling:
    """What a match writes back onto the lengths."""

    def test_legs_are_labelled_in_medley_order(self):
        df = _workout([_legs(analyze.IM_ORDER, per_leg=2)])
        df["predicted"] = "freestyle"
        df["confidence"] = 0.5
        found = patterns.detect_patterns(df, ratios={}, anchors=[
            dict(workout_id=1, rep_id=1, reverse=False, pace=dict(PROFILE.pace))])
        out = patterns.label_patterns(df, found)
        assert out["predicted"].tolist() == [
            "butterfly", "butterfly", "backstroke", "backstroke",
            "breaststroke", "breaststroke", "freestyle", "freestyle"]
        assert out["pattern"].unique().tolist() == ["IM"]
        assert out["im_continuous"].all()

    def test_a_partial_window_is_not_a_whole_medley(self):
        """A 150 of back/breast/free is three strokes of ordinary swimming, so
        its reps stay eligible for their own personal bests."""
        df = _workout([_legs(analyze.IM_ORDER[1:], per_leg=2)])
        df["predicted"] = "freestyle"
        df["confidence"] = 0.5
        out = patterns.label_patterns(df, patterns.detect_patterns(
            df, ratios={}, anchors=[dict(workout_id=1, rep_id=1, reverse=False,
                                         pace=dict(PROFILE.pace))]))
        assert out["pattern"].unique().tolist() == ["IM no fly"]
        assert not out["im_continuous"].any()

    def test_labelling_nothing_still_adds_the_columns(self):
        df = _workout([[22.0] * 4])
        df["predicted"] = "freestyle"
        df["confidence"] = 0.5
        out = patterns.label_patterns(df, [])
        assert {"pattern", "pattern_block", "im_continuous"} <= set(out.columns)
        assert out["pattern"].isna().all()

    def test_an_empty_frame_is_handled(self):
        assert patterns.detect_patterns(pd.DataFrame()) == []


class TestPipelineWiring:
    """The pattern labels have to survive everything that runs after them."""

    def _analysed(self):
        """An IM ladder, labelled as the pipeline would label it."""
        df = _workout([
            _legs(analyze.IM_ORDER[:2], per_leg=2),
            _legs(analyze.IM_ORDER[:3], per_leg=2),
            _legs(analyze.IM_ORDER, per_leg=2),
        ])
        df["predicted"] = "freestyle"
        df["confidence"] = 0.5
        found = patterns.detect_patterns(df, ratios={}, anchors=[
            dict(workout_id=1, rep_id=3, reverse=False, pace=dict(PROFILE.pace))])
        return patterns.label_patterns(df, found)

    def test_rep_consistency_does_not_collapse_a_pattern(self):
        """A 150 of back/breast/free changes stroke twice without stopping, which
        is exactly what the majority rule would erase."""
        df = self._analysed()
        out = analyze.enforce_rep_consistency(df)
        for _, g in out.groupby("rep_id"):
            assert g["predicted"].nunique() == len(g["pattern"].iloc[0].split("/")) \
                or g["pattern"].iloc[0].startswith("IM")
        assert out.loc[out["rep_id"] == 2, "predicted"].tolist() == [
            "butterfly", "butterfly", "backstroke", "backstroke",
            "breaststroke", "breaststroke"]

    def test_a_multi_stroke_rep_is_marked_mixed(self):
        df = self._analysed()
        assert df["mixed_rep"].all()

    def test_a_single_leg_member_is_not_named_a_pattern(self):
        """The 50 fly that opens a ladder is a genuine 50 fly: it keeps its own
        stroke, stays eligible for its own best, and is not called a pattern."""
        df = _workout([
            _legs(analyze.IM_ORDER[:1], per_leg=2),
            _legs(analyze.IM_ORDER, per_leg=2),
        ])
        df["predicted"] = "freestyle"
        df["confidence"] = 0.5
        out = patterns.label_patterns(df, patterns.detect_patterns(
            df, ratios={}, anchors=[dict(workout_id=1, rep_id=2, reverse=False,
                                         pace=dict(PROFILE.pace))]))
        first = out[out["rep_id"] == 1]
        assert first["predicted"].tolist() == ["butterfly", "butterfly"]
        assert first["pattern"].isna().all()
        assert not first["mixed_rep"].any()
