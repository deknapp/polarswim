"""Learning a stroke classifier from the swimmer's corrections.

The contract these tests defend, in order of precedence: a correction outranks
everything, a fitted model outranks the rules where it is confident, and the rules
survive wherever neither has earned the right to speak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from polarswim import learn


def _frame(n_per_class=12, workout_id=1):
    """Two clearly separable strokes: slow-and-cheap against slow-and-costly.

    Deliberately the pair the rules cannot split — the dead band between the two
    cost thresholds is exactly where corrections are worth having.
    """
    rows, idx = [], 1
    spec = [("breaststroke", 33.0, 13.0), ("backstroke", 31.0, 41.0)]
    rng = np.random.default_rng(0)
    for set_id, (stroke, pace, cost) in enumerate(spec, start=1):
        for _ in range(n_per_class):
            rows.append(dict(
                workout_id=workout_id, idx=idx, set_id=set_id, rep_id=idx,
                pace_s=pace + rng.normal(0, 0.6),
                hr_cost=cost + rng.normal(0, 1.5),
                set_rest_s=20.0, set_cv=0.05, pace_rel=1.0, set_size=n_per_class,
                predicted="undetermined", confidence=0.33, truth=stroke))
            idx += 1
    return pd.DataFrame(rows)


def _labels(df):
    return {(int(r.workout_id), int(r.idx)): r.truth for r in df.itertuples()}


class TestFitting:
    def test_a_model_learns_one_gaussian_per_stroke(self):
        df = _frame()
        m = learn.fit(df, _labels(df))
        assert set(m.classes) == {"breaststroke", "backstroke"}
        assert m.is_usable()

    def test_the_learned_means_match_the_labelled_data(self):
        df = _frame()
        m = learn.fit(df, _labels(df))
        pace = m.features.index("pace_s")
        assert m.means["breaststroke"][pace] == pytest.approx(33.0, abs=0.5)
        assert m.means["backstroke"][pace] == pytest.approx(31.0, abs=0.5)

    def test_too_few_labels_produces_no_usable_model(self):
        """A confident answer from four examples is worse than an honest one."""
        df = _frame(n_per_class=3)
        assert not learn.fit(df, _labels(df)).is_usable()

    def test_a_thin_class_is_left_out_rather_than_guessed(self):
        df = _frame()
        labels = _labels(df)
        # Demote all but a handful of one class.
        thin = [k for k, v in labels.items() if v == "backstroke"][:-2]
        for k in thin:
            labels.pop(k)
        m = learn.fit(df, labels)
        assert "backstroke" not in m.classes

    def test_undetermined_is_never_learned_as_a_stroke(self):
        """A swimmer never means 'this was undetermined stroke'."""
        df = _frame()
        labels = {k: "undetermined" for k in _labels(df)}
        assert learn.fit(df, labels).classes == []

    def test_a_constant_feature_cannot_become_infinite_confidence(self):
        """One set swum to a metronome gives zero variance, and a Gaussian turns
        that into certainty."""
        df = _frame()
        df["set_rest_s"] = 20.0
        m = learn.fit(df, _labels(df))
        assert all((v > 0).all() for v in m.variances.values())


class TestPrecedence:
    def test_the_model_overrides_the_rules_where_confident(self):
        df = _frame()
        out = learn.apply(df, learn.fit(df, _labels(df)))
        assert set(out["predicted"]) <= {"breaststroke", "backstroke"}
        assert (out["label_source"] == "model").any()

    def test_the_rules_survive_where_the_model_is_unsure(self):
        df = _frame()
        out = learn.apply(df, learn.fit(df, _labels(df)), min_confidence=1.01)
        assert (out["predicted"] == "undetermined").all()
        assert (out["label_source"] == "rules").all()

    def test_an_unusable_model_changes_nothing(self):
        df = _frame(n_per_class=3)
        out = learn.apply(df, learn.fit(df, _labels(df)))
        assert out["predicted"].tolist() == df["predicted"].tolist()

    def test_a_correction_outranks_the_model_that_trained_on_it(self):
        df = _frame()
        model = learn.fit(df, _labels(df))
        out = learn.apply(df, model)
        out = learn.apply_labels(out, {(1, 1): "butterfly"})
        row = out[out["idx"] == 1].iloc[0]
        assert row["predicted"] == "butterfly"
        assert row["confidence"] == 1.0
        assert row["label_source"] == "corrected"

    def test_lengths_without_a_correction_are_untouched(self):
        df = _frame()
        out = learn.apply_labels(df, {(1, 1): "butterfly"})
        assert (out[out["idx"] != 1]["predicted"] == "undetermined").all()


class TestRoundTrip:
    def test_a_model_survives_being_written_out_and_read_back(self):
        """It is stored as numbers in `model_params`, not a pickle, so this is the
        test that the stored form is complete."""
        df = _frame()
        original = learn.fit(df, _labels(df))
        restored = learn.from_params(original.as_params())
        assert set(restored.classes) == set(original.classes)
        for c in original.classes:
            assert restored.means[c] == pytest.approx(original.means[c])
            assert restored.variances[c] == pytest.approx(original.variances[c])

    def test_the_rule_thresholds_are_not_mistaken_for_a_class(self):
        """`_global` shares the table with the fitted model."""
        params = {"_global": {"pace_p30": 24.0, "cost_p67": 30.0}}
        assert learn.from_params(params).classes == []


class TestAccuracyReporting:
    def test_accuracy_is_measurable_once_there_are_corrections(self):
        df = _frame(n_per_class=20)
        # Four sets rather than two, so whole sets can be held out.
        df["set_id"] = (df["idx"] - 1) // 10 + 1
        result = learn.cross_validate(df, _labels(df))
        assert result["accuracy"] is not None
        assert result["accuracy"] > 50

    def test_it_holds_out_whole_sets_not_lengths(self):
        """Lengths within a set are near-duplicates; splitting on them leaks the
        answer across the fold boundary and flatters the score.

        With one set per stroke, holding a set out removes that stroke from
        training entirely — so there is nothing honest to report, and it says so
        rather than scoring the leaked split it could have taken instead."""
        df = _frame()
        result = learn.cross_validate(df, _labels(df))
        assert result["accuracy"] is None
        assert "hold any out" in result["reason"]

    def test_it_says_nothing_rather_than_a_number_when_labels_are_thin(self):
        df = _frame(n_per_class=4)
        result = learn.cross_validate(df, _labels(df))
        assert result["accuracy"] is None and result["n"] == 8

    def test_the_confusion_matrix_names_both_sides(self):
        df = _frame(n_per_class=20)
        df["set_id"] = (df["idx"] - 1) // 10 + 1
        for row in learn.cross_validate(df, _labels(df))["confusion"]:
            assert {"actual", "predicted", "n"} <= set(row)
