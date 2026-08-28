"""Learn a stroke classifier from the swimmer's own corrections.

The rule-based classifier in `analyze` works with no labels at all, which is the
only reason this project functions — the vendor labelled nothing and there was no
ground truth to train against. But rules pay for that independence with a dead
band: told a length is slow and its heart-rate cost sits between the two global
thresholds, they refuse to choose between breaststroke and backstroke and return
`undetermined`. That is honest, and it is also throwing away evidence once the
swimmer has said "this set was breaststroke" even once.

So: corrections train a model, and the model replaces the rules where it has
earned the right to.

**Gaussian naive Bayes, written out in numpy.** With six features and six classes,
a model fitted from tens of labels must be low-variance or it will memorise the
handful of sets it was given. A per-class Gaussian is about the simplest thing
that can express "your breaststroke is slow AND cheap, your backstroke is slow AND
expensive" — which is exactly the structure the rules encode by hand, learned from
data instead of asserted. Gradient boosting would fit the training labels better
and generalise worse.

Writing it here rather than importing scikit-learn keeps the project pip-only, but
the real reason is inspectability: the fitted parameters are per-class means and
variances, which go into `model_params` as numbers a person can read and sanity
check. A pickled estimator would make the model the one part of this database you
could not interrogate.

**It refuses to run on too little data.** Below `MIN_LABELS_PER_CLASS` examples a
class is not modelled at all, and below `MIN_LABELS_TOTAL` the whole model steps
aside for the rules. A confident answer from four examples is worse than an honest
`undetermined`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The axes the rules already use, plus the two they compute and ignore. Rest and
# set size were only ever read as tie-breakers; a fitted model can weigh them.
FEATURES = ("pace_s", "hr_cost", "set_rest_s", "set_cv", "pace_rel", "set_size")

MIN_LABELS_PER_CLASS = 8     # below this a class has no usable spread
MIN_LABELS_TOTAL = 20        # below this the rules are simply better
# Variance floor. One label set swum at a metronome pace gives a near-zero
# variance, which a Gaussian turns into infinite confidence for that class.
VAR_FLOOR = 1e-3


@dataclass
class StrokeModel:
    """Per-class means and variances over `FEATURES`, plus class priors."""
    classes: list[str] = field(default_factory=list)
    means: dict[str, np.ndarray] = field(default_factory=dict)
    variances: dict[str, np.ndarray] = field(default_factory=dict)
    priors: dict[str, float] = field(default_factory=dict)
    n_labels: int = 0
    features: tuple[str, ...] = FEATURES

    def is_usable(self) -> bool:
        return len(self.classes) >= 2 and self.n_labels >= MIN_LABELS_TOTAL

    def log_likelihood(self, x: np.ndarray) -> dict[str, float]:
        """Log P(class | features), up to the constant that cancels in argmax."""
        out = {}
        for c in self.classes:
            mu, var = self.means[c], self.variances[c]
            ok = ~np.isnan(x)                     # a missing feature abstains
            if not ok.any():
                continue
            ll = -0.5 * np.sum(np.log(2 * np.pi * var[ok])
                               + (x[ok] - mu[ok]) ** 2 / var[ok])
            out[c] = float(ll + np.log(self.priors[c]))
        return out

    def predict(self, x: np.ndarray) -> tuple[str, float] | None:
        """Best class and a calibrated-ish confidence, or None if it cannot say."""
        lls = self.log_likelihood(x)
        if not lls:
            return None
        names = list(lls)
        values = np.array([lls[n] for n in names])
        values -= values.max()                    # softmax, numerically safe
        p = np.exp(values)
        p /= p.sum()
        best = int(p.argmax())
        return names[best], float(p[best])

    def as_params(self) -> dict[str, dict[str, float]]:
        """Flatten into `model_params` rows, so the model is readable in SQL."""
        out: dict[str, dict[str, float]] = {}
        for c in self.classes:
            row = {"prior": self.priors[c], "n_labels": float(self.n_labels)}
            for i, name in enumerate(self.features):
                row[f"mean_{name}"] = float(self.means[c][i])
                row[f"var_{name}"] = float(self.variances[c][i])
            out[c] = row
        return out


def from_params(params: dict[str, dict[str, float]]) -> StrokeModel:
    """Rebuild a fitted model from its `model_params` rows.

    The model is stored, not refitted per request: fitting needs every labelled
    length in the history, so a per-workout view that refitted would pay for a
    full-database pass to render one card.
    """
    model = StrokeModel()
    for name, row in params.items():
        if name.startswith("_") or "prior" not in row:
            continue                     # the rule thresholds live here too
        means = np.array([row.get(f"mean_{f}", np.nan) for f in FEATURES])
        variances = np.array([row.get(f"var_{f}", np.nan) for f in FEATURES])
        if np.isnan(means).all():
            continue
        model.classes.append(name)
        model.means[name] = np.where(np.isnan(means), 0.0, means)
        model.variances[name] = np.where(
            np.isnan(variances) | (variances < VAR_FLOOR), VAR_FLOOR, variances)
        model.priors[name] = row["prior"]
        model.n_labels = int(row.get("n_labels", 0))
    return model


def fit(df: pd.DataFrame, labels: dict[tuple[int, int], str]) -> StrokeModel:
    """Fit per-class Gaussians to the labelled lengths in `df`.

    `df` is the analysed frame — repaired paces and set features already attached,
    so the model is fitted on exactly the numbers it will later be asked to
    classify. Fitting on raw observations instead would train it on merged lengths
    that the classifier never sees.
    """
    model = StrokeModel()
    if df.empty or not labels:
        return model

    key = list(zip(df["workout_id"], df["idx"]))
    truth = [labels.get(k) for k in key]
    # Not `_truth`: itertuples renames any leading-underscore column to a
    # positional `_1`, and the attribute access below would silently break.
    frame = df.assign(truth_stroke=truth)
    frame = frame[frame["truth_stroke"].notna()]
    if frame.empty:
        return model

    missing = [f for f in FEATURES if f not in frame.columns]
    if missing:
        return model

    model.n_labels = int(len(frame))
    for stroke, g in frame.groupby("truth_stroke"):
        # 'undetermined' is the absence of an answer, not an answer; a swimmer
        # never means "this was undetermined stroke".
        if stroke == "undetermined" or len(g) < MIN_LABELS_PER_CLASS:
            continue
        x = g[list(FEATURES)].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(x, axis=0)
            var = np.nanvar(x, axis=0)
        # A feature never observed for this class cannot inform it. Give it an
        # infinite variance so it contributes nothing rather than NaN.
        var = np.where(np.isnan(var) | (var < VAR_FLOOR), VAR_FLOOR, var)
        mu = np.where(np.isnan(mu), 0.0, mu)
        model.classes.append(str(stroke))
        model.means[str(stroke)] = mu
        model.variances[str(stroke)] = var
        model.priors[str(stroke)] = len(g) / len(frame)
    return model


def apply(df: pd.DataFrame, model: StrokeModel,
          min_confidence: float = 0.55) -> pd.DataFrame:
    """Overwrite rule predictions where the fitted model is confident.

    The rules keep every length the model is unsure about. This is the same
    principle as `undetermined`: a model that has seen twenty sets should improve
    the answers it has evidence for and leave the rest alone.
    """
    if not model.is_usable():
        return df

    df = df.copy()
    x = df[list(model.features)].to_numpy(dtype=float)
    predicted, confidence, source = [], [], []
    for i, row in enumerate(x):
        guess = model.predict(row)
        if guess and guess[1] >= min_confidence:
            predicted.append(guess[0])
            confidence.append(round(guess[1], 3))
            source.append("model")
        else:
            predicted.append(df["predicted"].iloc[i])
            confidence.append(float(df["confidence"].iloc[i]))
            source.append("rules")
    df["predicted"] = predicted
    df["confidence"] = confidence
    df["label_source"] = source
    return df


def apply_labels(df: pd.DataFrame, labels: dict[tuple[int, int], str]) -> pd.DataFrame:
    """Stamp the swimmer's own corrections over everything else.

    Last word, always. A correction is ground truth: it outranks the rules, the
    fitted model, and the structural medley detection, and re-running any of them
    must not disturb it.
    """
    df = df.copy()
    if "label_source" not in df.columns:
        df["label_source"] = "rules"
    if not labels:
        return df
    key = list(zip(df["workout_id"], df["idx"]))
    hit = [k in labels for k in key]
    if not any(hit):
        return df
    df.loc[hit, "predicted"] = [labels[k] for k in key if k in labels]
    df.loc[hit, "confidence"] = 1.0
    df.loc[hit, "label_source"] = "corrected"
    return df


def cross_validate(df: pd.DataFrame, labels: dict[tuple[int, int], str],
                   folds: int = 5) -> dict:
    """Held-out accuracy over the labelled lengths, and what it confuses.

    This is the number the project could not previously report at all: with no
    ground truth, self-consistency was measurable and accuracy was not. It is
    still only accuracy over the sets the swimmer chose to correct — which are
    disproportionately the ones the rules got wrong — so it reads pessimistically,
    and that is the honest direction to be wrong in.
    """
    key = list(zip(df["workout_id"], df["idx"]))
    frame = df.assign(truth_stroke=[labels.get(k) for k in key])
    frame = frame[frame["truth_stroke"].notna() & (frame["truth_stroke"] != "undetermined")]
    if len(frame) < MIN_LABELS_TOTAL:
        return {"n": int(len(frame)), "accuracy": None,
                "reason": f"need at least {MIN_LABELS_TOTAL} corrections"}

    # Split by SET, not by length. Lengths within a set are near-duplicates, so
    # splitting by length leaks the answer across the fold boundary and reports an
    # accuracy far better than the model would achieve on a new practice.
    groups = frame.groupby(["workout_id", "set_id"]).ngroup().to_numpy()
    unique = np.unique(groups)
    if len(unique) < 2:
        return {"n": int(len(frame)), "accuracy": None,
                "reason": "corrections cover only one set"}
    rng = np.random.default_rng(0)              # deterministic: a report, not a game
    rng.shuffle(unique)
    chunks = np.array_split(unique, min(folds, len(unique)))

    correct = 0
    total = 0
    confusion: dict[tuple[str, str], int] = {}
    for held in chunks:
        train = frame[~np.isin(groups, held)]
        test = frame[np.isin(groups, held)]
        if train.empty or test.empty:
            continue
        sub = {(int(r.workout_id), int(r.idx)): r.truth_stroke
               for r in train.itertuples()}
        model = fit(train, sub)
        if not model.is_usable():
            continue
        for r in test.itertuples():
            x = np.array([getattr(r, f) for f in FEATURES], dtype=float)
            guess = model.predict(x)
            if guess is None:
                continue
            total += 1
            correct += guess[0] == r.truth_stroke
            pair = (str(r.truth_stroke), guess[0])
            confusion[pair] = confusion.get(pair, 0) + 1

    if not total:
        return {"n": int(len(frame)), "accuracy": None,
                "reason": "not enough corrections per stroke to hold any out"}
    return {
        "n": int(len(frame)),
        "n_evaluated": total,
        "accuracy": round(100 * correct / total, 1),
        "confusion": [{"actual": a, "predicted": p, "n": n}
                      for (a, p), n in sorted(confusion.items(), key=lambda kv: -kv[1])],
    }
