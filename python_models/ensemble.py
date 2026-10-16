"""Soft-voting ensemble of already-fitted classifiers.

The ensemble averages the class probabilities of its members (optionally
weighted). All members must have been trained on the same integer-encoded
labels, so their ``predict_proba`` columns line up.
"""
from __future__ import annotations

import numpy as np


class SoftVotingEnsemble:
    def __init__(self, members: list[tuple[str, object]], weights: list[float] | None = None):
        self.members = members
        self.weights = np.asarray(weights if weights else [1.0] * len(members), dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.classes_ = getattr(members[0][1], "classes_", None)

    def predict_proba(self, X):
        total = None
        for (_, model), w in zip(self.members, self.weights):
            p = np.asarray(model.predict_proba(X), dtype=float) * w
            total = p if total is None else total + p
        return total

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def __repr__(self) -> str:
        names = ", ".join(n for n, _ in self.members)
        return f"SoftVotingEnsemble({names})"


class CriticalClassWeighting:
    """Decision weighting for safety-critical classes.

    Multiplies each class probability by a weight (1.0 = unchanged) and
    renormalises. The weights are tuned on the *validation* split to raise the
    recall of critical classes (a missed gunshot costs more than a false alarm,
    and false alarms are caught by manual review) while macro F1 may drop by
    at most one point. The test split is only used to report the result.
    """

    def __init__(self, base, weights):
        self.base = base
        self.weights = np.asarray(weights, dtype=float)
        self.classes_ = getattr(base, "classes_", None)

    def predict_proba(self, X):
        p = np.asarray(self.base.predict_proba(X), dtype=float) * self.weights
        return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def __repr__(self) -> str:
        return f"CriticalClassWeighting({self.base!r}, weights={np.round(self.weights, 2).tolist()})"
