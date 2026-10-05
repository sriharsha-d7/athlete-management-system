"""Smooth probability calibration.

Isotonic regression is the textbook choice but it produces a step function: small counterfactual
changes ("+0.5 h sleep") land on a flat plateau and the estimated effect collapses to exactly zero.
Platt scaling (a logistic regression on the model's log-odds) is strictly monotone and smooth, so
marginal effects survive calibration, which the recommendation engine depends on.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    def fit(self, raw_p, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_p)[:, None], y)
        return self

    def predict(self, raw_p):
        return self.lr.predict_proba(_logit(raw_p)[:, None])[:, 1]
