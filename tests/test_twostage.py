"""
election_prediction.twostage: the test year must never influence its own predictions,
sigma or probability calibration, and the residual model must only see training years.
"""

import numpy as np
import pandas as pd

from election_prediction.baseline import baseline_residuals
from election_prediction.twostage import (
    nested_two_stage,
    no_residual,
    probability_report,
    ridge_residual,
)

COLS = ["baseline", "resid_pred", "pred_margin", "sigma", "p_normal", "p_t", "p_logistic"]


def _table(seed=0, years=range(2010, 2026, 2), n=80):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        swing = rng.normal(0, 0.03)
        lean = rng.uniform(-0.5, 0.5, n)
        x = rng.normal(0, 1, n)
        poll = np.where(rng.random(n) < 0.3, lean + swing + rng.normal(0, 0.05, n), np.nan)
        rows.append(pd.DataFrame({"year": y, "lean": lean, "x": x, "poll_margin": poll,
                                  "margin": lean + swing + 0.04 * x + rng.normal(0, 0.05, n)}))
    return pd.concat(rows, ignore_index=True)


def test_test_year_never_influences_its_own_predictions():
    t = _table()
    a = nested_two_stage(t, ["lean"], ridge_residual(["x"]))
    t2 = t.copy()
    t2.loc[t2["year"] == 2016, "margin"] *= -3            # scramble one year's answers
    b = nested_two_stage(t2, ["lean"], ridge_residual(["x"]))
    m = t["year"] == 2016
    for c in COLS:                                         # its predictions, sigma, probabilities
        assert np.allclose(a.loc[m, c], b.loc[m, c]), c
    assert not np.allclose(a.loc[~m, "pred_margin"], b.loc[~m, "pred_margin"])  # others did use it


def test_residual_model_only_sees_training_years_with_inner_residuals():
    t = _table()
    seen = []

    def spy(train, test):
        (test_year,) = set(test["year"])
        assert test_year not in set(train["year"])
        # inner residuals: recomputed without the outer test year, so they differ from
        # the all-years out-of-fold residuals the old CSV would have used
        assert "residual" in train and train["residual"].notna().all()
        # poll_gap built from the fold's own baseline
        has = train["poll_margin"].notna()
        assert np.allclose(train.loc[has, "poll_gap"],
                           train.loc[has, "poll_margin"] - train.loc[has, "baseline"])
        seen.append((test_year, len(set(train["year"]))))
        return np.zeros(len(test))

    nested_two_stage(t, ["lean"], spy)
    outer = [s for s in seen if s[1] == 7]                 # 8 years: outer fits train on 7
    assert len(outer) == 8 and any(s[1] == 6 for s in seen)  # inner (sigma) fits train on 6


def test_inner_residuals_differ_from_the_saved_csv_kind():
    t = _table()
    saved = baseline_residuals(t, ["lean"])                # what the old CSV holds
    captured = {}

    def grab(train, test):
        if len(set(train["year"])) == 7:
            captured[int(test["year"].iloc[0])] = train["residual"].to_numpy()
        return np.zeros(len(test))
    nested_two_stage(t, ["lean"], grab)
    old = saved.loc[saved["year"] != 2016, "residual"].to_numpy()
    assert not np.allclose(captured[2016], old)            # rebuilt without 2016


def test_no_residual_equals_plain_baseline_and_ridge_helps():
    t = _table()
    base = nested_two_stage(t, ["lean"], no_residual)
    loyo = baseline_residuals(t, ["lean"])
    assert np.allclose(base["pred_margin"], loyo["baseline"])
    ridge = nested_two_stage(t, ["lean"], ridge_residual(["x"]))
    assert np.sqrt((ridge["error"] ** 2).mean()) < np.sqrt((base["error"] ** 2).mean())


def test_probabilities_are_valid_and_reported():
    res = nested_two_stage(_table(), ["lean"], no_residual)
    for m in ("normal", "t", "logistic"):
        assert res[f"p_{m}"].between(0, 1).all()
    rep = probability_report(res)
    assert set(rep["method"]) == {"normal", "t", "logistic"} and (rep["log_loss"] > 0).all()
