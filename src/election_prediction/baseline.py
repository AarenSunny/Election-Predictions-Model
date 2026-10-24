"""
Stage 1: baseline model.  *** SKELETON: not implemented on purpose. ***

    baseline_margin ~ lean + generic_ballot + approval_signed + midterm_flag
                      + midterm_pres_penalty + incumbency_x_lean

Plain linear regression fit leave-one-year-out; residual = actual - baseline is
the target for stage 2 (train.py). (Benchmark *strategies* to beat live in
benchmarks.py; this file is the stage-1 model.)

Helpers already available:
    features.approval_signed, features.midterm_flag, features.two_party_margin,
    features.midterm_pres_penalty, features.incumbency_x_lean
    splits.out_of_fold(df, fit_predict)       leak-free predictions per held-out year
    evaluate.margin_errors / errors_by_year   RMSE, bias, winner accuracy
    probability.sigma_components              uncertainty from the residuals

Checks to run once implemented:
    * linear beats uniform swing (lean + generic_ballot) on out-of-fold RMSE
    * coefficient on lean ~ +1, generic_ballot > 0
    * residuals average ~0; the 20 biggest have recognizable explanations
    * residual vs. lean plot is flat (curve -> try splines/GAM)
"""

from __future__ import annotations

import pandas as pd

# Default only; the pipeline reads the list from configs/baseline.yaml (baseline.features).
BASELINE_FEATURES = ["lean", "generic_ballot", "approval_signed", "midterm_flag",
                     "midterm_pres_penalty", "incumbency_x_lean"]


def fit_baseline(train: pd.DataFrame, features: list[str] = BASELINE_FEATURES):
    """Fit the stage-1 model on train rows; return a fitted model."""
    raise NotImplementedError("baseline model not built yet")


def predict_baseline(model, df: pd.DataFrame, features: list[str] = BASELINE_FEATURES):
    """Predicted D margin for each row of df."""
    raise NotImplementedError("baseline model not built yet")


def baseline_residuals(df: pd.DataFrame, features: list[str] = BASELINE_FEATURES) -> pd.DataFrame:
    """
    Out-of-fold baseline + residual columns. Glue only: wires fit/predict into
    splits.out_of_fold, so it works as soon as the two functions above do.
    """
    from .splits import out_of_fold

    def fit_predict(train, test):
        return predict_baseline(fit_baseline(train, features), test, features)

    out = df.copy()
    out["baseline"] = out_of_fold(out, fit_predict)
    out["residual"] = out["margin"] - out["baseline"]
    return out
