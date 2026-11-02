"""
Stage 1: baseline model. Plain linear regression on the inputs listed under
baseline.features in configs/baseline.yaml (v1: lean, generic_ballot,
incumbency_signed, midterm_x_pres; full input list in docs/BASELINE_CHANGES.md).

baseline_margin = a + b1 * lean + b2 * generic_ballot + b3 * incumbency_signed + b4 * midterm_x_pres

Plain linear regression fit leave-one-year-out 
residual = actual - baseline is the target for stage 2 (train.py). (Benchmark *strategies* to beat live in
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
from sklearn.linear_model import LinearRegression

# Default only; the pipeline reads the list from configs/baseline.yaml (baseline.features).
BASELINE_FEATURES = ["lean", "generic_ballot", "incumbency_signed", "midterm_x_pres"]

def fit_baseline(train: pd.DataFrame, features: list[str] = BASELINE_FEATURES):
    """Fit the stage-1 model on train rows ('margin' is the target); return a fitted model."""
    missing = [f for f in features + ["margin"] if f not in train.columns] #Check table has every required column
    if missing:                                                            #Shows which columns are missing, and insert missing
        raise KeyError(f"baseline training table is missing columns {missing}")
    rows = train.dropna(subset=features + ["margin"])                      #Removes the training rows with missing values
    return LinearRegression().fit(rows[features], rows["margin"])          #Creates model, learns intercept and coefficients, give trained model back to whoever called function


#Predict Democrat margin for each row of df
def predict_baseline(model, df: pd.DataFrame, features: list[str] = BASELINE_FEATURES):
    return model.predict(df[features])
#Defines a function that receives, trained model, tabke of races(df), inoput columns to use

#Coefficients fit on ALL years (for the sign/size check, not for predictions).
def describe_baseline(df: pd.DataFrame, features: list[str] = BASELINE_FEATURES) -> dict:
    m = fit_baseline(df, features)
    return {"intercept": float(m.intercept_), **dict(zip(features, map(float, m.coef_)))}
#defines a function that returns a dictionary of the model's intercept and coefficients
#return: intercept, coefficient for each feature
#map(float, m.coef) converts each coefficient to a float, zip pair each fature name with coefficient, pairs are turned into a dictionary


## Out-of-fold baseline + residual columns. Glue only: wires fit/predict into
## splits.out_of_fold, so it works as soon as the two functions above do.
def baseline_residuals(df: pd.DataFrame, features: list[str] = BASELINE_FEATURES) -> pd.DataFrame:  #evaluate on hidden years
    
    from .splits import out_of_fold
    
    def fit_predict(train, test):
        return predict_baseline(fit_baseline(train, features), test, features)

    out = df.copy()
    out["baseline"] = out_of_fold(out, fit_predict)
    out["residual"] = out["margin"] - out["baseline"]
    return out
