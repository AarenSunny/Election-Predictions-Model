"""
Predicted margin -> win probability, and the uncertainty (sigma) behind it.

Flow once a model exists:
    sig = sigma_components(oof_df)                      # out-of-fold errors
    p = prob_from_margin(pred_margin, sig["sigma_total"], dist="t")
    cal = fit_isotonic(p_oof, y_oof)                    # optional recalibration
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

EPS = 1e-6


def prob_from_margin(margin, sigma, dist: str = "normal", t_df: float = 5):
    """
    P(D wins) for a predicted D margin with uncertainty sigma (same units).
    dist="t": Student-t with the same standard deviation but fatter tails,
    i.e. more upsets far from 50/50, which usually fits elections better.
    """
    z = np.asarray(margin, dtype=float) / np.asarray(sigma, dtype=float)
    if dist == "normal":
        return stats.norm.cdf(z)
    if dist == "t":
        if t_df <= 2:
            raise ValueError("t_df must be > 2")
        return stats.t.cdf(z * np.sqrt(t_df / (t_df - 2)), df=t_df)
    raise ValueError("dist must be 'normal' or 't'")


def sigma_components(df: pd.DataFrame, error_col: str = "error", year_col: str = "year") -> dict:
    """
    Split out-of-fold errors (actual - predicted margin):
      sigma_national  spread of each year's average miss (whole country off together)
      sigma_race      spread within a year after removing that year's miss
      sigma_total     sqrt(sum of squares): use this per race for picks
    """
    yearly = df.groupby(year_col)[error_col].mean()
    within = df[error_col] - df.groupby(year_col)[error_col].transform("mean")
    s_nat = float(yearly.std(ddof=1)) if len(yearly) > 1 else 0.0
    s_race = float(within.std(ddof=1))
    return {"sigma_national": s_nat, "sigma_race": s_race,
            "sigma_total": float(np.hypot(s_nat, s_race)), "n_years": int(len(yearly))}


def fit_isotonic(p_oof, y_oof) -> IsotonicRegression:
    """Monotone recalibration. Fit ONLY on out-of-fold predictions."""
    iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
    iso.fit(np.asarray(p_oof, dtype=float), np.asarray(y_oof, dtype=float))
    return iso
