"""
Honest evaluation of baseline + residual model together, and of the win probabilities.

Why this file exists
--------------------
`data/processed/baseline_residuals.csv` is out-of-fold, but each of its residuals came
from a baseline fit on ALL other years. Training the residual model on that file and
testing on, say, 2024 would leak 2024: the 2018 residuals were computed by a baseline
that had seen 2024. So that file is for looking at misses only.

Here every test year is handled from scratch, using only the other years:

    for each test year Y:
        train = all years except Y
        1. inner residuals: for each training year Z, fit the baseline on train minus Z
           and predict Z  ->  residual = margin - that prediction   (Y never involved)
        2. fold-dependent features (e.g. poll_gap = poll margin - baseline) built from
           those inner baselines for training rows, from the step-3 baseline for Y
        3. fit the baseline on all of train -> baseline prediction for Y
        4. fit the residual model on train (target = inner residual) -> correction for Y.
           The residual model only ever receives training rows, so feature selection
           and early stopping inside it can't see Y.
        5. uncertainty for Y: repeat step 4 leaving out each training year Z in turn to
           get combined held-out errors for the training years; sigma for Y comes from
           those errors only (Y's own errors never set its sigma)
        6. probabilities for Y, three ways, each fit on training years only:
           normal(sigma), Student-t(sigma), logistic(predicted margin) calibration

Run:
    PYTHONPATH=src python3 -m election_prediction.twostage --config configs/baseline.yaml
    (add --residual linear to test the harness with a simple ridge residual model,
     --log to append to reports/experiments.csv)

Residual models are plugged in as  fit_predict(train_df, test_df) -> array, where
train_df has a 'residual' column (the inner residual) and both frames have the
fold-dependent features. 'none' predicts 0, so the combined model = the baseline.
"""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import pandas as pd

from .baseline import fit_baseline, predict_baseline
from .splits import leave_one_year_out, out_of_fold

ResidualFitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]
FoldFeatures = Callable[[pd.DataFrame, str], pd.DataFrame]

PROB_METHODS = ["normal", "t", "logistic"]


# ------------------------------------------------------------------ plug-ins
def no_residual(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Residual model that predicts 0: combined = baseline (the reference to beat)."""
    return np.zeros(len(test))


def ridge_residual(features: list[str], alpha: float = 10.0) -> ResidualFitPredict:
    """Simple stand-in residual model (ridge regression, blanks -> training mean).
    Used to test the harness; the real stage 2 is LightGBM (train.fit_residual_model)."""
    from sklearn.linear_model import Ridge

    def fit_predict(train, test):
        fill = train[features].mean()
        m = Ridge(alpha=alpha).fit(train[features].fillna(fill), train["residual"])
        return m.predict(test[features].fillna(fill))
    return fit_predict


def lightgbm_residual(features: list[str], params: dict) -> ResidualFitPredict:
    """Stage 2 from train.py. Its fit gets ONLY training rows (early stopping must
    hold out a training year itself, e.g. with last_training_year_split)."""
    from .train import fit_residual_model, predict_residual

    def fit_predict(train, test):
        return predict_residual(fit_residual_model(train, features, params), test, features)
    return fit_predict


def last_training_year_split(train: pd.DataFrame, year_col: str = "year"):
    """(fit_rows, validation_rows): hold out the latest training year, for early stopping
    inside a residual model without touching the test year."""
    last = train[year_col].max()
    return train[train[year_col] != last], train[train[year_col] == last]


def poll_gap_features(df: pd.DataFrame, baseline_col: str) -> pd.DataFrame:
    """Default fold-dependent features: poll_gap = poll_margin - the fold's baseline."""
    if "poll_margin" not in df:
        return df
    from .features import poll_gap
    out = df.copy()
    out["poll_gap"] = poll_gap(out["poll_margin"], out[baseline_col])
    return out


# ------------------------------------------------------------------ core
def _baseline_fit_predict(features):
    def fp(train, test):
        return predict_baseline(fit_baseline(train, features), test, features)
    return fp


def _probabilities(train_pred, train_y, sigma, test_pred, dist_t_df: float) -> dict:
    """Test-year P(D wins) three ways; anything fitted uses training-year rows only."""
    from sklearn.linear_model import LogisticRegression

    from .probability import prob_from_margin
    logit = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.asarray(train_pred).reshape(-1, 1), np.asarray(train_y))
    return {
        "normal": prob_from_margin(test_pred, sigma, dist="normal"),
        "t": prob_from_margin(test_pred, sigma, dist="t", t_df=dist_t_df),
        "logistic": logit.predict_proba(np.asarray(test_pred).reshape(-1, 1))[:, 1],
    }


def nested_two_stage(table: pd.DataFrame, baseline_features: list[str],
                     residual_fp: ResidualFitPredict = no_residual,
                     fold_features: FoldFeatures | None = poll_gap_features,
                     t_df: float = 5, year_col: str = "year") -> pd.DataFrame:
    """
    Held-out predictions for every row (see module docstring). Returns the input
    rows plus:
        baseline, resid_pred, pred_margin, error      margins (test year never seen)
        sigma_national, sigma_race, sigma              uncertainty set for that test year
        p_normal, p_t, p_logistic                      P(D wins) three ways
        d_won                                          1 if the Democrat won
    """
    fold_features = fold_features or (lambda d, _c: d)
    table = table.reset_index(drop=True)
    base_fp = _baseline_fit_predict(baseline_features)
    out = []
    for fold in leave_one_year_out(table, year_col):
        train = table.loc[fold.train_idx].copy()
        test = table.loc[fold.test_idx].copy()

        # 1-2: inner residuals + fold features for training rows (test year not involved)
        train["baseline"] = out_of_fold(train, base_fp, year_col)
        train["residual"] = train["margin"] - train["baseline"]
        train = fold_features(train, "baseline")

        # 3: baseline for the test year from all training years
        test["baseline"] = base_fp(train, test)
        test = fold_features(test, "baseline")

        # 4: residual model sees training rows only
        test["resid_pred"] = np.asarray(residual_fp(train, test), dtype=float)
        test["pred_margin"] = test["baseline"] + test["resid_pred"]

        # 5: combined held-out errors inside the training years -> sigma for this test year
        train["resid_pred"] = out_of_fold(train, residual_fp, year_col)
        train["pred_margin"] = train["baseline"] + train["resid_pred"]
        from .probability import sigma_components
        sig = sigma_components(train.assign(error=train["margin"] - train["pred_margin"]),
                               year_col=year_col)
        test["sigma_national"] = sig["sigma_national"]
        test["sigma_race"] = sig["sigma_race"]
        test["sigma"] = sig["sigma_total"]

        # 6: probabilities, calibrated on training years only
        probs = _probabilities(train["pred_margin"], (train["margin"] > 0).astype(int),
                               sig["sigma_total"], test["pred_margin"], t_df)
        for k, v in probs.items():
            test[f"p_{k}"] = v
        out.append(test)
    res = pd.concat(out).sort_index()
    res["error"] = res["margin"] - res["pred_margin"]
    res["d_won"] = (res["margin"] > 0).astype(int)
    return res


# ------------------------------------------------------------------ reports
def margin_report(res: pd.DataFrame, recent_from: int = 2018) -> pd.DataFrame:
    """RMSE / winner accuracy: baseline alone vs combined, all years and recent years."""
    from .evaluate import margin_errors
    rows = []
    for label, mask in (("all years", res["year"] > 0), (f"{recent_from}+", res["year"] >= recent_from)):
        b = margin_errors(res.loc[mask, "margin"], res.loc[mask, "baseline"])
        c = margin_errors(res.loc[mask, "margin"], res.loc[mask, "pred_margin"])
        rows.append({"years": label, "n": int(mask.sum()), "baseline_rmse": b["rmse"],
                     "combined_rmse": c["rmse"], "baseline_winner_acc": b["winner_accuracy"],
                     "combined_winner_acc": c["winner_accuracy"]})
    return pd.DataFrame(rows)


def probability_report(res: pd.DataFrame) -> pd.DataFrame:
    """Log loss and Brier (lower = better) for each probability method, on held-out years."""
    from .evaluate import brier, log_loss
    return pd.DataFrame([{"method": m, "log_loss": log_loss(res[f"p_{m}"], res["d_won"]),
                          "brier": brier(res[f"p_{m}"], res["d_won"])} for m in PROB_METHODS])


def run(cfg: dict, residual: str = "none", log: bool = False, verbose: bool = True) -> pd.DataFrame:
    from .config import repo_path, validate_config
    from .evaluate import (
        brier,
        calibration_table,
        errors_by_year,
        log_experiment,
        log_loss,
        underdog_check,
    )
    from .train import build_training_table

    validate_config(cfg)
    base_feats = list(cfg["baseline"]["features"])
    table = build_training_table(cfg, verbose=False)
    blank = table[base_feats].isna().any(axis=1)
    table = table[~blank].reset_index(drop=True)

    resid_feats = [f for f in cfg.get("residual_model", {}).get("features", []) if f in table]
    if residual == "none":
        fp = no_residual
    elif residual == "linear":
        if not resid_feats:
            raise ValueError("no residual_model.features found in the training table yet")
        fp = ridge_residual(resid_feats)
    elif residual == "lightgbm":
        fp = lightgbm_residual(resid_feats, cfg["residual_model"].get("lightgbm", {}))
    else:
        raise ValueError("residual must be none | linear | lightgbm")

    prob = cfg.get("probability", {})
    res = nested_two_stage(table, base_feats, fp, t_df=prob.get("t_df", 5))
    chosen = prob.get("dist", "t")

    if verbose:
        pd.set_option("display.width", 200)
        print(f"\n{len(res)} races, years {sorted(res['year'].unique())}")
        print(f"baseline features: {base_feats}")
        print(f"residual model: {residual}" + (f" on {resid_feats}" if residual != "none" else
                                               " (combined = baseline; reference to beat)"))
        print("\n== margins, every year held out (baseline refit inside each fold) ==")
        print(margin_report(res).round(4).to_string(index=False))
        print("\nby year (combined):")
        by = errors_by_year(res, pred_col="pred_margin")
        sig = res.groupby("year")[["sigma_national", "sigma_race", "sigma"]].first().reset_index()
        print(by.merge(sig, on="year").round(4).to_string(index=False))
        print("\n== win probabilities on held-out years (lower = better) ==")
        print(probability_report(res).round(4).to_string(index=False))
        print(f"\ncalibration, method '{chosen}' (config probability.dist): "
              "predicted vs actual D-win rate per bucket")
        print(calibration_table(res[f"p_{chosen}"], res["d_won"]).round(3).to_string(index=False))
        print("\nupset check: underdogs' predicted vs actual win rate")
        print(underdog_check(res[f"p_{chosen}"], res["d_won"]).round(3).to_string(index=False))

    dest = repo_path("data/processed/two_stage_oof.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(dest, index=False)
    if verbose:
        print(f"\nsaved {dest}")
    if log:
        log_experiment(repo_path("reports/experiments.csv"), f"two-stage ({residual})",
                       config=cfg.get("_config_path", ""), features="+".join(base_feats),
                       cv="nested leave-one-year-out",
                       rmse=float(np.sqrt(np.mean(res["error"] ** 2))),
                       log_loss=log_loss(res[f"p_{chosen}"], res["d_won"]),
                       brier=brier(res[f"p_{chosen}"], res["d_won"]),
                       notes=f"{len(res)} races; residual={residual}; probability={chosen}")
    return res


def main(argv=None):
    from .config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--residual", default="none", choices=["none", "linear", "lightgbm"])
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args(argv)
    # macOS numpy (Apple Accelerate) prints harmless matmul warnings inside sklearn
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        run(load_config(a.config), residual=a.residual, log=a.log)


if __name__ == "__main__":
    main()
