"""Small, regularized margin calibration with fully nested year validation.

No production files are overwritten. The fold predictor must refit BOTH stages on
its train argument only; calibration regenerates its training predictions inside
that boundary. Cached/global baseline residuals are not a valid substitute.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm, t

METHODS = ("normal", "t5", "sigmoid", "sigmoid_chamber")
EPS = 1e-6


def _finite(x, name):
    a = np.asarray(x, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


@dataclass
class MarginCalibrator:
    method: str = "sigmoid"
    coefficients: list = field(default_factory=list)
    chambers: list = field(default_factory=list)
    sigmas: dict = field(default_factory=dict)
    pooled_sigma: float = 0.1

    def fit(self, predictions: pd.DataFrame):
        """Fit on genuinely held-out margins from TRAINING years only.

        Columns: pred_margin, margin, year, chamber. No market is required.
        Sigmoid slope is constrained positive; chamber offsets are strongly shrunk.
        """
        if self.method not in METHODS:
            raise ValueError(f"unknown calibration method {self.method}")
        if len(predictions) < 10 or predictions.year.nunique() < 2:
            raise ValueError("calibration requires >=10 races from >=2 training years")
        x = _finite(predictions.pred_margin, "pred_margin")
        actual = _finite(predictions.margin, "margin")
        if (actual == 0).any():
            raise ValueError("tied results must be adjudicated or excluded before calibration")
        y = (actual > 0).astype(float)
        if len(np.unique(y)) != 2:
            raise ValueError("calibration requires both Democratic and Republican wins")
        self.chambers = sorted(predictions.chamber.astype(str).unique().tolist())
        errors = pd.Series(actual - x, index=predictions.index)
        annual = errors.groupby(predictions.year).mean()
        within = errors - errors.groupby(predictions.year).transform("mean")
        nat_var = float(annual.var(ddof=1))
        race_var = float(within.var(ddof=1))
        # A small floor prevents zero uncertainty on artificially perfect samples.
        self.pooled_sigma = max(0.01, float(np.sqrt(nat_var + race_var)))
        for ch in self.chambers:
            values = within[predictions.chamber.astype(str) == ch]
            local = float(values.var(ddof=1)) if len(values) > 1 else race_var
            shrunk = (len(values) * local + 200 * race_var) / (len(values) + 200)
            self.sigmas[ch] = max(0.01, float(np.sqrt(nat_var + shrunk)))
        if self.method.startswith("sigmoid"):
            design = self._design(predictions)
            penalty = np.array([1.0, 2.0] + [20.0] * (design.shape[1] - 2))

            def objective(beta):
                z = design @ beta
                loss = np.logaddexp(0, z).sum() - y @ z + 0.5 * (penalty * beta**2).sum()
                grad = design.T @ (expit(z) - y) + penalty * beta
                return float(loss), grad

            initial = np.zeros(design.shape[1])
            initial[0] = 2.0
            fitted = minimize(objective, initial, jac=True, method="L-BFGS-B",
                              bounds=[(0.001, 30)] + [(None, None)] * (len(initial) - 1))
            if not fitted.success:
                raise ValueError(f"calibration optimizer failed: {fitted.message}")
            self.coefficients = fitted.x.tolist()
        return self

    def _design(self, df):
        cols = [_finite(df.pred_margin, "pred_margin") / 0.1, np.ones(len(df))]
        if self.method == "sigmoid_chamber":
            cols += [(df.chamber.astype(str) == c).to_numpy(dtype=float) for c in self.chambers[1:]]
        return np.column_stack(cols)

    def predict(self, df):
        if not self.sigmas:
            raise ValueError("fit the calibrator before prediction")
        if self.method.startswith("sigmoid"):
            p = expit(self._design(df) @ np.asarray(self.coefficients))
        else:
            sigma = df.chamber.astype(str).map(self.sigmas).fillna(self.pooled_sigma).to_numpy()
            z = _finite(df.pred_margin, "pred_margin") / sigma
            p = norm.cdf(z) if self.method == "normal" else t.cdf(z * np.sqrt(5 / 3), df=5)
        return np.clip(p, EPS, 1 - EPS)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def nested_calibration(table, predict_fold, methods=METHODS, forward=False, test_years=None):
    """Evaluate the entire chain; outer test labels never enter calibration.

    predict_fold(train, test) -> margin array, trained from scratch on train only.
    For forward=True, outer fits use ONLY earlier years. Inner OOF holds out each
    training year; it may use later TRAINING years, never the outer future year.
    """
    table = table.reset_index(drop=True)
    years = sorted(table.year.unique())
    outputs, cache = [], {}

    def predict(train, test):
        key = (tuple(sorted(train.year.unique())), tuple(test.index))
        if key not in cache:
            p = _finite(predict_fold(train.copy(), test.copy()), "fold prediction")
            if p.shape != (len(test),):
                raise ValueError("fold predictor returned wrong number of rows")
            cache[key] = p
        return cache[key]

    for year in test_years or years:
        train = table[table.year < year] if forward else table[table.year != year]
        test = table[table.year == year].copy()
        if train.year.nunique() < 3 or test.empty:
            continue
        training_oof = []
        for inner_year in sorted(train.year.unique()):
            inner_train = train[train.year != inner_year]
            inner_test = train[train.year == inner_year].copy()
            inner_test["pred_margin"] = predict(inner_train, inner_test)
            training_oof.append(inner_test)
        calibration_train = pd.concat(training_oof)
        test["pred_margin"] = predict(train, test)
        for method in methods:
            test[f"p_{method}"] = MarginCalibrator(method).fit(calibration_train).predict(test)
        outputs.append(test)
    if not outputs:
        raise ValueError("no evaluable folds (need >=4 years)")
    return pd.concat(outputs).sort_index()


def calibration_metrics(predictions, methods=METHODS):
    from .evaluate import brier, log_loss
    rows = []
    for label, mask in [("all", np.ones(len(predictions), dtype=bool)),
                        ("2018+", predictions.year >= 2018),
                        ("competitive_predicted_margin", predictions.pred_margin.abs() <= 0.1)]:
        d = predictions.loc[mask]
        if d.empty:
            continue
        y = (d.margin > 0).astype(int)
        for method in methods:
            p = d[f"p_{method}"]
            rows.append(dict(subset=label, method=method, n=len(d),
                             log_loss=log_loss(p, y), brier=brier(p, y),
                             accuracy=float(((p >= 0.5) == y).mean())))
    return pd.DataFrame(rows)


def blend_market(p_model, p_market, model_weight):
    """Explicit sensitivity parameter, NOT a historically optimized weight."""
    if not 0 <= model_weight <= 1:
        raise ValueError("model_weight must be in [0, 1]")
    pm = _finite(p_model, "model probability")
    pk = np.asarray(p_market, dtype=float)
    valid = ~np.isnan(pk)
    if ((pm < 0) | (pm > 1)).any() or not np.isfinite(pk[valid]).all():
        raise ValueError("invalid probabilities")
    if ((pk[valid] < 0) | (pk[valid] > 1)).any():
        raise ValueError("market probability outside [0, 1]")
    out = pm.copy()
    # Exact endpoints preserve existing model or market probabilities.
    if model_weight == 0:
        out[valid] = pk[valid]
    elif model_weight != 1:
        out[valid] = expit(model_weight * logit(pm[valid].clip(EPS, 1-EPS))
                           + (1-model_weight) * logit(pk[valid].clip(EPS, 1-EPS)))
    return out


def decision_audit(df, model_col="p_model_D"):
    """Exact optimal per-race expected-points picks under the repository's rule.

    It maximizes expected SUM, not probability of winning a leaderboard. Missing
    market prices get a model-favorite fallback and no invented expected score.
    """
    from .decision import choose_side
    from .evaluate import points_if_correct
    out = df.copy()
    pm = _finite(out[model_col], model_col)
    pk = out.p_market_D.to_numpy(dtype=float)
    blend_market(pm, pk, 1)  # validate, allowing absent market values
    priced = np.isfinite(pk)
    rd, rr = points_if_correct(pk), points_if_correct(1-pk)
    out["p_D_break_even"] = np.divide(rr, rd+rr, out=np.full(len(out), np.nan), where=(rd+rr)>0)
    out["pick"] = np.where(pm >= 0.5, "D", "R")
    out.loc[priced, "pick"] = choose_side(pm[priced], pk[priced])
    out["ev_D"], out["ev_R"] = pm * rd, (1-pm) * rr
    out["pick_ev"] = np.where(out.pick == "D", out.ev_D, out.ev_R)
    out["ev_advantage"] = np.abs(out.ev_D - out.ev_R)
    out["probability_distance_to_flip"] = np.abs(pm - out.p_D_break_even)
    out["no_market"] = ~priced
    # Preserve known single-party handling, but do not fabricate probabilities.
    if "only_party" in out:
        forced = out.only_party.isin(["D", "R"])
        out.loc[forced, "pick"] = out.loc[forced, "only_party"]
        out["pick_ev"] = np.where(out.pick == "D", out.ev_D, out.ev_R)
    return out


def baseline_fold_predictor(features, residual_fp=None, fold_features=None, prepare_fold=None):
    """Adapter for baseline or Claude's residual callback; no shared-file edits."""
    from .baseline import baseline_residuals, fit_baseline, predict_baseline
    if fold_features is None:
        from .twostage import poll_gap_features
        fold_features = poll_gap_features

    def predict(train, test):
        if prepare_fold is not None:
            train, test = prepare_fold(train.copy(), test.copy())
        elif residual_fp is not None and "prev_overperf" in train and train.prev_overperf.notna().any():
            raise ValueError("prev_overperf needs prepare_fold to rebuild history within this split")
        pred = predict_baseline(fit_baseline(train, features), test, features)
        if residual_fp is not None:
            training_oof = baseline_residuals(train, features)
            training_oof = fold_features(training_oof, "baseline")
            heldout = test.assign(baseline=pred)
            heldout = fold_features(heldout, "baseline")
            pred = pred + np.asarray(residual_fp(training_oof, heldout))
        return pred
    return predict


def main(argv=None):
    from .baseline import baseline_residuals
    from .config import load_config
    from .evaluate import calibration_table
    from .train import build_training_table

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--out-dir", default="outputs/calibration")
    ap.add_argument("--method", choices=METHODS, default="t5")
    ap.add_argument("--training-table", help="CSV of raw race inputs; all model predictions are refit")
    ap.add_argument("--forecast", help="matching-model forecast CSV with pred_margin, chamber")
    ap.add_argument("--model-weight", type=float, default=1.0)
    ap.add_argument("--benchmark", help="official NOVEMBER 1 odds CSV, race keys + p_market_D")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    feats = cfg["baseline"]["features"]
    if args.training_table:
        table = pd.read_csv(args.training_table)
        table = table[table.year.isin(cfg["years"]["train"])]
        if cfg.get("baseline", {}).get("drop_specials", True) and "special" in table:
            if not pd.api.types.is_bool_dtype(table.special):
                raise ValueError("special column must contain True/False")
            table = table[~table.special]
    else:
        table = build_training_table(cfg, verbose=False)
    table = table.dropna(subset=feats).reset_index(drop=True)
    if "race_id" in table and table.race_id.duplicated().any():
        raise ValueError("duplicate training race IDs")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictor = baseline_fold_predictor(feats)
    results = nested_calibration(table, predictor)
    results.to_csv(out / "heldout_probabilities.csv", index=False)
    report = calibration_metrics(results)
    report.to_csv(out / "method_comparison.csv", index=False)
    for column in ("year", "chamber"):
        grouped = []
        for label, group in results.groupby(column):
            metrics = calibration_metrics(group)
            metrics[column] = label
            grouped.append(metrics)
        pd.concat(grouped).to_csv(out / f"by_{column}.csv", index=False)
    print(report.round(5).to_string(index=False))
    forward = nested_calibration(table, predictor, forward=True, test_years=[2024])
    calibration_metrics(forward).to_csv(out / "forward_2024.csv", index=False)
    print("\nForward-only 2024:")
    print(calibration_metrics(forward).round(5).to_string(index=False))
    calibration_table(results[f"p_{args.method}"], (results.margin > 0).astype(int)).to_csv(
        out / "reliability.csv", index=False)
    oof = baseline_residuals(table, feats).rename(columns={"baseline": "pred_margin"})
    fitted = MarginCalibrator(args.method).fit(oof)
    fitted.save(out / "calibrator.json")
    if args.forecast:
        races = pd.read_csv(args.forecast)
        from .baseline import fit_baseline, predict_baseline
        expected = predict_baseline(fit_baseline(table, feats), races, feats)
        if not np.allclose(expected, races.pred_margin, atol=1e-8, rtol=1e-6):
            raise ValueError("forecast margins do not match this baseline; use matching model OOF predictions")
        # This CLI currently fits a BASELINE calibrator: reject mislabeled residual forecasts.
        if "resid_pred" in races and np.any(np.abs(races.resid_pred) > 1e-12):
            raise ValueError("use the API with matching residual-model calibration predictions")
        # Forecast evidence can be newer; scoring rewards MUST use the frozen benchmark.
        races["p_evidence_D"] = races["p_market_D"]
        if args.benchmark:
            from .data import KEYS, standardize_keys
            frozen = standardize_keys(pd.read_csv(args.benchmark))
            races = races.drop(columns="p_market_D").merge(
                frozen[KEYS + ["p_market_D"]], on=KEYS, how="left", validate="one_to_one")
        races["benchmark_confirmed"] = bool(args.benchmark)
        races["p_calibrated_raw"] = fitted.predict(races)
        forced = races.get("only_party", pd.Series("", index=races.index)).isin(["D", "R"])
        races.loc[forced, "p_calibrated_raw"] = np.where(races.loc[forced, "only_party"] == "D", 0.999, 0.001)
        races["p_model_D"] = blend_market(races.p_calibrated_raw, races.p_evidence_D, args.model_weight)
        races.loc[forced, "p_model_D"] = races.loc[forced, "p_calibrated_raw"]
        audit = decision_audit(races)
        # Sensitivity across methods is not a confidence interval.
        alternatives = []
        for method in METHODS:
            q = MarginCalibrator(method).fit(oof).predict(races)
            q[forced.to_numpy()] = races.loc[forced, "p_calibrated_raw"]
            q = blend_market(q, races.p_evidence_D, args.model_weight)
            q[forced.to_numpy()] = races.loc[forced, "p_calibrated_raw"]
            alternatives.append(q)
        audit["p_method_min"] = np.min(alternatives, axis=0)
        audit["p_method_max"] = np.max(alternatives, axis=0)
        audit["pick_stable_across_methods"] = ((audit.p_method_min >= audit.p_D_break_even)
                                               | (audit.p_method_max < audit.p_D_break_even))
        audit.loc[forced, "pick_stable_across_methods"] = True
        audit.to_csv(out / "forecast_audit.csv", index=False)
        from .predict import validate_submission, write_submission
        validate_submission(audit, expected_count=cfg["submission"]["expected_count"])
        name = "submission_candidate.csv" if args.benchmark else "submission_PROVISIONAL.csv"
        write_submission(audit, out / name)
        if not args.benchmark:
            print("WARNING: benchmark not confirmed; provisional November-2 rewards must not be submitted.")
        sensitivity = []
        for weight in [0, 0.25, 0.5, 0.75, 1]:
            d = races.copy()
            d["p_model_D"] = blend_market(d.p_calibrated_raw, d.p_evidence_D, weight)
            d.loc[forced, "p_model_D"] = races.loc[forced, "p_calibrated_raw"]
            d = decision_audit(d)
            sensitivity.append(dict(model_weight=weight, n_priced=int((~d.no_market).sum()),
                                    n_D=int((d.pick == "D").sum()),
                                    expected_points_under_own_probabilities=float(d.pick_ev.sum())))
        pd.DataFrame(sensitivity).to_csv(out / "market_sensitivity.csv", index=False)
    metadata = dict(model="baseline", method=args.method, training_years=sorted(map(int, table.year.unique())),
                    baseline_features=feats, forecast=args.forecast, official_benchmark=args.benchmark,
                    benchmark_confirmed=bool(args.benchmark), model_weight=args.model_weight,
                    evaluation="nested year holdout; separate forward-2024 check",
                    caveat="model-weight scenarios are not historical evidence of contest edge")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nSaved to {out}; market weights are sensitivity scenarios, not validated edge.")


if __name__ == "__main__":
    main()
