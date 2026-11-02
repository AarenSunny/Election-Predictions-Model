"""
Stage 2: residual model + full pipeline.  *** SKELETON: model parts not implemented. ***

Pipeline (run with `python -m election_prediction.train --config configs/baseline.yaml`):
    1. load + validate data                  data.load_table / validate
    2. build features                        features.*
    3. stage-1 baseline, out-of-fold         baseline.baseline_residuals       (done; compare.py
                                                                                picks its inputs)
    4. stage-2 LightGBM on residuals         fit_residual_model below          [TODO model]
    5. margins -> probabilities              probability.sigma_components / prob_from_margin
    6. score + benchmark                     evaluate.*, benchmarks.backtest
    7. log the run                           evaluate.log_experiment -> reports/experiments.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import load_config


def fit_residual_model(train: pd.DataFrame, features: list[str], params: dict):
    """LightGBM regressor on the 'residual' column. Return a fitted model."""
    raise NotImplementedError("residual model not built yet")


def predict_residual(model, df: pd.DataFrame, features: list[str]):
    raise NotImplementedError("residual model not built yet")


def load_baseline_inputs(cfg: dict):
    """(national, incumbency) tables from the spreadsheets listed under data: in the config."""
    from .config import repo_path
    from .inputs import build_national, load_incumbency

    d = cfg["data"]
    national = build_national(repo_path(d["generic_ballot"]), repo_path(d["midterm_flag"]),
                              repo_path(d["president_flag"]), repo_path(d["approval"]))
    incumbency = load_incumbency(repo_path(d["incumbency"]))
    return national, incumbency


def build_training_table(cfg: dict, approval_source: str | None = None, verbose: bool = True,
                         inputs=None) -> pd.DataFrame:
    """
    Results + lean + every baseline input -> one row per contested, regular (non-special)
    past race in years.train. Columns: KEYS, race_id, margin (target), lean,
    generic_ballot, incumbency_signed, approval_x_pres, midterm_x_pres,
    approval_x_pres_midterm (+ the raw flags/approval they were built from).

    approval_source overrides baseline.approval_source (used by compare.py).
    inputs = (national, incumbency) skips re-reading the spreadsheets.
    """
    from .config import repo_path
    from .data import build_baseline_table
    from .features import add_baseline_inputs

    results = pd.read_csv(repo_path(cfg["data"]["results"]))
    lean = pd.read_csv(repo_path(cfg["data"]["lean"]))
    base = cfg.get("baseline", {})
    table = build_baseline_table(results, lean, verbose=verbose,
                                 drop_specials=base.get("drop_specials", True))
    table = table[table["year"].isin(cfg["years"]["train"])].reset_index(drop=True)
    national, incumbency = inputs if inputs is not None else load_baseline_inputs(cfg)
    source = approval_source or base.get("approval_source", "reliable_30d")
    table = add_baseline_inputs(table, national, incumbency, source)
    if verbose:
        miss = table["incumbency_signed"].isna()
        national_ok = table[["generic_ballot", "pres_flag", "midterm_flag", "approval"]].notna().all().all()
        print(f"\napproval series: {source}; generic ballot / flags / approval found for every year: "
              f"{bool(national_ok)}")
        print(f"incumbency matched {len(table) - miss.sum()} / {len(table)} races"
              + (f"; missing: {', '.join(table.loc[miss, 'race_id'].head(10))}" if miss.any() else ""))
    return table


def run(cfg: dict) -> pd.DataFrame:
    """Stage 1 only (baseline). Steps 4-7 come once the residual model exists."""
    from .baseline import baseline_residuals, describe_baseline
    from .config import repo_path, validate_config
    from .evaluate import errors_by_year, log_experiment, margin_errors
    from .probability import sigma_components

    validate_config(cfg)
    feats = list(cfg["baseline"]["features"])
    table = build_training_table(cfg)
    if table.empty:
        raise ValueError("no races left after joining results and lean; check data/raw/lean.csv")
    blank = table[feats].isna().any(axis=1)
    if blank.any():
        examples = ", ".join(table.loc[blank, "race_id"].head(5))
        print(f"dropping {blank.sum()} races with a blank input ({examples} ...)")
        table = table[~blank].reset_index(drop=True)
    print(f"\n{len(table)} races, chambers: {table['chamber'].value_counts().to_dict()}")
    print(f"baseline features: {feats}")

    out = baseline_residuals(table, feats)                 # leave-one-year-out
    overall = margin_errors(out["margin"], out["baseline"])
    no_model = margin_errors(out["margin"], 0.0 * out["margin"])
    sig = sigma_components(out.assign(error=out["residual"]))

    coefs = {k: round(v, 4) for k, v in describe_baseline(table, feats).items()}
    print("\ncoefficients (fit on all years):", coefs)
    print(f"out-of-fold RMSE {overall['rmse']:.4f}  (no-model guess of 0: {no_model['rmse']:.4f})")
    print(f"winner accuracy {overall['winner_accuracy']:.3f}")
    print("\nerrors by year:")
    print(errors_by_year(out, pred_col="baseline").round(4).to_string(index=False))
    print(f"\nsigma_national {sig['sigma_national']:.4f}  sigma_race {sig['sigma_race']:.4f}")
    print("\n20 biggest misses:")
    cols = ["race_id", "dem_name", "rep_name", "lean", "margin", "baseline", "residual"]
    top = out.reindex(out["residual"].abs().sort_values(ascending=False).index).head(20)
    print(top[[c for c in cols if c in top]].round(3).to_string(index=False))

    dest = repo_path(cfg["data"]["baseline_residuals"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\nsaved {dest}")

    log_experiment(repo_path("reports/experiments.csv"), cfg.get("run_name", "baseline"),
                   config=cfg.get("_config_path", ""), features="+".join(feats),
                   cv="leave-one-year-out", rmse=overall["rmse"],
                   notes=f"{len(out)} races; approval={cfg['baseline'].get('approval_source')}; "
                         f"sigma_national={sig['sigma_national']:.4f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    a = ap.parse_args(argv)
    # macOS numpy (Apple Accelerate) prints harmless "divide by zero / overflow in matmul"
    # warnings inside sklearn's predict; results are unaffected (a real problem shows up as NaN).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        run(load_config(a.config))


if __name__ == "__main__":
    main()
