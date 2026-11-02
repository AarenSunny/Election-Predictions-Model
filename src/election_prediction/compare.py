"""
Which baseline inputs (and which approval series) actually help?

Every run is scored the same way: leave-one-year-out (each election year predicted by a
model that never saw it), on the SAME races, so the numbers are directly comparable.
Lower out-of-fold RMSE = better. Run:

    python -m election_prediction.compare --config configs/baseline.yaml [--log]

Three tables:

1. Feature ladder: start from lean, add one input at a time (approval = the config's
   baseline.approval_source).
2. One at a time: lean + generic_ballot + ONE extra input, to see each input's own value.
3. Approval series: the 2 sheets x 3 windows of approval, each in the same feature set
   (with and without generic_ballot, since generic ballot soaks up most of the same signal).

How to read it: an input earns its place if it lowers RMSE by a clear margin (say
0.001+) AND its coefficient has the expected sign. Also watch rmse_2018+ (out-of-fold
error on 2018-2024 only): those years are most like 2026, and an input whose effect has
faded (incumbency did: ~0.23 margin in 1998-2004, ~0.05 since 2018) can lower the
all-years RMSE while hurting recent years. Only 14 election years exist, so
national inputs are each learned from 14 numbers; small gains can be noise.
--log appends every run to reports/experiments.csv.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .inputs import APPROVAL_SOURCES

LADDER = [
    ["lean"],
    ["lean", "generic_ballot"],
    ["lean", "generic_ballot", "incumbency_signed"],
    ["lean", "generic_ballot", "incumbency_signed", "approval_x_pres"],
    ["lean", "generic_ballot", "incumbency_signed", "approval_x_pres", "midterm_x_pres"],
    ["lean", "generic_ballot", "incumbency_signed", "approval_x_pres", "midterm_x_pres",
     "approval_x_pres_midterm"],
]
EXTRAS = ["incumbency_signed", "approval_x_pres", "midterm_x_pres", "approval_x_pres_midterm"]
APPROVAL_SETS = {
    "lean + approval": ["lean", "approval_x_pres"],
    "lean + generic + inc + approval": ["lean", "generic_ballot", "incumbency_signed", "approval_x_pres"],
}
# expected coefficient signs (positive = helps Democrats)
EXPECTED_SIGN = {"lean": 1, "generic_ballot": 1, "incumbency_signed": 1, "approval_x_pres": 1,
                 "midterm_x_pres": -1, "approval_x_pres_midterm": 1}


RECENT_FROM = 2018   # 'rmse_recent' = out-of-fold RMSE on these years only (most like 2026)


def score(table: pd.DataFrame, features: list[str], recent_from: int = RECENT_FROM) -> dict:
    """Leave-one-year-out score of one feature set + coefficients fit on all years."""
    from .baseline import baseline_residuals, describe_baseline
    from .evaluate import errors_by_year, margin_errors

    out = baseline_residuals(table, features)
    m = margin_errors(out["margin"], out["baseline"])
    by_year = errors_by_year(out, pred_col="baseline")
    coefs = describe_baseline(table, features)
    wrong = [f for f in features if f in EXPECTED_SIGN and coefs[f] * EXPECTED_SIGN[f] < 0]
    recent = out["year"] >= recent_from
    rmse_recent = (margin_errors(out.loc[recent, "margin"], out.loc[recent, "baseline"])["rmse"]
                   if recent.any() else float("nan"))
    return {"rmse": m["rmse"], "rmse_recent": rmse_recent, "mae": m["mae"],
            "winner_acc": m["winner_accuracy"],
            "worst_year_rmse": by_year["rmse"].max(), "year_bias_sd": by_year["bias"].std(),
            "n": len(out), "coefs": coefs, "wrong_sign": wrong}


def _row(label: str, r: dict, features: list[str]) -> dict:
    coefs = ", ".join(f"{f}={r['coefs'][f]:+.4g}" for f in features)
    return {"run": label, "rmse": r["rmse"], f"rmse_{RECENT_FROM}+": r["rmse_recent"],
            "mae": r["mae"], "winner_acc": r["winner_acc"],
            "worst_year": r["worst_year_rmse"], "year_bias_sd": r["year_bias_sd"],
            "coefficients": coefs + ("  [WRONG SIGN: " + ", ".join(r["wrong_sign"]) + "]"
                                     if r["wrong_sign"] else "")}


def compare(cfg: dict, log: bool = False, verbose: bool = True) -> dict[str, pd.DataFrame]:
    from .config import repo_path, validate_config
    from .evaluate import log_experiment
    from .train import build_training_table, load_baseline_inputs

    validate_config(cfg)
    inputs = load_baseline_inputs(cfg)
    default_src = cfg.get("baseline", {}).get("approval_source", "reliable_30d")
    tables = {src: build_training_table(cfg, approval_source=src, verbose=verbose and src == default_src,
                                        inputs=inputs)
              for src in APPROVAL_SOURCES}
    # identical race set for every run: drop any race with a blank input in ANY run
    allf = sorted({f for fs in LADDER for f in fs})
    keep = pd.concat([t[allf].isna().any(axis=1) for t in tables.values()], axis=1).any(axis=1)
    tables = {k: t[~keep].reset_index(drop=True) for k, t in tables.items()}
    n = len(tables[default_src])
    if verbose:
        print(f"\ncomparing on the same {n} races ({keep.sum()} dropped for a blank input)")

    results: dict[str, list] = {"ladder": [], "one_at_a_time": [], "approval_series": []}
    runs = []
    for fs in LADDER:
        runs.append(("ladder", " + ".join(fs), default_src, fs))
    for extra in EXTRAS:
        fs = ["lean", "generic_ballot", extra]
        runs.append(("one_at_a_time", f"lean + generic_ballot + {extra}", default_src, fs))
    for name, fs in APPROVAL_SETS.items():
        for src in APPROVAL_SOURCES:
            runs.append(("approval_series", f"{name} [{src}]", src, fs))

    for group, label, src, fs in runs:
        r = score(tables[src], fs)
        results[group].append(_row(label, r, fs))
        if log:
            log_experiment(repo_path("reports/experiments.csv"), f"compare: {label}",
                           config=cfg.get("_config_path", ""), features="+".join(fs),
                           cv="leave-one-year-out", rmse=r["rmse"],
                           notes=f"{r['n']} races; approval={src}; winner_acc={r['winner_acc']:.4f}")
    frames = {k: pd.DataFrame(v) for k, v in results.items()}
    if verbose:
        with pd.option_context("display.width", 250, "display.max_colwidth", 140):
            for k, f in frames.items():
                print(f"\n== {k.replace('_', ' ')} (lower rmse = better) ==")
                print(f.round(4).to_string(index=False))
    return frames


def main(argv=None):
    from .config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--log", action="store_true", help="append every run to reports/experiments.csv")
    a = ap.parse_args(argv)
    # macOS numpy (Apple Accelerate) prints harmless "divide by zero / overflow in matmul"
    # warnings inside sklearn's predict; results are unaffected (a real problem shows up as NaN).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        compare(load_config(a.config), log=a.log)


if __name__ == "__main__":
    main()
