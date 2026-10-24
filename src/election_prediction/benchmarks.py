"""
Benchmarks: simple strategies the model has to beat, and backtests.

Past Kalshi prices don't exist for most races, so backtests use a STAND-IN
MARKET built from past expert ratings (Cook / Sabato / Inside Elections) via
rating_to_prob_D(). Treat backtest totals as a comparison between strategies,
not a forecast of your 2026 score.

Strategies
----------
  always_favorite   market favorite every time
  always_underdog   market underdog every time   <- the real bar to clear
  model_favorite    whoever the model thinks is more likely
  model_ev          expected-value rule (your submission rule)

Input columns: year, p_model_D, p_market_D, winner ("D"/"R")
(or a 'rating' column instead of p_market_D, e.g. "Lean R")

See results from the command line (run from the repo root):
    python -m election_prediction.benchmarks outputs/oof_predictions.csv
    python -m election_prediction.benchmarks outputs/oof_predictions.csv --log --name "baseline v1"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .decision import choose_side
from .evaluate import expected_points, log_experiment, score_picks

RATING_PROBS = {"safe": 0.97, "solid": 0.97, "likely": 0.85, "lean": 0.70, "tilt": 0.60}


def rating_to_prob_D(rating) -> float:
    """'Safe D', 'Lean R', 'Likely Democratic', 'Toss-up' -> P(D wins). Unknown -> NaN."""
    if not isinstance(rating, str):
        return np.nan
    r = rating.strip().lower().replace("democratic", "d").replace("democrat", "d")
    r = r.replace("republican", "r")
    if r in ("toss-up", "tossup", "toss up"):
        return 0.5
    parts = r.split()
    if len(parts) != 2 or parts[0] not in RATING_PROBS or parts[1] not in ("d", "r"):
        return np.nan
    p = RATING_PROBS[parts[0]]
    return p if parts[1] == "d" else 1.0 - p


def strategy_picks(df: pd.DataFrame, model_col="p_model_D", market_col="p_market_D") -> dict:
    pk = df[market_col].to_numpy(dtype=float)
    pm = df[model_col].to_numpy(dtype=float)
    return {
        "always_favorite": np.where(pk >= 0.5, "D", "R"),
        "always_underdog": np.where(pk >= 0.5, "R", "D"),
        "model_favorite": np.where(pm >= 0.5, "D", "R"),
        "model_ev": choose_side(pm, pk),
    }


def backtest(df: pd.DataFrame, model_col="p_model_D", market_col="p_market_D",
             winner_col="winner", year_col="year") -> pd.DataFrame:
    """Realized points per strategy per year, plus TOTAL. Unpriced races skipped."""
    d = df.dropna(subset=[model_col, market_col, winner_col]).reset_index(drop=True)
    rows = []
    for name, pick in strategy_picks(d, model_col, market_col).items():
        pts = pd.Series(score_picks(pick, d[winner_col], d[market_col]))
        for yr, s in pts.groupby(d[year_col]):
            rows.append({"strategy": name, "year": str(yr), "points": s.sum()})
        rows.append({"strategy": name, "year": "TOTAL", "points": pts.sum()})
    out = pd.DataFrame(rows).pivot(index="strategy", columns="year", values="points")
    return out.sort_values("TOTAL", ascending=False)


def expected_totals(df: pd.DataFrame, model_col="p_model_D", market_col="p_market_D",
                    n_sims: int = 10_000, seed: int = 0) -> pd.DataFrame:
    """
    For 2026: each strategy's expected total and 10th/50th/90th percentile,
    IF the model's probabilities are true. Races simulated independently.
    """
    d = df.dropna(subset=[model_col, market_col]).reset_index(drop=True)
    pm = d[model_col].to_numpy(dtype=float)
    draws = np.random.default_rng(seed).random((n_sims, len(d))) < pm  # True = D wins
    rows = []
    for name, pick in strategy_picks(d, model_col, market_col).items():
        right = np.where(pick == "D", draws, ~draws)
        pts = score_picks(pick, pick, d[market_col])
        totals = (right * pts).sum(axis=1)
        rows.append({"strategy": name,
                     "expected": float(expected_points(pick, pm, d[market_col]).sum()),
                     "p10": float(np.percentile(totals, 10)),
                     "median": float(np.median(totals)),
                     "p90": float(np.percentile(totals, 90))})
    return pd.DataFrame(rows).sort_values("expected", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ command line
def run_backtest(path, name: str | None = None, log_path: str | Path | None = None,
                 notes: str = "", show: bool = True) -> pd.DataFrame:
    """
    Load a predictions CSV, print the backtest, and optionally append a row to
    experiments.csv. If the file has a 'rating' column but no 'p_market_D',
    ratings are converted with rating_to_prob_D first.
    """
    df = pd.read_csv(path)
    if "p_market_D" not in df.columns and "rating" in df.columns:
        df["p_market_D"] = df["rating"].map(rating_to_prob_D)
    missing = [c for c in ("year", "p_model_D", "p_market_D", "winner") if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")

    usable = df.dropna(subset=["p_model_D", "p_market_D", "winner"])
    bt = backtest(df)
    if show:
        print(f"\nBacktest: {len(usable)} scored races "
              f"({len(df) - len(usable)} skipped for missing values)\n")
        print(bt.round(1).to_string())
        years = [c for c in bt.columns if c != "TOTAL"]
        wins = int((bt.loc["model_ev", years] > bt.loc["always_underdog", years]).sum())
        print(f"\nmodel_ev beat always_underdog in {wins} of {len(years)} years.")

    if log_path:
        log_experiment(
            log_path, name or Path(path).stem, notes=notes, cv="leave-one-year-out",
            backtest_points=float(bt.loc["model_ev", "TOTAL"]),
            always_underdog_points=float(bt.loc["always_underdog", "TOTAL"]),
            always_favorite_points=round(float(bt.loc["always_favorite", "TOTAL"]), 2),
            n_races=len(usable),
        )
        if show:
            print(f"Logged to {log_path}")
    return bt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Backtest model predictions against simple strategies.")
    ap.add_argument("predictions", help="CSV with year, p_model_D, p_market_D (or rating), winner")
    ap.add_argument("--log", action="store_true", help="append the result to reports/experiments.csv")
    ap.add_argument("--log-path", default="reports/experiments.csv")
    ap.add_argument("--name", default=None, help="experiment name (default: file name)")
    ap.add_argument("--notes", default="", help="what changed in this run")
    a = ap.parse_args(argv)
    run_backtest(a.predictions, a.name, a.log_path if a.log else None, a.notes)


if __name__ == "__main__":
    main()
