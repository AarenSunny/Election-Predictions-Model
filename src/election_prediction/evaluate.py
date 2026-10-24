"""
Contest evaluation metric + prediction-error measures + experiment log.

Contest rule (info slides): correct pick earns -log2(p), p = Kalshi's probability
for the side you picked, floored at 1% (max 6.64 points); wrong pick earns 0.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

PROB_FLOOR = 0.01
MAX_POINTS = float(-np.log2(PROB_FLOOR))
EPS = 1e-6


# ------------------------------------------------------------- contest score
def floor_prob(p):
    return np.clip(np.asarray(p, dtype=float), PROB_FLOOR, 1.0)


def points_if_correct(p_market_side):
    return -np.log2(floor_prob(p_market_side))


def market_prob_of_side(pick, p_market_D):
    pick = np.asarray(pick)
    pD = np.asarray(p_market_D, dtype=float)
    return np.where(pick == "D", pD, 1.0 - pD)


def score_picks(pick, winner, p_market_D):
    """Realized points per race."""
    pick = np.asarray(pick)
    pts = points_if_correct(market_prob_of_side(pick, p_market_D))
    return np.where(pick == np.asarray(winner), pts, 0.0)


def expected_points(pick, p_model_D, p_market_D):
    """Expected points per race if p_model_D were the true probability."""
    pick = np.asarray(pick)
    pm = np.asarray(p_model_D, dtype=float)
    p_side = np.where(pick == "D", pm, 1.0 - pm)
    return p_side * points_if_correct(market_prob_of_side(pick, p_market_D))


def contest_score(df: pd.DataFrame, pick_col="pick", winner_col="winner",
                  market_col="p_market_D") -> dict:
    pts = score_picks(df[pick_col], df[winner_col], df[market_col])
    correct = df[pick_col].to_numpy() == df[winner_col].to_numpy()
    p_side = market_prob_of_side(df[pick_col], df[market_col])
    return {
        "total_points": float(pts.sum()),
        "n_races": int(len(df)),
        "n_correct": int(correct.sum()),
        "n_underdog_picks": int((p_side < 0.5).sum()),
        "n_upsets_called": int((correct & (p_side < 0.5)).sum()),
    }


# ------------------------------------------------------------- margin errors
def margin_errors(actual, predicted) -> dict:
    """RMSE / MAE / bias of predicted margins, plus winner accuracy."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    e = a - p
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "winner_accuracy": float(np.mean(np.sign(a) == np.sign(p))),
    }


def errors_by_year(df: pd.DataFrame, actual_col="margin", pred_col="pred_margin",
                   year_col="year") -> pd.DataFrame:
    rows = [{"year": y, **margin_errors(g[actual_col], g[pred_col]), "n": len(g)}
            for y, g in df.groupby(year_col)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------- probability errors
def log_loss(p, y):
    """y = 1 if D won. Lower is better; 0.693 = coin flip."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def calibration_table(p, y, bins=(0, .05, .1, .2, .35, .5, .65, .8, .9, .95, 1.0)) -> pd.DataFrame:
    """Predicted vs. actual D-win rate per bucket. Finer bins in the tails."""
    df = pd.DataFrame({"p": np.asarray(p, dtype=float), "y": np.asarray(y, dtype=float)})
    df["bucket"] = pd.cut(df["p"], bins=list(bins), include_lowest=True)
    out = df.groupby("bucket", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    out["gap"] = out["actual"] - out["predicted"]
    return out.reset_index()


def underdog_check(p_model_D, y, bins=(0, .05, .1, .2, .35, .5)) -> pd.DataFrame:
    """
    Predicted vs. actual UPSET rate. If actual runs above predicted in the low
    buckets, the tails are too thin -> use dist='t' in probability.prob_from_margin.
    """
    p = np.asarray(p_model_D, dtype=float)
    y = np.asarray(y, dtype=float)
    p_dog = np.minimum(p, 1 - p)
    dog_won = np.where(p < 0.5, y, 1 - y)
    return calibration_table(p_dog, dog_won, bins=bins).rename(
        columns={"predicted": "pred_upset_rate", "actual": "actual_upset_rate"})


# ------------------------------------------------------------- experiment log
EXPERIMENT_COLUMNS = ["timestamp", "name", "config", "features", "cv", "rmse", "log_loss",
                      "brier", "backtest_points", "always_underdog_points", "notes"]


def log_experiment(path: str | Path, name: str, notes: str = "", **metrics) -> None:
    """
    Append one row to reports/experiments.csv (header written if the file is
    empty or still holds the placeholder text). Unknown keys go into notes.
    """
    path = Path(path)
    row = {c: "" for c in EXPERIMENT_COLUMNS}
    row.update(timestamp=dt.datetime.now().isoformat(timespec="seconds"), name=name, notes=notes)
    extra = []
    for k, v in metrics.items():
        if k in row:
            row[k] = round(v, 5) if isinstance(v, float) else v
        else:
            extra.append(f"{k}={v}")
    if extra:
        row["notes"] = "; ".join(filter(None, [notes] + extra))
    needs_header = (not path.exists() or path.stat().st_size == 0
                    or not path.read_text().startswith("timestamp"))
    mode = "w" if needs_header else "a"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS)
        if needs_header:
            w.writeheader()
        w.writerow(row)
