"""
Produce and check the submission file.

    preds  = DataFrame with race keys + p_model_D (from the trained model)
    kalshi = race table from kalshi.race_probabilities / the Nov 1 snapshot
    sub = build_submission(preds, kalshi)
    validate_submission(sub, expected_races=race_list)
    write_submission(sub, "outputs/submission.csv")

The organizers' exact CSV columns aren't confirmed yet: change SUBMISSION_COLUMNS
(and the rename in write_submission) once they say.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import KEYS, race_id
from .decision import add_picks

SUBMISSION_COLUMNS = ["race_id", "chamber", "state", "district", "special",
                      "pick", "p_model_D", "p_market_D"]


def build_submission(preds: pd.DataFrame, kalshi: pd.DataFrame,
                     uncontested: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Join model probabilities to market prices and apply the EV pick rule.
      * race in preds but not priced on Kalshi -> pick the model favorite, flagged
      * uncontested races (keys + 'only_party' D/R) -> pick that party
    """
    df = preds[KEYS + ["p_model_D"]].merge(
        kalshi[KEYS + ["p_market_D"]], on=KEYS, how="left", validate="one_to_one")
    priced = df["p_market_D"].notna()
    df = pd.concat([add_picks(df[priced]), df[~priced].assign(
        pick=np.where(df.loc[~priced, "p_model_D"] >= 0.5, "D", "R"))], ignore_index=True)
    df["no_market"] = df["p_market_D"].isna()
    if uncontested is not None and len(uncontested):
        u = uncontested[KEYS + ["only_party"]]
        df = df.merge(u, on=KEYS, how="left")
        has = df["only_party"].isin(["D", "R"])
        df.loc[has, "pick"] = df.loc[has, "only_party"]
        df = df.drop(columns="only_party")
    df["race_id"] = race_id(df)
    return df.sort_values(["chamber", "state", "district"]).reset_index(drop=True)


class SubmissionError(ValueError):
    pass


def validate_submission(sub: pd.DataFrame, expected_races: pd.DataFrame | None = None,
                        expected_count: int | None = 470) -> pd.DataFrame:
    """Columns present, one row per race, picks D/R, probabilities in [0, 1], no gaps."""
    problems = []
    missing = [c for c in SUBMISSION_COLUMNS if c not in sub.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    if "race_id" in sub and sub["race_id"].duplicated().any():
        problems.append(f"duplicate race_ids: {sub.loc[sub['race_id'].duplicated(), 'race_id'].tolist()[:5]}")
    if "pick" in sub and not sub["pick"].isin(["D", "R"]).all():
        problems.append(f"{(~sub['pick'].isin(['D', 'R'])).sum()} picks that aren't D/R")
    for c in ("p_model_D", "p_market_D"):
        if c in sub:
            v = sub[c].dropna()
            if ((v < 0) | (v > 1)).any():
                problems.append(f"'{c}' outside [0, 1]")
    if "p_model_D" in sub and sub["p_model_D"].isna().any():
        problems.append(f"{sub['p_model_D'].isna().sum()} races with no model probability")
    if expected_count is not None and len(sub) != expected_count:
        problems.append(f"{len(sub)} rows, expected {expected_count}")
    if expected_races is not None:
        want = set(race_id(expected_races))
        have = set(sub["race_id"]) if "race_id" in sub else set()
        if want - have:
            problems.append(f"{len(want - have)} races missing, e.g. {sorted(want - have)[:5]}")
        if have - want:
            problems.append(f"{len(have - want)} unexpected races, e.g. {sorted(have - want)[:5]}")
    if problems:
        raise SubmissionError("; ".join(problems))
    return sub


def write_submission(sub: pd.DataFrame, path: str | Path, rename: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub[SUBMISSION_COLUMNS].rename(columns=rename or {}).to_csv(path, index=False)
    return path
