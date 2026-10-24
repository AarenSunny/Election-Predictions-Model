"""
Training/validation split rules.

Races in the same year share a national swing, so a random split leaks: the
model would see 2018 races while being tested on other 2018 races. Every split
here holds out WHOLE YEARS.

    for fold in leave_one_year_out(df):
        train, test = df.loc[fold.train_idx], df.loc[fold.test_idx]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class Fold:
    test_year: int
    train_idx: pd.Index
    test_idx: pd.Index


def leave_one_year_out(df: pd.DataFrame, year_col: str = "year",
                       years: list[int] | None = None) -> Iterator[Fold]:
    """Hold out each year once; train on all other years (past AND future)."""
    all_years = sorted(df[year_col].dropna().unique())
    for yr in years or all_years:
        test = df.index[df[year_col] == yr]
        train = df.index[df[year_col] != yr]
        fold = Fold(int(yr), train, test)
        check_fold(df, fold, year_col)
        yield fold


def forward_chaining(df: pd.DataFrame, year_col: str = "year",
                     min_train_years: int = 2) -> Iterator[Fold]:
    """
    Stricter check: train only on years BEFORE the test year, like real life.
    Fewer folds, more honest. Use it to confirm leave-one-year-out results.
    """
    all_years = sorted(df[year_col].dropna().unique())
    for i, yr in enumerate(all_years):
        if i < min_train_years:
            continue
        test = df.index[df[year_col] == yr]
        train = df.index[df[year_col] < yr]
        fold = Fold(int(yr), train, test)
        check_fold(df, fold, year_col, forbid_future=True)
        yield fold


def check_fold(df: pd.DataFrame, fold: Fold, year_col: str = "year",
               forbid_future: bool = False, id_col: str | None = "race_id") -> None:
    """Raise if a fold leaks: overlapping rows, test year in train, or future years."""
    if len(fold.test_idx) == 0:
        raise ValueError(f"fold {fold.test_year}: empty test set")
    if len(fold.train_idx.intersection(fold.test_idx)):
        raise ValueError(f"fold {fold.test_year}: rows appear in both train and test")
    train_years = set(df.loc[fold.train_idx, year_col])
    if fold.test_year in train_years:
        raise ValueError(f"fold {fold.test_year}: test year also in train")
    if forbid_future and train_years and max(train_years) >= fold.test_year:
        raise ValueError(f"fold {fold.test_year}: train contains future years")
    if id_col and id_col in df.columns:
        shared = set(df.loc[fold.train_idx, id_col]) & set(df.loc[fold.test_idx, id_col])
        if shared:
            raise ValueError(f"fold {fold.test_year}: race_ids in both sets, e.g. "
                             f"{sorted(shared)[:3]}")


def out_of_fold(df: pd.DataFrame, fit_predict, year_col: str = "year",
                folds: Iterator[Fold] | None = None) -> np.ndarray:
    """
    Generic out-of-fold predictions. fit_predict(train_df, test_df) -> array.
    Every row gets a prediction from a model that never saw its year.
    """
    preds = pd.Series(np.nan, index=df.index, dtype=float)
    for fold in folds or leave_one_year_out(df, year_col):
        preds.loc[fold.test_idx] = np.asarray(
            fit_predict(df.loc[fold.train_idx], df.loc[fold.test_idx]), dtype=float)
    if preds.isna().any():
        raise ValueError(f"{preds.isna().sum()} rows got no out-of-fold prediction")
    return preds.to_numpy()
