"""No overlap, no future information, every row predicted once."""

import numpy as np
import pandas as pd
import pytest

from election_prediction.splits import Fold, check_fold, forward_chaining, leave_one_year_out, out_of_fold


def _df():
    years = [2010, 2010, 2012, 2012, 2014, 2014]
    return pd.DataFrame({"year": years,
                         "race_id": [f"{y}-H-NY-{i:02d}" for i, y in enumerate(years)],
                         "x": np.arange(6.0)})


def test_loyo_holds_out_whole_years():
    df = _df()
    folds = list(leave_one_year_out(df))
    assert [f.test_year for f in folds] == [2010, 2012, 2014]
    for f in folds:
        assert set(df.loc[f.test_idx, "year"]) == {f.test_year}
        assert f.test_year not in set(df.loc[f.train_idx, "year"])
        assert len(f.train_idx) + len(f.test_idx) == len(df)


def test_forward_chaining_never_uses_future():
    df = _df()
    folds = list(forward_chaining(df, min_train_years=1))
    assert [f.test_year for f in folds] == [2012, 2014]
    for f in folds:
        assert df.loc[f.train_idx, "year"].max() < f.test_year


def test_check_fold_catches_leaks():
    df = _df()
    leaky = Fold(2012, df.index, df.index[df["year"] == 2012])
    with pytest.raises(ValueError):
        check_fold(df, leaky)


def test_out_of_fold_never_sees_its_own_year():
    df = _df()

    def fit_predict(train, test):
        assert not set(train["year"]) & set(test["year"])
        return np.full(len(test), train["x"].mean())

    preds = out_of_fold(df, fit_predict)
    assert preds.shape == (6,) and not np.isnan(preds).any()
    assert preds[0] == df.loc[df["year"] != 2010, "x"].mean()
