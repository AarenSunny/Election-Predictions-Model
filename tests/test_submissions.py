"""Correct columns, rows, value ranges in the final predictions file."""

import pandas as pd
import pytest

from election_prediction.predict import (
    SubmissionError,
    build_submission,
    validate_submission,
    write_submission,
)

KEYS = ["year", "chamber", "state", "district", "special"]


def _keys(n=3):
    return pd.DataFrame({"year": [2026] * n, "chamber": ["House"] * n, "state": ["NY"] * n,
                         "district": list(range(1, n + 1)), "special": [False] * n})


def test_build_submission_applies_picks_and_handles_gaps():
    preds = _keys().assign(p_model_D=[0.9, 0.5, 0.2])
    kalshi = _keys().assign(p_market_D=[0.9, 0.5, None])          # race 3 unpriced
    unc = _keys().iloc[[1]].assign(only_party="R")                # race 2 uncontested
    sub = build_submission(preds, kalshi, uncontested=unc).set_index("district")
    assert sub.loc[1, "pick"] == "R"   # model == market -> underdog has higher EV
    assert sub.loc[2, "pick"] == "R"   # uncontested override
    assert sub.loc[3, "pick"] == "R" and sub.loc[3, "no_market"]
    assert sub.loc[1, "race_id"] == "2026-H-NY-01"


def test_validate_submission_ok(tmp_path):
    sub = build_submission(_keys().assign(p_model_D=[0.6, 0.4, 0.5]),
                           _keys().assign(p_market_D=[0.5, 0.5, 0.5]))
    validate_submission(sub, expected_races=_keys(), expected_count=3)
    path = write_submission(sub, tmp_path / "sub.csv")
    assert len(pd.read_csv(path)) == 3


def test_validate_submission_catches_problems():
    sub = build_submission(_keys().assign(p_model_D=[0.6, 0.4, 0.5]),
                           _keys().assign(p_market_D=[0.5, 0.5, 0.5]))
    bad = pd.concat([sub, sub.iloc[[0]]])
    bad.loc[bad.index[-1], "pick"] = "X"
    with pytest.raises(SubmissionError) as e:
        validate_submission(bad, expected_races=_keys(4), expected_count=3)
    msg = str(e.value)
    assert "duplicate" in msg and "aren't D/R" in msg and "missing" in msg and "rows" in msg
